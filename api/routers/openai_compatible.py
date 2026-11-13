# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible router for text-to-speech API.
Implements endpoints compatible with OpenAI's TTS API specification.
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..structures.schemas import (
    OpenAISpeechRequest,
    ModelInfo,
    VoiceInfo,
    VoiceCloneRequest,
    VoiceCloneCapabilities,
)
from ..services.text_processing import normalize_text
from ..services.audio_encoding import encode_audio, get_content_type, DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)

# GPU lock: serializes TTS generation to prevent GPU contention
# (pattern from groxaxo vllm_omni backend)
_gpu_lock = asyncio.Lock()
# Serializes all speech generation — one request at a time (GPU-bound, model swaps are global).
_speech_sem = asyncio.Semaphore(int(os.environ.get("TTS_MAX_CONCURRENT", "1")))

# Voice library directory (same as VOICE_LIBRARY_DIR in start_server.sh)
VOICE_LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "./voice_library")).resolve()

# Cache for reference audio reads (profile_name -> (audio, sr))
_ref_audio_cache = {}


def _load_voice_profile(name_or_id: str) -> dict:
    """Load a voice profile by name or profile_id from the voice library.

    Returns dict with keys: ref_audio_path, ref_text, x_vector_only_mode, language.
    Raises ValueError if not found.
    """
    profiles_dir = VOICE_LIBRARY_DIR / "profiles"
    if not profiles_dir.exists():
        raise ValueError(f"Voice library not found: {profiles_dir}")

    # Search all profiles
    for child in profiles_dir.iterdir():
        if not child.is_dir():
            continue
        meta_file = child / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Match by profile_id or name (case-insensitive)
        if meta.get("profile_id") == name_or_id or \
           meta.get("name", "").lower() == name_or_id.lower():
            # Found it -- build result
            ref_filename = meta.get("ref_audio_filename", "")
            if not ref_filename:
                raise ValueError(f"Profile '{name_or_id}' has no reference audio")
            ref_path = child / ref_filename
            if not ref_path.exists():
                raise ValueError(f"Reference audio missing: {ref_path}")
            return {
                "ref_audio_path": str(ref_path),
                "ref_text": meta.get("ref_text", ""),
                "x_vector_only_mode": meta.get("x_vector_only_mode", False),
                "language": meta.get("language", "Auto"),
                "name": meta.get("name", name_or_id),
            }

    raise ValueError(f"Voice profile not found: '{name_or_id}'")


# Language code to language name mapping
LANGUAGE_CODE_MAPPING = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

# Available models (including language-specific variants)
AVAILABLE_MODELS = [
    ModelInfo(
        id="qwen3-tts",
        object="model",
        created=1737734400,  # 2025-01-24
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1-hd",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
    ModelInfo(
        id="qwen3-tts-hd",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
    # direct config.yaml checkpoint keys (see config.yaml models:)
    ModelInfo(id="0.6B-Base", object="model", created=1737734400, owned_by="qwen"),
    ModelInfo(id="1.7B-Base", object="model", created=1737734400, owned_by="qwen"),
    ModelInfo(id="0.6B-CustomVoice", object="model", created=1737734400, owned_by="qwen"),
    ModelInfo(id="1.7B-CustomVoice", object="model", created=1737734400, owned_by="qwen"),
]

# Add language-specific model variants
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    AVAILABLE_MODELS.extend([
        ModelInfo(
            id=f"tts-1-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
        ModelInfo(
            id=f"tts-1-hd-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
    ])

# Model name mapping (OpenAI -> internal). Values are unused beyond
# validation; the backend resolves the real checkpoint itself.
MODEL_MAPPING = {
    "tts-1": "qwen3-tts",
    "tts-1-hd": "qwen3-tts",
    "qwen3-tts": "qwen3-tts",
    "qwen3-tts-hd": "qwen3-tts",
    # direct config.yaml model keys — full checkpoint selection
    "0.6B-Base": "qwen3-tts",
    "1.7B-Base": "qwen3-tts",
    "0.6B-CustomVoice": "qwen3-tts",
    "1.7B-CustomVoice": "qwen3-tts",
}

# Add language-specific model mappings
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    MODEL_MAPPING[f"tts-1-{lang_code}"] = "qwen3-tts"
    MODEL_MAPPING[f"tts-1-hd-{lang_code}"] = "qwen3-tts"

# OpenAI voice mapping to Qwen voices
# Must map to voices that actually exist in the model:
# aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Serena",
    "nova": "Aiden",
    "onyx": "Eric",
    "shimmer": "Dylan",
}


def extract_language_from_model(model_name: str) -> Optional[str]:
    """Extract language from model name if it has a language suffix."""
    for lang_code, lang_name in LANGUAGE_CODE_MAPPING.items():
        suffix = f"-{lang_code}"
        if model_name.endswith(suffix):
            if model_name == f"tts-1{suffix}" or model_name == f"tts-1-hd{suffix}":
                return lang_name
    return None


async def get_tts_backend():
    """Get the TTS backend instance, initializing if needed."""
    from ..backends import get_backend, initialize_backend

    backend = get_backend()

    if not backend.is_ready():
        await initialize_backend()

    return backend


def get_voice_name(voice: str) -> str:
    """Map voice name to internal voice identifier."""
    if voice.lower() in VOICE_MAPPING:
        return VOICE_MAPPING[voice.lower()]
    return voice


async def generate_speech(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Generate speech from text using the configured TTS backend."""
    backend = await get_tts_backend()
    voice_name = get_voice_name(voice)

    try:
        audio, sr = await backend.generate_speech(
            text=text,
            voice=voice_name,
            language=language,
            instruct=instruct,
            speed=speed,
        )
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Speech generation failed: {e}")


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
):
    """OpenAI-compatible endpoint for text-to-speech. Serialized via _speech_sem."""
    async with _speech_sem:
        return await _create_speech_impl(request, client_request)


async def _create_speech_impl(
    request: OpenAISpeechRequest,
    client_request: Request,
):
    """OpenAI-compatible endpoint for text-to-speech."""
    logger.info(f"TTS request: model={request.model}, voice={request.voice}, format={request.response_format}, len={len(request.input)}")

    if request.model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}. Supported: {list(MODEL_MAPPING.keys())}",
                "type": "invalid_request_error",
            },
        )

    try:
        normalized_text = normalize_text(request.input, request.normalization_options)

        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )

        model_language = extract_language_from_model(request.model)
        language = model_language if model_language else (request.language or "Auto")

        # Voice profile: voice name starts with "clone:" -> load from voice library
        if request.voice.lower().startswith("clone:"):
            profile_name = request.voice[6:].strip()
            if not profile_name:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_voice",
                        "message": "clone: prefix requires a profile name, e.g. 'clone:MyVoice'",
                        "type": "invalid_request_error",
                    },
                )
            try:
                profile = _load_voice_profile(profile_name)
            except ValueError as e:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "profile_not_found",
                        "message": str(e),
                        "type": "invalid_request_error",
                    },
                )

            backend = await get_tts_backend()

            # Cache reference audio reads per profile
            ref_audio_path = profile["ref_audio_path"]
            if profile_name in _ref_audio_cache:
                ref_audio, ref_sr = _ref_audio_cache[profile_name]
            else:
                ref_audio, ref_sr = sf.read(ref_audio_path)
                if len(ref_audio.shape) > 1:
                    ref_audio = ref_audio.mean(axis=1)
                ref_audio = ref_audio.astype(np.float32)
                _ref_audio_cache[profile_name] = (ref_audio, ref_sr)
                logger.info(f"Reference audio cached: '{profile_name}'")

            clone_lang = language if language != "Auto" else profile["language"]
            logger.info(f"Voice profile '{profile['name']}': lang={clone_lang}, xvec_only={profile['x_vector_only_mode']}, stream={request.stream}")

            if request.stream:
                # True streaming: yield PCM chunks as model generates them
                fmt = request.response_format
                if fmt == "wav":
                    fmt = "pcm"
                content_type = get_content_type(fmt)

                async def clone_audio_stream():
                    gen_start = time.time()
                    first_chunk_time = None
                    total_samples = 0
                    chunk_count = 0
                    sample_rate = 24000
                    async for pcm_chunk, sr in backend.generate_voice_clone_streaming(
                        text=normalized_text,
                        ref_audio=ref_audio,
                        ref_audio_sr=ref_sr,
                        ref_text=profile["ref_text"] or None,
                        language=clone_lang,
                        x_vector_only_mode=profile["x_vector_only_mode"],
                        cache_key=profile_name,
                        model=request.model,
                    ):
                        if pcm_chunk is not None and len(pcm_chunk) > 0:
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - gen_start
                            total_samples += len(pcm_chunk)
                            sample_rate = sr
                            chunk_count += 1
                            yield encode_audio(pcm_chunk, fmt, sr)
                            await asyncio.sleep(0)
                    gen_time = time.time() - gen_start
                    audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                    rtf = gen_time / audio_dur if audio_dur > 0 else 0
                    logger.info(f"Voice clone stream: First-Byte={first_chunk_time:.2f}s Gesamt={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x Chunks={chunk_count}")

                return StreamingResponse(
                    clone_audio_stream(),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )
            else:
                # Non-streaming: generate all audio, then send (better RTF)
                gen_start = time.time()
                audio, sample_rate = await backend.generate_voice_clone(
                    text=normalized_text,
                    ref_audio=ref_audio,
                    ref_audio_sr=ref_sr,
                    ref_text=profile["ref_text"] or None,
                    language=clone_lang,
                    x_vector_only_mode=profile["x_vector_only_mode"],
                    speed=request.speed,
                    cache_key=profile_name,
                    model=request.model,
                )
                gen_time = time.time() - gen_start

                audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
                rtf = gen_time / audio_dur if audio_dur > 0 else 0
                logger.info(f"Voice clone: Gen={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x")

                fmt = request.response_format
                audio_bytes = encode_audio(audio, fmt, sample_rate)
                content_type = get_content_type(fmt)

                async def send_audio():
                    for i in range(0, len(audio_bytes), 4096):
                        yield audio_bytes[i:i + 4096]

                return StreamingResponse(
                    send_audio(),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )

        # Voice cloning: task_type=Base with ref_audio (Voice Studio)
        if request.task_type == "Base" and request.ref_audio:
            backend = await get_tts_backend()

            ref_audio_data = request.ref_audio
            if ref_audio_data.startswith("data:"):
                ref_audio_data = ref_audio_data.split(",", 1)[1]
            audio_bytes_raw = base64.b64decode(ref_audio_data)
            audio_buffer = io.BytesIO(audio_bytes_raw)
            ref_audio, ref_sr = sf.read(audio_buffer)
            if len(ref_audio.shape) > 1:
                ref_audio = ref_audio.mean(axis=1)
            ref_audio = ref_audio.astype(np.float32)

            if request.stream:
                # True streaming: yield PCM chunks as model generates them
                fmt = request.response_format
                if fmt == "wav":
                    fmt = "pcm"
                content_type = get_content_type(fmt)

                async def clone_ref_stream():
                    gen_start = time.time()
                    first_chunk_time = None
                    total_samples = 0
                    chunk_count = 0
                    sample_rate = 24000
                    async for pcm_chunk, sr in backend.generate_voice_clone_streaming(
                        text=normalized_text,
                        ref_audio=ref_audio,
                        ref_audio_sr=ref_sr,
                        ref_text=request.ref_text,
                        language=language,
                        x_vector_only_mode=request.x_vector_only_mode or False,
                        model=request.model,
                    ):
                        if pcm_chunk is not None and len(pcm_chunk) > 0:
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - gen_start
                            total_samples += len(pcm_chunk)
                            sample_rate = sr
                            chunk_count += 1
                            yield encode_audio(pcm_chunk, fmt, sr)
                            await asyncio.sleep(0)
                    gen_time = time.time() - gen_start
                    audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                    rtf = gen_time / audio_dur if audio_dur > 0 else 0
                    logger.info(f"Voice clone stream: First-Byte={first_chunk_time:.2f}s Gesamt={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x Chunks={chunk_count}")

                return StreamingResponse(
                    clone_ref_stream(),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )

            logger.info(f"Voice clone: lang={language}, ref_text={request.ref_text is not None}, xvec_only={request.x_vector_only_mode}")

            gen_start = time.time()
            audio, sample_rate = await backend.generate_voice_clone(
                text=normalized_text,
                ref_audio=ref_audio,
                ref_audio_sr=ref_sr,
                ref_text=request.ref_text,
                language=language,
                x_vector_only_mode=request.x_vector_only_mode or False,
                speed=request.speed,
                model=request.model,
            )
            gen_time = time.time() - gen_start

            audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
            rtf = gen_time / audio_dur if audio_dur > 0 else 0
            logger.info(f"Voice clone: Gen={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x")

            audio_bytes = encode_audio(audio, request.response_format, sample_rate)
            content_type = get_content_type(request.response_format)

            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",
                },
            )

        if request.stream:
            backend = await get_tts_backend()
            voice_name = get_voice_name(request.voice)
            fmt = request.response_format
            if fmt == "wav":
                fmt = "pcm"
            content_type = get_content_type(fmt)

            async def audio_stream():
                gen_start = time.time()
                first_chunk_time = None
                total_samples = 0
                chunk_count = 0
                sample_rate = 24000
                async for pcm_chunk, sr in backend.generate_speech_streaming(
                    text=normalized_text,
                    voice=voice_name,
                    language=language,
                    instruct=request.instruct,
                    model=request.model,
                ):
                    if pcm_chunk is not None and len(pcm_chunk) > 0:
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - gen_start
                        total_samples += len(pcm_chunk)
                        sample_rate = sr
                        chunk_count += 1
                        yield encode_audio(pcm_chunk, fmt, sr)
                        await asyncio.sleep(0)
                gen_time = time.time() - gen_start
                audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                rtf = gen_time / audio_dur if audio_dur > 0 else 0
                logger.info(f"TTS stream: First-Byte={first_chunk_time:.2f}s Gesamt={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x Chunks={chunk_count}")

            return StreamingResponse(
                audio_stream(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{fmt}",
                    "Cache-Control": "no-cache",
                },
            )
        else:
            # Non-streaming: generate in thread pool to keep event loop free
            # GPU lock serializes access (pattern from groxaxo vllm_omni backend)
            async with _gpu_lock:
                gen_start = time.time()
                loop = asyncio.get_event_loop()
                audio, sample_rate = await loop.run_in_executor(
                    None,
                    lambda: asyncio.run(generate_speech(
                        text=normalized_text,
                        voice=request.voice,
                        language=language,
                        instruct=request.instruct,
                        speed=request.speed,
                    ))
                )
                gen_time = time.time() - gen_start

            audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
            rtf = gen_time / audio_dur if audio_dur > 0 else 0
            logger.info(f"TTS: Gen={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x")

            audio_bytes = encode_audio(audio, request.response_format, sample_rate)
            content_type = get_content_type(request.response_format)

            async def send_audio():
                for i in range(0, len(audio_bytes), 4096):
                    yield audio_bytes[i:i + 4096]

            return StreamingResponse(
                send_audio(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"inline; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"TTS request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.get("/models")
async def list_models():
    """List all available TTS models."""
    return {
        "object": "list",
        "data": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/audio/models")
async def list_audio_models():
    """List TTS models in OpenWebUI-compatible format."""
    return {
        "models": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model.model_dump()

    raise HTTPException(
        status_code=404,
        detail={
            "error": "model_not_found",
            "message": f"Model '{model_id}' not found",
            "type": "invalid_request_error",
        },
    )


@router.get("/audio/voices")
@router.get("/voices")
async def list_voices():
    """List all available voices for text-to-speech."""
    openai_voices = [
        VoiceInfo(id="alloy", name="Alloy", description="OpenAI-compatible voice (maps to Vivian)"),
        VoiceInfo(id="echo", name="Echo", description="OpenAI-compatible voice (maps to Ryan)"),
        VoiceInfo(id="fable", name="Fable", description="OpenAI-compatible voice (maps to Serena)"),
        VoiceInfo(id="nova", name="Nova", description="OpenAI-compatible voice (maps to Aiden)"),
        VoiceInfo(id="onyx", name="Onyx", description="OpenAI-compatible voice (maps to Eric)"),
        VoiceInfo(id="shimmer", name="Shimmer", description="OpenAI-compatible voice (maps to Dylan)"),
    ]

    default_languages = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Russian", "Portuguese", "Italian"]

    try:
        backend = await get_tts_backend()
        speakers = backend.get_supported_voices()
        languages = backend.get_supported_languages()

        if speakers:
            voices = []
            for speaker in speakers:
                voice_info = VoiceInfo(
                    id=speaker,
                    name=speaker,
                    language=languages[0] if languages else "Auto",
                    description=f"Qwen3-TTS voice: {speaker}",
                )
                voices.append(voice_info.model_dump())
        else:
            voices = []

        # Add clone profiles from voice library
        clone_voices = []
        profiles_dir = VOICE_LIBRARY_DIR / "profiles"
        if profiles_dir.exists():
            for child in sorted(profiles_dir.iterdir()):
                meta_file = child / "meta.json"
                if not meta_file.exists():
                    continue
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if meta.get("task_type") == "Base" and meta.get("ref_audio_filename"):
                        clone_id = f"clone:{meta['name']}"
                        clone_voices.append(VoiceInfo(
                            id=clone_id,
                            name=clone_id,
                            description=f"Cloned voice: {meta['name']}",
                        ).model_dump())
                except Exception:
                    pass

        return {
            "voices": voices + clone_voices + [v.model_dump() for v in openai_voices],
            "languages": languages if languages else default_languages,
        }

    except Exception as e:
        logger.warning(f"Could not get voices from backend: {e}")
        return {
            "voices": [v.model_dump() for v in openai_voices],
            "languages": default_languages,
        }


@router.get("/audio/voice-clone/capabilities")
async def get_voice_clone_capabilities():
    """Get voice cloning capabilities of the current backend."""
    try:
        backend = await get_tts_backend()
        supports_cloning = backend.supports_voice_cloning()
        model_type = backend.get_model_type() if hasattr(backend, 'get_model_type') else "unknown"

        return VoiceCloneCapabilities(
            supported=supports_cloning,
            model_type=model_type,
            icl_mode_available=supports_cloning,
            x_vector_mode_available=supports_cloning,
        )
    except Exception as e:
        logger.warning(f"Could not get voice clone capabilities: {e}")
        return VoiceCloneCapabilities(
            supported=False,
            model_type="unknown",
            icl_mode_available=False,
            x_vector_mode_available=False,
        )


@router.post("/audio/voice-clone")
async def create_voice_clone(
    request: VoiceCloneRequest,
    client_request: Request,
):
    """Clone a voice from reference audio and generate speech. Serialized via _speech_sem."""
    async with _speech_sem:
        return await _create_voice_clone_impl(request, client_request)


async def _create_voice_clone_impl(
    request: VoiceCloneRequest,
    client_request: Request,
):
    """Clone a voice from reference audio and generate speech."""
    try:
        backend = await get_tts_backend()

        if not backend.supports_voice_cloning():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "voice_cloning_not_supported",
                    "message": "Voice cloning requires the Base model.",
                    "type": "invalid_request_error",
                },
            )

        if not request.x_vector_only_mode and not request.ref_text:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_ref_text",
                    "message": "ICL mode requires ref_text. Either provide ref_text or set x_vector_only_mode=True.",
                    "type": "invalid_request_error",
                },
            )

        try:
            audio_bytes = base64.b64decode(request.ref_audio)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_audio", "message": f"Failed to decode base64 audio: {e}", "type": "invalid_request_error"},
            )

        try:
            audio_buffer = io.BytesIO(audio_bytes)
            ref_audio, ref_sr = sf.read(audio_buffer)
            if len(ref_audio.shape) > 1:
                ref_audio = ref_audio.mean(axis=1)
            ref_audio = ref_audio.astype(np.float32)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "audio_processing_error", "message": f"Failed to process reference audio: {e}", "type": "invalid_request_error"},
            )

        normalized_text = normalize_text(request.input, request.normalization_options)
        if not normalized_text.strip():
            raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "Input text is empty after normalization", "type": "invalid_request_error"})

        audio, sample_rate = await backend.generate_voice_clone(
            text=normalized_text, ref_audio=ref_audio, ref_audio_sr=ref_sr,
            ref_text=request.ref_text, language=request.language or "Auto",
            x_vector_only_mode=request.x_vector_only_mode, speed=request.speed,
            model=request.model,
        )

        audio_bytes = encode_audio(audio, request.response_format, sample_rate)
        content_type = get_content_type(request.response_format)

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename=voice_clone.{request.response_format}", "Cache-Control": "no-cache"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail={"error": "processing_error", "message": str(e), "type": "server_error"})


# ---------------------------------------------------------------------------
# Voice profile CRUD (used by the web studio UI)
# ---------------------------------------------------------------------------

import shutil
import uuid
from datetime import datetime, timezone
from fastapi.responses import FileResponse

PROFILES_DIR = VOICE_LIBRARY_DIR / "profiles"

_PROFILE_EDITABLE_FIELDS = ("name", "language", "voice", "instructions", "ref_text", "x_vector_only_mode")


def _profile_meta_path(profile_id: str) -> Path:
    return PROFILES_DIR / profile_id / "meta.json"


def _find_profile_dir(name_or_id: str) -> Optional[Path]:
    """Locate a profile directory by profile_id or name (case-insensitive)."""
    if not PROFILES_DIR.exists():
        return None
    for child in PROFILES_DIR.iterdir():
        if not child.is_dir():
            continue
        meta_file = child / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("profile_id") == name_or_id or meta.get("name", "").lower() == name_or_id.lower():
            return child
    return None


def _profile_public(meta: dict, pid: str) -> dict:
    return {
        "profile_id": pid,
        "name": meta.get("name", pid),
        "task_type": meta.get("task_type", "Base"),
        "origin": meta.get("origin", ""),
        "created_at": meta.get("created_at", ""),
        "language": meta.get("language", "Auto"),
        "voice": meta.get("voice", ""),
        "instructions": meta.get("instructions", ""),
        "ref_text": meta.get("ref_text", ""),
        "x_vector_only_mode": bool(meta.get("x_vector_only_mode", False)),
        "has_ref_audio": bool(meta.get("ref_audio_filename")),
        "ref_audio_filename": meta.get("ref_audio_filename", ""),
        "image_url": meta.get("image_url", ""),
    }


def _decode_ref_audio(data_uri: str) -> bytes:
    """Decode a base64/data-URI audio payload to raw bytes."""
    payload = data_uri
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def _save_ref_audio(profile_dir: Path, audio_bytes: bytes, preferred_ext: str = ".wav") -> str:
    """Write ref audio bytes into the profile dir; returns filename."""
    ext = preferred_ext if preferred_ext.startswith(".") else f".{preferred_ext}"
    fname = f"ref{ext}"
    (profile_dir / fname).write_bytes(audio_bytes)
    return fname


@router.get("/voice-profiles")
async def list_voice_profiles():
    """List all saved voice profiles (metadata only, no audio)."""
    profiles = []
    if PROFILES_DIR.exists():
        for child in sorted(PROFILES_DIR.iterdir()):
            meta_file = child / "meta.json"
            if not child.is_dir() or not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                profiles.append(_profile_public(meta, child.name))
            except Exception:
                continue
    profiles.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return {"profiles": profiles}


@router.get("/voice-profiles/{profile_id}/ref-audio")
async def get_profile_ref_audio(profile_id: str):
    """Serve a profile's stored reference audio file."""
    pdir = _find_profile_dir(profile_id)
    if not pdir:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Profile not found"})
    meta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
    fname = meta.get("ref_audio_filename", "")
    fpath = pdir / fname if fname else None
    if not fpath or not fpath.exists():
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "No reference audio"})
    media_type = "audio/wav" if fpath.suffix.lower() in (".wav", ".wave") else "audio/mpeg"
    return FileResponse(str(fpath), media_type=media_type)


@router.post("/voice-profiles", status_code=201)
async def create_voice_profile(body: dict):
    """Create a voice profile. Body: name, task_type, language, voice,
    instructions, ref_text, x_vector_only_mode, ref_audio (data URI)."""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "name is required"})
    task_type = body.get("task_type") or "Base"
    ref_audio_uri = body.get("ref_audio") or ""
    if task_type == "Base" and not ref_audio_uri:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "ref_audio is required for Base profiles"})

    pid = uuid.uuid4().hex[:12]
    pdir = PROFILES_DIR / pid
    pdir.mkdir(parents=True, exist_ok=True)

    ref_filename = ""
    if ref_audio_uri:
        try:
            audio_bytes = _decode_ref_audio(ref_audio_uri)
            ext = ".wav"
            m = body.get("ref_audio_mime", "")
            if "mpeg" in m or "mp3" in m:
                ext = ".mp3"
            elif "ogg" in m or "opus" in m:
                ext = ".ogg"
            elif "webm" in m:
                ext = ".webm"
            ref_filename = _save_ref_audio(pdir, audio_bytes, ext)
        except Exception as e:
            shutil.rmtree(pdir, ignore_errors=True)
            raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": f"Bad ref_audio: {e}"})

    meta = {
        "profile_id": pid,
        "name": name,
        "task_type": task_type,
        "origin": body.get("origin") or "web-studio",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "language": body.get("language") or "Auto",
        "voice": body.get("voice") or "Vivian",
        "instructions": body.get("instructions") or "",
        "ref_text": body.get("ref_text") or "",
        "x_vector_only_mode": bool(body.get("x_vector_only_mode", False)),
        "ref_audio_filename": ref_filename,
    }
    (pdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _ref_audio_cache.pop(name, None)
    return {"profile": _profile_public(meta, pid)}


@router.put("/voice-profiles/{profile_id}")
async def update_voice_profile(profile_id: str, body: dict):
    """Update a voice profile's metadata; optionally replace ref audio."""
    pdir = _find_profile_dir(profile_id)
    if not pdir:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Profile not found"})
    meta_file = pdir / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    old_name = meta.get("name", "")

    for field in _PROFILE_EDITABLE_FIELDS:
        if field in body:
            meta[field] = body[field]
    if "name" in body and str(body["name"]).strip():
        meta["name"] = str(body["name"]).strip()

    ref_audio_uri = body.get("ref_audio") or ""
    if ref_audio_uri:
        try:
            audio_bytes = _decode_ref_audio(ref_audio_uri)
            m = body.get("ref_audio_mime", "")
            ext = ".mp3" if "mpeg" in m or "mp3" in m else (".ogg" if "ogg" in m or "opus" in m else (".webm" if "webm" in m else ".wav"))
            new_fname = _save_ref_audio(pdir, audio_bytes, ext)
            old_fname = meta.get("ref_audio_filename", "")
            if old_fname and old_fname != new_fname:
                old_path = pdir / old_fname
                if old_path.exists():
                    old_path.unlink()
            meta["ref_audio_filename"] = new_fname
        except Exception as e:
            raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": f"Bad ref_audio: {e}"})

    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _ref_audio_cache.pop(old_name, None)
    _ref_audio_cache.pop(meta.get("name", ""), None)
    return {"profile": _profile_public(meta, pdir.name)}


@router.delete("/voice-profiles/{profile_id}")
async def delete_voice_profile(profile_id: str):
    """Delete a voice profile and its files."""
    pdir = _find_profile_dir(profile_id)
    if not pdir:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Profile not found"})
    try:
        meta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
        _ref_audio_cache.pop(meta.get("name", ""), None)
    except Exception:
        pass
    shutil.rmtree(pdir)
    return {"deleted": True, "profile_id": pdir.name}


# ---------------------------------------------------------------------------
# Speech-to-text for clone reference transcription (faster-whisper, CPU)
# ---------------------------------------------------------------------------

_whisper_model = None
_whisper_lock = asyncio.Lock()


async def _get_whisper_model():
    """Lazy-load the faster-whisper model (CPU, int8)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    async with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail={"error": "unavailable", "message": "faster-whisper is not installed in this container"},
            )
        model_size = os.environ.get("TRANSCRIBE_MODEL", "small.en")
        logger.info(f"Loading whisper model '{model_size}' for transcription")
        _whisper_model = await asyncio.to_thread(
            WhisperModel, model_size, device="cpu", compute_type="int8"
        )
        return _whisper_model


@router.post("/transcribe")
async def transcribe_audio(body: dict):
    """Transcribe reference audio to text. Body: {audio: data-URI or base64}.

    Returns {text, segments}. Audio should be WAV (the studio UI resamples
    uploads/recordings to 24 kHz mono WAV before sending).
    """
    audio_uri = body.get("audio") or body.get("ref_audio") or ""
    if not audio_uri:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "audio is required"})
    try:
        audio_bytes = _decode_ref_audio(audio_uri)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": f"Bad audio payload: {e}"})

    import tempfile
    suffix = ".wav"
    mime = body.get("audio_mime", "")
    if "mpeg" in mime or "mp3" in mime:
        suffix = ".mp3"
    elif "ogg" in mime or "opus" in mime:
        suffix = ".ogg"
    elif "webm" in mime:
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(audio_bytes)
        tmp_path = tf.name
    try:
        model = await _get_whisper_model()
        import subprocess
        import numpy as np
        pcm = await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-v", "error", "-i", tmp_path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
            capture_output=True, check=True,
        )
        audio = np.frombuffer(pcm.stdout, dtype=np.float32)
        if audio.size == 0:
            raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "No decodable audio"})
        segments, info = await asyncio.to_thread(
            model.transcribe, audio, beam_size=5, vad_filter=True
        )
        segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
        text = " ".join(s["text"] for s in segs).strip()
        return {"text": text, "segments": segs, "language": info.language}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail={"error": "transcription_error", "message": str(e)})
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.post("/style-suggest")
async def style_suggest(body: dict):
    """Suggest a TTS style instruction from a vibe description via Lemonade LLM.

    Body: {vibe: str, text: str?} -> {instruct: str}
    """
    vibe = (body.get("vibe") or "").strip()
    sample = (body.get("text") or "").strip()
    if not vibe and not sample:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "vibe or text is required"})
    lemonade_url = os.environ.get("LEMONADE_URL", "").rstrip("/")
    if not lemonade_url:
        raise HTTPException(status_code=503, detail={"error": "not_configured", "message": "LEMONADE_URL is not set"})
    model = os.environ.get("LEMONADE_MODEL", "")
    prompt = vibe or f"Text to be spoken: {sample[:500]}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You write style prompts for a TTS engine. Given a vibe or sample text, output ONLY the style instruction phrase (under 12 words) describing HOW it should be spoken - tone, pace, emotion, delivery. No quotes, no explanation.\nExamples:\nVibe: announcing a product launch -> Excited, energetic, confident product presenter\nVibe: sleepy podcast host at 2am -> Slow, drowsy, intimate late-night radio voice\nVibe: horror story opening -> Low, ominous, slow with mounting dread\nVibe: grumpy old man complaining about modern technology -> Gruff, irritated, slow and dismissive\nVibe: customer support agent -> Warm, patient, reassuring and professional"},
            {"role": "user", "content": f"Vibe: {prompt}"},
        ],
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    def _call():
        import urllib.request
        req = urllib.request.Request(
            f"{lemonade_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req, timeout=60).read())
    try:
        resp = await asyncio.to_thread(_call)
        instruct = (resp["choices"][0]["message"].get("content") or "").strip().strip("\"'")
        if not instruct:
            raise HTTPException(status_code=502, detail={"error": "llm_error", "message": "Empty style suggestion"})
        return {"instruct": instruct}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Style suggest failed: {e}")
        raise HTTPException(status_code=502, detail={"error": "llm_error", "message": str(e)})


# ---------------------------------------------------------------------------
# Saved audio clips library
# ---------------------------------------------------------------------------

CLIPS_DIR = VOICE_LIBRARY_DIR / "clips"


def _clip_public(meta: dict, cid: str) -> dict:
    return {
        "clip_id": cid,
        "name": meta.get("name", cid),
        "created_at": meta.get("created_at", ""),
        "text": meta.get("text", ""),
        "voice": meta.get("voice", ""),
        "model": meta.get("model", ""),
        "format": meta.get("format", ""),
        "audio_filename": meta.get("audio_filename", ""),
        "tags": meta.get("tags", []),
    }


def _norm_tags(v) -> list:
    if isinstance(v, str):
        v = v.split(",")
    if not isinstance(v, list):
        return []
    out = []
    for t in v:
        t = str(t).strip().lower()
        if t and t not in out:
            out.append(t)
    return out[:20]


def _valid_clip_id(clip_id: str) -> bool:
    return bool(clip_id) and clip_id.replace("-", "").isalnum()


@router.get("/clips")
async def list_clips():
    clips = []
    if CLIPS_DIR.exists():
        for child in sorted(CLIPS_DIR.iterdir()):
            if not child.is_dir():
                continue
            meta_file = child / "meta.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            clips.append(_clip_public(meta, child.name))
    clips.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return {"clips": clips}


@router.post("/clips", status_code=201)
async def create_clip(body: dict):
    name = (body.get("name") or "").strip()
    audio_uri = body.get("audio") or ""
    if not name:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "name is required"})
    if not audio_uri:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "audio is required"})
    try:
        audio_bytes = _decode_ref_audio(audio_uri)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": f"Bad audio payload: {e}"})
    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "Audio payload too small"})
    if len(audio_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "Audio payload too large (50MB max)"})

    cid = uuid.uuid4().hex[:12]
    cdir = CLIPS_DIR / cid
    cdir.mkdir(parents=True, exist_ok=True)

    fmt = (body.get("format") or "wav").lower().lstrip(".")
    if fmt not in ("wav", "mp3", "flac", "ogg", "opus", "pcm"):
        fmt = "wav"
    audio_filename = f"audio.{fmt}"
    (cdir / audio_filename).write_bytes(audio_bytes)

    meta = {
        "clip_id": cid,
        "name": name,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "text": body.get("text") or "",
        "voice": body.get("voice") or "",
        "model": body.get("model") or "",
        "format": fmt,
        "audio_filename": audio_filename,
        "tags": _norm_tags(body.get("tags")),
    }
    (cdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"clip": _clip_public(meta, cid)}


@router.get("/clips/{clip_id}/audio")
async def get_clip_audio(clip_id: str):
    if not _valid_clip_id(clip_id):
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip not found"})
    cdir = CLIPS_DIR / clip_id
    meta_file = cdir / "meta.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip not found"})
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip not found"})
    fpath = cdir / (meta.get("audio_filename") or "")
    if not fpath.exists() or fpath.parent != cdir:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip audio missing"})
    media = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".opus": "audio/ogg", ".pcm": "audio/L16",
    }.get(fpath.suffix.lower(), "application/octet-stream")
    return FileResponse(str(fpath), media_type=media, filename=f"{meta.get('name', clip_id)}{fpath.suffix}")


@router.put("/clips/{clip_id}")
async def update_clip(clip_id: str, body: dict):
    """Rename a clip (and optionally patch text/voice/model metadata)."""
    cdir, meta = _load_clip(clip_id)
    name = (body.get("name") or "").strip()
    if name:
        meta["name"] = name
    for k in ("text", "voice", "model"):
        if k in body:
            meta[k] = body[k] or ""
    if "tags" in body:
        meta["tags"] = _norm_tags(body["tags"])
    (cdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"clip": _clip_public(meta, clip_id)}


@router.delete("/clips/{clip_id}")
async def delete_clip(clip_id: str):
    if not _valid_clip_id(clip_id):
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip not found"})
    cdir = CLIPS_DIR / clip_id
    if not cdir.exists() or not (cdir / "meta.json").exists():
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip not found"})
    shutil.rmtree(cdir, ignore_errors=True)
    return {"deleted": clip_id}


# ---------------------------------------------------------------------------
# Speaker playback via Home Assistant (Sonos etc.)
# ---------------------------------------------------------------------------

import httpx

HA_URL = os.environ.get("TTS_HA_URL", "").rstrip("/")
HA_TOKEN = os.environ.get("TTS_HA_TOKEN", "")
# Base URL speakers use to fetch clip audio — must be LAN-reachable, no auth
CLIP_PUBLIC_BASE = os.environ.get("TTS_PUBLIC_URL", "").rstrip("/")


def _load_clip(clip_id: str):
    """Return (cdir, meta) for a valid clip, or raise 404."""
    if not _valid_clip_id(clip_id):
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip not found"})
    cdir = CLIPS_DIR / clip_id
    meta_file = cdir / "meta.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip not found"})
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip not found"})
    return cdir, meta


@router.get("/speakers")
async def list_speakers():
    """List HA media_player entities that can play clips (Sonos etc.)."""
    if not HA_URL or not HA_TOKEN:
        return {"speakers": [], "configured": False}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{HA_URL}/api/states",
                            headers={"Authorization": f"Bearer {HA_TOKEN}"})
        if r.status_code != 200:
            return {"speakers": [], "configured": True,
                    "error": f"HA returned {r.status_code}"}
        states = r.json()
    except Exception as e:
        return {"speakers": [], "configured": True, "error": str(e)}
    # Sonos speakers are the media_players exposing a `group_members` attribute
    # (sonos integration grouping). Filters out sonos_cloud dupes, Alexa devices,
    # Roku/AppleTV/Chromecast entities — REST /api/states has no platform field.
    speakers = [
        {
            "entity_id": s["entity_id"],
            "name": s.get("attributes", {}).get("friendly_name", s["entity_id"]),
            "state": s.get("state", "unknown"),
        }
        for s in states
        if s.get("entity_id", "").startswith("media_player.")
        and "group_members" in s.get("attributes", {})
    ]
    speakers.sort(key=lambda x: x["name"].lower())
    return {"speakers": speakers, "configured": True}


@router.get("/clips/{clip_id}/stream.mp3")
async def get_clip_mp3(clip_id: str):
    """MP3-transcoded clip audio for Sonos/HA playback. Cached beside the clip.
    The .mp3 suffix matters — Sonos rejects URLs it can't type by extension."""
    cdir, meta = _load_clip(clip_id)
    src = cdir / (meta.get("audio_filename") or "")
    if not src.exists() or src.parent != cdir:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip audio missing"})
    mp3_path = cdir / "play.mp3"
    if not mp3_path.exists():
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(src), "-codec:a", "libmp3lame", "-q:a", "4",
            str(mp3_path),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        if proc.returncode != 0 or not mp3_path.exists():
            raise HTTPException(status_code=500, detail={"error": "transcode_failed", "message": "ffmpeg mp3 transcode failed"})
    return FileResponse(str(mp3_path), media_type="audio/mpeg",
                        filename=f"{meta.get('name', clip_id)}.mp3")


@router.post("/clips/{clip_id}/play")
async def play_clip(clip_id: str, body: dict):
    """Play a saved clip on HA media players (Sonos etc.).

    Body: {"entity_id": "media_player.x"} or {"entity_ids": [...]}.
    Speakers fetch the mp3 themselves — CLIP_PUBLIC_BASE must be LAN-reachable.
    """
    cdir, meta = _load_clip(clip_id)
    src = cdir / (meta.get("audio_filename") or "")
    if not src.exists():
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Clip audio missing"})

    entity_ids = body.get("entity_ids") or []
    if body.get("entity_id"):
        entity_ids = [body["entity_id"]] + [e for e in entity_ids if e != body["entity_id"]]
    entity_ids = [str(e) for e in entity_ids if str(e).strip()][:10]
    if not entity_ids:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "entity_id or entity_ids required"})
    if not HA_URL or not HA_TOKEN:
        raise HTTPException(status_code=503, detail={"error": "not_configured", "message": "Home Assistant is not configured (TTS_HA_URL/TTS_HA_TOKEN)"})
    if not CLIP_PUBLIC_BASE:
        raise HTTPException(status_code=503, detail={"error": "not_configured", "message": "TTS_PUBLIC_URL is not set — speakers need a LAN-reachable base URL for clip audio"})

    media_url = f"{CLIP_PUBLIC_BASE}/v1/clips/{clip_id}/stream.mp3"
    headers = {"Authorization": f"Bearer {HA_TOKEN}"}
    results = []
    async with httpx.AsyncClient(timeout=15) as c:
        for eid in entity_ids:
            try:
                r = await c.post(
                    f"{HA_URL}/api/services/media_player/play_media",
                    headers={**headers, "Content-Type": "application/json"},
                    json={"entity_id": eid, "media_content_id": media_url,
                          "media_content_type": "music"})
                if r.status_code == 200:
                    results.append({"entity_id": eid, "status": "ok"})
                    logger.info(f"Clip '{meta.get('name', clip_id)}' -> {eid}")
                else:
                    results.append({"entity_id": eid, "status": "error",
                                    "error": f"HA returned {r.status_code}: {r.text[:200]}"})
            except Exception as e:
                results.append({"entity_id": eid, "status": "error", "error": str(e)})
    all_ok = all(r["status"] == "ok" for r in results)
    return {"clip_id": clip_id, "media_url": media_url, "played_on": results,
            "status": "ok" if all_ok else "partial"}
