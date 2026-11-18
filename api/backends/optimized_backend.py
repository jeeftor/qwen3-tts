# coding=utf-8
"""
Optimized Qwen3-TTS backend with model switching based on request.
"""

import asyncio
import concurrent.futures
import logging
import os
import time
import yaml
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / "qwen3-tts" / "config.yaml"

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}

# Model mapping: OpenAI model name -> our model key
MODEL_MAPPING = {
    "tts-1": "0.6B-CustomVoice",
    "tts-1-hd": "1.7B-CustomVoice",
    "qwen3-tts": "0.6B-CustomVoice",
    "qwen3-tts-hd": "1.7B-CustomVoice",
}

class OptimizedQwen3TTSBackend:
    """Optimized backend with dynamic model switching."""

    def __init__(self):
        self.config = load_config()
        self.models = {}  # Cache loaded models
        self.current_model_key = None
        self.model = None
        self._ready = False
        self.device = None
        self.dtype = None
        self._voice_prompt_cache = {}  # cache_key -> VoiceClonePromptItem list
        self._last_gpu_activity = 0.0  # stamped on every GPU-thread task
        # Single worker thread: every blocking GPU call (model load, prompt
        # build, generate, stream next()) runs here, so the event loop stays
        # responsive and GPU access stays serialized.
        self._gpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts-gpu"
        )

    _DONE = object()

    async def _bridge_sync_gen(self, gen):
        """Yield items from a sync generator, running each next() on the GPU thread."""
        loop = asyncio.get_running_loop()
        it = iter(gen)
        try:
            while True:
                item = await loop.run_in_executor(self._gpu_executor, next, it, self._DONE)
                if item is self._DONE:
                    return
                yield item
        finally:
            # Best-effort: close the sync generator on the GPU thread so any
            # CUDA refs it holds are released even on client disconnect.
            closer = getattr(it, "close", None)
            if closer is not None:
                try:
                    await loop.run_in_executor(self._gpu_executor, closer)
                except BaseException:
                    pass

    def _get_model_key_for_request(self, openai_model: str) -> str:
        """Map OpenAI model name to our model key."""
        if openai_model in self.config.get("models", {}):
            return openai_model
        if openai_model in MODEL_MAPPING:
            return MODEL_MAPPING[openai_model]
        base_model = openai_model.rsplit('-', 1)[0] if '-' in openai_model else openai_model
        if base_model in MODEL_MAPPING:
            return MODEL_MAPPING[base_model]
        return self.config.get("default_model", "0.6B-CustomVoice")

    def get_available_models(self) -> List[str]:
        return list(self.config.get("models", {}).keys())

    def get_model_info(self, model_key: str) -> dict:
        return self.config.get("models", {}).get(model_key, {})

    def get_current_model_key(self) -> Optional[str]:
        return self.current_model_key

    async def _ensure_model_loaded(self, model_key: str) -> None:
        """Load model if not already loaded (runs on the GPU thread)."""
        if self.current_model_key == model_key and self.model is not None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._gpu_executor, self._load_model_sync, model_key)

    def _unload_model_sync(self) -> None:
        """Free the loaded model + prompt cache. Runs on the GPU thread."""
        import torch
        if self.model is None:
            return
        logger.info(f"Unloading {self.current_model_key}...")
        if self._voice_prompt_cache:
            logger.info(f"Clearing voice prompt cache ({len(self._voice_prompt_cache)} entries)")
            self._voice_prompt_cache.clear()
        del self.model
        self.model = None
        self.current_model_key = None
        torch.cuda.empty_cache()

    def touch_gpu_activity(self) -> None:
        self._last_gpu_activity = time.time()

    async def unload_if_idle(self, idle_seconds: float) -> bool:
        """Unload the model if no GPU work ran within idle_seconds.

        The check runs on the event loop; the unload itself is queued on the
        single GPU thread, so it can never interleave with an in-flight
        generate (each worker is one atomic executor task). Callers should
        additionally verify the request queue is empty before calling.
        """
        if self.model is None:
            return False
        if time.time() - self._last_gpu_activity < idle_seconds:
            return False
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._gpu_executor, self._unload_model_sync)
        return True

    def _load_model_sync(self, model_key: str) -> None:
        """Blocking model load/switch — always runs on the single GPU thread."""
        import torch

        self._last_gpu_activity = time.time()
        if self.current_model_key == model_key and self.model is not None:
            return

        model_info = self.config.get("models", {}).get(model_key, {})
        if not model_info:
            raise ValueError(f"Unknown model: {model_key}")

        hf_id = model_info["hf_id"]

        # Unload previous model
        self._unload_model_sync()

        logger.info(f"Loading {model_key} ({hf_id})...")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        from qwen_tts import Qwen3TTSModel
        torch.set_float32_matmul_precision('high')

        opt = self.config.get("optimization", {})
        attn_impl = opt.get("attention", "flash_attention_2")

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                hf_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_impl,
            )
            logger.info(f"Loaded with {attn_impl}")
        except Exception as e:
            logger.warning(f"Failed with {attn_impl}: {e}, trying sdpa")
            self.model = Qwen3TTSModel.from_pretrained(
                hf_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation="sdpa",
            )

        # Streaming optimizations — fast codebook + decode-window config apply
        # with or without torch.compile. Compile is impractical on ROCm
        # (warmup takes tens of minutes), so use_compile=false keeps the cheap wins.
        use_compile = bool(opt.get("use_compile", True)) and self.device != "cpu"
        try:
            streaming_opts = opt.get("streaming", {})
            self.model.enable_streaming_optimizations(
                decode_window_frames=streaming_opts.get("decode_window_frames", 80),
                use_compile=use_compile,
                use_cuda_graphs=opt.get("use_cuda_graphs", False) and use_compile,
                compile_mode=opt.get("compile_mode", "max-autotune"),
                use_fast_codebook=opt.get("use_fast_codebook", True),
                compile_codebook_predictor=opt.get("compile_codebook_predictor", True),
            )
            logger.info(f"Streaming optimizations: compile={use_compile}, mode={opt.get('compile_mode', 'max-autotune')}, fast_codebook={opt.get('use_fast_codebook', True)}")
        except Exception as e:
            logger.warning(f"Could not enable streaming optimizations: {e}")
            use_compile = False

        # torch.compile warmup — only when compilation is enabled; it exists
        # to trigger lazy kernel compilation (very slow on ROCm).
        if use_compile:
            try:
                # Mandatory warmup — dffdeeq examples ALWAYS run a generation
                # right after enable_streaming_optimizations() to trigger
                # actual kernel compilation (torch.compile is lazy).
                model_type = model_info.get("type", "customvoice")

                emit_every = streaming_opts.get("emit_every_frames", 4)
                decode_window = streaming_opts.get("decode_window_frames", 80)

                if model_type == "base":
                    # Base model warmup: must cover BOTH x_vector_only AND ICL code paths
                    # (ICL uses ref_code_context in stream_generate_pcm — different compilation)
                    dummy_audio = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)

                    # 1) x_vector_only path (non-streaming + streaming)
                    logger.info("Warmup 1/3: x_vector_only non-streaming...")
                    self.model.generate_voice_clone(
                        text="Warmup sentence for compilation.",
                        language="English",
                        ref_audio=(dummy_audio, 24000),
                        x_vector_only_mode=True,
                    )
                    logger.info("Warmup 1/3: x_vector_only streaming...")
                    for chunk, sr_out in self.model.stream_generate_voice_clone(
                        text="Streaming warmup for voice clone.",
                        language="English",
                        ref_audio=(dummy_audio, 24000),
                        x_vector_only_mode=True,
                        emit_every_frames=emit_every,
                        decode_window_frames=decode_window,
                    ):
                        pass

                    # 2) ICL path — same path as real clone:profile requests
                    logger.info("Warmup 2/3: ICL mode streaming (matches real usage)...")
                    for chunk, sr_out in self.model.stream_generate_voice_clone(
                        text="Second warmup with ICL mode to compile ref_code context path.",
                        language="English",
                        ref_audio=(dummy_audio, 24000),
                        ref_text="Warmup reference text.",
                        x_vector_only_mode=False,
                        emit_every_frames=emit_every,
                        decode_window_frames=decode_window,
                    ):
                        pass

                    # 3) Additional run to stabilize GPU power state (dffdeeq pattern)
                    logger.info("Warmup 3/3: GPU stabilization run...")
                    for chunk, sr_out in self.model.stream_generate_voice_clone(
                        text="Ein dritter Aufwärmdurchlauf um die GPU voll aufzuwärmen.",
                        language="German",
                        ref_audio=(dummy_audio, 24000),
                        x_vector_only_mode=True,
                        emit_every_frames=emit_every,
                        decode_window_frames=decode_window,
                    ):
                        pass
                    logger.info("All warmup runs complete (base)")
                else:
                    # CustomVoice model: 3 warmup runs with different text lengths
                    logger.info("Warmup 1/3: non-streaming...")
                    self.model.generate_custom_voice(
                        text="This is a warmup sentence to trigger torch compile kernel compilation.",
                        language="English",
                        speaker="Eric",
                    )
                    logger.info("Warmup 1/3: streaming...")
                    for chunk, sr_out in self.model.stream_generate_custom_voice(
                        text="Streaming warmup sentence for compiling the streaming code path.",
                        speaker="Eric",
                        language="English",
                        emit_every_frames=emit_every,
                        decode_window_frames=decode_window,
                    ):
                        pass

                    # Additional runs to stabilize GPU power state (dffdeeq pattern)
                    logger.info("Warmup 2/3: medium text...")
                    for chunk, sr_out in self.model.stream_generate_custom_voice(
                        text="Ein mittellanger Satz um verschiedene Tensor-Shapes aufzuwärmen.",
                        speaker="Eric",
                        language="German",
                        emit_every_frames=emit_every,
                        decode_window_frames=decode_window,
                    ):
                        pass

                    logger.info("Warmup 3/3: short text...")
                    for chunk, sr_out in self.model.stream_generate_custom_voice(
                        text="Short test.",
                        speaker="Eric",
                        language="English",
                        emit_every_frames=emit_every,
                        decode_window_frames=decode_window,
                    ):
                        pass
                    logger.info("All warmup runs complete (custom voice)")
            except Exception as e:
                logger.warning(f"Could not enable optimizations: {e}")

        self.current_model_key = model_key
        self._ready = True
        logger.info(f"Model {model_key} ready on {self.device}")

    async def initialize(self, model_key: Optional[str] = None) -> None:
        """Initialize with default or specified model."""
        if model_key is None:
            model_key = self.config.get("default_model", "0.6B-CustomVoice")
        await self._ensure_model_loaded(model_key)

    async def switch_model(self, model_key: str) -> None:
        """Switch to a different model."""
        await self._ensure_model_loaded(model_key)

    def _customvoice_model_key(self, model: str) -> str:
        """Resolve the CustomVoice model key for a request (with fallback)."""
        if not any(m.get("type") == "customvoice"
                   for m in self.config.get("models", {}).values()):
            raise ValueError(
                "preset voices unavailable — no CustomVoice model configured; "
                "use clone:<profile> instead")
        model_key = self._get_model_key_for_request(model)
        if self.config.get("models", {}).get(model_key, {}).get("type") != "customvoice":
            model_key = MODEL_MAPPING.get(model, "0.6B-CustomVoice")
        return model_key

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
        model: str = "tts-1",
    ) -> Tuple[np.ndarray, int]:
        """Generate speech, switching model if needed."""
        model_key = self._customvoice_model_key(model)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._gpu_executor, self._speech_sync, model_key, text, voice, language, instruct, speed
        )

    def _speech_sync(self, model_key, text, voice, language, instruct, speed):
        # Ensure + generate in one GPU-thread task: self.model can't be
        # swapped out from under us by another request's load.
        self._load_model_sync(model_key)
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=voice,
            instruct=instruct,
        )

        audio = wavs[0]
        if speed != 1.0:
            try:
                import librosa
                audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
            except ImportError:
                pass
        return audio, sr


    async def generate_speech_streaming(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: str = None,
        speed: float = 1.0,
        model: str = "tts-1",
    ):
        """Real token-by-token streaming via dffdeeq stream_generate_custom_voice.

        Yields (pcm_chunk, sample_rate) tuples as audio is generated.
        Uses dffdeeq's stream_generate_pcm under the hood.
        """
        model_key = self._customvoice_model_key(model)

        streaming_opts = self.config.get("optimization", {}).get("streaming", {})
        decode_window_frames = streaming_opts.get("decode_window_frames", 80)
        emit_every_frames = streaming_opts.get("emit_every_frames", 4)

        # Voice names arrive already resolved — the router's get_voice_name
        # owns OpenAI-alias mapping (VOICE_MAPPING); don't remap here.
        loop = asyncio.get_running_loop()
        gen = await loop.run_in_executor(
            self._gpu_executor, self._speech_stream_setup_sync,
            model_key, text, voice, language, instruct,
            decode_window_frames, emit_every_frames,
        )
        async for item in self._bridge_sync_gen(gen):
            yield item

    def _speech_stream_setup_sync(self, model_key, text, voice_name, language,
                                  instruct, decode_window_frames, emit_every_frames):
        """Ensure model + create the sync stream generator atomically on the
        GPU thread, so self.model can't be swapped between check and bind."""
        self._load_model_sync(model_key)
        return self.model.stream_generate_custom_voice(
            text=text,
            speaker=voice_name,
            language=language,
            instruct=instruct,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
        )


    def _clone_model_key(self, model: Optional[str]) -> str:
        """Resolve clone checkpoint: explicit Base-type config key wins, else clone_model."""
        models_cfg = self.config.get("models", {})
        if model and models_cfg.get(model, {}).get("type") == "base":
            return model
        return self.config.get("clone_model", "1.7B-Base")

    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
        cache_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Voice cloning requires Base model."""
        model_key = self._clone_model_key(model)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._gpu_executor,
            self._voice_clone_sync,
            model_key,
            text, ref_audio, ref_audio_sr, ref_text, language,
            x_vector_only_mode, speed, cache_key,
        )

    def _voice_clone_sync(self, model_key, text, ref_audio, ref_audio_sr, ref_text,
                          language, x_vector_only_mode, speed, cache_key):
        """Blocking clone generation — runs on the single GPU thread."""
        import time as _time

        # Ensure + generate in one GPU-thread task: self.model can't be
        # swapped out from under us by another request's load.
        self._load_model_sync(model_key)

        # Use cached voice prompt if available, otherwise compute and cache
        if cache_key and cache_key in self._voice_prompt_cache:
            prompt_items = self._voice_prompt_cache[cache_key]
            t0 = _time.time()
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
            )
            logger.info(f"Voice clone (cached): generate={_time.time()-t0:.3f}s")
        else:
            if cache_key:
                t0 = _time.time()
                prompt_items = self.model.create_voice_clone_prompt(
                    ref_audio=(ref_audio, ref_audio_sr),
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                )
                t_prompt = _time.time() - t0
                self._voice_prompt_cache[cache_key] = prompt_items
                logger.info(f"Voice prompt cached: '{cache_key}' (prompt_build={t_prompt:.3f}s)")
                t0 = _time.time()
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt_items,
                )
                logger.info(f"Voice clone (first): prompt_build={t_prompt:.3f}s generate={_time.time()-t0:.3f}s")
            else:
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    ref_audio=(ref_audio, ref_audio_sr),
                    ref_text=ref_text,
                    language=language,
                    x_vector_only_mode=x_vector_only_mode,
                )

        audio = wavs[0]
        if speed != 1.0:
            try:
                import librosa
                audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
            except ImportError:
                pass
        return audio, sr

    def _clone_prompt_sync(self, ref_audio, ref_audio_sr, ref_text,
                           x_vector_only_mode, cache_key):
        """Build (or fetch cached) voice clone prompt — GPU thread only."""
        import time as _time
        if cache_key and cache_key in self._voice_prompt_cache:
            return self._voice_prompt_cache[cache_key]
        t0 = _time.time()
        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=(ref_audio, ref_audio_sr),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        t_prompt = _time.time() - t0
        if cache_key:
            self._voice_prompt_cache[cache_key] = prompt_items
            logger.info(f"Voice prompt cached: '{cache_key}' (prompt_build={t_prompt:.3f}s)")
        else:
            logger.info(f"Voice prompt built (no cache): {t_prompt:.3f}s")
        return prompt_items

    async def generate_voice_clone_streaming(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        cache_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Streaming voice cloning via dffdeeq stream_generate_voice_clone.

        Yields (pcm_chunk, sample_rate) tuples as audio is generated.
        """
        model_key = self._clone_model_key(model)

        loop = asyncio.get_running_loop()
        streaming_opts = self.config.get("optimization", {}).get("streaming", {})
        decode_window_frames = streaming_opts.get("decode_window_frames", 80)
        emit_every_frames = streaming_opts.get("emit_every_frames", 4)

        gen = await loop.run_in_executor(
            self._gpu_executor,
            self._clone_stream_setup_sync,
            model_key,
            text, ref_audio, ref_audio_sr, ref_text, language,
            x_vector_only_mode, cache_key,
            decode_window_frames, emit_every_frames,
        )
        async for item in self._bridge_sync_gen(gen):
            yield item

    def _clone_stream_setup_sync(self, model_key, text, ref_audio, ref_audio_sr,
                                 ref_text, language, x_vector_only_mode,
                                 cache_key, decode_window_frames, emit_every_frames):
        """Ensure model + build/fetch voice prompt + create the sync stream
        generator atomically on the GPU thread, so self.model can't be
        swapped between check and bind."""
        self._load_model_sync(model_key)
        prompt_items = self._clone_prompt_sync(
            ref_audio, ref_audio_sr, ref_text, x_vector_only_mode, cache_key)
        return self.model.stream_generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt_items,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
        )

    def get_backend_name(self) -> str:
        return "optimized"

    def get_model_id(self) -> str:
        if self.current_model_key:
            info = self.config.get("models", {}).get(self.current_model_key, {})
            return info.get("hf_id", "unknown")
        return "not-loaded"

    def get_supported_voices(self) -> List[str]:
        return [v["name"] for v in self.config.get("voices", [])]

    def get_supported_languages(self) -> List[str]:
        return ["English", "Chinese", "Japanese", "Korean", "German",
                "French", "Spanish", "Russian", "Portuguese", "Italian"]

    def is_ready(self) -> bool:
        return self._ready

    def supports_voice_cloning(self) -> bool:
        return True

    def get_model_type(self) -> str:
        if not self.current_model_key:
            return "unknown"
        return self.config.get("models", {}).get(self.current_model_key, {}).get("type", "unknown")

    def get_device_info(self) -> Dict[str, Any]:
        import torch
        info = {"device": str(self.device) if self.device else "unknown", "gpu_available": False}
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_total"] = f"{props.total_memory / 1024**3:.1f} GB"
            info["vram_used"] = f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
            info["vram_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
        return info
