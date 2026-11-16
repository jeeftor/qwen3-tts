# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Audio encoding service for TTS API.
Handles conversion of raw audio to various formats (mp3, opus, aac, flac, wav, pcm).
"""

import io
import logging
import math
import os
import struct
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

# Default sample rate for Qwen3-TTS output
DEFAULT_SAMPLE_RATE = 24000

# --- Loudness normalization -------------------------------------------------
# Model output level varies noticeably between voices/models/profiles
# (~8 dB RMS spread measured across preset + clone voices). Normalize
# toward a consistent target so generated clips don't jump in volume.
# Disable with TTS_LOUDNESS_NORM=0; retune with TTS_LOUDNESS_TARGET_DBFS.
_TARGET_DBFS = float(os.environ.get("TTS_LOUDNESS_TARGET_DBFS", "-20"))
_ENABLED = os.environ.get("TTS_LOUDNESS_NORM", "1").strip().lower() not in ("0", "false", "off", "no")
_SILENCE_RMS = 10 ** (-45 / 20)  # per-frame RMS floor; below this = silence
_PEAK_CEILING = 0.95
_MAX_GAIN = 10 ** (15 / 20)      # +15 dB boost ceiling
_MIN_GAIN = 0.25                 # -12 dB attenuation floor


def _apply_gain(x: np.ndarray, gain: float) -> np.ndarray:
    """Scale by gain with a 0.95 peak ceiling."""
    out = x * gain
    if out.size:
        peak = float(np.max(np.abs(out)))
        if peak > _PEAK_CEILING:
            out = out * (_PEAK_CEILING / peak)
    return out


class LoudnessNormalizer:
    """Streaming loudness normalization (AGC).

    An exponential moving average of per-chunk power steers gain toward
    a target RMS. Chunks below the silence floor pass through raw and
    don't perturb the estimate, so the gain converges on the utterance
    level within ~a second and holds — preserving natural dynamics
    while evening out between-generation level differences. Feed it
    whole buffers or small stream chunks; behavior is equivalent.
    """

    def __init__(self, target_dbfs: Optional[float] = None, alpha: float = 0.25):
        self.target_rms = 10 ** ((target_dbfs if target_dbfs is not None else _TARGET_DBFS) / 20)
        self.alpha = alpha
        self._avg_pwr: Optional[float] = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if not _ENABLED or chunk is None or getattr(chunk, "size", 0) == 0:
            return chunk
        x = np.asarray(chunk, dtype=np.float32)
        pwr = float(np.mean(x.astype(np.float64) ** 2))
        if math.sqrt(pwr) < _SILENCE_RMS:
            return x
        if self._avg_pwr is None:
            self._avg_pwr = pwr
        else:
            self._avg_pwr += self.alpha * (pwr - self._avg_pwr)
        gain = min(max(self.target_rms / math.sqrt(self._avg_pwr), _MIN_GAIN), _MAX_GAIN)
        return _apply_gain(x, gain)


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    target_dbfs: Optional[float] = None,
) -> np.ndarray:
    """Whole-utterance loudness normalization for non-streaming paths.

    Frames the buffer and averages power over voiced frames only, so
    leading/trailing silence doesn't skew the estimate.
    """
    if not _ENABLED or audio is None or getattr(audio, "size", 0) == 0:
        return audio
    x = np.asarray(audio, dtype=np.float32)
    frame = max(1, int(0.05 * sample_rate))
    n_frames = x.size // frame
    if n_frames == 0:
        return LoudnessNormalizer(target_dbfs).process(x)
    frames = x[: n_frames * frame].reshape(n_frames, frame)
    pwrs = np.mean(frames.astype(np.float64) ** 2, axis=1)
    voiced = pwrs[pwrs > _SILENCE_RMS ** 2]
    if voiced.size == 0:
        return x
    speech_rms = math.sqrt(float(np.mean(voiced)))
    target = 10 ** ((target_dbfs if target_dbfs is not None else _TARGET_DBFS) / 20)
    gain = min(max(target / speech_rms, _MIN_GAIN), _MAX_GAIN)
    return _apply_gain(x, gain)


def get_content_type(audio_format: AudioFormat) -> str:
    """Get MIME content type for audio format."""
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    return content_types.get(audio_format, f"audio/{audio_format}")


def _float_to_int16(audio: np.ndarray) -> np.ndarray:
    """float audio -> int16 PCM, peak-normalizing anything above full scale."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.size:
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
    return (audio * 32767).astype(np.int16)


def convert_to_wav(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Convert numpy audio array to WAV format bytes.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample (8 or 16)
    
    Returns:
        WAV file bytes
    """
    # Convert to int16
    audio_int16 = _float_to_int16(audio)

    # Create WAV header
    num_samples = len(audio_int16)
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = num_samples * bytes_per_sample
    
    buffer = io.BytesIO()
    
    # RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))  # File size - 8
    buffer.write(b'WAVE')
    
    # Format chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Chunk size
    buffer.write(struct.pack('<H', 1))  # Audio format (PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    
    # Data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))
    buffer.write(audio_int16.tobytes())
    
    return buffer.getvalue()


def convert_to_pcm(
    audio: np.ndarray,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Convert numpy audio array to raw PCM bytes.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        bits_per_sample: Bits per sample (8 or 16)
    
    Returns:
        Raw PCM bytes
    """
    # Convert to int16
    audio_int16 = _float_to_int16(audio)

    return audio_int16.tobytes()


def encode_audio(
    audio: np.ndarray,
    format: AudioFormat = "mp3",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """
    Encode audio to the specified format.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        format: Target audio format
        sample_rate: Sample rate in Hz
    
    Returns:
        Encoded audio bytes
    """
    if format == "wav":
        return convert_to_wav(audio, sample_rate)
    
    if format == "pcm":
        return convert_to_pcm(audio)
    
    # For compressed formats, use pydub if available, otherwise fall back to wav
    try:
        from pydub import AudioSegment
        
        # Convert to WAV first
        wav_bytes = convert_to_wav(audio, sample_rate)
        
        # Load into pydub
        segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        
        # Export to target format
        output = io.BytesIO()
        
        format_params = {
            "mp3": {"format": "mp3", "bitrate": "192k"},
            "opus": {"format": "opus", "bitrate": "128k"},
            "aac": {"format": "adts", "bitrate": "192k"},  # AAC in ADTS container
            "flac": {"format": "flac"},
        }
        
        params = format_params.get(format, {"format": format})
        export_format = params.pop("format", format)
        
        segment.export(output, format=export_format, **params)
        return output.getvalue()
        
    except ImportError:
        # Fall back to WAV if pydub not available
        logger.warning(f"pydub not available, returning WAV instead of {format}")
        return convert_to_wav(audio, sample_rate)
    except Exception as e:
        # Fall back to WAV on any encoding error
        logger.warning(f"Failed to encode to {format} ({e}), returning WAV")
        return convert_to_wav(audio, sample_rate)


async def encode_audio_streaming(
    audio_generator,
    format: AudioFormat = "mp3",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
):
    """
    Async generator that encodes audio chunks to the specified format.
    
    Args:
        audio_generator: Async generator yielding audio chunks as numpy arrays
        format: Target audio format
        sample_rate: Sample rate in Hz
    
    Yields:
        Encoded audio chunks
    """
    async for audio_chunk in audio_generator:
        if audio_chunk is not None and len(audio_chunk) > 0:
            yield encode_audio(audio_chunk, format, sample_rate)
