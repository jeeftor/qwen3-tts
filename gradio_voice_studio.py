"""
Gradio Voice Studio for Qwen3-TTS.

This module defines a comprehensive Gradio-based UI that allows users to
create, manage, and export reusable voice profiles using the Qwen3-TTS
API.  The interface supports three primary workflows: preset voices
(CustomVoice), voice design (VoiceDesign), and voice cloning (Base).
Users can store profiles locally, preview and delete them, export
selected profiles to ZIP archives, and synthesize speech with saved
profiles via an interactive playground.

The entrypoint function ``build_app(base_url: str, library_dir: Path)``
returns a Gradio Blocks object configured with the desired API base URL
and profile storage directory.  See README or docstrings for usage.
"""

import argparse
import base64
import io
import json
import mimetypes
import os
import shutil
import tempfile
import time
import uuid
import wave
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import httpx


# -----------------------------------------------------------------------------
# Configuration and Defaults
# -----------------------------------------------------------------------------

# The default directory used to store voice profiles.  This can be overridden
# via the VOICE_LIBRARY_DIR environment variable or passed directly to
# ``build_app``.
DEFAULT_LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "./voice_library")).resolve()

# Base URL pointing to the running Qwen3-TTS API.  If not provided, this
# defaults to the local server on port 8880.  It may be overridden via the
# TTS_BASE_URL environment variable or passed to ``build_app``.
DEFAULT_TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8880").rstrip("/")

# Default timeout for API requests in seconds.  Can be customized via the
# TTS_TIMEOUT_S environment variable or overridden at runtime.
DEFAULT_TIMEOUT_S = float(os.environ.get("TTS_TIMEOUT_S", "300"))

# Sample rate of raw PCM chunks returned by /v1/audio/speech?stream=true
# (Qwen3-TTS emits 24 kHz mono s16le).
STREAM_SAMPLE_RATE = 24000

# Supported task types defined by the Qwen3-TTS API.  These values map to
# endpoint parameters used when creating and managing profiles.
SUPPORTED_TASK_TYPES = ["CustomVoice", "Base"]

# Fallback voices list used when the server does not provide a voices
# endpoint or fails to return names.
FALLBACK_VOICES = ["Vivian", "Ryan", "Serena", "Dylan", "Eric", "Aiden"]


@dataclass
class VoiceProfile:
    """Representation of a saved voice profile.

    Attributes:
        profile_id: Unique identifier for the profile.
        name: Human-friendly name assigned by the user.
        task_type: Type of profile (CustomVoice, VoiceDesign, Base).
        created_at: ISO8601 timestamp indicating when the profile was created.
        language: Language code or Auto for automatic detection.
        voice: Name of the voice (for CustomVoice or design clones).
        instructions: Style instructions (for CustomVoice or voice design).
        ref_text: Transcript of the reference audio (for Base profiles).
        x_vector_only_mode: Whether the clone is created without a transcript.
        ref_audio_filename: Filename of the reference audio stored in the profile directory.
        origin: Human-readable descriptor of how the profile was created.
    """

    profile_id: str
    name: str
    task_type: str
    created_at: str
    language: str = "Auto"
    voice: str = "Vivian"
    instructions: str = ""
    ref_text: str = ""
    x_vector_only_mode: bool = False
    ref_audio_filename: str = ""
    origin: str = ""


def ensure_dirs(library_dir: Path) -> Dict[str, Path]:
    """Ensure that the profiles and exports directories exist.

    Returns a dictionary mapping directory names to Path objects.
    """
    profiles_dir = library_dir / "profiles"
    exports_dir = library_dir / "exports"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    return {"profiles": profiles_dir, "exports": exports_dir}


def now_iso() -> str:
    """Return the current UTC time as an ISO8601-formatted string."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_profile_id() -> str:
    """Generate a short, unique profile identifier."""
    return uuid.uuid4().hex[:12]


def profile_dir(library_dir: Path, profile_id: str) -> Path:
    """Return the path to a given profile's directory."""
    return ensure_dirs(library_dir)["profiles"] / profile_id


def meta_path(library_dir: Path, profile_id: str) -> Path:
    """Return the path to a profile's metadata file."""
    return profile_dir(library_dir, profile_id) / "meta.json"


def load_profile(library_dir: Path, profile_id: str) -> VoiceProfile:
    """Load a profile from disk into a VoiceProfile instance."""
    p = meta_path(library_dir, profile_id)
    data = json.loads(p.read_text(encoding="utf-8"))
    return VoiceProfile(**data)


def save_profile(library_dir: Path, vp: VoiceProfile) -> None:
    """Persist a VoiceProfile to disk by writing its metadata file."""
    d = profile_dir(library_dir, vp.profile_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(vp.__dict__, indent=2), encoding="utf-8")


def delete_profile(library_dir: Path, profile_id: str) -> None:
    """Remove a profile directory and all its contents."""
    d = profile_dir(library_dir, profile_id)
    if d.exists():
        shutil.rmtree(d)


def list_profiles(library_dir: Path) -> List[VoiceProfile]:
    """Return all profiles stored in the library sorted by creation time."""
    dirs = ensure_dirs(library_dir)["profiles"]
    out: List[VoiceProfile] = []
    for child in sorted(dirs.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            mp = child / "meta.json"
            if mp.exists():
                try:
                    data = json.loads(mp.read_text(encoding="utf-8"))
                    out.append(VoiceProfile(**data))
                except Exception:
                    pass  # skip corrupted entries
    # Sort newest first
    out.sort(key=lambda x: x.created_at, reverse=True)
    return out


def profiles_table_rows(profiles: List[VoiceProfile]) -> List[List[Any]]:
    """Convert profiles list into table rows for the library tab."""
    rows = []
    for p in profiles:
        rows.append([
            p.profile_id,
            p.name,
            p.task_type,
            p.origin,
            p.language,
            p.voice,
            (p.instructions[:60] + "…") if len(p.instructions) > 60 else p.instructions,
            "yes" if bool(p.ref_audio_filename) else "no",
            p.created_at,
        ])
    return rows


def normalize_base_url(base_url: str) -> str:
    """Ensure the base URL does not end with a trailing slash."""
    return base_url.rstrip("/")


def data_uri_from_file(file_path: Path) -> str:
    """Encode a file's contents as a data URI with guessed MIME type."""
    mime, _ = mimetypes.guess_type(str(file_path))
    if not mime:
        mime = "audio/wav"
    b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def write_bytes_to_temp_audio(content: bytes, ext: str) -> str:
    """Write raw audio bytes to a temporary file and return its path."""
    ext = ext.lstrip(".")
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    os.close(fd)
    Path(path).write_bytes(content)
    return path


def request_tts(base_url: str, payload: Dict[str, Any], timeout_s: float) -> Tuple[bytes, str]:
    """Call the /v1/audio/speech endpoint and return audio bytes and extension."""
    url = normalize_base_url(base_url) + "/v1/audio/speech"
    response_format = payload.get("response_format") or "wav"
    payload["response_format"] = response_format
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
    ext = response_format.lower()
    if ext == "pcm":
        ext = "raw"
    return r.content, ext


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = STREAM_SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def convert_wav_bytes(wav_bytes: bytes, fmt: str) -> Tuple[bytes, str]:
    """Convert WAV bytes to another audio format via pydub/ffmpeg.

    Falls back to WAV when the target format cannot be produced.
    """
    fmt = (fmt or "wav").lower()
    if fmt in ("wav", "pcm"):
        return wav_bytes, "wav"
    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        out = io.BytesIO()
        export_fmt = "adts" if fmt == "aac" else fmt
        segment.export(out, format=export_fmt)
        return out.getvalue(), fmt
    except Exception:
        return wav_bytes, "wav"


def request_tts_stream(
    base_url: str, payload: Dict[str, Any], timeout_s: float
):
    """Stream /v1/audio/speech as raw PCM chunks.

    Yields ``(chunk_bytes, total_received, elapsed_seconds)`` tuples.  The
    server encodes each generated chunk independently, so we always request
    ``pcm`` — the only per-chunk-safe raw format — and containerize the
    result client-side.
    """
    url = normalize_base_url(base_url) + "/v1/audio/speech"
    req = dict(payload)
    req["stream"] = True
    req["response_format"] = "pcm"
    received = 0
    start = time.time()
    with httpx.Client(timeout=timeout_s) as client:
        with client.stream("POST", url, json=req) as r:
            if r.status_code >= 400:
                body = r.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"TTS server error {r.status_code}: {body}")
            for chunk in r.iter_bytes():
                if not chunk:
                    continue
                received += len(chunk)
                yield chunk, received, time.time() - start


def _stream_status(received: int, elapsed: float) -> str:
    audio_secs = received / (2 * STREAM_SAMPLE_RATE)
    return f"Streaming… {audio_secs:.1f}s of audio · {elapsed:.0f}s elapsed"


def _tts_stream_events(
    base_url: str, payload: Dict[str, Any], timeout_s: float, target_fmt: str = "wav"
):
    """Drive a streaming TTS request, yielding progress events.

    Yields ``{"type": "status", "text": ...}`` updates while audio arrives,
    then a final ``{"type": "done", "path": ..., "ext": ...}`` or
    ``{"type": "error", "text": ...}`` event.  Intended for use inside
    Gradio generator handlers.
    """
    yield {"type": "status", "text": "Generating… (first chunk can be slow if a model is loading)"}
    pcm = bytearray()
    last_status = 0.0
    try:
        for chunk, received, elapsed in request_tts_stream(base_url, payload, timeout_s):
            pcm.extend(chunk)
            if elapsed - last_status >= 0.3:
                last_status = elapsed
                yield {"type": "status", "text": _stream_status(received, elapsed)}
    except Exception as e:
        yield {"type": "error", "text": f"TTS generation failed: {e}"}
        return
    if not pcm:
        yield {"type": "error", "text": "TTS server returned no audio."}
        return
    wav_bytes = pcm_to_wav_bytes(bytes(pcm))
    audio_bytes, ext = convert_wav_bytes(wav_bytes, target_fmt)
    out_path = write_bytes_to_temp_audio(audio_bytes, ext)
    yield {"type": "done", "path": out_path, "ext": ext}


def try_fetch_voices(base_url: str, timeout_s: float) -> List[str]:
    """Attempt to fetch available voices from the /v1/voices endpoint."""
    url = normalize_base_url(base_url) + "/v1/voices"
    try:
        with httpx.Client(timeout=min(timeout_s, 20.0)) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
        voices: List[str] = []
        raw = data.get("voices") if isinstance(data, dict) else data
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    vid = item.get("id") or item.get("name")
                    if vid:
                        voices.append(str(vid))
                elif item:
                    voices.append(str(item))
            if voices:
                return voices
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            names: List[str] = []
            for item in data["data"]:
                if isinstance(item, dict) and "name" in item:
                    names.append(str(item["name"]))
            if names:
                return names
    except Exception:
        pass
    return FALLBACK_VOICES


def export_profiles_zip(library_dir: Path, profile_ids: Optional[List[str]] = None) -> str:
    """Create a ZIP archive of selected profiles (or all) and return its path."""
    ensure_dirs(library_dir)
    profiles = list_profiles(library_dir)
    if profile_ids:
        profiles = [p for p in profiles if p.profile_id in set(profile_ids)]
    exports_dir = ensure_dirs(library_dir)["exports"]
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    zip_path = exports_dir / f"voices_export_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        manifest: List[Dict[str, Any]] = []
        for p in profiles:
            d = profile_dir(library_dir, p.profile_id)
            z.write(d / "meta.json", arcname=f"{p.profile_id}/meta.json")
            if p.ref_audio_filename:
                ref_path = d / p.ref_audio_filename
                if ref_path.exists():
                    z.write(ref_path, arcname=f"{p.profile_id}/{p.ref_audio_filename}")
            manifest.append(p.__dict__)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    return str(zip_path)


# -----------------------------------------------------------------------------
# Gradio UI Construction
# -----------------------------------------------------------------------------

# Custom CSS to style the interface with a dark theme that complements the
# existing web UI of the Qwen3-TTS server.
CSS = """
:root {
  --bg0: #0b0f19;
  --bg1: #0f172a;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.10);
}
body, .gradio-container { background: radial-gradient(1200px 800px at 10% 10%, var(--bg1), var(--bg0)) !important; }
#header {
  padding: 18px 18px;
  border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(59,130,246,0.08));
  border-radius: 18px;
}
.card {
  border: 1px solid var(--border);
  background: var(--card);
  border-radius: 18px;
}
.small { opacity: 0.8; font-size: 0.95em; }
"""

TABLE_HEADERS = [
    "id",
    "name",
    "task_type",
    "origin",
    "language",
    "voice",
    "instructions",
    "has_ref_audio",
    "created_at",
]


def build_app(initial_base_url: str, initial_library_dir: Path) -> gr.Blocks:
    """Construct the Gradio Blocks interface for the Voice Studio.

    Args:
        initial_base_url: Base URL of the Qwen3-TTS API to use for requests.
        initial_library_dir: Directory where profiles will be stored.

    Returns:
        A gradio.Blocks object ready to be mounted on a FastAPI server or
        launched standalone via ``.launch()``.
    """
    ensure_dirs(initial_library_dir)
    with gr.Blocks(title="Qwen3 Voice Studio", css=CSS, theme=gr.themes.Soft()) as demo:
        # Shared state variables
        state_base_url = gr.State(initial_base_url)
        state_library_dir = gr.State(str(initial_library_dir))
        state_voices = gr.State([])

        # Header section
        gr.HTML(
            """
            <div id="header">
              <div style="font-size: 1.35rem; font-weight: 700;">Qwen3 Voice Studio</div>
              <div class="small">
                Create & save reusable voice profiles (preset or cloned) and export them for inference.
              </div>
            </div>
            """
        )

        # Settings accordion
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                base_url_in = gr.Textbox(
                    label="TTS Server Base URL",
                    value=initial_base_url,
                    placeholder="http://localhost:8880",
                )
                library_dir_in = gr.Textbox(
                    label="Voice Library Dir",
                    value=str(initial_library_dir),
                    placeholder="./voice_library",
                )
                timeout_in = gr.Number(
                    label="Request timeout (seconds)",
                    value=DEFAULT_TIMEOUT_S,
                    precision=0,
                )
            with gr.Row():
                refresh_voices_btn = gr.Button("Refresh voices from server", variant="primary")
                voices_status = gr.Markdown("", elem_classes=["small"])

        # Global log output
        with gr.Row():
            global_log = gr.Markdown("", elem_classes=["small"])

        # Tabs for create, library, and playground
        with gr.Tabs():
            # Create tab and nested subtabs
            with gr.Tab("Create"):
                with gr.Tabs():
                    # Preset (CustomVoice)
                    with gr.Tab("Preset Voice (CustomVoice)"):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=320):
                                preset_name = gr.Textbox(label="Profile name", placeholder="e.g. 'Vivian - Friendly NZ Support'")
                                preset_voice = gr.Dropdown(label="Voice", choices=FALLBACK_VOICES, value=FALLBACK_VOICES[0])
                                preset_model = gr.Dropdown(
                                    label="Model",
                                    choices=["Auto", "0.6B-CustomVoice", "1.7B-CustomVoice"],
                                    value="Auto",
                                )
                                preset_language = gr.Dropdown(
                                    label="Language",
                                    choices=[
                                        "Auto",
                                        "English",
                                        "Spanish",
                                        "Chinese",
                                        "Japanese",
                                        "Korean",
                                        "German",
                                        "French",
                                        "Russian",
                                        "Portuguese",
                                        "Italian",
                                    ],
                                    value="Auto",
                                )
                                preset_instructions = gr.Textbox(
                                    label="Style instructions (optional)",
                                    placeholder="e.g. Calm, confident, slightly upbeat, clear articulation.",
                                    lines=3,
                                )
                                preset_test_text = gr.Textbox(
                                    label="Test text",
                                    value="Hello! This is my saved preset voice profile.",
                                    lines=3,
                                )
                                preset_generate_btn = gr.Button("Generate", variant="primary")
                                preset_save_btn = gr.Button("Save profile", variant="secondary")
                            with gr.Column(scale=1, min_width=320):
                                with gr.Row():
                                    preset_audio = gr.Audio(label="Output audio", type="filepath", scale=4)
                                    preset_download = gr.DownloadButton(label="Download", scale=1)

                    # Voice clone (Base)
                    with gr.Tab("Voice Clone (Base)"):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=320):
                                clone_name = gr.Textbox(label="Profile name", placeholder="e.g. 'Facu - Mic Clone v1'")
                                clone_language = gr.Dropdown(
                                    label="Language",
                                    choices=[
                                        "Auto",
                                        "English",
                                        "Spanish",
                                        "Chinese",
                                        "Japanese",
                                        "Korean",
                                        "German",
                                        "French",
                                        "Russian",
                                        "Portuguese",
                                        "Italian",
                                    ],
                                    value="Auto",
                                )
                                clone_model_sel = gr.Dropdown(
                                    label="Clone model",
                                    choices=["Auto", "0.6B-Base", "1.7B-Base"],
                                    value="Auto",
                                )
                                clone_ref_audio = gr.Audio(
                                    label="Reference audio (upload or record)",
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                )
                                clone_xvec_only = gr.Checkbox(
                                    label="x_vector_only_mode (no transcript needed, usually lower quality)",
                                    value=False,
                                )
                                clone_ref_text = gr.Textbox(
                                    label="Reference transcript (recommended)",
                                    placeholder="Paste the transcript of the reference audio (or leave blank if x_vector_only_mode).",
                                    lines=3,
                                )
                                clone_test_text = gr.Textbox(
                                    label="Test text (what you want to synthesize)",
                                    value="Hello! This is a voice clone test.",
                                    lines=3,
                                )
                                clone_generate_btn = gr.Button("Generate", variant="primary")
                                clone_save_btn = gr.Button("Save clone profile", variant="secondary")
                            with gr.Column(scale=1, min_width=320):
                                with gr.Row():
                                    clone_audio = gr.Audio(label="Output audio", type="filepath", scale=4)
                                    clone_download = gr.DownloadButton(label="Download", scale=1)

            # Library tab
            with gr.Tab("Library"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=520):
                        library_table = gr.Dataframe(
                            headers=TABLE_HEADERS,
                            datatype=["str"] * len(TABLE_HEADERS),
                            label="Saved profiles",
                            interactive=False,
                            wrap=True,
                        )
                        with gr.Row():
                            library_refresh_btn = gr.Button("Refresh list", variant="primary")
                            export_selected_btn = gr.Button("Export selected → ZIP", variant="secondary")
                            export_all_btn = gr.Button("Export ALL → ZIP", variant="secondary")
                        export_file = gr.File(label="Export download")
                    with gr.Column(scale=1, min_width=340):
                        selected_id = gr.Textbox(label="Selected profile id", placeholder="Click a row to copy id here")
                        load_selected_btn = gr.Button("Load selected", variant="primary")
                        delete_selected_btn = gr.Button("Delete selected", variant="stop")
                        profile_details = gr.JSON(label="Profile details")
                        ref_preview = gr.Audio(label="Reference audio preview (if available)", type="filepath")
                        with gr.Accordion("Edit profile", open=False):
                            edit_name = gr.Textbox(label="Name")
                            edit_language = gr.Dropdown(
                                label="Language",
                                choices=[
                                    "Auto",
                                    "English",
                                    "Spanish",
                                    "Chinese",
                                    "Japanese",
                                    "Korean",
                                    "German",
                                    "French",
                                    "Russian",
                                    "Portuguese",
                                    "Italian",
                                ],
                            )
                            edit_voice = gr.Textbox(label="Voice (CustomVoice presets only)")
                            edit_instructions = gr.Textbox(label="Style instructions", lines=2)
                            edit_ref_text = gr.Textbox(
                                label="Reference transcript (Base clones — review carefully)",
                                lines=4,
                            )
                            edit_xvec = gr.Checkbox(label="x_vector_only_mode (no transcript needed)")
                            edit_ref_audio = gr.Audio(
                                label="Replace reference audio (optional — leave empty to keep current)",
                                type="filepath",
                            )
                            edit_save_btn = gr.Button("Save changes", variant="primary")
                            edit_status = gr.Markdown("", elem_classes=["small"])

            # Playground tab
            with gr.Tab("Playground"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        play_profile_id = gr.Dropdown(label="Pick a saved profile", choices=[], value=None)
                        play_text = gr.Textbox(label="Text to synthesize", value="Hello from the playground!", lines=4)
                        play_response_format = gr.Dropdown(
                            label="Audio format",
                            choices=["wav", "mp3", "flac", "aac", "opus", "pcm"],
                            value="wav",
                        )
                        play_model = gr.Dropdown(
                            label="Model override",
                            choices=["Auto", "0.6B-Base", "1.7B-Base", "0.6B-CustomVoice", "1.7B-CustomVoice"],
                            value="Auto",
                        )
                        play_speed = gr.Slider(label="Speed", minimum=0.25, maximum=4.0, value=1.0, step=0.05)
                        play_generate_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=1, min_width=360):
                        with gr.Row():
                            play_audio = gr.Audio(label="Output audio", type="filepath", scale=4)
                            play_download = gr.DownloadButton(label="Download", scale=1)
                        play_status = gr.Markdown("", elem_classes=["small"])

        # ------------------------------------------------------------------
        # Callback implementations
        # ------------------------------------------------------------------

        def on_refresh_voices(base_url: str, timeout_s: float):
            voices = try_fetch_voices(base_url, float(timeout_s))
            return (
                gr.State(base_url),
                gr.State(voices),
                gr.Dropdown(choices=voices, value=voices[0] if voices else None),
                f"✅ Loaded {len(voices)} voices from server (or fallback list).",
            )

        def on_generate_preset(base_url: str, timeout_s: float, voice: str, model_sel: str, language: str, instructions: str, text: str):
            payload = {
                "input": text,
                "voice": voice,
                "language": language,
                "task_type": "CustomVoice",
                "instruct": instructions or "",
                "response_format": "wav",
            }
            if model_sel != "Auto":
                payload["model"] = model_sel
            for ev in _tts_stream_events(base_url, payload, float(timeout_s)):
                if ev["type"] == "done":
                    yield ev["path"], ev["path"], "✅ Generated audio."
                elif ev["type"] == "error":
                    yield None, None, f"❌ {ev['text']}"
                else:
                    yield None, None, ev["text"]

        def on_save_preset(library_dir_str: str, name: str, voice: str, language: str, instructions: str):
            if not name.strip():
                raise gr.Error("Profile name is required.")
            vp = VoiceProfile(
                profile_id=safe_profile_id(),
                name=name.strip(),
                task_type="CustomVoice",
                origin="Preset(CustomVoice)",
                created_at=now_iso(),
                language=language,
                voice=voice,
                instructions=instructions or "",
            )
            save_profile(Path(library_dir_str), vp)
            return f"✅ Saved preset profile: {vp.profile_id}"

        def on_generate_clone(
            base_url: str,
            timeout_s: float,
            language: str,
            model_sel: str,
            ref_audio_path: str,
            ref_text: str,
            xvec_only: bool,
            text: str,
        ):
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise gr.Error("Reference audio is required.")
            if (not xvec_only) and (not ref_text.strip()):
                raise gr.Error("Reference transcript is required unless x_vector_only_mode is enabled.")
            ref_uri = data_uri_from_file(Path(ref_audio_path))
            payload = {
                "input": text,
                "voice": "Vivian",
                "language": language,
                "task_type": "Base",
                "ref_audio": ref_uri,
                "ref_text": ref_text.strip(),
                "x_vector_only_mode": bool(xvec_only),
                "response_format": "wav",
            }
            if model_sel != "Auto":
                payload["model"] = model_sel
            for ev in _tts_stream_events(base_url, payload, float(timeout_s)):
                if ev["type"] == "done":
                    yield ev["path"], ev["path"], "✅ Generated audio."
                elif ev["type"] == "error":
                    yield None, None, f"❌ {ev['text']}"
                else:
                    yield None, None, ev["text"]

        def on_save_clone_profile(
            library_dir_str: str,
            name: str,
            language: str,
            ref_audio_path: str,
            ref_text: str,
            xvec_only: bool,
        ):
            if not name.strip():
                raise gr.Error("Profile name is required.")
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise gr.Error("Reference audio is required.")
            if (not xvec_only) and (not ref_text.strip()):
                raise gr.Error("Reference transcript is required unless x_vector_only_mode is enabled.")
            pid = safe_profile_id()
            dest_dir = profile_dir(Path(library_dir_str), pid)
            dest_dir.mkdir(parents=True, exist_ok=True)
            ref_ext = Path(ref_audio_path).suffix or ".wav"
            ref_filename = f"ref_audio{ref_ext}"
            shutil.copy2(ref_audio_path, dest_dir / ref_filename)
            vp = VoiceProfile(
                profile_id=pid,
                name=name.strip(),
                task_type="Base",
                origin="Clone(Base)",
                created_at=now_iso(),
                language=language,
                voice="Vivian",
                instructions="",
                ref_text=ref_text.strip(),
                x_vector_only_mode=bool(xvec_only),
                ref_audio_filename=ref_filename,
            )
            save_profile(Path(library_dir_str), vp)
            return f"✅ Saved clone profile: {pid}"

        def on_library_refresh(library_dir_str: str):
            profiles = list_profiles(Path(library_dir_str))
            table = profiles_table_rows(profiles)
            choices = [p.profile_id for p in profiles]
            return table, gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        def on_load_selected(library_dir_str: str, pid: str):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            vp = load_profile(Path(library_dir_str), pid.strip())
            ref_path = ""
            if vp.ref_audio_filename:
                candidate = profile_dir(Path(library_dir_str), vp.profile_id) / vp.ref_audio_filename
                if candidate.exists():
                    ref_path = str(candidate)
            return (
                vp.__dict__,
                ref_path,
                vp.name,
                vp.language,
                vp.voice,
                vp.instructions,
                vp.ref_text,
                vp.x_vector_only_mode,
                None,
            )

        def on_update_profile(
            library_dir_str: str,
            pid: str,
            name: str,
            language: str,
            voice: str,
            instructions: str,
            ref_text: str,
            xvec_only: bool,
            new_ref_audio_path: Optional[str],
        ):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            lib = Path(library_dir_str)
            vp = load_profile(lib, pid.strip())
            if name.strip():
                vp.name = name.strip()
            if language:
                vp.language = language
            if voice.strip():
                vp.voice = voice.strip()
            vp.instructions = instructions or ""
            vp.ref_text = ref_text or ""
            vp.x_vector_only_mode = bool(xvec_only)
            note = ""
            if new_ref_audio_path and Path(str(new_ref_audio_path)).exists():
                src = Path(str(new_ref_audio_path))
                pdir = profile_dir(lib, vp.profile_id)
                current = (pdir / vp.ref_audio_filename) if vp.ref_audio_filename else None
                if not current or src.resolve() != current.resolve():
                    ext = src.suffix or ".wav"
                    fname = f"ref_audio{ext}"
                    shutil.copy2(src, pdir / fname)
                    if vp.ref_audio_filename and vp.ref_audio_filename != fname:
                        old = pdir / vp.ref_audio_filename
                        if old.exists():
                            old.unlink()
                    vp.ref_audio_filename = fname
                    note = f" — new reference audio saved as {fname}"
            save_profile(lib, vp)
            profiles = list_profiles(lib)
            table = profiles_table_rows(profiles)
            choices = [p.profile_id for p in profiles]
            return (
                table,
                gr.Dropdown(choices=choices, value=vp.profile_id),
                f"✅ Updated profile '{vp.name}'{note}",
            )

        def on_delete_selected(library_dir_str: str, pid: str):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            delete_profile(Path(library_dir_str), pid.strip())
            return "✅ Deleted profile."

        def on_export_selected(library_dir_str: str, pid: str):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            zip_path = export_profiles_zip(Path(library_dir_str), profile_ids=[pid.strip()])
            return zip_path, f"✅ Exported: {Path(zip_path).name}"

        def on_export_all(library_dir_str: str):
            zip_path = export_profiles_zip(Path(library_dir_str), profile_ids=None)
            return zip_path, f"✅ Exported: {Path(zip_path).name}"

        def on_play_generate(
            base_url: str,
            timeout_s: float,
            library_dir_str: str,
            pid: str,
            text: str,
            fmt: str,
            model_sel: str,
            speed: float,
        ):
            if not pid:
                raise gr.Error("Pick a saved profile.")
            vp = load_profile(Path(library_dir_str), pid)
            payload: Dict[str, Any] = {
                "input": text,
                "response_format": fmt,
                "speed": float(speed),
                "language": vp.language,
            }
            if model_sel != "Auto":
                payload["model"] = model_sel
            if vp.task_type == "CustomVoice":
                payload.update({
                    "task_type": "CustomVoice",
                    "voice": vp.voice,
                    "instruct": vp.instructions or "",
                })
            elif vp.task_type == "Base":
                payload.update({
                    "task_type": "Base",
                    "voice": vp.voice or "Vivian",
                    "x_vector_only_mode": bool(vp.x_vector_only_mode),
                })
                if vp.ref_audio_filename:
                    ref_file = profile_dir(Path(library_dir_str), vp.profile_id) / vp.ref_audio_filename
                    if not ref_file.exists():
                        raise gr.Error("This profile is missing its reference audio file.")
                    payload["ref_audio"] = data_uri_from_file(ref_file)
                else:
                    raise gr.Error("This Base profile has no stored ref_audio.")
                if not vp.x_vector_only_mode:
                    if not vp.ref_text.strip():
                        raise gr.Error("This profile needs ref_text unless x_vector_only_mode is enabled.")
                    payload["ref_text"] = vp.ref_text.strip()
            else:
                payload.update({
                    "task_type": "VoiceDesign",
                    "voice": vp.voice or "Vivian",
                    "instruct": vp.instructions or "",
                })
            if float(speed) != 1.0:
                # Streaming endpoints ignore speed — use the blocking path
                audio_bytes, ext = request_tts(base_url, payload, float(timeout_s))
                out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                yield out_path, out_path, "✅ Generated audio."
                return
            for ev in _tts_stream_events(base_url, payload, float(timeout_s), target_fmt=fmt):
                if ev["type"] == "done":
                    yield ev["path"], ev["path"], "✅ Generated audio."
                elif ev["type"] == "error":
                    yield None, None, f"❌ {ev['text']}"
                else:
                    yield None, None, ev["text"]

        # ------------------------------------------------------------------
        # Wire up UI interactions to callbacks
        # ------------------------------------------------------------------

        refresh_voices_btn.click(
            fn=on_refresh_voices,
            inputs=[base_url_in, timeout_in],
            outputs=[state_base_url, state_voices, preset_voice, voices_status],
        )

        preset_generate_btn.click(
            fn=on_generate_preset,
            inputs=[base_url_in, timeout_in, preset_voice, preset_model, preset_language, preset_instructions, preset_test_text],
            outputs=[preset_audio, preset_download, global_log],
        )
        preset_save_btn.click(
            fn=on_save_preset,
            inputs=[library_dir_in, preset_name, preset_voice, preset_language, preset_instructions],
            outputs=[global_log],
        )

        clone_generate_btn.click(
            fn=on_generate_clone,
            inputs=[base_url_in, timeout_in, clone_language, clone_model_sel, clone_ref_audio, clone_ref_text, clone_xvec_only, clone_test_text],
            outputs=[clone_audio, clone_download, global_log],
        )
        clone_save_btn.click(
            fn=on_save_clone_profile,
            inputs=[library_dir_in, clone_name, clone_language, clone_ref_audio, clone_ref_text, clone_xvec_only],
            outputs=[global_log],
        )

        library_refresh_btn.click(
            fn=on_library_refresh,
            inputs=[library_dir_in],
            outputs=[library_table, play_profile_id],
        )
        load_selected_btn.click(
            fn=on_load_selected,
            inputs=[library_dir_in, selected_id],
            outputs=[
                profile_details,
                ref_preview,
                edit_name,
                edit_language,
                edit_voice,
                edit_instructions,
                edit_ref_text,
                edit_xvec,
                edit_ref_audio,
            ],
        )
        edit_save_btn.click(
            fn=on_update_profile,
            inputs=[
                library_dir_in,
                selected_id,
                edit_name,
                edit_language,
                edit_voice,
                edit_instructions,
                edit_ref_text,
                edit_xvec,
                edit_ref_audio,
            ],
            outputs=[library_table, play_profile_id, edit_status],
        )
        delete_selected_btn.click(
            fn=on_delete_selected,
            inputs=[library_dir_in, selected_id],
            outputs=[global_log],
        ).then(
            fn=on_library_refresh,
            inputs=[library_dir_in],
            outputs=[library_table, play_profile_id],
        )

        export_selected_btn.click(
            fn=on_export_selected,
            inputs=[library_dir_in, selected_id],
            outputs=[export_file, global_log],
        )
        export_all_btn.click(
            fn=on_export_all,
            inputs=[library_dir_in],
            outputs=[export_file, global_log],
        )

        play_generate_btn.click(
            fn=on_play_generate,
            inputs=[base_url_in, timeout_in, library_dir_in, play_profile_id, play_text, play_response_format, play_model, play_speed],
            outputs=[play_audio, play_download, play_status],
        )

        demo.load(fn=on_library_refresh, inputs=[library_dir_in], outputs=[library_table, play_profile_id])

    return demo


def main():
    """Launch the Voice Studio as a standalone web app for testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_TTS_BASE_URL)
    parser.add_argument("--library-dir", default=str(DEFAULT_LIBRARY_DIR))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    app = build_app(args.base_url, Path(args.library_dir))
    app.queue(default_concurrency_limit=4).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
