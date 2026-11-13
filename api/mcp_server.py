# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
MCP (Model Context Protocol) endpoint for Qwen3-TTS.

Exposes the TTS service as MCP tools over streamable-HTTP at /mcp/.
Mounted by api/main.py only when the `mcp` package is installed.

Tools reuse the same code paths as the REST API — including the FIFO
generation queue — so MCP clients queue politely alongside UI/REST users.
"""

import asyncio
import base64
import json
import logging
import uuid

from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .routers import openai_compatible as oai
from .structures.schemas import OpenAISpeechRequest

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "qwen3-tts",
    instructions=(
        "Qwen3-TTS text-to-speech server. Generate speech with preset or "
        "cloned voices (voice='clone:<profile name>'), browse voice profiles "
        "and saved clips, check the GPU generation queue, transcribe style "
        "suggestions, and play saved clips on Home Assistant speakers "
        "(Sonos) when configured."
    ),
)


def _tool_error(e: HTTPException) -> ToolError:
    detail = e.detail
    if isinstance(detail, dict):
        msg = detail.get("message") or json.dumps(detail)
    else:
        msg = str(detail)
    return ToolError(f"HTTP {e.status_code}: {msg}")


async def _collect_response(resp) -> bytes:
    """Read a FastAPI Response/StreamingResponse body to bytes."""
    if hasattr(resp, "body_iterator"):
        chunks = [c async for c in resp.body_iterator]
        return b"".join(chunks)
    return resp.body or b""


@mcp.tool()
async def list_models() -> dict:
    """List available TTS models (e.g. 0.6B-Base, 1.7B-Base, CustomVoice variants)."""
    return await oai.list_models()


@mcp.tool()
async def list_voices() -> dict:
    """List preset voices plus saved clone profiles (voice='clone:<name>')."""
    return await oai.list_voices()


@mcp.tool()
async def list_voice_profiles() -> dict:
    """List saved voice-clone profiles with metadata (no audio)."""
    return await oai.list_voice_profiles()


@mcp.tool()
async def list_clips() -> dict:
    """List saved audio clips (name, tags, voice, model, text) newest first."""
    return await oai.list_clips()


@mcp.tool()
async def list_speakers() -> dict:
    """List Home Assistant media players (Sonos) clips can be played on.
    Returns configured=false when HA integration is not set up."""
    return await oai.list_speakers()


@mcp.tool()
async def queue_status(request_id: str = "") -> dict:
    """Generation queue board, or status of one request_id. States:
    queued (with position + estimated_wait_seconds), running, done, error,
    cancelled. Pass the X-Request-Id used on the speech request."""
    return oai._queue.status(request_id or None) or {"error": "unknown request_id"}


@mcp.tool()
async def suggest_style(vibe: str, text: str = "") -> dict:
    """Ask the configured LLM for a TTS style instruction for a vibe
    (e.g. 'grumpy old man', 'horror story opening'). Requires LEMONADE_URL."""
    try:
        return await oai.style_suggest({"vibe": vibe, "text": text})
    except HTTPException as e:
        raise _tool_error(e)


@mcp.tool()
async def generate_speech(
    text: str,
    voice: str = "Vivian",
    model: str = "qwen3-tts",
    response_format: str = "wav",
    speed: float = 1.0,
    instruct: str = "",
    save_clip: bool = False,
    clip_name: str = "",
) -> dict:
    """Generate speech audio. Returns base64-encoded audio plus metadata.

    voice: preset name or 'clone:<profile>' for a saved clone profile.
    model: 'qwen3-tts' aliases, or a config.yaml key like '1.7B-Base'.
    instruct: style instruction (CustomVoice models only — ignored for clones).
    save_clip: also persist to the clip library; returns clip_id.
    """
    req = OpenAISpeechRequest(
        model=model,
        input=text,
        voice=voice,
        response_format=response_format,
        speed=speed,
        stream=False,
        instruct=instruct or None,
    )
    rid = f"mcp-{uuid.uuid4().hex[:12]}"
    await oai._queue.acquire(rid, oai._queue_info(req))
    state = "done"
    try:
        resp = await oai._create_speech_impl(req, None)
    except HTTPException as e:
        state = "error"
        raise _tool_error(e)
    except asyncio.CancelledError:
        state = "cancelled"
        raise
    except Exception as e:
        state = "error"
        raise ToolError(f"generation failed: {e}")
    finally:
        await oai._queue.release(rid, state)

    data = await _collect_response(resp)
    out = {
        "audio_base64": base64.b64encode(data).decode(),
        "format": response_format,
        "bytes": len(data),
        "voice": voice,
        "model": model,
        "request_id": rid,
    }
    if save_clip:
        mime = getattr(resp, "media_type", "") or "audio/wav"
        clip = await oai.create_clip({
            "name": clip_name or text[:48] or "mcp clip",
            "audio": f"data:{mime};base64,{out['audio_base64']}",
            "format": response_format,
            "text": text,
            "voice": voice,
            "model": model,
            "tags": "mcp",
        })
        out["clip"] = clip.get("clip")
    return out


@mcp.tool()
async def play_clip(clip_id: str, entity_ids: list[str]) -> dict:
    """Play a saved clip on Home Assistant speakers (Sonos).
    entity_ids like 'media_player.living_room'. Requires TTS_HA_URL,
    TTS_HA_TOKEN and TTS_PUBLIC_URL to be configured."""
    try:
        return await oai.play_clip(clip_id, {"entity_ids": entity_ids})
    except HTTPException as e:
        raise _tool_error(e)
