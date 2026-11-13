# voice_library

Runtime data directory — **not source code**. This directory is gitignored;
clone the repo and it will be created empty on first run.

## Layout

```
voice_library/
├── studio.html          # optional UI override (shipped copy lives in api/static/)
├── profiles/<id>/       # saved voice-clone profiles
│   ├── meta.json        # name, language, transcript, voice settings
│   └── ref.wav          # 24 kHz mono reference audio
└── clips/<id>/          # saved generated clips
    ├── meta.json        # name, tags, text, voice, model, format, timestamps
    └── audio.<fmt>      # wav | mp3 | flac | ogg
```

## Profiles (`profiles/`)

Created via `POST /v1/voice-profiles` or the Studio UI's Clone tab.
`meta.json` fields: `profile_id`, `name`, `task_type` (`Base` = clone),
`language`, `ref_text` (reference transcript — improves clone quality),
`x_vector_only_mode`, `voice`, `instructions`, `description`,
`image_url`, `has_ref_audio`, `created_at`.

Profiles are usable as `clone:<name>` in `/v1/audio/speech` and listed by
`GET /v1/voice-profiles`. Reference audio is served at
`GET /v1/voice-profiles/{id}/ref-audio`.

## Clips (`clips/`)

Created via `POST /v1/clips` (base64/data-URI audio + metadata) or the
Studio UI's Save clip button. `meta.json` fields: `clip_id`, `name`,
`created_at`, `text`, `voice`, `model`, `format`, `audio_filename`,
`tags`.

Clips are playable at `GET /v1/clips/{id}/audio`, downloadable, and —
when `TTS_HA_URL`/`TTS_HA_TOKEN` are configured — playable on Sonos
speakers via `POST /v1/clips/{id}/play` and discoverable via
`GET /v1/speakers`.

## Environment

| Var | Default | Purpose |
|---|---|---|
| `VOICE_LIBRARY_DIR` | `./voice_library` | Root of this directory |
