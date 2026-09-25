# Voicebox Backend

FastAPI server powering voice cloning, speech generation, and audio processing. Runs locally as a Tauri sidecar or standalone via `python -m backend.main`.

## Running

```bash
# Via justfile (recommended)
just dev:server

# Standalone
python -m backend.main --host 127.0.0.1 --port 17493

# With custom data directory
python -m backend.main --data-dir /path/to/data
```

The server auto-initializes the SQLite database on first startup. Models are downloaded from HuggingFace on first use.

## Architecture

```
backend/
  app.py                  # FastAPI app factory, CORS, lifecycle events
  main.py                 # Entry point (imports app, runs uvicorn)
  config.py               # Data directory paths and configuration
  models.py               # Pydantic request/response schemas
  server.py               # Tauri sidecar launcher, parent-pid watchdog

  routes/                 # Thin HTTP handlers — validation, delegation, response formatting
  services/               # Business logic, CRUD, orchestration
  backends/               # TTS/STT engine implementations (MLX, PyTorch, etc.)
  database/               # ORM models, session management, migrations, seed data
  utils/                  # Shared utilities (audio, effects, caching, progress tracking)
```

### Request flow

```
HTTP request
  -> routes/        (validate input, parse params)
  -> services/      (business logic, database queries, orchestration)
  -> backends/      (TTS/STT inference)
  -> utils/         (audio processing, effects, caching)
```

Route handlers are intentionally thin. They validate input, delegate to a service function, and format the response. All business logic lives in `services/`.

### Key modules

**services/generation.py** -- Single `run_generation()` function that handles all three generation modes (generate, retry, regenerate). Manages model loading, voice prompt creation, chunked inference, normalization, effects, and version persistence.

**services/task_queue.py** -- Serial generation queue. Ensures only one GPU inference runs at a time. Background tasks are tracked to prevent garbage collection.

**backends/__init__.py** -- Protocol definitions (`TTSBackend`, `STTBackend`), model config registry, and factory functions. Adding a new engine means implementing the protocol and registering a config entry.

**backends/base.py** -- Shared utilities used across all engine implementations: HuggingFace cache checks, device detection, voice prompt combination, progress tracking.

**database/** -- SQLAlchemy ORM models with a re-exporting `__init__.py` for backward compatibility. Migrations run automatically on startup.

### Backend selection

The server detects the best inference backend at startup:

| Platform | Backend | Acceleration |
|----------|---------|-------------|
| macOS (Apple Silicon) | MLX | Metal / Neural Engine |
| Windows / Linux (NVIDIA) | PyTorch | CUDA |
| Linux (AMD) | PyTorch | ROCm |
| Intel Arc | PyTorch | IPEX / XPU |
| Windows (any GPU) | PyTorch | DirectML |
| Any | PyTorch | CPU fallback |

Detection is handled by `utils/platform_detect.py`. Both backends implement the same `TTSBackend` protocol, so the API layer is engine-agnostic.

## API

90 endpoints organized by domain. Full interactive documentation available at `http://localhost:17493/docs` when the server is running.

| Domain | Prefix | Description |
|--------|--------|-------------|
| Health | `/`, `/health` | Server status, GPU info, filesystem checks |
| Profiles | `/profiles` | Voice profile CRUD, samples, avatars, import/export |
| Channels | `/channels` | Audio channel management and voice assignment |
| Generation | `/generate` | TTS generation, retry, regenerate, status SSE |
| History | `/history` | Generation history, search, favorites, export |
| Transcription | `/transcribe` | Whisper-based audio-to-text |
| Stories | `/stories` | Multi-track timeline editor, audio export |
| Effects | `/effects` | Effect presets, preview, version management |
| Audio | `/audio`, `/samples` | Audio file serving |
| Models | `/models` | Load, unload, download, migrate, status |
| Tasks | `/tasks`, `/cache` | Active task tracking, cache management |
| CUDA | `/backend/cuda-*` | CUDA binary download and management |
| Auth | `/auth` | `whoami`, media tokens, API key management (admin) |

### OpenAI-compatible surface

`POST /v1/audio/speech`, `POST /v1/audio/transcriptions`, `GET /v1/models` and `GET /v1/voices` (`routes/openai_compat.py`) speak the OpenAI Audio API, so the OpenAI SDKs work with `base_url=".../v1"` and a client key. Voices are profile names or ids, models are registry names (`kokoro`, `qwen-tts-1.7B`, ...) or the `tts-1` aliases, and `response_format` covers mp3/opus/aac/flac/wav/pcm (`utils/encode.py`: ffmpeg pipe when available, libsndfile otherwise). Errors under `/v1` use the OpenAI `{"error": {...}}` envelope (`api_errors.py`), including the ones the auth middleware produces.

### Authentication

Every endpoint requires `Authorization: Bearer <key>`. Keys come from `VOICEBOX_API_KEY`, else the `api_key` file the server creates in its data directory on first start (`just api-key` prints the dev one), plus the hashed `api_keys.json` store managed with `python -m backend.keys` or the admin `/auth/keys` routes. `admin` keys can do everything; `client` keys (for your own apps) can generate, stream, speak, transcribe and read profiles. Browser loads that cannot send headers use `POST /auth/media-token` and `?token=`. `GET /health` answers `{"status": "healthy", "service": "voicebox"}` without a key. Details: `docs/content/docs/overview/api-keys.mdx`.

### Quick examples

```bash
export VOICEBOX_API_KEY="$(just api-key)"

# Generate speech
curl -X POST http://localhost:17493/generate \
  -H "Authorization: Bearer $VOICEBOX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "profile_id": "...", "language": "en"}'

# List profiles
curl -H "Authorization: Bearer $VOICEBOX_API_KEY" http://localhost:17493/profiles

# Stream generation status (SSE)
curl -H "Authorization: Bearer $VOICEBOX_API_KEY" http://localhost:17493/generate/{id}/status

# Stream audio while it is synthesized (WAV with unknown length, or "format": "pcm")
curl -N -X POST http://localhost:17493/generate/stream \
  -H "Authorization: Bearer $VOICEBOX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "First sentence. Second sentence.", "profile_id": "..."}' \
  | ffplay -nodisp -autoexit -

# Create a client key for one of your apps (printed once)
backend/venv/bin/python -m backend.keys create --id myapp --role client
```

## Data directory

```
{data_dir}/
  voicebox.db             # SQLite database
  profiles/{id}/          # Voice samples per profile
  generations/            # Generated audio files
  cache/                  # Voice prompt cache (memory + disk)
  backends/               # Downloaded CUDA binary (if applicable)
```

Default location is the OS-specific app data directory. Override with `--data-dir` or the `VOICEBOX_DATA_DIR` environment variable.

## Code quality

Linting and formatting are enforced by [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`. See `STYLE_GUIDE.md` for conventions.

```bash
just check-python       # lint + format check
just fix-python         # auto-fix lint issues + reformat
just test               # run pytest
```

## Server operation

Environment variables the server reads on a host or in Docker: `VOICEBOX_DATA_DIR` (or `--data-dir`), `VOICEBOX_PRELOAD_MODELS` (models loaded at boot; `GET /health/ready` is 503 until they are resident), `VOICEBOX_RETENTION_DAYS` (daily pruning of old generations and captures), `VOICEBOX_DRAIN_TIMEOUT_S` (how long a SIGTERM waits for queued generations), `LOG_LEVEL` / `VOICEBOX_LOG_LEVEL`, plus the key and limit variables from `auth/settings.py`. `python -m backend.preload <names>` downloads models ahead of time (`--list` prints the names). See `docs/content/docs/overview/deployment.mdx`.

## Dependencies

The Docker image installs `requirements.lock`, generated from `requirements.txt` plus `requirements-docker.in` by `scripts/lock-backend.sh` (`just lock-backend`, needs `uv`). torch and torchaudio are left out of the lock on purpose: the `Dockerfile` installs the chosen `PYTORCH_VARIANT`'s wheels first. Regenerate the lock whenever `requirements.txt` changes; the desktop and dev environments keep using `requirements.txt` directly.

### Desktop and development environment

Runtime dependencies are in `requirements.txt`. macOS-only MLX dependencies are in `requirements-mlx.txt`. Dev tools (ruff, pytest) are installed automatically by `just setup-python`.
