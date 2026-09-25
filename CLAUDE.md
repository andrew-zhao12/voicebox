# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Voicebox is a local-first AI voice studio: a FastAPI + PyTorch/MLX backend (`backend/`) that clones voices, generates speech with seven TTS engines, transcribes with Whisper, and runs a small local Qwen3 LLM; one shared React UI (`app/`) shipped both as a Tauri 2 desktop app (`tauri/`) and as a browser SPA (`web/`); and a built-in MCP server so agents can speak and transcribe. Everything talks HTTP to the backend on port **17493**. `landing/` (Next.js marketing site) and `docs/` (Fumadocs site with its own `bun.lock`) are separate concerns.

## Commands (run from the repo root)

Toolchain: `just`, Bun 1.3.8 (`packageManager` pin + `engine-strict`; not npm/yarn), Python 3.12 venv at `backend/venv` (created by `just setup-python`; plain pip, no lockfile), Rust stable for Tauri.

| Task | Command |
|---|---|
| First-time setup | `just setup` (= `setup-python` + `bun install`) |
| Backend only (uvicorn `--reload` on :17493) | `just dev-backend` |
| Print the dev admin API key (`data/api_key`, created on first use) | `just api-key` |
| Create a client API key for one of your apps | `backend/venv/bin/python -m backend.keys create --id myapp --role client` |
| Backend + desktop app | `just dev` (starts uvicorn unless :17493 already answers `/health`, then `tauri dev`; Vite on :5173, strictPort) |
| Backend + browser SPA | `just dev-web` |
| Python tests | `just test` (= `backend/venv/bin/python -m pytest backend/tests -v`) |
| Single test | `backend/venv/bin/python -m pytest backend/tests/test_cors.py -k <name> -v` |
| E2E: every TTS model against the frozen binary | `just test-models [--only kokoro]` (run `just build-server` first) |
| Lint + format check (Biome + ruff) | `just check`; auto-fix with `just fix` |
| TypeScript typecheck (app + web) | `bun run typecheck` |
| What GitHub CI runs | `bun run ci` (= typecheck + `build:web`). CI does **not** run pytest, ruff or Biome. |
| Sidecar binaries (PyInstaller) | `just build-server`, writes `tauri/src-tauri/binaries/voicebox-{server,mcp}-<triple>` |
| Desktop app / everything | `just build-tauri` / `just build` |
| GPU server variants | `python backend/build_binary.py --cuda` or `--rocm`, then `scripts/package_cuda.py` / `package_rocm.py` |
| Docker (CPU) | `docker compose up --build`, UI + API at http://localhost:17600 (host 17600 maps to container 17493) |
| Docker (AMD ROCm) | `docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build` |
| Docs site | `cd docs && bun install && bun run dev` |

Formatting: Biome for TS (`biome.json`: 2-space, width 100, single quotes); ruff for Python (`backend/pyproject.toml`: width 120, double quotes, py312). `.biomeignore` excludes all of `app/src/lib/api/`, including the hand-written client, so keep that folder tidy by hand.

## Architecture

### Monorepo layout
- `app/` is `@voicebox/app`: all React UI source. Not runnable alone (`app/src/main.tsx` has no `PlatformProvider`). Both shells alias `@` to `app/src`.
- `tauri/`: desktop shell. `src/main.tsx` + `src/platform/*` implement the `Platform` interface with Tauri APIs; Rust lives in `src-tauri/`.
- `web/`: browser shell with the same shape (`src/platform/*` are browser no-ops); builds to `web/dist`. The Dockerfile copies that to `/app/frontend`, which `backend/app.py` serves as an SPA when the directory exists.
- `backend/`: a Python package, not a Bun workspace. Import as `backend.*`; the uvicorn target is `backend.main:app`.
- `scripts/`: build/release helpers. `scripts/setup-dev-sidecar.js` writes placeholder sidecar binaries so `tauri dev` compiles without a PyInstaller build.

### Backend request flow (`backend/`)
- Entry points: `main.py` re-exports `app` (its `python -m backend.main` `--port` default is **8000**, not 17493). `server.py` is the PyInstaller entry: adds the `--parent-pid` watchdog and sets `VOICEBOX_BACKEND_VARIANT` from the binary name. `app.py::create_app()` composes the Voicebox and FastMCP lifespans, adds CORS + `ClientIdMiddleware`, calls `routes/__init__.py::register_routers`, mounts MCP at `/mcp`, then the optional SPA.
- Layers: `routes/` (thin handlers; decorators use absolute paths, only `cloud.py` and `settings.py` set a prefix) call `services/` (generation, task_queue, profiles, personality, refinement, captures, transcribe), which call `backends/` (engines), with `database/` (sync SQLAlchemy on SQLite) underneath. Pydantic request/response schemas live in `backend/models.py`.
- Engine registry is `backends/__init__.py`: `TTS_ENGINES` (qwen, qwen_custom_voice, luxtts, chatterbox, chatterbox_turbo, tada, kokoro), per-engine `ModelConfig` lists (HF repo ids, sizes), and the `get_tts_backend_for_engine` factory (one lazily created singleton per engine). `utils/platform_detect.get_backend_type()` returns `mlx` on Apple Silicon (Qwen3-TTS, Whisper and the LLM have MLX variants) else `pytorch`; `backends/base.get_torch_device()` tries cuda, xpu, directml, mps, cpu behind per-engine flags, and several engines force CPU on macOS. Cloning-capable engines are `services/profiles.py::CLONING_ENGINES`; kokoro and qwen_custom_voice are preset-voice engines.
- Generation: `POST /generate` writes a `generations` row, then `services/task_queue.enqueue_generation`; a single asyncio worker runs `services/generation.py`, which lazily loads the engine, splits long text in `utils/chunked_tts.py`, applies effects/normalize, and writes WAV under `{data}/generations`. Clients follow progress over SSE at `GET /generate/{id}/status`. `POST /generate/stream` instead runs `services/generation.py::run_generation_stream` as a `stream-*` job in the same queue and sends PCM16 per sentence chunk as it is synthesized (`utils/chunked_tts.py::generate_chunked_stream`, `utils/wav_stream.py`); it writes no DB row. All other live updates are SSE (`EventSource`); there are no WebSockets.
- Data dir: `config.py` defaults to `Path("data")` **relative to CWD** (`<repo>/data/` in dev); the Tauri sidecar passes `--data-dir`. DB is `{data}/voicebox.db`. Schema changes are hand-written idempotent migrations in `database/migrations.py`; Alembic is deliberately not used (the module docstring explains why and lists the steps).
- STT is Whisper via `services/transcribe.py`; the LLM is Qwen3 via `backends/qwen_llm_backend.py`, optional, used for personality compose/rewrite and capture refinement. Neither goes through the generation queue.
- GPU server variants are separate binaries the desktop app downloads at runtime (`services/cuda.py`, `services/rocm.py`, `routes/cuda.py`, `routes/rocm.py`).

### Frontend (`app/src`)
- The API client is hand-written: `lib/api/client.ts` + `lib/api/types.ts`. The generated `lib/api/{core,models,schemas,services}` folders are stale `just generate-api` output that nothing imports; do not extend them. `docs/openapi.json` is likewise a stale spec kept for the docs site.
- Server URL comes from `stores/serverStore.ts` (persisted): dev uses `http://127.0.0.1:17493`; web production uses `window.location.origin`; in Tauri, `App.tsx` overwrites it with the sidecar URL on every launch.
- State: zustand stores in `stores/` (`serverStore`, `uiStore`, `audioChannelStore` persist to localStorage); server data via TanStack Query hooks in `lib/hooks/`; TanStack Router in `router.tsx`; shadcn/Radix primitives in `components/ui/`; Tailwind v4 configured in `index.css` (no tailwind.config); i18next locales in `i18n/`.
- Platform differences go through the `Platform` interface in `platform/types.ts`, read with `usePlatform()` from `platform/PlatformContext.tsx`; components branch on `platform.metadata.isTauri`. The dictation/capture feature (`DictateWindow`, `CapturesTab`, the permission gates, `useChordSync`, `useCaptureRecordingSession`) is the existing exception that imports `@tauri-apps/api` directly; keep new platform-specific behavior behind `Platform`.
- Mic capture happens in the webview (`lib/hooks/useAudioRecording.ts`), not in Rust. `?view=dictate` mounts `components/DictateWindow` (the floating pill) instead of the main app. `CHANGELOG.md` is bundled into the UI at build time by `app/plugins/changelog.ts`.

### Desktop shell (`tauri/src-tauri/src`)
`main.rs` spawns the `voicebox-server` sidecar with `--data-dir --port 17493 --parent-pid`, and reuses any process already listening on 17493 that passes `/health`; that is how `just dev` attaches the shell to the uvicorn it started (the placeholder sidecar only exists so the Rust build compiles). Other modules: `hotkey_monitor.rs` (global chord via keytap), `focus_capture.rs` / `clipboard.rs` / `synthetic_keys.rs` (auto-paste; macOS and Windows only, the Linux functions are stubs), `audio_capture/` (system audio per OS), `audio_output.rs` (multi-device playback), `speak_monitor.rs` (relays `/events/speak` SSE to the pill). Port 17493 is `SERVER_PORT` there and repeated in the justfile, `serverStore.ts`, CORS defaults, `.mcp.json` and Docker; change all or none.

### MCP
`backend/mcp_server/` is a FastMCP server mounted at `/mcp` (Streamable HTTP) exposing `voicebox.speak`, `voicebox.transcribe`, `voicebox.list_captures`, `voicebox.list_profiles`; `speak` calls `routes.generations.generate_speech` in-process, and `POST /speak` is its REST twin. `ClientIdMiddleware` copies the `X-Voicebox-Client-Id` header into a ContextVar, which drives per-client voice bindings (`routes/mcp_bindings.py`, `mcp_server/resolve.py`). `backend/mcp_shim/` is a stdio-to-HTTP proxy built as the `voicebox-mcp` sidecar (env `VOICEBOX_HOST`, `VOICEBOX_PORT`, `VOICEBOX_CLIENT_ID`). The root `.mcp.json` points Claude Code at `http://127.0.0.1:17493/mcp`; it only works while a backend is running.

## Conventions and gotchas
- Run backend commands from the repo root. `backend/` uses relative imports, so `cd backend && uvicorn main:app` (as `docs/.../setup.mdx`, `remote-mode.mdx` and `scripts/generate-api.sh` say) fails on import.
- `just db-reset` deletes `backend/data/voicebox.db`, but the dev DB is `<repo>/data/voicebox.db`; remove it by hand.
- Env vars the backend actually reads: `VOICEBOX_API_KEY`, `VOICEBOX_API_KEY_FILE`, `VOICEBOX_API_KEYS_JSON`, `VOICEBOX_DISABLE_DOCS`, `VOICEBOX_RATE_LIMITING`, `VOICEBOX_MAX_QUEUE_DEPTH`, `VOICEBOX_MAX_BODY_MB`, `VOICEBOX_MEDIA_TOKEN_TTL` (all in `auth/settings.py`), `VOICEBOX_MODELS_DIR` (sets `HF_HUB_CACHE`), `VOICEBOX_CORS_ORIGINS` (comma-separated extras), `VOICEBOX_OFFLINE_PATCH=0` (disables `utils/hf_offline_patch.py`), `VOICEBOX_BACKEND_VARIANT`, `VOICEBOX_CLOUD_URL` / `VOICEBOX_CLOUD_API_URL`, plus `HF_HUB_OFFLINE`, `HSA_OVERRIDE_GFX_VERSION`, `MIOPEN_*`, and uvicorn's `FORWARDED_ALLOW_IPS`. Docs and `docker-compose.yml` mention `VOICEBOX_DATA_DIR`, `VOICEBOX_FORCE_CPU` and `LOG_LEVEL`; the code does not read them (use `--data-dir`).
- All TTS inference goes through `services/task_queue.py` (one worker; GPU work is serial), including `/generate/stream`. Never call a backend's `generate()` from a route. Cancelling a job waits for the in-flight engine call to finish (`utils/chunked_tts.py::_await_inflight`) because the worker thread cannot be interrupted; a backend may offer an optional `generate_stream` async generator for sub-sentence streaming (only `MLXTTSBackend` does).
- Models load lazily on first use and are never evicted across engines; every engine touched stays resident until `POST /models/{name}/unload`. `POST /models/download` also loads the model into memory.
- Every endpoint requires `Authorization: Bearer <key>`; `backend/auth/` enforces it as pure-ASGI middleware (stack: CORS → security headers → auth → rate limit → body limit → `ClientIdMiddleware`), so the `/mcp` mount, docs routes and unknown paths are covered too. Roles are `admin` (desktop UI, operators) and `client` (the user's apps); `backend/auth/policy.py` is the only place routes are classified and `tests/test_route_classification.py` fails when a new route is missing from it. Public without a key: `GET /health` (minimal body), `GET /cloud/callback`, the docs routes (unless `VOICEBOX_DISABLE_DOCS=1`) and, in Docker, the SPA shell. Keys come from `VOICEBOX_API_KEY`, else the auto-created `{data}/api_key` (admin; the desktop shell reads the same file), plus the hashed `{data}/api_keys.json` managed by `python -m backend.keys` and `/auth/keys`. Browser media and SSE loads use `POST /auth/media-token` and `?token=` (GET only, allowlisted paths). Keys are created in the lifespan, never at import, because `server.py`/`main.py` import the app before `--data-dir` is applied.
- Routes needing the caller use `get_principal()` (ContextVar set by the middleware); MCP tools read it from `get_http_request().scope["state"]["principal"]` instead because a stateful session's tool calls run in the task spawned at `initialize`. Queue caps and per-key limits are charged in the routes (`ensure_capacity`, `charge("tts_chars", ...)`); Whisper and the LLM go through `services/inference_slots.py` because they bypass the generation queue.
- `GenerationRequest.engine` defaults to `"qwen"`, so `profile.default_engine` only applies when a client sends an explicit `null` (as `/speak` does). A preset-voice profile called without `engine` gets a 400.
- FastAPI route order matters: register static paths before `/{id}` routes in the same module (e.g. `DELETE /history/failed` before `/history/{generation_id}`).
- Python style is `backend/STYLE_GUIDE.md`: heavy imports (torch, transformers, mlx) go inside functions with `# lazy: heavy import`; relative imports inside the package; `logging` with `%s` args, not `print`; a reason after every `noqa` / `type: ignore`; no ASCII section dividers.
- Tests: pytest reads `backend/pyproject.toml` (`asyncio_mode = "auto"`, no `conftest.py`) and tests import `backend.*`, another reason to run from the root. `backend/tests/` mixes unit tests with manual scripts; some (e.g. `test_generation_download.py`) expect a live server and fail without one. `test_cors.py` and the `test_auth_*` / `test_ratelimit.py` tests run without torch (they build a mini app with `backend/auth/install.py` via `tests/security_testapp.py`).
- Adding a TTS engine touches `backends/<name>_backend.py`, `backends/__init__.py` (ModelConfig, `TTS_ENGINES`, factory), the engine regex in `backend/models.py`, `requirements.txt`, the justfile, `release.yml`, `Dockerfile`, `build_binary.py` (+ `server.py` for frozen-build env vars), and the frontend engine lists (`lib/api/types.ts`, `lib/constants/languages.ts`, `components/Generation/EngineModelSelector.tsx`, `lib/hooks/useGenerationForm.ts`, `components/ServerSettings/ModelManagement.tsx`). Follow `.agents/skills/add-tts-engine/SKILL.md`; models that work under uvicorn routinely break in the frozen binary.
- Python target is 3.12 (`requires-python >= 3.12`, CI uses 3.12); the `Dockerfile` still builds on `python:3.11-slim`.
- Versioning/release: `.bumpversion.cfg` (currently 0.5.0) rewrites `tauri.conf.json`, `Cargo.toml`, every `package.json` and `backend/__init__.py`; never hand-edit versions. `CHANGELOG.md` is generated: update it only via the `draft-release-notes` skill and cut releases with `release-bump`. Pushing a `v*` tag runs `release.yml` (macOS arm64 MLX, macOS x64, Windows; no Linux).
- Repo skills live in `.agents/skills/` (`add-tts-engine`, `draft-release-notes`, `release-bump`, `triage-prs`), not `.claude/`, so open the relevant `SKILL.md` explicitly when doing those tasks.
- Commits follow Conventional Commits: `fix(scope): summary (#PR)`, `feat(...)`, `docs: ...`.

## Where to read more
- `backend/STYLE_GUIDE.md` (authoritative Python conventions), `backend/README.md`, `backend/mcp_server/README.md`.
- `docs/content/docs/developer/*.mdx`: `architecture`, `setup`, `building`, `tts-engines` (engine guide with PyInstaller pitfalls), `model-management`, `tts-generation`, `transcription`, `voice-profiles`, `effects-pipeline`.
- `docs/PROJECT_STATUS.md` is the living roadmap and known-bottleneck list; `docs/plans/*.md` hold design docs (MCP, Docker, OpenAI-compatible API, cloud).
- `backend/tests/E2E_MODEL_TEST_DESIGN.md` explains `just test-models`.
