"""A torch-free FastAPI app with stub routes at the real paths, wired through ``install_security``.

Used by the ``test_auth_*``, ``test_ratelimit.py`` and related tests so the
middlewares are exercised end to end without models or a database.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend import lifecycle
from backend.auth.install import SecurityRuntime, charge, install_security
from backend.auth.principal import Principal, current_principal
from backend.auth.settings import SecuritySettings


def build_test_app(
    tmp_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    frontend_dir: Path | None = None,
) -> tuple[FastAPI, SecurityRuntime]:
    environ = {
        "VOICEBOX_API_KEY_FILE": str(tmp_path / "api_key"),
        "VOICEBOX_API_KEYS_JSON": str(tmp_path / "api_keys.json"),
    }
    environ.update(env or {})
    settings = SecuritySettings.from_env(frontend_dir=frontend_dir, environ=environ)
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "voicebox API"}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "model_loaded": False, "gpu_available": True}

    @app.get("/health/ready")
    async def ready():
        ok, body = lifecycle.readiness(True)
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/profiles")
    async def profiles():
        return [{"id": "p1"}]

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        charge("tts_chars", len(body.get("text", "")))
        return {"chars": len(body.get("text", ""))}

    @app.get("/audio/{generation_id}")
    async def audio(generation_id: str):
        return PlainTextResponse(f"audio:{generation_id}")

    @app.get("/generate/{generation_id}/status")
    async def status(generation_id: str):
        async def events() -> AsyncIterator[bytes]:
            yield b'data: {"status": "completed"}\n\n'
            await asyncio.sleep(0)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        data = await file.read()
        return {"bytes": len(data)}

    @app.get("/cloud/callback")
    async def cloud_callback():
        return {"ok": True}

    @app.get("/captures")
    async def captures():
        return {"items": []}

    @app.get("/captures/{capture_id}/audio")
    async def capture_audio(capture_id: str):
        return PlainTextResponse(f"capture:{capture_id}")

    @app.post("/shutdown")
    async def shutdown():
        return {"message": "bye"}

    @app.get("/mcp/bindings")
    async def bindings():
        return []

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom with /secret/path")

    @app.get("/who")
    async def who(principal: Principal = Depends(current_principal)):
        return {"key_id": principal.key_id, "role": principal.role, "via": principal.via}

    async def mcp_endpoint(request: Request):
        return JSONResponse({"mcp": request.method})

    app.mount("/mcp", Starlette(routes=[Route("/", mcp_endpoint, methods=["GET", "POST"])]))

    runtime = install_security(app, settings)
    runtime.startup()
    return app, runtime
