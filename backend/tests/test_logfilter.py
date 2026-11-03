"""Media tokens and OAuth codes stay out of uvicorn's logs: access lines and WebSocket handshakes."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from backend.auth.logfilter import RedactQueryTokenFilter, install_log_redaction
from backend.tests.security_testapp import build_test_app

SECRET = "ZW52fGFkbWlufDE3OTIy.c2lnbmF0dXJl"


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def captured():
    """What a handler of uvicorn's access and error loggers is given."""
    handler = Capture()
    loggers = [logging.getLogger(name) for name in ("uvicorn.access", "uvicorn.error")]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    yield handler.lines
    for logger, level in zip(loggers, levels, strict=True):
        logger.removeHandler(handler)
        logger.setLevel(level)


def test_uvicorns_request_and_handshake_lines_are_redacted(captured):
    install_log_redaction()
    install_log_redaction()  # idempotent: one filter per logger
    access, error = logging.getLogger("uvicorn.access"), logging.getLogger("uvicorn.error")
    assert sum(isinstance(f, RedactQueryTokenFilter) for f in error.filters) == 1
    # The calls uvicorn makes: its HTTP access line, and the handshake lines of its WebSocket protocols.
    access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", f"/audio/x?token={SECRET}", "1.1", 200)
    error.info('%s - "WebSocket %s" [accepted]', "127.0.0.1:5001", f"/v1/realtime?intent=transcription&token={SECRET}")
    error.info('%s - "WebSocket %s" 403', "127.0.0.1:5002", f"/v1/realtime/transcription?token={SECRET}&model=x")
    error.info('%s - "WebSocket %s" %d', "127.0.0.1:5003", f"/cloud/callback?code={SECRET}&state={SECRET}", 500)
    assert captured == [
        '127.0.0.1:5000 - "GET /audio/x?token=[redacted] HTTP/1.1" 200',
        '127.0.0.1:5001 - "WebSocket /v1/realtime?intent=transcription&token=[redacted]" [accepted]',
        '127.0.0.1:5002 - "WebSocket /v1/realtime/transcription?token=[redacted]&model=x" 403',
        '127.0.0.1:5003 - "WebSocket /cloud/callback?code=[redacted]&state=[redacted]" 500',
    ]


def test_a_browser_socket_opened_with_a_media_token_does_not_log_it(tmp_path, captured):
    """The handshake line uvicorn itself writes, accepted and refused, for ``?token=`` on a realtime path."""
    app, _runtime = build_test_app(tmp_path)
    admin = (tmp_path / "api_key").read_text().strip()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, log_level="info", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started, "uvicorn did not start"
        host = f"127.0.0.1:{listener.getsockname()[1]}"
        minted = httpx.post(f"http://{host}/auth/media-token", headers={"Authorization": f"Bearer {admin}"})
        token = minted.json()["token"]
        with connect(f"ws://{host}/v1/realtime/transcription?token={token}") as ws:
            assert json.loads(ws.recv())["via"] == "token"
        with pytest.raises(InvalidStatus), connect(f"ws://{host}/v1/realtime/transcription?token={SECRET}"):
            pass
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()

    handshakes = [line for line in captured if '"WebSocket ' in line]
    assert len(handshakes) == 2, captured
    assert handshakes[0].endswith('"WebSocket /v1/realtime/transcription?token=[redacted]" [accepted]')
    assert handshakes[1].endswith('"WebSocket /v1/realtime/transcription?token=[redacted]" 403')
    assert not [line for line in captured if token in line or SECRET in line]
