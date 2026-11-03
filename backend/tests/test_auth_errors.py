"""How the 500 handler logs: a broken stream is one warning line, anything else a traceback (torch-free)."""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import AsyncIterator

import pytest
from fastapi.responses import StreamingResponse
from starlette.testclient import TestClient

from backend.auth import errors
from backend.tests.security_testapp import build_test_app


class StandInAbortError(RuntimeError):
    """Plays ``services.generation.StreamAbortedError``, whose module needs the audio stack."""


@pytest.fixture(params=["stand-in", "real"])
def aborted_error(request, monkeypatch) -> type[Exception]:
    if request.param == "real":
        generation = pytest.importorskip("backend.services.generation", reason="needs the audio stack")
        return generation.StreamAbortedError
    module = types.ModuleType(errors._GENERATION_MODULE)
    module.StreamAbortedError = StandInAbortError
    monkeypatch.setitem(sys.modules, errors._GENERATION_MODULE, module)
    return StandInAbortError


@pytest.fixture
def harness(tmp_path):
    app, _runtime = build_test_app(tmp_path)
    raised: dict[str, Exception] = {}

    async def stream():
        async def body() -> AsyncIterator[bytes]:
            yield b"first chunk"
            raise raised["error"]

        return StreamingResponse(body(), media_type="audio/pcm")

    app.add_api_route("/stream", stream)
    client = TestClient(app, raise_server_exceptions=False)
    key = (tmp_path / "api_key").read_text().strip()

    def get(path: str, error: Exception | None = None):
        if error is not None:
            raised["error"] = error
        return client.get(path, headers={"Authorization": f"Bearer {key}", "X-Request-Id": "req-1"})

    return get


def handler_records(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == errors.logger.name]


def test_a_stream_aborted_mid_way_is_logged_as_one_line_without_a_traceback(harness, aborted_error, caplog):
    caplog.set_level(logging.INFO)
    error = aborted_error("Stream stream-1 failed mid-stream: RuntimeError('engine exploded')")
    response = harness("/stream", error)

    assert response.status_code == 200  # the error came after the status line
    [record] = handler_records(caplog)
    assert record.levelno == logging.WARNING
    assert record.exc_info is None
    assert record.error_id == "req-1"
    assert record.getMessage() == (
        "Streamed response req-1 aborted: GET /stream: Stream stream-1 failed mid-stream: RuntimeError('engine exploded')"
    )


@pytest.mark.usefixtures("aborted_error")  # with the abort class loaded, so the lookup is exercised
@pytest.mark.parametrize(
    ("path", "error"),
    [
        pytest.param("/boom", None, id="before-the-response"),
        pytest.param("/stream", RuntimeError("encoder failed"), id="mid-stream"),
    ],
)
def test_every_other_error_keeps_its_traceback(harness, caplog, path, error):
    caplog.set_level(logging.INFO)
    response = harness(path, error)

    if error is None:
        assert response.status_code == 500
        assert response.json()["error_id"] == "req-1"
    [record] = handler_records(caplog)
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert type(record.exc_info[1]) is RuntimeError
    assert record.getMessage() == f"Unhandled error req-1: GET {path}"


def test_nothing_is_a_stream_abort_while_the_generation_service_is_not_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, errors._GENERATION_MODULE, raising=False)
    assert not errors._is_stream_abort(StandInAbortError("x"))
    assert errors._GENERATION_MODULE not in sys.modules  # looked up, never imported
