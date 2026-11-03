"""Echo stand-ins for the LLM: the agent answers "You said: <what it heard>".

So the agent's spoken reply carries the user's transcript, and one
transcription of the recorded reply checks Voicebox's STT and TTS together.
``EchoLLM`` plugs into an AgentSession (e2e_agent.py); ``EchoChatServer`` is
a minimal OpenAI-compatible chat endpoint for running the unmodified
../agent.py with ``LLM_BASE_URL`` pointed at it. Both append *tail* to the
first reply of a call, which gives the barge-in check a reply long enough to
talk over.

Run on its own, it serves the chat endpoint until interrupted, for trying
../agent.py without a real model:

    python echo.py [--port N]   # prints the LLM_BASE_URL to use
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    llm,
    utils,
)
from livekit.agents.llm import ChatChunk, ChoiceDelta

TOKEN_DELAY_S = 0.01


def reply_for(user_messages: list[str], tail: str = "") -> list[str]:
    """The reply to the latest user message, as text deltas.

    The echo comes word by word, like a model's tokens. *tail*, added to the
    first reply only, comes a sentence per delta, faster than any TTS speaks
    it, so the speech requests never wait for text.
    """
    heard = user_messages[-1].strip() if user_messages else ""
    words = (f"You said: {heard}" if heard else "I didn't catch that.").split(" ")
    deltas = [w if i == 0 else f" {w}" for i, w in enumerate(words)]
    if tail and len(user_messages) == 1:
        deltas += [f" {sentence.strip()}." for sentence in tail.split(".") if sentence.strip()]
    return deltas


class EchoLLM(llm.LLM):
    """An LLM that streams the deltas of :func:`reply_for`."""

    def __init__(self, *, tail: str = "") -> None:
        super().__init__()
        self.tail = tail

    @property
    def model(self) -> str:
        return "echo"

    @property
    def provider(self) -> str:
        return "voicebox-e2e"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        return _EchoStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _EchoStream(llm.LLMStream):
    async def _run(self) -> None:
        echo: EchoLLM = self._llm  # type: ignore[assignment]  # set by LLMStream.__init__
        users = [m.text_content or "" for m in self._chat_ctx.messages() if m.role == "user"]
        request_id = utils.shortuuid("echo_")
        for text in reply_for(users, echo.tail):
            await asyncio.sleep(TOKEN_DELAY_S)
            delta = ChoiceDelta(role="assistant", content=text)
            self._event_ch.send_nowait(ChatChunk(id=request_id, delta=delta))


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # [{"type": "text", "text": ...}, ...]
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class _ChatHandler(BaseHTTPRequestHandler):
    server: EchoChatServer

    def log_message(self, *args: Any) -> None:
        """Keep the driver's output clean: no access log."""

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._send_json(
                {"object": "list", "data": [{"id": "echo", "object": "model", "owned_by": "e2e"}]}
            )
        else:
            self._send_json({"error": {"message": "not found"}}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send_json({"error": {"message": "not found"}}, status=404)
            return
        users = [
            _text_of(m.get("content")) for m in body.get("messages", []) if m.get("role") == "user"
        ]
        deltas = reply_for(users, self.server.tail)
        text = "".join(deltas)
        self.server.requests += 1
        base = {
            "id": "chatcmpl-echo",
            "created": int(time.time()),
            "model": body.get("model", "echo"),
        }
        if not body.get("stream"):
            message = {"role": "assistant", "content": text}
            choice = {"index": 0, "message": message, "finish_reason": "stop"}
            self._send_json({**base, "object": "chat.completion", "choices": [choice]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        base["object"] = "chat.completion.chunk"
        for i, piece in enumerate(deltas):
            delta = {"content": piece, **({"role": "assistant"} if i == 0 else {})}
            self._send_event(
                {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            )
            time.sleep(TOKEN_DELAY_S)
        self._send_event({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        if (body.get("stream_options") or {}).get("include_usage"):
            usage = {
                "prompt_tokens": 1,
                "completion_tokens": len(text.split()),
                "total_tokens": 1 + len(text.split()),
            }
            self._send_event({**base, "choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _send_event(self, payload: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class EchoChatServer(ThreadingHTTPServer):
    """``/v1/chat/completions`` (streamed or not) and ``/v1/models`` on 127.0.0.1, in a thread."""

    daemon_threads = True

    def __init__(self, *, tail: str = "", port: int = 0) -> None:
        super().__init__(("127.0.0.1", port), _ChatHandler)
        self.tail = tail
        self.requests = 0
        self._thread = threading.Thread(target=self.serve_forever, name="echo-llm", daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"

    def start(self) -> EchoChatServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


def main() -> None:
    """Serve the echo chat endpoint in the foreground until Ctrl-C."""
    ap = argparse.ArgumentParser(description="An OpenAI-compatible chat API that echoes the user.")
    ap.add_argument("--port", type=int, default=0, help="port on 127.0.0.1; default: any free one")
    ap.add_argument("--tail", default="", help="text appended to the first reply of each call")
    args = ap.parse_args()
    server = EchoChatServer(tail=args.tail, port=args.port)
    print(f"echo chat API: LLM_BASE_URL={server.base_url} LLM_MODEL=echo", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
