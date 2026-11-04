"""Stream text to Voicebox and save complete PCM output as a WAV.

Install ``websockets>=13`` in a client environment, then run:
  VOICEBOX_CLIENT_KEY=... python scripts/speech_client.py --text "Hello from Voicebox." --out /tmp/hello.wav
  printf 'First sentence.\nSecond sentence.\n' | python scripts/speech_client.py --stdin --out /tmp/turns.wav

Each input line is one flush. Text arrives in small append messages; audio is
received concurrently. A rejected/cancelled flush fails the run and leaves only
the .partial.wav file, never a seemingly complete output.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import wave
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def socket_url(url: str) -> str:
    """Convert a server root into its incremental speech endpoint."""
    parts = urlsplit(url.rstrip("/"))
    if parts.scheme not in {"http", "https", "ws", "wss"} or not parts.hostname:
        raise ValueError("--url must be an HTTP or WebSocket server URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("--url must not contain credentials, a query, or a fragment")
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    path += "/v1/audio/speech/stream"
    return urlunsplit(("wss" if parts.scheme in {"https", "wss"} else "ws", parts.netloc, path, "", ""))


async def run(args) -> None:
    """Append incrementally, flush each line, and publish the WAV only after every done event."""
    from websockets.asyncio.client import connect

    key = os.getenv("VOICEBOX_CLIENT_KEY", "").strip()
    if not key:
        raise ValueError("Set VOICEBOX_CLIENT_KEY to a Voicebox client key")
    settings = {"voice": args.voice, "model": args.model}
    if args.pronunciations:
        settings["pronunciation_replacements"] = json.loads(Path(args.pronunciations).read_text())
    destination = Path(args.out)
    partial = destination.with_name(destination.stem + ".partial.wav")
    async with connect(
        socket_url(args.url), additional_headers={"Authorization": f"Bearer {key}"}, max_size=128 * 1024
    ) as socket:
        first = json.loads(await socket.recv())
        if first.get("type") != "session.created":
            raise RuntimeError(f"Session rejected: {first.get('error', first)}")
        await socket.send(json.dumps({"type": "session.update", "session": settings}))
        updated = json.loads(await socket.recv())
        if updated.get("type") != "session.updated":
            raise RuntimeError(f"Settings rejected: {updated.get('error', updated)}")
        completed: asyncio.Queue[str] = asyncio.Queue()
        expected_sequence: dict[str, int] = {}

        with wave.open(str(partial), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)

            async def receive():
                async for raw in socket:
                    event = json.loads(raw)
                    kind = event.get("type")
                    flush_id = event.get("flush_id")
                    if kind == "audio.started":
                        if event.get("sample_rate") != 24000 or event.get("format") != "pcm_s16le":
                            raise RuntimeError("Server returned an unexpected audio format")
                        expected_sequence[flush_id] = 0
                    elif kind == "audio.delta":
                        if expected_sequence.get(flush_id) != event["sequence"]:
                            raise RuntimeError("Audio chunks arrived out of order")
                        expected_sequence[flush_id] += 1
                        audio = base64.b64decode(event["audio"], validate=True)
                        if len(audio) % 2:
                            raise RuntimeError("Server returned an incomplete PCM sample")
                        wav.writeframesraw(audio)
                    elif kind == "audio.done":
                        expected_sequence.pop(flush_id, None)
                        await completed.put(flush_id)
                    elif kind in {"error", "audio.cancelled"}:
                        raise RuntimeError(f"Speech failed: {event.get('error', kind)}")
                    elif kind == "session.closed":
                        return

            async def send_lines():
                index = 0
                while True:
                    line = await asyncio.to_thread(sys.stdin.readline) if args.stdin else args.text
                    if not line:
                        break
                    if line.strip():
                        index += 1
                        flush_id = f"line-{index}"
                        for offset in range(0, len(line), args.chunk_chars):
                            await socket.send(
                                json.dumps(
                                    {"type": "input_text.append", "text": line[offset : offset + args.chunk_chars]}
                                )
                            )
                        await socket.send(json.dumps({"type": "input_text.flush", "flush_id": flush_id}))
                        received_id = await asyncio.wait_for(completed.get(), timeout=args.timeout)
                        if received_id != flush_id:
                            raise RuntimeError("Unexpected completed flush ID")
                        print(f"Completed {flush_id}", file=sys.stderr)
                    if not args.stdin:
                        break
                if index == 0:
                    raise ValueError("Provide non-empty text")
                await socket.send(json.dumps({"type": "session.close"}))

            receiver = asyncio.create_task(receive())
            sender = asyncio.create_task(send_lines())
            try:
                done, _ = await asyncio.wait([receiver, sender], return_when=asyncio.FIRST_COMPLETED)
                if receiver in done:
                    await receiver
                    if not sender.done():
                        raise RuntimeError("Server closed before all text was spoken")
                await sender
                await asyncio.wait_for(receiver, 5)
            finally:
                for task in (sender, receiver):
                    task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
    partial.replace(destination)
    print(f"Saved {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("VOICEBOX_URL", "http://127.0.0.1:17493"))
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--model", default="tts-1")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--stdin", action="store_true", help="Read one utterance per input line")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pronunciations", help="JSON file mapping literal terms to their pronunciations")
    parser.add_argument("--chunk-chars", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.chunk_chars < 1 or args.chunk_chars > 10000 or args.timeout <= 0:
        parser.error("--chunk-chars must be 1-10000 and --timeout must be positive")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        sys.exit(str(exc))
