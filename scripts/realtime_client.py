#!/usr/bin/env python3
"""Stream a WAV file to ``/v1/realtime/transcription`` at real-time pace and print the events.

    scripts/realtime_client.py --url ws://127.0.0.1:17493 --key vbx_... clip.wav [--model whisper-turbo]
                              [--language en] [--no-vad] [--speed 1.0]

The file is sent as PCM16 in 100 ms chunks (its own sample rate is declared
to the server); ``--no-vad`` sends one manual commit at the end instead of
letting the server detect turns.  Only ``websockets`` (already installed
with uvicorn) and ``soundfile`` are needed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time

import numpy as np
import soundfile as sf


def _append(pcm: np.ndarray) -> str:
    return json.dumps(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm.tobytes()).decode(),
        }
    )


async def run(args: argparse.Namespace) -> int:
    import websockets

    audio, rate = sf.read(args.file, dtype="int16", always_2d=True)
    pcm = audio[:, 0]  # mono
    url = args.url.rstrip("/") + "/v1/realtime/transcription"
    transcripts: list[str] = []
    pending: set[str] = set()  # committed utterances whose final has not arrived
    settled = asyncio.Event()  # set whenever pending becomes empty
    started = time.monotonic()

    async with websockets.connect(
        url, additional_headers={"Authorization": f"Bearer {args.key}"}
    ) as socket:
        print("<", json.loads(await socket.recv())["type"])
        session = {
            "input_sample_rate": int(rate),
            "input_audio_transcription": {
                "model": args.model,
                "language": args.language,
            },
            "turn_detection": None if args.no_vad else {"type": "server_vad"},
        }
        await socket.send(
            json.dumps({"type": "transcription_session.update", "session": session})
        )

        async def reader() -> None:
            async for message in socket:
                event = json.loads(message)
                kind = event["type"]
                if kind.endswith("transcription.delta"):
                    print(event["delta"], end="", flush=True)
                elif kind.endswith(("transcription.completed", "transcription.failed")):
                    if kind.endswith("completed"):
                        print(
                            f"\n< completed [{event['item_id']}]: {event['transcript']}"
                        )
                        transcripts.append(event["transcript"])
                    else:
                        print(f"\n< failed [{event['item_id']}]: {event['error']}")
                    pending.discard(event["item_id"])
                    if not pending:
                        settled.set()
                elif kind == "input_audio_buffer.committed":
                    pending.add(event["item_id"])
                    settled.clear()
                    print("<", kind, event["item_id"])
                elif kind == "error":
                    print("\n< error:", event["error"])
                else:
                    print(
                        "<",
                        kind,
                        {
                            k: v
                            for k, v in event.items()
                            if k not in ("type", "session")
                        },
                    )

        reader_task = asyncio.create_task(reader())
        chunk = int(rate * 0.1)
        for i in range(0, len(pcm), chunk):
            await socket.send(_append(pcm[i : i + chunk]))
            target = started + (i + chunk) / rate / args.speed
            await asyncio.sleep(max(0.0, target - time.monotonic()))
        if args.no_vad:
            await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        else:
            # Trailing silence lets the server close the last utterance.
            for _ in range(12):
                await socket.send(_append(np.zeros(chunk, dtype="<i2")))
                await asyncio.sleep(0.1)
        # Then wait for the final transcript of every committed utterance.
        await asyncio.sleep(0.5)
        if pending or not settled.is_set():
            try:
                await asyncio.wait_for(settled.wait(), timeout=args.wait)
            except TimeoutError:
                print(
                    f"\n! gave up after {args.wait:.0f} s with {len(pending)} utterance(s) still pending"
                )
        reader_task.cancel()
    print(f"\n{len(transcripts)} utterance(s) in {time.monotonic() - started:.1f} s")
    return 0 if transcripts else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", help="WAV (or any soundfile-readable) file to stream")
    parser.add_argument("--url", default="ws://127.0.0.1:17493")
    parser.add_argument("--key", required=True)
    parser.add_argument("--model", default="whisper-1")
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--no-vad", action="store_true", help="manual turn: commit once at the end"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="send faster than real time (2 = twice as fast)",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=120.0,
        help="seconds to wait for the outstanding finals",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
