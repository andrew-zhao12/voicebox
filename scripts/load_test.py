#!/usr/bin/env python3
"""Load-test a Voicebox server through ``POST /v1/audio/speech``.

    backend/venv/bin/python scripts/load_test.py --url http://127.0.0.1:17493 \\
        --key vbx_... --voice Narrator --model kokoro --concurrency 4 --requests 20

Fires ``--requests`` requests with ``--concurrency`` in flight, streams every
response and reports time to first byte, total time, realtime factor (audio
seconds produced per wall second; ``wav``/``pcm`` only, from the byte count and
the ``X-Voicebox-Sample-Rate`` header) and the 429/5xx counts.  Uses httpx,
which the backend already depends on, so run it from the project venv.  Put
the numbers into docs/content/docs/overview/remote-mode.mdx when you change
the performance table.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass

import httpx

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog while the sun sets behind the hills.",
    "Voicebox turns text into speech with the voice profile you choose, one sentence at a time.",
    "Our meeting starts at nine tomorrow; please bring the quarterly numbers and the revised plan.",
    "Streaming synthesis sends audio as soon as the first sentence is ready, so playback starts early.",
]


@dataclass
class Result:
    status: int
    ttfb_s: float
    total_s: float
    audio_s: float | None
    error: str | None = None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


async def one_request(client: httpx.AsyncClient, args, text: str) -> Result:
    body = {
        "model": args.model,
        "voice": args.voice,
        "input": text,
        "response_format": args.format,
    }
    started = time.perf_counter()
    ttfb = None
    size = 0
    try:
        async with client.stream("POST", "/v1/audio/speech", json=body) as response:
            sample_rate = int(response.headers.get("x-voicebox-sample-rate", "0") or 0)
            async for chunk in response.aiter_bytes():
                if ttfb is None:
                    ttfb = time.perf_counter() - started
                size += len(chunk)
            total = time.perf_counter() - started
            if response.status_code != 200:
                return Result(
                    response.status_code,
                    ttfb or total,
                    total,
                    None,
                    error=f"HTTP {response.status_code}",
                )
            audio_s = None
            if args.format in ("wav", "pcm") and sample_rate:
                payload = size - (44 if args.format == "wav" else 0)
                audio_s = payload / 2 / sample_rate
            return Result(200, ttfb or total, total, audio_s)
    except httpx.HTTPError as e:
        total = time.perf_counter() - started
        return Result(0, ttfb or total, total, None, error=type(e).__name__)


def load_texts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_TEXTS
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()] or DEFAULT_TEXTS


async def run(args, texts: list[str]) -> int:
    headers = {"Authorization": f"Bearer {args.key}"}
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    async with httpx.AsyncClient(
        base_url=args.url, headers=headers, timeout=args.timeout, limits=limits
    ) as client:
        if args.warmup:
            print(f"warm-up: {args.warmup} request(s)", file=sys.stderr)
            for i in range(args.warmup):
                warm = await one_request(client, args, texts[i % len(texts)])
                if warm.status != 200:
                    print(f"warm-up failed: {warm.error}", file=sys.stderr)
                    return 1

        semaphore = asyncio.Semaphore(args.concurrency)

        async def guarded(index: int) -> Result:
            async with semaphore:
                return await one_request(client, args, texts[index % len(texts)])

        wall_started = time.perf_counter()
        results = await asyncio.gather(*(guarded(i) for i in range(args.requests)))
        wall = time.perf_counter() - wall_started

    ok = [r for r in results if r.status == 200]
    rate_limited = sum(1 for r in results if r.status == 429)
    failed = [r for r in results if r.status not in (200, 429)]
    ttfb = [r.ttfb_s for r in ok]
    total = [r.total_s for r in ok]
    rtf = [r.audio_s / r.total_s for r in ok if r.audio_s]
    audio_total = sum(r.audio_s or 0 for r in ok)

    print(
        f"target        {args.url}  model={args.model} voice={args.voice!r} format={args.format}"
    )
    print(
        f"requests      {args.requests} total, {args.concurrency} concurrent, {len(ok)} ok, {rate_limited} x 429, {len(failed)} failed"
    )
    print(f"wall time     {wall:.1f} s  ({args.requests / wall * 60:.1f} requests/min)")
    if ok:
        print(
            f"first byte    p50 {percentile(ttfb, 50):.2f} s   p95 {percentile(ttfb, 95):.2f} s   max {max(ttfb):.2f} s"
        )
        print(
            f"total         p50 {percentile(total, 50):.2f} s   p95 {percentile(total, 95):.2f} s   max {max(total):.2f} s"
        )
    if rtf:
        print(
            f"realtime      median {statistics.median(rtf):.2f}x   (audio seconds per wall second, per request)"
        )
        print(
            f"audio         {audio_total:.1f} s produced, {audio_total / wall:.2f}x realtime across the run"
        )
    for failure in failed[:5]:
        print(f"failure       {failure.error}")
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:17493")
    parser.add_argument("--key", required=True, help="a client or admin API key")
    parser.add_argument("--voice", required=True, help="voice profile name or id")
    parser.add_argument(
        "--model",
        default="tts-1",
        help="tts-1 (profile default) or a model/engine name",
    )
    parser.add_argument(
        "--format", default="wav", choices=["wav", "pcm", "mp3", "opus", "aac", "flac"]
    )
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--warmup", type=int, default=1, help="requests to send first, not measured"
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--text-file", help="one input text per line; default is a built-in set"
    )
    args = parser.parse_args()
    return asyncio.run(run(args, load_texts(args.text_file)))


if __name__ == "__main__":
    sys.exit(main())
