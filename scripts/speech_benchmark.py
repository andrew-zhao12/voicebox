#!/usr/bin/env python3
"""Benchmark user-supplied speech fixtures and Voicebox HTTP capacity without loading models locally."""

import argparse
import asyncio
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import time
import unicodedata
import wave
from collections import Counter, defaultdict
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

PCM_RATE = 24000
METRIC_NAMES = {
    "voicebox_gpu_memory_bytes",
    "voicebox_queue_pending_jobs",
    "voicebox_queue_running_jobs",
    "voicebox_http_requests_total",
    "voicebox_generations_total",
}


def normalized(text: str) -> str:
    """Case-fold Unicode text and replace punctuation/symbols with word boundaries."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in folded).split())


def edit_distance(reference: list, actual: list) -> int:
    """Return Levenshtein distance using memory proportional to the hypothesis length."""
    row = list(range(len(actual) + 1))
    for i, token in enumerate(reference, 1):
        previous, row = row, [i]
        for j, received in enumerate(actual, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (token != received)))
    return row[-1]


def accuracy(reference: str, actual: str) -> dict:
    """Return error counts and rates; CER includes normalized spaces."""
    ref, hyp = normalized(reference), normalized(actual)
    words, characters = ref.split(), list(ref)
    word_errors = edit_distance(words, hyp.split())
    char_errors = edit_distance(characters, list(hyp))
    return {
        "word_errors": word_errors,
        "reference_words": len(words),
        "char_errors": char_errors,
        "reference_chars": len(characters),
        "wer": word_errors / max(len(words), 1),
        "cer": char_errors / max(len(characters), 1),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    """Return the nearest-rank percentile (including the maximum for small p95 samples)."""
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def required_string(value: Any, name: str) -> str:
    """Reject missing or blank manifest values before any network requests."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def load_manifest(path: Path, max_audio_bytes: int = 32 * 1024**2) -> dict:
    """Validate PCM16 WAV fixtures and record hashes without modifying input files."""
    content = path.read_bytes()
    manifest = json.loads(content)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("manifest must be an object with schema_version=1")
    required_string(manifest.get("name"), "name")
    clean = {"name": manifest["name"], "sha256": hashlib.sha256(content).hexdigest(), "audio": [], "tts": []}
    for kind in ("audio", "tts"):
        entries = manifest.get(kind, [])
        if not isinstance(entries, list):
            raise ValueError(f"{kind} must be an array")
        ids = set()
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError(f"{kind} entries must be objects")
            case_id = required_string(item.get("id"), f"{kind}.id")
            if case_id in ids:
                raise ValueError(f"duplicate {kind} id: {case_id}")
            ids.add(case_id)
            text = required_string(item.get("text"), f"{case_id}.text")
            if not normalized(text):
                raise ValueError(f"{case_id}.text must contain words or numbers")
            case = {"id": case_id, "text": text}
            if kind == "audio":
                source = item.get("source")
                if source not in {"real", "synthetic"}:
                    raise ValueError(f"{case_id}.source must explicitly be real or synthetic")
                audio_path = (path.parent / required_string(item.get("path"), f"{case_id}.path")).resolve()
                if not audio_path.is_file():
                    raise ValueError(f"{case_id}.path must be a regular file")
                size = audio_path.stat().st_size
                if size > max_audio_bytes:
                    raise ValueError(f"{case_id} exceeds the audio file size limit")
                data = audio_path.read_bytes()
                if len(data) > max_audio_bytes:
                    raise ValueError(f"{case_id} exceeds the audio file size limit")
                try:
                    with wave.open(io.BytesIO(data), "rb") as audio:
                        frames, rate = audio.getnframes(), audio.getframerate()
                        channels = audio.getnchannels()
                        if audio.getsampwidth() != 2 or channels not in {1, 2} or frames < 1 or rate < 8000:
                            raise ValueError(f"{case_id} needs nonempty mono/stereo PCM16 WAV at 8 kHz or higher")
                        if frames * channels * 2 > len(data):
                            raise ValueError(f"{case_id} has a truncated WAV payload")
                        payload = audio.readframes(frames)
                        if len(payload) != frames * channels * 2:
                            raise ValueError(f"{case_id} has a truncated WAV payload")
                except (wave.Error, EOFError) as exc:
                    raise ValueError(f"{case_id} is not a supported PCM16 WAV") from exc
                case.update(
                    source=source,
                    path=str(audio_path),
                    sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data),
                    audio_s=frames / rate,
                    sample_rate=rate,
                    channels=channels,
                )
                if "language" in item:
                    case["language"] = required_string(item["language"], f"{case_id}.language")
            else:
                case.update(
                    source="synthetic",
                    model=required_string(item.get("model"), f"{case_id}.model"),
                    voice=required_string(item.get("voice"), f"{case_id}.voice"),
                )
            clean[kind].append(case)
    if not clean["audio"] and not clean["tts"]:
        raise ValueError("manifest needs at least one audio or tts case")
    return clean


def parse_metrics(content: str) -> list[dict]:
    """Keep only the finite counters/gauges used by this benchmark."""
    values = []
    for line in content.splitlines():
        match = re.fullmatch(r"(\w+)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?", line)
        if not match or match[1] not in METRIC_NAMES:
            continue
        try:
            number = float(match[3])
        except ValueError:
            continue
        if not math.isfinite(number):
            continue
        labels = {
            key: json.loads(f'"{value}"') for key, value in re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', match[2] or "")
        }
        values.append({"name": match[1], "labels": labels, "value": number})
    return values


def cancellation_counts(metrics: list[dict]) -> dict[str, float]:
    """Read global cancellation counters; a request may increment more than one."""
    counts = {"speech_499": 0.0, "stream_cancelled": 0.0, "stream_abandoned": 0.0}
    for metric in metrics:
        labels, value = metric["labels"], metric["value"]
        if metric["name"] == "voicebox_http_requests_total":
            if labels.get("route") == "/v1/audio/speech" and labels.get("status") == "499":
                counts["speech_499"] += value
        elif metric["name"] == "voicebox_generations_total" and labels.get("kind") == "stream":
            status = labels.get("status")
            if status in {"cancelled", "abandoned"}:
                counts[f"stream_{status}"] += value
    return counts


class Observer:
    """Optional admin-only model and metrics observations; never infer missing GPU data."""

    def __init__(self, client: httpx.AsyncClient | None):
        self.client = client
        self.samples: list[dict] = []
        self.snapshots: list[dict] = []
        self.started = time.perf_counter()

    async def metrics(self) -> dict:
        """Read one bounded metrics response without persisting server error bodies."""
        sample = {"at_s": time.perf_counter() - self.started, "metrics": []}
        if self.client is None:
            return {**sample, "unavailable": "VOICEBOX_ADMIN_KEY not set"}
        try:
            response = await self.client.get("/metrics")
            if response.status_code != 200:
                return {**sample, "unavailable": f"HTTP {response.status_code}"}
            if len(response.content) > 2 * 1024**2:
                return {**sample, "unavailable": "metrics response exceeds 2 MiB"}
            sample["metrics"] = parse_metrics(response.text)
        except (httpx.HTTPError, ValueError) as exc:
            sample["unavailable"] = type(exc).__name__
        return sample

    async def snapshot(self, label: str) -> None:
        """Record model residency outside timed sweeps."""
        snapshot = {"label": label, **await self.metrics()}
        if self.client:
            try:
                response = await self.client.get("/models/status")
                snapshot["models"] = (
                    response.json() if response.status_code == 200 else {"status": response.status_code}
                )
            except (httpx.HTTPError, ValueError) as exc:
                snapshot["models"] = {"unavailable": type(exc).__name__}
        self.snapshots.append(snapshot)

    async def sample_until(self, done: asyncio.Event, interval: float) -> None:
        """Sample allocator gauges while the measured requests are in flight."""
        if self.client is None:
            return
        while not done.is_set():
            self.samples.append(await self.metrics())
            with suppress(TimeoutError):
                await asyncio.wait_for(done.wait(), interval)

    def report(self) -> dict:
        """Return observed allocator peaks, not total device VRAM or an inferred GPU size."""
        peaks = {}
        for sample in [*self.samples, *self.snapshots]:
            for metric in sample["metrics"]:
                if metric["name"] == "voicebox_gpu_memory_bytes":
                    kind = metric["labels"].get("kind", "unknown")
                    peaks[kind] = max(peaks.get(kind, 0), metric["value"])
        return {
            "snapshots": self.snapshots,
            "samples": self.samples,
            "gpu_allocator_peak_bytes": peaks or None,
            "gpu_scope": "sampled PyTorch CUDA allocator only; excludes other processes and non-PyTorch allocations",
        }


async def request_one(client: httpx.AsyncClient, case: dict, row: dict, timeout: float) -> dict:
    """Measure an HTTP inference request; failures never become successful truncated audio."""
    started = time.perf_counter()
    row.update(status=0, ok=False, first_audio_s=None, audio_bytes=0)
    try:
        async with asyncio.timeout(timeout):
            if row["kind"] == "stt":
                # Keep file descriptors local to each concurrent request and validate the hash
                # again so a recording edited during the run cannot silently change the corpus.
                data = Path(case["path"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != case["sha256"]:
                    raise ValueError("fixture_changed")
                form = {"model": row["model"], "response_format": "json"}
                if case.get("language"):
                    form["language"] = case["language"]
                response = await client.post(
                    "/v1/audio/transcriptions",
                    data=form,
                    files={"file": (Path(case["path"]).name, data, "audio/wav")},
                )
                row["status"] = response.status_code
                if response.status_code != 200:
                    row["error"] = f"HTTP {response.status_code}"
                    return row
                transcript = response.json().get("text")
                if not isinstance(transcript, str):
                    raise ValueError("missing_transcript")
                row.update(transcript=transcript, audio_s=case["audio_s"], **accuracy(case["text"], transcript))
            else:
                body = {"model": row["model"], "voice": case["voice"], "input": case["text"], "response_format": "pcm"}
                async with client.stream("POST", "/v1/audio/speech", json=body) as response:
                    row["status"] = response.status_code
                    if response.status_code != 200:
                        row["error"] = f"HTTP {response.status_code}"
                        return row
                    rate = int(response.headers.get("x-voicebox-sample-rate", PCM_RATE))
                    if rate != PCM_RATE:
                        raise ValueError("unexpected_pcm_sample_rate")
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            if row["first_audio_s"] is None:
                                row["first_audio_s"] = time.perf_counter() - started
                            row["audio_bytes"] += len(chunk)
                    if row["audio_bytes"] == 0 or row["audio_bytes"] % 2:
                        raise ValueError("empty_or_unaligned_pcm")
                    row["audio_s"] = row["audio_bytes"] / 2 / PCM_RATE
            row["ok"] = True
    except asyncio.CancelledError:
        row["error"] = "client_cancelled"
        raise
    except (httpx.HTTPError, TimeoutError, ValueError, OSError) as exc:
        # Exception strings can embed request URLs or response bodies. Report only classes.
        row["error"] = type(exc).__name__
    finally:
        row["latency_s"] = time.perf_counter() - started
        if row["ok"]:
            row["rtf"] = row["latency_s"] / row["audio_s"]
            row["audio_per_wall_x"] = row["audio_s"] / row["latency_s"]
    return row


def new_row(kind: str, model: str, case: dict, concurrency: int, repetition: int) -> dict:
    """Create report fields without credentials or raw input audio."""
    return {
        "kind": kind,
        "model": model,
        "case_id": case["id"],
        "source": case["source"],
        "voice": case.get("voice"),
        "concurrency": concurrency,
        "repetition": repetition,
    }


def summarize(rows: list[dict], wall: float, peak: int) -> dict:
    """Aggregate one model/source/concurrency sweep, weighting corpus errors by reference size."""
    first = rows[0]
    good = [row for row in rows if row["ok"]]
    summary = {key: first[key] for key in ("kind", "model", "source", "voice", "concurrency")}
    summary.update(
        requests=len(rows),
        successes=len(good),
        errors=len(rows) - len(good),
        rate_limited=sum(row["status"] == 429 for row in rows),
        status_counts=dict(Counter(str(row["status"]) for row in rows)),
        wall_s=wall,
        max_in_flight=peak,
        successful_requests_per_s=len(good) / wall,
        audio_s=sum(row["audio_s"] for row in good),
        audio_per_wall_x=sum(row["audio_s"] for row in good) / wall,
    )
    for field in ("latency_s", "first_audio_s", "rtf"):
        values = [row[field] for row in good if row.get(field) is not None]
        summary[f"{field}_p50"] = percentile(values, 0.5)
        summary[f"{field}_p95"] = percentile(values, 0.95)
    if first["kind"] == "stt":
        for rate, errors, count in (
            ("wer", "word_errors", "reference_words"),
            ("cer", "char_errors", "reference_chars"),
        ):
            denominator = sum(row[count] for row in good)
            summary[rate] = sum(row[errors] for row in good) / denominator if denominator else None
        summary["accuracy_scope"] = "successful requests only; inspect error count before comparing"
    return summary


async def sweep(
    client: httpx.AsyncClient,
    cases: list[dict],
    *,
    kind: str,
    model: str,
    concurrency: int,
    repeats: int,
    timeout: float,
) -> tuple[list[dict], dict]:
    """Run bounded workers so high request counts do not create unbounded waiting tasks."""
    pending = iter((case, repeat) for repeat in range(repeats) for case in cases)
    rows = []
    active = peak = 0

    async def worker() -> None:
        nonlocal active, peak
        for case, repeat in pending:
            active += 1
            peak = max(peak, active)
            row = new_row(kind, model, case, concurrency, repeat)
            try:
                rows.append(await request_one(client, case, row, timeout))
            finally:
                active -= 1

    started = time.perf_counter()
    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, repeats * len(cases))):
            group.create_task(worker())
    return rows, summarize(rows, time.perf_counter() - started, peak)


async def cancellation_probe(
    client: httpx.AsyncClient, observer: Observer, case: dict, *, after: float, observe: float, timeout: float
) -> dict:
    """Distinguish local connection closure from a later server cancellation observation."""
    before = await observer.metrics()
    baseline = cancellation_counts(before["metrics"])
    row = new_row("tts", case["model"], case, 1, 0)
    task = asyncio.create_task(request_one(client, case, row, timeout))
    started = time.perf_counter()
    try:
        await asyncio.sleep(after)
        if task.done():
            await task
            return {"request": row, "cancelled": False, "reason": "request_finished_before_cancel"}
        cancelled_at = time.perf_counter()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        result = {
            "request": row,
            "cancelled": True,
            "cancel_sent_at_s": cancelled_at - started,
            "client_close_s": time.perf_counter() - cancelled_at,
            "server_cancel_observed_s_upper_bound": None,
            "server_counter_delta": None,
            "evidence_scope": "global server counters; run on an otherwise idle replica",
        }
        if "unavailable" in before:
            result["server_evidence_unavailable"] = before["unavailable"]
            return result
        while time.perf_counter() - cancelled_at < observe:
            current = await observer.metrics()
            if "unavailable" in current:
                result["server_evidence_unavailable"] = current["unavailable"]
                break
            counts = cancellation_counts(current["metrics"])
            delta = {key: counts[key] - baseline[key] for key in counts}
            result["server_counter_delta"] = delta
            if any(value < 0 for value in delta.values()):
                result["server_evidence_unavailable"] = "counter_reset"
                break
            if any(value > 0 for value in delta.values()):
                result["server_cancel_observed_s_upper_bound"] = time.perf_counter() - cancelled_at
                break
            await asyncio.sleep(min(0.2, observe))
        return result
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write union columns, preserving nested diagnostic fields as JSON cells."""
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            )


async def run(args: argparse.Namespace, manifest: dict, *, key: str, admin_key: str | None) -> dict:
    """Measure each model sequentially, preserving resident models and all external resources."""
    report = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "server_url": args.url,
        "server_label": args.server_label,
        "client": {"python": platform.python_version(), "platform": platform.platform(), "httpx": httpx.__version__},
        "manifest": manifest,
        "settings": {
            name: getattr(args, name)
            for name in (
                "stt_models",
                "concurrency",
                "repeats",
                "warmup",
                "timeout",
                "sample_interval",
                "cancel_after",
                "cancel_observe",
            )
        },
        "requests": [],
        "summaries": [],
        "warmups": [],
        "cancellations": [],
        "definitions": {
            "rtf": "request wall seconds / audio seconds; lower is faster",
            "first_audio_s": "time to first nonempty PCM payload including leading silence; not audible-onset latency",
            "latency_s": "request start through complete response; includes upload, server queue and network",
            "capacity": "HTTP inference workload; does not include WebRTC, VAD, LLM or realtime WebSocket STT",
            "normalization": "Unicode NFKC+casefold, punctuation/symbols become spaces, whitespace collapsed; CER includes spaces",
        },
    }
    limits = httpx.Limits(
        max_connections=max(args.concurrency) + 2, max_keepalive_connections=max(args.concurrency) + 2
    )
    async with httpx.AsyncClient(
        base_url=args.url, headers={"Authorization": f"Bearer {key}"}, timeout=args.timeout, limits=limits
    ) as client:
        admin = (
            httpx.AsyncClient(
                base_url=args.url, headers={"Authorization": f"Bearer {admin_key}"}, timeout=min(args.timeout, 10)
            )
            if admin_key
            else None
        )
        observer = Observer(admin)
        done = asyncio.Event()
        sampling = None
        try:
            await observer.snapshot("before")
            sampling = asyncio.create_task(observer.sample_until(done, args.sample_interval))
            groups = []
            for model in args.stt_models:
                for source in ("real", "synthetic"):
                    cases = [case for case in manifest["audio"] if case["source"] == source]
                    if cases:
                        groups.append(("stt", model, cases))
            tts_groups = defaultdict(list)
            for case in manifest["tts"]:
                tts_groups[(case["model"], case["voice"])].append(case)
            groups.extend(("tts", model, cases) for (model, _voice), cases in tts_groups.items())
            for kind, model, cases in groups:
                warm_ok = True
                for index in range(args.warmup):
                    case = cases[index % len(cases)]
                    row = new_row(kind, model, case, 1, index)
                    report["warmups"].append(await request_one(client, case, row, args.timeout))
                    warm_ok = warm_ok and row["ok"]
                if warm_ok:
                    for concurrency in args.concurrency:
                        rows, summary = await sweep(
                            client,
                            cases,
                            kind=kind,
                            model=model,
                            concurrency=concurrency,
                            repeats=args.repeats,
                            timeout=args.timeout,
                        )
                        report["requests"].extend(rows)
                        report["summaries"].append(summary)
                        print(
                            f"{kind} {model} {cases[0]['source']} concurrency={concurrency}: {summary['successes']}/{summary['requests']} successful",
                            flush=True,
                        )
                await observer.snapshot(f"after {kind} {model} {cases[0]['source']}")
            if args.cancel_after > 0:
                for case in manifest["tts"]:
                    report["cancellations"].append(
                        await cancellation_probe(
                            client,
                            observer,
                            case,
                            after=args.cancel_after,
                            observe=args.cancel_observe,
                            timeout=args.timeout,
                        )
                    )
        finally:
            done.set()
            if sampling:
                await sampling
            await observer.snapshot("after")
            report["observations"] = observer.report()
            if admin:
                await admin.aclose()
    report["finished_at"] = datetime.now(UTC).isoformat()
    report["ok"] = all(row["ok"] for row in [*report["warmups"], *report["requests"]]) and bool(report["requests"])
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse reproducible workload settings; credentials come only from environment variables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--url", default=os.getenv("VOICEBOX_URL", "http://127.0.0.1:17493"))
    parser.add_argument("--output", type=Path, required=True, help="new report directory (must not already exist)")
    parser.add_argument("--server-label", default="unspecified", help="operator-supplied hardware/deployment label")
    parser.add_argument("--stt-models", default="whisper-1", help="comma-separated models, run sequentially")
    parser.add_argument("--concurrency", default="1,2,4", help="comma-separated positive concurrency levels")
    parser.add_argument("--repeats", type=int, default=3, help="repeat every case this many times per sweep")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured requests per model/source group")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--max-audio-mb", type=float, default=32)
    parser.add_argument(
        "--cancel-after", type=float, default=0, help="cancel a separate probe per TTS case after N seconds; 0 disables"
    )
    parser.add_argument("--cancel-observe", type=float, default=15, help="seconds to poll server cancellation counters")
    args = parser.parse_args(argv)
    try:
        parts = urlsplit(args.url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("--url must be an HTTP(S) server root without credentials, query or fragment")
        if parts.path not in {"", "/"}:
            raise ValueError("--url must be the server root, without /v1 or another path")
        args.url = args.url.rstrip("/")
        args.concurrency = list(dict.fromkeys(int(value) for value in args.concurrency.split(",")))
        if not args.concurrency or any(value < 1 or value > 128 for value in args.concurrency):
            raise ValueError("--concurrency values must be between 1 and 128")
        args.stt_models = list(
            dict.fromkeys(required_string(value.strip(), "STT model") for value in args.stt_models.split(","))
        )
        if args.repeats < 1 or args.warmup < 0:
            raise ValueError("--repeats must be positive and --warmup nonnegative")
        for name in ("timeout", "sample_interval", "max_audio_mb", "cancel_observe"):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
        if not math.isfinite(args.cancel_after) or args.cancel_after < 0:
            raise ValueError("--cancel-after must be finite and nonnegative")
        if args.cancel_after >= args.timeout:
            raise ValueError("--cancel-after must be less than --timeout")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    """Validate the corpus, run the benchmark, and save local JSON/CSV artifacts."""
    args = parse_args()
    key = os.getenv("VOICEBOX_CLIENT_KEY", "").strip()
    if not key:
        raise SystemExit("Set VOICEBOX_CLIENT_KEY; optional VOICEBOX_ADMIN_KEY enables metrics/model snapshots")
    manifest = load_manifest(args.manifest.resolve(), int(args.max_audio_mb * 1024**2))
    args.output.mkdir(parents=True, exist_ok=False)
    report = asyncio.run(run(args, manifest, key=key, admin_key=os.getenv("VOICEBOX_ADMIN_KEY") or None))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "requests.csv", report["requests"])
    write_csv(args.output / "summaries.csv", report["summaries"])
    write_csv(args.output / "cancellations.csv", report["cancellations"])
    print(f"Reports: {args.output.resolve()}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
