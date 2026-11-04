"""Offline optional-STT runtime checks: ``python -m backend.stt_preflight``.

Checks installed dependencies, decoder APIs and requested devices without fetching
weights. Nemotron is checked in its configured interpreter, never imported into
the main server's Transformers 4 environment.
"""

import argparse
import ctypes
import importlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

PROFILES = {
    "none": (),
    "faster-whisper": ("faster-whisper",),
    "nemotron": ("nemotron",),
    "all": ("faster-whisper", "nemotron"),
}


def required_runtimes(profile: str, preloads: str = "") -> tuple[str, ...]:
    """Select image runtimes plus optional engines named in startup preloads."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown VOICEBOX_STT_RUNTIMES={profile!r}; use {', '.join(PROFILES)}")
    selected = set(PROFILES[profile])
    for name in preloads.replace(",", " ").split():
        if name.startswith("faster-whisper-"):
            selected.add("faster-whisper")
        elif name.startswith("nemotron-streaming-"):
            selected.add("nemotron")
    return tuple(sorted(selected))


def _version(package: str, *, exact: str | None = None) -> str:
    value = importlib.metadata.version(package)
    if exact is not None and value != exact:
        raise RuntimeError(f"{package}=={exact} is required; found {value}")
    return value


def _cuda_libraries() -> None:
    """Make missing CTranslate2 CUDA dependencies fail before loading a model."""
    for name in ("libcublas.so.12", "libcudnn.so.9"):
        try:
            ctypes.CDLL(name)
        except OSError as exc:
            raise RuntimeError(f"Cannot load {name}; check the cu128 image and LD_LIBRARY_PATH") from exc


def _wants_cuda(runtime: str, *, check_device: bool) -> bool:
    key = "VOICEBOX_FASTER_WHISPER_DEVICE" if runtime == "faster-whisper" else "VOICEBOX_NEMOTRON_DEVICE"
    device = os.getenv(key, "auto")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"{key} must be auto, cpu, or cuda")
    globally_required = os.getenv("VOICEBOX_REQUIRE_GPU", "").lower() in ("1", "true", "yes")
    return check_device and (device == "cuda" or (device == "auto" and globally_required))


def probe(runtime: str, *, variant: str, check_device: bool) -> dict:
    """Import one runtime and inspect the exact APIs used by the backend; never load weights."""
    require_cuda = _wants_cuda(runtime, check_device=check_device)
    if runtime == "faster-whisper":
        versions = {
            package: _version(package, exact=version)
            for package, version in (
                ("faster-whisper", "1.2.1"),
                ("ctranslate2", "4.8.5"),
                ("av", "16.1.0"),
            )
        }
        main_transformers = _version("transformers")
        if main_transformers.split(".")[0] != "4":
            raise RuntimeError("faster-whisper must preserve the main server's Transformers 4 environment")
        versions["transformers"] = main_transformers
        ct2 = importlib.import_module("ctranslate2")
        model = importlib.import_module("faster_whisper").WhisperModel
        signature = inspect.signature(model.transcribe)
        required = {"initial_prompt", "hotwords", "word_timestamps"}
        if not required <= set(signature.parameters):
            raise RuntimeError("faster-whisper transcribe API does not expose required hint/alignment options")
        if variant == "cu128" or require_cuda:
            _cuda_libraries()
        if require_cuda and ct2.get_cuda_device_count() < 1:
            raise RuntimeError("faster-whisper requires CUDA but no CUDA device is available")
        return {"runtime": runtime, "versions": versions, "cuda_required": require_cuda}
    if runtime != "nemotron":
        raise ValueError(f"Unknown runtime: {runtime}")
    versions = {
        package: _version(package, exact=version)
        for package, version in (
            ("transformers", "5.13.0"),
            ("torch", None),
            ("numpy", None),
            ("librosa", None),
            ("soundfile", None),
        )
    }
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    # Access lazy exports so missing optional audio dependencies fail now.
    for name in ("AutoModelForRNNT", "AutoProcessor"):
        getattr(transformers, name)
    for module, cache in (
        ("nemotron_asr_streaming", "NemotronAsrStreamingRNNTDecoderCache"),
        ("nemotron3_5_asr", "Nemotron3_5AsrRNNTDecoderCache"),
    ):
        generation = importlib.import_module(f"transformers.models.{module}.generation_{module}")
        getattr(generation, cache)
        importlib.import_module(f"transformers.models.{module}.modeling_{module}")
    importlib.import_module("librosa")
    importlib.import_module("soundfile")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Nemotron requires CUDA but its isolated PyTorch runtime has no available CUDA device")
    return {"runtime": runtime, "versions": versions, "cuda_required": require_cuda}


def check_runtime(runtime: str, *, variant: str, check_device: bool, timeout: float = 120) -> dict:
    """Check each runtime in a subprocess with offline hub access and a bounded timeout."""
    executable = sys.executable
    if runtime == "nemotron":
        executable = os.getenv("VOICEBOX_NEMOTRON_PYTHON", "")
        if not executable or not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError("VOICEBOX_NEMOTRON_PYTHON must point to the separate Nemotron runtime interpreter")
    args = [executable, str(Path(__file__).resolve()), "--probe", runtime, "--variant", variant]
    if not check_device:
        args.append("--no-device-check")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{runtime} preflight exceeded {timeout:g} seconds") from exc
    try:
        report = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError(f"{runtime} preflight failed without a JSON report (exit {result.returncode})") from exc
    if result.returncode or "error" in report:
        raise RuntimeError(f"{runtime}: {report.get('error', 'preflight failed')}")
    return report


def main(argv: list[str] | None = None) -> int:
    """Return nonzero with a useful setup error before a container begins accepting work."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runtimes", default=os.getenv("VOICEBOX_STT_RUNTIMES", "none"))
    parser.add_argument("--variant", default=os.getenv("VOICEBOX_STT_VARIANT", "cpu"))
    parser.add_argument("--probe", choices=("faster-whisper", "nemotron"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-device-check", action="store_true", help="Build-time imports/library check without requiring GPU access"
    )
    parser.add_argument("--quiet", action="store_true", help="Only print failed checks")
    args = parser.parse_args(argv)
    try:
        if args.probe:
            report = probe(args.probe, variant=args.variant, check_device=not args.no_device_check)
        else:
            selected = required_runtimes(args.runtimes, os.getenv("VOICEBOX_PRELOAD_MODELS", ""))
            if selected and args.variant not in ("cpu", "cu128"):
                raise ValueError("Optional STT profiles support cpu or cu128 only")
            report = {
                "runtimes": [
                    check_runtime(runtime, variant=args.variant, check_device=not args.no_device_check)
                    for runtime in selected
                ]
            }
    except Exception as exc:
        report = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(report), file=sys.stdout if args.probe else sys.stderr)
        return 1
    if not args.quiet:
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
