"""Offline runtime validation and strict isolation of Docker STT dependencies."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import stt_preflight as preflight

INSTALLER = Path(__file__).resolve().parents[2] / "scripts" / "install_stt_runtimes.py"
spec = importlib.util.spec_from_file_location("install_stt_runtimes", INSTALLER)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.mark.parametrize(
    ("profile", "models", "expected"),
    [
        ("none", "kokoro,whisper-base", ()),
        ("none", "nemotron-streaming-en-0.6b,faster-whisper-base", ("faster-whisper", "nemotron")),
        ("none", "kokoro faster-whisper-base\tnemotron-streaming-en-0.6b", ("faster-whisper", "nemotron")),
        ("all", "", ("faster-whisper", "nemotron")),
        ("nemotron", "whisper-turbo", ("nemotron",)),
    ],
)
def test_runtime_selection_includes_preloads(profile, models, expected):
    assert preflight.required_runtimes(profile, models) == expected


def test_default_preflight_does_not_import_models(monkeypatch, capsys):
    monkeypatch.delenv("VOICEBOX_PRELOAD_MODELS", raising=False)
    monkeypatch.setattr(preflight, "check_runtime", lambda *a, **kw: pytest.fail("no optional runtime requested"))
    assert preflight.main(["--runtimes", "none"]) == 0
    assert json.loads(capsys.readouterr().out) == {"runtimes": []}


def test_unknown_profile_is_an_actionable_failure(capsys):
    assert preflight.main(["--runtimes", "typo"]) == 1
    assert "VOICEBOX_STT_RUNTIMES" in capsys.readouterr().err


def test_missing_nemotron_interpreter_fails_without_launching(monkeypatch):
    monkeypatch.delenv("VOICEBOX_NEMOTRON_PYTHON", raising=False)
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **kw: pytest.fail("must not spawn"))
    with pytest.raises(RuntimeError, match="separate Nemotron runtime"):
        preflight.check_runtime("nemotron", variant="cpu", check_device=True)


def test_nemotron_probe_runs_in_selected_interpreter_offline(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.touch(mode=0o755)
    monkeypatch.setenv("VOICEBOX_NEMOTRON_PYTHON", str(python))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, '{"runtime":"nemotron"}', "")

    monkeypatch.setattr(preflight.subprocess, "run", run)
    assert preflight.check_runtime("nemotron", variant="cu128", check_device=False)["runtime"] == "nemotron"
    args, kwargs = calls[0]
    assert args[0] == str(python)
    assert "--no-device-check" in args
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert kwargs["timeout"] == 120


def test_worker_reports_dependency_conflict(monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 1, '{"error":"wrong Transformers version"}', ""),
    )
    with pytest.raises(RuntimeError, match="wrong Transformers"):
        preflight.check_runtime("faster-whisper", variant="cpu", check_device=True)


def test_probe_timeout_is_bounded(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("probe", 0.1)

    monkeypatch.setattr(preflight.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="exceeded"):
        preflight.check_runtime("faster-whisper", variant="cpu", check_device=True, timeout=0.1)


def test_conflicting_transformers_is_rejected_before_model_import(monkeypatch):
    versions = {"faster-whisper": "1.2.1", "ctranslate2": "4.8.5", "av": "16.1.0", "transformers": "5.13.0"}
    monkeypatch.setattr(preflight.importlib.metadata, "version", versions.__getitem__)
    with pytest.raises(RuntimeError, match="Transformers 4"):
        preflight.probe("faster-whisper", variant="cpu", check_device=True)


def test_missing_cuda_libraries_are_explicit(monkeypatch):
    def missing(_name):
        raise OSError("not found")

    monkeypatch.setattr(preflight.ctypes, "CDLL", missing)
    with pytest.raises(RuntimeError, match=r"libcublas.so.12.*LD_LIBRARY_PATH"):
        preflight._cuda_libraries()


@pytest.mark.parametrize("runtime", ["faster-whisper", "nemotron"])
def test_explicit_cpu_offload_respects_runtime_device(monkeypatch, runtime):
    monkeypatch.setenv("VOICEBOX_REQUIRE_GPU", "1")
    key = "VOICEBOX_FASTER_WHISPER_DEVICE" if runtime == "faster-whisper" else "VOICEBOX_NEMOTRON_DEVICE"
    monkeypatch.setenv(key, "cpu")
    assert not preflight._wants_cuda(runtime, check_device=True)
    monkeypatch.setenv(key, "auto")
    assert preflight._wants_cuda(runtime, check_device=True)


def test_explicit_cuda_cannot_silently_fall_back_to_cpu(monkeypatch):
    versions = {"faster-whisper": "1.2.1", "ctranslate2": "4.8.5", "av": "16.1.0", "transformers": "4.57.3"}
    monkeypatch.setattr(preflight.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setenv("VOICEBOX_FASTER_WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(preflight, "_cuda_libraries", lambda: None)

    def transcribe(*, initial_prompt=None, hotwords=None, word_timestamps=False):
        pass

    modules = {
        "ctranslate2": SimpleNamespace(get_cuda_device_count=lambda: 0),
        "faster_whisper": SimpleNamespace(WhisperModel=SimpleNamespace(transcribe=transcribe)),
    }
    monkeypatch.setattr(preflight.importlib, "import_module", modules.__getitem__)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        preflight.probe("faster-whisper", variant="cu128", check_device=True)
    assert not preflight.probe("faster-whisper", variant="cu128", check_device=False)["cuda_required"]


@pytest.mark.parametrize(
    ("profile", "variant", "arch"),
    [("all", "rocm", "amd64"), ("all", "cu130", "amd64"), ("all", "cu128", "arm64"), ("typo", "cpu", "amd64")],
)
def test_unsupported_builds_fail_before_installing(tmp_path, monkeypatch, profile, variant, arch):
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **kw: pytest.fail("must reject before pip"))
    destination = tmp_path / "stt"
    with pytest.raises(ValueError, match=r"Unknown|support|require"):
        installer.install(
            profile=profile, variant=variant, architecture=arch, requirements=tmp_path, destination=destination
        )
    assert not destination.exists()


def test_default_profile_installs_nothing_and_keeps_rocm_usable(tmp_path, monkeypatch):
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **kw: pytest.fail("default must not install extras"))
    installer.install(profile="none", variant="rocm", architecture="amd64", requirements=tmp_path, destination=tmp_path)
    assert json.loads((tmp_path / "runtimes.json").read_text())["profile"] == "none"


def test_worker_install_is_isolated_and_variant_pinned(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://invalid.example/simple")
    monkeypatch.setattr(installer.subprocess, "run", lambda args, **kw: calls.append((args, kw)))
    installer.install(profile="all", variant="cu128", architecture="amd64", requirements=tmp_path, destination=tmp_path)
    main_install, make_env, torch_install, worker_install = [args for args, _ in calls]
    assert main_install[0] == installer.sys.executable
    assert "--no-deps" in main_install
    assert "requirements-faster-whisper.lock" in main_install[-1]
    assert make_env[-1] == str(tmp_path / "nemotron")
    assert torch_install[0] == str(tmp_path / "nemotron/bin/python")
    assert "torch==2.11.0+cu128" in torch_install
    assert "--no-deps" in worker_install
    assert worker_install[-1].endswith("requirements-nemotron.lock")
    assert calls[-1][1]["env"]["PIP_CONFIG_FILE"] == installer.os.devnull
    assert "PIP_EXTRA_INDEX_URL" not in calls[-1][1]["env"]
