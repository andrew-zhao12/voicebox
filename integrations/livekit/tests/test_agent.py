"""agent.py without a LiveKit or Voicebox server: its settings checks and its turn-taking."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from livekit.plugins import silero

import agent
from echo import EchoLLM

EXAMPLE_DIR = Path(agent.__file__).resolve().parent
# ``python -m livekit.agents`` (and ``lk agent``, which drives it) imports agent.py,
# so agent.py's ``__main__`` block does not run there.
LAUNCHERS = {
    "python agent.py start": ["agent.py", "start"],
    "python -m livekit.agents start": ["-m", "livekit.agents", "start", "agent.py"],
    "python -m livekit.agents start --dev": ["-m", "livekit.agents", "start", "agent.py", "--dev"],
}


def launch(args: list[str], **settings: str) -> subprocess.CompletedProcess[str]:
    """Run the example with *settings* and nothing else from the environment.

    Nothing listens at LIVEKIT_URL: a worker that starts anyway retries for
    minutes, and the timeout fails the test.
    """
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "LIVEKIT_URL": "ws://127.0.0.1:9",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "secret",
        "PYTHONDONTWRITEBYTECODE": "1",
        **settings,
    }
    return subprocess.run(
        [sys.executable, *args],
        cwd=EXAMPLE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("args", LAUNCHERS.values(), ids=LAUNCHERS.keys())
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({}, "VOICEBOX_CLIENT_KEY is not set"),
        ({"VOICEBOX_CLIENT_KEY": "vbx_test"}, "Set LLM_API_KEY (or OPENAI_API_KEY)"),
    ],
    ids=["no client key", "no LLM"],
)
def test_every_launcher_refuses_to_start_without_its_settings(args, settings, message):
    done = launch(args, **settings)
    assert done.returncode == 1
    assert done.stderr.startswith(message)
    assert "Traceback" not in done.stderr
    assert "registered worker" not in done.stdout + done.stderr


def test_the_missing_key_message_gives_the_readmes_key_command(monkeypatch):
    """The key it suggests has an agent's limits: a client key's defaults run out within minutes."""
    readme = (EXAMPLE_DIR / "README.md").read_text()
    block = readme.split("backend/venv/bin/python -m backend.keys create ", 1)[1].split("```", 1)[0]
    flags = " ".join(block.replace("\\", " ").split())
    assert "--limit inference=600" in flags
    monkeypatch.setattr(agent, "VOICEBOX_CLIENT_KEY", "")
    with pytest.raises(RuntimeError) as missing:
        agent.voicebox_key()
    assert f"python -m backend.keys create {flags}" in str(missing.value)


def test_turn_taking_stays_local_under_livekits_dev_defaults(monkeypatch):
    """``console`` and ``dev`` set LIVEKIT_DEV_MODE=1, where LiveKit's defaults run in its cloud."""
    monkeypatch.setenv("LIVEKIT_DEV_MODE", "1")
    monkeypatch.setattr(agent, "VOICEBOX_CLIENT_KEY", "vbx_test")

    async def build():
        return agent.build_session(vad=silero.VAD.load(), llm=EchoLLM())

    session = asyncio.run(build())
    assert session.turn_detection.model == "turn-detector-v1-mini"
    assert session.interruption_detection == "vad"


def test_native_transport_reads_pronunciations_and_speech_controls(monkeypatch, tmp_path):
    dictionary = tmp_path / "pronunciations.json"
    dictionary.write_text('{"SDK": "ess dee kay"}')
    monkeypatch.setattr(agent, "VOICEBOX_TTS_TRANSPORT", "websocket")
    monkeypatch.setattr(agent, "VOICEBOX_CLIENT_KEY", "vbx_test")
    monkeypatch.setenv("VOICEBOX_PRONUNCIATIONS_FILE", str(dictionary))
    monkeypatch.setenv("VOICEBOX_TTS_SPEED", "1.2")
    monkeypatch.setenv("VOICEBOX_TTS_INSTRUCTIONS", "Speak softly")
    monkeypatch.setenv("VOICEBOX_FIRST_CHUNK_CHARS", "40")
    monkeypatch.setenv("VOICEBOX_MAX_CHUNK_CHARS", "300")
    monkeypatch.setenv("VOICEBOX_TEXT_FLUSH_CHARS", "150")
    monkeypatch.setenv("VOICEBOX_TTS_TIMEOUT", "180")
    tts = agent.voicebox_tts()
    assert isinstance(tts, agent.VoiceboxTTS)
    assert tts.text_flush_chars == 150
    assert tts.speech_timeout == 180
    assert tts.settings["pronunciation_replacements"] == {"SDK": "ess dee kay"}
    assert tts.settings["speed"] == 1.2
    assert tts.settings["instructions"] == "Speak softly"
    assert tts.settings["first_chunk_chars"] == 40
    assert tts.settings["max_chunk_chars"] == 300


@pytest.mark.parametrize(
    ("transport", "variable", "value", "problem"),
    [
        ("websocket", "VOICEBOX_TTS_SPEED", "nan", "Speech speed"),
        ("websocket", "VOICEBOX_FIRST_CHUNK_CHARS", "5", "first_chunk_chars"),
        ("websocket", "VOICEBOX_TEXT_FLUSH_CHARS", "oops", "VOICEBOX_TEXT_FLUSH_CHARS"),
        ("websocket", "VOICEBOX_TTS_TIMEOUT", "0", "Speech progress timeout"),
        ("http", "VOICEBOX_FIRST_CHUNK_CHARS", "40", "require"),
        ("invalid", "VOICEBOX_TTS_SPEED", "1", "http or websocket"),
    ],
)
def test_invalid_speech_settings_fail_at_startup(monkeypatch, transport, variable, value, problem):
    monkeypatch.setattr(agent, "VOICEBOX_TTS_TRANSPORT", transport)
    monkeypatch.setattr(agent, "VOICEBOX_CLIENT_KEY", "vbx_test")
    monkeypatch.setattr(agent, "LLM_API_KEY", "llm_test")
    monkeypatch.setenv(variable, value)
    assert problem in agent.settings_problem()


def test_invalid_dictionary_is_reported_at_startup(monkeypatch, tmp_path):
    dictionary = tmp_path / "pronunciations.json"
    dictionary.write_text("invalid json")
    monkeypatch.setattr(agent, "VOICEBOX_TTS_TRANSPORT", "websocket")
    monkeypatch.setattr(agent, "VOICEBOX_CLIENT_KEY", "vbx_test")
    monkeypatch.setattr(agent, "LLM_API_KEY", "llm_test")
    monkeypatch.setenv("VOICEBOX_PRONUNCIATIONS_FILE", str(dictionary))
    assert agent.settings_problem().startswith("Invalid Voicebox speech settings:")
