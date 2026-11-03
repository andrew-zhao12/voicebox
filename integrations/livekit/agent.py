"""A LiveKit voice agent that listens and speaks through Voicebox.

LiveKit carries the call and decides when the user has finished speaking
(Silero VAD plus LiveKit's on-device end-of-turn model, so turn-taking never
calls LiveKit Cloud). Voicebox transcribes the user and speaks the replies
through its OpenAI-compatible API, using LiveKit's stock OpenAI plugin, and
any OpenAI-compatible LLM writes the replies.

Run it (Python 3.10 or newer):

    pip install -r requirements.txt
    python agent.py console   # talk to it from this terminal (local mic and speakers)
    python agent.py dev       # serve rooms on your LiveKit server, with debug logs
    python agent.py start     # production worker

LiveKit Agents 1.8 marks this built-in CLI as deprecated; it still works, and
``python -m livekit.agents start agent.py`` or the LiveKit CLI's ``lk agent
dev|console`` run the same file. Every one of them refuses to start without a
Voicebox client key or an LLM setting.

Environment:
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
                           your LiveKit server, for dev and start
    VOICEBOX_URL           Voicebox server root, default http://127.0.0.1:17493 (/v1 is appended)
    VOICEBOX_CLIENT_KEY    a Voicebox client key (required). Not VOICEBOX_API_KEY, which is
                           the server's own admin key.
    VOICEBOX_VOICE         the agent's voice: a preset voice id or a cloned voice's name or id,
                           default af_heart (GET /v1/voices lists them)
    VOICEBOX_TTS_MODEL     speech model or engine, default tts-1: the voice's own engine
                           (GET /v1/models lists the others; a preset voice works only with
                           its engine, a cloned voice only with a cloning engine)
    VOICEBOX_STT_MODEL     Whisper model, default whisper-1 (the server's current Whisper)
    VOICEBOX_STT_REALTIME  1 (default) streams the user's audio to Voicebox's realtime
                           transcription socket; 0 sends each utterance, cut by the VAD,
                           to /v1/audio/transcriptions
    VOICEBOX_LANGUAGE      language spoken by the user, default en
    LLM_BASE_URL           OpenAI-compatible chat API, default OpenAI's
                           (for Ollama: http://127.0.0.1:11434/v1)
    LLM_MODEL              chat model, default gpt-4.1-mini
    LLM_API_KEY            key for that API, default OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    UserInputTranscribedEvent,
    cli,
    inference,
)
from livekit.agents.llm import LLM
from livekit.agents.vad import VAD
from livekit.plugins import openai, silero

logger = logging.getLogger("voicebox-agent")

VOICEBOX_URL = os.environ.get("VOICEBOX_URL", "http://127.0.0.1:17493").rstrip("/")
VOICEBOX_CLIENT_KEY = os.environ.get("VOICEBOX_CLIENT_KEY", "")
VOICEBOX_VOICE = os.environ.get("VOICEBOX_VOICE", "af_heart")
VOICEBOX_TTS_MODEL = os.environ.get("VOICEBOX_TTS_MODEL", "tts-1")
VOICEBOX_STT_MODEL = os.environ.get("VOICEBOX_STT_MODEL", "whisper-1")
VOICEBOX_STT_REALTIME = os.environ.get("VOICEBOX_STT_REALTIME", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
VOICEBOX_LANGUAGE = os.environ.get("VOICEBOX_LANGUAGE", "en")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

GREETING = "Hello! I'm a voice assistant, and I'm listening. How can I help you today?"
INSTRUCTIONS = (
    "You are a friendly voice assistant. Your replies are spoken aloud, so answer in one to "
    "three short sentences of plain text, without markdown, lists, emoji or symbols."
)


MISSING_KEY = (
    "VOICEBOX_CLIENT_KEY is not set. Create a client key with an agent's limits on the Voicebox "
    "server, from its repository root (the README explains the limits):\n"
    "  python -m backend.keys create --id livekit-agent --role client --limit inference=600 "
    "--limit requests=1200 --limit max_pending_jobs=16 --limit max_realtime_sessions=8"
)
MISSING_LLM = "Set LLM_API_KEY (or OPENAI_API_KEY), or LLM_BASE_URL for a local server."


def voicebox_key() -> str:
    """The Voicebox client key; unset, the OpenAI plugin would send OPENAI_API_KEY instead."""
    if not VOICEBOX_CLIENT_KEY:
        raise RuntimeError(MISSING_KEY)
    return VOICEBOX_CLIENT_KEY


def settings_problem() -> str | None:
    """Why every call would fail (no Voicebox client key, no LLM), or ``None``."""
    if not VOICEBOX_CLIENT_KEY:
        return MISSING_KEY
    if not (LLM_API_KEY or LLM_BASE_URL):
        return MISSING_LLM
    return None


def voicebox_tts() -> openai.TTS:
    """Speech from Voicebox's ``/v1/audio/speech``.

    ``wav`` streams chunk by chunk without ffmpeg on the server, and its
    header carries the sample rate, so every engine plays at the right speed.
    """
    return openai.TTS(
        base_url=f"{VOICEBOX_URL}/v1",
        api_key=voicebox_key(),
        model=VOICEBOX_TTS_MODEL,
        voice=VOICEBOX_VOICE,
        response_format="wav",
    )


def voicebox_stt() -> openai.STT:
    """Transcription from Voicebox's realtime socket, or from its batch endpoint.

    The realtime socket (``/v1/realtime?intent=transcription``) hears the
    whole call and sends interim text. In batch mode AgentSession cuts each
    utterance with the VAD and posts it once the user stops.
    """
    return openai.STT(
        base_url=f"{VOICEBOX_URL}/v1",
        api_key=voicebox_key(),
        model=VOICEBOX_STT_MODEL,
        language=VOICEBOX_LANGUAGE,
        use_realtime=VOICEBOX_STT_REALTIME,
    )


def chat_llm() -> LLM:
    """The model that writes the replies: OpenAI, or any OpenAI-compatible server."""
    return openai.LLM(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL or NOT_GIVEN,
        # Local servers such as Ollama ignore the key, but the client needs one.
        api_key=LLM_API_KEY or ("unused" if LLM_BASE_URL else NOT_GIVEN),
    )


def build_session(*, vad: VAD, llm: LLM) -> AgentSession:
    """The voice pipeline: Voicebox for STT and TTS, local turn-taking, *llm* for the replies.

    Turn-taking is pinned to local models on purpose: left to its defaults,
    LiveKit Agents 1.8 uses LiveKit Cloud's end-of-turn and interruption
    models in ``console`` and ``dev`` mode.
    """
    return AgentSession(
        vad=vad,
        stt=voicebox_stt(),
        llm=llm,
        tts=voicebox_tts(),
        turn_handling={
            # v1-mini runs in-process; its weights ship with livekit-agents.
            "turn_detection": inference.TurnDetector(version="v1-mini"),
            "interruption": {"mode": "vad"},
        },
    )


def prewarm(proc: JobProcess) -> None:
    """Load the VAD once per worker process instead of once per call."""
    proc.userdata["vad"] = silero.VAD.load()


class VoiceboxAgentServer(AgentServer):
    """An ``AgentServer`` that refuses to start when ``settings_problem()`` finds one.

    Every launcher ends in ``run()``: ``python agent.py``, and also
    ``python -m livekit.agents start agent.py`` and ``lk agent``, which import
    this file, so its ``__main__`` block never runs there.
    """

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> None:
        problem = settings_problem()
        if problem is not None:
            # The launchers run this in a task; SystemExit raised here would stay on the
            # task and be logged again at exit, so a loop callback raises it instead.
            asyncio.get_running_loop().call_soon(sys.exit, problem)
            await asyncio.Event().wait()
        await super().run(devmode=devmode, unregistered=unregistered)


server = VoiceboxAgentServer(setup_fnc=prewarm)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Join the room, greet the user, then answer every turn."""
    session = build_session(vad=ctx.proc.userdata["vad"], llm=chat_llm())
    logger.debug(
        "turn detection: %s, interruptions: %s",
        getattr(session.turn_detection, "model", session.turn_detection),
        session.interruption_detection,
    )

    @session.on("user_input_transcribed")
    def _log_transcript(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final:
            logger.debug("heard: %s", ev.transcript)

    await ctx.connect()
    await session.start(agent=Agent(instructions=INSTRUCTIONS), room=ctx.room)
    session.say(GREETING)


if __name__ == "__main__":
    # Checked here too, before the CLI prints its notices or opens audio devices.
    serving = {"console", "dev", "start", "connect"} & set(sys.argv[1:])
    if serving and (problem := settings_problem()):
        sys.exit(problem)
    cli.run_app(server)
