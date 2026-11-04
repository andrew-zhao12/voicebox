"""Worker for the end-to-end check: ../agent.py's session with an echo LLM.

run_e2e.py starts it as ``python e2e_agent.py start``. The session comes
from ``agent.build_session`` (Voicebox STT and TTS, local turn-taking) with
EchoLLM in place of a real model, and the greeting is ``agent.GREETING``.
What the driver checks is logged as records carrying an ``e2e`` field. The
server options suit a test machine: one idle process, a random health port,
no CPU load limit and a short drain.

Environment: everything agent.py reads (except the LLM_* variables), plus
    LIVEKIT_AGENT_NAME   agent name for explicit dispatch (run_e2e.py sets e2e-agent-<run id>)
    LIVEKIT_DEV_MODE     run_e2e.py sets 1, LiveKit's console/dev defaults, which the
                         example's turn handling must override
    E2E_USER_IDENTITY    participant the session listens to, default e2e-user
    E2E_ECHO_TAIL        text appended to the first reply of each call
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    JobContext,
    UserInputTranscribedEvent,
    cli,
    room_io,
)
from livekit.agents.metrics import STTMetrics, TTSMetrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent as example  # the example, one directory up
from echo import EchoLLM

logger = logging.getLogger("voicebox-e2e")

USER_IDENTITY = os.environ.get("E2E_USER_IDENTITY", "e2e-user")
ECHO_TAIL = os.environ.get("E2E_ECHO_TAIL", "")


def _record(event: str, **fields: Any) -> None:
    logger.info("E2E %s", event, extra={"e2e": {"event": event, "wall": time.time(), **fields}})


def _report(session: AgentSession) -> None:
    """Log transcripts, agent states, chat items and TTS/STT metrics for the driver."""

    @session.on("user_input_transcribed")
    def _on_transcript(ev: UserInputTranscribedEvent) -> None:
        _record("transcript", final=ev.is_final, text=ev.transcript, at=ev.created_at)

    @session.on("agent_state_changed")
    def _on_state(ev: AgentStateChangedEvent) -> None:
        _record("agent_state", old=ev.old_state, new=ev.new_state, at=ev.created_at)

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if getattr(item, "type", "") == "message":
            _record(
                "item",
                role=item.role,
                text=item.text_content,
                interrupted=item.interrupted,
                metrics=dict(item.metrics),
                at=ev.created_at,
            )

    # Straight from the plugins: AgentSession's own metrics_collected is deprecated in 1.8.
    def _on_metrics(m: TTSMetrics | STTMetrics) -> None:
        _record("metrics", **m.model_dump(mode="json", exclude={"metadata"}))

    session.tts.on("metrics_collected", _on_metrics)
    session.stt.on("metrics_collected", _on_metrics)


server = AgentServer(
    port=0,  # the production default, 8081, is often taken on a workstation
    num_idle_processes=1,  # the production default is one per CPU
    load_threshold=float("inf"),  # never refuse a job because the test machine is busy
    drain_timeout=5,
    setup_fnc=example.prewarm,
)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """The example's entrypoint with EchoLLM, bound to the scripted participant."""
    llm = (
        example.chat_llm()
        if os.getenv("E2E_LLM") == "configured"
        else EchoLLM(tail=ECHO_TAIL, repeat_tail=os.getenv("E2E_REPEAT_TAIL") == "1")
    )
    session = example.build_session(vad=ctx.proc.userdata["vad"], llm=llm)
    _report(session)
    detector = session.turn_detection
    _record(
        "config",
        turn_detector=getattr(detector, "model", detector),
        interruption=str(session.interruption_detection),  # "vad", or NOT_GIVEN for the default
        stt_realtime=example.VOICEBOX_STT_REALTIME,
        tts_transport=example.VOICEBOX_TTS_TRANSPORT,
        greeting=example.GREETING,
    )
    # With participant_identity set, 1.8.8's RoomIO needs the room connected before start().
    await ctx.connect()
    await session.start(
        agent=Agent(instructions=example.INSTRUCTIONS),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            participant_identity=USER_IDENTITY, delete_room_on_close=True
        ),
    )
    await ctx.wait_for_participant(identity=USER_IDENTITY)
    session.say(example.GREETING)


if __name__ == "__main__":
    if not example.VOICEBOX_CLIENT_KEY:
        sys.exit("VOICEBOX_CLIENT_KEY is not set")
    cli.run_app(server)
