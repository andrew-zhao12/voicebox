"""Server-Sent-Event streams the frontend subscribes to.

``GET /events/speak`` — broadcasts ``speak-start`` / ``speak-end`` events
whenever an agent-initiated speak (MCP tool or POST /speak) runs. The
DictateWindow uses them to show the floating pill in a `speaking` state.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from .. import lifecycle
from ..mcp_server import events as mcp_events

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/events/speak")
async def speak_events(request: Request):
    """SSE stream of speak-start / speak-end events."""
    try:
        queue = mcp_events.subscribe()
    except mcp_events.TooManySubscribersError:
        raise HTTPException(
            status_code=429, detail="Too many speak-event subscribers", headers={"Retry-After": "5"}
        ) from None

    async def event_stream():
        try:
            # Immediate hello so EventSource knows the connection is live.
            yield {"event": "ready", "data": "{}"}
            last_ping = asyncio.get_running_loop().time()
            while True:
                if await request.is_disconnected() or lifecycle.is_draining():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    now = asyncio.get_running_loop().time()
                    if now - last_ping >= 15.0:
                        # Heartbeat so proxies don't reap idle streams.
                        yield {"event": "ping", "data": "{}"}
                        last_ping = now
                    continue
                last_ping = asyncio.get_running_loop().time()
                kind = event.pop("kind", "message")
                yield {"event": kind, "data": json.dumps(event)}
        finally:
            mcp_events.unsubscribe(queue)

    return EventSourceResponse(event_stream())
