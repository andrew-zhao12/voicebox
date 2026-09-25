"""In-memory pub/sub for speaking-pill SSE broadcasts.

MCP ``voicebox.speak`` calls and the REST ``POST /speak`` route publish
start/end events that DictateWindow subscribes to via /events/speak, so the
floating pill surfaces whenever an agent is speaking.
"""

import asyncio
from typing import Any

# Each subscriber gets its own queue. Bounded to drop oldest if a client lags.
_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

# The desktop shell and a couple of pills are the only legitimate listeners.
MAX_SUBSCRIBERS = 16


class TooManySubscribersError(Exception):
    pass


def subscribe() -> asyncio.Queue[dict[str, Any]]:
    """Register a new subscriber; caller must call unsubscribe() when done."""
    if len(_subscribers) >= MAX_SUBSCRIBERS:
        raise TooManySubscribersError(f"More than {MAX_SUBSCRIBERS} speak-event subscribers")
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue[dict[str, Any]]) -> None:
    _subscribers.discard(queue)


def publish(kind: str, payload: dict[str, Any]) -> None:
    """Fan out to all current subscribers. Non-blocking; drops on full queue.

    Each subscriber gets its own dict copy — the SSE consumer calls
    ``event.pop("kind", ...)``, so sharing a single dict between queues
    would mean the first consumer to drain its queue strips ``kind`` from
    the object the next consumer later reads.
    """
    for queue in list(_subscribers):
        event = {"kind": kind, **payload}
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # Slow subscriber — skip rather than block publishers.
            pass
