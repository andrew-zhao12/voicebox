"""Atomic in-process admission for speech and transcription WebSocket sessions."""

_counts: dict[str, dict[str, int]] = {}


def counts_for_kind(kind: str) -> dict[str, int]:
    """Return the shared counts for one service kind."""
    return _counts.setdefault(kind, {})


def open_sessions(kind: str | None = None) -> int:
    """Count one service's sessions, or all live speech and transcription sessions."""
    if kind is not None:
        return sum(counts_for_kind(kind).values())
    return sum(sum(counts.values()) for counts in _counts.values())


def open_for_key(key_id: str) -> int:
    """Count a key's sessions across all service kinds."""
    return sum(counts.get(key_id, 0) for counts in _counts.values())


def reserve(kind: str, key_id: str, *, per_key: int | None, max_global: int) -> bool:
    """Reserve without yielding, enforcing a shared per-key and per-kind global cap."""
    if open_sessions(kind) >= max_global or (per_key is not None and open_for_key(key_id) >= per_key):
        return False
    counts = counts_for_kind(kind)
    counts[key_id] = counts.get(key_id, 0) + 1
    return True


def release(kind: str, key_id: str) -> None:
    """Release one reserved session; remove keys once they no longer have sessions."""
    counts = counts_for_kind(kind)
    remaining = counts.get(key_id, 0) - 1
    if remaining > 0:
        counts[key_id] = remaining
    else:
        counts.pop(key_id, None)
