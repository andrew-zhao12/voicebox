"""Naive UTC timestamps for the database.

Every ``DateTime`` column and every stored timestamp is naive UTC (the
schema predates timezone-aware storage), so callers need the current time
in that form; ``datetime.utcnow()`` gave exactly that but is deprecated.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """The current UTC time as a naive ``datetime`` (what the schema stores)."""
    return datetime.now(UTC).replace(tzinfo=None)
