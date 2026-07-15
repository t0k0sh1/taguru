"""Retry policy: idempotency-first.

Most of taguru's POST routes are reads-with-a-body or replace/diff writes and
are safe to retry blindly. The exceptions carry ``UNSAFE_ON_AMBIGUOUS``: a
request that may already have executed (the response was lost mid-flight) is
NOT retried there, because re-applying would double state — ``add_associations``
accumulates weight. Statuses where the server states the request was never
executed (429 rate limit, 503 shed — both rejected before the handler ran)
retry regardless of class.
"""

from __future__ import annotations

import enum
import math
import random

DEFAULT_RETRIES = 2
BACKOFF_BASE_SECS = 0.5
BACKOFF_CAP_SECS = 8.0


class RetryClass(enum.Enum):
    SAFE = "safe"
    UNSAFE_ON_AMBIGUOUS = "unsafe_on_ambiguous"


def should_retry_status(status: int, retry_class: RetryClass) -> bool:
    """Whether a non-2xx status is retryable for this route class.

    429/503 mean the request was shed before executing — always retryable.
    502 is a transient embedding-provider failure ("retry later" per the
    protocol) — it only occurs on read-shaped routes, all of which are SAFE.
    501 is static configuration; retrying cannot help.
    """
    if status in (429, 503):
        return True
    if status == 502:
        return retry_class is RetryClass.SAFE
    return False


def should_retry_transport(ambiguous: bool, retry_class: RetryClass) -> bool:
    """Whether a transport failure is retryable.

    ``ambiguous`` means the request may have reached the server (failure after
    the connection was established); a pre-connect failure is always safe.
    """
    return not ambiguous or retry_class is RetryClass.SAFE


def backoff_delay(attempt: int) -> float:
    """Full-jitter exponential backoff for the given 0-based attempt."""
    return random.uniform(0.0, min(BACKOFF_CAP_SECS, BACKOFF_BASE_SECS * (2.0**attempt)))


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header. The server sends delay-seconds only.

    ``float`` already rejects a trailing tail ("5 seconds"), but it accepts
    "inf"/"nan"; require a finite, non-negative result so a malformed header
    falls back to the computed backoff instead of sleeping forever.
    """
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None
