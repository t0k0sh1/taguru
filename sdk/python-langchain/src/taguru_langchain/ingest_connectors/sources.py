"""Source id derivation and idempotency (ADR 0007 §6.1, issue #347).

Extends the source id convention ``taguru extract`` already established
(``path.to_string_lossy()``, src/extract.rs:481) and docs/long-running.html
already documented (``manual.pdf#installation``) to URL and object-storage
connectors, without inventing a new grammar.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .document import MAX_NAME_BYTES, Diagnostic

# A fixed, documented deny-list of well-known signed/temporary auth query
# parameters, matched case-insensitively on the key only — a URL connector
# strips these for two reasons: they are credential-shaped (ADR 0007 §9's
# "credential never reaches Taguru data" rule applies identically to a
# source id), and a rotating signature would otherwise make "the same
# resource" mint a different source id on every fetch, breaking §6.3's
# idempotency the same way any other unstable id breaks it.
_DENYLISTED_QUERY_KEYS: frozenset[str] = frozenset(
    {
        "signature",
        "sig",
        "token",
        "access_token",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        # AWS SigV4's other per-issuance companions: the same object
        # presigned twice keeps the same x-amz-credential but gets a fresh
        # x-amz-date/expires/algorithm/signedheaders each time, so leaving
        # these in would still mint a new source id per presign — exactly
        # the duplicate-ingestion this canonicalization exists to prevent.
        "x-amz-date",
        "x-amz-expires",
        "x-amz-algorithm",
        "x-amz-signedheaders",
        # GCS V4 signed URLs' equivalents of the above.
        "x-goog-signature",
        "x-goog-credential",
        "x-goog-date",
        "x-goog-expires",
        "x-goog-algorithm",
        "x-goog-signedheaders",
        "apikey",
        "api_key",
        # Deliberately NOT denylisted: Azure SAS's short keys (se, st, sp,
        # sv, sr). They churn per-issuance the same way, but they are also
        # short enough to collide with legitimate app query params on
        # arbitrary URLs (e.g. ?sr=1) — stripping innocent params from a
        # non-Azure URL was judged worse than accepting that Azure SAS
        # identity still churns across re-issuance.
    }
)


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def file_source_id(path: str) -> str:
    """The local-file source id — unchanged from ``taguru extract``'s own
    ``path.to_string_lossy()`` (src/extract.rs:481): the path string,
    verbatim."""
    return path


def sub_source_id(source: str, fragment: str) -> str:
    """A sub-document unit within ``source`` — ``manual.pdf#p12`` or
    ``manual.pdf#installation`` (already documented in
    docs/long-running.html), extended by ADR 0007 §6.1 to every connector
    kind: ``s3://bucket/key#p12``, ``https://example.com/report.html#p3``."""
    return f"{source}#{fragment}"


def check_source_id(source: str) -> Diagnostic | None:
    """``None`` when ``source`` fits within ``MAX_NAME_BYTES``; otherwise
    the ``source_id_too_long`` diagnostic a connector must refuse the
    object with — NEVER a silently truncated id, since truncation risks two
    distinct objects colliding on one source id (ADR 0007 §6.1)."""
    if _byte_len(source) <= MAX_NAME_BYTES:
        return None
    return Diagnostic(
        code="source_id_too_long",
        message=f"source id exceeds {MAX_NAME_BYTES} bytes",
        source=source,
    )


def canonicalize_url(url: str) -> str:
    """The one canonical, credential-stripped form of ``url`` — used for
    identity, storage, AND display alike (ADR 0007 §6.1). There is no
    separate "redacted display value": canonicalization already produces
    one value safe for every purpose, so any URL appearing in a log line or
    the observability summary uses this same form.

    - ``userinfo`` (``https://user:pass@host/...``) is always stripped —
      never meaningful to identity, always credential-shaped.
    - The deny-listed query parameters (module-level
      ``_DENYLISTED_QUERY_KEYS``) are stripped; every other query parameter
      is kept, in its original order.
    - Scheme, host (with port, and case preserved — canonicalization here
      is about credentials and stability, not host normalization), path,
      and fragment are otherwise untouched.

    The raw, uncanonicalized URL is the caller's to use for the fetch
    itself; it is never returned here and must never be persisted.
    """
    parts = urlsplit(url)
    netloc = parts.netloc.rpartition("@")[2] if "@" in parts.netloc else parts.netloc
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in _DENYLISTED_QUERY_KEYS
        ]
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


class SourceIdRegistry:
    """Detects "two distinct fetches canonicalize to the same source id" —
    a rare collision (e.g. two otherwise-identical URLs differing only in a
    stripped query parameter) this run must refuse the later one for,
    rather than silently overwrite (ADR 0007 §6.1) — the same
    collision-refusal ``taguru extract``'s own ``Run.claimed`` map already
    applies for batch file names (src/extract.rs:1273).

    Scoped to one run's lifetime; a connector constructs one instance per
    enumeration pass. Deciding which diagnostic code names a refused
    collision is left to the calling connector — ADR 0007 §8's closed
    vocabulary has no code specific to "duplicate source id" today, so a
    connector should pick the code that best matches why the object could
    not be safely ingested under that identity.
    """

    def __init__(self) -> None:
        self._claimed: set[str] = set()

    def claim(self, source: str) -> bool:
        """Returns ``True`` the first time ``source`` is claimed by this
        registry, ``False`` on every subsequent claim of the same id."""
        if source in self._claimed:
            return False
        self._claimed.add(source)
        return True
