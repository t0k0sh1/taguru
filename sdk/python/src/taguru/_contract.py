"""Wire-contract compatibility (ADR 0005 §3.8, §6).

This SDK decodes HTTP; it owns no wire shape of its own (ADR 0005 §3). So the
only thing it ever compares against a server is ``http_contract`` — never
``server``'s own SemVer (a compatible patch/minor difference must never be
refused) and never ``mcp_contract`` (this SDK speaks no MCP; that dimension
covers tool schemas and JSON-RPC conventions it is structurally blind to, so
comparing it would only produce false rejections).

Fails closed only on POSITIVE proof the two ranges are disjoint. Every
absence of information — a 404 (pre-0.6 servers, which are ``http_contract:
1`` in substance even though they predate this endpoint), a non-JSON body, a
missing ``http_contract`` key, an empty ``supported`` array — is fail-open:
none of those prove incompatibility, and refusing on them would be a worse
break than the one this check exists to prevent.

Deliberately NOT part of ``sdk/spec/check_versions.py``'s lockstep: ADR 0005
§3.8 makes the contract range independent of package version, on purpose —
routing it through that checker would silently re-couple them and undo the
point of the eight-dimension split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._errors import IncompatibleServerError

# The one `http_contract` version this SDK release decodes. A tuple of
# accepted versions, not a {min, max} range: it mirrors GET /version's own
# `supported` array (ADR 0005 §6) so the check is a plain set intersection,
# and it can express a future gap (e.g. `(1, 3)`) a min/max pair cannot.
SUPPORTED_HTTP_CONTRACTS: tuple[int, ...] = (1,)

# ADR 0005 §6: exempt from auth like the other probes, always 200.
VERSION_PATH = "/version"


@dataclass(slots=True, frozen=True)
class ServerContract:
    """The subset of ``GET /version``'s body this SDK reads."""

    server: str | None
    supported: tuple[int, ...]


def parse_version_body(payload: Any) -> ServerContract | None:
    """Parse a ``GET /version`` body, or ``None`` if it can't be read.

    ``None`` covers every shape this SDK does not recognize (a pre-0.6
    server's 404 body, a stray non-JSON response, a body missing
    ``http_contract`` or carrying a non-list ``supported``) — the caller
    treats ``None`` as "learned nothing," which is fail-open by construction.
    """
    if not isinstance(payload, dict):
        return None
    http_contract = payload.get("http_contract")
    if not isinstance(http_contract, dict):
        return None
    supported = http_contract.get("supported")
    if not isinstance(supported, list) or not all(isinstance(item, int) for item in supported):
        return None
    server = payload.get("server")
    return ServerContract(
        server=server if isinstance(server, str) else None,
        supported=tuple(supported),
    )


def incompatibility(seen: ServerContract, base_url: str) -> IncompatibleServerError | None:
    """The error to raise for ``seen``, or ``None`` if it's compatible.

    Compatible means "shares at least one `http_contract` version with this
    SDK" — an empty ``seen.supported`` is treated as no proof of anything
    (fail-open), not as an empty intersection.
    """
    if not seen.supported:
        return None
    if set(seen.supported) & set(SUPPORTED_HTTP_CONTRACTS):
        return None

    from . import __version__  # local: avoids a module-level import cycle

    sdk_versions = ", ".join(str(version) for version in SUPPORTED_HTTP_CONTRACTS)
    server_versions = ", ".join(str(version) for version in seen.supported)
    server_note = f" (taguru {seen.server})" if seen.server else ""

    if min(seen.supported) > max(SUPPORTED_HTTP_CONTRACTS):
        remedy = (
            f"Upgrade this SDK to a release that speaks http_contract "
            f"{server_versions}: pip install --upgrade 'taguru>={seen.server}'"
            if seen.server
            else f"Upgrade this SDK to a release that speaks http_contract {server_versions}."
        )
    elif seen.server and max(seen.supported) < min(SUPPORTED_HTTP_CONTRACTS):
        remedy = (
            f"Upgrade the server to {seen.server} or newer, or pin this SDK to the "
            f"server's release: pip install 'taguru=={seen.server.rsplit('.', 1)[0]}.*'"
        )
    else:
        remedy = (
            "Upgrade or downgrade one side to a pair that shares a contract version; "
            "this SDK's range is declared as taguru.SUPPORTED_HTTP_CONTRACTS."
        )

    message = (
        f"taguru SDK {__version__} speaks http_contract {sdk_versions}, but the "
        f"server at {base_url}{server_note} supports http_contract {server_versions} "
        f"— no version in common. {remedy}"
    )
    return IncompatibleServerError(
        message,
        sdk_version=__version__,
        server_version=seen.server,
        supported_contracts=SUPPORTED_HTTP_CONTRACTS,
        server_contracts=seen.supported,
    )
