"""Minimal HTTP server harness for `HtmlConnector`'s URL-fetch tests (issue
#349) — a stdlib `ThreadingHTTPServer` serving canned, in-memory responses,
synthesized at test time rather than depending on any external service or
recorded fixture, the same "no binary fixture" convention `tests/_pdfs.py`
already established for PDF bytes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Route:
    """One canned response. `location` set makes this a 302 redirect
    (`status`/`content_type`/`body` are then ignored); otherwise `status` is
    served with `body`, and `content_type` becomes the `Content-Type` header
    (omit to send no `Content-Type` header at all). `delay` (seconds) sleeps
    before responding at all — enough to exercise a caller's per-phase
    `timeout`. `chunk_delay` instead trickles `body` out in small pieces,
    sleeping between each — every individual gap stays under a per-phase
    `timeout`, so this is what exercises a caller's separate total-time
    budget instead; no `Content-Length` is sent for a trickled response
    (`protocol_version` defaults to HTTP/1.0, so the client reads until the
    connection closes)."""

    status: int = 200
    content_type: str | None = "text/html; charset=utf-8"
    body: bytes = b""
    delay: float = 0.0
    chunk_delay: float = 0.0
    location: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


class _Handler(BaseHTTPRequestHandler):
    server: _RouteServer  # narrows the base class's plain socketserver type

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        pass  # keep test output quiet; failures still show via assertions

    def do_GET(self) -> None:  # noqa: N802 - stdlib-mandated method name
        # `self.path` is the full request target, query string included —
        # route registration is by path only, so a request carrying a query
        # string (the credential/token-stripping tests deliberately send
        # one) must not be matched against it too.
        route = self.server.routes.get(urlsplit(self.path).path)
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        if route.delay:
            time.sleep(route.delay)
        if route.location is not None:
            self.send_response(302)
            self.send_header("Location", route.location)
            self.end_headers()
            return
        self.send_response(route.status)
        if route.content_type is not None:
            self.send_header("Content-Type", route.content_type)
        for name, value in route.headers.items():
            self.send_header(name, value)
        if route.chunk_delay:
            self.end_headers()
            chunk_size = 16
            for offset in range(0, len(route.body), chunk_size):
                self.wfile.write(route.body[offset : offset + chunk_size])
                self.wfile.flush()
                time.sleep(route.chunk_delay)
            return
        self.send_header("Content-Length", str(len(route.body)))
        self.end_headers()
        self.wfile.write(route.body)


class _RouteServer(ThreadingHTTPServer):
    def __init__(self, routes: dict[str, Route]) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.routes = routes

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def handle_error(self, request: object, client_address: object) -> None:
        # A client that gives up mid-response (the `timeout`/oversized-body
        # tests deliberately do this) leaves this server thread writing to a
        # closed socket; the base class's default `handle_error` prints a
        # full traceback to stderr for that, which is expected here, not a
        # real failure — swallow only the connection-teardown exceptions,
        # let anything else still print.
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


@contextmanager
def serve(routes: dict[str, Route]) -> Iterator[_RouteServer]:
    """Serves `routes` (path -> `Route`) on an ephemeral `127.0.0.1` port for
    the lifetime of the `with` block. Usage::

        with serve({"/page": Route(body=b"<html>...</html>")}) as server:
            document = HtmlConnector().read(f"{server.base_url}/page")
    """
    server = _RouteServer(routes)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
