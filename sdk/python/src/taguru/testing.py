"""Test helpers: spawn a real taguru server binary for integration tests.

Used by this repo's own SDK test suites (core and langchain), and usable by
applications that want to test against a real server. The spawn mirrors the
server's own integration harness (tests/http_api.rs): hermetic environment,
``TAGURU_ADDR=127.0.0.1:0``, address read from stdout.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

__all__ = ["SpawnedServer", "default_binary"]


def default_binary(repo_root: str | Path | None = None) -> Path:
    """The server binary: ``$TAGURU_TEST_BIN`` when set, else a
    ``cargo build --bin taguru`` of ``repo_root`` (required without the env)."""
    override = os.environ.get("TAGURU_TEST_BIN")
    if override:
        return Path(override)
    if repo_root is None:
        raise RuntimeError("TAGURU_TEST_BIN is not set and no repo_root was given")
    root = Path(repo_root)
    subprocess.run(["cargo", "build", "--quiet", "--bin", "taguru"], cwd=root, check=True)
    return root / "target" / "debug" / "taguru"


class SpawnedServer:
    """One spawned server process bound to an ephemeral port.

    The environment is hermetic: every ``TAGURU_*``/``OTEL_*`` variable from
    the calling shell is dropped before ``extra_env`` applies.
    """

    def __init__(self, binary: str | Path, data_dir: str | Path, extra_env: dict[str, str]) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("TAGURU_", "OTEL_"))
        }
        env.update(
            {
                "TAGURU_ADDR": "127.0.0.1:0",
                "TAGURU_DATA_DIR": str(data_dir),
                "TAGURU_FLUSH_SECS": "1",
            }
        )
        env.update(extra_env)
        self.process = subprocess.Popen(
            [str(binary)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
        )
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.startswith("listening on "):
                self.base_url = "http://" + line.removeprefix("listening on ").strip()
                break
        else:
            raise RuntimeError("server exited before printing its address")
        # Keep draining stdout so the server never blocks on a full pipe.
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        assert self.process.stdout is not None
        for _line in self.process.stdout:
            pass

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
