/**
 * Test helpers: spawn a real taguru server binary for integration tests
 * (Node-only; import from "taguru/testing"). The Python twin is
 * `taguru.testing`. Mirrors the server's own harness (tests/http_api.rs).
 */

import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

export const REPO_ROOT = resolve(import.meta.dirname, "../../..");

export const ADMIN_TOKEN = "test-admin-token";
export const READER_TOKEN = "test-reader-token";

export function serverBinary(repoRoot?: string): string {
  const override = process.env["TAGURU_TEST_BIN"];
  if (override) {
    return override;
  }
  // REPO_ROOT is right only while this file physically lives at
  // <repo>/sdk/typescript — a packed or published install sits under some
  // consumer's node_modules, where the guess points at nothing buildable.
  const root = repoRoot ?? REPO_ROOT;
  if (!existsSync(join(root, "Cargo.toml"))) {
    throw new Error(
      `${root} is not the taguru repository — pass serverBinary(repoRoot) or set TAGURU_TEST_BIN`,
    );
  }
  execFileSync("cargo", ["build", "--quiet", "--bin", "taguru"], {
    cwd: root,
    stdio: "inherit",
  });
  return join(root, "target", "debug", "taguru");
}

export interface SpawnedServer {
  baseUrl: string;
  child: ChildProcess;
  stop(): void;
}

export async function spawnServer(
  binary: string,
  extraEnv: Record<string, string>,
): Promise<SpawnedServer> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    // Hermetic: the developer shell must not leak embedding providers,
    // tokens, tracing, or a config file into tests.
    if (value !== undefined && !key.startsWith("TAGURU_") && !key.startsWith("OTEL_")) {
      env[key] = value;
    }
  }
  Object.assign(env, {
    TAGURU_ADDR: "127.0.0.1:0",
    TAGURU_DATA_DIR: mkdtempSync(join(tmpdir(), "taguru-ts-test-")),
    TAGURU_FLUSH_SECS: "1",
  });
  Object.assign(env, extraEnv);

  const child = spawn(binary, [], { env, stdio: ["ignore", "pipe", "ignore"] });
  const baseUrl = await new Promise<string>((resolvePromise, rejectPromise) => {
    let buffer = "";
    let resolved = false;
    child.stdout!.on("data", (chunk: Buffer) => {
      if (resolved) {
        return; // keep draining so the server never blocks on a full pipe
      }
      buffer += chunk.toString("utf-8");
      const match = /^listening on (.+)$/m.exec(buffer);
      if (match) {
        resolved = true;
        resolvePromise(`http://${match[1]!.trim()}`);
      }
    });
    child.on("exit", (code) => {
      if (!resolved) {
        rejectPromise(new Error(`server exited before printing its address (code ${code})`));
      }
    });
  });
  return {
    baseUrl,
    child,
    stop() {
      child.kill("SIGTERM");
    },
  };
}
