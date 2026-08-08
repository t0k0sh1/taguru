/**
 * FilesystemCheckpointStore and DocumentCheckpoints: file naming,
 * atomicity, and the fingerprint/degradation contract in isolation from
 * TaguruIngester. Mirrors the Python suite's test_checkpoint_store.py.
 */

import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import * as fsPromises from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { FakeListChatModel } from "@langchain/core/utils/testing";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  checkpointFileName,
  CheckpointLockedError,
  deriveModelIdentity,
  DocumentCheckpoints,
  FilesystemCheckpointStore,
  type CheckpointFingerprint,
  type CheckpointUnit,
} from "../../src/checkpoints.js";

vi.mock("node:fs/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs/promises")>();
  return { ...actual, rename: vi.fn(actual.rename) };
});

const encode = (text: string): Uint8Array => new TextEncoder().encode(text);
const decode = (data: Uint8Array): string => new TextDecoder().decode(data);

const fingerprint = (overrides: Partial<CheckpointFingerprint> = {}): CheckpointFingerprint => ({
  sha256: "deadbeef",
  model: "fake-model",
  prompt_version: 2,
  context: "sake",
  questions_n: 2,
  no_passage: false,
  description: "",
  fact_budget: 0,
  structured_output: "",
  lossy: false,
  schema_digest: "",
  ...overrides,
});

const unit = (overrides: Partial<CheckpointUnit> = {}): CheckpointUnit => ({
  chunk_index: 0,
  output: { associations: [], aliases: [], questions: [] },
  user: "u",
  answer: "a",
  ...overrides,
});

/** Reads the (private) unit map back out through the only public surface
 * DocumentCheckpoints exposes for it: its own serialization. */
const unitKeys = (checkpoints: DocumentCheckpoints): string[] =>
  Object.keys(JSON.parse(decode(checkpoints.toBytes())).units as Record<string, unknown>);

describe("FilesystemCheckpointStore", () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "taguru-ckpt-"));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it("round-trips units through save and load", async () => {
    const store = new FilesystemCheckpointStore(dir);
    expect(await store.load("docs/aomine.md")).toBeNull();
    await store.save("docs/aomine.md", encode('{"fingerprint": {}, "units": {}}'));
    expect(decode((await store.load("docs/aomine.md"))!)).toBe('{"fingerprint": {}, "units": {}}');
    await store.save("docs/aomine.md", encode('{"fingerprint": {}, "units": {"x": 1}}'));
    expect(decode((await store.load("docs/aomine.md"))!)).toBe(
      '{"fingerprint": {}, "units": {"x": 1}}',
    );
  });

  it("flattens path separators in the checkpoint file name", async () => {
    const name = await checkpointFileName("docs/a:b\\c.md");
    expect(name.startsWith("docs__a__b__c.md-")).toBe(true);
    expect(name.endsWith(".json")).toBe(true);
  });

  it("always appends a hash suffix, even for short names that flatten identically", async () => {
    // flattenSource alone is not injective — these three distinct source
    // ids all flatten to "a__b" — so the hash suffix (not the flattened
    // prefix) is what must keep their checkpoint files apart.
    const names = await Promise.all(
      ["a/b", "a:b", "a__b"].map((source) => checkpointFileName(source)),
    );
    expect(new Set(names).size).toBe(3);
    for (const name of names) {
      expect(name.startsWith("a__b-")).toBe(true);
    }
  });

  it("truncates long sources with a hash suffix at a UTF-8 boundary", async () => {
    // 50 Japanese characters * 3 bytes/char = 150 UTF-8 bytes, over the
    // 120-byte flatten threshold, and straddling the 96-byte truncation
    // point mid-codepoint (96 is not a multiple of 3) — exercises the
    // UTF-8 boundary backoff, not just byte-length truncation.
    const source = "文書/" + "あ".repeat(50) + ".md";
    const name = await checkpointFileName(source);
    expect(name.endsWith(".json")).toBe(true);
    const stem = name.slice(0, -".json".length);
    const dashIndex = stem.lastIndexOf("-");
    const prefix = stem.slice(0, dashIndex);
    const suffix = stem.slice(dashIndex + 1);
    expect(new TextEncoder().encode(prefix).length).toBeLessThanOrEqual(96);
    expect(suffix.length).toBe(16);
    // The prefix is a clean, valid-UTF-8 truncation of the flattened
    // source (no split codepoint) — it must itself be a text prefix of
    // "文書__" + 50 "あ"s, never a mangled partial character.
    expect(("文書__" + "あ".repeat(50)).startsWith(prefix)).toBe(true);
  });

  it("keeps distinct long sources distinct after truncation", async () => {
    // Two sources sharing the same >120-byte prefix must still map to
    // different files — the hash suffix, not the truncated prefix alone,
    // provides uniqueness.
    const base = "d".repeat(130);
    const nameA = await checkpointFileName(base + "-a");
    const nameB = await checkpointFileName(base + "-b");
    expect(nameA).not.toBe(nameB);
  });

  it("atomic write leaves no tmp litter", async () => {
    const store = new FilesystemCheckpointStore(dir);
    await store.save("docs/aomine.md", encode("{}"));
    // close() releases the advisory lock (removing its sidecar), so only
    // the checkpoint file itself remains.
    await store.close();
    expect(readdirSync(dir)).toEqual([await checkpointFileName("docs/aomine.md")]);
  });

  it("a failed rename keeps the old file and cleans up staging", async () => {
    const store = new FilesystemCheckpointStore(dir);
    await store.save("docs/aomine.md", encode("original"));
    vi.mocked(fsPromises.rename).mockRejectedValueOnce(new Error("simulated replace failure"));
    await expect(store.save("docs/aomine.md", encode("replacement"))).rejects.toThrow();
    expect(decode((await store.load("docs/aomine.md"))!)).toBe("original");
    await store.close();
    expect(readdirSync(dir)).toEqual([await checkpointFileName("docs/aomine.md")]);
  });

  it("a second store instance is locked out until the first releases", async () => {
    const first = new FilesystemCheckpointStore(dir);
    const second = new FilesystemCheckpointStore(dir);
    await first.save("docs/aomine.md", encode("one"));
    await expect(second.save("docs/aomine.md", encode("two"))).rejects.toThrow(
      CheckpointLockedError,
    );
    // The loser's failed save must not have clobbered the holder's state.
    expect(decode((await first.load("docs/aomine.md"))!)).toBe("one");
    // A DIFFERENT source is not contended.
    await second.save("docs/other.md", encode("independent"));
    await first.close();
    await second.save("docs/aomine.md", encode("two"));
    expect(decode((await second.load("docs/aomine.md"))!)).toBe("two");
    await second.close();
  });

  it("repeat saves by the lock holder are not self-contention", async () => {
    const store = new FilesystemCheckpointStore(dir);
    await store.save("docs/aomine.md", encode("one"));
    await store.save("docs/aomine.md", encode("two"));
    expect(decode((await store.load("docs/aomine.md"))!)).toBe("two");
    await store.close();
  });

  it("delete releases the lock along with the checkpoint", async () => {
    const first = new FilesystemCheckpointStore(dir);
    const second = new FilesystemCheckpointStore(dir);
    await first.save("docs/aomine.md", encode("one"));
    await first.delete("docs/aomine.md");
    await second.save("docs/aomine.md", encode("two"));
    await first.close();
    await second.close();
  });

  it("a dead holder's stale lock is reclaimed, never wedging the next run", async () => {
    const { writeFile, mkdir } = fsPromises;
    const store = new FilesystemCheckpointStore(dir);
    const lockPath = (await store.pathFor("docs/aomine.md")) + ".lock";
    await mkdir(dir, { recursive: true });
    // A PID no live process holds: spawn-and-reap is racy in a unit test,
    // so use garbage content — an unreadable holder is treated as stale
    // exactly like a dead one.
    await writeFile(lockPath, "not-a-pid", "utf-8");
    await store.save("docs/aomine.md", encode("recovered"));
    expect(decode((await store.load("docs/aomine.md"))!)).toBe("recovered");
    await store.close();
  });

  it("delete refuses to remove a checkpoint another live run holds", async () => {
    // A run whose every chunk was a cache hit never saved, so it holds no
    // lock — its cleanup must not destroy the holder's on-disk state.
    const first = new FilesystemCheckpointStore(dir);
    const second = new FilesystemCheckpointStore(dir);
    await first.save("docs/aomine.md", encode("one"));
    await expect(second.delete("docs/aomine.md")).rejects.toThrow(CheckpointLockedError);
    expect(decode((await first.load("docs/aomine.md"))!)).toBe("one");
    await first.close();
  });

  it("delete of a never-saved source is a no-op that creates nothing", async () => {
    const nested = join(dir, "never-created");
    const store = new FilesystemCheckpointStore(nested);
    await store.delete("docs/aomine.md");
    expect(existsSync(nested)).toBe(false);
  });

  it("concurrent saves on one instance do not self-conflict", async () => {
    const store = new FilesystemCheckpointStore(dir);
    await Promise.all([
      store.save("docs/aomine.md", encode("a")),
      store.save("docs/aomine.md", encode("b")),
    ]);
    expect(await store.load("docs/aomine.md")).not.toBeNull();
    await store.close();
  });

  it("a live holder's lock is respected", async () => {
    const { writeFile, mkdir } = fsPromises;
    const store = new FilesystemCheckpointStore(dir);
    const lockPath = (await store.pathFor("docs/aomine.md")) + ".lock";
    await mkdir(dir, { recursive: true });
    // This test process itself is the live holder.
    await writeFile(lockPath, String(process.pid), "utf-8");
    await expect(store.save("docs/aomine.md", encode("x"))).rejects.toThrow(
      CheckpointLockedError,
    );
  });

  it("load never creates the directory and save creates it on demand", async () => {
    const nested = join(dir, "checkpoints");
    const store = new FilesystemCheckpointStore(nested);
    expect(await store.load("docs/aomine.md")).toBeNull();
    expect(existsSync(nested)).toBe(false);
    await store.save("docs/aomine.md", encode("{}"));
    expect(existsSync(nested)).toBe(true);
  });

  it("delete of a missing file is a no-op", async () => {
    const store = new FilesystemCheckpointStore(dir);
    await expect(store.delete("docs/never-saved.md")).resolves.toBeUndefined();
  });
});

describe("deriveModelIdentity", () => {
  it("reads model/modelName/modelId/deploymentName in order and returns null otherwise", () => {
    const llm = new FakeListChatModel({ responses: ["x"] });
    expect(deriveModelIdentity(llm)).toBeNull();
    (llm as unknown as { model: string }).model = "demo-v1";
    expect(deriveModelIdentity(llm)).toBe("FakeListChatModel:demo-v1");
  });
});

describe("DocumentCheckpoints", () => {
  it("round-trips through bytes", () => {
    const fp = fingerprint();
    const u = unit();
    const checkpoints = DocumentCheckpoints.empty(fp);
    checkpoints.record("hash1", u);
    const restored = DocumentCheckpoints.fromBytes(checkpoints.toBytes(), fp);
    expect(restored.lookup("hash1")).toEqual(u);
  });

  it("degrades to empty on fingerprint mismatch", () => {
    const fp = fingerprint();
    const checkpoints = DocumentCheckpoints.empty(fp);
    checkpoints.record("hash1", unit());
    const other = fingerprint({ model: "a-different-model" });
    const restored = DocumentCheckpoints.fromBytes(checkpoints.toBytes(), other);
    expect(unitKeys(restored)).toEqual([]);
    expect(restored.fingerprint).toEqual(other);
  });

  it("degrades to empty on missing data", () => {
    const fp = fingerprint();
    expect(unitKeys(DocumentCheckpoints.fromBytes(null, fp))).toEqual([]);
  });

  it.each([
    ["invalid_json", "not json at all"],
    ["wrong_top_level_type", "[1, 2, 3]"],
    ["missing_fingerprint", '{"units": {}}'],
    ["units_wrong_type", '{"fingerprint": {}, "units": "nope"}'],
  ])("degrades to empty on malformed bytes (%s)", (_label, data) => {
    const fp = fingerprint();
    expect(unitKeys(DocumentCheckpoints.fromBytes(encode(data), fp))).toEqual([]);
  });

  it("a single corrupt unit invalidates the whole file", () => {
    const fp = fingerprint();
    const goodUnit = unit();
    const payload = {
      fingerprint: fp,
      units: {
        good: { chunk_index: goodUnit.chunk_index, output: goodUnit.output, user: goodUnit.user, answer: goodUnit.answer },
        bad: { chunk_index: 0 }, // missing output/user/answer
      },
    };
    const data = encode(JSON.stringify(payload));
    const restored = DocumentCheckpoints.fromBytes(data, fp);
    // The whole file degrades to empty — the still-good "good" unit is NOT
    // salvaged, matching the Rust/Python twins' single-parse posture.
    expect(unitKeys(restored)).toEqual([]);
  });
});
