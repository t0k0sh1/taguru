/**
 * Durable per-chunk checkpoint stores for `TaguruIngester` (ingest.ts;
 * issue #212, the TypeScript twin of the Python `taguru_langchain`
 * package's `checkpoints.py`, itself the twin of `taguru extract`'s issue
 * #210).
 *
 * The SDK owns every correctness-critical piece of checkpoint/resume — the
 * fingerprint gate that invalidates a whole document's cache on any
 * output-shaping settings change, the unit-content-hash keying that
 * survives a differently-chunked resume, and the JSON schema on disk. A
 * `CheckpointStore` is only durable blob storage keyed by source id;
 * implementing one for object storage or a database is three small async
 * methods, and none of them can cause a false reuse even in principle —
 * the ingester parses, structurally compares the fingerprint, and degrades
 * silently to "nothing cached" on anything it does not recognize.
 *
 * `FilesystemCheckpointStore` is the batteries-included default: one JSON
 * file per document, the SDK analogue of `taguru extract`'s
 * `.extract-checkpoints/` directory.
 *
 * Constraint: no top-level `node:` import in this module. Hashing uses the
 * Web Crypto global (`crypto.subtle`, available in Node >= 20 with no
 * import and browser-safe), so `ingest.ts` can statically import the
 * fingerprint/unit machinery without dragging Node builtins into a
 * non-Node bundle. `FilesystemCheckpointStore` alone needs `node:fs`/
 * `node:path`; it imports them dynamically inside its methods, the same
 * pattern the core SDK's `Context.exportToFile` already uses
 * (sdk/typescript/src/client.ts).
 */

import type { BaseChatModel } from "@langchain/core/language_models/chat_models";

import type { ModelOutput } from "./extract.js";

/**
 * Durable blob storage for one document's checkpoint state, keyed by
 * source id.
 *
 * Contract the ingester depends on:
 *
 * - `save` MUST be atomic: a concurrent or post-crash `load` returns
 *   either a previously saved value in full or the new one in full, never
 *   a partial write. `FilesystemCheckpointStore` gets this via a
 *   temp-file-then-rename.
 * - `load` resolves to `null` when nothing was ever saved for `source` —
 *   the ordinary first-run/never-checkpointed case, not a failure.
 * - A `load`/`save`/`delete` call MAY reject. `load`/`save` failures are
 *   caught by the ingester and reported via `console.warn`; the run
 *   proceeds as though nothing were cached (`load`) or the completed
 *   chunk still counts for this run (`save`). A `delete` failure is
 *   silently ignored — nothing correctness-critical depends on prompt
 *   cleanup.
 */
export interface CheckpointStore {
  /** Resolve to the last value saved for `source`, or `null` if none was ever saved. */
  load(source: string): Promise<Uint8Array | null>;
  /** Atomically persist `data` as the sole value for `source`, replacing whatever was saved before. */
  save(source: string, data: Uint8Array): Promise<void>;
  /** Remove any saved value for `source`. Deleting an already-empty source is not an error. */
  delete(source: string): Promise<void>;
}

/** SHA-256 of `text`'s UTF-8 bytes, as lowercase hex — the one hash primitive
 * every checkpoint identity (fingerprint content hash, unit hash, long
 * source-name suffix) is built from. */
export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

function flattenSource(source: string): string {
  return source.replaceAll("/", "__").replaceAll("\\", "__").replaceAll(":", "__");
}

/**
 * The largest prefix of `text` whose UTF-8 encoding is at most `maxBytes`
 * long, backing off to a valid character boundary rather than splitting a
 * multi-byte codepoint — mirrors the Rust/Python twins' truncation exactly.
 * `TextDecoder` alone can't do this: decoding a byte-sliced prefix directly
 * would substitute U+FFFD for a codepoint split mid-sequence instead of
 * dropping it, so back off over UTF-8 continuation bytes (`10xxxxxx`,
 * i.e. `(byte & 0xc0) === 0x80`) first.
 */
function truncateUtf8Prefix(text: string, maxBytes: number): string {
  const bytes = new TextEncoder().encode(text);
  if (bytes.length <= maxBytes) {
    return text;
  }
  let end = maxBytes;
  while (end > 0 && (bytes[end]! & 0xc0) === 0x80) {
    end -= 1;
  }
  return new TextDecoder().decode(bytes.subarray(0, end));
}

/**
 * Path separators and `:` flatten to `__` so the checkpoint directory stays
 * flat and the name stays human-readable, then a 16-hex-character
 * content-hash suffix is ALWAYS appended — flattening alone is not
 * injective (`"a/b"`, `"a:b"`, and `"a__b"` all flatten to `"a__b"`), so
 * without the suffix, distinct short source ids could collide on the same
 * file and silently overwrite each other's checkpoint progress. A
 * flattened name over 120 UTF-8 bytes also truncates to a <=96-byte prefix
 * so long source paths never blow a filesystem's name-length limit; the
 * hash suffix alone (not the truncated prefix) is what keeps two such
 * names apart. Same scheme as `taguru extract`'s `checkpoint_file_name`
 * (src/extract.rs) and `langchain-taguru`'s `_checkpoint_file_name`
 * (checkpoints.py), which originally only suffixed past the 120-byte
 * threshold and shared this same short-name collision risk until issue
 * #227 fixed all three ports to always suffix.
 */
export async function checkpointFileName(source: string): Promise<string> {
  const flattened = flattenSource(source);
  const prefix =
    new TextEncoder().encode(flattened).length > 120
      ? truncateUtf8Prefix(flattened, 96)
      : flattened;
  const suffix = (await sha256Hex(source)).slice(0, 16);
  return `${prefix}-${suffix}.json`;
}

/**
 * Best-effort parent-directory fsync: the rename in `writeAtomic` is itself
 * an entry in the parent directory's own data, so without this a crash
 * could forget the rename even though the file's own contents reached
 * disk. Some platforms (Windows) cannot open a directory for fsync at all
 * — silently skipped there; the file's own fsync is still the durability
 * floor.
 */
async function fsyncDir(directory: string): Promise<void> {
  const { open } = await import("node:fs/promises");
  let handle;
  try {
    handle = await open(directory, "r");
  } catch {
    return;
  }
  try {
    await handle.sync();
  } catch {
    // best-effort
  } finally {
    await handle.close();
  }
}

/**
 * Writes via a temporary sibling file, fsync, and rename — a crash
 * mid-write leaves the previous version intact, and power loss right
 * after this call resolves cannot tear or lose the new one. Mirrors
 * src/storage.rs's `write_atomic`. The temp file is closed before the
 * rename (unlike the Python twin's still-open `fdopen` scope): Windows
 * cannot rename a file that is still open.
 */
async function writeAtomic(path: string, data: Uint8Array): Promise<void> {
  const { mkdir, open, rename, unlink } = await import("node:fs/promises");
  const { dirname, basename, join } = await import("node:path");
  const dir = dirname(path);
  await mkdir(dir, { recursive: true });
  const staged = join(dir, `.${basename(path)}.${crypto.randomUUID()}.tmp`);
  const handle = await open(staged, "w");
  try {
    try {
      await handle.write(data);
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(staged, path);
  } catch (error) {
    // The staged file never became `path` — clean up its temporary name
    // rather than leave partial-write litter behind. A failure after a
    // successful `rename` never reaches here (`staged` no longer exists
    // under its temporary name at that point).
    await unlink(staged).catch(() => {});
    throw error;
  }
  await fsyncDir(dir);
}

/**
 * One JSON file per document under `directory` — the SDK analogue of
 * `taguru extract`'s `.extract-checkpoints/`. `directory` is never created
 * until the first `save`; a source with nothing checkpointed yet never
 * gets a file.
 */
export class FilesystemCheckpointStore implements CheckpointStore {
  private readonly directory: string;

  constructor(directory: string) {
    this.directory = directory;
  }

  /**
   * The file `source` maps to — public so an operator can inspect or
   * remove one document's checkpoints by hand (the SDK has no `--force`
   * equivalent; deleting this file is the manual invalidation path).
   */
  async pathFor(source: string): Promise<string> {
    const { join } = await import("node:path");
    return join(this.directory, await checkpointFileName(source));
  }

  async load(source: string): Promise<Uint8Array | null> {
    const { readFile } = await import("node:fs/promises");
    try {
      return await readFile(await this.pathFor(source));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return null;
      }
      throw error;
    }
  }

  async save(source: string, data: Uint8Array): Promise<void> {
    await writeAtomic(await this.pathFor(source), data);
  }

  async delete(source: string): Promise<void> {
    const { rm } = await import("node:fs/promises");
    await rm(await this.pathFor(source), { force: true });
  }
}

/**
 * Best-effort model identity for the checkpoint fingerprint, prefixed with
 * the wrapper's class name so two providers exposing the same model string
 * (e.g. two different "llama3" backends) can never collide. Checked once
 * at `TaguruIngester` construction time so an ingester with an
 * unidentifiable model fails fast (`checkpoint_model_id` is then required)
 * rather than checkpointing under an unstable or degenerate key.
 *
 * Attribute names are the LangChain.js-idiomatic camelCase siblings of the
 * Python twin's `model`/`model_name`/`model_id`/`deployment_name` walk:
 * `model` is current on ChatOpenAI/ChatAnthropic/ChatOllama/
 * ChatBedrockConverse; `modelName` is still populated as a legacy alias on
 * ChatOpenAI/ChatVertexAI; `modelId`/`deploymentName` cover AWS/Azure-style
 * wrappers (AzureChatOpenAI itself inherits ChatOpenAI's `model`, so this
 * walk already resolves it without a fifth name).
 */
export function deriveModelIdentity(llm: BaseChatModel): string | null {
  const attrs = ["model", "modelName", "modelId", "deploymentName"] as const;
  for (const attr of attrs) {
    const value = (llm as unknown as Record<string, unknown>)[attr];
    if (typeof value === "string" && value !== "") {
      return `${llm.constructor.name}:${value}`;
    }
  }
  return null;
}

/**
 * The same compatibility inputs a batch's manifest entry would check,
 * minus anything that shapes only how many corrective turns a chunk takes
 * rather than the validity of an accepted output. Any field mismatch
 * (content edited, model/prompt/questions/fact_budget/structured_output/
 * lossy changed) treats the whole checkpoint file as absent — a settings
 * change can never silently reuse an incompatible cached chunk. Mirrors
 * src/extract.rs's `CheckpointFingerprint` (minus `max_output_tokens`,
 * which has no TypeScript equivalent, and `max_attempts`/`chunk_bytes`,
 * excluded for the same reason as the Rust/Python twins: a validated
 * output does not depend on retry budget, and a chunk-size change just
 * makes today's pieces hash differently — a safe cache miss, not a
 * correctness hazard).
 *
 * Field names are snake_case (matching the SDKs' wire-vocabulary
 * convention) and equal to their on-disk JSON keys.
 */
export interface CheckpointFingerprint {
  sha256: string;
  model: string;
  prompt_version: number;
  context: string;
  questions_n: number;
  no_passage: boolean;
  description: string;
  fact_budget: number;
  structured_output: string;
  lossy: boolean;
  /**
   * The fetched schema document's digest (`""` = no schema). A checkpoint
   * file written before this field existed defaults to `""` on load
   * (`fingerprintFromJson`) — matches a schema-less rerun, the same
   * "new field defaults to the value that changes today's behavior least"
   * precedent every other field here already sets. Mirrors
   * src/extract.rs's `CheckpointFingerprint.schema_digest`.
   */
  schema_digest: string;
}

function fingerprintFromJson(data: unknown): CheckpointFingerprint | null {
  if (typeof data !== "object" || data === null) {
    return null;
  }
  const raw = data as Record<string, unknown>;
  const { sha256, model, prompt_version, context, questions_n } = raw;
  const { no_passage, description, fact_budget, structured_output, lossy } = raw;
  const { schema_digest } = raw;
  if (
    typeof sha256 !== "string" ||
    typeof model !== "string" ||
    !Number.isInteger(prompt_version) ||
    typeof context !== "string" ||
    !Number.isInteger(questions_n) ||
    typeof no_passage !== "boolean" ||
    typeof description !== "string" ||
    !Number.isInteger(fact_budget) ||
    typeof structured_output !== "string" ||
    typeof lossy !== "boolean" ||
    (schema_digest !== undefined && typeof schema_digest !== "string")
  ) {
    return null;
  }
  return {
    sha256,
    model,
    prompt_version: prompt_version as number,
    context,
    questions_n: questions_n as number,
    no_passage,
    description,
    fact_budget: fact_budget as number,
    structured_output,
    lossy,
    schema_digest: typeof schema_digest === "string" ? schema_digest : "",
  };
}

function fingerprintsEqual(a: CheckpointFingerprint, b: CheckpointFingerprint): boolean {
  return (
    a.sha256 === b.sha256 &&
    a.model === b.model &&
    a.prompt_version === b.prompt_version &&
    a.context === b.context &&
    a.questions_n === b.questions_n &&
    a.no_passage === b.no_passage &&
    a.description === b.description &&
    a.fact_budget === b.fact_budget &&
    a.structured_output === b.structured_output &&
    a.lossy === b.lossy &&
    a.schema_digest === b.schema_digest
  );
}

/**
 * One durable unit of extraction work — a top-level chunk (TypeScript has
 * no length-ladder split rung yet, but the schema and the unit-hash keying
 * already anticipate one, exactly like the Rust/Python twins'
 * `CheckpointUnit` anticipated issue #179's amendment before #210
 * implemented it). Keyed by the unit's OWN content hash rather than
 * `chunk_index` alone, so a resumed run whose chunking differs from a
 * prior one never misattributes a piece's output to the wrong text.
 * `user`/`answer` are stored (not just `output`) so a reused unit can
 * still participate in Stage 2 cross-chunk correction exactly like a
 * freshly-extracted one — `correctCrossChunkIssues` rebuilds a chunk's own
 * conversation from these.
 */
export interface CheckpointUnit {
  /** The chunk's coordinates in ITS run, kept only for the same "chunk i/n"
   * reporting ChunkCompleted already carries — not part of the cache key. */
  chunk_index: number;
  output: ModelOutput;
  /** The exact user turn that produced `output`. */
  user: string;
  /** The model's own final accepted answer text. */
  answer: string;
}

function modelOutputFromJson(data: unknown): ModelOutput | null {
  if (typeof data !== "object" || data === null) {
    return null;
  }
  const raw = data as Record<string, unknown>;
  const { associations, aliases, questions } = raw;
  if (
    !Array.isArray(associations) ||
    !Array.isArray(aliases) ||
    !Array.isArray(questions) ||
    associations.some((item) => typeof item !== "object" || item === null) ||
    aliases.some((item) => typeof item !== "object" || item === null) ||
    questions.some((item) => typeof item !== "object" || item === null)
  ) {
    return null;
  }
  return {
    associations: associations as ModelOutput["associations"],
    aliases: aliases as ModelOutput["aliases"],
    questions: questions as ModelOutput["questions"],
  };
}

function checkpointUnitFromJson(data: unknown): CheckpointUnit | null {
  if (typeof data !== "object" || data === null) {
    return null;
  }
  const raw = data as Record<string, unknown>;
  const { chunk_index, output, user, answer } = raw;
  if (!Number.isInteger(chunk_index) || typeof user !== "string" || typeof answer !== "string") {
    return null;
  }
  const parsedOutput = modelOutputFromJson(output);
  if (parsedOutput === null) {
    return null;
  }
  return { chunk_index: chunk_index as number, output: parsedOutput, user, answer };
}

/**
 * One document's durable checkpoint state: the settings it was extracted
 * under, and every unit completed so far, keyed by content hash.
 * Serialized as one small JSON object, rewritten in full (`writeAtomic`)
 * after every new unit lands — small enough that a read-modify-write-
 * whole-file is simpler, and just as durable, as an append-only log.
 *
 * `units` is a `Map` rather than a plain object so a maliciously- or
 * accidentally-shaped on-disk file (e.g. a `"__proto__"` key) can never
 * touch anything but the map's own entries.
 */
export class DocumentCheckpoints {
  readonly fingerprint: CheckpointFingerprint;
  private readonly units: Map<string, CheckpointUnit>;

  private constructor(fingerprint: CheckpointFingerprint, units: Map<string, CheckpointUnit>) {
    this.fingerprint = fingerprint;
    this.units = units;
  }

  static empty(fingerprint: CheckpointFingerprint): DocumentCheckpoints {
    return new DocumentCheckpoints(fingerprint, new Map());
  }

  /**
   * Missing, unreadable, corrupt, or fingerprint-mismatched checkpoints all
   * degrade to "nothing cached" — never an error, and never a false reuse
   * of an incompatible output. A single parse failure anywhere in the file
   * (including one corrupted unit) invalidates the WHOLE file rather than
   * salvaging the units that happen to still parse, mirroring the Rust
   * twin's single-parse posture: a partially-trustworthy checkpoint file
   * is treated exactly like an absent one.
   */
  static fromBytes(
    data: Uint8Array | null,
    fingerprint: CheckpointFingerprint,
  ): DocumentCheckpoints {
    const parsed = DocumentCheckpoints.parse(data, fingerprint);
    return parsed ?? DocumentCheckpoints.empty(fingerprint);
  }

  private static parse(
    data: Uint8Array | null,
    fingerprint: CheckpointFingerprint,
  ): DocumentCheckpoints | null {
    if (data === null) {
      return null;
    }
    let raw: unknown;
    try {
      raw = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(data));
    } catch {
      return null;
    }
    if (typeof raw !== "object" || raw === null) {
      return null;
    }
    const container = raw as Record<string, unknown>;
    const loadedFingerprint = fingerprintFromJson(container.fingerprint);
    if (loadedFingerprint === null || !fingerprintsEqual(loadedFingerprint, fingerprint)) {
      return null;
    }
    const rawUnits = container.units;
    if (typeof rawUnits !== "object" || rawUnits === null || Array.isArray(rawUnits)) {
      return null;
    }
    const units = new Map<string, CheckpointUnit>();
    for (const [key, value] of Object.entries(rawUnits as Record<string, unknown>)) {
      const unit = checkpointUnitFromJson(value);
      if (unit === null) {
        return null;
      }
      units.set(key, unit);
    }
    return new DocumentCheckpoints(fingerprint, units);
  }

  toBytes(): Uint8Array {
    const payload = {
      fingerprint: this.fingerprint,
      units: Object.fromEntries(this.units),
    };
    return new TextEncoder().encode(JSON.stringify(payload));
  }

  lookup(unitHash: string): CheckpointUnit | null {
    return this.units.get(unitHash) ?? null;
  }

  record(unitHash: string, unit: CheckpointUnit): void {
    this.units.set(unitHash, unit);
  }
}
