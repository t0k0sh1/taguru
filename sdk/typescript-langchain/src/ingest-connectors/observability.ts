/**
 * Cross-connector observability (ADR 0007 §11, issue #353): the one event
 * shape, run summary, and `dry_run` posture every connector driver — a
 * future `syncReferences`/`syncObjectStorage` alike — emits, instead of
 * each connector inventing its own. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.observability` module (issue #415).
 *
 * Two ideas, unchanged from the Python twin:
 *
 * - **Per-source record is an event log, not a snapshot** (ADR 0007 §11):
 *   one `SourceEvent` per phase transition, append-only. `RunReport`
 *   counts are a tally of each source's *last* phase event only — a source
 *   that reached `imported` counts once, under `imported`, never
 *   separately under `discovered`/`parsed` too.
 * - **The summary references the events, it does not inline them** — a
 *   `RunReport` from a 100k-object bucket sync stays O(1) whether or not
 *   `eventsOut` was given; `RunReport.events` is only populated when a
 *   caller asks for it (`keepEvents: true`, the default) and
 *   `RunReport.eventsPath` names where the full JSONL sidecar landed.
 *
 * Deliberate deviation from the Python port: `SourceEventSink` uses Node's
 * synchronous `fs` API (`openSync`/`writeSync`/`closeSync`) rather than
 * `fs/promises`, and imports `node:fs` statically rather than dynamically
 * inside each method (contrast checkpoints.ts's `FilesystemCheckpointStore`).
 * Python's `SourceEventSink`/`RunRecorder` are side-effecting and
 * synchronous top to bottom — the sink opens (and truncates) its file
 * eagerly in `__init__`, and `RunRecorder.discovered`/`record`/`duplicate`
 * are ordinary synchronous methods a connector calls with no `await`.
 * Mirroring that with `fs/promises` would force the file open into an
 * async factory (a JS constructor cannot `await`), which in turn would
 * force `RunRecorder`'s own constructor and every recording method to
 * become `async` — rippling through the whole connector-driver API this
 * module exists to give a shared shape to. Synchronous `fs` keeps the
 * class shape identical to the Python original and keeps the "flushed,
 * readable before `close()`" posture trivially true (no interleaving with
 * the event loop to reason about). This file does real filesystem I/O by
 * design and is never imported by ingest.ts's browser-safe core path, so
 * the bundle-safety concern that motivates checkpoints.ts's dynamic
 * `import("node:fs/promises")` does not apply here.
 */

import * as fs from "node:fs";

import { Diagnostic } from "./document.js";
import type { IngestEvent, IngestEventCallback } from "../events.js";
import type { TaguruIngester } from "../ingest.js";

/**
 * Checked for equality, like `document.ts`'s own `CONNECTOR_DOCUMENT_VERSION`
 * — a protocol-breaking change to `RunReport.toDict`'s shape bumps this,
 * never loosens the check to a range.
 */
export const RUN_SUMMARY_VERSION = 1;

export const SOURCE_EVENT_KIND = "source_event";
export const RUN_SUMMARY_KIND = "run_summary";

/** ADR 0007 §11's seven-state per-source phase vocabulary, verbatim. */
export const PHASES = [
  "discovered",
  "unchanged",
  "parsed",
  "extracted",
  "imported",
  "skipped",
  "failed",
] as const;

export type Phase = (typeof PHASES)[number];

/**
 * One phase transition for one source (ADR 0007 §11) — a source's full
 * per-run history is the ordered subsequence of a run's events sharing its
 * `source`.
 *
 * `elapsedMs` is milliseconds since this *source's* first event (always
 * `0.0` on `discovered`), not since the run started — `RunReport.durationMs`
 * covers the whole run. `bytes` is the raw byte count of the source
 * object/file where a driver knows it for free (a listing's `size`, an
 * `fs.stat().size`) — never the parsed text's byte count, which would make
 * a sum over this column meaningless across phases that carry no text at
 * all.
 */
export class SourceEvent {
  readonly source: string;
  readonly phase: Phase;
  readonly elapsedMs: number;
  readonly bytes: number;
  readonly parser: string | null;
  readonly diagnostic: Diagnostic | null;

  constructor(fields: {
    source: string;
    phase: Phase;
    elapsedMs: number;
    bytes?: number;
    parser?: string | null;
    diagnostic?: Diagnostic | null;
  }) {
    this.source = fields.source;
    this.phase = fields.phase;
    this.elapsedMs = fields.elapsedMs;
    this.bytes = fields.bytes ?? 0;
    this.parser = fields.parser ?? null;
    this.diagnostic = fields.diagnostic ?? null;
  }

  toDict(): Record<string, unknown> {
    return {
      kind: SOURCE_EVENT_KIND,
      source: this.source,
      phase: this.phase,
      elapsed_ms: this.elapsedMs,
      bytes: this.bytes,
      parser: this.parser,
      diagnostic: this.diagnostic === null ? null : this.diagnostic.toDict(),
    };
  }
}

/**
 * One run's machine-readable summary (ADR 0007 §11) — always constructed,
 * never behind an opt-in flag (contrast `taguru extract`'s
 * `--diagnostics-out`, where an unset flag means no record is ever
 * assembled at all).
 *
 * Every connector's summary shares this one key set — `tagsDropped`/
 * `deletedDetected`/`retracted` are specific to the S3 connector (ADR 0007
 * §9) but stay present (structurally zero) for every other driver, so a
 * consumer parsing one shape never has to branch on which connector
 * produced it.
 */
export class RunReport {
  readonly connector: string;
  readonly durationMs: number;
  readonly interrupted: boolean;
  readonly discovered: number;
  readonly unchanged: number;
  readonly parsed: number;
  readonly extracted: number;
  readonly imported: number;
  readonly skipped: number;
  readonly failed: number;
  readonly locatorsDropped: number;
  readonly sectionsDropped: number;
  readonly tagsDropped: number;
  readonly deletedDetected: number;
  readonly retracted: number;
  readonly events: readonly SourceEvent[];
  readonly eventsPath: string | null;

  constructor(
    fields: {
      connector?: string;
      durationMs?: number;
      interrupted?: boolean;
      discovered?: number;
      unchanged?: number;
      parsed?: number;
      extracted?: number;
      imported?: number;
      skipped?: number;
      failed?: number;
      locatorsDropped?: number;
      sectionsDropped?: number;
      tagsDropped?: number;
      deletedDetected?: number;
      retracted?: number;
      events?: readonly SourceEvent[];
      eventsPath?: string | null;
    } = {},
  ) {
    this.connector = fields.connector ?? "";
    this.durationMs = fields.durationMs ?? 0.0;
    this.interrupted = fields.interrupted ?? false;
    this.discovered = fields.discovered ?? 0;
    this.unchanged = fields.unchanged ?? 0;
    this.parsed = fields.parsed ?? 0;
    this.extracted = fields.extracted ?? 0;
    this.imported = fields.imported ?? 0;
    this.skipped = fields.skipped ?? 0;
    this.failed = fields.failed ?? 0;
    this.locatorsDropped = fields.locatorsDropped ?? 0;
    this.sectionsDropped = fields.sectionsDropped ?? 0;
    this.tagsDropped = fields.tagsDropped ?? 0;
    this.deletedDetected = fields.deletedDetected ?? 0;
    this.retracted = fields.retracted ?? 0;
    this.events = Object.freeze([...(fields.events ?? [])]);
    this.eventsPath = fields.eventsPath ?? null;
  }

  toDict(): Record<string, unknown> {
    return {
      kind: RUN_SUMMARY_KIND,
      connector_run: RUN_SUMMARY_VERSION,
      connector: this.connector,
      duration_ms: this.durationMs,
      interrupted: this.interrupted,
      discovered: this.discovered,
      unchanged: this.unchanged,
      parsed: this.parsed,
      extracted: this.extracted,
      imported: this.imported,
      skipped: this.skipped,
      failed: this.failed,
      locators_dropped: this.locatorsDropped,
      sections_dropped: this.sectionsDropped,
      tags_dropped: this.tagsDropped,
      deleted_detected: this.deletedDetected,
      retracted: this.retracted,
      events: this.events.length,
      events_path: this.eventsPath,
    };
  }

  /**
   * Renders `events` as newline-delimited JSON, one line per phase
   * transition, each terminated by its own `\n` — the same on-disk shape a
   * `SourceEventSink` writes incrementally (every `SourceEventSink.write`
   * appends one line plus a trailing newline), so this method's output is
   * byte-identical to what `eventsOut` would have written for the same
   * run. Only meaningful when `keepEvents: true` collected them
   * (`RunRecorder`'s default); `""` when there are none.
   */
  eventsJsonl(): string {
    return this.events.map((event) => JSON.stringify(event.toDict()) + "\n").join("");
  }
}

/**
 * Deprecated alias kept for issue #351/#352's published name — combines a
 * value export (`S3SyncReport === RunReport`) with a type export
 * (`S3SyncReport` as a type name for `RunReport`), the TS analogue of the
 * Python module's `S3SyncReport = RunReport` binding.
 */
export const S3SyncReport = RunReport;
export type S3SyncReport = RunReport;

/**
 * Where a `SourceEventSink` writes: a path (opened, truncated, and owned by
 * the sink — `close()` closes it) or a caller-owned file descriptor (never
 * closed by the sink, only ever written to — the caller opened it, the
 * caller closes it, exactly as `TaguruIngester.on_event` never assumes
 * ownership of a stream a caller handed it).
 *
 * This narrows the Python port's `str | Path | TextIO`: Node has no
 * synchronous, flush-on-write text stream analogous to a buffered Python
 * file object, so the caller-owned case is a raw `fs` file descriptor
 * (`fs.openSync`'s return value) instead — `fs.writeSync`/`fs.closeSync`
 * against a descriptor is the deterministic, synchronously-observable
 * equivalent this module's posture (flushed, readable before `close()`; a
 * write to an already-closed descriptor degrades instead of raising)
 * depends on. `path` is optional the same way Python's `getattr(target,
 * "name", None)` degrades to `None` for a nameless stream.
 */
export type SourceEventSinkTarget = string | { readonly fd: number; readonly path?: string };

/**
 * Append-only per-source JSONL sidecar (ADR 0007 §11), one line per phase
 * transition, written the moment it lands — mirroring `taguru extract`'s
 * own `DiagnosticsSink` (src/extract.rs:2372) field for field:
 * truncate-on-open ("describes THIS run, never a prior one appended to"),
 * one write per record, and a write/open failure warns exactly once for the
 * run then silently drops every later record rather than raising mid-sync.
 *
 * A `string` target is opened (truncating) and owned — `close` closes it.
 * A caller-supplied file descriptor is written to but never closed — the
 * caller opened it, the caller closes it.
 */
export class SourceEventSink {
  private fd: number | null;
  private readonly ownsHandle: boolean;
  private readonly pathValue: string | null;
  private warned: boolean;

  constructor(target: SourceEventSinkTarget) {
    this.warned = false;
    if (typeof target === "string") {
      this.pathValue = target;
      this.ownsHandle = true;
      try {
        this.fd = fs.openSync(target, "w");
      } catch (error) {
        console.warn(
          `SourceEventSink: opening ${JSON.stringify(target)} raised ${String(error)}; ` +
            "events will not be recorded to a file this run",
        );
        this.fd = null;
        this.warned = true;
      }
    } else {
      this.fd = target.fd;
      this.pathValue = target.path ?? null;
      this.ownsHandle = false;
    }
  }

  get path(): string | null {
    return this.pathValue;
  }

  write(event: SourceEvent): void {
    if (this.fd === null) {
      return;
    }
    try {
      fs.writeSync(this.fd, JSON.stringify(event.toDict()) + "\n");
    } catch (error) {
      // A caller-supplied descriptor this sink never owns (see the class
      // doc comment) may already be closed by the time a later `write`
      // lands; that must degrade exactly like any other write failure,
      // never throw mid-sync.
      if (!this.warned) {
        console.warn(
          `SourceEventSink: writing to ${JSON.stringify(this.pathValue)} raised ${String(error)}; ` +
            "later events this run will not be recorded",
        );
        this.warned = true;
      }
      this.fd = null;
    }
  }

  /** Safe to call more than once; a caller-supplied descriptor is never closed here. */
  close(): void {
    if (this.ownsHandle && this.fd !== null) {
      try {
        fs.closeSync(this.fd);
      } catch {
        // best-effort
      }
      this.fd = null;
    }
  }
}

interface SourceState {
  started: number;
  phase: Phase;
  bytes: number;
}

export interface RunRecorderFields {
  connector: string;
  eventsOut?: SourceEventSinkTarget;
  /** Default `true`, chosen so an existing caller sees no change. */
  keepEvents?: boolean;
}

/**
 * Collects one run's `SourceEvent` history and derives its `RunReport` (ADR
 * 0007 §11) — the shared bookkeeping every connector driver uses instead of
 * each maintaining its own tally.
 *
 * `eventsOut`, when given, is honored even under a dry run: the sidecar is
 * the dry run's entire product, and the caller named the path explicitly —
 * writing to it does not violate dry-run's "touch nothing" contract the way
 * writing to the corpus or a checkpoint store would (ADR 0007 §11.1).
 */
export class RunRecorder {
  private readonly connector: string;
  private readonly keepEvents: boolean;
  private readonly sink: SourceEventSink | null;
  private readonly events: SourceEvent[];
  private readonly lastPhase: Map<string, Phase>;
  private readonly sources: Map<string, SourceState>;
  private readonly runStarted: number;
  private locatorsDropped: number;
  private sectionsDropped: number;
  private tagsDropped: number;
  private deletedDetected: number;
  private retracted: number;

  constructor(fields: RunRecorderFields) {
    this.connector = fields.connector;
    this.keepEvents = fields.keepEvents ?? true;
    this.sink = fields.eventsOut === undefined ? null : new SourceEventSink(fields.eventsOut);
    this.events = [];
    this.lastPhase = new Map();
    this.sources = new Map();
    this.runStarted = performance.now();
    this.locatorsDropped = 0;
    this.sectionsDropped = 0;
    this.tagsDropped = 0;
    this.deletedDetected = 0;
    this.retracted = 0;
  }

  private emit(
    source: string,
    phase: Phase,
    options: {
      elapsedMs: number;
      sourceBytes: number;
      parser: string | null;
      diagnostic: Diagnostic | null;
    },
  ): void {
    const event = new SourceEvent({
      source,
      phase,
      elapsedMs: options.elapsedMs,
      bytes: options.sourceBytes,
      parser: options.parser,
      diagnostic: options.diagnostic,
    });
    if (this.keepEvents) {
      this.events.push(event);
    }
    this.sink?.write(event);
    this.lastPhase.set(source, phase);
  }

  /**
   * Records `source`'s `discovered` event. Always `elapsedMs === 0.0` —
   * every later `record` for the same source is timed from this call, not
   * from the run's own start. `sourceBytes` is remembered and reused by
   * `record` for later phases of the same source, since an object's own
   * size does not change between phases (unless a later call overrides it
   * explicitly).
   */
  discovered(source: string, options?: { sourceBytes?: number }): void {
    const sourceBytes = options?.sourceBytes ?? 0;
    this.sources.set(source, { started: performance.now(), phase: "discovered", bytes: sourceBytes });
    this.emit(source, "discovered", { elapsedMs: 0.0, sourceBytes, parser: null, diagnostic: null });
  }

  /**
   * Records `source`'s transition to `phase`. `sourceBytes` defaults to
   * whatever `discovered` recorded for this source when omitted — pass it
   * explicitly only when a later phase learns a size `discovered` did not
   * have (e.g. after a redirect); the explicit value is then remembered for
   * any phase still to come, the same way `discovered`'s own value is.
   */
  record(
    source: string,
    phase: Phase,
    options?: { sourceBytes?: number; parser?: string | null; diagnostic?: Diagnostic | null },
  ): void {
    const state = this.sources.get(source);
    const started = state?.started ?? performance.now();
    const elapsedMs = performance.now() - started;
    let resolvedBytes = options?.sourceBytes ?? 0;
    if (state !== undefined) {
      state.phase = phase;
      if (options?.sourceBytes !== undefined) {
        state.bytes = options.sourceBytes;
      } else {
        resolvedBytes = state.bytes;
      }
    } else {
      this.sources.set(source, { started, phase, bytes: resolvedBytes });
    }
    this.emit(source, phase, {
      elapsedMs,
      sourceBytes: resolvedBytes,
      parser: options?.parser ?? null,
      diagnostic: options?.diagnostic ?? null,
    });
  }

  /**
   * Renames `old` to `new` in this run's tally and timing state without
   * emitting a new event — for a source whose final identity is only known
   * after a fetch (ADR 0007 §11.1, e.g. an HTTP redirect).
   *
   * `old`'s phase history stops counting under `old` — the next `record`
   * for either name only ever tallies once, as one reference. Returns
   * `true` when the rename applied normally (including the no-op cases:
   * `old === new`, or `old` was never discovered under this run).
   *
   * Returns `false` — a POST-fetch collision — when `new` is already a
   * DIFFERENT source this run has its own, independent history for (two
   * distinct inputs redirecting to the same final URL, e.g.). Silently
   * overwriting `new`'s prior state in that case is exactly `duplicate`'s
   * own problem one layer later: if the first input already reached
   * `imported`, blindly moving the second input's still-`discovered` state
   * onto `new` would reset `new`'s last phase back to `discovered`, then
   * the caller's own subsequent `record` calls for the second input —
   * which necessarily continue under `new` too, since that is now the
   * fetched document's identity — would drive it through `parsed`/
   * `unchanged` again, silently erasing the real `imported` outcome from
   * the tally. Instead: `old` is recorded as a duplicate of `new`
   * (`duplicate`, so the tally is untouched), and the caller MUST treat
   * `false` as "this reference is already fully handled" — skip any
   * further recording, checkpoint write, or import attempt for it.
   */
  retarget(oldSource: string, newSource: string): boolean {
    if (oldSource === newSource) {
      return true;
    }
    if (this.sources.has(newSource)) {
      this.sources.delete(oldSource);
      this.lastPhase.delete(oldSource);
      this.duplicate(oldSource, { of: newSource });
      return false;
    }
    const state = this.sources.get(oldSource);
    if (state === undefined) {
      return true;
    }
    this.sources.delete(oldSource);
    this.sources.set(newSource, state);
    const phase = this.lastPhase.get(oldSource);
    this.lastPhase.delete(oldSource);
    if (phase !== undefined) {
      this.lastPhase.set(newSource, phase);
    }
    return true;
  }

  /**
   * Records `reference` as a rejected duplicate of the already-claimed
   * source id `of` (ADR 0007 §11.1, the `duplicate_source` diagnostic) —
   * visible in the JSONL trail as its own `skipped` event, but deliberately
   * keyed by `reference` itself and NEVER folded into `record`'s
   * last-phase tally for `of`.
   *
   * This is not an oversight: `of` already has its own, possibly still
   * in-progress, phase history from the reference that claimed it first —
   * a second reference resolving to the same id is not a distinct source
   * in this run, so it must never be able to overwrite `of`'s real outcome
   * (e.g. turning an already-recorded `imported` back into `skipped` just
   * because a later, redundant input happened to name the same thing).
   * Calling `record` instead of this method for a known duplicate is the
   * bug this method exists to make impossible.
   *
   * A consumer that needs to separate a duplicate event from `of`'s own
   * events must check `diagnostic.code === "duplicate_source"` rather than
   * assuming `source` alone identifies one JSONL sub-sequence — `reference`
   * and `of` can be the same string (e.g. a duplicate listing key from one
   * `store.list()` pass), so grouping strictly by `source` could misread
   * that source's history as an interrupted-then-recovered run rather than
   * "one duplicate listing entry, one real import." The tally itself stays
   * correct either way; this method never touches it.
   */
  duplicate(reference: string, options: { of: string }): void {
    const event = new SourceEvent({
      source: reference,
      phase: "skipped",
      elapsedMs: 0.0,
      diagnostic: new Diagnostic({
        code: "duplicate_source",
        message: `already claimed as ${JSON.stringify(options.of)} earlier in this run`,
        source: options.of,
      }),
    });
    if (this.keepEvents) {
      this.events.push(event);
    }
    this.sink?.write(event);
    // Deliberately no `lastPhase`/`sources` write for either `reference`
    // or `of` — see the doc comment above.
  }

  /**
   * Accumulates ADR 0007 §5's `locatorsDropped`/`sectionsDropped`
   * (surfaced via `IngestOutcome`, which a driver otherwise has no reason
   * to inspect) and §9's `tagsDropped` into this run's totals.
   */
  addDropped(options?: { locators?: number; sections?: number; tags?: number }): void {
    this.locatorsDropped += options?.locators ?? 0;
    this.sectionsDropped += options?.sections ?? 0;
    this.tagsDropped += options?.tags ?? 0;
  }

  /** Accumulates ADR 0007 §9's deletion-policy counters. */
  addDeletions(options: { detected: number; retracted: number }): void {
    this.deletedDetected += options.detected;
    this.retracted += options.retracted;
  }

  /**
   * Translates one `IngestEvent` into this run's `extracted` phase.
   *
   * `ImportStarted` is the only variant this cares about: it fires exactly
   * once per document, exactly after every chunk has been extracted and
   * the batch rendered, immediately before the network `importBatches`
   * call (ingest.ts's `ingestText`) — precisely the honest moment
   * `extracted` needs, and the only one available without synthesizing a
   * timestamp (ADR 0007 §11.1 rejects that as "the quietly degraded output
   * #217 forbids"). Every other event variant is ignored here — a driver
   * that wants finer-grained reporting installs its own `on_event` and
   * calls this one from it, or uses `attach`/`attached` to chain both.
   */
  onIngestEvent(event: IngestEvent): void {
    if (event.kind === "import_started") {
      this.record(event.source, "extracted");
    }
  }

  /**
   * Chain-attaches this recorder's `onIngestEvent` onto `ingester`'s
   * `on_event` callback (chaining onto whatever callback was already
   * installed, never replacing it) and returns a `detach` function that
   * restores the original callback — the disposable primitive `attached`
   * is built on. `TaguruIngester`'s own event dispatch already wraps every
   * callback in try/catch + `console.warn`, so a fault in this recorder
   * can never break the ingest it is observing.
   *
   * `TaguruIngester.on_event` is a private field in TypeScript (unlike the
   * Python twin's public, freely reassignable attribute), so this reaches
   * it through a runtime cast rather than a typed public setter —
   * TypeScript's `private`/`readonly` are compile-time-only, and the
   * underlying field is an ordinary JS property, so this is a deliberate,
   * narrowly-scoped exception to the rule that this module never edits
   * ingest.ts. Not re-entrant and not thread-safe: at most one driver
   * attaches to a given ingester at a time.
   */
  attach(ingester: TaguruIngester): () => void {
    const target = ingester as unknown as { on_event: IngestEventCallback | undefined };
    const previous = target.on_event;
    const combined: IngestEventCallback = (event) => {
      this.onIngestEvent(event);
      if (previous !== undefined) {
        previous(event);
      }
    };
    target.on_event = combined;
    return () => {
      target.on_event = previous;
    };
  }

  /**
   * Runs `fn` with this recorder's `onIngestEvent` attached to `ingester`'s
   * `on_event` for the duration of the call, restoring the original
   * callback in `finally` even if `fn` throws or its returned promise
   * rejects — the TS mirror of the Python port's `with
   * recorder.attached(ingester): ...` context manager.
   */
  async attached<T>(ingester: TaguruIngester, fn: () => T | Promise<T>): Promise<T> {
    const detach = this.attach(ingester);
    try {
      return await fn();
    } finally {
      detach();
    }
  }

  /**
   * Derives this run's `RunReport` — counts are a tally of each source's
   * LAST phase event only (ADR 0007 §11), never a cumulative count across
   * every phase it passed through.
   */
  finish(options?: { interrupted?: boolean }): RunReport {
    const tallies: Record<Phase, number> = {
      discovered: 0,
      unchanged: 0,
      parsed: 0,
      extracted: 0,
      imported: 0,
      skipped: 0,
      failed: 0,
    };
    for (const phase of this.lastPhase.values()) {
      tallies[phase] += 1;
    }
    return new RunReport({
      connector: this.connector,
      durationMs: performance.now() - this.runStarted,
      interrupted: options?.interrupted ?? false,
      discovered: tallies.discovered,
      unchanged: tallies.unchanged,
      parsed: tallies.parsed,
      extracted: tallies.extracted,
      imported: tallies.imported,
      skipped: tallies.skipped,
      failed: tallies.failed,
      locatorsDropped: this.locatorsDropped,
      sectionsDropped: this.sectionsDropped,
      tagsDropped: this.tagsDropped,
      deletedDetected: this.deletedDetected,
      retracted: this.retracted,
      events: [...this.events],
      eventsPath: this.sink === null ? null : this.sink.path,
    });
  }

  /** Closes the events sink, if this recorder owns one. Safe to call more than once. */
  close(): void {
    this.sink?.close();
  }
}
