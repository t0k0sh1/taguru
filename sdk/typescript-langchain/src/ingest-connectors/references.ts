/**
 * The cross-connector local-file/URL driver (ADR 0007 §11, issue #353): the
 * one entry point every non-object-storage connector (`.md`/`.txt`, PDF,
 * HTML, DOCX, PPTX) syncs through, instead of a caller hand-rolling its own
 * loop over `Connector.read` (protocol.ts) calls with no shared event log,
 * counts, or `dryRun` posture. `syncObjectStorage` (not yet ported to this
 * SDK) is this module's object-storage twin — the two share
 * `observability.ts` but not this module, since object-storage listing
 * (bucket/prefix enumeration, its own fingerprint tiers) has no local-file/
 * URL analogue. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.references` module (issue #415).
 *
 * Two ideas this module exists to get right, both real bugs found while
 * writing the Python original rather than hypotheticals:
 *
 * - **Dispatch must classify a reference's KIND before asking any connector
 *   whether it "supports" the reference.** Every non-HTML connector's own
 *   `supports()` looks at the extension alone (`TextFileConnector.supports`,
 *   text.ts, is exactly `SUPPORTED_SUFFIXES.has(pathSuffix(reference))`) —
 *   so `supports("https://example.com/a.md")` is `true`, and would hand a
 *   URL to a filesystem read as though it were a local path.
 *   `planReferences` classifies `path` vs. `url` first (mirroring exactly
 *   `HtmlConnector.read`'s own Windows-drive-path-then-scheme check,
 *   html.ts) and only probes a URL-kind reference against connectors that
 *   are scheme-aware in the first place (see `connectorForUrl`).
 * - **A duplicate reference must never be tallied under the source id it
 *   collides with.** The obvious approach — record the duplicate's
 *   `skipped`/`duplicate_source` event under the SAME source id the earlier
 *   reference already claimed — corrupts `RunReport`'s (observability.ts)
 *   last-phase-only tally: because `planReferences` always resolves
 *   duplicates in input order, the earlier (claiming) reference is always
 *   processed first, so by the time the duplicate is reached the claimed
 *   source may already have reached `imported` — recording `skipped` under
 *   that same key would silently turn a successful import back into a
 *   failure in the summary. `RunRecorder.duplicate` (observability.ts)
 *   exists specifically to make this impossible.
 *
 * Deliberate deviations from the Python original, all forced by real
 * differences this SDK's already-ported sibling modules already establish:
 *
 * - **`defaultConnectors()` never needs to catch an "optional dependency
 *   missing" error.** Python's `PdfConnector`/`DocxConnector`/
 *   `PptxConnector` raise `ImportError` synchronously at CONSTRUCTION,
 *   since `import pypdf`/etc. happens at Python module load — so
 *   `default_connectors()` wraps each optional connector's construction in
 *   `try/except ImportError: pass`. This SDK's pdf.ts/docx.ts/pptx.ts
 *   deliberately load their optional npm packages via a dynamic `import()`
 *   INSIDE `read()`, never at construction (each module's own doc comment
 *   states this) — so `new PdfConnector()` (etc.) never throws for a
 *   missing package; only calling `read()` without the package installed
 *   would. `defaultConnectors()` here therefore constructs all five
 *   connectors unconditionally, with no `try`/`catch` at all.
 * - **`isWindowsDrivePath`/`stripFragment` are duplicated from html.ts, not
 *   imported.** Python's `references.py` imports `_is_windows_drive_path`/
 *   `_strip_fragment` directly from `taguru_langchain.ingest_connectors.
 *   html` (both are module-private-by-convention, i.e. leading-underscore,
 *   but freely importable within the package). `html.ts` keeps its
 *   equivalents as genuinely non-exported (unprefixed) module-local
 *   functions — TypeScript's export list is the real privacy boundary, and
 *   this port must not edit html.ts to widen it. This module therefore
 *   carries its own small, byte-identical-logic copies (below), documented
 *   at their definitions.
 * - **A single `string` reference type, not `str | Path`.** This SDK has no
 *   `pathlib.Path` analogue in general use (every sibling connector's
 *   `read(reference: string)` already takes a plain string); `references`
 *   here is `Iterable<string>`, matching every connector `read()`'s own
 *   parameter type instead of introducing one.
 */

import type { CheckpointStore } from "../checkpoints.js";
import type { ShouldStop, TaguruIngester } from "../ingest.js";
import { ingestConnectorDocument, normalizeStop } from "./bridge.js";
import { ConnectorCheckpoint, FileProbe, FileProbeCheckpoint } from "./checkpoint.js";
import { type ConnectorDocument, Diagnostic, type DiagnosticCode } from "./document.js";
import { DocxConnector } from "./docx.js";
import { HtmlConnector } from "./html.js";
import { RunRecorder, type RunReport, type SourceEventSinkTarget } from "./observability.js";
import { PdfConnector } from "./pdf.js";
import { PptxConnector } from "./pptx.js";
import type { Connector } from "./protocol.js";
import { SourceIdRegistry, canonicalizeUrl, checkSourceId, fileSourceId } from "./sources.js";
import { TextFileConnector } from "./text.js";

export const PARSER_NAME = "taguru-reference-sync";
/** This driver's own identity in `RunReport.connector` (observability.ts) —
 * it is not itself a format connector (a delegate's own
 * `fingerprintInputs.parser` is left untouched, exactly as a future
 * `syncObjectStorage` port leaves a delegate's fingerprint untouched under
 * its own S3 identity re-stamping). */

/** What kind of identity a reference resolves to (ADR 0007 §11's per-kind
 * `dryRun` table is keyed on exactly this distinction) — a reference whose
 * scheme names an object-storage backend (`s3://`, `gs://`, `az://`, ...)
 * is classified `"url"` for bookkeeping purposes but always carries an
 * `unsupported_format` diagnostic pointing at a future `syncObjectStorage`
 * instead: this driver has no listing/fingerprint-tier story of its own for
 * those schemes. */
export type ReferenceKind = "path" | "url";

// -- local duplicates of html.ts's own (non-exported) reference-kind helpers --
// -- (see this module's own doc comment for why these are copies, not ----------
// -- imports) --------------------------------------------------------------

// A Windows drive-letter path (`C:\docs\a.html`, `C:/docs/a.html`) parses
// under a bare scheme-prefix regex as scheme `"c"` — indistinguishable, by
// scheme alone, from a genuine (if unusual) single-letter URL scheme.
// Checked before any scheme dispatch so a Windows absolute path is never
// misrouted into the URL-fetch path or refused as `unsupported_format`.
const WINDOWS_DRIVE_RE = /^[A-Za-z]:[\\/]/;

function isWindowsDrivePath(reference: string): boolean {
  return WINDOWS_DRIVE_RE.test(reference);
}

function stripFragment(url: string): string {
  const hashIndex = url.indexOf("#");
  return hashIndex === -1 ? url : url.slice(0, hashIndex);
}

const SCHEME_RE = /^([A-Za-z][A-Za-z0-9+.-]*):/;

function urlScheme(reference: string): string {
  const match = SCHEME_RE.exec(reference);
  return match === null ? "" : match[1]!.toLowerCase();
}

function pathBasename(reference: string): string {
  const trimmed = reference.replace(/[/\\]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return index === -1 ? trimmed : trimmed.slice(index + 1);
}

function pathSuffix(reference: string): string {
  const name = pathBasename(reference);
  const dotIndex = name.lastIndexOf(".");
  return dotIndex <= 0 ? "" : name.slice(dotIndex);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** `fs.stat(reference, { bigint: true })`, imported dynamically inside the
 * function (never at module scope — src/ modules stay import-free of
 * `node:*` at the top level so this package can still be loaded somewhere
 * that never calls a file-touching function). `bigint: true` mirrors
 * `os.stat().st_mtime_ns` at full nanosecond precision, needed for
 * `FileProbe.mtimeNs`. */
async function statFile(reference: string) {
  const { stat } = await import("node:fs/promises");
  return stat(reference, { bigint: true });
}

/**
 * The installed reference connectors, in ADR 0007's own PDF/HTML/DOCX/PPTX
 * order plus the original `.md`/`.txt` reference. See this module's own doc
 * comment for why this never needs to catch a missing-optional-dependency
 * error the way the Python original does.
 */
export function defaultConnectors(): readonly Connector[] {
  return Object.freeze([
    new TextFileConnector(),
    new HtmlConnector(),
    new PdfConnector(),
    new DocxConnector(),
    new PptxConnector(),
  ]);
}

function probeReference(scheme: string): string {
  // No installed connector's `supports()` inspects anything about a
  // reference beyond its scheme (for a URL) or extension (for a path) — a
  // fabricated, suffix-less URL is therefore enough to ask "are you
  // scheme-aware at all," the same trick a future `syncObjectStorage` port
  // plays in the other direction with a fabricated `"object<suffix>"` name.
  return `${scheme}://taguru.invalid/probe`;
}

function connectorForPath(connectors: readonly Connector[], reference: string): Connector | null {
  for (const connector of connectors) {
    if (connector.supports(reference)) {
      return connector;
    }
  }
  return null;
}

function connectorForUrl(
  connectors: readonly Connector[],
  reference: string,
  scheme: string,
): Connector | null {
  const probe = probeReference(scheme);
  for (const connector of connectors) {
    // Both checks matter: `supports(probe)` rejects every connector that
    // (like every non-HTML connector today) only ever looks at a
    // reference's extension, since the probe carries none; `supports
    // (reference)` then leaves room for a future scheme-aware connector
    // that only claims a URL scheme for certain paths.
    if (connector.supports(probe) && connector.supports(reference)) {
      return connector;
    }
  }
  return null;
}

/**
 * One reference's dispatch decision, made before any fetch — the "enumerate
 * all discovered work before the first fetch" half of ADR 0007 §11,
 * materialized as data so `syncReferences` can report every `discovered`
 * event up front regardless of how far the run gets.
 *
 * `diagnostic` set means this reference will never reach `read()` — an
 * unreadable/oversized/duplicate/unrecognized reference, reported `skipped`
 * (or, for a duplicate, not tallied at all — see `RunRecorder.duplicate`,
 * observability.ts) without ever calling a connector. `connector` is `null`
 * exactly when `diagnostic` is set.
 */
export class ReferencePlan {
  readonly reference: string;
  readonly source: string;
  readonly kind: ReferenceKind;
  readonly sourceBytes: number;
  readonly connector: Connector | null;
  readonly diagnostic: Diagnostic | null;

  constructor(fields: {
    reference: string;
    source: string;
    kind: ReferenceKind;
    sourceBytes: number;
    connector: Connector | null;
    diagnostic: Diagnostic | null;
  }) {
    this.reference = fields.reference;
    this.source = fields.source;
    this.kind = fields.kind;
    this.sourceBytes = fields.sourceBytes;
    this.connector = fields.connector;
    this.diagnostic = fields.diagnostic;
  }
}

function duplicateDiagnostic(source: string): Diagnostic {
  return new Diagnostic({
    code: "duplicate_source",
    message: "source id already claimed earlier in this run",
    source,
  });
}

function unsupported(source: string, message: string): Diagnostic {
  const code: DiagnosticCode = "unsupported_format";
  return new Diagnostic({ code, message, source });
}

async function planPath(
  reference: string,
  source: string,
  connectors: readonly Connector[],
): Promise<ReferencePlan> {
  const diagnostic = checkSourceId(source);
  if (diagnostic !== null) {
    return new ReferencePlan({
      reference,
      source,
      kind: "path",
      sourceBytes: 0,
      connector: null,
      diagnostic,
    });
  }
  let statResult: Awaited<ReturnType<typeof statFile>>;
  try {
    statResult = await statFile(reference);
  } catch (error) {
    return new ReferencePlan({
      reference,
      source,
      kind: "path",
      sourceBytes: 0,
      connector: null,
      diagnostic: new Diagnostic({ code: "unreadable", message: errorMessage(error), source }),
    });
  }
  if (!statResult.isFile()) {
    return new ReferencePlan({
      reference,
      source,
      kind: "path",
      sourceBytes: 0,
      connector: null,
      diagnostic: new Diagnostic({ code: "unreadable", message: "not a regular file", source }),
    });
  }
  const connector = connectorForPath(connectors, reference);
  if (connector === null) {
    const suffix = pathSuffix(reference) || "(none)";
    return new ReferencePlan({
      reference,
      source,
      kind: "path",
      sourceBytes: Number(statResult.size),
      connector: null,
      diagnostic: unsupported(
        source,
        `no installed connector supports extension ${JSON.stringify(suffix)}`,
      ),
    });
  }
  return new ReferencePlan({
    reference,
    source,
    kind: "path",
    sourceBytes: Number(statResult.size),
    connector,
    diagnostic: null,
  });
}

function planUrl(
  reference: string,
  source: string,
  scheme: string,
  connectors: readonly Connector[],
): ReferencePlan {
  const diagnostic = checkSourceId(source);
  if (diagnostic !== null) {
    return new ReferencePlan({
      reference,
      source,
      kind: "url",
      sourceBytes: 0,
      connector: null,
      diagnostic,
    });
  }
  const connector = connectorForUrl(connectors, reference, scheme);
  if (connector === null) {
    return new ReferencePlan({
      reference,
      source,
      kind: "url",
      sourceBytes: 0,
      connector: null,
      diagnostic: unsupported(
        source,
        `no installed connector supports scheme ${JSON.stringify(scheme)}`,
      ),
    });
  }
  return new ReferencePlan({
    reference,
    source,
    kind: "url",
    sourceBytes: 0,
    connector,
    diagnostic: null,
  });
}

/**
 * Classifies and dispatches every reference in `references`, in order,
 * before any of them is fetched.
 *
 * Classification mirrors `HtmlConnector.read`'s (html.ts) own local-path-
 * vs-URL check exactly (Windows drive path, then `http`/`https` scheme,
 * then bare path) — an `http`/`https` reference's planned `source` is
 * therefore the same value `HtmlConnector`'s own URL read computes as its
 * pre-fetch source, so a fetch that hits no redirect needs no
 * `RunRecorder.retarget` (observability.ts) at all. A reference naming any
 * OTHER scheme (`s3://`, `gs://`, `az://`, ...) is planned with an
 * `unsupported_format` diagnostic pointing at a future `syncObjectStorage`.
 *
 * A reference whose planned `source` was already claimed by an earlier
 * reference in this same call is planned with a `duplicate_source`
 * diagnostic and no connector — `syncReferences` never calls
 * `RunRecorder.discovered` for it, only `RunRecorder.duplicate`.
 */
export async function planReferences(
  references: Iterable<string>,
  options: { connectors: readonly Connector[] },
): Promise<ReferencePlan[]> {
  const registry = new SourceIdRegistry();
  const plans: ReferencePlan[] = [];
  for (const raw of references) {
    const reference = String(raw);
    let source: string;
    let kind: ReferenceKind;
    let scheme = "";
    if (isWindowsDrivePath(reference)) {
      source = fileSourceId(reference);
      kind = "path";
    } else {
      scheme = urlScheme(reference);
      if (scheme === "http" || scheme === "https") {
        source = canonicalizeUrl(stripFragment(reference));
        kind = "url";
      } else if (scheme) {
        plans.push(
          new ReferencePlan({
            reference,
            source: reference,
            kind: "url",
            sourceBytes: 0,
            connector: null,
            diagnostic: unsupported(
              reference,
              `scheme ${JSON.stringify(scheme)} is not synced by syncReferences ` +
                "(use syncObjectStorage for s3://, gs://, az://, ...)",
            ),
          }),
        );
        continue;
      } else {
        source = fileSourceId(reference);
        kind = "path";
      }
    }

    if (!registry.claim(source)) {
      plans.push(
        new ReferencePlan({
          reference,
          source,
          kind,
          sourceBytes: 0,
          connector: null,
          diagnostic: duplicateDiagnostic(source),
        }),
      );
      continue;
    }

    if (kind === "path") {
      plans.push(await planPath(reference, source, options.connectors));
    } else {
      plans.push(planUrl(reference, source, scheme, options.connectors));
    }
  }
  return plans;
}

function isDuplicatePlan(plan: ReferencePlan): boolean {
  return plan.diagnostic !== null && plan.diagnostic.code === "duplicate_source";
}

async function saveFileProbe(
  fileProbeCheckpoint: FileProbeCheckpoint,
  plan: ReferencePlan,
  document: ConnectorDocument,
): Promise<void> {
  if (plan.kind !== "path") {
    return;
  }
  let statResult: Awaited<ReturnType<typeof statFile>>;
  try {
    statResult = await statFile(plan.reference);
  } catch {
    // Best-effort: a probe write failure only costs the NEXT dry-run an
    // honest "parsed" instead of "unchanged" — never a false hit.
    return;
  }
  await fileProbeCheckpoint.save(
    document.source,
    new FileProbe({
      size: Number(statResult.size),
      mtimeNs: statResult.mtimeNs,
      parser: document.fingerprintInputs.parser,
      parserVersion: document.fingerprintInputs.parserVersion,
      parseOptionsDigest: document.fingerprintInputs.parseOptionsDigest,
    }),
  );
}

async function dryRunPlan(
  plan: ReferencePlan,
  options: { recorder: RunRecorder; fileProbeCheckpoint: FileProbeCheckpoint },
): Promise<void> {
  if (plan.kind === "url") {
    // ADR 0007 §11: a URL performs NO network access under --dry-run, not
    // even a HEAD — "would attempt to fetch" is already the correct,
    // honest answer, and the only one available without one.
    options.recorder.record(plan.source, "parsed");
    return;
  }

  const connector = plan.connector;
  if (connector === null) {
    // Unreachable by construction (plan.diagnostic is null here, and
    // planReferences only ever omits a diagnostic when it also resolved a
    // connector) — mirrors the Python original's own `assert`.
    throw new Error("dryRunPlan: a diagnostic-free path plan carried no connector");
  }
  let statResult: Awaited<ReturnType<typeof statFile>>;
  try {
    statResult = await statFile(plan.reference);
  } catch {
    // The file vanished between planning and this dry-run pass — still
    // honestly "would attempt to fetch" (and would then fail), never a
    // manufactured `unchanged`.
    options.recorder.record(plan.source, "parsed");
    return;
  }
  const probe = await options.fileProbeCheckpoint.load(plan.source);
  const candidate = new FileProbe({
    size: Number(statResult.size),
    mtimeNs: statResult.mtimeNs,
    parser: connector.parser,
    parserVersion: connector.parserVersion,
    parseOptionsDigest: await connector.parseOptionsDigest(),
  });
  if (probe !== null && probe.equals(candidate)) {
    options.recorder.record(plan.source, "unchanged");
  } else {
    // No probe, or ANY mismatch (including a metadata match but a changed
    // parser option) — never a false `unchanged`, the one direction ADR
    // 0007 §11 allows dry-run's local-file check to err.
    options.recorder.record(plan.source, "parsed");
  }
}

/**
 * Reads every local-file or `http(s)://` reference in `references` and
 * syncs it into `ingester`, applying the connector-native checkpoint (skip
 * re-fetch/re-parse of byte-identical content) and the local-file dry-run
 * probe (ADR 0007 §11), and reporting one `RunReport` (observability.ts).
 *
 * `dryRun` here means exactly ADR 0007 §11's driver-level contract: no
 * network fetch, no local file read beyond a cheap stat, and no write to
 * the corpus OR to either checkpoint store. This is a DIFFERENT, stricter
 * meaning than `TaguruIngester.ingestText`'s own `dry_run` (which still
 * calls the model and only skips the network import call) — this driver's
 * `dryRun` is therefore never forwarded into `ingestConnectorDocument`
 * (bridge.ts); a real (non-dry) call to that bridge always runs with its
 * own default `dryRun: false`.
 *
 * `eventsOut` is honored even under `dryRun: true`: the sidecar it names is
 * the dry run's entire product, and the caller named the path explicitly,
 * so writing to it does not violate dry-run's "touch nothing else"
 * contract.
 *
 * Every reference is classified and dispatched (`planReferences`) — and
 * every non-duplicate reference reported `discovered` — before any of them
 * is read, so an interruption (`shouldStop`) after the Nth reference still
 * leaves every later reference visible as `discovered` in the report rather
 * than absent.
 */
export async function syncReferences(
  references: Iterable<string>,
  options: {
    ingester: TaguruIngester;
    checkpoints: CheckpointStore;
    connectors?: readonly Connector[];
    dryRun?: boolean;
    shouldStop?: ShouldStop;
    eventsOut?: SourceEventSinkTarget;
  },
): Promise<RunReport> {
  const stop = normalizeStop(options.shouldStop);
  const activeConnectors = options.connectors ?? defaultConnectors();
  const connectorCheckpoint = new ConnectorCheckpoint(options.checkpoints);
  const fileProbeCheckpoint = new FileProbeCheckpoint(options.checkpoints);

  // try/finally, not a bare construct-then-close: `eventsOut`'s file handle
  // must close even when something below throws (a connector's `read()`
  // catches its own exceptions, but `ingestConnectorDocument`,
  // `connectorCheckpoint.load`/`save`, or `recorder.record` itself do not)
  // — `recorder.close()` at the end of a straight-line function body never
  // runs in that case. Mirrors the Python original's `with RunRecorder(...)
  // as recorder:`.
  const recorder = new RunRecorder({ connector: PARSER_NAME, eventsOut: options.eventsOut });
  try {
    const plans = await planReferences(references, { connectors: activeConnectors });

    const work: ReferencePlan[] = [];
    for (const plan of plans) {
      if (isDuplicatePlan(plan)) {
        recorder.duplicate(plan.reference, { of: plan.source });
        continue;
      }
      recorder.discovered(plan.source, { sourceBytes: plan.sourceBytes });
      work.push(plan);
    }

    let interrupted = false;
    for (const plan of work) {
      if (stop()) {
        interrupted = true;
        break;
      }

      if (plan.diagnostic !== null) {
        recorder.record(plan.source, "skipped", { diagnostic: plan.diagnostic });
        continue;
      }

      const connector = plan.connector;
      if (connector === null) {
        // Unreachable by construction (planReferences only omits a
        // diagnostic when it also resolved a connector) — handled
        // defensively rather than throwing mid-run.
        recorder.record(plan.source, "failed", {
          diagnostic: new Diagnostic({
            code: "unreadable",
            message: "no connector resolved",
            source: plan.source,
          }),
        });
        continue;
      }

      if (options.dryRun) {
        await dryRunPlan(plan, { recorder, fileProbeCheckpoint });
        continue;
      }

      let document: ConnectorDocument;
      try {
        document = await connector.read(plan.reference);
      } catch (error) {
        // A connector bug must not abort the run.
        recorder.record(plan.source, "failed", {
          diagnostic: new Diagnostic({
            code: "unreadable",
            message: errorMessage(error),
            source: plan.source,
          }),
        });
        continue;
      }

      if (document.source !== plan.source) {
        if (!recorder.retarget(plan.source, document.source)) {
          // A post-fetch collision (e.g. two different URLs redirecting to
          // the same final URL) — already recorded as a duplicate of the
          // winning source with no effect on its tally; this reference is
          // fully handled, never imported under a competing identity.
          continue;
        }
      }
      const source = document.source;

      if (!document.text) {
        const diagnostic = document.diagnostics.length > 0 ? document.diagnostics[0]! : null;
        recorder.record(source, "failed", { diagnostic });
        continue;
      }

      recorder.record(source, "parsed", { parser: document.fingerprintInputs.parser });

      if ((await connectorCheckpoint.load(source, document.fingerprintInputs)) !== null) {
        await saveFileProbe(fileProbeCheckpoint, plan, document);
        recorder.record(source, "unchanged");
        continue;
      }

      const outcome = await recorder.attached(options.ingester, () =>
        ingestConnectorDocument(options.ingester, document, { shouldStop: options.shouldStop }),
      );

      recorder.addDropped({
        locators: outcome.locators_dropped,
        sections: outcome.sections_dropped,
      });

      if (outcome.interrupted) {
        recorder.record(source, "skipped");
        interrupted = true;
        break;
      }
      if (!outcome.ok) {
        recorder.record(source, "failed", {
          diagnostic: new Diagnostic({
            code: "unreadable",
            message: outcome.error ?? "ingest failed",
            source,
          }),
        });
        continue;
      }

      await connectorCheckpoint.save(document);
      await saveFileProbe(fileProbeCheckpoint, plan, document);
      recorder.record(source, "imported", { parser: document.fingerprintInputs.parser });
    }

    return recorder.finish({ interrupted });
  } finally {
    recorder.close();
  }
}
