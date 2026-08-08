/**
 * The bridge from a connector's normalized document to `TaguruIngester`
 * (ingest.ts) (ADR 0007, issue #347) — the "end to end" wiring the issue
 * asks for: a `ConnectorDocument` goes in, an `IngestOutcome` comes out,
 * through the exact same retract-then-apply batch/import path any other
 * source uses. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.bridge` module (issue #415).
 *
 * Depends on `../ingest.js` in one direction only — `TaguruIngester` itself
 * knows nothing about connectors; a connector never grows special-cased
 * methods on the ingester it feeds.
 *
 * One deliberate deviation from the Python original, forced by this SDK's
 * own, already-established shape rather than chosen here:
 *
 * - **One async function per Python sync/async pair, not two.** Python
 *   exposes `ingest_connector_document`/`aingest_connector_document` and
 *   `ingest_connector_documents`/`aingest_connector_documents` because
 *   `TaguruIngester.ingest_text`/`ingest_documents` themselves come in sync
 *   and async flavors. `ingest.ts`'s own `TaguruIngester` is async-only (its
 *   class doc: "identical here, modulo TS being async-only — no separate
 *   `a`-prefixed variants"), so this module follows the same convention:
 *   `ingestConnectorDocument`/`ingestConnectorDocuments` are the only two
 *   functions, both `async`, each covering both of its Python twin's roles.
 */

import type { ConnectorDocument } from "./document.js";
import type { IngestOutcome, ShouldStop, TaguruIngester } from "../ingest.js";

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * `undefined` (no `should_stop` given) normalizes to "never stop" — mirrors
 * `ingest.ts`'s own (module-private) `normalizeStop`, duplicated here
 * (rather than imported — `ingest.ts` does not export it) because
 * `ingestConnectorDocuments` below needs the exact same
 * predicate-or-`AbortSignal` normalization `TaguruIngester` itself applies
 * to `should_stop`/`ShouldStop`. Exported so `references.ts` (issue #415's
 * other module in this pair) can share this one implementation rather than
 * a third copy — mirroring how the Python twin's `references.py` imports
 * `taguru_langchain.ingest`'s own (module-private-by-convention)
 * `_normalize_stop` directly.
 */
export function normalizeStop(shouldStop: ShouldStop | undefined): () => boolean {
  if (shouldStop === undefined) {
    return () => false;
  }
  if (typeof shouldStop === "function") {
    return shouldStop;
  }
  return () => shouldStop.aborted;
}

function baseOutcome(source: string): IngestOutcome {
  return {
    source,
    ok: false,
    ndjson: null,
    created: false,
    retracted: 0,
    associations: 0,
    aliases: 0,
    passage_stored: false,
    questions_stored: 0,
    sections_stored: 0,
    sections_dropped: 0,
    locators_stored: 0,
    locators_dropped: 0,
    duplicates_dropped: 0,
    invalid_dropped: 0,
    llm_calls: 0,
    chunks: 0,
    correction_attempts: 0,
    lossless_repairs: [],
    error: null,
    embeddings_refresh_warning: null,
    interrupted: false,
    chunks_reused: 0,
  };
}

function diagnosticsOnlyOutcome(document: ConnectorDocument): IngestOutcome {
  const summary = document.diagnostics.map((d) => `${d.code}: ${d.message}`).join("; ");
  return {
    ...baseOutcome(document.source),
    error: summary || "connector document has empty text and no diagnostic",
  };
}

/**
 * Ingests one `ConnectorDocument` through `ingester`.
 *
 * A document whose `text` is empty (diagnostics-only — nothing usable was
 * extracted, ADR 0007 §5/§10) never reaches the network: it is reported as
 * a failed `IngestOutcome` carrying every diagnostic's `code`/`message` in
 * `error` — the same "never silently empty" posture the normalized document
 * contract itself enforces at construction time.
 */
export async function ingestConnectorDocument(
  ingester: TaguruIngester,
  document: ConnectorDocument,
  options?: { dryRun?: boolean; shouldStop?: ShouldStop },
): Promise<IngestOutcome> {
  if (!document.text) {
    return diagnosticsOnlyOutcome(document);
  }
  return ingester.ingestText(document.text, {
    source: document.source,
    sections: document.sections.map((entry) => ({
      paragraph: entry.paragraph,
      section: entry.section,
    })),
    locators: document.locators.map((entry) => ({
      paragraph: entry.paragraph,
      locator: entry.locator,
    })),
    dry_run: options?.dryRun,
    should_stop: options?.shouldStop,
  });
}

/**
 * Ingests each document independently — one failure never stops the rest
 * (set `ingester.raise_on_error: true` to fail fast), the same posture
 * `TaguruIngester.ingestDocuments` already takes for LangChain `Document`s.
 * `shouldStop` (issue #211/#212) is checked between documents too, and once
 * a document's own `ingestConnectorDocument` reports `interrupted` the run
 * stops there.
 *
 * A diagnostics-only document (empty `text`) is never a `raise_on_error`
 * trigger — it is expected connector output, not a programming error, so
 * the run always continues past it to the next document regardless of
 * `raise_on_error`.
 */
export async function ingestConnectorDocuments(
  ingester: TaguruIngester,
  documents: readonly ConnectorDocument[],
  options?: { dryRun?: boolean; shouldStop?: ShouldStop },
): Promise<IngestOutcome[]> {
  const stop = normalizeStop(options?.shouldStop);
  const outcomes: IngestOutcome[] = [];
  for (const document of documents) {
    if (stop()) {
      break;
    }
    let outcome: IngestOutcome;
    try {
      outcome = await ingestConnectorDocument(ingester, document, options);
    } catch (error) {
      if (ingester.raise_on_error) {
        throw error;
      }
      outcomes.push({ ...baseOutcome(document.source), error: errorMessage(error) });
      continue;
    }
    outcomes.push(outcome);
    if (outcome.interrupted) {
      break;
    }
  }
  return outcomes;
}
