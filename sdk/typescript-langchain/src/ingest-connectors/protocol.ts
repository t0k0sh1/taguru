/**
 * The connector extension point (ADR 0007 §5, issue #347): the interface
 * any format connector — present or future — implements to produce a
 * `ConnectorDocument`. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.protocol` module (issue #415).
 *
 * A structural interface, not a base class — the same extension-point
 * convention `CheckpointStore` (checkpoints.ts) already uses: structural
 * typing, no forced inheritance, no runtime `instanceof` check.
 */

import type { ConnectorDocument } from "./document.js";

/** Reads one reference (a path, URL, or object key) into a `ConnectorDocument`. */
export interface Connector {
  /**
   * The connector's own identity, stamped into every document's
   * `fingerprint_inputs.parser` (ADR 0007 §5).
   */
  readonly parser: string;

  /**
   * The connector's own version, stamped into
   * `fingerprint_inputs.parser_version`.
   */
  readonly parserVersion: string;

  /**
   * The digest `read` would stamp into
   * `fingerprint_inputs.parse_options_digest` for this instance's current
   * effective config — computable without reading anything. Added for
   * issue #351's S3 connector: its own connector-level checkpoint (ADR
   * 0007 §6.3) must decide whether re-parsing a just-fetched object can
   * be skipped BEFORE calling `read`, which needs this value as part of
   * the candidate fingerprint it checks the checkpoint against. A
   * `Promise` (unlike the Python twin's plain property) because hashing
   * uses the Web Crypto global, the same constraint checkpoints.ts
   * documents.
   */
  parseOptionsDigest(): Promise<string>;

  /**
   * Whether this connector can read `reference` at all (extension, MIME,
   * or content sniffing) — a caller dispatching across several connectors
   * checks this before calling `read`.
   */
  supports(reference: string): boolean;

  /**
   * Reads `reference` into a normalized document.
   *
   * Never rejects for an ordinary parse failure — an unreadable,
   * encrypted, corrupt, or unsupported object is reported as a
   * `ConnectorDocument` with an empty `text` and a non-empty
   * `diagnostics` (ADR 0007 §5), not an exception. Rejecting is reserved
   * for a programming error (e.g. calling `read` on a reference
   * `supports` already rejected).
   */
  read(reference: string): Promise<ConnectorDocument>;
}
