# 0007. Standard ingest connectors: protocol, packaging boundary, and citation locators

- **Status**: Accepted
- **Date**: 2026-08-01
- **Issue**: #345
- **Related**: #217, #179, #195, #211, #212, #346, #347, #348, #349, #350,
  #351, #352, #353, #354, ADR 0001 §7, ADR 0005 §4, ADR 0005 §8
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

The connector / normalized-document protocol, source-id and fingerprint
contract, citation-locator wire representation, diagnostics vocabulary,
object-storage checkpoint and deletion semantics, OCR boundary, observability
shape, and packaging boundary for #217's standard ingest connectors — the
nine open decisions #345 lists before #346 (citation locator wire contract)
and #347–#354 (connector implementations, documentation) can start without
each guessing independently and risking a later, backward-incompatible
rewrite.

No code changes ship with this ADR. It is design and decision only — #346
through #354 are the implementation, exactly as #302/ADR 0006 preceded
#303–#308 and #271/ADR 0004 preceded #272–#279.

Out of scope, filed to the follow-up this ADR names rather than answered
here:

- XLSX / tabular data connectors (§13) — spreadsheet data does not fit the
  paragraph model `src/paragraph.rs` defines, and forcing it in would corrupt
  that model for every other connector.
- GCS, Azure Blob, and a local directory-watcher connector (§13) — this ADR
  fixes the protocol boundary that lets them be added later without a design
  redo; it does not implement them.
- `langchain-typescript`'s own connector parity (§13) — deliberately deferred,
  the same posture ADR 0006 §13.3 took for `langchain-taguru` reranker parity.
- Any answer-generation behavior. Connectors normalize input into Taguru's
  existing batch/import contract; they do not read, summarize, or answer.

## 2. Context

### 2.1 What exists today, and what doesn't

`taguru extract` (`src/extract.rs`) is the only in-tree document producer,
and it reads exactly `.md`/`.txt` (`docs/extract.html`, "PDFs are not among
the formats above"). `docs/local-rag-walkthrough.html` documents the
work-around explicitly: "`taguru extract` does not read PDFs... Converting a
PDF to text is this pipeline's job — plain `pypdf`, no LangChain document
loader needed for this much — before any Taguru code runs at all."
`docs/long-running.html`'s work-unit section repeats the same boundary for a
multi-section PDF: "split each PDF by heading first (a runner
responsibility — **Taguru has no PDF reader**)."

This was a deliberate, not an accidental, boundary — the credential and
packaging discipline it protects is exactly what §4 weighs here. What has
changed since that boundary was drawn is #211/#212: `TaguruIngester` in both
`langchain-taguru` packages now has real chunk-level checkpoint/resume,
typed progress events, and a cooperative-stop contract of its own (§2.4).
That machinery is a natural landing point for connector output, which is why
§4's recommended option builds on it rather than duplicating it.

### 2.2 The batch/import contract connectors must normalize into

`src/ingest.rs`'s module doc states the rule connectors must honor
unmodified: "One file states one source's COMPLETE truth: applying it first
retracts the source, then adds the file's facts, so re-importing a file is
idempotent." The wire shape (`src/ingest.rs:1054` `Batch`, `:1135` `Header`,
`:1219` `AssociationLine`, `:1245` `PassageLine`, `:1260` `QuestionLine`,
`:1267` `SectionLine`) is `#[serde(deny_unknown_fields)]` throughout, so any
new field a connector needs (§7) must be added intentionally to these
structs, not smuggled in.

A source id lives in the header (`Batch.source`, capped at
`MAX_NAME_BYTES` = 1024 bytes, `src/api.rs:1137`) and is stamped onto every
line in the file — an association line may not carry its own `source`
precisely so that retract-then-apply stays exact. `taguru extract` sets
`source = path.to_string_lossy()` verbatim (`src/extract.rs:483`).
`docs/long-running.html` already documents the extension of this scheme
below file granularity: `manual.pdf#installation` for a heading-split PDF
section, "each section is then an independent batch, its own manifest row,
its own retract-then-apply unit."

### 2.3 What citation locators exist today, and what's missing

Position information in a batch is exactly one kind today:

```jsonc
{"paragraph": N, "section": "見出し名"}      // src/ingest.rs:1260-1273
```

`PassageRecord` (`src/passages.rs:198-230`) stores `paragraphs`,
`questions`, and `sections` — nothing else. The response side,
`Citation { text, source, section: Option<String> }`
(`src/api/sources.rs:100`), has the same shape; `section` is documented as
"never omitted — `null` when the paragraph precedes every marker or the
source has none," which is honest about what exists (a free-text heading)
and equally honest about what doesn't (a page number, a slide number, a
sheet cell, a table's own address).

#217's acceptance criterion "page / section / slide などの位置が citation
まで追跡できる" cannot be met by a connector alone: there is nowhere in the
wire contract to put a PDF page number or a PPTX slide number today. §7
decides the fix.

### 2.4 The checkpoint/resume machinery connectors should reuse, not reinvent

Two independent layers already solve "don't redo unchanged work," and
connectors need a third that composes with both rather than a fourth that
duplicates either:

- **`taguru extract`'s own layer** (`src/extract.rs`): a document-level
  manifest (`ManifestEntry`, `:4907`, matched field-for-field including
  `sha256 = sha256_hex(BOM-stripped UTF-8 text)`, `:1282`) and a
  chunk-level checkpoint keyed by **the chunk's own content hash**, never
  its index (`CheckpointFingerprint`/`CheckpointUnit`/`DocumentCheckpoints`,
  `:5107`–`:5190`). This is untouched by anything in this ADR: if a
  connector's output is the same normalized text as last time, extract's own
  fingerprint already recognizes that and skips the model call, with zero
  connector-side cooperation required (§6 explains why).
- **`TaguruIngester`'s `CheckpointStore`** (`sdk/python-langchain/src/
  taguru_langchain/checkpoints.py:34`): a 3-method protocol —
  `load(source) -> bytes | None`, `save(source, data) -> None` (must be
  atomic), `delete(source) -> None` — with `FilesystemCheckpointStore`
  (`:143`) as the reference implementation, atomic-write-then-rename exactly
  like `src/storage.rs::write_atomic`. Fingerprint mismatch degrades to "no
  cache," never a false hit; this is the same posture `src/extract.rs`
  takes and this ADR requires connectors to take too (§6).

A connector needs its own checkpoint only for the layer neither of the above
covers: **did I already download/parse this object**, independent of
whether extraction has run. §6 defines that layer's shape by extending
`CheckpointStore`'s contract, not by inventing a new one.

### 2.5 The object-storage primitives already proven in this repository

`src/ship.rs:176` `open_store(url)` opens `s3://`, `gs://`, `az://`, and
`file://` uniformly: `AmazonS3Builder::from_env()` /
`GoogleCloudStorageBuilder::from_env()` / `MicrosoftAzureBuilder::from_env()`
— each cloud's own default credential chain, explicitly *not* the
credential-free `parse_url` path (`:169`'s comment: "`parse_url` alone
builds *without* env credentials, so cloud schemes go through `from_env()`
deliberately"). Listing is `store.list(Some(&prefix))` drained with
`futures_util::StreamExt`. Retry is `ShipError { Io(io::Error), Fenced }`
where `Io` is transient-by-assumption because cursors never advance past
unshipped data, so the next cycle retries for free
(`src/ship.rs:400` and the `spawn` driver at `:1358`). Tests use a real
`file://` bucket rather than minio/localstack
(`tests/http_api/replication.rs:1`: "the store client is the same code for
all four schemes; what differs per cloud is auth and the wire, which a test
without credentials cannot reach"). §9 and #351 reuse all four of these
verbatim rather than re-deriving them.

### 2.6 The wire-compatibility rule this ADR must satisfy

ADR 0005 §4's classification table is binding here. Two rows matter most:

> Add an optional response field → compatible (`adr/0005-wire-contract-
> compatibility.md:233`)
> Add an optional request field → compatible (`:237`, "no request body
> denies unknown fields today")

§7's locator decision is designed to land inside these rows: additive fields
only, never a reshaped container, never a new requirement on an existing
field.

## 3. Options considered — packaging boundary

The `scratch`-image constraint is concrete, not aspirational: `Dockerfile`
builds only `--bin taguru` (`Dockerfile:17`, `cargo auditable build --release
--locked --bin taguru`), and the runtime stage copies exactly that one
binary onto `FROM scratch`. Whatever a connector needs — a PDF parser, an
HTML boilerplate stripper, an Office reader, a cloud SDK — must not force
its way into that build, or `deny.toml`'s audit surface, the image's attack
surface, and `cargo install taguru`'s build time all grow for every server
operator, including the overwhelming majority who ingest nothing but
`.md`/`.txt`.

Three placements were compared:

| | A: second Rust binary (`taguru-ingest`) | B: Python SDK (`TaguruIngester` input side) | C: connector protocol + Python reference implementation |
|---|---|---|---|
| **Server core (`scratch`, `Cargo.lock`, `deny.toml`)** | Untouched — a new `[[bin]]` target does not enter the `taguru` image (confirmed: no `[[bin]]` section exists today, both binaries are auto-discovered by `src/main.rs`/`src/bin/*.rs`) — but `Cargo.lock` is still shared, so every new parser/cloud dependency still widens `cargo audit`'s and `cargo install taguru`'s surface for operators who never invoke `taguru-ingest` | Untouched — nothing lands in `Cargo.toml` at all | Untouched, and `Cargo.lock` stays untouched too — no Rust dependency added anywhere |
| **Parser ecosystem maturity** | Rust PDF/DOCX/PPTX crates trail Python's (`pypdf`, `python-docx`, `python-pptx`, `trafilatura`) by a wide, well-known margin — more edge-case documents silently mis-parse or panic | Python's ecosystem is the mature one | Same as B; the reference implementation gets to pick the mature ecosystem |
| **Reuse of existing checkpoint/resume machinery** | Would reimplement `ManifestEntry`/chunk-checkpoint semantics a third time in Rust, or import `src/extract.rs`'s private types across a binary boundary awkwardly | Reuses `TaguruIngester` + `CheckpointStore` (§2.4) directly — this is the layer it was built for | Same as B for the reference implementation; other-language implementations get an equivalent contract to reuse, not this exact code |
| **Cross-SDK CI parity burden** | N/A | `sdk/spec/surface.yaml` — confirmed by reading it — enumerates only the core `Taguru` client (`recall`, `import_batches`, `contexts`, …); `TaguruIngester` and everything in `langchain-taguru`/`langchain-typescript` is **not** in that checked surface. Adding connector code to `TaguruIngester` therefore does not obligate a simultaneous TypeScript port | Same — the protocol is language-neutral by construction; a TypeScript reference implementation is a scope decision (§13), not a CI requirement, either way |
| **Future GCS/Azure/directory-watcher extension (#217's stated goal)** | Extends inside one Rust binary; each new backend still competes for the same binary's dependency budget | Extends inside one Python package; same competition, one language | Extends by anyone implementing the protocol, in whatever language or runtime fits that backend — no shared binary to keep small |
| **When this option would be the right call instead** | If Rust's PDF/DOCX ecosystem matures to parity with Python's, or if a deployment specifically cannot run Python at all (air-gapped, Rust-only toolchain mandate) — the protocol from §5 still applies, only the reference implementation's language changes | If the reference implementation should be distributed as a pure client SDK add-on with no separate CLI to install — viable but redundant with C once C exists, since C's reference *is* a Python package | — |

**Decision: C.** The protocol (§5) is the frozen, language-neutral
contract; the reference implementation for #348–#352 lives in
`langchain-taguru` (a new `taguru_langchain.ingest_connectors` — or similarly
named — submodule beside `ingest.py`, not a new top-level package), reusing
`TaguruIngester`/`CheckpointStore` directly. This is "connector protocol and
reference implementations are separate," the third option #217 itself listed.
It is not a rejection of A: if Rust's parser ecosystem or an air-gapped
Rust-only requirement changes the calculus, a second reference
implementation can be added later without touching the protocol, because the
protocol was never Python-specific to begin with.

## 4. Decision

1. Packaging: Option C (§3). No new Rust dependency, no new binary, no
   change to the `scratch` image or its audit surface. The reference
   connector implementation lives in `langchain-taguru`
   (mirrored to `langchain-typescript` only per the follow-up scope in §13).
2. The connector protocol is the normalized-document contract in §5 —
   implemented once (#347), consumed by every format connector (#348–#352).
3. Source id and fingerprint follow §6: connector-native ids (`path`,
   `s3://bucket/key`, `url#fragment`) plus a connector-level checkpoint that
   composes with, not replaces, `CheckpointStore` (§2.4).
4. Citation locators get a new optional, additive line/field (§7) — never a
   repurposing of `section`.
5. Diagnostics are a closed, versioned code enum (§8), following the same
   "rename is breaking" posture `ErrorCode` already declares
   (`src/api.rs:154-159`).
6. Object storage reuses `src/ship.rs`'s `open_store`/listing/retry pattern
   verbatim from the *connector* side (calling the same crate, not
   duplicating its logic in Python) — via a thin CLI/subprocess or FFI
   boundary #351 designs — and follows §9's checkpoint-fingerprint priority
   and default-report-only deletion policy.
7. OCR is never bundled; §10 fixes the detection/adapter boundary.
8. Observability follows §11 — one machine-readable event/summary shape
   shared by every connector.
9. Nothing in this ADR changes `taguru extract`'s or `POST /import`'s
   behavior for an existing caller who sends no new field (§12).

### 4.6 Implementation note (#351): object storage stays entirely in Python

Item 6 above named a thin CLI/subprocess or FFI boundary as #351's way to
reuse `src/ship.rs`'s `open_store`/listing/retry pattern "from the
*connector* side." Implementing #351 surfaced a conflict this ADR did not
resolve: any such boundary needs a new subcommand (or a new binary) in
`src/`, which contradicts §3/§4's own Option C decision and this ADR's own
requirement traceability table (§Appendix), which states "#347–#352 は
`src/` に触れない." Issue #351's own acceptance criterion — an integration
test using a `file://` bucket, no minio/localstack — also needs a `file://`
implementation that lives wherever the rest of the connector does, i.e. in
Python, not behind a Rust-only boundary.

#351 resolves this in favor of §3/§4: object storage access is a thin
Python-side `ObjectStore` protocol (`taguru_langchain.ingest_connectors.
objectstore`) with two implementations — `S3ObjectStore` (an optional
`boto3` dependency via a new `s3` extra, the same packaging posture
`PdfConnector`/`DocxConnector` already take for their own optional parser
dependencies) and `FileObjectStore` (stdlib-only). Nothing here calls
`object_store` (the Rust crate) or any code in `src/`; what carries over
from `src/ship.rs` is the *shape* only — parse a URL, dispatch on scheme,
return a store scoped to one bucket/directory plus the prefix the URL
itself named, and the same refusal posture (a missing `file://` directory
is an error, never a silent `mkdir`; an unsupported scheme names the ones
that are supported). Credentials still come only from `boto3`'s own
standard chain (mirroring `AmazonS3Builder::from_env()`, never a bespoke
path), and §9's checkpoint-fingerprint priority, deletion policy, and
transient/permanent failure split are all implemented exactly as designed
below — only the "calls the same crate" clause of item 6 does not hold.
This is an implementation-time amendment, not a re-opening of §3/§4's own
packaging decision: the packaging boundary (no new Rust dependency, no new
binary, `src/` untouched) is upheld *more* strictly than item 6 originally
described, not less.

## 5. Normalized document contract (for #347)

The one shape every connector — present or future — produces, and the one
shape `taguru extract`/`TaguruIngester` consume from a connector instead of
reading files directly:

```json
{
  "connector_document": 1,
  "source": "s3://reports/2026/q1.pdf#p12",
  "text": "…paragraph-joined plain text, ready for `taguru extract`'s own paragraph.rs splitter…",
  "locators": [
    {"paragraph": 0, "locator": {"kind": "page", "value": "12"}},
    {"paragraph": 3, "locator": {"kind": "page", "value": "13"}}
  ],
  "sections": [
    {"paragraph": 0, "section": "四半期業績"}
  ],
  "metadata": {
    "origin_uri": "s3://reports/2026/q1.pdf",
    "display_name": "q1.pdf",
    "title": "2026年第1四半期業績報告",
    "canonical_url": null,
    "tags": ["quarterly-report"],
    "content_type": "application/pdf"
  },
  "fingerprint_inputs": {
    "raw_content_sha256": "…sha256 of the fetched/opened raw bytes, before parsing, hex…",
    "parser": "taguru-pdf-connector",
    "parser_version": "1.0.0",
    "parse_options_digest": "…sha256 of the connector's own effective config, hex…"
  },
  "diagnostics": []
}
```

Rules:

- `text` is plain paragraph-joined text — the *same* blank-line convention
  `src/paragraph.rs::split` already defines. A connector never numbers its
  own paragraphs; it produces text and lets `paragraph.rs` be, as its module
  doc already insists, "THE one function that decides where a passage's
  paragraphs begin and end." `locators`/`sections` index into whatever
  `paragraph.rs::split(text)` yields downstream, exactly as `taguru extract`
  already assumes for `--questions`/section lines.
- `locators` and `sections` are independent, both optional, both
  paragraph-indexed — never merged into one field (§7 explains why).
- A `locators`/`sections` entry naming a paragraph index outside the range
  `paragraph.rs::split(text)` actually produces is dropped and counted —
  `locators_dropped`/`sections_dropped` in the observability summary (§11) —
  never a hard failure, matching the existing `questions_dropped`/
  `sections_dropped` posture `src/ingest.rs` already takes for the same
  out-of-range case (§7.2 defines the equivalent duplicate-paragraph rule).
- `metadata` carries only what §2.3's `Citation`/`SourceEntry` can already
  represent or what §7 adds; a connector must not invent metadata the
  contract doesn't declare.
- `fingerprint_inputs.raw_content_sha256` hashes the **raw bytes fetched
  from the origin, before parsing** — deliberately not `text`, so a
  connector can detect "the object is byte-identical to last time" without
  re-parsing at all. It is consumed by §6's connector-native checkpoint,
  never by `taguru extract`'s own manifest, which independently computes and
  fingerprints `sha256_hex(text)` itself (§6.2) — the two hashes cover two
  different, deliberately non-interchangeable questions ("did the source
  object change" vs. "did the parsed text change") and neither substitutes
  for the other.
- `diagnostics` is `[]` on a clean parse; §8 defines its non-empty shape,
  and a non-empty `diagnostics` with an *empty* `text` is the required
  encoding of "nothing usable was extracted" (scanned PDF, encrypted file,
  unsupported format) — never a silently empty `text` with no diagnostic,
  which is exactly the "quietly degraded output" #217 forbids.
- `connector_document` is a version integer, checked for equality like
  `BATCH_VERSION`/`GROUP_VERSION` (`src/ingest.rs:114,119`) — a
  protocol-breaking change bumps it.

## 6. Source id and idempotency (for #347, #351)

### 6.1 Source id grammar

Extending the existing convention (§2.2) rather than inventing a new one:

| Connector | Source id |
|---|---|
| local file | `path.to_string_lossy()` — unchanged from `taguru extract` today |
| local file, sub-document unit | `path#locator`, e.g. `manual.pdf#p12` or `manual.pdf#installation` — already documented in `docs/long-running.html` |
| URL | the **canonicalized** URL (below), e.g. `https://example.com/report.html` |
| S3 object | `s3://bucket/key`, sub-unit `s3://bucket/key#p12` |

All capped at `MAX_NAME_BYTES` (1024 bytes, `src/api.rs:1137`) — a
connector that would exceed this refuses the object with a `source_id_too_
long` diagnostic (§8) rather than silently truncating, since truncation
would risk two distinct objects colliding on one source id.

**URL canonicalization is mandatory, not cosmetic.** A raw URL can carry
`userinfo` (`https://user:pass@host/...`) or a signed/temporary query
parameter (a presigned-URL signature, `?token=…`, `?X-Amz-Signature=…`) —
both are credential-shaped, and §9's "credential never reaches Taguru data"
rule (already stated there for S3) applies identically here, to source id,
connector checkpoint, batch file, and log line alike. A URL connector
therefore:

- Always strips `userinfo` before deriving the source id — it is never
  meaningful to identity and always a credential.
- Strips a fixed, documented deny-list of well-known signed/temporary
  auth query parameters (`signature`, `sig`, `token`, `access_token`,
  `x-amz-signature`, `x-amz-credential`, `x-amz-security-token`, `apikey`,
  `api_key`, matched case-insensitively) before deriving the source id —
  not only for the credential rule, but because a rotating signature would
  otherwise make the "same resource" produce a different source id on every
  fetch, breaking §6.3's idempotency the same way an unstable id breaks it
  anywhere else. The raw, uncanonicalized URL is used only transiently for
  the fetch itself and is never persisted.
- If two distinct fetches canonicalize to the same source id (a rare
  collision, e.g. two otherwise-identical URLs differing only in a stripped
  parameter), the connector refuses the later one with a diagnostic (§8)
  rather than silently overwriting — the same collision-refusal `taguru
  extract`'s `Run.claimed` map already applies for batch file names
  (`src/extract.rs:1273`).
- Any URL that must appear in a log line or the observability summary (§11)
  uses this same canonicalized, credential-stripped form — there is no
  separate "redacted display value," because canonicalization already
  produces one value safe for every purpose (identity, storage, and
  display).

### 6.2 Extract's fingerprint covers `text` changes; `locators`/`sections` need one more field

`taguru extract`'s manifest fingerprint is `sha256_hex` of the document
**text it reads**. Once a connector's normalized `text` is what extract
reads, any change in `text` — whether from new document content or from a
parser upgrade that extracts differently — already changes the hash and
already triggers re-extraction. A bare parser-version field inside
`ManifestEntry`/`CheckpointFingerprint` would therefore be redundant for
`text` changes specifically: the existing content hash already is the
correct signal there.

That reasoning does **not** extend to `locators`/`sections` (§7): a parser
upgrade can legitimately produce the *same* `text` with *different*
positional metadata (a corrected page-boundary heuristic, say), and a batch
file's `locator`/`section` lines are populated from `connector_document`
regardless of whether extraction itself re-ran. If the manifest fingerprint
only ever looks at `text`, that case is invisible to it — the manifest says
"unchanged," extraction is skipped, and the previously-written batch file
(and therefore the stored source) keeps its stale locators forever, which is
exactly the silently-wrong outcome this ADR's whole citation-fidelity
concern (§7) exists to prevent.

The fix stays inside this ADR's own scope (`ManifestEntry`/
`CheckpointFingerprint` are Taguru-internal structs, not the wire contract
ADR 0005 governs, so no compatibility classification applies): both structs
gain one more field, `locator_digest` — `sha256_hex` of a canonical
serialization of `connector_document.locators` + `.sections` — compared for
equality exactly like every other fingerprint field already is
(`Manifest::matches`, `src/extract.rs:4982`). A locator-only change now
correctly invalidates the manifest entry and re-triggers the write (not
necessarily a full model re-call: the per-chunk checkpoint, keyed on chunk
*text* content, still legitimately reuses the model's prior answers — only
the manifest-level "is the written batch file already correct" decision
needs to see `locators` too). `locator_digest` defaults to a fixed empty-set
hash for any pre-#347 manifest entry, so an old entry degrades to "differs,
re-extract once" rather than a false match, the same posture every other
new field in this ADR takes.

### 6.3 The connector's own checkpoint

What extract's fingerprint *cannot* see is "did I already fetch and parse
this object," which matters because fetching/parsing (network I/O, a heavy
PDF/DOCX parse) is its own expensive, resumable stage — the same reasoning
`docs/long-running.html` gives for extract's own chunk checkpoints, one
level upstream. A connector-native checkpoint therefore:

- Is keyed by source id (§6.1), and stores `fingerprint_inputs` (§5) —
  `raw_content_sha256` of the **fetched raw bytes** (not the parsed `text` —
  parsing itself is what's being skipped), `parser`, `parser_version`,
  `parse_options_digest`, plus (for S3 objects) §9's fingerprint fields.
- Reuses `CheckpointStore`'s 3-method contract (§2.4) — `load`/`save`/
  `delete` — rather than a bespoke format, so `FilesystemCheckpointStore`
  and any future backend work unmodified for connector checkpoints too.
- Degrades exactly like `TaguruIngester`'s existing checkpoints (§2.4): a
  missing, corrupt, or fingerprint-mismatched entry means "treat as unseen,"
  never a false hit.
- Composes with, never replaces, extract's own manifest/chunk checkpoint —
  a full run is: connector checkpoint (skip re-fetch/re-parse) → normalized
  `text` → extract's manifest/chunk checkpoint (skip re-extraction) →
  `taguru import` (retract-then-apply). Three independent, individually
  resumable stages, matching `docs/long-running.html`'s "Taguru provides /
  your runner provides" split — the connector checkpoint is squarely a
  runner responsibility, the same category extract's own checkpoints
  already sit in from `taguru import`'s point of view.
- On content change (same source id, different `raw_content_sha256`), the
  connector re-parses and produces a new `text`; the downstream
  retract-then-apply contract (§2.2) is what makes that replacement safe —
  the connector does nothing source-identity-specific beyond emitting the
  same source id again.

## 7. Citation fidelity (for #346, #348, #349, #350, #352)

### 7.1 Decision: an additive, typed locator — `section`'s meaning is unchanged

`section` stays exactly what it is today: a free-text heading label
(`{"paragraph": N, "section": "見出し名"}`). Overloading it with `"p.12"` or
`"slide 4"` was considered and rejected — it would make heading text and
positional metadata indistinguishable on the wire, break every existing
consumer's assumption that `section` is prose, and still not be
machine-filterable ("show me everything from slide 4" needs a typed field,
not string parsing).

Instead, a new, independent, paragraph-indexed, optional line:

```json
{"paragraph": N, "locator": {"kind": "page", "value": "12"}}
```

`kind` is an open string (`"page"`, `"slide"`, `"sheet"`, `"table"`, and
whatever a future connector needs — deliberately not a closed enum, so a
new `kind` is an additive, compatible change under ADR 0005 §4's "add a new
value to an enum-like field" row, which is compatible in Python
unconditionally and requires no TypeScript closed-union prerequisite because
this is a plain string, not one of ADR 0005 §2.5's seven closed unions).
`value` is a free-text string — `"12"`, not `12` — because a locator's
natural representation varies (`"A1:C4"` for a spreadsheet range is exactly
as valid as `"12"` for a page).

### 7.2 Storage and propagation

- `PassageRecord` (`src/passages.rs:198`) gains `locators: Vec<(u32,
  Locator)>` beside its existing `sections`, following the exact same
  "sorted, at most one... extends to the next marker" pattern only where it
  makes sense — unlike `section`, a `locator` does **not** extend to the
  next paragraph; a table's locator names only the paragraphs the table
  itself produced, and each connector emits one locator line per paragraph
  it needs to place, not one per range boundary (avoiding the "does it
  extend" ambiguity `section` deliberately embraces).
- At most one `locator` is stored per paragraph, mirroring `Citation.locator`
  being singular (`Option<Locator>`, never a list). When a batch names two
  `locator` lines for the same paragraph — a connector bug, or two connector
  runs disagreeing — resolution follows the same mechanism
  `PassageRecord::new`'s doc comment already cites for `sections`
  (`src/passages.rs:203-207`, "see `filter_sections` for how a paragraph
  claimed more than once is resolved"): the same last-write-wins rule
  extended to `locators`, with every displaced duplicate dropped and counted
  in `locators_dropped` (§5) rather than silently discarded.
- `Citation` (`src/api/sources.rs:100`) gains `locator: Option<Locator>`,
  alongside its existing `section: Option<String>`, following the same
  "never omitted, `null` when absent" rule already documented for `section`.
- `recall`/`explore`/`activate`/`unreachable_from`'s `attributions[]`
  resolve `locator` the same way they already resolve `section` — both are
  paragraph-locator lookups against the same `PassageRecord`.
- Wire fixtures (`tests/fixtures/wire/http/*.json`) and both SDKs' typed
  decode gain `locator` as an additional optional field — a compatible
  change per ADR 0005 §4 (§2.6). #346 owns this wire-level work; connectors
  (#348–#350, #352) only ever *produce* locators via §5's
  `connector_document.locators`.

### 7.3 Non-paragraph positions (tables, footnotes, speaker notes)

A table, footnote, or speaker-notes block is represented as its own
paragraph (or paragraph run) in `text` — never folded into the paragraph
next to it — precisely so it can carry its own `locator`
(`{"kind": "table", "value": "1"}`, `{"kind": "speaker_notes", "value":
"4"}`) distinguishing it from ordinary body text at the same citation
granularity everything else uses. This is a rule for §5's connectors to
follow when producing `text`+`locators`, not a new storage mechanism.

### 7.4 Chunk boundary vs. citation locator — the rule connectors must not blur

`taguru extract`'s own chunking (`chunk`/`labeled_document`,
`src/extract.rs:4809`/`:1875`) is a *model-input* concern — it exists to
keep individual LLM calls under `CHUNK_BYTES` and is invisible past
extraction. A connector's `locator` is a *human citation* concern — it must
survive into the final `Citation` response regardless of how extract later
chunks the text for its own model calls. Connectors therefore never derive
locators from anything chunk-shaped; they derive them from the source
document's own structure (a PDF's page boundaries, a PPTX's slide
boundaries), independent of and prior to extract's chunking.

## 8. Diagnostics (for #347, #348, #349, #350, #351, #352)

A closed, versioned code enum, following `ErrorCode`'s own declared posture
("a rename is a breaking change... like a response-shape change,"
`src/api.rs:154-159`):

| Code | Meaning |
|---|---|
| `unreadable` | object could not be fetched/opened at all — transport/I/O (retryable per §9) or permission/credential/ACL denial (terminal per §9, never auto-retried) |
| `unsupported_format` | extension/MIME/content sniffing found no matching connector |
| `encrypted` | the document requires a password/key this connector does not have |
| `corrupt` | the document's own structure fails to parse (truncated, malformed) |
| `ocr_required` | text extraction found no usable text layer (§10) — never silently emit empty text |
| `source_id_too_long` | the derived source id would exceed `MAX_NAME_BYTES` (§6.1) |
| `content_too_large` | the object exceeds the connector's or Taguru's size cap (`MAX_PASSAGE_BYTES`, 8 MiB) |
| `partial_extraction` | some content was recovered but a known-incomplete region was skipped (named, not silent) |

Each diagnostic record carries `code`, a human-readable `message`, and
`source` (the id that would have been used) — the same three fields
`taguru extract`'s own diagnostics sidecar
(`DiagnosticsSink`/`AttemptRecord`, `src/extract.rs:2372`/`:2566`) already
uses for its own per-chunk/per-document records, so a downstream tool that
already parses one JSONL diagnostics shape does not need a second parser.
Adding a new `code` value is additive (compatible, ADR 0005 §4); renaming or
repurposing an existing one is breaking, exactly like `ErrorCode`
(`src/api.rs:154-159`).

## 9. Object storage boundary (for #351)

- **Checkpoint fingerprint priority**: `version id` (when the bucket has
  versioning enabled) > `content hash` (when cheaply available, e.g. a
  `Content-MD5`/checksum the store already returns) > `(size, last-modified)`
  pair > `ETag` alone (last resort only — many S3-compatible stores compute
  `ETag` in ways that are not a reliable content hash, e.g. multipart
  uploads). Never `ETag` alone as the *first* choice, per #217's explicit
  requirement.
- **Credential boundary**: the connector's S3 access uses the standard AWS
  credential chain (mirroring `src/ship.rs:212`'s `AmazonS3Builder::
  from_env()`, not a bespoke credential path) and never writes any
  credential material into the connector checkpoint (§6.3), the emitted
  batch file, any log line, or `metadata` (§5) — the same rule `open_store`
  already enforces for the replication path.
- **Metadata mapping**: S3 object tags/metadata map into `connector_document.
  metadata.tags` (§5) only after the same caps batch import already enforces
  (`MAX_TAG_BYTES`/`MAX_TAGS_PER_SOURCE`, `src/api.rs:1166,1172`) — an
  oversized or excess tag is dropped and counted, never truncated silently,
  matching the existing `questions_dropped`/`sections_dropped` posture
  (`src/ingest.rs`).
- **Pagination/retry, and transient vs. permanent failure**: reuse
  `src/ship.rs`'s `store.list`/`ShipError` pattern (§2.5), but `ShipError::
  Io`'s "retry on the next pass" assumption does not transfer wholesale — it
  holds for `src/ship.rs`'s own replication path only because every failure
  it retries is transport-shaped (network blip, throttling, 5xx) under a
  credential the caller controls end-to-end. A connector additionally faces
  failures that are **not** transient: invalid/expired/revoked credentials,
  and object-level permission denial (403/`AccessDenied`-shaped responses).
  Retrying those on every future enumeration pass forever (§9's `Io`
  default) would silently mask a real, fixable problem behind an endless
  quiet retry loop. Connectors therefore classify object-storage failures
  into exactly two kinds: **transient** (network/timeout/5xx/throttling —
  keeps `src/ship.rs`'s existing retry-on-next-pass behavior) and
  **permanent** (authentication/authorization failure — terminal
  immediately, reported once as `unreadable` (§8) and not retried without an
  explicit operator re-run). A connector-level retry counter or backoff
  policy beyond this two-way split is left to #351's implementation, not
  fixed here.
- **Deletion detection needs a listable inventory, not just per-source
  checkpoints**: §6.3's `CheckpointStore` is deliberately keyed by one
  source id at a time (`load`/`save`/`delete`) — it has no "list every
  source this connector has ever seen" operation, which is exactly what
  detecting "an object present last run is absent from this run's listing"
  requires, especially across a process restart. The S3 connector therefore
  persists one additional, S3-connector-specific artifact alongside the
  per-source checkpoints: a **prefix inventory** — the flat list of source
  ids the connector's *last fully completed* enumeration of a given
  bucket/prefix returned, written atomically (the same write-then-rename
  discipline as `FilesystemCheckpointStore`, §2.4) only once a listing pass
  finishes without error, never incrementally mid-pass. Each run diffs the
  current listing against the prior inventory: an id present before and
  absent now is a candidate deletion, handled per the policy below; the
  inventory is then overwritten with the current listing. A missing or
  corrupt inventory degrades to "no prior run to compare against" — every
  object reads as newly discovered, never as a false deletion — the same
  "unreadable state never masquerades as a confident answer" posture every
  other checkpoint in this ADR already takes.
- **Deletion policy**: default is `report-only` — a deleted object is named
  in the observability summary (§11) but never triggers a `retract`.
  `--retract` (explicit retraction of the corresponding source) and
  `--mirror` (retract on every sync) are both opt-in flags, never defaults,
  matching #217's explicit "default で破壊的同期を行わない."

## 10. OCR boundary (for #348, #352)

- No OCR engine ships in any connector or the reference implementation.
- A PDF (or any format) whose text layer is empty or near-empty (a
  connector-defined, documented threshold — e.g. fewer than N extractable
  characters per page after whitespace normalization) is diagnosed
  `ocr_required` (§8) with an **empty `text`** in its
  `connector_document` — never a low-quality near-empty `text` passed
  through silently, which is exactly the failure mode #217 names and
  forbids.
- The external OCR adapter is a narrow interface: given the raw document
  bytes (or a rendered page image), return recovered text plus the same
  locator shape (§7) any other connector produces — i.e., an OCR adapter is
  itself just another producer of §5's normalized-document contract for the
  pages it recovers. #352 implements this interface and the PDF connector's
  (§348's) call-out to it when configured; absent a configured adapter, the
  `ocr_required` diagnostic is terminal for that document, not retried.

## 11. Observability (for #353)

- **Per-source record is an event log, not a snapshot**: a source moves
  through the seven-state vocabulary #217 names verbatim (`discovered`,
  `unchanged`, `parsed`, `extracted`, `imported`, `skipped`, `failed`) over
  the course of one run — `discovered` then `parsed` then `extracted` then
  `imported`, for example — so one JSONL line per *source* cannot hold one
  `phase` field without losing every earlier transition. The per-source
  JSONL is therefore append-only: **one line per phase transition**,
  `{source, phase, elapsed_ms, bytes, parser, diagnostic}` (`diagnostic`,
  §8, present only on `failed`/`skipped`), mirroring `taguru extract`'s own
  `DocumentRecord` shape (`src/extract.rs:2621`) at the level of one record
  per meaningful event, not one record per source.
- **Counts** are derived, not independently tracked: the run summary's
  `discovered`/`unchanged`/`parsed`/`extracted`/`imported`/`skipped`/
  `failed` totals are a tally of each source's *last* phase event in the
  run — a source that reached `imported` counts once, under `imported`,
  never separately under `discovered`/`parsed`/`extracted` too. This keeps
  "how many sources landed" and "what happened to source X, in order" as
  two different, non-conflicting reads of the same event log rather than
  two independently-maintained counters that could drift apart.
- **`--dry-run`**: reports `discovered`/planned `parsed` (i.e. which sources
  *would* be fetched/parsed) without performing any network fetch, parse,
  or write — the same "touches nothing" contract `taguru import --dry-run`
  and `taguru extract --dry-run` already guarantee. Whether a source can be
  honestly reported as `unchanged` under `--dry-run` depends on what
  comparison is available without a fetch (§6.3's checkpoint again, decided
  per source kind):
  - **Local file**: cheap metadata (size, mtime) is read without touching
    file contents; `unchanged` is reported only when that metadata matches
    what the connector checkpoint (§6.3) recorded from the last real run.
    Metadata alone cannot prove content equality (a touch-without-edit
    changes mtime; some tools preserve mtime across an edit) — the same
    inherent limitation ETag-less HTTP caching already has — so a
    metadata mismatch is reported as `parsed` (would re-fetch and compare
    `raw_content_sha256`, §6.3), never a false `unchanged`, and a metadata
    match is a best-effort `unchanged` that a real (non-dry-run) pass may
    still revise if a hash comparison ever disagrees with it.
  - **URL**: `--dry-run` performs no network access at all, including no
    `HEAD` request, so there is no cheap signal to compare against and
    `unchanged` can never be honestly reported. A URL source under
    `--dry-run` is always reported as `parsed` (an unconditional "would
    attempt to fetch") — never `unchanged`, and never a distinct `unknown`
    state, since "would attempt to fetch" is already the correct, honest
    answer without adding a fourth dry-run outcome to reason about.
  - **S3 object**: the bucket listing itself already carries the
    fingerprint fields §9 defines (version id / size / last-modified /
    ETag) at no extra request cost, so `unchanged` under `--dry-run` is as
    reliable as it is on a real run — no degradation needed here.
- **Machine-readable summary**: one JSON object per run — total counts, run
  duration, and a reference to the per-source JSONL — emitted always, not
  behind an opt-in flag, since #353's whole purpose is making this the
  default rather than `extract`'s current opt-in `--diagnostics-out`
  posture. (Unlike extract's diagnostics sidecar, which stays opt-in because
  it may embed raw model answers — §8's diagnostics carry no comparably
  sensitive payload.)
- Responsibility split follows `docs/long-running.html`'s existing table
  ("Taguru provides / your runner provides") verbatim: the connector commits
  a source's outcome the moment it lands (never batches successes up to lose
  together on a later failure), and enumerates all discovered work before
  the first fetch — the same two properties that make extract/import's own
  long-running story converge.

## 12. Backward compatibility, scope boundary, and privacy

- `taguru extract` and `POST /import`'s behavior for a caller sending no new
  field is **unchanged** — every wire addition in §7 is optional
  (ADR 0005 §4, §2.6). No existing batch file, no existing SDK version,
  needs any change to keep working.
- The server's HTTP retrieval path gains no parser, no cloud SDK, no
  connector code — §3's Option C is chosen specifically to keep this true.
  `src/lib.rs`'s public surface (`context`, `deadline`) does not grow.
- No connector persists or logs credential material anywhere Taguru data
  reaches (§9) — this is the same boundary `docs/extract.html` already
  states for `TAGURU_EXTRACT_*`: "The server holds no model credentials...
  It never touches the data directory."
- Nothing here enables or implies answer generation; connectors terminate at
  the normalized-document contract (§5), which flows only into
  `taguru extract`/`TaguruIngester` and from there into the existing
  batch/import path.

## 13. Consequences and follow-up

| Follow-up | Why deferred |
|---|---|
| XLSX / tabular connector | Table data does not fit `paragraph.rs`'s model; needs its own normalized shape, not shoehorned into `text`. A future ADR, not a #217 sub-issue. |
| GCS / Azure Blob / local directory-watcher connectors | §5's protocol and §9's checkpoint/fingerprint pattern already generalize to them (object_store already supports all three schemes per `src/ship.rs`); no design blocker remains, only implementation effort deliberately left for when there's a concrete need. |
| `langchain-typescript` connector parity | Same posture ADR 0006 §13.3 took: a behavior-adding decision for an already-shipped SDK, left to its own follow-up rather than forced by this ADR. Not required by `sdk/spec/surface.yaml` (§3). |
| A second (Rust) reference implementation | Revisit if Rust's PDF/Office parser ecosystem matures, or an air-gapped/Rust-only deployment concretely needs it (§3's Option A trigger condition). |

## 14. Documentation impact

No documentation ships with this ADR. #354 is responsible for the connector
reference page, and for updating `README.md`, `docs/extract.html` (the
"PDFs are not among the formats above" passage), `docs/long-running.html`
(work-unit examples), and `docs/local-rag-walkthrough.html` (its current
"plain `pypdf`... before any Taguru code runs" framing, which predates this
ADR) — following ADR 0005 §2.6's own rule that the PR adding a capability is
the PR that documents it, not the ADR that designed it.

## Appendix: requirement traceability

| #217 acceptance criterion | Section | Owning sub-issue |
|---|---|---|
| connector protocol と normalized document contract が定義されている | §5 | #347 |
| PDF、HTML、DOCX の代表 fixture が citation locator を保って取り込める | §7 | #348, #349, #350 |
| S3 bucket/prefix から少なくとも PDF、HTML、DOCX を dispatch して取り込める | §9 | #351 |
| 同一入力の再実行はモデル呼び出しと import を skip できる | §6.2, §6.3 | #347, #351 |
| 内容変更時は同じ source identity で安全に置換される | §6.1, §6.3 | #347 |
| 中断後に完了済み unit を再処理せず resume できる | §6.3, §9 | #347, #351 |
| 暗号化 PDF、破損文書、OCR 必須、未対応形式が構造化 diagnostic になる | §8, §10 | #347, #348, #352 |
| page / section / slide などの位置が citation まで追跡できる | §7 | #346, #348, #349, #350, #352 |
| S3 の削除は default では retract せず、dry-run/report を経て明示的に選択する | §9 | #351 |
| credential が checkpoint、batch、log、Taguru source metadata へ漏れない | §9, §12 | #351 |
| server コアの軽量性と credential boundary を維持する | §3, §4, §12 | (this ADR; enforced by #347–#352 not touching `src/`) |
| unit/integration tests に加え、任意の実 S3 互換環境を使う end-to-end 手順がある | §9 | #351 |
| README と ingest/extract docs から標準 connector へ到達できる | §14 | #354 |

| #345 completion criterion | Section |
|---|---|
| packaging boundary の3案比較と結論 | §3, §4.1 |
| connector / normalized-document protocol の JSON 形まで確定 | §5 |
| source id / fingerprint 契約 | §6 |
| citation locator の wire 変更の要否と形 | §7 |
| diagnostics のコード一覧と変更姿勢 | §8 |
| object storage の checkpoint fingerprint / credential / 削除方針 | §9 |
| OCR 境界(検出条件・空テキスト禁止・adapter interface) | §10 |
| observability の event/summary 形 | §11 |
| 初期対象からの除外方針(XLSX、GCS/Azure/directory watcher) | §13 |
