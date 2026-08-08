# langchain-taguru (TypeScript/JavaScript)

Official LangChain.js integration for the [Taguru](https://github.com/t0k0sh1/taguru)
long-term semantic memory server. The Python twin (`langchain-taguru` on PyPI)
exposes the identical surface — method names differ only by casing convention;
configuration fields are snake_case in both.

```sh
npm install langchain-taguru @langchain/core
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { TaguruIngester, TaguruRetriever } from "langchain-taguru";

// Write: an LLM decomposes documents into the association graph
// (the LangChain twin of `taguru extract`; per-source replace, idempotent).
const ingester = new TaguruIngester({
  context: "sake",
  llm: new ChatOpenAI({ model: "gpt-4.1", temperature: 0 }),
  create_context: true,
  context_description: "青嶺酒造という架空の酒蔵の知識",
});
await ingester.ingestDocuments(docs); // docs[*].metadata.source required

// Read: graph lane (resolve → activate → citations) + text lane
// (searchPassages), merged by Reciprocal Rank Fusion.
const retriever = new TaguruRetriever({ context: "sake", k: 8 });
const documents = await retriever.invoke("青嶺酒造");
```

Runnable use-case examples (RAG QA with citations, governed ingestion,
conversational long-term memory — each mirrored in Python) live in
[examples/langchain](https://github.com/t0k0sh1/taguru/tree/main/examples/langchain);
they work offline, no API key needed.

`TaguruIngester` takes an optional `on_event` callback for live progress —
document/chunk/attempt/import/embedding-refresh events, including *why* a
corrective attempt fired. Useful with slow local models, where a single
`ingestText()` call can otherwise look like one long silent block:

```typescript
const ingester = new TaguruIngester({
  ...,
  on_event: (event) => console.log(event.kind),
});
```

## Checkpoint/resume for spot and preemptible instances

Pass `checkpoint_store` to survive an interruption mid-document (a killed
process, a reclaimed spot instance) without losing every chunk already
extracted for it:

```typescript
import { FilesystemCheckpointStore, TaguruIngester } from "langchain-taguru";

const checkpointStore = new FilesystemCheckpointStore(".taguru-checkpoints");
const ingester = new TaguruIngester({
  ...,
  checkpoint_store: checkpointStore,
});
```

Each chunk's accepted output is durably persisted (keyed by the chunk's own
content hash) before the next chunk starts; rerunning the same
`ingestText()`/`ingestDocuments()` call after an interruption resumes
without re-calling the model for chunks already completed. Changing the
document's content, the model, or any output-shaping setting (`fact_budget`,
`structured_output`, `questions`, ...) invalidates the whole cache rather
than risking a silent reuse of an incompatible output. The checkpoint is
cleared once the document's batch actually lands in `/import`, and kept if
the document ultimately fails — so a `dry_run: true` call, which never
imports, still records checkpoints but never deletes them. Pass
`should_stop` (a zero-argument function, or an `AbortSignal`) to stop
cooperatively between chunks; `IngestOutcome.interrupted` reports whether
that happened.

`checkpoint_store` accepts anything implementing the three-method
`CheckpointStore` interface (`load`/`save`/`delete`, keyed by source id), so
object storage or a database work as a drop-in replacement for
`FilesystemCheckpointStore` on an ephemeral instance with no durable local
disk:

```typescript
interface S3CheckpointStore extends CheckpointStore {
  load(source: string): Promise<Uint8Array | null>;
  save(source: string, data: Uint8Array): Promise<void>; // must be atomic
  delete(source: string): Promise<void>;
}
```

To force a full re-extraction ignoring whatever is cached, delete that
source's checkpoint yourself — `await checkpointStore.delete(source)`, or
remove the file at `await checkpointStore.pathFor(source)`.

For composing this with a bounded, resumable runner (time/item windows,
signal handling, torn-import repair), see
[long-running ingestion](https://t0k0sh1.github.io/taguru/long-running.html).

`TaguruIngester` also takes an optional `structured_output` flag (default
`false`) that asks the chat model for JSON-schema-constrained generation —
`llm.withStructuredOutput(MODEL_OUTPUT_JSON_SCHEMA, { includeRaw: true })`
— instead of parsing a free-text answer. Strictly opt-in and provider/model
dependent: a chat model that cannot bind tools raises out of the
constructor immediately, before any document is ingested, rather than
surfacing later as a per-attempt failure. Either way the answer still goes
through the same lenient validation walk and business-rule checks a
free-text answer gets — a schema only narrows what shape a well-behaved
provider can return.

By default, a business-rule-invalid item (a bad weight, a dangling alias,
an out-of-range question, ...) never gets silently dropped and reported as
a success: it earns one targeted, path-addressed corrective turn naming
exactly which fields are wrong, and the source fails outright (no
`/import` call) if it's still invalid afterward. Pass `lossy: true` to
restore the old drop-and-proceed behavior instead — the source still
imports, and `IngestOutcome.invalid_dropped` counts what got silently
discarded.

`TaguruIngester` also takes the same bounded structured-output controls as
`taguru extract` and the Python SDK, all defaulting to today's unbounded,
2-attempt, full-replay behavior:

- `fact_budget` — ask the model to keep each chunk's answer to at most N
  associations, folded into the system prompt (default: unbounded).
- `max_attempts` — total attempts (1 initial + corrections) at getting the
  model to answer with the JSON object asked for, `1..=10` (default `2`;
  `1` skips the corrective turn entirely).
- `corrective_context_bytes` — how much of the model's own prior bad
  answer is replayed back to it in the next attempt's corrective turn:
  unset replays it in full (default), `0` omits it behind a placeholder,
  any other value truncates it to that many bytes.

When the provider's `AIMessage.response_metadata` says a malformed answer
was cut off at its output-length cap (`finish_reason`/`done_reason`
`"length"`, or Anthropic's `stop_reason: "max_tokens"`), the corrective
ask switches from "try again" to "try again shorter," naming
`fact_budget` when one is set.

## Standard ingest connectors (ADR 0007)

The `ingest-connectors` modules (issue #415, the mechanical mirror of the
Python package's `taguru_langchain.ingest_connectors`) normalize a source
document — plain text plus paragraph-indexed section headings and typed
citation locators (page/slide/sheet/table) — into the one shape
`TaguruIngester` consumes, instead of a bare `pageContent` string. The
reference connector reads `.md`/`.txt`, extracting ATX headings as
sections:

```ts
import { TextFileConnector, ingestConnectorDocument } from "langchain-taguru";

const document = await new TextFileConnector().read("docs/manual.md");
if (document.diagnostics.length > 0) {
  // encrypted, corrupt, unsupported format, OCR required, ... — never a
  // silently empty passage
}
const outcome = await ingestConnectorDocument(ingester, document);
```

`ConnectorDocument.sections`/`.locators` round-trip losslessly through
`/import` into `Citation.section`/`.locator` on every
`recall`/`explore`/`activate`/`citePassage` response. A connector's own
fetch/parse work is independently resumable via `ConnectorCheckpoint`,
layered over the same `CheckpointStore` `checkpoint_store` already uses
above — a distinct `namespace` keeps the two from colliding when they
share one `FilesystemCheckpointStore` directory.

The format connectors mirror their Python twins connector for connector,
with each parser dependency an optional npm peer kept out of the default
install per ADR 0007 §3/§4 — a caller ingesting nothing but `.md`/`.txt`
never pays for a PDF parser:

- `PdfConnector` (issue #348; `npm install pdfjs-dist`) — one
  `{"kind": "page", "value": ...}` locator per paragraph from the PDF's
  own page boundaries; the outline (bookmarks) becomes `sections` and the
  document `title`. A page whose text layer is under `minCharsPerPage`
  (default 16) non-whitespace characters is named in an `ocr_required`
  diagnostic; encrypted and corrupt PDFs are `encrypted`/`corrupt` —
  never a thrown exception.
- `HtmlConnector` (issue #349; no extra install — the tolerant HTML
  tokenizer is built in) — local `.html`/`.htm`/`.xhtml` files and
  `http(s)://` URLs. Boilerplate is stripped, the heading hierarchy
  becomes a breadcrumb `section` (`"Guide > Installation"`), and each
  heading's `id` becomes a `{"kind": "fragment", "value": ...}` locator.
  The source id is the final, fragment-stripped, canonicalized URL (ADR
  0007 §6.1). A URL fetch refuses private/loopback/link-local
  destinations (including via redirects) unless `allowPrivateNetworks`
  is set.
- `DocxConnector` (issue #350; `npm install fflate fast-xml-parser`) —
  body walked in real document order; heading breadcrumbs become
  `sections`; each table becomes exactly one paragraph carrying a
  `{"kind": "table", "value": ...}` locator. Password-protected packages
  are `encrypted`, unreachable footnote/comment/text-box content is one
  `partial_extraction` diagnostic, and `.doc`/`.docm` are
  `unsupported_format`.
- `PptxConnector` (issue #352; same two packages as `DocxConnector`) —
  shapes walked in document order; every body paragraph carries a
  `{"kind": "slide", "value": ...}` locator, speaker notes carry
  `{"kind": "speaker_notes", ...}`, and slide titles double as
  `sections`.

No OCR engine ships in this package or any connector (ADR 0007 §10):
`OcrAdapter` is the external boundary `PdfConnector` calls out to when one
is configured (`ocrAdapter`), offered exactly the pages its own threshold
found unusable; an adapter's failure leaves those pages `ocr_required`, as
if none had been configured.

Object storage (issues #351/#414) mirrors `src/ship.rs`'s scheme set:
`openObjectStore("s3://...")`/`"gs://..."`/`"az://..."`/`"file://..."`
returns an `ObjectStore` scoped to one bucket/prefix. Each cloud store
dynamically imports its own optional peer (`@aws-sdk/client-s3`,
`@google-cloud/storage`, `@azure/storage-blob`) and reads only that
cloud's standard credential chain — no parameter anywhere in this path
accepts a key or secret directly, and no credential ever reaches a
checkpoint, batch, log line, or `metadata`. `FileObjectStore` (no
dependency at all) is the test/air-gapped backend.

`syncObjectStorage` lists a bucket/prefix and dispatches each object to
whichever connector its extension (or content-type) names. Re-running is
cheap twice over: an object whose listing metadata (version id / content
hash / size+last-modified, `ETag` only as a last resort) is unchanged is
never fetched; one whose bytes are unchanged despite a metadata bump is
fetched but never re-ingested. Deleted objects are never retracted by
default — `report.deletedDetected` names them; `deletionPolicy:
"retract"` and `"mirror"` are explicit opt-ins, and `dryRun: true`
touches nothing. `syncReferences` is the non-S3 twin for local-file and
`http(s)://` reference lists, and `watchDirectory` (issue #414) runs the
same sync over a local directory on an interval. Both drivers return the
same `RunReport` (ADR 0007 §11): last-phase-only counts over
`discovered`/`unchanged`/`parsed`/`extracted`/`imported`/`skipped`/
`failed`, the full per-source `SourceEvent` history, and an optional
`eventsOut` JSONL sidecar streamed as the run happens.

Not provided, deliberately: a VectorStore facade (Taguru's retrieval is
structural-first — `similaritySearch` would misrepresent it), a Memory class
(deprecated upstream in favor of LangGraph state), and agent Tools (the MCP
bridge `taguru-mcp` already serves the identical tools).

The behavioral contract is the server's protocol document (`GET /protocol`);
the ingestion prompt/validation mirror `taguru extract` (PROMPT_VERSION kept
in sync with `src/extract.rs`).
