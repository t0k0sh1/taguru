# langchain-taguru (Python)

Official LangChain integration for the [Taguru](https://github.com/t0k0sh1/taguru)
long-term semantic memory server. The TypeScript twin (`langchain-taguru` on
npm) exposes the identical surface.

```sh
pip install langchain-taguru
```

```python
from langchain_openai import ChatOpenAI
from taguru_langchain import TaguruIngester, TaguruRetriever

# Write: an LLM decomposes documents into the association graph
# (the LangChain twin of `taguru extract`; per-source replace, idempotent).
ingester = TaguruIngester(
    context="sake",
    llm=ChatOpenAI(model="gpt-4.1", temperature=0),
    create_context=True,
    context_description="青嶺酒造という架空の酒蔵の知識",
)
ingester.ingest_documents(docs)          # docs[*].metadata["source"] required

# Read: graph lane (resolve → activate → citations) + text lane
# (search_passages), merged by Reciprocal Rank Fusion.
retriever = TaguruRetriever(context="sake", k=8)
documents = retriever.invoke("青嶺酒造")
```

Runnable use-case examples (RAG QA with citations, governed ingestion,
conversational long-term memory — each mirrored in TypeScript) live in
[examples/langchain](https://github.com/t0k0sh1/taguru/tree/main/examples/langchain);
they work offline, no API key needed.

`TaguruIngester` takes an optional `on_event` callback for live progress —
document/chunk/attempt/import/embedding-refresh events, including *why* a
corrective attempt fired. Useful with slow local models, where a single
`ingest_text()` call can otherwise look like one long silent block:

```python
ingester = TaguruIngester(..., on_event=lambda event: print(event.kind))
```

## Checkpoint/resume for spot and preemptible instances

Pass `checkpoint_store` to survive an interruption mid-document (a killed
process, a reclaimed spot instance) without losing every chunk already
extracted for it:

```python
from taguru_langchain import FilesystemCheckpointStore

ingester = TaguruIngester(
    ...,
    checkpoint_store=FilesystemCheckpointStore(".taguru-checkpoints"),
)
```

Each chunk's accepted output is durably persisted (keyed by the chunk's own
content hash) before the next chunk starts; rerunning the same
`ingest_text()`/`ingest_documents()` call after an interruption resumes
without re-calling the model for chunks already completed. Changing the
document's content, the model, or any output-shaping setting (`fact_budget`,
`structured_output`, `questions`, ...) invalidates the whole cache rather
than risking a silent reuse of an incompatible output. The checkpoint is
cleared once the document's batch actually lands in `/import`, and kept if
the document ultimately fails — so a `dry_run=True` call, which never
imports, still records checkpoints but never deletes them. Pass
`should_stop` (a zero-argument callable, or a `threading.Event`) to stop
cooperatively between chunks; `IngestOutcome.interrupted` reports whether
that happened.

`checkpoint_store` accepts anything implementing the three-method
`CheckpointStore` protocol (`load`/`save`/`delete`, keyed by source id), so
object storage or a database work as a drop-in replacement for
`FilesystemCheckpointStore` on an ephemeral instance with no durable local
disk:

```python
class S3CheckpointStore:
    def load(self, source: str) -> bytes | None: ...
    def save(self, source: str, data: bytes) -> None: ...  # must be atomic
    def delete(self, source: str) -> None: ...
```

To force a full re-extraction ignoring whatever is cached, delete that
source's checkpoint yourself — `store.delete(source)`, or
`FilesystemCheckpointStore.path_for(source).unlink()`.

For composing this with a bounded, resumable runner (time/item windows,
signal handling, torn-import repair), see
[long-running ingestion](https://t0k0sh1.github.io/taguru/long-running.html).

## Standard ingest connectors (ADR 0007)

`taguru_langchain.ingest_connectors` normalizes a source document — plain
text plus paragraph-indexed section headings and typed citation locators
(page/slide/sheet/table) — into the one shape `TaguruIngester` consumes,
instead of a bare `page_content` string. The reference connector reads
`.md`/`.txt`, extracting ATX headings as sections:

```python
from taguru_langchain.ingest_connectors import TextFileConnector, ingest_connector_document

document = TextFileConnector().read("docs/manual.md")
if document.diagnostics:
    ...  # encrypted, corrupt, unsupported format, OCR required, ... — never a silently empty passage
outcome = ingest_connector_document(ingester, document)
```

`ConnectorDocument.sections`/`.locators` round-trip losslessly through
`/import` into `Citation.section`/`.locator` on every
`recall`/`explore`/`activate`/`cite_passage` response. A connector's own
fetch/parse work is independently resumable via `ConnectorCheckpoint`,
layered over the same `CheckpointStore` `checkpoint_store=` already uses
above — a distinct `namespace=` keeps the two from colliding when they
share one `FilesystemCheckpointStore` directory:

```python
from taguru_langchain.ingest_connectors import ConnectorCheckpoint

checkpoint = ConnectorCheckpoint(
    FilesystemCheckpointStore(".taguru-checkpoints"), namespace="connector"
)
cached = checkpoint.load(source, fresh_fingerprint)  # None on any mismatch — never a false hit
if cached is None:
    document = TextFileConnector().read(path)
    checkpoint.save(document)
```

`PdfConnector` (issue #348) reads `.pdf` files the same way — `pip install
"langchain-taguru[pdf]"` for its `pypdf` dependency, kept out of the
default install per ADR 0007 §3/§4 so a caller ingesting nothing but
`.md`/`.txt` never pays for a PDF parser. One `{"kind": "page", "value":
...}` locator is emitted per paragraph, derived from the PDF's own page
boundaries; its outline (bookmarks), if any, becomes `sections` and the
document `title`. No OCR engine ships here: a page whose extracted text
has fewer than `min_chars_per_page` (default 16) non-whitespace characters
is named in an `ocr_required` diagnostic instead of silently passed
through as low-quality text — raise the threshold for a corpus of
mostly-image PDFs, or configure `ocr_adapter=` (issue #352, see below) to
recover exactly the pages it names. Encrypted and corrupt PDFs are
reported the same structured way
(`encrypted`/`corrupt`), never a raised exception:

```python
from taguru_langchain.ingest_connectors import PdfConnector

document = PdfConnector().read("docs/manual.pdf")
if document.diagnostics:
    ...  # encrypted, corrupt, ocr_required, ... — never a silently empty passage
outcome = ingest_connector_document(ingester, document)
```

`HtmlConnector` (issue #349) reads both a local `.html`/`.htm`/`.xhtml` file
and an `http(s)://` URL — no extra to install, parsing is stdlib
`html.parser` only. Boilerplate (script/style/nav/aside, and a page's own
header/footer when nothing scopes the content to a `<main>`/`<article>`) is
stripped before `text` is built; the heading hierarchy survives as a
breadcrumb `section` per paragraph (`"Guide > Installation"`, since
`sections` is flat); and each heading's own `id` (or its nearest
`id`-bearing ancestor's) becomes a `{"kind": "fragment", "value": ...}`
locator on every paragraph up to the next heading — combined with
`metadata.canonical_url`, a citation can point at a real in-page deep link.
A URL fetch's source id is the *final*, fragment-stripped, canonicalized
URL (ADR 0007 §6.1) — `<link rel="canonical">`, when present, only ever
populates `metadata.canonical_url`, never the source id itself. A page with
no extractable text after boilerplate removal (image-only, an unrendered
JS-shell SPA) is `ocr_required` with empty `text`, and a 4xx/5xx response,
a non-HTML `Content-Type`, or a raw body over `max_file_bytes` are each
their own diagnostic — never a raised exception. By default, a URL fetch
also refuses any destination (including one reached only via a redirect)
that resolves to a private, loopback, link-local, or multicast address —
`HtmlConnector` still assumes the caller controls or trusts the URL itself,
but this stops an otherwise-trusted URL from being turned into a probe of
`localhost` or a cloud metadata endpoint by a redirect the origin server
controls. Pass `allow_private_networks=True` to fetch one intentionally
(a local test server, an internal document server on a private network):

```python
from taguru_langchain.ingest_connectors import HtmlConnector

document = HtmlConnector().read("https://example.com/guide")
if document.diagnostics:
    ...  # unreadable, unsupported_format, ocr_required, ... — never a silently empty passage
outcome = ingest_connector_document(ingester, document)
```

`DocxConnector` (issue #350) reads `.docx` files — `pip install
"langchain-taguru[docx]"` for its `python-docx` dependency, kept out of the
default install per ADR 0007 §3/§4 for the same reason `PdfConnector`'s
`pypdf` is. The document body is walked in real document order (paragraphs
and tables interleaved, never `document.paragraphs`/`.tables` separately);
a heading's breadcrumb becomes a `section`, the same `"Guide >
Installation"` convention `HtmlConnector` uses. A table — top-level or
nested inside another table's cell — becomes exactly one paragraph (rows
joined with `\n`, cells with `" | "`) carrying a `{"kind": "table", "value":
...}` locator (`"3"`, or `"3.1"` for a table nested inside table 3's own
cell); an ordinary body paragraph never carries a locator, which is what
makes "this paragraph has a locator" mean "this paragraph is a table" when
reading this connector's `locators`. No OCR engine ships: a document left
with no extractable text (an image-only `.docx`) is `ocr_required` with
empty `text`. A password-protected `.docx` is recognized by its container's
own signature and reported `encrypted` before ever being opened as a zip; a
merely corrupt/truncated package is `corrupt`. Footnote/endnote/comment
text and text-box content are each unreachable through this connector's own
paragraph walk — named in a single `partial_extraction` diagnostic rather
than silently short-changed. Only `.docx` is read; `.doc` (legacy binary)
and `.docm` (macro-enabled) are both `unsupported_format`:

```python
from taguru_langchain.ingest_connectors import DocxConnector

document = DocxConnector().read("docs/manual.docx")
if document.diagnostics:
    ...  # encrypted, corrupt, ocr_required, partial_extraction, ... — never a silently empty passage
outcome = ingest_connector_document(ingester, document)
```

`PptxConnector` (issue #352) reads `.pptx` files — `pip install
"langchain-taguru[pptx]"` for its `python-pptx` dependency, kept out of the
default install per ADR 0007 §3/§4 for the same reason `DocxConnector`'s
`python-docx` is. A slide's shapes are walked in document order, recursing
into a group shape's own nested shapes; every non-empty text-frame
paragraph and every table (rows joined with `\n`, cells with `" | "`, one
paragraph per table) carries a `{"kind": "slide", "value": ...}` locator —
unlike `DocxConnector` (whose one-locator-per-paragraph budget goes to
tables, since a DOCX has no page-like structure of its own to spend it on
instead), a slide already has a number to spend that budget naming, so
here it goes to distinguishing a slide's body from its speaker notes
instead: a slide's notes are read as their own paragraph(s), each carrying
`{"kind": "speaker_notes", "value": ...}`. A slide's title is read like any
other paragraph (same `slide` locator as its neighbors) and additionally
becomes the paragraph-anchored `section`. No OCR engine, no rasterizer: a
presentation left with no extractable text (an image-only deck) is
`ocr_required` with empty `text`. A chart, a SmartArt diagram, and an
embedded/linked OLE object are each unreachable through this connector's
own shape walk — named in a single `partial_extraction` diagnostic rather
than silently short-changed. Only `.pptx` is read; `.ppt` (legacy binary)
and `.pptm` (macro-enabled) are both `unsupported_format`:

```python
from taguru_langchain.ingest_connectors import PptxConnector

document = PptxConnector().read("docs/deck.pptx")
if document.diagnostics:
    ...  # encrypted, corrupt, ocr_required, partial_extraction, ... — never a silently empty passage
outcome = ingest_connector_document(ingester, document)
```

No OCR engine ships in this package, or in any connector (ADR 0007 §10).
`OcrAdapter` (issue #352) is the external boundary a connector calls out to
when one is configured — given the raw document bytes and the locators
naming which pages/units are unusable, an adapter returns whatever text it
could recover for them, each still tagged with the same locator it was
asked about; `PdfConnector` is the one connector wired to call one today
(`ocr_adapter=`), offering it exactly the pages its own
`min_chars_per_page` threshold found unusable, never the whole document.
An adapter's own failure — an exception, or simply recovering nothing for
a page — leaves that page exactly `ocr_required`, as if no adapter had
been configured at all:

```python
from taguru_langchain.ingest_connectors import OcrRecoveredUnit, OcrRequest, OcrResult

class MyOcrAdapter:
    name = "my-ocr-engine"
    version = "1.0"

    def recognize(self, request: OcrRequest) -> OcrResult:
        units = tuple(
            OcrRecoveredUnit(locator=locator, text=my_engine.recognize(request.content, locator))
            for locator in request.locators
        )
        return OcrResult(units=units)

document = PdfConnector(ocr_adapter=MyOcrAdapter()).read("docs/scanned.pdf")
```

`S3Connector`/`sync_object_storage` (issue #351) list an S3 bucket/prefix
and dispatch each object to whichever installed connector above its
extension (or, failing that, its content-type) names — `pip install
"langchain-taguru[s3]"` for its `boto3` dependency, the same opt-in extra
`PdfConnector`/`DocxConnector` already use. Credentials always come from
`boto3`'s own standard chain (environment, shared config/credentials file,
an EC2/ECS/Lambda role, ...); there is no parameter anywhere in this path
that accepts an access key or secret directly:

```python
from taguru_langchain import FilesystemCheckpointStore
from taguru_langchain.ingest_connectors import open_object_store, sync_object_storage

store, prefix = open_object_store("s3://my-bucket/reports/")
report = sync_object_storage(
    store,
    prefix,
    ingester=ingester,
    checkpoints=FilesystemCheckpointStore(".taguru-checkpoints"),
)
print(report.discovered, report.imported, report.failed, report.deleted_detected)
```

Re-running the same call is cheap twice over: an object whose bucket-listing
metadata (version id / content hash / size+last-modified, with a bare
`ETag` only as the last resort — some S3-compatible stores compute it in a
way that isn't a reliable content hash) is unchanged from last time is
never even fetched; one whose bytes are unchanged despite a metadata bump
(a tag edit, a copy-in-place) is fetched but never re-ingested. Both
checkpoints live in the same `checkpoints` store passed above — no second
store to configure.

Deleted objects are never retracted by default — `report.deleted_detected`
names them, but nothing changes in `taguru` until you opt in explicitly:
`sync_object_storage(..., deletion_policy="retract")` withdraws exactly the
objects this connector's own prior listing no longer sees; `"mirror"` goes
further and reconciles against the context's actual source list too, so it
self-heals even the first time it runs, with no prior listing to diff
against. `dry_run=True` reports what a real pass would discover/fetch/skip
— including an honest `unchanged` verdict, since S3's own listing already
carries every fingerprint field needed — without ever touching the network,
`taguru`, or the checkpoint/inventory store.

An S3-compatible endpoint (MinIO, Cloudflare R2, ...) works the same way —
pass `endpoint_url`/`region_name`/`profile_name` to `open_object_store`, or
build `S3ObjectStore` directly. This package's own tests use a `file://`
bucket (`FileObjectStore`, stdlib-only — no `boto3` needed at all) in place
of a live bucket; point at a real MinIO container the same way for an
end-to-end check outside CI:

```sh
docker run -d -p 9000:9000 -e MINIO_ROOT_USER=test -e MINIO_ROOT_PASSWORD=testtest quay.io/minio/minio server /data
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=testtest AWS_DEFAULT_REGION=us-east-1
python -c "
import boto3
c = boto3.client('s3', endpoint_url='http://localhost:9000')
c.create_bucket(Bucket='reports')
c.put_object(Bucket='reports', Key='q1.pdf', Body=open('q1.pdf', 'rb').read())
"
python -c "
from taguru_langchain.ingest_connectors import open_object_store, sync_object_storage
store, prefix = open_object_store('s3://reports/', endpoint_url='http://localhost:9000')
# ... sync_object_storage(store, prefix, ingester=..., checkpoints=...)
"
```

### Observability (ADR 0007 §11)

`sync_object_storage` and `sync_references` (issue #353's cross-connector
driver for every local-file/`http(s)://` reference — `.md`/`.txt`/PDF/
HTML/DOCX/PPTX, the non-S3 twin of `sync_object_storage` above) both
return the same `RunReport`: `discovered`/`unchanged`/`parsed`/
`extracted`/`imported`/`skipped`/`failed` counts (each source tallied
once, under its LAST phase only — never separately under every phase it
passed through), plus `duration_ms` and `interrupted`. `report.events` is
the full per-source phase history when you need it (`SourceEvent`:
`source`, `phase`, `elapsed_ms`, `bytes`, `parser`, `diagnostic`); pass
`events_out=` a path (or an already-open text stream) to also stream it as
JSONL — one line per phase transition, written the moment it happens — as
the run happens. "Append-only" describes the sidecar within ONE run: a
path is opened fresh (truncating any prior content) every call, so it
records exactly that run's own events, never a log accumulated across
multiple calls; pass an already-open stream instead if you want to
control that yourself (e.g. to append across runs). Written even under
`dry_run=True`, since the sidecar is a dry run's whole product and the
path is one you named explicitly:

```python
from taguru_langchain.ingest_connectors import sync_references

report = sync_references(
    ["docs/manual.md", "docs/manual.pdf", "https://example.com/notes.html"],
    ingester=ingester,
    checkpoints=FilesystemCheckpointStore(".taguru-checkpoints"),
    events_out="sync.jsonl",
)
print(report.to_dict())  # one JSON object: counts, duration_ms, an events_path reference
```

`sync_references`'s own `dry_run=True` means exactly "no fetch beyond a
local file's cheap `stat`, no write anywhere, including the checkpoint
stores" — a stricter, driver-level meaning than `TaguruIngester.
ingest_text`'s own `dry_run` (which still calls the model and only skips
`import_batches`); a local file reports `unchanged` only when its `size`,
`mtime_ns`, `parser`, `parser_version`, AND `parse_options_digest` all
still match what the last real run recorded — a changed parser or parsing
option is enough to report `parsed` even with byte-identical file
metadata — `parsed` on any of the five mismatching (never a false
`unchanged`), and a URL is always `parsed` — no `HEAD`, no network access
at all under `dry_run`. `S3SyncReport` is now a deprecated alias of
`RunReport` — `sync_object_storage`'s own
`tags_dropped`/`deleted_detected`/`retracted` counters are part of that
one shared shape, present (structurally zero) on every driver's report.

Three more constructor arguments bound how a chunk's structured-output
retry behaves, all optional and all unchanged by default: `fact_budget`
asks the model to keep a chunk's answer to at most N associations;
`max_attempts` (default 2, 1-10) raises or lowers the total attempts at
valid JSON per chunk before the document fails; and
`corrective_context_bytes` caps how much of a malformed answer gets
replayed back on the next attempt (`0` omits it behind a placeholder;
left unset, the default, replays it in full). Worth raising
`max_attempts` or setting `fact_budget`/`corrective_context_bytes` on slow
local models, where a large malformed answer near the output cap can
otherwise stall a chunk for minutes.

`TaguruIngester` also takes an optional `structured_output` flag (default
`False`) that asks the chat model for JSON-schema-constrained generation —
`llm.with_structured_output(MODEL_OUTPUT_JSON_SCHEMA, include_raw=True)` —
instead of parsing a free-text answer. Strictly opt-in and provider/model
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
`/import` call) if it's still invalid afterward. Pass `lossy=True` to
restore the old drop-and-proceed behavior instead — the source still
imports, and `IngestOutcome.invalid_dropped` counts what got silently
discarded.

Not provided, deliberately: a VectorStore facade (Taguru's retrieval is
structural-first — `similarity_search` would misrepresent it), a Memory class
(deprecated upstream in favor of LangGraph state), and agent Tools (the MCP
bridge `taguru-mcp` already serves the identical tools; pair it with
`langchain-mcp-adapters`).

The behavioral contract is the server's protocol document (`GET /protocol`);
the ingestion prompt/validation mirror `taguru extract` (PROMPT_VERSION is
kept in sync with `src/extract.rs`).
