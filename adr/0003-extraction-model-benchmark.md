# 0003. Extraction-model benchmark: execution, artifacts, and CLI shape

- **Status**: Accepted
- **Date**: 2026-07-26
- **Issue**: #255
- **Related**: #189, ADR 0001, ADR 0002
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How #189's benchmark drives `taguru extract` across a model matrix, what
`benchmark-results/` contains and how it is versioned, and what CLI surface
exposes it — the four decisions #255 requires before any implementation
issue (#256–#261) can start without risking a later, backward-incompatible
rewrite of the artifact shapes. Out of scope: the metric definitions and
their computation (#257 owns the prose; this ADR only fixes the artifact
that carries them), the association-identity matching algorithm (#258),
the CLI's broader flag/environment-variable consistency audit (ADR 0002
§12.1's #248, which this ADR narrows and defers to exactly as ADR 0002
did for itself), and a single composite score or model ranking — #189's
own acceptance criteria rule that out categorically, and §9 makes it
structurally impossible to produce by accident.

## 2. Context

### 2.1 Provider selection is environment-only, and the environment always wins

`taguru extract` has no flag naming a model or an endpoint.
`ChatClient::from_env` (src/extract.rs:1985-2005) reads `TAGURU_EXTRACT_URL`
and `TAGURU_EXTRACT_MODEL` from the process environment — both required,
process startup fails without either — plus optional
`TAGURU_EXTRACT_API_KEY` and `TAGURU_EXTRACT_TIMEOUT_SECS`. Nine more
settings resolve the same way inside `run()`: `TAGURU_EXTRACT_PARALLEL`,
`_FACT_BUDGET`, `_MAX_ATTEMPTS`, `_CORRECTIVE_CONTEXT_BYTES`,
`_STRUCTURED_OUTPUT`, `_MAX_OUTPUT_TOKENS`, `_LOSSY`, `_DIAGNOSTICS`,
`_DIAGNOSTICS_RAW_BYTES`. The one file-based override, `--config FILE`
(`load_config`, src/config.rs:114-153), applies its `KEY=VALUE` pairs with
`std::env::set_var` but gives the **ambient environment precedence** — a
key already exported is left alone and a warning is printed
(src/config.rs:139-142: *"the real environment wins: a `docker run -e` or
shell export must override the file"*). `load_config` is also documented as
sound only single-threaded (src/config.rs:113-115), so applying it a
second time mid-process, between benchmark cells, is not an option `taguru
extract`'s own code permits.

### 2.2 Two mechanisms silently no-op a repeated run

`.extract-manifest.json` (`MANIFEST_NAME`, src/extract.rs:187) is written
at `args.out.join(MANIFEST_NAME)` — rooted at `--out`. `Manifest::matches`
(src/extract.rs:4644-4658) skips a document when eleven fields agree:
`sha256`, `model`, `prompt_version`, `context`, `questions_n`,
`no_passage`, `description`, `fact_budget`, `structured_output`,
`max_output_tokens`, `lossy`. **Endpoint URL, timeout, `--parallel`, and
`--max-attempts` are not among them.** Two models registered under the
same `TAGURU_EXTRACT_MODEL` string but served from different endpoints —
routine when comparing a local build against a hosted copy of the same
weights — would silently skip each other's documents in a shared
directory.

`.extract-checkpoints/` (`CHECKPOINT_DIR_NAME`, src/extract.rs:4746) is
likewise rooted at `--out` (`Run::load_checkpoints`, src/extract.rs:1145-
1156) and gated by a `CheckpointFingerprint` (src/extract.rs:4780-4791)
built from the same eleven-minus-`output` fields. `--force` bypasses both
— but `load_checkpoints` treats `--force` as "redo this document over"
and returns an **empty** store (src/extract.rs:1140-1144), discarding
issue #179's resume capability entirely.

### 2.3 What the diagnostics sidecar records, and what it drops

`--diagnostics-out` writes one `AttemptRecord` (src/extract.rs:2269-2288)
per completion call: `kind` (always `"attempt"` today), `source`, `stage`,
`chunk_index`, `attempt`/`max_attempts`, `state`, `length_limited`,
`elapsed_seconds`, `provider_metadata` (`finish_reason`, token counts),
`parse_error`, `validation_issues`, and three conditionally-omitted
fields. Its own doc comment (src/extract.rs:2260-2267) states the field
names deliberately mirror
`sdk/python-langchain/src/taguru_langchain/events.py`'s `AttemptFailed`/
`ProviderMetadata` *structurally*, checked by
`attempt_record_serializes_the_shared_key_set`
(src/extract.rs:4951) and its Python twin.

Nothing in the sidecar identifies a chunk beyond its index and nothing
identifies a document beyond its path string. `chunk()` (src/extract.rs:
4471-4494) is applied to `labeled_document(&text, CHUNK_BYTES)`
(src/extract.rs:1751-1770), not to the original file — its own doc
comment says why reassembly is unneeded: *"Chunks are prompt input only …
so exact reassembly does not matter"* (src/extract.rs:4466-4469). A byte
offset carried by `chunk()` would therefore be an offset into a
`[N] `-labeled derived rendering, shifted by every label, not into the
document a reader has on disk. `Run::report` (src/extract.rs:1710-1739)
prints one human-readable summary line per document — association count,
alias count, duplicates, dropped — to stdout, not to any structured
sidecar record.

### 2.4 CLI surface and dependency audit

`taguru` hand-rolls its argument parsing on purpose — no `clap` dependency
exists anywhere in `Cargo.lock`. `dispatch()` (src/cli.rs:293-324) is a
flat `match args.first()` over fourteen single-word verbs (`serve`,
`route`, `version`, `help`, `health`, `inspect`, `estimate`, `import`,
`export`, `compact`, `restore`, `extract`, `calibrate`, `communities`);
its own module doc states the reasoning: *"Hand-rolled on purpose — a
default `serve`, three offline subcommands, and one flag do not need an
argument framework; the same reasoning that keeps the metrics and BM25
in-tree"* (src/cli.rs:1-4). No verb is nested and none is hyphenated.

ADR 0002 §2.1 documents the existing pattern for a CLI verb that talks to
a remote server: `health`, `calibrate`, and `communities` all resolve one
target through `default_base_url()` (src/cli.rs:485-499) — a positional
URL, else `TAGURU_ADDR` — and authenticate with `TAGURU_API_TOKEN`,
falling back to the first `name:token` entry of `TAGURU_API_TOKENS`
(src/calibrate.rs:65-67 documents this as *"the same variables the server
reads"*).

`toml` (and its transitive `toml_edit`, `toml_datetime`, `serde_spanned`,
`winnow`) is absent from `Cargo.lock` entirely. `serde_json` is already a
direct dependency, already carrying the `raw_value` feature
(Cargo.toml:22-25), and is the format of every artifact this ADR governs
plus `taguru extract`'s own on-disk state
(`.extract-manifest.json`). Every dependency in this tree carries a
justification comment (Cargo.toml:31-58 for examples); the project's own
practice is to prefer what is already in the tree.

`src/api.rs` declares its submodules private by default — `mod sources;`
(src/api.rs:36) — with exactly one exception, `pub(crate) mod
communities;` (src/api.rs:26-28), annotated: *"`taguru communities` (the
CLI) shares the manifest shape and the artifact naming/label constants
with the verbs."* `src/lib.rs:9-12` states the crate's public library
surface is `context` alone; `api`, `registry`, and `storage` are declared
only in `src/main.rs` (`mod api;`, src/main.rs:1) and are unreachable even
from the binary's own library dependencies.

### 2.5 Prior art: ADR 0001's harness

`adr/0001/harness.py` and `adr/0001/rollup.py` already ran a model ×
mechanism × document × ceiling × repetition matrix against a live
endpoint, journaling one JSON object per line to
`adr/0001/results/attempts.jsonl` (doubling as both event log and resume
index) and rolling it up into eight CSVs. ADR 0001 §13 names this
prototype a usable seed: *"#189's benchmark tooling may treat it as a
seed/prototype; nothing here constrains #189's design"*
(adr/0001-structured-extraction-reliability.md:571-572).

## 3. Options considered

### A. Execution: link the harness against `extract`'s internals and drive cells in-process

- Evidence: §2.1 — every provider setting `ChatClient::from_env` reads
  comes from the process environment, and `load_config`'s `set_var` is
  documented sound only single-threaded (src/config.rs:113-115). Running
  cell after cell in one process means either mutating the environment
  between cells (ruled out by that same comment) or threading a new
  settings value through `run()`, the twenty-odd-field `Run` struct, and
  every call site of a module past six thousand lines.
- Evidence: the refactor's only consumer would be the benchmark, and its
  effect is a measured code path that is not the code path an operator
  runs — exactly what #189's acceptance criterion 1 ("複数モデルを同一
  fixture・prompt・設定で実行") exists to rule out. A benchmark's strongest
  claim to credibility is that it ran what users run.
- Verdict: **Rejected.**

### B. Execution: spawn `taguru extract --diagnostics-out` once per (model, run) cell

- Evidence: every verb is already a pure `args -> exit code` function
  (`Some("extract") => exit(crate::extract::run(&args[1..]))`,
  src/cli.rs:317); a subprocess is the seam this codebase already
  exposes, not one built for the occasion.
- Evidence: `--diagnostics-out` already emits per-attempt state, retry
  count, finish reason, and token counts (§2.3) — acceptance criteria 3
  and 4 are already produced by the production binary and need only be
  collected, not invented.
- Evidence: exit codes are an existing machine-readable contract — 0
  success, 1 operation failure, 2 usage error (src/cli.rs:6-7) — so a
  cell's outcome needs no prose parsing.
- Evidence: §2.5 — `adr/0001/harness.py` already validated this exact
  matrix shape against a live endpoint; this option is that harness with
  `taguru extract` substituted for hand-built HTTP request bodies.
- Cost, accepted: one process spawn per cell (negligible against LLM
  latency), and — as §7 makes explicit — this does **not** leave
  `src/extract.rs` untouched.
- Verdict: **Adopted.**

### C. Execution: add `--url`/`--model` flags to `taguru extract` and configure cells from the command line

- Evidence: ADR 0002 §4's organizing principle — the command line, never
  an environment variable, decides which target a verb means — extends
  coherently to `extract`'s LLM endpoint.
- Evidence: but ADR 0002 §1 and §12.1 already assign "the CLI's broader
  flag/environment-variable consistency" to #248 and explicitly defer it
  there rather than decide it as a side effect of a narrower ADR. Adding
  these flags here would preempt that audit for one caller's convenience.
  A spawned child's environment is already fully determined per cell by
  its parent (§5), so the benchmark does not need the flags to work.
- Verdict: **Rejected for this ADR; left to #248.** §9.1's manifest
  records resolved settings as a key/value map specifically so a future
  move to flags needs no artifact version bump.

### D. CLI: a two-level `benchmark` namespace

`taguru benchmark extract` runs the matrix; `taguru benchmark compare`
derives artifacts from a finished results directory.

- Evidence: §2.4 counts fourteen flat, unnested verbs and zero
  precedent for a second level. The benchmark family needs at least two
  operations now (#256 runs; #257/#258/#259 derive) and plausibly a third
  (#260, optional retrieval impact).
- Evidence: **ADR 0002 §3 B's rejection of a `remote` namespace does not
  transfer, and the reason is specific, not general.** That verdict rests
  on an installed base: *"the three existing remote verbs — `health`,
  `calibrate`, and `communities` — already live at the top level, not
  under any namespace, and are wired into deployment
  (`docker-compose.yml`, Kubernetes `HEALTHCHECK`) that way. A new
  namespace would put remote access in two incompatible places in the
  same binary"* (adr/0002-remote-cli-access.md:142-149). Benchmarking has
  no top-level verb today and nothing in any deployment manifest to keep
  compatible — the premise that killed option B there is absent here.
- Evidence: what *does* transfer is the structural half of that
  verdict — `dispatch()` is a flat match, hand-rolled on purpose
  (src/cli.rs:1-4), and nesting a sub-parser is the shape that comment
  set out not to grow. This ADR accepts that cost at its actual size: one
  `match args.first()` inside `benchmark::run`, the same shape every
  existing verb already hand-rolls for its own flags, one level down —
  one new arm in `dispatch()`, not a parsing framework. ADR 0002 §3 B's
  second strand — a *separate binary* duplicating argument parsing and
  `bearer_token()` — is honored by staying inside the one `taguru`
  binary.
- Verdict: **Adopted.**

### E. CLI: flat, hyphenated verbs (`taguru bench-extract`, `taguru bench-compare`, …)

- Evidence: hyphenated verb names have no precedent in this tree — all
  fourteen existing verbs are single words (src/cli.rs:293-324). This
  trades a nesting level the codebase has never had for a naming style it
  has also never had, and repeats the second cost once per operation
  instead of paying the first cost once.
- Evidence: "benchmark" already names three things in this repository —
  the `[[bench]] name = "context"` Criterion-style target
  (`benches/context.rs`), `examples/benchmark/`, and
  `examples/http_benchmark/`. A `benchmark` namespace confines the
  collision to one documented word; several `bench-*` verbs spread it.
- Verdict: **Rejected.**

### F. Ship the harness outside the binary (an `adr/0003/harness.py`, or an `examples/` program)

- Evidence: real precedent exists — §2.5's `adr/0001/harness.py`, and
  `.gitignore`'s generic `adr/*/raw/` rule (line 8) already anticipates
  further ADR evidence harnesses.
- Evidence: but #189's own 責務の境界 (division of responsibility) section
  lists "公平な実行条件を作るハーネス" and "再現に必要な manifest" as what
  *Taguru itself* provides. ADR 0001 §13 calls `harness.py` a prototype
  and seed for #189, making #189 its productized successor, not its
  continuation. `taguru` ships as one binary; a Python harness would make
  acceptance criterion 11 (usable with a local Ollama or OpenAI-
  compatible endpoint) conditional on a repository checkout and a Python
  toolchain a binary-only user does not have.
- Verdict: **Rejected as the destination**, but its discipline is
  kept — `adr/0001/harness.py`/`rollup.py` remain the vocabulary seed for
  #256/#257 and nothing here obliges rewriting or removing them.

### G. Ship the runner as a verb but leave aggregation to external tools

`taguru benchmark extract` writes `runs/*.jsonl`; no `compare` verb ships.

- Evidence: keeps metric definitions out of the binary's compatibility
  surface while #257/#258/#259 are still being designed.
- Evidence: but acceptance criterion 8 requires metric definitions and
  their computation to be documented, and 責務の境界 lists "測定値と定義"
  as Taguru's own responsibility. Shipping only the raw journal would
  move the one artifact users actually compare models with —
  `measurements.json` — outside the product.
- Verdict: **Rejected as the whole answer**, but its discipline holds:
  §9 requires every derived artifact to be a pure function of the
  results directory alone (R4 below), reproducible by `benchmark compare`
  or by any external tool reading the same versioned files — never by
  state that existed only while models were being called.

## 4. Decision

**Option B for execution and Option D for CLI shape. `taguru benchmark
extract` spawns one `taguru extract --diagnostics-out` child process per
(model, run) cell, injecting that cell's provider settings into the
child's environment and giving every cell a fresh `--out` directory.
`taguru benchmark compare` reads a finished results directory and derives
`measurements.json`/`.csv` and `differences.jsonl`, calling no model and
touching no network. The model matrix is described by a `models.json`
file whose per-model records name provider identity and capability only,
never task settings, and reference API keys by environment-variable name,
never by value. Chunk provenance in `runs/*.jsonl` is keyed by paragraph
index, not byte offset, because `extract.rs`'s chunker operates on a
derived, labeled rendering of the document (§7). Choosing the subprocess
model does not leave `src/extract.rs` untouched: §7 requires a bounded,
additive change to its diagnostics sidecar, carried by a new prerequisite
issue (#262) rather than folded into #256.**

Four rules follow, stated once because later sections rely on them:

> **R1 — One binary, one path.** A benchmark cell runs the exact
> `taguru extract` an operator runs, selected the same way, so a
> measurement is evidence about the product, not about the harness.
>
> **R2 — The cell owns its environment.** Every setting reaching a child
> is set by the parent explicitly; nothing is inherited by accident.
> `manifest.json` records what the child was given because the parent
> built both from the same value.
>
> **R3 — A cell writes only into its own directory.** Fairness between
> runs is a property of the filesystem layout, not a flag anyone has to
> remember to pass.
>
> **R4 — Everything downstream of the models is a pure function of the
> results directory.** `compare` can be re-run, or reimplemented
> elsewhere, without re-spending model time.

## 5. The per-cell configuration contract

**Environment injection is adopted; writing a per-cell `--config` file is
rejected.** `load_config` gives the ambient environment precedence over
the file (src/config.rs:139-142) — an operator benchmarking extraction
models is overwhelmingly likely to already have `TAGURU_EXTRACT_MODEL`/
`_URL` exported, since that is the only way to run `taguru extract`
today. A cell would then silently run the *shell's* model while
`manifest.json` recorded the *file's* — the warning
(src/config.rs:135-136) lands in the child's own stderr, unreviewed. This
makes a config file the strictly worse reproduction record: it states
intent, and the environment decides reality. Writing `TAGURU_EXTRACT_API_KEY`
into a file inside the results directory is also a needless secret leak
(§8's secrets rule generalizes the same argument).

`std::process::Command::env` sets a key on the child only, overriding
whatever the parent process itself inherited, with no precedence rule to
reason about — the parent builds one settings map, passes it to the
child, and serializes the same map into `manifest.json`.

**`TAGURU_EXTRACT_*` scrub — scoped fairness, not full-environment
reproducibility.** `Command::env_clear()` is too blunt — it would also
remove `PATH`, `HOME`, and proxy variables a real endpoint may need, and
building a general-purpose allowlist for those is a materially larger
undertaking §255 did not ask this ADR to take on. The rule instead is
deny-by-default over exactly the namespace §2.1 showed `taguru extract`
itself reads: start from the parent's environment; `env_remove` every
`TAGURU_EXTRACT_*` key listed in `KNOWN_KEYS` (src/config.rs:51-\*, the
file's own typo-lint list — reused here as the enumeration of what
`extract` reads); then `env` the cell's resolved values explicitly,
including ones left at their defaults. This closes exactly the hazard R2
names — a stray `TAGURU_EXTRACT_LOSSY=1` in the operator's shell cannot
silently make one cell unfair, and no `TAGURU_EXTRACT_*` key's value is
"whatever happened to be inherited." It deliberately does **not** close a
different, broader hazard: `HTTP_PROXY`, locale, TLS trust settings, and
every other non-`TAGURU_EXTRACT_*` variable pass through from the
harness's own process unscrubbed, and `manifest.json` records only their
*names*, not their values — so two cells run under different proxy or
locale settings would not be distinguishable from the manifest alone.
Nothing observed in this codebase makes that variance likely (`extract`
reads no such variable itself), but the guarantee this section makes is
scoped to `TAGURU_EXTRACT_*` fairness, not to bit-for-bit environment
reproducibility; a future ADR revision that wants the latter needs its
own explicit allowlist-with-redacted-fingerprints design, not an
extension of this one's scrub.

**Reflected in #256**: construct and record a complete
`TAGURU_EXTRACT_*` map per cell; never write a per-cell config file;
scrub the namespace before setting it.

## 6. Run isolation and output layout

**A fresh `--out` directory per (model, run) cell is adopted; `--force`
is rejected.** §2.2's two skip mechanisms are both rooted at `--out`. A
directory no prior cell wrote into contains no manifest and no
checkpoints, so run 2..N of an identical configuration cannot be
skipped — isolation becomes a structural property of the layout, not a
flag to remember. It also neutralizes §2.2's finding that endpoint URL,
timeout, `--parallel`, and `--max-attempts` are absent from
`Manifest::matches`: two differently-configured cells sharing a
`model` string cannot cross-skip if they never share a directory.
`--force` was rejected because it empties the checkpoint store
(src/extract.rs:1140-1144) — issue #179's resume capability disappears for
the whole matrix, exactly the workload most likely to run for hours and
be interrupted — and it still does not stop two cells in one directory
from overwriting each other's raw batch files, which acceptance criterion
3 requires be retained.

### Layout

```text
<output>/
  manifest.json                          # #256 — reproduction record, document/chunk dictionary
  models.lock.json                       # models.json fully resolved (defaults folded in, no secrets)
  runs/<model_id>.run<NN>.jsonl          # #256 — one file per cell, the AttemptRecord superset
  cells/<model_id>/run<NN>/              # passed verbatim as `taguru extract --out`
      .extract-manifest.json             #   written by extract, untouched by the harness
      .extract-checkpoints/              #   written by extract
      <batch files>                      #   raw extraction output (acceptance criterion 3)
      diagnostics.jsonl                  #   passed as --diagnostics-out
      stdout.log  stderr.log  exit_code  #   the cell's own record
  measurements.json  measurements.csv    # #257
  differences.jsonl                      # #259
  report.html                            # #261, optional
```

`model_id` is an operator-chosen slug from `models.json` (§8), not the
wire model name — model names carry `:` and `/` (`qwen2.5:7b`,
`hf.co/org/repo:Q4_K_M`) that are not portable path components. `run<NN>`
is zero-padded so lexical order is run order. Nothing outside `cells/` is
written by a child process, and nothing inside `cells/` is edited by the
harness — that boundary is what keeps R4 (§4) checkable.

**Vocabulary accumulation makes document order prompt-affecting.**
`Run.vocabulary` accumulates as documents are processed and is folded
into every subsequent chunk's system prompt (src/extract.rs:1312, 1346,
1406, 1465, 3399-3404) — including from documents the manifest skipped
(`absorb_vocabulary`). The prompt a document sees therefore depends on
which documents preceded it in that process. Consequence: one cell must
be a single `taguru extract` invocation over the whole corpus, never one
invocation per document, and the corpus must reach every cell in
identical order. The harness passes the corpus as **one directory
argument** (directory expansion sorts entries, src/extract.rs:1837) rather
than as separately-ordered positional files, and records the resolved
document order in `manifest.json` (§9.1).

**`--parallel` defaults to 1 for the matrix.** It does not change
prompts, but it does change measured latency through contention and makes
diagnostics line order nondeterministic (fan-out onto one
`Mutex<BufWriter<File>>`, src/extract.rs's `DiagnosticsSink`). Joins in
`runs/*.jsonl` are always by key — `(document_id, chunk_index)`, per the
definition in §9.1 — never by line position.

**The structured-output probe is a per-cell fact, recorded, not
assumed stable.** Under `auto`, `resolve_response_format` probes the live
endpoint once per process (src/extract.rs:2306). `manifest.json` records
both the requested mode and the resolved rung per cell so `compare` can
flag a model whose rung differed between runs rather than silently
conflating a rung difference with a model difference.

**Reflected in #256**: fresh `cells/<model_id>/run<NN>/` per cell, never
`--force`; corpus passed as one sorted directory with resolved order
recorded in the manifest; `--parallel` global, default 1; every join by
`(document_id, chunk_index)`, never by line order or by `source`.

## 7. Chunk provenance: the unavoidable, bounded change to `extract.rs`

#255 requires `runs/*.jsonl` to carry chunk provenance — source,
`chunk_index`, a locator into the document, and a chunk hash — from the
start, because #259's paired diff cannot add it retroactively. §2.3
already showed the sidecar cannot supply it today, and re-deriving it
inside the harness by re-implementing `paragraph::split`'s packing rule a
second time is rejected outright: it duplicates the one function
`src/paragraph.rs`'s own module doc calls *"THE one function that decides
where a passage's paragraphs begin and end"* (src/paragraph.rs:1-2), in a
second place that must be kept bit-identical forever and would drift
silently the day `CHUNK_BYTES` or the packing rule changes.

**The coordinate is the paragraph index, not a byte range.** `chunk()`
(src/extract.rs:4471) is applied to `labeled_document(&text, CHUNK_BYTES)`
(src/extract.rs:1210 calls it this way), which prefixes every paragraph
with a `[N] ` label before chunking — so any byte offset `chunk()` could
report would describe the labeled rendering, shifted by every prefix,
never the original file. `chunk()`'s own comment states reassembly was
never a design goal (src/extract.rs:4466-4469). The paragraph index,
in contrast, is exactly what survives: `labeled_document` numbers
paragraphs directly from `crate::paragraph::split(text)`
(src/extract.rs:1751-1753), which returns `ParagraphSpan{index, start,
end, hash}` — byte offsets into the **original** text
(src/paragraph.rs:16-30) — and whose module doc states the intended
posture explicitly: *"Spans are not persisted anywhere; they are
recomputed when a passage becomes resident"* (src/paragraph.rs:9-11).
This is the same coordinate system the batch's `paragraph` locator, the
passage store, the BM25 lane, and the vector lane all already share.
Given `(document_id, paragraph)` and the manifest's pinned document
`sha256`, a byte range and the verbatim text follow from one deterministic,
offline call to `paragraph::split` — no persistence, no drift risk.

**What `runs/*.jsonl` carries and what it does not**: `paragraph_first`/
`paragraph_last` (inclusive) and a `chunk_sha256` of the chunk text as
sent to the model. No byte offset into the labeled rendering is emitted,
because it would not be dereferenceable against anything a reader has on
disk.

**What `src/extract.rs` must add — nothing on `AttemptRecord`.** Its
existing `kind` field (src/extract.rs:2270) already makes the sidecar a
tagged union; new record kinds are additive by construction. Two are
introduced (owned by #262, blocking #256 and #259, not folded into either
because it touches the production extraction path and ADR 0001 §10's
diagnostics-retention territory, and #256 is already large without it):

1. `kind: "chunk"` — once per chunk, before its first attempt: `source`,
   `chunk_index`, `chunk_total`, `chunk_sha256`, `chunk_bytes`,
   `paragraph_first`, `paragraph_last`.
2. `kind: "document"` — a structured record built at the same call site
   as `Run::report` (src/extract.rs:1710-1739), from the same
   `Extraction` value already in scope there: `associations.len()`,
   `concepts.len()`, `labels.len()`, `questions.len()`, `duplicates`,
   `dropped`. `Run::report` itself only ever *prints*
   `concepts.len() + labels.len()` combined as one "alias(es)" count for
   a human reader; the new record computes the two `BTreeMap` lengths
   separately, which costs nothing extra since both maps are already in
   scope — it is not a claim that `Run::report`'s own body already emits
   six separate numbers.

`AttemptRecord` itself gains no field. This keeps
`attempt_record_serializes_the_shared_key_set` (src/extract.rs:4951) and
its Python twin exactly as they are — no SDK-side change accompanies this
ADR. The harness denormalizes `kind: "chunk"` into every `kind: "attempt"`
line it writes to `runs/*.jsonl` (§9.2); `--diagnostics-out` itself stays
normalized (one `chunk` record per chunk), which also means this change
costs existing `--diagnostics-out` consumers nothing beyond filtering on
`kind == "attempt"` — the same discriminator already required for
today's single `kind` value, now doing double duty.

## 8. Model configuration file

**Format: `models.json`, not `models.toml` or `KEY=VALUE`.**

- `models.toml` is rejected: `toml`, `toml_edit`, `toml_datetime`,
  `serde_spanned`, and `winnow` are absent from `Cargo.lock` in their
  entirety (§2.4) — a genuinely new dependency subtree for one dev-tool
  config file, against a project whose `Cargo.toml` justifies every
  dependency in a comment and that hand-rolled its CLI parser, its BM25,
  and its metrics rather than add one. TOML's real advantage — comments —
  is answered with an optional `"note"` string per model instead.
- `KEY=VALUE` (`crate::config::parse_config`, src/config.rs:164-\*) is
  rejected: it is flat and single-valued, so N models need index-mangled
  keys (`TAGURU_BENCH_MODEL_0_URL`) that `KNOWN_KEYS`'s typo lint
  structurally cannot cover, and `load_config` applies the file to the
  *process environment*, letting the ambient environment win
  (src/config.rs:139-142) — the exact hazard §5 rejected for cells, here
  promoted to the matrix definition itself.
- `models.json` is adopted: `serde_json` is already direct, already
  carries `raw_value` (Cargo.toml:22-25), and is the format of every
  other artifact this ADR governs and of `extract`'s own on-disk state.
  One parser, one version-stamp discipline (§10), zero new dependencies.

### The fairness invariant

> A per-model record may describe only what the provider *is* or *can
> do*. Everything that shapes the task — corpus, `--context`,
> `--questions`, `--fact-budget`, `--no-passage`, `--lossy`,
> `--description`, `--parallel`, `--max-output-tokens`, run count — is
> global to the matrix and is a flag on `taguru benchmark extract`, never
> a per-model key.

Stated as an invariant, not a convention, because acceptance criterion 1
requires the same fixture, prompt, and settings across models: if task
settings were expressible per model, a `models.json` could describe an
unfair comparison that **no downstream artifact could detect** —
`measurements.json` would show a model with a larger `fact_budget` as
simply more productive. Making the settings inexpressible per model means
the file cannot encode that error. `max_output_tokens` is exactly this
kind of setting, not a capability: a lower cap directly suppresses the
association counts and raises the `length_limited` rate #257 aggregates,
the same shape of unfairness `fact_budget` already guards against — so it
is global, set once by `--max-output-tokens` on `taguru benchmark
extract`, and `extraction_settings` in `manifest.json` (§9.1) carries the
single value every cell ran under.

`structured_output` is the one deliberate exception, and its
justification does not extend to `max_output_tokens`: ADR 0001
established the structured-output rung as a provider capability, and
`auto` already probes the live endpoint per run
(`resolve_response_format`, src/extract.rs:2306; `probe_structured_output`,
src/extract.rs:2360) — the resolved rung varies per model whether or not
the file says so, and forcing one rung globally would exclude models that
cannot honor `json_schema` or handicap those that can. §6 already
requires the resolved rung be recorded per cell so a rung difference is
never mistaken for a model difference. `max_output_tokens` has no
equivalent auto-probe: nothing discovers a provider's true output ceiling
at startup, so unlike the structured-output rung, a per-model value here
would be the operator's unverified guess, not a measured capability. If a
provider's hard limit sits below the global `--max-output-tokens`, that
is a startup usage error naming the model and the limit, not a silent
per-model downgrade — the same "fail fast rather than encode an
undetectable unfairness" posture the invariant states above.

### Schema

```json
{
  "taguru_benchmark_models": 1,
  "defaults": {
    "timeout_secs": 300,
    "structured_output": "auto"
  },
  "models": [
    {
      "id": "qwen25-7b-q4",
      "label": "Qwen2.5 7B Instruct (Q4_K_M, local Ollama)",
      "model": "qwen2.5:7b",
      "url": "http://localhost:11434/v1/chat/completions",
      "api_key_env": null,
      "structured_output": "auto",
      "timeout_secs": 300,
      "note": "baseline"
    },
    {
      "id": "hosted-120b",
      "label": "gpt-oss-120b (hosted, OpenAI-compatible)",
      "model": "gpt-oss-120b",
      "url": "https://example.internal/v1/chat/completions",
      "api_key_env": "BENCH_KEY_HOSTED",
      "structured_output": "json-schema"
    }
  ]
}
```

`max_output_tokens` is deliberately absent from this schema — it is a
global `--max-output-tokens` flag on `taguru benchmark extract`, not a
per-model field (see the fairness invariant above).

- `taguru_benchmark_models` follows the batch/group/communities
  version-stamp convention (§10).
- `id` matches `^[a-z0-9][a-z0-9._-]{0,63}$`, must be unique, and is the
  only value used as a path component.
- `defaults` supplies any per-model key omitted from an entry; the
  *resolved* record (defaults folded in) is what `models.lock.json` and
  `manifest.json` record, so no reader has to re-apply defaults.
- `structured_output` takes `StructuredOutputMode`'s closed vocabulary —
  `auto`, `json-schema`, `json-object`, `off`
  (`manifest_value`, src/extract.rs:853-860) — and an unrecognized value
  is a hard usage error, matching `extract`'s own treatment of
  `TAGURU_EXTRACT_STRUCTURED_OUTPUT`.
- An unknown top-level or per-model key earns a warning naming it,
  following `load_config`'s own typo-lint posture (src/config.rs:135-136):
  a hand-edited file is exactly where a misspelled knob silently becomes
  a no-op.

### Secrets

> No API key value ever appears in `models.json`, in `models.lock.json`,
> in any artifact under the results directory, or in any file the
> harness writes.

`api_key_env` names an environment variable; the harness reads it in its
own process and passes the value to the child via `Command::env` —
memory only, never a file (the second, independent reason §5 rejects
per-cell config files). A named variable that is unset is a usage error
at matrix startup, before any model is called. `models.lock.json` and
`manifest.json` record `api_key_env`'s *name* and whether it was set,
never the value. A `models.json` carrying a key-shaped value where
`api_key_env` belongs, or an `url` whose authority component carries
inline `user:password@` userinfo, is a hard usage error at parse time —
naming the correct field and refusing to run any cell — rather than a
value the harness silently redacts on the way out. Stripping only when
*writing* an artifact was rejected: it leaves the credential sitting in
the user-authored `models.json` itself (read, not written, by the
harness, so the blockquote's guarantee would otherwise not cover it) for
as long as that file exists, and still forwards it to the child
unredacted in the meantime. Rejecting it at input time is also the
cheaper rule to enforce for the same reason §8's other secrets guard is
cheap: the check runs once, before any model is called, not once per
artifact write site.

**Reflected in #256**: parse and version-check `models.json`; enforce the
fairness invariant by construction (no per-model task-setting field
exists to parse); reject any `url` carrying inline userinfo at parse
time; resolve defaults into `models.lock.json`; validate every
`api_key_env` up front; never write a key value to disk.

## 9. Artifact schemas

### 9.1 `manifest.json`

**Obtainability triage** — every field is either always available, only
available after a provider probe, or never available on this codebase's
current execution path; the schema marks each explicitly rather than
guessing.

| #189 requirement | Availability |
|---|---|
| model name, provider, taguru version, prompt version | always |
| input/chunk hash | always (via #262's `chunk_plan`, called in-process since `benchmark` shares the binary with `extract`) |
| temperature, output cap, extraction settings | always — the `CheckpointFingerprint` set (src/extract.rs:4780-4791) minus `sha256` |
| schema hash | conditional — `null` when structured output resolved to `off` |
| model digest, quantization, context window | probe-dependent — Ollama's `/api/show` exposes these; an OpenAI-compatible `/v1/models` does not |
| environment/hardware | partly probe-dependent — OS/arch from `std`; CPU/GPU detail needs a platform probe |
| SDK versions | not applicable — the benchmark drives `taguru extract` directly, no SDK on the path; recorded as an empty object, not omitted |

**Null policy**: a probe-dependent field is present and `null`, never
omitted, mirroring `AttemptRecord.provider_metadata`/`parse_error`/
`validation_issues`, which carry no `skip_serializing_if` precisely so a
reader can distinguish "asked, nothing came back" from "this writer never
had such a field" (src/extract.rs:2279-2281 vs. 2282-2287's
`skip_serializing_if` trio). Every nullable field here is paired with a
`provider_probe` block naming what was attempted. Omission
(`skip_serializing_if`) is reserved for whole optional sub-objects whose
absence has one unambiguous meaning, matching `SearchContextPlan`'s own
`filter` field (src/api/sources.rs:480-481).

```json
{
  "taguru_benchmark_manifest": 1,
  "run_id": "20260726T091422Z-3f7a1c",
  "started_at": "2026-07-26T09:14:22Z",
  "finished_at": "2026-07-26T10:02:51Z",
  "taguru_version": "0.4.0",
  "sdk_versions": {},
  "harness": {
    "execution": "subprocess",
    "runs_per_model": 3,
    "documents_root": "corpus",
    "document_order": ["brewery.md", "pairing.md"],
    "config_path": "models.json",
    "config_sha256": "6b1f…"
  },
  "extraction_settings": {
    "prompt_version": 2,
    "chunk_bytes": 24576,
    "temperature": 0,
    "context": "bench",
    "questions": 0,
    "fact_budget": 0,
    "no_passage": false,
    "description": "",
    "max_output_tokens": null,
    "max_attempts": 2,
    "parallel": 1,
    "timeout_secs": 300
  },
  "documents": [
    {
      "document_id": "brewery",
      "path": "corpus/brewery.md",
      "bytes": 8134,
      "sha256": "9c2e…",
      "paragraph_count": 27,
      "chunk_total": 1,
      "chunks": [
        {"chunk_index": 0, "chunk_sha256": "41ab…", "chunk_bytes": 8215,
         "paragraph_first": 0, "paragraph_last": 26}
      ]
    }
  ],
  "models": [
    {
      "model_id": "qwen25-7b-q4",
      "model_name": "qwen2.5:7b",
      "endpoint": "http://127.0.0.1:11434/v1/chat/completions",
      "digest": "sha256:2c1a…",
      "quantization": "Q4_K_M",
      "context_window": 32768,
      "structured_output_requested": "auto",
      "provider_probe": {"attempted": ["GET /api/show"], "ok": true, "note": null}
    },
    {
      "model_id": "hosted-120b",
      "model_name": "gpt-oss-120b",
      "endpoint": "https://example.internal/v1/chat/completions",
      "digest": null,
      "quantization": null,
      "context_window": null,
      "structured_output_requested": "json-schema",
      "provider_probe": {"attempted": ["GET /v1/models"], "ok": true,
        "note": "the OpenAI-compatible model list carries no digest, quantization, or context-window field"}
    }
  ],
  "cells": [
    {"cell_id": "qwen25-7b-q4.run1", "model_id": "qwen25-7b-q4", "run_index": 1,
     "runs_file": "runs/qwen25-7b-q4.run1.jsonl", "cell_dir": "cells/qwen25-7b-q4/run1",
     "structured_output_resolved": "json_schema",
     "started_at": "2026-07-26T09:14:23Z", "finished_at": "2026-07-26T09:31:04Z",
     "outcome": "complete"}
  ],
  "environment": {
    "os": "linux", "arch": "x86_64",
    "cpu_model": "…", "cpu_cores": 12, "memory_bytes": 34359738368,
    "gpu": null, "hostname_hash": "d41d…"
  }
}
```

`environment.hostname_hash` is a hash, not the hostname itself, because a
manifest is meant to be shareable. The document/chunk dictionary lives
here and only here — chunking depends on `CHUNK_BYTES` and the document
alone, never on the model, so repeating it per cell would be N×M copies
of one fact; `runs/*.jsonl` denormalizes only what identifies a line
without a join (§9.2).

**`document_id` is the join key, not `source`.** The harness derives it
once, when the corpus is enumerated to build `documents[]` above: the
source path relative to the corpus root, with `/` replaced by `__` and
the extension stripped, deduplicated with a short content-hash suffix on
collision — the same flatten-then-hash scheme `checkpoint_file_name`
already uses for exactly this reason (src/extract.rs:4761-4770). It is
computed once and reused verbatim by every cell, which is what makes it
stable across models and runs; `source` is carried alongside every record
purely for human-readable display and is never assumed unique or
canonical on its own — `taguru extract` accepts whatever path string a
cell happened to be invoked with, and two cells could in principle spell
the same file differently.

### 9.2 `runs/<model_id>.run<NN>.jsonl`

One file per cell — the cell is the unit of execution, failure, retry,
and resume; a per-cell file can be deleted and reproduced in isolation,
and each is self-describing via its own header line. `kind` is the
discriminator (matching `AttemptRecord.kind`, src/extract.rs:2270):

| `kind` | Cardinality | Purpose |
|---|---|---|
| `header` | 1, line 1 | version stamp + cell identity |
| `document` | 2 per document attempted (`phase: "start"`, `phase: "end"`) | identity, then §7's structured `Run::report` |
| `chunk` | 1 per chunk | §7's provenance, before that chunk's first attempt |
| `attempt` | 1 per completion call | `AttemptRecord` verbatim + harness envelope + denormalized chunk keys |
| `cell` | 0 or 1, last line | cell totals — **its absence marks an interrupted cell** |

A `document` pair follows the same "absence marks incomplete" convention
as `kind: "cell"`: the `phase: "start"` record carries identity only
(`document_id`, `source`, `document_sha256`, `chunk_total`) and is
written before the first chunk is attempted; the `phase: "end"` record
carries §7's `Run::report`-derived counts and is written only once the
document finishes (successfully or not). A `document_id` with a `start`
record but no matching `end` record marks a document abandoned mid-run —
the same interruption signal `kind: "cell"`'s absence gives at the whole
cell's scope, one level down.

```json
{"kind":"header","taguru_benchmark_runs":1,"run_id":"20260726T091422Z-3f7a1c","cell_id":"qwen25-7b-q4.run1","model_id":"qwen25-7b-q4","model_name":"qwen2.5:7b","run_index":1,"prompt_version":2}
{"kind":"document","ts":1784787733.041,"cell_id":"qwen25-7b-q4.run1","document_id":"brewery","source":"corpus/brewery.md","document_sha256":"9c2e…","chunk_total":1,"phase":"start"}
{"kind":"chunk","ts":1784787733.042,"cell_id":"qwen25-7b-q4.run1","document_id":"brewery","source":"corpus/brewery.md","chunk_index":0,"chunk_total":1,"chunk_sha256":"41ab…","chunk_bytes":8215,"paragraph_first":0,"paragraph_last":26}
{"kind":"attempt","ts":1784787747.268,"cell_id":"qwen25-7b-q4.run1","document_id":"brewery","document_sha256":"9c2e…","chunk_sha256":"41ab…","paragraph_first":0,"paragraph_last":26,"source":"corpus/brewery.md","stage":"item","chunk_index":0,"attempt":1,"max_attempts":2,"state":"stop_malformed","length_limited":false,"elapsed_seconds":14.226,"provider_metadata":{"finish_reason":"stop","input_tokens":2105,"output_tokens":870,"total_tokens":2975},"parse_error":"expected `,` or `}` at line 31 column 4","validation_issues":null}
{"kind":"document","ts":1784787760.004,"cell_id":"qwen25-7b-q4.run1","document_id":"brewery","source":"corpus/brewery.md","document_sha256":"9c2e…","phase":"end","outcome":"written","batch_path":"cells/qwen25-7b-q4/run1/brewery.md.jsonl","associations":41,"concepts":6,"labels":2,"questions":0,"duplicates":3,"dropped":0}
{"kind":"cell","ts":1784787961.550,"cell_id":"qwen25-7b-q4.run1","outcome":"complete","documents_written":12,"attempts_total":31,"exit_code":0}
```

An `attempt` line is one merged JSON object built from three layers of
fields, not a copy of any single upstream record. **Layer 1 carries every
`AttemptRecord` field through verbatim, unrenamed and unmodified** —
`kind`, `source`, `stage`, `chunk_index`, `attempt`, `max_attempts`,
`state`, `length_limited`, `elapsed_seconds`, `provider_metadata`,
`parse_error`, `validation_issues`, and the three conditionally-omitted
fields all keep the exact value `--diagnostics-out` would have written,
so a consumer reading only those keys sees `AttemptRecord` unchanged.
**Layer 2** is the harness envelope, added alongside layer 1's keys: `ts`,
`cell_id`, `model_id`, `run_index`, `document_id`. **Layer 3** is
denormalized from
that chunk's own `kind: "chunk"` record: `document_sha256`,
`chunk_sha256`, `paragraph_first`, `paragraph_last`. The harness
denormalizes rather than `extract.rs` emitting it directly, so
`--diagnostics-out` stays normalized (one `chunk` record per chunk) and
`AttemptRecord` never changes; folding five keys onto a line the harness
is already annotating is free, and it is what lets any single
`runs/*.jsonl` line be dereferenced without a cross-`kind` join. Under
`--parallel > 1` lines interleave, so every consumer joins by key, never
by position.

`AttemptRecord`'s `state` vocabulary (ADR 0001 §7) collapses a JSON
syntax failure and a Stage 1 validation rejection into one
`stop_malformed`; only `validation_issues != null` tells them apart. The
legacy (non-ladder) path additionally never emits `length_limited` or
`refusal` as a `state`, reporting `stop_valid` with `length_limited: true`
instead. Both are carried forward as `caveat` strings on the affected
metric definitions (§9.3) rather than silently smoothed over.

**Reflected in #256**: consume #262's `chunk`/`document` records;
denormalize into `runs/*.jsonl`; join on `(document_id, chunk_index)`.

### 9.3 `measurements.json` / `measurements.csv`

**Structure enforces "no single score, no ranking" rather than trusting a
reviewer to catch one.**

1. Per-model results are a lexicographically sorted map keyed by
   `model_id` (a `BTreeMap`, the collection `Run.vocabulary` already uses
   at src/extract.rs:1081) — no array position reads as a rank.
2. No field is scoped across models: no `rank`, `score`, `winner`,
   `best`, `recommended`, `overall`, or `delta_vs_*` anywhere in either
   artifact, enforced by a unit test asserting the emitted key set
   intersects a banned-name list emptily. Every number is scoped to one
   cell, model, or document and is uninterpretable without its
   `definitions` entry.
3. `measurements.csv` is long/tidy, never wide — `model_id` is always a
   data column, never a header column, matching every one of
   `adr/0001/rollup.py`'s eight CSVs (`model` is a data column in all
   eight).
4. `report.html` (#261) can only render what is here — with no rank
   field in the data, there is nothing for it to draw as one.

**The `Distribution` shape**, used everywhere a distribution appears:

```json
{"n": 372, "min": 4.12, "p50": 12.44, "p90": 31.87, "p99": 58.10, "max": 61.02, "mean": 15.23, "sum": 5665.6}
```

Percentiles are **nearest-rank, no interpolation**: for ascending
`x[0..n-1]`, the `p`th percentile is `x[ceil(p/100 · n) − 1]` — always an
observed value, float-free in its index arithmetic, exactly reproducible
by an external re-aggregator. `sum` is present wherever a rate derives
from it, so a reader can re-derive the rate rather than trust it. Both
rules are stated in the artifact's own `definitions` block, not only in
this ADR.

**`n == 0` is a defined shape, not an omission.** A cell where every
attempt failed before producing a sample (every completion timed out,
say) still emits the metric, with `n: 0` and `min`/`p50`/`p90`/`p99`/
`max`/`mean`/`sum` all `null` — the same "present and `null`, never
omitted" policy §9.1 already states for probe-dependent fields, so a
reader can tell "measured, zero samples" from "this metric does not
apply to this scope" by the key's presence alone. A ratio metric (the `{"value", "n", "numerator"}` shape
`attempt.state_rate.*` uses below) follows the same rule at its own
denominator: `n: 0` pairs with `value: null` and `numerator: null`, never
a divide-by-zero `NaN` and never a silently misleading `0.0`.

```json
{
  "taguru_benchmark_measurements": 1,
  "run_id": "20260726T091422Z-3f7a1c",
  "generated_at": "2026-07-26T10:05:12Z",
  "percentile_method": "nearest-rank",
  "inputs": {"runs": ["runs/qwen25-7b-q4.run1.jsonl"], "cells": "cells/"},
  "definitions": {
    "latency.attempt_seconds": {
      "unit": "second", "statistic": "distribution", "scopes": ["cell", "model", "document"],
      "description": "Wall time of one completion call, request start to parsed response.",
      "source": "runs/*.jsonl kind=attempt .elapsed_seconds", "caveat": null
    },
    "attempt.state_rate.stop_malformed": {
      "unit": "ratio", "statistic": "ratio", "scopes": ["cell", "model"],
      "description": "Attempts whose terminal state was stop_malformed, over all attempts in scope.",
      "source": "runs/*.jsonl kind=attempt .state",
      "caveat": "Conflates a JSON syntax failure with a Stage 1 validation rejection; .validation_issues != null separates them (ADR 0001 §7; src/extract.rs:2269)."
    }
  },
  "cells": {
    "qwen25-7b-q4.run1": {
      "model_id": "qwen25-7b-q4", "run_index": 1,
      "latency.attempt_seconds": {"n": 31, "min": 4.12, "p50": 12.44, "p90": 31.87, "p99": 58.10, "max": 61.02, "mean": 15.23, "sum": 472.1},
      "attempt.state_rate.stop_malformed": {"value": 0.0968, "n": 31, "numerator": 3}
    }
  },
  "models": {"qwen25-7b-q4": {"…": "same metric keys, aggregated over that model's cells"}},
  "documents": {"qwen25-7b-q4": {"brewery": {"…": "…"}}}
}
```

`measurements.csv` is one tidy file, not eight:

```csv
scope,model_id,run_index,document_id,metric,stat,value,unit,n
cell,qwen25-7b-q4,1,,latency.attempt_seconds,p50,12.44,second,31
model,qwen25-7b-q4,,,latency.attempt_seconds,p50,12.91,second,93
document,qwen25-7b-q4,1,brewery,extraction.associations,value,41,count,1
```

`scope` is `cell` \| `model` \| `document`; `metric` is a dotted name
keying `definitions`; `stat` is `value` \| `min` \| `p50` \| `p90` \|
`p99` \| `max` \| `mean` \| `sum` \| `n` \| `numerator`. Adding a metric
adds rows, not columns, so `taguru_benchmark_measurements` never has to
rev just because #257 grows a metric, and the shape is inherently
rank-free — there is no column to put a rank in. `measurements.csv`
carries no version stamp of its own; its version is
`measurements.json`'s, and the pair is written atomically together.

`measurements.csv` is a **value projection** of `measurements.json`, not
a lossless flattening of it: every numeric field of every `Distribution`
and ratio (including `numerator`, so an empty-sample cell's `n: 0` and
`null`s from the rule above round-trip exactly) becomes one row, but the
artifact's `definitions` block — unit, statistic, description, source,
`caveat` — and its `inputs`/`run_id`/`generated_at` metadata stay
JSON-only. A tool that needs a metric's caveat (the `stop_malformed`
state-conflation note from §9.2, for one) reads `measurements.json`; the
CSV is for spreadsheets and `pandas`/`sqlite`, not for reconstructing the
JSON it was derived from.

`runs/*.jsonl` alone cannot supply extraction-shape metrics that need the
full item list (positive/negative weight split, distinct subjects,
relation reuse, orphan/duplicate counts) — `kind: "document"` supplies
counts `Run::report` already computes at no extra cost, and #257 re-reads
`cells/**` for the vocabulary-shape metrics rather than duplicating whole
batches into `runs/*.jsonl`. `measurements.json.inputs` names both
sources so the derivation stays auditable.

**Reflected in #257**: lives under `taguru benchmark compare`; a pure
function of the results directory; `BTreeMap` ordering; banned-key unit
test; nearest-rank percentiles; `definitions` embedded in the artifact,
not only documented in prose elsewhere.

### 9.4 `differences.jsonl`

**Vocabulary constraint, made mechanical, not just intended.** #189 and
#259 forbid verdict language — no side is a baseline or an "expected"
answer without gold data. Sides are `a`/`b`, bound to `model_id`, never
`baseline`/`candidate`. `present_in` is an array of `model_id`s (length 1
for a one-sided record); counts are `n_present`, never `n_missing`. A
banned-lexicon unit test asserts no emitted key name or `kind` value
matches
`/(miss|error|wrong|incorrect|fail|omit|expected|gold|truth|false.?positive|recall|precision|better|worse)/` —
the same mechanism §9.3 uses for the rank ban, so the constraint survives
a future contributor who has not read this ADR.

```json
{"kind":"header","taguru_benchmark_differences":1,"run_id":"20260726T091422Z-3f7a1c","pairs":[{"pair_id":"hosted-120b__qwen25-7b-q4","a":"hosted-120b","b":"qwen25-7b-q4"}],"matching":{"module":"benchmark::identity","case_fold":true,"unicode_normalization":"NFKC","alias_expansion":"batch-local","weight_tolerance":0.0},"text_included":false}
{"kind":"association_shared","pair_id":"hosted-120b__qwen25-7b-q4","present_in":["hosted-120b","qwen25-7b-q4"],"key":{"subject":"青嶺酒造","label":"located in","object":"雲居県霧沢町"},"sides":{"a":{"runs":[1,2,3],"n_present":3},"b":{"runs":[1,2,3],"n_present":3}},"locator":{"document_id":"brewery","source":"corpus/brewery.md","document_sha256":"9c2e…","paragraph":0,"chunk_index":0,"chunk_sha256":"41ab…","text":null,"text_truncated":false}}
{"kind":"association_single_side","pair_id":"hosted-120b__qwen25-7b-q4","present_in":["qwen25-7b-q4"],"key":{"subject":"青嶺","label":"type of","object":"日本酒"},"sides":{"a":null,"b":{"runs":[1,3],"n_present":2}},"locator":{"document_id":"brewery","source":"corpus/brewery.md","document_sha256":"9c2e…","paragraph":4,"chunk_index":0,"chunk_sha256":"41ab…","text":null,"text_truncated":false}}
```

`sides.a`/`sides.b` are `null` when that side has no matching item under
the key — stated as absence of a match, never as a fault. `locator`
carries `paragraph` (nullable — the extraction attached no locator, or
`--no-passage` stripped it), `chunk_index`, and `chunk_sha256`, but no
byte span: the writer **recomputes** the span by validating the file
against `document_sha256` and running `crate::paragraph::split` at diff
time, matching `paragraph.rs`'s stated "recomputed, not persisted"
posture (§7) — a locator can never point at bytes that moved. `text` is
opt-in via `--with-text` (default `null`), the same posture
`TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES` already takes for
`AttemptRecord.response_text` — embedding source text into a shareable
artifact should be a deliberate act, and the caller already has the
corpus with its hash pinned.

The header's `matching` block is mandatory: a difference set is
uninterpretable without knowing whether case folding, NFKC, and alias
expansion were applied before "present in one side only" was decided.
Recording #258's parameters there makes the whole file re-derivable. One
file covers every model pair; each record carries `pair_id`.

**Reflected in #258**: matching parameters are exactly what the header
records. **Reflected in #259**: `document_id`/`paragraph`/`chunk_index`
from `runs/*.jsonl` and `manifest.json` are sufficient to locate 元文書・
節・原文 — no byte range is threaded through from extraction.

## 10. Versioning and compatibility

| Artifact | Key | Type | Initial |
|---|---|---|---|
| `models.json` (input) | `taguru_benchmark_models` | `u64` | `1` |
| `models.lock.json` | `taguru_benchmark_models` (reused, see below) | `u64` | `1` |
| `manifest.json` | `taguru_benchmark_manifest` | `u64` | `1` |
| `runs/*.jsonl` | `taguru_benchmark_runs` | `u64` | `1` (header record, line 1) |
| `measurements.json` | `taguru_benchmark_measurements` | `u64` | `1` |
| `differences.jsonl` | `taguru_benchmark_differences` | `u64` | `1` (header record, line 1) |

`models.lock.json` carries no independent format of its own — it is
`models.json` with `defaults` folded into every entry and no other
transformation, so it reuses `models.json`'s exact stamp key and value
rather than minting a sixth. A reader that understands one schema
understands both; nothing about resolving defaults changes the shape a
version number would need to describe.

**Five independent stamps across six files, not one shared stamp.**
`src/ingest.rs:115-117`
states the precedent directly: `GROUP_VERSION` is kept *"separate from
[`BATCH_VERSION`] so either shape can rev without dragging the other
along."* The artifacts here have different producers (the user or
tooling upstream of the harness writes `models.json`; #256 writes
`models.lock.json`/`manifest.json`/`runs/`; #257 writes
`measurements.*`; #259 writes `differences.jsonl`) and different
consumers; one shared stamp would mean a one-column addition to
`measurements.csv` invalidates every archived `runs/*.jsonl` that never
changed. `u64` matches every JSON version key
already in the tree (`taguru_batch`, `taguru_group`, `taguru_communities`,
src/ingest.rs:113,118; src/api/communities.rs:106); `u32` stays reserved
for stamps on the binary wire (`IMAGE_VERSION`, src/context/image.rs:51;
`PROMPT_VERSION`, src/extract.rs:122).

**Compatibility check: range acceptance, not equality — the opposite of
`taguru_batch`, deliberately.** Two precedents already coexist in this
tree and diverge for a principled reason. `taguru_batch`/`taguru_group`/
`taguru_communities` reject any value but the exact one this build reads
(src/ingest.rs:1422-1427; src/communities.rs:718-721) — because an import
batch may be handed to taguru by *anyone*, and taguru must refuse to
guess at a shape it was not built for. `IMAGE_VERSION` instead accepts
any value from 1 through the current constant
(`(1..=IMAGE_VERSION).contains(&version)`, src/context/image.rs:219),
because an image is written *and later re-read* by taguru itself, and
compatibility across the tool's own evolution is the entire point of
keeping one on disk — its doc comment keeps a version-history log
(src/context/image.rs:34-50) for exactly this reason. A
`benchmark-results/` directory is squarely the second case: no tool but
taguru writes one, and re-reading an old one — "外部ツールでも再集計でき
る" — is what #189 asks for. **`manifest.json`, `runs/*.jsonl`,
`measurements.json`, `differences.jsonl`, and `models.lock.json`
therefore accept the `IMAGE_VERSION` posture**: readers carry
`#[serde(default)]` per field with a documented "least behavior change
for an old file" rationale, the same discipline `ManifestEntry`'s own
fields already follow (src/extract.rs:4568-4624 — each field's doc
comment states what its default preserves). Within an accepted range a
revision may only *add* a field; removing or repurposing one drops the
old version out of the accepted range and forces a bump instead.

`models.json` and §11's `eval.jsonl` are the two exceptions, and go the
other way (**equality**) for the same reason as each other: both are
*authored by the user*, not written and later re-read by taguru itself,
so the `taguru_batch` reasoning applies to them, not `IMAGE_VERSION`'s —
taguru must refuse to guess at a shape a person wrote by hand rather than
silently defaulting missing fields into a hand-edited matrix definition.
`models.lock.json` stays on the range-acceptance side of that split
despite reusing `models.json`'s stamp key, because taguru itself is what
writes and re-reads *it*.

**JSONL stamps live on the header record only, never on every line** —
`taguru_batch`'s own placement as line 1 of a batch
(src/ingest.rs:1422, header parsed first) is the precedent. A per-record
stamp would let one file claim two versions at once, which is
unresolvable; a header stamp is checked exactly once and constrains the
whole file.

## 11. Boundary with #215, and the shared evaluation dataset

> #260 and #215 measure retrieval for different reasons and must not be
> merged. #215 evaluates one already-populated corpus against labeled
> expectations — a quality gate, owning rank-based metrics, citation
> metrics, graded relevance, per-case thresholds, and a non-zero CI exit;
> its comparison is *this corpus today versus the same corpus after a
> configuration change*. #260 compares N corpora that differ only in
> which model extracted them, holding the question set, import path, and
> retrieval configuration fixed; its comparison is *model A's corpus
> versus model B's corpus*, and absent user-supplied expectations it
> reports only differences — empty-result rate, top-k overlap, rank
> movement, source diversity, per-lane hit counts — asserting nothing
> about which set of results is better. The two share exactly one thing,
> the labeled dataset: when a case carries `expected_sources` or
> `expected_concepts`, #260 computes recall@k and MRR from the same
> fields and definitions #215 uses, so one `eval.jsonl` serves both.
> Everything #215 adds beyond that subset — graded relevance for nDCG,
> citation-locator expectations, expected labels/associations, per-case
> retrieval options, regression thresholds — #260 carries through
> untouched and does not interpret.

**Shared dataset**: `eval.jsonl`, stamp `taguru_eval` (`u64`, initial
`1`), header line 1, **equality-checked** — the one format in this ADR's
scope whose producer is the user, so §10's `taguru_batch` reasoning
applies rather than its `IMAGE_VERSION` reasoning.

```json
{"taguru_eval":1,"name":"sake retrieval cases","default_target":{"context":"sake"}}
{"case_id":"brand-origin-001","query":"青嶺はどこの蔵の酒か","cues":["青嶺"],"expected_sources":[{"source":"corpus/brewery.md","paragraphs":[0],"relevance":3}],"expected_concepts":["青嶺酒造"],"options":{"limit":10}}
```

Core fields, used by both #260 and #215: `case_id` (stable, unique join
key), `query`, `cues`, `expected_sources[]` (`source`, `paragraphs[]`,
`relevance` 0–3 default 1 — presence is what switches recall@k/MRR on),
`expected_concepts[]`, `options.limit`. `target.context`/`target.groups`
are a **binding, not a case property** — #260 always overrides them, one
per-model corpus per model, so their absence is normal for a dataset
meant to be shared. #215-only extensions — `expected_labels[]`,
`expected_associations[]`, `expected_citations[]`, `options.floor`/
`sources`/`since` — are carried through by #260 without interpretation;
#260 warns once per run, not once per case, if it sees them and proceeds.
Case structs use `#[serde(deny_unknown_fields)]`
(matching `Header`/`GroupLine`, src/ingest.rs:1037,1062) so a typo in a
hand-written dataset is a reported error, not a silently ignored
expectation.

**Lane-hit evidence for #260**: `SearchPlan`/`PassageLanes`/
`LaneEvidence` (src/api/sources.rs:410-514) are `pub`, but `mod sources;`
(src/api.rs:36) is private — unreachable even from inside the binary, let
alone the library — and the search entry points are axum handlers over
`State<AppState>`, so an in-process call would additionally mean
constructing server state offline and re-implementing routing, grant, and
filter logic: measuring a path no user takes, exactly what #215 rules out
(*"Evaluate the public retrieval behavior, not private scoring
helpers"*). §2.4's `default_base_url()` precedent is the answer instead:
**#260 drives retrieval over HTTP** — `taguru import` builds each model's
corpus offline as it already does, a server serves it, and #260 calls
`POST /contexts/{name}/sources/search` per case with the same
`--url`/`TAGURU_API_TOKEN` resolution `health`/`calibrate`/`communities`
already use. One visibility change accompanies this:
**`pub(crate) mod sources;`**, so responses deserialize into the real
`SearchPlan`/`PassageLanes`/`LaneEvidence` types instead of hand-copied
mirrors that would drift silently the first time a lane is added — the
exact precedent `pub(crate) mod communities;` (src/api.rs:26-28) already
sets for the same reason. `src/lib.rs`'s public surface stays `context`
alone (src/lib.rs:9-12) — no wider export is made for this. If no server
is reachable, #260 emits every non-lane comparison and sets lane fields
to `null` with a stated reason rather than failing the run — a documented
degradation, not a hard requirement, matching #189's own "任意" framing
of this tier. ADR 0002 §12.1's #248 already tracks unifying `calibrate`'s
and `communities`'s near-identical hand-written HTTP clients; #260 and
#215 should land on that one client rather than adding a third and
fourth.

## 12. Consequences

### 12.1 Follow-up issues

| Issue | Title | Depends on | Implements |
|---|---|---|---|
| [#262](https://github.com/t0k0sh1/taguru/issues/262) | extract: diagnostics sidecar に chunk/document 由来レコードを追加する | none | §7 |
| [#256](https://github.com/t0k0sh1/taguru/issues/256) | benchmark: 複数モデル・複数 run の実行ハーネスと manifest.json / runs/\*.jsonl | #255, #262 | §5, §6, §8, §9.1, §9.2, §10 |
| [#257](https://github.com/t0k0sh1/taguru/issues/257) | benchmark: 測定値集計(measurements.json/csv)と指標定義の文書化 | #256 | §9.3, §10 |
| [#258](https://github.com/t0k0sh1/taguru/issues/258) | benchmark: association 同一性マッチングと run 間安定性指標 | #256 | §9.4 |
| [#259](https://github.com/t0k0sh1/taguru/issues/259) | benchmark: モデル間 paired diff(differences.jsonl)と元文書・節・原文への参照 | #258, #262 | §7, §9.4 |
| [#260](https://github.com/t0k0sh1/taguru/issues/260) | benchmark: モデル別コーパスへの同一質問セット実行と検索影響の比較(任意) | #256 | §11 |
| [#261](https://github.com/t0k0sh1/taguru/issues/261) | benchmark: 自己完結の静的 report.html(任意) | #257, #259 | §9.3 |

Explicitly and deliberately **not filed** as follow-ups, with the reason
recorded here so a future PR does not reopen these without first reading
why they were set aside: an `extract.rs` configuration-seam refactor for
in-process execution (§3 A — the subprocess model was chosen, so the seam
#189's own split note anticipated is not needed; #262 above is a
narrower, different issue — diagnostics provenance, not a config seam);
`taguru extract --url`/`--model` flags (§3 C — owned by ADR 0002 §12.1's
#248); a byte-range-carrying chunker refactor (§7 — the paragraph index
is sufficient and `paragraph::split` is already re-derivable offline); a
wider `src/lib.rs` export (§11 — `pub(crate) mod sources;` is enough);
splitting `AttemptRecord`'s `stop_malformed` into separate syntax/
validation states (§9.2 — an ADR 0001 §7 vocabulary question, out of this
ADR's scope, carried forward as a `caveat` string in #257's metric
definitions instead).

### 12.2 Migration and API compatibility

- `taguru extract`'s existing flags, environment variables, and
  `.extract-manifest.json`/checkpoint formats are unchanged by this ADR.
  `--diagnostics-out` gains two additive record kinds (#262); any
  existing consumer filtering on `kind == "attempt"` is unaffected, and
  `AttemptRecord`'s own fields do not change, so the Python SDK parity
  test needs no update.
- `taguru benchmark` is an entirely new top-level verb; no existing verb,
  flag, or environment variable changes meaning as a result of this ADR.
- All four benchmark artifacts start at version 1; there is no prior
  format to migrate from.

### 12.3 Documentation impact

- `docs/extract.html`'s `--diagnostics-out` section gains a note that the
  sidecar is multi-`kind` once #262 lands; #262 owns that edit, not this
  ADR's own PR.
- A new `docs/benchmark.html` (or an addition to an existing operations
  page) documenting `taguru benchmark extract`/`compare`,
  `models.json`, and the four artifact schemas is #256's/#257's
  responsibility, written against the schemas this ADR fixes — not
  written speculatively here before the verb exists.
- `CONTRIBUTING.md`'s "Design decisions" section needs no edit; ADR 0003
  follows the same `adr/000N-*.md` convention it already describes.

## Appendix: requirement traceability

| #255 requirement | Section |
|---|---|
| 1. 実行方式(サブプロセス vs in-process) | §3 A–C, §4 |
| 2. 公平な run の分離(出力ディレクトリ分離規則) | §6 |
| 3. 成果物スキーマの版付け | §9, §10 |
| 3. `runs/*.jsonl` のチャンク由来情報を最初から含める | §7, §9.2 |
| 3. モデル設定ファイルの形式 | §8 |
| 4. #215 との境界 | §11 |
| 非目標: ハーネス実装・集計実装そのもの | out of scope — carried to #256/#257 |
| 非目標: 総合スコア・モデル順位付けの設計 | §9.3's structural ban; not attempted here |
