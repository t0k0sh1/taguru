# 0009. Optional per-context schema: entity types and relation domain/range

- **Status**: Accepted
- **Date**: 2026-08-03
- **Issue**: #378
- **Related**: #218, #182, #187, #192, #199, #217, ADR 0005 §4, ADR 0006 §1,
  ADR 0007
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

The entity-type model, the schema document's shape and persistence,
`off`/`warn`/`strict` validation semantics and the batch-ordering guarantee,
the error contract shared by every write path, relation-constraint staging,
producer guidance for `taguru extract` and both LangChain SDK ingesters,
the read-side minimum, and backward compatibility for #218's optional
per-context schema/ontology layer — the open design questions #378 lists
before the ten implementation sub-issues (§Appendix) can start without each
guessing independently and risking a later, backward-incompatible rewrite.

No code changes ship with this ADR. It is design and decision only — the
sub-issues in §15 are the implementation, exactly as #345/ADR 0007 preceded
#346–#354 and #302/ADR 0006 preceded #303–#308.

Out of scope, filed to the follow-up §14 names rather than answered here:

- `cardinality` (one / optional-one / many) — §9.4.
- `inverse` and `symmetric` relations, and any transitive-closure or
  RDF/OWL-style reasoning — §14; associations never *infer* new
  associations in this design, only validate the ones a source asserts.
- `deprecated` relation labels with a replacement — a producer-guidance and
  audit signal, not an enforcement rule; §14.
- Bulk retype / corpus-wide relation-rename migration tooling — the audit
  (§10) reports candidates, never applies them; a dedicated migration
  feature is its own issue, as #218 itself allows.
- Using entity types to score or rank retrieval candidates — §12.4.
- A schema registry shared across contexts — each context owns its own
  document, exactly as each context owns its own `dice_floor`.

## 2. Context

### 2.1 What exists today, and what doesn't

Taguru's association graph (`src/context.rs:576` `Context`) accepts
`(subject, label, object, weight, source)` with no notion of what kind of
thing a concept is or what types a relation label connects. Three
mechanisms already fight adjacent symptoms without fixing the underlying
gap:

- **`resolve`/alias resolution** (`src/context/alias.rs`) collapses spelling
  variants of the *same* concept or label onto one canonical name — it says
  nothing about what *kind* a concept is or what a relation *expects*.
- **`audit_vocabulary`** (`src/api/vocabulary.rs:116`) surfaces lexical/
  semantic twin *candidates* — "candidates for review, not verdicts"
  (`:37-40`) — for a human to resolve via alias. It has no concept of
  domain/range and cannot distinguish "these are spelling variants" from
  "these are two different types that happen to share a label."
- **`DriftAudit`** (`src/api/vocabulary.rs:187`, `audit_drift` at `:196`)
  reports unsourced edges and dead aliases — again no type or
  relation-shape information.

None of these can express or check the issue's own motivating examples: a
`企業` and a `人物` both taking `設立年`, `杜氏 --所属--> 酒蔵` reversing
direction across producers, or `所在地`/`本社所在地`/`所在` proliferating as
unrelated labels rather than one relation with alias variants.

### 2.2 The write-path shape a schema check must fit into

`ingest.rs`'s module doc states Taguru's central write invariant: "One file
states one source's COMPLETE truth: applying it first retracts the source,
then adds the file's facts, so re-importing a file is idempotent"
(`src/ingest.rs:1644-1648`). `apply_batch` (`src/ingest.rs:2308`) is
two-phase: everything up to and including `predicted_alias_rejection`
(`:2243`) is pure, read-only, and provably precedes any write —
`ApplyRefusal::wrote_anything()` (`:2145`) certifies exactly two variants,
`NoContext` and `Rejected`, as preceding every durable mutation. Everything
after the import marker opens (`:2299-2307`) is *not* cross-store atomic;
the four mutations (retract, store passages, add associations, add aliases)
are each individually durable, and the module doc calls this "deliberately
not attempted" rather than a gap. `preview_batch` (`:2520`) is
`apply_batch`'s read-only twin for `?dry_run=true`, sharing every pure
check so the two "can never disagree" (`:2247-2248`).

`predicted_alias_rejection` is the load-bearing precedent this ADR reuses
structurally: a pure, stateful, pre-write check against the live context,
shared verbatim between `apply_batch` and `preview_batch`. A schema check
is the same shape, and §7 places it right beside it.

The HTTP associations path (`POST /contexts/{name}/associations`,
`src/api/associations.rs:205`) is different: `interpret_associations`
(`:167`) is a *pure* function with no `AppState`, and its handler's own
partial-write arm (`:263`) is explicitly non-atomic — "items before the
failing one are applied." §8 accounts for this asymmetry.

### 2.3 The error contract already in place (#182's machinery)

`ErrorCode` (`src/api.rs:169`) is a closed, stable enum whose own doc
states "renaming or repurposing a variant is a breaking change... like a
response-shape change." `Issue` (`src/api.rs:308`, constructors `:316-390`)
is, in its own words, "the machine-actionable twin of one clause in the
accompanying prose `error`" — four fields: `path`, `kind`, `expected`,
`actual`. `RefusalDetail` (`:452`) bundles `issues`, `integrity`,
`durable_batches`, `retryable_after_correction`. `alias_rejection_issue`
(`src/api/import.rs:118-140`) is the precedent for discriminating a
structured refusal into `Issue`s without inventing a new `ErrorCode`
variant: it maps two distinct `AliasError` arms onto two *existing* `kind`
tokens (`unknown_reference`, `conflict`). §8 follows this precedent rather
than adding a variant.

MCP needs essentially no bespoke work: `route_tool` (`src/mcp/route.rs:24`)
is a pure mapping onto the HTTP routes, and `ToolError.structured`
(`src/mcp/protocol.rs:164`) forwards the `issues` array verbatim to any MCP
host. Only `tool_definitions()` prose (`src/mcp/schema.rs:56`) needs new
words, and new tool entries for the schema management routes (§6.4).

### 2.4 Two precedents for where per-context state lives

- **Sidecar config**: `dice_floor` (`src/context.rs:646`, doc: "This is
  config, not knowledge, so it is NOT part of the persistent image") lives
  in `ContextMeta` (`src/registry.rs:108-126`), persisted in `MetaFile`
  (`:340`), and is re-applied on every load (`:2603-2605`).
- **Standalone per-context file**: `GroupRecord` (`src/groups.rs:52`)
  persists as its own `{stem}.group` file, with atomic write-then-rename
  (`write_group`, `:279-281`), a rename marker (`:271-273`), and a
  `{stem}.group.corrupt` quarantine for a file that reads but does not
  parse (`:303-309`). Its nesting guard, `validate_nesting`
  (`:88-92`)/`nesting_depths` (`:99-108`), is one memoized `O(groups +
  edges)` walk returning a deterministic `NestingViolation::Cycle`/
  `TooDeep`, capped at `MAX_GROUP_DEPTH = 3` (`:34`) with the stated
  rationale "deep taxonomies are filing, not addressing."

`context_files(stem) -> [String; 9]` (`src/registry.rs:2829`) is the single
source of truth every consumer (delete, boot sweep, `move_context_files`
`:2859`, `hydrate::FamilySig::of` `src/hydrate.rs:582`) reads — "a file kind
added there cannot silently miss this list." Replication ships any
non-skipped file automatically (`classify`, `src/ship.rs:616`), but
hydration only sees files this array names.

### 2.5 Retrieval-cache invalidation is lane-based, not free-form

`ContextRevision { graph, passages, config }` (`src/registry.rs:285`) feeds
`op_lanes(op, revision) -> [u64; 2]` (`src/registry/retrieval_cache.rs:216`)
— every retrieval op gets exactly two lanes, and the pairing is
load-bearing: `Recall | Query → [graph, passages]`. `cache_identity`
(`src/registry.rs:690`, re-minted by `invalidate_cache_identity` `:737`) is
the coarser instrument for "content switched lineage under an unmoved
revision" — already used by compaction (`src/registry/context_io.rs:
243-253`) for exactly this class of problem.

### 2.6 The producer side (#199's corrective retry, mirrored in three languages)

`system_prompt` (`src/extract.rs:3698`) already emits a vocabulary block
(`:3742-3752`, "reuse these exact spellings... instead of coining a
synonym"), capped at `VOCABULARY_CAP = 200` (`:144`). `cross_output_issues`
(`:4515`) is the existing precedent for a check that needs the *union*
across every model output before judging any one of them: it collects
`concept_names`/`label_names` across all outputs first, then judges each
output against the completed set, grouped per-output-index "so the caller
can address a single targeted corrective turn per offending output"
(`:4511-4514`). `corrective_validation_message` (`:3615`) is declared "the
cross-language corrective-text baseline #180/#181 mirror byte for byte."
`model_output_json_schema` (`:4387`) documents, at `:4368-4382`, exactly
what it deliberately does *not* encode (byte caps, weight bounds, per-
document paragraph bounds, cross-item rules) — §11 extends that list rather
than the schema.

Both LangChain SDKs (`sdk/python-langchain/src/taguru_langchain/_extract.py`,
`sdk/typescript-langchain/src/extract.ts`) mirror these functions byte for
byte, per ADR 0001 §7. `TaguruIngester._fetch_vocabulary`
(`sdk/python-langchain/src/taguru_langchain/ingest.py:1042`) already pulls
the context's live label vocabulary into the prompt with a best-effort
`except NotFoundError: return []` posture — the precedent §11.4 extends for
schema vocabulary.

## 3. Options considered — where a type assertion lives

Three placements, weighed against the properties that matter: survival
through compaction, WAL replay, export/import, replication, source
attribution, and whether the assertion is knowledge (retractable, sourced)
or config.

| | (a) ordinary association, reserved label | (b) new fixed-width table, image v7 | (c) sidecar metadata |
|---|---|---|---|
| **Compaction** | Free — `Context::compacted` (`src/context/write.rs:485`) iterates `self.query_any(&[], &[], &[])` (`:490`), which already yields every live edge. | Silently dropped unless taught: `compacted()` starts from `Context::default()` and re-carries only edges + aliases; the caller separately re-applies `applied_seq`/`dice_floor` (`src/registry/context_io.rs:238-240`). Two edits, both silent-loss-if-missed. | N/A — never in the image. |
| **WAL replay** | Free — `WalOp::Associate` (`src/wal.rs:38`) already carries it; `apply_op` (`src/registry.rs:2681`) already applies it. No new variant, so the downgrade hazard the WAL doc states (`src/wal.rs:47-51`: "a DOWNGRADED binary reading a log that holds one of the newer ops refuses the boot as corruption") is never triggered. | Needs a new `WalOp` variant, triggering exactly that downgrade hazard, plus a `grows()` classification (`:75`). | Config writes bypass the WAL; breaks the crash-consistency story `applied_seq` (`src/context.rs:655`) exists for. |
| **Export/import** | Free — an ordinary `AssociationLine` round-trips exactly today. | New stream record kind, new version stamp, new dispatch arm. | Same, plus the schema-record path. |
| **Replication/hydration** | Free — already in `.ctx`, already index 0 of `context_files`. | Same as (a) — also in `.ctx`. | A new file, duplicating §5's problem. |
| **Retrieval-cache lane** | Free — a graph write bumps the `graph` lane `op_lanes` already gives `Recall`/`Query`. | Same, if it bumps `graph`. | Needs a lane decision per assertion — untenable. |
| **Source attribution** | Free — `associate_from` writes an `AttributionRecord`, so weight/count/paragraph/source all attach identically to any other fact. | Needs its own attribution chain or loses provenance. | Loses provenance entirely. |
| **Knowledge vs. config** | Knowledge — retractable by `retract_source`, exactly the `applied_seq` side of the split, not the `dice_floor` side. | Also knowledge — this axis doesn't separate (a) from (b). | Fails: `ContextMeta` is parsed wholesale at boot for every context and rewritten on every config bump; unbounded per-concept data there taxes every context, not just typed ones. |

**Decision: (a).** Type assertions are ordinary associations under one
reserved label. This is the only option where every survival property holds
*by construction*, and the only one under which types are searchable
knowledge from day one: `describe`, `query_any`, `retract_source`, and the
drift audit all see them the moment the label ships, with no per-surface
plumbing.

**The cost this decision accepts, and how §6 bounds it.** Because the
object of a type assertion is an ordinary concept, a type name (`Brewery`,
`Organization`) becomes a concept itself: it can appear in `resolve`
candidates, become a high-degree hub on its incoming chain
(`ConceptRecord.incoming_count`, `src/context.rs:388`), and register as a
spelling-drift candidate in `audit_vocabulary`. §6.3 fixes three exclusions
as part of this Decision, not left to an implementer's judgment.

**Rejected: (b), a new fixed-width `Context` table, image v7.** It buys
exactly one thing — the type namespace is structurally separate from the
concept namespace — and pays for it with `IMAGE_SECTIONS` 9→10
(`src/context/image.rs:57`), a new compile-time layout assert (`:12-26`), a
version-conditional load beside the three that exist (`:246`, `:351`,
`:363`), a legacy `to_bytes_as_version` path (`:138`), a new `WalOp` with
its downgrade-refuses-boot consequence, a new export/import record, *and*
the two compaction edits whose omission is silent data loss. **Trigger for
a successor ADR**: if type assertions are ever measured to materially
distort `resolve` ranking or community detection, or if per-assertion
metadata beyond `(source, weight, count, paragraph)` becomes necessary,
(b) is the right redo — and (a)'s data migrates into it by an ordinary
`compacted()`-shaped rebuild.

**Rejected: (c), sidecar metadata.** Fails on volume, on provenance, and on
the knowledge/config split `dice_floor`'s own doc states outright.

## 4. Decision

1. **Entity types are multiple per concept**, a set (possibly empty),
   asserted as ordinary associations under the reserved label
   `schema:type` (§3, §6). `IMAGE_VERSION` stays `6`; no new `WalOp`
   variant; no new `Context` table.
2. **`is_a` is a shallow DAG declared only inside the schema document**,
   never as graph edges — multiple parents allowed, cycles refused at
   `PUT` time, depth capped, ancestor closure precomputed once per
   document install (§6.2).
3. **The schema document is a new standalone per-context file**,
   `{stem}.schema.json`, following `GroupRecord`'s exact shape —
   `context_files` grows to `[String; 10]` (§5).
4. **Its revision rides the existing `config` lane**, plus a
   `cache_identity` re-mint on every mutation; no fourth
   `ContextRevision` lane (§5.2).
5. **Validation modes are `off` (default) / `warn` / `strict`**, stored
   inside the document, changeable at any time including while
   violations exist (§7.1).
6. **The ordering guarantee** — a batch that both assigns a type and uses
   an association requiring it validates correctly regardless of line
   order — is met by validating the *union* of live state and the
   batch's own type assertions, computed in full before any write (§7.2).
7. **The error contract adds no new `ErrorCode` variant**, reusing
   `InvalidArgument`, and exactly two new `Issue.kind` tokens: `domain`
   and `range` (§8).
8. **Relation constraints ship as `domain`/`range` only** in this design;
   cardinality, inverse/symmetric, and deprecated-relation guidance are
   named follow-up, not this ADR's scope (§9).
9. **Relation canonicalization is not a schema feature** — it continues
   to be `label_aliases`' job; the schema document names relations by
   canonical spelling only (§9.5).
10. **Producer guidance is additive prompt text and a new cross-output
    check**, never a change to the JSON-schema-constrained output shape
    (§11).
11. **The read-side minimum** is `describe`/`resolve` returning types and
    `query` gaining a type filter — never type-weighted scoring (§12).
12. **Nothing here changes behavior for a context with no schema file.**
    Every old image, old batch, and existing API caller behaves exactly
    as today (§13).

## 5. Schema document contract (for S1, S2)

### 5.1 Persistence: a standalone file, not the sidecar

**Decision: `{stem}.schema.json`, a new file, copying `GroupRecord`'s
pattern field for field, with one deliberate divergence** — atomic
write-then-rename (`write_atomic`, the same primitive `write_group` uses)
and a rename marker analogous to `group_renaming_marker_path`, but **not**
`scan_groups`' "malformed keeps its name and loses its content" recovery
(`src/groups.rs:292-309`).

**Why the divergence: an empty schema is not a safe default to fail
into.** For a group, replacing an unparseable file with a fresh empty
record and continuing the boot is safe — the consequence is "no
membership," a routing-only concern. For a schema, the equivalent
"replace with a fresh empty record" is indistinguishable from `mode: off`
(§5.3's document shape), which **silently disables `strict` enforcement**
for a context whose operator explicitly turned it on — the exact failure
mode §5.3 already refuses to allow for a version mismatch, and this ADR
will not allow it for a parse failure either. A schema file present but
unparseable therefore **refuses the boot** — the same posture already
used for an unreadable file, extended to cover a malformed one too — with
the mangled bytes quarantined to `{stem}.schema.corrupt` for hand recovery,
exactly as `scan_groups` already does before the refusal. There is no
"boot with an empty schema and log a warning" path: a context that had a
schema comes back with that schema, or it does not come back.

**Corollary for `move_context_files`' best-effort straggler tolerance**
(below, this section). `move_context_files` (`src/registry.rs:2859`) already
returns the first sidecar-move error to the caller and keeps the rename
marker in place for retry rather than silently dropping a straggler — this
was true before this ADR and needs no new mechanism. What this Decision
adds is only that the schema file must be covered by that same guarantee
like every other sidecar: a schema move failure is reported and retried
until it completes, never treated as "file absent, so `off`."

**Rejected: a fifth `ContextMeta` field.** `MetaFile` (`src/registry.rs:
340`) is parsed at boot for every context to seed the directory and
rewritten on every `bump_config_revision` (full `write_meta`, fsync +
rename) and every embedding refresh. A schema with hundreds of types and
relations riding in that file taxes every context's boot and every config
bump, whether or not that context uses a schema, and leaks into every
`GET /contexts` directory row via `DirectoryEntry` (`src/registry.rs:358`).

**Rejected: an image section.** Named in §3; the category argument is that
`image_body` (`src/context/image.rs:84-113`) writes fixed-width record
tables plus one string arena — a nested JSON policy document has no
fixed-width form, and the schema is not knowledge asserted by a source, not
retractable by `retract_source`.

**Placement in `context_files` and why it goes last.** `context_files`
(`src/registry.rs:2829`) grows from `[String; 9]` to `[String; 10]`, with
the schema file **at index 9, the end**. `move_context_files` (`:2859`)
treats index 0 (`.ctx`) as the pivot everything else follows — a lagging
schema file must never block a context rename, so it goes where a
straggler is already tolerated as best-effort. "Best-effort" here means
what it already means for every other sidecar: the rename keeps trying
and the caller learns of a straggler through the returned error and the
retained rename marker, **not** that a straggler is silently treated as
absent — the corollary above applies at both ends, source and
destination stem. Replication needs zero changes: `classify`
(`src/ship.rs:616`) already ships any file that is neither the lock, the
replication record, nor a `*.tmp{N}` stager, as `Published`.

### 5.2 Revision and cache invalidation

**Decision: reuse the `config` lane; additionally re-mint `cache_identity`
on every schema mutation. No fourth `ContextRevision` lane.**

`op_lanes` (`src/registry/retrieval_cache.rs:216-234`) returns `[u64; 2]`
for every op — the pairing is load-bearing (`Recall | Query`'s `passages`
half exists because marker resolution runs through the passage store). A
fourth lane could only reach `Recall`/`Query` by widening the array to
`[u64; 3]` everywhere or displacing `passages` — and doing neither would be
worse than doing nothing: a schema change that alters what `query` returns
(§12.3's type filter) would keep serving stale cache entries while looking
handled.

`cache_identity` (`src/registry.rs:690`) is the correct instrument instead:
it already exists for "content switched lineage under an unmoved revision"
and is already used this way by compaction (`src/registry/context_io.rs:
243-253`, "without a fresh identity, a retrieval-cache key minted before
this compact stays valid and keeps answering with content this context no
longer holds"). `PUT /contexts/{name}/schema` therefore does, under the
entry write lock, exactly what `bump_config_revision` (`:2274-2295`)
already does for `dice_floor`, plus `invalidate_cache_identity` (`:737`).

**Write order across the two durable artifacts, and why it is not free.**
`dice_floor`'s content and its `config_revision` live in the *same*
`MetaFile`, written by the *same* `write_meta` call — genuinely atomic, no
crash window possible between them. §5.1 deliberately puts the schema's
content in a *separate* file from its revision (which still lives in
`MetaFile`), so that same atomicity does not come for free here and must
be designed in: **the durable write order is revision-then-content** —
`bump_config_revision`'s `write_meta` (`:2274-2295`) lands *before*
`{stem}.schema.json`'s `write_atomic`, both still under the one entry
write lock so no in-process reader observes a moment where they disagree
(`bump_config_revision`'s own doc: "Called AFTER the publish... so a
reader observing the new value always sees the new content" — mirrored
here as *before* the new content lands, not after, since content ordering
is reversed relative to that caller). A crash between the two durable
writes therefore always fails toward **extra**, never **missed**,
invalidation: `MetaFile.revision.config` already advanced while
`{stem}.schema.json` still holds its pre-`PUT` bytes, so a replica or a
restarted process that notices the new revision fetches and finds nothing
new — a wasted refetch, not a stale enforcement. The mirror order
(content first, revision second) is the one this ADR refuses: a crash in
that window would leave a durably-changed schema behind a revision number
that never advanced, which is exactly the "changed but nobody was told"
failure `bump_config_revision`'s own "a failed sidecar write... heals by
the next flush" contract does not cover for a genuine process crash (that
healing pattern deliberately tolerates the reverse: the revision may lag
the content that's identically stored right there in the SAME file — it
never tolerates a *separate* content file changing durably underneath an
unmoved revision, which is precisely what §5.1's split introduces and
this ordering closes). A `PUT` that never returns 200 to its caller is,
under this order, always safe to retry — the schema document is replaced
wholesale, not by delta, so retrying is idempotent regardless of which
side of the crash window the previous attempt reached.

**The write order alone still leaves one window open, and this closes
it: what a crash-then-clean-restart boots into.** The write order above
guarantees a crash mid-`PUT` never advances the revision past
unpublished content *while the process keeps running* — a live reader
never observes the mismatch. But a genuine process crash followed by a
restart is a different case: the restarted process does not "notice"
anything is stale, it simply loads whatever bytes are durably on disk.
If the crash landed between `write_meta` and `write_atomic`,
`{stem}.schema.json` is still perfectly *valid* JSON — just the
**previous** `PUT`'s content, not the one the operator's request
intended — so §5.1's "refuses the boot" only fires for unreadable or
malformed bytes, and this file is neither. Left unaddressed, the context
would come back up enforcing `strict` under the *old* constraints while
`MetaFile.config_revision` already claims the *new* one — content and
revision durably disagree, and nothing before this paragraph would
detect it.

**Fix: `MetaFile` carries a schema content digest, written atomically
with the revision it describes — never separately.** `ContextMeta`
gains `schema_digest: Option<String>` (`sha256_hex` of
`{stem}.schema.json`'s bytes, `None` for a schema-free context), set in
the *same* `write_meta` call that bumps `config_revision` (§5.1's
revision-then-content order: the digest recorded is the *target*
content's digest, computed before `write_atomic` runs, exactly as the
revision itself is bumped before that same write). At every point this
ADR already reads the schema file — boot, and the read path
`schema_issues`/`predicted_schema_rejection` share (§7.2) — the file's
actual digest is checked against `MetaFile.schema_digest`. **A mismatch,
in either direction (stale content under a newer digest, or a recorded
digest with no corresponding file), fails exactly like §5.1's unreadable
or malformed case: refuses the boot**, rather than serving a `strict`
context whose enforced rules do not match what its own revision claims
to be enforcing. This is the "fail-closed on undetectable interrupted
update" contract directly: the digest is the one artifact §5.1's write
order does not otherwise provide a way to cross-check, and it costs one
extra field in a file already rewritten on every config bump, not a new
persistence mechanism (no WAL record, no manifest entry).

**Cost accepted**: re-minting invalidates every cached retrieval for the
context, not only graph-lane ones, and is process-local — but a schema
`PUT` is a rare operator action, not a per-write hot path, so coarse
invalidation is the right trade. **Trigger for a successor ADR**: if schema
mutation ever becomes high-frequency, the coarse cost earns a real fourth
lane.

### 5.3 Document shape and its own version stamp

**Decision: `SCHEMA_VERSION: u64 = 1`, independent of `BATCH_VERSION`
(`src/ingest.rs:134`), `GROUP_VERSION` (`:139`), and `IMAGE_VERSION`
(`src/context/image.rs:51`)** — `GROUP_VERSION`'s own doc gives the
justification verbatim: separate "so either shape can rev without dragging
the other along." Surfaced in `version_facts()` (`src/api.rs:141`) as
`schema_formats: [SCHEMA_VERSION]` beside `batch_formats`.

```jsonc
{
  "schema": 1,                            // SCHEMA_VERSION — an unread value is a hard refusal
  "mode": "off",                          // off | warn | strict   (§7.1)
  "closed_labels": false,                 // §6.4
  "types":     { "Brewery": {"is_a": ["Organization"]}, "Organization": {}, "Person": {} },
  "relations": { "杜氏": {"domain": ["Brewery"], "range": ["Person"]} }
}
```

**Version handling is symmetric, on purpose — this ADR does *not* copy
`GroupRecord`'s tolerant-default convention.** `GroupRecord` uses
`#[serde(default)]` at rest so an older binary can silently load a file a
newer one wrote with an extra field (`src/groups.rs:51-60`, "the
struct-level `serde(default)` keeps every pre-nesting group file loading
unchanged") — safe there because a dropped `groups` field degrades to "no
nesting," a routing-only consequence. That degradation is exactly the
failure mode §5.1 already refuses to allow for a schema: an older binary
silently dropping a field it doesn't recognize is indistinguishable from
under-enforcing `strict`, whether the field silently disappears because
the binary is older or because the file failed to parse at all. So both
directions get the same hard-refusal treatment: **`deny_unknown_fields`
applies to the persisted struct exactly as it does to the wire** (the
`PUT` body and the `taguru_schema` export record, §13.3) — an on-disk
file naming a field this binary does not know is a hard refusal identical
in wording to an unread `schema` stamp, not a silent drop. Consequently
**`SCHEMA_VERSION` bumps on every shape change, additive or breaking** —
unlike `BATCH_VERSION`/`GROUP_VERSION`, which only bump for a
breaking change and rely on `serde(default)` tolerance for additive ones.
This is a stricter discipline than either of those two deliberately
accepts, and it is deliberate here for the same reason `strict` itself
exists: a validation feature must fail loud, never quiet, when it cannot
prove it is applying the constraints it was told to.

**Caps**, each a named constant sized in `MAX_GROUP_MEMBERS`'s style
(`src/groups.rs:44`): `MAX_SCHEMA_TYPES`, `MAX_SCHEMA_RELATIONS`,
`MAX_RELATION_TYPES` (per `domain`/`range` list — load-bearing for §8's
bounded `expected` string), `MAX_TYPE_DEPTH` (§6.2), plus the existing
`MAX_NAME_BYTES` for every name.

## 6. Type assertions and the reserved label (for S3)

### 6.1 Cardinality of types, and untyped concepts

**A concept carries a set of types, possibly empty (multiple, not
exactly-one).** Enforcing "exactly one" would require reading the
concept's existing outgoing chain filtered to the reserved label on every
write — the same shape §9.4 defers for cardinality, with the same
retract-then-apply ambiguity — and it makes the issue's own worked example
(`Brewery: {is_a: ["Organization"]}` — a brewery is *both*) unrepresentable
without inference, which §14 puts out of scope.

**Untyped concepts are legal in every mode, including `strict`.** `strict`
constrains *associations*, not concepts: it refuses one only when (i) its
label has a relation definition with non-empty `domain`/`range`, **and**
(ii) the subject or object has a non-empty type set, **and** (iii) that set
(after `is_a` closure) is disjoint from the constraint. Requiring every
mentioned concept to be typed first would make `strict` unreachable on any
real corpus and would turn every flip of the mode into an instant
whole-graph violation, contradicting §7.1's "changeable while violating."
The untyped population is the **audit's** business (§10), reported as
candidates, never enforced.

**Undefined relation labels are not a violation by default**, gated by one
explicit document flag, `closed_labels: bool` (default `false`). When set,
an association whose label has no relation definition reuses the existing
`Issue::unknown_reference` kind (`src/api.rs:382-390`) rather than adding a
third token.

### 6.2 `is_a` hierarchy

**A shallow DAG, declared inside the schema document only — never as graph
edges.** Multiple parents allowed; a cycle is refused at `PUT` time; depth
is capped at `MAX_TYPE_DEPTH = 8`; the ancestor closure is precomputed once
when the document installs.

This structurally copies `GroupRecord.groups`' nesting: `validate_nesting`
(`src/groups.rs:88-92`)/`nesting_depths` (`:99-108`) is one memoized
`O(groups + edges)` walk naming the offending group deterministically on
`Cycle`/`TooDeep`. The one divergence is the depth cap: `MAX_GROUP_DEPTH =
3` is justified there by "deep taxonomies are filing, not addressing,"
which does not hold for entity types — `Brewery ⊂ Manufacturer ⊂ Company ⊂
Organization ⊂ Agent` is a legitimate five-deep chain. Because the closure
is computed once at document install (the document is immutable between
`PUT`s), §7.2's per-association check is a set-intersection against a
precomputed set, never a live walk — the depth cap bounds `PUT`-time work
only, never the write path.

**Rejected: `is_a` as ordinary graph edges under a second reserved label.**
It would put the hierarchy where `activate`/`explore` traverse it, and
make the closure a live graph walk on the write path — the one thing that
must stay fast. **Rejected: no hierarchy.** Every `domain` would have to
enumerate every subtype by hand, which the issue's own example already
rules out.

**A `schema:type` object naming a type that is not a key in `types` is
always accepted, never a violation, in every mode.** This extends the
same philosophy §6.1 already states for untyped concepts — a schema is
meant to be adoptable incrementally on a live corpus, and an
undeclared-but-asserted type name is exactly that kind of incremental
fact: a source calling something `"Distillery"` before the schema
document formally lists `Distillery` should not be refused for saying so.
Concretely, for an undeclared type name: `is_a` closure lookup treats it
as its own singleton ancestor set (absent from the precomputed map,
§7.2 step 5's set union simply has nothing to add); `Issue.kind`
(§8.1) never fires for it, in any mode, `closed_labels` or not —
`closed_labels` (§6.4) governs *relation* label vocabulary only and does
not extend to type names, on purpose, to avoid overloading one boolean
with two different closed-world questions. What `strict` still enforces
is the `domain`/`range` disjointness test (§7.2 step 6) against whatever
type set the concept in fact carries, declared or not — an undeclared
type behaves exactly like a declared one there, just without a hierarchy
above it. §10's audit gains one more, always-on informational line
(not gated by `closed_labels`, since this is not an enforcement question):
type names asserted but absent from `types` — a signal for a schema
author who wants to know what vocabulary is actually in use, reported
the same "candidate for review" way as every other audit finding.

### 6.3 The reserved label, and the three exclusions its cost requires

**Type assertions are ordinary associations under the reserved label
`schema:type`**: `{"subject": "青嶺酒造", "label": "schema:type", "object":
"Brewery", "source": "kura.md", "paragraph": 3}`. Because `associate_from`
writes an `AttributionRecord` for every edge, this carries source, weight,
count, and paragraph identically to any other fact, with no new mechanism
— `retract_source` withdraws a source's type claims with its facts, and
the drift audit's unsourced-edge sweep already reports an unsourced type
the same way it reports any other unsourced edge.

**Reserved-name grammar, following the existing precedent exactly**:
`EMPTY_SOURCE = "export:unsourced"` (`src/export.rs:37-39`) is a
`namespace:value` id whose collision with a real user id is a hard refusal
naming a rename (`src/export.rs:315`, `:353`). `schema:type` gets the same
three guards, fixed as part of this Decision:

1. An association line whose label is `schema:type` in a context with **no
   installed schema document** is an ordinary association and nothing
   more — no special meaning imposed on a context that never opted in.
   This is Decision #12's byte-identical guarantee applied to the one
   label an operator could otherwise coincidentally already be using.
2. **No path may resolve any label to `schema:type` once a schema
   exists** — this is one guard applied at every place §7.2 step 3
   already says a label gets resolved, not `add_label_alias` alone:
   - `add_label_alias` refuses to *create* a persisted alias whose
     canonical target is `schema:type`.
   - A batch's own inline `batch.labels` declaration (§7.2 step 3's
     other half of the union) is checked by the same
     `schema_issues`/`predicted_schema_rejection` pass that checks
     everything else in that step — a batch-local alias mapping some
     spelling to `schema:type` is refused the same way a persisted one
     is, not silently accepted because it never touched
     `add_label_alias`'s API.
   - **`PUT /contexts/{name}/schema`, on the transition from no schema
     (or `off`) to an installed one, refuses if any *already-persisted*
     `label_alias` resolves to `schema:type`.** An alias created before
     any schema existed cannot have been refused by the first two
     bullets — there was nothing to refuse against — so the install
     itself is where a coincidental pre-existing alias meeting the
     reserved label is caught, symmetric with guard 3's own refusal of
     a `relations` entry named `schema:type` at the same `PUT`.
   Whichever path a violation is caught on, the message names the
   conflicting alias and instructs a rename, mirroring
   `EMPTY_SOURCE`'s own collision wording (`src/export.rs:315`, `:353`).
3. `PUT /schema` refuses a relation definition named `schema:type`.

**One gate for guard 1 and the three exclusions below, stated once so
they cannot drift apart: "installed schema document," not "`mode !=
off`."** Whether a schema *validates* writes is §7.1/§7.2's separate,
mode-gated question — `mode == off` already short-circuits the entire
check at §7.2 step 1, for every label, not only `schema:type`. Whether
`schema:type` is read, audited, or traversed *specially at all* is this
gate, and it does not borrow mode's threshold: an operator who installs a
schema but leaves it in `off` while drafting types has already committed
to the reserved label meaning something, and `off` only means "don't
enforce it yet," not "pretend the label is ordinary." Conflating the two
gates is exactly what would make guard 1 above disagree with §12's
read-side population of `types` below (both must key off the same
condition) — this paragraph is the single place that condition is
defined; every consumer below cites it rather than restating its own
threshold.

**The three exclusions this Decision fixes, so type-name concepts do not
quietly distort unrelated features whenever a schema document exists for
the context** (§12 extends this same gate to `describe`/`resolve`):

1. `activate`/`explore` do not traverse `schema:type` by default.
2. `schema:type` is excluded from the vocabulary block `system_prompt`
   emits (`src/extract.rs:3742-3752`) and from `list_labels`'s default
   page.
3. Type-name concepts are excluded from `audit_vocabulary`'s twin sweep
   (`src/api/vocabulary.rs:116`), so `Organization`/`Organisation` is a
   schema-authoring question, never a spelling-drift candidate.

### 6.4 `closed_labels`

Named in §6.1: one boolean, default `false`. When `true`, an association
whose label has no entry in `relations` is a violation reusing
`Issue::unknown_reference`. Closed-world label enforcement is a genuinely
different posture from open-world domain/range checking, and this ADR
fixes it as an explicit opt-in rather than something an implementer
guesses at.

**`closed_labels` never applies to `schema:type` itself.** §6.3 guard 3
forbids a relation definition named `schema:type` from ever existing, so
under a naive "any label absent from `relations` is unknown" reading,
every legitimate type assertion would itself be flagged the moment
`closed_labels` is set — silently defeating the type model `closed_labels`
was never meant to touch. `closed_labels`, like every other check in
§7.2, is scoped to `fact_ops` only (§7.2 step 3's partition); `type_ops`
are exempt by construction, not by a special case bolted onto the check.

## 7. Validation semantics and the ordering guarantee (for S3, S4, S5)

### 7.1 Mode: storage, default, and changing it mid-corpus

**Mode lives inside `{stem}.schema.json`, not `ContextMeta`.** A mode
without a document is meaningless; keeping it in `ContextMeta` would let
`PATCH /contexts/{name}` flip a context to `strict` with no schema behind
it. Keeping mode and document in one file makes "what the schema says" and
"how hard it is enforced" one atomic write and one revision bump.
`DirectoryEntry` (`src/registry.rs:358`) may still *echo* a read-only
`schema_mode: Option<String>` (an additive response field, compatible per
ADR 0005 §4) so a client can route without a second call — echoed, never
authoritative.

**Default is `off`, meaning no file was ever written for this context —
byte-identical to today's behavior.** This is not merely a default; it is
the mechanism by which the issue's own first acceptance criterion
("schemaなしcontextは現在と同じ動作を維持する") holds by construction. It is
a narrower condition than "no file is currently readable": per §5.1, a
context whose schema file exists but is corrupt, quarantined, or mid-move
refuses the boot rather than silently falling back to this default —
"never had one" and "have one I currently cannot read" are kept
distinguishable on purpose, so `off` never means anything but "never
opted in."

**Mode may change while violations exist — deliberately, and this is the
part to document rather than let someone discover.** `PUT /schema` never
inspects the graph: refusing a flip to `strict` while a violation exists
would require an O(edges) sweep inside a `PUT`, and would make `strict`
permanently unreachable on any corpus carrying one legacy violation — the
opposite of the incrementally-adoptable schema the issue asks for. `strict`
therefore means **"from now on"**: existing edges are grandfathered, and
pre-existing violations are visible only through the explicitly-invoked,
read-only `POST /contexts/{name}/schema/audit` (§10). The pre-flight the
issue also asks for ("schema更新前に既存dataへの影響をdry-runできる") is
`POST /contexts/{name}/schema/validate` — a *proposed* document evaluated
against the live graph, in the same "candidates for review, not verdicts"
framing `audit_vocabulary` already uses (`src/api/vocabulary.rs:37-40`).

**Rejected: refusing a `PUT` to `strict` while violations exist.** Named
above.

### 7.2 The ordering guarantee — the concrete algorithm

**Decision: validate against `(post-retraction live state) ∪ (this batch's
own type assertions)`, computed in full before any write.** Batch line
order is irrelevant by construction, because nothing is applied
incrementally during the check.

New function `predicted_schema_rejection(state: &AppState, batch: &Batch)`,
called as the first statement of `apply_batch` (`src/ingest.rs:2308`),
beside `predicted_alias_rejection` (`:2243`), sharing its exact properties
— pure, read-only, pre-write, stateful against live context state, shared
verbatim with `preview_batch` (`:2520`) so the two can never disagree
(`:2247-2248`).

1. **Fast exit.** No schema, or `mode == off` ⇒ return before reading a
   single association — the zero-cost path every existing context takes.
2. **Use the corrected op list**, `corrected_associations(batch, …).0`
   (`:2205`), not the raw batch, so the check sees exactly the ops the
   write will apply — the same reasoning that function's own doc gives for
   being shared with `preview_batch`.
3. **Partition.** Resolve each op's label through live `label_aliases` ∪
   this batch's own declared `batch.labels` — the same union
   `predicted_alias_rejection` already reasons over (`:2249-2251`, `:2277`)
   — and note aliases apply *last* in `apply_batch` (`:2296-2298`), which
   is exactly why the union, not the live set alone, is the correct
   resolution basis. Ops resolving to `schema:type` are `type_ops`; the
   rest are `fact_ops`.
4. **Build `TypeEnv`** — the union that gives this section its name:
   - **Live half**: for each concept named as subject or object in
     `fact_ops`, read its existing `schema:type` edges from the resident
     `Context`, **excluding any type assertion whose only attribution is
     `batch.source`** — because `apply_batch` retracts the source before
     applying (`:2294`), the judgment must be against the state the graph
     is *about to be in*, not the state it is currently in. Without this
     exclusion, a batch legitimately narrowing a source's own type claims
     would be judged against claims it is about to withdraw.
   - **Batch half**: add every `(subject → object)` from `type_ops`.
5. **Expand** each concept's type set through the schema's precomputed
   `is_a` ancestor closure (§6.2) — a set union against a precomputed map,
   never a walk.
6. **Judge, collect-all — `fact_ops` only, `type_ops` are never judged.**
   For each `fact_op` whose resolved label has a relation definition: a
   domain violation is `domain` non-empty **and** the subject's expanded
   set non-empty **and** the two disjoint; symmetrically for
   `range`/object. An empty type set never violates (§6.1); an empty
   `domain`/`range` never constrains. When `closed_labels` is set, a
   `fact_op` whose resolved label has *no* relation definition is
   additionally a violation (`Issue::unknown_reference`, §6.4) — scoped to
   `fact_ops` by this same partition, so a `type_op` (always labeled
   `schema:type`, which §6.3 guard 3 forbids from ever appearing in
   `relations`) can never be flagged by `closed_labels`. Collect **every**
   violation, following `interpret_associations`' own discipline
   (`src/api/associations.rs:167-176`, collecting every item's issues in
   one pass rather than rejecting at the first bad field), bounded by
   `MAX_LISTED_ISSUES` (`src/api.rs:450`).
7. **Dispatch.** `strict` + any violation ⇒ `Err(ApplyRefusal::Rejected(...))`
   — before the create, before the batch marker, before the retraction,
   inheriting the guarantee `wrote_anything()` already grants that arm
   (`:2145`; "a predicted rejection opens no marker at all: nothing ran yet
   for it to bracket," `:2303-2304`). `warn` ⇒ the write proceeds untouched
   and the violations ride out in the response (§8.3).

**Why line order cannot matter.** `TypeEnv` is complete *before* step 6
begins — there is no incremental application inside the check. A batch
stating a domain constraint on line 3 and the concept's type on line 40
validates identically to the reverse order. This is the same
union-before-judgment shape `cross_output_issues`
(`src/extract.rs:4515`) already uses on the producer side.

**The boundary this guarantee does not cover, stated explicitly.** The
union is **per-batch**, because `apply_batch` is per-batch and each batch
is independently durable. A `fact_op` in one batch typed only by a later
batch of the same stream is a violation — this is the direct consequence
of "one batch owns one source's truth" (`:1644-1648`), not a gap to close
later. Producers already satisfy the same scoping for aliases.

### 7.3 `POST /contexts/{name}/associations` — the non-atomic entrance

**Decision: strict mode never reaches the non-atomic partial-write arm.
The entire schema check runs pre-write and returns `integrity:
"nothing_written"`.**

The partial arm (`src/api/associations.rs:263-269`) is honest about being
non-atomic. Strict stays honest by deciding *before*
`tokio::task::block_in_place(|| state.add_associations(...))` — the same
position the existing shape-validation refusal already occupies
(`:230-247`, `RefusalDetail { issues, integrity: Some("nothing_written"),
retryable_after_correction: Some(true), .. }`).

`interpret_associations` (`:167`) stays pure and unchanged. A second, also
pure, function `schema_issues(env: &SchemaEnv, ops: &[AssocOp]) -> Vec<Issue>`
is added, where `SchemaEnv` is the schema document plus the live type sets
for exactly the concepts these ops mention, gathered by the handler in one
`state.read_context` call between the two pure passes. `schema_issues` is
the *same* function `predicted_schema_rejection` calls — the structural
guarantee that keeps the HTTP and import entrances incapable of drifting,
one level up from `predicted_alias_rejection`/`preview_batch`'s own
sharing. Two details specific to this entrance: (i) the request's own
array may itself contain type assertions, so `SchemaEnv` is built as live
∪ this-request's-own type ops (§7.2 step 4); (ii) there is no source
retraction on this path (each item carries its own optional `source`), so
the exclusion clause in step 4 does not apply — a plain union.

**Rejected: making the write path itself atomic.** That is #187's scope,
and would need cross-store transaction machinery `apply_batch` already
declines to attempt (`:2305-2307`).

## 8. Error contract (for S4, S5)

### 8.1 No new `ErrorCode`; two new `Issue.kind` tokens

**Decision: reuse `ErrorCode::InvalidArgument` (`src/api.rs:175-178`,
status 400). Exactly two new `Issue.kind` tokens: `"domain"` and
`"range"`.**

`ErrorCode`'s own doc states renaming or repurposing a variant is a
breaking, CHANGELOG-level change; `InvalidArgument` already means
precisely "the request parsed, but a value was refused," and the
machine-actionable discrimination the issue's acceptance criterion asks
for ("subject、label、object、expected type、actual type") belongs in
`Issue`, not in a new top-level code — the precedent is
`alias_rejection_issue` discriminating two `AliasError` arms into two
*existing* kinds without inventing a variant. This deliberately differs
from an alias rejection's own status (409, a state conflict): a schema
violation is a refused value, so 400.

Two kinds, not one, because `Issue` has exactly one `path`
(`src/api.rs:308-313`) and the acceptance criterion requires knowing
*which side* — subject or object — failed. Encoding that in `kind` (a
"stable machine key," the same role `inspect::Notice.kind`,
`src/inspect.rs:98`, already plays) is more stable than making a client
parse it out of `path`. A third `"unknown_label"` token is not added now:
when `closed_labels` fires (§6.4) it reuses `Issue::unknown_reference`.

### 8.2 Path, `expected`, `actual`

Paths point at what the caller must correct — the concept, never the
label, since the label is well-formed and only the typing disagrees:

| Entrance | domain violation | range violation |
|---|---|---|
| `POST /contexts/{name}/associations` | `associations[{i}].subject` | `associations[{i}].object` |
| `POST /import` / `taguru import` | `batches[{b}].associations[{a}].subject` | `batches[{b}].associations[{a}].object` |

The HTTP form reuses `interpret_associations`' own grammar
(`format!("associations[{index}]")`, `src/api/associations.rs:174`); the
import form extends `alias_rejection_issue`'s grammar
(`batches[{index}].{namespace}['{alias}']`, `src/api/import.rs:119-123`)
with the per-item index an alias path had no need for.

```text
expected: "'青嶺酒造' typed as one of [Brewery] (or a subtype), for relation '杜氏'"
actual:   "typed as [Person]"
```

`expected` enumerates the *declared* `domain`/`range` and says "or a
subtype" rather than expanding the closure — the expanded set's size is a
schema-authoring accident, not information the corrector needs. Both
strings are bounded (`MAX_RELATION_TYPES`, §5.3). `actual` is never
`"untyped"` — an untyped concept never violates (§6.1).

`RefusalDetail.retryable_after_correction` is `Some(true)` always — a
schema violation is by definition correctable. `integrity` on the
associations entrance is `Some("nothing_written")`; on import it is
whatever `stream_integrity` (`src/api/import.rs:150-158`) already returns,
unchanged, since a schema refusal is a `Rejected`-class refusal — one of
the two arms `wrote_anything()` already certifies.

### 8.3 `warn` mode: response shape and CLI

**Decision: `ApiResponse` gains `issues: Vec<Issue>`
(`skip_serializing_if = "Vec::is_empty"`), mirroring `ApiError.issues`
field-for-field. `Applied` (`src/ingest.rs:2015`) and `ImportOutcome` gain
`schema_violations: usize`. `Issue` values are byte-identical between
`warn` and `strict` — only the HTTP status differs.**

Putting warnings in the envelope rather than reshaping the result avoids a
breaking response-shape change (`POST /contexts/{name}/associations`
today answers a bare `usize` result) and gives structured warnings one
representation across every endpoint that could ever produce them — absent
on every response with nothing to say, which is all of `off` mode and all
of today's traffic. The counter exists because `MAX_LISTED_ISSUES`
truncates the list and the tally must survive truncation — the same reason
`Applied.association_paragraphs_dropped` (`src/ingest.rs:2044`) already
turns a silent loss into "a reported number." Identical `Issue` values in
both modes mean a client's violation handler is the same code whether the
context runs `warn` or `strict` — a mode flip becomes an operational
decision, not a client rewrite.

`taguru import`'s report line gains `schema warnings: N` beside the
existing `*_dropped` counts. `taguru inspect` gains an `inspect::Notice`
(`src/inspect.rs:98`) only for the schema **file's** own health —
unreadable, quarantined, or an unread `schema` version — never for graph
violations: `inspect` names "an alteration boot would make when it next
loads the data" (`:88-92`), and a schema violation alters nothing at boot.

## 9. Relation constraints — staging (for the whole implementation split)

### 9.1 In scope now

The schema document, `off`/`warn`/`strict`, persistence and revision,
entity types, `is_a`, and `domain`/`range` validation at every write
entrance, the standing audit and dry-run, producer vocabulary, and the
read-side minimum.

### 9.2–9.4 Deferred, and why

| Deferred | Why |
|---|---|
| `cardinality` (one / optional-one / many) | Requires a per-`(subject, label)` existing-edge read on the write path, whose correct answer under retract-then-apply is genuinely ambiguous: a batch replacing a source's facts transiently violates a one-to-one during the retract→apply window (`src/ingest.rs:2294-2299`). Needs its own design. |
| `inverse`, `symmetric` | Both are **inference** — asserting edges nobody stated — which §14 puts out of scope outright. A purely validating symmetric check ("refuse `A r B` unless `B r A` exists") is cardinality-shaped and carries the same batch-ordering hazard §7.2 solves for types, without §7.2's clean union answer. |
| `deprecated` relation + replacement | Would be an audit/producer-guidance signal, never a refusal — but §5.3's document shape has no field to mark a relation deprecated or name its replacement, so there is nothing for §10's audit or §11's producer guidance to read yet. Fully deferred: a follow-up ADR fixes the field shape before either surface can act on it, not an implicit part of this design. |
| self-loop policy, negative-weight policy | Both are relation-independent flags expressible later inside `relations[label]` with no shape change; nothing about `domain`/`range` forecloses them. |
| transitive closure, OWL reasoning | The issue itself defers these; §14. |

### 9.5 Relation canonicalization is not a schema feature

**Decision: the schema document names relations by canonical spelling
only. §7.2 step 3 resolves the label through `label_aliases` (plus the
batch's own) before consulting the document.**

Taguru already owns this problem end to end: `AliasRecord`
(`src/context.rs:427`, persisted since image v2), `WalOp::AliasLabel`,
batch-level declaration (`batch.labels`) with pre-write prediction
(`predicted_alias_rejection`), and `DriftAudit.dead_label_aliases`. A
`canonical:` key inside the schema document would create a **second**
answer to "what does `本社所在地` resolve to," resolvable in a different
order on the read path than on the write path — exactly the class of bug
that is undiscoverable until a query returns the wrong facts. The schema's
actual contribution to the `所在地`/`本社所在地`/`所在` drift the issue
names is not a new mechanism: `audit_vocabulary`'s twin-candidate framing
becomes *actionable* once the schema tells an operator which spelling is
canonical, and the fix is an ordinary alias.

## 10. Audit and dry-run (for S7)

**`POST /contexts/{name}/schema/audit`** reports, over the live graph:
untyped concepts, unknown relation labels (only meaningful with
`closed_labels`, and never `schema:type` itself, per §6.4), type names
asserted but absent from `types` (§6.2 — always reported, unconditional
on `closed_labels`, since an undeclared type is never a violation, only
a signal), and domain/range violations — several independent checks in
one response, in `DriftAudit`'s shape (`src/api/vocabulary.rs:187`): one
paged section (mirroring `page_by`, `:237`), and framed, like
`audit_vocabulary` itself (`:37-40`), as "candidates for review, not
verdicts" — this audit never auto-applies a fix. Deprecated-relation
usage is **not** in this list — §9.2 defers it because the document has
no field to mark a relation deprecated yet; a follow-up ADR adds both
the field and this audit's check for it together, rather than the audit
inventing a convention the schema shape does not yet define.

**`POST /contexts/{name}/schema/validate`** takes a *proposed* document and
evaluates it against the live graph without persisting it — the pre-flight
§7.1 promises before a `strict` flip. Both routes are O(edges) with no
cheap variant, joining the unconditional heavy-ops group
(`src/main.rs:764-779`) rather than `audit_drift`'s conditional-extension
pattern, which exists only because that route has a cheap default path.

## 11. Producer guidance (for S8)

### 11.1 `system_prompt` — one new block

**Decision: `system_prompt` (`src/extract.rs:3698`) gains one block after
the existing vocabulary block (`:3742-3752`), emitted only when a schema
exists and `mode != off`**: the allowed entity type names (`.take(CAP)`,
the same shape `VOCABULARY_CAP = 200` already uses), one `label: domain →
range` line per constrained relation (budget-capped, live-vocabulary
relations first), and the instruction to emit type assertions on the
reserved label with the same "reuse these exact spellings" framing.

`PROMPT_VERSION` (`src/extract.rs:132`, currently `2`) **must bump to
`3`** — non-negotiable, since chunk-checkpoint reuse is keyed on it
(`entry.prompt_version == PROMPT_VERSION`), and a cached output produced
under a schema-free prompt must not be silently reused under a
schema-bearing one.

### 11.2 `ItemRules` gains nothing; the check is a `cross_output_issues` sibling

**Decision: `ItemRules` (`src/extract.rs:3817`) is unchanged.** Its own doc
frames it as the two pieces of per-**document** context no single item
carries — a schema is per-**context** and per-**answer-set**: a type
asserted in output 3 licenses a fact in output 1, which no single-output,
Stage-1 check can see. The schema check is instead a new
`schema_output_issues(outputs, &schema) -> Vec<(usize, Vec<String>)>`,
structurally a sibling of `cross_output_issues` (`:4515`) — identical
per-output-index grouping and identical two-pass shape (collect every name
across every output, then judge each output), the exact producer-side
mirror of §7.2's union-before-judgment rule. It feeds
`corrective_validation_message` (`:3615`) unchanged.

### 11.3 `model_output_json_schema` gains documentation, not fields

**Decision: `model_output_json_schema` (`:4387`) is unchanged**, except for
three additions to its existing "what this schema does NOT encode" list
(`:4368-4382`): a concept's type set is known only per-context at
validation time, never at schema-authoring time (the same argument the
paragraph-count entry already makes); JSON Schema cannot express "the
object of relation R must be a concept some other item in this answer
typed as T" (a cross-item rule that list already has a bucket for); and
allowed relation labels are deliberately **not** rendered as a JSON Schema
`enum`, because a structurally-constrained model would then be unable to
propose a new relation — which the issue explicitly requires stays
possible ("既存contextとschemaなしの自由な利用方法は維持する"). Constraining
the model's *shape* is not the same as constraining its *content*.

### 11.4 SDK parity

`_extract.py`/`extract.ts` mirror the `system_prompt` block and corrective
wording byte for byte, per §2.6. `TaguruIngester._fetch_vocabulary`
(`ingest.py:1042`) gains a `_fetch_schema` twin against `GET
/contexts/{name}/schema` with the **identical best-effort posture** —
`except NotFoundError: return []` — which is exactly what makes a
schema-unaware server, or a schema-free context, work unchanged from the
SDK's point of view.

## 12. Retrieval (for S9)

### 12.1–12.3 Which endpoints, filter or field

**Decision**: `describe` returns types; `resolve` returns types on top
candidates only; `query` gains a type filter. Both halves of the
acceptance criterion ("取得・filter") are covered, each on the endpoint
where it fits its existing shape. **All three are gated by §6.3's single
condition — an installed schema document for the context — not by
mode**: a context with no schema populates none of this (§6.3 guard 1
applies identically here), and a schema in `off` still reports whatever
types it defines, exactly as `describe`/`resolve` already report
ungated facts regardless of any other server-side policy toggle.

- **`describe`** — `ConceptDescription` (`src/context.rs:356`) gains
  `types: Vec<String>`, populated inside `Context::describe`
  (`src/context/query.rs:176`)'s existing outgoing-chain walk, filtered to
  `schema:type` and reusing the same dead-edge skip — empty for a context
  with no schema, per the gate above. `describe`'s own doc already frames
  it as "the 'what is known about X' overview... BEFORE fetching facts" —
  a concept's types are exactly that.
- **`resolve`** — `TieredResolution` (`src/api/resolve.rs:39`) gains
  `types: Option<Vec<String>>` beside `kind`/`gloss`, attached to the same
  top candidates `gloss` already is, for the same cost reason. **Not a
  filter**: `ResolveRequest` (`:20`) is cue + floors + limit, and a type
  filter changes what resolve *means* (entry-point lookup by spelling) —
  a caller who already knows the type knows more than the cue does.
- **`query`** — optional `subject_types`/`object_types` on the request; the
  only read that returns a list large enough for a filter to matter.
  `AssociationOut` (`src/api.rs:1552`) is unchanged.

### 12.4 Explicitly not in scope: type-aware scoring

Using types to *score* — the issue's "relation domain/rangeを使った曖昧候補
の抑制" and its own explainability requirement — is deferred by name. A
score contribution that is not explainable is what the issue itself
forbids, and the explain surface (`/resolve/explain`, `src/auth.rs:786-789`)
is its own design problem.

### 12.5 Authorization

`required_role` (`src/auth.rs:760-813`) fails closed — an unclassified
route demands `Admin` and is replica-refused. Classification, fixed here:

| Route | Role | Sits beside |
|---|---|---|
| `GET /contexts/{name}/schema` | `Read` | other context GETs |
| `POST /contexts/{name}/schema/validate` | `Read` | `/vocabulary/audit`, `/drift/audit` |
| `POST /contexts/{name}/schema/audit` | `Read` | same |
| `PUT /contexts/{name}/schema` | `Write` | `PATCH /contexts/{name}` |

`PUT` is `Write`, not `Admin`: it is an ingest-loop verb an agent performs,
the same classification context creation already gets.

## 13. Backward compatibility, scope boundary, and privacy

- **Old image**: `IMAGE_VERSION` stays `6` (§3); every existing image loads
  exactly as it does today, into a context with no schema file, i.e.
  `mode == off`, i.e. today's behavior byte for byte.
- **Old batch**: a batch with no type assertions validates against
  whatever schema the context holds; a schema-free context exits at §7.2
  step 1 before reading a single association.
- **A new file read by an old server — the one real hazard of §5.1, named
  rather than implied away.** An older binary's `context_files` is
  `[String; 9]`, so a stray `{stem}.schema.json` is not deleted with its
  context, not moved on rename, and not hydrated. It is inert litter, not
  corruption — the boot scan registers a context by `.ctx` (index 0, the
  pivot); on a replica it is actively swept, since `hydrate_shared`
  removes local files the manifest does not know; on a writer, downgrading
  across this change requires deleting stray `.schema.json` files, which
  belongs in the CHANGELOG exactly like any other file-family addition.
- **A schema-carrying export hitting a schema-unaware server**:
  1. `export::render` (`src/export.rs:267`) emits a `taguru_schema` record
     only when the context has a schema and `mode != off` — a schema-free
     export is byte-identical to today.
  2. An old server today would hit `parse_stream`'s dispatch
     (`src/ingest.rs:1636-1637`) and fall through to the "not a batch
     header" error (`:1650-1655`) — a hard refusal, but a misleading one.
  3. **The explicit pre-flight is the real answer**: `version_facts()`
     gains `schema_formats` beside `batch_formats`; `Api::
     warn_on_version_skew("export")` (`src/export.rs:749`) and `taguru
     import --url` check it before a byte ships, refusing with both
     sides' `schema_formats` named — this is the acceptance criterion's
     "明示的なcompatibility error," delivered where it is actionable.
  4. A new server reading an unread `schema` version refuses in
     `parse_group`'s exact wording shape (`src/ingest.rs:1438-1443`);
     `deny_unknown_fields` on the record struct makes an unknown field a
     hard refusal too — this ADR's job is making the *message* honest, not
     adding a mechanism that already exists.
- `taguru extract`, `POST /import`, and `POST /contexts/{name}/associations`
  behave identically for a caller sending no new field. No existing batch
  file, no existing SDK version, needs a change to keep working.
- No connector, extractor, or SDK gains a new credential surface; nothing
  in this design touches `TAGURU_EXTRACT_*`'s existing boundary.

## 14. Consequences and follow-up

| Follow-up | Why deferred |
|---|---|
| `cardinality` (one / optional-one / many) | §9.2 — needs its own retract-then-apply-aware design. |
| `inverse`, `symmetric` relations | §9.2 — inference, out of scope; a validating-only symmetric check still needs a batch-ordering answer this ADR doesn't derive. |
| `deprecated` relation + replacement | §9.2 — a signal for audit/producer guidance, not enforcement; a small follow-up once §10/§11 exist. |
| Transitive closure / RDF-OWL reasoning | §1 — explicitly never in scope; associations are never inferred, only validated. |
| Bulk retype / relation-rename migration tooling | §1 — the audit reports candidates; a dedicated migration feature is its own issue. |
| Type-aware retrieval scoring | §12.4 — needs its own explainability design. |
| Cross-context schema sharing / a schema registry | §1 — each context owns its own document. |

## 15. Documentation impact

No documentation ships with this ADR. The owning sub-issue (S10, below)
is responsible for a new reference page beside `docs/modeling.html`,
updates to `README.md`, `docs/import.html`, `docs/extract.html`, and the
CHANGELOG — following ADR 0005 §2.6's rule that the PR adding a capability
documents it, not the ADR that designed it.

## Appendix: sub-issue split and requirement traceability

Ten sub-issues, filed under #218, mapping the issue's own six stages.
Stage 5 (cardinality/inverse/symmetric) produces no sub-issue — deferred
per §14, the same posture ADR 0007 §13 took for XLSX/GCS connectors.

| # | Title | One-line scope | Stage | Section |
|---|---|---|---|---|
| S1 | `schema: {stem}.schema.json, document shape, SCHEMA_VERSION, and the file family` | Standalone file on `GroupRecord`'s pattern, `context_files` → `[String; 10]`, `is_a` cycle/depth validation + closure precompute, `version_facts().schema_formats`. No enforcement, no HTTP. | 1 | §5, §6.2 |
| S2 | `schema: GET/PUT /contexts/{name}/schema, mode, revision, cache invalidation` | Management routes, auth classification, `config` bump + `cache_identity` re-mint, `DirectoryEntry.schema_mode` echo. | 1 | §5.2, §7.1, §12.5 |
| S3 | `schema: the reserved type label and the shared pre-write check` | `schema:type` + its three namespace guards, `TypeEnv` union (§7.2 steps 2–6), pure `schema_issues`, the two new `Issue` kinds. Library-level; no entrance wired. | 2 | §6, §7.2, §8.1 |
| S4 | `schema: strict/warn on POST /import and taguru import` | `predicted_schema_rejection` beside `predicted_alias_rejection`, shared with `preview_batch`, `Applied.schema_violations`, `ApiResponse.issues`, `import_refusal` mapping, CLI report line. | 3 | §7.2, §8.2, §8.3 |
| S5 | `schema: strict/warn on POST /contexts/{name}/associations and MCP` | Pre-write arm before `state.add_associations`, `integrity: "nothing_written"`, four new `tool_definitions()` entries. | 3 | §7.3, §2.3 |
| S6 | `schema: the taguru_schema export/import record and replication parity` | `export::render` emission gated on `mode != off`, `parse_stream` dispatch arm, version-refusal message, `warn_on_version_skew`, `FamilySig` coverage test, replica round-trip test. | 1, 3 | §13 |
| S7 | `schema: POST /contexts/{name}/schema/validate and /schema/audit` | Dry-run of a proposed document; the standing audit, `DriftAudit`-shaped, paged, heavy-ops gated. | 4 | §10 |
| S8 | `schema: producer vocabulary in extract and both SDK ingesters` | `system_prompt` block, `PROMPT_VERSION` 2→3, `schema_output_issues`, `_fetch_schema` twins, `model_output_json_schema` doc additions. | 3 | §11 |
| S9 | `schema: types on resolve/describe, type filters on query` | §12 exactly. | 6 | §12 |
| S10 | `schema: metrics and documentation` | Closed-enum `SchemaOutcome {Ok, Warned, Refused}` (zeros always emitted), per-context violation counts behind the existing `PerContextMetrics {Off, All, Top(n)}` opt-in-and-bounded pattern; reference page, README, docs, CHANGELOG. | all | §15 |

**Ordering.** Strictly ordered: `S1 → S2`; `S1 → S3`; `{S2, S3} → S4`;
`S3 → S5`; `{S1, S4} → S6`; `S3 → S7`; `S2 → S8`; `S3 → S9`; everything
`→ S10`. Parallel once S3 lands: `S4 ∥ S5 ∥ S7 ∥ S9` — they share
`schema_issues` and touch disjoint files. S8 runs in parallel from the
moment S2 lands and touches no Rust validation code at all. Critical path:
`S1 → S2 → S3 → S4 → S6 → S10`.

**The one sequencing trap worth stating plainly**: S3 must land *before*
S4/S5, not alongside them. If the two write entrances build their own
schema checks in parallel they will diverge — precisely the failure
`predicted_alias_rejection`/`preview_batch` sharing and
`corrected_associations` sharing were each written to prevent.

| #218 acceptance criterion | Section | Owning sub-issue |
|---|---|---|
| schemaなしcontextは現在と同じ動作を維持する | §7.1, §13 | S1, S2 |
| context単位でentity typesとrelation domain/rangeを定義できる | §5, §6 | S1, S2 |
| conceptへ型を付与し、source attribution方針が明文化されている | §6.1, §6.3 | S3 |
| `warn`はwriteを受理して構造化warningを返す | §8.3 | S4, S5 |
| `strict`は違反batchをatomicに拒否し、path-specific errorを返す | §7.2, §7.3, §8 | S3, S4, S5 |
| associations API、import、MCP、extract、Python/TypeScript ingesterでvalidation semanticsが一致する | §7.2, §7.3, §2.3, §11.4 | S3, S4, S5, S8 |
| extract producerがschema vocabularyを受け取り、違反をcorrective retryできる | §11 | S8 |
| 既存graphを変更せずにschema auditできる | §10 | S7 |
| schema更新前に既存dataへの影響をdry-runできる | §7.1, §10 | S7 |
| schemaとrevisionがimage、WAL、export/import、replicationで保持される | §3, §5, §13 | S1, S6 |
| schema変更で関連するretrieval cacheが無効化される | §5.2 | S2 |
| resolve/describe/queryの少なくとも一部でentity typeを取得・filterできる | §12 | S9 |
| violation diagnosticsとmetricsが追加される | §8, §10 | S4, S5, S7, S10 |
| 完全なRDF/OWL互換を約束せず、初期スコープ外が文書化される | §1, §14 | (this ADR) |
| migrationと後方互換性がdocumentationされる | §13, §14, §15 | S10 |
