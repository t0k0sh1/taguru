# 0010. taguru-code: the offline code map — universe, packaging, and fact model

- **Status**: Accepted
- **Date**: 2026-08-06
- **Issue**: #447
- **Related**: #443, #444, #445, ADR 0004 §5, ADR 0005, ADR 0007 §3, ADR
  0007 §7
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

The design decisions behind `taguru-code` (#442, merged as #445): what set of
files the code map covers, how the tool is packaged, where its data lives,
what facts a source file produces, how lookup works without aliases, and why
the whole pipeline is deterministic with a built-in accuracy gate.

Unlike ADR 0007, which preceded its implementation, this document records
decisions that were made and *validated* during the #442 spike — a
prototype-first path the issue chose deliberately, with a four-axis
evaluation (accuracy gate, a face-off against the incumbent tool, freshness
tests, ingestion cost) as the graduation bar. The decisions are fixed here so
later changes supersede them explicitly instead of drifting; the evaluation
record lives in `examples/skills/taguru-code-nav/results-2026-08-06.md`.

Out of scope, tracked elsewhere rather than answered here:

- Performance work — content-fingerprint skip, fsync cadence, trimming the
  dual-inclusion set (#443).
- A `watch` mode that follows edits automatically (#444).
- Grammars beyond Rust. §6 fixes the contract that makes a new language one
  file plus one table row; it does not implement any.
- Call graphs, module-dependency facts, or any semantic analysis. The map
  answers "where is X, what does Y contain" — nothing else.

## 2. The problem, and the shape of the answer

A coding agent locating a symbol greps repeatedly: candidate patterns, then
disambiguation, then the definition among usages. The incumbent
knowledge-graph tool (graphify) answers location questions with a
multi-thousand-token graph traversal that mixes in unrelated hub nodes, and
misses method-on-type and typo'd-name questions (8/12 on the evaluation set,
against 12/12 for the shape chosen here).

The answer is a map, not a search: one deterministic pass extracts every
definition's location and containment into the association graph taguru
already has, and lookup is a single command whose whole output is
`kind qualified-name file:lines [tier]` — one line per hit, exit 1 with an
explicit fall-back-to-grep instruction when the map does not know. Agents
consume tools through their failure modes as much as their successes; an
honest miss is part of the contract.

## 3. The universe: exactly what ripgrep sees

The map covers **tracked plus untracked files, minus everything .gitignore
excludes, read as the bytes on disk** — staged and unstaged edits included.
This is ripgrep's default universe, and that equivalence is the point: the
tool's visibility matches the agent's other eyes, so "taguru-code knows
everything rg would search" is a rule an agent can trust without caveats.

- `git ls-files --cached --others --exclude-standard` is the one authority
  for membership. There is no hand-rolled directory walker anywhere — a
  bespoke walker would re-implement ignore semantics wrong, and gitignored
  files (secrets, build output, local settings) must never enter the map.
  A directory that is not a git work tree is a refusal, not a fallback.
- HEAD is only the **incremental anchor**, never a content source: a sync
  touches `git diff <last synced commit>..HEAD` (the committed churn) plus
  the working tree's dirty set — the current one and the one recorded at the
  last sync, so a revert, an untracked file's deletion, or a
  dirty-then-committed file all heal to their real content. A gc-orphaned
  anchor degrades to a full re-sync with a warning; idempotency (§5) is the
  recovery mechanism, so the degradation is safe by construction.
- The data directory itself is excluded from the universe unconditionally.
  `.taguru/` is untracked; without this rule its own files would enter the
  dirty set on every run and "up to date" could never hold. Gitignoring it
  is advised, but correctness must not depend on advice.

The first spike scoped the universe to committed state only (simpler
anchoring). It was widened before merge because the committed-only map went
blind exactly where an agent looks most — the code it just wrote this
session. The cost is honest: the map is a snapshot as of the last sync, and
a mid-edit file may parse partially until the next one. `status` reports
both pending committed changes and pending dirty files so staleness is
visible, and #444 exists to automate the refresh.

## 4. Packaging: a third binary over the server's own modules

`taguru-code` is a **standalone binary** (`src/bin/taguru-code.rs`), built
from the same crate by dual inclusion — the `#[path]` pattern `taguru-mcp`
established, scaled to the import web's closure. Two alternatives were
implemented-or-designed far enough to reject on evidence:

- **A Python SDK connector** (the ADR 0007 §3 route, and the first full
  design for this feature). Rejected on installation friction: `pip install`
  plus per-language extras is real cost for a polyglot developer, and the
  SDK path had no offline query surface at all — reads would have required a
  running server. The decisive requirement was *one command, zero
  configuration, fully offline*.
- **A `taguru` subcommand**. Rejected for surface separation: the code map
  is an agent-facing tool with its own verbs, defaults, and failure
  vocabulary; the server binary's CLI is an operator surface. They share the
  crate and the batch contract, not an argv namespace.

Putting `tree-sitter` + `tree-sitter-rust` into the crate's dependencies
deliberately revisits ADR 0007 §3, which kept document parsers *out* of
`src/` — that decision weighed the PDF/DOCX ecosystem's Rust immaturity,
`Cargo.lock` audit surface, and image bloat. tree-sitter fails the analogy
on the first point (mature, Rust-native, MIT, cargo-deny clean) and the
boundary itself is different: ADR 0007 protected the **server** from
parser dependencies, and the server's wire contract is untouched here —
everything taguru-code writes goes through the standard batch/import
contract a client could have sent (ADR 0005 posture). The accepted costs
are binary size (grammar objects link into both binaries) and deny.toml
audit surface, both named in #442; trimming the inclusion set is #443's
item, not a reason to block.

## 5. Data directory: `$PROJECT_ROOT/.taguru`, a normal data dir

The default data directory is `.taguru/` at the repository root — never the
server's `./data` default, which collides with real project directories.
Deliberately, it is a **standard taguru data directory**: the same images,
sidecars, and locks every other entrance uses, so `taguru serve` can serve
the same map over HTTP/MCP later with zero migration. The one file
taguru-code owns inside it (`code-sync.json`: last synced commit + last
dirty set) has a non-`.ctx` extension the registry's boot scan ignores.

Two write-path choices follow from the sync being **idempotent by the batch
contract** (one file = one source; re-import retracts then re-applies):

- Per-op WAL durability is off for sync (`wal_enabled: false`, the
  flush-interval window). A crash mid-sync is recovered by re-running, not
  by replay; the sync point only advances on success. Measured effect:
  85 s → 17 s on the first full debug-build sync.
- A lost or corrupt state file, or an unusable anchor, degrades to a full
  re-sync. There is no state that cannot be rebuilt from the repository.

## 6. Fact model: location and structure, nothing cleverer

A grammar (tree-sitter, one `Grammar` impl per language behind an extension
table) turns one file into symbols; the fact builder is a pure function from
symbols to batch lines. Per file, one source (`<repo-relative path>`, the
retract-then-apply unit) carrying:

- **Names**: a symbol is `<repo-relative path>::<name path>` — the file path
  IS the namespace, so identity is language-agnostic and collision-free
  without any module-resolution cleverness (Rust's `mod` graph, Python's
  packages). Files and directories are their bare paths; `contains` edges
  chain uniformly from directory to file to symbol, and an
  `impl OtherFilesType` scope anchors under its file so methods never float
  free of the tree.
- **Edges**: `defined_in` (symbol → file, the "where is X" product) and
  `contains` (structure). Nothing else in v1 — no call graph, no
  module-dependency facts. Coarse facts age slowly; fine-grained ones would
  churn on every edit and duplicate what LSP-shaped tools already do well.
- **Line ranges** as citation locators `{"kind": "lines", "value": "a-b"}` —
  the open locator vocabulary ADR 0007 §7 fixed, carrying code line ranges
  with zero server change, exactly the extension mechanism it was designed
  to be.
- **The passage is NOT the raw file.** `src/paragraph.rs` splits on blank
  lines and stores at most one locator per paragraph; code bodies contain
  blank lines, so "one symbol = one paragraph" cannot survive raw source —
  locators would silently mis-attach, precisely the failure mode ADR 0007's
  locator design exists to prevent. Instead each symbol contributes its
  one-line signature as one paragraph, in source order: paragraph *i* is
  symbol *i* by construction, every symbol keeps its own locator and
  section heading, and BM25 indexes the identifier-dense line that matters.

## 7. Lookup without aliases

Short-name aliases (`parse_batch` → the qualified name) were the obvious
design and are **rejected**: the alias store is append-only (`retract_source`
never removes one), and an alias conflict refuses the *entire* batch before
any write. In a codebase, short names collide constantly (`new`, `run`,
`tests`) — naive minting would refuse a large fraction of files, and
first-claimant-wins minting would leave permanently stale claims after every
rename. A one-shot, irreversible claim is the wrong primitive for names that
churn.

Instead `find` ranks at read time, over one in-memory scan of the
`defined_in` edges: exact tail segment, qualified suffix (`Type::method`,
or `file.rs::name` via a `/`-boundary), tail prefix/substring, a
file/directory basename tier for path cues, and a tail-segment bigram-Dice
tier as the typo fallback with a floor — below it, the honest answer is "no
match, fall back to grep", never plausible-looking junk. The generic
`resolve()` tiers are unsuitable here by design, not by defect: they score
whole stored names, and a qualified name's path drowns its tail.

One consequence of the platform's append-only graph surfaced as a bug and is
now a rule: retracting a source empties an edge's attributions but keeps the
edge record, so **read paths must filter attribution-less edges** — a
renamed file's old symbols are ghosts no source attests anymore.

## 8. Deterministic, with the accuracy gate built in

No LLM appears anywhere in the pipeline — extraction is AST-mechanical,
which is what makes the accuracy gate free: `evalset` samples symbols from
the same parse and emits labeled cases (tail, qualified, and path cues;
tails shared by more than a few symbols sample as qualified cues, because no
tool can answer `tests` from the bare name and no agent would ask that way),
and `eval` replays them through the same `find` core the agent uses,
scoring hit@1 / hit@10 / locator drift. `--thresholds` turns a completed
run into a pass/fail gate that exits 3 on regression — ADR 0004 §5's CI
convention, offline. The gate hard-errors on unknown threshold keys or
mis-typed values and reports how many thresholds applied: a silently hollow
gate is worse than none.

Acceptance at merge: hit@1 0.915, hit@10 1.00, drift 0 on this repository's
200-case set.

## 9. Consequences

- The map is a snapshot; freshness is a sync away, visible via `status`,
  automated by #444 later. Mid-edit files may parse partially — tree-sitter
  degrades to the symbols it can see, and the next sync heals.
- Rust-only at merge. The grammar contract (`Grammar` trait + extension
  table + pure fact builder) confines a new language to one file and one
  row; nothing above the grammar knows a language exists.
- Both binaries carry the grammar objects; deny.toml's surface widened.
  Accepted in #442, revisited in #443.
- `find`'s scan is O(edges) per call — measured 67 ms at 5.6 k symbols,
  fine to ~50 k; a prefix index is a #443 item gated on a real monorepo
  need, not built speculatively.
- The agent-facing contract (use before grep, trust-then-verify at the
  site, fall back on exit 1) ships as a skill template in
  `examples/skills/taguru-code-nav/`, beside the evaluation protocol and
  results that justified graduation.
