# 0015. Context-vocabulary steering at extract time

- **Status**: Accepted
- **Date**: 2026-08-08
- **Issue**: #496 (S3)
- **Related**: ADR 0014 (S2 — the in-document half), ADR 0013 (S1 —
  the occurrence check this interacts with), ADR 0012 §4 (twin
  detection, the layer this prevents work for), ADR 0009 §13 (the
  file-not-server precedent)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How `taguru extract` steers a new document's spellings toward the
TARGET CONTEXT's existing vocabulary (#496 S3's "extract 時の語彙
resolve"). Out of scope: S4 (coverage verification), the SDK producers,
and relevance-ranked vocabulary selection (§4).

## 2. Context

S2 (ADR 0014) prevents spelling variance WITHIN one document: the
candidate block anchors the model to the document's own surface forms.
It cannot touch the cross-document twin — `cargo-nextest` extracted in
session 1, `nextest` in session 2 — because each run only sees its own
documents. Today that twin is caught after import by the consolidation
audit (ADR 0012 §4): detection, at merge-proposal cost, after both
spellings already live in the graph. The 2026-08-08 rehearsal measured
exactly this: the `cargo-nextest`/`nextest` merge was the audit's
canonical finding and a manual re-import was needed to fold it.

The reference vocabulary has to come from somewhere, and ADR 0009 §13
already ruled on the shape of that question for `--schema`: extract is
an offline producer with no server credential surface, so the operator
hands it a file. The same ruling holds here, for the same reason.

## 3. Decision

**`--vocabulary PATH` / `TAGURU_EXTRACT_VOCABULARY` loads exported
batch stream(s) and steers the run toward the harvested names — via
the prompt for spelling choice, and via the ADR 0013 allowlist for
acceptance. Off by default; the harvested name set is a fingerprinted
computation input.**

1. **Source: exported batch streams, file or directory** — the format
   taguru already owns (`taguru export --out DIR`, or
   `GET /contexts/{name}/export` saved to disk). Harvested per batch:
   association subjects/objects and alias CANONICALS as concept names;
   association labels and label-alias canonicals as label vocabulary.
   Alias *spellings* are never harvested — they are exactly the
   variants a canonical exists to fold, and offering them would seed
   the twin this control prevents. A path that loads nothing is a hard
   startup error (the `--schema` posture): the operator explicitly
   asked for steering, and drifting silently defeats it.
2. **Labels ride the existing machinery**: the harvested labels seed
   the run's label vocabulary, so the long-standing "relation labels
   already in use" block carries them from the FIRST document — before
   this, that block was empty until the run's own first document
   landed. No new prompt surface for labels.
3. **Concept names get their own block**, after the label block,
   carrying ADR 0014's measured contract verbatim (prose list — no
   re-encoding; data framing; the anti-checklist clause;
   non-restriction) plus the one instruction that is S3's point:
   *use the context's exact spelling even when the document spells the
   same entity differently*. Capped at 200 names, alphabetically —
   arbitrary but deterministic; §4 names the upgrade.
4. **The ADR 0013 occurrence check admits vocabulary spellings**: a
   subject/object spelled the context's way is not a fabrication,
   however the document spells the entity — without this, the steering
   and the mechanical validation would fight, and the check would
   remove exactly the resolutions this control exists to produce. The
   allowlist is the FULL harvested set (not the capped prompt list):
   a context spelling is legitimate whether or not it fit the prompt.
5. **Fingerprinted like a schema**: the digest of the harvested name
   sets (content-addressed — same names, same digest, whatever file
   layout produced them) rides the manifest and checkpoint
   fingerprints (`""` = off). `benchmark extract` forwards
   `--vocabulary` as a global task setting and records the digest in
   `extraction_settings.vocabulary_sha256`.

## 4. Staged on purpose

v1 offers names without ranking them. A context whose vocabulary
exceeds the prompt cap gets an alphabetical prefix — deterministic,
honest, and crude. The planned upgrade is relevance selection:
embedding similarity between the document and the vocabulary picks the
names that could plausibly matter for THIS document (the fastembed
feature or TAGURU_EMBED_URL is the natural provider). That stage is
bought the way S2's morphological analyzer would be (ADR 0014 §3.4):
against a measured corpus where the alphabetical prefix demonstrably
drops the names that mattered — not before. The occurrence allowlist
is deliberately exempt from all capping, so scaling pressure lands
only on the prompt list, never on acceptance.

## 5. Consequences

- Layering (restating ADR 0014 §5 from the other side): S2 prevents
  in-document variance, S3 prevents cross-document variance against
  the context, and the consolidation audit (ADR 0012) remains the
  detection net under both — prevention shrinks its workload, never
  replaces it.
- Rust-only, like `--schema` and `--candidates`; SDK follow-ups
  inherit all three together.
- A vocabulary export goes stale as the context grows; a stale one
  still steers correctly toward every name it knows and simply misses
  newer ones — degradation is gradual and the audit catches the rest.
  Refreshing the export is the operator's loop, documented in
  docs/extract.html.
