# 0056. A community's id is valid within one analysis; its fingerprint is the identity across re-runs

- **Status**: Accepted
- **Date**: 2026-11-16
- **Issue**: #1050 (decisions recorded on the issue, 2026-11-16; under #958)
- **Related**: ADR 0045 (an id apart from the name), ADR 0054 (a source's
  id is the string its caller chose), ADR 0055 (an API key's name is its
  id), #851 (a column that points at a record says which record's id it
  holds), #166 (community detection and `taguru communities`)
- **Supersedes**: ADR 0006, in part — the §9 statement that
  "`CommunityHit.community` is always prefixed `community:{id}`". §9's
  conclusion stands (a passage hit and a community hit never share a
  `(source, paragraph)` locator), for the reason given in §2.4 below. /
  **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

Community detection (`louvain-cc/1`) names each community
`L{level}-{index}`, where the index renumbers communities by their first
member in dense concept order. The partition is deterministic — the same
graph gives the same ids — but the ids are positions: when the graph
changes, the indices shift, and a later analysis can hand `L0-3` to a
different cluster.

`taguru communities` stores each community's summary in the artifact
context under the source `community:{id}`. A re-run reuses a summary
whose content fingerprint (FNV-1a over a leaf's members and induced
edges, or a parent's child fingerprints) is unchanged, but writes it
under the *new* id and retracts the ids no new community carries. So a
caller that kept `L0-3` from a search hit, or the source
`community:L0-3`, can later find a different cluster under it.

The fingerprint is already the cross-run identity: equal exactly when the
summarized content is equal. The analysis stream (`GET
/contexts/{id}/communities`) and the artifact's manifest carry it, but a
search hit (`CommunityHit`, also embedded in evidence items) did not.

Two heavier alternatives were considered and not taken: minting a UUID
per fingerprint lineage (needs a heuristic for which changed community
"continues" which, plus rules for splits and merges, and persisted
lineage), and using the fingerprint itself as the id (an id that moves
whenever one fact inside the community changes).

## 2. Decision

1. **A community's id stays positional and is valid only within the one
   analysis that produced it.** The docs, `GET /protocol`, the MCP tool
   description and the SDK models say so.
2. **`CommunityHit` carries the summary's own `fingerprint`**, and a
   caller keeping a community beyond one analysis compares fingerprints.
   `taguru communities` stores each summary with a
   `community-fingerprint:{fp}` tag, written in the same import batch as
   the text, and a search reads the tag from the same passage record as
   the hit's text. The manifest cannot supply it: it is written last, so
   mid-rewrite the summary under `community:L0-0` can already be the new
   run's while the manifest's `L0-0` entry is still the old run's, and a
   positional id cannot tell the two apart. For the same reason the
   manifest's other facts (`level`, `parent_id`, `concept_count`) join a
   hit only when the manifest entry's fingerprint equals the summary's.
   A summary written before summaries carried the tag reports no
   fingerprint and keeps the id-only join.
3. **Columns that point at a community say `_id`** (#851):
   `CommunityHit.community` → `community_id`, `CommunityHit.parent` →
   `parent_id`; in the analysis stream, `parent` → `parent_id` and
   `children` → `child_ids`. A row's own key stays `id`. Communities get
   no display `name`.
4. **The wire carries the bare id** (`"L0-3"`), and the artifact's source
   for it is `community:{id}`. An evidence candidate built from a
   community hit is keyed on that source, which no passage source uses —
   so ADR 0006 §9's "a passage and a community candidate never share a
   locator" holds through the source, not through the hit field.
5. **The artifact's stored manifest (`communities:manifest`) is
   unchanged**, `parent` included. It is a record inside the artifact, not
   a response, and renaming it would make every existing artifact fail to
   parse on its next re-run.
6. **A search hit exposes the passage record it was cut from** (inside
   the server only), so reading a source's metadata beside its text needs
   no second read and cannot straddle two writes.

## 3. Consequences

- Breaking wire change on the unreleased `http_contract` 2: the
  `communities/search` hit fields, evidence items' embedded `community`,
  the analysis stream's `parent`/`children`, and both SDKs'
  `CommunityHit`.
- Existing artifacts keep working without a re-run, but their hits carry
  no `fingerprint` until the next `taguru communities` run tags the
  summaries. That run reuses every summary whose fingerprint is unchanged,
  so it costs no LLM calls.
- Positional ids can still move between runs; the contract now says so
  instead of implying stability.
