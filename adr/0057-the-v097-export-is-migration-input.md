# 0057. The v0.9.7 release's export is this build's migration input

- **Status**: Accepted
- **Date**: 2026-11-17
- **Issue**: pre-release review for 0.10.0 (branch `fix/pre-release-0.10.0`,
  under #958)
- **Related**: ADR 0042 (`type`/`version`/`id` columns), ADR 0043 (the
  schema document states type and version), ADR 0050–0053 (groups by id),
  ADR 0054 (a source's id is the caller's string), ADR 0056 (community ids
  per analysis)
- **Supersedes**: ADR 0045 §2.7 in part (how the "export, then import"
  crossing works), ADR 0053 §3 in part (the crossing through the builds in
  between) / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

ADR 0045 §2.7 said a data directory without ids is not read, and that "the
operator exports with the previous release and imports with this one, which
assigns ids". No build ever did that last step. From the first build that read
`type` columns (format `2026-09-17`, ADR 0042) on, import refused the v0.9.7
release's stream at line 1: its records say `taguru_batch`, `taguru_schema` and
`taguru_group`, name their `context` by display name, and name group members by
name. ADR 0053 §3 then described a crossing through the builds in between. Those
builds were never released, so for an operator on v0.9.7, the last release, the
crossing did not exist. 0.10.0 would have shipped with no upgrade path.

The two shapes are close. The operation lines (associations, aliases, passages
with `stored_at`/`date`/`tags`, questions, sections, locators) have not changed
since v0.9.7. The schema document's body (`mode`, `closed_labels`, `types`,
`relations`) has not changed either. What changed is the three record headers,
and the fact that everything they name is now an id.

A related gap showed up in the same review. The `{stem}.schema.json` files and
the stored derivation manifests that the unreleased builds stamped with
`2026-09-17` or `2026-10-01` were refused outright, even though the schema
document's shape had not changed since ADR 0043.

## 2. Decision

1. **This build reads the v0.9.7 release's export format as migration
   input.** That covers `{"taguru_batch":1,"context":<name>,"source":<id>,
   "create"?:{description,pinned,dice_floor,semantic_floor}}` headers,
   `{"taguru_schema":1,"context":<name>,…}` schema records, and
   `{"taguru_group":1,"name":…,"description"?,"contexts"?:[names],
   "groups"?:[names]}` group records, followed by the unchanged operation
   lines. Two entrances read it: `taguru import FILE|DIR...` offline and
   `POST /import` on a single instance. Entrances that cannot settle names
   against one registry refuse it and name those two: `import --url`, whose
   chunks would each settle separately, `taguru router`, whose shards would
   each mint a different id for one name, and every internal
   `parse_stream` caller.
2. **Names are settled on ids against the receiving registry before
   anything applies.** This happens across every v0.9.7 file of one
   invocation (one CLI run, or one `POST /import` body).
   - A context name that exactly one standing `context` carries reuses that
     id.
   - An absent name gets a minted id. Every batch of that name carries the
     first create block the invocation brings for it.
   - An absent name with no create block anywhere in the invocation is
     refused by name, as v0.9.7 refused such a batch.
   - A name several standing `contexts` carry is refused (`409 conflict`
     over HTTP), never guessed.
   - Group names resolve the same way, with an absent group minted. Member
     contexts resolve by name. A child group resolves to a record of the
     same invocation or to the one standing group with that name.
   - The group records then restore as one set, children first, so file
     order never matters.
   - A refusal applies nothing. A re-import is idempotent, because the
     names now resolve to the ids the first import minted.
3. **One file is one format.** A file whose stream-level records mix
   v0.9.7 and current-format records is refused at the first record of the
   other format. Separate files of both formats may share one run, and the
   v0.9.7 files apply after the current ones.
4. **Only the v0.9.7 release's shapes are read.** Streams written by the
   unreleased `2026-09-17` and `2026-10-01` builds are not import input:
   their header carries a `version` this build refuses, and supporting it
   would be different code for data no release wrote. Export always writes
   the current format; the v0.9.7 shapes are read-only.
5. **Earlier format revisions are named.** `format::EARLIER_VERSIONS` lists
   `2026-09-17` and `2026-10-01`. A reader whose record did not change across
   them may read one as its own: the at-rest `{stem}.schema.json`, whose
   shape is field-for-field this build's since ADR 0043. A schema file in
   the shape before ADR 0043 (a numeric `schema` stamp) refuses the boot
   with the migration hint, not as a corrupt file. A reader whose record
   did change treats one as written by an earlier build. The stored
   communities and consolidation manifests are incomparable, so
   `taguru communities` and `taguru consolidation` rebuild, the same
   `rebuilt_from_scratch` path as an algorithm change. The v0.9.7 release's
   untyped `taguru_communities`/`taguru_consolidation` manifests are
   treated the same way.
6. **Every pre-id refusal names this crossing.** `format::migration_hint()`
   (the boot refusal and `taguru inspect`) says: export with v0.9.7, start
   this build on an empty data directory, `taguru import DIR`, see
   `docs/import.html#upgrading-from-0-9-7`.

## 3. Consequences

- An operator upgrading from v0.9.7 runs one export on v0.9.7 and one
  import on this build. ADR 0045 §2.7's intent holds: the data directory
  itself is still not read.
- `ParsedStream` (current or legacy) and `resolve_legacy` sit beside the
  stream parser. The legacy records share the current parser's line loop,
  caps, and operation-line parsing, so the two cannot drift.
- Ids are minted at import, so a migrated `context` has an id no v0.9.7
  setting knew. Grants and quotas must be rewritten by id after the import.
- Derived artifacts (`NAME::communities`, `NAME::consolidation`) arrive as
  plain contexts under minted ids. This build derives an artifact's id from
  its source's id, so one `taguru communities`/`taguru consolidation` run
  per source rebuilds them, with LLM calls.
- Behind a router, each shard is upgraded on its own. A group spanning
  shards then has one id per shard and is recreated once through the router.
- Supporting a second release's format later would mean another reader
  beside this one. No such reader is planned. A release that drops this
  one is a new ADR.
