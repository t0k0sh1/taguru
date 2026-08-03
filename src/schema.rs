//! The per-context schema document and its file: one optional
//! `{stem}.schema.json`, holding entity types (an `is_a` hierarchy) and
//! relation `domain`/`range` constraints (ADR 0009). This module owns
//! the document's shape, its on-disk round trip, and the `is_a`
//! validation/closure precompute; [`check`] owns the pure
//! `schema_issues` every write entrance shares (S3, #381) — nothing
//! *here*, in the document module itself, enforces anything against a
//! graph write.
//!
//! Structurally this follows [`crate::groups::GroupRecord`]'s pattern —
//! atomic write-then-rename, a rename marker the registry already
//! provides, `BTreeMap`/`BTreeSet` for deterministic output — with one
//! deliberate divergence a group does not need: **a schema file that is
//! present but cannot be trusted refuses the boot, it is never replaced
//! with a fresh empty record.** A `GroupRecord` may safely reset to
//! empty on a parse failure (the consequence is "no nesting," a
//! routing-only concern); the schema's empty shape is indistinguishable
//! from `mode: off`, and silently landing there would silently disable
//! `strict` enforcement for a context whose operator explicitly turned
//! it on. So every trouble case here — unreadable, does not parse, does
//! not validate, or its digest disagrees with what `ContextMeta`
//! recorded — is a hard refusal, never a fallback.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::api::MAX_NAME_BYTES;
use crate::registry::{schema_corrupt_path, schema_path};
use crate::sha256::sha256_hex;
use crate::storage::write_atomic;

// `schema_issues` (S3, #381) is called from the associations pre-write
// arm (S5, #383, `src/api/associations.rs`). `predicted_schema_rejection`
// (S4, #382, the import entrance) has not landed yet, so this is still
// the only caller.
mod check;
pub(crate) use check::{SchemaCheckInput, SchemaEnv, schema_issues};

/// This binary's only readable document shape. Independent of
/// `BATCH_VERSION`/`GROUP_VERSION`/`IMAGE_VERSION` — [`GroupRecord`]'s
/// own doc gives the justification verbatim: separate "so either shape
/// can rev without dragging the other along." Unlike those siblings,
/// this bumps on every shape change, additive or breaking: the struct
/// below is `deny_unknown_fields` AT REST as well as on the wire, so an
/// older binary silently dropping a field it doesn't recognize —
/// indistinguishable from under-enforcing `strict` — never happens
/// quietly; it is instead an unread `schema` stamp or a parse refusal,
/// both hard boot failures (see the module doc).
///
/// [`GroupRecord`]: crate::groups::GroupRecord
pub(crate) const SCHEMA_VERSION: u64 = 1;

/// The reserved relation label type assertions ride under (ADR 0009
/// §6.3) — an ordinary association carrying `{subject, "schema:type",
/// object}`, exactly like `UNSOURCED_SOURCE`/`EMPTY_SOURCE`
/// (`src/context.rs:195`, `src/export.rs:110`) are reserved
/// `namespace:value` ids. Three guards keep any path from resolving a
/// label to this one once a schema document exists (never gated on
/// mode — see [`check::SchemaCheck::reserved`]'s own doc for why):
/// `install` refuses a document that names it as a relation (guard 3,
/// checked here); `PUT /schema`'s
/// migration-boundary check refuses on any already-persisted
/// `label_alias` that resolves to it (guard 2's install-time half,
/// `src/registry/lifecycle.rs`); and the aliases handler plus
/// `schema_issues`' `SchemaCheck::reserved` refuse `add_label_alias`
/// and a batch's own inline `batch.labels` respectively (guard 2's
/// other two halves, guard 1 is what happens in their absence — this
/// label is inert in a schema-free context).
pub(crate) const SCHEMA_TYPE_LABEL: &str = "schema:type";

/// Ceiling on the number of declared types — a schema is meant to be a
/// small, hand-authored vocabulary, not an import target; this bounds
/// `PUT`-time validation work and every `GET`'s response size.
pub(crate) const MAX_SCHEMA_TYPES: usize = 500;

/// Ceiling on the number of declared relations, same rationale as
/// [`MAX_SCHEMA_TYPES`].
pub(crate) const MAX_SCHEMA_RELATIONS: usize = 500;

/// Ceiling on one relation's `domain` or `range` list — load-bearing
/// for §8's bounded `expected` string in a future issue's violation
/// diagnostics, so it is fixed here where the shape is defined.
pub(crate) const MAX_RELATION_TYPES: usize = 64;

/// The `is_a` nesting ceiling. `GroupRecord`'s `MAX_GROUP_DEPTH = 3` is
/// justified there by "deep taxonomies are filing, not addressing,"
/// which does not hold for entity types: `Brewery ⊂ Manufacturer ⊂
/// Company ⊂ Organization ⊂ Agent` is a legitimate five-deep chain, so
/// this cap sits well above the group one.
pub(crate) const MAX_TYPE_DEPTH: usize = 8;

/// off | warn | strict — storage only; nothing here enforces it. `Off`
/// is the default, meaning "no file was ever written for this context,"
/// which is also what a schema-free context's absent file already
/// means — the two are the same state on purpose (see the module doc).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SchemaMode {
    #[default]
    Off,
    Warn,
    Strict,
}

impl SchemaMode {
    /// The same spelling `#[serde(rename_all = "lowercase")]` gives this
    /// enum on the wire — for `DirectoryEntry.schema_mode`'s echo (#380),
    /// which carries the mode as a plain string rather than a nested
    /// enum so an older client reads it without knowing this type.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Warn => "warn",
            Self::Strict => "strict",
        }
    }
}

/// One declared entity type: its direct parents, if any. `Organization:
/// {}` (no `is_a`) is a legal leaf-and-root type, hence the struct-level
/// `default` — but an unrecognized field is still a hard refusal
/// ([`SchemaDocument`]'s doc explains why).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(crate) struct TypeDef {
    pub(crate) is_a: BTreeSet<String>,
}

/// One declared relation label: the type sets `strict` allows on each
/// side of an association carrying this label. Both empty (the
/// default) means the relation is declared but unconstrained.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(crate) struct RelationDef {
    pub(crate) domain: BTreeSet<String>,
    pub(crate) range: BTreeSet<String>,
}

/// `{stem}.schema.json`'s whole content. Every top-level field is
/// required — no struct-level `default`, unlike [`TypeDef`]/
/// [`RelationDef`] above — because a document missing one is exactly
/// the ambiguous-with-`mode:off` shape the module doc refuses to allow
/// a parse failure to produce silently; requiring the field instead
/// makes an incomplete document a parse refusal (quarantined, boot
/// stops) rather than a silently-defaulted one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SchemaDocument {
    pub(crate) schema: u64,
    pub(crate) mode: SchemaMode,
    pub(crate) closed_labels: bool,
    pub(crate) types: BTreeMap<String, TypeDef>,
    pub(crate) relations: BTreeMap<String, RelationDef>,
}

/// Why a document cannot install. Carries enough to name the offending
/// type/relation deterministically — the map and every child set
/// iterate in name order, so which one gets blamed never depends on
/// hash-map iteration order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SchemaViolation {
    UnknownVersion(u64),
    TooManyTypes(usize),
    TooManyRelations(usize),
    /// A relation definition literally named [`SCHEMA_TYPE_LABEL`] — ADR
    /// 0009 §6.3 guard 3, symmetric with guard 2's refusal of a
    /// `label_alias` resolving to the same reserved label (checked by
    /// `PUT /schema`, not here: an alias is registry state, not part of
    /// this document).
    ReservedRelation,
    NameTooLong(String),
    TooManyRelationTypes {
        relation: String,
        side: &'static str,
        count: usize,
    },
    /// The named type reaches itself through its own `is_a` chain — the
    /// type on the cycle the walk was at when it closed the loop.
    Cycle(String),
    /// A chain of more than [`MAX_TYPE_DEPTH`] types; the name is the
    /// TOP of that chain, reported the same whether the cap is hit
    /// descending or on the way back up (mirrors
    /// [`crate::groups::NestingViolation::TooDeep`]).
    TooDeep(String),
}

impl std::fmt::Display for SchemaViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownVersion(version) => write!(
                f,
                "schema version {version} is not {SCHEMA_VERSION}, the only version this \
                 binary reads"
            ),
            Self::TooManyTypes(count) => {
                write!(f, "{count} types is over the {MAX_SCHEMA_TYPES}-type cap")
            }
            Self::TooManyRelations(count) => write!(
                f,
                "{count} relations is over the {MAX_SCHEMA_RELATIONS}-relation cap"
            ),
            Self::ReservedRelation => write!(
                f,
                "relation '{SCHEMA_TYPE_LABEL}' is reserved for type assertions — rename it"
            ),
            Self::NameTooLong(name) => {
                write!(f, "'{name}' is over the {MAX_NAME_BYTES}-byte name cap")
            }
            Self::TooManyRelationTypes {
                relation,
                side,
                count,
            } => write!(
                f,
                "relation '{relation}' names {count} {side} types, over the \
                 {MAX_RELATION_TYPES}-type cap"
            ),
            Self::Cycle(name) => write!(f, "type '{name}' is_a reaches itself through a cycle"),
            Self::TooDeep(name) => write!(
                f,
                "the is_a chain under type '{name}' stacks more than {MAX_TYPE_DEPTH} types"
            ),
        }
    }
}

/// A validated [`SchemaDocument`] plus its precomputed `is_a` ancestor
/// closures — the document is immutable between installs, so every
/// later membership test (`strict`'s domain/range check, #381) is a
/// set lookup against this, never a live walk.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InstalledSchema {
    document: SchemaDocument,
    /// Every DECLARED type's transitive `is_a` ancestors, excluding
    /// itself. A name absent here is either undeclared (see
    /// `closure_of`) or declared with no ancestors (an empty set).
    ancestors: BTreeMap<String, BTreeSet<String>>,
}

impl InstalledSchema {
    /// The validated document — `GET /schema` (#380) serves this
    /// verbatim, and `DirectoryEntry.schema_mode` (#380) echoes its
    /// `mode` field alone.
    pub(crate) fn document(&self) -> &SchemaDocument {
        &self.document
    }

    /// The membership test §7.2 (ADR 0009) runs on every association: a
    /// declared type answers itself plus every ancestor `is_a` reaches;
    /// an UNDECLARED type name — legal everywhere, never a violation on
    /// its own (§6.2) — answers its own singleton, since it has no
    /// hierarchy above it to consult.
    #[allow(dead_code)] // called by check::SchemaEnv::build; see mod check's own allow
    pub(crate) fn closure_of(&self, type_name: &str) -> BTreeSet<String> {
        let mut closure = self.ancestors.get(type_name).cloned().unwrap_or_default();
        closure.insert(type_name.to_string());
        closure
    }
}

/// Validates a parsed document and precomputes its `is_a` closures —
/// the one gate between "bytes that parsed" and "a schema this server
/// will enforce." Every check here can only be reached with a
/// hand-edited data directory: nothing this server writes (once a
/// `PUT` route exists) can produce a document this refuses.
pub(crate) fn install(document: SchemaDocument) -> Result<InstalledSchema, SchemaViolation> {
    if document.schema != SCHEMA_VERSION {
        return Err(SchemaViolation::UnknownVersion(document.schema));
    }
    if document.types.len() > MAX_SCHEMA_TYPES {
        return Err(SchemaViolation::TooManyTypes(document.types.len()));
    }
    if document.relations.len() > MAX_SCHEMA_RELATIONS {
        return Err(SchemaViolation::TooManyRelations(document.relations.len()));
    }
    if document.relations.contains_key(SCHEMA_TYPE_LABEL) {
        return Err(SchemaViolation::ReservedRelation);
    }
    // Every name the document mentions, not just the declared keys:
    // an `is_a` parent or a `domain`/`range` entry may legitimately
    // name a type or relation this document never declares (§6.2's
    // "always accepted, never a violation" undeclared-name posture),
    // and an undeclared name is exactly as unbounded as a declared one
    // unless checked here too — otherwise `MAX_RELATION_TYPES`'s own
    // doc ("load-bearing for §8's bounded `expected` string") would not
    // hold: bounding the COUNT of a domain/range list says nothing
    // about each entry's length.
    for name in document
        .types
        .keys()
        .chain(document.relations.keys())
        .chain(document.types.values().flat_map(|def| &def.is_a))
        .chain(
            document
                .relations
                .values()
                .flat_map(|def| def.domain.iter().chain(&def.range)),
        )
    {
        if name.len() > MAX_NAME_BYTES {
            return Err(SchemaViolation::NameTooLong(name.clone()));
        }
    }
    for (relation, def) in &document.relations {
        for (side, set) in [("domain", &def.domain), ("range", &def.range)] {
            if set.len() > MAX_RELATION_TYPES {
                return Err(SchemaViolation::TooManyRelationTypes {
                    relation: relation.clone(),
                    side,
                    count: set.len(),
                });
            }
        }
    }
    let ancestors = type_ancestors(&document.types)?;
    Ok(InstalledSchema {
        document,
        ancestors,
    })
}

/// [`install`]'s `is_a` pass: one memoized `O(types + edges)` walk,
/// structurally [`crate::groups::nesting_depths`]'s `chain_depth` with
/// the memo widened from a depth to a `(depth, ancestors)` pair — the
/// depth cap still bounds the walk, and the ancestor set is what
/// [`InstalledSchema::closure_of`] serves afterward.
fn type_ancestors(
    types: &BTreeMap<String, TypeDef>,
) -> Result<BTreeMap<String, BTreeSet<String>>, SchemaViolation> {
    let mut settled: BTreeMap<&str, (usize, BTreeSet<String>)> = BTreeMap::new();
    let mut visiting: Vec<&str> = Vec::new();
    for name in types.keys() {
        closure_of_declared(types, name, &mut settled, &mut visiting)?;
    }
    Ok(settled
        .into_iter()
        .map(|(name, (_, ancestors))| (name.to_string(), ancestors))
        .collect())
}

/// One type's `(chain depth, transitive ancestors)`, itself excluded
/// from the ancestor set. An `is_a` parent that is not itself a
/// declared key (a `types` entry) is accepted as a leaf ancestor with
/// nothing further above it — the same "undeclared is never a
/// violation" posture §6.2 states for a concept's asserted type, applied
/// consistently to a schema author naming a not-yet-declared parent.
fn closure_of_declared<'a>(
    types: &'a BTreeMap<String, TypeDef>,
    name: &'a str,
    settled: &mut BTreeMap<&'a str, (usize, BTreeSet<String>)>,
    visiting: &mut Vec<&'a str>,
) -> Result<(usize, BTreeSet<String>), SchemaViolation> {
    if let Some(cached) = settled.get(name) {
        return Ok(cached.clone());
    }
    let Some(def) = types.get(name) else {
        return Ok((1, BTreeSet::new()));
    };
    if visiting.contains(&name) {
        return Err(SchemaViolation::Cycle(name.to_string()));
    }
    if visiting.len() >= MAX_TYPE_DEPTH {
        // The path down to here already stacks MAX + 1 types counting
        // this one; whatever hangs above cannot make that legal.
        return Err(SchemaViolation::TooDeep(visiting[0].to_string()));
    }
    visiting.push(name);
    let mut depth = 1;
    let mut ancestors = BTreeSet::new();
    for parent in &def.is_a {
        ancestors.insert(parent.clone());
        if types.contains_key(parent.as_str()) {
            let (parent_depth, parent_ancestors) =
                closure_of_declared(types, parent, settled, visiting)?;
            depth = depth.max(parent_depth + 1);
            ancestors.extend(parent_ancestors);
        }
    }
    visiting.pop();
    if depth > MAX_TYPE_DEPTH {
        // Name the TOP of the offending chain, like the descent guard
        // above. `visiting` holds `name`'s ancestors here (it was just
        // popped); its first frame is the chain's root, or `name`
        // itself when `name` is a root.
        let blame = visiting.first().copied().unwrap_or(name);
        return Err(SchemaViolation::TooDeep(blame.to_string()));
    }
    settled.insert(name, (depth, ancestors.clone()));
    Ok((depth, ancestors))
}

/// The exact bytes [`write_schema_bytes`] writes for a document — split
/// out from it so `PUT /schema` (#380) can hash the SAME bytes it is
/// about to persist (`sha256_hex`, for `MetaFile::schema_digest`)
/// instead of re-serializing and risking the two silently drifting
/// apart (e.g. a future serde attribute changing output without both
/// call sites being touched).
pub(crate) fn document_bytes(document: &SchemaDocument) -> io::Result<Vec<u8>> {
    serde_json::to_vec_pretty(document).map_err(io::Error::from)
}

/// Persists one schema document's already-serialized bytes via the
/// registry's staged write (fsync, rename, parent fsync): a crash
/// mid-write leaves the previous version intact. Takes bytes rather
/// than a `&SchemaDocument` so a caller that already hashed the bytes
/// (`PUT /schema`, #380) writes exactly what it hashed.
pub(crate) fn write_schema_bytes(dir: &Path, stem: &str, bytes: &[u8]) -> io::Result<()> {
    write_atomic(&schema_path(dir, stem), bytes)
}

/// Serializes then persists in one call — the test round trip's
/// convenience wrapper; `PUT /schema` (#380) uses the two halves above
/// directly so it can hash what it writes.
#[cfg(test)]
fn write_schema(dir: &Path, stem: &str, document: &SchemaDocument) -> io::Result<()> {
    write_schema_bytes(dir, stem, &document_bytes(document)?)
}

/// Loads and verifies `{stem}.schema.json` against what `ContextMeta`
/// recorded as its digest — the boot-time consistency check ADR 0009
/// §5.2 requires so a crash between `write_meta`'s revision bump and
/// this file's own `write_atomic` (a window §5.2 cannot close by
/// ordering alone, since the two are separate durable writes) is always
/// detectable rather than serving a `strict` context whose enforced
/// rules do not match what its own revision claims to be enforcing.
///
/// `Ok(None)` is the one non-refusal outcome: no file and no recorded
/// digest, i.e. a context that never had a schema — `off`, byte-
/// identical to every context before this file family existed. Every
/// other outcome is `Err`, with a message naming what is wrong and how
/// to recover, mirroring [`crate::groups::scan_groups`]'s tone:
///
/// - unreadable file → refuse (a transient permission hiccup must not
///   be mistaken for "never had one")
/// - a recorded digest with no file, or a file whose digest disagrees
///   with the recorded one (in EITHER direction) → refuse: the digest
///   is the one artifact that can catch an interrupted `PUT` a valid-
///   looking file would otherwise hide
/// - bytes that read but do not parse → the mangled bytes are set aside
///   as `{stem}.schema.corrupt` for hand recovery (evidence, like
///   `scan_groups`' own quarantine) when `set_aside_corrupt` is `true`,
///   and refuse either way — UNLIKE `scan_groups`, there is no
///   fresh-empty-record fallback here (see the module doc)
/// - a document [`install`] refuses → refuse (a hand-edited file only)
///
/// `set_aside_corrupt` gates the ONE side effect this otherwise
/// read-only function has: `taguru inspect` audits directories —
/// often backups, sometimes on read-only media — and must never write
/// to what it is auditing, so it passes `false`. Every other caller
/// (boot, a cold load) passes `true`. Even then, a failed quarantine
/// write is only logged, never returned as this call's error: the
/// caller asked "does this schema load," and a hand-recovery copy
/// failing to land must not overwrite that diagnosis with an
/// unrelated write failure.
pub(crate) fn load_schema(
    dir: &Path,
    stem: &str,
    recorded_digest: Option<&str>,
    set_aside_corrupt: bool,
) -> io::Result<Option<InstalledSchema>> {
    let path = schema_path(dir, stem);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return match recorded_digest {
                None => Ok(None),
                Some(digest) => Err(io::Error::other(format!(
                    "schema digest '{digest}' is recorded for '{stem}' but {} is missing — \
                     restore the file, or clear the recorded digest by rewriting the schema \
                     and start again",
                    path.display()
                ))),
            };
        }
        Err(error) => {
            return Err(io::Error::new(
                error.kind(),
                format!(
                    "schema '{}' is unreadable: {error} — restore the file (or move it out \
                     of the data directory) and start again",
                    path.display()
                ),
            ));
        }
    };
    let actual_digest = sha256_hex(&bytes);
    if recorded_digest != Some(actual_digest.as_str()) {
        return Err(io::Error::other(format!(
            "schema '{}' digest does not match what its context recorded (recorded {}, on \
             disk {actual_digest}) — an update may have been interrupted mid-write; restore \
             the intended file so the two agree, or clear both and start again",
            path.display(),
            recorded_digest.unwrap_or("none"),
        )));
    }
    let document: SchemaDocument = match serde_json::from_slice(&bytes) {
        Ok(document) => document,
        Err(error) => {
            let set_aside = schema_corrupt_path(dir, stem);
            if set_aside_corrupt {
                tracing::warn!(
                    schema = %path.display(),
                    bytes_at = %set_aside.display(),
                    %error,
                    "schema file does not parse; refusing the boot and setting the bytes aside"
                );
                // Evidence, not the diagnosis: a failed quarantine write
                // must not replace "does not parse" with an unrelated
                // io error below (see this function's own doc).
                if let Err(set_aside_error) = fs::write(&set_aside, &bytes) {
                    tracing::warn!(
                        schema = %path.display(),
                        bytes_at = %set_aside.display(),
                        error = %set_aside_error,
                        "schema bytes not set aside"
                    );
                }
            }
            return Err(io::Error::other(format!(
                "schema '{}' does not parse: {error} — a schema never falls back to an \
                 empty (i.e. mode: off) record on parse failure, since that would silently \
                 disable strict enforcement{}",
                path.display(),
                if set_aside_corrupt {
                    format!(
                        " (the bytes were set aside as {} for hand recovery)",
                        set_aside.display()
                    )
                } else {
                    String::new()
                },
            )));
        }
    };
    install(document).map(Some).map_err(|violation| {
        io::Error::other(format!(
            "schema '{}' does not validate: {violation} — a hand-edited file only; nothing \
             this server writes can produce this shape",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, private scratch directory for one test — same shape as
    /// `crate::registry::test_support::scratch_dir`, duplicated here
    /// rather than imported: that module is private to `registry`.
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("taguru-schema-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn valid() -> SchemaDocument {
        SchemaDocument {
            schema: SCHEMA_VERSION,
            mode: SchemaMode::Strict,
            closed_labels: false,
            types: BTreeMap::from([
                (
                    "Brewery".to_string(),
                    TypeDef {
                        is_a: BTreeSet::from(["Organization".to_string()]),
                    },
                ),
                ("Organization".to_string(), TypeDef::default()),
                ("Person".to_string(), TypeDef::default()),
            ]),
            relations: BTreeMap::from([(
                "杜氏".to_string(),
                RelationDef {
                    domain: BTreeSet::from(["Brewery".to_string()]),
                    range: BTreeSet::from(["Person".to_string()]),
                },
            )]),
        }
    }

    #[test]
    fn a_valid_document_installs_and_the_closure_includes_ancestors() {
        let installed = install(valid()).unwrap();
        assert_eq!(
            installed.closure_of("Brewery"),
            BTreeSet::from(["Brewery".to_string(), "Organization".to_string()])
        );
        assert_eq!(
            installed.closure_of("Organization"),
            BTreeSet::from(["Organization".to_string()])
        );
    }

    #[test]
    fn an_undeclared_type_name_closes_over_itself_only_and_is_never_a_violation() {
        let installed = install(valid()).unwrap();
        assert_eq!(
            installed.closure_of("Distillery"),
            BTreeSet::from(["Distillery".to_string()])
        );
    }

    #[test]
    fn an_unread_version_refuses() {
        let mut document = valid();
        document.schema = SCHEMA_VERSION + 1;
        assert_eq!(
            install(document),
            Err(SchemaViolation::UnknownVersion(SCHEMA_VERSION + 1))
        );
    }

    #[test]
    fn a_relation_named_the_reserved_type_label_refuses() {
        let mut document = valid();
        document
            .relations
            .insert(SCHEMA_TYPE_LABEL.to_string(), RelationDef::default());
        assert_eq!(install(document), Err(SchemaViolation::ReservedRelation));
    }

    #[test]
    fn document_bytes_hashes_to_what_write_schema_bytes_persists() {
        let dir = scratch_dir("schema-bytes");
        fs::create_dir_all(&dir).unwrap();
        let document = valid();
        let bytes = document_bytes(&document).unwrap();
        write_schema_bytes(&dir, "sake", &bytes).unwrap();
        assert_eq!(fs::read(schema_path(&dir, "sake")).unwrap(), bytes);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unknown_field_at_rest_refuses_to_parse() {
        let bytes = br#"{"schema":1,"mode":"off","closed_labels":false,"types":{},"relations":{},"extra":true}"#;
        assert!(serde_json::from_slice::<SchemaDocument>(bytes).is_err());
    }

    #[test]
    fn a_missing_top_level_field_refuses_to_parse() {
        let bytes = br#"{"schema":1,"mode":"off","types":{},"relations":{}}"#;
        assert!(serde_json::from_slice::<SchemaDocument>(bytes).is_err());
    }

    #[test]
    fn too_many_types_refuses() {
        let mut document = valid();
        document.types = (0..MAX_SCHEMA_TYPES + 1)
            .map(|n| (format!("T{n}"), TypeDef::default()))
            .collect();
        assert_eq!(
            install(document),
            Err(SchemaViolation::TooManyTypes(MAX_SCHEMA_TYPES + 1))
        );
    }

    #[test]
    fn too_many_relations_refuses() {
        let mut document = valid();
        document.relations = (0..MAX_SCHEMA_RELATIONS + 1)
            .map(|n| (format!("R{n}"), RelationDef::default()))
            .collect();
        assert_eq!(
            install(document),
            Err(SchemaViolation::TooManyRelations(MAX_SCHEMA_RELATIONS + 1))
        );
    }

    #[test]
    fn too_many_relation_types_on_one_side_refuses() {
        let mut document = valid();
        document.relations.insert(
            "多対".to_string(),
            RelationDef {
                domain: (0..MAX_RELATION_TYPES + 1)
                    .map(|n| format!("D{n}"))
                    .collect(),
                range: BTreeSet::new(),
            },
        );
        assert_eq!(
            install(document),
            Err(SchemaViolation::TooManyRelationTypes {
                relation: "多対".to_string(),
                side: "domain",
                count: MAX_RELATION_TYPES + 1,
            })
        );
    }

    #[test]
    fn a_name_over_the_byte_cap_refuses() {
        let mut document = valid();
        let long = "x".repeat(MAX_NAME_BYTES + 1);
        document.types.insert(long.clone(), TypeDef::default());
        assert_eq!(install(document), Err(SchemaViolation::NameTooLong(long)));
    }

    #[test]
    fn a_direct_self_cycle_refuses() {
        let mut document = valid();
        document.types.insert(
            "Loop".to_string(),
            TypeDef {
                is_a: BTreeSet::from(["Loop".to_string()]),
            },
        );
        assert_eq!(
            install(document),
            Err(SchemaViolation::Cycle("Loop".to_string()))
        );
    }

    #[test]
    fn a_two_type_cycle_refuses() {
        let mut document = valid();
        document.types.insert(
            "A".to_string(),
            TypeDef {
                is_a: BTreeSet::from(["B".to_string()]),
            },
        );
        document.types.insert(
            "B".to_string(),
            TypeDef {
                is_a: BTreeSet::from(["A".to_string()]),
            },
        );
        assert!(matches!(install(document), Err(SchemaViolation::Cycle(_))));
    }

    /// A chain of exactly [`MAX_TYPE_DEPTH`] types is legal; one more
    /// is refused. `T0` (no `is_a`) sorts first and is the walk's own
    /// entry point for T1..T8 too — each resolves its single parent as
    /// an immediate memo hit within the very same `install` call, so
    /// `T8`, the one new frame that pushes the depth over the cap, is
    /// also the walk's own outermost frame at that moment: it blames
    /// itself. See the next test for the case where blame must bubble
    /// PAST a memoized node to an outer frame instead.
    #[test]
    fn a_full_depth_chain_installs_and_one_more_is_refused() {
        let mut document = valid();
        document.types.clear();
        document.relations.clear();
        // T0 <- T1 <- ... <- T(MAX_TYPE_DEPTH - 1), a chain MAX_TYPE_DEPTH long.
        for n in 0..MAX_TYPE_DEPTH {
            let is_a = if n == 0 {
                BTreeSet::new()
            } else {
                BTreeSet::from([format!("T{}", n - 1)])
            };
            document.types.insert(format!("T{n}"), TypeDef { is_a });
        }
        assert!(install(document.clone()).is_ok());

        document.types.insert(
            format!("T{MAX_TYPE_DEPTH}"),
            TypeDef {
                is_a: BTreeSet::from([format!("T{}", MAX_TYPE_DEPTH - 1)]),
            },
        );
        assert_eq!(
            install(document),
            Err(SchemaViolation::TooDeep(format!("T{MAX_TYPE_DEPTH}")))
        );
    }

    /// `crate::groups`' own
    /// `too_deep_blames_the_chain_top_even_through_a_memoized_child`,
    /// mirrored: `a0..a{MAX_TYPE_DEPTH-1}` settles at exactly
    /// `MAX_TYPE_DEPTH` (legal) while walked first in name order. Then
    /// `p_top -> q_mid -> a{last}` is discovered later; `q_mid`'s
    /// combined depth (the cached chain's depth plus itself) already
    /// exceeds the cap the moment it looks up the memoized `a{last}`,
    /// so the descent guard never fires — the cap trips on `q_mid`'s
    /// own way back up, one frame short of where the NEW discovery
    /// started. The blame must still be the TOP of the newly-discovered
    /// chain (`p_top`), not the memoized-lookup site `q_mid`.
    #[test]
    fn too_deep_blames_the_new_chains_top_even_through_a_memoized_type() {
        let mut document = valid();
        document.types.clear();
        document.relations.clear();
        for n in 0..MAX_TYPE_DEPTH {
            let is_a = if n == 0 {
                BTreeSet::new()
            } else {
                BTreeSet::from([format!("a{}", n - 1)])
            };
            document.types.insert(format!("a{n}"), TypeDef { is_a });
        }
        document.types.insert(
            "q_mid".to_string(),
            TypeDef {
                is_a: BTreeSet::from([format!("a{}", MAX_TYPE_DEPTH - 1)]),
            },
        );
        document.types.insert(
            "p_top".to_string(),
            TypeDef {
                is_a: BTreeSet::from(["q_mid".to_string()]),
            },
        );
        assert_eq!(
            install(document),
            Err(SchemaViolation::TooDeep("p_top".to_string())),
            "names the new chain's top, not the memoized-lookup site q_mid"
        );
    }

    #[test]
    fn write_then_load_round_trips_and_matches_the_written_digest() {
        let dir = scratch_dir("schema-roundtrip");
        fs::create_dir_all(&dir).unwrap();
        let document = valid();
        write_schema(&dir, "sake", &document).unwrap();
        let bytes = fs::read(schema_path(&dir, "sake")).unwrap();
        let digest = sha256_hex(&bytes);

        let loaded = load_schema(&dir, "sake", Some(&digest), true)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.document(), &document);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_file_and_no_recorded_digest_is_off() {
        let dir = scratch_dir("schema-off");
        fs::create_dir_all(&dir).unwrap();
        assert!(load_schema(&dir, "sake", None, true).unwrap().is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_recorded_digest_with_no_file_refuses() {
        let dir = scratch_dir("schema-missing");
        fs::create_dir_all(&dir).unwrap();
        let error = load_schema(&dir, "sake", Some("deadbeef"), true).unwrap_err();
        assert!(error.to_string().contains("is missing"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_digest_mismatch_refuses_in_either_direction() {
        let dir = scratch_dir("schema-mismatch");
        fs::create_dir_all(&dir).unwrap();
        write_schema(&dir, "sake", &valid()).unwrap();

        // Stale recorded digest under fresh content.
        let error = load_schema(&dir, "sake", Some("deadbeef"), true).unwrap_err();
        assert!(error.to_string().contains("does not match"), "{error}");

        // A stray file with nothing recorded for it at all.
        let error = load_schema(&dir, "sake", None, true).unwrap_err();
        assert!(error.to_string().contains("does not match"), "{error}");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unparseable_bytes_are_quarantined_and_the_boot_refuses() {
        let dir = scratch_dir("schema-corrupt");
        fs::create_dir_all(&dir).unwrap();
        let path = schema_path(&dir, "sake");
        fs::write(&path, b"not json").unwrap();
        let digest = sha256_hex(b"not json");

        let error = load_schema(&dir, "sake", Some(&digest), true).unwrap_err();
        assert!(error.to_string().contains("does not parse"), "{error}");
        assert_eq!(
            fs::read(schema_corrupt_path(&dir, "sake")).unwrap(),
            b"not json"
        );
        // The original path is untouched — no empty-record fallback.
        assert_eq!(fs::read(&path).unwrap(), b"not json");

        let _ = fs::remove_dir_all(&dir);
    }

    /// `set_aside_corrupt: false` is `taguru inspect`'s contract: it
    /// audits directories (often backups, sometimes read-only media)
    /// and must never write to them, even to quarantine evidence.
    #[test]
    fn set_aside_corrupt_false_never_writes_the_quarantine_file() {
        let dir = scratch_dir("schema-corrupt-readonly");
        fs::create_dir_all(&dir).unwrap();
        let path = schema_path(&dir, "sake");
        fs::write(&path, b"not json").unwrap();
        let digest = sha256_hex(b"not json");

        let error = load_schema(&dir, "sake", Some(&digest), false).unwrap_err();
        assert!(error.to_string().contains("does not parse"), "{error}");
        assert!(
            !schema_corrupt_path(&dir, "sake").exists(),
            "inspect must never write, not even a quarantine copy"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// `load_schema`'s last failure path: the file reads, its digest
    /// matches, and it parses — but `install` refuses it (a hand-edited
    /// file naming an unread `schema` version, here). No quarantine:
    /// the bytes are not mangled, so there is nothing to set aside.
    #[test]
    fn a_parseable_but_invalid_document_refuses_without_quarantine() {
        let dir = scratch_dir("schema-invalid");
        fs::create_dir_all(&dir).unwrap();
        let bytes =
            br#"{"schema":99,"mode":"off","closed_labels":false,"types":{},"relations":{}}"#;
        fs::write(schema_path(&dir, "sake"), bytes).unwrap();
        let digest = sha256_hex(bytes);

        let error = load_schema(&dir, "sake", Some(&digest), true).unwrap_err();
        assert!(error.to_string().contains("does not validate"), "{error}");
        assert!(
            !schema_corrupt_path(&dir, "sake").exists(),
            "a parseable file is not corrupt bytes; nothing is set aside"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
