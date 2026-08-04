//! ADR 0009 §7.2's `TypeEnv` union and the one pure `schema_issues`
//! function every write entrance shares. The ADR's own Appendix names
//! the trap this module exists to close: "S3 must land *before* S4/S5,
//! not alongside them. If the two write entrances build their own
//! schema checks in parallel they will diverge." `predicted_alias_rejection`/
//! `preview_batch` sharing (`src/ingest.rs`) is the existing precedent
//! for this shape, one level down.
//!
//! Nothing in this module writes anything or reads anything but the one
//! `Context` it is handed — `schema_issues` is pure, and [`SchemaEnv`]'s
//! builder is the only side that touches the graph, once, before any op
//! is judged.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use taguru::context::{Association, Context};

use crate::api::Issue;
use crate::registry::AssocOp;

use super::{InstalledSchema, SCHEMA_TYPE_LABEL, SchemaMode};

/// Cap on how many type names a violation message enumerates — a
/// relation's declared `domain`/`range` is itself capped
/// ([`super::MAX_RELATION_TYPES`]) but a concept's *asserted* type set
/// is not (nothing stops a source from asserting `schema:type` on one
/// concept many times over), so the message side needs its own,
/// independent bound.
const MAX_ISSUE_TYPE_NAMES: usize = 8;

/// What [`SchemaEnv::build`] needs beyond the live `Context` it is
/// handed — bundled into one struct so the builder itself stays a
/// two-argument function rather than tripping clippy's
/// too-many-arguments lint.
pub(crate) struct SchemaCheckInput<'a> {
    /// The context's installed document, already validated (`super::install`).
    pub(crate) schema: Arc<InstalledSchema>,
    /// The exact ops this write is about to apply — `corrected_associations`'
    /// output on the import path (ADR 0009 §7.2 step 2), the request's own
    /// array on the associations path (§7.3).
    pub(crate) ops: &'a [AssocOp],
    /// This write's own inline label-alias declaration — `batch.labels`
    /// on the import path, empty on the associations path (which has no
    /// such field). §7.2 step 3's "other half of the union," and ADR
    /// 0009 §6.3 guard 2's batch-local bullet.
    pub(crate) declared_labels: &'a BTreeMap<String, String>,
    /// The source this write is about to retract before applying, if
    /// any. `Some` on the import path (`apply_batch` retracts
    /// `batch.source` before applying — `src/ingest.rs:2354-2357`);
    /// `None` on the associations path, which never retracts (§7.3) —
    /// there the union is a plain one, with no exclusion clause.
    pub(crate) retracted_source: Option<&'a str>,
}

/// One concept's type population for this check: what was actually
/// asserted (declared or not — an undeclared name is still legal,
/// ADR 0009 §6.2), and that set's `is_a` closure — the only form
/// [`side_violates`] ever tests against. Kept separate rather than
/// storing only the closure because §8.2's `actual` message must name
/// what was asserted, never the expanded closure ("the expanded set's
/// size is a schema-authoring accident, not information the corrector
/// needs" — the same reasoning §8.2 gives for `expected` not expanding).
struct TypeAssertions {
    asserted: BTreeSet<String>,
    expanded: BTreeSet<String>,
}

/// The union ADR 0009 §7.2 names this section after: `(post-retraction
/// live state) ∪ (this write's own type assertions)`, built once,
/// before any op is judged — so line order inside a batch cannot
/// matter (§7.2's own "why line order cannot matter" paragraph).
///
/// Every map here is scoped to exactly the spellings `SchemaCheckInput::ops`
/// mentions — [`SchemaEnv`] is not a general-purpose alias/type index,
/// it is the minimal read a `schema_issues` call over those exact ops
/// needs, no more.
pub(crate) struct SchemaEnv {
    schema: Arc<InstalledSchema>,
    /// Every op's `label` spelling, resolved through live
    /// `label_aliases` ∪ `declared_labels` (§7.2 step 3). A spelling
    /// absent here (only possible when `mode == off`, see
    /// [`SchemaEnv::build`]) resolves to itself.
    resolved_labels: BTreeMap<String, String>,
    /// Every op's `subject`/`object` spelling, resolved to its
    /// canonical concept name via [`Context::canonical_concept`] — or
    /// itself, when the spelling is not yet interned (it will become
    /// the literal canonical the moment this write lands). Concept
    /// resolution is exact, never through `declared_labels`-style
    /// batch-local aliasing: a batch's own `concepts` aliases can never
    /// merge a spelling its own associations use (`add_alias` refuses
    /// that as a `Conflict`, caught upstream by
    /// `predicted_alias_rejection`), so live aliasing is the only basis
    /// that applies here.
    concept_canon: BTreeMap<String, String>,
    /// Canonical concept name → its type population — §7.2 steps 4–5.
    /// Absent means untyped (never a violation, §6.1).
    types: BTreeMap<String, TypeAssertions>,
    /// This write's own inline label-alias declaration, unresolved —
    /// used only by [`schema_issues`]' guard-2 pass, which asks "does
    /// this declaration itself name the reserved label," not "what
    /// does some other spelling resolve to."
    declared_labels: BTreeMap<String, String>,
}

impl SchemaEnv {
    /// Builds the union `schema_issues` judges against. `context` is
    /// read exactly once for the live half (a single
    /// [`Context::query_any`] call, only when `mode != off` and at
    /// least one op mentions a concept in fact position) — ADR 0009
    /// §7.2 step 1's "return before reading a single association" is
    /// honored by skipping that read entirely, not by discarding its
    /// result, when the document's mode is `off`. `declared_labels` is
    /// captured either way, since guard 2 (ADR 0009 §6.3) refuses a
    /// reserved-label batch alias regardless of mode.
    pub(crate) fn build(context: &Context, input: SchemaCheckInput<'_>) -> Self {
        let SchemaCheckInput {
            schema,
            ops,
            declared_labels,
            retracted_source,
        } = input;
        let declared_labels = declared_labels.clone();

        if schema.document().mode == SchemaMode::Off {
            return Self {
                schema,
                resolved_labels: BTreeMap::new(),
                concept_canon: BTreeMap::new(),
                types: BTreeMap::new(),
                declared_labels,
            };
        }

        // Live label_aliases() is already flattened to true canonicals
        // (`add_alias` resolves through the id map at creation time,
        // `src/context/alias.rs:47`) — snapshotted as owned strings so
        // it can outlive the borrow of `context`.
        let live_labels: BTreeMap<String, String> = context
            .label_aliases()
            .into_iter()
            .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
            .collect();

        let mut resolved_labels = BTreeMap::new();
        for op in ops {
            resolved_labels.entry(op.label.clone()).or_insert_with(|| {
                resolve_label(&op.label, &live_labels, &declared_labels).to_string()
            });
        }

        let mut concept_canon = BTreeMap::new();
        for op in ops {
            for spelling in [op.subject.as_str(), op.object.as_str()] {
                if concept_canon.contains_key(spelling) {
                    continue;
                }
                let canonical = context
                    .canonical_concept(spelling)
                    .map(str::to_string)
                    .unwrap_or_else(|| spelling.to_string());
                concept_canon.insert(spelling.to_string(), canonical);
            }
        }

        let is_type_op = |op: &AssocOp| {
            resolved_labels.get(&op.label).map(String::as_str) == Some(SCHEMA_TYPE_LABEL)
        };

        let mut asserted: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

        // Batch half (§7.2 step 4): every type_op's (subject → object),
        // unconditionally — cheap, since it only reads `ops` and the
        // `concept_canon` map already built above. The object is a
        // type NAME by convention, but `Context` interns it exactly
        // like any other association object — a live concept alias of
        // it (e.g. "Organisation" → "Organization") resolves at write
        // time just as the subject side does, so both sides go through
        // `concept_canon` here. Skipping this on the object would file
        // the assertion under the alias spelling, and `closure_of`
        // would then see an undeclared singleton disjoint from any
        // `domain`/`range` naming the true canonical type — a false
        // `strict` refusal of a write that is in fact valid.
        for op in ops {
            if is_type_op(op) {
                let subject = concept_canon
                    .get(&op.subject)
                    .cloned()
                    .unwrap_or_else(|| op.subject.clone());
                let object_type = concept_canon
                    .get(&op.object)
                    .cloned()
                    .unwrap_or_else(|| op.object.clone());
                asserted.entry(subject).or_default().insert(object_type);
            }
        }

        // Live half: only concepts fact_ops actually mention — the
        // narrowest read that answers §7.2 step 4's question.
        let fact_concepts: BTreeSet<&str> = ops
            .iter()
            .filter(|op| !is_type_op(op))
            .flat_map(|op| [op.subject.as_str(), op.object.as_str()])
            .map(|spelling| {
                concept_canon
                    .get(spelling)
                    .map(String::as_str)
                    .unwrap_or(spelling)
            })
            .collect();

        let subjects: Vec<&str> = fact_concepts.into_iter().collect();
        for (concept, names) in live_type_assertions(context, &subjects, retracted_source) {
            asserted.entry(concept).or_default().extend(names);
        }

        let types = asserted
            .into_iter()
            .map(|(concept, names)| {
                let mut expanded = BTreeSet::new();
                for name in &names {
                    expanded.extend(schema.closure_of(name));
                }
                (
                    concept,
                    TypeAssertions {
                        asserted: names,
                        expanded,
                    },
                )
            })
            .collect();

        Self {
            schema,
            resolved_labels,
            concept_canon,
            types,
            declared_labels,
        }
    }

    fn types_for(&self, spelling: &str) -> Option<&TypeAssertions> {
        let canonical = self
            .concept_canon
            .get(spelling)
            .map(String::as_str)
            .unwrap_or(spelling);
        self.types.get(canonical)
    }
}

/// The live half of ADR 0009 §7.2 step 4's union, factored out of
/// [`SchemaEnv::build`] so ADR 0009 §12's read-side query filter can
/// share it verbatim rather than re-deriving "what is this concept's
/// asserted type" by hand: every live `schema:type` edge whose subject
/// is one of `concepts`, minus whichever ones this write is about to
/// retract with its own source (the exclusion clause; `None` on every
/// read-only caller and on the associations write path, which never
/// retracts — a plain union there). Scoped to exactly the spellings the
/// caller cares about, never a general-purpose type index — an empty
/// `concepts` costs one allocation, not a full scan (`Context::query_any`
/// with all three positions empty returns every edge in the context,
/// `src/context/query.rs:69-73`).
pub(crate) fn live_type_assertions(
    context: &Context,
    concepts: &[&str],
    retracted_source: Option<&str>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut asserted: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    if concepts.is_empty() {
        return asserted;
    }
    for assoc in context.query_any(concepts, &[SCHEMA_TYPE_LABEL], &[]) {
        if !survives_retraction(&assoc, retracted_source) {
            continue;
        }
        asserted
            .entry(assoc.subject.clone())
            .or_default()
            .insert(assoc.object.clone());
    }
    asserted
}

/// The read-side sibling of [`SchemaEnv::build`]'s live half (ADR 0009
/// §12): every live-asserted type on `concepts`, `is_a`-expanded through
/// `schema`'s precomputed ancestor closures — the exact predicate §7.2
/// step 5 already applies to `domain`/`range`, reused here so `query`'s
/// `subject_types`/`object_types` filter can never disagree with what
/// `strict` itself would treat as this concept's type. No retraction
/// exclusion (a read has no pending write to exclude) and no batch half
/// (a read has no ops of its own) — a plain live union, `is_a`-expanded.
pub(crate) fn expanded_type_sets(
    context: &Context,
    schema: &InstalledSchema,
    concepts: &[&str],
) -> BTreeMap<String, BTreeSet<String>> {
    live_type_assertions(context, concepts, None)
        .into_iter()
        .map(|(concept, names)| {
            let mut expanded = BTreeSet::new();
            for name in &names {
                expanded.extend(schema.closure_of(name));
            }
            (concept, expanded)
        })
        .collect()
}

/// §7.2 step 3's label resolution: live `label_aliases` ∪ this write's
/// own `declared` (`batch.labels`). `declared` is consulted first — it
/// is what this write is about to make true — falling back to `live`.
/// A live alias is always already flattened to its true canonical (see
/// [`SchemaEnv::build`]'s doc), so in the overwhelming case at most one
/// hop through `declared` followed by at most one hop through `live`
/// resolves a spelling; the small bounded loop below exists only to be
/// defensive against a same-write chain across multiple `declared`
/// entries, not because deeper chains are expected or specially
/// supported.
fn resolve_label<'a>(
    spelling: &'a str,
    live: &'a BTreeMap<String, String>,
    declared: &'a BTreeMap<String, String>,
) -> &'a str {
    const MAX_HOPS: usize = 4;
    let mut current = spelling;
    for _ in 0..MAX_HOPS {
        let next = declared
            .get(current)
            .or_else(|| live.get(current))
            .map(String::as_str);
        match next {
            Some(next) if next != current => current = next,
            _ => break,
        }
    }
    current
}

/// Whether a live type edge survives the retraction this write is
/// about to perform before applying — ADR 0009 §7.2 step 4's live-half
/// exclusion: "excluding any type assertion whose only attribution is
/// `batch.source`... the judgment must be against the state the graph
/// is *about to be in*." `Context::retract_source` subtracts exactly
/// this source's own count from the edge
/// (`edge.count.saturating_sub(record.count)`,
/// `src/context/sources.rs:143-150`), so the precise predicate is
/// "count minus this source's count is still positive" — stricter than
/// the ADR's own prose reading ("whose only attribution is
/// `batch.source`"), which would wrongly drop an edge that also carries
/// unsourced weight surviving the retraction. Deliberately reads
/// `count`, never `weight`: `Attribution::weight` is a source's raw
/// cumulative sum, `Association::weight` is the edge's *average*
/// (`sum / count`) — neither is what "how many times" means here.
fn survives_retraction(assoc: &Association, retracted_source: Option<&str>) -> bool {
    if assoc.count == 0 {
        // Already dead (`query_any` returns retracted edges too); never
        // a live type assertion regardless of retraction.
        return false;
    }
    let Some(source) = retracted_source else {
        return true;
    };
    let retracted_count = assoc
        .attributions
        .iter()
        .find(|attribution| attribution.source == source)
        .map(|attribution| attribution.count)
        .unwrap_or(0);
    assoc.count.saturating_sub(retracted_count) > 0
}

/// A `domain`/`range` violation is `declared` non-empty (an unconstrained
/// side never violates, §7.2 step 6), the concept's expanded type set
/// non-empty (untyped never violates, §6.1), and the two disjoint. An
/// undeclared type name behaves exactly like a declared one here — its
/// closure is just its own singleton (§6.2) — so no special case is
/// needed for it.
fn side_violates(declared: &BTreeSet<String>, types: Option<&TypeAssertions>) -> bool {
    if declared.is_empty() {
        return false;
    }
    let Some(types) = types else {
        return false;
    };
    if types.expanded.is_empty() {
        return false;
    }
    declared.is_disjoint(&types.expanded)
}

/// The result of one `schema_issues` call, kept as two lists rather
/// than one flat `Vec<Issue>` because they answer to different gates
/// and different write-entrance dispatch (ADR 0009 §7.2 step 7, §6.3):
/// a caller must never fold them together before deciding what to do
/// with each.
pub(crate) struct SchemaCheck {
    /// ADR 0009 §6.3 guard 2's batch-local bullet: this write's own
    /// `declared_labels` names the reserved label as some spelling's
    /// canonical. Gated on "a schema document exists," never on mode —
    /// a refusal in `off` and `warn` alike, since guard 2 is about
    /// namespace ownership, not enforcement. Non-empty here means the
    /// whole write refuses regardless of what `violations` holds.
    pub(crate) reserved: Vec<Issue>,
    /// ADR 0009 §7.2 step 6's domain/range judgment plus §6.4's
    /// `closed_labels` check, over `fact_ops` only. Empty whenever
    /// `mode == off`. What a caller does with a non-empty list is
    /// mode-dispatched by §7.2 step 7: `strict` rejects, `warn` reports
    /// and proceeds.
    pub(crate) violations: Vec<Issue>,
}

impl SchemaCheck {
    /// §7.2 step 7's dispatch, named once (#388, S10 of #218's ADR
    /// 0009 split §15) so the two write entrances' metrics can never
    /// describe the same check differently than the check itself
    /// dispatched. `mode` only breaks the `Warned`/`Refused` tie for a
    /// non-empty `violations` — `reserved` refuses regardless of mode
    /// (guard 2's own rule), and `violations` is empty by construction
    /// whenever `mode == off`, so that case falls straight to `Ok`.
    pub(crate) fn outcome(&self, mode: SchemaMode) -> crate::metrics::SchemaOutcome {
        use crate::metrics::SchemaOutcome;
        if !self.reserved.is_empty() {
            SchemaOutcome::Refused
        } else if self.violations.is_empty() {
            SchemaOutcome::Ok
        } else if mode == SchemaMode::Strict {
            SchemaOutcome::Refused
        } else {
            SchemaOutcome::Warned
        }
    }
}

/// `Issue.path`'s grammar. A write entrance names a coordinate within
/// its own request body (ADR 0009 §8.2's two prefixes); ADR 0009 §10's
/// audit has no such coordinate — the offending association travels
/// alongside its issues in `SchemaViolationOut`, so `path` there need
/// only name which side of that one association fired.
pub(crate) enum IssuePath<'a> {
    /// `{prefix}associations[{index}].{side}` — `""` for
    /// `POST /contexts/{name}/associations` (paths read
    /// `associations[{i}]...`), `"batches[{b}]."` for
    /// `POST /import`/`taguru import` (paths read
    /// `batches[{b}].associations[{a}]...`).
    Request { prefix: &'a str },
    /// `subject` / `object` / `label` alone — §10's audit calls
    /// [`schema_issues`] once per association (`ops.len() == 1`), so no
    /// request-body index ever applies.
    Edge,
}

impl IssuePath<'_> {
    fn associations_field(&self, index: usize, side: &str) -> String {
        match self {
            Self::Request { prefix } => format!("{prefix}associations[{index}].{side}"),
            Self::Edge => side.to_string(),
        }
    }

    fn labels_field(&self, alias: &str) -> String {
        match self {
            Self::Request { prefix } => format!("{prefix}labels['{alias}']"),
            Self::Edge => format!("labels['{alias}']"),
        }
    }
}

/// The one function every schema-checking write entrance calls — S4
/// (#382, `predicted_schema_rejection`/`preview_batch`) and S5 (#383,
/// the associations handler's pre-write arm) alike, so the two
/// entrances cannot drift apart (this module's own doc); S7 (#385)'s
/// audit reuses it too, one association at a time, via
/// [`IssuePath::Edge`]. `env` must have been built from this exact
/// `ops` slice — [`SchemaEnv`]'s maps only cover the spellings those
/// ops mention.
pub(crate) fn schema_issues(env: &SchemaEnv, ops: &[AssocOp], path: IssuePath<'_>) -> SchemaCheck {
    let reserved = env
        .declared_labels
        .iter()
        .filter(|(_, canonical)| canonical.as_str() == SCHEMA_TYPE_LABEL)
        .map(|(alias, _)| reserved_alias_issue(&path, alias))
        .collect();

    let document = env.schema.document();
    if document.mode == SchemaMode::Off {
        return SchemaCheck {
            reserved,
            violations: Vec::new(),
        };
    }

    let mut violations = Vec::new();
    for (index, op) in ops.iter().enumerate() {
        let resolved_label = env
            .resolved_labels
            .get(&op.label)
            .map(String::as_str)
            .unwrap_or(&op.label);
        // type_ops are never judged (§7.2 step 6) — this is exactly
        // what makes the reserved-label partition load-bearing rather
        // than cosmetic.
        if resolved_label == SCHEMA_TYPE_LABEL {
            continue;
        }
        match document.relations.get(resolved_label) {
            Some(relation) => {
                let subject_types = env.types_for(&op.subject);
                if side_violates(&relation.domain, subject_types) {
                    violations.push(Issue::domain(
                        path.associations_field(index, "subject"),
                        expected_types(&op.subject, &relation.domain, resolved_label),
                        actual_types(subject_types),
                    ));
                }
                let object_types = env.types_for(&op.object);
                if side_violates(&relation.range, object_types) {
                    violations.push(Issue::range_type(
                        path.associations_field(index, "object"),
                        expected_types(&op.object, &relation.range, resolved_label),
                        actual_types(object_types),
                    ));
                }
            }
            // `closed_labels` (§6.4) is scoped to fact_ops by this same
            // partition — a type_op is never reachable here at all,
            // never mind whether `closed_labels` is set.
            None if document.closed_labels => {
                violations.push(Issue::undeclared_label(
                    path.associations_field(index, "label"),
                    "a label declared in this context's schema (closed_labels)",
                ));
            }
            None => {}
        }
    }

    SchemaCheck {
        reserved,
        violations,
    }
}

fn reserved_alias_issue(path: &IssuePath<'_>, alias: &str) -> Issue {
    Issue::conflict(
        path.labels_field(alias),
        "a canonical spelling other than the one reserved for type assertions",
        format!(
            "resolves to '{SCHEMA_TYPE_LABEL}', the relation label reserved for type \
             assertions (ADR 0009 §6.3) — rename the alias"
        ),
    )
}

/// §8.2's `expected` shape: the declared set as written, never
/// expanded through the `is_a` closure — "the expanded set's size is a
/// schema-authoring accident, not information the corrector needs."
fn expected_types(name: &str, declared: &BTreeSet<String>, relation: &str) -> String {
    format!(
        "'{name}' typed as one of [{}] (or a subtype), for relation '{relation}'",
        join_capped(declared.iter())
    )
}

/// §8.2's `actual` shape: what was asserted, never the closure — and
/// never `"untyped"` (an untyped concept never violates, so
/// `side_violates` never lets this run with `types: None`).
fn actual_types(types: Option<&TypeAssertions>) -> String {
    match types {
        Some(types) => format!("typed as [{}]", join_capped(types.asserted.iter())),
        None => "typed as []".to_string(),
    }
}

fn join_capped<'a>(names: impl Iterator<Item = &'a String>) -> String {
    let names: Vec<&str> = names.map(String::as_str).collect();
    if names.len() <= MAX_ISSUE_TYPE_NAMES {
        names.join(", ")
    } else {
        format!(
            "{}, … (+{} more)",
            names[..MAX_ISSUE_TYPE_NAMES].join(", "),
            names.len() - MAX_ISSUE_TYPE_NAMES
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{RelationDef, SchemaDocument, TypeDef};

    /// Local twin of `crate::registry::test_support::assoc_op` — that
    /// helper lives in a private sibling module (`registry`'s own
    /// `test_support`) unreachable from here, and `AssocOp`'s fields are
    /// all `pub`, so building one directly is simpler than trying to
    /// share it.
    fn assoc_op(
        subject: &str,
        label: &str,
        object: &str,
        weight: f64,
        source: Option<&str>,
    ) -> AssocOp {
        AssocOp {
            subject: subject.to_string(),
            label: label.to_string(),
            object: object.to_string(),
            weight,
            source: source.map(String::from),
            paragraph: None,
        }
    }

    fn doc(mode: SchemaMode, closed_labels: bool) -> SchemaDocument {
        SchemaDocument {
            schema: super::super::SCHEMA_VERSION,
            mode,
            closed_labels,
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

    fn installed(document: SchemaDocument) -> Arc<InstalledSchema> {
        Arc::new(super::super::install(document).unwrap())
    }

    fn env(context: &Context, schema: Arc<InstalledSchema>, ops: &[AssocOp]) -> SchemaEnv {
        let declared = BTreeMap::new();
        SchemaEnv::build(
            context,
            SchemaCheckInput {
                schema,
                ops,
                declared_labels: &declared,
                retracted_source: None,
            },
        )
    }

    #[test]
    fn mode_off_yields_reserved_but_never_violations() {
        let mut context = Context::default();
        context
            .associate_from("青嶺酒造", "杜氏", "山田太郎", 1.0, "a.md", None)
            .unwrap();
        context
            .associate_from("青嶺酒造", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        let schema = installed(doc(SchemaMode::Off, false));
        let ops = [assoc_op("青嶺酒造", "杜氏", "山田太郎", 1.0, None)];
        let declared = BTreeMap::from([("種別".to_string(), SCHEMA_TYPE_LABEL.to_string())]);
        let env = SchemaEnv::build(
            &context,
            SchemaCheckInput {
                schema,
                ops: &ops,
                declared_labels: &declared,
                retracted_source: None,
            },
        );
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(check.violations.len(), 0, "off never judges");
        assert_eq!(
            check.reserved.len(),
            1,
            "guard 2 fires regardless of mode: {:?}",
            check.reserved.first().map(|issue| &issue.actual)
        );
    }

    #[test]
    fn reserved_batch_alias_fires_in_warn_and_strict_too() {
        let context = Context::default();
        for mode in [SchemaMode::Warn, SchemaMode::Strict] {
            let schema = installed(doc(mode, false));
            let ops: Vec<AssocOp> = Vec::new();
            let declared = BTreeMap::from([("型".to_string(), SCHEMA_TYPE_LABEL.to_string())]);
            let env = SchemaEnv::build(
                &context,
                SchemaCheckInput {
                    schema,
                    ops: &ops,
                    declared_labels: &declared,
                    retracted_source: None,
                },
            );
            let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
            assert_eq!(check.reserved.len(), 1, "mode {mode:?}");
            assert!(
                check.reserved[0].path.contains("型"),
                "{}",
                check.reserved[0].path
            );
        }
    }

    #[test]
    fn domain_violation_is_reported_on_the_subject_path() {
        let mut context = Context::default();
        context
            .associate_from("山田太郎", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op("山田太郎", "杜氏", "鈴木一郎", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(
            check.violations.len(),
            1,
            "{:?}",
            check.violations.first().map(|i| &i.path)
        );
        let issue = &check.violations[0];
        assert_eq!(issue.kind, "domain");
        assert_eq!(issue.path, "associations[0].subject");
        assert!(issue.expected.contains("Brewery"), "{}", issue.expected);
        assert!(issue.actual.contains("Person"), "{}", issue.actual);
    }

    /// [`IssuePath::Edge`]'s twin of the test above (#385, ADR 0009 §10's
    /// schema audit): the path names only the side, never a request-body
    /// coordinate — the audit already travels the offending association
    /// alongside the issue, so there is no index for `path` to carry.
    #[test]
    fn edge_path_names_the_side_alone() {
        let mut context = Context::default();
        context
            .associate_from("山田太郎", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op("山田太郎", "杜氏", "鈴木一郎", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Edge);
        assert_eq!(check.violations.len(), 1);
        assert_eq!(check.violations[0].path, "subject");
    }

    #[test]
    fn range_violation_is_reported_on_the_object_path() {
        let mut context = Context::default();
        context
            .associate_from("青嶺酒造", SCHEMA_TYPE_LABEL, "Brewery", 1.0, "a.md", None)
            .unwrap();
        context
            .associate_from(
                "醸造所",
                SCHEMA_TYPE_LABEL,
                "Organization",
                1.0,
                "a.md",
                None,
            )
            .unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op("青嶺酒造", "杜氏", "醸造所", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(check.violations.len(), 1);
        let issue = &check.violations[0];
        assert_eq!(issue.kind, "range");
        assert_eq!(issue.path, "associations[0].object");
    }

    #[test]
    fn empty_domain_never_constrains() {
        let mut context = Context::default();
        let schema = installed(doc(SchemaMode::Strict, false));
        // No relation definition for this label at all.
        context
            .associate("青嶺酒造", "所在地", "広島", 1.0)
            .unwrap();
        let ops = [assoc_op("青嶺酒造", "所在地", "広島", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(check.violations.len(), 0);
    }

    #[test]
    fn untyped_subject_never_violates() {
        let context = Context::default();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op("青嶺酒造", "杜氏", "山田太郎", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(
            check.violations.len(),
            0,
            "untyped never violates: {:?}",
            check.violations.first().map(|i| &i.actual)
        );
    }

    #[test]
    fn is_a_subtype_satisfies_the_parent_domain() {
        let mut context = Context::default();
        context
            .associate_from("青嶺酒造", SCHEMA_TYPE_LABEL, "Brewery", 1.0, "a.md", None)
            .unwrap();
        let mut document = doc(SchemaMode::Strict, false);
        document.relations.insert(
            "所属".to_string(),
            RelationDef {
                domain: BTreeSet::from(["Organization".to_string()]),
                range: BTreeSet::new(),
            },
        );
        let schema = installed(document);
        let ops = [assoc_op("青嶺酒造", "所属", "山田太郎", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(
            check.violations.len(),
            0,
            "Brewery is_a Organization: {:?}",
            check.violations.first().map(|i| &i.actual)
        );
    }

    #[test]
    fn undeclared_type_name_is_its_own_singleton_closure_never_a_violation() {
        let mut context = Context::default();
        context
            .associate_from(
                "青嶺酒造",
                SCHEMA_TYPE_LABEL,
                "Distillery",
                1.0,
                "a.md",
                None,
            )
            .unwrap();
        let mut document = doc(SchemaMode::Strict, false);
        document.relations.insert(
            "所属".to_string(),
            RelationDef {
                domain: BTreeSet::from(["Distillery".to_string()]),
                range: BTreeSet::new(),
            },
        );
        let schema = installed(document);
        let ops = [assoc_op("青嶺酒造", "所属", "山田太郎", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(check.violations.len(), 0);
    }

    #[test]
    fn type_ops_are_never_judged() {
        let context = Context::default();
        let mut document = doc(SchemaMode::Strict, true);
        document.relations.clear();
        let schema = installed(document);
        // schema:type is forbidden as a `relations` key (guard 3), so a
        // type_op could never match one anyway — this pins that
        // closed_labels also never fires for it.
        let ops = [assoc_op(
            "青嶺酒造",
            SCHEMA_TYPE_LABEL,
            "Brewery",
            1.0,
            None,
        )];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(check.violations.len(), 0);
    }

    #[test]
    fn closed_labels_flags_an_undeclared_fact_label_but_not_schema_type() {
        let context = Context::default();
        let schema = installed(doc(SchemaMode::Strict, true));
        let ops = [
            assoc_op("青嶺酒造", "所在地", "広島", 1.0, None),
            assoc_op("青嶺酒造", SCHEMA_TYPE_LABEL, "Brewery", 1.0, None),
        ];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(check.violations.len(), 1, "{:?}", check.violations);
        assert_eq!(check.violations[0].kind, "unknown_reference");
        assert_eq!(check.violations[0].path, "associations[0].label");
    }

    #[test]
    fn alias_false_negative_a_live_alias_subject_still_reads_its_type() {
        let mut context = Context::default();
        context
            .associate_from("青嶺酒造", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        context.add_concept_alias("青嶺", "青嶺酒造").unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        // The op uses the alias spelling, not the canonical one.
        let ops = [assoc_op("青嶺", "杜氏", "山田太郎", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(
            check.violations.len(),
            1,
            "the alias spelling must still resolve to the typed canonical concept"
        );
    }

    #[test]
    fn alias_false_positive_the_type_op_object_lands_on_the_canonical_type() {
        let mut context = Context::default();
        // Interns "Organization" as a concept so it can be aliased below.
        context
            .associate("Organization", "杜氏", "dummy", 1.0)
            .unwrap();
        context
            .add_concept_alias("Organisation", "Organization")
            .unwrap();
        // The type op asserts the alias spelling "Organisation"; once
        // written, `青嶺酒造 schema:type Organization` is what actually
        // lands (concept aliases resolve at write time), so this must
        // NOT be judged as if the concept were typed "Organisation" — a
        // singleton disjoint from `domain: [Organization]`.
        let mut document = doc(SchemaMode::Strict, false);
        document.relations.insert(
            "所属".to_string(),
            RelationDef {
                domain: BTreeSet::from(["Organization".to_string()]),
                range: BTreeSet::new(),
            },
        );
        let schema = installed(document);
        let ops = [
            assoc_op("青嶺酒造", SCHEMA_TYPE_LABEL, "Organisation", 1.0, None),
            assoc_op("青嶺酒造", "所属", "山田太郎", 1.0, None),
        ];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(
            check.violations.len(),
            0,
            "a type op naming an aliased type must not read as an undeclared singleton: {:?}",
            check.violations.first().map(|i| &i.actual)
        );
    }

    #[test]
    fn retraction_excludes_a_type_edge_only_this_source_asserted() {
        let mut context = Context::default();
        context
            .associate_from(
                "青嶺酒造",
                SCHEMA_TYPE_LABEL,
                "Person",
                1.0,
                "gone.md",
                None,
            )
            .unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op("青嶺酒造", "杜氏", "山田太郎", 1.0, None)];
        let declared = BTreeMap::new();
        let env = SchemaEnv::build(
            &context,
            SchemaCheckInput {
                schema,
                ops: &ops,
                declared_labels: &declared,
                retracted_source: Some("gone.md"),
            },
        );
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(
            check.violations.len(),
            0,
            "the only attribution is about to be retracted, so the concept reads as untyped: {:?}",
            check.violations.first().map(|i| &i.actual)
        );
    }

    #[test]
    fn retraction_does_not_exclude_a_type_edge_another_source_also_asserted() {
        let mut context = Context::default();
        context
            .associate_from(
                "青嶺酒造",
                SCHEMA_TYPE_LABEL,
                "Person",
                1.0,
                "gone.md",
                None,
            )
            .unwrap();
        context
            .associate_from(
                "青嶺酒造",
                SCHEMA_TYPE_LABEL,
                "Person",
                1.0,
                "stays.md",
                None,
            )
            .unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op("青嶺酒造", "杜氏", "山田太郎", 1.0, None)];
        let declared = BTreeMap::new();
        let env = SchemaEnv::build(
            &context,
            SchemaCheckInput {
                schema,
                ops: &ops,
                declared_labels: &declared,
                retracted_source: Some("gone.md"),
            },
        );
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(
            check.violations.len(),
            1,
            "another source's attribution survives the retraction, so the concept is still typed"
        );
    }

    #[test]
    fn retracted_source_none_is_a_plain_union() {
        let mut context = Context::default();
        context
            .associate_from("青嶺酒造", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op("青嶺酒造", "杜氏", "山田太郎", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(check.violations.len(), 1);
    }

    #[test]
    fn a_retracted_type_edge_count_zero_is_skipped_regardless_of_retraction() {
        let mut context = Context::default();
        context
            .associate_from("青嶺酒造", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        context.retract_source("a.md");
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op("青嶺酒造", "杜氏", "山田太郎", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(
            check.violations.len(),
            0,
            "a dead edge is never a live type assertion"
        );
    }

    #[test]
    fn negative_weight_is_still_judged() {
        let mut context = Context::default();
        context
            .associate_from("山田太郎", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op("山田太郎", "杜氏", "鈴木一郎", -1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(
            check.violations.len(),
            1,
            "weight sign must not exempt an op from judgment"
        );
    }

    #[test]
    fn every_violation_survives_past_twenty_no_truncation_here() {
        let mut context = Context::default();
        context
            .associate_from("山田太郎", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops: Vec<AssocOp> = (0..25)
            .map(|i| assoc_op("山田太郎", "杜氏", &format!("弟子{i}"), 1.0, None))
            .collect();
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(
            check.violations.len(),
            25,
            "truncation is the caller's job (MAX_LISTED_ISSUES), not schema_issues'"
        );
    }

    #[test]
    fn the_import_path_prefix_folds_into_one_grammar() {
        let mut context = Context::default();
        context
            .associate_from("山田太郎", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op("山田太郎", "杜氏", "鈴木一郎", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(
            &env,
            &ops,
            IssuePath::Request {
                prefix: "batches[3].",
            },
        );
        assert_eq!(
            check.violations[0].path,
            "batches[3].associations[0].subject"
        );
    }

    #[test]
    fn build_with_no_fact_concepts_reads_nothing_from_the_graph() {
        // A batch consisting only of type_ops must not trigger the
        // query_any(&[], ...) degenerate case (§7.2 step 4's guard) —
        // if it did, every edge in the context would spuriously appear
        // as a "type," and this context has an unrelated fact edge that
        // would prove it.
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        let schema = installed(doc(SchemaMode::Strict, false));
        let ops = [assoc_op(
            "青嶺酒造",
            SCHEMA_TYPE_LABEL,
            "Brewery",
            1.0,
            None,
        )];
        let env = env(&context, schema, &ops);
        // `violations` is empty either way, because a type_op is never
        // judged (§7.2 step 6) — that alone can't distinguish the
        // guarded path from the degenerate one. Only `env.types` can:
        // had `query_any(&[], ...)` run, the unrelated `私 好き りんご`
        // fact would have landed in it as if it were a type assertion.
        assert_eq!(
            env.types.keys().collect::<Vec<_>>(),
            vec!["青嶺酒造"],
            "only the batch half's own type_op may populate `types`"
        );
        assert_eq!(
            schema_issues(&env, &ops, IssuePath::Request { prefix: "" })
                .violations
                .len(),
            0
        );
    }

    /// `SchemaCheck::outcome`'s four branches (#388, S10 of #218's ADR
    /// 0009 split §15) — reserved refuses regardless of mode, a clean
    /// check (including every `off` check, since `violations` is empty
    /// there) is `Ok`, and a non-empty `violations` splits on mode:
    /// `strict` refuses, `warn` rides it out.
    #[test]
    fn outcome_dispatches_exactly_like_the_write_entrances_do() {
        use crate::metrics::SchemaOutcome;

        // off + reserved: guard 2 refuses even though `violations` is
        // always empty in `off` (`mode_off_yields_reserved_but_never_
        // violations` above).
        let mut context = Context::default();
        context
            .associate_from("青嶺酒造", "杜氏", "山田太郎", 1.0, "a.md", None)
            .unwrap();
        let schema = installed(doc(SchemaMode::Off, false));
        let ops = [assoc_op("青嶺酒造", "杜氏", "山田太郎", 1.0, None)];
        let declared = BTreeMap::from([("種別".to_string(), SCHEMA_TYPE_LABEL.to_string())]);
        let reserved_env = SchemaEnv::build(
            &context,
            SchemaCheckInput {
                schema,
                ops: &ops,
                declared_labels: &declared,
                retracted_source: None,
            },
        );
        let check = schema_issues(&reserved_env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(check.outcome(SchemaMode::Off), SchemaOutcome::Refused);

        // A clean batch against an installed schema is Ok in every mode.
        let mut context = Context::default();
        context
            .associate_from("山田太郎", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        context
            .associate_from("青嶺酒造", SCHEMA_TYPE_LABEL, "Brewery", 1.0, "a.md", None)
            .unwrap();
        for mode in [SchemaMode::Off, SchemaMode::Warn, SchemaMode::Strict] {
            let schema = installed(doc(mode, false));
            let ops = [assoc_op("青嶺酒造", "杜氏", "山田太郎", 1.0, None)];
            let env = env(&context, schema, &ops);
            let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
            assert_eq!(check.outcome(mode), SchemaOutcome::Ok, "mode {mode:?}");
        }

        // A domain violation: warn rides it out, strict refuses.
        let mut context = Context::default();
        context
            .associate_from("山田太郎", SCHEMA_TYPE_LABEL, "Person", 1.0, "a.md", None)
            .unwrap();
        let schema = installed(doc(SchemaMode::Warn, false));
        let ops = [assoc_op("山田太郎", "杜氏", "鈴木一郎", 1.0, None)];
        let env = env(&context, schema, &ops);
        let check = schema_issues(&env, &ops, IssuePath::Request { prefix: "" });
        assert_eq!(check.outcome(SchemaMode::Warn), SchemaOutcome::Warned);
        assert_eq!(check.outcome(SchemaMode::Strict), SchemaOutcome::Refused);
    }
}
