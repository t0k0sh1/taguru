//! Pre-apply rejection prediction and the write/preview pair that
//! shares it: [`apply_batch`] and its read-only twin [`preview_batch`]
//! both run [`predicted_alias_rejection`] then [`predicted_schema_rejection`]
//! before touching anything, so a dry run and a real import can never
//! disagree about whether a batch's aliases or associations would be
//! refused.

use super::*;

/// What one batch accomplished — the CLI formats it into a report
/// line, `POST /import` serializes it into the response.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct Applied {
    pub(crate) created: bool,
    pub(crate) retracted: usize,
    pub(crate) associations: usize,
    pub(crate) aliases: usize,
    pub(crate) passage_stored: bool,
    /// A previously stored passage for this source was retracted and the
    /// batch carried no replacement. With `passage_stored` this is a
    /// routine replace; without it, the batch just erased passage text —
    /// surfaced so that loss is never silent.
    pub(crate) passage_dropped: bool,
    pub(crate) questions_stored: usize,
    /// Questions naming a paragraph their passage's split does not
    /// have — most often a producer's index drifting from the server's
    /// canonical split.
    pub(crate) questions_dropped: usize,
    pub(crate) sections_stored: usize,
    /// Sections naming a paragraph their passage's split does not have
    /// (same convention and same likely cause as `questions_dropped`),
    /// plus any but the last of two or more sections claiming the same
    /// paragraph — a start marker governs until the next one, so only
    /// one can ever apply.
    pub(crate) sections_dropped: usize,
    pub(crate) locators_stored: usize,
    /// Locators naming a paragraph their passage's split does not have
    /// (same convention and same likely cause as `sections_dropped`),
    /// plus any but the last of two or more locators claiming the same
    /// paragraph — unlike a section's start marker, a locator names
    /// only its own paragraph, but the same one-per-paragraph
    /// last-write-wins rule applies.
    pub(crate) locators_dropped: usize,
    /// Association paragraph locators naming a spot this batch's own
    /// passage split does not have. Dropped exactly as `questions_dropped`
    /// and `sections_dropped` are — the association's fact still lands,
    /// only the paragraph pointer is cleared — and surfaced for the same
    /// reason: so the loss is a reported number, not a silent one.
    pub(crate) association_paragraphs_dropped: usize,
    /// `warn`-mode schema violations this batch's associations raised
    /// (ADR 0009 §8.3) — the true count, surviving truncation, mirrored
    /// into `ImportOutcome.schema_violations`. Always 0 for `off`, no
    /// schema, or `strict` (a `strict` violation refuses the batch
    /// instead — see [`ApplyRefusal::Schema`] — so it never reaches
    /// here).
    pub(crate) schema_violations: usize,
    /// The same violations, capped at `MAX_LISTED_ISSUES` and with
    /// batch-relative paths (`associations[{i}]...`, no `batches[{b}].`
    /// prefix — that is `src/api/import.rs`'s to add, once it knows this
    /// batch's stream position). Not part of `ImportOutcome` itself: the
    /// HTTP handler reads this to build the response envelope's
    /// `issues`; the CLI only reports the count.
    pub(crate) schema_issues: Vec<crate::api::Issue>,
}

/// Why a batch did not (fully) apply — one shape for both entrances:
/// the CLI prints [`ApplyRefusal::text`], the HTTP endpoint maps the
/// variant onto a status and sends the same words.
#[cfg_attr(test, derive(Debug))]
pub(crate) enum ApplyRefusal {
    /// The context does not exist and the batch brought no create
    /// block (404 over HTTP).
    NoContext(String),
    /// Filesystem trouble creating the context or persisting the
    /// passage (500).
    Io(String),
    /// The registry refused access (mapped like every other write).
    Access(AccessError),
    /// The library rejected an op partway; `applied` counts what
    /// landed first, `full` distinguishes capacity (507) from
    /// conflict (409). The retraction makes a corrected retry exact.
    Partial {
        applied: usize,
        message: String,
        full: bool,
    },
    /// Predicted before anything mutated: this batch's own alias
    /// operations would resolve to `AliasError::UnknownCanonical` or
    /// `Conflict` once actually applied, so the whole batch is
    /// refused up front (409) — no context created, no marker opened,
    /// no retraction, nothing. Distinct from `Partial { applied: 0,
    /// .. }`, which can only follow the retraction (itself a write)
    /// already landing. Structured (issue #182) rather than a bare
    /// message, so the HTTP endpoint can name the offending alias as a
    /// path-addressed `Issue` instead of prose alone.
    Rejected(AliasRejection),
    /// Predicted before anything mutated, same position as `Rejected`
    /// (checked right after it): this batch's own associations would
    /// violate a `strict` context's schema, or its own `labels`
    /// declares the reserved `schema:type` alias (ADR 0009 §6.3 guard
    /// 2, §7.2 step 7). Structured for the same reason `Rejected` is —
    /// path-addressed `Issue`s an MCP host corrects and resends.
    Schema(SchemaRejection),
}

/// Which alias namespace a predicted rejection concerns — concepts
/// intern subjects/objects, labels intern relation names.
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(crate) enum AliasNamespace {
    Concept,
    Label,
}

impl AliasNamespace {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Concept => "concepts",
            Self::Label => "labels",
        }
    }
}

/// A predicted alias rejection (issue #182): this batch's own alias
/// operations would resolve to [`AliasError::UnknownCanonical`] or
/// [`AliasError::Conflict`] once actually applied — named precisely
/// enough to build a structured `Issue` from, not just the prose
/// [`AliasRejection::text`] already reported.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct AliasRejection {
    pub(crate) namespace: AliasNamespace,
    pub(crate) alias: String,
    pub(crate) canonical: String,
    pub(crate) error: AliasError,
}

impl AliasRejection {
    pub(crate) fn text(&self) -> String {
        format!(
            "{} alias '{}' → '{}': {}; nothing was applied",
            match self.namespace {
                AliasNamespace::Concept => "concept",
                AliasNamespace::Label => "label",
            },
            self.alias,
            self.canonical,
            self.error,
        )
    }
}

impl ApplyRefusal {
    /// Whether the batch may have durably written anything before the
    /// refusal. Only [`ApplyRefusal::NoContext`], [`ApplyRefusal::Rejected`],
    /// and [`ApplyRefusal::Schema`] provably precede the first write —
    /// all three are predicted before the context is even created.
    /// Everything past that point starts with the source retraction,
    /// itself a durable write, so a later refusal (a passage that
    /// would not persist, a partial prefix of associations or aliases)
    /// leaves real changes behind. `Io` from a failed create or a
    /// failed batch-marker write is the over-approximation (both
    /// precede the first graph write); the refresh pass answers an
    /// absent context with its no-op `None` arm anyway.
    pub(crate) fn wrote_anything(&self) -> bool {
        !matches!(
            self,
            Self::NoContext(_) | Self::Rejected(_) | Self::Schema(_)
        )
    }

    /// How many ops this refusal's batch durably wrote before failing.
    /// Only [`ApplyRefusal::Partial`] carries a count — association or
    /// alias ops that landed in the WAL before the op that tripped the
    /// refusal. Feeds `ops_since_flush` in the import loop: a run
    /// dominated by partial failures (a capacity cap hit over and
    /// over) still needs its mid-run flushes on schedule, or the very
    /// WAL growth `FLUSH_EVERY_OPS` exists to bound goes unwatched.
    pub(crate) fn ops_written(&self) -> usize {
        match self {
            Self::Partial { applied, .. } => *applied,
            Self::NoContext(_) | Self::Io(_) | Self::Access(_) | Self::Rejected(_) => 0,
            Self::Schema(_) => 0,
        }
    }

    pub(crate) fn text(&self) -> String {
        match self {
            Self::NoContext(context) => {
                format!("context '{context}' does not exist and the batch brought no create block")
            }
            Self::Io(message) => message.clone(),
            Self::Access(AccessError::NotFound) => {
                "the context was deleted out from under the batch".to_string()
            }
            Self::Access(AccessError::Load(error)) => {
                format!("the context image would not load: {error}")
            }
            Self::Access(AccessError::Unpersisted(error)) => {
                format!("the WAL refused the write: {error}")
            }
            // `import_refusal` (api.rs) routes the Access variant to
            // `access_error_noted` directly and never calls `text()`
            // on it; the CLI import path runs with
            // Deadline::unbounded(). Unreachable either way, kept for
            // exhaustiveness.
            Self::Access(AccessError::DeadlineExceeded) => "deadline exceeded".to_string(),
            // Same unreachability, other leg: the CLI import boots with
            // no quota declaration (offline commands run as the
            // operator), and the HTTP path never calls `text()` on
            // Access.
            Self::Access(AccessError::QuotaExceeded(message)) => message.clone(),
            Self::Partial { message, .. } => message.clone(),
            Self::Rejected(rejection) => rejection.text(),
            Self::Schema(rejection) => rejection.text(),
        }
    }
}

/// Association paragraph locators corrected against this batch's own
/// passage split: a locator naming a spot the split does not have is
/// meaningless, so it is cleared (the association's fact still lands)
/// and counted as dropped. A batch with no passage has nothing to
/// check a locator against, so every op passes through unchanged.
/// Shared between the write path ([`apply_batch`]) and its read-only
/// preview ([`preview_batch`]) so the two can never disagree.
/// `paragraph_count`, when already known (`preview_batch` needs it for
/// its own question/section drop counts), is reused instead of
/// re-splitting the same passage text.
fn corrected_associations(batch: &Batch, paragraph_count: Option<usize>) -> (Vec<AssocOp>, usize) {
    let Some(text) = &batch.passage else {
        return (batch.associations.clone(), 0);
    };
    let paragraph_count = paragraph_count.unwrap_or_else(|| crate::paragraph::split(text).len());
    let mut dropped = 0;
    let corrected = batch
        .associations
        .iter()
        .cloned()
        .map(|mut op| {
            if op.paragraph.is_some_and(|p| p as usize >= paragraph_count) {
                op.paragraph = None;
                dropped += 1;
            }
            op
        })
        .collect();
    (corrected, dropped)
}

/// Predicts, without writing anything, whether this batch's own alias
/// operations would resolve to `AliasError::UnknownCanonical` or
/// `Conflict` once actually applied — the only purely content-driven
/// (non-capacity) rejections anywhere in the four-step apply pipeline.
/// Shared between [`apply_batch`] and [`preview_batch`] so a dry run
/// can never disagree with the real import about this call.
///
/// Checks concepts before labels, mirroring the WAL op order
/// `add_aliases` actually writes in, so a predicted message names the
/// same operation that would be the first to fail for real.
///
/// A context that does not exist yet has no aliases and no
/// associations to seed fresh names with, so a batch with a `create`
/// block is checked against an empty [`Context::default`] — exactly
/// the value `AppState::create` seeds a new context with. A context
/// that does not exist and brings no `create` block is left to the
/// ordinary `NoContext` refusal that follows this check.
fn predicted_alias_rejection(
    state: &AppState,
    batch: &Batch,
    seeds: Option<&PreviewSeeds>,
) -> Option<AliasRejection> {
    if batch.concepts.is_empty() && batch.labels.is_empty() {
        return None;
    }
    // Only THIS context's seeds may vouch for a canonical: a stream
    // can interleave contexts, and a name an earlier batch interned
    // into a sibling context proves nothing about this one — the real
    // apply would still refuse `UnknownCanonical` here.
    let seeded = seeds.and_then(|seeds| seeds.interned_in(&batch.context));
    let concepts = batch
        .associations
        .iter()
        .flat_map(|op| [op.subject.as_str(), op.object.as_str()])
        .chain(
            seeded
                .into_iter()
                .flat_map(|seeds| seeds.concepts.iter().map(String::as_str)),
        );
    let labels = batch.associations.iter().map(|op| op.label.as_str()).chain(
        seeded
            .into_iter()
            .flat_map(|seeds| seeds.labels.iter().map(String::as_str)),
    );
    let check = move |context: &Context| -> Option<AliasRejection> {
        if let Err((alias, canonical, error)) =
            context.check_concept_aliases(&batch.concepts, concepts)
        {
            return Some(AliasRejection {
                namespace: AliasNamespace::Concept,
                alias: alias.to_string(),
                canonical: canonical.to_string(),
                error,
            });
        }
        if let Err((alias, canonical, error)) = context.check_label_aliases(&batch.labels, labels) {
            return Some(AliasRejection {
                namespace: AliasNamespace::Label,
                alias: alias.to_string(),
                canonical: canonical.to_string(),
                error,
            });
        }
        None
    };

    if state.directory_entry(&batch.context).is_none() {
        // An earlier previewed batch reaching this context stands in
        // for the create block the real stream's first batch carries.
        return if batch.create.is_some() || seeds.is_some_and(|seeds| seeds.reaches(&batch.context))
        {
            check(&Context::default())
        } else {
            None
        };
    }
    state.read_context(&batch.context, check).ok().flatten()
}

/// What earlier batches of the SAME previewed stream would intern —
/// the dry run's stand-in for the batch-by-batch interning a real
/// apply performs, kept PER CONTEXT because a stream can interleave
/// contexts and interning is per context. Export puts every alias on
/// the LAST batch while the canonicals are interned by earlier ones,
/// so without this a dry run of a stream the real import applies
/// cleanly refuses with a spurious `UnknownCanonical` — breaking "a
/// dry run refuses exactly what the real import would". Cross-batch
/// alias CONFLICTS remain un-predicted (a preview holds no simulated
/// alias table); that gap only lets a preview pass what the real run
/// would refuse, the same advisory direction as the capacity caps.
#[derive(Default)]
pub(crate) struct PreviewSeeds {
    /// Context → what this stream's earlier batches intern there. A
    /// context's PRESENCE also stands in for its creation: a
    /// restore's create block rides only the FIRST batch, so without
    /// it every later batch of a fresh-name restore previews a
    /// spurious `NoContext` the real import (whose first batch
    /// actually creates) never raises.
    contexts: BTreeMap<String, ContextSeeds>,
}

/// One context's share of a [`PreviewSeeds`]: the names its earlier
/// batches would intern.
#[derive(Default)]
pub(crate) struct ContextSeeds {
    concepts: BTreeSet<String>,
    labels: BTreeSet<String>,
}

impl PreviewSeeds {
    /// Records what `batch` would intern once applied — call after the
    /// batch previews clean, before the next batch previews.
    pub(crate) fn absorb(&mut self, batch: &Batch) {
        let seeds = self.contexts.entry(batch.context.clone()).or_default();
        seeds.concepts.extend(batch.concept_vocabulary());
        seeds.labels.extend(batch.label_vocabulary());
    }

    /// Whether an earlier batch of this previewed stream already
    /// landed in `context` — creating it if it did not exist.
    fn reaches(&self, context: &str) -> bool {
        self.contexts.contains_key(context)
    }

    /// What this stream's earlier batches intern in `context` — and
    /// ONLY there; a sibling context's names never vouch here.
    fn interned_in(&self, context: &str) -> Option<&ContextSeeds> {
        self.contexts.get(context)
    }
}

/// `warn`-mode schema violations this batch's own associations raised
/// (ADR 0009 §8.3), capped like every other collect-all pass — empty
/// whenever the batch is clean, the context has no schema, or the
/// schema's mode is `off`.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct SchemaWarnings {
    pub(crate) issues: Vec<crate::api::Issue>,
    pub(crate) total: usize,
}

impl SchemaWarnings {
    fn none() -> Self {
        Self {
            issues: Vec::new(),
            total: 0,
        }
    }
}

/// A predicted schema rejection (ADR 0009 §7.2, §6.3): this batch's own
/// associations would violate a `strict` context's domain/range
/// constraints, or this batch's own `labels` declares the reserved
/// `schema:type` alias — named precisely enough to build path-addressed
/// `Issue`s from, exactly like [`AliasRejection`] beside it. `reserved`
/// tells [`ApplyRefusal::text`] and the HTTP status
/// (`src/api/import.rs`) which of the two this is: a reserved-label
/// collision is a namespace conflict (409, like an alias `Conflict`), a
/// domain/range violation is a refused value (400, ADR 0009 §8.1).
#[cfg_attr(test, derive(Debug))]
pub(crate) struct SchemaRejection {
    pub(crate) issues: Vec<crate::api::Issue>,
    pub(crate) total: usize,
    pub(crate) reserved: bool,
}

impl SchemaRejection {
    /// What the issues are about, for the refusal's prose.
    pub(crate) fn what(&self) -> &'static str {
        if self.reserved {
            "this batch's label aliases"
        } else {
            "this batch's associations"
        }
    }

    /// Every issue, listed — `issues` is the complete set here (#863):
    /// the offline CLI prints this and has no other channel for the
    /// remainder, so nothing is cut; the HTTP entrance truncates for
    /// the wire itself (`src/api/import.rs`), where `issues_total`
    /// carries the count.
    pub(crate) fn text(&self) -> String {
        crate::api::collected_validation_message(self.what(), &self.issues, self.total)
    }
}

/// Which caller is running [`predicted_schema_rejection`] (#388, S10 of
/// #218's ADR 0009 split §15): only [`Apply`](CheckPurpose::Apply) is a
/// real write gate, so only it feeds `taguru_schema_checks_total` —
/// [`Preview`](CheckPurpose::Preview) runs the identical check for
/// `?dry_run=true`/`preview_batch`, and counting it too would let a
/// validate-then-apply workflow double-count the same refusal.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CheckPurpose {
    Apply,
    Preview,
}

/// Predicts, without writing anything, whether this batch's own
/// associations would violate a `strict` context's schema, or its own
/// `labels` declares the reserved `schema:type` alias (ADR 0009 §6.3
/// guard 2's batch-local bullet, checked regardless of mode) — the
/// schema twin of [`predicted_alias_rejection`], run right after it: an
/// alias conflict is caught first, so by the time this runs every
/// concept/label spelling this batch's own `associations` use is
/// already known not to collide with its own `concepts`/`labels`
/// declarations, which `schema::check::SchemaEnv` relies on (its own
/// doc, `src/schema/check.rs:88-94`). Shared between [`apply_batch`]
/// and [`preview_batch`] for the same reason `predicted_alias_rejection`
/// is: a dry run can never disagree about this call either. `purpose`
/// distinguishes them for metrics only — the judgment itself is
/// identical either way. `ops` is the corrected association list the
/// caller built with `corrected_associations(batch, None)` (ADR 0009
/// §7.2 step 2 — no paragraph count exists yet at prediction time for
/// either entrance): passing it in lets `apply_batch` build the list
/// once for prediction and apply alike, so the two can never drift.
///
/// No schema installed for this context — including one that does not
/// exist yet — returns `Ok` before a single lock is taken
/// (`AppState::schema_of`'s own fast path for `schema_digest.is_none()`):
/// the zero-cost path every schema-free context takes, ADR 0009 §7.2
/// step 1. A schema recorded but currently unreadable is never treated
/// as schema-free — `src/schema.rs`'s own module doc fixes that as a
/// hard refusal, never a silent fallback — so this maps such a read
/// failure to [`ApplyRefusal::Io`] instead of proceeding.
fn predicted_schema_rejection(
    state: &AppState,
    batch: &Batch,
    ops: &[AssocOp],
    purpose: CheckPurpose,
) -> Result<SchemaWarnings, ApplyRefusal> {
    let schema = match state.schema_of(&batch.context) {
        None | Some(Ok(None)) => return Ok(SchemaWarnings::none()),
        Some(Ok(Some(schema))) => schema,
        Some(Err(message)) => {
            return Err(ApplyRefusal::Io(format!(
                "schema for context '{}' could not be read: {message}",
                batch.context
            )));
        }
    };

    let check = state
        .read_context(&batch.context, |context| {
            let env = crate::schema::SchemaEnv::build(
                context,
                crate::schema::SchemaCheckInput {
                    schema: schema.clone(),
                    ops,
                    declared_labels: &batch.labels,
                    // `apply_batch` retracts `batch.source` before
                    // applying (`:2354-2357` at the time of writing) —
                    // the live-half exclusion this passes on to
                    // `SchemaEnv::build` judges against the graph state
                    // this write is about to leave behind, not its
                    // current one (ADR 0009 §7.2 step 4).
                    retracted_source: Some(&batch.source),
                },
            );
            crate::schema::schema_issues(
                &env,
                ops,
                crate::schema::IssuePath::Request { prefix: "" },
            )
        })
        .map_err(ApplyRefusal::Access)?;

    let mode = schema.document().mode;
    if purpose == CheckPurpose::Apply {
        state.note_schema_check(&batch.context, check.outcome(mode), check.violations.len());
    }

    // Complete lists, never truncated here (#863): the offline CLI
    // prints every issue, and the HTTP entrance cuts to
    // MAX_LISTED_ISSUES with the true total beside it at the point it
    // builds the response.
    if !check.reserved.is_empty() {
        let total = check.reserved.len();
        return Err(ApplyRefusal::Schema(SchemaRejection {
            issues: check.reserved,
            total,
            reserved: true,
        }));
    }

    let issues = check.violations;
    let total = issues.len();
    if mode == crate::schema::SchemaMode::Strict && total > 0 {
        return Err(ApplyRefusal::Schema(SchemaRejection {
            issues,
            total,
            reserved: false,
        }));
    }
    // `off` and a clean `strict` batch both fall through here with an
    // empty `issues`/`total` — constructing `SchemaWarnings` either way
    // rather than special-casing keeps this function's one dispatch
    // point exactly what ADR 0009 §7.2 step 7 describes.
    Ok(SchemaWarnings { issues, total })
}

/// Applies one validated batch: ensure the context, retract the
/// source, then land passage → associations → aliases. Aliases go
/// last on purpose — an alias needs its canonical interned, and the
/// associations just before are what intern it. Before any of that,
/// [`predicted_alias_rejection`] checks whether this batch's own alias
/// operations would resolve to a conflict, then [`predicted_schema_rejection`]
/// checks whether they would violate the context's schema; either
/// predicted rejection refuses the whole batch ([`ApplyRefusal::Rejected`]
/// / [`ApplyRefusal::Schema`]) up front, so a bad alias or a schema
/// violation no longer surfaces only after the associations (or the
/// retraction) have already landed.
///
/// Past that point, the four mutations are separately durable, so a
/// crash between them leaves the source half-applied with every store
/// individually consistent — undetectable after the fact. A
/// batch-open marker brackets them: written before the retraction,
/// removed only after the aliases, so boot and `taguru inspect` can
/// name any batch that never finished. A batch refused after the
/// marker opens (capacity, disk trouble) keeps its marker too — the
/// refusal is reported once, the marker keeps saying so until the
/// documented repair (re-import, or retract the source) actually
/// runs. A predicted rejection opens no marker at all: nothing ran
/// yet for it to bracket. Cross-store atomicity is deliberately not
/// attempted for what prediction cannot catch: per-source
/// retract-then-apply idempotency already makes the repair exact, so
/// detection is the remaining gap.
pub(crate) fn apply_batch(
    state: &AppState,
    batch: &Batch,
    deadline: Deadline,
) -> Result<Applied, ApplyRefusal> {
    // No seeds: earlier batches of a real stream have actually landed,
    // so the live context already holds whatever they interned.
    if let Some(rejection) = predicted_alias_rejection(state, batch, None) {
        return Err(ApplyRefusal::Rejected(rejection));
    }
    // Built once, judged and applied alike: the same silent paragraph
    // clamp `corrected_associations` documents below, shared with the
    // schema prediction so the two see the identical op list (and the
    // passage is paragraph-split once, not twice).
    let (corrected, association_paragraphs_dropped) = corrected_associations(batch, None);
    let schema_warnings =
        predicted_schema_rejection(state, batch, &corrected, CheckPurpose::Apply)?;

    let mut created = false;
    if state.directory_entry(&batch.context).is_none() {
        let Some(meta) = &batch.create else {
            return Err(ApplyRefusal::NoContext(batch.context.clone()));
        };
        match state.create(&batch.context, meta.clone()) {
            Ok(()) => created = true,
            // Another writer got between the check and the create —
            // possible on the live server, harmless everywhere: the
            // context exists now, which is all the batch needed.
            Err(CreateError::AlreadyExists) => {}
            // Unreachable in practice — `parse_header` already refused an
            // empty context name — but the registry guards it too, so the
            // match must speak for it.
            Err(CreateError::InvalidName) => {
                return Err(ApplyRefusal::Io(format!(
                    "context name '{}' is not usable (empty)",
                    batch.context
                )));
            }
            Err(CreateError::Io(io_error)) => {
                return Err(ApplyRefusal::Io(format!(
                    "creating context '{}': {io_error}",
                    batch.context
                )));
            }
        }
    }

    // The marker precedes the first mutation or the batch does not
    // run: starting untracked would silently reopen the exact
    // undetectable-tear window it exists to close.
    if let Err(error) = state.open_import_marker(&batch.context, &batch.source) {
        return Err(ApplyRefusal::Io(format!(
            "import marker not persisted: {error} — nothing was applied"
        )));
    }

    // Not `retract_source`: this batch's own marker (opened above)
    // already brackets this call along with every step that follows —
    // clearing it here too would reopen the batch to the exact gap it
    // exists to close.
    let (retracted, passage_removed, passage_removal_errored) = state
        .retract_source_unmarked(&batch.context, &batch.source)
        .map_err(ApplyRefusal::Access)?;
    // `passage_removed` alone is unconditional — true whenever a prior
    // passage existed and was removed, with no notion of a forthcoming
    // replacement. `Applied::passage_dropped` promises the narrower
    // "and the batch carried no replacement," same as `preview_batch`.
    let passage_dropped = passage_removed && batch.passage.is_none();

    // A genuine passage-store failure here only self-heals when this
    // batch carries a replacement passage: `store_passages` below then
    // overwrites whatever stale copy the failed retraction left
    // behind. With no replacement coming, that stale passage would
    // survive under a marker this function is about to clear as if
    // the source's truth were fully applied — refuse instead, leaving
    // the marker (and the documented repair) in place.
    if passage_removal_errored && batch.passage.is_none() {
        return Err(ApplyRefusal::Io(format!(
            "old passage for source '{}' could not be retracted and this batch carries no \
             replacement passage to overwrite it with — its truth may be half-applied",
            batch.source
        )));
    }

    let mut questions_stored = 0;
    let mut questions_dropped = 0;
    let mut sections_stored = 0;
    let mut sections_dropped = 0;
    let mut locators_stored = 0;
    let mut locators_dropped = 0;
    if let Some(text) = &batch.passage {
        let outcome = state
            .store_passages(
                &batch.context,
                BTreeMap::from([(
                    batch.source.clone(),
                    crate::passages::PassageSubmission {
                        text: text.clone(),
                        questions: batch.questions.clone(),
                        sections: batch.sections.clone(),
                        locators: batch.locators.clone(),
                        meta: crate::passages::SourceMeta {
                            stored_at: batch.stored_at,
                            date: batch.date,
                            tags: batch.tags.clone(),
                        },
                    },
                )]),
            )
            .ok_or(ApplyRefusal::Access(AccessError::NotFound))?
            .map_err(|error| match error {
                // The policy refusal keeps its shape (507 over HTTP,
                // via the same Access mapping every graph gate uses);
                // only genuine disk trouble flattens to Io.
                crate::registry::PassagesWriteError::QuotaExceeded(message) => {
                    ApplyRefusal::Access(AccessError::QuotaExceeded(message))
                }
                crate::registry::PassagesWriteError::Io(io_error) => {
                    ApplyRefusal::Io(format!("passage not persisted: {io_error}"))
                }
            })?;
        questions_stored = outcome.questions_stored;
        questions_dropped = outcome.questions_dropped;
        sections_stored = outcome.sections_stored;
        sections_dropped = outcome.sections_dropped;
        locators_stored = outcome.locators_stored;
        locators_dropped = outcome.locators_dropped;
    }

    // Same rule as questions/sections above, applied silently: a
    // paragraph naming a spot this batch's own passage does not have
    // is meaningless, so it is dropped rather than persisted — the
    // association itself (subject/label/object/weight) still lands
    // (`corrected`, built once above, before the schema prediction).
    // The caller's deadline rides into every chunk: a long batch on a
    // spent budget refuses here (the marker stays, the retry is
    // exact) instead of writing unbounded past the checkpoints the
    // HTTP import loop keeps between batches.
    let associations_to_apply: &[AssocOp] = &corrected;

    let mut associations = 0;
    for chunk in associations_to_apply.chunks(MAX_ASSOCIATIONS_PER_REQUEST) {
        match state
            .add_associations(&batch.context, chunk.to_vec(), deadline)
            .map_err(ApplyRefusal::Access)?
        {
            Ok(applied) => associations += applied,
            Err(partial) => {
                let applied = associations + partial.applied;
                return Err(ApplyRefusal::Partial {
                    applied,
                    message: format!(
                        "applied {applied} association(s), then: {} — fix the batch and \
                         re-import; the retraction makes the retry exact",
                        partial.message
                    ),
                    full: partial.full,
                });
            }
        }
    }

    let mut aliases = 0;
    if !batch.concepts.is_empty() || !batch.labels.is_empty() {
        match state
            .add_aliases(&batch.context, &batch.concepts, &batch.labels)
            .map_err(ApplyRefusal::Access)?
        {
            Ok(applied) => aliases += applied,
            Err(partial) => {
                return Err(ApplyRefusal::Partial {
                    // Same running total the association arm above
                    // reports: `applied` is the batch's cumulative
                    // count, not just this call's — a batch whose
                    // associations landed but whose first alias
                    // didn't must not report 0 (`partial.applied`
                    // alone) when `associations` ops are already
                    // durable.
                    applied: associations + partial.applied,
                    message: format!(
                        "applied {} alias(es), then: {}",
                        partial.applied, partial.message
                    ),
                    full: partial.full,
                });
            }
        }
    }

    // Only now is the source's stated truth fully on disk.
    state.clear_import_marker(&batch.context, &batch.source);

    state.note_write(&batch.context);
    Ok(Applied {
        created,
        retracted,
        associations,
        aliases,
        passage_stored: batch.passage.is_some(),
        passage_dropped,
        questions_stored,
        questions_dropped,
        sections_stored,
        sections_dropped,
        locators_stored,
        locators_dropped,
        association_paragraphs_dropped,
        schema_violations: schema_warnings.total,
        schema_issues: schema_warnings.issues,
    })
}

/// The read-only twin of [`apply_batch`], for `POST
/// /import?dry_run=true`: reports what a batch WOULD do without
/// writing anything — no context created, no marker opened, no source
/// retracted. Runs the same [`predicted_alias_rejection`] and
/// [`predicted_schema_rejection`] checks first, in the same order
/// `apply_batch` does, so a batch whose aliases would conflict or whose
/// associations would violate the schema is refused here exactly as it
/// would be by `apply_batch` — the two entrances can never disagree on
/// either call. Every other write step in `apply_batch` has a cheap
/// read-only counterpart here, except the `associations` and `aliases`
/// counts, which stay OPTIMISTIC (every op this batch carries,
/// corrected the same way `apply_batch` corrects them): a capacity cap
/// (507) can only surface by actually applying the op, so those two
/// COUNTS remain advisory even though an alias CONFLICT or a schema
/// VIOLATION no longer is — the real import can still apply fewer
/// associations or aliases than previewed. Every other field
/// (`retracted`, the drop counts, `schema_violations`) reads through to
/// the same state a real batch would query, so it matches exactly.
pub(crate) fn preview_batch(
    state: &AppState,
    batch: &Batch,
    seeds: &PreviewSeeds,
) -> Result<Applied, ApplyRefusal> {
    if let Some(rejection) = predicted_alias_rejection(state, batch, Some(seeds)) {
        return Err(ApplyRefusal::Rejected(rejection));
    }
    // The prediction judges the SAME list `apply_batch`'s own
    // prediction judges — `None`, not this preview's `paragraph_count`
    // below, which exists only here (ADR 0009 §7.2 step 2).
    let (predicted_ops, _) = corrected_associations(batch, None);
    let schema_warnings =
        predicted_schema_rejection(state, batch, &predicted_ops, CheckPurpose::Preview)?;

    let exists = state.directory_entry(&batch.context).is_some();
    // An earlier batch of this previewed stream reaching the context
    // stands in for its create — the real stream's first batch will
    // have created it by the time this one applies.
    let seeded = !exists && seeds.reaches(&batch.context);
    let created = !exists && !seeded;
    if created && batch.create.is_none() {
        return Err(ApplyRefusal::NoContext(batch.context.clone()));
    }

    // A context about to be created — by this batch or an earlier one
    // of the same previewed stream — has nothing to retract from yet.
    let retracted = if exists {
        state
            .count_source_edges(&batch.context, &batch.source)
            .map_err(ApplyRefusal::Access)?
    } else {
        0
    };
    // Mirrors apply_batch's tolerance for a passage-store read that
    // fails: retract_source warns and reports no removal rather than
    // failing the whole batch, so the preview falls back the same way.
    let had_passage = exists
        && state
            .passage_sources(&batch.context)
            .and_then(Result::ok)
            .is_some_and(|sources| sources.contains(&batch.source));
    let passage_dropped = had_passage && batch.passage.is_none();

    let paragraph_count = batch
        .passage
        .as_deref()
        .map(|text| crate::paragraph::split(text).len());
    let (questions_dropped, sections_dropped, locators_dropped) = match paragraph_count {
        Some(paragraph_count) => crate::passages::preview_drops(
            paragraph_count,
            &batch.questions,
            &batch.sections,
            &batch.locators,
        ),
        None => (0, 0, 0),
    };

    let (corrected, association_paragraphs_dropped) =
        corrected_associations(batch, paragraph_count);

    Ok(Applied {
        created,
        retracted,
        associations: corrected.len(),
        aliases: batch.concepts.len() + batch.labels.len(),
        passage_stored: batch.passage.is_some(),
        passage_dropped,
        questions_stored: batch.questions.len() - questions_dropped,
        questions_dropped,
        sections_stored: batch.sections.len() - sections_dropped,
        sections_dropped,
        locators_stored: batch.locators.len() - locators_dropped,
        locators_dropped,
        association_paragraphs_dropped,
        schema_violations: schema_warnings.total,
        schema_issues: schema_warnings.issues,
    })
}
