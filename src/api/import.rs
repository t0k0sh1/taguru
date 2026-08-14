use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use serde::{Deserialize, Serialize};

use taguru::context::AliasError;
use taguru::deadline::Deadline;

use crate::groups::GroupRecord;
use crate::ingest::AliasRejection;
use crate::metrics::ErrorKind;
use crate::registry::{AccessError, AppState};

use super::groups::{scope_refusal, scoped_member_contexts};
use super::{
    AppBytes, AppPath, AppQuery, ErrorCode, Issue, RefusalDetail, access_error, access_error_noted,
    deadline_exceeded, error, group_not_found, key_name, nesting_error_code, ok,
    ok_with_issues_total, validation_error,
};

/// `POST /import`'s query string.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ImportQuery {
    /// Report what the stream would do without writing anything — no
    /// context created, no source retracted, nothing stored. See
    /// [`crate::ingest::preview_batch`] for which counts are exact and
    /// which are advisory.
    pub dry_run: bool,
}

/// What `POST /import` accomplished — the same numbers the CLI's
/// per-file report line carries.
#[derive(Serialize)]
pub struct ImportOutcome {
    pub context: String,
    pub source: String,
    pub created: bool,
    pub retracted: usize,
    pub associations: usize,
    pub aliases: usize,
    pub passage_stored: bool,
    /// A stored passage for this source was retracted with no replacement
    /// in the batch — `passage_stored: false` alone cannot tell "never
    /// had one" from "just erased one", so callers get this explicitly.
    pub passage_dropped: bool,
    pub questions_stored: usize,
    pub questions_dropped: usize,
    pub sections_stored: usize,
    pub sections_dropped: usize,
    pub locators_stored: usize,
    pub locators_dropped: usize,
    /// Association paragraph locators dropped for naming a spot the
    /// batch's passage split does not have — the association itself
    /// still landed. Reported like `questions_dropped`/`sections_dropped`.
    pub association_paragraphs_dropped: usize,
    /// `warn`-mode schema violations this batch's associations raised
    /// (ADR 0009 §8.3) — the true count, surviving the response
    /// envelope's own `issues` truncation. Always 0 for `off`, no
    /// schema, or `strict` (a `strict` violation refuses the batch
    /// instead, so it never reaches an `ImportOutcome` at all).
    pub schema_violations: usize,
}

/// What a `POST /import` body accomplished — one [`ImportOutcome`] per
/// batch, in stream order. Every import answers this shape, a
/// single-batch body included: one shape to parse, not two.
#[derive(Serialize)]
pub struct ImportStreamOutcome {
    pub batches: Vec<ImportOutcome>,
    /// One entry per `taguru_schema` record, stream order — absent
    /// entirely for a stream that carried none, keeping the
    /// pre-schema response byte-identical (ADR 0009 §13).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub schemas: Vec<SchemaImportOutcome>,
    /// One entry per `taguru_group` record, stream order — absent
    /// entirely for a stream that carried none, keeping the pre-group
    /// response byte-identical.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupImportOutcome>,
}

/// What installing one `taguru_schema` record accomplished. No
/// outcome verb (unlike [`GroupImportOutcome`]'s "created"/"replaced"/
/// "unchanged"): `put_schema` cannot itself distinguish an install
/// from a no-op PUT of the identical document, and a guessed verb
/// would be worse than none (ADR 0009 §13).
#[derive(Serialize)]
pub struct SchemaImportOutcome {
    pub context: String,
    pub mode: &'static str,
    pub types: usize,
    pub relations: usize,
}

/// `pub(crate)`: `import --json` reuses this to build the exact same
/// shape `POST /import`'s response carries, from the same
/// [`crate::schema::SchemaDocument`] [`crate::ingest::
/// apply_schema_record`] returns — one computation, not two that
/// could drift.
pub(crate) fn schema_import_outcome(
    context: &str,
    document: &crate::schema::SchemaDocument,
) -> SchemaImportOutcome {
    SchemaImportOutcome {
        context: context.to_string(),
        mode: document.mode.as_str(),
        types: document.types.len(),
        relations: document.relations.len(),
    }
}

/// What restoring one group record accomplished. A restore is a
/// replace of the whole record; the label says what it replaced.
#[derive(Serialize)]
pub struct GroupImportOutcome {
    pub name: String,
    /// `"created"` | `"replaced"` | `"unchanged"` — the last says the
    /// standing record already matched, so nothing was rewritten.
    pub outcome: &'static str,
    /// Member counts of the record as restored.
    pub contexts: usize,
    pub groups: usize,
}

/// `pub(crate)`: `import --json` (issue #371) reuses this to build the
/// exact same shape `POST /import`'s response carries, from the same
/// `Batch`/`Applied` inputs `report()`'s human line already reads —
/// one computation, not two that could drift.
pub(crate) fn import_outcome(
    batch: &crate::ingest::Batch,
    applied: &crate::ingest::Applied,
) -> ImportOutcome {
    ImportOutcome {
        context: batch.context.clone(),
        source: batch.source.clone(),
        created: applied.created,
        retracted: applied.retracted,
        associations: applied.associations,
        aliases: applied.aliases,
        passage_stored: applied.passage_stored,
        passage_dropped: applied.passage_dropped,
        questions_stored: applied.questions_stored,
        questions_dropped: applied.questions_dropped,
        sections_stored: applied.sections_stored,
        sections_dropped: applied.sections_dropped,
        locators_stored: applied.locators_stored,
        locators_dropped: applied.locators_dropped,
        association_paragraphs_dropped: applied.association_paragraphs_dropped,
        schema_violations: applied.schema_violations,
    }
}

/// Names one predicted alias rejection (issue #182) as a path-addressed
/// [`Issue`]: `batches[{index}]` locates which stream element failed,
/// the alias namespace and spelling locate the mapping, and the kind
/// mirrors what [`AliasError`] itself distinguishes.
fn alias_rejection_issue(batch_index: usize, rejection: &AliasRejection) -> Issue {
    let path = format!(
        "batches[{batch_index}].{}['{}']",
        rejection.namespace.as_str(),
        rejection.alias
    );
    match rejection.error {
        AliasError::UnknownCanonical => Issue::unknown_reference(
            path,
            format!(
                "'{}' already interned as a concept or label this batch's associations use",
                rejection.canonical
            ),
        ),
        AliasError::Conflict => Issue::conflict(
            path,
            format!("resolves to '{}'", rejection.canonical),
            "already resolves to a different record",
        ),
        AliasError::Full(_) => {
            Issue::conflict(path, "context capacity available", "context is full")
        }
    }
}

/// Extends `schema::schema_issues`' batch-relative paths
/// (`associations[{i}]...`, `labels['{alias}']`) with the stream
/// position [`crate::ingest::predicted_schema_rejection`] itself has no
/// way to know (ADR 0009 §8.2's import grammar,
/// `batches[{b}].associations[{a}]...`) — the schema twin of
/// [`alias_rejection_issue`]'s own `batches[{index}]` prefix. Shared by
/// both the strict-refusal arm below and the warn-mode success path, so
/// the two present identical `Issue` values for the same violation (ADR
/// 0009 §8.3).
pub(super) fn schema_issues_in_batch(batch_index: usize, issues: Vec<Issue>) -> Vec<Issue> {
    issues
        .into_iter()
        .map(|issue| Issue {
            path: format!("batches[{batch_index}].{}", issue.path),
            ..issue
        })
        .collect()
}

/// The `integrity` verdict for a mid-stream import refusal (issue
/// #182): `nothing_written` when no batch before this one landed (or
/// this is a `dry_run`, which never writes anything at all regardless
/// of how many batches "previewed clean"), otherwise a
/// `durable_prefix` naming exactly how many did — never implying a
/// subset of THIS batch's own writes was accepted, since every batch
/// is all-or-nothing.
pub(super) fn stream_integrity(
    previewed_or_landed: usize,
    dry_run: bool,
) -> (&'static str, Option<usize>) {
    if dry_run || previewed_or_landed == 0 {
        ("nothing_written", None)
    } else {
        ("durable_prefix", Some(previewed_or_landed))
    }
}

/// The refusal a budget spent partway through [`import_batch`]'s loop
/// answers: each landed batch is durable (retract-then-apply), so the
/// stop is a resumable prefix — the note names the position, and the
/// [`stream_integrity`] fields carry the same claim machine-readably.
pub(super) fn import_budget_refusal(
    index: usize,
    total: usize,
    batch: &crate::ingest::Batch,
    landed: usize,
    dry_run: bool,
    started_at: Instant,
) -> Response {
    let note = import_batch_note(
        index,
        total,
        batch,
        landed,
        dry_run,
        ("not previewed", "not attempted"),
        (
            "re-running the preview with more time or a narrower stream is exact",
            "re-POSTing the remaining stream is exact (each batch replaces its \
             own source)",
        ),
    );
    let (integrity, durable_batches) = stream_integrity(landed, dry_run);
    validation_error(
        ErrorCode::Timeout,
        format!(
            "{note}request exceeded its budget partway through a multi-batch \
             import (TAGURU_REQUEST_TIMEOUT_SECS tunes this)"
        ),
        RefusalDetail {
            integrity: Some(integrity),
            durable_batches,
            ..Default::default()
        },
        started_at,
    )
}

/// Maps one batch's [`ApplyRefusal`](crate::ingest::ApplyRefusal) onto
/// the response, `note` naming which batch of a stream refused (empty
/// for a single-batch body, keeping that path's responses exactly as
/// they were). `batch_index` and `durable_batches` (batches before this
/// one that already landed) feed the structured detail (issue #182) the
/// two provably-atomic arms (`NoContext`, `Rejected`) can carry;
/// `Partial`/`Io`/`Access` cannot prove how much of THIS batch landed,
/// so their prose stays the only account, as before.
#[allow(clippy::too_many_arguments)] // the batch's stream position, spread flat like call_inner's outer context
pub(super) fn import_refusal(
    state: &AppState,
    batch: &crate::ingest::Batch,
    refusal: crate::ingest::ApplyRefusal,
    note: &str,
    batch_index: usize,
    durable_batches: usize,
    dry_run: bool,
    started_at: Instant,
) -> Response {
    match refusal {
        // The three AccessError arms — status, metric, message — live
        // in access_error_noted; import just supplies the batch note.
        crate::ingest::ApplyRefusal::Access(failure) => {
            access_error_noted(state, failure, &batch.context, note, started_at)
        }
        refusal @ crate::ingest::ApplyRefusal::NoContext(_) => {
            let (integrity, durable_batches) = stream_integrity(durable_batches, dry_run);
            validation_error(
                ErrorCode::NoContext,
                format!("{note}{}", refusal.text()),
                RefusalDetail {
                    issues: vec![Issue::missing(
                        format!("batches[{batch_index}].create"),
                        "a create block, since the named context does not exist yet",
                    )],
                    integrity: Some(integrity),
                    durable_batches,
                    retryable_after_correction: Some(true),
                },
                started_at,
            )
        }
        refusal @ crate::ingest::ApplyRefusal::Io(_) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
                format!("{note}{}", refusal.text()),
                started_at,
            )
        }
        crate::ingest::ApplyRefusal::Partial {
            applied,
            message,
            full,
        } => {
            // The batch got partway: what landed counts as a write,
            // and the status keeps the capacity/conflict distinction
            // every partial write reports.
            if applied > 0 {
                state.note_write(&batch.context);
            }
            let code = if full {
                ErrorCode::StorageFull
            } else {
                ErrorCode::Conflict
            };
            error(code, format!("{note}{message}"), started_at)
        }
        // Predicted before anything mutated — no create, no marker,
        // no retraction, so unlike `Partial` above there is no write
        // to note.
        crate::ingest::ApplyRefusal::Rejected(rejection) => {
            let (integrity, durable_batches) = stream_integrity(durable_batches, dry_run);
            let issue = alias_rejection_issue(batch_index, &rejection);
            let message = format!("{note}{}", rejection.text());
            validation_error(
                ErrorCode::Conflict,
                message,
                RefusalDetail {
                    issues: vec![issue],
                    integrity: Some(integrity),
                    durable_batches,
                    retryable_after_correction: Some(true),
                },
                started_at,
            )
        }
        // Predicted before anything mutated, same position as
        // `Rejected` above — no create, no marker, no retraction, so
        // there is no write to note here either. ADR 0009 §8.1: a
        // reserved-label collision is a namespace conflict (409, like
        // an alias `Conflict`); a domain/range violation is a refused
        // value (400).
        crate::ingest::ApplyRefusal::Schema(rejection) => {
            let (integrity, durable_batches) = stream_integrity(durable_batches, dry_run);
            let code = if rejection.reserved {
                ErrorCode::Conflict
            } else {
                ErrorCode::InvalidArgument
            };
            let message = format!("{note}{}", rejection.text());
            let issues = schema_issues_in_batch(batch_index, rejection.issues);
            validation_error(
                code,
                message,
                RefusalDetail {
                    issues,
                    integrity: Some(integrity),
                    durable_batches,
                    retryable_after_correction: Some(true),
                },
                started_at,
            )
        }
    }
}

/// Maps a refused group restore onto the response — the group half of
/// [`import_refusal`]. Groups apply after every batch, so the note
/// says what already landed; re-POSTing the corrected stream is exact
/// (batches replace their sources, records their groups).
pub(super) fn restore_refusal(
    state: &AppState,
    refusal: crate::registry::RestoreGroupsError,
    batches_landed: usize,
    started_at: Instant,
) -> Response {
    use crate::registry::RestoreGroupsError;
    let code = match &refusal {
        RestoreGroupsError::InvalidName | RestoreGroupsError::Duplicate(_) => {
            ErrorCode::InvalidArgument
        }
        RestoreGroupsError::NoSuchContext { .. } => ErrorCode::NoContext,
        RestoreGroupsError::NoSuchChild { .. } => ErrorCode::NoGroup,
        RestoreGroupsError::OverCap { .. } => ErrorCode::OverLimit,
        RestoreGroupsError::Nesting(violation) => nesting_error_code(violation),
        RestoreGroupsError::Io { .. } => {
            state.metrics().record_error(ErrorKind::Io);
            ErrorCode::Internal
        }
        // A spent budget is not a server fault — the batch-loop timeout
        // records no error either, and the client resumes by re-POSTing.
        RestoreGroupsError::Timeout { .. } => ErrorCode::Timeout,
    };
    // A budget that ran out mid-restore is a resumable prefix, not a
    // rejected set: some records may have landed, and the batches
    // before them did. The validation arms, by contrast, wrote nothing
    // in the group phase — every batch landed, so the group phase's
    // own refusal is provably a durable prefix (issue #182), never a
    // partial group write of its own (`restore_groups` validates the
    // whole set before applying any of it).
    match &refusal {
        RestoreGroupsError::Timeout { .. } => {
            // Same machine-readable claim `import_budget_refusal` makes
            // for the batch loop: the fields speak for the BATCHES (all
            // durable), which is what a resuming importer keys on — the
            // group phase itself stays a re-POST, as the note says.
            let (integrity, durable_batches) = stream_integrity(batches_landed, false);
            validation_error(
                code,
                format!(
                    "group restore exceeded its budget with {batches_landed} batch(es) durable \
                     (TAGURU_REQUEST_TIMEOUT_SECS tunes this); {}",
                    refusal.text()
                ),
                RefusalDetail {
                    integrity: Some(integrity),
                    durable_batches,
                    ..Default::default()
                },
                started_at,
            )
        }
        _ => {
            let (integrity, durable_batches) = stream_integrity(batches_landed, false);
            validation_error(
                code,
                format!(
                    "group records refused with every batch landed ({batches_landed} durable); \
                     fixing the stream and re-POSTing it whole is exact: {}",
                    refusal.text()
                ),
                RefusalDetail {
                    integrity: Some(integrity),
                    durable_batches,
                    retryable_after_correction: Some(true),
                    ..Default::default()
                },
                started_at,
            )
        }
    }
}

/// Maps one `taguru_schema` record's [`crate::ingest::SchemaApplyError`]
/// onto the response — the schema twin of [`import_refusal`], simpler
/// because `put_schema` is atomic (no partial-write "how much of this
/// one record landed" question a batch's own refusal has to answer).
/// Every batch of the stream already landed durably by the time a
/// schema record is even reached; nothing past this record (later
/// schemas, every group) applies.
pub(super) fn schema_import_refusal(
    state: &AppState,
    context: &str,
    failure: crate::ingest::SchemaApplyError,
    durable_batches_count: usize,
    applied_schemas: usize,
    started_at: Instant,
) -> Response {
    // `integrity` must reflect everything durable so far in the
    // stream, batches AND any schema record that installed before
    // this one failed — `put_schema` is atomic, so an earlier schema
    // of the same stream can already be durably persisted even when
    // zero batches landed (a schemas-only stream). `durable_batches`
    // itself still names only the batch count: a schema is not a
    // batch, and folding the two into one number would mislead a
    // client that uses this field to know how many batches to skip on
    // a corrected resend.
    let integrity = if durable_batches_count + applied_schemas == 0 {
        "nothing_written"
    } else {
        "durable_prefix"
    };
    let durable_batches = (durable_batches_count > 0).then_some(durable_batches_count);
    match &failure {
        crate::ingest::SchemaApplyError::NoContext => validation_error(
            ErrorCode::NoContext,
            format!(
                "schema record: context '{context}' {failure} — nothing past this point \
                 was applied"
            ),
            RefusalDetail {
                issues: vec![Issue::missing(
                    format!("context '{context}'"),
                    "an earlier batch of this stream (or a previous request) creating it, \
                     since a schema record's context must already exist",
                )],
                integrity: Some(integrity),
                durable_batches,
                retryable_after_correction: Some(true),
            },
            started_at,
        ),
        crate::ingest::SchemaApplyError::ReservedAlias(_) => validation_error(
            ErrorCode::InvalidArgument,
            format!(
                "schema record: context '{context}' {failure} — nothing past this point \
                 was applied"
            ),
            RefusalDetail {
                integrity: Some(integrity),
                durable_batches,
                retryable_after_correction: Some(true),
                ..Default::default()
            },
            started_at,
        ),
        crate::ingest::SchemaApplyError::Load(_) => {
            // Unlike `put_schema`'s own Load arm (`src/api/schema.rs`),
            // this event's target is not `taguru::search` — ADR 0008
            // §8's one OTel export exclusion — so it reaches the
            // export layer whenever OTLP is enabled. The detail can
            // name a filesystem path (forbidden by §8) and the context
            // name is never recorded on an event either (§8's
            // per-item rule), so only a low-cardinality reason rides
            // the trace; the response text below still names the
            // context, since the CALLER already sent it.
            tracing::warn!(
                taguru.reason = "schema_load_failed",
                "import schema record failed"
            );
            state.metrics().record_error(ErrorKind::Load);
            error(
                ErrorCode::Internal,
                format!(
                    "schema record: context '{context}' could not be loaded — see server \
                     logs; every batch before it is durable"
                ),
                started_at,
            )
        }
        crate::ingest::SchemaApplyError::Io(_) => {
            tracing::warn!(
                taguru.reason = "schema_write_failed",
                "import schema record failed"
            );
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
                format!(
                    "schema record: context '{context}' schema not persisted — see server \
                     logs; every batch before it is durable"
                ),
                started_at,
            )
        }
    }
}

/// The refusal a budget spent partway through the schema-record loop
/// answers (issue #620) — [`import_budget_refusal`]'s twin for this
/// second phase of the stream, which previously had no deadline check
/// at all: a stream carrying many `taguru_schema` records could run
/// past `TAGURU_REQUEST_TIMEOUT_SECS` entirely, since each record's
/// `put_schema` (context hydration plus two fsyncs) is not free.
/// Accounting mirrors [`schema_import_refusal`]'s own: `durable_batches`
/// names only the batch count (a schema is not a batch), `integrity`
/// folds in `applied_schemas` too, since an earlier schema record of
/// this same stream can already be durable when the budget runs out on
/// a later one.
pub(super) fn schema_import_budget_refusal(
    index: usize,
    total: usize,
    context: &str,
    durable_batches_count: usize,
    applied_schemas: usize,
    started_at: Instant,
) -> Response {
    let integrity = if durable_batches_count + applied_schemas == 0 {
        "nothing_written"
    } else {
        "durable_prefix"
    };
    let durable_batches = (durable_batches_count > 0).then_some(durable_batches_count);
    validation_error(
        ErrorCode::Timeout,
        format!(
            "schema record {} of {total} (context '{context}') not attempted — request \
             exceeded its budget partway through a multi-record schema install \
             (TAGURU_REQUEST_TIMEOUT_SECS tunes this); every batch and schema record \
             before it is durable",
            index + 1,
        ),
        RefusalDetail {
            integrity: Some(integrity),
            durable_batches,
            ..Default::default()
        },
        started_at,
    )
}

/// The "N batches already landed" prefix a multi-batch import prepends
/// to a mid-stream failure — shared by the deadline-exhaustion and the
/// refused-batch cases in [`import_batch`]'s loop, which differ only in
/// the verb for THIS batch and the exact fix to name. A single-batch
/// stream has nothing before it to report, so the note is empty.
pub(super) fn import_batch_note(
    index: usize,
    total: usize,
    batch: &crate::ingest::Batch,
    done: usize,
    dry_run: bool,
    verb: (&str, &str),
    next_step: (&str, &str),
) -> String {
    if total <= 1 {
        return String::new();
    }
    let (verb, clause, next_step) = if dry_run {
        (
            verb.0,
            "previewed clean (dry_run=true; nothing is written)",
            next_step.0,
        )
    } else {
        (verb.1, "landed durably", next_step.1)
    };
    format!(
        "batch {} of {total} (context '{}', source '{}') {verb} — the {done} batch(es) \
         before it {clause}; {next_step}: ",
        index + 1,
        batch.context,
        batch.source,
    )
}

/// `POST /import` — the batch-file contract (docs/import.html) over
/// HTTP: the body IS one batch file — or a whole stream of batches,
/// as `GET /contexts/{name}/export` renders — applied to the live
/// server with the same validate-first, retract-then-apply semantics
/// as `taguru import`. Each batch states one source's complete truth,
/// so bulk loads and restores reach a running server without a
/// downtime window. The body cap, auth, timeout, and rate limit apply
/// as on any endpoint; embeddings ride the next flush
/// (`TAGURU_EMBED_AUTO`) exactly as live writes do.
///
/// `taguru_schema` records (ADR 0009 §13) ride the same stream and
/// install AFTER every batch, BEFORE any group — a schema record's
/// context may exist only because a batch of this same body just
/// created it, and a schema landing before groups restore lets a
/// group's own member-existence check see the context as its final
/// self. Each schema record is independent (one context each), so the
/// first one that fails refuses the request right there; unlike
/// groups' whole-set validation, later schema records and every group
/// are simply never reached.
///
/// `taguru_group` records ride the same stream and apply LAST — after
/// every batch, wherever they sat — so a group and the member
/// contexts it names can travel in one body in any order. Restoring a
/// record replaces the whole group; the set is validated whole and a
/// refusal applies no group, with every batch already durable.
///
/// The response is `{batches: [...]}` in stream order — a single-batch
/// body answers the same shape with one entry, a stream that carried
/// schema records adds `schemas: [...]`, and one that carried group
/// records adds `groups: [...]`. A refusal partway through a stream
/// stops there — the batches before it landed durably, and because
/// every batch is retract-then-apply, re-POSTing the whole corrected
/// stream is exact, never
/// double-counted.
///
/// `?dry_run=true` reports the same `{batches: [...]}` shape without
/// writing anything — parsing and scope checks still run in full, so a
/// malformed or forbidden stream is refused exactly as it would be for
/// real. Each batch's checks are seeded with what the batches before
/// it would intern and create ([`crate::ingest::PreviewSeeds`]), so a
/// restore whose aliases trail their canonicals — every export — and
/// a fresh-name restore's post-first batches preview clean, exactly
/// as they apply. Two counts per batch, `associations` and `aliases`,
/// are optimistic (see [`crate::ingest::preview_batch`]); every other
/// field is exact. `taguru_schema` and `taguru_group` records are both
/// a known gap: they apply through a path (`put_schema`,
/// `restore_groups`) that dry-run does not preview, so a stream
/// carrying either is parsed and scope-checked like normal but not
/// previewed — the response omits `schemas`/`groups` entirely rather
/// than report a guess.
pub async fn import_batch(
    State(state): State<AppState>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppQuery(query): AppQuery<ImportQuery>,
    AppBytes(body): AppBytes,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let stream = match crate::ingest::parse_stream(&body[..]) {
        Ok(stream) => stream,
        // Line-numbered, like the CLI's validation pass. Nothing in the
        // stream has been touched yet — a fixed-up resend is exact.
        Err(message) => {
            return validation_error(
                ErrorCode::MalformedRequest,
                message,
                RefusalDetail {
                    integrity: Some("nothing_written"),
                    retryable_after_correction: Some(true),
                    ..Default::default()
                },
                started_at,
            );
        }
    };
    // Import's contexts live in the BODY, out of the route-level
    // authorization check's reach — a context-scoped key is judged
    // here instead, before anything applies.
    if let Some(axum::Extension(scope)) = &scope
        && let Some(refused) = stream
            .batches
            .iter()
            .find(|batch| !scope.allows_context(&batch.context))
    {
        return validation_error(
            ErrorCode::Forbidden,
            format!(
                "key '{}' has no grant on context '{}' (batch source '{}'); nothing \
                 was applied",
                key_name(&key),
                refused.context,
                refused.source
            ),
            RefusalDetail {
                integrity: Some("nothing_written"),
                ..Default::default()
            },
            started_at,
        );
    }
    // Schema records are judged by the same context grant as a batch,
    // before anything applies — one step earlier than the group check
    // just below, since schemas install before groups restore (ADR
    // 0009 §13).
    if let Some(axum::Extension(scope)) = &scope
        && let Some((context, _)) = stream
            .schemas
            .iter()
            .find(|(context, _)| !scope.allows_context(context))
    {
        return validation_error(
            ErrorCode::Forbidden,
            format!(
                "key '{}' has no grant on context '{context}' (a schema record); nothing \
                 was applied",
                key_name(&key)
            ),
            RefusalDetail {
                integrity: Some("nothing_written"),
                ..Default::default()
            },
            started_at,
        );
    }
    // Group records are judged the way every group write is: by the
    // context closure — the standing record's and the prospective
    // one's both — before anything applies. Gated on the scope first,
    // [`scoped_group_refusal`]'s discipline: an unscoped key never
    // pays for the closure read.
    if scope.is_some()
        && !stream.groups.is_empty()
        && let Some(refusal) = scope_refusal(
            &scope,
            &key,
            &state.group_restore_involves(&stream.groups),
            started_at,
        )
    {
        return refusal;
    }
    let total = stream.batches.len();
    // Each batch is a create/retract_source/store_passages/add_associations/
    // add_aliases sequence — fsync-bearing writes, same as every other
    // mutating endpoint — run back to back; keep the whole loop off the
    // async worker rather than just one call in it.
    let outcome = tokio::task::block_in_place(|| {
        let mut outcomes: Vec<ImportOutcome> = Vec::with_capacity(total);
        // `warn`-mode schema violations (ADR 0009 §8.3), accumulated
        // across every batch in the stream with this batch's own
        // `batches[{index}]` prefix already applied. `warn_issues` is
        // capped as it is collected, not just at the end — a large
        // `warn`-mode stream must not first pile up a multiple of
        // `MAX_LISTED_ISSUES` (one batch's worth each) only to discard
        // the excess. `warn_total`, unlike the list, is never capped —
        // `ok_with_issues_total` (`src/api.rs`) needs the true
        // cross-batch count, which `warn_issues.len()` alone could no
        // longer answer once the cap trims a later batch's
        // contribution; each batch's own `ImportOutcome.schema_violations`
        // carries its individual count regardless of either total.
        let mut warn_issues: Vec<Issue> = Vec::new();
        let mut warn_total: usize = 0;
        // Dry run only: what earlier batches of this stream would have
        // interned by the time each batch applies for real — see
        // [`crate::ingest::PreviewSeeds`].
        let mut seeds = crate::ingest::PreviewSeeds::default();
        for (index, batch) in stream.batches.iter().enumerate() {
            // Each landed batch is durable (retract-then-apply), so a
            // budget that runs out partway is safe to report as a
            // resumable prefix rather than an all-or-nothing failure.
            if deadline.expired() {
                return Err(Box::new(import_budget_refusal(
                    index,
                    total,
                    batch,
                    outcomes.len(),
                    query.dry_run,
                    started_at,
                )));
            }
            // The storage-quota pre-check (issue #136), batch-granular
            // like the deadline above and skipped on dry runs — a
            // preview's capacity answers are documented as advisory
            // ("can only surface by actually applying"). Only batches
            // that CARRY GROWTH are gated: a header-only batch is a
            // pure source retraction, the import-shaped way down in
            // size, and must stay open at the ceiling exactly as the
            // write path keeps retract/unalias open. The WAL lanes
            // behind the check are live, so re-checking per batch IS
            // the running remaining-budget check: used bytes advance
            // as batches land, and the import stops at the first batch
            // past the line — a resumable prefix, like a spent
            // deadline. The gates inside the write path still stand
            // behind this for the batch that crosses mid-apply.
            if !query.dry_run
                && batch.carries_growth()
                && let Some((used, ceiling)) = state.storage_quota_refusal(&batch.context)
            {
                state.metrics().record_storage_quota_refusal();
                let note = import_batch_note(
                    index,
                    total,
                    batch,
                    outcomes.len(),
                    query.dry_run,
                    ("not previewed", "not attempted"),
                    (
                        "re-running the preview against a shrunk context is exact",
                        "retracting or compacting the context (or raising its quota), then \
                         re-POSTing the remaining stream is exact (each batch replaces its \
                         own source)",
                    ),
                );
                let (integrity, durable_batches) = stream_integrity(outcomes.len(), query.dry_run);
                return Err(Box::new(validation_error(
                    ErrorCode::StorageFull,
                    format!(
                        "{note}{}",
                        crate::registry::storage_quota_message(&batch.context, used, ceiling)
                    ),
                    RefusalDetail {
                        integrity: Some(integrity),
                        durable_batches,
                        ..Default::default()
                    },
                    started_at,
                )));
            }
            let applied = if query.dry_run {
                let previewed = crate::ingest::preview_batch(&state, batch, &seeds);
                if previewed.is_ok() {
                    seeds.absorb(batch);
                }
                previewed
            } else {
                crate::ingest::apply_batch(&state, batch)
            };
            match applied {
                Ok(applied) => {
                    // Import is retract-then-apply — destructive — and both
                    // the context and the replaced source live in the BODY,
                    // out of the access log's reach. One audit line each.
                    // `dry_run` marks the ones that wrote nothing.
                    tracing::info!(
                        target: "taguru::audit",
                        key = %key_name(&key),
                        context = %batch.context,
                        source = %batch.source,
                        created = applied.created,
                        retracted = applied.retracted,
                        associations = applied.associations,
                        dry_run = query.dry_run,
                        "import batch applied",
                    );
                    outcomes.push(import_outcome(batch, &applied));
                    warn_total += applied.schema_violations;
                    if warn_issues.len() < crate::api::MAX_LISTED_ISSUES {
                        warn_issues.extend(schema_issues_in_batch(index, applied.schema_issues));
                        warn_issues.truncate(crate::api::MAX_LISTED_ISSUES);
                    }
                }
                Err(refusal) => {
                    let note = import_batch_note(
                        index,
                        total,
                        batch,
                        outcomes.len(),
                        query.dry_run,
                        ("would be refused", "refused"),
                        (
                            "fixing the stream and re-running the preview is exact",
                            "fixing the stream and re-POSTing it whole is exact (each batch \
                             replaces its own source)",
                        ),
                    );
                    return Err(Box::new(import_refusal(
                        &state,
                        batch,
                        refusal,
                        &note,
                        index,
                        outcomes.len(),
                        query.dry_run,
                        started_at,
                    )));
                }
            }
        }
        // Schemas install after every batch, before groups restore
        // (ADR 0009 §13): a schema record's context may exist only
        // because a batch of THIS SAME stream just created it. Like
        // `taguru_group` records, `put_schema` has no read-only twin
        // to preview through, so a dry run skips this entirely — the
        // response omits `schemas` rather than report a guess. Unlike
        // groups (one whole SET validated together), each schema
        // record names one independent context, so the first one that
        // fails refuses the request right there — every batch before
        // it stays durable, nothing past it (later schemas, every
        // group) applies.
        let mut schema_outcomes: Vec<SchemaImportOutcome> = Vec::new();
        if !query.dry_run {
            for (index, (context, installed)) in stream.schemas.iter().enumerate() {
                // Batch-granular like the loop above (issue #620):
                // each landed batch AND each installed schema record
                // is durable, so a budget that runs out partway is
                // safe to report as a resumable prefix.
                if deadline.expired() {
                    return Err(Box::new(schema_import_budget_refusal(
                        index,
                        stream.schemas.len(),
                        context,
                        outcomes.len(),
                        schema_outcomes.len(),
                        started_at,
                    )));
                }
                match crate::ingest::apply_schema_record(&state, context, installed.clone()) {
                    Ok(document) => {
                        tracing::info!(
                            target: "taguru::audit",
                            key = %key_name(&key),
                            context = %context,
                            mode = document.mode.as_str(),
                            "import schema record installed",
                        );
                        schema_outcomes.push(schema_import_outcome(context, &document));
                    }
                    Err(failure) => {
                        return Err(Box::new(schema_import_refusal(
                            &state,
                            context,
                            failure,
                            outcomes.len(),
                            schema_outcomes.len(),
                            started_at,
                        )));
                    }
                }
            }
        }
        // Groups apply LAST — after every batch — so a record and the
        // member contexts it names can ride one stream in any order.
        // `taguru_group` records apply through a separate path
        // (`restore_groups`) that has no read-only twin, so a dry run
        // skips them entirely — the response omits `groups` rather than
        // report a guess.
        let mut group_outcomes: Vec<GroupImportOutcome> = Vec::new();
        if !stream.groups.is_empty() && !query.dry_run {
            match state.restore_groups(&stream.groups, deadline) {
                Ok(restored) => {
                    for ((name, record), (_, applied)) in stream.groups.iter().zip(&restored) {
                        // A restore replaces the whole record —
                        // destructive like the batches above, and just
                        // as far out of the access log's reach.
                        tracing::info!(
                            target: "taguru::audit",
                            key = %key_name(&key),
                            group = %name,
                            outcome = applied.as_str(),
                            contexts = record.contexts.len(),
                            children = record.groups.len(),
                            "import group record applied",
                        );
                        group_outcomes.push(GroupImportOutcome {
                            name: name.clone(),
                            outcome: applied.as_str(),
                            contexts: record.contexts.len(),
                            groups: record.groups.len(),
                        });
                    }
                }
                Err(refusal) => {
                    return Err(Box::new(restore_refusal(
                        &state, refusal, total, started_at,
                    )));
                }
            }
        }
        Ok((
            outcomes,
            schema_outcomes,
            group_outcomes,
            warn_issues,
            warn_total,
        ))
    });
    let (outcomes, schema_outcomes, group_outcomes, warn_issues, warn_total) = match outcome {
        Ok(applied) => applied,
        Err(refusal) => return *refusal,
    };
    ok_with_issues_total(
        ImportStreamOutcome {
            batches: outcomes,
            schemas: schema_outcomes,
            groups: group_outcomes,
        },
        warn_issues,
        warn_total,
        started_at,
    )
}

/// `POST /contexts/{name}/compact` — rebuild the image without the
/// dead weight the append-only format accumulates (retracted edges,
/// unlinked attributions, arena slack), persisting the result before
/// answering. An admin verb (the role table's fail-closed default);
/// the context's own requests wait out the rebuild, every other
/// context is untouched. Content is preserved — the response says
/// what was shed and what the footprint became.
pub async fn compact_context(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match tokio::task::block_in_place(|| state.compact_context(&name, deadline)) {
        Ok(outcome) => {
            // Maintenance that rewrites the image is audit-worthy even
            // though no knowledge changes.
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                bytes_before = outcome.bytes_before,
                bytes_after = outcome.bytes_after,
                dead_edges = outcome.dead_edges,
                aliases_dropped = outcome.aliases_dropped,
                image_persisted = outcome.image_persisted,
                "context compacted",
            );
            ok(outcome, started_at)
        }
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

/// `GET /contexts/{name}/export` — the context back out as the import
/// batch stream (docs/import.html): one batch per source in source-id
/// order, the create block on the first, the alias table on the last,
/// sourceless weight in a reserved `export:unsourced` batch. The
/// response body IS the stream (JSON Lines, not the JSON envelope):
/// save it, and `taguru import` or `POST /import` restores it —
/// per-source retract-then-apply, so re-importing is idempotent.
/// Materialized under one registry fence (a concurrent write cannot
/// shear the graph against the passages) and rendered off the async
/// runtime, the way vocabulary/audit steps aside.
pub async fn export_context(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let rendered = tokio::task::block_in_place(|| {
        state
            .export_context(&name, deadline)
            .map(|snapshot| crate::export::render(&name, &snapshot, deadline))
    });
    export_response(&state, &name, rendered, deadline, started_at)
}

/// Maps [`export_context`]'s materialize-and-render outcome onto the
/// response — the stream itself, or which refusal a render failure is.
pub(super) fn export_response(
    state: &AppState,
    name: &str,
    rendered: Result<Result<crate::export::Rendered, String>, AccessError>,
    deadline: Deadline,
    started_at: Instant,
) -> Response {
    match rendered {
        Ok(Ok(rendered)) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "application/x-ndjson; charset=utf-8",
            )],
            rendered.stream,
        )
            .into_response(),
        // Mirrors search_passages: render()'s own loop over a large
        // context's associations/aliases can outlast the budget after
        // export_context's entry check already passed — reclassify by
        // asking the same deadline again rather than by matching
        // render()'s message text.
        Ok(Err(_)) if deadline.expired() => deadline_exceeded(started_at),
        // A real source colliding with a reserved export id — the one
        // thing a context can hold that the stream cannot say.
        Ok(Err(message)) => error(ErrorCode::Conflict, message, started_at),
        Err(failure) => access_error(state, failure, name, started_at),
    }
}

/// `GET /groups/{name}/export` — the group back out as its import
/// record: one `taguru_group` line (JSON Lines body, not the JSON
/// envelope), [`export_context`]'s twin one storey up. `POST /import`
/// (or `taguru import`) restores it by replacing the whole record, so
/// re-importing is idempotent; the members must exist at import time,
/// and batches of the same stream apply first. For a context-scoped
/// key the members are the grant's slice, exactly as
/// `GET /groups/{name}` serves them — the export IS that key's view,
/// and restoring it elsewhere carries only what the key could see.
pub async fn export_group(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
) -> Response {
    let started_at = Instant::now();
    let Some(record) = state.group(&name) else {
        return group_not_found(&name, started_at);
    };
    let filtered = GroupRecord {
        description: record.description,
        contexts: scoped_member_contexts(record.contexts, &scope),
        // Child names stay whole, as on the row: labels, not content.
        groups: record.groups,
    };
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "application/x-ndjson; charset=utf-8",
        )],
        crate::export::render_group(&name, &filtered),
    )
        .into_response()
}
