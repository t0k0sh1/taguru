use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::groups::GroupRecord;
use crate::metrics::ErrorKind;
use crate::registry::AppState;

use super::groups::{scope_refusal, scoped_member_contexts};
use super::{
    AppBytes, AppPath, AppQuery, ErrorCode, access_error, access_error_noted, deadline_exceeded,
    error, group_not_found, key_name, nesting_error_code, ok,
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
    /// Association paragraph locators dropped for naming a spot the
    /// batch's passage split does not have — the association itself
    /// still landed. Reported like `questions_dropped`/`sections_dropped`.
    pub association_paragraphs_dropped: usize,
}

/// What a `POST /import` body accomplished — one [`ImportOutcome`] per
/// batch, in stream order. Every import answers this shape, a
/// single-batch body included: one shape to parse, not two.
#[derive(Serialize)]
pub struct ImportStreamOutcome {
    pub batches: Vec<ImportOutcome>,
    /// One entry per `taguru_group` record, stream order — absent
    /// entirely for a stream that carried none, keeping the pre-group
    /// response byte-identical.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupImportOutcome>,
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

fn import_outcome(batch: &crate::ingest::Batch, applied: &crate::ingest::Applied) -> ImportOutcome {
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
        association_paragraphs_dropped: applied.association_paragraphs_dropped,
    }
}

/// Maps one batch's [`ApplyRefusal`](crate::ingest::ApplyRefusal) onto
/// the response, `note` naming which batch of a stream refused (empty
/// for a single-batch body, keeping that path's responses exactly as
/// they were).
fn import_refusal(
    state: &AppState,
    batch: &crate::ingest::Batch,
    refusal: crate::ingest::ApplyRefusal,
    note: &str,
    started_at: Instant,
) -> Response {
    match refusal {
        // The three AccessError arms — status, metric, message — live
        // in access_error_noted; import just supplies the batch note.
        crate::ingest::ApplyRefusal::Access(failure) => {
            access_error_noted(state, failure, &batch.context, note, started_at)
        }
        refusal @ crate::ingest::ApplyRefusal::NoContext(_) => error(
            ErrorCode::NoContext,
            format!("{note}{}", refusal.text()),
            started_at,
        ),
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
    // in the group phase.
    let message = match &refusal {
        RestoreGroupsError::Timeout { .. } => format!(
            "group restore exceeded its budget with {batches_landed} batch(es) durable \
             (TAGURU_REQUEST_TIMEOUT_SECS tunes this); {}",
            refusal.text()
        ),
        _ => format!(
            "group records refused with every batch landed ({batches_landed} durable); \
             fixing the stream and re-POSTing it whole is exact: {}",
            refusal.text()
        ),
    };
    error(code, message, started_at)
}

/// The "N batches already landed" prefix a multi-batch import prepends
/// to a mid-stream failure — shared by the deadline-exhaustion and the
/// refused-batch cases in [`import_batch`]'s loop, which differ only in
/// the verb for THIS batch and the exact fix to name. A single-batch
/// stream has nothing before it to report, so the note is empty.
fn import_batch_note(
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
/// `taguru_group` records ride the same stream and apply LAST — after
/// every batch, wherever they sat — so a group and the member
/// contexts it names can travel in one body in any order. Restoring a
/// record replaces the whole group; the set is validated whole and a
/// refusal applies no group, with every batch already durable.
///
/// The response is `{batches: [...]}` in stream order — a single-batch
/// body answers the same shape with one entry, and a stream that
/// carried group records adds `groups: [...]`. A refusal partway
/// through a stream stops there — the batches before it landed
/// durably, and because every batch is retract-then-apply,
/// re-POSTing the whole corrected stream is exact, never
/// double-counted.
///
/// `?dry_run=true` reports the same `{batches: [...]}` shape without
/// writing anything — parsing and scope checks still run in full, so a
/// malformed or forbidden stream is refused exactly as it would be for
/// real. Two counts per batch, `associations` and `aliases`, are
/// optimistic (see [`crate::ingest::preview_batch`]); every other
/// field is exact. `taguru_group` records are a known gap: they apply
/// through a separate path (`restore_groups`) that dry-run does not
/// preview, so a stream carrying any is parsed and scope-checked like
/// normal but its `groups` are silently not previewed — the response
/// omits `groups` entirely rather than report a guess.
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
        // Line-numbered, like the CLI's validation pass.
        Err(message) => return error(ErrorCode::MalformedRequest, message, started_at),
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
        return error(
            ErrorCode::Forbidden,
            format!(
                "key '{}' has no grant on context '{}' (batch source '{}'); nothing \
                 was applied",
                key_name(&key),
                refused.context,
                refused.source
            ),
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
        for (index, batch) in stream.batches.iter().enumerate() {
            // Each landed batch is durable (retract-then-apply), so a
            // budget that runs out partway is safe to report as a
            // resumable prefix rather than an all-or-nothing failure.
            if deadline.expired() {
                let note = import_batch_note(
                    index,
                    total,
                    batch,
                    outcomes.len(),
                    query.dry_run,
                    ("not previewed", "not attempted"),
                    (
                        "re-running the preview with more time or a narrower stream is exact",
                        "re-POSTing the remaining stream is exact (each batch replaces its \
                         own source)",
                    ),
                );
                return Err(Box::new(error(
                    ErrorCode::Timeout,
                    format!(
                        "{note}request exceeded its budget partway through a multi-batch \
                         import (TAGURU_REQUEST_TIMEOUT_SECS tunes this)"
                    ),
                    started_at,
                )));
            }
            // The storage-quota pre-check (issue #136), batch-granular
            // like the deadline above and skipped on dry runs — a
            // preview's capacity answers are documented as advisory
            // ("can only surface by actually applying"). The WAL lanes
            // behind the check are live, so re-checking per batch IS
            // the running remaining-budget check: used bytes advance
            // as batches land, and the import stops at the first batch
            // past the line — a resumable prefix, like a spent
            // deadline. The gates inside the write path still stand
            // behind this for the batch that crosses mid-apply.
            if !query.dry_run
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
                return Err(Box::new(error(
                    ErrorCode::StorageFull,
                    format!(
                        "{note}{}",
                        crate::registry::storage_quota_message(&batch.context, used, ceiling)
                    ),
                    started_at,
                )));
            }
            let applied = if query.dry_run {
                crate::ingest::preview_batch(&state, batch)
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
                        &state, batch, refusal, &note, started_at,
                    )));
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
        Ok((outcomes, group_outcomes))
    });
    let (outcomes, group_outcomes) = match outcome {
        Ok(applied) => applied,
        Err(refusal) => return *refusal,
    };
    ok(
        ImportStreamOutcome {
            batches: outcomes,
            groups: group_outcomes,
        },
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
        // the entry check above already passed — reclassify by asking
        // the same deadline again rather than by matching render()'s
        // message text.
        Ok(Err(_)) if deadline.expired() => deadline_exceeded(started_at),
        // A real source colliding with a reserved export id — the one
        // thing a context can hold that the stream cannot say.
        Ok(Err(message)) => error(ErrorCode::Conflict, message, started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
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
