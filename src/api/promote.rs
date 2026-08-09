//! `POST /contexts/{name}/promote` (ADR 0018): graph-path promotion —
//! the named scratch sources move into an established destination
//! context as export → filter → re-head → import → audit, one
//! request, built from the same machinery the manual runbook uses
//! (docs/promotion.html). No LLM anywhere in the path: the scratch's
//! structure is already right, so the transfer is the export/import
//! round trip, never a re-extraction. The judgment points stay with
//! the caller — WHICH sources to promote (this request's input), what
//! to do with the audit's candidates (its output), and when to retire
//! the promoted scratch (`retract_source`, never here).

use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;
use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::metrics::ErrorKind;
use crate::registry::{AccessError, AppState};

use super::consolidation::{
    CONSOLIDATION_DETECTOR, ConsolidationAudit, DEFAULT_EVIDENCE_CAP, contradiction_section,
    merge_section, staleness_section,
};
use super::import::{
    ImportOutcome, import_batch_note, import_outcome, import_refusal, schema_issues_in_batch,
    stream_integrity,
};
use super::{
    AppJson, AppPath, AppQuery, ErrorCode, Issue, RefusalDetail, access_error, deadline_exceeded,
    error, key_name, ok_with_issues_total, validation_error,
};

#[derive(Debug, Deserialize)]
pub struct PromoteRequest {
    /// The destination context — must already exist; promote never
    /// creates one.
    pub into: String,
    /// The scratch source ids to promote. Every one must exist in the
    /// promoting context (a passage or a live attribution) or the
    /// request refuses whole — under retract-then-apply a mistyped
    /// session id would otherwise no-op silently.
    pub sources: Vec<String>,
    /// Omitted means true: after a real apply, the destination gets
    /// the default consolidation audit (ADR 0012 — all three checks,
    /// default ceilings) and its candidates ride back under `audit`.
    pub audit: Option<bool>,
}

/// `?dry_run=true`: preview the same `{batches: [...]}` with nothing
/// written — [`crate::ingest::preview_batch`]'s counts, no audit
/// (nothing landed to audit).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct PromoteQuery {
    pub dry_run: bool,
}

/// What one promotion accomplished: `/import`'s per-batch outcome
/// shape (one entry per promoted source, source-id order), the alias
/// accounting, and the landing zone's audit candidates.
#[derive(Serialize)]
pub struct PromoteOutcome {
    pub batches: Vec<ImportOutcome>,
    /// Scratch aliases whose canonical had no live edge in the
    /// promoted slice — left behind, and counted rather than silent
    /// (the export render's own accounting).
    pub aliases_dropped: usize,
    /// The destination's consolidation audit after the apply —
    /// candidates for the caller to judge, never applications. Absent
    /// on `dry_run`, on `audit: false`, and when the audit itself
    /// could not run (`audit_skipped` says why).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audit: Option<ConsolidationAudit>,
    /// Why `audit` is absent when it was supposed to run — the writes
    /// above are already durable by audit time, so an audit failure
    /// degrades to this note instead of failing a request whose
    /// batches landed. `audit_consolidation` re-runs it standalone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audit_skipped: Option<&'static str>,
}

pub async fn promote_sources(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppQuery(query): AppQuery<PromoteQuery>,
    AppJson(request): AppJson<PromoteRequest>,
) -> Response {
    let started_at = Instant::now();
    if request.sources.is_empty() {
        return error(
            ErrorCode::InvalidArgument,
            "'sources' must name at least one source id to promote — choosing which \
             sources are the keepers is the caller's judgment, never a default",
            started_at,
        );
    }
    if request.into == name {
        return error(
            ErrorCode::InvalidArgument,
            format!(
                "'into' names the promoting context '{name}' itself — promotion moves \
                 sources into a DIFFERENT, established context"
            ),
            started_at,
        );
    }
    // The destination lives in the BODY, out of the route-level
    // authorization check's reach — a context-scoped key is judged
    // here instead, before anything applies (`/import`'s discipline).
    if let Some(axum::Extension(scope)) = &scope
        && !scope.allows_context(&request.into)
    {
        return validation_error(
            ErrorCode::Forbidden,
            format!(
                "key '{}' has no grant on context '{}' ('into'); nothing was applied",
                key_name(&key),
                request.into
            ),
            RefusalDetail {
                integrity: Some("nothing_written"),
                ..Default::default()
            },
            started_at,
        );
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Promote never creates the destination — checked up front so a
    // typo'd name refuses before the scratch is even materialized (a
    // deletion between here and the apply is still caught: the create
    // blocks are stripped below, so `apply_batch` answers NoContext).
    if !state.context_exists(&request.into) {
        return validation_error(
            ErrorCode::NoContext,
            format!(
                "context '{}' ('into') does not exist — promotion lands in an \
                 established context and never creates one; create it first",
                request.into
            ),
            RefusalDetail {
                integrity: Some("nothing_written"),
                retryable_after_correction: Some(true),
                ..Default::default()
            },
            started_at,
        );
    }
    let requested: BTreeSet<String> = request.sources.iter().cloned().collect();
    let snapshot = match tokio::task::block_in_place(|| state.export_context(&name, deadline)) {
        Ok(snapshot) => snapshot,
        Err(failure) => return access_error(&state, failure, &name, started_at),
    };
    // Every requested id must exist in the scratch — as a passage or a
    // live attribution — or the request refuses whole, naming the
    // absentees path-addressed. Nothing has been written yet.
    let available: BTreeSet<&str> = snapshot
        .passages
        .iter()
        .map(|(source, _)| source.as_str())
        .chain(snapshot.associations.iter().flat_map(|association| {
            association
                .attributions
                .iter()
                .filter(|attribution| attribution.count > 0)
                .map(|attribution| attribution.source.as_str())
        }))
        .collect();
    let issues: Vec<Issue> = request
        .sources
        .iter()
        .enumerate()
        .filter(|(_, source)| !available.contains(source.as_str()))
        .map(|(index, source)| {
            Issue::unknown_reference(
                format!("sources[{index}]"),
                format!("'{source}' stored in context '{name}' as a passage or a live attribution"),
            )
        })
        .collect();
    if !issues.is_empty() {
        return validation_error(
            ErrorCode::NoSource,
            format!(
                "{} of the named source id(s) exist(s) nowhere in context '{name}' — \
                 under retract-then-apply a mistyped id would no-op silently, so the \
                 whole request refuses instead; nothing was applied",
                issues.len()
            ),
            RefusalDetail {
                issues,
                integrity: Some("nothing_written"),
                retryable_after_correction: Some(true),
                ..Default::default()
            },
            started_at,
        );
    }
    // The transfer IS the export/import round trip: filter the
    // snapshot to the keepers, render an ordinary import stream headed
    // at the destination, parse it back, apply batch by batch.
    let filtered = crate::export::filter_to_sources(snapshot, &requested);
    let rendered = match tokio::task::block_in_place(|| {
        crate::export::render(&request.into, &filtered, deadline)
    }) {
        Ok(rendered) => rendered,
        Err(_) if deadline.expired() => return deadline_exceeded(started_at),
        // A scratch source colliding with a reserved export id — the
        // one thing the slice can hold that the stream cannot say.
        Err(message) => return error(ErrorCode::Conflict, message, started_at),
    };
    let mut stream = match crate::ingest::parse_stream(rendered.stream.as_bytes()) {
        Ok(stream) => stream,
        // Unreachable by construction — the stream is our own render —
        // but a refusal here must not pretend anything was applied.
        Err(message) => {
            state.metrics().record_error(ErrorKind::Io);
            return error(
                ErrorCode::Internal,
                format!("promotion stream failed to re-parse: {message}"),
                started_at,
            );
        }
    };
    let total = stream.batches.len();
    let outcome = tokio::task::block_in_place(|| {
        let mut outcomes: Vec<ImportOutcome> = Vec::with_capacity(total);
        // `warn`-mode schema violations from the DESTINATION's schema,
        // accumulated exactly as `/import` accumulates them.
        let mut warn_issues: Vec<Issue> = Vec::new();
        let mut warn_total: usize = 0;
        // Dry run only: what earlier batches of this stream would have
        // interned by the time each batch applies for real — see
        // [`crate::ingest::PreviewSeeds`].
        let mut seeds = crate::ingest::PreviewSeeds::default();
        for (index, batch) in stream.batches.iter_mut().enumerate() {
            // Promotion lands in an established context, never mints
            // one — see [`crate::ingest::Batch::strip_create`].
            batch.strip_create();
            if deadline.expired() {
                let note = import_batch_note(
                    index,
                    total,
                    batch,
                    outcomes.len(),
                    query.dry_run,
                    ("not previewed", "not attempted"),
                    (
                        "re-running the preview with more time or fewer sources is exact",
                        "re-calling promote with the same sources is exact (each batch \
                         replaces its own source)",
                    ),
                );
                let (integrity, durable_batches) = stream_integrity(outcomes.len(), query.dry_run);
                return Err(Box::new(validation_error(
                    ErrorCode::Timeout,
                    format!(
                        "{note}request exceeded its budget partway through the promotion \
                         (TAGURU_REQUEST_TIMEOUT_SECS tunes this)"
                    ),
                    RefusalDetail {
                        integrity: Some(integrity),
                        durable_batches,
                        ..Default::default()
                    },
                    started_at,
                )));
            }
            // The destination's storage-quota pre-check, `/import`'s
            // own batch-granular discipline: only growth is gated, the
            // lanes are live, and the stop is a resumable prefix.
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
                        "re-running the preview against a shrunk destination is exact",
                        "retracting or compacting the destination (or raising its quota), \
                         then re-calling promote is exact (each batch replaces its own \
                         source)",
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
                    // Retract-then-apply into the destination is as
                    // destructive as an import, and both context and
                    // source live in the body — one audit line each,
                    // naming the scratch the truth came FROM.
                    tracing::info!(
                        target: "taguru::audit",
                        key = %key_name(&key),
                        from = %name,
                        context = %batch.context,
                        source = %batch.source,
                        retracted = applied.retracted,
                        associations = applied.associations,
                        dry_run = query.dry_run,
                        "promote batch applied",
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
                            "fixing the scratch (or the destination) and re-running the \
                             preview is exact",
                            "fixing the scratch (or the destination) and re-calling \
                             promote is exact (each batch replaces its own source)",
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
        Ok((outcomes, warn_issues, warn_total))
    });
    let (outcomes, warn_issues, warn_total) = match outcome {
        Ok(applied) => applied,
        Err(refusal) => return *refusal,
    };
    // The landing zone's audit (runbook step 4), bundled so the step
    // people skip runs by default — candidates only; judging them and
    // applying judgments stay ordinary follow-up calls. By this point
    // every batch above is durable, so an audit failure degrades to
    // `audit_skipped` instead of failing a request that already wrote.
    let (audit, audit_skipped) = if query.dry_run || !request.audit.unwrap_or(true) {
        (None, None)
    } else {
        match tokio::task::block_in_place(|| landing_audit(&state, &request.into, deadline)) {
            Ok(audit) => (Some(audit), None),
            Err(reason) => (None, Some(reason)),
        }
    };
    ok_with_issues_total(
        PromoteOutcome {
            batches: outcomes,
            aliases_dropped: rendered.aliases_dropped,
            audit,
            audit_skipped,
        },
        warn_issues,
        warn_total,
        started_at,
    )
}

/// The destination's consolidation audit with `audit_consolidation`'s
/// own defaults — all three checks, limit 100, evidence cap
/// [`DEFAULT_EVIDENCE_CAP`], floors 0.6. Tuned re-runs belong to the
/// standalone endpoint (fingerprint reuse makes them cheap); the
/// error is a machine-readable reason for [`PromoteOutcome::
/// audit_skipped`], never a refusal.
fn landing_audit(
    state: &AppState,
    into: &str,
    deadline: Deadline,
) -> Result<ConsolidationAudit, &'static str> {
    let hidden = state.hidden_label(into);
    let effective: HashMap<String, u64> = match state.source_effective_times(into) {
        None => return Err("no_context"),
        Some(Err(io_error)) => {
            tracing::warn!(context = %into, "source metadata read failed: {io_error}");
            state.metrics().record_error(ErrorKind::Load);
            return Err("metadata_unreadable");
        }
        Some(Ok(map)) => map,
    };
    let merge = merge_section(
        state,
        into,
        0.6,
        0.6,
        DEFAULT_EVIDENCE_CAP,
        100,
        hidden,
        deadline,
    )
    .map_err(audit_skip_reason)?;
    let contradiction = state
        .read_context(into, |context| {
            contradiction_section(context, &effective, 100, deadline)
        })
        .and_then(std::convert::identity)
        .map_err(audit_skip_reason)?;
    let staleness = state
        .read_context(into, |context| {
            staleness_section(context, &effective, 0, 100)
        })
        .map_err(audit_skip_reason)?;
    Ok(ConsolidationAudit {
        detector: CONSOLIDATION_DETECTOR.to_string(),
        merge: Some(merge),
        contradiction: Some(contradiction),
        staleness: Some(staleness),
    })
}

fn audit_skip_reason(failure: AccessError) -> &'static str {
    match failure {
        AccessError::DeadlineExceeded => "deadline_exceeded",
        AccessError::NotFound => "no_context",
        _ => "unavailable",
    }
}
