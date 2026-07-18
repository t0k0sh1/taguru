use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::registry::{AppState, AssocOp};

use super::{
    AppJson, AppPath, ErrorCode, MAX_ASSOCIATION_WEIGHT, MAX_ASSOCIATIONS_PER_REQUEST,
    MAX_NAME_BYTES, access_error, deadline_exceeded, empty, error, key_name, ok, oversized,
    partial_write_error,
};

/// A batch of assertions — the arguments of `associate` /
/// `associate_from` as JSON fields ([`AssocOp`], shared with the WAL).
/// A batch is the natural write unit: one document's extracted facts,
/// one request, one lock acquisition, one WAL fsync.
pub async fn add_associations(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(associations): AppJson<Vec<AssocOp>>,
) -> Response {
    let started_at = Instant::now();
    // Refused before the write lock is even taken: nothing of an
    // oversized batch is applied.
    if associations.len() > MAX_ASSOCIATIONS_PER_REQUEST {
        return error(
            ErrorCode::OverLimit,
            format!(
                "batch of {} associations exceeds the per-request limit of \
                 {MAX_ASSOCIATIONS_PER_REQUEST}; split the ingest",
                associations.len()
            ),
            started_at,
        );
    }
    // Also refused whole: a weight the graph must never accumulate.
    // JSON cannot carry Infinity or NaN, but it carries 1e300 just
    // fine, and saturation never washes out of an edge.
    for (index, op) in associations.iter().enumerate() {
        if !op.weight.is_finite() || op.weight.abs() > MAX_ASSOCIATION_WEIGHT {
            return error(
                ErrorCode::InvalidArgument,
                format!(
                    "associations[{index}].weight {} is outside the accepted range \
                     (finite, |weight| <= {MAX_ASSOCIATION_WEIGHT}); nothing was applied",
                    op.weight
                ),
                started_at,
            );
        }
        // And a name the graph must never intern — see MAX_NAME_BYTES.
        let source = op.source.as_deref().unwrap_or("");
        for (field, value) in [
            ("subject", op.subject.as_str()),
            ("label", op.label.as_str()),
            ("object", op.object.as_str()),
            ("source", source),
        ] {
            if let Some(refusal) = oversized(
                &format!("associations[{index}].{field}"),
                value,
                MAX_NAME_BYTES,
                started_at,
            ) {
                return refusal;
            }
        }
        // subject/label/object name the triple itself and must carry
        // something. A source may be OMITTED — that is the ordinary
        // unsourced-association case (see AssocOp::source) — but a
        // source that is PRESENT is a name like any other: an empty
        // string would intern a real, permanent source id that every
        // later attribution list displays and that unrelated callers'
        // mistakes silently merge into.
        for (field, value) in [
            ("subject", Some(op.subject.as_str())),
            ("label", Some(op.label.as_str())),
            ("object", Some(op.object.as_str())),
            ("source", op.source.as_deref()),
        ] {
            let Some(value) = value else {
                continue;
            };
            if let Some(refusal) =
                empty(&format!("associations[{index}].{field}"), value, started_at)
            {
                return refusal;
            }
        }
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let total = associations.len();
    // The write stages ops in the WAL and fsyncs before returning, so
    // keep it off the async worker like `store_passages` and the flush.
    match tokio::task::block_in_place(|| state.add_associations(&name, associations, deadline)) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(Ok(applied)) => {
            // An empty batch reaches here as `applied == 0`: nothing
            // was written, so the counter must not move either — the
            // same rule the partial-write arm below already applies.
            if applied > 0 {
                state.note_write(&name);
            }
            ok(applied, started_at)
        }
        // Items before the failing one are applied (each item is
        // all-or-nothing in the library); report how far the batch got.
        // Association writes only fail on capacity today, but the
        // shared mapping must not assume that.
        Ok(Err(partial)) => {
            partial_write_error(&state, &name, partial, started_at, |applied, message| {
                format!("applied {applied} of {total} associations, then: {message}")
            })
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RetractAssociationRequest {
    pub subject: String,
    pub label: String,
    pub object: String,
}

/// What one association retraction accomplished. `retracted: false`
/// means the triple named no live edge — unknown names, no such edge,
/// or one already fully retracted — and nothing was changed, the same
/// found-nothing honesty `retract_source` answers with.
#[derive(Serialize)]
pub struct RetractAssociationOutcome {
    pub retracted: bool,
    /// How many per-source attribution records were unlinked with the
    /// edge (0 for an edge carrying only unsourced weight).
    pub attributions_removed: usize,
}

/// Withdraws one `(subject, label, object)` association outright —
/// the surgical correction for a fact that should never have been
/// asserted, where `retract_source` would discard the whole
/// document's contribution. A fact that is merely CONTESTED wants a
/// negative-weight assertion instead, which preserves the dispute.
pub async fn retract_association(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<RetractAssociationRequest>,
) -> Response {
    let started_at = Instant::now();
    for (field, value) in [
        ("subject", &request.subject),
        ("label", &request.label),
        ("object", &request.object),
    ] {
        if let Some(refusal) = empty(field, value, started_at) {
            return refusal;
        }
        if let Some(refusal) = oversized(field, value, MAX_NAME_BYTES, started_at) {
            return refusal;
        }
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // The write stages a WAL op and fsyncs before returning; keep it
    // off the async worker like every other write path.
    let outcome = tokio::task::block_in_place(|| {
        state.retract_association(&name, &request.subject, &request.label, &request.object)
    });
    match outcome {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(unlinked) => {
            // The retracted TRIPLE lives in the body, so the access log
            // alone cannot say what was withdrawn — the audit line can
            // (destructive, like retract_source and the deletes).
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                subject = %request.subject,
                label = %request.label,
                object = %request.object,
                retracted = unlinked.is_some(),
                attributions_removed = unlinked.unwrap_or(0),
                "association retracted",
            );
            // A retraction that found nothing changed nothing; only an
            // effective one counts as a write.
            if unlinked.is_some() {
                state.note_write(&name);
            }
            ok(
                RetractAssociationOutcome {
                    retracted: unlinked.is_some(),
                    attributions_removed: unlinked.unwrap_or(0),
                },
                started_at,
            )
        }
    }
}
