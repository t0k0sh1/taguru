use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use taguru::deadline::Deadline;

use crate::registry::{AppState, AssocOp};

use super::{
    AppJson, AppPath, ErrorCode, Issue, MAX_ASSOCIATION_WEIGHT, MAX_ASSOCIATIONS_PER_REQUEST,
    MAX_NAME_BYTES, RefusalDetail, access_error, collected_validation_message, deadline_exceeded,
    describe_value, empty, error, key_name, ok, oversized, partial_write_error, truncate_issues,
    validation_error,
};

/// Reads one `associations[index]` item's required, non-empty,
/// within-cap string field — the raw-JSON twin of what a
/// `Deserialize<String>` plus the old post-hoc `oversized`/`empty`
/// checks did together. Working from `Value` (issue #182) instead of a
/// typed struct means a wrong-typed field is diagnosed alongside every
/// other item's issues in one pass, rather than rejecting the whole
/// batch at the JSON-extractor layer before this handler ever runs.
fn interpret_required_string(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
    issues: &mut Vec<Issue>,
) -> String {
    let full_path = format!("{path}.{key}");
    match obj.get(key) {
        None | Some(Value::Null) => {
            issues.push(Issue::missing(full_path, "a non-empty string"));
            String::new()
        }
        Some(Value::String(text)) if text.is_empty() => {
            issues.push(Issue::empty(full_path));
            String::new()
        }
        Some(Value::String(text)) if text.len() > MAX_NAME_BYTES => {
            issues.push(Issue::too_long(full_path, MAX_NAME_BYTES, text.len()));
            String::new()
        }
        Some(Value::String(text)) => text.clone(),
        Some(other) => {
            issues.push(Issue::wrong_type(full_path, "a non-empty string", other));
            String::new()
        }
    }
}

/// `source`'s shape: omitted (or null) means the ordinary unsourced
/// association — see [`AssocOp::source`] — but a source that IS
/// present is a name like any other and must carry something, or it
/// would intern a real, permanent source id that unrelated callers'
/// mistakes then silently merge into.
fn interpret_source(
    obj: &serde_json::Map<String, Value>,
    path: &str,
    issues: &mut Vec<Issue>,
) -> Option<String> {
    match obj.get("source") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) if text.is_empty() => {
            issues.push(Issue::empty(format!("{path}.source")));
            None
        }
        Some(Value::String(text)) if text.len() > MAX_NAME_BYTES => {
            issues.push(Issue::too_long(
                format!("{path}.source"),
                MAX_NAME_BYTES,
                text.len(),
            ));
            None
        }
        Some(Value::String(text)) => Some(text.clone()),
        Some(other) => {
            issues.push(Issue::wrong_type(
                format!("{path}.source"),
                "a string",
                other,
            ));
            None
        }
    }
}

/// A weight the graph must never accumulate, refused whole like every
/// other rejected item: JSON cannot carry Infinity or NaN, but it
/// carries 1e300 just fine, and saturation never washes out of an edge.
fn interpret_weight(
    obj: &serde_json::Map<String, Value>,
    path: &str,
    issues: &mut Vec<Issue>,
) -> f64 {
    let full_path = format!("{path}.weight");
    let expected = format!("a finite number with |weight| <= {MAX_ASSOCIATION_WEIGHT}");
    match obj.get("weight") {
        None | Some(Value::Null) => {
            issues.push(Issue::missing(full_path, expected));
            0.0
        }
        Some(Value::Number(number)) => {
            let weight = number.as_f64().unwrap_or(f64::NAN);
            if !weight.is_finite() || weight.abs() > MAX_ASSOCIATION_WEIGHT {
                issues.push(Issue::range(
                    full_path,
                    expected,
                    describe_value(obj.get("weight").unwrap()),
                ));
                0.0
            } else {
                weight
            }
        }
        Some(other) => {
            issues.push(Issue::wrong_type(full_path, expected, other));
            0.0
        }
    }
}

/// Zero-based paragraph position within `source` — meaningless without
/// one, so `apply_op` only ever honors it alongside `source`.
fn interpret_paragraph(
    obj: &serde_json::Map<String, Value>,
    path: &str,
    issues: &mut Vec<Issue>,
) -> Option<u32> {
    match obj.get("paragraph") {
        None | Some(Value::Null) => None,
        Some(value @ Value::Number(number)) => {
            match number.as_u64().and_then(|value| u32::try_from(value).ok()) {
                Some(paragraph) => Some(paragraph),
                None => {
                    issues.push(Issue::wrong_type(
                        format!("{path}.paragraph"),
                        "a non-negative integer paragraph index",
                        value,
                    ));
                    None
                }
            }
        }
        Some(other) => {
            issues.push(Issue::wrong_type(
                format!("{path}.paragraph"),
                "a non-negative integer paragraph index",
                other,
            ));
            None
        }
    }
}

/// Interprets the `associations` request body as a lenient JSON walk
/// (issue #182), collecting every item's issues in one pass instead of
/// rejecting the whole batch at the first bad field — mirroring, for
/// this REST write, the same collect-all discipline ADR 0001 §8 already
/// gives a retrying LLM's answer (`extract.rs::interpret_model_output`).
/// Returns the built ops on a clean pass; the built-so-far ops are
/// discarded (never returned) the moment any issue is found, since the
/// whole batch is refused together — `nothing_written`.
fn interpret_associations(value: &Value) -> Result<Vec<AssocOp>, Vec<Issue>> {
    let Some(array) = value.as_array() else {
        return Err(vec![Issue::wrong_type("associations", "an array", value)]);
    };
    let mut issues = Vec::new();
    let mut ops = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let path = format!("associations[{index}]");
        let Some(obj) = item.as_object() else {
            issues.push(Issue::wrong_type(path, "an object", item));
            continue;
        };
        let subject = interpret_required_string(obj, "subject", &path, &mut issues);
        let label = interpret_required_string(obj, "label", &path, &mut issues);
        let object = interpret_required_string(obj, "object", &path, &mut issues);
        let weight = interpret_weight(obj, &path, &mut issues);
        let source = interpret_source(obj, &path, &mut issues);
        let paragraph = interpret_paragraph(obj, &path, &mut issues);
        ops.push(AssocOp {
            subject,
            label,
            object,
            weight,
            source,
            paragraph,
        });
    }
    if issues.is_empty() {
        Ok(ops)
    } else {
        Err(issues)
    }
}

/// A batch of assertions — the arguments of `associate` /
/// `associate_from` as JSON fields ([`AssocOp`], shared with the WAL).
/// A batch is the natural write unit: one document's extracted facts,
/// one request, one lock acquisition, one WAL fsync.
pub async fn add_associations(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(body): AppJson<Value>,
) -> Response {
    let started_at = Instant::now();
    // Refused before the write lock is even taken, and before the
    // per-item walk below: nothing of an oversized batch is applied,
    // and there is no point diagnosing thousands of items nothing will
    // ever write.
    if let Some(array) = body.as_array()
        && array.len() > MAX_ASSOCIATIONS_PER_REQUEST
    {
        return error(
            ErrorCode::OverLimit,
            format!(
                "batch of {} associations exceeds the per-request limit of \
                 {MAX_ASSOCIATIONS_PER_REQUEST}; split the ingest",
                array.len()
            ),
            started_at,
        );
    }
    let associations = match interpret_associations(&body) {
        Ok(associations) => associations,
        Err(issues) => {
            let (issues, total) = truncate_issues(issues);
            let message = collected_validation_message("the associations batch", &issues, total);
            return validation_error(
                ErrorCode::InvalidArgument,
                message,
                RefusalDetail {
                    issues,
                    integrity: Some("nothing_written"),
                    retryable_after_correction: Some(true),
                    ..Default::default()
                },
                started_at,
            );
        }
    };
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
