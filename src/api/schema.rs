//! `GET`/`PUT /contexts/{name}/schema` (#380, S2 of #218's ADR 0009
//! split) — the management routes over the schema document S1 (#379)
//! already knows how to validate and persist. Nothing here enforces
//! anything against the graph; that is `schema_issues`' job (S3, #381).

use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use taguru::deadline::Deadline;

use crate::metrics::ErrorKind;
use crate::registry::{AppState, PutSchemaError};
use crate::schema::{self, SchemaDocument};

use super::{AppJson, AppPath, ErrorCode, deadline_exceeded, error, key_name, not_found, ok};

/// One directory row by name's schema-document twin — the resident
/// document as `install`ed, or a 404 distinguishing "this context has
/// never installed a schema" ([`ErrorCode::NoSchema`]) from "this
/// context does not exist" ([`ErrorCode::NoContext`]), the boundary ADR
/// 0009 §6.3 requires stay load-bearing rather than collapsing both
/// into one shape.
pub async fn get_schema(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // `schema_of` can load the full context (a cold entry whose schema
    // is recorded but not yet resolved locally) — real disk IO, so it
    // steps off the async worker like every other mutating or
    // load-bearing handler.
    match tokio::task::block_in_place(|| state.schema_of(&name)) {
        None => not_found(&name, started_at),
        Some(Ok(Some(installed))) => ok(installed.document(), started_at),
        Some(Ok(None)) => error(
            ErrorCode::NoSchema,
            format!("context '{name}' has no schema document"),
            started_at,
        ),
        Some(Err(message)) => {
            // The detail (which can name the data directory's own
            // filesystem path — `schema::load_schema`'s digest-mismatch
            // and unreadable-file messages both do) is for the operator,
            // not an authenticated HTTP client: logged here, never
            // forwarded into the response.
            tracing::warn!(context = %name, error = %message, "schema load failed");
            state.metrics().record_error(ErrorKind::Load);
            error(
                ErrorCode::Internal,
                format!("context '{name}' schema could not be loaded — see server logs"),
                started_at,
            )
        }
    }
}

/// Installs `body` as `name`'s schema document wholesale — there is no
/// delta form, so a retry after any failure below is always safe
/// regardless of which side of it the previous attempt reached (ADR
/// 0009 §5.2). `body` is `#[serde(deny_unknown_fields)]` on
/// [`SchemaDocument`] itself, so an unread field or an unknown `mode`
/// spelling is a 400 from [`AppJson`] before this handler even runs;
/// [`schema::install`] catches everything shape-valid-but-refused
/// (version, caps, `is_a` cycles/depth, the reserved relation label).
pub async fn put_schema(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(document): AppJson<SchemaDocument>,
) -> Response {
    let started_at = Instant::now();
    let installed = match schema::install(document) {
        Ok(installed) => installed,
        Err(violation) => {
            return error(
                ErrorCode::InvalidArgument,
                violation.to_string(),
                started_at,
            );
        }
    };
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Writes the sidecar (fsync + rename) then the schema file itself
    // (also fsync + rename), and may load the context first to check
    // its live label-alias table; keep all of that off the async
    // worker like every other mutating endpoint.
    match tokio::task::block_in_place(|| state.put_schema(&name, installed)) {
        None => not_found(&name, started_at),
        Some(Ok(document)) => {
            // Every destructive-ish operator action leaves one
            // self-contained `taguru::audit` line — who, what, to which
            // object — mirroring `delete_context`/`rename_context`.
            // Installing (or replacing) a schema is not destructive to
            // the graph, but it does change what `strict` refuses from
            // this point on, which is exactly the kind of change an
            // incident review wants to find by grepping one target.
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                mode = document.mode.as_str(),
                "context schema installed",
            );
            ok(document, started_at)
        }
        Some(Err(PutSchemaError::ReservedAlias(alias))) => error(
            ErrorCode::InvalidArgument,
            format!(
                "label alias '{alias}' resolves to '{}', the relation label reserved for \
                 type assertions (ADR 0009 §6.3) — rename the alias before installing this \
                 schema",
                schema::SCHEMA_TYPE_LABEL
            ),
            started_at,
        ),
        Some(Err(PutSchemaError::Load(message))) => {
            // Same posture as `get_schema`'s Load arm just above: the
            // detail can name a filesystem path, so it is logged, not
            // returned.
            tracing::warn!(context = %name, error = %message, "schema load failed");
            state.metrics().record_error(ErrorKind::Load);
            error(
                ErrorCode::Internal,
                format!("context '{name}' could not be loaded — see server logs"),
                started_at,
            )
        }
        Some(Err(PutSchemaError::Io(io_error))) => {
            tracing::warn!(context = %name, error = %io_error, "schema write failed");
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
                format!("context '{name}' schema not persisted — see server logs"),
                started_at,
            )
        }
    }
}
