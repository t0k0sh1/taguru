//! The axum-facing surface: the RED-metrics/access-log middleware and
//! the `/live`, `/health`, `/version`, `/metrics` handlers. Wired by
//! path from `main.rs` and `route/server.rs`, so every one of these
//! items keeps the exact visibility its `crate::metrics::` callers
//! depend on.

use super::*;

/// Access-log + RED-metrics middleware, one pass per request.
/// `MatchedPath` comes as `Option` deliberately: the required form
/// rejects before the fallback runs and would hijack 404 handling.
/// Unmatched requests all land in one `<unmatched>` series so a path
/// scanner cannot mint unbounded label values — and the method is
/// folded to a fixed allowlist ([`normalized_method`]) for the same
/// reason: an extension-method token is just as attacker-chosen as a
/// path, and this middleware runs ahead of auth.
///
/// With span export configured this is also where the request span is
/// born — parented from the inbound trace context, named per HTTP
/// semconv, its trace id stamped into the access log so a log line
/// finds its trace and vice versa. Without it, the disabled branch
/// leaves the response path and the log shape exactly as before.
pub async fn track_http(
    State(state): State<AppState>,
    matched: Option<MatchedPath>,
    request: Request,
    next: Next,
) -> Response {
    let method = crate::trace::normalized_method(request.method());
    let route = matched
        .as_ref()
        .map(|matched| matched.as_str().to_string())
        .unwrap_or_else(|| "<unmatched>".to_string());
    // The object the request addressed, when the route names one. The
    // route TEMPLATE keeps metric cardinality bounded, but a log line
    // is no series: without the real name here, "which contexts did
    // this key delete" has no answer after the fact. "-" mirrors the
    // key convention below. The names are identifiers, not memory
    // content — the registry's own warnings already print them.
    // Extracted by hand: a route's params can't ride the signature as
    // an extractor the way MatchedPath (which supports optional
    // extraction) does. `path_param` decodes the same way a handler's
    // `AppPath` would, so a percent-encoded name logs decoded here too.
    let (mut parts, body) = request.into_parts();
    let name = crate::api::path_param(&mut parts, "name")
        .await
        .unwrap_or_else(|| "-".to_string());
    // The name lands in the column matching its kind — on the group
    // routes `{name}` is a GROUP — so a log query over `context=`
    // never silently matches group names (the audit lines and the
    // /metrics gauges keep the same split).
    let (context, group) = if route.starts_with("/groups") {
        ("-".to_string(), name)
    } else {
        (name, "-".to_string())
    };
    let request = Request::from_parts(parts, body);
    let started = Instant::now();

    let (response, trace_id) = if crate::trace::enabled() {
        crate::trace::traced_request(method, &route, request, next).await
    } else {
        (next.run(request).await, None)
    };

    let elapsed = started.elapsed();
    let status = response.status().as_u16();
    state.metrics().record_http(method, &route, status, elapsed);
    // Which credential made the request — stamped on the response by
    // the auth layer. "-" = unauthenticated (exempt path, auth off, or
    // a rejection).
    let key = response
        .extensions()
        .get::<crate::auth::AuthKey>()
        .map_or("-", |key| key.0.as_ref())
        .to_string();
    match trace_id {
        Some(trace_id) => tracing::info!(
            method = %method,
            route = %route,
            // `escape_debug`, not a bare Display: `context`/`group`
            // decode straight from the URL path (`path_param`
            // percent-decodes), so a segment carrying an encoded
            // control character (e.g. `%0A`) would otherwise land in
            // the log line raw. A plain `?context` (Debug) would
            // escape it too, but ALSO wraps the whole value in an
            // extra pair of quotes under `TAGURU_LOG_FORMAT=json`,
            // double-encoding what JSON already escapes on its own —
            // `escape_debug` gets the same control-character escaping
            // with no format-dependent quoting either layer has to
            // undo.
            context = %context.escape_debug(),
            group = %group.escape_debug(),
            status,
            key = %key,
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            trace_id = %trace_id,
            "http",
        ),
        None => tracing::info!(
            method = %method,
            route = %route,
            context = %context.escape_debug(),
            group = %group.escape_debug(),
            status,
            key = %key,
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            "http",
        ),
    }
    response
}

/// GET /live: pure liveness — 200 for as long as the process answers
/// at all, deliberately unconditional. A failing flush is a DISK
/// problem (that is `/health`'s signal): restarting the process fixes
/// no disk and costs a full pinned preload, so an orchestrator's
/// liveness probe belongs here, its readiness probe on `/health`.
/// Wiring both probes at `/health` turns every transient disk stall
/// into a restart loop.
pub async fn live() -> &'static str {
    "ok"
}

/// GET /health: 200 `{"status": "ok", "version": "<CARGO_PKG_VERSION>"}`
/// while the write path is healthy, 503 in the ApiError shape when the
/// most recent image flush failed — or the flusher tick that would
/// have flushed it panicked instead of running. The check is the
/// flusher's own outcome, so an orchestrator's probe turns red within
/// one flush interval of the disk going bad — and green again one
/// interval after it recovers. (An idle server with nothing dirty
/// reports its last known state.) The readiness signal: stop routing
/// traffic while the disk is bad, resume when it heals — liveness
/// lives at `/live`. The `version` field lets a remote CLI (ADR 0002
/// §10) detect a minor-version skew from the one request it already
/// sends.
pub async fn health(State(state): State<AppState>) -> Response {
    if state.metrics().maintenance_active() {
        return crate::api::error(
            crate::api::ErrorCode::Maintenance,
            "a maintenance compaction sweep is running — this is an intentional \
             pause, not a fault"
                .to_string(),
            Instant::now(),
        );
    }
    if state.metrics().flush_is_healthy() {
        return axum::Json(serde_json::json!({
            "status": "ok",
            "version": env!("CARGO_PKG_VERSION"),
        }))
        .into_response();
    }
    let reason = if state.metrics().flusher_panicked() {
        "the flusher task panicked on its last tick — this is a bug, not a disk \
         fault; check the server log"
            .to_string()
    } else {
        match state.metrics().last_flush_success_epoch() {
            0 => "the last image flush failed, and none has succeeded since boot — \
                  check disk space and the server log"
                .to_string(),
            epoch => format!(
                "the last image flush failed; the last success was at unix {epoch} — \
                 check disk space and the server log"
            ),
        }
    };
    crate::api::error(crate::api::ErrorCode::Unhealthy, reason, Instant::now())
}

/// GET /version: contract-version discovery (ADR 0005 §6) — the
/// machine-readable answer to "which wire shapes does this server
/// speak," so a caller can detect an incompatible SDK/server pairing
/// before it reaches a decode error. Bare JSON, not the `ApiResponse`
/// envelope, matching `/health`. Unlike `/health`, this always
/// answers 200 even while the write path is degraded — a
/// compatibility check has to run from something that isn't itself
/// affected by the fault it might be diagnosing. No `State` needed:
/// every field is a compile-time constant.
pub async fn version() -> Response {
    axum::Json(crate::api::version_facts()).into_response()
}

/// GET /metrics: the whole registry in Prometheus text format.
pub async fn render(State(state): State<AppState>) -> impl IntoResponse {
    let body = state.metrics().render_prometheus(&state.gauge_snapshot());
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}
