//! Router-owned endpoints: `/health`, `/metrics`, and the `/protocol`
//! proxy, plus the `urlencode` helper the group verbs share.

use super::*;

pub(super) async fn health(State(state): State<RouterState>) -> Response {
    // The router itself has no degraded write path to report; shard
    // health belongs to the shards' own probes (and surfaces here per
    // request as pass-through errors and `unreached` labels). `version`
    // names the router's own build (ADR 0002 §10) — same binary as
    // `serve`, so the same field means the same thing here.
    axum::Json(json!({
        "status": "ok",
        "router": true,
        "shards": state.map().shards.len(),
        "version": env!("CARGO_PKG_VERSION"),
    }))
    .into_response()
}

pub(super) async fn render_metrics(State(state): State<RouterState>) -> Response {
    let mut out = String::new();
    out.push_str("# TYPE taguru_router gauge\ntaguru_router 1\n");
    out.push_str(&format!(
        "# TYPE taguru_router_shards gauge\ntaguru_router_shards {}\n",
        state.map().shards.len()
    ));
    out.push_str("# TYPE taguru_router_requests_total counter\n");
    for ((route, status), count) in state.inner.metrics.http.lock().iter() {
        out.push_str(&format!(
            "taguru_router_requests_total{{route=\"{route}\",status=\"{status}\"}} {count}\n"
        ));
    }
    out.push_str("# TYPE taguru_router_shard_requests_total counter\n");
    for ((shard, outcome), count) in state.inner.metrics.shard.lock().iter() {
        out.push_str(&format!(
            "taguru_router_shard_requests_total{{shard=\"{}\",outcome=\"{outcome}\"}} {count}\n",
            state.map().url(*shard)
        ));
    }
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        out,
    )
        .into_response()
}

/// `GET /protocol`, proxied from the first shard that answers — the
/// manual is deploy-wide and shards are expected to be homogeneous
/// (same version, same embedding configuration); the router itself
/// has no embedding tier to describe.
pub(super) async fn proxy_protocol(
    State(state): State<RouterState>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    let mut unreached = Vec::new();
    for shard in state.map().all() {
        match state
            .call_shard(
                shard,
                Method::GET,
                "/protocol",
                &HeaderMap::new(),
                None,
                deadline,
            )
            .await
        {
            Ok(answer) if answer.status.is_success() => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "text/markdown; charset=utf-8")],
                    answer.body,
                )
                    .into_response();
            }
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => unreached.push(Unreached {
                shard: state.map().url(shard).to_string(),
                contexts: Vec::new(),
                error,
            }),
        }
    }
    unreachable_refusal(&unreached, started_at)
}

/// Percent-encodes one path segment the way the stdio bridge does:
/// enough for a context/group name to survive the round trip.
pub(super) fn urlencode(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char)
            }
            other => encoded.push_str(&format!("%{other:02X}")),
        }
    }
    encoded
}
