//! The streaming proxy for `context`-scoped verbs: same method, path,
//! and body streamed straight to the owning shard, headers minus the
//! hop-by-hop set — the response is the shard's own bytes.

use super::*;

/// Strips what must not cross a proxy hop: the RFC 9110 hop-by-hop
/// set, `Host` (reqwest recomputes it for the shard), and the length/
/// framing headers the outbound client re-derives from the body it
/// actually sends.
fn hop_headers(headers: &HeaderMap) -> HeaderMap {
    let mut forwarded = headers.clone();
    for name in [
        header::CONNECTION,
        header::PROXY_AUTHENTICATE,
        header::PROXY_AUTHORIZATION,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
        header::HOST,
        header::CONTENT_LENGTH,
    ] {
        forwarded.remove(name);
    }
    forwarded.remove("keep-alive");
    forwarded
}

/// Whether `path` carries a dot segment — `.` or `..`, raw or with
/// either dot percent-encoded (`%2e`, `%2E`), which the URL parser on
/// the outbound side decodes and resolves alike. The proxy forwards
/// the inbound path verbatim, and the shard-side URL parse applies
/// RFC 3986 dot-segment removal to it: without this check,
/// `POST /contexts/sake/../../import` matches the `{*rest}` wildcard,
/// picks shard-of(`sake`), and reaches that shard as `POST /import` —
/// past the router's own routing of every batch to its owning shard,
/// and onto any endpoint or unmapped context the shard hosts. A dot
/// segment has no legitimate reading here (no context-scoped verb's
/// path contains one), so it is refused whole rather than resolved.
fn has_dot_segment(path: &str) -> bool {
    path.split('/').any(|segment| {
        let decoded = segment.replace("%2E", ".").replace("%2e", ".");
        decoded == "." || decoded == ".."
    })
}

/// The refusal for a dot segment: the single-instance `invalid_argument`
/// shape, naming what was refused but never resolving it.
fn dot_segment_refusal(path: &str, started_at: Instant) -> Response {
    api::error(
        ErrorCode::InvalidArgument,
        format!("path '{path}' carries a dot segment ('.' or '..'); not proxied"),
        started_at,
    )
}

pub(super) async fn proxy_context_root(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    request: Request,
) -> Response {
    proxy_context(state, name, deadline, request).await
}

pub(super) async fn proxy_context_sub(
    State(state): State<RouterState>,
    axum::extract::Path((name, _rest)): axum::extract::Path<(String, String)>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    request: Request,
) -> Response {
    proxy_context(state, name, deadline, request).await
}

/// The transparent hop: same method, same path and query, headers
/// minus the hop-by-hop set, body streamed out and the shard's answer
/// streamed back — the response is the shard's own bytes, error
/// shapes included.
async fn proxy_context(
    state: RouterState,
    name: String,
    deadline: Deadline,
    request: Request,
) -> Response {
    let started_at = Instant::now();
    // Before the map is consulted: a dot segment is refused whatever
    // shard the context name would pick, and the refusal names the
    // path as received, encoded segments included.
    if has_dot_segment(request.uri().path()) {
        return dot_segment_refusal(request.uri().path(), started_at);
    }
    let map = state.map();
    let Some(shard) = map.shard_of(&name) else {
        // No entry and no fallback: for a read this context cannot
        // exist anywhere the router routes — the single-instance
        // not-found, byte for byte. A PUT is asking to CREATE it, and
        // the honest answer is that the map decides where new contexts
        // go, not that something wasn't found.
        return if request.method() == Method::PUT {
            api::error(
                ErrorCode::InvalidArgument,
                format!(
                    "no shard owns context '{name}': add a route-map entry for it, or a \
                     '*' fallback for unmapped contexts (TAGURU_ROUTE_MAP)"
                ),
                started_at,
            )
        } else {
            api::error(
                ErrorCode::NoContext,
                format!("context '{name}' not found"),
                started_at,
            )
        };
    };
    let (parts, body) = request.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|paq| paq.as_str().to_string())
        .unwrap_or_else(|| parts.uri.path().to_string());
    let url = format!("{}{}", map.url(shard), path_and_query);
    // A causal child of the router's request span, same shape as the
    // fan-out's `call_shard` (issue #696): the shard's BYTES stay
    // untouched, but the trace headers are the router's own outbound
    // call — `inject_current_trace` overwrites the W3C pair so the
    // shard's request span parents under this hop instead of skipping
    // it (and, with no inbound `traceparent`, instead of starting a
    // parentless trace of its own). A no-op with export off, which is
    // what keeps the bare pass-through in that mode. Unlike
    // `call_shard`, the response is streamed, not buffered: this span
    // closes once the shard's status and headers are in, so it times
    // the dispatch, not the body transfer.
    let span = crate::trace::span!(
        "taguru.shard_call",
        otel.kind = "client",
        otel.name = %format!("{} -> shard {shard}", parts.method),
        taguru.shard.index = shard as i64,
        http.request.method = %parts.method,
        http.response.status_code = tracing::field::Empty,
        taguru.shard.outcome = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
    );
    // Header injection has to see `span`, and building the request is
    // synchronous — so it happens inside `in_scope`, while the
    // dispatch below rides `.instrument` (same split as `call_shard`).
    let mut outbound = span.in_scope(|| {
        let mut forwarded = hop_headers(&parts.headers);
        inject_current_trace(&mut forwarded);
        state
            .inner
            .client
            .request(parts.method.clone(), url)
            .headers(forwarded)
            .body(reqwest::Body::wrap_stream(body.into_data_stream()))
    });
    if let Some(limit) = budget(deadline) {
        outbound = outbound.timeout(limit);
    }
    match outbound.send().instrument(span.clone()).await {
        Ok(answer) => {
            let shard_outcome = if answer.status().is_success() {
                "ok"
            } else {
                "http_error"
            };
            state
                .inner
                .metrics
                .record_shard(map.url(shard), shard_outcome);
            let status = answer.status();
            span.record("http.response.status_code", i64::from(status.as_u16()));
            // An HTTP error status from the shard is an answer this
            // proxy relays verbatim, not a client-span failure — only
            // the transport arm below marks this span ERROR.
            span.record("taguru.shard.outcome", shard_outcome);
            let headers = hop_headers(answer.headers());
            let mut response = Response::builder().status(status);
            if let Some(response_headers) = response.headers_mut() {
                *response_headers = headers;
            }
            response
                .body(Body::from_stream(answer.bytes_stream()))
                .unwrap_or_else(|error| {
                    api::error(
                        ErrorCode::Internal,
                        format!("could not assemble the proxied response: {error}"),
                        started_at,
                    )
                })
        }
        Err(error) => {
            state
                .inner
                .metrics
                .record_shard(map.url(shard), "unreached");
            span.record("taguru.shard.outcome", "unreached");
            span.record("otel.status_code", "ERROR");
            api::error(
                ErrorCode::ShardUnreachable,
                format!(
                    "shard {} (owning context '{name}') is unreachable: {error}",
                    map.url(shard)
                ),
                started_at,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dot segment, raw or percent-encoded in either case, is one;
    /// a segment that merely contains dots (`.well-known`, `a..b`,
    /// `..foo`) is not.
    #[test]
    fn a_dot_segment_is_a_whole_segment_raw_or_percent_encoded() {
        for path in [
            "/contexts/sake/..",
            "/contexts/sake/../../import",
            "/contexts/../x",
            "/contexts/sake/./recall",
            "/contexts/sake/%2e%2e/flush",
            "/contexts/sake/%2E%2E/flush",
            "/contexts/sake/.%2e/flush",
            "/contexts/%2e/x",
            "/contexts/sake/associations/..",
        ] {
            assert!(has_dot_segment(path), "{path}");
        }
        for path in [
            "/contexts/sake",
            "/contexts/sake/recall",
            "/contexts/.well-known/x",
            "/contexts/a..b/recall",
            "/contexts/..foo/recall",
            "/contexts/foo../recall",
            "/contexts/sake/associations",
            "/contexts/sake//recall",
            "/contexts/%E6%97%A5%E6%9C%AC%E9%85%92/recall",
        ] {
            assert!(!has_dot_segment(path), "{path}");
        }
    }

    #[tokio::test]
    async fn the_refusal_is_the_invalid_argument_shape_naming_the_path() {
        let response = dot_segment_refusal("/contexts/sake/../../import", Instant::now());
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "error");
        assert_eq!(body["code"], "invalid_argument");
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("/contexts/sake/../../import"),
            "{body}"
        );
    }
}
