//! Router process state: metrics counters, the shared inner handle, and
//! the per-shard dispatch primitives (`call_shard`, `fan_out`) every
//! other verb builds on.

use super::*;

/// Router-mode counters, rendered at `GET /metrics` as
/// `taguru_router_*` — deliberately router-shaped, not server-shaped:
/// a stateless proxy has no flusher, no WAL, no cache to report on.
#[derive(Default)]
pub(super) struct RouterMetrics {
    pub(super) http: Mutex<BTreeMap<(String, u16), u64>>,
    pub(super) shard: Mutex<BTreeMap<(usize, &'static str), u64>>,
}

impl RouterMetrics {
    pub(super) fn record_http(&self, route: &str, status: u16) {
        *self
            .http
            .lock()
            .entry((route.to_string(), status))
            .or_insert(0) += 1;
    }

    pub(super) fn record_shard(&self, shard: usize, outcome: &'static str) {
        *self.shard.lock().entry((shard, outcome)).or_insert(0) += 1;
    }
}

pub(super) struct RouterInner {
    pub(super) map: RouteMap,
    pub(super) client: reqwest::Client,
    pub(super) metrics: RouterMetrics,
    /// The MCP `initialize` manual, fetched once from the first shard
    /// that answers `GET /protocol` — a cache of immutable-per-deploy
    /// text, not state. Until a shard answers, initialize falls back
    /// to the local manual without the shard's configuration trailer.
    pub(super) instructions: OnceLock<Arc<String>>,
}

#[derive(Clone)]
pub(crate) struct RouterState {
    pub(super) inner: Arc<RouterInner>,
}

impl RouterState {
    pub(super) fn map(&self) -> &RouteMap {
        &self.inner.map
    }
}

/// Request headers a NON-proxy (fan-out) call forwards: the caller's
/// identity and trace context; everything else is the router's own
/// call.
fn forward_headers(headers: &HeaderMap) -> HeaderMap {
    let mut forwarded = HeaderMap::new();
    for name in [
        header::AUTHORIZATION,
        header::HeaderName::from_static("traceparent"),
        // Vendor sampling/routing state: a shard that gets
        // `traceparent` without it loses whatever the upstream tracer
        // decided about this trace (ADR 0008 §10).
        header::HeaderName::from_static("tracestate"),
        header::HeaderName::from_static("x-amzn-trace-id"),
    ] {
        if let Some(value) = headers.get(&name) {
            forwarded.insert(name, value.clone());
        }
    }
    // The router's OWN current span — `taguru.shard_call`, entered by
    // the caller before this runs — not the inbound header: a shard
    // must parent under the span that dispatched it, or the fan-out
    // collapses into one flat level and the router's own time vanishes
    // from the trace. Overwrites the two W3C headers copied above; the
    // AWS form is left as the caller sent it, since Taguru never mints
    // that spelling. A no-op with export off, which is what keeps the
    // copy above meaningful in that mode (ADR 0008 §10).
    crate::trace::inject_current(&mut forwarded);
    // `TraceContextPropagator::inject_context` always sets `tracestate`
    // alongside `traceparent`, even when there is none to carry — an
    // empty-but-present header rather than an absent one. Drop it in
    // that case so a shard that got no inbound `tracestate` doesn't
    // receive an empty one just because export happens to be on.
    if forwarded
        .get(header::HeaderName::from_static("tracestate"))
        .is_some_and(|value| value.is_empty())
    {
        forwarded.remove(header::HeaderName::from_static("tracestate"));
    }
    forwarded
}

impl RouterState {
    /// One buffered round trip to a shard — the fan-out building
    /// block. `Err` is TRANSPORT failure only (connect, timeout, torn
    /// body); an HTTP error status is an answer, not an `Err`.
    pub(super) async fn call_shard(
        &self,
        shard: usize,
        method: Method,
        path_and_query: &str,
        headers: &HeaderMap,
        body: Option<Bytes>,
        deadline: Deadline,
    ) -> Result<ShardAnswer, String> {
        // A causal CHILD of one router request with a fully contained
        // lifetime — not a span link, which would lose that
        // containment (ADR 0008 §3, §7). Created before
        // `forward_headers` so `inject_current` picks up this span,
        // not whatever the router's own request span happens to be.
        let span = crate::trace::span!(
            "taguru.shard_call",
            otel.kind = "client",
            otel.name = %format!("{method} -> shard {shard}"),
            taguru.shard.index = shard,
            http.request.method = %method,
            http.response.status_code = tracing::field::Empty,
            taguru.shard.outcome = tracing::field::Empty,
            otel.status_code = tracing::field::Empty,
        );
        let url = format!("{}{}", self.map().url(shard), path_and_query);
        // Header injection has to see `span`, and building the request
        // is synchronous — so it happens inside `in_scope`, while the
        // round trip below rides `.instrument` instead (a thread-local
        // guard cannot survive `fan_out`'s `join_all` interleaving).
        let mut request = span.in_scope(|| {
            self.inner
                .client
                .request(method, url)
                .headers(forward_headers(headers))
        });
        if let Some(limit) = budget(deadline) {
            request = request.timeout(limit);
        }
        if let Some(body) = body {
            request = request
                .header(header::CONTENT_TYPE, "application/json")
                .body(body);
        }
        let outcome = async {
            let response = request.send().await?;
            let status = response.status();
            let body = response.bytes().await?;
            Ok::<ShardAnswer, reqwest::Error>(ShardAnswer { status, body })
        }
        .instrument(span.clone())
        .await;
        match outcome {
            Ok(answer) => {
                let shard_outcome = if answer.status.is_success() {
                    "ok"
                } else {
                    "http_error"
                };
                self.inner.metrics.record_shard(shard, shard_outcome);
                span.record(
                    "http.response.status_code",
                    i64::from(answer.status.as_u16()),
                );
                span.record("taguru.shard.outcome", shard_outcome);
                // An HTTP error status from a shard is an answer, not
                // a client-span failure — only the transport (`Err`)
                // arm below marks this span ERROR.
                Ok(answer)
            }
            Err(error) => {
                self.inner.metrics.record_shard(shard, "unreached");
                span.record("taguru.shard.outcome", "unreached");
                span.record("otel.status_code", "ERROR");
                Err(error.to_string())
            }
        }
    }

    /// [`Self::call_shard`] across a shard set, concurrently; answers
    /// come back labeled by shard index.
    pub(super) async fn fan_out<F>(
        &self,
        shards: &[usize],
        method: Method,
        path_and_query: &str,
        headers: &HeaderMap,
        body_for: F,
        deadline: Deadline,
    ) -> Vec<(usize, Result<ShardAnswer, String>)>
    where
        F: Fn(usize) -> Option<Bytes>,
    {
        let calls = shards.iter().map(|&shard| {
            let method = method.clone();
            let body = body_for(shard);
            async move {
                (
                    shard,
                    self.call_shard(shard, method, path_and_query, headers, body, deadline)
                        .await,
                )
            }
        });
        futures_util::future::join_all(calls).await
    }

    /// The MCP manual: the first shard's `GET /protocol`, cached for
    /// the process lifetime once one answers; the local text (no
    /// configuration trailer) until then.
    pub(super) async fn mcp_instructions(&self, deadline: Deadline) -> Arc<String> {
        if let Some(cached) = self.inner.instructions.get() {
            return Arc::clone(cached);
        }
        for shard in self.map().all() {
            let fetch = self
                .call_shard(
                    shard,
                    Method::GET,
                    "/protocol",
                    &HeaderMap::new(),
                    None,
                    deadline,
                )
                .await;
            if let Ok(answer) = fetch
                && answer.status.is_success()
                && let Ok(text) = std::str::from_utf8(&answer.body)
            {
                let manual = Arc::new(text.to_string());
                let _ = self.inner.instructions.set(Arc::clone(&manual));
                return Arc::clone(self.inner.instructions.get().unwrap_or(&manual));
            }
        }
        Arc::new(api::protocol_text(None))
    }
}
