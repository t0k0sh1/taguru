//! ADR 0008 §6, §10: the router's own request span, and
//! `taguru.shard_call` as the fan-out's child span — not a header
//! pass-through, injected from the router's own current span.

use serde_json::json;

use crate::support::*;

#[test]
fn a_router_request_span_parents_each_shard_call_and_shard_request_span() {
    let collector = FakeCollector::start();
    let otel_env = [
        ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
        ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
        ("OTEL_BSP_SCHEDULE_DELAY", "100"),
    ];
    // Real shards, not FakeShard — so the trace continues into each
    // shard's own request span, proving W3C context (not just a
    // forwarded header) actually reached it.
    let shard_a = Server::start_with_env("tracing-router-shard-a", &otel_env);
    let shard_b = Server::start_with_env("tracing-router-shard-b", &otel_env);
    let router = Server::start_router(
        "tracing-router-fanout",
        &format!("a = {}\nb = {}\n", shard_a.base, shard_b.base),
        &otel_env,
    );

    let (status, body) = router.call("GET", "/contexts", None);
    assert_eq!(status, 200, "{body}");

    let _ = router.stop_gracefully();
    let _ = shard_a.stop_gracefully();
    let _ = shard_b.stop_gracefully();
    let tree = SpanTree::new(collector.spans());

    // Three processes all name their request span "GET /contexts", so
    // `one` can't disambiguate — identify the router's OWN span as the
    // one with no parent (the test sent no inbound `traceparent`, so
    // only the trace ROOT has an empty parentSpanId; both shards'
    // spans are parented under their own taguru.shard_call).
    let roots: Vec<&serde_json::Value> = tree
        .by_name("GET /contexts")
        .into_iter()
        .filter(|span| {
            span["parentSpanId"].as_str().unwrap_or_default().is_empty()
                || span["parentSpanId"].is_null()
        })
        .collect();
    let router_request = match roots.as_slice() {
        [span] => *span,
        _ => panic!("expected exactly one root GET /contexts span, found {roots:?}"),
    };
    let shard_calls = tree.children(router_request);
    assert_eq!(
        shard_calls.len(),
        2,
        "one taguru.shard_call per shard: {shard_calls:?}"
    );
    for shard_call in &shard_calls {
        // `otel.name` overrides the exported name to `"{method} -> shard
        // {index}"` (same "otel.name wins" pattern the HTTP request span
        // uses) — the span's underlying macro name, `taguru.shard_call`,
        // never reaches the wire, so assert the override instead.
        let name = shard_call["name"].as_str().unwrap_or_default();
        assert!(
            name.starts_with("GET -> shard "),
            "unexpected shard_call name: {name:?}"
        );
        // Each shard's OWN request span must be parented under the
        // taguru.shard_call that dispatched it — not under the
        // router's request span directly, and not left as a fresh
        // trace (the fan-out collapsing to one flat level is exactly
        // what a bare header pass-through, instead of injecting from
        // the router's own span, would produce).
        let downstream: Vec<&serde_json::Value> = tree
            .children(shard_call)
            .into_iter()
            .filter(|span| span["name"] == "GET /contexts")
            .collect();
        assert_eq!(
            downstream.len(),
            1,
            "shard_call {:?} must parent exactly one downstream request span: {shard_calls:?}",
            shard_call["spanId"]
        );
        assert_eq!(downstream[0]["traceId"], router_request["traceId"]);
    }
}

#[test]
fn the_router_injects_its_own_span_and_forwards_tracestate() {
    let collector = FakeCollector::start();
    let shard = FakeShard::start(json!({"result": {"total": 0, "contexts": []}}));
    let router = Server::start_router(
        "tracing-router-header-inject",
        &format!("a = {}\n", shard.endpoint),
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );

    let request = ureq::http::Request::builder()
        .method("GET")
        .uri(format!("{}/contexts", router.base))
        .header(
            "traceparent",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        )
        .header("tracestate", "vendor=abc123")
        .body(())
        .unwrap();
    let response = test_agent().run(request).expect("router must answer");
    assert_eq!(response.status(), 200);

    let _ = router.stop_gracefully();
    let received = shard.requests();
    assert_eq!(received.len(), 1, "{received:?}");
    let headers = &received[0];

    // tracestate rides through untouched.
    assert_eq!(
        headers.get("tracestate").map(String::as_str),
        Some("vendor=abc123")
    );

    // traceparent is NOT the caller's own — the router must have
    // regenerated it from its own current span (a bare pass-through
    // would keep the inbound trace-id but never mint a new parent
    // span-id matching a `taguru.shard_call` span; here we just prove
    // it changed from the caller's literal span-id, which only
    // `inject_current` explains).
    let forwarded = headers
        .get("traceparent")
        .unwrap_or_else(|| panic!("no traceparent forwarded: {headers:?}"));
    assert!(
        forwarded.starts_with("00-0af7651916cd43dd8448eb211c80319c-"),
        "trace id must be preserved: {forwarded}"
    );
    assert!(
        !forwarded.contains("b7ad6b7169203331"),
        "span id must be the router's OWN span, not the caller's: {forwarded}"
    );
}
