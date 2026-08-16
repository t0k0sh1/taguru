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

#[test]
fn the_router_omits_tracestate_when_the_caller_sent_none() {
    // Regression test: `TraceContextPropagator::inject_context` always
    // sets `tracestate` alongside `traceparent`, even with nothing to
    // carry — an empty-but-present header rather than an absent one.
    // A shard that got no inbound `tracestate` must not receive an
    // empty one just because export happens to be on.
    let collector = FakeCollector::start();
    let shard = FakeShard::start(json!({"result": {"total": 0, "contexts": []}}));
    let router = Server::start_router(
        "tracing-router-no-tracestate",
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
        // No `tracestate` header on the inbound request.
        .body(())
        .unwrap();
    let response = test_agent().run(request).expect("router must answer");
    assert_eq!(response.status(), 200);

    let _ = router.stop_gracefully();
    let received = shard.requests();
    assert_eq!(received.len(), 1, "{received:?}");
    let headers = &received[0];

    assert!(
        !headers.contains_key("tracestate"),
        "no tracestate should have been forwarded: {headers:?}"
    );
    assert!(
        headers.get("traceparent").is_some(),
        "traceparent must still be injected: {headers:?}"
    );
}

/// issue #696: the transparent proxy route (`/contexts/{name}`) must
/// dispatch under a `taguru.shard_call` span exactly like the fan-out
/// verbs — before the fix it forwarded headers bare, so the shard's
/// request span either skipped the router's hop or (with no inbound
/// `traceparent`) started a parentless trace of its own.
#[test]
fn a_proxied_context_verb_parents_the_shard_request_span_under_a_shard_call() {
    let collector = FakeCollector::start();
    let otel_env = [
        ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
        ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
        ("OTEL_BSP_SCHEDULE_DELAY", "100"),
    ];
    let shard = Server::start_with_env("tracing-proxy-shard", &otel_env);
    let router = Server::start_router(
        "tracing-proxy-router",
        &format!("sake = {}\n", shard.base),
        &otel_env,
    );
    // Seeded on the shard DIRECTLY, so exactly one request — the GET
    // below — crosses the router's proxy hop.
    shard.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let (status, body) = router.call("GET", "/contexts/sake", None);
    assert_eq!(status, 200, "{body}");

    let _ = router.stop_gracefully();
    let _ = shard.stop_gracefully();
    let tree = SpanTree::new(collector.spans());

    // Both processes name their request span "GET /contexts/{name}";
    // the router's is the one with no parent (the test sent no inbound
    // `traceparent`).
    let roots: Vec<&serde_json::Value> = tree
        .by_name("GET /contexts/{name}")
        .into_iter()
        .filter(|span| {
            span["parentSpanId"].as_str().unwrap_or_default().is_empty()
                || span["parentSpanId"].is_null()
        })
        .collect();
    let router_request = match roots.as_slice() {
        [span] => *span,
        _ => panic!("expected exactly one root GET /contexts/{{name}} span, found {roots:?}"),
    };
    let shard_calls = tree.children(router_request);
    let shard_call = match shard_calls.as_slice() {
        [span] => *span,
        _ => panic!("expected exactly one taguru.shard_call child, found {shard_calls:?}"),
    };
    assert_eq!(shard_call["name"], "GET -> shard 0", "{shard_call:?}");
    assert_eq!(
        attribute(shard_call, "taguru.shard.outcome").map(|v| v["stringValue"].clone()),
        Some(json!("ok"))
    );
    assert_eq!(
        attribute(shard_call, "http.response.status_code").map(|v| v["intValue"].clone()),
        Some(json!("200")),
        "{shard_call:?}"
    );
    // The shard's OWN request span continues the same trace under the
    // proxy's shard_call — W3C context actually reached it, not just a
    // pass-through of the (absent) inbound header.
    let downstream: Vec<&serde_json::Value> = tree
        .children(shard_call)
        .into_iter()
        .filter(|span| span["name"] == "GET /contexts/{name}")
        .collect();
    assert_eq!(
        downstream.len(),
        1,
        "the shard's request span must parent under the proxy's shard_call: {downstream:?}"
    );
    assert_eq!(downstream[0]["traceId"], router_request["traceId"]);
}

/// issue #696, header half: the proxied hop overwrites `traceparent`
/// with the router's own span (same trace, new parent span-id) and
/// rides `tracestate` through untouched.
#[test]
fn the_proxy_injects_its_own_span_and_forwards_tracestate() {
    let collector = FakeCollector::start();
    let shard = FakeShard::start(json!({"status": "ok"}));
    let router = Server::start_router(
        "tracing-proxy-header-inject",
        &format!("sake = {}\n", shard.endpoint),
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );

    let request = ureq::http::Request::builder()
        .method("GET")
        .uri(format!("{}/contexts/sake", router.base))
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

    assert_eq!(
        headers.get("tracestate").map(String::as_str),
        Some("vendor=abc123")
    );
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

/// issue #696: the proxy path shares `inject_current_trace`'s
/// empty-`tracestate` cleanup with the fan-out (see the sibling
/// fan-out test above for the propagator behavior that makes the
/// cleanup necessary).
#[test]
fn the_proxy_omits_tracestate_when_the_caller_sent_none() {
    let collector = FakeCollector::start();
    let shard = FakeShard::start(json!({"status": "ok"}));
    let router = Server::start_router(
        "tracing-proxy-no-tracestate",
        &format!("sake = {}\n", shard.endpoint),
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );

    let request = ureq::http::Request::builder()
        .method("GET")
        .uri(format!("{}/contexts/sake", router.base))
        .header(
            "traceparent",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        )
        // No `tracestate` header on the inbound request.
        .body(())
        .unwrap();
    let response = test_agent().run(request).expect("router must answer");
    assert_eq!(response.status(), 200);

    let _ = router.stop_gracefully();
    let received = shard.requests();
    assert_eq!(received.len(), 1, "{received:?}");
    let headers = &received[0];

    assert!(
        !headers.contains_key("tracestate"),
        "no tracestate should have been forwarded: {headers:?}"
    );
    assert!(
        headers.get("traceparent").is_some(),
        "traceparent must still be injected: {headers:?}"
    );
}

/// issue #696, outcome half: a shard that answers an HTTP error is an
/// answer the proxy relays (`http_error`, status UNSET); a shard that
/// cannot be reached marks the proxy's shard_call ERROR — the same
/// split the fan-out's own two tests below pin.
#[test]
fn a_proxied_http_error_and_a_transport_failure_mark_the_shard_call_differently() {
    let collector = FakeCollector::start();
    let shard = FakeShard::start_with_status(500, json!({"status": "error", "error": "boom"}));
    let router = Server::start_router(
        "tracing-proxy-shard-outcomes",
        // `sake` answers 500; `ghost`'s shard refuses the connection
        // synchronously (nothing listens on the discard port).
        &format!("sake = {}\nghost = http://127.0.0.1:9\n", shard.endpoint),
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );

    let (status, _) = router.call("GET", "/contexts/sake", None);
    assert_eq!(status, 500);
    let (status, _) = router.call("GET", "/contexts/ghost", None);
    assert_eq!(status, 502);

    let _ = router.stop_gracefully();
    let tree = SpanTree::new(collector.spans());

    // Two shards, two indices — identify each shard_call by its
    // outcome attribute rather than assuming which map line got which
    // index.
    let calls: Vec<&serde_json::Value> = tree
        .by_name("GET -> shard 0")
        .into_iter()
        .chain(tree.by_name("GET -> shard 1"))
        .collect();
    let outcome = |span: &serde_json::Value| {
        attribute(span, "taguru.shard.outcome").map(|v| v["stringValue"].clone())
    };
    let http_error = calls
        .iter()
        .find(|span| outcome(span) == Some(json!("http_error")))
        .unwrap_or_else(|| panic!("no http_error shard_call among {calls:?}"));
    assert_eq!(status_code(http_error), 0, "{http_error:?}");
    assert_eq!(
        attribute(http_error, "http.response.status_code").map(|v| v["intValue"].clone()),
        Some(json!("500")),
        "{http_error:?}"
    );
    let unreached = calls
        .iter()
        .find(|span| outcome(span) == Some(json!("unreached")))
        .unwrap_or_else(|| panic!("no unreached shard_call among {calls:?}"));
    assert_eq!(status_code(unreached), 2, "{unreached:?}");
}

#[test]
fn a_transport_failure_marks_the_shard_call_span_unreached_and_error() {
    let collector = FakeCollector::start();
    let router = Server::start_router(
        "tracing-router-shard-unreached",
        // Nothing listens on 9 (the discard service) — a synchronous
        // connection refusal, not a hang.
        "a = http://127.0.0.1:9\n",
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );

    let _ = router.call("GET", "/contexts", None);

    let _ = router.stop_gracefully();
    let tree = SpanTree::new(collector.spans());

    let shard_call = tree.one("GET -> shard 0");
    assert_eq!(
        attribute(shard_call, "taguru.shard.outcome").map(|v| v["stringValue"].clone()),
        Some(json!("unreached"))
    );
    assert_eq!(status_code(shard_call), 2, "{shard_call:?}");
}

#[test]
fn a_shard_http_error_marks_the_shard_call_span_but_not_error() {
    let collector = FakeCollector::start();
    let shard = FakeShard::start_with_status(500, json!({"status": "error", "error": "boom"}));
    let router = Server::start_router(
        "tracing-router-shard-http-error",
        &format!("a = {}\n", shard.endpoint),
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );

    let _ = router.call("GET", "/contexts", None);

    let _ = router.stop_gracefully();
    let tree = SpanTree::new(collector.spans());

    let shard_call = tree.one("GET -> shard 0");
    assert_eq!(
        attribute(shard_call, "taguru.shard.outcome").map(|v| v["stringValue"].clone()),
        Some(json!("http_error"))
    );
    // An HTTP error status from a shard is an answer, not a
    // client-span failure — only a transport error marks this span
    // ERROR (see the sibling test above).
    assert_eq!(status_code(shard_call), 0, "{shard_call:?}");
}
