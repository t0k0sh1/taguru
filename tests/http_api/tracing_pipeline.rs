//! ADR 0008: the composed retrieval pipeline's span tree, over a real
//! OTLP wire (`FakeCollector`) — root/phase shape, skip events, cache
//! hits as childless spans, and the two privacy/error-status
//! regressions the ADR itself found and fixed (§2.5).

use serde_json::json;

use crate::support::*;

fn seed_sake_context(server: &Server) {
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の記憶"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "docs/kura.md": "青嶺酒造は雲居県霧沢町の蔵元である。杜氏は高瀬である。"
        }})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0,
             "source": "docs/kura.md", "paragraph": 0},
        ])),
    );
}

#[test]
fn a_composed_retrieve_exports_one_root_with_a_phase_span_per_step() {
    let collector = FakeCollector::start();
    let server = Server::start_with_env(
        "tracing-retrieve-tree",
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );
    seed_sake_context(&server);

    let traceparent = "00-11112222333344445555666677778888-1111222233334444-01";
    let mut request = ureq::http::Request::builder()
        .method("POST")
        .uri(format!("{}/mcp", server.base))
        .header("traceparent", traceparent);
    request = request.header("Content-Type", "application/json");
    let body = json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "retrieve", "arguments": {
            "context": "sake", "origins": ["青嶺酒造"],
            "text_fallback_query": "杜氏は誰か", "text_fallback_only_if_empty": false
        }}
    });
    let response = test_agent()
        .run(request.body(body.to_string()).unwrap())
        .expect("retrieve tool call must answer");
    assert_eq!(response.status(), 200);

    let _ = server.stop_gracefully();
    let tree = SpanTree::new(collector.spans());

    let request_span = tree.one("POST /mcp");
    assert_eq!(request_span["traceId"], "11112222333344445555666677778888");

    let root = tree.one("taguru.retrieve");
    assert_eq!(root["traceId"], request_span["traceId"], "{root:?}");
    assert_eq!(
        root["parentSpanId"], request_span["spanId"],
        "taguru.retrieve must be the request span's child"
    );
    assert_eq!(
        attribute(root, "taguru.transport").map(|v| v["stringValue"].clone()),
        Some(json!("remote_mcp"))
    );

    // One phase span per step that actually ran: resolve (always),
    // describe (default on), activate (anchors resolved), citations
    // (default on), passage_fallback (forced via
    // text_fallback_only_if_empty: false). query is absent — no
    // labels were given.
    for phase in [
        "taguru.resolve",
        "taguru.describe",
        "taguru.activate",
        "taguru.citations",
        "taguru.passage_fallback",
    ] {
        let span = tree.one(phase);
        assert_eq!(
            span["traceId"], request_span["traceId"],
            "{phase} must share the request's trace"
        );
        assert_eq!(
            span["parentSpanId"], root["spanId"],
            "{phase} must be taguru.retrieve's child"
        );
    }
    assert!(
        tree.by_name("taguru.query").is_empty(),
        "no labels were given; taguru.query must not exist"
    );
}

#[test]
fn skipped_steps_are_recorded_as_events_with_stable_reason_codes() {
    let collector = FakeCollector::start();
    let server = Server::start_with_env(
        "tracing-retrieve-skips",
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );
    seed_sake_context(&server);

    let result = server.call_tool(
        1,
        "retrieve",
        json!({
            "context": "sake", "origins": ["青嶺酒造"],
            "describe_first": false, "fetch_citations": false
        }),
    );
    assert!(result.get("isError").is_none(), "{result}");

    let _ = server.stop_gracefully();
    let tree = SpanTree::new(collector.spans());

    let root = tree.one("taguru.retrieve");
    let skip_events: Vec<&str> = root["events"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|event| event["name"] == "taguru.skip")
        .filter_map(event_reason)
        .collect();
    for expected in ["describe_disabled", "labels_absent", "citations_disabled"] {
        assert!(
            skip_events.contains(&expected),
            "expected {expected:?} among {skip_events:?}"
        );
    }
    // No text_fallback_query at all — a fourth, distinct skip reason.
    assert!(
        skip_events.contains(&"fallback_not_requested"),
        "{skip_events:?}"
    );

    for absent in [
        "taguru.describe",
        "taguru.query",
        "taguru.citations",
        "taguru.passage_fallback",
    ] {
        assert!(
            tree.by_name(absent).is_empty(),
            "{absent} must not exist when its step was skipped"
        );
    }
}

#[test]
fn a_cache_hit_answers_with_one_span_and_no_lane_children() {
    let collector = FakeCollector::start();
    let server = Server::start_with_env(
        "tracing-passage-cache-hit",
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );
    seed_sake_context(&server);

    let search = |server: &Server| {
        server.ok(
            "POST",
            "/contexts/sake/sources/search",
            Some(json!({"query": "杜氏"})),
        )
    };
    let first = search(&server);
    assert!(!first["hits"].as_array().unwrap().is_empty(), "{first}");
    let second = search(&server);
    assert_eq!(first, second);

    let _ = server.stop_gracefully();
    let tree = SpanTree::new(collector.spans());

    let searches = tree.by_name("taguru.passage_search");
    assert_eq!(searches.len(), 2, "one span per call: {searches:?}");
    // Order isn't guaranteed by export batching alone, so identify the
    // hit by its own attribute rather than position.
    let hit = searches
        .iter()
        .find(|span| attribute(span, "taguru.cache.result") == Some(&json!({"stringValue": "hit"})))
        .unwrap_or_else(|| panic!("no cache-hit span among {searches:?}"));
    let miss = searches
        .iter()
        .find(|span| {
            attribute(span, "taguru.cache.result") == Some(&json!({"stringValue": "miss"}))
        })
        .unwrap_or_else(|| panic!("no cache-miss span among {searches:?}"));
    // Proves the childless assertion below actually distinguishes hit
    // from miss, rather than both trivially having no children.
    assert!(
        !tree.children(miss).is_empty(),
        "a cache miss must run lane children: {:?}",
        tree.children(miss)
    );
    assert!(
        tree.children(hit).is_empty(),
        "a cache hit must answer with no lane children: {:?}",
        tree.children(hit)
    );
}

#[test]
fn a_provider_degrade_leaves_the_root_and_lane_span_unset_not_error() {
    let collector = FakeCollector::start();
    let server = Server::start_with_env(
        "tracing-embed-degrade",
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
            // Unreachable — every query embedding call fails, forcing
            // the semantic lane to degrade while BM25 still answers.
            ("TAGURU_EMBED_URL", "http://127.0.0.1:9/v1/embeddings"),
            ("TAGURU_EMBED_MODEL", "unreachable-model"),
            ("TAGURU_EMBED_PASSAGES", "1"),
        ],
    );
    seed_sake_context(&server);

    let found = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "杜氏"})),
    );
    assert!(!found["hits"].as_array().unwrap().is_empty(), "{found}");

    let _ = server.stop_gracefully();
    let tree = SpanTree::new(collector.spans());

    // Regression test for ADR 0008 §2.5(a): the degrade must not
    // color the request span (or any ancestor) ERROR — asserted as
    // "UNSET" (code 0) via the span's real OTLP `status`, not the
    // `otel.status_code` tracing FIELD, which `tracing-opentelemetry`
    // consumes entirely into that status rather than leaving as an
    // attribute (so `attribute(span, "otel.status_code")` is always
    // `None`, regardless of the actual status).
    let request_span = tree.one("POST /contexts/{name}/sources/search");
    assert_eq!(status_code(request_span), 0, "{request_span:?}");
    let passage_search = tree.one("taguru.passage_search");
    assert_eq!(status_code(passage_search), 0, "{passage_search:?}");

    let degrade_reasons: Vec<&str> = passage_search["events"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(
            tree.children(passage_search)
                .into_iter()
                .flat_map(|child| child["events"].as_array().into_iter().flatten()),
        )
        .filter(|event| event["name"] == "taguru.degrade")
        .filter_map(event_reason)
        .collect();
    assert!(
        degrade_reasons.contains(&"vector_query_embedding_failed"),
        "{degrade_reasons:?}"
    );
}

#[test]
fn no_question_concept_source_or_passage_text_reaches_the_collector() {
    let collector = FakeCollector::start();
    let server = Server::start_with_env(
        "tracing-privacy-sentinel",
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
            ("TAGURU_LOG_SEARCHES", "1"),
        ],
    );
    // Every nonce below is unique and would only appear on the wire if
    // some span/event carried raw content instead of counts/reasons.
    const CONCEPT_NONCE: &str = "SENTINEL-CONCEPT-3f9a";
    const SOURCE_NONCE: &str = "sentinel-source-7b2c.md";
    const PASSAGE_NONCE: &str = "SENTINEL-PASSAGE-TEXT-91de";
    const QUERY_NONCE: &str = "SENTINEL-QUERY-c4a1";

    server.ok(
        "PUT",
        "/contexts/sentinel",
        Some(json!({"description": "d"})),
    );
    server.ok(
        "POST",
        "/contexts/sentinel/sources",
        Some(json!({"passages": {SOURCE_NONCE: format!("{PASSAGE_NONCE} {CONCEPT_NONCE}")}})),
    );
    server.ok(
        "POST",
        "/contexts/sentinel/associations",
        Some(json!([{
            "subject": CONCEPT_NONCE, "label": "sentinel-label", "object": "sentinel-object",
            "weight": 1.0, "source": SOURCE_NONCE, "paragraph": 0
        }])),
    );
    server.ok(
        "POST",
        "/contexts/sentinel/sources/search",
        Some(json!({"query": QUERY_NONCE})),
    );
    server.call_tool(
        1,
        "retrieve",
        json!({"context": "sentinel", "origins": [CONCEPT_NONCE]}),
    );

    let _ = server.stop_gracefully();
    let bodies = collector.raw_bodies();
    // Without this, a collector that received nothing at all (export
    // failed, or never flushed) would pass the loop below vacuously —
    // proving nothing about the one thing this test exists to check.
    assert!(!bodies.is_empty(), "no OTLP payload was collected at all");
    for nonce in [CONCEPT_NONCE, SOURCE_NONCE, PASSAGE_NONCE, QUERY_NONCE] {
        assert!(
            bodies.iter().all(|body| !body.contains(nonce)),
            "sentinel {nonce:?} leaked into an OTLP payload"
        );
    }
}
