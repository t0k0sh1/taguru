//! The Prometheus metrics text and the usage counters that feed it.

use serde_json::json;

use crate::support::*;

#[test]
fn metrics_expose_prometheus_text_reflecting_traffic() {
    let server = Server::start("metrics");

    // Two health probes, then two recalls against DIFFERENT context
    // names on the same route template (both 404 — routing happened,
    // which is all the label needs).
    server.call("GET", "/health", None);
    server.call("GET", "/health", None);
    server.call("POST", "/contexts/nope1/recall", Some(json!({"cue": "x"})));
    server.call("POST", "/contexts/nope2/recall", Some(json!({"cue": "x"})));
    // And one path that matches no route at all.
    server.call("GET", "/definitely/not/a/route", None);

    let (status, body) = server.call("GET", "/metrics", None);
    assert_eq!(status, 200);
    let text = body.as_str().expect("metrics body is text, not JSON");

    // Counted traffic, keyed by route template.
    assert!(
        text.contains(
            "taguru_http_requests_total{method=\"GET\",route=\"/health\",status=\"200\"} 2"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "taguru_http_requests_total{method=\"POST\",route=\"/contexts/{name}/recall\",status=\"404\"} 2"
        ),
        "two context names must fold into ONE templated series: {text}"
    );
    // The raw paths never become label values; unmatched requests all
    // share one bucket.
    assert!(!text.contains("nope1"), "raw path leaked into labels");
    assert!(!text.contains("/definitely/not/a/route"));
    assert!(text.contains("route=\"<unmatched>\""));

    // Histogram, domain counters, and gauges are all present.
    assert!(text.contains("taguru_http_request_duration_seconds_bucket"));
    assert!(text.contains("taguru_flush_total{outcome=\"ok\"}"));
    assert!(text.contains("taguru_contexts_registered 0"));
}

#[test]
fn search_outcomes_and_resolve_tiers_land_in_the_metrics_text() {
    let server = Server::start("searchmetrics");
    server.ok("PUT", "/contexts/sm", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sm/associations",
        Some(json!([{
            "subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
            "weight": 1.0, "source": "p1"
        }])),
    );

    // One hit and one empty recall; one confident resolve and one miss
    // (no embedding provider in the harness, so nothing rescues it).
    server.ok(
        "POST",
        "/contexts/sm/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    server.ok("POST", "/contexts/sm/recall", Some(json!({"cue": "qqqq"})));
    server.ok(
        "POST",
        "/contexts/sm/resolve",
        Some(json!({"cue": "青嶺酒造"})),
    );
    server.ok("POST", "/contexts/sm/resolve", Some(json!({"cue": "qqqq"})));

    let (status, body) = server.call("GET", "/metrics", None);
    assert_eq!(status, 200);
    let text = body.as_str().expect("metrics body is text, not JSON");

    assert!(
        text.contains("taguru_searches_total{op=\"recall\",outcome=\"hit\"} 1"),
        "{text}"
    );
    assert!(
        text.contains("taguru_searches_total{op=\"recall\",outcome=\"empty\"} 1"),
        "{text}"
    );
    assert!(
        text.contains("taguru_searches_total{op=\"resolve\",outcome=\"hit\"} 1"),
        "{text}"
    );
    assert!(
        text.contains("taguru_resolves_total{tier=\"lexical\"} 1"),
        "{text}"
    );
    assert!(
        text.contains("taguru_resolves_total{tier=\"miss\"} 1"),
        "{text}"
    );
}

#[test]
fn usage_counters_track_reads_writes_and_empties_per_context() {
    let server = Server::start("usage");
    server.ok("PUT", "/contexts/used", Some(json!({"description": "d"})));
    server.ok("PUT", "/contexts/idle", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/used/associations",
        Some(json!([{
            "subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
            "weight": 1.0, "source": "p1"
        }])),
    );
    server.ok(
        "POST",
        "/contexts/used/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    server.ok(
        "POST",
        "/contexts/used/recall",
        Some(json!({"cue": "qqqq"})),
    );
    server.ok(
        "POST",
        "/contexts/used/query",
        Some(json!({"subject": "青嶺酒造"})),
    );
    // The registry groups unreachable_from with the association reads
    // above; the usage counters must agree. Zero orphans is the audit
    // succeeding, so it counts as a read but never as an empty one.
    server.ok(
        "POST",
        "/contexts/used/unreachable_from",
        Some(json!({"origins": ["青嶺酒造"]})),
    );

    let used = server.ok("GET", "/contexts/used", None);
    assert_eq!(used["usage"]["writes"], json!(1), "{used}");
    assert_eq!(used["usage"]["reads"], json!(4), "{used}");
    assert_eq!(used["usage"]["empty_reads"], json!(1), "{used}");
    assert!(used["usage"]["last_read_epoch"].as_u64().unwrap() > 0);
    assert!(used["usage"]["last_write_epoch"].as_u64().unwrap() > 0);

    // The untouched context shows exactly that — the "never chosen"
    // signal the directory exists to expose.
    let idle = server.ok("GET", "/contexts/idle", None);
    assert_eq!(idle["usage"]["reads"], json!(0), "{idle}");
    assert_eq!(idle["usage"]["writes"], json!(0), "{idle}");
    assert_eq!(idle["usage"]["last_read_epoch"], json!(0), "{idle}");
}

/// An empty associations or aliases batch applies nothing (`applied ==
/// 0`), so it must not bump the write counter — the same rule the
/// partial-write arm already applies via `partial.applied > 0`.
#[test]
fn empty_association_and_alias_batches_do_not_bump_the_write_counter() {
    let server = Server::start("empty-batch-writes");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let applied = server.ok("POST", "/contexts/sake/associations", Some(json!([])));
    assert_eq!(applied, json!(0));

    let applied = server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {}, "labels": {}})),
    );
    assert_eq!(applied, json!(0));

    let entry = server.ok("GET", "/contexts/sake", None);
    assert_eq!(
        entry["usage"]["writes"],
        json!(0),
        "empty batches must not count as writes: {entry}"
    );

    // A non-empty batch still counts — proving the counter isn't just
    // stuck at zero regardless of what reaches it.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{
            "subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
            "weight": 1.0, "source": "p1"
        }])),
    );
    let entry = server.ok("GET", "/contexts/sake", None);
    assert_eq!(entry["usage"]["writes"], json!(1), "{entry}");
}

#[test]
fn usage_counters_survive_a_graceful_restart_even_for_read_only_sessions() {
    let server = Server::start("usagerestart");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{
            "subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
            "weight": 1.0, "source": "p1"
        }])),
    );
    server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    let data_dir = server.stop_gracefully();

    // Second boot performs READS ONLY: nothing dirties the graph, so
    // no image flush ever writes the sidecar — the shutdown sweep is
    // the only thing standing between these counters and oblivion.
    let server = Server::start_on("usagerestart", data_dir);
    server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    let data_dir = server.stop_gracefully();

    let server = Server::start_on("usagerestart", data_dir);
    let entry = server.ok("GET", "/contexts/sake", None);
    assert_eq!(entry["usage"]["reads"], json!(2), "{entry}");
    assert_eq!(entry["usage"]["writes"], json!(1), "{entry}");
}
