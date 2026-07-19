//! The exact-match retrieval cache end to end: hits skip the search
//! path (visible on /metrics), invalidation follows exactly the
//! revision lanes each surface depends on, scoped keys share entries
//! only when their grants resolve a request identically, and a
//! recreated name never serves its old incarnation's results.

use serde_json::{Value, json};

use crate::support::*;

/// One counter/gauge value scraped off /metrics. `line_start` is the
/// full name-plus-labels prefix, so `taguru_x{a="b"}` never matches
/// `taguru_x_longer{...}`.
fn metric(server: &Server, line_start: &str) -> u64 {
    let (status, body) = server.call("GET", "/metrics", None);
    assert_eq!(status, 200);
    let text = body.as_str().expect("metrics body is text");
    text.lines()
        .find_map(|line| {
            line.strip_prefix(line_start)
                .and_then(|rest| rest.trim().parse::<u64>().ok())
        })
        .unwrap_or_else(|| panic!("metric '{line_start}' not found in scrape"))
}

fn cache_hits(server: &Server, op: &str) -> u64 {
    metric(
        server,
        &format!("taguru_retrieval_cache_total{{op=\"{op}\",outcome=\"hit\"}}"),
    )
}

fn cache_misses(server: &Server, op: &str) -> u64 {
    metric(
        server,
        &format!("taguru_retrieval_cache_total{{op=\"{op}\",outcome=\"miss\"}}"),
    )
}

fn assoc(subject: &str, label: &str, object: &str) -> Value {
    json!([{"subject": subject, "label": label, "object": object,
            "weight": 1.0, "source": "a.md"}])
}

/// The core loop: an identical recall answers from the cache with a
/// byte-identical page, a graph write invalidates exactly that
/// context's entries while a sibling's survive, and every page of a
/// cursor walk caches independently. `taguru_searches_total` keeps
/// counting served retrievals across hits, so dashboards read
/// continuously whichever way a response was computed.
#[test]
fn an_identical_recall_hits_the_cache_and_a_write_invalidates_only_its_context() {
    let server = Server::start("rcache-recall");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(assoc("蔵", "杜氏", "高瀬")),
    );

    let recall = |context: &str| {
        server.ok(
            "POST",
            &format!("/contexts/{context}/recall"),
            Some(json!({"cue": "蔵"})),
        )
    };
    let first = recall("sake");
    let second = recall("sake");
    assert_eq!(first, second, "a hit serves the identical page");
    assert_eq!(cache_misses(&server, "recall"), 1);
    assert_eq!(cache_hits(&server, "recall"), 1);
    assert_eq!(
        metric(
            &server,
            "taguru_searches_total{op=\"recall\",outcome=\"hit\"}"
        ),
        2,
        "the searches family counts served retrievals, hits included"
    );

    // A graph write moves the graph lane: the next identical recall
    // recomputes and sees the new edge.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(assoc("蔵", "創業", "1832")),
    );
    let third = recall("sake");
    assert_eq!(third["total"], 2, "the recomputed page carries the write");
    assert_eq!(cache_misses(&server, "recall"), 2);
    assert_eq!(cache_hits(&server, "recall"), 1);

    // A sibling's entries ride out the other context's writes.
    server.ok("PUT", "/contexts/bunko", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/bunko/associations",
        Some(assoc("蔵", "所蔵", "古文書")),
    );
    recall("bunko");
    recall("bunko");
    assert_eq!(cache_hits(&server, "recall"), 2);
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(assoc("蔵", "銘柄", "青嶺")),
    );
    recall("bunko");
    assert_eq!(
        cache_hits(&server, "recall"),
        3,
        "a write to sake must not invalidate bunko's entry"
    );

    // Cursor pages are ordinary key parameters: each page caches on
    // its own, and repeating a page hits it.
    let paged = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "蔵", "limit": 1})),
    );
    let after = &paged["matches"][0];
    let page_two = json!({"cue": "蔵", "limit": 1, "after": {
        "weight": after["weight"], "subject": after["subject"],
        "label": after["label"], "object": after["object"]}});
    let first_serve = server.ok("POST", "/contexts/sake/recall", Some(page_two.clone()));
    let second_serve = server.ok("POST", "/contexts/sake/recall", Some(page_two));
    assert_eq!(first_serve, second_serve);
    assert_eq!(cache_hits(&server, "recall"), 4);

    assert!(
        metric(&server, "taguru_retrieval_cache_entries") > 0,
        "filled entries show on the gauge"
    );
    assert!(metric(&server, "taguru_retrieval_cache_bytes") > 0);
}

/// The per-lane contract, both directions: a passage search ignores
/// graph writes but recomputes on passage and config (floor, refresh)
/// changes — and a recall ignores config changes. The lane-contribution
/// family keeps counting per served response across hits.
#[test]
fn passage_search_and_recall_invalidate_along_their_own_lanes_only() {
    let server = Server::start("rcache-lanes");
    server.ok("PUT", "/contexts/pass", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/pass/sources",
        Some(json!({"passages": {"a.md": "青嶺は端麗辛口の酒である。"}})),
    );

    let search = || {
        server.ok(
            "POST",
            "/contexts/pass/sources/search",
            Some(json!({"query": "端麗"})),
        )
    };
    let first = search();
    let second = search();
    assert_eq!(first, second);
    assert_eq!(cache_misses(&server, "search_passages"), 1);
    assert_eq!(cache_hits(&server, "search_passages"), 1);
    assert_eq!(
        metric(
            &server,
            "taguru_passage_lane_contributions_total{lane=\"bm25_only\"}"
        ),
        2,
        "lane contributions count served hits, cached serves included"
    );

    // A graph write is invisible to the passage lanes.
    server.ok(
        "POST",
        "/contexts/pass/associations",
        Some(assoc("青嶺", "味", "端麗辛口")),
    );
    search();
    assert_eq!(
        cache_hits(&server, "search_passages"),
        2,
        "a graph write must not invalidate a passage search"
    );

    // A passage store moves the passages lane...
    server.ok(
        "POST",
        "/contexts/pass/sources",
        Some(json!({"passages": {"b.md": "端麗な味わいの純米酒。"}})),
    );
    search();
    assert_eq!(cache_misses(&server, "search_passages"), 2);
    // ...and a config change (the context's own floor) the config lane.
    server.ok(
        "PATCH",
        "/contexts/pass",
        Some(json!({"semantic_floor": 0.4})),
    );
    search();
    assert_eq!(cache_misses(&server, "search_passages"), 3);

    // The reverse direction: recall keys on (graph, passages), so a
    // config change leaves its entries standing.
    let recall = || {
        server.ok(
            "POST",
            "/contexts/pass/recall",
            Some(json!({"cue": "青嶺"})),
        )
    };
    recall();
    recall();
    assert_eq!(cache_hits(&server, "recall"), 1);
    server.ok(
        "PATCH",
        "/contexts/pass",
        Some(json!({"semantic_floor": 0.6})),
    );
    recall();
    assert_eq!(
        cache_hits(&server, "recall"),
        2,
        "a config change must not invalidate a recall"
    );
}

/// Scope isolation falls out of resolved-target keying: two keys whose
/// grants slice a group differently resolve different target lists and
/// never share an entry — while two keys reaching the identical
/// single-context request do share, which is safe because the
/// middleware already vetted both.
#[test]
fn scoped_keys_share_entries_exactly_when_their_grants_resolve_alike() {
    let server = Server::start_with_env(
        "rcache-scopes",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,narrow:ntok,wide:wtok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"narrow": {"role": "read", "contexts": ["x"]},
                    "wide": {"role": "read", "contexts": ["x", "y"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        let (status, parsed) = server.call_with_token(method, path, body, Some(token));
        assert_eq!(status, 200, "{method} {path} -> {parsed}");
        parsed["result"].clone()
    };
    for context in ["x", "y"] {
        call(
            "PUT",
            &format!("/contexts/{context}"),
            Some(json!({"description": "d"})),
            "atok",
        );
        call(
            "POST",
            &format!("/contexts/{context}/associations"),
            Some(assoc("蔵", "在処", context)),
            "atok",
        );
    }
    call(
        "PUT",
        "/groups/g",
        Some(json!({"description": "", "contexts": ["x", "y"], "groups": []})),
        "atok",
    );

    let cross = json!({"groups": ["g"], "cue": "蔵"});
    let wide_page = call("POST", "/recall", Some(cross.clone()), "wtok");
    assert_eq!(wide_page["total"], 2, "the wide grant reaches both members");
    let narrow_page = call("POST", "/recall", Some(cross.clone()), "ntok");
    assert_eq!(
        narrow_page["total"], 1,
        "the narrow grant sees its slice of the group, never the wide key's page"
    );
    assert_eq!(
        cache_hits(&server, "recall"),
        0,
        "different resolved target lists must not share an entry"
    );
    assert_eq!(cache_misses(&server, "recall"), 2);
    call("POST", "/recall", Some(cross.clone()), "ntok");
    call("POST", "/recall", Some(cross), "wtok");
    assert_eq!(
        cache_hits(&server, "recall"),
        2,
        "each grant hits its own entry on repeat"
    );

    // Single-context requests pass the middleware's grant check before
    // the handler, so identical requests share safely across keys.
    let single = json!({"cue": "蔵"});
    call("POST", "/contexts/x/recall", Some(single.clone()), "ntok");
    call("POST", "/contexts/x/recall", Some(single), "wtok");
    assert_eq!(
        cache_hits(&server, "recall"),
        3,
        "the wide key reuses the narrow key's single-context fill"
    );
}

/// Revisions restart at zero on delete+recreate, so the counters alone
/// could collide a recreated context's key with the old incarnation's
/// entry — the per-incarnation identity in the key is what keeps the
/// old page unreachable.
#[test]
fn a_recreated_context_never_serves_the_old_incarnations_results() {
    let server = Server::start("rcache-recreate");
    server.ok("PUT", "/contexts/re", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/re/associations",
        Some(assoc("蔵", "杜氏", "高瀬")),
    );
    let recall = || server.ok("POST", "/contexts/re/recall", Some(json!({"cue": "蔵"})));
    let cached = recall();
    recall();
    assert_eq!(cached["matches"][0]["object"], "高瀬");
    assert_eq!(cache_hits(&server, "recall"), 1);

    // Recreate and drive the graph lane back to the exact value the
    // cached entry was keyed at (one applied op → graph = 1).
    server.ok("DELETE", "/contexts/re", None);
    server.ok("PUT", "/contexts/re", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/re/associations",
        Some(assoc("蔵", "杜氏", "別人")),
    );
    let fresh = recall();
    assert_eq!(
        fresh["matches"][0]["object"], "別人",
        "colliding revisions must not resurrect the old incarnation's page"
    );
    assert_eq!(cache_hits(&server, "recall"), 1, "no further hit");
    assert_eq!(cache_misses(&server, "recall"), 2);
}

/// `TAGURU_RETRIEVAL_CACHE_BYTES=0` is the operator escape hatch:
/// every request computes fresh, and the cache families stay silent —
/// no fake misses from a cache nobody is running.
#[test]
fn a_zero_budget_disables_the_cache_and_its_counters() {
    let server = Server::start_with_env("rcache-off", &[("TAGURU_RETRIEVAL_CACHE_BYTES", "0")]);
    server.ok("PUT", "/contexts/off", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/off/associations",
        Some(assoc("蔵", "杜氏", "高瀬")),
    );
    let first = server.ok("POST", "/contexts/off/recall", Some(json!({"cue": "蔵"})));
    let second = server.ok("POST", "/contexts/off/recall", Some(json!({"cue": "蔵"})));
    assert_eq!(first, second, "disabled changes how, never what");
    assert_eq!(cache_hits(&server, "recall"), 0);
    assert_eq!(cache_misses(&server, "recall"), 0);
    assert_eq!(metric(&server, "taguru_retrieval_cache_entries"), 0);
}

/// The resolved target list is part of the key IN ORDER: the cross
/// passage merge breaks rank ties by target-list order, so the same
/// set named in a different order is a different result — and must be
/// a different entry.
#[test]
fn cross_passage_search_keys_on_target_order() {
    let server = Server::start("rcache-order");
    for context in ["cx", "cy"] {
        server.ok(
            "PUT",
            &format!("/contexts/{context}"),
            Some(json!({"description": "d"})),
        );
        server.ok(
            "POST",
            &format!("/contexts/{context}/sources"),
            Some(json!({"passages": {"a.md": format!("{context}の蔵は端麗な酒を醸す。")}})),
        );
    }
    let search = |contexts: Value| {
        server.ok(
            "POST",
            "/sources/search",
            Some(json!({"contexts": contexts, "query": "端麗"})),
        )
    };
    search(json!(["cx", "cy"]));
    search(json!(["cx", "cy"]));
    assert_eq!(cache_hits(&server, "search_passages"), 1);
    search(json!(["cy", "cx"]));
    assert_eq!(
        cache_misses(&server, "search_passages"),
        2,
        "a reordered target list is a different result, so a different key"
    );
}
