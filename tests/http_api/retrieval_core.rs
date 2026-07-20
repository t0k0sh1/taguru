//! The documented retrieve loop and protocol tier reporting, driven over HTTP.

use serde_json::{Value, json};

use crate::support::*;

/// GET /protocol carries a live-configuration trailer once the semantic
/// tier is configured: an agent must learn from the manual itself that
/// `refresh_embeddings` is worth calling here (or already automatic) —
/// the static text alone leaves a configured tier dark.
#[test]
fn protocol_reports_the_semantic_tier_when_configured() {
    // /protocol never calls the provider, so a dead endpoint serves.
    let embed_env = [
        ("TAGURU_EMBED_URL", "http://127.0.0.1:9/v1/embeddings"),
        ("TAGURU_EMBED_MODEL", "proto-test-model"),
    ];
    let server = Server::start_with_env("proto-embed", &embed_env);
    let (status, protocol) = server.call("GET", "/protocol", None);
    assert_eq!(status, 200);
    let text = protocol.as_str().unwrap();
    assert!(text.contains("## This server"));
    assert!(text.contains("`proto-test-model`"));
    assert!(text.contains("calling `refresh_embeddings`"));
    assert!(!text.contains("auto-refreshes"));

    let mut auto_env = embed_env.to_vec();
    auto_env.push(("TAGURU_EMBED_AUTO", "1"));
    let server = Server::start_with_env("proto-auto", &auto_env);
    let (_, protocol) = server.call("GET", "/protocol", None);
    assert!(protocol.as_str().unwrap().contains("auto-refreshes"));
}

/// A request blocked mid-embed used to hold SIGTERM's graceful drain
/// for the provider's timeout ladder (60s × 3 attempts at the
/// defaults) — the deploy manifests carried 200s grace periods for
/// exactly this. The stop signal now abandons the provider wait: the
/// blocked search degrades to its lexical lane, answers during the
/// drain, and the process exits in seconds.
#[test]
fn sigterm_drains_promptly_past_an_in_flight_embed() {
    use std::time::Duration;

    // A provider that accepts and never answers, sockets held open —
    // announcing each arrival, so the test KNOWS when the search is
    // blocked inside the provider call rather than guessing by sleep.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let provider_addr = listener.local_addr().unwrap();
    let (arrived, provider_reached) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut held = Vec::new();
        while let Ok((stream, _)) = listener.accept() {
            let _ = arrived.send(());
            held.push(stream);
        }
    });

    let provider_url = format!("http://{provider_addr}/v1/embeddings");
    let server = Server::start_with_env(
        "drain-embed",
        &[
            ("TAGURU_EMBED_URL", provider_url.as_str()),
            ("TAGURU_EMBED_MODEL", "black-hole-model"),
            ("TAGURU_EMBED_PASSAGES", "1"),
            ("TAGURU_EMBED_TIMEOUT_SECS", "60"),
            ("TAGURU_REQUEST_TIMEOUT_SECS", "60"),
        ],
    );
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "docs/aomine.md": "原料米には山田錦を使い、精米歩合は50パーセントまで磨く。"
        }})),
    );

    // The vector lane embeds the query before anything else, so this
    // search blocks inside the black-holed provider call.
    let base = server.base.clone();
    let searcher = std::thread::spawn(move || {
        let response = test_agent()
            .post(format!("{base}/contexts/sake/sources/search"))
            .header("Content-Type", "application/json")
            .send(r#"{"query": "精米歩合", "limit": 3}"#);
        finish(response, "POST", "/contexts/sake/sources/search")
    });
    // Only signal once the search is provably blocked in the provider.
    provider_reached
        .recv_timeout(Duration::from_secs(30))
        .expect("the search must reach the provider");

    let started = std::time::Instant::now();
    let _dir = server.stop_gracefully();
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(20),
        "the drain held {elapsed:?}; the stop signal must abandon the in-flight \
         provider wait instead of waiting out its 60s ceiling"
    );

    // Answered during the drain, not reset — by the lexical lane
    // alone, the vector lane visibly absent from the hit, and the plan
    // confessing the abandoned embedding instead of hiding it.
    let (status, body) = searcher.join().unwrap();
    assert_eq!(status, 200, "{body}");
    let hits = body["result"]["hits"].as_array().expect("hits array");
    assert!(!hits.is_empty(), "{body}");
    assert!(hits[0]["lanes"]["bm25"].is_object(), "{body}");
    assert!(hits[0]["lanes"]["vector"].is_null(), "{body}");
    let vector_plan = &body["result"]["plan"]["contexts"][0]["lanes"]["vector"];
    assert_eq!(vector_plan["ran"], json!(false), "{body}");
    assert!(
        vector_plan["reason"]
            .as_str()
            .unwrap()
            .starts_with("the query embedding failed"),
        "{body}"
    );
}

#[test]
fn full_retrieval_loop_over_http() {
    let server = Server::start("loop");

    // Health, playbook, empty directory.
    let (status, health) = server.call("GET", "/health", None);
    assert_eq!((status, health), (200, Value::String("ok".into())));
    let (status, protocol) = server.call("GET", "/protocol", None);
    assert_eq!(status, 200);
    assert!(protocol.as_str().unwrap().contains("# Taguru"));
    // Lexical-only server: no live-configuration trailer to act on.
    assert!(!protocol.as_str().unwrap().contains("## This server"));
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["total"], json!(0));
    assert_eq!(directory["contexts"], json!([]));

    // Create; duplicates conflict; unknown contexts 404.
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の知識", "dice_floor": 0.3})),
    );
    let (status, _) = server.call("PUT", "/contexts/sake", Some(json!({})));
    assert_eq!(status, 409);
    let (status, _) = server.call("POST", "/contexts/nope/recall", Some(json!({"cue": "x"})));
    assert_eq!(status, 404);

    // Ingest a batch plus its passage.
    let applied = server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "第1段落"},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0, "source": "第2段落"},
            {"subject": "青嶺酒造", "label": "仕込み水", "object": "雲居山の伏流水", "weight": 1.0, "source": "第2段落"},
            {"subject": "青嶺酒造", "label": "仕込み水", "object": "雲居山の伏流水", "weight": 1.0, "source": "第5段落"},
            {"subject": "高瀬", "label": "出身", "object": "南部杜氏", "weight": 1.0, "source": "第3段落"},
            {"subject": "青嶺酒造", "label": "行う", "object": "大量生産", "weight": -1.0, "source": "第2段落"},
        ])),
    );
    assert_eq!(applied, json!(6));
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "第2段落": "青嶺酒造は、仕込み水に雲居山の伏流水を使う。杜氏は高瀬である。",
        }})),
    );
    let sources = server.ok("GET", "/contexts/sake/sources", None);
    assert_eq!(sources["total"], json!(1));
    assert_eq!(sources["sources"], json!(["第2段落"]));
    // The listing's metadata view (#167): same page, name plus the
    // server's stored_at stamp (its value is the wall clock's).
    assert_eq!(sources["entries"][0]["name"], "第2段落");
    assert!(sources["entries"][0]["stored_at"].is_u64());

    // recall/query pages carry totals; query takes OR-sets per position.
    let page = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造", "limit": 3})),
    );
    assert_eq!(page["total"], json!(4));
    assert_eq!(page["matches"].as_array().unwrap().len(), 3);
    // Truncation keeps the strongest |weight| first.
    assert_eq!(page["matches"][0]["label"], json!("杜氏"));
    let narrowed = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": ["代表銘柄", "杜氏"]})),
    );
    assert_eq!(narrowed["total"], json!(2));

    // describe outlines without materializing; corroboration shows in
    // attributions through query.
    let outline = server.ok(
        "POST",
        "/contexts/sake/describe",
        Some(json!({"concept": "青嶺酒造"})),
    );
    assert_eq!(outline["as_subject"][0]["label"], json!("代表銘柄")); // count ties -> label insertion order
    let water = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "仕込み水"})),
    );
    // Two sources each asserting 1.0 average to 1.0 — corroboration is
    // visible via count and the two attributions below, not via weight
    // alone.
    assert_eq!(water["matches"][0]["weight"], json!(1.0));
    assert_eq!(water["matches"][0]["count"], json!(2));
    assert_eq!(
        water["matches"][0]["attributions"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    // resolve tiers: exact is lexical; a typo lands through the fuzzy
    // tier; the per-call floor tightens it away.
    let exact = server.ok(
        "POST",
        "/contexts/sake/resolve",
        Some(json!({"cue": "青嶺酒造"})),
    );
    assert_eq!(exact[0]["tier"], json!("lexical"));
    assert_eq!(exact[0]["score"], json!(1.0));
    let typo = server.ok(
        "POST",
        "/contexts/sake/resolve",
        Some(json!({"cue": "青嶺酒蔵"})),
    );
    assert_eq!(typo[0]["name"], json!("青嶺酒造"));
    let strict = server.ok(
        "POST",
        "/contexts/sake/resolve",
        Some(json!({"cue": "青嶺酒蔵", "dice_floor": 0.9})),
    );
    assert!(
        !strict
            .as_array()
            .unwrap()
            .iter()
            .any(|hit| hit["name"] == json!("青嶺酒造"))
    );

    // Walks carry paths; strengths rank magnitude (the negative fact
    // outranks weight-1 facts).
    let ranked = server.ok(
        "POST",
        "/contexts/sake/activate",
        Some(json!({"origins": ["青嶺酒造"], "limit": 3})),
    );
    assert_eq!(ranked["matches"][0]["association"]["label"], json!("杜氏"));
    assert_eq!(ranked["matches"][0]["path"], json!(["青嶺酒造"]));
    let walked = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["青嶺酒造"], "max_depth": 2})),
    );
    assert!(
        walked["matches"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["distance"] == json!(2) && r["path"] == json!(["青嶺酒造", "高瀬"]))
    );

    // Aliases resolve at entry, answer with canonical spellings, and
    // refuse to shadow existing spellings.
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(
            json!({"concepts": {"Aomine Brewery": "青嶺酒造"}, "labels": {"蔵元の責任者": "杜氏"}}),
        ),
    );
    let via_alias = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "Aomine Brewery", "label": "蔵元の責任者"})),
    );
    assert_eq!(via_alias["matches"][0]["subject"], json!("青嶺酒造"));
    assert_eq!(via_alias["matches"][0]["object"], json!("高瀬"));
    let (status, _) = server.call(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"青嶺": "青嶺酒造"}})),
    );
    assert_eq!(status, 409, "shadowing an existing concept must conflict");

    // Coverage audit, passage lookup and search, retraction.
    let orphans = server.ok(
        "POST",
        "/contexts/sake/unreachable_from",
        Some(json!({"origins": ["青嶺酒造"]})),
    );
    assert_eq!(orphans, json!({"total": 0, "matches": []}));
    let passages = server.ok(
        "POST",
        "/contexts/sake/sources/lookup",
        Some(json!({"sources": ["第2段落", "第9段落"]})),
    );
    assert!(
        passages["passages"]["第2段落"]
            .as_str()
            .unwrap()
            .contains("伏流水")
    );
    assert_eq!(passages["missing"], json!(["第9段落"]));
    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "仕込み水はどこの水?"})),
    );
    assert_eq!(hits["hits"][0]["source"], json!("第2段落"));
    let retracted = server.ok(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "第5段落"})),
    );
    assert_eq!(retracted["associations_touched"], json!(1));
    let water = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "仕込み水"})),
    );
    assert_eq!(water["matches"][0]["weight"], json!(1.0));

    // Metadata edits show up in the directory; embeddings without a
    // provider are refused as unimplemented.
    server.ok(
        "PATCH",
        "/contexts/sake",
        Some(json!({"pinned": true, "semantic_floor": 0.2})),
    );
    let listed = server.ok("GET", "/contexts", None)["contexts"].clone();
    assert_eq!(listed[0]["pinned"], json!(true));
    assert_eq!(listed[0]["semantic_floor"], json!(0.2));
    assert_eq!(listed[0]["stats"]["associations"], json!(5));
    // top_concepts uses the same {label, count} object shape as describe's
    // as_subject/as_object, not a positional [name, count] tuple.
    assert_eq!(
        listed[0]["stats"]["top_concepts"][0],
        json!({"label": "青嶺酒造", "count": 4})
    );
    // The single-context row says the same thing without the listing.
    let single = server.ok("GET", "/contexts/sake", None);
    assert_eq!(single["name"], json!("sake"));
    assert_eq!(single["stats"]["associations"], json!(5));
    let (status, _) = server.call("POST", "/contexts/sake/embeddings/refresh", None);
    assert_eq!(status, 501);

    // Deletion removes the context and its files.
    server.ok("DELETE", "/contexts/sake", None);
    assert_eq!(server.ok("GET", "/contexts", None)["total"], json!(0));
    let (status, _) = server.call("GET", "/contexts/sake", None);
    assert_eq!(status, 404);
}
