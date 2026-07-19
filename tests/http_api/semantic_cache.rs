//! The semantic retrieval tier end to end: a paraphrased passage query
//! serves its canonical's cached page, every guard tripwire
//! (negation, number, entity) splits instead of serving, a write turns
//! a held claim stale and the next fill re-canonicalizes the cluster,
//! scoped keys share claims exactly when their grants resolve alike,
//! and the tier stays silent unless both the threshold and the
//! embedding lane are configured.
//!
//! The stub provider maps each test query to a fixed unit vector by
//! angle in one plane, so every pairwise cosine below is chosen, not
//! discovered: paraphrase pairs sit a few degrees apart (cosine well
//! above the 0.9 test threshold), unrelated queries tens of degrees
//! apart (well below), and unknown text lands on the orthogonal axis.

use serde_json::{Value, json};

use crate::support::*;

/// One counter/gauge value scraped off /metrics (same helper as the
/// exact-tier tests).
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

fn semantic(server: &Server, outcome: &str) -> u64 {
    metric(
        server,
        &format!("taguru_semantic_cache_total{{outcome=\"{outcome}\"}}"),
    )
}

fn exact(server: &Server, outcome: &str) -> u64 {
    metric(
        server,
        &format!("taguru_retrieval_cache_total{{op=\"search_passages\",outcome=\"{outcome}\"}}"),
    )
}

/// The pair-controlled embeddings endpoint: every known query string
/// gets a unit vector at a fixed angle in the x-y plane; anything else
/// (passage text, unknown queries) sits on z, orthogonal to all of
/// them.
fn vector_for(text: &str) -> Vec<f64> {
    let degrees = match text {
        "does the mill produce oysters" => Some(0.0),
        "is the mill producing oysters" => Some(14.0),
        "does the mill not produce oysters" => Some(10.0),
        "did the mill sell 20 oysters" => Some(40.0),
        "did the mill sell 30 oysters" => Some(44.0),
        "does Acme run the mill" => Some(72.0),
        "does Globex run the mill" => Some(76.0),
        _ => None,
    };
    match degrees {
        Some(degrees) => {
            let radians = f64::to_radians(degrees);
            vec![radians.cos(), radians.sin(), 0.0]
        }
        None => vec![0.0, 0.0, 1.0],
    }
}

fn spawn_paired_embeddings() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            std::thread::spawn(move || {
                use std::io::{Read, Write};
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let body_start = loop {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                    if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        break at + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&buffer[..body_start]).to_string();
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())?
                    })
                    .unwrap_or(0);
                while buffer.len() < body_start + length {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                }
                let request: serde_json::Value =
                    serde_json::from_slice(&buffer[body_start..body_start + length]).unwrap();
                let data: Vec<serde_json::Value> = request["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|text| json!({ "embedding": vector_for(text.as_str().unwrap()) }))
                    .collect();
                let body = json!({ "data": data }).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            });
        }
    });
    format!("http://{addr}/v1/embeddings")
}

fn semantic_env(provider: &str) -> Vec<(&'static str, String)> {
    vec![
        ("TAGURU_EMBED_URL", provider.to_string()),
        ("TAGURU_EMBED_MODEL", "paired-model".to_string()),
        ("TAGURU_EMBED_PASSAGES", "1".to_string()),
        ("TAGURU_SEMANTIC_CACHE_THRESHOLD", "0.9".to_string()),
    ]
}

/// Every #153 verify checkbox in one corpus: the paraphrase serves the
/// canonical's identical page while negation, numeric, and entity
/// flips each probe above the cosine threshold and are refused by the
/// guard — visible as `guarded`, never as a serve. The rewritten
/// lookup a hit rides stays out of the exact family's counters.
#[test]
fn a_paraphrase_serves_the_canonical_page_and_every_tripwire_splits() {
    let provider = spawn_paired_embeddings();
    let env = semantic_env(&provider);
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let server = Server::start_with_env("semcache-guard", &env);
    server.ok("PUT", "/contexts/mill", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/mill/sources",
        Some(json!({"passages": {"docs/mill.md":
            "The mill produces fresh oysters. Acme sells them while Globex ships 20 or 30 boxes."}})),
    );

    let search = |query: &str| {
        server.ok(
            "POST",
            "/contexts/mill/sources/search",
            Some(json!({"query": query})),
        )
    };

    // Fill, then prove the exact tier still fronts the semantic one.
    let base = search("does the mill produce oysters");
    assert!(!base.as_array().unwrap().is_empty(), "the corpus matches");
    assert_eq!(semantic(&server, "miss"), 1, "an empty bucket is a miss");
    search("does the mill produce oysters");
    assert_eq!(exact(&server, "hit"), 1);
    assert_eq!(
        semantic(&server, "hit"),
        0,
        "an exact hit never reaches the semantic tier"
    );

    // The paraphrase: cosine 0.97 to the canonical, no tripwires.
    let paraphrase = search("is the mill producing oysters");
    assert_eq!(paraphrase, base, "the canonical's page, unchanged");
    assert_eq!(semantic(&server, "hit"), 1);

    // Negation, number, entity: each pair sits ABOVE the cosine
    // threshold, so only the guard separates them.
    search("does the mill not produce oysters");
    assert_eq!(semantic(&server, "guarded"), 1, "negation split");
    search("did the mill sell 20 oysters");
    assert_eq!(semantic(&server, "miss"), 2, "a new cluster opens");
    search("did the mill sell 30 oysters");
    assert_eq!(semantic(&server, "guarded"), 2, "numeric split");
    search("does Acme run the mill");
    assert_eq!(semantic(&server, "miss"), 3);
    search("does Globex run the mill");
    assert_eq!(semantic(&server, "guarded"), 3, "entity split");

    assert_eq!(semantic(&server, "stale"), 0);
    assert_eq!(
        metric(&server, "taguru_semantic_cache_entries"),
        6,
        "every fresh fill registered a claim; the served paraphrase did not"
    );
    // Seven distinct query strings each missed their own exact key
    // once; the semantic hit's rewritten lookup was counted in ITS
    // family, not here.
    assert_eq!(exact(&server, "miss"), 7);
    assert_eq!(exact(&server, "hit"), 1);
}

/// A write bumps the passages lane: the claim still holds but its
/// rewritten key misses (`stale`), the fresh fill re-canonicalizes the
/// cluster onto the newest wording, and the ORIGINAL wording then
/// serves from the new canonical — both directions of the equivalence
/// stay live across corpus motion, and the cluster stays one slot.
#[test]
fn a_write_turns_the_claim_stale_and_the_next_fill_recanonicalizes() {
    let provider = spawn_paired_embeddings();
    let env = semantic_env(&provider);
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let server = Server::start_with_env("semcache-stale", &env);
    server.ok("PUT", "/contexts/mill", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/mill/sources",
        Some(json!({"passages": {"docs/mill.md": "The mill produces fresh oysters daily."}})),
    );

    let search = |query: &str| {
        server.ok(
            "POST",
            "/contexts/mill/sources/search",
            Some(json!({"query": query})),
        )
    };

    search("does the mill produce oysters");
    search("is the mill producing oysters");
    assert_eq!(semantic(&server, "hit"), 1);

    // The corpus moves: a second document lands in the passages lane.
    server.ok(
        "POST",
        "/contexts/mill/sources",
        Some(json!({"passages": {"docs/news.md": "News: the mill exports oysters overseas."}})),
    );
    let refreshed = search("is the mill producing oysters");
    assert_eq!(
        semantic(&server, "stale"),
        1,
        "the claim held; the corpus had moved on"
    );
    assert!(
        refreshed
            .as_array()
            .unwrap()
            .iter()
            .any(|hit| hit["source"] == "docs/news.md"),
        "the stale fall-through recomputed against the new corpus: {refreshed}"
    );

    // The original wording now serves from the re-canonicalized claim.
    let original_again = search("does the mill produce oysters");
    assert_eq!(semantic(&server, "hit"), 2);
    assert_eq!(
        original_again, refreshed,
        "the cluster's newest fill answers for every member"
    );
    assert_eq!(
        metric(&server, "taguru_semantic_cache_entries"),
        1,
        "re-canonicalization replaces; it never accumulates"
    );
}

/// The cross surface keys claims on the RESOLVED target list, so two
/// grants share a claim exactly when they resolve a request
/// identically — the same posture as the exact tier — while the
/// single-context surface shares freely behind the middleware's own
/// grant check.
#[test]
fn scoped_keys_share_claims_exactly_when_their_grants_resolve_alike() {
    let provider = spawn_paired_embeddings();
    let mut env = semantic_env(&provider);
    env.push((
        "TAGURU_API_TOKENS",
        "boss:atok,narrow:ntok,wide:wtok".to_string(),
    ));
    env.push((
        "TAGURU_KEY_SCOPES",
        r#"{"narrow": {"role": "read", "contexts": ["x"]},
            "wide": {"role": "read", "contexts": ["x", "y"]}}"#
            .to_string(),
    ));
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let server = Server::start_with_env("semcache-scopes", &env);
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
            &format!("/contexts/{context}/sources"),
            Some(json!({"passages": {"docs/a.md": "The mill produces fresh oysters daily."}})),
            "atok",
        );
    }
    call(
        "PUT",
        "/groups/g",
        Some(json!({"description": "", "contexts": ["x", "y"], "groups": []})),
        "atok",
    );

    let ask = |query: &str| json!({"groups": ["g"], "query": query});
    let wide_page = call(
        "POST",
        "/sources/search",
        Some(ask("does the mill produce oysters")),
        "wtok",
    );
    let narrow_page = call(
        "POST",
        "/sources/search",
        Some(ask("is the mill producing oysters")),
        "ntok",
    );
    assert!(
        narrow_page
            .as_array()
            .unwrap()
            .iter()
            .all(|hit| hit["context"] == "x"),
        "the narrow grant sees its slice, never the wide fill: {narrow_page}"
    );
    assert_eq!(
        semantic(&server, "hit"),
        0,
        "different resolved target lists must not share a claim"
    );
    assert_eq!(semantic(&server, "miss"), 2);

    // Same grant, paraphrased: each resolves as before and hits its
    // own claim — in both directions.
    let wide_para = call(
        "POST",
        "/sources/search",
        Some(ask("is the mill producing oysters")),
        "wtok",
    );
    assert_eq!(wide_para, wide_page);
    call(
        "POST",
        "/sources/search",
        Some(ask("does the mill produce oysters")),
        "ntok",
    );
    assert_eq!(semantic(&server, "hit"), 2);

    // Single-context requests pass the middleware's grant check before
    // the handler, so a paraphrase shares across keys safely — and the
    // cross claims above never bleed into this differently-tagged
    // bucket (the fill below is a miss, not a hit off the cross slot).
    let narrow_single = call(
        "POST",
        "/contexts/x/sources/search",
        Some(json!({"query": "does the mill produce oysters"})),
        "ntok",
    );
    assert_eq!(semantic(&server, "miss"), 3);
    let wide_single = call(
        "POST",
        "/contexts/x/sources/search",
        Some(json!({"query": "is the mill producing oysters"})),
        "wtok",
    );
    assert_eq!(
        semantic(&server, "hit"),
        3,
        "the wide key reuses the narrow key's single-context claim"
    );
    assert_eq!(wide_single, narrow_single);
}

/// Unset threshold, or a configured threshold without the embedding
/// lane: the tier must not run, must not call the provider, and must
/// keep every one of its counters at zero — the exact tier alone
/// decides how those requests are served.
#[test]
fn the_tier_stays_silent_without_a_threshold_or_without_the_lane() {
    let provider = spawn_paired_embeddings();
    // Embedding lane on, no threshold: paraphrases recompute.
    let unset = [
        ("TAGURU_EMBED_URL", provider.as_str()),
        ("TAGURU_EMBED_MODEL", "paired-model"),
        ("TAGURU_EMBED_PASSAGES", "1"),
    ];
    let server = Server::start_with_env("semcache-unset", &unset);
    server.ok("PUT", "/contexts/mill", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/mill/sources",
        Some(json!({"passages": {"docs/mill.md": "The mill produces fresh oysters daily."}})),
    );
    server.ok(
        "POST",
        "/contexts/mill/sources/search",
        Some(json!({"query": "does the mill produce oysters"})),
    );
    server.ok(
        "POST",
        "/contexts/mill/sources/search",
        Some(json!({"query": "is the mill producing oysters"})),
    );
    for outcome in ["hit", "stale", "guarded", "miss"] {
        assert_eq!(semantic(&server, outcome), 0, "unset threshold: {outcome}");
    }
    assert_eq!(metric(&server, "taguru_semantic_cache_entries"), 0);
    assert_eq!(exact(&server, "miss"), 2, "the exact tier still fills");

    // Threshold set, embedding lane off: same silence.
    let no_lane = [("TAGURU_SEMANTIC_CACHE_THRESHOLD", "0.9")];
    let server = Server::start_with_env("semcache-nolane", &no_lane);
    server.ok("PUT", "/contexts/mill", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/mill/sources",
        Some(json!({"passages": {"docs/mill.md": "The mill produces fresh oysters daily."}})),
    );
    server.ok(
        "POST",
        "/contexts/mill/sources/search",
        Some(json!({"query": "does the mill produce oysters"})),
    );
    server.ok(
        "POST",
        "/contexts/mill/sources/search",
        Some(json!({"query": "is the mill producing oysters"})),
    );
    for outcome in ["hit", "stale", "guarded", "miss"] {
        assert_eq!(semantic(&server, outcome), 0, "lane off: {outcome}");
    }
}
