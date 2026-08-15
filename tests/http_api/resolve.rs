//! `explain_resolve_verdict`'s semantic-tier verdicts and
//! `resolve_tiers`'s embedding-provider degrade paths (issue #627):
//! the real owner of `/resolve/explain` HTTP coverage is
//! `groups_cross_mcp.rs`, whose one test configures NO embedding
//! provider at all — every lexical verdict is pinned, but
//! `semantic_below_floor`, both `below_cutoff` arms reached through
//! the semantic lane, and `resolve_tiers`'s `EmbeddingsFailed`/
//! `Timeout`/weak-lexical-degrade arms were all unexercised.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

use serde_json::{Value, json};

use crate::support::*;

/// A cue whose raw text an embeddings stub axis-matches directly
/// (`cue_vector` embeds the bare cue string, unlike a concept's
/// gloss-wrapped text) — shares no bigram with anything this file
/// stores, so the lexical tier never finds it a candidate at all
/// (`target_score: None`, the precondition every semantic verdict
/// needs to be reached ahead of the lexical ones).
const CUE: &str = "とくちょう";

/// An embeddings stub that returns an exact, deliberately-constructed
/// cosine for each `(name, cosine)` pair in `targets` — not a
/// heuristic axis match, an exact unit-circle construction: `CUE`
/// embeds as `(1, 0)`, and a concept named `name` (gloss text starts
/// `"{name}。"`) embeds as `(cosine, sqrt(1 - cosine^2))`, so the raw
/// dot product (both are already unit vectors; `similarity` never
/// renormalizes) is exactly `cosine`, not an incidental value subject
/// to float rounding near a boundary. Anything else (the association
/// objects, any label) lands on `(0, 1)` — orthogonal to `CUE`,
/// cosine 0, never a false positive.
fn spawn_exact_cosine_embeddings(targets: &'static [(&'static str, f32)]) -> String {
    fn vector_for(text: &str, targets: &[(&str, f32)]) -> Vec<f32> {
        if text == CUE {
            return vec![1.0, 0.0];
        }
        for &(name, cosine) in targets {
            if text.starts_with(&format!("{name}。")) {
                let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
                return vec![cosine, sine];
            }
        }
        vec![0.0, 1.0]
    }

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            std::thread::spawn(move || {
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
                let request: Value =
                    serde_json::from_slice(&buffer[body_start..body_start + length]).unwrap();
                let data: Vec<Value> = request["input"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .map(|input| {
                        json!({ "embedding": vector_for(input.as_str().unwrap_or_default(), targets) })
                    })
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

fn exact_cosine_server(tag: &str, targets: &'static [(&'static str, f32)]) -> Server {
    Server::start_with_env(
        tag,
        &[
            (
                "TAGURU_EMBED_URL",
                spawn_exact_cosine_embeddings(targets).as_str(),
            ),
            ("TAGURU_EMBED_MODEL", "cosine-mock"),
        ],
    )
}

/// One concept per `(name, _)` pair in `targets`, each with a single
/// distinct object so its own gloss is exactly `"{name}。{label}は
/// {object}。"` — the text `spawn_exact_cosine_embeddings` matches by
/// prefix. `CUE` shares no bigram with any of these names (a
/// hiragana string against kanji/katakana names), so the lexical tier
/// never finds a candidate — every semantic verdict's precondition.
fn seed_concepts(server: &Server, name: &str, targets: &[(&str, f32)]) {
    server.ok(
        "PUT",
        &format!("/contexts/{name}"),
        Some(json!({"description": "d"})),
    );
    let batch: Vec<Value> = targets
        .iter()
        .map(|(concept, _)| {
            json!({
                "subject": concept, "label": "l",
                "object": format!("{concept}-obj"),
                "weight": 1.0, "source": "a.md",
            })
        })
        .collect();
    server.ok(
        "POST",
        &format!("/contexts/{name}/associations"),
        Some(Value::Array(batch)),
    );
    server.ok(
        "POST",
        &format!("/contexts/{name}/embeddings/refresh"),
        None,
    );
}

/// `semantic_below_floor` (`resolve.rs:926-933`): the expected name's
/// gloss cosine (0.2) sits under the default semantic floor (0.35,
/// `DEFAULT_SEMANTIC_FLOOR`, `src/registry.rs`).
#[test]
fn explain_reports_semantic_below_floor() {
    const TARGETS: &[(&str, f32)] = &[("低調", 0.2)];
    let server = exact_cosine_server("resolve-semantic-below-floor", TARGETS);
    seed_concepts(&server, "sake", TARGETS);

    let explained = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": CUE, "expected": "低調"})),
    );
    assert_eq!(
        explained["verdict"],
        json!("semantic_below_floor"),
        "{explained}"
    );
    assert_eq!(explained["semantic"]["entered"], json!(true), "{explained}");
    let cosine = explained["semantic"]["cosine"].as_f64().unwrap();
    assert!((cosine - 0.2).abs() < 1e-3, "{explained}");
    let floor = explained["semantic"]["floor"].as_f64().unwrap();
    assert!(cosine < floor, "{explained}");
}

/// `below_cutoff`'s semantic-cap arm (`resolve.rs:935-943`): six
/// concepts outrank the expected one, so its rank (7) exceeds
/// `SEMANTIC_RESOLVE_LIMIT` (5, `cap` in the response).
#[test]
fn explain_reports_below_cutoff_via_the_semantic_cap() {
    const TARGETS: &[(&str, f32)] = &[
        ("首位一", 1.0),
        ("首位二", 1.0),
        ("首位三", 1.0),
        ("首位四", 1.0),
        ("首位五", 1.0),
        ("首位六", 1.0),
        ("次点", 0.5),
    ];
    let server = exact_cosine_server("resolve-semantic-cap", TARGETS);
    seed_concepts(&server, "sake", TARGETS);

    let explained = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": CUE, "expected": "次点"})),
    );
    assert_eq!(explained["verdict"], json!("below_cutoff"), "{explained}");
    assert_eq!(explained["semantic"]["rank"], json!(7), "{explained}");
    assert_eq!(explained["semantic"]["cap"], json!(5), "{explained}");
    assert!(
        explained["summary"]
            .as_str()
            .unwrap()
            .contains("its top 5 only"),
        "{explained}"
    );
}

/// `below_cutoff`'s own-limit arm (`resolve.rs:947-954`): the expected
/// name clears the floor and sits within the semantic tier's own cap
/// (rank 2 of 5), but the REQUEST's own `limit: 1` is what actually
/// excludes it from `served` — a different reason than the cap arm
/// above, and the response must say so.
#[test]
fn explain_reports_below_cutoff_via_the_requests_own_limit() {
    const TARGETS: &[(&str, f32)] = &[("首位", 1.0), ("次点", 0.5)];
    let server = exact_cosine_server("resolve-semantic-own-limit", TARGETS);
    seed_concepts(&server, "sake", TARGETS);

    let explained = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": CUE, "expected": "次点", "limit": 1})),
    );
    assert_eq!(explained["verdict"], json!("below_cutoff"), "{explained}");
    assert_eq!(explained["semantic"]["rank"], json!(2), "{explained}");
    assert_eq!(explained["semantic"]["cap"], json!(5), "{explained}");
    assert!(
        explained["summary"]
            .as_str()
            .unwrap()
            .contains("the request served only 1"),
        "{explained}"
    );

    // Confirms the DISTINCT reason from the cap test above: raising
    // the request's own limit (without touching the corpus) now
    // serves it.
    let wider = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": CUE, "expected": "次点", "limit": 2})),
    );
    assert_eq!(wider["verdict"], json!("served"), "{wider}");
}

// --- resolve_tiers's embedding-provider degrade paths -----------------------

/// What one stub connection does with a QUERY-purpose embed request
/// (the cue) — an INDEX-purpose one (concept glosses, at refresh
/// time) always succeeds, so the sidecar is real and non-empty by the
/// time `resolve`/`explain` actually calls the provider again for the
/// cue itself. Mirrors `coverage.rs`'s `Fault`, gated on purpose
/// instead of content.
#[derive(Clone, Copy)]
enum QueryFault {
    Fail,
    Sleep,
}

fn spawn_query_faulty_embeddings(fault: QueryFault) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            std::thread::spawn(move || {
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
                let is_query = headers.lines().any(|line| {
                    line.to_ascii_lowercase()
                        .starts_with("x-taguru-embed-purpose: query")
                });
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
                if is_query {
                    match fault {
                        QueryFault::Fail => {
                            let body = "provider on fire";
                            let response = format!(
                                "HTTP/1.1 500 Internal Server Error\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                                body.len()
                            );
                            let _ = stream.write_all(response.as_bytes());
                            return;
                        }
                        QueryFault::Sleep => {
                            std::thread::sleep(Duration::from_secs(3));
                        }
                    }
                }
                let request: Value =
                    serde_json::from_slice(&buffer[body_start..body_start + length]).unwrap();
                let data: Vec<Value> = request["input"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .map(|_| json!({ "embedding": [1.0f32, 0.0] }))
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

/// A minimal corpus, refreshed against a QUERY-faulty provider while
/// it still answers INDEX requests fine — the sidecar is real, so a
/// later resolve's cue embed is what actually fails.
fn query_faulty_server(tag: &str, fault: QueryFault) -> Server {
    let server = Server::start_with_env(
        tag,
        &[
            (
                "TAGURU_EMBED_URL",
                spawn_query_faulty_embeddings(fault).as_str(),
            ),
            ("TAGURU_EMBED_MODEL", "query-fault-mock"),
        ],
    );
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "住所", "object": "京都",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
    server.ok("POST", "/contexts/sake/embeddings/refresh", None);
    server
}

/// `resolve_tiers`'s catch-all `Err` arm (`resolve.rs:285-292`):
/// `bounded` (lexical candidates) is empty — `CUE` shares no bigram
/// with anything stored — and the cue's own embed call fails outright
/// (502 `embeddings_failed`), not a Timeout (the deadline is nowhere
/// near expired).
#[test]
fn resolve_reports_embeddings_failed_when_the_cue_embed_is_unreachable() {
    let server = query_faulty_server("resolve-query-failed", QueryFault::Fail);

    let (status, body) = server.call("POST", "/contexts/sake/resolve", Some(json!({"cue": CUE})));
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["code"], json!("embeddings_failed"), "{body}");
}

/// `resolve_tiers`'s `Timeout` arm (`resolve.rs:274-284`): same empty-
/// `bounded` precondition, but the cue embed outlives the request
/// budget instead of failing outright.
#[test]
fn resolve_reports_timeout_when_the_cue_embed_is_slow() {
    let server = Server::start_with_env(
        "resolve-query-timeout",
        &[
            (
                "TAGURU_EMBED_URL",
                spawn_query_faulty_embeddings(QueryFault::Sleep).as_str(),
            ),
            ("TAGURU_EMBED_MODEL", "query-slow-mock"),
            ("TAGURU_REQUEST_TIMEOUT_SECS", "1"),
        ],
    );
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "住所", "object": "京都",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
    server.ok("POST", "/contexts/sake/embeddings/refresh", None);

    let (status, body) = server.call("POST", "/contexts/sake/resolve", Some(json!({"cue": CUE})));
    assert_eq!(status, 408, "{body}");
    assert_eq!(body["code"], json!("timeout"), "{body}");
}

/// `resolve_tiers`'s degrade-not-fail arm (`resolve.rs:270-273`):
/// `bounded` is NON-empty (the cue has a real, if unconfident, lexical
/// match) when the cue embed fails — the response still succeeds
/// (200), serving the weak lexical results alone rather than failing
/// the whole request over a best-effort tier.
#[test]
fn resolve_serves_weak_lexical_results_when_the_cue_embed_fails_but_lexical_matches_exist() {
    let server = query_faulty_server("resolve-query-degrade", QueryFault::Fail);
    // Shares the 青嶺/酒造 bigrams with 青嶺酒造 without containing it —
    // a real, unconfident (score 0.5 < LEXICAL_CONFIDENCE) fuzzy match,
    // proven by groups_cross_mcp.rs's own below_floor test to produce
    // exactly this shape.
    let cue = "青嶺の酒造り";

    let (status, body) = server.call("POST", "/contexts/sake/resolve", Some(json!({"cue": cue})));
    assert_eq!(status, 200, "{body}");
    let candidates = body["result"].as_array().expect("candidates array");
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate["name"] == json!("青嶺酒造")
                && candidate["tier"] == json!("lexical")
                && candidate["kind"] == json!("fuzzy")),
        "the weak lexical match must still be served: {body}"
    );
}
