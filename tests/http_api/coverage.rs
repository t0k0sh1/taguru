//! `POST /contexts/{name}/embeddings/refresh`'s failure arms (issue
//! #626): `errors.rs:237-244` only ever reaches `EmbeddingsUnconfigured`
//! (501, no provider at all). Nothing in `tests/` ever configures a
//! provider that answers but FAILS — `ErrorCode::EmbeddingsFailed`
//! (502) and the `Timeout` (408) arm, on either the gloss half or the
//! passage half, plus `RefreshBreakdown.skipped_over_limit` and
//! `GET /contexts/{name}/embeddings`'s shape once a provider AND real
//! passages both exist (`calibrate.rs`/`contract.rs` only ever probe
//! the no-provider shape).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

use serde_json::{Value, json};

use crate::support::*;

/// What one stub connection does with a request whose `input` matched
/// its trigger.
#[derive(Clone, Copy)]
enum Fault {
    /// Answer HTTP 500 immediately — `ureq::Error::StatusCode`, a
    /// retryable transport-level refusal (`embedding.rs`'s
    /// `AttemptCall::run`: any `>= 500` status is retryable, so this
    /// still costs the two built-in retries before the caller sees it).
    Fail,
    /// Never answer within the caller's effective timeout — the
    /// client-side `ureq` timeout (`self.timeout.min(deadline.
    /// remaining())`) fires first, so this must simply outlast
    /// whatever `TAGURU_REQUEST_TIMEOUT_SECS` the test sets, not the
    /// literal duration given.
    Sleep,
}

/// An embeddings stub that answers ordinary success for every request
/// EXCEPT one whose `input` array contains `trigger` — `Some("")`
/// (every string contains the empty string) applies the fault to
/// EVERY request, `Some(marker)` applies it only to requests carrying
/// that marker (which is how the gloss half — embeds concept glosses,
/// no marker — and the passage half — embeds passage/question text,
/// seeded with the marker — are told apart; purpose alone cannot,
/// since `refresh_embeddings`/`refresh_passage_embeddings` both call
/// the provider with the same `EmbedPurpose::Index`), and `None` never
/// triggers it at all (an always-succeeding provider).
fn spawn_faulty_embeddings(trigger: Option<&'static str>, fault: Fault) -> String {
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
                let inputs: Vec<&str> = request["input"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|value| value.as_str())
                    .collect();
                let triggered =
                    trigger.is_some_and(|trigger| inputs.iter().any(|text| text.contains(trigger)));
                if triggered {
                    match fault {
                        Fault::Fail => {
                            let body = "provider on fire";
                            let response = format!(
                                "HTTP/1.1 500 Internal Server Error\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                                body.len()
                            );
                            let _ = stream.write_all(response.as_bytes());
                            return;
                        }
                        Fault::Sleep => {
                            std::thread::sleep(Duration::from_secs(3));
                            // Fall through and answer 200 — by the time
                            // this write happens the client's own
                            // request timeout has already given up and
                            // closed its side; the write is a no-op the
                            // dropped/reset connection swallows.
                        }
                    }
                }
                let data: Vec<Value> = inputs
                    .iter()
                    .map(|_| json!({ "embedding": [1.0f32, 0.0, 0.0, 0.0] }))
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

/// An embeddings stub that always succeeds, fixed width 4 — for the
/// tests below that need a working provider (the over-limit skip, and
/// `GET /embeddings` with real sidecars) rather than a failing one.
fn spawn_ok_embeddings() -> String {
    spawn_faulty_embeddings(None, Fault::Fail)
}

/// A minimal context with a live association (two concepts, one
/// label), so the gloss half has something to embed.
fn seed_one_concept(server: &Server, name: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{name}"),
        Some(json!({"description": "d"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{name}/associations"),
        Some(json!([
            {"subject": "青嶺酒造", "label": "住所", "object": "京都",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
}

/// The gloss half fails with a provider that is configured but
/// unreachable (dead port, no retries can succeed) — the `deadline`
/// is nowhere near expired (default 30s, no override), so this is
/// `EmbeddingsFailed`, not `Timeout`.
#[test]
fn refresh_reports_embeddings_failed_when_the_gloss_provider_is_unreachable() {
    let server = Server::start_with_env(
        "coverage-gloss-failed",
        &[
            ("TAGURU_EMBED_URL", "http://127.0.0.1:9/v1/embeddings"),
            ("TAGURU_EMBED_MODEL", "dead-mock"),
        ],
    );
    seed_one_concept(&server, "sake");

    let (status, body) = server.call("POST", "/contexts/sake/embeddings/refresh", None);
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["code"], json!("embeddings_failed"), "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .starts_with("embedding refresh failed: "),
        "{body}"
    );
}

/// The gloss half's own request outlives the request budget: the
/// client-side timeout (bounded by `TAGURU_REQUEST_TIMEOUT_SECS`) cuts
/// it, `deadline.expired()` is now true, and `refresh_embeddings`
/// picks the `Timeout` arm instead of `EmbeddingsFailed` for the exact
/// same underlying `Err(String)`.
#[test]
fn refresh_reports_timeout_when_the_gloss_provider_is_slow() {
    let server = Server::start_with_env(
        "coverage-gloss-timeout",
        &[
            (
                "TAGURU_EMBED_URL",
                spawn_faulty_embeddings(Some(""), Fault::Sleep).as_str(),
            ),
            ("TAGURU_EMBED_MODEL", "slow-mock"),
            ("TAGURU_REQUEST_TIMEOUT_SECS", "1"),
        ],
    );
    seed_one_concept(&server, "sake");

    let (status, body) = server.call("POST", "/contexts/sake/embeddings/refresh", None);
    assert_eq!(status, 408, "{body}");
    assert_eq!(body["code"], json!("timeout"), "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .starts_with("embedding refresh failed: "),
        "{body}"
    );
}

/// The passage half fails AFTER the gloss half already landed
/// successfully — the marker-triggered stub answers gloss requests
/// normally and only 500s the passage text carrying the trigger. The
/// message names the partial-progress guarantee explicitly, distinct
/// from the gloss-side message above.
#[test]
fn refresh_reports_embeddings_failed_on_the_passage_half_after_glosses_land() {
    let server = Server::start_with_env(
        "coverage-passage-failed",
        &[
            (
                "TAGURU_EMBED_URL",
                spawn_faulty_embeddings(Some("SENTINEL_PASSAGE_TEXT"), Fault::Fail).as_str(),
            ),
            ("TAGURU_EMBED_MODEL", "partial-mock"),
            ("TAGURU_EMBED_PASSAGES", "1"),
        ],
    );
    seed_one_concept(&server, "sake");
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"docs/a.md": "SENTINEL_PASSAGE_TEXT triggers the stub."}})),
    );

    let (status, body) = server.call("POST", "/contexts/sake/embeddings/refresh", None);
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["code"], json!("embeddings_failed"), "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .starts_with("passage embedding refresh failed partway (progress is saved): "),
        "{body}"
    );

    // The gloss half's own success is not undone by the passage
    // half's failure — the identity exposure shows the sidecar it
    // already built.
    let status_body = server.ok("GET", "/contexts/sake/embeddings", None);
    // seed_one_concept's one association mints two concepts (subject
    // AND object) plus one label.
    assert_eq!(
        status_body["glosses"]["concepts"],
        json!(2),
        "{status_body}"
    );
    assert_eq!(status_body["glosses"]["labels"], json!(1), "{status_body}");
    assert!(status_body["passages"].is_null(), "{status_body}");
}

/// The passage half's own timeout arm, mirroring the gloss-side
/// timeout test above.
#[test]
fn refresh_reports_timeout_on_the_passage_half_after_glosses_land() {
    let server = Server::start_with_env(
        "coverage-passage-timeout",
        &[
            (
                "TAGURU_EMBED_URL",
                spawn_faulty_embeddings(Some("SENTINEL_PASSAGE_TEXT"), Fault::Sleep).as_str(),
            ),
            ("TAGURU_EMBED_MODEL", "partial-slow-mock"),
            ("TAGURU_EMBED_PASSAGES", "1"),
            ("TAGURU_REQUEST_TIMEOUT_SECS", "1"),
        ],
    );
    seed_one_concept(&server, "sake");
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"docs/a.md": "SENTINEL_PASSAGE_TEXT triggers the stub."}})),
    );

    let (status, body) = server.call("POST", "/contexts/sake/embeddings/refresh", None);
    assert_eq!(status, 408, "{body}");
    assert_eq!(body["code"], json!("timeout"), "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .starts_with("passage embedding refresh failed partway (progress is saved): "),
        "{body}"
    );
}

/// `RefreshBreakdown.skipped_over_limit` (`coverage.rs:37`, set at the
/// passage half's own row-budget check) — never asserted anywhere.
/// `TAGURU_PASSAGE_VECTOR_LIMIT=1` caps the corpus to its first row;
/// the gloss half carries none (`glosses.skipped_over_limit` is
/// always `None` — `coverage.rs`'s own `RefreshOutcome` construction
/// hard-codes it).
#[test]
fn refresh_reports_passages_skipped_over_the_row_limit() {
    let server = Server::start_with_env(
        "coverage-over-limit",
        &[
            ("TAGURU_EMBED_URL", spawn_ok_embeddings().as_str()),
            ("TAGURU_EMBED_MODEL", "cap-mock"),
            ("TAGURU_EMBED_PASSAGES", "1"),
            ("TAGURU_PASSAGE_VECTOR_LIMIT", "1"),
        ],
    );
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "docs/a.md": "最初の段落。",
            "docs/b.md": "二番目の段落。",
        }})),
    );

    let refreshed = server.ok("POST", "/contexts/sake/embeddings/refresh", None);
    // No concepts in this corpus, but `glosses` still runs (and rides
    // in the response) whenever the passage lane is enabled — `None`
    // is reserved for the "passage embedding disabled" shape only.
    // `RefreshBreakdown.skipped_over_limit` is hard-coded `None` on
    // the gloss half (`coverage.rs`'s own `RefreshOutcome`
    // construction), so it never appears on the wire there.
    assert_eq!(refreshed["glosses"]["embedded"], json!(0), "{refreshed}");
    assert_eq!(
        refreshed["glosses"]["skipped_over_limit"],
        json!(null),
        "{refreshed}"
    );
    assert_eq!(
        refreshed["passages"]["skipped_over_limit"],
        json!(1),
        "{refreshed}"
    );
    assert_eq!(refreshed["passages"]["embedded"], json!(1), "{refreshed}");
}

/// `GET /contexts/{name}/embeddings`'s shape once a provider is
/// configured AND both sidecars hold real rows — `calibrate.rs`'s own
/// use of this route only ever probes the gloss-only, no-passages
/// shape; `contract.rs`'s wire fixture is the no-provider shape.
#[test]
fn embeddings_status_reports_both_sidecars_once_both_are_built() {
    let server = Server::start_with_env(
        "coverage-status-both",
        &[
            ("TAGURU_EMBED_URL", spawn_ok_embeddings().as_str()),
            ("TAGURU_EMBED_MODEL", "both-mock"),
            ("TAGURU_EMBED_PASSAGES", "1"),
        ],
    );
    seed_one_concept(&server, "sake");
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"docs/a.md": "段落一つ。"}})),
    );
    server.ok("POST", "/contexts/sake/embeddings/refresh", None);

    let status = server.ok("GET", "/contexts/sake/embeddings", None);
    assert_eq!(status["provider_model"], json!("both-mock"), "{status}");
    assert_eq!(status["glosses"]["model"], json!("both-mock"), "{status}");
    assert_eq!(status["glosses"]["width"], json!(4), "{status}");
    // seed_one_concept's one association mints two concepts (subject
    // AND object) plus one label.
    assert_eq!(status["glosses"]["concepts"], json!(2), "{status}");
    assert_eq!(status["glosses"]["labels"], json!(1), "{status}");
    assert_eq!(status["passages"]["model"], json!("both-mock"), "{status}");
    assert_eq!(status["passages"]["width"], json!(4), "{status}");
    assert_eq!(status["passages"]["rows"], json!(1), "{status}");
}
