//! A provider changing vector width behind a stable model name (#133):
//! a `dimensions` setting is a request-time parameter on modern models,
//! so the same model id can legitimately answer a different width. The
//! sidecars record the width they were built with, the serve paths
//! refuse to score across a mismatch AND SAY SO — before this, resolve
//! answered `[]` with explain prescribing a floor no value could
//! satisfy, and search served an empty page whose plan claimed the
//! vector lane ran — and the next refresh detects the change, wipes,
//! and re-embeds both stores, counted on /metrics.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use crate::support::*;

/// One counter value scraped off /metrics.
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

fn width_rebuilds(server: &Server, store: &str) -> u64 {
    metric(
        server,
        &format!("taguru_embedding_width_rebuilds_total{{store=\"{store}\"}}"),
    )
}

/// An embeddings stub whose vector width is adjustable at runtime:
/// every input answers with a unit vector [1, 0, ...] of the current
/// width — the states under test are width gates, not similarity
/// arithmetic.
fn spawn_width_embeddings(width: Arc<AtomicUsize>) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let width = Arc::clone(&width);
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
                let request: Value =
                    serde_json::from_slice(&buffer[body_start..body_start + length]).unwrap();
                let input_len = request["input"].as_array().map_or(0, Vec::len);
                let dim = width.load(Ordering::Relaxed);
                let data: Vec<Value> = (0..input_len)
                    .map(|_| {
                        let mut vector = vec![0.0; dim];
                        vector[0] = 1.0;
                        json!({ "embedding": vector })
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

/// The whole arc, in the order an operator lives it: sidecars built at
/// one width serve; the provider's width changes behind the same model
/// name; every serve surface names the mismatch instead of shipping
/// silent zeros; one refresh wipes, re-embeds (counted), and heals.
#[test]
fn a_width_change_is_named_on_every_surface_and_one_refresh_heals_it() {
    let width = Arc::new(AtomicUsize::new(3));
    let provider = spawn_width_embeddings(Arc::clone(&width));
    let server = Server::start_with_env(
        "width-change",
        &[
            ("TAGURU_EMBED_URL", provider.as_str()),
            ("TAGURU_EMBED_MODEL", "titan-v2"),
            ("TAGURU_EMBED_PASSAGES", "1"),
        ],
    );
    server.ok("PUT", "/contexts/mine", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/mine/associations",
        Some(json!([{"subject": "琥珀", "label": "分類", "object": "宝石", "weight": 1.0}])),
    );
    server.ok(
        "POST",
        "/contexts/mine/sources",
        Some(json!({"passages": {"docs/ore.md": "琥珀は樹脂の化石である。"}})),
    );
    server.ok("POST", "/contexts/mine/embeddings/refresh", None);

    // Both tiers serve at the built width: a cue/query sharing no
    // spelling with anything stored reaches vectors alone (the stub's
    // identical unit vectors put cosine at 1.0).
    let resolved = server.ok(
        "POST",
        "/contexts/mine/resolve",
        Some(json!({"cue": "べっこう"})),
    );
    assert_eq!(resolved[0]["tier"], json!("semantic"), "{resolved}");
    let page = server.ok(
        "POST",
        "/contexts/mine/sources/search",
        Some(json!({"query": "こはくいろ"})),
    );
    assert_eq!(page["hits"][0]["source"], json!("docs/ore.md"), "{page}");

    // The provider now answers width 4 behind the same model name.
    // Fresh cue/query strings on every call below: the cue cache must
    // not mask the change.
    width.store(4, Ordering::Relaxed);

    // Search: an empty page whose plan NAMES the mismatch — before
    // #133 it claimed `{"ran": true}` here.
    let reason = "stored vectors are 3-dimensional but the model now answers 4 \
                  (a dimensions setting changed behind its name) — they are \
                  never served, and the next refresh re-embeds";
    let page = server.ok(
        "POST",
        "/contexts/mine/sources/search",
        Some(json!({"query": "とうめいなたから"})),
    );
    assert_eq!(page["hits"], json!([]), "{page}");
    assert_eq!(
        page["plan"]["contexts"][0]["lanes"]["vector"],
        json!({"ran": false, "reason": reason}),
        "{page}"
    );

    // Search explain: the same sentence, not `Ran {cosine: None}`'s
    // "no current embedding yet" misdiagnosis.
    let explained = server.ok(
        "POST",
        "/contexts/mine/sources/search/explain",
        Some(json!({"query": "とうめいなたから", "source": "docs/ore.md"})),
    );
    assert_eq!(explained["vector"]["ran"], json!(false), "{explained}");
    assert_eq!(explained["vector"]["reason"], json!(reason), "{explained}");

    // Resolve folds to empty exactly like a model change (the tier is
    // best-effort by contract)…
    let resolved = server.ok(
        "POST",
        "/contexts/mine/resolve",
        Some(json!({"cue": "むかしのじゅし"})),
    );
    assert_eq!(resolved, json!([]), "{resolved}");
    // …and explain names the width, where it used to report the 0.0
    // sentinel as a measured cosine and prescribe an unreachable floor.
    let explained = server.ok(
        "POST",
        "/contexts/mine/resolve/explain",
        Some(json!({"cue": "むかしのじゅし", "expected": "琥珀"})),
    );
    assert_eq!(
        explained["verdict"],
        json!("semantic_not_run"),
        "{explained}"
    );
    assert_eq!(
        explained["semantic"]["reason"],
        json!(
            "gloss vectors are 3-dimensional but the model now answers 4 \
             (a dimensions setting changed behind its name) — awaiting re-embed"
        ),
        "{explained}"
    );
    assert!(explained["semantic"]["cosine"].is_null(), "{explained}");

    // One refresh heals both stores — and the wipes are counted, not
    // just warned about.
    assert_eq!(width_rebuilds(&server, "gloss"), 0);
    assert_eq!(width_rebuilds(&server, "passages"), 0);
    let refreshed = server.ok("POST", "/contexts/mine/embeddings/refresh", None);
    assert_eq!(refreshed["glosses"]["embedded"], json!(3), "{refreshed}");
    assert_eq!(refreshed["passages"]["embedded"], json!(1), "{refreshed}");
    assert_eq!(width_rebuilds(&server, "gloss"), 1);
    assert_eq!(width_rebuilds(&server, "passages"), 1);

    let resolved = server.ok(
        "POST",
        "/contexts/mine/resolve",
        Some(json!({"cue": "アンバー"})),
    );
    assert_eq!(resolved[0]["tier"], json!("semantic"), "{resolved}");
    let page = server.ok(
        "POST",
        "/contexts/mine/sources/search",
        Some(json!({"query": "アンバーのそう"})),
    );
    assert_eq!(page["hits"][0]["source"], json!("docs/ore.md"), "{page}");
    assert_eq!(
        page["plan"]["contexts"][0]["lanes"]["vector"]["ran"],
        json!(true),
        "{page}"
    );
}
