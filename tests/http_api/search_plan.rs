//! The execution plan riding every search response (#151): each
//! vector-lane state names itself in `plan` exactly as `explain` would
//! phrase it — off, nothing embedded yet, model changed, provider
//! refused — a zero-hit page stays distinguishable from a lane that
//! never ran, a transiently failed embedding is confessed AND kept out
//! of the retrieval caches (provider recovery bumps no revision lane,
//! so a fill would pin the degraded page), and the graph searches list
//! their resolved targets in effective order.

use serde_json::{Value, json};

use crate::support::*;

/// One counter value scraped off /metrics (the cache tests' helper).
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

fn exact(server: &Server, outcome: &str) -> u64 {
    metric(
        server,
        &format!("taguru_retrieval_cache_total{{op=\"search_passages\",outcome=\"{outcome}\"}}"),
    )
}

/// The vector-lane entry of a single-context response's plan.
fn vector_plan(page: &Value) -> Value {
    page["plan"]["contexts"][0]["lanes"]["vector"].clone()
}

/// An embeddings stub that answers every input with the same unit
/// vector — the lane states under test here are gates and failures,
/// not similarity arithmetic.
fn spawn_flat_embeddings() -> String {
    spawn_embeddings_with(|input_len| {
        let data: Vec<Value> = (0..input_len)
            .map(|_| json!({ "embedding": [1.0, 0.0, 0.0] }))
            .collect();
        let body = json!({ "data": data }).to_string();
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    })
}

/// An embeddings stub that refuses every call — the transient-failure
/// provider.
fn spawn_refusing_embeddings() -> String {
    spawn_embeddings_with(|_| {
        "HTTP/1.1 500 Internal Server Error\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            .to_string()
    })
}

/// The shared one-shot HTTP machinery of the two stubs above: read one
/// request whole, hand the input count to `respond`, write its bytes.
fn spawn_embeddings_with(respond: fn(usize) -> String) -> String {
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
                let request: Value =
                    serde_json::from_slice(&buffer[body_start..body_start + length]).unwrap();
                let input_len = request["input"].as_array().map_or(0, Vec::len);
                let _ = stream.write_all(respond(input_len).as_bytes());
            });
        }
    });
    format!("http://{addr}/v1/embeddings")
}

/// Every stable vector-lane state, walked in the order an operator
/// would meet them: lane off (provider configured), nothing embedded
/// yet, ran under the default floor, and — after a restart under a
/// renamed model — the model-changed refusal. The reason strings are
/// asserted verbatim: they are the same prose `explain` emits, and the
/// plan's contract is that the two never drift.
#[test]
fn the_plan_names_every_stable_vector_lane_state() {
    let provider = spawn_flat_embeddings();

    // Provider configured, lane off: the plan says which.
    let off = Server::start_with_env(
        "plan-lane-off",
        &[
            ("TAGURU_EMBED_URL", provider.as_str()),
            ("TAGURU_EMBED_MODEL", "alpha"),
        ],
    );
    off.ok("PUT", "/contexts/mine", Some(json!({"description": "d"})));
    off.ok(
        "POST",
        "/contexts/mine/sources",
        Some(json!({"passages": {"docs/ore.md": "琥珀は樹脂の化石である。"}})),
    );
    let page = off.ok(
        "POST",
        "/contexts/mine/sources/search",
        Some(json!({"query": "琥珀"})),
    );
    assert_eq!(
        vector_plan(&page),
        json!({"ran": false, "reason": "passage embedding is off (TAGURU_EMBED_PASSAGES)"}),
        "{page}"
    );

    // Zero hits with the plan still present: "nothing matched" and
    // "lane didn't run" are different sentences now.
    let none = off.ok(
        "POST",
        "/contexts/mine/sources/search",
        Some(json!({"query": "真珠"})),
    );
    assert_eq!(none["hits"], json!([]), "{none}");
    assert_eq!(
        none["plan"]["contexts"][0]["lanes"]["bm25"]["ran"],
        json!(true),
        "{none}"
    );

    // A query with no searchable terms: both lanes honestly not run.
    let empty = off.ok(
        "POST",
        "/contexts/mine/sources/search",
        Some(json!({"query": "、。"})),
    );
    assert_eq!(
        empty["plan"]["contexts"][0]["lanes"]["vector"],
        json!({"ran": false, "reason": "the query yields no searchable terms"}),
        "{empty}"
    );

    // Lane on, nothing embedded yet.
    let server = Server::start_with_env(
        "plan-lane-states",
        &[
            ("TAGURU_EMBED_URL", provider.as_str()),
            ("TAGURU_EMBED_MODEL", "alpha"),
            ("TAGURU_EMBED_PASSAGES", "1"),
        ],
    );
    server.ok("PUT", "/contexts/mine", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/mine/sources",
        Some(json!({"passages": {"docs/ore.md": "琥珀は樹脂の化石である。"}})),
    );
    let unrefreshed = server.ok(
        "POST",
        "/contexts/mine/sources/search",
        Some(json!({"query": "琥珀"})),
    );
    assert_eq!(
        vector_plan(&unrefreshed),
        json!({"ran": false,
               "reason": "no paragraph vectors exist yet — the embedding refresh has not covered this context"}),
        "{unrefreshed}"
    );

    // Refreshed: the lane runs and names the effective (default) floor.
    server.ok("POST", "/contexts/mine/embeddings/refresh", None);
    let ran = server.ok(
        "POST",
        "/contexts/mine/sources/search",
        Some(json!({"query": "琥珀"})),
    );
    assert_eq!(vector_plan(&ran)["ran"], json!(true), "{ran}");
    assert!(
        (vector_plan(&ran)["floor"].as_f64().unwrap() - 0.35).abs() < 1e-6,
        "{ran}"
    );

    // The same plan through the MCP transport — the tool passes the
    // handler's envelope through verbatim, plan included.
    let reply = server.call_tool(
        1,
        "search_passages",
        json!({"context": "mine", "query": "琥珀"}),
    );
    assert!(reply.get("isError").is_none(), "{reply}");
    let text = reply["content"][0]["text"].as_str().unwrap();
    let envelope: Value = serde_json::from_str(text).unwrap();
    assert_eq!(
        envelope["result"]["plan"]["contexts"][0]["lanes"]["vector"]["ran"],
        json!(true),
        "{envelope}"
    );

    // A restart under a renamed model: the stored rows are refused by
    // name, and the plan says whose they were.
    let dir = server.stop_gracefully();
    let renamed = Server::start_on_with_env(
        "plan-model-renamed",
        dir,
        &[
            ("TAGURU_EMBED_URL", provider.as_str()),
            ("TAGURU_EMBED_MODEL", "beta"),
            ("TAGURU_EMBED_PASSAGES", "1"),
        ],
    );
    let changed = renamed.ok(
        "POST",
        "/contexts/mine/sources/search",
        Some(json!({"query": "琥珀"})),
    );
    assert_eq!(
        vector_plan(&changed),
        json!({"ran": false,
               "reason": "stored vectors belong to model 'alpha' but the provider is \
                          'beta' — they are never served, and the next refresh re-embeds"}),
        "{changed}"
    );
    let _ = std::fs::remove_dir_all(renamed.stop_gracefully());
}

/// The one lane state that recovers with no revision bump: a refused
/// query embedding degrades the answer to BM25 — and the plan confesses
/// it — but the degraded page is NOT filled into the retrieval cache
/// (nor registered as a semantic claim), so the provider's recovery is
/// not pinned out by a stale fill. Identical searches keep missing:
/// the pre-cache status quo, not a poisoned hit.
#[test]
fn a_refused_query_embedding_is_confessed_and_never_cached() {
    let provider = spawn_refusing_embeddings();
    let server = Server::start_with_env(
        "plan-embed-refused",
        &[
            ("TAGURU_EMBED_URL", provider.as_str()),
            ("TAGURU_EMBED_MODEL", "alpha"),
            ("TAGURU_EMBED_PASSAGES", "1"),
        ],
    );
    server.ok("PUT", "/contexts/mine", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/mine/sources",
        Some(json!({"passages": {"docs/ore.md": "琥珀は樹脂の化石である。"}})),
    );

    let search = || {
        server.ok(
            "POST",
            "/contexts/mine/sources/search",
            Some(json!({"query": "琥珀"})),
        )
    };
    let first = search();
    assert_eq!(
        first["hits"][0]["source"],
        json!("docs/ore.md"),
        "the lexical lane still answers: {first}"
    );
    let vector = vector_plan(&first);
    assert_eq!(vector["ran"], json!(false), "{first}");
    assert!(
        vector["reason"]
            .as_str()
            .unwrap()
            .starts_with("the query embedding failed"),
        "{first}"
    );

    // The second refusal arrives through the circuit breaker, so its
    // reason names the open circuit instead of the raw HTTP failure —
    // the hits are identical, the account honestly is not.
    let second = search();
    assert_eq!(first["hits"], second["hits"], "same degraded hits");
    assert!(
        vector_plan(&second)["reason"]
            .as_str()
            .unwrap()
            .starts_with("the query embedding failed"),
        "{second}"
    );
    assert_eq!(
        (exact(&server, "miss"), exact(&server, "hit")),
        (2, 0),
        "a transiently degraded page must not be pinned into the cache"
    );
}

/// The graph searches' plan: the contexts actually consulted, in
/// effective order — for the cross variants the RESOLVED list (direct
/// names in request order, then group-reached members in name order,
/// deduped), which the tagged matches alone cannot reconstruct when a
/// target comes up empty.
#[test]
fn graph_plans_list_the_resolved_targets_in_effective_order() {
    let server = Server::start("plan-graph");
    for context in ["amber", "quartz"] {
        server.ok(
            "PUT",
            &format!("/contexts/{context}"),
            Some(json!({"description": "d"})),
        );
    }
    server.ok(
        "PUT",
        "/groups/minerals",
        Some(json!({"description": "", "contexts": ["amber", "quartz"], "groups": []})),
    );

    let single = server.ok(
        "POST",
        "/contexts/amber/recall",
        Some(json!({"cue": "何もない"})),
    );
    assert_eq!(single["total"], json!(0), "{single}");
    assert_eq!(
        single["plan"],
        json!({"contexts": ["amber"]}),
        "zero matches still carry the plan: {single}"
    );

    // quartz is named directly AND reached through the group: seated
    // once, first; the group's other member follows in name order.
    let cross = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["quartz"], "groups": ["minerals"], "cue": "何もない"})),
    );
    assert_eq!(
        cross["plan"],
        json!({"contexts": ["quartz", "amber"]}),
        "{cross}"
    );

    let queried = server.ok(
        "POST",
        "/query",
        Some(json!({"contexts": ["quartz"], "groups": ["minerals"], "subject": "誰か"})),
    );
    assert_eq!(
        queried["plan"],
        json!({"contexts": ["quartz", "amber"]}),
        "{queried}"
    );
}
