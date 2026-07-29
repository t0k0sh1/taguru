//! #278: a decisive, deterministic offline fixture for `taguru
//! evaluate` (issue #273, ADR 0004) — the graph-only, BM25-only, fused
//! lexical/vector, source-metadata-filter, grouping-boundary, citation,
//! known-miss, and explicitly-irrelevant-candidate paths #215's
//! acceptance criteria name, plus the two structural guarantees ADR
//! 0004 §12 requires to land with this issue: `LanePlan{ran:false,
//! reason:"..."}`'s verbatim `vector_off_reason` string appearing in
//! `evaluation.json` when no provider is configured, and (the source
//! text check for "no answer-generation LLM" itself lives beside
//! `evaluate.rs`, in `src/evaluate/tests.rs`) the wire proof that the
//! *server* — not just `evaluate`'s own process — actually had none.
//!
//! `tests/http_api/evaluate.rs` already covers the harness's wiring
//! (both lanes run, a read-only key completes a run, the preflight
//! refusal, an ambiguous position, the thresholds gate's pass/fail/
//! stable-by-default paths); this file is deliberately a separate
//! module so that smoke suite stays small and this one can grow a
//! richer, purpose-built corpus without either crowding the other.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use crate::support::*;

fn eval_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "taguru-evaluate-fixture-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("eval scratch dir must be creatable");
    dir
}

fn write_eval_file(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("eval.jsonl");
    std::fs::write(&path, contents).unwrap();
    path
}

fn write_thresholds(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("thresholds.json");
    std::fs::write(&path, contents).unwrap();
    path
}

// ============================ Recording/mutating proxies ============================
//
// Both proxies share one one-shot-per-connection HTTP skeleton with
// the embeddings stubs elsewhere in this test cluster
// (`passages.rs::spawn_fruity_embeddings`, `search_plan.rs::
// spawn_flat_embeddings`): read one request whole off a fresh
// `TcpListener::incoming()` connection, act on it, write one response,
// `connection: close`. Unlike those stubs — which answer AS the
// embeddings endpoint — these sit IN FRONT of a real `taguru` server
// and forward every request to it unchanged (`spawn_recording_proxy`)
// or unchanged-but-for-one-injected-write (`spawn_mutating_proxy`).

/// Reads one HTTP/1.1 request off `stream`: method, path (with any
/// query string), and body bytes (by `Content-Length`; 0 when absent,
/// which handles the harness's bodyless GETs same as a POST).
fn read_proxied_request(stream: &mut TcpStream) -> Option<(String, String, Vec<u8>)> {
    // A stalled peer must surface as a fast `None` (and so a readable
    // test failure), never an indefinite block that reads as an
    // opaque CI hang.
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let body_start = loop {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return None,
            Ok(n) => buffer.extend_from_slice(&chunk[..n]),
        }
        if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..body_start]).to_string();
    let mut lines = head.lines();
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let length: usize = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())?
        })
        .unwrap_or(0);
    while buffer.len() < body_start + length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => buffer.extend_from_slice(&chunk[..n]),
        }
    }
    let end = (body_start + length).min(buffer.len());
    Some((method, path, buffer[body_start..end].to_vec()))
}

/// One outbound call to `target_base` — the machinery both proxies use
/// to forward a client's request, and the mutating proxy also uses to
/// make its own injected write.
fn relay(target_base: &str, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let request = ureq::http::Request::builder()
        .method(method)
        .uri(format!("{target_base}{path}"));
    let response = match body {
        Some(body) => request
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .map(|request| test_agent().run(request)),
        None => request.body(()).map(|request| test_agent().run(request)),
    };
    let mut response = response
        .expect("relay request must assemble")
        .expect("relay request must run");
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().unwrap_or_default();
    (status, text)
}

fn relay_and_respond(
    stream: &mut TcpStream,
    target_base: &str,
    method: &str,
    path: &str,
    body: &[u8],
) {
    let body_text = (!body.is_empty()).then(|| String::from_utf8_lossy(body).to_string());
    let (status, text) = relay(target_base, method, path, body_text.as_deref());
    let out = format!(
        "HTTP/1.1 {status} relay\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}",
        text.len()
    );
    let _ = stream.write_all(out.as_bytes());
}

/// Every `(method, path)` a proxy has forwarded so far.
type CallLog = Arc<Mutex<Vec<(String, String)>>>;

/// A plain pass-through reverse proxy that records every `(method,
/// path)` `taguru evaluate` sends. ADR 0004 §7's module doc claims
/// `recall`/`activate`/`explore`/`describe` are never called — an
/// architectural fact (an HTTP client cannot call a route it never
/// names), but this checks it against the ACTUAL wire traffic of a
/// real run rather than trusting the doc comment alone.
fn spawn_recording_proxy(target_base: String) -> (String, CallLog) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let calls_for_thread = calls.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let target_base = target_base.clone();
            let calls = calls_for_thread.clone();
            std::thread::spawn(move || {
                let Some((method, path, body)) = read_proxied_request(&mut stream) else {
                    return;
                };
                calls.lock().unwrap().push((method.clone(), path.clone()));
                relay_and_respond(&mut stream, &target_base, &method, &path, &body);
            });
        }
    });
    (format!("http://{addr}"), calls)
}

/// [`spawn_recording_proxy`]'s sibling for ADR 0004 §12's `corpus.
/// stable == false` gate: on the FIRST request whose path ends
/// `/sources/search` — guaranteed to land after the run's opening `GET
/// /contexts/{name}` and before its closing one — it injects one write
/// directly against `target_base`, then forwards the original request
/// unchanged. `taguru evaluate` itself exposes no pause hook between
/// its two revision reads, so this is the only way to reproduce an
/// actual mid-run write end-to-end rather than driving `stable` by
/// hand (which `src/evaluate/thresholds.rs`'s own unit tests already
/// do for the threshold-evaluation logic in isolation).
fn spawn_mutating_proxy(target_base: String, context: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let injected = Arc::new(AtomicBool::new(false));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let target_base = target_base.clone();
            let context = context.clone();
            let injected = injected.clone();
            std::thread::spawn(move || {
                let Some((method, path, body)) = read_proxied_request(&mut stream) else {
                    return;
                };
                let path_only = path.split('?').next().unwrap_or(&path);
                if path_only.ends_with("/sources/search") && !injected.swap(true, Ordering::SeqCst)
                {
                    let write =
                        json!({"passages": {"corpus/mid-run.md": "後から追加された文章。"}});
                    let _ = relay(
                        &target_base,
                        "POST",
                        &format!("/contexts/{context}/sources"),
                        Some(&write.to_string()),
                    );
                }
                relay_and_respond(&mut stream, &target_base, &method, &path, &body);
            });
        }
    });
    format!("http://{addr}")
}

// ================================ Offline corpus (3.1/3.2) ================================
//
// Five sources, deliberately vocabulary-disjoint (Japanese BM25 terms
// are adjacent-character bigrams — `src/registry/terms.rs` — so
// distinct kanji runs with no shared two-character sequence are
// structurally invisible to each other's queries, not just
// coincidentally unmatched):
//
// - `corpus/brewery.md` — the one real hit, carries a section marker
//   for the citation lane's three-valued section check.
// - `corpus/filter-a.md`/`filter-b.md` — share a literal lexical
//   prefix so an untagged query matches both; `options.tags` then
//   narrows to one (#167's pre-lane filter).
// - `corpus/ledger.md` — a real, preflight-passing source that no
//   case's actual query ever retrieves: the "known miss" fixture.
// - `corpus/unrelated.md` — named in no case's `expected_sources` and
//   sharing no bigram with any query in this file: the "explicitly
//   irrelevant candidate" that must never appear in any hit set.
fn seed_offline_corpus(server: &Server, context: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{context}"),
        Some(json!({"description": "#278 offline fixture"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{context}/sources"),
        Some(json!({
            "passages": {
                "corpus/brewery.md": "青嶺は青嶺酒造が造る銘柄です。",
                "corpus/filter-a.md": "共通見出し語句。蔵元の説明文はここにある。",
                "corpus/filter-b.md": "共通見出し語句。原料米の説明文はここにある。",
                "corpus/ledger.md": "杜氏見習いの日誌帳を几帳面につける。",
                "corpus/unrelated.md": "梅雨明けの空に入道雲が広がった。",
            },
            "tags": {
                "corpus/filter-a.md": ["蔵"],
                "corpus/filter-b.md": ["原料"],
            },
            "sections": {
                "corpus/brewery.md": [{"paragraph": 0, "section": "由来"}],
            },
        })),
    );
    server.ok(
        "POST",
        &format!("/contexts/{context}/associations"),
        Some(json!([
            {"subject": "青嶺酒造", "label": "醸造元", "object": "蔵元",
             "weight": 1.0, "source": "corpus/brewery.md", "paragraph": 0},
        ])),
    );
}

/// Five cases, one `taguru evaluate` run: graph-only (no
/// `expected_sources` at all — recall/lane_cross must both stay
/// absent), BM25-only (the vector lane's honest-off reason, asserted
/// verbatim), the source-metadata filter, a known miss (a real,
/// preflight-passing source the case's own query never actually
/// retrieves), and the citation lane's full outcome variety (resolved
/// with a matching quote, a matching section, a mismatched section, a
/// mismatched quote, `no_paragraph`, `no_source`) run without any
/// `expected_sources` of its own — ADR 0004 §8's orthogonality claim.
fn write_offline_eval(dir: &Path) -> PathBuf {
    let lines = [
        r#"{"taguru_eval":1,"name":"evaluate fixture: offline corpus"}"#,
        r#"{"case_id":"graph-only-001","query":"青嶺酒造とは","cues":["青嶺酒造"],"expected_concepts":["青嶺酒造"],"expected_associations":[{"subject":"青嶺酒造","label":"醸造元","object":"蔵元"}]}"#,
        r#"{"case_id":"bm25-only-002","query":"青嶺","expected_sources":[{"source":"corpus/brewery.md","relevance":3}]}"#,
        r#"{"case_id":"tag-filter-003","query":"共通見出し語句","expected_sources":[{"source":"corpus/filter-a.md","relevance":3}],"options":{"tags":["蔵"]}}"#,
        r#"{"case_id":"known-miss-004","query":"青嶺","expected_sources":[{"source":"corpus/ledger.md","relevance":3}]}"#,
        r#"{"case_id":"citations-005","query":"存在しないクエリ","expected_citations":[{"source":"corpus/brewery.md","paragraph":0,"quote":"青嶺酒造"},{"source":"corpus/brewery.md","paragraph":0,"section":"由来"},{"source":"corpus/brewery.md","paragraph":0,"section":"別セクション"},{"source":"corpus/brewery.md","paragraph":0,"quote":"存在しない引用文"},{"source":"corpus/brewery.md","paragraph":99},{"source":"corpus/does-not-exist.md","paragraph":0}]}"#,
    ];
    write_eval_file(dir, &(lines.join("\n") + "\n"))
}

#[test]
fn evaluate_fixture_covers_graph_bm25_filter_known_miss_and_citation_paths() {
    let server = Server::start("evaluate-fixture-offline");
    seed_offline_corpus(&server, "sake");
    // Routed through a recording proxy rather than straight at
    // `server.base`: this one run also carries ADR 0004 §7's "never
    // recall/activate/explore/describe" claim (checked at the bottom
    // of this test) — no separate harness needed for it.
    let (proxy_base, calls) = spawn_recording_proxy(server.base.clone());
    let dir = eval_dir("offline");
    let eval_path = write_offline_eval(&dir);
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "sake",
            "--url",
            &proxy_base,
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");

    let evaluation: Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    let cases = evaluation["cases"].as_array().unwrap();
    let case = |id: &str| -> &Value {
        cases
            .iter()
            .find(|c| c["case_id"] == id)
            .unwrap_or_else(|| panic!("missing case '{id}': {evaluation}"))
    };

    // --- graph-only: resolve -> query, never recall/query paging; no
    // recall/lane_cross block since no expected_sources was declared.
    let graph = case("graph-only-001");
    assert!(graph["recall"].is_null(), "{graph}");
    assert!(graph["lane_cross"].is_null(), "{graph}");
    let cue = &graph["structural"]["cues"][0];
    assert_eq!(cue["cue"], "青嶺酒造", "{cue}");
    assert_eq!(cue["kind"], "concept", "{cue}");
    // ADR 0004 §7 step 1: an explicit limit of 5, never the ceiling.
    assert_eq!(cue["limit"], 5, "{cue}");
    assert!(
        cue["resolved_names"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "青嶺酒造"),
        "{cue}"
    );
    let assoc = &graph["structural"]["associations"][0];
    assert_eq!(assoc["subject"]["outcome"], "resolved", "{assoc}");
    assert_eq!(assoc["label"]["outcome"], "resolved", "{assoc}");
    assert_eq!(assoc["object"]["outcome"], "resolved", "{assoc}");
    assert_eq!(assoc["query"]["outcome"], "queried", "{assoc}");
    assert_eq!(assoc["query"]["total"], 1, "{assoc}");
    assert_eq!(graph["coverage"]["concepts"]["value"], 1.0, "{graph}");
    assert_eq!(graph["coverage"]["associations"]["value"], 1.0, "{graph}");

    // --- BM25-only: the vector lane's honest-off reason, verbatim
    // (AC 3, ADR 0004 §12's stronger check — this proves the SERVER
    // had no provider configured, not merely that `evaluate` never
    // imports one).
    let bm25 = case("bm25-only-002");
    assert_eq!(
        bm25["passage"]["plan"]["lanes"]["bm25"]["ran"], true,
        "{bm25}"
    );
    assert_eq!(
        bm25["passage"]["plan"]["lanes"]["vector"]["ran"], false,
        "{bm25}"
    );
    assert_eq!(
        bm25["passage"]["plan"]["lanes"]["vector"]["reason"], "no embedding provider is configured",
        "{bm25}"
    );
    assert_eq!(bm25["recall"]["recall_at_k"], 1.0, "{bm25}");
    let hit = &bm25["passage"]["hits"][0];
    assert_eq!(hit["source"], "corpus/brewery.md", "{hit}");
    assert!(hit["lanes"]["bm25"].is_object(), "{hit}");
    assert!(hit["lanes"]["vector"].is_null(), "{hit}");

    // --- source metadata filter (#167): only the tag-eligible source
    // answers, even though both filter-a/filter-b lexically match.
    let filtered = case("tag-filter-003");
    let filtered_hits = filtered["passage"]["hits"].as_array().unwrap();
    assert!(!filtered_hits.is_empty(), "{filtered}");
    assert!(
        filtered_hits
            .iter()
            .all(|hit| hit["source"] == "corpus/filter-a.md"),
        "{filtered}"
    );
    assert_eq!(filtered["recall"]["recall_at_k"], 1.0, "{filtered}");
    let filter_plan = &filtered["passage"]["plan"]["filter"];
    assert_eq!(filter_plan["eligible_sources"], 1, "{filter_plan}");
    assert_eq!(filter_plan["total_sources"], 5, "{filter_plan}");

    // --- known miss: `corpus/ledger.md` is a real, preflight-passing
    // source (seeded above) that this case's own query never actually
    // retrieves — a genuine regression signal, not a preflight defect.
    let miss = case("known-miss-004");
    assert_eq!(miss["recall"]["recall_at_k"], 0.0, "{miss}");
    assert_eq!(miss["recall"]["mrr"], 0.0, "{miss}");
    assert!(
        miss["missed"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m.as_str().unwrap().contains("corpus/ledger.md")),
        "{miss}"
    );

    // --- citation lane: full outcome variety, run without
    // expected_sources (ADR 0004 §8 orthogonality).
    let citations = &case("citations-005")["citations"]["checks"];
    let checks = citations.as_array().unwrap();
    assert_eq!(checks.len(), 6, "{checks:?}");
    assert_eq!(checks[0]["outcome"], "resolved", "{checks:?}");
    assert_eq!(checks[0]["quote"]["matched"], true, "{checks:?}");
    assert_eq!(checks[1]["outcome"], "resolved", "{checks:?}");
    assert_eq!(checks[1]["section"]["check"], "matched", "{checks:?}");
    assert_eq!(checks[2]["outcome"], "resolved", "{checks:?}");
    assert_eq!(checks[2]["section"]["check"], "mismatched", "{checks:?}");
    assert_eq!(checks[3]["outcome"], "resolved", "{checks:?}");
    assert_eq!(checks[3]["quote"]["matched"], false, "{checks:?}");
    assert_eq!(checks[4]["outcome"], "unresolved", "{checks:?}");
    assert_eq!(checks[4]["code"], "no_paragraph", "{checks:?}");
    assert_eq!(checks[5]["outcome"], "unresolved", "{checks:?}");
    assert_eq!(checks[5]["code"], "no_source", "{checks:?}");
    // ADR 0004 §11: never the served paragraph body, even beside a
    // resolved/matched check.
    for check in checks {
        assert!(check.get("text").is_none(), "{check}");
    }

    // --- explicitly irrelevant candidate: named in no case's
    // expected_sources, shares no bigram with any query above — must
    // never appear in any case's hit set.
    for c in cases {
        if let Some(hits) = c["passage"]["hits"].as_array() {
            assert!(
                hits.iter().all(|h| h["source"] != "corpus/unrelated.md"),
                "case '{}' surfaced the explicitly irrelevant candidate: {c}",
                c["case_id"]
            );
        }
    }

    // --- ADR 0004 §7: recall/activate/explore/describe are never
    // called, over this run's ACTUAL wire traffic (not just the
    // module doc's claim) — resolve/resolve_label/query are, from the
    // graph-only case above.
    let called = calls.lock().unwrap();
    let paths: Vec<&str> = called
        .iter()
        .map(|(_, path)| path.split('?').next().unwrap_or(path.as_str()))
        .collect();
    for verb in ["/recall", "/activate", "/explore", "/describe"] {
        assert!(
            paths.iter().all(|path| !path.ends_with(verb)),
            "'{verb}' must never be called by evaluate; saw {paths:?}"
        );
    }
    assert!(
        paths.iter().any(|path| path.ends_with("/resolve")),
        "the graph-only case's coverage cue must call /resolve; saw {paths:?}"
    );
    assert!(
        paths.iter().any(|path| path.ends_with("/query")),
        "the graph-only case's association probe must call /query; saw {paths:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ================================ Grouping boundary (3.4) ================================

/// ADR 0004 §1: `taguru evaluate` is a quality gate over ONE corpus,
/// never a cross-corpus comparison (that is `taguru benchmark search`,
/// #260) — `--context NAME` names exactly one context, and `eval.jsonl`
/// has no group/contexts field. This locks that boundary in against a
/// context that DOES belong to a group sharing a sibling's vocabulary:
/// evaluating `sake` never surfaces `beer`'s sources, even though both
/// carry "青嶺" and both are members of the same group.
#[test]
fn evaluate_never_crosses_into_a_sibling_group_members_sources() {
    let server = Server::start("evaluate-fixture-grouping");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"corpus/brewery.md": "青嶺は青嶺酒造が造る銘柄です。"}})),
    );
    server.ok("PUT", "/contexts/beer", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/beer/sources",
        Some(json!({"passages": {"corpus/hops.md": "青嶺麦酒のホップは芳醇である。"}})),
    );
    server.ok(
        "PUT",
        "/groups/beverages",
        Some(json!({"description": "飲料", "contexts": ["sake", "beer"]})),
    );

    let dir = eval_dir("grouping");
    let eval_path = write_eval_file(
        &dir,
        r#"{"taguru_eval":1,"name":"grouping boundary"}
{"case_id":"boundary-001","query":"青嶺","expected_sources":[{"source":"corpus/brewery.md","relevance":3}]}
"#,
    );
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "sake",
            "--url",
            &server.base,
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");
    let evaluation: Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    let hits = evaluation["cases"][0]["passage"]["hits"]
        .as_array()
        .unwrap();
    assert!(!hits.is_empty(), "{evaluation}");
    assert!(
        hits.iter().all(|h| h["source"] != "corpus/hops.md"),
        "sibling group member's source leaked into a single-context run: {evaluation}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The same boundary from the preflight side: an `expected_sources`
/// entry naming a sibling group member's source is refused exactly
/// like any other source `--context sake` does not itself carry —
/// group membership is not an exemption.
#[test]
fn evaluate_preflight_refuses_a_sibling_group_members_source() {
    let server = Server::start("evaluate-fixture-grouping-preflight");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"corpus/brewery.md": "青嶺は青嶺酒造が造る銘柄です。"}})),
    );
    server.ok("PUT", "/contexts/beer", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/beer/sources",
        Some(json!({"passages": {"corpus/hops.md": "青嶺麦酒のホップは芳醇である。"}})),
    );
    server.ok(
        "PUT",
        "/groups/beverages",
        Some(json!({"description": "飲料", "contexts": ["sake", "beer"]})),
    );

    let dir = eval_dir("grouping-preflight");
    let eval_path = write_eval_file(
        &dir,
        "{\"taguru_eval\":1,\"name\":\"grouping preflight\"}\n\
         {\"case_id\":\"ghost-001\",\"query\":\"青嶺\",\
         \"expected_sources\":[{\"source\":\"corpus/hops.md\",\"relevance\":1}]}\n",
    );
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "sake",
            "--url",
            &server.base,
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("sake"), "{stderr}");
    assert!(stderr.contains("corpus/hops.md"), "{stderr}");
    assert!(!out_path.exists());

    let _ = std::fs::remove_dir_all(&dir);
}

// ================================ Provider suite (3.3) ================================
//
// A stub embeddings endpoint, same TCP skeleton as
// `passages.rs::spawn_fruity_embeddings`: every `input` text that
// mentions the grape orchard (either the passage's own words or a
// paraphrase sharing none of them) gets the same unit vector; anything
// else is orthogonal. Deterministic, in-process, no network egress —
// stays inside the default repository gate exactly like every other
// stub in this cluster.
fn spawn_orchard_embeddings() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            std::thread::spawn(move || {
                let Some((_, _, body)) = read_proxied_request(&mut stream) else {
                    return;
                };
                let request: Value = serde_json::from_slice(&body).unwrap();
                let data: Vec<Value> = request["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|text| {
                        let text = text.as_str().unwrap();
                        let vector = if text.contains("ぶどう") || text.contains("果樹園") {
                            vec![1.0, 0.0, 0.0]
                        } else {
                            vec![0.0, 0.0, 1.0]
                        };
                        json!({ "embedding": vector })
                    })
                    .collect();
                let response_body = json!({ "data": data }).to_string();
                let out = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
                    response_body.len()
                );
                let _ = stream.write_all(out.as_bytes());
            });
        }
    });
    format!("http://{addr}/v1/embeddings")
}

/// AC 7: fused lexical/vector ranking, semantic paraphrase retrieval
/// under an embedding provider, and provider/model identity recorded
/// in the artifact — all opt-in via the existing stub-provider
/// pattern (`TAGURU_EMBED_URL`/`TAGURU_EMBED_MODEL`/
/// `TAGURU_EMBED_PASSAGES`), never a new feature flag, and fully
/// in-process/offline so this stays inside the default gate.
#[test]
fn evaluate_fixture_covers_fusion_and_semantic_paraphrase_under_a_provider() {
    const MODEL: &str = "orchard-stub-v1";
    let provider = spawn_orchard_embeddings();
    let server = Server::start_with_env(
        "evaluate-fixture-provider",
        &[
            ("TAGURU_EMBED_URL", provider.as_str()),
            ("TAGURU_EMBED_MODEL", MODEL),
            ("TAGURU_EMBED_PASSAGES", "1"),
        ],
    );
    server.ok(
        "PUT",
        "/contexts/orchard",
        Some(json!({"description": "果樹園"})),
    );
    server.ok(
        "POST",
        "/contexts/orchard/sources",
        Some(json!({"passages": {"corpus/grape.md": "ぶどう畑で収穫の準備が進む。"}})),
    );
    server.ok("POST", "/contexts/orchard/embeddings/refresh", None);

    let dir = eval_dir("provider");
    let eval_path = write_eval_file(
        &dir,
        r#"{"taguru_eval":1,"name":"provider suite"}
{"case_id":"fusion-001","query":"ぶどう畑","expected_sources":[{"source":"corpus/grape.md","relevance":3}]}
{"case_id":"paraphrase-002","query":"果樹園の様子","expected_sources":[{"source":"corpus/grape.md","relevance":3}]}
"#,
    );
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "orchard",
            "--url",
            &server.base,
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");
    let evaluation: Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    let cases = evaluation["cases"].as_array().unwrap();
    let case = |id: &str| -> &Value {
        cases
            .iter()
            .find(|c| c["case_id"] == id)
            .unwrap_or_else(|| panic!("missing case '{id}': {evaluation}"))
    };

    // --- fusion: the query shares the passage's own words (a lexical
    // hit) AND clears the semantic floor (a vector hit) — both lanes
    // must be present on the SAME hit.
    let fusion = case("fusion-001");
    assert_eq!(fusion["recall"]["recall_at_k"], 1.0, "{fusion}");
    let fused_hit = &fusion["passage"]["hits"][0];
    assert!(fused_hit["lanes"]["bm25"].is_object(), "{fused_hit}");
    assert!(fused_hit["lanes"]["vector"].is_object(), "{fused_hit}");
    let vector_plan = &fusion["passage"]["plan"]["lanes"]["vector"];
    assert_eq!(vector_plan["ran"], true, "{vector_plan}");
    assert!(vector_plan["floor"].is_number(), "{vector_plan}");

    // --- semantic paraphrase: zero lexical overlap with the passage
    // (no shared bigram), retrieved on the vector lane alone.
    let paraphrase = case("paraphrase-002");
    assert_eq!(paraphrase["recall"]["recall_at_k"], 1.0, "{paraphrase}");
    let paraphrase_hit = &paraphrase["passage"]["hits"][0];
    assert!(
        paraphrase_hit["lanes"]["bm25"].is_null(),
        "{paraphrase_hit}"
    );
    assert!(
        paraphrase_hit["lanes"]["vector"].is_object(),
        "{paraphrase_hit}"
    );

    // --- provider/model identity, recorded once for the whole run.
    assert_eq!(
        evaluation["corpus"]["embeddings"]["provider_model"], MODEL,
        "{evaluation}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The offline mirror of the identity assertion above: `GET
/// /contexts/{name}/embeddings` succeeds even with no provider
/// configured (`EmbeddingsStatus.provider_model: None`), so `evaluate`
/// still records a `corpus.embeddings` block — just with a `null`
/// `provider_model`, distinguishing "no provider" from "a provider
/// whose identity is merely unrecorded" (the latter would instead omit
/// the whole block, which only happens when the GET itself fails).
#[test]
fn evaluate_records_a_null_provider_model_when_no_provider_is_configured() {
    let server = Server::start("evaluate-fixture-no-provider");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"corpus/brewery.md": "青嶺は青嶺酒造が造る銘柄です。"}})),
    );
    let dir = eval_dir("no-provider");
    let eval_path = write_eval_file(
        &dir,
        "{\"taguru_eval\":1,\"name\":\"no provider\"}\n\
         {\"case_id\":\"c1\",\"query\":\"青嶺\",\
         \"expected_sources\":[{\"source\":\"corpus/brewery.md\",\"relevance\":1}]}\n",
    );
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "sake",
            "--url",
            &server.base,
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");
    let evaluation: Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    // `serde_json` indexing yields `Null` for an absent key too, which
    // would not tell "block present with a null provider_model" apart
    // from "no block at all" — the presence check below is what
    // actually locks in this test's own claim.
    assert!(
        evaluation["corpus"]["embeddings"].is_object(),
        "the corpus.embeddings block must be recorded: {evaluation}"
    );
    assert_eq!(
        evaluation["corpus"]["embeddings"]["provider_model"],
        Value::Null,
        "{evaluation}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ================================ stable:false (3.5) ================================

/// ADR 0004 §12: a write landing mid-run is bracketed, not hidden —
/// `corpus.stable == false`, and (with the default
/// `allow_unstable_corpus: false`) a `--thresholds` gate fails on it
/// even when every other bound is satisfied. Reproduced end-to-end via
/// [`spawn_mutating_proxy`] rather than driving `stable` by hand.
#[test]
fn evaluate_fails_the_gate_when_a_write_lands_mid_run() {
    let server = Server::start("evaluate-fixture-unstable");
    server.ok("PUT", "/contexts/watch", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/watch/sources",
        Some(json!({"passages": {"corpus/seed.md": "初期状態の文章です。"}})),
    );
    let proxy_base = spawn_mutating_proxy(server.base.clone(), "watch".to_string());

    let dir = eval_dir("unstable");
    let eval_path = write_eval_file(
        &dir,
        "{\"taguru_eval\":1,\"name\":\"mid-run write\"}\n\
         {\"case_id\":\"stability-001\",\"query\":\"初期状態\",\
         \"expected_sources\":[{\"source\":\"corpus/seed.md\",\"relevance\":1}]}\n",
    );
    let thresholds_path = write_thresholds(&dir, "{\"taguru_evaluate_thresholds\":1}");
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "watch",
            "--url",
            &proxy_base,
            "--out",
            out_path.to_str().unwrap(),
            "--thresholds",
            thresholds_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 3, "{stderr}");
    assert!(out_path.exists(), "the artifact must still be written");

    let evaluation: Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    assert_eq!(evaluation["corpus"]["stable"], false, "{evaluation}");
    assert_ne!(
        evaluation["corpus"]["revision_before"], evaluation["corpus"]["revision_after"],
        "{evaluation}"
    );
    assert_eq!(evaluation["thresholds"]["passed"], false, "{evaluation}");
    let violations = evaluation["thresholds"]["violations"].as_array().unwrap();
    let violation = violations
        .iter()
        .find(|v| v["metric"] == "corpus.stable")
        .unwrap_or_else(|| panic!("no corpus.stable violation recorded: {evaluation}"));
    assert_eq!(violation["scope"], "corpus", "{violation}");
    assert!(
        violation["reason"]
            .as_str()
            .unwrap()
            .contains("allow_unstable_corpus"),
        "{violation}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
