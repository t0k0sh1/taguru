//! HTTP integration tests: the real binary, spawned on a free port with
//! a scratch data directory, driven through the same retrieval loop the
//! protocol documents. Everything here was once verified by hand with
//! curl; this pins it so handler wiring and response shapes cannot
//! regress silently.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use serde_json::{Value, json};

/// One running server on its own port and data directory, killed and
/// cleaned up on drop.
struct Server {
    child: Child,
    base: String,
    data_dir: PathBuf,
}

impl Server {
    fn start(tag: &str) -> Self {
        Self::start_with_env(tag, &[])
    }

    fn start_with_env(tag: &str, extra_env: &[(&str, &str)]) -> Self {
        let data_dir =
            std::env::temp_dir().join(format!("taguru-http-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data_dir);
        Self::spawn(tag, data_dir, extra_env)
    }

    fn start_on(tag: &str, data_dir: PathBuf) -> Self {
        Self::spawn(tag, data_dir, &[])
    }

    fn spawn(tag: &str, data_dir: PathBuf, extra_env: &[(&str, &str)]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
        command
            .env("TAGURU_ADDR", "127.0.0.1:0")
            .env("TAGURU_DATA_DIR", &data_dir)
            .env("TAGURU_FLUSH_SECS", "1")
            .env_remove("TAGURU_EMBED_URL") // lexical-only, hermetic
            .env_remove("TAGURU_API_TOKEN"); // unauthenticated unless a test opts in
        for (key, value) in extra_env {
            command.env(key, value);
        }
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("server binary must spawn");

        // The server prints its resolved address; read until it appears.
        let stdout = child.stdout.take().expect("stdout must be piped");
        let mut lines = BufReader::new(stdout).lines();
        let base = loop {
            let line = lines
                .next()
                .unwrap_or_else(|| panic!("server '{tag}' exited before listening"))
                .expect("server stdout must be readable");
            if let Some(addr) = line.strip_prefix("listening on ") {
                break format!("http://{addr}");
            }
        };
        // Keep draining stdout so the server never blocks on a full pipe.
        std::thread::spawn(move || for _ in lines {});

        Self {
            child,
            base,
            data_dir,
        }
    }

    /// One request; returns (status, parsed body). Non-JSON bodies come
    /// back as JSON strings.
    fn call(&self, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
        self.call_with_token(method, path, body, None)
    }

    /// [`Server::call`] with an explicit bearer token attached.
    fn call_with_token(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
        token: Option<&str>,
    ) -> (u16, Value) {
        let mut request = ureq::AgentBuilder::new()
            .build()
            .request(method, &format!("{}{path}", self.base));
        if let Some(token) = token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        let response = match body {
            Some(body) => request
                .set("Content-Type", "application/json")
                .send_string(&body.to_string()),
            None => request.call(),
        };
        finish(response, method, path)
    }

    /// A raw request: the body goes out as-is, with a Content-Type only
    /// when one is given — for the header-omission cases the JSON
    /// helpers cannot express.
    fn call_raw(
        &self,
        method: &str,
        path: &str,
        body: Option<&str>,
        content_type: Option<&str>,
    ) -> (u16, Value) {
        let mut request = ureq::AgentBuilder::new()
            .build()
            .request(method, &format!("{}{path}", self.base));
        if let Some(content_type) = content_type {
            request = request.set("Content-Type", content_type);
        }
        let response = match body {
            Some(body) => request.send_string(body),
            None => request.call(),
        };
        finish(response, method, path)
    }

    fn ok(&self, method: &str, path: &str, body: Option<Value>) -> Value {
        let (status, parsed) = self.call(method, path, body);
        assert_eq!(status, 200, "{method} {path} -> {parsed}");
        parsed["result"].clone()
    }

    /// Graceful stop (SIGTERM), waiting for the shutdown flush.
    fn stop_gracefully(self) -> PathBuf {
        self.stop_with("-TERM")
    }

    /// Hard kill (SIGKILL): no shutdown flush, no cleanup — whatever
    /// durability the server claims must come from the disk alone.
    fn stop_hard(self) -> PathBuf {
        self.stop_with("-KILL")
    }

    fn stop_with(mut self, signal: &str) -> PathBuf {
        let pid = self.child.id().to_string();
        Command::new("kill")
            .args([signal, &pid])
            .status()
            .expect("kill must run");
        let _ = self.child.wait();
        let data_dir = self.data_dir.clone();
        // Drop must not re-kill or delete the directory we hand back.
        std::mem::forget(self);
        data_dir
    }
}

/// Shared response tail: status plus parsed JSON body (or the raw
/// text when it is not JSON).
fn finish(response: Result<ureq::Response, ureq::Error>, method: &str, path: &str) -> (u16, Value) {
    let (status, text) = match response {
        Ok(response) => {
            let status = response.status();
            (status, response.into_string().unwrap_or_default())
        }
        Err(ureq::Error::Status(status, response)) => {
            (status, response.into_string().unwrap_or_default())
        }
        Err(error) => panic!("request {method} {path} failed: {error}"),
    };
    let parsed = serde_json::from_str(&text).unwrap_or(Value::String(text));
    (status, parsed)
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.data_dir);
    }
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
    assert_eq!(water["matches"][0]["weight"], json!(2.0));
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
    assert_eq!(ranked[0]["association"]["label"], json!("杜氏"));
    assert_eq!(ranked[0]["path"], json!(["青嶺酒造"]));
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
    assert_eq!(hits[0]["source"], json!("第2段落"));
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

#[test]
fn bearer_token_gates_every_route_except_health_and_metrics() {
    let server = Server::start_with_env("auth", &[("TAGURU_API_TOKEN", "s3cret")]);

    // Liveness and the scrape answer with zero credentials.
    assert_eq!(server.call("GET", "/health", None).0, 200);
    assert_eq!(server.call("GET", "/metrics", None).0, 200);

    // Everything else refuses a missing or wrong token with the API's
    // own error shape, and accepts the right one.
    let (status, body) = server.call("GET", "/contexts", None);
    assert_eq!(status, 401);
    assert_eq!(body["status"], json!("error"));
    let (status, _) = server.call_with_token("GET", "/contexts", None, Some("wrong"));
    assert_eq!(status, 401);
    let (status, _) = server.call_with_token("GET", "/contexts", None, Some("s3cret"));
    assert_eq!(status, 200);

    // Writes are gated the same way.
    let (status, _) = server.call("PUT", "/contexts/sake", Some(json!({})));
    assert_eq!(status, 401);
    let (status, _) =
        server.call_with_token("PUT", "/contexts/sake", Some(json!({})), Some("s3cret"));
    assert_eq!(status, 200);
}

#[test]
fn graph_writes_survive_a_hard_kill() {
    // A one-hour flush interval: the periodic flusher provably cannot
    // have persisted anything before the SIGKILL lands — whatever
    // comes back after the restart came through the WAL.
    let server = Server::start_with_env("hardkill", &[("TAGURU_FLUSH_SECS", "3600")]);
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "第1段落"},
        ])),
    );

    let data_dir = server.stop_hard();
    let server = Server::start_on("hardkill2", data_dir);
    let recalled = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    assert_eq!(recalled["matches"][0]["object"], json!("1907年"));
}

#[test]
fn an_association_batch_over_the_cap_is_rejected_before_any_write() {
    let server = Server::start("batchcap");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    let batch: Vec<Value> = (0..10_001)
        .map(|i| json!({"subject": format!("s{i}"), "label": "l", "object": "o", "weight": 1.0}))
        .collect();
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(Value::Array(batch)),
    );
    assert_eq!(status, 400, "{body}");

    // The guard ran before the write lock: nothing was applied.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));
}

#[test]
fn an_insane_weight_is_rejected_before_any_write() {
    let server = Server::start("weightcap");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    // Finite but absurd: two of these would saturate an edge to
    // +Infinity, and a later retract would mint Inf − Inf = NaN — a
    // fact nothing can read or reset again.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "a", "label": "l", "object": "b", "weight": 1.0},
            {"subject": "青嶺酒造", "label": "生産量", "object": "無限", "weight": 1.0e300},
        ])),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("associations[1].weight"),
        "the message must point at the offending item: {body}"
    );
    // Refused whole, before the write lock: not even the sane first
    // item landed.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));

    // The documented boundary stays usable, negation included.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "a", "label": "l", "object": "b", "weight": 1.0e6},
            {"subject": "a", "label": "l2", "object": "b", "weight": -1.0e6},
        ])),
    );
}

#[test]
fn the_directory_pages_by_name_and_serves_single_contexts() {
    let server = Server::start("dirpage");
    for name in ["apple", "banana", "cherry"] {
        server.ok(
            "PUT",
            &format!("/contexts/{name}"),
            Some(json!({"description": name})),
        );
    }

    let page = server.ok("GET", "/contexts?limit=2", None);
    assert_eq!(page["total"], json!(3), "total names the full count");
    let names: Vec<&str> = page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context| context["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["apple", "banana"], "name order, first page");

    let page = server.ok("GET", "/contexts?limit=2&after=banana", None);
    assert_eq!(page["total"], json!(3));
    let names: Vec<&str> = page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context| context["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["cherry"], "keyset picks up after the cursor");

    let single = server.ok("GET", "/contexts/banana", None);
    assert_eq!(single["name"], json!("banana"));
    assert_eq!(single["description"], json!("banana"));
    let (status, body) = server.call("GET", "/contexts/nope", None);
    assert_eq!(status, 404);
    assert_eq!(body["status"], json!("error"));
}

#[test]
fn a_present_body_is_parsed_whatever_the_content_type_says() {
    let server = Server::start("rawbody");

    // requests.put(url, data=json.dumps(...)) territory: a JSON body
    // with no JSON Content-Type. The description must land — this
    // used to silently drop the body and create with every field
    // defaulted, under a 200.
    let (status, body) = server.call_raw(
        "PUT",
        "/contexts/sake",
        Some(r#"{"description":"青嶺酒造の記憶","pinned":true}"#),
        None,
    );
    assert_eq!(status, 200, "{body}");
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(
        directory["contexts"][0]["description"],
        json!("青嶺酒造の記憶")
    );
    assert_eq!(directory["contexts"][0]["pinned"], json!(true));

    // A present body that is not JSON is an error, never defaults.
    let (status, body) =
        server.call_raw("PUT", "/contexts/beer", Some("definitely not json"), None);
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["status"], json!("error"));

    // An absent body still means defaults — the documented shape.
    let (status, body) = server.call_raw("PUT", "/contexts/beer", None, None);
    assert_eq!(status, 200, "{body}");

    // The other optional-body endpoint follows the same contract.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some("also not json"),
        None,
    );
    assert_eq!(status, 400, "{body}");
    let (status, body) = server.call_raw("POST", "/contexts/sake/vocabulary/audit", None, None);
    assert_eq!(status, 200, "{body}");
}

#[test]
fn off_axis_errors_speak_the_api_error_shape_too() {
    let server = Server::start("errshape");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    // Unknown path → 404 in the error shape.
    let (status, body) = server.call("GET", "/contextz", None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert!(body["time"].is_number(), "{body}");

    // Known path, wrong method → 405 in the error shape.
    let (status, body) = server.call("DELETE", "/contexts/sake/recall", None);
    assert_eq!(status, 405, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");

    // Malformed JSON on a JSON-required endpoint → 400 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some("{not json"),
        Some("application/json"),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");

    // Wrong media type → 415 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some("cue=x"),
        Some("application/x-www-form-urlencoded"),
    );
    assert_eq!(status, 415, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");

    // Well-formed JSON of the wrong type → 422 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some(r#"{"cue": 42}"#),
        Some("application/json"),
    );
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
}

#[test]
fn oversized_names_are_rejected_at_every_write_boundary() {
    let server = Server::start("namecap");
    let long = "字".repeat(400); // 1200 bytes, over the 1024-byte name cap

    // A context name becomes a file stem (percent-encoded ×3): 64
    // bytes is the cap.
    let (status, body) = server.call(
        "PUT",
        &format!("/contexts/{}", "n".repeat(65)),
        Some(json!({})),
    );
    assert_eq!(status, 400, "{body}");

    // The description rides in every directory listing.
    let (status, body) = server.call(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "d".repeat(5000)})),
    );
    assert_eq!(status, 400, "{body}");

    server.ok("PUT", "/contexts/sake", Some(json!({})));
    let (status, body) = server.call(
        "PATCH",
        "/contexts/sake",
        Some(json!({"description": "d".repeat(5000)})),
    );
    assert_eq!(status, 400, "{body}");

    // Graph names: the top-concepts snapshot carries them into every
    // GET /contexts response, far outside the cache budget.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": long, "label": "l", "object": "o", "weight": 1.0}])),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("associations[0].subject"),
        "the message must point at the offending field: {body}"
    );

    // Aliases and passage source ids persist names too.
    let mut concepts = serde_json::Map::new();
    concepts.insert(long.clone(), json!("x"));
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": concepts, "labels": {}})),
    );
    assert_eq!(status, 400, "{body}");
    let mut passages = serde_json::Map::new();
    passages.insert(long.clone(), json!("原文"));
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": passages})),
    );
    assert_eq!(status, 400, "{body}");

    // Nothing landed anywhere.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));

    // The boundary itself stays usable.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{
            "subject": "s".repeat(1024), "label": "l", "object": "o", "weight": 1.0
        }])),
    );
}

#[test]
fn unreachable_from_pages_like_recall_and_query() {
    let server = Server::start("orphanpage");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0},
            // Three islands no walk from the origin can reach.
            {"subject": "x1", "label": "l", "object": "y1", "weight": 1.0},
            {"subject": "x2", "label": "l", "object": "y2", "weight": 1.0},
            {"subject": "x3", "label": "l", "object": "y3", "weight": 1.0},
        ])),
    );

    let audit = server.ok(
        "POST",
        "/contexts/sake/unreachable_from",
        Some(json!({"origins": ["青嶺酒造"], "limit": 2})),
    );
    assert_eq!(audit["total"], json!(3));
    assert_eq!(audit["matches"].as_array().unwrap().len(), 2);
}

#[test]
fn explore_without_max_depth_stops_at_the_server_ceiling() {
    let server = Server::start("depthcap");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    // A 15-hop chain: c0 → c1 → … → c15.
    let chain: Vec<Value> = (0..15)
        .map(|i| {
            json!({"subject": format!("c{i}"), "label": "next", "object": format!("c{}", i + 1), "weight": 1.0})
        })
        .collect();
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(Value::Array(chain)),
    );

    let walked = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["c0"]})),
    );
    let deepest = walked["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["distance"].as_u64().unwrap())
        .max()
        .unwrap();
    assert_eq!(deepest, 10, "omitted max_depth must stop at the ceiling");
}

#[test]
fn explore_pages_and_keeps_the_closest_past_the_limit() {
    let server = Server::start("explorepage");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    // A hub with four direct neighbours; one leads a hop further to a
    // heavy edge.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "hub", "label": "l", "object": "n1", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "n2", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "n3", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "n4", "weight": 1.0},
            {"subject": "n1", "label": "l", "object": "far", "weight": 9.0},
        ])),
    );

    let walked = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["hub"], "limit": 3})),
    );
    assert_eq!(walked["total"], json!(5));
    let matches = walked["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 3);
    // The cut keeps the closest structure, not the heaviest weight:
    // the distance-2 edge (weight 9.0) is what falls off.
    assert!(
        matches.iter().all(|r| r["distance"] == json!(1)),
        "{walked}"
    );
}

#[test]
fn the_wal_cap_env_refuses_writes_rather_than_growing_forever() {
    // Flushes effectively never run, so the log can only grow; a
    // 1-byte cap trips on the second write.
    let server = Server::start_with_env(
        "walcap",
        &[("TAGURU_WAL_MAX_BYTES", "1"), ("TAGURU_FLUSH_SECS", "3600")],
    );
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "a", "label": "l", "object": "b", "weight": 1.0}])),
    );
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "c", "label": "l", "object": "d", "weight": 1.0}])),
    );
    assert_eq!(status, 500, "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("write-ahead log"),
        "{body}"
    );
}

#[test]
fn a_bind_failure_exits_with_a_diagnosis_not_a_panic() {
    // Occupy a port, then ask the server to bind it.
    let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = holder.local_addr().unwrap().to_string();
    let data_dir =
        std::env::temp_dir().join(format!("taguru-http-bindfail-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);

    let output = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .env("TAGURU_ADDR", &addr)
        .env("TAGURU_DATA_DIR", &data_dir)
        .env_remove("TAGURU_EMBED_URL")
        .env_remove("TAGURU_API_TOKEN")
        .output()
        .expect("server binary must spawn");

    assert!(!output.status.success(), "a failed bind must exit nonzero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot bind"), "{stderr}");
    assert!(
        !stderr.contains("panicked"),
        "an operator mistake must not read as a crash: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&data_dir);
}

#[test]
#[cfg(unix)]
fn health_reports_503_while_flushes_fail_and_recovers_after() {
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant};

    let server = Server::start("health503");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    // The first write creates the WAL file while the directory is
    // still writable; afterwards appends only need the existing file.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "a", "label": "l", "object": "b", "weight": 1.0}])),
    );

    std::fs::set_permissions(&server.data_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    // Keep the context dirty each round so a tick between our calls
    // cannot leave the flusher idle-and-green.
    let deadline = Instant::now() + Duration::from_secs(10);
    let degraded = loop {
        let _ = server.call(
            "POST",
            "/contexts/sake/associations",
            Some(json!([{"subject": "a", "label": "l", "object": "b", "weight": 0.001}])),
        );
        let (status, body) = server.call("GET", "/health", None);
        if status == 503 {
            break body;
        }
        assert!(Instant::now() < deadline, "health never degraded");
        std::thread::sleep(Duration::from_millis(200));
    };
    assert_eq!(degraded["status"], json!("error"), "{degraded}");

    std::fs::set_permissions(&server.data_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let _ = server.call(
            "POST",
            "/contexts/sake/associations",
            Some(json!([{"subject": "a", "label": "l", "object": "b", "weight": 0.001}])),
        );
        let (status, _) = server.call("GET", "/health", None);
        if status == 200 {
            break;
        }
        assert!(Instant::now() < deadline, "health never recovered");
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[test]
fn a_body_over_the_configured_limit_is_rejected_with_413() {
    let server = Server::start_with_env("bodycap", &[("TAGURU_MAX_BODY_BYTES", "16")]);
    let (status, _) = server.call(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "この説明は16バイトよりずっと長い"})),
    );
    assert_eq!(status, 413);
}

#[test]
fn a_custom_request_timeout_does_not_disturb_fast_requests() {
    // The deadline actually firing is unit-tested in limits.rs; this
    // pins the wiring — a tight budget must not break normal traffic.
    let server = Server::start_with_env("timeout", &[("TAGURU_REQUEST_TIMEOUT_SECS", "1")]);
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    assert_eq!(server.call("GET", "/contexts", None).0, 200);
}

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
fn log_output_is_structured_when_json_format_is_requested() {
    let data_dir = std::env::temp_dir().join(format!("taguru-jsonlog-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let mut child = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &data_dir)
        .env("TAGURU_LOG_FORMAT", "json")
        .env_remove("TAGURU_EMBED_URL")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server binary must spawn");

    // The first stderr line is already a log record (boot logging runs
    // before the listener binds); it must be one JSON object with the
    // standard fields, not pretty-format text.
    let stderr = child.stderr.take().expect("stderr must be piped");
    let line = BufReader::new(stderr)
        .lines()
        .next()
        .expect("a log line must appear")
        .expect("server stderr must be readable");
    let parsed: Value =
        serde_json::from_str(&line).unwrap_or_else(|_| panic!("stderr is not JSON: {line}"));
    assert!(parsed["level"].is_string(), "{parsed}");

    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&data_dir);
}

#[test]
fn data_survives_a_graceful_restart() {
    let server = Server::start("restart");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "再起動テスト"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "第1段落"},
        ])),
    );

    // SIGTERM triggers the shutdown flush; the same data directory must
    // come back with the knowledge intact.
    let data_dir = server.stop_gracefully();
    let server = Server::start_on("restart2", data_dir);
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["name"], json!("sake"));
    assert_eq!(
        directory["contexts"][0]["description"],
        json!("再起動テスト")
    );
    let recalled = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    assert_eq!(recalled["matches"][0]["object"], json!("1907年"));
}
