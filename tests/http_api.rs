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
        let data_dir = std::env::temp_dir().join(format!("arag-http-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data_dir);
        Self::start_on(tag, data_dir)
    }

    fn start_on(tag: &str, data_dir: PathBuf) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_associative-rag"))
            .env("ARAG_ADDR", "127.0.0.1:0")
            .env("ARAG_DATA_DIR", &data_dir)
            .env("ARAG_FLUSH_SECS", "1")
            .env_remove("ARAG_EMBED_URL") // lexical-only, hermetic
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
        let request = ureq::AgentBuilder::new()
            .build()
            .request(method, &format!("{}{path}", self.base));
        let response = match body {
            Some(body) => request
                .set("Content-Type", "application/json")
                .send_string(&body.to_string()),
            None => request.call(),
        };
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

    fn ok(&self, method: &str, path: &str, body: Option<Value>) -> Value {
        let (status, parsed) = self.call(method, path, body);
        assert_eq!(status, 200, "{method} {path} -> {parsed}");
        parsed["result"].clone()
    }

    /// Graceful stop (SIGTERM), waiting for the shutdown flush.
    fn stop_gracefully(mut self) -> PathBuf {
        let pid = self.child.id().to_string();
        Command::new("kill")
            .args(["-TERM", &pid])
            .status()
            .expect("kill must run");
        let _ = self.child.wait();
        let data_dir = self.data_dir.clone();
        // Drop must not re-kill or delete the directory we hand back.
        std::mem::forget(self);
        data_dir
    }
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
    assert!(protocol.as_str().unwrap().contains("# AssociativeRAG"));
    assert_eq!(server.ok("GET", "/contexts", None), json!([]));

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
        walked
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
    assert_eq!(orphans, json!([]));
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
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory[0]["pinned"], json!(true));
    assert_eq!(directory[0]["semantic_floor"], json!(0.2));
    assert_eq!(directory[0]["stats"]["associations"], json!(5));
    let (status, _) = server.call("POST", "/contexts/sake/embeddings/refresh", None);
    assert_eq!(status, 501);

    // Deletion removes the context and its files.
    server.ok("DELETE", "/contexts/sake", None);
    assert_eq!(server.ok("GET", "/contexts", None), json!([]));
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
    assert_eq!(directory[0]["name"], json!("sake"));
    assert_eq!(directory[0]["description"], json!("再起動テスト"));
    let recalled = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    assert_eq!(recalled["matches"][0]["object"], json!("1907年"));
}
