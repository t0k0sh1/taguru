//! Continuous-replication integration: a real server shipping to a
//! real (file://) bucket, restored by the real CLI, verified by the
//! real inspect — the round trip issue #127's acceptance names, minus
//! the cloud endpoints (the store client is the same code for all
//! four schemes; what differs per cloud is auth and the wire, which a
//! test without credentials cannot reach).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::json;

use crate::support::Server;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("taguru-repl-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn bucket_url(dir: &Path) -> String {
    format!("file://{}", dir.display())
}

/// Polls until `check` passes — shipping is asynchronous by design, so
/// the tests wait on observable bucket/metric state, never on sleeps
/// sized by hope alone.
fn wait_for(what: &str, mut check: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if check() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("timed out waiting for {what}");
}

/// Every file under `dir` (one level, files only), name → bytes.
fn dir_contents(dir: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .map(|entry| {
            (
                entry.file_name().into_string().unwrap(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

fn run_cli(args: &[&str], extra_env: &[(&str, &str)]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    crate::support::common::scrub_taguru_env(&mut command);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    command.args(args).output().expect("CLI must run")
}

#[test]
fn shipped_bucket_restores_to_an_equivalent_directory() {
    let bucket = scratch("roundtrip-bucket");
    let server = Server::start_with_env(
        "repl-roundtrip",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
        ],
    );

    // A context exercising both log lanes and the alias table…
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の知識"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0, "source": "第2段落"},
            {"subject": "高瀬", "label": "出身", "object": "南部杜氏", "weight": 1.0, "source": "第3段落"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "第2段落": "青嶺酒造は、仕込み水に雲居山の伏流水を使う。杜氏は高瀬である。",
        }})),
    );
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"あおみね": "青嶺酒造"}})),
    );
    // …a group, and a stand-in for the grant store (the shipper moves
    // every published file the same way; a live OAuth flow would only
    // change who wrote the bytes).
    server.ok(
        "PUT",
        "/groups/breweries",
        Some(json!({"contexts": ["sake"]})),
    );
    std::fs::write(server.data_dir.join("oauth.json"), b"{\"grants\":[]}").unwrap();

    // The bucket catches up: baseline complete, and the graph lane's
    // first segment present.
    wait_for("the baseline to complete", || {
        bucket
            .join("gen-00000000000000000001")
            .join("complete")
            .exists()
    });

    // Writes that land AFTER the baseline ride the tail path.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "第1段落"},
        ])),
    );

    // A graceful stop flushes, then drains the shipper — the bucket
    // must end as current as the disk.
    let data_dir = server.stop_gracefully();

    // Restore with the real CLI…
    let restored = scratch("roundtrip-restored");
    let restore = run_cli(
        &[
            "restore",
            "--out",
            &restored.display().to_string(),
            &bucket_url(&bucket),
        ],
        &[],
    );
    assert!(
        restore.status.success(),
        "restore failed: {}",
        String::from_utf8_lossy(&restore.stderr)
    );

    // …verify it with the real inspect…
    let inspect = run_cli(&["inspect", &restored.display().to_string()], &[]);
    assert!(
        inspect.status.success(),
        "inspect refused the restored directory: {}\n{}",
        String::from_utf8_lossy(&inspect.stdout),
        String::from_utf8_lossy(&inspect.stderr)
    );

    // …and prove equivalence where it is defined: the export streams
    // (graph + passages + groups) of source and restore are identical.
    let export_source = scratch("roundtrip-export-source");
    let out = run_cli(
        &["export", "--out", &export_source.display().to_string()],
        &[("TAGURU_DATA_DIR", &data_dir.display().to_string())],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let export_restored = scratch("roundtrip-export-restored");
    let out = run_cli(
        &["export", "--out", &export_restored.display().to_string()],
        &[("TAGURU_DATA_DIR", &restored.display().to_string())],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        dir_contents(&export_source),
        dir_contents(&export_restored),
        "the restored directory must export byte-identical streams"
    );

    // The grant store rode along, owner-only like the server writes it.
    let grants = restored.join("oauth.json");
    assert_eq!(std::fs::read(&grants).unwrap(), b"{\"grants\":[]}");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&grants).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "the restored grant store must stay owner-only");
    }

    // Restore refuses to layer onto a non-empty directory.
    let refused = run_cli(
        &[
            "restore",
            "--out",
            &restored.display().to_string(),
            &bucket_url(&bucket),
        ],
        &[],
    );
    assert!(!refused.status.success());
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("not empty"),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );

    for dir in [bucket, data_dir, restored, export_source, export_restored] {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn a_second_writer_fences_the_first_which_fail_stops_but_keeps_serving() {
    let bucket = scratch("fence-bucket");
    let first = Server::start_with_env(
        "repl-fenced",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
        ],
    );
    first.ok("PUT", "/contexts/sake", Some(json!({})));
    wait_for("the first writer's baseline", || {
        bucket
            .join("gen-00000000000000000001")
            .join("complete")
            .exists()
    });

    // A second instance pointed at the same bucket. Since issue #128
    // deposing a live writer takes stated intent (the takeover guard
    // has its own test below); past the guard, fencing works exactly
    // as before.
    let second = Server::start_with_env(
        "repl-usurper",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
            ("TAGURU_TAKEOVER", "1"),
        ],
    );
    wait_for("the second writer's claim", || {
        bucket.join("fence").join("00000000000000000002").exists()
    });

    // The first writer discovers its deposition on its next DIRTY
    // cycle — give it one.
    first.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "a", "label": "b", "object": "c", "weight": 1.0},
        ])),
    );
    wait_for("the fenced metric to latch", || {
        let (status, metrics) = first.call("GET", "/metrics", None);
        assert_eq!(status, 200);
        metrics
            .as_str()
            .unwrap_or_default()
            .contains("taguru_replication_fenced 1")
    });

    // Fail-stop means the SHIPPER stopped — the serve path keeps
    // answering from its local truth.
    let page = first.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "a", "limit": 3})),
    );
    assert_eq!(page["total"], json!(1));

    // And nothing the fenced writer did after deposition reached the
    // successor's namespace.
    assert!(
        !bucket
            .join("gen-00000000000000000002")
            .join("files")
            .join("sake.wal.jsonl")
            .exists(),
        "a fenced writer must never write into its successor's generation"
    );

    let _ = std::fs::remove_dir_all(first.stop_gracefully());
    let _ = std::fs::remove_dir_all(second.stop_gracefully());
    let _ = std::fs::remove_dir_all(bucket);
}

#[test]
fn an_empty_disk_boots_from_the_bucket_and_serves_the_lineage() {
    let bucket = scratch("bootstrap-bucket");
    let first = Server::start_with_env(
        "repl-boot-source",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
        ],
    );

    // Two contexts — one pinned (the eager half of hydration), one
    // not (the lazy half) — plus passages and a group, so every file
    // family crosses the bucket.
    first.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の知識"})),
    );
    first.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0, "source": "第2段落"},
        ])),
    );
    first.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "第2段落": "青嶺酒造は、仕込み水に雲居山の伏流水を使う。杜氏は高瀬である。",
        }})),
    );
    first.ok(
        "PUT",
        "/contexts/glossary",
        Some(json!({"description": "用語集"})),
    );
    first.ok("PATCH", "/contexts/glossary", Some(json!({"pinned": true})));
    first.ok(
        "POST",
        "/contexts/glossary/associations",
        Some(json!([
            {"subject": "杜氏", "label": "意味", "object": "醸造責任者", "weight": 1.0},
        ])),
    );
    first.ok(
        "PUT",
        "/groups/breweries",
        Some(json!({"contexts": ["sake"]})),
    );
    wait_for("the source baseline", || {
        bucket
            .join("gen-00000000000000000001")
            .join("complete")
            .exists()
    });
    let first_dir = first.stop_gracefully();
    assert!(
        bucket
            .join("gen-00000000000000000001")
            .join("retired")
            .exists(),
        "a graceful stop retires its generation"
    );

    // An EMPTY directory against the same bucket: the successor
    // hydrates the lineage instead of starting blank. The retired
    // marker is why no takeover acknowledgment is needed.
    let second = Server::start_with_env(
        "repl-boot-successor",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
        ],
    );

    // Readiness semantics: the port opened, so the pinned context is
    // already local (preload hydrates it eagerly, before binding).
    assert!(
        second.data_dir.join("glossary.ctx").exists(),
        "a pinned context hydrates before the port opens"
    );

    // The whole directory is enumerable — including the lazy context.
    let page = second.ok("GET", "/contexts", None);
    let names: Vec<&str> = page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["glossary", "sake"], "{page}");

    // First touch serves the hydrated truth: recall and passages both.
    let recall = second.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造", "limit": 5})),
    );
    assert_eq!(recall["total"], json!(1), "{recall}");
    let hits = second.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "伏流水", "limit": 3})),
    );
    assert!(!hits.as_array().unwrap().is_empty(), "{hits}");
    assert_eq!(hits[0]["source"], "第2段落", "{hits}");

    // The successor owns the lineage: once every family settles, its
    // own generation completes — the manifest gate in action.
    wait_for("the successor's own complete generation", || {
        bucket
            .join("gen-00000000000000000002")
            .join("complete")
            .exists()
    });
    let second_dir = second.stop_gracefully();

    // Equivalence where it is defined: both directories export
    // byte-identical batch streams.
    let export_first = scratch("bootstrap-export-first");
    let out = run_cli(
        &["export", "--out", &export_first.display().to_string()],
        &[("TAGURU_DATA_DIR", &first_dir.display().to_string())],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let export_second = scratch("bootstrap-export-second");
    let out = run_cli(
        &["export", "--out", &export_second.display().to_string()],
        &[("TAGURU_DATA_DIR", &second_dir.display().to_string())],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        dir_contents(&export_first),
        dir_contents(&export_second),
        "a bucket boot must serve exactly the lineage it inherited"
    );

    for dir in [bucket, first_dir, second_dir, export_first, export_second] {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn deposing_a_live_writer_needs_stated_intent_but_a_retired_one_does_not() {
    let bucket = scratch("intent-bucket");
    let first = Server::start_with_env(
        "repl-intent-holder",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
        ],
    );
    first.ok("PUT", "/contexts/sake", Some(json!({})));
    wait_for("the holder's baseline", || {
        bucket
            .join("gen-00000000000000000001")
            .join("complete")
            .exists()
    });

    // While the first writer lives (fresh heartbeat, no retired
    // marker), an empty-disk boot against its bucket refuses fast and
    // names the acknowledgment it wants.
    let refused_dir = scratch("intent-refused");
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    crate::support::common::scrub_taguru_env(&mut command)
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &refused_dir)
        .env("TAGURU_REPLICATE_URL", bucket_url(&bucket))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let mut child = command.spawn().expect("server binary must spawn");
    let deadline = Instant::now() + Duration::from_secs(15);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("the takeover guard must refuse promptly, not boot");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut stderr = String::new();
    use std::io::Read as _;
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(!status.success(), "the boot must be refused");
    assert!(stderr.contains("take-over"), "{stderr}");

    // A clean stop retires the generation, and the same empty-disk
    // boot proceeds with no acknowledgment at all.
    let first_dir = first.stop_gracefully();
    let second = Server::start_with_env(
        "repl-intent-successor",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
        ],
    );
    let page = second.ok("GET", "/contexts", None);
    assert_eq!(
        page["contexts"].as_array().unwrap().len(),
        1,
        "the retired lineage hydrates freely: {page}"
    );

    for dir in [bucket, refused_dir, first_dir, second.stop_gracefully()] {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The /metrics body as text (it is not JSON; the harness hands it
/// back as a JSON string).
fn metrics_text(server: &Server) -> String {
    let (status, body) = server.call("GET", "/metrics", None);
    assert_eq!(status, 200);
    body.as_str().expect("metrics is text").to_string()
}

/// The names GET /contexts currently answers with.
fn context_names(server: &Server) -> Vec<String> {
    let page = server.ok("GET", "/contexts", None);
    page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["name"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn a_replica_serves_reads_tails_the_writer_and_refuses_writes() {
    let bucket = scratch("reader-bucket");
    let writer = Server::start_with_env(
        "repl-reader-writer",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
        ],
    );

    // The same spread the bucket-boot test ships: a pinned context
    // (the eager half), a lazy one with passages, and a group.
    writer.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の知識"})),
    );
    writer.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0, "source": "第2段落"},
        ])),
    );
    writer.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "第2段落": "青嶺酒造は、仕込み水に雲居山の伏流水を使う。杜氏は高瀬である。",
        }})),
    );
    writer.ok(
        "PUT",
        "/contexts/glossary",
        Some(json!({"description": "用語集"})),
    );
    writer.ok("PATCH", "/contexts/glossary", Some(json!({"pinned": true})));
    writer.ok(
        "PUT",
        "/groups/breweries",
        Some(json!({"contexts": ["sake"]})),
    );
    // The writer keeps running (unlike the bucket-boot test's graceful
    // stop), so "complete exists" alone could be an EARLIER cycle's
    // manifest: wait until it carries the whole spread — the pinned
    // meta's current version rode the same (or an earlier) cycle as
    // the group file, since every cycle ships all changed files.
    wait_for("the writer to ship the whole spread", || {
        std::fs::read_to_string(bucket.join("gen-00000000000000000001").join("complete"))
            .map(|manifest| {
                manifest.contains("breweries.group") && manifest.contains("glossary.ctx")
            })
            .unwrap_or(false)
    });

    // A replica against the same bucket, on an empty directory: it
    // hydrates like a bucket boot (pinned before the port opens),
    // then keeps tailing.
    let replica = Server::start_with_env(
        "repl-reader-replica",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
            ("TAGURU_REPLICA", "1"),
            ("TAGURU_WRITER_URL", "http://writer.internal:8248"),
        ],
    );
    assert!(
        replica.data_dir.join("glossary.ctx").exists(),
        "a pinned context hydrates before the replica's port opens"
    );
    assert_eq!(context_names(&replica), ["glossary", "sake"]);

    // Every read verb family: graph recall, passage search, the group
    // directory — all serve the lineage.
    let recall = replica.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造", "limit": 5})),
    );
    assert_eq!(recall["total"], json!(1), "{recall}");
    let hits = replica.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "伏流水", "limit": 3})),
    );
    assert_eq!(hits[0]["source"], "第2段落", "{hits}");
    let groups = replica.ok("GET", "/groups", None);
    assert_eq!(groups["groups"][0]["name"], "breweries", "{groups}");

    // The scrape carries the replica shape from boot; the generation
    // gauge lands with the tailer's first poll.
    assert!(
        metrics_text(&replica).contains("taguru_replica 1"),
        "the role gauge is a boot fact"
    );
    wait_for("the generation gauge to land", || {
        metrics_text(&replica).contains("taguru_replica_generation 1")
    });

    // Writes refuse crisply, naming the writer — the ingest loop, the
    // operator verbs, and a dispatched MCP write tool alike. Read
    // tools on /mcp work unchanged.
    for (method, path, body) in [
        ("PUT", "/contexts/fresh", Some(json!({}))),
        (
            "POST",
            "/contexts/sake/associations",
            Some(json!([{"subject": "a", "label": "l", "object": "o", "weight": 1.0}])),
        ),
        ("DELETE", "/contexts/sake", None),
        ("POST", "/flush", None),
        ("POST", "/contexts/sake/compact", None),
    ] {
        let (status, answer) = replica.call(method, path, body);
        assert_eq!(status, 403, "{method} {path} -> {answer}");
        assert_eq!(answer["code"], "read_only_replica", "{answer}");
        assert!(
            answer["error"]
                .as_str()
                .unwrap()
                .contains("http://writer.internal:8248"),
            "the refusal must name the writer: {answer}"
        );
    }
    // The OAuth grant surface is merged onto the router AFTER
    // routes(), so it exercises the OUTER copy of the gate — minting
    // credentials is a write, and it refuses like one. OAuth needs an
    // API key to boot, so a dedicated replica carries both (its
    // /oauth/register is bearer-exempt by design: the refusal below
    // is the replica gate's, not auth's).
    let oauth_replica = Server::start_with_env(
        "repl-reader-oauth",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
            ("TAGURU_REPLICA", "1"),
            ("TAGURU_API_TOKENS", "ops:sesame"),
            ("TAGURU_PUBLIC_URL", "http://replica.internal:8248"),
        ],
    );
    let (status, answer) = oauth_replica.call(
        "POST",
        "/oauth/register",
        Some(json!({"client_name": "probe", "redirect_uris": ["http://127.0.0.1/cb"]})),
    );
    assert_eq!(status, 403, "{answer}");
    assert_eq!(answer["code"], "read_only_replica", "{answer}");
    drop(oauth_replica);

    let read_tool = replica.call_tool(1, "recall", json!({"context": "sake", "cue": "青嶺酒造"}));
    assert_ne!(read_tool["isError"], json!(true), "{read_tool}");
    let write_tool = replica.call_tool(
        2,
        "add_associations",
        json!({"context": "sake", "associations": [
            {"subject": "a", "label": "l", "object": "o", "weight": 1.0}
        ]}),
    );
    assert_eq!(write_tool["isError"], json!(true), "{write_tool}");
    assert!(
        write_tool["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("read replica"),
        "{write_tool}"
    );

    // Live tailing: the writer moves — a new fact, a new context —
    // and the replica follows within shipping lag + poll interval.
    writer.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "銘柄", "object": "雲居", "weight": 1.0},
        ])),
    );
    writer.ok(
        "PUT",
        "/contexts/news",
        Some(json!({"description": "新着"})),
    );
    wait_for("the replica to tail the new fact", || {
        let recall = replica.ok(
            "POST",
            "/contexts/sake/recall",
            Some(json!({"cue": "青嶺酒造", "limit": 5})),
        );
        recall["total"] == json!(2)
    });
    wait_for("the replica to tail the new context", || {
        context_names(&replica).contains(&"news".to_string())
    });

    // The lag arithmetic drains to zero on display.
    wait_for("the lag metrics to drain", || {
        let scrape = metrics_text(&replica);
        scrape.contains("taguru_replica_behind_seconds{context=\"sake\",lane=\"graph\"} 0")
    });

    // A deletion propagates: the context leaves the directory AND its
    // files leave the replica's disk.
    writer.ok("DELETE", "/contexts/news", None);
    wait_for("the deletion to propagate", || {
        !context_names(&replica).contains(&"news".to_string())
    });
    wait_for("the deleted family to leave the replica's disk", || {
        !replica.data_dir.join("news.ctx").exists()
            && !replica.data_dir.join("news.meta.json").exists()
    });

    let writer_dir = writer.stop_gracefully();
    let replica_dir = replica.stop_gracefully();
    for dir in [bucket, writer_dir, replica_dir] {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn promotion_rehearsal_the_standby_drains_flips_and_the_pool_follows() {
    let bucket = scratch("promote-bucket");
    let writer = Server::start_with_env(
        "repl-promote-writer",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
        ],
    );
    writer.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の知識"})),
    );
    writer.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0},
        ])),
    );
    // The very first complete can predate the seed (an empty writer's
    // first cycle manifests an empty directory): wait until the
    // manifest actually carries the context.
    wait_for("the writer to ship the seed", || {
        std::fs::read_to_string(bucket.join("gen-00000000000000000001").join("complete"))
            .map(|manifest| manifest.contains("sake.ctx"))
            .unwrap_or(false)
    });

    // Two replicas: the standby that will be promoted, and a pool
    // member that must ride through the promotion.
    let standby_env = [
        ("TAGURU_REPLICATE_URL", bucket_url(&bucket)),
        ("TAGURU_REPLICATE_INTERVAL_MS", "100".to_string()),
        ("TAGURU_REPLICA", "1".to_string()),
    ];
    let standby_env: Vec<(&str, &str)> = standby_env
        .iter()
        .map(|(key, value)| (*key, value.as_str()))
        .collect();
    let standby = Server::start_with_env("repl-promote-standby", &standby_env);
    let pool = Server::start_with_env("repl-promote-pool", &standby_env);

    // The writer advances once more; both replicas tail it — the
    // steady state a promotion starts from.
    writer.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "銘柄", "object": "雲居", "weight": 1.0},
        ])),
    );
    for (name, replica) in [("standby", &standby), ("pool", &pool)] {
        wait_for(&format!("the {name} to reach the writer's tip"), || {
            let recall = replica.ok(
                "POST",
                "/contexts/sake/recall",
                Some(json!({"cue": "青嶺酒造", "limit": 5})),
            );
            recall["total"] == json!(2)
        });
    }

    // Runbook step 1 — the writer dies, uncleanly (no retired marker,
    // heartbeat fresh: the worst case the guard exists for).
    let writer_dir = writer.stop_hard();

    // Step 2 — drain: the standby's lag reads zero against the bucket.
    wait_for("the standby to drain the shipped tail", || {
        let scrape = metrics_text(&standby);
        scrape.contains("taguru_replica_behind_seconds{context=\"sake\",lane=\"graph\"} 0")
    });

    // Step 3+4 — flip: stop the standby, restart its directory as the
    // writer. The crashed predecessor is why bare `serve` refuses and
    // asks for the acknowledgment…
    let standby_dir = standby.stop_gracefully();
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    crate::support::common::scrub_taguru_env(&mut command)
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &standby_dir)
        .env("TAGURU_REPLICATE_URL", bucket_url(&bucket))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let mut child = command.spawn().expect("server binary must spawn");
    let deadline = Instant::now() + Duration::from_secs(15);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("promoting over a crashed writer must demand stated intent");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut stderr = String::new();
    use std::io::Read as _;
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(!status.success());
    assert!(stderr.contains("take-over"), "{stderr}");

    // …and with it, the standby's cache becomes the writer: it claims
    // the next generation (fencing the dead predecessor wherever it
    // is) and takes writes.
    let promoted = Server::start_on_with_env(
        "repl-promote-promoted",
        standby_dir,
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
            ("TAGURU_TAKEOVER", "1"),
        ],
    );
    promoted.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "创業", "object": "1897年", "weight": 1.0},
        ])),
    );
    let recall = promoted.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造", "limit": 5})),
    );
    assert_eq!(
        recall["total"],
        json!(3),
        "the promoted writer serves the whole history plus the new write: {recall}"
    );
    wait_for("the promoted writer's own complete generation", || {
        bucket
            .join("gen-00000000000000000002")
            .join("complete")
            .exists()
    });

    // The pool replica re-aims at the new generation live — no
    // restart — and serves the post-promotion write, while still
    // refusing writes itself.
    wait_for("the pool replica to follow the promoted lineage", || {
        let recall = pool.ok(
            "POST",
            "/contexts/sake/recall",
            Some(json!({"cue": "青嶺酒造", "limit": 5})),
        );
        recall["total"] == json!(3)
    });
    wait_for("the pool replica's generation gauge to advance", || {
        metrics_text(&pool).contains("taguru_replica_generation 2")
    });
    let (status, answer) = pool.call("PUT", "/contexts/fresh", Some(json!({})));
    assert_eq!(status, 403, "{answer}");
    assert_eq!(answer["code"], "read_only_replica", "{answer}");

    let promoted_dir = promoted.stop_gracefully();
    let pool_dir = pool.stop_gracefully();
    for dir in [bucket, writer_dir, promoted_dir, pool_dir] {
        let _ = std::fs::remove_dir_all(dir);
    }
}
