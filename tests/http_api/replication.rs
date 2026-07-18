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

    // A second instance pointed at the same bucket — the exact
    // misconfiguration fencing exists for. Its claim deposes the
    // first writer.
    let second = Server::start_with_env(
        "repl-usurper",
        &[
            ("TAGURU_REPLICATE_URL", &bucket_url(&bucket)),
            ("TAGURU_REPLICATE_INTERVAL_MS", "100"),
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
