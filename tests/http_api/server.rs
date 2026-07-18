//! Process-lifecycle behavior: crash/restart/hard-kill durability and startup diagnostics.

use std::process::Command;

use serde_json::json;

use crate::support::*;

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
fn post_flush_persists_dirty_contexts_on_demand() {
    // The periodic flusher is effectively off: the endpoint is the
    // only thing that can move the image.
    let server = Server::start_with_env("forceflush", &[("TAGURU_FLUSH_SECS", "3600")]);
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "a", "label": "l", "object": "b", "weight": 1.0}])),
    );

    let flushed = server.ok("POST", "/flush", None);
    assert_eq!(flushed, json!(["sake"]), "the dirty context must flush");
    // Nothing left dirty: an immediate second call is a no-op.
    assert_eq!(server.ok("POST", "/flush", None), json!([]));
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

    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .env("TAGURU_ADDR", &addr)
        .env("TAGURU_DATA_DIR", &data_dir);
    let output = command.output().expect("server binary must spawn");

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
