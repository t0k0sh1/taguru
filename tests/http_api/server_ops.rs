//! Cross-cutting server operational behavior: health, body limits, request timeouts.

use serde_json::json;

use crate::support::*;

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
    let (status, body) = server.call(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "この説明は16バイトよりずっと長い"})),
    );
    assert_eq!(status, 413);
    // The cap breach speaks the one JSON error shape like every other
    // axis (it used to be axum's plain-text rejection).
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("payload_too_large"), "{body}");
}

#[test]
fn a_custom_request_timeout_does_not_disturb_fast_requests() {
    // The deadline actually firing is unit-tested in limits.rs; this
    // pins the wiring — a tight budget must not break normal traffic.
    let server = Server::start_with_env("timeout", &[("TAGURU_REQUEST_TIMEOUT_SECS", "1")]);
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    assert_eq!(server.call("GET", "/contexts", None).0, 200);
}

/// The wall-clock proof for the `block_in_place` deadline work: a
/// multi-batch import large enough that landing every batch takes far
/// longer than the configured budget must still answer in roughly the
/// budget's time, not in the time the whole loop would take to drain.
#[test]
fn a_tight_timeout_cuts_a_multi_batch_import_short_instead_of_running_it_to_completion() {
    const BATCH_COUNT: usize = 8_000;
    let server = Server::start_with_env("timeout-import", &[("TAGURU_REQUEST_TIMEOUT_SECS", "1")]);
    let mut stream = String::new();
    stream.push_str(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-0\", \
         \"create\": {\"description\": \"d\"}}\n",
    );
    stream
        .push_str("{\"subject\": \"s0\", \"label\": \"l\", \"object\": \"o0\", \"weight\": 1.0}\n");
    for i in 1..BATCH_COUNT {
        stream.push_str(&format!(
            "{{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-{i}\"}}\n"
        ));
        stream.push_str(&format!(
            "{{\"subject\": \"s{i}\", \"label\": \"l\", \"object\": \"o{i}\", \"weight\": 1.0}}\n"
        ));
    }

    let started = std::time::Instant::now();
    let (status, body) = post_import(&server, &stream, None);
    let elapsed = started.elapsed();

    assert_eq!(status, 408, "{body}");
    assert_eq!(body["code"], json!("timeout"), "{body}");
    // Each fsync-bearing batch costs roughly 10ms, so draining all
    // 8,000 would take over a minute; answering near the 1-second
    // budget instead is the point of this test.
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "took {elapsed:?} — the deadline check inside the batch loop should have cut \
         this short instead of letting every batch land first"
    );
}
