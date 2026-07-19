//! Key rotation without a restart (issue #134): SIGHUP and the
//! `--config` file watch swap `TAGURU_API_TOKEN(S)` /
//! `TAGURU_KEY_SCOPES` on the running binary — fail closed (a broken
//! edit keeps the previous table, and a reload can never disarm
//! auth), audited by name only, with in-flight requests seeing the
//! old table or the new one and never a dropped response.
//!
//! The config-file watch polls every ~5s, so it may fire alongside
//! any SIGHUP these tests send after a rewrite; counter assertions
//! are therefore `>=`, never `==`. Every spawn disables the
//! per-source-IP failed-auth throttle: these tests poll and hammer
//! with not-yet-armed (and just-retired) tokens by design, and the
//! throttle's 429s would otherwise drown the 401-vs-200 verdicts
//! under scrutiny.

use std::time::{Duration, Instant};

use crate::support::Server;

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("taguru-reload-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Polls until `check` passes or the budget elapses — reloads are
/// asynchronous with respect to the signal / file write that asks for
/// them.
fn eventually(budget: Duration, what: &str, mut check: impl FnMut() -> bool) {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if check() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("timed out waiting for {what}");
}

/// One counter's value out of a /metrics render.
fn counter(metrics_text: &str, name: &str) -> u64 {
    metrics_text
        .lines()
        .find_map(|line| {
            line.strip_prefix(name)
                .and_then(|rest| rest.strip_prefix(' '))
        })
        .unwrap_or_else(|| panic!("{name} not rendered"))
        .parse()
        .unwrap()
}

/// SIGHUP applies a rewritten config: the rotated key's NEW bytes
/// authenticate, the removed key and the old bytes die, the reloaded
/// scope demotes the key live, and the audit line carries names —
/// never token bytes.
#[cfg(unix)]
#[test]
fn sighup_rotates_keys_and_scopes_and_audits_names_only() {
    let dir = scratch("sighup");
    let config = dir.join("taguru.env");
    let stderr = dir.join("stderr.log");
    std::fs::write(
        &config,
        "TAGURU_API_TOKENS=ci:sekrit-old,laptop:sekrit-laptop\n",
    )
    .unwrap();
    let server = Server::start_with_config(
        "reload-sighup",
        &config,
        &stderr,
        &[("TAGURU_AUTH_FAIL_LIMIT_PER_MIN", "0")],
    );

    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-old"))
            .0,
        200
    );
    assert_eq!(server.call("GET", "/contexts", None).0, 401);

    // Rotate ci's bytes, drop laptop, and demote ci to read-only.
    std::fs::write(
        &config,
        "TAGURU_API_TOKENS=ci:sekrit-new\nTAGURU_KEY_SCOPES={\"ci\": \"read\"}\n",
    )
    .unwrap();
    server.signal("-HUP");
    eventually(
        Duration::from_secs(10),
        "the rotated key to authenticate",
        || {
            server
                .call_with_token("GET", "/contexts", None, Some("sekrit-new"))
                .0
                == 200
        },
    );
    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-old"))
            .0,
        401
    );
    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-laptop"))
            .0,
        401
    );
    // The reloaded scope binds immediately: read works above, write
    // is now beyond ci's grant.
    let (refused, body) = server.call_with_token(
        "PUT",
        "/contexts/rotated",
        Some(serde_json::json!({})),
        Some("sekrit-new"),
    );
    assert_eq!(refused, 403, "{body}");

    // The audit line: on the audit target, trigger and name lists
    // spelled out, and not one token byte anywhere in the log.
    let log = std::fs::read_to_string(&stderr).unwrap();
    let audit = log
        .lines()
        .find(|line| line.contains("keyring reloaded"))
        .unwrap_or_else(|| panic!("no audit line in:\n{log}"));
    assert!(audit.contains("taguru::audit"), "{audit}");
    assert!(audit.contains("trigger=\"sighup\""), "{audit}");
    assert!(audit.contains("removed=[\"laptop\"]"), "{audit}");
    assert!(audit.contains("rotated=[\"ci\"]"), "{audit}");
    assert!(audit.contains("rescoped=[\"ci\"]"), "{audit}");
    assert!(
        !log.contains("sekrit"),
        "token bytes must never reach the log:\n{log}"
    );

    let (_, metrics) = server.call("GET", "/metrics", None);
    let text = metrics.as_str().unwrap();
    assert!(counter(text, "taguru_keyring_reloads_total") >= 1, "{text}");
    assert_eq!(counter(text, "taguru_keyring_reload_refusals_total"), 0);
}

/// Every fail-closed path over the wire: a malformed rewrite and a
/// rewrite that would disarm authentication both keep the previous
/// table working — the old key keeps answering, anonymous stays
/// refused — and each refusal counts.
#[cfg(unix)]
#[test]
fn a_broken_reload_keeps_the_previous_table_armed() {
    let dir = scratch("failclosed");
    let config = dir.join("taguru.env");
    let stderr = dir.join("stderr.log");
    std::fs::write(&config, "TAGURU_API_TOKENS=ci:sekrit-keep\n").unwrap();
    let server = Server::start_with_config(
        "reload-failclosed",
        &config,
        &stderr,
        &[("TAGURU_AUTH_FAIL_LIMIT_PER_MIN", "0")],
    );
    let refusals = || {
        std::fs::read_to_string(&stderr)
            .unwrap()
            .matches("keyring reload refused")
            .count()
    };

    // A malformed table: refused, nothing changes.
    std::fs::write(&config, "TAGURU_API_TOKENS=token-pasted-alone\n").unwrap();
    server.signal("-HUP");
    eventually(
        Duration::from_secs(10),
        "the malformed-table refusal",
        || refusals() >= 1,
    );
    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-keep"))
            .0,
        200
    );

    // A table with no keys at all while auth is armed: the one
    // transition a reload must never perform.
    std::fs::write(&config, "# rotated away every key by accident\n").unwrap();
    let already = refusals();
    server.signal("-HUP");
    eventually(Duration::from_secs(10), "the would-disable refusal", || {
        refusals() > already
    });
    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-keep"))
            .0,
        200
    );
    assert_eq!(server.call("GET", "/contexts", None).0, 401);

    let (_, metrics) = server.call("GET", "/metrics", None);
    let text = metrics.as_str().unwrap();
    assert!(
        counter(text, "taguru_keyring_reload_refusals_total") >= 2,
        "{text}"
    );
}

/// The `--config` watch alone — no signal anywhere — picks up a
/// rotated file within its poll interval: the Kubernetes
/// secret-volume flow, where nothing can send SIGHUP into the pod.
#[test]
fn the_config_watch_picks_up_a_rotation_without_a_signal() {
    let dir = scratch("watch");
    let config = dir.join("taguru.env");
    let stderr = dir.join("stderr.log");
    std::fs::write(&config, "TAGURU_API_TOKENS=ci:sekrit-w1\n").unwrap();
    let server = Server::start_with_config(
        "reload-watch",
        &config,
        &stderr,
        &[("TAGURU_AUTH_FAIL_LIMIT_PER_MIN", "0")],
    );

    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-w1"))
            .0,
        200
    );

    // A different LENGTH too, so even a filesystem with whole-second
    // mtimes cannot make this rewrite invisible to the (mtime, len)
    // probe.
    std::fs::write(&config, "TAGURU_API_TOKENS=ci:sekrit-w2-rotated\n").unwrap();
    eventually(
        Duration::from_secs(20),
        "the watch to apply the rotation",
        || {
            server
                .call_with_token("GET", "/contexts", None, Some("sekrit-w2-rotated"))
                .0
                == 200
        },
    );
    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-w1"))
            .0,
        401
    );
    let log = std::fs::read_to_string(&stderr).unwrap();
    assert!(log.contains("trigger=\"config-watch\""), "{log}");
}

/// A variable the SHELL set keeps beating the file across reloads —
/// the same precedence boot applies. The file rotates underneath; the
/// armed table must not move, and the reload says "no change".
#[cfg(unix)]
#[test]
fn a_shell_set_variable_keeps_winning_over_the_file_across_reloads() {
    let dir = scratch("shellwins");
    let config = dir.join("taguru.env");
    let stderr = dir.join("stderr.log");
    std::fs::write(&config, "TAGURU_API_TOKENS=ci:sekrit-file\n").unwrap();
    let server = Server::start_with_config(
        "reload-shellwins",
        &config,
        &stderr,
        &[
            ("TAGURU_API_TOKENS", "ci:sekrit-shell"),
            ("TAGURU_AUTH_FAIL_LIMIT_PER_MIN", "0"),
        ],
    );

    // Boot precedence: the real environment won.
    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-shell"))
            .0,
        200
    );
    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-file"))
            .0,
        401
    );

    std::fs::write(&config, "TAGURU_API_TOKENS=ci:sekrit-file-rotated\n").unwrap();
    server.signal("-HUP");
    eventually(Duration::from_secs(10), "the no-change reload", || {
        std::fs::read_to_string(&stderr)
            .unwrap()
            .contains("keyring reload: no change")
    });
    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-shell"))
            .0,
        200
    );
    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-file-rotated"))
            .0,
        401
    );
}

/// The acceptance race: requests hammering the gate straight through
/// a rotation are never dropped and never see a torn table — every
/// response is a definitive 200 or 401, the retiring key's thread
/// flips 200→401 exactly once, and the incoming key's flips 401→200
/// exactly once.
#[cfg(unix)]
#[test]
fn a_reload_never_drops_or_mixes_answers_for_racing_requests() {
    let dir = scratch("race");
    let config = dir.join("taguru.env");
    let stderr = dir.join("stderr.log");
    std::fs::write(&config, "TAGURU_API_TOKENS=ci:sekrit-before\n").unwrap();
    let server = Server::start_with_config(
        "reload-race",
        &config,
        &stderr,
        &[("TAGURU_AUTH_FAIL_LIMIT_PER_MIN", "0")],
    );
    assert_eq!(
        server
            .call_with_token("GET", "/contexts", None, Some("sekrit-before"))
            .0,
        200
    );

    // A dropped connection panics the sender — "old or new, never
    // none" includes "never a transport error".
    fn get_status(base: &str, token: &str) -> u16 {
        let request = ureq::http::Request::builder()
            .method("GET")
            .uri(format!("{base}/contexts"))
            .header("Authorization", format!("Bearer {token}"))
            .body(())
            .unwrap();
        match crate::support::test_agent().run(request) {
            Ok(response) => response.status().as_u16(),
            Err(error) => panic!("request dropped mid-reload: {error}"),
        }
    }

    let stop = std::sync::atomic::AtomicBool::new(false);
    let base = server.base.clone();
    let (before, after) = std::thread::scope(|scope| {
        let hammer = |token: &'static str| {
            let base = &base;
            let stop = &stop;
            scope.spawn(move || {
                let mut statuses = Vec::new();
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    statuses.push(get_status(base, token));
                }
                statuses
            })
        };
        let before = hammer("sekrit-before");
        let after = hammer("sekrit-after");

        std::fs::write(&config, "TAGURU_API_TOKENS=ci:sekrit-after\n").unwrap();
        server.signal("-HUP");
        eventually(Duration::from_secs(10), "the rotation to land", || {
            get_status(&base, "sekrit-after") == 200
        });
        // A few more requests on both threads AFTER the swap, then stop.
        std::thread::sleep(Duration::from_millis(200));
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        (before.join().unwrap(), after.join().unwrap())
    });

    // Per thread the flip is monotonic: each request completes before
    // the next starts, and every request judges against one whole
    // table — so the retiring key may never re-succeed after its
    // first 401, nor the incoming key fail after its first 200.
    let flips = |statuses: &[u16], from: u16, to: u16| {
        assert!(
            statuses.iter().all(|status| [from, to].contains(status)),
            "only definitive verdicts allowed: {statuses:?}"
        );
        statuses
            .windows(2)
            .all(|pair| pair[0] == from || pair[1] == to)
    };
    assert!(
        flips(&before, 200, 401),
        "the retiring key re-authenticated after dying: {before:?}"
    );
    assert!(
        flips(&after, 401, 200),
        "the incoming key failed after arming: {after:?}"
    );
    assert!(after.last() == Some(&200), "{after:?}");
}
