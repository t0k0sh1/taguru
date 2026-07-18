//! The opt-in search-event log.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use serde_json::{Value, json};

use crate::support::*;

/// One request against a manually spawned server — the JSON-log
/// sessions below run outside the `Server` harness so they can own
/// stderr. Returns the status; bodies are irrelevant to log tests.
fn raw_call(base: &str, method: &str, path: &str, body: Option<Value>) -> u16 {
    let request = ureq::http::Request::builder()
        .method(method)
        .uri(format!("{base}{path}"));
    let response = match body {
        Some(body) => request
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .map(|request| test_agent().run(request)),
        None => request.body(()).map(|request| test_agent().run(request)),
    };
    match response.expect("request must assemble") {
        Ok(response) => response.status().as_u16(),
        Err(error) => panic!("{method} {path} failed: {error}"),
    }
}

/// Spawns the binary with JSON logs on a piped stderr, runs `drive`,
/// stops the server gracefully, and returns every stderr line that
/// parsed as JSON. The child has exited before the scan, so an absent
/// line is a real absence, not a race.
fn json_log_session(tag: &str, extra_env: &[(&str, &str)], drive: impl FnOnce(&str)) -> Vec<Value> {
    let data_dir = std::env::temp_dir().join(format!("taguru-log-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &data_dir)
        .env("TAGURU_LOG_FORMAT", "json");
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server binary must spawn");

    let stdout = child.stdout.take().expect("stdout must be piped");
    let (addr, _stdout_lines) = common::read_listen_line("server", stdout);
    let base = format!("http://{addr}");
    // Drain stderr concurrently: a session can log more than one pipe
    // buffer holds, and a full pipe blocks the server's workers.
    let stderr = child.stderr.take().expect("stderr must be piped");
    let reader = std::thread::spawn(move || {
        BufReader::new(stderr)
            .lines()
            .map_while(Result::ok)
            .filter_map(|line| serde_json::from_str::<Value>(&line).ok())
            .collect::<Vec<Value>>()
    });

    drive(&base);

    Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .expect("kill must run");
    let _ = child.wait();
    let lines = reader.join().expect("stderr reader must finish");
    let _ = std::fs::remove_dir_all(&data_dir);
    lines
}

#[test]
fn search_events_carry_cue_and_hits_when_opted_in() {
    let lines = json_log_session("searchlog-on", &[("TAGURU_LOG_SEARCHES", "1")], |base| {
        assert_eq!(
            raw_call(
                base,
                "PUT",
                "/contexts/s",
                Some(json!({"description": "d"}))
            ),
            200
        );
        assert_eq!(
            raw_call(
                base,
                "POST",
                "/contexts/s/associations",
                Some(json!([{
                    "subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
                    "weight": 1.0, "source": "p1"
                }]))
            ),
            200
        );
        assert_eq!(
            raw_call(
                base,
                "POST",
                "/contexts/s/recall",
                Some(json!({"cue": "青嶺酒造"}))
            ),
            200
        );
        assert_eq!(
            raw_call(
                base,
                "POST",
                "/contexts/s/resolve",
                Some(json!({"cue": "qqqq"}))
            ),
            200
        );
    });

    let searches: Vec<&Value> = lines
        .iter()
        .filter(|line| line["target"] == "taguru::search")
        .collect();
    let recall = searches
        .iter()
        .find(|line| line["fields"]["op"] == "recall")
        .expect("a recall event must be logged");
    assert_eq!(recall["fields"]["context"], json!("s"), "{recall}");
    assert_eq!(recall["fields"]["cue"], json!("青嶺酒造"), "{recall}");
    assert_eq!(recall["fields"]["hits"], json!(1), "{recall}");
    let resolve = searches
        .iter()
        .find(|line| line["fields"]["op"] == "resolve")
        .expect("a resolve event must be logged");
    assert_eq!(resolve["fields"]["cue"], json!("qqqq"), "{resolve}");
    assert_eq!(resolve["fields"]["hits"], json!(0), "{resolve}");
    assert_eq!(resolve["fields"]["tier"], json!("miss"), "{resolve}");
}

#[test]
fn search_events_stay_absent_without_the_opt_in() {
    let lines = json_log_session("searchlog-off", &[], |base| {
        assert_eq!(
            raw_call(
                base,
                "PUT",
                "/contexts/s",
                Some(json!({"description": "d"}))
            ),
            200
        );
        assert_eq!(
            raw_call(
                base,
                "POST",
                "/contexts/s/recall",
                Some(json!({"cue": "秘匿の合い言葉"}))
            ),
            200
        );
    });

    // The stream is alive — access-log lines prove the scan saw real
    // output — yet carries no search events, and so no cue content.
    assert!(
        lines.iter().any(|line| line["fields"]["message"] == "http"),
        "expected access-log lines in the scanned stderr"
    );
    assert!(
        lines.iter().all(|line| line["target"] != "taguru::search"),
        "a search event leaked without the opt-in"
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.to_string().contains("秘匿の合い言葉")),
        "cue content leaked into the default log stream"
    );
}
