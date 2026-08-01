//! Structured logging and span/trace-id propagation to an OTEL collector.
//!
//! Named `observability`, not `tracing`, so this module never shadows
//! the `tracing` crate the binary itself logs through.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use serde_json::{Value, json};

use crate::support::*;

#[test]
fn log_output_is_structured_when_json_format_is_requested() {
    let data_dir = std::env::temp_dir().join(format!("taguru-jsonlog-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &data_dir)
        .env("TAGURU_LOG_FORMAT", "json")
        .env("RUST_LOG", "info");
    let mut child = command
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
fn a_request_span_reaches_the_collector_carrying_the_inbound_trace_identity() {
    let collector = FakeCollector::start();
    let server = Server::start_with_env(
        "otlp",
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );

    // The upstream (a mesh, another service) already started a trace.
    let response = test_agent()
        .get(&format!("{}/health", server.base))
        .header(
            "traceparent",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        )
        .call()
        .expect("health must answer");
    assert_eq!(response.status(), 200);

    // Graceful shutdown flushes the batch exporter before exit, so by
    // the time the process is gone the span has been delivered.
    let _ = server.stop_gracefully();

    let spans = collector.spans();
    let span = spans
        .iter()
        .find(|span| span["name"] == "GET /health")
        .unwrap_or_else(|| panic!("no GET /health span among {spans:?}"));

    // Same trace, parented under the caller's span: this is the whole
    // point of accepting inbound context.
    assert_eq!(span["traceId"], "0af7651916cd43dd8448eb211c80319c");
    assert_eq!(span["parentSpanId"], "b7ad6b7169203331");
    assert_eq!(
        attribute(span, "http.route").map(|value| value["stringValue"].clone()),
        Some(json!("/health"))
    );
    assert_eq!(
        attribute(span, "http.response.status_code").cloned(),
        Some(json!({"intValue": "200"}))
    );

    // The resource names the service — the default when
    // OTEL_SERVICE_NAME is unset.
    let service = span["resource"]["attributes"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|attribute| attribute["key"] == "service.name")
        .map(|attribute| attribute["value"]["stringValue"].clone());
    assert_eq!(service, Some(json!("taguru")));
}

/// Drains `reader` on a side thread and waits up to `timeout` (wall
/// clock, not a line count) for a JSON log record whose
/// `fields.message` is `"http"` and whose `fields.route` matches
/// `route`. Panics with every line collected so far plus what it was
/// looking for — instead of blocking forever — when the record never
/// appears (a dropped log level, a stalled process, a malformed
/// stream): the reader thread keeps blocking on its own read, but the
/// deadline here is wall-clock and fires regardless.
fn wait_for_access_log_record<R>(reader: R, route: &str, timeout: std::time::Duration) -> Value
where
    R: std::io::Read + Send + 'static,
{
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            let Ok(line) = line else { break };
            if sender.send(line).is_err() {
                break;
            }
        }
    });

    let deadline = std::time::Instant::now() + timeout;
    let mut collected = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let received = if remaining.is_zero() {
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        } else {
            receiver.recv_timeout(remaining)
        };
        match received {
            Ok(line) => {
                collected.push(line.clone());
                if let Ok(parsed) = serde_json::from_str::<Value>(&line)
                    && parsed["fields"]["message"] == "http"
                    && parsed["fields"]["route"] == route
                {
                    return parsed;
                }
            }
            Err(_) => panic!(
                "no access-log record with message \"http\" and route {route:?} appeared \
                 within {timeout:?}; stderr collected so far ({} line(s)):\n{}",
                collected.len(),
                collected.join("\n")
            ),
        }
    }
}

#[test]
fn the_access_log_carries_the_trace_id_when_export_is_configured() {
    let data_dir = std::env::temp_dir().join(format!("taguru-tracelog-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    // The endpoint only needs to be configured, not alive: spans are
    // created (and the log correlated) regardless of delivery.
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &data_dir)
        .env("TAGURU_LOG_FORMAT", "json")
        .env("RUST_LOG", "info")
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9");
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server binary must spawn");

    let stdout = child.stdout.take().expect("stdout must be piped");
    let stderr = child.stderr.take().expect("stderr must be piped");

    // Kills the child and cleans the data dir on every exit path —
    // success, an assertion failure, or a panic out of the bounded
    // wait below — so a failing run never leaves an orphaned server.
    struct KillOnDrop<'a>(&'a mut std::process::Child, &'a std::path::Path);
    impl Drop for KillOnDrop<'_> {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
            let _ = std::fs::remove_dir_all(self.1);
        }
    }
    let _cleanup = KillOnDrop(&mut child, &data_dir);

    let (addr, _stdout_lines) = common::read_listen_line("server", stdout);
    let base = format!("http://{addr}");

    let response = test_agent()
        .get(&format!("{base}/health"))
        .header(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        )
        .call()
        .expect("health must answer");
    assert_eq!(response.status(), 200);

    // The access-log line for that request must carry the same trace
    // id the caller minted — the log↔trace join key. Waited for with a
    // wall-clock deadline, not a line-count cap: a dropped log level
    // or a stalled child must fail this test, not block the suite.
    let record = wait_for_access_log_record(stderr, "/health", std::time::Duration::from_secs(10));
    assert_eq!(
        record["fields"]["trace_id"],
        json!("4bf92f3577b34da6a3ce929d0e0e4736"),
        "{record}"
    );
}

/// #311: a server that never emits the expected record (log level
/// dropped, process stalled) must fail this test at the deadline
/// instead of blocking the reader thread — and thus the suite —
/// forever.
#[test]
fn wait_for_access_log_record_times_out_instead_of_blocking_forever() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener must bind");
    let addr = listener.local_addr().unwrap();
    // Kept alive so the accepted stream sees an open connection with
    // no data — never EOF — the same shape as a live child's pipe.
    let _keep_open = std::net::TcpStream::connect(addr).expect("client must connect");
    let (server_side, _) = listener.accept().expect("listener must accept");

    let started = std::time::Instant::now();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        wait_for_access_log_record(
            server_side,
            "/health",
            std::time::Duration::from_millis(200),
        )
    }));
    let elapsed = started.elapsed();

    assert!(
        outcome.is_err(),
        "must fail, not succeed, when no record ever appears"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "must fail near its own deadline, not block indefinitely: {elapsed:?}"
    );
    let message = outcome
        .unwrap_err()
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("/health"), "{message}");
    assert!(message.contains("200ms"), "{message}");
}

/// #311: malformed (non-JSON) lines must be skipped, not crash the
/// parser or count as a match — and the deadline must still fire when
/// nothing valid ever follows them, with the malformed lines visible
/// in the diagnostic for debugging.
#[test]
fn wait_for_access_log_record_skips_malformed_lines_and_still_times_out() {
    use std::io::Write;

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener must bind");
    let addr = listener.local_addr().unwrap();
    let mut keep_open = std::net::TcpStream::connect(addr).expect("client must connect");
    let (server_side, _) = listener.accept().expect("listener must accept");

    writeln!(keep_open, "not json").unwrap();
    writeln!(keep_open, "{{\"fields\":{{\"message\":\"http\"}}}}").unwrap();
    keep_open.flush().unwrap();

    let started = std::time::Instant::now();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        wait_for_access_log_record(
            server_side,
            "/health",
            std::time::Duration::from_millis(200),
        )
    }));
    let elapsed = started.elapsed();

    assert!(
        outcome.is_err(),
        "must fail, not succeed, when no line ever matches"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "malformed input must not defeat the deadline: {elapsed:?}"
    );
    let message = outcome
        .unwrap_err()
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("not json"), "{message}");
}
