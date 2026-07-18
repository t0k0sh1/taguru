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
        .env("TAGURU_LOG_FORMAT", "json");
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

/// A single-purpose OTLP/HTTP sink: accepts POSTs, stores every body,
/// answers 200. Runs until the test process exits.
struct FakeCollector {
    endpoint: String,
    bodies: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl FakeCollector {
    fn start() -> Self {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("collector must bind");
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let bodies = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = bodies.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                // Read headers, then exactly Content-Length body bytes.
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let header_end = loop {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break None,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                    if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        break Some(at + 4);
                    }
                };
                let Some(header_end) = header_end else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())?
                    })
                    .unwrap_or(0);
                while buffer.len() < header_end + length {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                }
                let body = String::from_utf8_lossy(&buffer[header_end..]).to_string();
                sink.lock().unwrap().push(body);
                let _ = stream.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                      Content-Length: 2\r\nConnection: close\r\n\r\n{}",
                );
            }
        });
        Self { endpoint, bodies }
    }

    /// Every span object exported so far, flattened across batches.
    fn spans(&self) -> Vec<Value> {
        let mut spans = Vec::new();
        for body in self.bodies.lock().unwrap().iter() {
            let Ok(parsed) = serde_json::from_str::<Value>(body) else {
                continue;
            };
            for resource_spans in parsed["resourceSpans"].as_array().into_iter().flatten() {
                for scope_spans in resource_spans["scopeSpans"]
                    .as_array()
                    .into_iter()
                    .flatten()
                {
                    for span in scope_spans["spans"].as_array().into_iter().flatten() {
                        let mut span = span.clone();
                        span["resource"] = resource_spans["resource"].clone();
                        spans.push(span);
                    }
                }
            }
        }
        spans
    }
}

/// One attribute value out of the OTLP attribute list shape
/// `[{"key": ..., "value": {"stringValue": ...}}]`.
fn attribute<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
    span["attributes"]
        .as_array()?
        .iter()
        .find(|attribute| attribute["key"] == key)
        .map(|attribute| &attribute["value"])
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
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9");
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server binary must spawn");

    let stdout = child.stdout.take().expect("stdout must be piped");
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
    // id the caller minted — the log↔trace join key.
    let stderr = child.stderr.take().expect("stderr must be piped");
    let mut found = None;
    for line in BufReader::new(stderr).lines().take(200) {
        let Ok(line) = line else { break };
        let Ok(parsed) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if parsed["fields"]["message"] == "http" && parsed["fields"]["route"] == "/health" {
            found = Some(parsed);
            break;
        }
    }
    let record = found.expect("an access-log line for /health must appear");
    assert_eq!(
        record["fields"]["trace_id"],
        json!("4bf92f3577b34da6a3ce929d0e0e4736"),
        "{record}"
    );

    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&data_dir);
}
