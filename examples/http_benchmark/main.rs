//! HTTP-level load benchmark: the whole server — routing, auth, locks,
//! WAL, serialization — under concurrent clients, not the bare
//! `Context` the `benchmark` example times. Drives a RUNNING server
//! (never starts one: measure the deployment you actually run, over
//! the network you actually cross) through three phases — seed writes,
//! concurrent reads, a mixed read/write blend — and reports
//! throughput, latency percentiles, and every non-2xx.
//!
//! ```sh
//! # a scratch server to point at, if none is running:
//! #   TAGURU_DATA_DIR=$(mktemp -d) cargo run --release &
//! cargo run --release --example http_benchmark -- \
//!     --url http://127.0.0.1:8248 --concurrency 8 --requests 2000
//! ```
//!
//! Treat the numbers as RELATIVE — across commits, across
//! configurations (WAL on/off, cache sizes), across `--concurrency` —
//! never as absolutes. The benchmark writes into (and afterwards
//! deletes) one context named `http-benchmark`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

struct Config {
    url: String,
    token: Option<String>,
    concurrency: usize,
    requests: usize,
    seed: usize,
}

const CONTEXT: &str = "http-benchmark";

fn main() {
    let config = parse_args();
    let client = Client {
        agent: ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(30))
            .build(),
        url: config.url.clone(),
        token: config.token.clone(),
    };

    // Fresh context, always: a leftover from an aborted run would
    // skew the seed phase.
    let _ = client.call("DELETE", &format!("/contexts/{CONTEXT}"), None);
    client
        .call(
            "PUT",
            &format!("/contexts/{CONTEXT}"),
            Some(serde_json::json!({"description": "http_benchmark scratch"})),
        )
        .expect("create must succeed — is the server up, and the token right?");

    println!(
        "target {} · concurrency {} · {} requests per phase\n",
        config.url, config.concurrency, config.requests
    );

    // Phase 1 — seed writes: batches of 100 associations over a pool
    // of subjects, the shape extract-driven ingest produces.
    let seeded = AtomicUsize::new(0);
    let batches = config.seed.div_ceil(100);
    let write_stats = run_phase("seed writes (batches of 100)", &config, batches, |_| {
        let start = seeded.fetch_add(100, Ordering::Relaxed);
        let batch: Vec<serde_json::Value> = (start..start + 100)
            .map(|index| {
                serde_json::json!({
                    "subject": format!("概念{}", index % (config.seed / 20).max(1)),
                    "label": format!("関係{}", index % 97),
                    "object": format!("対象{index}"),
                    "weight": 1.0,
                    "source": format!("doc-{}.md", index % 500),
                })
            })
            .collect();
        client.call(
            "POST",
            &format!("/contexts/{CONTEXT}/associations"),
            Some(serde_json::Value::Array(batch)),
        )
    });
    write_stats.print(batches * 100, "association");

    // Phase 2 — concurrent reads: recall + activate over the seeded
    // vocabulary, the retrieval loop's two ends.
    let turn = AtomicUsize::new(0);
    let read_stats = run_phase("reads (recall/activate)", &config, config.requests, |_| {
        let index = turn.fetch_add(1, Ordering::Relaxed);
        let cue = format!("概念{}", index % (config.seed / 20).max(1));
        if index.is_multiple_of(2) {
            client.call(
                "POST",
                &format!("/contexts/{CONTEXT}/recall"),
                Some(serde_json::json!({"cue": cue})),
            )
        } else {
            client.call(
                "POST",
                &format!("/contexts/{CONTEXT}/activate"),
                Some(serde_json::json!({"origins": [cue], "limit": 20})),
            )
        }
    });
    read_stats.print(config.requests, "request");

    // Phase 3 — mixed: 90% reads, 10% single-association writes; the
    // same-context write lock and the WAL fsync now sit inside the
    // read path's world.
    let turn = AtomicUsize::new(0);
    let mixed_stats = run_phase(
        "mixed (90% read / 10% write)",
        &config,
        config.requests,
        |_| {
            let index = turn.fetch_add(1, Ordering::Relaxed);
            let cue = format!("概念{}", index % (config.seed / 20).max(1));
            if index.is_multiple_of(10) {
                client.call(
                    "POST",
                    &format!("/contexts/{CONTEXT}/associations"),
                    Some(serde_json::json!([{
                        "subject": cue, "label": "追記", "object": format!("値{index}"),
                        "weight": 1.0, "source": "mixed.md",
                    }])),
                )
            } else {
                client.call(
                    "POST",
                    &format!("/contexts/{CONTEXT}/recall"),
                    Some(serde_json::json!({"cue": cue})),
                )
            }
        },
    );
    mixed_stats.print(config.requests, "request");

    let _ = client.call("DELETE", &format!("/contexts/{CONTEXT}"), None);
}

struct Client {
    agent: ureq::Agent,
    url: String,
    token: Option<String>,
}

impl Client {
    fn call(
        &self,
        method: &str,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<(), String> {
        let mut request = self.agent.request(method, &format!("{}{path}", self.url));
        if let Some(token) = &self.token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        let response = match body {
            Some(body) => request
                .set("Content-Type", "application/json")
                .send_string(&body.to_string()),
            None => request.call(),
        };
        match response {
            Ok(reply) => {
                // Drain so the connection returns to the pool.
                let _ = reply.into_string();
                Ok(())
            }
            Err(ureq::Error::Status(code, _)) => Err(format!("HTTP {code}")),
            Err(error) => Err(format!("transport: {error}")),
        }
    }
}

/// Latencies in micros plus the failure tally, merged across workers.
struct PhaseStats {
    label: &'static str,
    latencies_us: Vec<u64>,
    failures: usize,
    wall: Duration,
}

fn run_phase(
    label: &'static str,
    config: &Config,
    total: usize,
    one_request: impl Fn(usize) -> Result<(), String> + Sync,
) -> PhaseStats {
    let issued = AtomicUsize::new(0);
    let failures = AtomicUsize::new(0);
    let one_request = &one_request;
    let started = Instant::now();
    let latencies: Vec<u64> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..config.concurrency)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine: Vec<u64> = Vec::new();
                    loop {
                        let index = issued.fetch_add(1, Ordering::Relaxed);
                        if index >= total {
                            break mine;
                        }
                        let call_started = Instant::now();
                        if let Err(error) = one_request(index)
                            && failures.fetch_add(1, Ordering::Relaxed) < 3
                        {
                            eprintln!("{label}: {error}");
                        }
                        mine.push(call_started.elapsed().as_micros() as u64);
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("worker must not panic"))
            .collect()
    });
    PhaseStats {
        label,
        latencies_us: latencies,
        failures: failures.load(Ordering::Relaxed),
        wall: started.elapsed(),
    }
}

impl PhaseStats {
    fn print(mut self, units: usize, unit: &str) {
        self.latencies_us.sort_unstable();
        let calls = self.latencies_us.len().max(1);
        let percentile = |p: f64| -> f64 {
            let index = ((calls as f64 * p).ceil() as usize).clamp(1, calls) - 1;
            self.latencies_us[index] as f64 / 1000.0
        };
        println!(
            "{:<32} {:>8.0} {unit}s/s · p50 {:>7.2} ms · p95 {:>7.2} ms · p99 {:>7.2} ms{}",
            self.label,
            units as f64 / self.wall.as_secs_f64(),
            percentile(0.50),
            percentile(0.95),
            percentile(0.99),
            match self.failures {
                0 => String::new(),
                failed => format!(" · {failed} FAILED"),
            },
        );
    }
}

fn parse_args() -> Config {
    let mut config = Config {
        url: std::env::var("TAGURU_URL").unwrap_or_else(|_| "http://127.0.0.1:8248".to_string()),
        token: std::env::var("TAGURU_API_TOKEN").ok(),
        concurrency: 8,
        requests: 2000,
        seed: 10_000,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .unwrap_or_else(|| panic!("{name} needs a value"))
        };
        match arg.as_str() {
            "--url" => config.url = value("--url"),
            "--token" => config.token = Some(value("--token")),
            "--concurrency" => {
                config.concurrency = value("--concurrency").parse().expect("--concurrency")
            }
            "--requests" => config.requests = value("--requests").parse().expect("--requests"),
            "--seed-associations" => {
                config.seed = value("--seed-associations").parse().expect("--seed")
            }
            other => panic!("unknown argument '{other}' (see examples/http_benchmark/README.md)"),
        }
    }
    config.concurrency = config.concurrency.max(1);
    config
}
