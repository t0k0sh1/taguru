//! `taguru calibrate` (issue #131) end to end: a deterministic
//! topic-vector stub, a real server, a real probe file — the bands
//! measure, contaminated probes are excluded loudly, overlapping
//! bands refuse to invent a floor, and the states that make
//! calibration impossible abort with their reason instead of a
//! fabricated report.

use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};

use crate::support::*;

/// An embeddings stub with one axis per stored name and a shared
/// component: a gloss (which always opens with its own name — the
/// `名前。facts` shape the server embeds) lands on that name's axis,
/// a cue lands on the axis its keyword aims at, and every distinct
/// pair of axes meets at cosine 0.25/1.25 = 0.2 through the shared
/// component. In-topic cosines are 1.0 — two clean bands, exactly
/// what a calibration must find. `flat` collapses every text onto one
/// axis instead: upper == lower, the overlap case.
fn spawn_topic_embeddings(flat: Arc<AtomicBool>) -> String {
    const AXES: [&str; 6] = ["琥珀", "樹脂化石", "ダイヤモンド", "輝き", "分類", "特徴"];
    const CUE_KEYWORDS: [(&str, usize); 3] = [("じゅし", 0), ("かせき", 0), ("かがやき", 2)];
    // AXES plus one axis for unrecognized text, plus the shared
    // component's own dimension.
    const WIDTH: usize = AXES.len() + 2;

    fn vector(text: &str, flat: bool) -> Vec<f32> {
        let mut vector = vec![0.0f32; WIDTH];
        if flat {
            vector[0] = 1.0;
            return vector;
        }
        let axis = AXES
            .iter()
            .position(|name| text.starts_with(&format!("{name}。")))
            .or_else(|| {
                CUE_KEYWORDS
                    .iter()
                    .find(|(keyword, _)| text.contains(keyword))
                    .map(|&(_, axis)| axis)
            })
            .unwrap_or(AXES.len());
        vector[axis] = 1.0;
        vector[WIDTH - 1] = 0.5;
        vector
    }

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let flat = Arc::clone(&flat);
            std::thread::spawn(move || {
                use std::io::{Read, Write};
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let body_start = loop {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                    if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        break at + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&buffer[..body_start]).to_string();
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())?
                    })
                    .unwrap_or(0);
                while buffer.len() < body_start + length {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                }
                let request: Value =
                    serde_json::from_slice(&buffer[body_start..body_start + length]).unwrap();
                let flat = flat.load(Ordering::Relaxed);
                let data: Vec<Value> = request["input"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .map(|input| {
                        json!({ "embedding": vector(input.as_str().unwrap_or_default(), flat) })
                    })
                    .collect();
                let body = json!({ "data": data }).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            });
        }
    });
    format!("http://{addr}/v1/embeddings")
}

/// One `taguru calibrate` run, hermetic like every other binary spawn.
fn run_calibrate(args: &[&str], env: &[(&str, &str)]) -> (i32, String, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .arg("calibrate")
        .args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command.output().expect("calibrate must run");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// A server with two single-topic concept clusters embedded through
/// the topic stub — the smallest corpus with a real distractor band.
fn calibration_server(tag: &str, flat: Arc<AtomicBool>) -> Server {
    let provider = spawn_topic_embeddings(flat);
    let server = Server::start_with_env(
        tag,
        &[
            ("TAGURU_EMBED_URL", provider.as_str()),
            ("TAGURU_EMBED_MODEL", "titan-mock"),
        ],
    );
    server.ok(
        "PUT",
        "/contexts/cal",
        Some(json!({"description": "calibration corpus"})),
    );
    server.ok(
        "POST",
        "/contexts/cal/associations",
        Some(json!([
            {"subject": "琥珀", "label": "分類", "object": "樹脂化石", "weight": 1.0},
            {"subject": "ダイヤモンド", "label": "特徴", "object": "輝き", "weight": 1.0},
        ])),
    );
    server.ok("POST", "/contexts/cal/embeddings/refresh", None);
    server
}

fn write_probes(tag: &str, lines: &str) -> std::path::PathBuf {
    let dir = batch_dir(tag);
    let path = dir.join("probes.tsv");
    std::fs::write(&path, lines).expect("probe file must be writable");
    path
}

/// The main arc: three clean paraphrase probes measure two bands
/// (1.0 vs 0.2 by stub construction), three contaminated probes are
/// excluded each with its own diagnosis, and the suggested floor
/// lands mid-gap — with the report identity taken from the new
/// `GET /contexts/{name}/embeddings` exposure.
#[test]
fn a_calibration_run_measures_bands_excludes_contamination_and_suggests_a_floor() {
    let server = calibration_server("calibrate-bands", Arc::new(AtomicBool::new(false)));

    // The identity exposure the report stamps itself with (#131/#133):
    // provider setting beside what the sidecar actually holds.
    let status = server.ok("GET", "/contexts/cal/embeddings", None);
    assert_eq!(status["provider_model"], json!("titan-mock"), "{status}");
    assert_eq!(status["glosses"]["model"], json!("titan-mock"), "{status}");
    assert_eq!(status["glosses"]["width"], json!(8), "{status}");
    assert_eq!(status["glosses"]["concepts"], json!(4), "{status}");
    assert_eq!(status["glosses"]["labels"], json!(2), "{status}");
    assert!(status["passages"].is_null(), "{status}");

    let probes = write_probes(
        "calibrate-bands",
        "# 較正プローブ — 上3本が本物、下3本は汚染の各種\n\
         むかしのじゅし\t琥珀\n\
         おおむかしのかせき\t琥珀\n\
         かがやきのひかり\tダイヤモンド\n\
         琥珀いろ\t琥珀\n\
         ダイヤモンド\t輝き\n\
         ふしぎなたからもの\t翡翠\n",
    );
    let (code, stdout, stderr) = run_calibrate(
        &[
            "--context",
            "cal",
            "--probes",
            probes.to_str().unwrap(),
            "--json",
            &server.base,
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let report: Value = serde_json::from_str(&stdout).expect("calibrate --json prints JSON");

    // The identity and the current serving floor.
    assert_eq!(report["model"], json!("titan-mock"), "{report}");
    assert_eq!(report["width"], json!(8), "{report}");
    assert!(
        (report["effective_floor"].as_f64().unwrap() - 0.35).abs() < 1e-6,
        "{report}"
    );

    // Three measured, three excluded — each exclusion naming its own
    // failure, not a generic shrug.
    assert_eq!(report["probes"], json!(6), "{report}");
    assert_eq!(report["measured"], json!(3), "{report}");
    let excluded = report["excluded"].as_array().unwrap();
    assert_eq!(excluded.len(), 3, "{report}");
    let reason_for = |cue: &str| -> String {
        excluded
            .iter()
            .find(|row| row["cue"] == json!(cue))
            .unwrap_or_else(|| panic!("no exclusion for '{cue}' in {report}"))["reason"]
            .as_str()
            .unwrap()
            .to_string()
    };
    assert!(
        reason_for("琥珀いろ").contains("lexically resolves with confidence"),
        "{report}"
    );
    assert!(
        reason_for("ダイヤモンド").contains("itself a stored spelling"),
        "{report}"
    );
    assert!(
        reason_for("ふしぎなたからもの").contains("not stored in this context"),
        "{report}"
    );

    // The bands the stub constructs: expected cosines at 1.0, best
    // distractors at 0.2, floor mid-gap.
    let upper = &report["upper"];
    assert_eq!(upper["count"], json!(3), "{report}");
    assert!(upper["min"].as_f64().unwrap() > 0.99, "{report}");
    let lower = &report["lower"];
    assert_eq!(lower["count"], json!(3), "{report}");
    let lower_max = lower["max"].as_f64().unwrap();
    assert!((0.19..0.21).contains(&lower_max), "{report}");
    assert_eq!(report["verdict"], json!("suggested"), "{report}");
    let floor = report["suggested_floor"].as_f64().unwrap();
    assert!((0.55..0.65).contains(&floor), "{report}");
    let gap = report["gap"].as_f64().unwrap();
    assert!((0.75..0.85).contains(&gap), "{report}");

    // Every measured probe carries its evidence: the canonical, the
    // cosine, and the strongest distractor by name.
    let per_probe = report["per_probe"].as_array().unwrap();
    assert_eq!(per_probe.len(), 3, "{report}");
    for row in per_probe {
        assert!(row["cosine"].as_f64().unwrap() > 0.99, "{report}");
        let other = row["best_other"]["cosine"].as_f64().unwrap();
        assert!((0.19..0.21).contains(&other), "{report}");
    }

    // The human report ends in the one line an operator pastes.
    let (code, stdout, stderr) = run_calibrate(
        &[
            "--context",
            "cal",
            "--probes",
            probes.to_str().unwrap(),
            &server.base,
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains(&format!("TAGURU_SEMANTIC_FLOOR={floor}")),
        "human report must print the env line verbatim: {stdout}"
    );
    assert!(stdout.contains("titan-mock"), "{stdout}");
}

/// A model that cannot separate the bands (every text embeds
/// identically) earns an overlap verdict and NO suggested floor — the
/// honest answer the issue demands instead of a fabricated number.
#[test]
fn overlapping_bands_refuse_to_invent_a_floor() {
    let server = calibration_server("calibrate-overlap", Arc::new(AtomicBool::new(true)));
    let probes = write_probes(
        "calibrate-overlap",
        "むかしのじゅし\t琥珀\nかがやきのひかり\tダイヤモンド\n",
    );
    let (code, stdout, stderr) = run_calibrate(
        &[
            "--context",
            "cal",
            "--probes",
            probes.to_str().unwrap(),
            "--json",
            &server.base,
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let report: Value = serde_json::from_str(&stdout).expect("calibrate --json prints JSON");
    assert_eq!(report["verdict"], json!("overlap"), "{report}");
    assert!(report["suggested_floor"].is_null(), "{report}");
    // Upper and lower both sit at 1.0 — the overlap is total.
    assert!(report["upper"]["min"].as_f64().unwrap() > 0.99, "{report}");
    assert!(report["lower"]["max"].as_f64().unwrap() > 0.99, "{report}");
}

/// The two server states no probe set can fix, each named before any
/// probe spends provider money: embeddings off entirely, and a
/// context whose vectors were never refreshed.
#[test]
fn servers_that_cannot_calibrate_say_why() {
    // No provider at all.
    let bare = Server::start("calibrate-off");
    bare.ok(
        "PUT",
        "/contexts/cal",
        Some(json!({"description": "no embeddings here"})),
    );
    let probes = write_probes("calibrate-off", "むかしのじゅし\t琥珀\n");
    let (code, _, stderr) = run_calibrate(
        &[
            "--context",
            "cal",
            "--probes",
            probes.to_str().unwrap(),
            "--json",
            &bare.base,
        ],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("no embedding provider"), "{stderr}");

    // A provider, but nothing embedded yet.
    let provider = spawn_topic_embeddings(Arc::new(AtomicBool::new(false)));
    let unrefreshed = Server::start_with_env(
        "calibrate-unrefreshed",
        &[
            ("TAGURU_EMBED_URL", provider.as_str()),
            ("TAGURU_EMBED_MODEL", "titan-mock"),
        ],
    );
    unrefreshed.ok(
        "PUT",
        "/contexts/cal",
        Some(json!({"description": "vectors never refreshed"})),
    );
    unrefreshed.ok(
        "POST",
        "/contexts/cal/associations",
        Some(json!([{"subject": "琥珀", "label": "分類", "object": "樹脂化石", "weight": 1.0}])),
    );
    let (code, _, stderr) = run_calibrate(
        &[
            "--context",
            "cal",
            "--probes",
            probes.to_str().unwrap(),
            "--json",
            &unrefreshed.base,
        ],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("no gloss vectors yet"), "{stderr}");

    // An unknown context names itself too.
    let (code, _, stderr) = run_calibrate(
        &[
            "--context",
            "nope",
            "--probes",
            probes.to_str().unwrap(),
            "--json",
            &bare.base,
        ],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("404"), "{stderr}");
}

/// Calibrate authenticates the way the server does: TAGURU_API_TOKEN
/// (or the first TAGURU_API_TOKENS key) rides every request, and a
/// missing token is the server's own 401, not a hang or a panic.
#[test]
fn an_authenticated_server_accepts_the_environment_token() {
    let flat = Arc::new(AtomicBool::new(false));
    let provider = spawn_topic_embeddings(Arc::clone(&flat));
    let server = Server::start_with_env(
        "calibrate-auth",
        &[
            ("TAGURU_EMBED_URL", provider.as_str()),
            ("TAGURU_EMBED_MODEL", "titan-mock"),
            ("TAGURU_API_TOKEN", "sekrit"),
        ],
    );
    let with_token = |method: &str, path: &str, body: Option<Value>| {
        let (status, parsed) = server.call_with_token(method, path, body, Some("sekrit"));
        assert_eq!(status, 200, "{method} {path} -> {parsed}");
        parsed["result"].clone()
    };
    with_token(
        "PUT",
        "/contexts/cal",
        Some(json!({"description": "auth calibration"})),
    );
    with_token(
        "POST",
        "/contexts/cal/associations",
        Some(json!([
            {"subject": "琥珀", "label": "分類", "object": "樹脂化石", "weight": 1.0},
            {"subject": "ダイヤモンド", "label": "特徴", "object": "輝き", "weight": 1.0},
        ])),
    );
    with_token("POST", "/contexts/cal/embeddings/refresh", None);

    let probes = write_probes("calibrate-auth", "むかしのじゅし\t琥珀\n");
    let (code, stdout, stderr) = run_calibrate(
        &[
            "--context",
            "cal",
            "--probes",
            probes.to_str().unwrap(),
            "--json",
            &server.base,
        ],
        &[("TAGURU_API_TOKEN", "sekrit")],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let report: Value = serde_json::from_str(&stdout).expect("calibrate --json prints JSON");
    assert_eq!(report["verdict"], json!("suggested"), "{report}");

    let (code, _, stderr) = run_calibrate(
        &[
            "--context",
            "cal",
            "--probes",
            probes.to_str().unwrap(),
            "--json",
            &server.base,
        ],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("401"), "{stderr}");
}
