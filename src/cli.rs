//! Command-line surface. Hand-rolled on purpose — a default `serve`,
//! three offline subcommands, and one flag do not need an argument
//! framework; the same reasoning that keeps the metrics and BM25
//! in-tree.
//!
//! Exit codes: 0 success · 1 operation failure (corruption found,
//! server error) · 2 usage error.

use std::process::exit;

const USAGE: &str = concat!(
    "taguru ",
    env!("CARGO_PKG_VERSION"),
    " — long-term semantic memory for LLMs

USAGE:
  taguru [serve]                        start the HTTP server (the default)
  taguru version                        print the version
  taguru --help                         this text

ENVIRONMENT (every knob; unset = the shown default):
  TAGURU_ADDR                  bind address (127.0.0.1:8248; port 0 = pick free)
  TAGURU_DATA_DIR              data directory (./data)
  TAGURU_CACHE_BYTES           resident budget for unpinned contexts (512 MiB)
  TAGURU_FLUSH_SECS            image flush interval (5)
  TAGURU_WAL                   fsync write-ahead log, 0/false = off (on)
  TAGURU_WAL_MAX_BYTES         per-context WAL ceiling, 0 = none (256 MiB)
  TAGURU_API_TOKEN             bearer token; unset = UNAUTHENTICATED
  TAGURU_MAX_BODY_BYTES        request body cap (8 MiB)
  TAGURU_REQUEST_TIMEOUT_SECS  per-request budget (30)
  TAGURU_EMBED_URL             OpenAI-compatible /embeddings endpoint (off)
  TAGURU_EMBED_MODEL           embedding model name
  TAGURU_EMBED_API_KEY         embedding provider credential
  TAGURU_EMBED_AUTO            1 = refresh embeddings with each flush (off)
  RUST_LOG                     log filter, EnvFilter syntax (info)
  TAGURU_LOG_FORMAT            json for JSON log lines (pretty)
  TAGURU_LOG_SEARCHES          1 = per-search event log; cues are memory
                               CONTENT, so this is opt-in (off)
  OTEL_EXPORTER_OTLP_ENDPOINT  turns on OTLP/HTTP span export (off)

EXIT CODES: 0 ok · 1 failure or corruption found · 2 usage error
"
);

/// What `main` should do once the arguments are understood. Offline
/// subcommands never return — they print and exit before any runtime,
/// listener, or telemetry exists.
pub struct ServeArgs {}

/// Parses the process arguments, running and exiting for everything
/// except `serve`, whose settings it returns.
pub fn dispatch() -> ServeArgs {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => parse_serve(&[]),
        Some("serve") => parse_serve(&args[1..]),
        Some("version") => {
            refuse_extras("version", &args[1..]);
            println!("taguru {}", env!("CARGO_PKG_VERSION"));
            exit(0)
        }
        Some("help") | Some("--help") | Some("-h") => {
            print!("{USAGE}");
            exit(0)
        }
        Some(other) => {
            eprintln!("taguru: unknown argument '{other}' — try 'taguru --help'");
            exit(2)
        }
    }
}

/// `serve` takes no arguments (yet — the config flag arrives with the
/// configuration file).
fn parse_serve(args: &[String]) -> ServeArgs {
    if let Some(arg) = args.first() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                exit(0)
            }
            other => usage_error(&format!("unknown argument '{other}'")),
        }
    }
    ServeArgs {}
}

fn refuse_extras(command: &str, extras: &[String]) {
    if let Some(extra) = extras.first() {
        usage_error(&format!("'{command}' takes no argument, got '{extra}'"));
    }
}

fn usage_error(message: &str) -> ! {
    eprintln!("taguru: {message} — try 'taguru --help'");
    exit(2)
}
