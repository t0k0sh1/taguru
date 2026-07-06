//! Command-line surface. Hand-rolled on purpose — a default `serve`,
//! three offline subcommands, and one flag do not need an argument
//! framework; the same reasoning that keeps the metrics and BM25
//! in-tree.
//!
//! Exit codes: 0 success · 1 operation failure (corruption found,
//! server error) · 2 usage error.

use std::path::{Path, PathBuf};
use std::process::exit;

const USAGE: &str = concat!(
    "taguru ",
    env!("CARGO_PKG_VERSION"),
    " — long-term semantic memory for LLMs

USAGE:
  taguru [serve] [--config FILE]        start the HTTP server (the default)
  taguru version                        print the version
  taguru health [URL]                   exit 0 iff a running server's /health
                                        answers 200 — the container
                                        HEALTHCHECK; URL defaults to TAGURU_ADDR
  taguru inspect PATH                   verify a data directory or one .ctx
                                        image offline (backup check) — the
                                        same validating load the server runs
  taguru estimate --associations N ...  size memory/disk for a target corpus
                                        by building and measuring one
                                        (see: taguru estimate --help)
  taguru --help                         this text

CONFIGURATION FILE (--config FILE, or TAGURU_CONFIG=FILE):
  KEY=VALUE per line, # comments and blank lines ignored — the same
  dialect `docker run --env-file` reads, so one file serves both. Real
  environment variables always win over the file; unknown TAGURU_*
  keys are flagged as probable typos.

ENVIRONMENT (every knob; unset = the shown default):
  TAGURU_ADDR                  bind address (127.0.0.1:8248; port 0 = pick free)
  TAGURU_DATA_DIR              data directory (./data)
  TAGURU_CACHE_BYTES           resident budget for unpinned contexts (512 MiB)
  TAGURU_FLUSH_SECS            image flush interval (5)
  TAGURU_WAL                   fsync write-ahead log, 0/false = off (on)
  TAGURU_WAL_MAX_BYTES         per-context WAL ceiling, 0 = none (256 MiB)
  TAGURU_API_TOKEN             bearer token; unset = UNAUTHENTICATED
  TAGURU_API_TOKENS            named keys 'ci:tokA,laptop:tokB' — the access
                               log carries the key name; rotate by overlap
  TAGURU_PUBLIC_URL            public base URL; enables OAuth key delegation
                               on /mcp (claude.ai custom connectors)
  TAGURU_MAX_BODY_BYTES        request body cap (8 MiB)
  TAGURU_REQUEST_TIMEOUT_SECS  per-request budget (30)
  TAGURU_RATE_LIMIT_PER_MIN    per-key request budget; past it 429 +
                               Retry-After (0 = off)
  TAGURU_EMBED_URL             OpenAI-compatible /embeddings endpoint (off)
  TAGURU_EMBED_MODEL           embedding model name
  TAGURU_EMBED_API_KEY         embedding provider credential
  TAGURU_EMBED_AUTO            1 = refresh embeddings with each flush (off)
  TAGURU_SEMANTIC_FLOOR        semantic entry floor when neither the call nor
                               the context sets one (0.35, calibrated for
                               text-embedding-3-large; model-dependent)
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
pub struct ServeArgs {
    pub config: Option<PathBuf>,
}

/// Parses the process arguments, running and exiting for everything
/// except `serve`, whose settings it returns.
pub fn dispatch() -> ServeArgs {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => parse_serve(&[]),
        Some("serve") => parse_serve(&args[1..]),
        Some("--config") => parse_serve(&args),
        Some("version") => {
            refuse_extras("version", &args[1..]);
            println!("taguru {}", env!("CARGO_PKG_VERSION"));
            exit(0)
        }
        Some("help") | Some("--help") | Some("-h") => {
            print!("{USAGE}");
            exit(0)
        }
        Some("health") => exit(health(&args[1..])),
        Some("inspect") => exit(crate::inspect::run(&args[1..])),
        Some("estimate") => exit(crate::estimate::run(&args[1..])),
        Some(other) => {
            eprintln!("taguru: unknown argument '{other}' — try 'taguru --help'");
            exit(2)
        }
    }
}

/// `serve` takes exactly one optional `--config FILE`.
fn parse_serve(args: &[String]) -> ServeArgs {
    let mut config = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => usage_error("--config given twice"),
                None => usage_error("--config needs a file path"),
            },
            "--help" | "-h" => {
                print!("{USAGE}");
                exit(0)
            }
            other => usage_error(&format!("unknown argument '{other}'")),
        }
    }
    // The flag beats the variable, so a shell override works even when
    // a container image bakes TAGURU_CONFIG in.
    let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    ServeArgs { config }
}

/// `taguru health [URL]`: one GET against a running server's /health,
/// exit 0 iff it answers 200. This exists for container HEALTHCHECKs —
/// a scratch image has no curl, but it always has taguru itself.
/// /health is exempt from bearer auth, so no token is needed here.
fn health(args: &[String]) -> i32 {
    let base = match args {
        [] => default_base_url(),
        [flag] if flag == "--help" || flag == "-h" => {
            println!("usage: taguru health [URL]   exit 0 iff GET URL/health answers 200");
            return 0;
        }
        [url] => url.trim_end_matches('/').to_string(),
        [_, extra, ..] => usage_error(&format!("'health' takes one optional URL, got '{extra}'")),
    };
    let url = format!("{base}/health");
    // The agent timeout stays under HEALTHCHECK's own 5s deadline so
    // the verdict (and its message) comes from here, not from a kill.
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(4))
        .build();
    match agent.get(&url).call() {
        Ok(response) => {
            println!("{}", response.into_string().unwrap_or_default().trim());
            0
        }
        Err(ureq::Error::Status(code, response)) => {
            let body = response.into_string().unwrap_or_default();
            eprintln!("taguru: health: {url} answered {code}: {}", body.trim());
            1
        }
        Err(error) => {
            eprintln!("taguru: health: {error}");
            1
        }
    }
}

/// The URL `health` probes when none is given: TAGURU_ADDR, with an
/// unspecified bind address read as its loopback — 0.0.0.0 is
/// reachable at 127.0.0.1 from inside the same network namespace, and
/// inside the namespace is exactly where a HEALTHCHECK runs.
fn default_base_url() -> String {
    let addr = std::env::var("TAGURU_ADDR").unwrap_or_else(|_| "127.0.0.1:8248".to_string());
    format!("http://{}", loopback_of(&addr))
}

fn loopback_of(addr: &str) -> String {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    match addr.parse::<SocketAddr>() {
        Ok(mut socket) if socket.ip().is_unspecified() => {
            socket.set_ip(match socket.ip() {
                IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
                IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::LOCALHOST),
            });
            socket.to_string()
        }
        _ => addr.to_string(),
    }
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

/// Human-readable byte count: exact under a KiB, one decimal above —
/// these sit in report lines, not accounting ledgers.
pub fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 3] = [("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)];
    for (unit, size) in UNITS {
        if bytes >= size {
            return format!("{:.1} {unit}", bytes as f64 / size as f64);
        }
    }
    format!("{bytes} B")
}

/// Every variable the server reads, for typo detection: a config file
/// is where a misspelled knob silently becomes a no-op, and unlike the
/// shell it is worth linting.
const KNOWN_KEYS: [&str; 20] = [
    "TAGURU_ADDR",
    "TAGURU_DATA_DIR",
    "TAGURU_CACHE_BYTES",
    "TAGURU_FLUSH_SECS",
    "TAGURU_WAL",
    "TAGURU_WAL_MAX_BYTES",
    "TAGURU_API_TOKEN",
    "TAGURU_API_TOKENS",
    "TAGURU_PUBLIC_URL",
    "TAGURU_MAX_BODY_BYTES",
    "TAGURU_REQUEST_TIMEOUT_SECS",
    "TAGURU_RATE_LIMIT_PER_MIN",
    "TAGURU_EMBED_URL",
    "TAGURU_EMBED_MODEL",
    "TAGURU_EMBED_API_KEY",
    "TAGURU_EMBED_AUTO",
    "TAGURU_SEMANTIC_FLOOR",
    "TAGURU_LOG_FORMAT",
    "TAGURU_LOG_SEARCHES",
    "TAGURU_CONFIG",
];

/// Reads a configuration file into the process environment. Exits with
/// a usage error on an unreadable file or a malformed line — a config
/// the operator pointed at explicitly must never be half-applied.
///
/// Call this before the async runtime exists: applying the file means
/// `std::env::set_var`, which is only sound while the process is
/// single-threaded.
pub fn load_config(path: &Path) {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => usage_error(&format!("cannot read config {}: {error}", path.display())),
    };
    let pairs = match parse_config(&text) {
        Ok(pairs) => pairs,
        Err(message) => usage_error(&format!("config {}: {message}", path.display())),
    };
    for (key, value) in pairs {
        if key.starts_with("TAGURU_") && !KNOWN_KEYS.contains(&key.as_str()) {
            eprintln!("taguru: config: {key} is not a variable taguru reads (typo?)");
        }
        // The real environment wins: a `docker run -e` or shell export
        // must override the file, exactly as it overrides an image's
        // baked-in defaults.
        if std::env::var_os(&key).is_some() {
            eprintln!("taguru: config: {key} set in the environment; the file value is ignored");
            continue;
        }
        // SAFETY: the caller runs this before the tokio runtime (or
        // any other thread) starts, so no concurrent environment
        // access exists.
        unsafe { std::env::set_var(&key, &value) };
    }
}

/// The `docker run --env-file` dialect: one KEY=VALUE per line, `#`
/// comments and blank lines ignored, values taken verbatim (no quoting
/// or expansion). Returned in file order.
fn parse_config(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut pairs = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!(
                "line {}: expected KEY=VALUE, got '{raw}'",
                index + 1
            ));
        };
        let key = key.trim();
        if key.is_empty() || key.contains(char::is_whitespace) {
            return Err(format!(
                "line {}: '{key}' is not a variable name",
                index + 1
            ));
        }
        pairs.push((key.to_string(), value.to_string()));
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_lines_parse_the_env_file_dialect() {
        let text =
            "# comment\n\nTAGURU_ADDR=127.0.0.1:0\nTAGURU_API_TOKEN=a=b=c\n  TAGURU_WAL=1  \n";
        let pairs = parse_config(text).unwrap();
        assert_eq!(
            pairs,
            vec![
                ("TAGURU_ADDR".to_string(), "127.0.0.1:0".to_string()),
                // The first '=' splits; the value keeps the rest verbatim.
                ("TAGURU_API_TOKEN".to_string(), "a=b=c".to_string()),
                ("TAGURU_WAL".to_string(), "1".to_string()),
            ]
        );
    }

    #[test]
    fn config_lines_without_a_separator_are_refused_with_the_line_number() {
        let error = parse_config("TAGURU_WAL=1\nnot a pair\n").unwrap_err();
        assert!(error.contains("line 2"), "{error}");
    }

    #[test]
    fn config_keys_with_spaces_are_refused() {
        let error = parse_config("TAGURU WAL=1\n").unwrap_err();
        assert!(error.contains("line 1"), "{error}");
    }

    #[test]
    fn an_unspecified_bind_address_probes_via_loopback() {
        assert_eq!(loopback_of("0.0.0.0:8248"), "127.0.0.1:8248");
        assert_eq!(loopback_of("[::]:8248"), "[::1]:8248");
        assert_eq!(loopback_of("127.0.0.1:8248"), "127.0.0.1:8248");
        // Hostnames don't parse as socket addresses; pass them through.
        assert_eq!(loopback_of("localhost:8248"), "localhost:8248");
    }

    #[test]
    fn every_documented_variable_is_a_known_key() {
        // The usage text and the typo lint must agree: a variable
        // documented in --help but missing from KNOWN_KEYS would warn
        // on a perfectly valid config.
        for line in USAGE.lines() {
            let Some(name) = line.split_whitespace().next() else {
                continue;
            };
            if name.starts_with("TAGURU_") {
                assert!(KNOWN_KEYS.contains(&name), "{name} missing from KNOWN_KEYS");
            }
        }
    }
}
