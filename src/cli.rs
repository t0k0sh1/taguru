//! Command-line surface. Hand-rolled on purpose — a default `serve`,
//! three offline subcommands, and one flag do not need an argument
//! framework; the same reasoning that keeps the metrics and BM25
//! in-tree.
//!
//! Exit codes: 0 success · 1 operation failure (corruption found,
//! server error) · 2 usage error.

use std::path::PathBuf;
use std::process::exit;

#[cfg(test)]
use crate::config::KNOWN_KEYS;
use crate::config::{load_config, usage_error};

const USAGE: &str = concat!(
    "taguru ",
    env!("CARGO_PKG_VERSION"),
    " — long-term semantic memory for LLMs

USAGE:
  taguru [serve] [--config FILE] [--take-over] [--replica]
                                        start the HTTP server (the default).
                                        With TAGURU_REPLICATE_URL set and an
                                        empty data directory, the server boots
                                        FROM the bucket: shared files and
                                        pinned contexts hydrate before the
                                        port opens, the rest on first touch.
                                        --take-over acknowledges deposing a
                                        recently-live writer on that bucket
                                        (TAGURU_TAKEOVER=1 says the same).
                                        --replica (TAGURU_REPLICA=1) serves
                                        the bucket lineage read-only instead:
                                        it tails newer manifests continuously,
                                        never claims a generation, and refuses
                                        every write naming the writer — reads
                                        scale horizontally, and the replica
                                        doubles as the warm standby a manual
                                        promotion turns into the next writer
  taguru route [--config FILE]          start the stateless scatter-gather
                                        router over sharded instances:
                                        TAGURU_ROUTE_MAP names a file of
                                        'context = shard-url' lines (plus an
                                        optional '* = shard-url' fallback);
                                        context verbs proxy to the owning
                                        shard, cross-context recall/query/
                                        sources/search and groups span every
                                        shard with the single-instance merge
                                        semantics, and /mcp works unchanged.
                                        No data directory, no state — scale
                                        routers freely behind one LB. Auth is
                                        pass-through: the shards enforce keys
  taguru version                        print the version
  taguru health [--config FILE] [URL]   exit 0 iff a running server's /health
                                        answers 200 — the container
                                        HEALTHCHECK; URL defaults to TAGURU_ADDR
                                        (the config file is applied first, so a
                                        --config deployment probes its own port)
  taguru inspect PATH                   verify a data directory, one .ctx
                                        image, or one .group record offline
                                        (backup check) — the same validating
                                        load the server runs
  taguru estimate --associations N ...  size memory/disk for a target corpus
                                        by building and measuring one
                                        (see: taguru estimate --help)
  taguru import [--dry-run] FILE|DIR... apply JSONL batch files to the data
                                        directory offline — bulk/initial
                                        loads (see: taguru import --help);
                                        the directory lock refuses to run
                                        beside a live server
  taguru export --out DIR [CONTEXT...]  write contexts — and, on a full
                                        export, groups — back out of the data
                                        directory as import batch streams,
                                        the portable backup (see: taguru
                                        export --help); a running server
                                        serves the same at
                                        GET /contexts/{name}/export and
                                        GET /groups/{name}/export
  taguru compact [CONTEXT...]           rewrite context images without the
                                        dead weight the append-only format
                                        accumulates (see: taguru compact
                                        --help); live servers use
                                        POST /contexts/{name}/compact
  taguru restore --out DIR [URL]        materialize a data directory from a
                                        replication bucket's newest complete
                                        generation (see: taguru restore
                                        --help); URL defaults to
                                        TAGURU_REPLICATE_URL — verify the
                                        result with taguru inspect
  taguru extract --context NAME --out DIR FILE|DIR...
                                        decompose documents into batch files
                                        through an OpenAI-compatible chat
                                        model (see: taguru extract --help)
  taguru calibrate --context NAME --probes FILE [--json] [URL]
                                        measure the semantic-floor bands of a
                                        running server's embedding model with
                                        (cue, expected) probe pairs and print
                                        the floor between them — the floor is
                                        a property of the model, remeasured
                                        per switch (see: taguru calibrate
                                        --help); URL defaults to TAGURU_ADDR
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
  TAGURU_PASSAGES_WAL_MAX_BYTES  passage-log backstop, engages only when
                               compaction is stuck; 0 = none (1 GiB)
  TAGURU_REPLICATE_URL         object-storage bucket for continuous
                               replication — s3://, gs://, az://, or
                               file:// — with each cloud's default
                               credential chain; unset = off. Ships every
                               file family and both log lanes, epoch-
                               fenced; restore with taguru restore, or
                               boot an empty directory straight from the
                               bucket (lazy, pinned-first hydration)
  TAGURU_REPLICATE_INTERVAL_MS replication poll cadence, the steady-state
                               RPO knob (1000)
  TAGURU_TAKEOVER              1 = same acknowledgment as serve's
                               --take-over: depose the bucket's newest
                               writer even though it was alive within the
                               last 300s and did not stop cleanly (off)
  TAGURU_REPLICA               1 = serve --replica: read-only, tailing the
                               bucket lineage; TAGURU_REPLICATE_INTERVAL_MS
                               is the poll cadence — staleness is bounded
                               by the writer's shipping lag plus it (off)
  TAGURU_WRITER_URL            where a replica's write-refusal points
                               clients (the writer's own base URL or LB
                               name); unset = the refusal names only the
                               bucket's fence holder
  TAGURU_ROUTE_MAP             route mode only: the context→shard map file,
                               'context = shard-url' per line, # comments,
                               optional '* = shard-url' for contexts the map
                               does not name; edits take a router restart
  TAGURU_API_TOKEN             bearer token; unset = UNAUTHENTICATED
  TAGURU_API_TOKENS            named keys 'ci:tokA,laptop:tokB' — the access
                               log carries the key name; rotate by overlap
  TAGURU_KEY_SCOPES            JSON grants per key name: {\"ci\": \"read\",
                               \"bot\": {\"role\": \"write\", \"contexts\":
                               [\"sake\"]}} — roles read ⊂ write ⊂ admin;
                               unnamed keys keep the full historical grant
  TAGURU_PUBLIC_URL            public base URL; enables OAuth key delegation
                               on /mcp (claude.ai custom connectors)
  TAGURU_MAX_BODY_BYTES        request body cap (8 MiB)
  TAGURU_MCP_MAX_RESULT_BYTES  POST /mcp per-tool-result buffering cap; past
                               it a tool call fails with the export escape
                               hatches named instead of buffering forever
                               (8 MiB)
  TAGURU_REQUEST_TIMEOUT_SECS  per-request budget (30)
  TAGURU_RATE_LIMIT_PER_MIN    per-key request budget; past it 429 +
                               Retry-After (0 = off)
  TAGURU_AUTH_FAIL_LIMIT_PER_MIN  failed-auth attempts per source IP before
                               429 (10; 0 = off; coarse behind a proxy)
  TAGURU_MAX_CONCURRENT_REQUESTS  in-flight request ceiling — past it new
                               requests are shed with 503 + Retry-After;
                               /health and /metrics exempt (256; 0 = off)
  TAGURU_MAX_CONCURRENT_HEAVY_OPS  shared ceiling for audit_vocabulary,
                               audit_drift's include_twins, and
                               compact_context; excess calls are shed with
                               503 + Retry-After (2; 0 = off)
  TAGURU_AUTO_COMPACT          ratio-triggered auto-compaction: each flush
                               tick rebuilds at most the one worst context
                               whose dead ratio exceeds the trigger, behind
                               the heavy-ops ceiling (on; 0/false = manual
                               compaction only)
  TAGURU_AUTO_COMPACT_RATIO    that trigger: compact once dead edges /
                               total edges exceeds this (0.5 — dead weight
                               outgrew live content)
  TAGURU_CROSS_SEARCH_CONCURRENCY  member contexts searched in parallel by
                               a single cross-context (group) query (4)
  TAGURU_EMBED_URL             OpenAI-compatible /embeddings endpoint (off)
  TAGURU_EMBED_MODEL           embedding model name
  TAGURU_EMBED_API_KEY         embedding provider credential
  TAGURU_EMBED_TIMEOUT_SECS    per-attempt provider ceiling (60); a request's
                               remaining budget bounds an attempt further,
                               and transient failures retry twice with backoff
  TAGURU_EMBED_PASSAGES        1/true also embeds stored paragraphs — the
                               semantic passage lane; opt-in spend (off)
  TAGURU_PASSAGE_VECTOR_LIMIT  max embedded rows (paragraphs + their doc2query
                               questions) per context (20000); past it the
                               lexical lane still serves every paragraph
  TAGURU_EMBED_AUTO            1 = refresh embeddings with each flush (off)
  TAGURU_EMBED_PARALLEL        concurrent 128-item chunk dispatch for gloss
                               and passage embedding refresh (1 = old
                               sequential behavior); raise to match the
                               provider's rate limit, not the core count —
                               bounds a single context's refresh only;
                               concurrent refreshes across contexts aren't
                               serialized and multiply this
  TAGURU_SEMANTIC_FLOOR        semantic entry floor when neither the call nor
                               the context sets one (0.35, calibrated for
                               text-embedding-3-large; model-dependent —
                               'taguru calibrate' measures the right value)
  TAGURU_EXTRACT_URL           OpenAI-compatible /chat/completions endpoint,
                               read only by 'taguru extract' (off)
  TAGURU_EXTRACT_MODEL         extraction model name
  TAGURU_EXTRACT_API_KEY       extraction provider credential
  TAGURU_EXTRACT_TIMEOUT_SECS  extract's per-completion budget; local models
                               may need more; 0 = no limit (300)
  TAGURU_EXTRACT_PARALLEL      concurrent chunk completions per document (1)
  RUST_LOG                     log filter, EnvFilter syntax (info)
  TAGURU_LOG_FORMAT            json for JSON log lines (pretty)
  TAGURU_LOG_SEARCHES          1 = per-search event log; cues are memory
                               CONTENT, so this is opt-in (off)
  TAGURU_METRICS_PER_CONTEXT   taguru_context_* gauges on /metrics: 1/all =
                               every context, N = top-N by disk size (off —
                               per-context labels cost Prometheus series)
  OTEL_EXPORTER_OTLP_ENDPOINT  turns on OTLP/HTTP span export (off)

EXIT CODES: 0 ok · 1 failure or corruption found · 2 usage error
"
);

/// What `main` should do once the arguments are understood. Offline
/// subcommands never return — they print and exit before any runtime,
/// listener, or telemetry exists; the two server modes come back here
/// so `main` can load the config file first.
pub enum Command {
    Serve(ServeArgs),
    Route(RouteArgs),
}

impl Command {
    /// The config file to load into the environment before the server
    /// boots, whichever mode is starting.
    pub fn config(&self) -> Option<&PathBuf> {
        match self {
            Command::Serve(args) => args.config.as_ref(),
            Command::Route(args) => args.config.as_ref(),
        }
    }
}

/// `taguru route`'s settings: the config file alone — the map itself
/// rides `TAGURU_ROUTE_MAP` like every other knob rides a variable.
pub struct RouteArgs {
    pub config: Option<PathBuf>,
}

pub struct ServeArgs {
    pub config: Option<PathBuf>,
    /// `--take-over`: the operator's stated intent to depose a
    /// recently-live writer on the replication bucket (see
    /// `crate::hydrate`'s takeover guard). `TAGURU_TAKEOVER=1` says
    /// the same thing where flags are awkward (a container manifest).
    pub take_over: bool,
    /// `--replica`: serve the replication bucket's lineage read-only,
    /// tailing it continuously (issue #129) — `TAGURU_REPLICA=1` says
    /// the same.
    pub replica: bool,
}

/// Parses the process arguments, running and exiting for everything
/// except the server modes (`serve`, `route`), whose settings it
/// returns.
pub fn dispatch() -> Command {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => Command::Serve(parse_serve(&[])),
        Some("serve") => Command::Serve(parse_serve(&args[1..])),
        Some("--config") => Command::Serve(parse_serve(&args)),
        Some("route") => Command::Route(parse_route(&args[1..])),
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
        Some("import") => exit(crate::ingest::run(&args[1..])),
        Some("export") => exit(crate::export::run(&args[1..])),
        Some("compact") => exit(crate::compact::run(&args[1..])),
        Some("restore") => exit(crate::ship::run(&args[1..])),
        Some("extract") => exit(crate::extract::run(&args[1..])),
        Some("calibrate") => exit(crate::calibrate::run(&args[1..])),
        Some(other) => {
            eprintln!("taguru: unknown argument '{other}' — try 'taguru --help'");
            exit(2)
        }
    }
}

/// `route` takes one optional `--config FILE` and nothing else — the
/// shard map is a variable (`TAGURU_ROUTE_MAP`), not an argument.
fn parse_route(args: &[String]) -> RouteArgs {
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
            other => usage_error(&format!("'route' does not take '{other}'")),
        }
    }
    let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    RouteArgs { config }
}

/// `serve` takes one optional `--config FILE`, the `--take-over`
/// acknowledgment, and the `--replica` role flag.
fn parse_serve(args: &[String]) -> ServeArgs {
    let mut config = None;
    let mut take_over = false;
    let mut replica = false;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => usage_error("--config given twice"),
                None => usage_error("--config needs a file path"),
            },
            "--take-over" => take_over = true,
            "--replica" => replica = true,
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
    ServeArgs {
        config,
        take_over,
        replica,
    }
}

/// `taguru health [--config FILE] [URL]`: one GET against a running
/// server's /health, exit 0 iff it answers 200. This exists for
/// container HEALTHCHECKs — a scratch image has no curl, but it always
/// has taguru itself. /health is exempt from bearer auth, so no token
/// is needed here.
///
/// The config file (`--config`, or `TAGURU_CONFIG` like serve) is
/// applied before the default URL is resolved: in a deployment whose
/// TAGURU_ADDR lives in that file, the probe must aim at the port the
/// server actually bound, not at the built-in default — a health
/// check that asks the wrong door reports a healthy server unhealthy
/// forever.
fn health(args: &[String]) -> i32 {
    let mut config: Option<PathBuf> = None;
    let mut explicit_url: Option<String> = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    "usage: taguru health [--config FILE] [URL]   \
                     exit 0 iff GET URL/health answers 200"
                );
                return 0;
            }
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => usage_error("--config given twice"),
                None => usage_error("--config needs a file path"),
            },
            flag if flag.starts_with('-') => {
                usage_error(&format!("'health' does not take '{flag}'"))
            }
            url => {
                if explicit_url
                    .replace(url.trim_end_matches('/').to_string())
                    .is_some()
                {
                    usage_error(&format!("'health' takes one optional URL, got '{url}'"));
                }
            }
        }
    }
    // The flag beats the variable, both beat the built-in default —
    // the same rule serve applies. Sound here for the same reason:
    // dispatch() runs before any runtime or second thread exists.
    let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    if let Some(path) = &config {
        load_config(path);
    }
    let base = match explicit_url {
        Some(url) => url,
        None => match default_base_url() {
            Ok(url) => url,
            Err(error) => {
                eprintln!("taguru: health: {error}");
                return 2;
            }
        },
    };
    let url = format!("{base}/health");
    // The agent timeout stays under HEALTHCHECK's own 5s deadline so
    // the verdict (and its message) comes from here, not from a kill.
    // Error statuses come back as responses so their body reaches the
    // verdict message.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(4)))
        .http_status_as_error(false)
        .build()
        .into();
    match agent.get(&url).call() {
        Ok(mut response) if response.status().as_u16() < 400 => {
            let body = response.body_mut().read_to_string().unwrap_or_default();
            println!("{}", body.trim());
            0
        }
        Ok(mut response) => {
            let code = response.status().as_u16();
            let body = response.body_mut().read_to_string().unwrap_or_default();
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
///
/// Errs when that resolves to port 0: TAGURU_ADDR documents port 0 as
/// "pick free" (the OS assigns an ephemeral port at bind time), but
/// that assignment is invisible to a second, later process — probing
/// port 0 itself always fails, reporting a healthy server unhealthy
/// forever rather than just once.
///
/// `calibrate` resolves its target the same way, for the same reason:
/// one rule for "which server does a CLI verb mean".
pub(crate) fn default_base_url() -> Result<String, String> {
    let addr = std::env::var("TAGURU_ADDR").unwrap_or_else(|_| "127.0.0.1:8248".to_string());
    base_url_for(&addr)
}

fn base_url_for(addr: &str) -> Result<String, String> {
    let loopback = loopback_of(addr);
    if loopback.ends_with(":0") {
        return Err(format!(
            "TAGURU_ADDR ({addr}) binds to port 0 (OS-assigned) — the actual port \
             can't be discovered from here; pass the server's real URL explicitly: \
             'taguru health http://host:PORT'"
        ));
    }
    Ok(format!("http://{loopback}"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unspecified_bind_address_probes_via_loopback() {
        assert_eq!(loopback_of("0.0.0.0:8248"), "127.0.0.1:8248");
        assert_eq!(loopback_of("[::]:8248"), "[::1]:8248");
        assert_eq!(loopback_of("127.0.0.1:8248"), "127.0.0.1:8248");
        // Hostnames don't parse as socket addresses; pass them through.
        assert_eq!(loopback_of("localhost:8248"), "localhost:8248");
    }

    #[test]
    fn a_port_0_bind_address_refuses_to_guess_the_real_port() {
        let error =
            base_url_for("0.0.0.0:0").expect_err("port 0 cannot resolve to a probeable URL");
        assert!(error.contains("port 0"), "{error}");
        // A concrete port still resolves normally.
        assert_eq!(
            base_url_for("0.0.0.0:8248").unwrap(),
            "http://127.0.0.1:8248"
        );
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
