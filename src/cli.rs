//! Command-line surface. Hand-rolled on purpose — a default `serve`,
//! three offline subcommands, and one flag do not need an argument
//! framework; the same reasoning that keeps the metrics and BM25
//! in-tree.
//!
//! Exit codes: 0 success · 1 operation failure (corruption found,
//! server error) · 2 usage error · 3 `taguru evaluate` completed and a
//! `--thresholds` bound was violated (ADR 0004 §5) — the one verb in
//! this tree that returns it.

use std::path::PathBuf;
use std::process::exit;

#[cfg(test)]
use crate::config::KNOWN_KEYS;
use crate::config::{load_config, usage_error};
use crate::remote::default_base_url;

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
  taguru router [--config FILE]         start the stateless scatter-gather
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
                                        pass-through: the shards enforce keys.
                                        'route' still works as a deprecated
                                        alias (a warning goes to stderr)
  taguru version                        print this binary's version; a running
                                        server reports its own version in its
                                        GET /health response body
  taguru health [--config FILE] [--url URL] [URL]
                                        exit 0 iff a running server's /health
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
                                        beside a live server, or point
                                        import itself at one with --url
  taguru export --out DIR [CONTEXT...]  write contexts — and, on a full
                                        export, groups — back out of the data
                                        directory as import batch streams,
                                        the portable backup (see: taguru
                                        export --help); a running server
                                        serves the same at
                                        GET /contexts/{name}/export and
                                        GET /groups/{name}/export, or point
                                        export itself at one with --url
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
  taguru benchmark extract --models FILE --context NAME --out DIR CORPUS_DIR
                                        run taguru extract across a model
                                        matrix, one subprocess per (model,
                                        run) cell, and assemble manifest.json/
                                        runs/*.jsonl (ADR 0003; see: taguru
                                        benchmark extract --help)
  taguru benchmark compare [--with-text] RESULTS_DIR
                                        derive measurements.json/
                                        measurements.csv and differences.jsonl
                                        from a finished results directory
                                        (ADR 0003 §9.3/§9.4; see: taguru
                                        benchmark compare --help)
  taguru benchmark search --eval FILE [--url URL] [--config FILE]
                          [--run N] [--context-prefix NAME] [--skip-import]
                          RESULTS_DIR
                                        build one per-model corpus from a
                                        finished results directory and
                                        compare their search results (ADR
                                        0003 §11; see: taguru benchmark
                                        search --help)
  taguru evaluate --eval FILE --context NAME [--url URL] [--config FILE]
                  [--thresholds FILE] [--assembly] [--max-items N]
                  [--max-bytes N] [--max-tokens N] [--rerank MODEL]
                                        run eval.jsonl's cases against one
                                        context's live retrieval endpoints
                                        (passage search, then resolve/query)
                                        and write evaluation.json — a quality
                                        gate that calls no answer-generation
                                        LLM (ADR 0004). Report-only (exit 0)
                                        without --thresholds; with it, exits
                                        3 on a violated bound. --assembly
                                        swaps the passage lane for budgeted
                                        evidence assembly (ADR 0006 §14) at
                                        equal --max-items/--max-bytes/
                                        --max-tokens against a baseline run
                                        (see: taguru evaluate --help)
  taguru evaluate compare BASE.json HEAD.json [--out FILE]
                                        compare two evaluation.json runs and
                                        write changes.jsonl: improved/
                                        regressed/added/removed cases (ADR
                                        0004 §9.2; see: taguru evaluate
                                        compare --help)
  taguru anchoring BATCH_OR_DIR... [--vocabulary PATH] [--json FILE]
                                        judge extraction batches against their
                                        own passage text: the anchoring rate
                                        (subject and object present in the
                                        cited paragraph, else the passage;
                                        strict and alias-group variants) and
                                        locator validity — offline, no server
                                        (see: taguru anchoring --help)
  taguru calibrate --context NAME --probes FILE [--json] [--url URL] [URL]
                                        measure the semantic-floor bands of a
                                        running server's embedding model with
                                        (cue, expected) probe pairs and print
                                        the floor between them — the floor is
                                        a property of the model, remeasured
                                        per switch (see: taguru calibrate
                                        --help); --url and the positional URL
                                        are aliases, name the target either
                                        way; unnamed, it defaults to
                                        TAGURU_ADDR
  taguru communities --context NAME [--into NAME] [--dry-run] [--json]
                     [--url URL] [URL]
  taguru communities --group NAME [--dry-run] [--json] [--url URL] [URL]
                                        derive (or refresh) a community-
                                        summaries artifact from a running
                                        server's context: server-side
                                        detection, LLM summaries of what
                                        changed only, written back as an
                                        ordinary context (default
                                        'NAME::communities', or --into's
                                        name) that
                                        POST /contexts/{name}/communities/search
                                        serves with a staleness verdict;
                                        --group derives one artifact per
                                        member context instead, transitively,
                                        with no cross-context merge (see:
                                        taguru communities --help); --url and
                                        the positional URL are aliases, name
                                        the target either way; unnamed, it
                                        defaults to TAGURU_ADDR
  taguru consolidation --context NAME [--checks LIST] [--into NAME]
                       [--dry-run] [--url URL] [URL]
                                        judge a running server's
                                        consolidation-audit candidates
                                        (merge/contradiction/staleness,
                                        ADR 0012) with the extract LLM and
                                        store the judgments as an ordinary
                                        derived context (default
                                        'NAME::consolidation') keyed by each
                                        candidate's fingerprint — re-runs
                                        reuse stored judgments, dismissals
                                        included, until the evidence moves;
                                        proposals only, applied by the
                                        operator through ordinary writes
                                        (see: taguru consolidation --help)
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
  TAGURU_RETRIEVAL_CACHE_BYTES  exact-match result cache for recall/query/
                               passage search, invalidated by the revision
                               counters (32 MiB; 0 = off; below ~16 KiB is
                               not a smaller cache but pure cost — no
                               response fits, so keys keep getting minted
                               and only ever miss; a boot-time warning
                               fires under that floor)
  TAGURU_SEMANTIC_CACHE_THRESHOLD  semantic tier over the exact cache,
                               passage search only: a paraphrased query
                               clearing this cosine floor (plus a negation/
                               number/entity guard) serves the earlier
                               query's cached result; needs the exact cache
                               and TAGURU_EMBED_PASSAGES (unset = off)
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
  TAGURU_ROUTE_MAP             router mode only: the context→shard map file,
                               'context = shard-url' per line, # comments,
                               optional '* = shard-url' for contexts the map
                               does not name; edits hot-reload like the auth
                               table (SIGHUP, or the file's own watch, ~5s)
                               — a broken edit keeps the previous map
  TAGURU_API_TOKEN             bearer token; unset = UNAUTHENTICATED
  TAGURU_API_TOKENS            named keys 'ci:tokA,laptop:tokB' — the access
                               log carries the key name; rotate by overlap
  TAGURU_KEY_SCOPES            JSON grants per key name: {\"ci\": \"read\",
                               \"bot\": {\"role\": \"write\", \"contexts\":
                               [\"sake\"]}} — roles read ⊂ write ⊂ admin;
                               unnamed keys keep the full historical grant
                               These three (the auth table) hot-reload:
                               SIGHUP, or an edited --config file (picked
                               up within ~5s), swaps them live — fail
                               closed, so a broken edit keeps the previous
                               table and a reload can never disarm auth.
                               Everything else stays boot-time.
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
  TAGURU_CONTEXT_QUOTAS        JSON ceilings per context name:
                               {\"sake\": {\"storage_bytes\": 1073741824,
                               \"cache_bytes\": 134217728}} — storage refuses
                               growth writes at the ceiling (507; retract/
                               compact/delete stay open), cache evicts the
                               over-share context first under pressure; a
                               broken declaration refuses boot (off)
  TAGURU_CROSS_SEARCH_CONCURRENCY  member contexts searched in parallel by
                               a single cross-context (group) query (4)
  TAGURU_EMBED_URL             OpenAI-compatible /embeddings endpoint (off);
                               'local' runs the model in-process (default
                               builds; slim --no-default-features builds and
                               the Docker image warn and keep the lane off)
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
  TAGURU_RERANK_URL            Cohere/Jina-compatible /rerank endpoint for
                               opt-in evidence reranking (#307); off keeps
                               POST /contexts/{name}/evidence fully
                               deterministic (off)
  TAGURU_RERANK_MODEL          reranker model name
  TAGURU_RERANK_API_KEY        reranker provider credential
  TAGURU_RERANK_TIMEOUT_SECS   per-attempt provider ceiling (5); a request's
                               remaining budget bounds an attempt further,
                               and one transient failure retries with backoff
  TAGURU_EXTRACT_URL           OpenAI-compatible /chat/completions endpoint,
                               read only by 'taguru extract' (off)
  TAGURU_EXTRACT_MODEL         extraction model name
  TAGURU_EXTRACT_API_KEY       extraction provider credential
  TAGURU_EXTRACT_TIMEOUT_SECS  extract's per-completion budget; local models
                               may need more; 0 = no limit (300)
  TAGURU_EXTRACT_PARALLEL      concurrent chunk completions per document (1)
  TAGURU_EXTRACT_FACT_BUDGET   default for --fact-budget (0, off)
  TAGURU_EXTRACT_MAX_ATTEMPTS  total attempts at valid JSON per chunk, 1-10
                               (2)
  TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES  cap a corrective turn's replay of
                               the model's own prior bad answer to this many
                               bytes; 0 omits it entirely (unset: replay it
                               in full)
  TAGURU_EXTRACT_STRUCTURED_OUTPUT  default for --structured-output (off)
  TAGURU_EXTRACT_MAX_OUTPUT_TOKENS  default for --max-output-tokens (unset)
  TAGURU_EXTRACT_ESCALATION_FACTOR  cap of the escalated resend, as a multiple
                      of --max-output-tokens; 0 = uncapped (2)
  TAGURU_EXTRACT_RUNAWAY_RATIO  fail a piece whose length-limited answer
                      outgrows this multiple of its bytes; 0 = off (8)
  TAGURU_EXTRACT_CHUNK_BYTES  default for --chunk-bytes (24576)
  TAGURU_EXTRACT_CHUNK_CONTEXT  default for --chunk-context (off)
  TAGURU_EXTRACT_TRACE_ATTEMPTS  `off` disables the per-document attempts log (ADR 0025)
  TAGURU_EXTRACT_LOSSY         default for --lossy (0/false)
  TAGURU_EXTRACT_CANDIDATES    default for --candidates (0/false)
  TAGURU_EXTRACT_VOCABULARY    default for --vocabulary (unset, off)
  TAGURU_EXTRACT_COVERAGE      default for --coverage (0/false)
  TAGURU_EXTRACT_DIAGNOSTICS   default for --diagnostics-out (unset, off)
  TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES  attach the model's raw answer text
                               to each diagnostics record, capped to this
                               many bytes; unset or 0 = never attach it
                               (metadata only)
  TAGURU_EXTRACT_SCHEMA        default for extract's --schema (unset, off)
  TAGURU_EXTRACT_REPLAY        default for extract's --replay (off)
  TAGURU_EXTRACT_REPLAY_FROM   default for extract's --replay-from
                               (OUT/.extract-trace)
  RUST_LOG                     log filter, EnvFilter syntax (info)
  TAGURU_LOG_FORMAT            json for JSON log lines (pretty)
  TAGURU_LOG_SEARCHES          1 = per-search event log; cues are memory
                               CONTENT, so this is opt-in (off)
  TAGURU_METRICS_PER_CONTEXT   taguru_context_* gauges on /metrics: 1/all =
                               every context, N = top-N by disk size (off —
                               per-context labels cost Prometheus series)
  OTEL_EXPORTER_OTLP_ENDPOINT  turns on OTLP/HTTP span export (off)

EXIT CODES: 0 ok · 1 failure or corruption found · 2 usage error ·
            3 threshold violation (taguru evaluate --thresholds only)
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

/// `taguru router`'s settings: the config file alone — the map itself
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
/// except the server modes (`serve`, `router`), whose settings it
/// returns.
pub fn dispatch() -> Command {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => Command::Serve(parse_serve(&[])),
        Some("serve") => Command::Serve(parse_serve(&args[1..])),
        Some("--config") | Some("--take-over") | Some("--replica") => {
            Command::Serve(parse_serve(&args))
        }
        Some("router") => Command::Route(parse_route(&args[1..])),
        // Deprecated alias (issue #248 item 9): the module and every
        // design doc call this "the router"; the subcommand name
        // itself lagged behind. Removing the alias is its own
        // follow-up issue, filed once the rename lands.
        Some("route") => {
            eprintln!(
                "taguru: 'route' is a deprecated alias for 'router' and will be removed in a \
                 future release"
            );
            Command::Route(parse_route(&args[1..]))
        }
        Some("version") => {
            refuse_extras("version", &args[1..]);
            println!("taguru {}", env!("CARGO_PKG_VERSION"));
            exit(0)
        }
        Some("help") | Some("--help") | Some("-h") => {
            print!("{USAGE}");
            exit(0)
        }
        Some("benchmark") => exit(crate::benchmark::run(&args[1..])),
        Some("evaluate") => exit(crate::evaluate::run(&args[1..])),
        Some("health") => exit(health(&args[1..])),
        Some("inspect") => exit(crate::inspect::run(&args[1..])),
        Some("estimate") => exit(crate::estimate::run(&args[1..])),
        Some("import") => exit(crate::ingest::run(&args[1..])),
        Some("export") => exit(crate::export::run(&args[1..])),
        Some("compact") => exit(crate::compact::run(&args[1..])),
        Some("restore") => exit(crate::ship::run(&args[1..])),
        Some("extract") => exit(crate::extract::run(&args[1..])),
        Some("calibrate") => exit(crate::calibrate::run(&args[1..])),
        Some("anchoring") => exit(crate::anchoring::run(&args[1..])),
        Some("communities") => exit(crate::communities::run(&args[1..])),
        Some("consolidation") => exit(crate::consolidation::run(&args[1..])),
        Some(other) => {
            eprintln!("taguru: unknown argument '{other}' — try 'taguru --help'");
            exit(2)
        }
    }
}

/// `router` (or its deprecated alias `route`) takes one optional
/// `--config FILE` and nothing else — the shard map is a variable
/// (`TAGURU_ROUTE_MAP`), not an argument.
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
            other => usage_error(&format!("'router' does not take '{other}'")),
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

/// `taguru health [--config FILE] [--url URL] [URL]`: one GET against
/// a running server's /health, exit 0 iff it answers 200. This exists
/// for container HEALTHCHECKs — a scratch image has no curl, but it
/// always has taguru itself. /health is exempt from bearer auth, so no
/// token is needed here.
///
/// `--url` is an alias for the positional form (issue #248 item 1) —
/// either names the target, never both. The config file (`--config`,
/// or `TAGURU_CONFIG` like serve) is applied before the default URL is
/// resolved: in a deployment whose TAGURU_ADDR lives in that file, the
/// probe must aim at the port the server actually bound, not at the
/// built-in default — a health check that asks the wrong door reports
/// a healthy server unhealthy forever.
fn health(args: &[String]) -> i32 {
    let mut config: Option<PathBuf> = None;
    let mut explicit_url: Option<String> = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    "usage: taguru health [--config FILE] [--url URL] [URL]   \
                     exit 0 iff GET URL/health answers 200 — --url and the \
                     positional form are aliases, name the target either way"
                );
                return 0;
            }
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => usage_error("--config given twice"),
                None => usage_error("--config needs a file path"),
            },
            "--url" => match rest.next() {
                Some(url) if explicit_url.is_none() && !url.starts_with('-') => {
                    explicit_url = Some(url.trim_end_matches('/').to_string());
                }
                Some(_) if explicit_url.is_none() => usage_error("--url needs a server URL"),
                Some(_) => usage_error("'health' takes either --url or a positional URL, not both"),
                None => usage_error("--url needs a server URL"),
            },
            flag if flag.starts_with('-') => {
                usage_error(&format!("'health' does not take '{flag}'"))
            }
            url => {
                if explicit_url
                    .replace(url.trim_end_matches('/').to_string())
                    .is_some()
                {
                    usage_error("'health' takes either --url or a positional URL, not both");
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
        // Exactly 200, matching this fn's own doc ("exit 0 iff a
        // running server's /health answers 200") — a 2xx/3xx that
        // isn't literally 200 (a 204/202 from a load balancer or
        // service-mesh sidecar sitting in front of the port, say) is
        // not this server itself answering healthy, and must not read
        // as one to a HEALTHCHECK.
        Ok(mut response) if response.status().as_u16() == 200 => {
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

fn refuse_extras(command: &str, extras: &[String]) {
    if let Some(extra) = extras.first() {
        usage_error(&format!("'{command}' takes no argument, got '{extra}'"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_documented_variable_is_a_known_key() {
        // The usage text and the typo lint must agree: a variable
        // documented in --help but missing from KNOWN_KEYS would warn
        // on a perfectly valid config.
        crate::config::assert_usage_vars_are_known_keys(USAGE);
    }

    #[test]
    fn every_known_key_is_documented() {
        // The reverse of `every_documented_variable_is_a_known_key`: the
        // ENVIRONMENT section claims to list "every knob", so a key added
        // to KNOWN_KEYS without a matching --help entry must fail here
        // instead of silently going undocumented. `TAGURU_CONFIG` is
        // documented in the CONFIGURATION FILE section as prose
        // (`TAGURU_CONFIG=FILE`), not an `ENVIRONMENT` line, so this
        // checks for the identifier anywhere in `USAGE` rather than
        // restricting to a line's first token like the reverse test does.
        for name in KNOWN_KEYS {
            assert!(
                crate::config::documented_as_whole_word(USAGE, name),
                "{name} is in KNOWN_KEYS but not documented in --help"
            );
        }
    }
}
