mod api;
mod auth;
mod bm25;
mod cli;
mod compact;
mod config;
#[cfg(test)]
mod context_proptest;
// The same file lib.rs includes — see src/crc32c.rs for why the
// checksum primitive is dual-included instead of exported.
mod crc32c;
mod embedding;
mod env;
mod estimate;
mod export;
mod extract;
mod groups;
mod hash;
mod hydrate;
mod ingest;
mod inspect;
mod limits;
mod mcp;
mod metrics;
mod oauth;
mod oauth_http;
mod paragraph;
mod passages;
mod registry;
mod remote_mcp;
mod replica;
mod ship;
mod storage;
mod trace;
mod wal;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::routing::{get, post};
use embedding::EmbeddingProvider;
use env::{
    DEFAULT_MAX_BODY_BYTES, DEFAULT_MCP_MAX_RESULT_BYTES, env_bool, env_number,
    needs_off_loopback_warning, resolve_body_bytes, resolve_flush_secs, resolve_heavy_ops,
    resolve_mcp_max_result_bytes, resolve_per_minute, resolve_timeout_secs,
};
use registry::AppState;
use taguru::deadline::Deadline;
use tokio::net::TcpListener;
use tower_http::catch_panic::CatchPanicLayer;
use tracing::{error, info, warn};

/// Configuration comes from the environment — or from a KEY=VALUE file
/// (`--config FILE` / `TAGURU_CONFIG=FILE`, the `docker --env-file`
/// dialect; real environment variables win over the file):
/// - `TAGURU_DATA_DIR`: where context images and sidecars live (default
///   `data`). Disk is the source of truth; memory is a cache over it.
/// - `TAGURU_CACHE_BYTES`: resident budget for unpinned loaded contexts
///   (default 512 MiB). Past it, least-recently-used contexts are
///   flushed and dropped; pinned contexts live outside the budget.
/// - `TAGURU_FLUSH_SECS`: how often dirty contexts are persisted (default
///   5). With the WAL on this is image-freshness cadence; writes also
///   persist on eviction and on graceful shutdown (SIGINT/SIGTERM).
/// - `TAGURU_WAL`: per-context write-ahead log for acknowledged graph
///   writes (default on). `0`/`false` restores the flush-interval
///   crash-loss window.
/// - `TAGURU_ADDR`: bind address (default 127.0.0.1:8248 — "TAGU" on a
///   phone keypad, clear of the defaults Taguru tends to sit beside:
///   3000 Next.js/Grafana, 8000 FastAPI/Chroma, 8080 Weaviate, 11434
///   Ollama, 6333 Qdrant. Port 0 picks a free port and the resolved
///   address is printed).
/// - `RUST_LOG`: log filter (default `info`), standard EnvFilter syntax.
/// - `TAGURU_LOG_FORMAT`: `json` for one JSON object per log line;
///   anything else keeps the human-readable format. Logs go to stderr.
/// - `TAGURU_LOG_SEARCHES`: `1` emits one `taguru::search` event line
///   per retrieval (context, op, cue, hits) for keyword analysis in
///   the log pipeline. Off by default: cues are memory content, and
///   the standard log stream carries no content.
/// - `TAGURU_MAX_CONCURRENT_HEAVY_OPS`: shared ceiling for concurrent
///   vocabulary audits (including a drift audit's `include_twins`) and
///   per-context compactions (default 2; 0 disables). Excess calls are
///   shed immediately with 503 + `Retry-After`.
/// - `OTEL_EXPORTER_OTLP_ENDPOINT` (or the `_TRACES_` variant): turns
///   on OTLP/HTTP span export — one span per request, parented from an
///   inbound `traceparent` or `X-Amzn-Trace-Id`, `trace_id` stamped
///   into the access log. The other standard `OTEL_*` variables
///   (service name, headers, batch cadence) apply. Unset: no tracing.
fn main() {
    // Argument handling runs before ANY runtime, listener, or
    // telemetry exists: `taguru version` and friends must never start
    // a server (`--help` used to).
    let serve_args = cli::dispatch();
    // The config file becomes environment variables HERE, while the
    // process is still single-threaded (set_var's soundness condition)
    // and before init_telemetry reads RUST_LOG/OTEL_* — file values
    // must steer those too.
    if let Some(path) = &serve_args.config {
        config::load_config(path);
    }
    serve(serve_args);
}

#[tokio::main]
async fn serve(serve_args: cli::ServeArgs) {
    // The subscriber must exist before anything can log — the
    // env_number warnings just below would otherwise be dropped
    // silently (tracing has no default subscriber and no buffering).
    let tracer_provider = init_telemetry();

    let config = registry::BootConfig::from_env();
    let flush_secs = resolve_flush_secs(env_number("TAGURU_FLUSH_SECS", 5));
    let max_body_bytes =
        resolve_body_bytes(env_number("TAGURU_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES));
    let mcp_max_result_bytes = resolve_mcp_max_result_bytes(env_number(
        "TAGURU_MCP_MAX_RESULT_BYTES",
        DEFAULT_MCP_MAX_RESULT_BYTES,
    ));
    let timeout_secs = resolve_timeout_secs(env_number("TAGURU_REQUEST_TIMEOUT_SECS", 30));
    let rate_per_minute = resolve_per_minute(
        "TAGURU_RATE_LIMIT_PER_MIN",
        env_number("TAGURU_RATE_LIMIT_PER_MIN", 0),
    );
    let rate_limiter = Arc::new(limits::RateLimiter::new(rate_per_minute));
    let rate_limit_disabled = rate_limiter.is_disabled();
    if !rate_limit_disabled {
        info!(per_minute = rate_per_minute, "per-key rate limit enabled");
    }
    // The in-flight ceiling defaults ON: without it nothing bounds
    // concurrent request growth under overload but the OS itself. 256
    // is far above any healthy load for a single node while still
    // being a real valve; 0 disables.
    let max_concurrent = env_number("TAGURU_MAX_CONCURRENT_REQUESTS", 256);
    if max_concurrent > 0 {
        info!(
            max_in_flight = max_concurrent,
            "in-flight request ceiling enabled (past it: 503 + Retry-After)"
        );
    }
    let max_concurrent_heavy = resolve_heavy_ops(env_number("TAGURU_MAX_CONCURRENT_HEAVY_OPS", 2));
    if max_concurrent_heavy > 0 {
        info!(
            max_concurrent_heavy,
            "heavy-operation ceiling enabled for vocabulary audits and context compactions"
        );
    }
    // Failed-auth attempts are throttled per source IP so the gate
    // cannot be brute-forced for free (default 10/min; 0 disables).
    let auth_fail_per_minute = resolve_per_minute(
        "TAGURU_AUTH_FAIL_LIMIT_PER_MIN",
        env_number("TAGURU_AUTH_FAIL_LIMIT_PER_MIN", 10),
    );
    let fail_limiter = Arc::new(limits::RateLimiter::new(auth_fail_per_minute));
    if !fail_limiter.is_disabled() {
        info!(
            per_minute = auth_fail_per_minute,
            "failed-auth throttle enabled (per source IP)"
        );
    }

    // Misconfigured credentials refuse to boot: a keyring that
    // silently dropped an entry would surface as an auth hole.
    let keyring = match auth::Keyring::parse(
        std::env::var("TAGURU_API_TOKEN").ok(),
        std::env::var("TAGURU_API_TOKENS").ok(),
    )
    .and_then(|mut keyring| {
        // Scopes are part of the same credential story: a grant that
        // silently failed to arm is an authorization hole.
        keyring.apply_scopes(std::env::var("TAGURU_KEY_SCOPES").ok().as_deref())?;
        Ok(keyring)
    }) {
        Ok(keyring) => Arc::new(keyring),
        Err(error) => {
            tracing::error!(%error, "refusing to start with broken credentials");
            std::process::exit(1);
        }
    };
    let auth_configured = !keyring.is_disabled();
    if auth_configured {
        info!(
            keys = keyring.key_count(),
            scoped = keyring.scoped_key_count(),
            "bearer auth enabled"
        );
    } else {
        warn!(
            "TAGURU_API_TOKEN(S) is not set: the API accepts UNAUTHENTICATED requests \
             (fine on localhost, never on an exposed address)"
        );
    }

    // TAGURU_PUBLIC_URL turns on OAuth for the remote MCP transport —
    // the URL doubles as issuer and audience, so OAuth without it
    // would mint tokens for an address nobody can verify.
    let oauth = match std::env::var("TAGURU_PUBLIC_URL") {
        Ok(url) if !url.trim().is_empty() => {
            if !auth_configured {
                tracing::error!(
                    "TAGURU_PUBLIC_URL is set but no API key is: the OAuth consent \
                     step delegates an EXISTING key, so configure TAGURU_API_TOKEN(S) first"
                );
                std::process::exit(1);
            }
            let oauth = Arc::new(oauth::Oauth::open(&url, &config.data_dir));
            info!(issuer = %oauth.public_url(), "oauth enabled for remote MCP");
            Some(oauth)
        }
        Ok(_) => {
            // Present-but-blank is almost always a templating accident
            // (e.g. `${PUBLIC_URL:-}` with an unset upstream var), not an
            // intentional opt-out — unlike every other env var this file
            // reads, silently treating it as absent would leave OAuth
            // disabled with no boot-time signal that anything was wrong.
            warn!(
                "TAGURU_PUBLIC_URL is set but empty: treating OAuth as disabled — \
                 unset the variable entirely if that's intended"
            );
            None
        }
        Err(_) => None,
    };

    let embedder = embedding::HttpEmbeddings::from_env();
    if let Some(embedder) = &embedder {
        info!(model = embedder.model(), "semantic entry tier enabled");
        // The provider call runs in block_in_place, which the request
        // timeout cannot preempt — a budget under the provider ceiling
        // 408s spuriously whenever the provider is merely slow. And one
        // embed() makes up to MAX_EMBED_ATTEMPTS sequential attempts on
        // transient failures, so the worst-case hold is that many times
        // the per-attempt ceiling, not one. This trap used to live only
        // in a code comment; size it and say it at boot.
        let embed_timeout = env_number("TAGURU_EMBED_TIMEOUT_SECS", 60).max(1);
        let worst_case = embed_timeout.saturating_mul(embedding::MAX_EMBED_ATTEMPTS);
        if timeout_secs <= worst_case {
            warn!(
                request_timeout_secs = timeout_secs,
                embed_timeout_secs = embed_timeout,
                embed_worst_case_secs = worst_case,
                embed_attempts = embedding::MAX_EMBED_ATTEMPTS,
                "TAGURU_REQUEST_TIMEOUT_SECS is at or under the embedding provider's \
                 worst-case hold (TAGURU_EMBED_TIMEOUT_SECS × retries) — raise it above \
                 that, or slow provider calls will answer 408 after the work was \
                 already done"
            );
        }
        // The same worst-case hold applies to shutdown: the provider call
        // runs in block_in_place, so a SIGTERM arriving mid-embed leaves
        // axum's graceful drain waiting on it, not the request timeout
        // above. A container orchestrator's own stop grace period
        // (Kubernetes terminationGracePeriodSeconds, Docker
        // stop_grace_period) is invisible to this process, so it can't be
        // checked the way TAGURU_REQUEST_TIMEOUT_SECS was above — only
        // said, once, here.
        info!(
            embed_worst_case_secs = worst_case,
            "an in-flight embed call can hold graceful shutdown's drain for up to this \
             long — size the container orchestrator's stop grace period at least this \
             large, or a SIGKILL cuts the drain short (every write is still safe in the \
             WAL; only the final flush and unwritten usage counters are lost)"
        );
    }
    let auto_embed = embedder.is_some() && env_bool("TAGURU_EMBED_AUTO", false);
    if auto_embed {
        info!("auto embedding refresh enabled (runs with each flush)");
    }
    let embedder = embedder.map(|provider| Arc::new(provider) as Arc<dyn EmbeddingProvider>);
    // /protocol tells connecting agents which optional tiers are live;
    // the model name is captured here because the provider itself moves
    // into the state.
    let protocol_trailer = api::protocol_trailer(
        embedder.as_ref().map(|provider| provider.model()),
        auto_embed,
    );
    // MCP initialize hands out exactly what GET /protocol serves —
    // both transports, one manual.
    let mcp_instructions = Arc::new(api::protocol_text(protocol_trailer.as_deref()));
    if let Some(floor) = config.semantic_floor {
        info!(floor, "semantic floor default recalibrated");
    }

    // Replication: resolve the URL BEFORE boot so a typo'd
    // TAGURU_REPLICATE_URL refuses to start (a misspelled bucket
    // silently shipping nowhere is the one failure an operator cannot
    // see coming), while an unreachable-but-well-formed bucket only
    // degrades — the shipper retries in the background.
    let replicate =
        ship::ReplicateConfig::from_env().map(|replicate| match ship::open_store(&replicate.url) {
            Ok((store, root)) => (replicate, store, root),
            Err(error) => {
                tracing::error!(%error, "TAGURU_REPLICATE_URL is not usable");
                std::process::exit(1);
            }
        });
    // Read replica (issue #129): same bucket machinery, opposite role —
    // hydrate continuously, never claim, never ship, refuse writes.
    let replica_mode = serve_args.replica || env_bool("TAGURU_REPLICA", false);
    if replica_mode && replicate.is_none() {
        tracing::error!(
            "--replica needs TAGURU_REPLICATE_URL: a replica IS the bucket lineage, \
             served locally — without a bucket there is nothing to replicate"
        );
        std::process::exit(1);
    }
    let ship_progress = (!replica_mode)
        .then(|| {
            replicate
                .as_ref()
                .map(|_| Arc::new(ship::ShipProgress::new()))
        })
        .flatten();

    // Boot-from-bucket (issue #128): decide once — before the registry
    // opens — whether this directory is the lineage's own tip (boot as
    // ever), a cache to re-verify, or empty and about to hydrate; and
    // whether starting here needs the operator's stated takeover
    // intent. A refusal here is deliberate: it is the guard rail that
    // keeps "starting a writer against a bucket" an explicit act. A
    // replica takes the sibling path: always a cache, never a claimant,
    // no guard to trip (it deposes nobody).
    let take_over = serve_args.take_over || env_bool("TAGURU_TAKEOVER", false);
    let hydrator = match &replicate {
        Some((replicate, store, root)) if replica_mode => {
            match hydrate::prepare_replica(store, root, &replicate.url, &config.data_dir).await {
                Ok(hydrator) => Some(hydrator),
                Err(error) => {
                    tracing::error!(%error, "refusing to start");
                    std::process::exit(1);
                }
            }
        }
        Some((replicate, store, root)) => {
            match hydrate::prepare(store, root, &replicate.url, &config.data_dir, take_over).await {
                Ok(hydrator) => hydrator,
                Err(error) => {
                    tracing::error!(%error, "refusing to start");
                    std::process::exit(1);
                }
            }
        }
        None => None,
    };
    let replica_info = replica_mode.then(|| {
        Arc::new(replica::ReplicaInfo::new(
            std::env::var("TAGURU_WRITER_URL")
                .ok()
                .map(|url| url.trim().to_string())
                .filter(|url| !url.is_empty()),
        ))
    });

    let state = match config.boot(
        embedder,
        ship_progress.clone(),
        hydrator.clone(),
        replica_info.clone(),
    ) {
        Ok(state) => state,
        // A held lock or an unreadable directory is an operator
        // problem, not a bug: one line, no backtrace.
        Err(error) => {
            tracing::error!(%error, "data directory is not usable");
            std::process::exit(1);
        }
    };

    if replica_mode {
        // No flusher: a replica never has dirty state to persist, and
        // its disk belongs to the tailer. The scrape flips into
        // replica shape here, once, for the process's lifetime.
        state.metrics().set_replica_mode();
    } else {
        spawn_flusher(state.clone(), flush_secs, auto_embed);
    }

    // AFTER boot, so the pinned preload's eager hydration went first —
    // the fill mops up whatever first touch has not reached, closing
    // the window in which the bucket's only complete generation is
    // the predecessor's. A replica has no fill thread: its tailer's
    // first pass does exactly this job, and keeps doing it forever.
    if !replica_mode && let Some(hydrator) = &hydrator {
        hydrator.spawn_background_fill();
    }

    let shipper = match &replicate {
        Some((replicate, store, root)) if !replica_mode => {
            info!(
                url = %replicate.url,
                interval_ms = replicate.interval.as_millis() as u64,
                "replication enabled — shipping the data directory continuously"
            );
            Some(ship::spawn(
                Arc::clone(store),
                root.clone(),
                ship::ReplicateConfig {
                    url: replicate.url.clone(),
                    interval: replicate.interval,
                },
                config.data_dir.clone(),
                ship_progress
                    .clone()
                    .expect("progress exists whenever a writer replicates"),
                state.clone(),
                hydrator.clone(),
            ))
        }
        _ => None,
    };
    let tailer = match &replicate {
        Some((replicate, store, root)) if replica_mode => {
            info!(
                url = %replicate.url,
                interval_ms = replicate.interval.as_millis() as u64,
                "replica mode — tailing the bucket lineage; writes are refused"
            );
            Some(replica::spawn(
                Arc::clone(store),
                root.clone(),
                ship::ReplicateConfig {
                    url: replicate.url.clone(),
                    interval: replicate.interval,
                },
                config.data_dir.clone(),
                state.clone(),
                Arc::clone(
                    hydrator
                        .as_ref()
                        .expect("a replica boot always has a hydrator"),
                ),
                Arc::clone(
                    replica_info
                        .as_ref()
                        .expect("replica info exists whenever replica_mode does"),
                ),
            ))
        }
        _ => None,
    };

    let app = routes(
        protocol_trailer,
        limits::HeavyOpsLimiter::new(max_concurrent_heavy),
        state.clone(),
    )
    .with_state(state.clone());

    // POST /mcp speaks the MCP Streamable HTTP transport over these
    // same routes. The dispatch handle is captured BEFORE the outer
    // middleware stack goes on, so a tool call that already passed
    // auth, the timeout, and the body cap at /mcp is not re-charged
    // inside — one client request, one budget, one log line — but it
    // DOES carry its own authorization layer, so a scoped key's grant
    // is enforced on each dispatched tool call exactly as on the raw
    // API. ("No body cap inside" is explicit: an extractor that finds
    // no DefaultBodyLimit extension falls back to axum's hardcoded
    // 2 MiB default, which would silently cap dispatched tool calls
    // below the operator's TAGURU_MAX_BODY_BYTES.)
    let mcp_dispatch = app
        .clone()
        .layer(axum::extract::DefaultBodyLimit::disable())
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&keyring),
            auth::enforce_authorization,
        ));
    let app = app.route(
        "/mcp",
        post(
            move |deadline: axum::Extension<Deadline>,
                  key: Option<axum::Extension<auth::AuthKey>>,
                  body: axum::body::Bytes| {
                remote_mcp::serve(
                    mcp_dispatch.clone(),
                    Arc::clone(&mcp_instructions),
                    key.map(|extension| extension.0),
                    body,
                    mcp_max_result_bytes,
                    deadline.0,
                )
            },
        ),
    );
    // OAuth's discovery and grant endpoints ride the same stack (rate
    // limited, logged) but are exempted from bearer auth inside the
    // gate — they exist to create credentials.
    let app = match &oauth {
        Some(oauth) => app.merge(oauth_http::router(oauth_http::OauthState {
            oauth: Arc::clone(oauth),
            keyring: Arc::clone(&keyring),
            // The consent form shares the bearer gate's failed-auth
            // throttle: a wrong key there is a wrong credential too.
            fail_limiter: Arc::clone(&fail_limiter),
        })),
        None => app,
    };
    // The replica gate AGAIN, outside the merges: a `.layer()` wraps
    // only the routes already on the router it is called on, so the
    // copy inside `routes()` (which the /mcp dispatch clone needs)
    // never reaches the OAuth router merged above — whose grant and
    // token POSTs would otherwise mutate a replica's cache-owned
    // credential store. Idempotent on the shared routes (both copies
    // judge identically); the OAuth discovery GETs still pass, the
    // mutating POSTs refuse like every other write.
    let app = app.layer(axum::middleware::from_fn_with_state(
        state.clone(),
        api::replica_gate,
    ));
    // `routes()` applies its own CatchPanicLayer (see the comment at
    // its end) before this function ever adds the /mcp route or merges
    // the OAuth router above, so — for the exact same reason the auth
    // layer below has to come after both — that inner layer never
    // reaches either surface. Re-applying it here, still innermost
    // relative to auth/rate-limit/timeout/concurrency/metrics below,
    // means a panic dispatched through /mcp or raised inside an OAuth
    // handler answers 500 through `api::panic_response` instead of
    // dropping the connection out from under those layers.
    let app = app.layer({
        let state = state.clone();
        CatchPanicLayer::custom(move |payload| api::panic_response(payload, &state))
    });
    // Authorization wraps every OUTER route — the context API, the
    // /mcp endpoint itself, and the merged OAuth routes — now that all
    // are registered. `.layer()` only covers routes already on the
    // router, so this MUST come after the /mcp route and the oauth
    // merge above; layering it earlier (as this once did) silently
    // left both surfaces unchecked. Idempotent on auth-exempt routes:
    // enforce_authorization no-ops when no AuthKey extension is present.
    let app = app.layer(axum::middleware::from_fn_with_state(
        Arc::clone(&keyring),
        auth::enforce_authorization,
    ));
    let gate = Arc::new(auth::Gate {
        keyring,
        oauth: oauth.clone(),
        fail_limiter,
    });
    let app = app
        // Layers wrap only the routes registered above and nest by
        // call order — later .layer() calls sit outside earlier ones.
        // Body limit innermost, then the per-key rate gate (inside
        // auth: budget is spent only by authenticated keys), then
        // auth, then the timeout (its budget covers auth and body
        // handling too), and metrics outermost: every response — 401,
        // 408, 413, 429 — lands in the access log and the RED metrics.
        .layer(axum::extract::DefaultBodyLimit::max(max_body_bytes))
        .layer(axum::middleware::from_fn_with_state(
            rate_limiter,
            limits::enforce_rate_limit,
        ))
        .layer(axum::middleware::from_fn_with_state(
            gate,
            auth::require_bearer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            Duration::from_secs(timeout_secs as u64),
            limits::enforce_timeout,
        ))
        // The in-flight ceiling sits outside the timeout: a shed must
        // be the cheapest possible response — no auth, no body read,
        // no handler — while still landing in the access log below.
        .layer(axum::middleware::from_fn_with_state(
            (max_concurrent, state.clone()),
            limits::enforce_concurrency,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            metrics::track_http,
        ));

    let addr = std::env::var("TAGURU_ADDR").unwrap_or_else(|_| "127.0.0.1:8248".to_string());
    let listener = match TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(bind_error) => {
            // The one boot failure an operator hits routinely deserves
            // a diagnosis, not a panic backtrace.
            tracing::error!(
                %addr,
                error = %bind_error,
                "cannot bind — is the port already in use, or the address not local? \
                 Set TAGURU_ADDR to change where the server listens"
            );
            std::process::exit(1);
        }
    };
    // Print the RESOLVED address: with port 0 the OS picks one, and
    // whoever spawned us (integration tests included) reads it here.
    // This stdout line is a contract — logging changes must not touch it.
    println!("listening on {}", listener.local_addr().unwrap());
    // Off-loopback with no per-key budget: one chatty client owns the
    // whole server. README says to turn the limit on when the server
    // leaves localhost — say it where the operator is looking. Pulled
    // into a predicate so the condition has a unit test; the message
    // hedges on 0.0.0.0 since the Dockerfile binds it by default just
    // so `-p`/a Service can publish the port, which isn't proof this
    // process is actually reachable beyond its own namespace.
    if needs_off_loopback_warning(listener.local_addr().unwrap().ip(), rate_limit_disabled) {
        warn!(
            "listening beyond loopback with TAGURU_RATE_LIMIT_PER_MIN off — 0.0.0.0 \
             alone doesn't mean exposed (a container needs it just to publish the \
             port at all); set a per-key budget once the server is actually \
             reachable beyond this host"
        );
    }
    // "ready" only after the socket exists: everything an operator
    // needs to sanity-check the deployment, in one line, at the moment
    // requests can actually arrive.
    info!(
        addr = %listener.local_addr().unwrap(),
        contexts = state.context_count(),
        data_dir = %config.data_dir.display(),
        cache_mib = config.cache_bytes / (1024 * 1024),
        flush_secs,
        wal_enabled = config.wal_enabled,
        wal_max_mib = config.wal_max_bytes / (1024 * 1024),
        passages_wal_max_mib = config.passages_wal_max_bytes / (1024 * 1024),
        max_body_mib = max_body_bytes / (1024 * 1024),
        mcp_max_result_mib = mcp_max_result_bytes / (1024 * 1024),
        timeout_secs,
        auth_enabled = auth_configured,
        "server ready",
    );
    // `with_connect_info` puts each caller's SocketAddr in the request
    // extensions — the per-source-IP throttles (failed-auth in the gate,
    // the anonymous request budget) read it from there. Without it those
    // fall back to a single shared bucket.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .unwrap();

    // Nothing dirty may outlive the process: one final flush.
    tokio::task::block_in_place(|| state.flush_dirty());
    // Usage counters skip the WAL by design; the graceful stop is
    // where purely-read contexts get theirs onto disk.
    tokio::task::block_in_place(|| state.persist_usage());
    info!("flushed dirty contexts on shutdown");
    // AFTER the final flush, so the shipper's last cycle carries the
    // images that flush just published — the bucket ends as current as
    // the disk. Best-effort by nature: an unreachable bucket already
    // has the un-shipped tail counted in its lag metric, and holding
    // shutdown hostage to it would trade a bounded RPO for an unbounded
    // stop.
    if let Some(shipper) = shipper {
        shipper.shutdown().await;
        info!("replication drained on shutdown");
    }
    // The tailer stops between polls (or mid-apply, at the next stem
    // boundary): nothing to drain — a replica's disk state is exactly
    // as durable as the last applied diff, and the next boot
    // re-verifies it against the bucket anyway.
    if let Some(tailer) = tailer {
        tokio::task::block_in_place(|| tailer.shutdown());
        info!("replica tailer stopped on shutdown");
    }
    // The batch worker still owns the last spans; hand them to the
    // collector before the process ends.
    if let Some(provider) = tracer_provider
        && let Err(error) = tokio::task::block_in_place(|| provider.shutdown())
    {
        warn!(error = %error, "trace export flush on shutdown failed");
    }
}

/// Every HTTP route the server answers, mapped to its api:: handler —
/// the same table `POST /mcp` dispatches into. Off-axis errors answer
/// in the ApiError shape too: unknown paths, and known paths hit with
/// the wrong verb.
fn routes(
    protocol_trailer: Option<String>,
    heavy_ops_limiter: limits::HeavyOpsLimiter,
    state: AppState,
) -> Router<AppState> {
    let heavy_routes = Router::new()
        .route("/contexts/{name}/compact", post(api::compact_context))
        .route(
            "/contexts/{name}/vocabulary/audit",
            post(api::audit_vocabulary),
        )
        .route_layer(axum::middleware::from_fn_with_state(
            heavy_ops_limiter.clone(),
            limits::enforce_heavy_ops,
        ));

    // `include_twins` runs the same CPU-bound pairwise scan
    // audit_vocabulary does, but only when the request body asks for
    // it — the default drift audit is a cheap O(n) sweep. So this
    // route carries the limiter as an extension instead of sharing
    // the unconditional gate above; `audit_drift` only spends a
    // permit while its expensive branch actually runs.
    let drift_audit_route = Router::new()
        .route("/contexts/{name}/drift/audit", post(api::audit_drift))
        .route_layer(axum::Extension(heavy_ops_limiter));

    Router::new()
        .route("/health", get(metrics::health))
        .route("/live", get(metrics::live))
        .route("/metrics", get(metrics::render))
        .route(
            "/protocol",
            get(move || api::protocol(protocol_trailer.clone())),
        )
        .route("/flush", post(api::flush_all))
        .route("/maintenance/compact", post(api::maintenance_compact))
        .route("/import", post(api::import_batch))
        .route("/contexts", get(api::list_contexts))
        .route(
            "/contexts/{name}",
            get(api::get_context)
                .put(api::create_context)
                .patch(api::update_context)
                .delete(api::delete_context),
        )
        .route("/groups", get(api::list_groups))
        .route(
            "/groups/{name}",
            get(api::get_group)
                .put(api::create_group)
                .patch(api::update_group)
                .delete(api::delete_group),
        )
        .route("/groups/{name}/export", get(api::export_group))
        .route("/groups/{name}/rename", post(api::rename_group))
        // The cross-context searches: the per-context operation of the
        // same name, run across several contexts named in the body.
        .route("/recall", post(api::cross_recall))
        .route("/query", post(api::cross_query))
        .route("/sources/search", post(api::cross_search_passages))
        .route("/contexts/{name}/export", get(api::export_context))
        .route("/contexts/{name}/rename", post(api::rename_context))
        .route("/contexts/{name}/associations", post(api::add_associations))
        .route(
            "/contexts/{name}/associations/retract",
            post(api::retract_association),
        )
        .route("/contexts/{name}/recall", post(api::recall))
        .route("/contexts/{name}/query", post(api::query))
        .route("/contexts/{name}/describe", post(api::describe))
        .route("/contexts/{name}/explore", post(api::explore))
        .route("/contexts/{name}/activate", post(api::activate))
        .route("/contexts/{name}/resolve", post(api::resolve))
        .route(
            "/contexts/{name}/resolve/explain",
            post(api::explain_resolve),
        )
        .route("/contexts/{name}/resolve_label", post(api::resolve_label))
        .route(
            "/contexts/{name}/resolve_label/explain",
            post(api::explain_resolve_label),
        )
        .route("/contexts/{name}/labels", get(api::labels))
        .route(
            "/contexts/{name}/aliases",
            get(api::list_aliases)
                .post(api::add_aliases)
                .delete(api::remove_aliases),
        )
        .route(
            "/contexts/{name}/sources",
            get(api::list_sources).post(api::store_passages),
        )
        .route(
            "/contexts/{name}/sources/lookup",
            post(api::lookup_passages),
        )
        .route(
            "/contexts/{name}/sources/search",
            post(api::search_passages),
        )
        .route(
            "/contexts/{name}/sources/search/explain",
            post(api::explain_search_passages),
        )
        .route(
            "/contexts/{name}/sources/retract",
            post(api::retract_source),
        )
        .route("/contexts/{name}/citations", post(api::citation))
        .route(
            "/contexts/{name}/embeddings/refresh",
            post(api::refresh_embeddings),
        )
        .route(
            "/contexts/{name}/unreachable_from",
            post(api::unreachable_from),
        )
        .merge(heavy_routes)
        .merge(drift_audit_route)
        .fallback(api::unknown_path)
        .method_not_allowed_fallback(api::method_not_allowed)
        // Innermost on purpose: `routes()` is cloned for the /mcp
        // in-process dispatch too (see `mcp_dispatch` below), so a
        // panic inside a dispatched tool call is caught right here,
        // same as one from an ordinary HTTP handler — neither takes
        // down the connection task out from under the metrics/
        // access-log/trace middleware layered on further out. Also why
        // this needs `state` passed in rather than reusing `serve()`'s
        // own outer CatchPanicLayer below: `record_error` inside
        // `panic_response` is the only signal a dispatched call's panic
        // leaves behind, since the /mcp response itself stays 200 (see
        // `panic_response`'s doc comment).
        .layer(CatchPanicLayer::custom({
            let state = state.clone();
            move |payload| api::panic_response(payload, &state)
        }))
        // Outside the panic catcher, inside everything `serve()` adds:
        // a no-op on a writer, the write-refusal on a replica — and,
        // because the /mcp dispatch clones this router, the refusal
        // covers dispatched write TOOLS exactly like raw HTTP verbs.
        .layer(axum::middleware::from_fn_with_state(
            state,
            api::replica_gate,
        ))
}

/// The periodic flusher: every `flush_secs`, persist what is dirty —
/// and, when auto embedding is on, refresh what just flushed. Best
/// effort: a failed refresh is retried the next time a write dirties
/// the context (the gloss diff is idempotent), and the manual endpoint
/// always remains.
fn spawn_flusher(state: AppState, flush_secs: usize, auto_embed: bool) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(flush_secs as u64));
        ticker.tick().await; // the first tick fires immediately; skip it
        loop {
            ticker.tick().await;
            // A maintenance sweep is already rewriting images under its
            // own lock discipline; skip this tick rather than race it.
            // Safe either way (the entry's `image_generation` guards a
            // stale republish), but skipping avoids flushing an entry
            // mid-compaction for nothing.
            if state.metrics().maintenance_active() {
                continue;
            }
            // Guarded: this loop's `JoinHandle` is discarded above, so
            // an unguarded panic here (a bug, not a disk fault) would
            // silently kill the flusher forever — no log, no crash,
            // and `/health` stuck reporting whatever it last saw since
            // the loop that was supposed to keep disproving it is gone.
            // Catching it keeps the loop alive for the next tick and
            // lets `record_flusher_tick` turn it into a 503 instead.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_flush_tick(&state, auto_embed)
            }));
            state.metrics().record_flusher_tick(outcome.is_ok());
            if let Err(payload) = outcome {
                let message = if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else {
                    "flusher tick panicked with a non-string payload".to_string()
                };
                error!(
                    %message,
                    "flusher tick panicked; /health will report unhealthy until the next clean tick"
                );
            }
        }
    });
}

/// One flusher tick's actual work, split out of `spawn_flusher` so it
/// can run behind `catch_unwind` as a plain synchronous call — the
/// guarded closure must not straddle an `.await`, which rules out
/// wrapping the loop body in place.
fn run_flush_tick(state: &AppState, auto_embed: bool) {
    // Serialization and fsyncs are blocking work; keep the
    // async workers free while images land on disk.
    let flushed = tokio::task::block_in_place(|| state.flush_dirty());
    // Opt-in: a flushed context just changed on disk, so its
    // glosses may have changed too — re-embed the difference.
    if auto_embed && !flushed.is_empty() {
        tokio::task::block_in_place(|| {
            for name in &flushed {
                match state.refresh_embeddings(name, Deadline::unbounded()) {
                    None | Some(Ok((0, _))) => {}
                    Some(Ok((embedded, _))) => {
                        info!(context = %name, embedded, "auto-embedded glosses");
                    }
                    Some(Err(error)) => {
                        warn!(context = %name, error, "auto embedding refresh failed");
                    }
                }
            }
        });
    }
    // Passages ride their own dirty flag: storing text never
    // marks the GRAPH dirty, so the flush list alone would
    // miss passage-only ingest.
    if auto_embed && state.passage_embedding_enabled() {
        let stale = state.passage_embed_dirty_names();
        if !stale.is_empty() {
            tokio::task::block_in_place(|| {
                for name in &stale {
                    match state.refresh_passage_embeddings(name, Deadline::unbounded()) {
                        None => {}
                        Some(Ok(outcome)) if outcome.embedded == 0 => {}
                        Some(Ok(outcome)) => {
                            info!(
                                context = %name,
                                embedded = outcome.embedded,
                                "auto-embedded paragraphs"
                            );
                        }
                        Some(Err(error)) => {
                            warn!(
                                context = %name,
                                error,
                                "auto passage embedding failed"
                            );
                        }
                    }
                }
            });
        }
    }
}

/// Installs the global tracing subscriber: `RUST_LOG` filtering
/// (default `info`), human-readable or JSON lines per
/// `TAGURU_LOG_FORMAT`, written to stderr — stdout stays reserved for
/// the bootstrap contract lines tests parse. When an OTLP endpoint is
/// configured, an export layer rides alongside; `RUST_LOG` shapes only
/// the stderr log, never what is exported.
fn init_telemetry() -> Option<opentelemetry_sdk::trace::SdkTracerProvider> {
    use tracing_subscriber::Layer as _;
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    let filter = tracing_subscriber::EnvFilter::try_from_env("RUST_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let (json, unrecognized_log_format) = match std::env::var("TAGURU_LOG_FORMAT") {
        Ok(value) if value.eq_ignore_ascii_case("json") => (true, None),
        Ok(value) => (false, Some(value)),
        Err(_) => (false, None),
    };

    let (provider, exporter_error) = trace::provider();
    let otel_layer = provider.as_ref().map(|provider| {
        use opentelemetry::trace::TracerProvider as _;
        // INFO keeps the export layer from re-enabling debug/trace
        // callsites the stderr filter would otherwise leave off.
        tracing_opentelemetry::layer()
            .with_tracer(provider.tracer("taguru"))
            .with_filter(tracing::level_filters::LevelFilter::INFO)
    });

    let registry = tracing_subscriber::registry().with(otel_layer);
    let stderr_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);
    if json {
        registry
            .with(stderr_layer.json().with_filter(filter))
            .init();
    } else {
        registry.with(stderr_layer.with_filter(filter)).init();
    }

    // Deferred from the TAGURU_LOG_FORMAT read above: logging works only now.
    if let Some(value) = unrecognized_log_format {
        warn!(
            "ignoring TAGURU_LOG_FORMAT={value}: not a recognized format (json); using human-readable"
        );
    }
    // Deferred from trace::provider(): logging works only now.
    if let Some(error) = exporter_error {
        warn!(
            error,
            "span export disabled: the OTLP exporter failed to build"
        );
    }
    if provider.is_some() {
        info!("OTLP span export enabled");
    }
    provider
}

/// Resolves on the first SIGINT (Ctrl+C) or SIGTERM — the signal that
/// starts axum's graceful drain — and, before returning, arms a one-shot
/// task that force-exits on the SECOND such signal. The drain waits on
/// in-flight requests; if one hangs, an impatient operator's second
/// Ctrl+C would otherwise be swallowed (tokio's installed handler
/// overrides the default terminate disposition, and nothing awaits it),
/// leaving SIGKILL as the only way out. The re-armed listener restores
/// the familiar escape hatch: press again to quit now. A forced exit
/// skips the final image flush, but every write is already in the WAL and
/// replays on the next boot — only unwritten usage counters are lost,
/// the price the operator chose by not waiting for the clean stop.
///
/// The signal streams are registered ONCE and the second wait reuses
/// them. Re-registering per wait (the old shape) dropped the SIGTERM
/// stream when the first signal fired and installed a new one only after
/// the spawn — a window in which a second signal would land on no handler
/// and be lost, defeating the very escape hatch this arms.
async fn shutdown_signal() {
    let mut signals = TerminateSignals::install();
    signals.recv().await;
    tokio::spawn(async move {
        signals.recv().await;
        warn!("second shutdown signal received — forcing an immediate exit");
        // 128 + SIGINT(2), the shell convention for signal-terminated.
        std::process::exit(130);
    });
}

/// The SIGINT/SIGTERM streams axum's graceful drain listens on, held for
/// the whole shutdown sequence so [`recv`](Self::recv) can be awaited
/// again for a second signal without a re-registration gap.
struct TerminateSignals {
    #[cfg(unix)]
    interrupt: Option<tokio::signal::unix::Signal>,
    #[cfg(unix)]
    terminate: Option<tokio::signal::unix::Signal>,
}

impl TerminateSignals {
    /// Registers the terminate signals once. A stream that fails to
    /// register degrades to one that never fires — the other signal (and
    /// a second Ctrl+C) still works — rather than crashing the server as
    /// it starts serving.
    fn install() -> Self {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let register = |kind: SignalKind| match signal(kind) {
                Ok(stream) => Some(stream),
                Err(error) => {
                    warn!(%error, "could not register a shutdown signal handler");
                    None
                }
            };
            Self {
                interrupt: register(SignalKind::interrupt()),
                terminate: register(SignalKind::terminate()),
            }
        }
        #[cfg(not(unix))]
        {
            Self {}
        }
    }

    /// Completes on the next SIGINT or SIGTERM (Ctrl+C on non-Unix).
    async fn recv(&mut self) {
        #[cfg(unix)]
        {
            let Self {
                interrupt,
                terminate,
            } = self;
            let interrupt = wait_signal(interrupt.as_mut());
            let terminate = wait_signal(terminate.as_mut());
            tokio::select! {
                () = interrupt => {}
                () = terminate => {}
            }
        }
        #[cfg(not(unix))]
        {
            // No persistent Unix stream to drop here; ctrl_c()'s handler
            // is a global that stays installed, so re-awaiting is safe.
            let _ = tokio::signal::ctrl_c().await;
        }
    }
}

/// Awaits an optional signal stream, or pends forever when its
/// registration failed so the surrounding select falls to the other arm.
#[cfg(unix)]
async fn wait_signal(stream: Option<&mut tokio::signal::unix::Signal>) {
    match stream {
        Some(stream) => {
            stream.recv().await;
        }
        None => std::future::pending().await,
    }
}
