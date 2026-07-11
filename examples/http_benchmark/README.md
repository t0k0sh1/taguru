# http_benchmark — the whole server under concurrent load

Where `benchmark` times the bare `Context`, this drives a **running
server** over HTTP with concurrent clients — routing, auth, the
per-context locks, the WAL fsync, and serialization all included —
and answers the capacity-planning question the library benchmark
cannot: what does one instance sustain, at what tail latency, under
this configuration?

Three phases against one scratch context (`http-benchmark`, created
and deleted by the run):

1. **seed writes** — batches of 100 associations, the shape
   extract-driven ingest produces;
2. **reads** — recall/activate over the seeded vocabulary;
3. **mixed** — 90% reads with 10% single-association writes, so the
   same-context write lock and the WAL sit inside the read path's
   world.

Each phase reports throughput, p50/p95/p99 latency, and every
non-2xx.

## Run

```sh
# a scratch server to point at, if none is running:
TAGURU_DATA_DIR=$(mktemp -d) cargo run --release &

cargo run --release --example http_benchmark -- \
    --url http://127.0.0.1:8248 --concurrency 8 --requests 2000
# --token TOKEN / TAGURU_API_TOKEN when the target has auth on
# --seed-associations 10000 sizes the corpus the reads run against
```

## Reading the numbers

- Treat everything as **relative** — across commits, across
  configurations (WAL on/off, `TAGURU_CACHE_BYTES`, embedding tier
  on/off), across `--concurrency` sweeps — never as absolutes.
- Run it over the network path production will cross. Localhost
  numbers flatter the transport; they only isolate the server's own
  cost.
- Watch the server's own `/metrics` during a run —
  `taguru_inflight_requests`, the route histograms, and (past the
  in-flight ceiling) `taguru_requests_shed_total` tell you which side
  saturated first.
- A `--concurrency` sweep (1, 2, 4, 8, …) finds the knee: throughput
  that stops rising while p99 keeps climbing means the server is
  queueing, and the ceiling (`TAGURU_MAX_CONCURRENT_REQUESTS`) is
  what keeps that queue honest in production.
