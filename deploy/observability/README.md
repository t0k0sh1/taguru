# Prometheus + Grafana, worked example

`GET /metrics` (main README, "Health and metrics") is Prometheus text
— this directory is the other half: an actual Prometheus scraping it
and a Grafana dashboard on top, provisioned so `docker compose up -d`
is the whole setup. Every step below was run against a real `taguru
serve` while writing this, not just written from reading the code.

## What's here

- [`docker-compose.yml`](docker-compose.yml) — Prometheus + Grafana
  only. taguru is **not** started by this compose file — bring your
  own (see below) and point [`prometheus.yml`](prometheus.yml) at it.
- [`prometheus.yml`](prometheus.yml) — one scrape job, `taguru`,
  targeting `host.docker.internal:8248` by default (Docker Desktop
  resolves this natively; the compose file's `extra_hosts` line is
  what makes the same address work on Linux).
- [`grafana/provisioning/`](grafana/provisioning) — a Prometheus
  datasource (`uid: prometheus`, pointed at the `prometheus` service
  on the compose network) and a dashboard provider, both loaded on
  boot — no manual "Add data source" / "Import dashboard" clicking.
- [`grafana/dashboards/taguru.json`](grafana/dashboards/taguru.json)
  — request rate/latency by route, in-flight requests, requests shed
  by the rate limiter, build info, errors by kind, search outcomes by
  op, retrieval cache hit ratio, and per-context disk bytes. Every
  panel expression uses a metric name that actually appears in
  `src/metrics.rs`.

## Procedure

1. **Start taguru.** Any of these work, since only `/metrics` matters
   here:
   - locally: `TAGURU_API_TOKENS='ops:demo-token' TAGURU_METRICS_PER_CONTEXT=all cargo run --release`
   - the single-host compose:
     `TAGURU_API_TOKENS='ops:demo-token' docker compose -f ../docker-compose.yml up -d`
     — then point `prometheus.yml` at `taguru:8248` and join this
     compose to that one's network (`docker network connect`), since
     they're separate compose projects by default.
   - a real deployment — point `prometheus.yml` at its `TAGURU_ADDR`.

   `TAGURU_METRICS_PER_CONTEXT=all` is optional but turns on the
   per-context disk/resident panels; without it those two panels stay
   empty.

2. **Bring up the stack:**

   ```sh
   cd deploy/observability
   docker compose up -d
   ```

3. **Check the scrape landed** (Prometheus, no auth):

   ```sh
   curl -s 'http://localhost:9090/api/v1/query?query=up{job="taguru"}'
   ```

   `value` should read `1`. If it's absent or `0`, `docker compose logs
   prometheus` and the target page at
   `http://localhost:9090/targets` say why — almost always taguru
   not listening where `prometheus.yml` expects it.

4. **Open Grafana** at `http://localhost:3000` — `admin` / `admin`,
   change the password on first login (the compose file doesn't
   disable anonymous access implicitly; it's off by default here).
   The **taguru** dashboard is already in the **taguru** folder, no
   import step. Generate a bit of traffic against taguru first
   (`curl` a few `recall`/`activate` calls) if the panels look flat —
   they're rate()-windowed, so a server that's been idle for 5
   minutes legitimately shows zero.

5. **Tear down:** `docker compose down` removes both containers;
   nothing here persists state outside them (Prometheus's TSDB and
   Grafana's sqlite both live in anonymous container volumes, so
   `down` — without `-v`, which isn't needed since none are named —
   already discards them on the next `up`).

## Verified, not just documented

While building this the actual sequence was: build the release
binary, start it with a real context and a few `PUT`/`POST` calls
against it, bring the compose stack up, and confirm through
Prometheus's HTTP API that `up{job="taguru"}` was `1` and that
`taguru_searches_total{op="activate",outcome="hit"}` reflected the
real call count — then confirm the same numbers reading through
Grafana's datasource proxy (`/api/datasources/proxy/uid/prometheus/...`),
not just Prometheus directly. `taguru_context_disk_bytes` came back
with the actual on-disk byte counts for the context created during
the run. That's the chain this directory automates.

## Extending the dashboard

New panels are ordinary PromQL against whatever's in
`src/metrics.rs`'s `render_prometheus` — the metric names there are
the source of truth, and `render_prometheus`'s own tests
(`src/metrics.rs`, the `#[cfg(test)]` block) show the exact label sets
each one carries. Add a panel object to `taguru.json`'s `panels`
array; Grafana's file-provider reloads it within
`updateIntervalSeconds` (30s here) without a restart.
