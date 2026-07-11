# Deploying taguru

Worked examples of the model the main README documents under
"Deployment and availability": one node, one writer, one data
directory.

- [`docker-compose.yml`](docker-compose.yml) — a single host: named
  volume, loopback port, graceful stop.
- [`kubernetes.yaml`](kubernetes.yaml) — `replicas: 1` +
  `strategy: Recreate` on a ReadWriteOnce volume, with the probe
  wiring (`startupProbe`/`livenessProbe` on `/live`, `readinessProbe`
  on `/health`) and the scratch image's uid handled.

Both pin the image version on purpose (`latest` moves), keep the
credentials out of the manifest, and leave TLS to the layer in front —
a bearer token is the whole credential, so nothing here publishes the
port beyond loopback or the cluster.

Backups and restores are the same everywhere: `POST /flush`, snapshot
the volume (or `taguru export` for the portable stream), verify with
`taguru inspect`. Rehearse the restore — availability on this model
is restore time.
