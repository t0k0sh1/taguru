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
- [`kubernetes-stateless.yaml`](kubernetes-stateless.yaml) — the same
  single writer with the bucket as the source of truth: emptyDir
  instead of the PVC, the pod hydrates from `TAGURU_REPLICATE_URL` at
  boot (pinned contexts first, the rest lazily). Its header states
  what the variant trades away (a crashed pod's un-shipped tail) and
  why it bakes `TAGURU_TAKEOVER=1` in.
- [`kubernetes-replicas.yaml`](kubernetes-replicas.yaml) — a read
  pool beside either writer manifest: `replicas: N` of
  `TAGURU_REPLICA=1` pods tailing the same bucket behind their own
  `taguru-read` Service. Reads scale with the pool; writes answer
  403 naming the writer (`TAGURU_WRITER_URL`); per-context lag —
  the promotion-time RPO — is on `/metrics`. Its header carries the
  manual-promotion runbook in short form.

When several writers each own a disjoint set of contexts, `taguru
route` puts one front door on the fleet: a stateless router (no
volume, no keys — shards enforce auth) whose `TAGURU_ROUTE_MAP` file
says which shard owns which context, with groups and cross-context
search spanning every shard under the single-instance merge
semantics. The `docker-compose.yml` carries a commented example;
the topology notes live in the
[Kubernetes page](https://t0k0sh1.github.io/taguru/kubernetes.html#sharding).

[`kustomize/`](kustomize/) packages the same manifests for
`kubectl apply -k`: the raw files above as verbatim bases, overlays
for the stateless / replicas / router variants (the router overlay is
the two-shard fleet worked out — writer shards, the route-map
ConfigMap, the front-door Deployment), and the knob retunes (image
tag, storage, resources, probes) as documented patches. Its README
records why kustomize over a Helm chart; `verify.sh` — run by CI on
every PR touching `deploy/` — keeps the in-tree manifest copies
byte-identical to the raw files, schema-validates every rendered
configuration, and keeps the base render-equivalent to
`kubernetes.yaml`.
Moving a context, in order: quiesce its writes → `taguru export` →
DELETE it through the router (the old shard drops it, group
projections included) → map edit + rolling router restart →
re-import through the router, which now routes it to the new shard.

All pin the image version on purpose (`latest` moves), keep the
credentials out of the manifest, and leave TLS to the layer in front —
a bearer token is the whole credential, so nothing here publishes the
port beyond loopback or the cluster.

Backups and restores are the same everywhere: set
`TAGURU_REPLICATE_URL` for continuous shipping to object storage
(recover with `taguru restore` — or just start a server on an empty
directory with the same URL and let it boot from the bucket; RPO ≈
seconds of shipping lag), or `POST /flush` and snapshot the volume for
a point-in-time copy (or `taguru export` for the portable stream);
verify either with `taguru inspect`. With a replica pool tailing the
bucket, availability is promotion time: drain the replica's lag
metrics to zero, start a writer against the bucket, flip the name —
the runbook is in the architecture docs, and what the bucket never
received (the dead writer's un-shipped tail) is the honestly-stated
RPO. Without replicas, availability is restore time — rehearse
whichever is the plan. Starting a writer against a bucket IS the
takeover act: while the previous writer looks alive (no clean stop, a
heartbeat within the last 300s), the boot demands `--take-over` /
`TAGURU_TAKEOVER=1` before it deposes them. A replica never trips
that guard — it deposes nobody.
