# Kustomize packaging

`kubectl apply -k` over the reference manifests one directory up —
the variant matrix (writer / stateless / replicas / router) and the
knobs people retune, without copy-editing one YAML per combination.

```
base/                  the single-writer PVC model   (kubernetes.yaml)
overlays/stateless/    boot-from-bucket, emptyDir    (kubernetes-stateless.yaml)
overlays/replicas/     writer + bucket-tailing read pool, one apply
overlays/router/       two writer shards + the taguru route front door
verify.sh              CI's checks, runnable locally
```

```sh
kubectl create namespace taguru
kubectl -n taguru create secret generic taguru-keys \
  --from-literal=TAGURU_API_TOKENS='ops:CHANGE-ME'
kubectl -n taguru apply -k deploy/kustomize/base   # or an overlay
```

## Why kustomize, not a Helm chart

Recorded per issue #139, which asked for the packaging to be chosen
deliberately:

- **The reference manifests ARE the documentation** — their comments
  carry the reasoning (why `Recreate`, why the grace period, what the
  stateless variant trades away). Kustomize consumes that YAML as-is;
  a chart would rewrite it into templates, and `{{ }}` is where those
  comments go to die. `helm template` output has no comments at all.
- **No new tool for consumers**: `kubectl apply -k` ships inside
  kubectl. A chart adds a toolchain for what is, today, four small
  variants of one manifest.
- **The retuned knobs are patch-shaped.** Image tag, storage
  class/size, resources, probe budgets, grace period, secret names —
  each is a strategic-merge patch or an `images:` stanza (worked
  examples below), not a templating problem. A values.yaml would
  re-document what the manifests and `taguru --help` already say.
- Choosing kustomize forecloses nothing: a chart can still arrive
  later as a separate consumer artifact if demand shows. Per the
  issue: one, not both.

**Why the manifest copies exist**: kustomize's load restrictor
refuses `../../kubernetes.yaml` references and `kubectl apply -k` has
no flag to relax it (GitOps tools default restricted too), so the
files the kustomizations consume must live in-tree. The copies are
byte-identical to `deploy/*.yaml` and `verify.sh` — run by CI on
every PR touching `deploy/` — fails if they drift: the reference
files stay the single source of truth, and editing one means `cp`ing
it over its copy. `verify.sh` also asserts the base renders
equivalent to `kubectl apply -f kubernetes.yaml` and schema-validates
every rendered object (kubeconform).

## Retuning the knobs

Make your own overlay; never edit the reference manifests. A complete
example that pins a different image, resizes the volume, and widens
the startup budget:

```yaml
# my-fleet/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../deploy/kustomize/base        # path from YOUR repo to this one
images:
  - name: ghcr.io/t0k0sh1/taguru
    newTag: "0.4.0"                 # releases move the in-repo pin; you can too
patches:
  - target: { kind: PersistentVolumeClaim, name: taguru-data }
    patch: |-
      - op: replace
        path: /spec/resources/requests/storage
        value: 50Gi
      # storage class, when the cluster default is wrong for RWO block:
      # - op: add
      #   path: /spec/storageClassName
      #   value: gp3
  - target: { kind: Deployment, name: taguru }
    patch: |-
      # A bigger corpus: size memory with `taguru estimate`, and give
      # the pinned preload a longer startup runway.
      - op: replace
        path: /spec/template/spec/containers/0/resources/limits/memory
        value: 4Gi
      - op: replace
        path: /spec/template/spec/containers/0/startupProbe/failureThreshold
        value: 120
```

The same shapes cover the rest: `terminationGracePeriodSeconds` when
the embedding tier's timeout is widened, a different secret name via a
patch on `envFrom`, probe periods per fleet. The router overlay's map
(`overlays/router/route-map`) is a generated ConfigMap, so editing it
and re-applying rolls the routers by itself — and adding a shard is
copying a `shards/` directory, not templating one.
