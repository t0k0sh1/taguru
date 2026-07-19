#!/usr/bin/env bash
# The packaging's own test suite — run from anywhere, no arguments.
# CI runs it on every PR touching deploy/ (.github/workflows/deploy.yml);
# run it locally after editing anything here.
#
# Three properties, each load-bearing:
#   1. The in-tree manifest copies are BYTE-IDENTICAL to the reference
#      files (kustomize's load restrictor is why copies exist at all —
#      see base/kustomization.yaml); a drifted copy fails loudly here
#      instead of shipping a fork.
#   2. Every configuration renders, and the rendered objects validate
#      against the Kubernetes schemas (kubeconform, when installed).
#   3. The base renders EQUIVALENT to `kubectl apply -f kubernetes.yaml`
#      — asserted by normalizing the reference through the same
#      kustomize and diffing the outputs, which also catches a patch
#      sneaking into the base (the base must stay transformer-free).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
deploy="$(dirname "$here")"

if command -v kustomize >/dev/null 2>&1; then
    build() { kustomize build "$1"; }
else
    build() { kubectl kustomize "$1"; }
fi

failed=0

# 1. Copies must not drift from the reference manifests.
while read -r reference copy; do
    if diff -u "$deploy/$reference" "$here/$copy" >/dev/null; then
        echo "ok: $copy == deploy/$reference"
    else
        echo "FAIL: $copy has drifted from deploy/$reference — edit the reference, then cp it over the copy" >&2
        diff -u "$deploy/$reference" "$here/$copy" >&2 || true
        failed=1
    fi
done <<'PAIRS'
kubernetes.yaml base/kubernetes.yaml
kubernetes-stateless.yaml overlays/stateless/kubernetes-stateless.yaml
kubernetes-replicas.yaml overlays/replicas/kubernetes-replicas.yaml
PAIRS

# 2. Every configuration renders; validate when kubeconform is around.
for config in base overlays/stateless overlays/replicas overlays/router; do
    if ! rendered="$(build "$here/$config")"; then
        echo "FAIL: $config does not render" >&2
        failed=1
        continue
    fi
    if command -v kubeconform >/dev/null 2>&1; then
        if echo "$rendered" | kubeconform -strict -summary; then
            echo "ok: $config renders and validates"
        else
            echo "FAIL: $config rendered objects failed schema validation" >&2
            failed=1
        fi
    else
        echo "ok: $config renders (kubeconform not installed — schema validation skipped)"
    fi
done

# 3. base ≡ the raw reference manifest, modulo kustomize's own
# normalization: wrap the reference in a throwaway kustomization so
# BOTH sides pass through the identical normalizer, then byte-diff.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp "$deploy/kubernetes.yaml" "$tmp/"
printf 'resources:\n  - kubernetes.yaml\n' > "$tmp/kustomization.yaml"
if diff -u <(build "$tmp") <(build "$here/base") >/dev/null; then
    echo "ok: base renders equivalent to deploy/kubernetes.yaml"
else
    echo "FAIL: base render diverges from deploy/kubernetes.yaml — the base must stay transformer-free" >&2
    diff -u <(build "$tmp") <(build "$here/base") >&2 || true
    failed=1
fi

exit "$failed"
