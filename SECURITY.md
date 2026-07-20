# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities through [GitHub Private Vulnerability
Reporting](https://github.com/t0k0sh1/taguru/security/advisories/new)
— the "Report a vulnerability" button under this repository's
**Security** tab. It opens a private advisory visible only to the
maintainer and you, with threaded discussion, until a fix ships; this
is the preferred path, since it also drives the eventual public
advisory. If GitHub is not workable for you, email
**takashi.yamashina@gmail.com** instead.

Either way, include what makes a report actionable: affected
version/commit, reproduction steps, and the impact you assess (what
an attacker gains, what access they need). Please do not open a
public issue for a suspected vulnerability.

## Scope

In scope: the `taguru` server and library crate, the `taguru-mcp`
bridge, the `taguru` / `langchain-taguru` SDKs (Python and
TypeScript), and the published Docker image
(`ghcr.io/t0k0sh1/taguru`) — everything this repository builds and
ships.

Out of scope: a vulnerability in a third-party dependency with no
taguru-specific exploitation path (report it upstream; `cargo-deny`
and Dependabot already track known advisories against this tree), and
issues that only arise from a misconfigured deployment (an API token
committed to source control, a server exposed beyond localhost without
TLS — see [Running in production](README.md#running-in-production)
for the intended setup).

## Verifying a release

Every version tag ships a signed multi-arch image with a bill of
materials and build provenance attached, and CI re-verifies all three
from a clean runner — no registry login, no signing credentials —
before the release counts as done (the `verify` job in
[docker.yml](.github/workflows/docker.yml)). The same checks run from
any machine.

The image is signed keylessly ([Sigstore](https://www.sigstore.dev/)):
the identity is this repository's Docker workflow running on the
version tag. Pin it exactly — an unanchored pattern like
`github.com/t0k0sh1/taguru` also matches a look-alike repository
named `t0k0sh1/taguru-anything`:

```console
cosign verify ghcr.io/t0k0sh1/taguru:0.3.0 \
  --certificate-identity 'https://github.com/t0k0sh1/taguru/.github/workflows/docker.yml@refs/tags/v0.3.0' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

The verified output names the manifest-list digest — deploy that, not
the bare tag: a tag can be re-pushed, a digest names one immutable
manifest ([deploy/README.md](deploy/README.md) shows the pinning
syntax for each manifest).

The SBOM (SPDX) and provenance (SLSA v1, naming the exact CI run that
built the image) are BuildKit attestations on the image index itself,
so they travel under the digest the signature binds and plain buildx
reads them — no cosign required:

```console
docker buildx imagetools inspect ghcr.io/t0k0sh1/taguru:0.3.0 \
  --format '{{ json .SBOM }}'        # every crate, with its version
docker buildx imagetools inspect ghcr.io/t0k0sh1/taguru:0.3.0 \
  --format '{{ json .Provenance }}'  # which Actions run built it, from what
```

The crate list comes from the binary, not the filesystem: `taguru` is
built with [`cargo auditable`](https://github.com/rust-secure-code/cargo-auditable),
so the dependency inventory is embedded in the executable and works
outside any container too — `cargo audit bin taguru` audits a bare
binary against RustSec. (Releases up to and including 0.3.0 predate
`cargo auditable`: their attached SBOM is present but lists no crates
— a scratch image gives the scanner nothing else to read. Signature
and provenance verify on those releases regardless.)

## Response

taguru has a single maintainer, not a security team on a paid SLA —
expect an initial response within a few days. A confirmed
vulnerability gets a fix released as soon as practical; disclosure
timing is coordinated through the private advisory, and reporters are
credited in the eventual writeup unless they ask otherwise.

## Disclosure

Shipped fixes are recorded under a `### Security` heading in
[CHANGELOG.md](CHANGELOG.md) — see the 0.2.0 entries for the level of
detail to expect (what was wrong, what changed, whether upgrading
requires any action on your part).
