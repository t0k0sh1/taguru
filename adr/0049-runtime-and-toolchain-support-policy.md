# 0049. Runtime and toolchain support: drop three months before upstream EOL; the MSRV is the release toolchain; SDK CI installs from the lockfiles

- **Status**: Accepted
- **Date**: 2026-11-11
- **Issue**: #1013
- **Related**: #1012 (Python 3.10 dropped under advisory pressure), #1001 /
  #910 (pdfjs-dist blocked on the Node 20 floor), #1015 (stale `uv.lock`
  pins that CI never saw)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

Until now, each supported runtime version was decided only when
something broke:

- Python 3.10 was dropped in #1012 because the fixed releases of
  `pillow` and `filelock` no longer installed on it.
- The TS SDK still declared `node >=20` and tested Node 20 six months
  after Node 20's EOL (2026-04-30). That kept pdfjs-dist on 6.2.108,
  because 6.3+ calls `Promise.withResolvers`, a Node 22 API.
- `rust-version = "1.89"` was never built in CI. Every job ran unpinned
  stable while the release image built on `rust:1.96-alpine`, so nothing
  checked the declared MSRV.
- The Python SDK CI resolved fresh from PyPI on every run:
  - An unbounded dev dependency's new release could fail an unrelated
    PR.
  - The committed `uv.lock` files kept advisory-flagged pins that no
    job ever installed (#1015).

## 2. Decision

1. **Drop a runtime three months before its upstream EOL.** This
   applies to Python and Node.js. A version whose upstream end-of-life
   is three months away or less is removed from the declared range
   (`requires-python`, `engines`) and from the CI matrices in the first
   release cut inside that window. The drop is recorded in the CHANGELOG
   as a breaking change.
2. **Node.js floor: `^22.13.0 || >=24`.** Node 20 is past EOL. 22.13 is
   pdfjs-dist's own `engines` floor. The SDK CI matrix is 22 and 24.
3. **Rust: the MSRV is the release toolchain.** Rust stable has no
   per-version EOL, so rule 1 does not apply to it. Instead,
   `rust-version` in `Cargo.toml` equals the Dockerfile's
   `rust:<version>` and moves only with it, starting at 1.96. CI's
   `msrv` job fails if the two differ, then runs the full test suite at
   that version. The `check` job's fmt and clippy stay on unpinned
   stable.
4. **SDK CI installs from the lockfiles.** The Python SDK's lint,
   typecheck, test and mutation jobs install with `uv sync --locked`
   from each package's `uv.lock`.
   - A lockfile that no longer matches its `pyproject.toml` fails the
     job.
   - The audit runs on the locked environments, so a lock that still
     pins a vulnerable version fails as well.
   - One job (`python-fresh`) keeps the fresh resolve an installing user
     gets, with its own audit and test runs.

## 3. Consequences

Dates per the upstream schedules as of 2026-11-11:

| Runtime | Upstream EOL | Dropped from releases cut on or after |
|---|---|---|
| Python 3.11 | 2027-10 (late October) | 2027-07-24 |
| Node.js 22 | 2027-04-30 | 2027-01-30 |
| Python 3.12 | 2028-10 | 2028-07 |
| Node.js 24 | 2028-04-30 | 2028-01-30 |

- Node 22 enters its drop window about two and a half months after this
  ADR. A release cut from 2027-01-30 on therefore makes the TS SDK
  `>=24` only, which is another breaking change for its users.
- Bumping the Dockerfile's Rust image is now an MSRV bump too, and CI
  enforces it. `dependabot.yml` has no `docker` ecosystem entry, so no
  automated bump can split the two.
- Unrelated PRs no longer fail because of a new PyPI release in the
  lint, test or mutation jobs. Such a release now surfaces in only two
  places: `python-fresh`, and the Dependabot `uv` group bumps that move
  the lockfiles.
- Still waiting on upstream, outside this decision (tracked in #1013):
  - vitest 5, blocked on `@stryker-mutator/vitest-runner`.
  - The `minimatch` advisory, held at `^10.2.2` by
    `@typescript-eslint/typescript-estree`.
