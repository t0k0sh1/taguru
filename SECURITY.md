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
