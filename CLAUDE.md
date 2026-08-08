## Top-Level Rules

- Only an explicit fix instruction authorizes fixing. Investigating never rolls into fixing on its own — not because the fix is small, not because a command or skill you ran also fixes. Asked to investigate: stop at findings, ask before changing anything.
- Once fixing is authorized, fix every problem you find by default — whatever its source (your own discovery, the user, a PR review, a review tool) and whatever its scope. Handle obvious ones silently; for anything large or out of scope, list them as checkboxes and have the user deselect what to defer. Default is fix, not defer.
- A fix isn't complete until `cargo fmt`, `cargo clippy` (issues resolved), and `cargo test` (full suite passing) all succeed.

## Pre-PR Mutation Gate

- Before `gh pr create` — and before pushing fix commits to an existing PR — run the same diff-scoped mutation check CI runs (mutants-diff.yml), so missed mutants are resolved locally instead of surfacing as a CI round-trip. Fetch first so the diff matches the PR's actual base, and count before running — the full run only happens within CI's per-PR budget (60):
  ```sh
  git fetch origin main &&
    diff=$(mktemp) && git diff origin/main...HEAD > "$diff" &&
    if [ "$(cargo mutants --in-diff "$diff" --list | wc -l)" -le 60 ]; then
      cargo mutants --profile=mutants --in-diff "$diff"
    else
      echo "over budget: dispatch a module sweep (mutants-sweep.yml) instead"
    fi
  ```
  Exit codes 2 (missed) and 3 (timeouts) are findings, not infrastructure failures. Resolve every missed mutant before the PR goes up: add a test that kills it, or mark it `#[mutants::skip]` with a reason comment.
- Over-budget diffs are new-module territory: CI skips them too, so cover that ground with a dispatched module sweep (mutants-sweep.yml) instead of a local run.

## Disk Hygiene

- At the start of a session, if free disk space is under 20 GiB or `target/` exceeds 30 GiB, run `cargo clean` — but only after confirming no other build is running (another session's `cargo test`, `cargo mutants`, etc.). Otherwise leave `target/` alone: an unconditional clean forces a full rebuild and costs more than it saves.
- Root cause of past 100+ GiB bloat and its fix (dev-profile `split-debuginfo`/dependency debuginfo) are documented in `Cargo.toml` above `[profile.dev]`.
