## Top-Level Rules

- Only an explicit fix instruction authorizes fixing. Investigating never rolls into fixing on its own — not because the fix is small, not because a command or skill you ran also fixes. Asked to investigate: stop at findings, ask before changing anything.
- Once fixing is authorized, fix every problem you find by default — whatever its source (your own discovery, the user, a PR review, a review tool) and whatever its scope. Handle obvious ones silently; for anything large or out of scope, list them as checkboxes and have the user deselect what to defer. Default is fix, not defer.
- A fix isn't complete until `cargo fmt`, `cargo clippy` (issues resolved), and `cargo test` (full suite passing) all succeed.

## Disk Hygiene

- At the start of a session, if free disk space is under 20 GiB or `target/` exceeds 30 GiB, run `cargo clean` — but only after confirming no other build is running (another session's `cargo test`, `cargo mutants`, etc.). Otherwise leave `target/` alone: an unconditional clean forces a full rebuild and costs more than it saves.
- Root cause of past 100+ GiB bloat and its fix (dev-profile `split-debuginfo`/dependency debuginfo) are documented in `Cargo.toml` above `[profile.dev]`.
