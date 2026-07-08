## Top-Level Rules

- Only an explicit fix instruction authorizes fixing. Investigating never rolls into fixing on its own — not because the fix is small, not because a command or skill you ran also fixes. Asked to investigate: stop at findings, ask before changing anything.
- Once fixing is authorized, fix every problem you find by default — whatever its source (your own discovery, the user, a PR review, a review tool) and whatever its scope. Handle obvious ones silently; for anything large or out of scope, list them as checkboxes and have the user deselect what to defer. Default is fix, not defer.
- A fix isn't complete until `cargo fmt`, `cargo clippy` (issues resolved), and `cargo test` (full suite passing) all succeed.
