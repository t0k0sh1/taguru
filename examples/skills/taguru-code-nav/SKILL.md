---
name: taguru-code-nav
description: "Locate symbols, files, and structure in this repository without grepping: taguru-code answers 'where is X defined?' and 'what does Y contain?' from a pre-built code map. Use BEFORE Grep/Glob for any location question about code — function/struct/trait definitions, file contents, module layout."
---

# taguru-code-nav

This repository keeps an offline code map in `.taguru/`, maintained by
the `taguru-code` binary. It answers location questions in one call
where grepping takes several, and every answer carries `file:lines`
you can open directly.

## When to use

Reach for `taguru-code` FIRST when the question is any of:

- "Where is `parse_batch` defined?" — a symbol's location
- "Where is `Context::resolve`?" — a method on a type
- "What's in `src/ingest/`?" / "What does `model.rs` define?" — structure
- Fixing a typo'd or half-remembered name (`parse_bacth`)

Do NOT use it for: full-text search of arbitrary strings, comments,
or string literals — that is Grep's job. It indexes definitions and
structure, not every occurrence.

## Commands

```bash
taguru-code find <cue>            # locate a symbol: kind, qualified name, file:lines, tier
taguru-code find <cue> --json     # same, machine-readable
taguru-code tree                  # top-level directories
taguru-code tree src/ingest       # one level: what a directory/file/symbol contains
taguru-code tree src/api.rs       # symbols a file defines
taguru-code status                # is the map in sync with HEAD?
taguru-code sync                  # refresh after commits
```

`find` output, one hit per line:

```
fn        src/ingest/model.rs::parse_batch   src/ingest/model.rs:323-346  [exact 1.00]
```

The bracketed tier tells you how the cue matched: `exact` (tail name
matched exactly), `qualified` (`Type::method` suffix), `prefix` /
`contains` (partial), `path` (file/directory), `fuzzy` (typo
correction — treat as a suggestion, verify before relying on it).

## Rules

1. **Trust, then verify at the site.** An `exact` hit's `file:lines`
   is where the definition was at the last sync — open the file at
   that line. If the code has drifted (see rule 3), the name is still
   correct; re-locate with Grep in that one file.
2. **Fall back honestly.** Exit code 1 with "no match" means the map
   does not know the name — fall back to Grep/Glob immediately; do
   not retry variations more than once. Symbols in generated code or
   non-Rust files are not in the map.
3. **The map is committed-state only.** Uncommitted edits are
   invisible to it. After commits (yours or a pull), run
   `taguru-code sync` — it is incremental and takes seconds. If
   `taguru-code status` says "behind HEAD", sync before trusting line
   numbers.
4. **Disambiguate with a qualified cue.** A bare `new` or `tests`
   matches many symbols; qualify it (`Api::new`,
   `deadline.rs`-then-tree) instead of paging through hits.
5. **Structure questions go to `tree`, not `find`.** "What handlers
   does src/api/ have" is one `tree src/api` call.

## Setup (once per clone)

```bash
taguru-code sync          # builds .taguru/ from committed state
echo '.taguru/' >> .gitignore   # if not already ignored
```
