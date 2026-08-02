//! `taguru compact`: rewrite context images without their dead
//! weight. The image is append-only by design — retraction unlinks
//! attribution records but never reclaims them, alias removal leaves
//! arena bytes behind — so a long-lived context with heavy revision
//! traffic grows monotonically. Compaction rebuilds each image from
//! its live content alone ([`taguru::context::Context::compacted`])
//! and persists the result; a running server serves the same at
//! `POST /contexts/{name}/compact`, and by default also runs the
//! policy itself, ratio-triggered from the flusher tick
//! (`TAGURU_AUTO_COMPACT`, issue #135) — this offline command remains
//! for opted-out deployments and for reclaiming without booting a
//! server. Passages need neither: their store compacts itself
//! ratio-triggered from inside the write path.
//!
//! `--url` (issue #246) makes this verb dual-mode, the same way
//! `export --url` (#245) is: absent, the local path above runs
//! unchanged; given a URL, CONTEXT arguments are compacted one at a
//! time through `POST /contexts/{name}/compact`, and no arguments
//! trigger the server's own `POST /maintenance/compact` sweep instead
//! of an enumerate-then-call loop — the sweep picks its own
//! candidates, worst dead ratio first.
//!
//! `--dry-run` and `--json` (issue #371) answer "is compacting this
//! worth it" without rewriting anything: `GET /contexts` (offline,
//! `state.directory()`) already carries each context's standing
//! `dead_edges`/`arena_slack`/… — the same numbers `POST
//! .../compact` would shed — so a preview needs no server-side
//! simulation of the rebuild it isn't running. See [`DeadWeight`].

use std::path::PathBuf;

use serde::Serialize;
use serde_json::json;

use crate::registry::{
    AccessError, CompactOutcome, DirectoryEntry, MaintenanceCompactionEntry,
    MaintenanceCompactionOutcome,
};
use crate::remote::Api;
use taguru::deadline::Deadline;

const USAGE: &str = "\
usage: taguru compact [--config FILE] [--url URL] [--parallel N]
                       [--dry-run] [--json] [CONTEXT...]

Rewrites context images in TAGURU_DATA_DIR without the dead weight the
append-only format accumulates (retracted edges, unlinked attribution
records, arena slack) — offline; the directory lock refuses to run
beside a live server. No CONTEXT arguments means every context. Live
systems use POST /contexts/{name}/compact instead (admin role; the
context's own requests wait out the rebuild). Content is preserved:
counts and paragraph locators exactly, per-source weights within
float re-accumulation error; aliases whose canonical no longer
carries any live association are dropped and counted.

  --config F     read KEY=VALUE environment from F (same dialect as serve)
  --url URL      compact a running server instead of TAGURU_DATA_DIR
                 directly. CONTEXT arguments each call their own
                 POST /contexts/{name}/compact; with none, calls the
                 server's POST /maintenance/compact sweep instead (its
                 own candidate list, not every context — --parallel
                 has no effect on this single request). Both need the
                 admin role (TAGURU_API_TOKEN or TAGURU_API_TOKENS). A
                 503 (the heavy-op ceiling, or a sweep already
                 running) is shown as reported, Retry-After included
                 when the server sent one — never retried
                 automatically; compaction is safe to re-run.
  --parallel N   compact up to N contexts at once (default 1, sequential);
                 output is reordered to match the sequential run byte for
                 byte, regardless of N or thread scheduling. With --url,
                 parallelizes the per-context HTTP calls the same way;
                 the server's own heavy-op ceiling (default 2 concurrent)
                 may shed the rest as 503s rather than queue them.
  --dry-run      report each context's standing dead weight (dead
                 edges, arena slack, unlinked attributions) and touch
                 nothing. With --url and no CONTEXT arguments, walks
                 GET /contexts instead of running the server's sweep —
                 every context's dead weight, not just the sweep's own
                 candidate list. Cannot predict bytes_after (the
                 rebuilt image size): this reports what is there to
                 shed, not what a rewrite would produce.
  --json         with --dry-run, one object per context (the same
                 fields as above, plus dead_ratio and whether the
                 numbers are a live read or a saved snapshot); without
                 --dry-run, one object per rewritten context in the
                 same shape POST /contexts/{name}/compact answers with.
";

pub(crate) fn run(args: &[String]) -> i32 {
    let mut config: Option<PathBuf> = None;
    let mut url: Option<String> = None;
    let mut parallel: Option<usize> = None;
    let mut dry_run = false;
    let mut as_json = false;
    let mut names: Vec<String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return 0;
            }
            "--dry-run" => dry_run = true,
            "--json" => as_json = true,
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => {
                    return crate::config::subcommand_usage_error(
                        "compact",
                        "--config given twice",
                    );
                }
                None => {
                    return crate::config::subcommand_usage_error(
                        "compact",
                        "--config needs a file path",
                    );
                }
            },
            // Trailing '/' trimmed the same way export/calibrate/communities
            // already do — a joined path segment must not double up.
            "--url" => match rest.next() {
                Some(value) if url.is_none() => {
                    url = Some(value.trim_end_matches('/').to_string());
                }
                Some(_) => {
                    return crate::config::subcommand_usage_error("compact", "--url given twice");
                }
                None => {
                    return crate::config::subcommand_usage_error(
                        "compact",
                        "--url needs a server URL",
                    );
                }
            },
            "--parallel" => match rest.next().map(|value| value.parse::<usize>()) {
                Some(_) if parallel.is_some() => {
                    return crate::config::subcommand_usage_error(
                        "compact",
                        "--parallel given twice",
                    );
                }
                Some(Ok(n)) if n >= 1 => parallel = Some(n),
                _ => {
                    return crate::config::subcommand_usage_error(
                        "compact",
                        "--parallel needs an integer of at least 1",
                    );
                }
            },
            other if other.starts_with('-') => {
                return crate::config::subcommand_usage_error(
                    "compact",
                    &format!("unknown flag '{other}'"),
                );
            }
            name => names.push(name.to_string()),
        }
    }
    // SAFETY (same contract as serve/import/export): applied while the
    // process is still single-threaded — no runtime ever starts here.
    // Loaded before the --url dispatch below: a config file is the
    // usual place a deployment's TAGURU_API_TOKEN lives, and Api::new
    // reads the bearer from the environment at construction.
    if let Some(path) = &config {
        crate::config::load_config(path);
    }

    let parallel = parallel.unwrap_or(1);
    match url {
        // ADR 0002 §5/§6: `--url` is the only way `compact` goes
        // remote — no positional URL argument, no TAGURU_URL or
        // default_base_url() fallback the way `health`/`calibrate`/
        // `communities` have one. Absent, this is exactly the local
        // path that ran before this flag existed (§12.2).
        Some(base) => {
            if dry_run {
                run_remote_dry_run(&base, names, as_json)
            } else {
                run_remote(&base, names, parallel, as_json)
            }
        }
        None => {
            if dry_run {
                run_local_dry_run(names, as_json)
            } else {
                run_local(names, parallel, as_json)
            }
        }
    }
}

fn run_local(names: Vec<String>, parallel: usize, as_json: bool) -> i32 {
    crate::ingest::init_logging();
    let state = match crate::registry::BootConfig::from_env().boot(None, None, None, None, None) {
        Ok(state) => state,
        Err(error) => {
            eprintln!("taguru: compact: {error}");
            return 1;
        }
    };

    let names = if names.is_empty() {
        let all: Vec<String> = state
            .directory()
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        if all.is_empty() {
            eprintln!("taguru: compact: the data directory holds no contexts");
            return 1;
        }
        all
    } else {
        names
    };

    let mut failures = 0usize;
    let mut reports: Vec<MaintenanceCompactionEntry> = Vec::new();
    if parallel <= 1 {
        for name in &names {
            let outcome = state.compact_context(name, Deadline::unbounded());
            if !report_outcome(name, &outcome, as_json, &mut reports) {
                failures += 1;
            }
        }
    } else {
        // Same shared work-queue pattern `preload_pinned` uses to load
        // pinned contexts at boot: independent per-entry locks mean the
        // workers never contend with each other.
        let indexed: Vec<(usize, &String)> = names.iter().enumerate().collect();
        let mut collected = crate::registry::parallel_map(indexed, parallel, |(index, name)| {
            (index, state.compact_context(name, Deadline::unbounded()))
        });
        // Reordered to the original argument order so `--parallel N`'s
        // stdout is byte-for-byte identical to the sequential run,
        // whatever N is or however the workers happened to race.
        collected.sort_by_key(|(index, _)| *index);
        for (name, (_, outcome)) in names.iter().zip(collected) {
            if !report_outcome(name, &outcome, as_json, &mut reports) {
                failures += 1;
            }
        }
    }
    if as_json {
        print_json(&reports);
    } else {
        println!(
            "compact: {} of {} context(s) rewritten",
            names.len() - failures,
            names.len()
        );
    }
    if failures > 0 { 1 } else { 0 }
}

/// `compact --dry-run` (offline): each context's standing dead weight,
/// read straight off `state.directory()` — the same live-for-hot,
/// snapshot-for-cold stats `GET /contexts` serves — with nothing
/// opened or rewritten.
fn run_local_dry_run(names: Vec<String>, as_json: bool) -> i32 {
    crate::ingest::init_logging();
    let state = match crate::registry::BootConfig::from_env().boot(None, None, None, None, None) {
        Ok(state) => state,
        Err(error) => {
            eprintln!("taguru: compact: {error}");
            return 1;
        }
    };

    let rows: Vec<(String, Option<DirectoryEntry>)> = if names.is_empty() {
        let all = state.directory();
        if all.is_empty() {
            eprintln!("taguru: compact: the data directory holds no contexts");
            return 1;
        }
        all.into_iter()
            .map(|entry| (entry.name.clone(), Some(entry)))
            .collect()
    } else {
        names
            .into_iter()
            .map(|name| {
                let entry = state.directory_entry(&name);
                (name, entry)
            })
            .collect()
    };

    let mut failures = 0usize;
    let mut weights = Vec::new();
    for (name, entry) in &rows {
        match entry {
            Some(entry) => weights.push(dead_weight_of(entry)),
            None => {
                eprintln!("taguru: compact: context '{name}': no such context");
                failures += 1;
            }
        }
    }
    finish_dry_run(weights, rows.len(), failures, as_json)
}

/// Prints one context's outcome in the shape both the sequential and
/// `--parallel` paths share, so the two can never drift apart. Returns
/// whether it succeeded, for the caller's failure count. `--json`
/// skips the print and accumulates into `reports` instead — collected
/// once, whole, so `--parallel`'s reordering (the caller's
/// responsibility) still lands the array in argument order.
fn report_outcome(
    name: &str,
    result: &Result<CompactOutcome, AccessError>,
    as_json: bool,
    reports: &mut Vec<MaintenanceCompactionEntry>,
) -> bool {
    match result {
        Ok(outcome) => {
            if as_json {
                reports.push(MaintenanceCompactionEntry {
                    name: name.to_string(),
                    outcome: *outcome,
                });
            } else {
                println!("{}", success_line(name, outcome));
            }
            true
        }
        Err(failure) => {
            let message = match failure {
                AccessError::NotFound => "no such context".to_string(),
                AccessError::Load(error) => error.clone(),
                AccessError::Unpersisted(error) => error.clone(),
                // The CLI runs with Deadline::unbounded(), which never
                // expires — unreachable in practice, kept for
                // exhaustiveness.
                AccessError::DeadlineExceeded => "deadline exceeded".to_string(),
                // Same unreachability: the offline CLI boots with no
                // quota declaration (and compaction shrinks anyway).
                AccessError::QuotaExceeded(error) => error.clone(),
            };
            eprintln!("taguru: compact: context '{name}': {message}");
            false
        }
    }
}

/// One context's report line — shared by the sequential path, the
/// `--parallel` path, and both `--url` paths (per-context and the
/// maintenance sweep), so the four can never drift apart.
fn success_line(name: &str, outcome: &CompactOutcome) -> String {
    format!(
        "context '{name}': {} → {} ({} dead edge(s) shed{})",
        crate::config::fmt_bytes(outcome.bytes_before as u64),
        crate::config::fmt_bytes(outcome.bytes_after as u64),
        outcome.dead_edges,
        match outcome.aliases_dropped {
            0 => String::new(),
            dropped => format!(", {dropped} alias(es) dropped (canonical has no live association)"),
        },
    )
}

/// One context's standing dead weight — what `--dry-run` reports and
/// `--dry-run --json` serializes. Deliberately NOT [`CompactOutcome`]:
/// that type answers "what did the rewrite shed", this one answers
/// "what is there to shed", and `bytes_after` (the rebuilt image size)
/// is not predictable from stats alone.
#[derive(Serialize)]
struct DeadWeight {
    context: String,
    associations: usize,
    dead_edges: usize,
    dead_ratio: f64,
    dead_attributions: usize,
    arena_slack: usize,
    footprint_bytes: usize,
    /// True when `entry.loaded` was false: a cold context's stats are
    /// the last-saved snapshot, not a live recomputation
    /// (`describe_entry`, registry.rs) — never presented as live
    /// without saying so.
    stats_are_snapshot: bool,
}

fn dead_weight_of(entry: &DirectoryEntry) -> DeadWeight {
    DeadWeight {
        context: entry.name.clone(),
        associations: entry.stats.associations,
        dead_edges: entry.stats.dead_edges,
        dead_ratio: entry.stats.dead_ratio(),
        dead_attributions: entry.stats.dead_attributions,
        arena_slack: entry.stats.arena_slack,
        footprint_bytes: entry.stats.footprint_bytes,
        stats_are_snapshot: !entry.loaded,
    }
}

/// [`success_line`]'s `--dry-run` twin — offline and remote share it
/// the same way the two `success_line` callers do.
fn dead_weight_line(weight: &DeadWeight) -> String {
    format!(
        "context '{}': {} associations, {} dead edge(s) ({:.1}% dead), {} unlinked \
         attribution(s), {} arena slack{}",
        weight.context,
        weight.associations,
        weight.dead_edges,
        weight.dead_ratio * 100.0,
        weight.dead_attributions,
        crate::config::fmt_bytes(weight.arena_slack as u64),
        if weight.stats_are_snapshot {
            " (snapshot: context is not currently loaded)"
        } else {
            ""
        },
    )
}

/// Serializes a `--json` report list to stdout as one document.
/// Shared by every `--json` exit path (dry-run and real, local and
/// remote) so a serialization failure is handled once.
fn print_json<T: Serialize>(items: &[T]) -> bool {
    match serde_json::to_string_pretty(items) {
        Ok(text) => {
            println!("{text}");
            true
        }
        Err(error) => {
            eprintln!("taguru: compact: report did not serialize: {error}");
            false
        }
    }
}

/// The shared tail of every `--dry-run` path (local and remote): print
/// or serialize what was gathered, then report exit status the same
/// way the real compaction paths do — failures from missing/unreadable
/// contexts count same as a serialization failure.
fn finish_dry_run(weights: Vec<DeadWeight>, total: usize, failures: usize, as_json: bool) -> i32 {
    let mut ok = true;
    if as_json {
        ok = print_json(&weights);
    } else {
        let carrying_weight = weights.iter().filter(|w| w.dead_edges > 0).count();
        for weight in &weights {
            println!("{}", dead_weight_line(weight));
        }
        println!(
            "dry run: {carrying_weight} of {total} context(s) carry dead weight, nothing rewritten"
        );
    }
    if failures > 0 || !ok { 1 } else { 0 }
}

/// The remote twin of [`run_local`]: CONTEXT arguments each call
/// their own `POST /contexts/{name}/compact`; no arguments call the
/// server's `POST /maintenance/compact` sweep instead, which picks
/// its own candidates rather than enumerating every context first
/// (ADR 0002 §6).
fn run_remote(base: &str, names: Vec<String>, parallel: usize, as_json: bool) -> i32 {
    // ADR 0002 §7: caught before any request leaves the process.
    if let Err(message) = crate::remote::reject_userinfo(base) {
        return crate::config::subcommand_usage_error("compact", &message);
    }
    let api = Api::new(base.to_string());
    // ADR 0002 §5: every remote, mutating invocation prints its target
    // before sending anything — the one line that lets an operator
    // notice a stale --url value pasted from history before the
    // request lands, without adding a confirmation prompt.
    eprintln!("compact → {base}");
    api.warn_on_version_skew("compact");

    if names.is_empty() {
        run_remote_sweep(&api, as_json)
    } else {
        run_remote_contexts(&api, &names, parallel, as_json)
    }
}

/// One `POST /contexts/{name}/compact`, decoded back into the same
/// [`CompactOutcome`] the local path reports — the server's `result`
/// and the library's in-process return value are the same shape.
fn remote_compact_one(api: &Api, name: &str) -> Result<CompactOutcome, String> {
    let result = api.post(&["contexts", name, "compact"], &json!({}))?;
    serde_json::from_value(result)
        .map_err(|error| format!("unrecognized compact response: {error}"))
}

fn run_remote_contexts(api: &Api, names: &[String], parallel: usize, as_json: bool) -> i32 {
    let mut failures = 0usize;
    let mut reports: Vec<MaintenanceCompactionEntry> = Vec::new();
    if parallel <= 1 {
        for name in names {
            if !report_remote_outcome(name, &remote_compact_one(api, name), as_json, &mut reports) {
                failures += 1;
            }
        }
    } else {
        // Same index-then-sort shape run_local's own --parallel path
        // uses: independent HTTP calls, reordered back to the
        // original argument order so stdout doesn't depend on which
        // request happened to answer first.
        let indexed: Vec<(usize, &String)> = names.iter().enumerate().collect();
        let mut collected = crate::registry::parallel_map(indexed, parallel, |(index, name)| {
            (index, remote_compact_one(api, name))
        });
        collected.sort_by_key(|(index, _)| *index);
        for (name, (_, outcome)) in names.iter().zip(collected) {
            if !report_remote_outcome(name, &outcome, as_json, &mut reports) {
                failures += 1;
            }
        }
    }
    let mut ok = true;
    if as_json {
        ok = print_json(&reports);
    } else {
        println!(
            "compact: {} of {} context(s) rewritten",
            names.len() - failures,
            names.len()
        );
    }
    if failures > 0 || !ok { 1 } else { 0 }
}

/// [`report_outcome`]'s remote twin: same success line, but the
/// failure side carries whatever message the server (or the HTTP
/// layer) produced instead of an [`AccessError`] — a per-item failure
/// the run continues past, same as `export --url`.
fn report_remote_outcome(
    name: &str,
    result: &Result<CompactOutcome, String>,
    as_json: bool,
    reports: &mut Vec<MaintenanceCompactionEntry>,
) -> bool {
    match result {
        Ok(outcome) => {
            if as_json {
                reports.push(MaintenanceCompactionEntry {
                    name: name.to_string(),
                    outcome: *outcome,
                });
            } else {
                println!("{}", success_line(name, outcome));
            }
            true
        }
        Err(message) => {
            eprintln!("taguru: compact: context '{name}': {message}");
            false
        }
    }
}

/// `POST /maintenance/compact`: the server sweeps its own candidates
/// (worst dead ratio first) instead of the CLI enumerating every
/// context and calling each one — the one request this issues either
/// succeeds as a whole or fails as a whole, so `--parallel` has
/// nothing to parallelize here.
fn run_remote_sweep(api: &Api, as_json: bool) -> i32 {
    match api.post(&["maintenance", "compact"], &json!({})) {
        Err(message) => {
            eprintln!("taguru: compact: {message}");
            1
        }
        Ok(result) => match serde_json::from_value::<MaintenanceCompactionOutcome>(result) {
            Err(error) => {
                eprintln!("taguru: compact: unrecognized sweep response: {error}");
                1
            }
            Ok(outcome) => {
                let mut ok = true;
                if as_json {
                    ok = print_json(&outcome.contexts);
                } else {
                    for entry in &outcome.contexts {
                        println!("{}", success_line(&entry.name, &entry.outcome));
                    }
                    println!(
                        "compact: server sweep rewrote {} context(s)",
                        outcome.contexts.len()
                    );
                }
                if outcome.deadline_exceeded {
                    eprintln!(
                        "taguru: compact: the sweep's deadline expired before every candidate \
                         was compacted — re-run to continue (compaction is safe to repeat)"
                    );
                    return 1;
                }
                if ok { 0 } else { 1 }
            }
        },
    }
}

/// `compact --dry-run --url`: no CONTEXT arguments walks every row of
/// `GET /contexts` (issue #371) rather than the server's own sweep —
/// the sweep picks its own worst-first candidates, but a preview must
/// answer for every context, since the point is deciding whether a
/// context is worth compacting in the first place. CONTEXT arguments
/// each fetch their own `GET /contexts/{name}` instead.
fn run_remote_dry_run(base: &str, names: Vec<String>, as_json: bool) -> i32 {
    if let Err(message) = crate::remote::reject_userinfo(base) {
        return crate::config::subcommand_usage_error("compact", &message);
    }
    let api = Api::new(base.to_string());
    eprintln!("compact --dry-run → {base}");
    api.warn_on_version_skew("compact");

    let mut failures = 0usize;
    let mut weights = Vec::new();
    let total;
    if names.is_empty() {
        match api.list_context_entries() {
            Ok(entries) => {
                total = entries.len();
                if entries.is_empty() {
                    eprintln!("taguru: compact: the server holds no contexts");
                    return 1;
                }
                weights.extend(entries.iter().map(dead_weight_of));
            }
            Err(message) => {
                eprintln!("taguru: compact: {message}");
                return 1;
            }
        }
    } else {
        total = names.len();
        for name in &names {
            match api.get(&["contexts", name]) {
                Ok(value) => match serde_json::from_value::<DirectoryEntry>(value) {
                    Ok(entry) => weights.push(dead_weight_of(&entry)),
                    Err(error) => {
                        eprintln!(
                            "taguru: compact: context '{name}': unrecognized response: {error}"
                        );
                        failures += 1;
                    }
                },
                Err(message) => {
                    eprintln!("taguru: compact: context '{name}': {message}");
                    failures += 1;
                }
            }
        }
    }
    finish_dry_run(weights, total, failures, as_json)
}

#[cfg(test)]
mod tests {
    use super::{CompactOutcome, MaintenanceCompactionOutcome, success_line};

    /// A `POST /maintenance/compact` response decodes into the exact
    /// struct the local path already reports through — proof that the
    /// `#[serde(flatten)]` on [`crate::registry::MaintenanceCompactionEntry`]
    /// round-trips, and that `success_line`'s output for a sweep entry
    /// cannot silently drift from the per-context report line.
    #[test]
    fn a_sweep_response_deserializes_into_the_same_report_line_as_a_per_context_one() {
        let body = serde_json::json!({
            "contexts": [
                {
                    "name": "sake",
                    "bytes_before": 1000,
                    "bytes_after": 400,
                    "dead_edges": 3,
                    "aliases_dropped": 1,
                }
            ],
            "deadline_exceeded": false,
        });
        let outcome: MaintenanceCompactionOutcome =
            serde_json::from_value(body).expect("a sweep response must decode");
        assert_eq!(outcome.contexts.len(), 1);
        assert!(!outcome.deadline_exceeded);

        let entry = &outcome.contexts[0];
        assert_eq!(entry.name, "sake");
        assert_eq!(
            success_line(&entry.name, &entry.outcome),
            success_line(
                "sake",
                &CompactOutcome {
                    bytes_before: 1000,
                    bytes_after: 400,
                    dead_edges: 3,
                    aliases_dropped: 1,
                }
            ),
        );
        assert!(
            success_line(&entry.name, &entry.outcome).contains("1 alias(es) dropped"),
            "{}",
            success_line(&entry.name, &entry.outcome)
        );
    }
}
