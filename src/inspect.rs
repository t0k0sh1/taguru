//! `taguru inspect`: offline verification of a data directory, one
//! `.ctx` image, or one `.group` record — the backup check that needs
//! no server. Every image goes through the same fully validating
//! parser the server boots with, every WAL through the same replay
//! parser, and every group file through the same record parse, so
//! "inspect says ok" and "the server will load it" are one statement.
//! Exits 1 when anything holding acknowledged data is corrupt — a
//! group file that would not parse included, because boot answers
//! that by resetting the record (the membership is acknowledged data,
//! and this is the tool that must say so BEFORE a restore spends it).
//!
//! `--json` (issue #371) is a second rendering of the exact same
//! facts the text report prints, never a second computation of them:
//! every count that reaches the human line also reaches
//! [`InspectReport`], built alongside it in the same scope from the
//! same local variables, so the two can never disagree about what
//! inspect actually found.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Serialize;
use taguru::context::Context;

use crate::config::fmt_bytes;
use crate::groups::{
    GroupRecord, MAX_GROUP_DEPTH, MAX_GROUP_MEMBERS, repair_nesting, trim_membership,
};
use crate::registry::{
    IMPORT_MARKER_EXTENSION, ImportMarker, bm25_path, meta_path, name_from_stem, passages_path,
    passages_wal_path, pvectors_path, scanned_stem_and_name, sources_path, vectors_path, wal_path,
};
use crate::wal;

const USAGE: &str = "\
usage: taguru inspect PATH [--json]   (a data directory, one .ctx image, or one .group record)

  --json   one JSON document instead of per-context/group text lines:
           {target, kind, contexts, groups, notices, total, corrupt}.
           The exit code and what counts as CORRUPT are unchanged —
           only how the result is rendered.
";

pub fn run(args: &[String]) -> i32 {
    let mut as_json = false;
    let mut positionals: Vec<&str> = Vec::new();
    for arg in args {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return 0;
            }
            "--json" => as_json = true,
            other if other.starts_with('-') => {
                eprint!("{USAGE}");
                return 2;
            }
            value => positionals.push(value),
        }
    }
    let path = match positionals.as_slice() {
        [path] => Path::new(*path),
        _ => {
            eprint!("{USAGE}");
            return 2;
        }
    };
    if path.is_dir() {
        inspect_directory(path, as_json)
    } else if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("group") {
            inspect_group_file(path, as_json)
        } else {
            inspect_file(path, as_json)
        }
    } else {
        eprintln!(
            "taguru: inspect: {} is neither a file nor a directory",
            path.display()
        );
        2
    }
}

/// One `level`/`kind`/`subject`/`message` finding — a warning names an
/// alteration boot would make when it next loads the data (a dropped
/// reference, a reset record); a note names something already healed
/// or otherwise informational. `kind` is a stable machine key; `message`
/// is the same prose the human report prints (verbatim in every case
/// where a single note maps to one printed line — see call sites for
/// the few that combine several notes' messages into one suffix).
#[derive(Serialize)]
struct Notice {
    level: &'static str,
    kind: &'static str,
    subject: String,
    message: String,
}

impl Notice {
    fn warning(kind: &'static str, subject: impl Into<String>, message: impl Into<String>) -> Self {
        Notice {
            level: "warning",
            kind,
            subject: subject.into(),
            message: message.into(),
        }
    }

    fn note(kind: &'static str, subject: impl Into<String>, message: impl Into<String>) -> Self {
        Notice {
            level: "note",
            kind,
            subject: subject.into(),
            message: message.into(),
        }
    }
}

/// One `.ctx` image's report — the same fields [`stats_line`] prints,
/// plus (directory scans only) the sidecar sizes the text report's
/// per-context line appends. `status` other than `"ok"` means every
/// field below `error` is absent: a corrupt image, WAL, or passage
/// store cannot report counts it never finished loading.
#[derive(Serialize)]
struct ContextRow {
    name: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    associations: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    concepts: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sources: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    footprint_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    applied_seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dead_edges: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dead_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dead_attributions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arena_slack: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unsourced_edges: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unsourced_weight: Option<f64>,
    /// Sidecar fields: present for a directory scan (which reads WAL,
    /// vectors, index, and passages alongside the image), absent for a
    /// bare single-file `inspect PATH.ctx` (which reads only the image).
    #[serde(skip_serializing_if = "Option::is_none")]
    wal_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    wal_pending: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vector_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    passage_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    passage_count: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    notes: Vec<Notice>,
}

/// The per-context sidecar sizes a directory scan (not a bare
/// single-file inspect) reads alongside the image.
struct Sidecars {
    wal_bytes: u64,
    wal_pending: usize,
    vector_bytes: u64,
    index_bytes: u64,
    passage_bytes: u64,
    passage_count: usize,
}

impl ContextRow {
    fn ok(
        name: String,
        context: &Context,
        image_bytes: u64,
        generation: &str,
        sidecars: Option<Sidecars>,
    ) -> Self {
        let (unsourced_edges, unsourced_weight) = context.unsourced_summary();
        let (wal_bytes, wal_pending, vector_bytes, index_bytes, passage_bytes, passage_count) =
            match sidecars {
                Some(s) => (
                    Some(s.wal_bytes),
                    Some(s.wal_pending),
                    Some(s.vector_bytes),
                    Some(s.index_bytes),
                    Some(s.passage_bytes),
                    Some(s.passage_count),
                ),
                None => (None, None, None, None, None, None),
            };
        ContextRow {
            name,
            status: "ok",
            error: None,
            image_bytes: Some(image_bytes),
            generation: Some(generation.to_string()),
            associations: Some(context.association_count()),
            concepts: Some(context.concept_count()),
            labels: Some(context.label_count()),
            sources: Some(context.source_count()),
            footprint_bytes: Some(context.footprint() as u64),
            applied_seq: Some(context.applied_seq()),
            dead_edges: Some(context.dead_edges()),
            dead_ratio: Some(context.dead_ratio()),
            dead_attributions: Some(context.dead_attributions()),
            arena_slack: Some(context.arena_slack() as u64),
            unsourced_edges: Some(unsourced_edges),
            unsourced_weight: Some(unsourced_weight),
            wal_bytes,
            wal_pending,
            vector_bytes,
            index_bytes,
            passage_bytes,
            passage_count,
            notes: Vec::new(),
        }
    }

    fn corrupt(name: String, status: &'static str, error: String) -> Self {
        ContextRow {
            name,
            status,
            error: Some(error),
            image_bytes: None,
            generation: None,
            associations: None,
            concepts: None,
            labels: None,
            sources: None,
            footprint_bytes: None,
            applied_seq: None,
            dead_edges: None,
            dead_ratio: None,
            dead_attributions: None,
            arena_slack: None,
            unsourced_edges: None,
            unsourced_weight: None,
            wal_bytes: None,
            wal_pending: None,
            vector_bytes: None,
            index_bytes: None,
            passage_bytes: None,
            passage_count: None,
            notes: Vec::new(),
        }
    }
}

/// One `.group` record's report. `status` other than `"ok"` means the
/// record never parsed, so `contexts`/`groups` read 0 rather than a
/// guess.
#[derive(Serialize)]
struct GroupRow {
    name: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    contexts: usize,
    groups: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    notes: Vec<Notice>,
}

impl GroupRow {
    fn ok(name: String, record: &GroupRecord) -> Self {
        GroupRow {
            name,
            status: "ok",
            error: None,
            contexts: record.contexts.len(),
            groups: record.groups.len(),
            notes: Vec::new(),
        }
    }

    fn trouble(name: String, status: &'static str, error: String) -> Self {
        GroupRow {
            name,
            status,
            error: Some(error),
            contexts: 0,
            groups: 0,
            notes: Vec::new(),
        }
    }
}

/// The directory scan's footer line, structured — same six sums the
/// text report's `total:` line prints.
#[derive(Serialize)]
struct Totals {
    contexts: usize,
    groups: usize,
    image_bytes: u64,
    wal_bytes: u64,
    vector_bytes: u64,
    index_bytes: u64,
    passage_bytes: u64,
    footprint_bytes: u64,
}

/// `--json`'s whole answer, one document regardless of which of the
/// three targets (directory / image / group) `PATH` named.
#[derive(Serialize)]
struct InspectReport {
    target: String,
    kind: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    contexts: Vec<ContextRow>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    groups: Vec<GroupRow>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    notices: Vec<Notice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<Totals>,
    corrupt: usize,
}

fn print_json(report: &InspectReport) {
    match serde_json::to_string_pretty(report) {
        Ok(text) => println!("{text}"),
        Err(error) => eprintln!("taguru: inspect: report did not serialize: {error}"),
    }
}

/// One `.group` file through the parse boot runs, read and parse kept
/// apart — [`load_image`]'s twin one storey up, except the two failure
/// classes cost differently (an unreadable file refuses a boot;
/// unparseable bytes reset the record), so both callers must hear
/// which.
fn load_group(path: &Path) -> Result<GroupRecord, GroupFileTrouble> {
    let bytes = std::fs::read(path).map_err(GroupFileTrouble::Unreadable)?;
    serde_json::from_slice(&bytes).map_err(GroupFileTrouble::Corrupt)
}

enum GroupFileTrouble {
    Unreadable(std::io::Error),
    Corrupt(serde_json::Error),
}

/// One bare `.group` record: the "is this group file I restored
/// intact" question, [`inspect_file`]'s twin one storey up. Reference
/// checks need the directory around it, so a single file answers for
/// its own parse alone.
fn inspect_group_file(path: &Path, as_json: bool) -> i32 {
    let target = path.display().to_string();
    match load_group(path) {
        Ok(record) => {
            if as_json {
                print_json(&InspectReport {
                    target,
                    kind: "group",
                    contexts: Vec::new(),
                    groups: vec![GroupRow::ok(path.display().to_string(), &record)],
                    notices: Vec::new(),
                    total: None,
                    corrupt: 0,
                });
            } else {
                println!(
                    "{}: ok  {} member context(s) · {} child group(s)",
                    path.display(),
                    record.contexts.len(),
                    record.groups.len()
                );
            }
            0
        }
        Err(GroupFileTrouble::Unreadable(error)) => {
            if as_json {
                print_json(&InspectReport {
                    target,
                    kind: "group",
                    contexts: Vec::new(),
                    groups: vec![GroupRow::trouble(
                        path.display().to_string(),
                        "unreadable",
                        error.to_string(),
                    )],
                    notices: Vec::new(),
                    total: None,
                    corrupt: 1,
                });
            } else {
                eprintln!("{}: UNREADABLE — {error}", path.display());
            }
            1
        }
        Err(GroupFileTrouble::Corrupt(error)) => {
            if as_json {
                print_json(&InspectReport {
                    target,
                    kind: "group",
                    contexts: Vec::new(),
                    groups: vec![GroupRow::trouble(
                        path.display().to_string(),
                        "corrupt",
                        error.to_string(),
                    )],
                    notices: Vec::new(),
                    total: None,
                    corrupt: 1,
                });
            } else {
                eprintln!("{}: CORRUPT — {error}", path.display());
            }
            1
        }
    }
}

/// One bare image, no sidecars: the "is this .ctx I restored intact"
/// question.
fn inspect_file(path: &Path, as_json: bool) -> i32 {
    let target = path.display().to_string();
    match load_image(path) {
        Ok((context, image_bytes, generation)) => {
            if as_json {
                let row = ContextRow::ok(
                    path.display().to_string(),
                    &context,
                    image_bytes,
                    &generation,
                    None,
                );
                print_json(&InspectReport {
                    target,
                    kind: "image",
                    contexts: vec![row],
                    groups: Vec::new(),
                    notices: Vec::new(),
                    total: None,
                    corrupt: 0,
                });
            } else {
                println!(
                    "{}: ok  {}",
                    path.display(),
                    stats_line(&context, image_bytes, &generation)
                );
            }
            0
        }
        Err(error) => {
            if as_json {
                print_json(&InspectReport {
                    target,
                    kind: "image",
                    contexts: vec![ContextRow::corrupt(
                        path.display().to_string(),
                        "corrupt_image",
                        error.clone(),
                    )],
                    groups: Vec::new(),
                    notices: Vec::new(),
                    total: None,
                    corrupt: 1,
                });
            } else {
                eprintln!("{}: CORRUPT — {error}", path.display());
            }
            1
        }
    }
}

fn inspect_directory(dir: &Path, as_json: bool) -> i32 {
    // One listing serves both halves: the .ctx stems here, the .group
    // files handed to `inspect_groups` below.
    let mut entries: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
        Ok(read) => read
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .collect(),
        Err(error) => {
            eprintln!("taguru: inspect: cannot read {}: {error}", dir.display());
            return 2;
        }
    };
    entries.sort();
    let stems: Vec<String> = entries
        .iter()
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("ctx"))
        .filter_map(|path| path.file_stem().and_then(|s| s.to_str()).map(String::from))
        .collect();
    // Context EXISTENCE is file presence — a corrupt image still
    // occupies its name at boot — so this set is what the group
    // reference warnings below judge against.
    let context_names: BTreeSet<String> = stems
        .iter()
        .filter_map(|stem| name_from_stem(stem))
        .collect();

    let mut failures = 0usize;
    let mut contexts = 0usize;
    let mut image_total = 0u64;
    let mut footprint_total = 0u64;
    let mut wal_total = 0u64;
    let mut vectors_total = 0u64;
    let mut index_total = 0u64;
    let mut passages_total = 0u64;
    let mut context_rows: Vec<ContextRow> = Vec::new();
    let mut notices: Vec<Notice> = Vec::new();

    for stem in &stems {
        let name = match name_from_stem(stem) {
            Some(name) => name,
            None => {
                // Not a failure: the server skips it too — but a backup
                // holding files the server will never serve is worth a line.
                if as_json {
                    notices.push(Notice::warning(
                        "undecodable_stem",
                        format!("{stem}.ctx"),
                        "stem does not decode; the server will skip it",
                    ));
                } else {
                    println!("{stem}.ctx: WARNING — stem does not decode; the server will skip it");
                }
                continue;
            }
        };
        let image = dir.join(format!("{stem}.ctx"));
        let (context, image_bytes, generation) = match load_image(&image) {
            Ok(loaded) => loaded,
            Err(error) => {
                if as_json {
                    context_rows.push(ContextRow::corrupt(name.clone(), "corrupt_image", error));
                } else {
                    println!("{name}: CORRUPT image — {error}");
                }
                failures += 1;
                continue;
            }
        };

        // The same parse a boot-time replay would run, but READ-ONLY:
        // inspect audits a directory (often a backup) and must never
        // mutate it, so where the server would truncate a torn tail it
        // uses `replay_readonly`, which heals nothing and hands back the
        // torn size to report. Records at or below the image's watermark
        // are inert; the ones above are acknowledged writes the image
        // does not carry yet.
        let (pending, wal_torn, wal_unchecked) =
            match wal::replay_readonly::<wal::WalOp>(&wal_path(dir, stem), context.applied_seq()) {
                Ok((ops, _, torn, unchecked)) => (ops.len(), torn, unchecked),
                Err(error) => {
                    if as_json {
                        context_rows.push(ContextRow::corrupt(
                            name.clone(),
                            "corrupt_wal",
                            error.to_string(),
                        ));
                    } else {
                        println!("{name}: CORRUPT WAL — {error}");
                    }
                    failures += 1;
                    continue;
                }
            };

        // The passage snapshot and its log hold acknowledged text that
        // exists nowhere else — run the same strict load the server uses
        // (legacy .sources.json keeps its lenient contract inside it),
        // but with heal=false so this stays read-only. Vectors stay a
        // size: a derived cache's corruption costs a re-embed, never data.
        let (passage_count, passages_torn, passages_unchecked) =
            match crate::passages::PassageStore::load(
                passages_path(dir, stem),
                &sources_path(dir, stem),
                passages_wal_path(dir, stem),
                0,
                false,
            ) {
                Ok((store, torn, unchecked)) => (store.source_ids().len(), torn, unchecked),
                Err(error) => {
                    if as_json {
                        context_rows.push(ContextRow::corrupt(
                            name.clone(),
                            "corrupt_passages",
                            error.to_string(),
                        ));
                    } else {
                        println!("{name}: CORRUPT passages — {error}");
                    }
                    failures += 1;
                    continue;
                }
            };

        // Meta is self-healing on the server side (defaults + warning),
        // so a broken one is reported without failing the inspection.
        let meta_unparseable = matches!(
            std::fs::read(meta_path(dir, stem)),
            Ok(bytes) if serde_json::from_slice::<serde_json::Value>(&bytes).is_err()
        );
        let meta_note = if meta_unparseable {
            " · WARNING: meta.json unparseable (description/usage will reset)"
        } else {
            ""
        };

        // A non-empty tail is a crash mid-append, not corruption, but the
        // two TornTail shapes are healed — and mean — different things:
        // a Discarded fragment never happened as far as the log is
        // concerned (never acknowledged, healed by truncation), while a
        // Recovered record is complete and already counted in `pending`
        // above (healed by appending its missing newline). Report
        // whichever it is (inspect left the file untouched) rather than
        // failing — but never let it pass silently, and never blur the
        // two together.
        fn torn_tail_note(label: &str, torn: wal::TornTail) -> String {
            match torn {
                wal::TornTail::Discarded { bytes } => format!(
                    "{label} torn tail ({bytes} bytes) — crash mid-append, healed by \
                     truncation on the server's next load"
                ),
                wal::TornTail::Recovered { bytes } => format!(
                    "{label} tail missing its trailing newline ({bytes} bytes) — the \
                     record itself is complete and already counted above, healed by \
                     appending the newline on the server's next load"
                ),
            }
        }
        // Computed once, read by both the human suffix below and the
        // JSON notes further down — one string, not two that could say
        // different things about the same torn tail.
        let wal_torn_note = wal_torn.map(|torn| torn_tail_note("WAL", torn));
        let passages_torn_note = passages_torn.map(|torn| torn_tail_note("passages WAL", torn));
        let mut torn_parts = Vec::new();
        if let Some(note) = &wal_torn_note {
            torn_parts.push(note.clone());
        }
        if let Some(note) = &passages_torn_note {
            torn_parts.push(note.clone());
        }
        let torn_note = if torn_parts.is_empty() {
            String::new()
        } else {
            format!(
                " · NOTE: {} (inspect left it untouched)",
                torn_parts.join("; ")
            )
        };

        // Checksummed records were verified byte-for-byte by the replay
        // above; records from a pre-checksum writer can only be parsed.
        // Say so — "inspect says ok" must not overclaim on a log that
        // predates verifiability.
        let mut unverified_parts = Vec::new();
        if wal_unchecked > 0 {
            unverified_parts.push(format!("{wal_unchecked} WAL record(s)"));
        }
        if passages_unchecked > 0 {
            unverified_parts.push(format!("{passages_unchecked} passages WAL record(s)"));
        }
        let unverified_note = if unverified_parts.is_empty() {
            String::new()
        } else {
            format!(
                " · NOTE: {} predate checksums — parsed, but not verifiable bit-for-bit",
                unverified_parts.join(" and ")
            )
        };

        let wal_bytes = file_size(&wal_path(dir, stem));
        // The vector sidecars and the BM25 index are derived caches —
        // size-only here.
        let vector_bytes =
            file_size(&vectors_path(dir, stem)) + file_size(&pvectors_path(dir, stem));
        let index_bytes = file_size(&bm25_path(dir, stem));
        let passage_bytes = file_size(&passages_path(dir, stem))
            + file_size(&passages_wal_path(dir, stem))
            + file_size(&sources_path(dir, stem));

        if as_json {
            let mut row = ContextRow::ok(
                name.clone(),
                &context,
                image_bytes,
                &generation,
                Some(Sidecars {
                    wal_bytes,
                    wal_pending: pending,
                    vector_bytes,
                    index_bytes,
                    passage_bytes,
                    passage_count,
                }),
            );
            if meta_unparseable {
                row.notes.push(Notice::warning(
                    "meta_unparseable",
                    name.clone(),
                    "meta.json unparseable (description/usage will reset)",
                ));
            }
            if let Some(note) = wal_torn_note {
                row.notes
                    .push(Notice::note("wal_torn_tail", name.clone(), note));
            }
            if let Some(note) = passages_torn_note {
                row.notes
                    .push(Notice::note("passages_wal_torn_tail", name.clone(), note));
            }
            if wal_unchecked > 0 {
                row.notes.push(Notice::note(
                    "unverified_pre_checksum_wal",
                    name.clone(),
                    format!(
                        "{wal_unchecked} WAL record(s) predate checksums — parsed, but not \
                         verifiable bit-for-bit"
                    ),
                ));
            }
            if passages_unchecked > 0 {
                row.notes.push(Notice::note(
                    "unverified_pre_checksum_passages_wal",
                    name.clone(),
                    format!(
                        "{passages_unchecked} passages WAL record(s) predate checksums — \
                         parsed, but not verifiable bit-for-bit"
                    ),
                ));
            }
            context_rows.push(row);
        } else {
            println!(
                "{name}: ok  {} · WAL {} ({pending} pending) · vectors {} · index {} · passages {} \
                 ({passage_count} sources){meta_note}{torn_note}{unverified_note}",
                stats_line(&context, image_bytes, &generation),
                fmt_bytes(wal_bytes),
                fmt_bytes(vector_bytes),
                fmt_bytes(index_bytes),
                fmt_bytes(passage_bytes),
            );
        }

        contexts += 1;
        image_total += image_bytes;
        footprint_total += context.footprint() as u64;
        wal_total += wal_bytes;
        vectors_total += vector_bytes;
        index_total += index_bytes;
        passages_total += passage_bytes;
    }

    // Surviving import batch markers: each names a source whose
    // multi-store import (retract → passages → associations → aliases)
    // opened and never finished. The stores are individually
    // consistent, so the marker is the only witness that the source's
    // truth may be half-applied. A warning, not a failure: the bytes
    // are intact and the repair is documented — re-import the batch
    // file (per-source retract-then-apply is idempotent) or retract
    // the source.
    for path in &entries {
        if path.extension().and_then(|e| e.to_str()) != Some(IMPORT_MARKER_EXTENSION) {
            continue;
        }
        let file = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
        let parsed = std::fs::read(path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ImportMarker>(&bytes).ok());
        match parsed {
            Some(marker) if context_names.contains(&marker.context) => {
                if as_json {
                    notices.push(Notice::warning(
                        "incomplete_import",
                        marker.context.clone(),
                        format!(
                            "the import of source '{}' never completed; its truth may be \
                             half-applied — re-import its batch file or retract the source",
                            marker.source
                        ),
                    ));
                } else {
                    println!(
                        "{}: WARNING — the import of source '{}' never completed; its truth \
                         may be half-applied — re-import its batch file or retract the source",
                        marker.context, marker.source
                    );
                }
            }
            Some(marker) => {
                if as_json {
                    notices.push(Notice::note(
                        "orphan_import_marker",
                        file.to_string(),
                        format!(
                            "import marker for context '{}', which no longer exists here (the \
                             server's next boot removes it)",
                            marker.context
                        ),
                    ));
                } else {
                    println!(
                        "{file}: NOTE — import marker for context '{}', which no longer \
                         exists here (the server's next boot removes it)",
                        marker.context
                    );
                }
            }
            None => {
                if as_json {
                    notices.push(Notice::warning(
                        "unreadable_import_marker",
                        file.to_string(),
                        "unreadable import marker; an import batch may be half-applied, but \
                         which source is unrecoverable",
                    ));
                } else {
                    println!(
                        "{file}: WARNING — unreadable import marker; an import batch may be \
                         half-applied, but which source is unrecoverable"
                    );
                }
            }
        }
    }

    let (group_count, group_failures, group_rows, group_notices) =
        inspect_groups(&entries, &context_names, as_json);
    failures += group_failures;
    notices.extend(group_notices);

    if stems.is_empty() && group_count == 0 && group_failures == 0 {
        if as_json {
            // `notices` (import-marker warnings/notes especially) can
            // be non-empty even with zero .ctx images and zero groups
            // — an orphaned `.importing` marker needs neither — so
            // this must carry the already-collected notices, not a
            // fresh empty Vec that would silently drop them.
            print_json(&InspectReport {
                target: dir.display().to_string(),
                kind: "directory",
                contexts: Vec::new(),
                groups: Vec::new(),
                notices,
                total: None,
                corrupt: 0,
            });
        } else {
            println!("no .ctx images under {}", dir.display());
        }
        return 0;
    }

    if as_json {
        print_json(&InspectReport {
            target: dir.display().to_string(),
            kind: "directory",
            contexts: context_rows,
            groups: group_rows,
            notices,
            total: Some(Totals {
                contexts,
                groups: group_count,
                image_bytes: image_total,
                wal_bytes: wal_total,
                vector_bytes: vectors_total,
                index_bytes: index_total,
                passage_bytes: passages_total,
                footprint_bytes: footprint_total,
            }),
            corrupt: failures,
        });
    } else {
        println!(
            "total: {contexts} contexts · {group_count} groups · images {} · WAL {} · vectors {} · \
             index {} · passages {} · footprint if all resident {}",
            fmt_bytes(image_total),
            fmt_bytes(wal_total),
            fmt_bytes(vectors_total),
            fmt_bytes(index_total),
            fmt_bytes(passages_total),
            fmt_bytes(footprint_total),
        );
    }
    if failures > 0 {
        eprintln!("taguru: inspect: {failures} corrupt");
        return 1;
    }
    0
}

/// The `.group` half of a directory inspection, same read-only
/// discipline: every file (from the caller's one listing) goes
/// through the parse boot runs, and what boot would ALTER is reported
/// instead of healed — as a failure where the alteration loses
/// acknowledged data (bytes that do not parse reset the record; an
/// unreadable file refuses the boot itself), as a warning where it
/// drops what is already stale (dangling references, over-cap sets,
/// an ill-shaped nesting). Returns (groups parsed, failures, the
/// per-group JSON rows, and the standalone notices that don't belong
/// to any one group).
fn inspect_groups(
    entries: &[std::path::PathBuf],
    context_names: &BTreeSet<String>,
    as_json: bool,
) -> (usize, usize, Vec<GroupRow>, Vec<Notice>) {
    let mut failures = 0usize;
    let mut records: BTreeMap<String, GroupRecord> = BTreeMap::new();
    let mut group_rows: Vec<GroupRow> = Vec::new();
    let mut notices: Vec<Notice> = Vec::new();
    // Names already reported as CORRUPT above — inserted into `records`
    // as an empty placeholder so a sibling's dangling-reference check
    // sees the same shape boot will (comment below), but the JSON
    // preview loop must not ALSO emit an "ok" row for a name that's
    // already a `GroupRow::trouble` row; the human line below still
    // prints one, since a person reads the CORRUPT line right above it
    // and the empty "ok" line's own zero counts as a caveat, not new
    // information a JSON reader could act on.
    let mut reported_corrupt: BTreeSet<String> = BTreeSet::new();
    for path in entries {
        let file = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
        if file.ends_with(".group.corrupt") {
            if as_json {
                notices.push(Notice::note(
                    "set_aside_group_bytes",
                    file.to_string(),
                    "bytes an earlier boot set aside from a group that did not parse (evidence \
                     for hand recovery; every scan ignores it)",
                ));
            } else {
                println!(
                    "{file}: NOTE — bytes an earlier boot set aside from a group that did not \
                     parse (evidence for hand recovery; every scan ignores it)"
                );
            }
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("group") {
            continue;
        }
        let Some((stem, name)) = scanned_stem_and_name(path) else {
            if as_json {
                notices.push(Notice::warning(
                    "undecodable_stem",
                    file.to_string(),
                    "stem does not decode; the server will skip it",
                ));
            } else {
                println!("{file}: WARNING — stem does not decode; the server will skip it");
            }
            continue;
        };
        match load_group(path) {
            Ok(record) => {
                records.insert(name, record);
            }
            Err(GroupFileTrouble::Unreadable(error)) => {
                if as_json {
                    group_rows.push(GroupRow::trouble(
                        name.clone(),
                        "unreadable",
                        error.to_string(),
                    ));
                } else {
                    println!(
                        "{name}: UNREADABLE group — {error} (a boot refuses to start while \
                         this file cannot be read)"
                    );
                }
                failures += 1;
            }
            Err(GroupFileTrouble::Corrupt(error)) => {
                if as_json {
                    group_rows.push(GroupRow::trouble(
                        name.clone(),
                        "corrupt",
                        error.to_string(),
                    ));
                } else {
                    println!(
                        "{name}: CORRUPT group — {error} (a boot keeps the name, sets these \
                         bytes aside as {stem}.group.corrupt, and resets the record to empty)"
                    );
                }
                failures += 1;
                // scan_groups registers the name with an empty record
                // rather than dropping it — the reference checks below
                // must see the same shape boot will, or a parent naming
                // this group as a child gets a false "boot drops this
                // reference" warning (boot only drops references whose
                // name is genuinely absent), and the shape preview below
                // would prune the edge before repair_nesting ever runs,
                // skewing the cycle/depth check for unrelated edges too.
                reported_corrupt.insert(name.clone());
                records.insert(name, GroupRecord::default());
            }
        }
    }

    // The healing preview: what boot's reconciliation would drop.
    for (name, record) in &records {
        let dangling_contexts = record
            .contexts
            .iter()
            .filter(|context| !context_names.contains(*context))
            .count();
        let dangling_children = record
            .groups
            .iter()
            .filter(|child| !records.contains_key(*child))
            .count();
        // Built once regardless of --json: the human suffix below reads
        // the same `Notice::message`s this feeds into the JSON row, so
        // the two renderings cannot disagree about what was found.
        let mut group_notes: Vec<Notice> = Vec::new();
        if dangling_contexts > 0 {
            group_notes.push(Notice::warning(
                "dangling_context_reference",
                name.clone(),
                format!(
                    "{dangling_contexts} member context(s) have no context here (boot drops \
                     the references)"
                ),
            ));
        }
        if dangling_children > 0 {
            group_notes.push(Notice::warning(
                "dangling_child_group_reference",
                name.clone(),
                format!(
                    "{dangling_children} child group(s) have no group here (boot drops the \
                     references)"
                ),
            ));
        }
        for (field, len) in [
            ("member contexts", record.contexts.len()),
            ("child groups", record.groups.len()),
        ] {
            if len > MAX_GROUP_MEMBERS {
                let kind = if field == "member contexts" {
                    "over_cap_members"
                } else {
                    "over_cap_children"
                };
                group_notes.push(Notice::warning(
                    kind,
                    name.clone(),
                    format!(
                        "{len} {field} where a group holds at most {MAX_GROUP_MEMBERS} (boot \
                         keeps the first {MAX_GROUP_MEMBERS})"
                    ),
                ));
            }
        }
        if as_json {
            if !reported_corrupt.contains(name) {
                let mut row = GroupRow::ok(name.clone(), record);
                row.notes = group_notes;
                group_rows.push(row);
            }
        } else {
            println!(
                "{name}: ok  {} member context(s) · {} child group(s){}",
                record.contexts.len(),
                record.groups.len(),
                if group_notes.is_empty() {
                    String::new()
                } else {
                    format!(
                        " · WARNING: {}",
                        group_notes
                            .iter()
                            .map(|note| note.message.as_str())
                            .collect::<Vec<_>>()
                            .join("; ")
                    )
                }
            );
        }
    }

    // The shape preview runs the REAL repair on a scratch copy — dangling
    // references dropped first, then the membership cap, then nesting
    // shape, the exact order boot's `reconcile_groups` runs in (a dangling
    // edge is not a shape violation, so the repair would keep it; an
    // over-cap set can still change which edges the shape repair sees) —
    // and names every edge boot would drop, not just the first violation a
    // validator walk happens to hit.
    let mut swept = records.clone();
    for record in swept.values_mut() {
        record.groups.retain(|child| records.contains_key(child));
    }
    trim_membership(&mut swept, MAX_GROUP_MEMBERS);
    let mut repaired = swept.clone();
    repair_nesting(&mut repaired);
    for (name, record) in &swept {
        for child in record.groups.difference(&repaired[name].groups) {
            if as_json {
                notices.push(Notice::warning(
                    "dropped_nesting_edge",
                    "groups",
                    format!(
                        "boot drops the nesting edge '{name}' → '{child}' (it would close a \
                         cycle or stack more than {MAX_GROUP_DEPTH} groups)"
                    ),
                ));
            } else {
                println!(
                    "groups: WARNING — boot drops the nesting edge '{name}' → '{child}' \
                     (it would close a cycle or stack more than {MAX_GROUP_DEPTH} groups)"
                );
            }
        }
    }
    (records.len(), failures, group_rows, notices)
}

fn load_image(path: &Path) -> Result<(Context, u64, String), String> {
    let bytes = std::fs::read(path).map_err(|error| format!("unreadable: {error}"))?;
    let context = Context::from_bytes(&bytes).map_err(|error| error.to_string())?;
    // From the same bytes the successful load just proved: v6+ means
    // the checksum was verified in that load, older versions have no
    // checksum to verify — say which, because a backup check that
    // "says ok" must not overclaim on pre-checksum bytes.
    let generation = match Context::image_generation(&bytes) {
        Some((version, true)) => format!("v{version}, checksum verified"),
        Some((version, false)) => format!("v{version} — predates checksums, parsed unverified"),
        None => unreachable!("from_bytes checked the magic"),
    };
    Ok((context, bytes.len() as u64, generation))
}

fn stats_line(context: &Context, image_bytes: u64, generation: &str) -> String {
    let (unsourced_edges, unsourced_weight) = context.unsourced_summary();
    format!(
        "image {} ({generation}) · {} associations · {} concepts · {} labels · {} sources · \
         footprint {} · applied_seq {} · {} dead edge(s) ({:.1}% dead) · \
         {} unlinked attribution(s) · {} arena slack · \
         {unsourced_edges} unsourced edge(s) (weight {unsourced_weight:.1})",
        fmt_bytes(image_bytes),
        context.association_count(),
        context.concept_count(),
        context.label_count(),
        context.source_count(),
        fmt_bytes(context.footprint() as u64),
        context.applied_seq(),
        context.dead_edges(),
        context.dead_ratio() * 100.0,
        context.dead_attributions(),
        fmt_bytes(context.arena_slack() as u64),
    )
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
}
