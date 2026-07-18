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

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use taguru::context::Context;

use crate::cli::fmt_bytes;
use crate::groups::{
    GroupRecord, MAX_GROUP_DEPTH, MAX_GROUP_MEMBERS, repair_nesting, trim_membership,
};
use crate::registry::{
    IMPORT_MARKER_EXTENSION, ImportMarker, bm25_path, meta_path, name_from_stem, passages_path,
    passages_wal_path, pvectors_path, scanned_stem_and_name, sources_path, vectors_path, wal_path,
};
use crate::wal;

const USAGE: &str =
    "usage: taguru inspect PATH   (a data directory, one .ctx image, or one .group record)\n";

pub fn run(args: &[String]) -> i32 {
    let path = match args {
        [flag] if flag == "--help" || flag == "-h" => {
            print!("{USAGE}");
            return 0;
        }
        [path] => Path::new(path.as_str()),
        _ => {
            eprint!("{USAGE}");
            return 2;
        }
    };
    if path.is_dir() {
        inspect_directory(path)
    } else if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("group") {
            inspect_group_file(path)
        } else {
            inspect_file(path)
        }
    } else {
        eprintln!(
            "taguru: inspect: {} is neither a file nor a directory",
            path.display()
        );
        2
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
fn inspect_group_file(path: &Path) -> i32 {
    match load_group(path) {
        Ok(record) => {
            println!(
                "{}: ok  {} member context(s) · {} child group(s)",
                path.display(),
                record.contexts.len(),
                record.groups.len()
            );
            0
        }
        Err(GroupFileTrouble::Unreadable(error)) => {
            eprintln!("{}: UNREADABLE — {error}", path.display());
            1
        }
        Err(GroupFileTrouble::Corrupt(error)) => {
            eprintln!("{}: CORRUPT — {error}", path.display());
            1
        }
    }
}

/// One bare image, no sidecars: the "is this .ctx I restored intact"
/// question.
fn inspect_file(path: &Path) -> i32 {
    match load_image(path) {
        Ok((context, image_bytes, generation)) => {
            println!(
                "{}: ok  {}",
                path.display(),
                stats_line(&context, image_bytes, &generation)
            );
            0
        }
        Err(error) => {
            eprintln!("{}: CORRUPT — {error}", path.display());
            1
        }
    }
}

fn inspect_directory(dir: &Path) -> i32 {
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

    for stem in &stems {
        let name = match name_from_stem(stem) {
            Some(name) => name,
            None => {
                // Not a failure: the server skips it too — but a backup
                // holding files the server will never serve is worth a line.
                println!("{stem}.ctx: WARNING — stem does not decode; the server will skip it");
                continue;
            }
        };
        let image = dir.join(format!("{stem}.ctx"));
        let (context, image_bytes, generation) = match load_image(&image) {
            Ok(loaded) => loaded,
            Err(error) => {
                println!("{name}: CORRUPT image — {error}");
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
                    println!("{name}: CORRUPT WAL — {error}");
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
                    println!("{name}: CORRUPT passages — {error}");
                    failures += 1;
                    continue;
                }
            };

        // Meta is self-healing on the server side (defaults + warning),
        // so a broken one is reported without failing the inspection.
        let meta_note = match std::fs::read(meta_path(dir, stem)) {
            Ok(bytes) if serde_json::from_slice::<serde_json::Value>(&bytes).is_err() => {
                " · WARNING: meta.json unparseable (description/usage will reset)"
            }
            _ => "",
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
        let mut torn_parts = Vec::new();
        if let Some(torn) = wal_torn {
            torn_parts.push(torn_tail_note("WAL", torn));
        }
        if let Some(torn) = passages_torn {
            torn_parts.push(torn_tail_note("passages WAL", torn));
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
        println!(
            "{name}: ok  {} · WAL {} ({pending} pending) · vectors {} · index {} · passages {} \
             ({passage_count} sources){meta_note}{torn_note}{unverified_note}",
            stats_line(&context, image_bytes, &generation),
            fmt_bytes(wal_bytes),
            fmt_bytes(vector_bytes),
            fmt_bytes(index_bytes),
            fmt_bytes(passage_bytes),
        );

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
                println!(
                    "{}: WARNING — the import of source '{}' never completed; its truth \
                     may be half-applied — re-import its batch file or retract the source",
                    marker.context, marker.source
                );
            }
            Some(marker) => {
                println!(
                    "{file}: NOTE — import marker for context '{}', which no longer \
                     exists here (the server's next boot removes it)",
                    marker.context
                );
            }
            None => {
                println!(
                    "{file}: WARNING — unreadable import marker; an import batch may be \
                     half-applied, but which source is unrecoverable"
                );
            }
        }
    }

    let (group_count, group_failures) = inspect_groups(&entries, &context_names);
    failures += group_failures;

    if stems.is_empty() && group_count == 0 && group_failures == 0 {
        println!("no .ctx images under {}", dir.display());
        return 0;
    }

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
/// an ill-shaped nesting). Returns (groups parsed, failures).
fn inspect_groups(
    entries: &[std::path::PathBuf],
    context_names: &BTreeSet<String>,
) -> (usize, usize) {
    let mut failures = 0usize;
    let mut records: BTreeMap<String, GroupRecord> = BTreeMap::new();
    for path in entries {
        let file = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
        if file.ends_with(".group.corrupt") {
            println!(
                "{file}: NOTE — bytes an earlier boot set aside from a group that did not \
                 parse (evidence for hand recovery; every scan ignores it)"
            );
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("group") {
            continue;
        }
        let Some((stem, name)) = scanned_stem_and_name(path) else {
            println!("{file}: WARNING — stem does not decode; the server will skip it");
            continue;
        };
        match load_group(path) {
            Ok(record) => {
                records.insert(name, record);
            }
            Err(GroupFileTrouble::Unreadable(error)) => {
                println!(
                    "{name}: UNREADABLE group — {error} (a boot refuses to start while \
                     this file cannot be read)"
                );
                failures += 1;
            }
            Err(GroupFileTrouble::Corrupt(error)) => {
                println!(
                    "{name}: CORRUPT group — {error} (a boot keeps the name, sets these \
                     bytes aside as {stem}.group.corrupt, and resets the record to empty)"
                );
                failures += 1;
                // scan_groups registers the name with an empty record
                // rather than dropping it — the reference checks below
                // must see the same shape boot will, or a parent naming
                // this group as a child gets a false "boot drops this
                // reference" warning (boot only drops references whose
                // name is genuinely absent), and the shape preview below
                // would prune the edge before repair_nesting ever runs,
                // skewing the cycle/depth check for unrelated edges too.
                records.insert(name, GroupRecord::default());
            }
        }
    }

    // The healing preview: what boot's reconciliation would drop.
    for (name, record) in &records {
        let mut warnings: Vec<String> = Vec::new();
        let dangling_contexts = record
            .contexts
            .iter()
            .filter(|context| !context_names.contains(*context))
            .count();
        if dangling_contexts > 0 {
            warnings.push(format!(
                "{dangling_contexts} member context(s) have no context here (boot drops \
                 the references)"
            ));
        }
        let dangling_children = record
            .groups
            .iter()
            .filter(|child| !records.contains_key(*child))
            .count();
        if dangling_children > 0 {
            warnings.push(format!(
                "{dangling_children} child group(s) have no group here (boot drops the \
                 references)"
            ));
        }
        for (field, len) in [
            ("member contexts", record.contexts.len()),
            ("child groups", record.groups.len()),
        ] {
            if len > MAX_GROUP_MEMBERS {
                warnings.push(format!(
                    "{len} {field} where a group holds at most {MAX_GROUP_MEMBERS} (boot \
                     keeps the first {MAX_GROUP_MEMBERS})"
                ));
            }
        }
        println!(
            "{name}: ok  {} member context(s) · {} child group(s){}",
            record.contexts.len(),
            record.groups.len(),
            if warnings.is_empty() {
                String::new()
            } else {
                format!(" · WARNING: {}", warnings.join("; "))
            }
        );
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
            println!(
                "groups: WARNING — boot drops the nesting edge '{name}' → '{child}' \
                 (it would close a cycle or stack more than {MAX_GROUP_DEPTH} groups)"
            );
        }
    }
    (records.len(), failures)
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
