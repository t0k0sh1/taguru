//! `taguru inspect`: offline verification of a data directory or one
//! `.ctx` image — the backup check that needs no server. Every image
//! goes through the same fully validating parser the server boots
//! with, and every WAL through the same replay parser, so "inspect
//! says ok" and "the server will load it" are one statement. Exits 1
//! when anything holding acknowledged data is corrupt.

use std::path::Path;

use taguru::context::Context;

use crate::cli::fmt_bytes;
use crate::registry::{
    bm25_path, meta_path, name_from_stem, passages_path, passages_wal_path, pvectors_path,
    sources_path, vectors_path, wal_path,
};
use crate::wal;

const USAGE: &str = "usage: taguru inspect PATH   (a data directory, or one .ctx image)\n";

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
        inspect_file(path)
    } else {
        eprintln!(
            "taguru: inspect: {} is neither a file nor a directory",
            path.display()
        );
        2
    }
}

/// One bare image, no sidecars: the "is this .ctx I restored intact"
/// question.
fn inspect_file(path: &Path) -> i32 {
    match load_image(path) {
        Ok((context, image_bytes)) => {
            println!(
                "{}: ok  {}",
                path.display(),
                stats_line(&context, image_bytes)
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
    let mut stems: Vec<String> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("ctx"))
            .filter_map(|path| path.file_stem().and_then(|s| s.to_str()).map(String::from))
            .collect(),
        Err(error) => {
            eprintln!("taguru: inspect: cannot read {}: {error}", dir.display());
            return 2;
        }
    };
    stems.sort();
    if stems.is_empty() {
        println!("no .ctx images under {}", dir.display());
        return 0;
    }

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
        let (context, image_bytes) = match load_image(&image) {
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
        let (pending, wal_torn) =
            match wal::replay_readonly::<wal::WalOp>(&wal_path(dir, stem), context.applied_seq()) {
                Ok((ops, _, torn)) => (ops.len(), torn),
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
        let (passage_count, passages_torn) = match crate::passages::PassageStore::load(
            passages_path(dir, stem),
            &sources_path(dir, stem),
            passages_wal_path(dir, stem),
            0,
            false,
        ) {
            Ok((store, torn)) => (store.source_ids().len(), torn),
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

        // A torn tail is a crash mid-append, not corruption: the record
        // was never acknowledged, and the server heals it on its next
        // load. Report it (inspect left the file untouched) rather than
        // failing — but never let it pass silently.
        let mut torn_parts = Vec::new();
        if let Some(bytes) = wal_torn {
            torn_parts.push(format!("WAL torn tail ({bytes} bytes)"));
        }
        if let Some(bytes) = passages_torn {
            torn_parts.push(format!("passages WAL torn tail ({bytes} bytes)"));
        }
        let torn_note = if torn_parts.is_empty() {
            String::new()
        } else {
            format!(
                " · NOTE: {} — crash mid-append, healed on the server's next load \
                 (inspect left it untouched)",
                torn_parts.join(" and ")
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
             ({passage_count} sources){meta_note}{torn_note}",
            stats_line(&context, image_bytes),
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

    println!(
        "total: {contexts} contexts · images {} · WAL {} · vectors {} · index {} · passages {} · \
         footprint if all resident {}",
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

fn load_image(path: &Path) -> Result<(Context, u64), String> {
    let bytes = std::fs::read(path).map_err(|error| format!("unreadable: {error}"))?;
    let context = Context::from_bytes(&bytes).map_err(|error| error.to_string())?;
    Ok((context, bytes.len() as u64))
}

fn stats_line(context: &Context, image_bytes: u64) -> String {
    format!(
        "image {} · {} associations · {} concepts · {} labels · {} sources · \
         footprint {} · applied_seq {}",
        fmt_bytes(image_bytes),
        context.association_count(),
        context.concept_count(),
        context.label_count(),
        context.source_count(),
        fmt_bytes(context.footprint() as u64),
        context.applied_seq(),
    )
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
}
