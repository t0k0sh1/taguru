//! Document discovery and chunking: expanding CLI paths into files,
//! reading a document's text, and splitting it into `ChunkDescriptor`
//! pieces at paragraph boundaries.

use super::*;

/// One chunk, with the paragraph-index provenance a diagnostics
/// consumer needs to point back at the original document — issue
/// #262. `text` is exactly what the model sees; `sha256` is the same
/// value a checkpoint stores its unit under
/// ([`CheckpointStore::lookup`]), so a caller that already has a
/// [`ChunkDescriptor`] never re-hashes to consult the checkpoint.
pub(crate) struct ChunkDescriptor {
    pub(crate) text: String,
    pub(crate) sha256: String,
    /// Inclusive, matching [`crate::paragraph::split`]'s own
    /// `ParagraphSpan.index` — the coordinate system the batch's
    /// `paragraph` locator, passage store, BM25 lane, and vector lane
    /// all already share (ADR 0003 §7). Never a byte offset: `chunk()`
    /// works on [`labeled_document`]'s derived, relabeled rendering, so
    /// any byte offset it could report would describe that rendering,
    /// shifted by every `[N] ` prefix, never the original file.
    pub(crate) paragraph_first: u32,
    pub(crate) paragraph_last: u32,
}

/// Plans `text`'s chunks exactly as [`Run::extract_document`] will
/// split and send them, annotated with each chunk's paragraph range
/// and content hash (issue #262). This is a read of [`chunk`]'s and
/// [`labeled_document`]'s own output, not a second implementation of
/// their packing rule — the one function ADR 0003 §7 says must never
/// be duplicated — so the chunks this returns are byte-for-byte
/// [`Run::extract_document`]'s own.
///
/// Also the seam #256's benchmark harness calls in-process (same
/// binary as `extract`) to build its manifest's document/chunk
/// dictionary, without going through a subprocess.
pub(crate) fn chunk_plan(text: &str) -> Vec<ChunkDescriptor> {
    chunk_plan_with_cap(text, CHUNK_BYTES)
}

pub(super) fn chunk_plan_with_cap(text: &str, cap: usize) -> Vec<ChunkDescriptor> {
    chunk(&labeled_document(text, cap), cap)
        .into_iter()
        .map(|piece| {
            // Every block in a chunk is `[N] `-labeled and no block is
            // ever re-split by chunk() (labeled_document reserves the
            // label's own room, so a labeled block never exceeds cap) —
            // splitting on "\n\n" therefore always yields whole,
            // labeled blocks, never a bare continuation.
            let first = piece
                .split("\n\n")
                .next()
                .expect("split(\"\\n\\n\") yields at least one piece");
            let last = piece
                .rsplit("\n\n")
                .next()
                .expect("rsplit(\"\\n\\n\") yields at least one piece");
            ChunkDescriptor {
                sha256: sha256_hex(piece.as_bytes()),
                paragraph_first: leading_paragraph_number(first),
                paragraph_last: leading_paragraph_number(last),
                text: piece,
            }
        })
        .collect()
}

/// Parses the `[N] ` label [`labeled_document`] prefixes onto every
/// block. A block without one would mean `chunk()`/`labeled_document`
/// stopped upholding the invariant [`chunk_plan_with_cap`] relies on —
/// a taguru bug, not a condition a caller can act on, so this panics
/// rather than fabricating provenance.
pub(super) fn leading_paragraph_number(block: &str) -> u32 {
    block
        .strip_prefix('[')
        .and_then(|rest| rest.split_once("] "))
        .and_then(|(digits, _)| digits.parse().ok())
        .expect("labeled_document prefixes every block with its [N] label")
}

/// The document's text, refused early when it could never ride as a
/// batch passage: unreadable, over the 8 MiB passage cap, or not UTF-8.
/// Size is checked from metadata BEFORE the read for the common case —
/// an oversized document (a mispointed path, a multi-GB log file) is
/// refused without ever buffering its bytes. That check alone would
/// still race a file that grows past the cap between the stat and the
/// read (TOCTOU) — or, for something like a FIFO, one whose metadata
/// length never reflected its content at all — so the read itself is
/// also bounded: at most one byte over the cap is ever buffered, just
/// enough to detect an overflow the stat missed without letting an
/// unbounded stream through.
///
/// `pub(crate)` so `benchmark`'s preflight hashes a document's text the
/// exact same way `extract`'s own manifest does (BOM-stripped, size-
/// capped, UTF-8-validated) — the document dictionary in
/// `manifest.json` (ADR 0003 §9.1) must agree with what a cell's own
/// `.extract-manifest.json` records, or a resumed matrix could not tell
/// "unchanged" from "drifted."
pub(crate) fn read_document(path: &Path) -> Result<String, String> {
    let size = fs::metadata(path).map_err(|error| error.to_string())?.len();
    if size > MAX_PASSAGE_BYTES as u64 {
        return Err(format!(
            "{size} bytes exceeds the {MAX_PASSAGE_BYTES}-byte \
             document cap — split the document"
        ));
    }
    let file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    file.take(MAX_PASSAGE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > MAX_PASSAGE_BYTES as u64 {
        return Err(format!(
            "exceeds the {MAX_PASSAGE_BYTES}-byte document cap — split the document"
        ));
    }
    let text = String::from_utf8(bytes).map_err(|_| "not UTF-8".to_string())?;
    // A leading BOM is invisible in an editor but would otherwise become
    // the first character of paragraph 0 — silently breaking any exact
    // match against the document's true opening text.
    Ok(match text.strip_prefix('\u{FEFF}') {
        Some(rest) => rest.to_string(),
        None => text,
    })
}

/// Explicit files are taken as given; a directory contributes its
/// `.md` and `.txt` files in name order — the same shape as import's
/// expansion, and an empty directory is likewise a mistake.
///
/// `pub(crate)` so `benchmark` enumerates a corpus directory the exact
/// same way the `extract` child process it spawns will (ADR 0003 §6:
/// "the corpus must reach every cell in identical order"), rather than
/// re-deriving the same sort rule a second time.
pub(crate) fn expand_documents(paths: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for raw in paths {
        let path = Path::new(raw);
        if path.is_file() {
            files.push(path.to_path_buf());
        } else if path.is_dir() {
            let mut found: Vec<PathBuf> = fs::read_dir(path)
                .map_err(|error| format!("cannot read {raw}: {error}"))?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|p| {
                    p.is_file()
                        && matches!(
                            p.extension().and_then(|e| e.to_str()),
                            Some("md") | Some("txt")
                        )
                })
                .collect();
            if found.is_empty() {
                return Err(format!("no .md or .txt files under {raw}"));
            }
            found.sort();
            files.append(&mut found);
        } else {
            return Err(format!("{raw} is neither a file nor a directory"));
        }
    }
    Ok(files)
}
