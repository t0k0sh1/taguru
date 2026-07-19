//! Per-context passage store: the original text behind each source id,
//! resident in memory and durable through its own append-log.
//!
//! The predecessor kept passages in `{stem}.sources.json` and rewrote
//! the WHOLE file on every store — importing N documents wrote O(N²)
//! bytes, and every search re-read the file from disk. This store is
//! the same shape the graph itself uses: a compacted snapshot
//! (`{stem}.passages.bin`, watermark in the header) plus an append-log
//! (`{stem}.passages.wal.jsonl`, fsynced per batch), replayed above the
//! watermark on load. Unlike the graph's log, this one is not optional
//! hardening — it IS the write path; `TAGURU_WAL` does not apply here.
//!
//! Compaction is self-triggered from inside the write path (imports
//! have no flush ticker), and the trigger is a RATIO of the last
//! snapshot's size, not a fixed byte threshold: with a fixed threshold
//! K, a growing context compacts every K bytes and rewrites its whole
//! (growing) snapshot each time — Σ i·K is the same O(N²) this store
//! exists to remove, just with a friendlier constant. A ratio trigger
//! amortizes exactly like `Vec` doubling: each compaction writes at
//! most (1 + RATIO)× the previous snapshot, so total rewrite cost stays
//! linear in total stored bytes.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use serde::{Deserialize, Serialize};

use crate::paragraph::{self, ParagraphSpan};
use crate::wal;

/// One source's submitted passage: the text, and optionally the
/// doc2query questions and section markers riding with it —
/// (paragraph index, question) and (paragraph index, label) pairs a
/// producer (`taguru extract --questions`, the ingest batch format,
/// or any client) attached so question-shaped queries can land on
/// answer-shaped paragraphs, and paragraphs can carry a section
/// label, from the index side too.
#[derive(Debug, Clone)]
pub(crate) struct PassageSubmission {
    pub(crate) text: String,
    pub(crate) questions: Vec<(u32, String)>,
    pub(crate) sections: Vec<(u32, String)>,
}

impl PassageSubmission {
    /// A bare passage — the shape most tests submit.
    #[cfg(test)]
    pub(crate) fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            questions: Vec::new(),
            sections: Vec::new(),
        }
    }
}

/// What one store call accomplished, question and section bookkeeping
/// included: out-of-range questions and sections are dropped one by
/// one (the paragraph they name does not exist in the text they rode
/// in with), never failing the passages they accompanied.
#[derive(Debug, Default)]
pub(crate) struct StoreOutcome {
    pub(crate) stored: usize,
    pub(crate) questions_stored: usize,
    pub(crate) questions_dropped: usize,
    pub(crate) sections_stored: usize,
    pub(crate) sections_dropped: usize,
}

/// How many of `questions`/`sections` name a paragraph outside
/// `paragraph_count` — the same range check [`PassageRecord::new`]
/// applies when it filters, exposed standalone so a caller can get the
/// count without building the record: `preview_batch`'s read-only
/// twin of `apply_batch` calls this to report what a real store would
/// drop. Sections additionally count paragraphs claimed more than
/// once — see [`filter_sections`].
pub(crate) fn preview_drops(
    paragraph_count: usize,
    questions: &[(u32, String)],
    sections: &[(u32, String)],
) -> (usize, usize) {
    let out_of_range = |&&(paragraph, _): &&(u32, String)| paragraph as usize >= paragraph_count;
    (
        questions.iter().filter(out_of_range).count(),
        filter_sections(sections, paragraph_count).1,
    )
}

/// Keeps only sections naming a paragraph within `paragraph_count`,
/// then collapses same-paragraph entries to the last one submitted —
/// a section is a start marker, and [`PassageRecord::section_for`]'s
/// binary search can only ever surface one label per paragraph, so an
/// earlier marker for a paragraph that has a later one can never be
/// reached again once stored. Dropping it here instead (rather than
/// letting it ride into storage unreachable forever) matches the rest
/// of the store's last-write-wins convention and keeps the drop count
/// honest. Returns the survivors, sorted by paragraph as
/// `PassageRecord` stores them, and how many entries — out of range or
/// shadowed — were dropped. The one place this is decided, so
/// `PassageRecord::new` and `preview_drops` can never disagree about
/// which sections a submission keeps.
fn filter_sections(
    sections: &[(u32, String)],
    paragraph_count: usize,
) -> (Vec<(u32, String)>, usize) {
    let mut seen = std::collections::HashSet::new();
    let mut kept: Vec<(u32, String)> = sections
        .iter()
        .rev()
        .filter(|&&(paragraph, _)| (paragraph as usize) < paragraph_count && seen.insert(paragraph))
        .cloned()
        .collect();
    kept.sort_by_key(|&(paragraph, _)| paragraph);
    let dropped = sections.len() - kept.len();
    (kept, dropped)
}

/// One source's resident passage: the byte-exact text, its paragraph
/// spans (computed once per residency, never persisted — the split
/// function is deterministic, so the spans are as durable as the text
/// they index), its stored questions, and its stored section markers
/// — both validated against those spans and sorted by paragraph.
/// Handed out as `Arc`, so search and the index/vector lanes walk
/// paragraphs without holding the store lock and without copying
/// text.
#[derive(Debug)]
pub(crate) struct PassageRecord {
    pub(crate) text: Arc<str>,
    pub(crate) paragraphs: Vec<ParagraphSpan>,
    pub(crate) questions: Vec<(u32, String)>,
    /// (paragraph index of section start, label) pairs, sorted by
    /// paragraph, at most one per paragraph — a section implicitly
    /// extends until the next marker or the end of the passage.
    /// Populated by the ingest batch format's `section` line (see
    /// `ingest.rs`); resolved through `section_for`. See
    /// [`filter_sections`] for how a paragraph claimed more than once
    /// is resolved before it ever reaches here.
    pub(crate) sections: Vec<(u32, String)>,
}

impl PassageRecord {
    /// Builds the record, keeping only questions and section markers
    /// whose paragraph exists in THIS text's split — the one
    /// validation point both the live path and replay go through, so
    /// they can never disagree. Sections are additionally collapsed to
    /// one per paragraph (see [`filter_sections`]). Returns how many
    /// questions and how many sections were dropped.
    pub(crate) fn new(
        text: Arc<str>,
        questions: Vec<(u32, String)>,
        sections: Vec<(u32, String)>,
    ) -> (Arc<Self>, usize, usize) {
        let paragraphs = paragraph::split(&text);
        let questions_offered = questions.len();
        let mut questions: Vec<(u32, String)> = questions
            .into_iter()
            .filter(|&(paragraph, _)| (paragraph as usize) < paragraphs.len())
            .collect();
        questions.sort_by_key(|&(paragraph, _)| paragraph);
        let questions_dropped = questions_offered - questions.len();
        let (sections, sections_dropped) = filter_sections(&sections, paragraphs.len());
        (
            Arc::new(Self {
                text,
                paragraphs,
                questions,
                sections,
            }),
            questions_dropped,
            sections_dropped,
        )
    }

    /// Constructor for sibling modules' unit tests.
    #[cfg(test)]
    pub(crate) fn for_tests(text: &str) -> Arc<Self> {
        Self::new(Arc::from(text), Vec::new(), Vec::new()).0
    }

    /// [`Self::for_tests`] with doc2query questions attached.
    #[cfg(test)]
    pub(crate) fn for_tests_with_questions(text: &str, questions: Vec<(u32, String)>) -> Arc<Self> {
        Self::new(Arc::from(text), questions, Vec::new()).0
    }

    /// The paragraph texts behind the spans, in order.
    pub(crate) fn paragraph_texts(&self) -> impl Iterator<Item = (&ParagraphSpan, &str)> {
        self.paragraphs
            .iter()
            .map(|span| (span, &self.text[span.start as usize..span.end as usize]))
    }

    /// One paragraph's span and text by its index — the single-lookup
    /// counterpart of `paragraph_texts`, and the slice both
    /// `search_passages` and the citation endpoint share so their
    /// excerpts can never drift apart.
    pub(crate) fn paragraph(&self, index: usize) -> Option<(&ParagraphSpan, &str)> {
        self.paragraphs
            .get(index)
            .map(|span| (span, &self.text[span.start as usize..span.end as usize]))
    }

    /// The section governing a paragraph — the nearest start marker at
    /// or before `index`, extending until the next marker or the end
    /// of the passage. `None` before the first marker, past the end,
    /// or when the record carries no sections at all. Used by
    /// `AppState::resolve_sections` to label attributions on
    /// association reads (recall, query, explore, activate,
    /// unreachable_from), and by `AppState::citation` to label its
    /// single excerpt.
    pub(crate) fn section_for(&self, index: usize) -> Option<&str> {
        if index >= self.paragraphs.len() {
            return None;
        }
        let pos = self
            .sections
            .partition_point(|&(start, _)| (start as usize) <= index);
        pos.checked_sub(1).map(|i| self.sections[i].1.as_str())
    }
}

/// One passage mutation — this log's whole vocabulary. Same
/// internally-tagged shape as the graph's `WalOp`, but its own enum in
/// its own file: each log speaks one vocabulary (see `wal::WalRecord`).
/// Both ops are last-write-wins, so a replay overlapping the snapshot
/// re-applies harmlessly — the watermark still gates it, but a bug
/// there degrades to wasted work, not corruption (unlike `associate`,
/// which accumulates). `questions` and `sections` are both additive
/// (older logs simply have none) and carry the submission AS OFFERED —
/// validation happens in `PassageRecord::new`, identically on the live
/// path and on replay.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum PassageOp {
    Store {
        source: String,
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        questions: Vec<(u32, String)>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        sections: Vec<(u32, String)>,
    },
    Retract {
        source: String,
    },
}

/// Per-source-id rough constant for the resident estimate, matching
/// the spirit of `VectorStore::footprint`'s entry overhead.
const SOURCE_OVERHEAD: usize = 64;

/// Compaction floor: below this much pending log, never compact — a
/// young context should not rewrite its snapshot over every trickle.
const COMPACT_FLOOR_BYTES: u64 = 4 * 1024 * 1024;
/// Compact when the pending log outgrows RATIO × the last snapshot
/// (see the module doc for why a ratio, never a fixed threshold).
const COMPACT_RATIO: u64 = 1;

const SNAPSHOT_MAGIC: &[u8; 8] = b"TAGURUS4";
/// The pre-checksum snapshot format: S4's layout minus the CRC-32C
/// footer, so its bytes can only be parsed, never verified. Read
/// forever, written never again.
const S3_SNAPSHOT_MAGIC: &[u8; 8] = b"TAGURUS3";
/// The pre-sections snapshot format: same layout minus the per-source
/// section blocks. Read forever, written never again.
const S2_SNAPSHOT_MAGIC: &[u8; 8] = b"TAGURUS2";
/// The pre-doc2query snapshot format: same layout minus both the
/// question and section blocks. Read forever, written never again.
const LEGACY_SNAPSHOT_MAGIC: &[u8; 8] = b"TAGURUS1";

#[derive(Debug)]
struct PassageStoreInner {
    /// source id → resident record (byte-exact text + paragraph
    /// spans). `Arc` so a bulk reader clones handles, not text, and
    /// walks them without holding this lock.
    sources: BTreeMap<String, Arc<PassageRecord>>,
    /// The next log sequence this store hands out. Lives INSIDE this
    /// lock, beside the map it numbers: compaction must read "the map
    /// and how far the log got" as one consistent snapshot, or it could
    /// truncate records the map state it wrote never contained.
    next_seq: u64,
    /// Bytes appended to the log since the last successful truncation.
    /// Only a truncation that actually ran resets it — a compaction
    /// that skipped truncating (a write landed mid-serialize) must not,
    /// or the counter drifts below the file's real size.
    log_bytes: u64,
    /// Size of the last snapshot written — the ratio trigger's base.
    snapshot_bytes: u64,
    /// Running Σ(key + text + overhead), so `footprint` is O(1).
    resident_bytes: usize,
    /// Lifetime observability: how many compactions ran, and how many
    /// snapshot bytes they wrote in total. The linearity regression
    /// test pins the latter against total stored bytes.
    compactions: u64,
    snapshot_bytes_written: u64,
}

/// One context's passages. `writer` serializes mutators so log order
/// equals seq order and a compaction never truncates under a
/// concurrent append's feet; readers only ever take `inner`, so a
/// search is never blocked by an fsync.
#[derive(Debug)]
pub(crate) struct PassageStore {
    writer: Mutex<()>,
    inner: RwLock<PassageStoreInner>,
    snapshot_path: PathBuf,
    log_path: PathBuf,
    /// The pre-migration `.sources.json`, retired by the first
    /// successful compaction — its contents are in the snapshot from
    /// that moment, and only then may the old file go.
    legacy_path: PathBuf,
    /// Backstop ceiling on the pending log (0 = unlimited). A healthy
    /// store compacts at RATIO × snapshot, so the log legitimately
    /// reaches that size; the ceiling exists for a compaction that is
    /// failing outright and engages only past BOTH this value and 2×
    /// the last snapshot — a big context near its natural trigger is
    /// never refused.
    max_log_bytes: u64,
}

/// Tally from one `append_and_apply_locked` call. A named struct
/// rather than a same-typed `usize` tuple — four positional counts
/// this size apart are easy to transpose without the compiler
/// noticing.
#[derive(Debug, Default)]
struct AppliedCounts {
    questions_stored: usize,
    questions_dropped: usize,
    sections_stored: usize,
    sections_dropped: usize,
}

impl PassageStore {
    /// Loads a context's passages, `ensure_hot`-shaped: pick the base —
    /// the snapshot if one exists, else the legacy `.sources.json`
    /// (watermark 0), else empty (watermark 0) — then UNCONDITIONALLY
    /// replay the log above the base's watermark. Legacy data thereby
    /// needs no special migration step: writes logged after a legacy
    /// load are ordinary above-watermark records, correct across any
    /// crash before the first compaction.
    ///
    /// The snapshot and the log hold acknowledged writes that exist
    /// nowhere else, so corruption in either is an error, not a shrug.
    /// The legacy file keeps its historical contract (unreadable means
    /// empty) — refusing to boot over a file the old code tolerated
    /// would turn an upgrade into an outage.
    /// `heal` heals a torn WAL tail in place (the server's boot) or
    /// leaves the file untouched and returns the tail's `TornTail` shape
    /// so a read-only caller — `taguru inspect` — can report it (`Ok`'s
    /// second element; `None` when the log ends clean or was healed).
    /// `Ok`'s third element counts log records that carried no checksum
    /// (pre-checksum writers) — meaningful only on the heal=false path,
    /// where inspect reports how much of the log was actually verified;
    /// the healing path reports 0 without counting.
    pub(crate) fn load(
        snapshot_path: PathBuf,
        legacy_path: &Path,
        log_path: PathBuf,
        max_log_bytes: usize,
        heal: bool,
    ) -> io::Result<(Self, Option<wal::TornTail>, usize)> {
        let (sources, watermark, snapshot_bytes) = match fs::read(&snapshot_path) {
            Ok(bytes) => {
                let size = bytes.len() as u64;
                let (sources, watermark) = snapshot_from_bytes(&bytes).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "corrupt passage snapshot at {} — it holds acknowledged \
                             passages, restore it from backup",
                            snapshot_path.display()
                        ),
                    )
                })?;
                (sources, watermark, size)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let mut sources = BTreeMap::new();
                for (source, text) in read_legacy(legacy_path) {
                    let text = bounded_text(&source, text, "the legacy sources file")?;
                    sources.insert(
                        source,
                        PassageRecord::new(Arc::from(text), Vec::new(), Vec::new()).0,
                    );
                }
                (sources, 0, 0)
            }
            Err(error) => return Err(error),
        };

        let mut sources: BTreeMap<String, Arc<PassageRecord>> = sources;
        // The server heals a torn tail in place (and `replay` logs it);
        // inspect passes heal=false to leave the file as it found it and
        // learn the torn size instead. Both decode the same ops, so an
        // inspection that says ok means the server's load will succeed too.
        let (ops, top, torn, unchecked) = if heal {
            let (ops, top) = wal::replay::<PassageOp>(&log_path, watermark)?;
            (ops, top, None, 0)
        } else {
            wal::replay_readonly::<PassageOp>(&log_path, watermark)?
        };
        for op in ops {
            match op {
                PassageOp::Store {
                    source,
                    text,
                    questions,
                    sections,
                } => {
                    let text = bounded_text(&source, text, "the passages WAL")?;
                    sources.insert(
                        source,
                        PassageRecord::new(Arc::from(text), questions, sections).0,
                    );
                }
                PassageOp::Retract { source } => {
                    sources.remove(&source);
                }
            }
        }
        let log_bytes = match fs::metadata(&log_path) {
            Ok(meta) => meta.len(),
            Err(_) => 0,
        };
        let resident_bytes = sources
            .iter()
            .map(|(source, record)| record_bytes(source, record))
            .sum();

        Ok((
            Self {
                writer: Mutex::new(()),
                inner: RwLock::new(PassageStoreInner {
                    sources,
                    next_seq: top + 1,
                    log_bytes,
                    snapshot_bytes,
                    resident_bytes,
                    compactions: 0,
                    snapshot_bytes_written: 0,
                }),
                snapshot_path,
                log_path,
                legacy_path: legacy_path.to_path_buf(),
                max_log_bytes: max_log_bytes as u64,
            },
            torn,
            unchecked,
        ))
    }

    /// Merge-upserts a batch, log-first: fsync the ops, then apply —
    /// on `Err` nothing was applied and nothing must be assumed
    /// durable. `stored` keeps the historical contract (how many the
    /// request carried, not how many keys were new); the question
    /// tallies say how many rode in and how many named a paragraph
    /// that does not exist in their own text.
    pub(crate) fn store(
        &self,
        passages: BTreeMap<String, PassageSubmission>,
    ) -> io::Result<StoreOutcome> {
        // Paragraph spans use u32 offsets. Every ingress today sits far
        // below this (the 8 MiB import cap, the HTTP body cap), but the
        // body cap is operator-tunable — refuse cleanly here rather
        // than let a 4 GiB text reach the splitter's assert.
        if let Some((source, submission)) = passages
            .iter()
            .find(|(_, submission)| submission.text.len() > u32::MAX as usize)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "passage '{source}' is {} bytes; passages are capped at 4 GiB",
                    submission.text.len()
                ),
            ));
        }
        let ops: Vec<PassageOp> = passages
            .into_iter()
            .map(|(source, submission)| PassageOp::Store {
                source,
                text: submission.text,
                questions: submission.questions,
                sections: submission.sections,
            })
            .collect();
        let count = ops.len();
        if count == 0 {
            return Ok(StoreOutcome::default());
        }
        let writer = self.writer.lock().unwrap();
        // The backstop, checked under `writer` so it is exact: refuse
        // growth only when the log is past the operator ceiling AND
        // past 2× the last snapshot — the second condition keeps a big
        // context approaching its NATURAL ratio trigger from being
        // refused; both together mean compaction has demonstrably been
        // failing (each failure already warned). Retractions stay
        // allowed: they are how an operator shrinks the store.
        {
            let inner = self.inner.read().unwrap();
            if self.max_log_bytes > 0
                && inner.log_bytes >= self.max_log_bytes
                && inner.log_bytes >= 2 * inner.snapshot_bytes
            {
                return Err(io::Error::other(format!(
                    "the passage log is at {} bytes (cap {}) with compaction failing — \
                     check disk space and the server log",
                    inner.log_bytes, self.max_log_bytes
                )));
            }
        }
        let counts = self.append_and_apply_locked(&writer, &ops)?;
        Ok(StoreOutcome {
            stored: count,
            questions_stored: counts.questions_stored,
            questions_dropped: counts.questions_dropped,
            sections_stored: counts.sections_stored,
            sections_dropped: counts.sections_dropped,
        })
    }

    /// Withdraws one source's passage. Returns whether one existed —
    /// an absent source appends nothing (the check is stable: `writer`
    /// is held across check and append).
    pub(crate) fn retract(&self, source: &str) -> io::Result<bool> {
        let writer = self.writer.lock().unwrap();
        if !self.inner.read().unwrap().sources.contains_key(source) {
            return Ok(false);
        }
        let ops = [PassageOp::Retract {
            source: source.to_string(),
        }];
        self.append_and_apply_locked(&writer, &ops)?;
        Ok(true)
    }

    fn append_and_apply_locked(
        &self,
        writer: &std::sync::MutexGuard<'_, ()>,
        ops: &[PassageOp],
    ) -> io::Result<AppliedCounts> {
        let first_seq = self.inner.read().unwrap().next_seq;
        let appended = wal::append_batch(&self.log_path, first_seq, ops)?;
        let mut counts = AppliedCounts::default();
        let over_ratio = {
            let mut inner = self.inner.write().unwrap();
            for op in ops {
                match op {
                    PassageOp::Store {
                        source,
                        text,
                        questions,
                        sections,
                    } => {
                        let (record, questions_dropped, sections_dropped) = PassageRecord::new(
                            Arc::from(text.as_str()),
                            questions.clone(),
                            sections.clone(),
                        );
                        counts.questions_stored += record.questions.len();
                        counts.questions_dropped += questions_dropped;
                        counts.sections_stored += record.sections.len();
                        counts.sections_dropped += sections_dropped;
                        inner.resident_bytes += record_bytes(source, &record);
                        if let Some(previous) = inner.sources.insert(source.clone(), record) {
                            inner.resident_bytes -= record_bytes(source, &previous);
                        }
                    }
                    PassageOp::Retract { source } => {
                        if let Some(previous) = inner.sources.remove(source) {
                            inner.resident_bytes -= record_bytes(source, &previous);
                        }
                    }
                }
            }
            inner.next_seq = first_seq + ops.len() as u64;
            inner.log_bytes += appended;
            inner.log_bytes > COMPACT_FLOOR_BYTES.max(COMPACT_RATIO * inner.snapshot_bytes)
        };
        if over_ratio {
            // The write itself is already durable; compaction is
            // housekeeping, and a failure must not turn a persisted
            // write into a client-facing error. A stuck compaction
            // shows up as unbounded log growth instead.
            if let Err(error) = self.compact_locked(writer) {
                tracing::warn!(
                    "passage log at {} not compacted (will retry on a later write): {error}",
                    self.log_path.display()
                );
            }
        }
        Ok(counts)
    }

    /// Rewrites the snapshot from the current map and truncates the
    /// log if — and only if — no write landed while the bytes were in
    /// flight (the same watermark re-check `flush_entry` runs).
    ///
    /// Takes `writer` itself rather than assuming a caller already
    /// holds it: eviction calls this directly, with no
    /// `append_and_apply_locked` in between. Without the lock here,
    /// that direct call and a ratio-triggered compaction from a
    /// concurrent `store`/`retract` can both read a consistent
    /// `(sources, seen_seq)` snapshot and both reach `write_atomic` —
    /// but whichever lands SECOND wins on disk regardless of which
    /// one actually saw the newer state. If the first (newer) one to
    /// finish already truncated the log on its own, then-correct
    /// watermark check, the second (staler) one overwriting the
    /// snapshot afterward leaves the write it missed nowhere: not in
    /// the snapshot, not in the (already-emptied) log. An
    /// acknowledged, fsynced write vanishes silently — and eviction
    /// clearing `entry.passages` right after this call means the very
    /// next access reloads from disk and confirms the loss.
    pub(crate) fn compact(&self) -> io::Result<()> {
        let writer = self.writer.lock().unwrap();
        self.compact_locked(&writer)
    }

    /// [`compact`]'s body, for a caller that already holds `writer` —
    /// `append_and_apply_locked` calls this instead of `compact()` so
    /// its own ratio-triggered compaction does not try to re-lock a
    /// `std::sync::Mutex`, which is not reentrant.
    ///
    /// The serialize-and-write happens with no OTHER lock held: the
    /// map is `Arc<str>` handles, so the one consistent read is a
    /// cheap clone, and megabytes of text land on disk while readers
    /// proceed (writers wait behind `writer`, same as any mutator).
    fn compact_locked(&self, _writer: &std::sync::MutexGuard<'_, ()>) -> io::Result<()> {
        let (sources, seen_seq) = {
            let inner = self.inner.read().unwrap();
            (inner.sources.clone(), inner.next_seq)
        };
        #[cfg(test)]
        COMPACT_PAUSE_AFTER_READ.with(|cell| {
            if let Some(hook) = cell.borrow_mut().take() {
                hook();
            }
        });
        let bytes = snapshot_to_bytes(&sources, seen_seq - 1);
        crate::storage::write_atomic(&self.snapshot_path, &bytes)?;

        // The snapshot's rename is durable, so the legacy file's
        // contents live there now (any base it fed is baked in) and the
        // loader will never consult it again — retire it. Ordering is
        // the whole point: removing it BEFORE a durable snapshot exists
        // would strand the passages of a crash in between. A failed
        // unlink lingers harmlessly and retries next compaction.
        if let Err(error) = crate::storage::remove_persisted_file(&self.legacy_path)
            && error.kind() != io::ErrorKind::NotFound
        {
            tracing::warn!(
                "legacy passages at {} not retired (will retry): {error}",
                self.legacy_path.display()
            );
        }

        let mut inner = self.inner.write().unwrap();
        inner.snapshot_bytes = bytes.len() as u64;
        inner.compactions += 1;
        inner.snapshot_bytes_written += bytes.len() as u64;
        if inner.next_seq == seen_seq {
            match wal::reset(&self.log_path) {
                Ok(()) => inner.log_bytes = 0,
                Err(error) => {
                    // Harmless: the fresh watermark makes every logged
                    // record replay-inert; the log is just longer than
                    // it needs to be until the next truncation lands.
                    tracing::warn!(
                        "passage log at {} not truncated (harmless): {error}",
                        self.log_path.display()
                    );
                }
            }
        }
        Ok(())
    }

    /// Bytes of log the last compaction has not covered — the signal
    /// eviction uses to decide whether a best-effort compaction would
    /// save the next load a replay.
    pub(crate) fn pending_log_bytes(&self) -> u64 {
        self.inner.read().unwrap().log_bytes
    }

    /// The last log sequence this store handed out (0 = nothing ever
    /// logged) — the passage half of the context revision. Durable by
    /// construction: appends are log-first and compaction bakes it
    /// into the snapshot, so a reload always resumes the numbering.
    pub(crate) fn watermark(&self) -> u64 {
        self.inner.read().unwrap().next_seq - 1
    }

    pub(crate) fn get(&self, source: &str) -> Option<Arc<PassageRecord>> {
        self.inner.read().unwrap().sources.get(source).cloned()
    }

    /// The registered source ids, ASCENDING — `sources` is a
    /// `BTreeMap`, so its keys come out sorted; callers paging by id
    /// (`list_sources`) rely on this rather than re-sorting per page.
    pub(crate) fn source_ids(&self) -> Vec<String> {
        self.inner.read().unwrap().sources.keys().cloned().collect()
    }

    /// Every passage as (source, record-handle) — the bulk reader's
    /// entrance. The lock is held for the clone only; text and spans
    /// ride out as `Arc` handles, never copies.
    pub(crate) fn snapshot(&self) -> Vec<(String, Arc<PassageRecord>)> {
        self.inner
            .read()
            .unwrap()
            .sources
            .iter()
            .map(|(source, record)| (source.clone(), Arc::clone(record)))
            .collect()
    }

    /// Rough resident bytes, for the cache budget and the gauges.
    pub(crate) fn footprint(&self) -> usize {
        self.inner.read().unwrap().resident_bytes
    }

    /// (compactions run, snapshot bytes they wrote) over this store's
    /// residency — the observability behind the linearity guarantee.
    /// Test-only until a metrics counter picks it up.
    #[cfg(test)]
    pub(crate) fn compaction_totals(&self) -> (u64, u64) {
        let inner = self.inner.read().unwrap();
        (inner.compactions, inner.snapshot_bytes_written)
    }
}

// Test-only interleaving control for `PassageStore::compact_locked`:
// armed on a thread via `pause_next_compact_after_read`, it fires once
// — right after that thread's next `compact` captures its `(sources,
// seen_seq)` snapshot — then clears itself. Lets a test force a second,
// concurrent `compact`/`store` to run to completion in that gap,
// deterministically, rather than hoping a race window lands.
#[cfg(test)]
thread_local! {
    static COMPACT_PAUSE_AFTER_READ: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn pause_next_compact_after_read(hook: impl FnOnce() + 'static) {
    COMPACT_PAUSE_AFTER_READ.with(|cell| *cell.borrow_mut() = Some(Box::new(hook)));
}

/// The u32-offset bound `store()` enforces on the live path, re-checked
/// on what disk hands back. The WAL's records and the legacy sources
/// file are JSON with no inherent length bound; no live write can
/// produce an oversized text (`store()` refuses first), but a tampered
/// or corrupted file could, and it would otherwise sail into the
/// splitter's assert and panic the whole load. Corruption is this
/// module's `InvalidData` error, never a panic.
fn bounded_text(source: &str, text: String, origin: &str) -> io::Result<String> {
    if text.len() > u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "passage '{source}' from {origin} is {} bytes; passages are capped at 4 GiB",
                text.len()
            ),
        ));
    }
    Ok(text)
}

/// The legacy whole-file JSON map. Unreadable or absent means empty —
/// that file's historical contract, kept for the one load that
/// migrates it.
fn read_legacy(path: &Path) -> BTreeMap<String, String> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|_| {
            tracing::warn!("ignoring unreadable legacy passages at {}", path.display());
            BTreeMap::new()
        }),
        Err(_) => BTreeMap::new(),
    }
}

/// What one resident record costs, for the running footprint: key,
/// text, spans, questions, sections, and the per-entry overhead
/// constant.
pub(crate) fn record_bytes(source: &str, record: &PassageRecord) -> usize {
    source.len()
        + record.text.len()
        + record.paragraphs.len() * std::mem::size_of::<ParagraphSpan>()
        + record
            .questions
            .iter()
            .map(|(_, question)| question.len() + 8)
            .sum::<usize>()
        + record
            .sections
            .iter()
            .map(|(_, label)| label.len() + 8)
            .sum::<usize>()
        + SOURCE_OVERHEAD
}

fn snapshot_to_bytes(sources: &BTreeMap<String, Arc<PassageRecord>>, watermark: u64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(SNAPSHOT_MAGIC);
    out.extend_from_slice(&watermark.to_le_bytes());
    out.extend_from_slice(&(sources.len() as u32).to_le_bytes());
    // BTreeMap iterates sorted (and a record's questions/sections are
    // sorted by paragraph at construction): identical content is
    // byte-identical. Text, questions, and sections land on disk —
    // spans are recomputed on load.
    for (source, record) in sources {
        write_chunk(&mut out, source.as_bytes());
        write_chunk(&mut out, record.text.as_bytes());
        out.extend_from_slice(&(record.questions.len() as u32).to_le_bytes());
        for (paragraph, question) in &record.questions {
            out.extend_from_slice(&paragraph.to_le_bytes());
            write_chunk(&mut out, question.as_bytes());
        }
        out.extend_from_slice(&(record.sections.len() as u32).to_le_bytes());
        for (paragraph, label) in &record.sections {
            out.extend_from_slice(&paragraph.to_le_bytes());
            write_chunk(&mut out, label.as_bytes());
        }
    }
    // The CRC-32C footer, same discipline as the context image: the
    // snapshot holds acknowledged passages with no other copy, and
    // structural parsing alone cannot tell bit-rot from truth.
    let crc = crate::crc32c::crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

fn snapshot_from_bytes(bytes: &[u8]) -> Option<(BTreeMap<String, Arc<PassageRecord>>, u64)> {
    let mut pos = 0usize;
    let magic = bytes.get(..8)?;
    // TAGURUS1 predates questions and sections; TAGURUS2 predates only
    // sections; TAGURUS3 predates the checksum footer. Each older
    // layout is the newest one minus its trailing pieces, so one
    // parser reads all four.
    let (questions_on_disk, sections_on_disk, checksummed) = if magic == SNAPSHOT_MAGIC {
        (true, true, true)
    } else if magic == S3_SNAPSHOT_MAGIC {
        (true, true, false)
    } else if magic == S2_SNAPSHOT_MAGIC {
        (true, false, false)
    } else if magic == LEGACY_SNAPSHOT_MAGIC {
        (false, false, false)
    } else {
        return None;
    };
    // Verify before trusting anything past the magic: pre-checksum
    // generations can only be parsed, the current one is proven the
    // bytes that were written.
    let bytes = if checksummed {
        let body_len = bytes.len().checked_sub(4)?;
        let (body, footer) = bytes.split_at(body_len);
        let stored = u32::from_le_bytes(footer.try_into().ok()?);
        (crate::crc32c::crc32c(body) == stored).then_some(body)?
    } else {
        bytes
    };
    pos += 8;
    let watermark = u64::from_le_bytes(bytes.get(pos..pos + 8)?.try_into().ok()?);
    pos += 8;
    let count = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
    pos += 4;
    let mut sources = BTreeMap::new();
    for _ in 0..count {
        let source = String::from_utf8(read_chunk(bytes, &mut pos)?.to_vec()).ok()?;
        let text = std::str::from_utf8(read_chunk(bytes, &mut pos)?)
            .ok()?
            .to_string();
        let mut questions = Vec::new();
        if questions_on_disk {
            let question_count =
                u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
            pos += 4;
            for _ in 0..question_count {
                let paragraph = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?);
                pos += 4;
                let question = String::from_utf8(read_chunk(bytes, &mut pos)?.to_vec()).ok()?;
                questions.push((paragraph, question));
            }
        }
        let mut sections = Vec::new();
        if sections_on_disk {
            let section_count =
                u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
            pos += 4;
            for _ in 0..section_count {
                let paragraph = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?);
                pos += 4;
                let label = String::from_utf8(read_chunk(bytes, &mut pos)?.to_vec()).ok()?;
                sections.push((paragraph, label));
            }
        }
        sources.insert(
            source,
            PassageRecord::new(Arc::from(text), questions, sections).0,
        );
    }
    (pos == bytes.len()).then_some((sources, watermark))
}

fn write_chunk(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn read_chunk<'b>(bytes: &'b [u8], pos: &mut usize) -> Option<&'b [u8]> {
    let len = u32::from_le_bytes(bytes.get(*pos..*pos + 4)?.try_into().ok()?) as usize;
    *pos += 4;
    let chunk = bytes.get(*pos..*pos + len)?;
    *pos += len;
    Some(chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("taguru-passages-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn open(dir: &Path) -> PassageStore {
        open_capped(dir, 0)
    }

    fn open_capped(dir: &Path, max_log_bytes: usize) -> PassageStore {
        PassageStore::load(
            dir.join("t.passages.bin"),
            &dir.join("t.sources.json"),
            dir.join("t.passages.wal.jsonl"),
            max_log_bytes,
            true,
        )
        .unwrap()
        .0
    }

    fn batch(entries: &[(&str, &str)]) -> BTreeMap<String, PassageSubmission> {
        entries
            .iter()
            .map(|&(source, text)| (source.to_string(), PassageSubmission::plain(text)))
            .collect()
    }

    /// The stored text behind a source, for assertions.
    fn text(store: &PassageStore, source: &str) -> Option<String> {
        store.get(source).map(|record| record.text.to_string())
    }

    #[test]
    fn passage_store_upsert_is_idempotent_last_write_wins() {
        let dir = scratch_dir("upsert");
        let store = open(&dir);
        assert_eq!(
            store.store(batch(&[("第1段落", "旧版")])).unwrap().stored,
            1
        );
        assert_eq!(
            store
                .store(batch(&[("第1段落", "新版"), ("第2段落", "追加")]))
                .unwrap()
                .stored,
            2,
            "the return value counts the batch, not just new keys"
        );
        assert_eq!(text(&store, "第1段落").as_deref(), Some("新版"));
        assert_eq!(text(&store, "第2段落").as_deref(), Some("追加"));
        assert_eq!(store.source_ids(), vec!["第1段落", "第2段落"]);
    }

    #[test]
    fn passage_store_retract_removes_and_is_a_noop_if_the_source_is_absent() {
        let dir = scratch_dir("retract");
        let store = open(&dir);
        store.store(batch(&[("a", "text")])).unwrap();
        assert!(store.retract("a").unwrap());
        assert!(store.get("a").is_none());
        assert!(
            !store.retract("a").unwrap(),
            "absent means false, not an error"
        );
        assert!(!store.retract("never-stored").unwrap());
    }

    #[test]
    fn a_crash_between_a_wal_append_and_compaction_replays_correctly_on_restart() {
        let dir = scratch_dir("replay");
        {
            let store = open(&dir);
            store
                .store(batch(&[("a", "первый"), ("b", "第二")]))
                .unwrap();
            store.retract("a").unwrap();
            // Dropped without any compaction — everything lives in the
            // log alone, exactly the state a crash leaves behind.
            assert!(!dir.join("t.passages.bin").exists());
        }
        let reborn = open(&dir);
        assert!(reborn.get("a").is_none());
        assert_eq!(text(&reborn, "b").as_deref(), Some("第二"));
    }

    #[test]
    fn compaction_truncates_the_log_and_a_reload_reads_the_snapshot() {
        let dir = scratch_dir("compact");
        {
            let store = open(&dir);
            store
                .store(batch(&[("a", "本文A"), ("b", "本文B")]))
                .unwrap();
            store.compact().unwrap();
            assert_eq!(
                fs::metadata(dir.join("t.passages.wal.jsonl"))
                    .unwrap()
                    .len(),
                0,
                "the covered log truncates"
            );
            // Writes after the compaction land in the log again.
            store.store(batch(&[("c", "本文C")])).unwrap();
        }
        let reborn = open(&dir);
        assert_eq!(text(&reborn, "a").as_deref(), Some("本文A"));
        assert_eq!(text(&reborn, "b").as_deref(), Some("本文B"));
        assert_eq!(text(&reborn, "c").as_deref(), Some("本文C"));
    }

    /// A regression for a stale-snapshot race in `compact`. Before the
    /// fix, `compact` took no lock of its own — it trusted every caller
    /// to already hold `writer`, true for the ratio-triggered call
    /// inside `append_and_apply_locked` but false for eviction's direct
    /// `store.compact()` (`AppState::evict_entry`, taken straight off
    /// `entry.passages` with no `writer` lock in sight). Two uncoordinated
    /// `compact` calls could then interleave: one (the "fresh" one below)
    /// reads a `(sources, seen_seq)` snapshot that already includes a new
    /// write, finishes first, and correctly truncates the log now that
    /// the write is safely captured in its snapshot; the other (the
    /// "stale" one) had read an OLDER snapshot, missing that write, and
    /// finishes second — overwriting the fresh, correct snapshot with its
    /// stale one. The write "stale" never saw is now nowhere: not in the
    /// snapshot it just wrote, not in the log ("fresh" already truncated
    /// it, correctly from "fresh"'s own vantage).
    ///
    /// Rather than hope two racing threads happen to land in that
    /// interleaving, [`pause_next_compact_after_read`] pins "stale" right
    /// after it captures its snapshot — before the write below exists —
    /// and holds it there until "fresh" (which does that write, then
    /// compacts) has had every chance to run. Fixed, "fresh" cannot even
    /// start its own `store` until "stale" — which now takes `writer`
    /// before pausing — releases it, so the sleep just elapses with
    /// "fresh" blocked; unfixed, "stale" holds nothing, so "fresh" races
    /// ahead and finishes well within it.
    #[test]
    fn a_paused_compact_never_overwrites_a_write_a_racing_compact_already_persisted() {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let dir = scratch_dir("compact-race");
        let store = Arc::new(open(&dir));
        store.store(batch(&[("seed", "seed text")])).unwrap();

        let (paused_tx, paused_rx) = mpsc::channel::<()>();
        let (resume_tx, resume_rx) = mpsc::channel::<()>();

        // Mirrors eviction's direct, once-unlocked `compact()` call:
        // pauses right after reading `(sources, seen_seq)` — before
        // "late" below exists — and does not write that now-stale
        // snapshot until told to.
        let stale = {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                pause_next_compact_after_read(move || {
                    paused_tx.send(()).unwrap();
                    resume_rx.recv().unwrap();
                });
                store.compact()
            })
        };
        paused_rx.recv().unwrap();

        // Mirrors a ratio-triggered compact from a concurrent `store`/
        // `retract`: stores "late", then compacts, seeing it and — no
        // further write having landed since — truncating the log.
        let fresh = {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                store
                    .store(batch(&[(
                        "late",
                        "landed and was compacted while the stale compact was paused",
                    )]))
                    .unwrap();
                store.compact()
            })
        };
        // Ample time for `fresh` to fully finish on the unfixed code,
        // where it races `stale` freely; on the fixed code it simply
        // blocks on `writer` for the same span, since `stale` holds it
        // throughout the pause.
        thread::sleep(Duration::from_millis(300));

        resume_tx.send(()).unwrap();
        stale.join().unwrap().unwrap();
        fresh.join().unwrap().unwrap();

        drop(store);
        let reloaded = open(&dir);
        assert_eq!(text(&reloaded, "seed").as_deref(), Some("seed text"));
        assert_eq!(
            text(&reloaded, "late").as_deref(),
            Some("landed and was compacted while the stale compact was paused"),
            "a write a concurrent compact already read, persisted, and \
             truncated the log for must survive a stale compact's later overwrite"
        );
    }

    #[test]
    fn storing_thousands_of_small_sources_touches_the_snapshot_file_a_bounded_number_of_times_not_once_per_call()
     {
        let dir = scratch_dir("small");
        let store = open(&dir);
        for i in 0..2000 {
            store
                .store(batch(&[(
                    format!("doc-{i}").as_str(),
                    "ひと言だけの短い本文。",
                )]))
                .unwrap();
        }
        let (compactions, _) = store.compaction_totals();
        assert_eq!(
            compactions, 0,
            "2000 small stores stay under the floor — the old design \
             rewrote the whole file 2000 times"
        );
        assert!(!dir.join("t.passages.bin").exists());
        assert_eq!(store.source_ids().len(), 2000);
    }

    #[test]
    fn compacting_many_large_passages_keeps_total_rewritten_bytes_linear_in_total_stored_bytes() {
        let dir = scratch_dir("linear");
        let store = open(&dir);
        // Enough volume to force several compactions: 32 sources of
        // 1 MiB against a 4 MiB floor with RATIO=1.
        let text = "あ".repeat(1024 * 1024 / 3);
        let mut total_stored = 0u64;
        for i in 0..32 {
            store
                .store(batch(&[(format!("doc-{i}").as_str(), text.as_str())]))
                .unwrap();
            total_stored += text.len() as u64;
        }
        let (compactions, rewritten) = store.compaction_totals();
        assert!(
            compactions >= 2,
            "the test must provoke repeated compactions to prove anything \
             (got {compactions})"
        );
        // Geometric amortization: with RATIO=1 the rewrite series is
        // bounded by ~2× total; 4× leaves headroom for framing bytes.
        // A fixed-threshold trigger fails this with Θ(N²/K) rewrites.
        assert!(
            rewritten <= 4 * total_stored,
            "compaction rewrote {rewritten} bytes for {total_stored} stored — quadratic?"
        );
    }

    #[test]
    fn legacy_sources_json_is_the_base_and_the_log_replays_on_top_of_it() {
        let dir = scratch_dir("legacy");
        let legacy: BTreeMap<&str, &str> = [("old", "旧ファイルの本文"), ("both", "旧版")]
            .into_iter()
            .collect();
        fs::write(
            dir.join("t.sources.json"),
            serde_json::to_vec_pretty(&legacy).unwrap(),
        )
        .unwrap();
        {
            let store = open(&dir);
            assert_eq!(text(&store, "old").as_deref(), Some("旧ファイルの本文"));
            store
                .store(batch(&[("both", "新版"), ("new", "追記")]))
                .unwrap();
        }
        // A crash here leaves the legacy base plus a pending log; the
        // reload replays the log over the legacy base, watermark 0.
        let reborn = open(&dir);
        assert_eq!(text(&reborn, "old").as_deref(), Some("旧ファイルの本文"));
        assert_eq!(text(&reborn, "both").as_deref(), Some("新版"));
        assert_eq!(text(&reborn, "new").as_deref(), Some("追記"));
    }

    #[test]
    fn legacy_sources_json_is_retired_only_after_the_first_compaction_succeeds() {
        let dir = scratch_dir("legacy-retire");
        let legacy_path = dir.join("t.sources.json");
        let legacy: BTreeMap<&str, &str> = [("old", "旧本文")].into_iter().collect();
        fs::write(&legacy_path, serde_json::to_vec_pretty(&legacy).unwrap()).unwrap();

        let store = open(&dir);
        store.store(batch(&[("new", "新本文")])).unwrap();
        assert!(
            legacy_path.exists(),
            "before any compaction the legacy file IS the base — removing \
             it early would strand its passages across a crash"
        );

        store.compact().unwrap();
        assert!(
            !legacy_path.exists(),
            "a durable snapshot carries the legacy contents; the old file retires"
        );
        drop(store);
        let reborn = open(&dir);
        assert_eq!(text(&reborn, "old").as_deref(), Some("旧本文"));
        assert_eq!(text(&reborn, "new").as_deref(), Some("新本文"));
    }

    #[test]
    fn passage_record_keeps_questions_validated_and_sorted_by_paragraph() {
        let dir = scratch_dir("questions");
        {
            let store = open(&dir);
            let mut passages = BTreeMap::new();
            passages.insert(
                "doc".to_string(),
                PassageSubmission {
                    text: "一つ目。\n\n二つ目。".to_string(),
                    questions: vec![
                        (1, "二番目は何?".to_string()),
                        (0, "最初は何?".to_string()),
                        (9, "存在しない段落への質問?".to_string()),
                    ],
                    sections: Vec::new(),
                },
            );
            let outcome = store.store(passages).unwrap();
            assert_eq!(
                (
                    outcome.stored,
                    outcome.questions_stored,
                    outcome.questions_dropped
                ),
                (1, 2, 1),
                "the out-of-range question drops without failing its passage"
            );
            let record = store.get("doc").unwrap();
            assert_eq!(
                record.questions,
                vec![(0, "最初は何?".to_string()), (1, "二番目は何?".to_string())]
            );
        }
        // Questions are acknowledged writes: the log replays them.
        let reborn = open(&dir);
        assert_eq!(reborn.get("doc").unwrap().questions.len(), 2);
    }

    #[test]
    fn passage_record_keeps_sections_validated_and_sorted_by_start() {
        let (record, _questions_dropped, sections_dropped) = PassageRecord::new(
            Arc::from("一つ目。\n\n二つ目。"),
            Vec::new(),
            vec![
                (1, "後編".to_string()),
                (0, "前編".to_string()),
                (9, "存在しない段落への見出し".to_string()),
            ],
        );
        assert_eq!(
            record.sections,
            vec![(0, "前編".to_string()), (1, "後編".to_string())],
            "the out-of-range section drops and the rest sort by paragraph"
        );
        assert_eq!(sections_dropped, 1);
    }

    #[test]
    fn passage_record_collapses_a_paragraph_claimed_by_more_than_one_section() {
        let sections = vec![
            (1, "前編".to_string()),
            (1, "後編".to_string()),
            (9, "存在しない段落への見出し".to_string()),
        ];
        let (record, _questions_dropped, sections_dropped) = PassageRecord::new(
            Arc::from("一つ目。\n\n二つ目。"),
            Vec::new(),
            sections.clone(),
        );
        assert_eq!(
            record.sections,
            vec![(1, "後編".to_string())],
            "the earlier marker for a paragraph that has a later one is dropped, \
             not kept alongside it — only one label can ever govern a paragraph"
        );
        assert_eq!(
            sections_dropped, 2,
            "one out-of-range plus one shadowed by a later marker for the same paragraph"
        );
        assert_eq!(
            record.section_for(1),
            Some("後編"),
            "the surviving marker is the one section_for already resolved to"
        );
        assert_eq!(
            preview_drops(2, &[], &sections).1,
            sections_dropped,
            "preview must count the same drops apply does"
        );
    }

    #[test]
    fn a_passage_store_accepts_sections_and_reports_the_bookkeeping() {
        let dir = scratch_dir("sections-store");
        {
            let store = open(&dir);
            let mut passages = BTreeMap::new();
            passages.insert(
                "doc".to_string(),
                PassageSubmission {
                    text: "導入。\n\n本編。".to_string(),
                    questions: Vec::new(),
                    sections: vec![
                        (1, "本編".to_string()),
                        (9, "存在しない段落への見出し".to_string()),
                    ],
                },
            );
            let outcome = store.store(passages).unwrap();
            assert_eq!(
                (
                    outcome.stored,
                    outcome.sections_stored,
                    outcome.sections_dropped
                ),
                (1, 1, 1),
                "the out-of-range section drops without failing its passage"
            );
            let record = store.get("doc").unwrap();
            assert_eq!(record.sections, vec![(1, "本編".to_string())]);
        }
        // Sections are acknowledged writes: the log replays them.
        let reborn = open(&dir);
        assert_eq!(
            reborn.get("doc").unwrap().sections,
            vec![(1, "本編".to_string())]
        );
    }

    #[test]
    fn a_stored_section_survives_a_restart_via_wal_replay() {
        // Constructs the WAL op directly rather than through
        // `store()` — the same shape `registry.rs`'s generation tests
        // use to drive `PassageOp` without a live store, exercising
        // `load()`'s replay path on its own.
        let dir = scratch_dir("sections-wal");
        let log_path = dir.join("t.passages.wal.jsonl");
        wal::append_batch(
            &log_path,
            1,
            &[PassageOp::Store {
                source: "doc".to_string(),
                text: "導入。\n\n本編。".to_string(),
                questions: Vec::new(),
                sections: vec![(1, "本編".to_string())],
            }],
        )
        .unwrap();
        let store = PassageStore::load(
            dir.join("t.passages.bin"),
            &dir.join("t.sources.json"),
            log_path,
            0,
            true,
        )
        .unwrap()
        .0;
        assert_eq!(
            store.get("doc").unwrap().sections,
            vec![(1, "本編".to_string())]
        );
    }

    #[test]
    fn a_legacy_s1_snapshot_loads_with_empty_questions_and_upgrades_on_compaction() {
        let dir = scratch_dir("s1-upgrade");
        // S1 byte-for-byte: magic, watermark, count, [key, text] chunks
        // — the S2 layout minus the question blocks.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"TAGURUS1");
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        for chunk in ["old-doc", "旧本文。"] {
            bytes.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            bytes.extend_from_slice(chunk.as_bytes());
        }
        fs::write(dir.join("t.passages.bin"), &bytes).unwrap();

        let store = open(&dir);
        let record = store.get("old-doc").unwrap();
        assert_eq!(record.text.as_ref(), "旧本文。");
        assert!(record.questions.is_empty());
        store.compact().unwrap();
        drop(store);

        let head = fs::read(dir.join("t.passages.bin")).unwrap();
        assert_eq!(&head[..8], b"TAGURUS4", "the rewrite upgrades the format");
        let reborn = open(&dir);
        assert_eq!(text(&reborn, "old-doc").as_deref(), Some("旧本文。"));
    }

    #[test]
    fn a_legacy_s2_snapshot_loads_with_empty_sections_and_upgrades_on_compaction() {
        let dir = scratch_dir("s2-upgrade");
        // S2 byte-for-byte: magic, watermark, count, [key, text,
        // question block] — the S3 layout minus the section blocks.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"TAGURUS2");
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        for chunk in ["old-doc", "旧本文。"] {
            bytes.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            bytes.extend_from_slice(chunk.as_bytes());
        }
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one question
        bytes.extend_from_slice(&0u32.to_le_bytes()); // at paragraph 0
        let question = "旧本文とは?".as_bytes();
        bytes.extend_from_slice(&(question.len() as u32).to_le_bytes());
        bytes.extend_from_slice(question);
        fs::write(dir.join("t.passages.bin"), &bytes).unwrap();

        let store = open(&dir);
        let record = store.get("old-doc").unwrap();
        assert_eq!(record.text.as_ref(), "旧本文。");
        assert_eq!(record.questions, vec![(0, "旧本文とは?".to_string())]);
        assert!(record.sections.is_empty());
        store.compact().unwrap();
        drop(store);

        let head = fs::read(dir.join("t.passages.bin")).unwrap();
        assert_eq!(&head[..8], b"TAGURUS4", "the rewrite upgrades the format");
        let reborn = open(&dir);
        assert_eq!(text(&reborn, "old-doc").as_deref(), Some("旧本文。"));
    }

    #[test]
    fn passage_snapshot_rejects_truncated_or_garbage_bytes() {
        let sources: BTreeMap<String, Arc<PassageRecord>> = [(
            "a".to_string(),
            PassageRecord::new(
                Arc::from("text"),
                vec![(0, "何のtext?".to_string())],
                Vec::new(),
            )
            .0,
        )]
        .into_iter()
        .collect();
        let bytes = snapshot_to_bytes(&sources, 7);
        let (parsed, watermark) = snapshot_from_bytes(&bytes).unwrap();
        assert_eq!(watermark, 7);
        assert_eq!(parsed["a"].text.as_ref(), "text");
        assert_eq!(parsed["a"].questions, vec![(0, "何のtext?".to_string())]);

        assert!(snapshot_from_bytes(b"garbage").is_none());
        assert!(snapshot_from_bytes(&bytes[..bytes.len() - 1]).is_none());
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(
            snapshot_from_bytes(&padded).is_none(),
            "trailing bytes are corruption, not slack"
        );
    }

    #[test]
    fn snapshot_checksum_rejects_a_flipped_byte_and_s3_loads_unchecked() {
        let sources: BTreeMap<String, Arc<PassageRecord>> = [(
            "doc".to_string(),
            PassageRecord::new(Arc::from("第1段落。"), Vec::new(), Vec::new()).0,
        )]
        .into_iter()
        .collect();
        let bytes = snapshot_to_bytes(&sources, 7);
        assert!(bytes.starts_with(SNAPSHOT_MAGIC));
        assert!(snapshot_from_bytes(&bytes).is_some());

        // Flip one watermark byte: structurally the snapshot still
        // parses perfectly — it just carries a different watermark.
        // Only the checksum can refuse it.
        let mut flipped = bytes.clone();
        flipped[9] ^= 0x01;
        assert!(
            snapshot_from_bytes(&flipped).is_none(),
            "a parseable bitflip must fail the checksum"
        );

        // The same bytes as an S3 snapshot — body without the footer,
        // pre-checksum magic — load exactly as they did before the
        // footer existed: parsed, unverifiable.
        let mut s3 = bytes[..bytes.len() - 4].to_vec();
        s3[..8].copy_from_slice(S3_SNAPSHOT_MAGIC);
        let (parsed, watermark) = snapshot_from_bytes(&s3).expect("S3 must keep loading");
        assert_eq!(watermark, 7);
        assert_eq!(parsed["doc"].text.as_ref(), "第1段落。");
    }

    #[test]
    fn passage_snapshot_round_trips_populated_sections() {
        let sources: BTreeMap<String, Arc<PassageRecord>> = [(
            "a".to_string(),
            PassageRecord::new(
                Arc::from("導入。\n\n本編。"),
                Vec::new(),
                vec![(1, "本編".to_string())],
            )
            .0,
        )]
        .into_iter()
        .collect();
        let bytes = snapshot_to_bytes(&sources, 3);
        let (parsed, watermark) = snapshot_from_bytes(&bytes).unwrap();
        assert_eq!(watermark, 3);
        assert_eq!(parsed["a"].sections, vec![(1, "本編".to_string())]);
    }

    #[test]
    fn a_corrupt_snapshot_is_a_load_error_not_an_empty_store() {
        let dir = scratch_dir("strict");
        fs::write(dir.join("t.passages.bin"), b"not a snapshot").unwrap();
        let error = PassageStore::load(
            dir.join("t.passages.bin"),
            &dir.join("t.sources.json"),
            dir.join("t.passages.wal.jsonl"),
            0,
            true,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn the_log_backstop_refuses_stores_but_not_retracts_and_recovers_after_compaction() {
        let dir = scratch_dir("backstop");
        let store = open_capped(&dir, 1);
        store
            .store(batch(&[("a", "本文A"), ("b", "本文B")]))
            .unwrap();

        // Log is past the (absurdly low) cap with no snapshot at all —
        // the "compaction stuck" shape. Growth is refused; shrinking
        // and reading are not.
        let error = store.store(batch(&[("c", "本文C")])).unwrap_err();
        assert!(error.to_string().contains("cap 1"), "{error}");
        assert!(store.retract("b").unwrap(), "retracts must stay allowed");
        assert_eq!(text(&store, "a").as_deref(), Some("本文A"));

        // A compaction that finally lands truncates the log; stores flow
        // again (the snapshot also lifts the 2× condition).
        store.compact().unwrap();
        store.store(batch(&[("c", "本文C")])).unwrap();
        assert_eq!(text(&store, "c").as_deref(), Some("本文C"));
    }

    #[test]
    fn paragraph_returns_the_span_and_text_by_index_and_none_past_the_end() {
        let record = PassageRecord::for_tests("最初の段落。\n\n次の段落。");
        let (span, text) = record.paragraph(0).unwrap();
        assert_eq!(span.index, 0);
        assert_eq!(text, "最初の段落。");
        let (span, text) = record.paragraph(1).unwrap();
        assert_eq!(span.index, 1);
        assert_eq!(text, "次の段落。");
        assert!(
            record.paragraph(2).is_none(),
            "past the end is None, not a panic"
        );
    }

    #[test]
    fn section_for_resolves_the_governing_start_marker_or_none() {
        let record = PassageRecord::new(
            Arc::from("序。\n\n本編一。\n\n本編二。\n\n結び。"),
            Vec::new(),
            vec![(1, "本編".to_string()), (3, "結び".to_string())],
        )
        .0;
        assert_eq!(record.section_for(0), None, "before the first marker");
        assert_eq!(record.section_for(1), Some("本編"), "right on a marker");
        assert_eq!(
            record.section_for(2),
            Some("本編"),
            "extends until the next marker"
        );
        assert_eq!(record.section_for(3), Some("結び"));
        assert_eq!(
            record.section_for(4),
            None,
            "past the end is None, not a panic"
        );
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn proptest_config() -> ProptestConfig {
            let cases = std::env::var("PROPTEST_CASES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32);
            ProptestConfig {
                cases,
                ..ProptestConfig::default()
            }
        }

        fn word_strategy() -> impl Strategy<Value = &'static str> {
            prop_oneof![
                Just("蔵元"),
                Just("山田錦"),
                Just("精米歩合"),
                Just("杜氏"),
                Just("南部"),
                Just("state"),
                Just("impl"),
                Just("boot"),
                Just("query"),
                Just("index"),
            ]
        }

        fn text_strategy() -> impl Strategy<Value = String> {
            prop::collection::vec(
                prop::collection::vec(word_strategy(), 1..8).prop_map(|words| words.join("")),
                1..5,
            )
            .prop_map(|paragraphs| paragraphs.join("\n\n"))
        }

        fn source_name_strategy() -> impl Strategy<Value = String> {
            "[a-z]{1,8}\\.md"
        }

        /// (paragraph, text) markers for `questions`/`sections`. The
        /// paragraph half is deliberately wider than any generated
        /// text's actual paragraph count, so `PassageRecord::new`
        /// exercises both the keep and the silent-drop path for
        /// out-of-range markers — the same filtering the live store
        /// and WAL replay both depend on.
        fn marker_strategy() -> impl Strategy<Value = (u32, String)> {
            (0u32..10, word_strategy().prop_map(str::to_string))
        }

        fn record_strategy() -> impl Strategy<Value = Arc<PassageRecord>> {
            (
                text_strategy(),
                prop::collection::vec(marker_strategy(), 0..4),
                prop::collection::vec(marker_strategy(), 0..4),
            )
                .prop_map(|(text, questions, sections)| {
                    PassageRecord::new(Arc::from(text), questions, sections).0
                })
        }

        fn corpus_strategy() -> impl Strategy<Value = BTreeMap<String, Arc<PassageRecord>>> {
            prop::collection::vec((source_name_strategy(), record_strategy()), 0..6)
                .prop_map(|entries| entries.into_iter().collect())
        }

        /// Recomputes the CRC-32C footer after a mutation tampers with
        /// an S4 snapshot's body — same discipline as `Context`'s
        /// `resealed` test helper (context.rs), and for the same
        /// reason: without resealing, every mutation stops at the
        /// checksum and the structural parser underneath goes
        /// unexercised.
        fn resealed(mut bytes: Vec<u8>) -> Vec<u8> {
            let body = bytes.len() - 4;
            let crc = crate::crc32c::crc32c(&bytes[..body]);
            bytes[body..].copy_from_slice(&crc.to_le_bytes());
            bytes
        }

        proptest! {
            #![proptest_config(proptest_config())]

            #[test]
            fn snapshot_round_trips_through_bytes(
                sources in corpus_strategy(),
                watermark in any::<u64>(),
            ) {
                let bytes = snapshot_to_bytes(&sources, watermark);
                let (restored, restored_watermark) = snapshot_from_bytes(&bytes)
                    .expect("a freshly serialized snapshot must always decode");
                prop_assert_eq!(restored_watermark, watermark);
                prop_assert_eq!(restored.len(), sources.len());
                for (source, original) in &sources {
                    let record = restored.get(source).expect("every source must survive");
                    prop_assert_eq!(record.text.as_ref(), original.text.as_ref());
                    prop_assert_eq!(&record.paragraphs, &original.paragraphs);
                    prop_assert_eq!(&record.questions, &original.questions);
                    prop_assert_eq!(&record.sections, &original.sections);
                }
            }

            #[test]
            fn snapshot_from_bytes_never_panics_on_arbitrary_bytes(
                bytes in prop::collection::vec(any::<u8>(), 0..512),
            ) {
                let _ = snapshot_from_bytes(&bytes);
            }

            #[test]
            fn snapshot_from_bytes_never_panics_on_mutated_s4_bytes(
                sources in corpus_strategy(),
                watermark in any::<u64>(),
                mutations in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 0..24),
            ) {
                let mut bytes = snapshot_to_bytes(&sources, watermark);
                for (pick, value) in mutations {
                    *pick.get_mut(&mut bytes) = value;
                }
                let mutated = resealed(bytes);
                let _ = snapshot_from_bytes(&mutated);
            }

            #[test]
            fn snapshot_from_bytes_never_panics_on_mutated_s3_bytes(
                sources in corpus_strategy(),
                watermark in any::<u64>(),
                mutations in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 0..24),
            ) {
                let sealed = snapshot_to_bytes(&sources, watermark);
                let mut bytes = sealed[..sealed.len() - 4].to_vec();
                bytes[..8].copy_from_slice(S3_SNAPSHOT_MAGIC);
                for (pick, value) in mutations {
                    *pick.get_mut(&mut bytes) = value;
                }
                let _ = snapshot_from_bytes(&bytes);
            }
        }
    }
}
