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

/// One source's resident passage: the byte-exact text and its
/// paragraph spans, computed once per residency (never persisted —
/// the split function is deterministic, so the spans are as durable
/// as the text they index). Handed out as `Arc`, so search and the
/// index/vector lanes walk paragraphs without holding the store lock
/// and without copying text.
#[derive(Debug)]
pub(crate) struct PassageRecord {
    pub(crate) text: Arc<str>,
    pub(crate) paragraphs: Vec<ParagraphSpan>,
}

impl PassageRecord {
    fn new(text: Arc<str>) -> Arc<Self> {
        let paragraphs = paragraph::split(&text);
        Arc::new(Self { text, paragraphs })
    }

    /// Constructor for sibling modules' unit tests.
    #[cfg(test)]
    pub(crate) fn for_tests(text: &str) -> Arc<Self> {
        Self::new(Arc::from(text))
    }

    /// The paragraph texts behind the spans, in order.
    pub(crate) fn paragraph_texts(&self) -> impl Iterator<Item = (&ParagraphSpan, &str)> {
        self.paragraphs
            .iter()
            .map(|span| (span, &self.text[span.start as usize..span.end as usize]))
    }
}

/// One passage mutation — this log's whole vocabulary. Same
/// internally-tagged shape as the graph's `WalOp`, but its own enum in
/// its own file: each log speaks one vocabulary (see `wal::WalRecord`).
/// Both ops are last-write-wins, so a replay overlapping the snapshot
/// re-applies harmlessly — the watermark still gates it, but a bug
/// there degrades to wasted work, not corruption (unlike `associate`,
/// which accumulates).
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum PassageOp {
    Store { source: String, text: String },
    Retract { source: String },
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

const SNAPSHOT_MAGIC: &[u8; 8] = b"TAGURUS1";

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
    pub(crate) fn load(
        snapshot_path: PathBuf,
        legacy_path: &Path,
        log_path: PathBuf,
        max_log_bytes: usize,
    ) -> io::Result<Self> {
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
                let legacy = read_legacy(legacy_path);
                (
                    legacy
                        .into_iter()
                        .map(|(source, text)| (source, PassageRecord::new(Arc::from(text))))
                        .collect(),
                    0,
                    0,
                )
            }
            Err(error) => return Err(error),
        };

        let mut sources: BTreeMap<String, Arc<PassageRecord>> = sources;
        let (ops, top) = wal::replay::<PassageOp>(&log_path, watermark)?;
        for op in ops {
            match op {
                PassageOp::Store { source, text } => {
                    sources.insert(source, PassageRecord::new(Arc::from(text)));
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

        Ok(Self {
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
        })
    }

    /// Merge-upserts a batch, log-first: fsync the ops, then apply —
    /// on `Err` nothing was applied and nothing must be assumed
    /// durable. Returns the batch size (the historical contract: how
    /// many the request carried, not how many keys were new).
    pub(crate) fn store(&self, passages: BTreeMap<String, String>) -> io::Result<usize> {
        let ops: Vec<PassageOp> = passages
            .into_iter()
            .map(|(source, text)| PassageOp::Store { source, text })
            .collect();
        let count = ops.len();
        if count == 0 {
            return Ok(0);
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
        self.append_and_apply_locked(&writer, &ops)?;
        Ok(count)
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
        _writer: &std::sync::MutexGuard<'_, ()>,
        ops: &[PassageOp],
    ) -> io::Result<()> {
        let first_seq = self.inner.read().unwrap().next_seq;
        let appended = wal::append_batch(&self.log_path, first_seq, ops)?;
        let over_ratio = {
            let mut inner = self.inner.write().unwrap();
            for op in ops {
                match op {
                    PassageOp::Store { source, text } => {
                        let record = PassageRecord::new(Arc::from(text.as_str()));
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
            if let Err(error) = self.compact() {
                tracing::warn!(
                    "passage log at {} not compacted (will retry on a later write): {error}",
                    self.log_path.display()
                );
            }
        }
        Ok(())
    }

    /// Rewrites the snapshot from the current map and truncates the
    /// log if — and only if — no write landed while the bytes were in
    /// flight (the same watermark re-check `flush_entry` runs). The
    /// serialize-and-write happens with NO lock held: the map is
    /// `Arc<str>` handles, so the one consistent read is a cheap
    /// clone, and megabytes of text land on disk while readers and
    /// writers proceed.
    pub(crate) fn compact(&self) -> io::Result<()> {
        let (sources, seen_seq) = {
            let inner = self.inner.read().unwrap();
            (inner.sources.clone(), inner.next_seq)
        };
        let bytes = snapshot_to_bytes(&sources, seen_seq - 1);
        crate::registry::write_atomic(&self.snapshot_path, &bytes)?;

        // The snapshot's rename is durable, so the legacy file's
        // contents live there now (any base it fed is baked in) and the
        // loader will never consult it again — retire it. Ordering is
        // the whole point: removing it BEFORE a durable snapshot exists
        // would strand the passages of a crash in between. A failed
        // unlink lingers harmlessly and retries next compaction.
        if let Err(error) = fs::remove_file(&self.legacy_path)
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

    pub(crate) fn get(&self, source: &str) -> Option<Arc<PassageRecord>> {
        self.inner.read().unwrap().sources.get(source).cloned()
    }

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
/// text, spans, and the per-entry overhead constant.
fn record_bytes(source: &str, record: &PassageRecord) -> usize {
    source.len()
        + record.text.len()
        + record.paragraphs.len() * std::mem::size_of::<ParagraphSpan>()
        + SOURCE_OVERHEAD
}

fn snapshot_to_bytes(sources: &BTreeMap<String, Arc<PassageRecord>>, watermark: u64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(SNAPSHOT_MAGIC);
    out.extend_from_slice(&watermark.to_le_bytes());
    out.extend_from_slice(&(sources.len() as u32).to_le_bytes());
    // BTreeMap iterates sorted: identical content is byte-identical.
    // Only the text lands on disk — spans are recomputed on load.
    for (source, record) in sources {
        write_chunk(&mut out, source.as_bytes());
        write_chunk(&mut out, record.text.as_bytes());
    }
    out
}

fn snapshot_from_bytes(bytes: &[u8]) -> Option<(BTreeMap<String, Arc<PassageRecord>>, u64)> {
    let mut pos = 0usize;
    if bytes.get(..8)? != SNAPSHOT_MAGIC {
        return None;
    }
    pos += 8;
    let watermark = u64::from_le_bytes(bytes.get(pos..pos + 8)?.try_into().ok()?);
    pos += 8;
    let count = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
    pos += 4;
    let mut sources = BTreeMap::new();
    for _ in 0..count {
        let source = String::from_utf8(read_chunk(bytes, &mut pos)?.to_vec()).ok()?;
        let text = std::str::from_utf8(read_chunk(bytes, &mut pos)?).ok()?;
        sources.insert(source, PassageRecord::new(Arc::from(text)));
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
        )
        .unwrap()
    }

    fn batch(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|&(source, text)| (source.to_string(), text.to_string()))
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
        assert_eq!(store.store(batch(&[("第1段落", "旧版")])).unwrap(), 1);
        assert_eq!(
            store
                .store(batch(&[("第1段落", "新版"), ("第2段落", "追加")]))
                .unwrap(),
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
    fn passage_snapshot_rejects_truncated_or_garbage_bytes() {
        let sources: BTreeMap<String, Arc<PassageRecord>> =
            [("a".to_string(), PassageRecord::new(Arc::from("text")))]
                .into_iter()
                .collect();
        let bytes = snapshot_to_bytes(&sources, 7);
        let (parsed, watermark) = snapshot_from_bytes(&bytes).unwrap();
        assert_eq!(watermark, 7);
        assert_eq!(parsed["a"].text.as_ref(), "text");

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
    fn a_corrupt_snapshot_is_a_load_error_not_an_empty_store() {
        let dir = scratch_dir("strict");
        fs::write(dir.join("t.passages.bin"), b"not a snapshot").unwrap();
        let error = PassageStore::load(
            dir.join("t.passages.bin"),
            &dir.join("t.sources.json"),
            dir.join("t.passages.wal.jsonl"),
            0,
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
}
