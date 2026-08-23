//! The `--diagnostics-out`/`TAGURU_EXTRACT_DIAGNOSTICS` JSONL sidecar:
//! `DiagnosticsSink` and the record shapes it writes.

use super::*;

/// The `--diagnostics-out`/`TAGURU_EXTRACT_DIAGNOSTICS` JSONL sidecar
/// (issue #200, ADR 0001 §10): a tagged stream of records — `kind`
/// discriminates `chunk` (once per chunk, before its first attempt),
/// `attempt` (one per LLM attempt, the original and still the only
/// `kind` most consumers need), and `document` (once per document
/// written) — opt-in, metadata-only by default (issue #262, ADR 0003
/// §7). `File::create` truncates on open — the sidecar describes THIS
/// run, never a prior one appended to, so a skipped-everything rerun
/// leaves it empty rather than stale (docs/extract.html says so).
/// `Mutex`-guarded because `--parallel` dispatches chunk workers
/// concurrently onto the same file
/// (`crate::registry::dispatch_chunks_concurrently`); each emitted
/// record is one `write_all` + `flush` so a killed run keeps every
/// completed line — no fsync, unlike `wal.rs`'s crash-durable records:
/// this sidecar is advisory, and a document's own batch file and the
/// manifest are what "written" actually means.
pub(super) struct DiagnosticsSink {
    pub(super) writer: Mutex<std::io::BufWriter<fs::File>>,
    /// `None`: never attach `response_text`. `Some(n)`, always `n > 0`
    /// ([`DiagnosticsSink::open`] folds `Some(0)` to `None`): cap raw
    /// text at `n` bytes, [`corrective_assistant_turn`]'s treatment.
    pub(super) raw_cap: Option<usize>,
    pub(super) path: PathBuf,
    /// Set on the first write failure so the one warning line prints
    /// once for the run, not once per dropped record.
    pub(super) warned: AtomicBool,
}

impl DiagnosticsSink {
    /// Opens (truncating) and writes the `kind: "run"` record first
    /// (ADR 0023 §3.3): `run_id` is what joins this run's `attempt`
    /// records to the per-document trace files it wrote.
    pub(super) fn open(
        path: PathBuf,
        raw_cap: Option<usize>,
        run_id: &str,
    ) -> std::io::Result<Self> {
        let file = fs::File::create(&path)?;
        let sink = Self {
            writer: Mutex::new(std::io::BufWriter::new(file)),
            raw_cap: raw_cap.filter(|&n| n > 0),
            path,
            warned: AtomicBool::new(false),
        };
        sink.write_record(&RunRecord {
            kind: "run",
            run_id: run_id.to_string(),
        });
        Ok(sink)
    }

    /// Assembles and writes one record. A diagnostics write never fails
    /// extraction itself (requirement 4 only binds stdout/stderr when
    /// the FLAG is absent) — an I/O error here earns one stderr warning
    /// and every later record on this sink is silently dropped.
    pub(super) fn emit(&self, attempt: DiagnosticsAttempt) {
        let provider_metadata = attempt.response.map(|response| ProviderMetadataRecord {
            finish_reason: response.finish_reason.clone(),
            input_tokens: response.usage.and_then(|usage| usage.input_tokens),
            output_tokens: response.usage.and_then(|usage| usage.output_tokens),
            total_tokens: response.usage.and_then(|usage| usage.total_tokens),
        });
        let response_text = attempt
            .response
            .and_then(|response| self.capture_raw(&response.content));
        let record = AttemptRecord {
            kind: "attempt",
            run_id: attempt.attempt_ref.run_id.clone(),
            attempt_seq: attempt.attempt_ref.attempt_seq,
            piece_id: attempt.piece_id.to_string(),
            source: attempt.source.to_string(),
            stage: attempt.stage,
            chunk_index: attempt.chunk_index,
            attempt: attempt.attempt,
            max_attempts: attempt.max_attempts,
            state: attempt.state,
            length_limited: attempt.length_limited,
            elapsed_seconds: attempt.elapsed.as_secs_f64(),
            provider_metadata,
            parse_error: attempt.parse_error.map(str::to_string),
            validation_issues: attempt.validation_issues.map(<[String]>::to_vec),
            removed_items: attempt.removed_items,
            piece_bytes: attempt.piece_bytes,
            requested_max_tokens: attempt.requested_max_tokens,
            response_text,
        };
        self.write_record(&record);
    }

    /// One `kind: "chunk"` record, before that chunk's first attempt
    /// (issue #262, ADR 0003 §7): `source`/`chunk_index`/`chunk_total`
    /// identify it the same way an `attempt` record does, and
    /// `chunk_sha256`/`paragraph_first`/`paragraph_last` are exactly
    /// [`ChunkDescriptor`]'s own provenance fields, unmodified.
    pub(super) fn emit_chunk(
        &self,
        source: &str,
        chunk_index: usize,
        chunk_total: usize,
        descriptor: &ChunkDescriptor,
    ) {
        self.write_record(&ChunkRecord {
            kind: "chunk",
            source: source.to_string(),
            chunk_index,
            chunk_total,
            chunk_sha256: descriptor.sha256.clone(),
            chunk_bytes: descriptor.text.len(),
            paragraph_first: descriptor.paragraph_first,
            paragraph_last: descriptor.paragraph_last,
        });
    }

    /// One `kind: "document"` record, built at the same call site as
    /// [`Run::report`] from the same `Extraction` value already in
    /// scope there (issue #262, ADR 0003 §7) — a structured version of
    /// what `report` only ever prints as one human-readable line.
    /// Unlike `report`, `concepts` and `labels` are counted separately
    /// rather than combined into one "alias(es)" figure, since both
    /// `BTreeMap`s are already in scope at no extra cost. Written only
    /// once a document lands successfully — a document that fails
    /// never reaches this call site, so its absence here marks exactly
    /// that, the same "absence marks incomplete" convention `kind:
    /// "cell"` uses at the harness's cell scope (ADR 0003 §9.2).
    pub(super) fn emit_document(
        &self,
        source: &str,
        extraction: &Extraction,
        removed: usize,
        uncovered: usize,
        out_path: &Path,
    ) {
        self.write_record(&DocumentRecord {
            kind: "document",
            source: source.to_string(),
            associations: extraction.associations.len(),
            concepts: extraction.concepts.len(),
            labels: extraction.labels.len(),
            questions: extraction.questions.len(),
            duplicates: extraction.duplicates,
            dropped: extraction.dropped,
            removed,
            uncovered,
            batch_path: out_path.display().to_string(),
        });
    }

    /// Serializes and appends one record, shared by [`Self::emit`],
    /// [`Self::emit_chunk`], and [`Self::emit_document`] — the
    /// serialize-then-append-then-warn-once mechanics are identical
    /// across all three `kind`s; only the record shape differs.
    pub(super) fn write_record(&self, record: &impl serde::Serialize) {
        let mut line = match serde_json::to_string(record) {
            Ok(line) => line,
            // Every record type here is plain, always-serializable
            // fields — this would be a taguru bug, not a runtime
            // condition; never seen in practice, worth 0 diagnostics
            // rather than a panic mid-extraction.
            Err(_) => return,
        };
        line.push('\n');
        let mut writer = match self.writer.lock() {
            Ok(writer) => writer,
            Err(poisoned) => poisoned.into_inner(),
        };
        let wrote = writer
            .write_all(line.as_bytes())
            .and_then(|()| writer.flush());
        if let Err(error) = wrote
            && !self.warned.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "taguru: extract: diagnostics: writing {}: {error} — further records are \
                 dropped",
                self.path.display()
            );
        }
    }

    /// The raw-text opt-in's byte cap, applied at capture time — exactly
    /// [`corrective_assistant_turn`]'s treatment of a prior bad answer.
    pub(super) fn capture_raw(&self, content: &str) -> Option<String> {
        let cap = self.raw_cap?;
        Some(if cap >= content.len() {
            content.to_string()
        } else {
            format!(
                "{}… [truncated to {cap} bytes]",
                &content[..floor_char_boundary(content, cap)]
            )
        })
    }
}

/// One attempt's diagnostics, gathered at the call site — legacy
/// `extract_chunk`, ladder `extract_round`, and Stage 2
/// `correct_cross_output_issues` each classify an attempt differently
/// (ADR 0001 §7), so each builds this itself rather than
/// [`DiagnosticsSink::emit`] trying to re-derive it from a shared
/// shape.
pub(super) struct DiagnosticsAttempt<'a> {
    pub(super) source: &'a str,
    /// ADR 0023: this completion's identity, taken from
    /// [`ChatClient::next_attempt`] right before the call it names.
    pub(super) attempt_ref: &'a AttemptRef,
    /// ADR 0023: the piece this completion asked about (for Stage 2,
    /// the piece whose output is being corrected).
    pub(super) piece_id: &'a str,
    pub(super) stage: &'static str,
    pub(super) chunk_index: usize,
    pub(super) attempt: usize,
    pub(super) max_attempts: usize,
    /// ADR 0001 §7's vocabulary: `stop_valid`, `stop_malformed`,
    /// `length_limited`, `empty`, `refusal`, `timeout`, `transport`.
    pub(super) state: &'static str,
    pub(super) length_limited: bool,
    pub(super) elapsed: Duration,
    /// `None` exactly for `timeout`/`transport` — no response exists.
    pub(super) response: Option<&'a ChatCompletion>,
    pub(super) parse_error: Option<&'a str>,
    pub(super) validation_issues: Option<&'a [String]>,
    /// ADR 0013: the mechanical pass's removals on a `stop_valid`
    /// attempt — `Some` exactly when the accepted answer had items
    /// removed; every other state is `None`.
    pub(super) removed_items: Option<Vec<String>>,
    /// Ladder-only: the byte length of the piece this round asked
    /// about, distinguishing split sub-pieces that share one
    /// `chunk_index`.
    pub(super) piece_bytes: Option<usize>,
    /// Ladder-only: this round's `max_tokens`, when one was sent.
    pub(super) requested_max_tokens: Option<usize>,
}

/// One JSONL line of the `--diagnostics-out` sidecar. Field names
/// mirror `taguru-langchain`'s `AttemptFailed`/`ProviderMetadata`
/// events (sdk/python-langchain/src/taguru_langchain/events.py)
/// wherever the concept matches, so a parity test can compare the two
/// shapes structurally instead of through a name-mapping table — see
/// this module's `attempt_record_serializes_the_shared_key_set` test
/// and its Python twin in `tests/unit/test_events.py`. Metadata only:
/// `response_text` exists exactly when
/// TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES opted in, byte-capped at
/// capture — never chain-of-thought, only the assistant's final text
/// (ADR 0001 §10).
#[derive(serde::Serialize)]
pub(super) struct AttemptRecord {
    pub(super) kind: &'static str,
    /// ADR 0023: always present — `(run_id, attempt_seq)` is the key a
    /// trace file's `piece.attempt` joins on, `piece_id` the key its
    /// items join on.
    pub(super) run_id: String,
    pub(super) attempt_seq: u64,
    pub(super) piece_id: String,
    pub(super) source: String,
    pub(super) stage: &'static str,
    pub(super) chunk_index: usize,
    pub(super) attempt: usize,
    pub(super) max_attempts: usize,
    pub(super) state: &'static str,
    pub(super) length_limited: bool,
    pub(super) elapsed_seconds: f64,
    pub(super) provider_metadata: Option<ProviderMetadataRecord>,
    pub(super) parse_error: Option<String>,
    pub(super) validation_issues: Option<Vec<String>>,
    /// Rust-only, like `piece_bytes` below: absent (never null) when
    /// the attempt removed nothing — the flagless shape is unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) removed_items: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) piece_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) requested_max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) response_text: Option<String>,
}

/// `removed_items`'s value for an accepted answer: each removal's
/// display string, `Some` exactly when anything was removed (ADR
/// 0013's record, unchanged by #786's structured `Removal`).
pub(super) fn removed_item_texts(removed: &[Removal]) -> Option<Vec<String>> {
    (!removed.is_empty()).then(|| removed.iter().map(ToString::to_string).collect())
}

/// The sidecar's first line (ADR 0023 §3.3): which run the `attempt`
/// records below belong to. Written by [`DiagnosticsSink::open`]
/// itself, so no sidecar of this version lacks it.
#[derive(serde::Serialize)]
pub(super) struct RunRecord {
    pub(super) kind: &'static str,
    pub(super) run_id: String,
}

/// [`ChatCompletion`]'s `finish_reason` and [`TokenUsage`], nested to
/// match `ProviderMetadata`'s serialized shape on the Python side.
#[derive(serde::Serialize)]
pub(super) struct ProviderMetadataRecord {
    pub(super) finish_reason: Option<String>,
    pub(super) input_tokens: Option<u64>,
    pub(super) output_tokens: Option<u64>,
    pub(super) total_tokens: Option<u64>,
}

/// One `kind: "chunk"` JSONL line (issue #262, ADR 0003 §7): the
/// provenance a benchmark harness or any other `--diagnostics-out`
/// consumer needs to point an attempt back at the original document —
/// `paragraph_first`/`paragraph_last` are a `crate::paragraph::split`
/// index range, never a byte offset (see [`ChunkDescriptor`]).
/// `AttemptRecord` gains no field for this; the two stay joinable by
/// `(source, chunk_index)`, matching the key an `attempt` record
/// already carries.
#[derive(serde::Serialize)]
pub(super) struct ChunkRecord {
    pub(super) kind: &'static str,
    pub(super) source: String,
    pub(super) chunk_index: usize,
    pub(super) chunk_total: usize,
    pub(super) chunk_sha256: String,
    pub(super) chunk_bytes: usize,
    pub(super) paragraph_first: u32,
    pub(super) paragraph_last: u32,
}

/// One `kind: "document"` JSONL line (issue #262, ADR 0003 §7): the
/// structured counterpart of [`Run::report`]'s single human-readable
/// line, written once a document lands successfully.
#[derive(serde::Serialize)]
pub(super) struct DocumentRecord {
    pub(super) kind: &'static str,
    pub(super) source: String,
    pub(super) associations: usize,
    pub(super) concepts: usize,
    pub(super) labels: usize,
    pub(super) questions: usize,
    pub(super) duplicates: usize,
    pub(super) dropped: usize,
    /// ADR 0013: how many items the mechanical pass removed across the
    /// document's accepted answers (Stage 1 and the Stage 2 alias
    /// prune together) — the count `Run::report` prints as "removed
    /// (mechanical validation)".
    pub(super) removed: usize,
    /// ADR 0016: how many candidate-pair sentences no accepted
    /// association covered — the count `Run::report` prints as
    /// "uncovered (coverage)"; always 0 when `--coverage` is off.
    pub(super) uncovered: usize,
    pub(super) batch_path: String,
}
