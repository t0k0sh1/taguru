//! The per-document trace file (ADR 0023): `--out/.extract-trace/
//! <batch name>`, one JSONL line per chunk, piece, and batch item,
//! joining every item of the written batch to the piece of text and
//! the completion that produced it. Written in the same step as the
//! batch, with the batch's lifecycle; never read by `extract` itself.

use super::*;

/// Directory (under `--out`, hidden like `.extract-checkpoints/` and
/// a subdirectory so `taguru import DIR`'s `*.jsonl` expansion never
/// reads a trace as a batch — ADR 0023 §3.3).
pub(super) const TRACE_DIR_NAME: &str = ".extract-trace";

/// ADR 0023 §3.4's `document` record — the file's first line.
#[derive(serde::Serialize)]
pub(super) struct TraceDocument<'a> {
    pub(super) kind: &'static str,
    pub(super) run_id: &'a str,
    pub(super) source: &'a str,
    pub(super) document_sha256: &'a str,
    pub(super) batch_path: String,
    pub(super) chunk_total: usize,
}

/// ADR 0023 §3.4's `chunk` record: exactly the diagnostics `chunk`
/// record's provenance fields (ADR 0003 §7), minus `source` (the file
/// is already one document's).
#[derive(serde::Serialize)]
pub(super) struct TraceChunk<'a> {
    pub(super) kind: &'static str,
    pub(super) chunk_index: usize,
    pub(super) chunk_total: usize,
    pub(super) chunk_sha256: &'a str,
    pub(super) chunk_bytes: usize,
    pub(super) paragraph_first: u32,
    pub(super) paragraph_last: u32,
}

/// ADR 0023 §3.4's `piece` record: one per output of the chunk loop,
/// in output order. `attempt` is the completion whose answer the
/// output IS — Stage 2's corrective answer when one replaced the
/// Stage 1 answer — and `null` only for a unit reused from a
/// pre-0023 checkpoint. `paragraph_first`/`paragraph_last` are
/// `null` when the piece text carries no `[N] ` labels (never on a
/// real run; tests build bare pieces).
#[derive(serde::Serialize)]
pub(super) struct TracePiece<'a> {
    pub(super) kind: &'static str,
    pub(super) piece_id: &'a str,
    pub(super) chunk_index: usize,
    pub(super) chunk_sha256: Option<&'a str>,
    pub(super) piece_bytes: usize,
    pub(super) paragraph_first: Option<u32>,
    pub(super) paragraph_last: Option<u32>,
    pub(super) reused: bool,
    pub(super) attempt: Option<&'a AttemptRef>,
}

/// ADR 0023 §3.4's `item` record: the item's content key (§3.1), as
/// the batch line spells it, plus the `piece_id` it joins on.
#[derive(serde::Serialize)]
pub(super) struct TraceItem<'a> {
    pub(super) kind: &'static str,
    pub(super) item: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) subject: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) label: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) object: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) alias: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) canonical: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) paragraph: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) question: Option<&'a str>,
    /// `None` would mean an item `merge` kept without recording its
    /// origin — a taguru bug; serialized as `null` rather than hidden.
    pub(super) piece_id: Option<&'a str>,
}

impl<'a> TraceItem<'a> {
    fn blank(item: &'static str, piece_id: Option<&'a str>) -> Self {
        Self {
            kind: "item",
            item,
            subject: None,
            label: None,
            object: None,
            alias: None,
            canonical: None,
            paragraph: None,
            question: None,
            piece_id,
        }
    }
}

/// What the chunk loop knew about one output, kept past `merge` (which
/// consumes the `ChunkOutput`s) so the trace can name it.
pub(super) struct PieceOrigin {
    pub(super) piece_id: String,
    pub(super) chunk_index: usize,
    pub(super) piece_bytes: usize,
    pub(super) paragraph_range: Option<(u32, u32)>,
    pub(super) reused: bool,
    pub(super) attempt: Option<AttemptRef>,
}

impl PieceOrigin {
    /// Read off a [`ChunkOutput`] before it is consumed: the piece text
    /// is the user turn's document part, byte for byte
    /// (`user_message_document` is `user_message`'s inverse). `reused`
    /// is derived from the attempt's run: an output this run produced
    /// carries this run's id; a checkpointed one carries the producing
    /// run's (or none, pre-0023).
    pub(super) fn of(output: &ChunkOutput, run_id: &str) -> Self {
        let piece = user_message_document(&output.user);
        Self {
            piece_id: output.piece_id.clone(),
            chunk_index: output.chunk_index,
            piece_bytes: piece.len(),
            paragraph_range: paragraph_range(piece),
            reused: output
                .attempt
                .as_ref()
                .is_none_or(|attempt| attempt.run_id != run_id),
            attempt: output.attempt.clone(),
        }
    }
}

/// Renders one document's trace (ADR 0023 §3.4), in file order:
/// `document`, every `chunk`, every `piece`, every `item` in batch
/// order. `chunks` is the plan's descriptors; `pieces` the chunk
/// loop's outputs in `merge`'s input order, so [`Extraction::origins`]
/// indexes straight into it.
pub(super) fn render_trace(
    run_id: &str,
    source: &str,
    document_sha256: &str,
    batch_path: &Path,
    chunks: &[ChunkDescriptor],
    pieces: &[PieceOrigin],
    extraction: &Extraction,
) -> String {
    let mut lines = Vec::new();
    let mut push = |record: &dyn erased_serialize::Serialize| {
        // Every record here is plain, always-serializable fields — a
        // failure would be a taguru bug; the line is skipped rather
        // than the document failed (ADR 0023 §3.6: the trace is
        // advisory).
        if let Ok(line) = record.to_json_line() {
            lines.push(line);
        }
    };
    push(&TraceDocument {
        kind: "document",
        run_id,
        source,
        document_sha256,
        batch_path: batch_path.display().to_string(),
        chunk_total: chunks.len(),
    });
    for (chunk_index, descriptor) in chunks.iter().enumerate() {
        push(&TraceChunk {
            kind: "chunk",
            chunk_index,
            chunk_total: chunks.len(),
            chunk_sha256: &descriptor.sha256,
            chunk_bytes: descriptor.text.len(),
            paragraph_first: descriptor.paragraph_first,
            paragraph_last: descriptor.paragraph_last,
        });
    }
    for piece in pieces {
        push(&TracePiece {
            kind: "piece",
            piece_id: &piece.piece_id,
            chunk_index: piece.chunk_index,
            chunk_sha256: chunks
                .get(piece.chunk_index)
                .map(|descriptor| descriptor.sha256.as_str()),
            piece_bytes: piece.piece_bytes,
            paragraph_first: piece.paragraph_range.map(|(first, _)| first),
            paragraph_last: piece.paragraph_range.map(|(_, last)| last),
            reused: piece.reused,
            attempt: piece.attempt.as_ref(),
        });
    }
    let piece_of = |key: &ItemKey| -> Option<&str> {
        extraction
            .origins
            .get(key)
            .and_then(|&origin| pieces.get(origin))
            .map(|piece| piece.piece_id.as_str())
    };
    // Batch order (render_batch): questions, associations, concepts,
    // labels.
    for (paragraph, question) in &extraction.questions {
        let mut item = TraceItem::blank(
            "question",
            piece_of(&ItemKey::Question(*paragraph, question.clone())),
        );
        item.paragraph = Some(*paragraph);
        item.question = Some(question);
        push(&item);
    }
    for fact in &extraction.associations {
        // `Fact::origin` is the same number `origins` holds for the
        // triple — read from the fact, which already carries it.
        let mut item = TraceItem::blank(
            "association",
            pieces.get(fact.origin).map(|piece| piece.piece_id.as_str()),
        );
        item.subject = Some(&fact.subject);
        item.label = Some(&fact.label);
        item.object = Some(&fact.object);
        push(&item);
    }
    for (namespace, kind, map) in [
        (
            "concept",
            ItemKey::Concept as fn(String) -> ItemKey,
            &extraction.concepts,
        ),
        (
            "label",
            ItemKey::Label as fn(String) -> ItemKey,
            &extraction.labels,
        ),
    ] {
        for (alias, canonical) in map {
            let mut item = TraceItem::blank(namespace, piece_of(&kind(alias.clone())));
            item.alias = Some(alias);
            item.canonical = Some(canonical);
            push(&item);
        }
    }
    lines.join("\n") + "\n"
}

/// The `[N] ` labels' range of a labeled piece, or `None` when the
/// text is not a [`labeled_document`] rendering — the lenient twin of
/// [`leading_paragraph_number`], for a record that prefers `null` to a
/// panic.
pub(super) fn paragraph_range(piece: &str) -> Option<(u32, u32)> {
    let number = |block: &str| -> Option<u32> {
        block
            .strip_prefix('[')
            .and_then(|rest| rest.split_once("] "))
            .and_then(|(digits, _)| digits.parse().ok())
    };
    let first = number(piece.split("\n\n").next()?)?;
    let last = number(piece.rsplit("\n\n").next()?)?;
    Some((first, last))
}

/// Writes the trace beside the batch: `--out/.extract-trace/<batch
/// name>`, atomically. A failure is reported once on stderr and never
/// fails the document (ADR 0023 §3.6).
pub(super) fn write_trace(out: &Path, file_name: &str, body: &str) {
    let dir = out.join(TRACE_DIR_NAME);
    let result = fs::create_dir_all(&dir)
        .and_then(|()| crate::storage::write_atomic(&dir.join(file_name), body.as_bytes()));
    if let Err(error) = result {
        eprintln!(
            "taguru: extract: trace: writing {}: {error} — the batch is written; its trace \
             is not",
            dir.join(file_name).display()
        );
    }
}

/// A tiny object-safe serialization seam so [`render_trace`] can push
/// four record shapes through one closure without boxing them.
mod erased_serialize {
    pub(super) trait Serialize {
        fn to_json_line(&self) -> serde_json::Result<String>;
    }
    impl<T: serde::Serialize> Serialize for T {
        fn to_json_line(&self) -> serde_json::Result<String> {
            serde_json::to_string(self)
        }
    }
}
