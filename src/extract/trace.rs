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

/// ADR 0038 §3.6: what the read masked, addressed by rule and
/// paragraph — the record deliberately has no `raw`.
#[derive(serde::Serialize)]
struct TraceRedaction<'a> {
    kind: &'static str,
    rule: &'a str,
    paragraph: u32,
    placeholder: &'a str,
    bytes: usize,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    preexisting: bool,
}

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

/// ADR 0024 (#786): one item the model wrote that the batch does
/// not hold, with the original text it was about. `text` is the cited
/// paragraph when the item cited a valid one, else the whole piece the
/// model was shown — the loss is always readable in the original.
#[derive(serde::Serialize)]
pub(super) struct TraceLoss<'a> {
    pub(super) kind: &'static str,
    /// `association` | `alias` | `question`.
    pub(super) item: &'static str,
    /// `removed` (mechanical, ADR 0013 — Stage 1 or a Stage 2 prune)
    /// | `dropped` (merge's contract) | `duplicate` (merge folded it).
    pub(super) reason: &'static str,
    /// The rule or finding, in the report's own words.
    pub(super) rule: &'a str,
    /// `removed` only: the item's path in the accepted answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) path: Option<&'a str>,
    /// The item as the model wrote it.
    pub(super) raw: &'a serde_json::Value,
    pub(super) piece_id: &'a str,
    pub(super) attempt: Option<&'a AttemptRef>,
    pub(super) paragraph: Option<u32>,
    pub(super) text: &'a str,
    /// `duplicate` only: the piece whose copy the batch holds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) kept_piece_id: Option<&'a str>,
}

/// ADR 0027 (#789): what taguru itself put into the prompt to steer
/// the answer, exactly as prompted (each list carries the prompt's own
/// ranking and caps — computed by the same functions that render the
/// blocks, so record and prompt cannot drift). One record per
/// document today, `chunk_index: null`: every chunk of a document sees
/// the same steering (ADR 0014: candidates come from the whole
/// document, once; the vocabulary grows only between documents). When
/// #782 adds per-chunk context, its records set `chunk_index` — a
/// chunk's steering is the document-wide record plus its own.
#[derive(serde::Serialize)]
pub(super) struct TraceSteering<'a> {
    pub(super) kind: &'static str,
    pub(super) chunk_index: Option<usize>,
    /// ADR 0014's candidate names, as offered (empty: `--candidates`
    /// off, or a document with none).
    pub(super) candidates: &'a [String],
    /// The system prompt actually sent, by hash — always present,
    /// pinned or recomputed alike (ADR 0031 §3.6).
    pub(super) system_sha256: &'a str,
    /// ADR 0031 §3.6: the run_id the system prompt was pinned from,
    /// when `--replay` pinned it (its `ReplayIndex` named exactly one
    /// distinct recorded system) — the field is absent when this run
    /// computed its own, pin declined, or no `--replay` at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) pinned_from: Option<&'a str>,
    /// #759's reuse list, in prompt order with the prompted counts
    /// (empty: first document of a run with no `--vocabulary`).
    pub(super) vocabulary: Vec<VocabularyEntry<'a>>,
    /// ADR 0015's target-context concept names, as prompted.
    pub(super) context_names: &'a [String],
    /// ADR 0009 §11.1's schema block lists; `null` when no schema
    /// block was prompted.
    pub(super) schema: Option<SteeringSchema<'a>>,
}

#[derive(serde::Serialize)]
pub(super) struct VocabularyEntry<'a> {
    pub(super) label: &'a str,
    pub(super) count: usize,
}

#[derive(serde::Serialize)]
pub(super) struct SteeringSchema<'a> {
    pub(super) types: Vec<&'a str>,
    pub(super) constrained_relations: Vec<&'a str>,
}

/// ADR 0026 (#787): one canonical paragraph's coverage — `covered`
/// when at least one kept item cites it — with the paragraph's own
/// text exactly when it is NOT covered, so the unreflected side of
/// the document is listable in the original without re-splitting the
/// batch passage. `bytes` weights the coverage rate.
#[derive(serde::Serialize)]
pub(super) struct TraceParagraph<'a> {
    pub(super) kind: &'static str,
    pub(super) paragraph: u32,
    pub(super) bytes: usize,
    /// How many kept items (associations and questions) cite it.
    pub(super) items: usize,
    pub(super) covered: bool,
    /// The paragraph's text, present exactly when `covered` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) text: Option<&'a str>,
}

/// ADR 0026 (#787): one ADR 0016 coverage gap — a sentence holding a
/// candidate pair that no accepted association covers — with the FULL
/// sentence (stderr's quote is byte-capped; this is not) and the
/// paragraph's text. Written only under `--coverage`, exactly when
/// stderr names the gap.
#[derive(serde::Serialize)]
pub(super) struct TraceUncovered<'a> {
    pub(super) kind: &'static str,
    pub(super) paragraph: u32,
    pub(super) sentence: &'a str,
    pub(super) text: &'a str,
    /// The chunk whose range holds the paragraph (the first, for an
    /// oversized paragraph straddling several — ADR 0003 §7 repeats
    /// its number across them). `null` if no chunk range holds it —
    /// a taguru bug, never invented.
    pub(super) chunk_index: Option<usize>,
    pub(super) chunk_sha256: Option<&'a str>,
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
    /// The piece text the model was shown — a loss without a valid
    /// paragraph citation is shown against this.
    pub(super) text: String,
    /// ADR 0013's removals from this output, as #786 records them.
    pub(super) removed: Vec<Removal>,
    /// Lossy mode's parse-time drops (ADR 0024 §3.6).
    pub(super) unparsed: Vec<Removal>,
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
            text: piece.to_string(),
            removed: output.removed.clone(),
            unparsed: output.unparsed.clone(),
        }
    }
}

/// Renders one document's trace (ADR 0023 §3.4), in file order:
/// `document`, every `chunk`, every `piece`, every `item` in batch
/// order, then (ADR 0024) every `loss` — removals piece by piece, then
/// merge's drops and duplicates — then (ADR 0026) one `paragraph`
/// record per canonical paragraph (text attached exactly when no kept
/// item cites it) and one `uncovered` record per `--coverage` gap,
/// with the full sentence. `chunks` is the plan's descriptors;
/// `pieces` the chunk loop's outputs in `merge`'s input order, so
/// [`Extraction::origins`] and [`Loss::origin`] index straight into
/// it; `paragraphs` the document's canonical paragraphs' text, the
/// coordinate every `paragraph` field cites.
#[allow(clippy::too_many_arguments)] // one call site; a struct would only rename the eight
pub(super) fn render_trace(
    run_id: &str,
    source: &str,
    document_sha256: &str,
    batch_path: &Path,
    redactions: &[crate::sensitive::Redaction],
    chunks: &[ChunkDescriptor],
    pieces: &[PieceOrigin],
    paragraphs: &[&str],
    extraction: &Extraction,
    uncovered: &[CoverageGap],
    steering: &TraceSteering,
    units: &[Unit],
    blocks: &[Option<ContextBlock>],
    overview: &[Option<OverviewAnswer>],
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
    // ADR 0038 §3.6: one `redaction` record per match the read masked
    // (and per placeholder the input already carried), right after the
    // document record — rule, paragraph, placeholder, bytes; never the
    // matched text (no `raw`, on purpose).
    for redaction in redactions {
        push(&TraceRedaction {
            kind: "redaction",
            rule: &redaction.rule,
            paragraph: redaction.paragraph,
            placeholder: &redaction.placeholder,
            bytes: redaction.bytes,
            preexisting: redaction.preexisting,
        });
    }
    // ADR 0027: the prompt's steering lists, right after the document
    // record — they hold for every chunk below.
    push(steering);
    // ADR 0033 §3.4: the document's structural units, before the
    // chunks that cite them by index.
    for unit in units {
        push(&TraceStructure {
            kind: "structure",
            unit,
        });
    }
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
        // ADR 0033 §3.5: what the overview pass answered for this
        // chunk, then §3.6.4: what the chunk was told.
        if let Some(answer) = overview.get(chunk_index).and_then(Option::as_ref) {
            push(&TraceOverview {
                kind: "overview",
                chunk_index,
                answer,
            });
        }
        if let Some(block) = blocks.get(chunk_index).and_then(Option::as_ref) {
            push(&TraceChunkContext {
                kind: "chunk_context",
                chunk_index,
                block,
            });
        }
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
    // Losses (ADR 0024): each readable in the original — the cited
    // paragraph, else the piece.
    let cited = |paragraph: Option<u32>| -> Option<(u32, &str)> {
        let paragraph = paragraph?;
        paragraphs
            .get(paragraph as usize)
            .map(|text| (paragraph, *text))
    };
    for piece in pieces {
        let removed = piece.removed.iter().map(|removal| ("removed", removal));
        let unparsed = piece.unparsed.iter().map(|removal| ("dropped", removal));
        for (reason, removal) in removed.chain(unparsed) {
            let paragraph = cited(
                removal
                    .item
                    .get("paragraph")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|n| u32::try_from(n).ok()),
            );
            push(&TraceLoss {
                kind: "loss",
                item: removal.item_kind(),
                reason,
                rule: &removal.reason,
                path: Some(&removal.path),
                raw: &removal.item,
                piece_id: &piece.piece_id,
                attempt: piece.attempt.as_ref(),
                paragraph: paragraph.map(|(index, _)| index),
                text: paragraph.map_or(piece.text.as_str(), |(_, text)| text),
                kept_piece_id: None,
            });
        }
    }
    for loss in &extraction.losses {
        let Some(piece) = pieces.get(loss.origin) else {
            continue; // a taguru bug; the line is skipped, not invented
        };
        let paragraph = cited(loss.paragraph);
        push(&TraceLoss {
            kind: "loss",
            item: loss.kind,
            reason: loss.reason,
            rule: &loss.rule,
            path: None,
            raw: &loss.item,
            piece_id: &piece.piece_id,
            attempt: piece.attempt.as_ref(),
            paragraph: paragraph.map(|(index, _)| index),
            text: paragraph.map_or(piece.text.as_str(), |(_, text)| text),
            kept_piece_id: loss
                .kept_origin
                .and_then(|origin| pieces.get(origin))
                .map(|piece| piece.piece_id.as_str()),
        });
    }
    // ADR 0026: paragraph coverage — cited-by-kept-items, byte-
    // weighted by `bytes`; the text of exactly the paragraphs nothing
    // cites, so the unreflected side is readable in the original.
    let mut citations = vec![0usize; paragraphs.len()];
    for fact in &extraction.associations {
        if let Some(paragraph) = fact.paragraph
            && let Some(count) = citations.get_mut(paragraph as usize)
        {
            *count += 1;
        }
    }
    for (paragraph, _) in &extraction.questions {
        if let Some(count) = citations.get_mut(*paragraph as usize) {
            *count += 1;
        }
    }
    for (paragraph, (text, items)) in paragraphs.iter().zip(&citations).enumerate() {
        let covered = *items > 0;
        push(&TraceParagraph {
            kind: "paragraph",
            paragraph: paragraph as u32,
            bytes: text.len(),
            items: *items,
            covered,
            text: (!covered).then_some(*text),
        });
    }
    for gap in uncovered {
        let chunk = chunks.iter().enumerate().find(|(_, descriptor)| {
            (descriptor.paragraph_first..=descriptor.paragraph_last).contains(&gap.paragraph)
        });
        push(&TraceUncovered {
            kind: "uncovered",
            paragraph: gap.paragraph,
            sentence: &gap.sentence,
            text: paragraphs
                .get(gap.paragraph as usize)
                .copied()
                .unwrap_or_default(),
            chunk_index: chunk.map(|(index, _)| index),
            chunk_sha256: chunk.map(|(_, descriptor)| descriptor.sha256.as_str()),
        });
    }
    lines.join("\n") + "\n"
}

/// How many characters of a `piece_id` a human-facing line prints
/// (ADR 0037): enough to tell a document's pieces apart and to paste
/// back into `taguru inspect --piece`, which accepts any prefix.
pub(crate) const PIECE_ID_SHORT: usize = 12;

/// The printed form of a piece id — cut by characters, never bytes,
/// so an id read back from a hand-edited log prints oddly instead of
/// panicking.
pub(crate) fn short_piece_id(piece_id: &str) -> &str {
    piece_id
        .char_indices()
        .nth(PIECE_ID_SHORT)
        .map_or(piece_id, |(end, _)| &piece_id[..end])
}

/// The `[N] ` labels' range of a labeled piece, or `None` when the
/// text is not a [`labeled_document`] rendering — the lenient twin of
/// [`leading_paragraph_number`], for a record that prefers `null` to a
/// panic.
pub(crate) fn paragraph_range(piece: &str) -> Option<(u32, u32)> {
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
