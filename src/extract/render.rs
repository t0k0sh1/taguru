//! Rendering an `Extraction` into the JSONL batch text, and the
//! chunking helpers `render_batch`'s callers share.

use super::*;

/// Serializes the batch: header, passage (the document itself), the
/// facts, then aliases. serde_json strings never contain raw newlines,
/// so every `to_string` is one line by construction.
pub(super) fn render_batch(
    context: &str,
    source: &str,
    description: Option<&str>,
    extraction: &Extraction,
    passage: Option<&str>,
    date: Option<u64>,
    tags: &[String],
) -> String {
    let mut header = serde_json::json!({
        "taguru_batch": 1,
        "context": context,
        "source": source,
    });
    if let Some(text) = description {
        header["create"] = serde_json::json!({ "description": text });
    }
    let mut lines = vec![header.to_string()];
    if let Some(text) = passage {
        // #466 S1 (ADR 0017): the runbook's source metadata rides the
        // passage line, exactly where the import wire format carries it
        // (docs/import.html). Absent fields are omitted, keeping the
        // no-flags batch byte for byte today's.
        let mut line = serde_json::json!({ "passage": text });
        if let Some(date) = date {
            line["date"] = serde_json::json!(date);
        }
        if !tags.is_empty() {
            line["tags"] = serde_json::json!(tags);
        }
        lines.push(line.to_string());
        for (paragraph, question) in &extraction.questions {
            lines.push(
                serde_json::json!({ "paragraph": paragraph, "question": question }).to_string(),
            );
        }
    }
    for fact in &extraction.associations {
        let mut line = serde_json::json!({
            "subject": fact.subject,
            "label": fact.label,
            "object": fact.object,
            "weight": fact.weight,
        });
        // A paragraph locator attaches to THIS batch's passage line;
        // with the passage stripped (--no-passage) there is nothing to
        // locate into, and import refuses the dangling reference — so
        // strip the locators with the text they pointed at.
        if passage.is_some()
            && let Some(paragraph) = fact.paragraph
        {
            line["paragraph"] = serde_json::json!(paragraph);
        }
        lines.push(line.to_string());
    }
    for (alias, canonical) in &extraction.concepts {
        lines.push(
            serde_json::json!({"alias": alias, "canonical": canonical, "kind": "concept"})
                .to_string(),
        );
    }
    for (alias, canonical) in &extraction.labels {
        lines.push(
            serde_json::json!({"alias": alias, "canonical": canonical, "kind": "label"})
                .to_string(),
        );
    }
    lines.join("\n") + "\n"
}

/// Splits a document at paragraph boundaries into chunks of at most
/// `cap` bytes (an oversized paragraph splits at line, then char
/// boundaries). Chunks are prompt input only — the passage stays the
/// verbatim document — so exact reassembly does not matter; keeping
/// sentences whole does. A blank document yields no chunks.
pub(super) fn chunk(text: &str, cap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for paragraph in text.split("\n\n") {
        for piece in split_oversized(paragraph, cap) {
            if !current.is_empty() && current.len() + 2 + piece.len() > cap {
                chunks.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(piece);
        }
    }
    chunks.push(current);
    chunks.retain(|chunk| !chunk.trim().is_empty());
    chunks
}

/// Re-chunks one already-labeled piece to a smaller cap for the
/// ladder's split rung. [`chunk`] alone would carry an oversized
/// block's continuation to the model unlabeled — exactly what
/// [`labeled_document`] exists to prevent — so oversized blocks are
/// pre-split here with their `[N] ` label repeated on every piece:
/// the same discipline, at a smaller cap.
pub(super) fn split_labeled_piece(piece: &str, cap: usize) -> Vec<String> {
    let mut blocks = Vec::new();
    for block in piece.split("\n\n") {
        if block.len() <= cap {
            blocks.push(block.to_string());
            continue;
        }
        let label_length = block
            .starts_with('[')
            .then(|| block.find("] ").map(|index| index + 2))
            .flatten()
            .unwrap_or(0);
        let (label, content) = block.split_at(label_length);
        let piece_cap = cap.saturating_sub(label.len()).max(1);
        for sub in split_oversized(content, piece_cap) {
            blocks.push(format!("{label}{}", sub.trim_end_matches('\n')));
        }
    }
    chunk(&blocks.join("\n\n"), cap)
}

pub(super) fn split_oversized(paragraph: &str, cap: usize) -> Vec<&str> {
    if paragraph.len() <= cap {
        return vec![paragraph];
    }
    let mut pieces = Vec::new();
    let mut rest = paragraph;
    while rest.len() > cap {
        // Prefer the last line break inside the window; fall back to
        // the last char boundary, and always make progress.
        let window = &rest[..floor_char_boundary(rest, cap)];
        let mut cut = window
            .rfind('\n')
            .map(|index| index + 1)
            .unwrap_or(window.len());
        if cut == 0 {
            cut = rest
                .chars()
                .next()
                .map(char::len_utf8)
                .unwrap_or(rest.len());
        }
        pieces.push(&rest[..cut]);
        rest = &rest[cut..];
    }
    if !rest.is_empty() {
        pieces.push(rest);
    }
    pieces
}

pub(super) fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}
