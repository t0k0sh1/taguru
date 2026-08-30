//! ADR 0033 (#782): chunk context — what a chunk is told about where
//! it sits in its document and what came before it, so the model
//! reads `同法`, `前条`, `この製品`, or a bare `## Borrowing` section
//! knowing what they refer to. "Context" here is 文脈, never a Taguru
//! context (the namespace `--context NAME` targets); everything
//! user-facing says *chunk context* and every identifier carries
//! `chunk_context`.
//!
//! `--chunk-context structure` (this module, ADR 0033 §3.3 stage a) is
//! entirely mechanical: the document's structure is detected from its
//! own lines (§3.4), and each chunk's block is rendered from that
//! structure and the paragraphs before the chunk (§3.6). No model
//! call, no dictionary, deterministic — the same text always yields
//! the same block, so the block is a computation input only through
//! the mode name.
//!
//! The block is prompt input only. It carries no `[N]` paragraph
//! label, so nothing in it can be tagged as a fact's paragraph (ADR
//! 0003 §7 untouched); it is text the document itself holds, so the
//! occurrence check reads it (ADR 0013, extended by §3.6.2); and it is
//! bounded to a fraction of the chunk cap so the chunk is never
//! squeezed.

use super::*;

/// `--chunk-context MODE`, cumulative (ADR 0033 §3.3). Only the modes
/// this version implements parse; `overview` and `ingested` are named
/// by the ADR and land with their own PRs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ChunkContextMode {
    /// Today's prompt, byte for byte.
    #[default]
    Off,
    /// Position + overlap + mechanically resolved references, and
    /// structure-aware chunk boundaries. No model call.
    Structure,
}

impl ChunkContextMode {
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "structure" => Some(Self::Structure),
            _ => None,
        }
    }

    /// The manifest/checkpoint/settings record of the mode: `""` when
    /// off — the "new field defaults to the value that changes today's
    /// behavior least" precedent (`structured_output`/`candidates`), so
    /// every entry written before ADR 0033 keeps matching a default
    /// run — and the mode name otherwise.
    pub(super) fn manifest_value(self) -> &'static str {
        match self {
            Self::Off => "",
            Self::Structure => "structure",
        }
    }

    pub(super) fn is_on(self) -> bool {
        self != Self::Off
    }
}

/// The accepted spellings, for usage text and errors.
pub(super) const CHUNK_CONTEXT_MODES: &str = "off, structure";

/// One structural unit of a document (ADR 0033 §3.4): a heading and
/// the paragraphs it governs, up to the next heading. `level` is
/// 1-based, outermost first, in whatever numbering the document's own
/// convention gives (Markdown `#` count; 章 1 / 節 2 / 条 3; ◆ 1 / ○ 2;
/// a numbered heading's depth). `paragraph_last` is inclusive and can
/// equal the next unit's `paragraph_first` when two headings share one
/// canonical paragraph (a statute's `第二章` and `第三条` lines with no
/// blank line between them); a heading that opens its paragraph ends
/// the previous unit at the paragraph before.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(super) struct Unit {
    pub(super) unit: usize,
    pub(super) level: u8,
    pub(super) heading: String,
    pub(super) paragraph_first: u32,
    pub(super) paragraph_last: u32,
    /// The byte offset (into the document) just past the heading line
    /// — where the unit's opening text starts. Not serialized: the
    /// trace's `structure` record is paragraph-addressed.
    #[serde(skip)]
    body_start: usize,
}

/// What one line is, structurally, when it heads a unit.
pub(super) fn heading_of(line: &str) -> Option<(u8, String)> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.len() > 200 {
        return None;
    }
    // Markdown ATX heading.
    let hashes = trimmed.bytes().take_while(|b| *b == b'#').count();
    if (1..=6).contains(&hashes) && trimmed[hashes..].starts_with(' ') {
        let text = trimmed[hashes..].trim().trim_end_matches('#').trim();
        if !text.is_empty() {
            return Some((hashes as u8, text.to_string()));
        }
    }
    // Japanese statute: 第N章 / 第N節 / 第N条 (with an optional の二 …).
    if let Some(rest) = trimmed.strip_prefix('第') {
        let digits = rest
            .chars()
            .take_while(|c| "一二三四五六七八九十百千〇0123456789０１２３４５６７８９".contains(*c))
            .count();
        if digits > 0 {
            let after: String = rest.chars().skip(digits).collect();
            let (level, kind) = match after.chars().next() {
                Some('章') => (1u8, '章'),
                Some('節') => (2, '節'),
                Some('条') => (3, '条'),
                _ => (0, ' '),
            };
            if level > 0 {
                let number: String = rest.chars().take(digits).collect();
                // 第三条の二: the branch number rides on the article.
                let mut heading = format!("第{number}{kind}");
                let mut tail: String = after.chars().skip(1).collect();
                if kind == '条'
                    && let Some(branch) = tail.strip_prefix('の')
                {
                    let branch_digits: String = branch
                        .chars()
                        .take_while(|c| "一二三四五六七八九十".contains(*c))
                        .collect();
                    if !branch_digits.is_empty() {
                        heading.push('の');
                        heading.push_str(&branch_digits);
                        tail = branch[branch_digits.len()..].to_string();
                    }
                }
                // What follows the number is either nothing, or a
                // title/body after whitespace (`第一条　この法律は`);
                // `第二条の用語により` and `第二章の規定` are prose
                // that merely opens with a reference.
                if !tail.is_empty() && !tail.starts_with(char::is_whitespace) {
                    return None;
                }
                if kind != '条' {
                    // A chapter/section line carries its title.
                    // U+3000 is whitespace to `trim`, so 第二章　題 splits here.
                    let title = tail.trim();
                    if !title.is_empty() {
                        heading.push('　');
                        heading.push_str(title);
                    }
                }
                return Some((level, heading));
            }
        }
    }
    // A statute's supplementary provisions: `附則` / `附　則`, a
    // chapter-level unit — so an amendment's 第一条（施行期日） sits
    // under it, never beside the act's own 第一条 (#780's law-13).
    if trimmed.starts_with("附則") || trimmed.starts_with("附　則") {
        return Some((1, one_line(trimmed)));
    }
    // Minutes: ◆ section (speaker or metadata block), ○ utterance.
    if let Some(rest) = trimmed.strip_prefix('◆') {
        let text = rest.trim();
        if !text.is_empty() {
            return Some((1, format!("◆{text}")));
        }
    }
    if let Some(rest) = trimmed.strip_prefix('○') {
        let speaker: String = rest
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '　')
            .take(30)
            .collect();
        if !speaker.is_empty() {
            return Some((2, format!("○{speaker}")));
        }
    }
    // Numbered heading: `1.`, `1.2`, `1.2.3 Title`, `§3`, `§ 3.1`.
    // Short lines only (a numbered list item is a sentence; a heading
    // is a label) and never a bare number.
    let numbered = trimmed
        .strip_prefix('§')
        .map(|rest| rest.trim_start())
        .unwrap_or(trimmed);
    let mut depth = 0u8;
    let mut cursor = numbered;
    loop {
        let digits = cursor.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            break;
        }
        depth += 1;
        cursor = &cursor[digits..];
        match cursor.strip_prefix('.') {
            Some(rest) => cursor = rest,
            None => break,
        }
    }
    // `1. Title` alone is indistinguishable from a list item, so a
    // numbered heading needs either the § sign or a dotted depth
    // (`1.2 Title`); a trailing full stop marks a sentence, not a label.
    if depth > 0 && trimmed.len() <= 80 {
        let title = cursor.trim();
        let is_section = trimmed.starts_with('§');
        if !title.is_empty()
            && (is_section || depth > 1)
            && !title.ends_with('。')
            && !title.ends_with('.')
        {
            return Some((depth.min(6), trimmed.to_string()));
        }
    }
    None
}

/// ADR 0033 §3.4: the document's structure, from its own lines. Every
/// line is classified; a heading line opens a unit at its paragraph.
/// A parenthesized statute 見出し line (`（目的）`) immediately before a
/// 第N条 line names that article. A document with no heading yields no
/// units, and every chunk of it gets no position (and no references).
pub(super) fn detect_units(text: &str, spans: &[crate::paragraph::ParagraphSpan]) -> Vec<Unit> {
    let mut units: Vec<Unit> = Vec::new();
    let mut pending_caption: Option<String> = None;
    // Inside an HTML comment (`<!-- … -->`, the shape a bilingual
    // Markdown source keeps its original headings in) nothing is a
    // heading: the comment is not what the reader reads.
    let mut in_comment = false;
    let mut first_line = true;
    for span in spans {
        let start = span.start as usize;
        let paragraph = &text[start..span.end as usize];
        let mut offset = start;
        for line in paragraph.split_inclusive('\n') {
            let line_start = offset;
            offset += line.len();
            let trimmed = line.trim();
            let was_first = std::mem::take(&mut first_line);
            if in_comment {
                if trimmed.contains("-->") {
                    in_comment = false;
                }
                continue;
            }
            if trimmed.starts_with("<!--") {
                if !trimmed.contains("-->") {
                    in_comment = true;
                }
                continue;
            }
            // The document's title: its first line when that is not a
            // heading of any kind — an act's name, a meeting's
            // identity — as a level-0 unit every position path opens
            // with, so a chunk always knows which document it reads
            // (the "which act's supplementary provisions" gap).
            if was_first && heading_of(line).is_none() && looks_like_title(trimmed) {
                units.push(Unit {
                    unit: 0,
                    level: 0,
                    heading: one_line(trimmed),
                    paragraph_first: span.index,
                    paragraph_last: span.index,
                    // Never quoted as a reference (§3.1: the title is
                    // on every path), so its opening is moot.
                    body_start: span.end as usize,
                });
                continue;
            }
            if let Some(caption) = trimmed
                .strip_prefix('（')
                .and_then(|rest| rest.strip_suffix('）'))
                .filter(|inner| !inner.is_empty() && inner.chars().count() <= 40)
            {
                pending_caption = Some(caption.to_string());
                continue;
            }
            let Some((level, mut heading)) = heading_of(line) else {
                pending_caption = None;
                continue;
            };
            if heading.starts_with('第')
                && heading.contains('条')
                && let Some(caption) = pending_caption.take()
            {
                heading = format!("{heading}（{caption}）");
            }
            pending_caption = None;
            if let Some(last) = units.last_mut() {
                // A heading that opens its paragraph ends the previous
                // unit at the paragraph before; one further down a
                // paragraph shares that paragraph with it.
                last.paragraph_last = if line_start == start {
                    span.index.saturating_sub(1)
                } else {
                    span.index
                };
            }
            units.push(Unit {
                unit: units.len(),
                level,
                heading,
                paragraph_first: span.index,
                paragraph_last: span.index,
                body_start: line_start + line.len(),
            });
        }
    }
    if let (Some(last), Some(final_span)) = (units.last_mut(), spans.last()) {
        last.paragraph_last = final_span.index;
    }
    units
}

/// The paragraphs at which the outermost level present in the
/// document opens a unit — where `plan` prefers to end a chunk (ADR
/// 0033 §3.4). Empty when there is no structure.
pub(super) fn preferred_breaks(units: &[Unit]) -> HashSet<u32> {
    // The title unit (level 0) opens the document and is not a
    // boundary; the outermost real level is what chapters are.
    let Some(outermost) = units
        .iter()
        .map(|unit| unit.level)
        .filter(|level| *level > 0)
        .min()
    else {
        return HashSet::new();
    };
    units
        .iter()
        .filter(|unit| unit.level == outermost)
        .map(|unit| unit.paragraph_first)
        .collect()
}

/// The heading path in force at `paragraph` (ADR 0033 §3.1 position):
/// the units whose ranges hold it, outermost first, resolved as a
/// stack — a heading at level L closes every unit at level ≥ L.
pub(super) fn position(units: &[Unit], paragraph: u32) -> Vec<&Unit> {
    let mut stack: Vec<&Unit> = Vec::new();
    for unit in units {
        if unit.paragraph_first > paragraph {
            break;
        }
        while stack.last().is_some_and(|open| open.level >= unit.level) {
            stack.pop();
        }
        stack.push(unit);
    }
    stack
}

/// The units a chunk refers to by name and does not itself hold (ADR
/// 0033 §3.1 references), in first-mention order, capped: statute
/// articles by number (`第三条`, `第三条の二`), `前条` (the article before
/// the chunk's own), sections by `§N`/`第N章`/`第N節`, and any other
/// heading quoted verbatim when it is long enough not to be a common
/// word. Units already on the position path are not references.
pub(super) fn references<'a>(
    units: &'a [Unit],
    chunk_text: &str,
    chunk_first: u32,
    chunk_last: u32,
    on_path: &[&Unit],
    cap: usize,
) -> Vec<&'a Unit> {
    let in_chunk =
        |unit: &Unit| unit.paragraph_first <= chunk_last && unit.paragraph_last >= chunk_first;
    let mut found: Vec<(usize, &Unit)> = Vec::new();
    let mut note = |at: usize, unit: &'a Unit| {
        if in_chunk(unit) || on_path.iter().any(|open| open.unit == unit.unit) {
            return;
        }
        if found.iter().any(|(_, seen)| seen.unit == unit.unit) {
            return;
        }
        found.push((at, unit));
    };
    // One candidate per key: a statute's supplementary provisions
    // repeat the act's own article numbers, so a mentioned `第一条`
    // resolves to the nearest such unit at or before the chunk's
    // start (statutes refer backward), or the first one after when
    // none precedes — and to nothing when that unit is the chunk's own.
    let mut by_key: Vec<(String, &Unit)> = Vec::new();
    for unit in units {
        // Speaker and section labels of minutes are positions, never
        // references; the title is on every path already. An unnumbered
        // heading under 4 chars is too common a word to be a quoted
        // reference (a numbered one — `第二条`, `§ 1 x`, `1.2 x` — is
        // always at least three).
        // (The level-0 title needs no clause of its own: it heads
        // every position path, and `note` drops path units.)
        if unit.heading.starts_with('◆') || unit.heading.starts_with('○') {
            continue;
        }
        let key = reference_key(&unit.heading);
        let is_named = unit.heading.starts_with('第')
            || unit.heading.starts_with('§')
            || unit
                .heading
                .bytes()
                .next()
                .is_some_and(|b| b.is_ascii_digit());
        if !is_named && key.chars().count() < 4 {
            continue;
        }
        match by_key.iter_mut().find(|(seen, _)| *seen == key) {
            Some((_, chosen)) => {
                let before = |candidate: &Unit| candidate.paragraph_first <= chunk_first;
                if (before(unit) && unit.paragraph_first >= chosen.paragraph_first)
                    || (!before(chosen) && unit.paragraph_first < chosen.paragraph_first)
                {
                    *chosen = unit;
                }
            }
            None => by_key.push((key, unit)),
        }
    }
    for (key, unit) in &by_key {
        if let Some(at) = chunk_text.find(key.as_str()) {
            note(at, unit);
        }
    }
    if let Some(at) = chunk_text.find("前条") {
        let current = position(units, chunk_first)
            .into_iter()
            .rev()
            .find(|unit| unit.heading.starts_with('第') && unit.heading.contains('条'));
        if let Some(current) = current
            && let Some(previous) = units[..current.unit]
                .iter()
                .rev()
                .find(|unit| unit.heading.starts_with('第') && unit.heading.contains('条'))
        {
            note(at, previous);
        }
    }
    found.sort_by_key(|(at, unit)| (*at, unit.unit));
    found.into_iter().map(|(_, unit)| unit).take(cap).collect()
}

/// The part of a heading a chunk would quote to refer to it: a
/// statute article's number without its 見出し (`第三条` of
/// `第三条（定義）`), a chapter's `第N章`, a Markdown/numbered heading's
/// text as written, a minutes speaker as written.
fn reference_key(heading: &str) -> String {
    if heading.starts_with('第') {
        return heading
            .split(['（', '　'])
            .next()
            .unwrap_or(heading)
            .to_string();
    }
    heading.to_string()
}

/// What one chunk's block was built from — the `chunk_context` trace
/// record's contribution list (ADR 0033 §3.6.4), so #783 can credit a
/// gained fact to a kind.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(super) struct ContextBlock {
    /// The block text as placed in the user turn: one preamble
    /// section, no blank line inside (so `user_message_document`'s
    /// first-blank-line rule still finds the chunk).
    #[serde(skip)]
    pub(super) text: String,
    pub(super) sha256: String,
    pub(super) bytes: usize,
    /// Units on the position path, outermost first.
    pub(super) position: Vec<usize>,
    /// Units quoted as references, in mention order.
    pub(super) references: Vec<usize>,
    /// The preceding paragraphs carried as overlap, inclusive, or
    /// absent when the chunk opens the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) overlap_paragraphs: Option<(u32, u32)>,
}

/// The block's byte cap: a quarter of the chunk cap, never under 512.
pub(super) fn block_cap(chunk_cap: usize) -> usize {
    (chunk_cap / 4).max(512)
}

/// How many references one block carries at most.
const REFERENCE_CAP: usize = 4;
/// How much of a referenced unit's opening is quoted.
const REFERENCE_OPENING_BYTES: usize = 240;

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `<!-- … -->` spans removed — a bilingual Markdown source keeps the
/// original text in comments, which is not what the reader reads.
pub(super) fn strip_html_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find("<!--") {
        out.push_str(&rest[..open]);
        match rest[open..].find("-->") {
            Some(close) => rest = &rest[open + close + 3..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Whether a first line reads as a document title: short, not
/// markup, not a caption, and holding at least one letter.
fn looks_like_title(line: &str) -> bool {
    (4..=120).contains(&line.len())
        && !line.starts_with(['<', '[', '|', '-', '*', '`', '{', '（', '(', '#', '>'])
        && line.chars().any(char::is_alphabetic)
}

pub(super) fn truncate_at_char(text: &str, cap: usize) -> &str {
    if text.len() <= cap {
        return text;
    }
    // 0 is always a boundary, so this terminates without a guard.
    let mut end = cap;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// ADR 0033 §3.6: one chunk's block — position, references, then
/// overlap, each added only while the cap holds, each truncated at
/// an entry or paragraph boundary. `None` when nothing applies (a
/// structureless document's first chunk).
pub(super) fn render_block(
    text: &str,
    spans: &[crate::paragraph::ParagraphSpan],
    units: &[Unit],
    chunk_first: u32,
    chunk_last: u32,
    chunk_text: &str,
    cap: usize,
) -> Option<ContextBlock> {
    let path = position(units, chunk_first);
    let refs = references(
        units,
        chunk_text,
        chunk_first,
        chunk_last,
        &path,
        REFERENCE_CAP,
    );
    let mut lines: Vec<String> = Vec::new();
    let mut used = 0usize;
    let mut position_ids = Vec::new();
    if !path.is_empty() {
        let line = format!(
            "Position: {}",
            path.iter()
                .map(|unit| one_line(&unit.heading))
                .collect::<Vec<_>>()
                .join(" › ")
        );
        used += line.len() + 1;
        lines.push(line);
        position_ids = path.iter().map(|unit| unit.unit).collect();
    }
    // Every line's exact cost — prefix, separators, and the newline
    // before it — is charged against the cap, so the block never
    // exceeds it (ADR 0033 §3.6.3; the docs promise a quarter of
    // `--chunk-bytes`, not roughly that).
    const REFERENCES_PREFIX: &str = "References: ";
    const REFERENCES_SEPARATOR: &str = " | ";
    const PRECEDING_PREFIX: &str = "Preceding text: ";
    let mut reference_ids = Vec::new();
    let mut reference_entries = Vec::new();
    let mut references_len = 0usize;
    for unit in &refs {
        let end = spans
            .get(unit.paragraph_last as usize)
            .map(|span| span.end as usize)
            .unwrap_or(text.len())
            .max(unit.body_start);
        let opening = one_line(truncate_at_char(
            &strip_html_comments(&text[unit.body_start..end]),
            REFERENCE_OPENING_BYTES,
        ));
        let entry = if opening.is_empty() {
            one_line(&unit.heading)
        } else {
            format!("{} — {opening}", one_line(&unit.heading))
        };
        // The line as it would stand with this entry added, newline
        // included, must still fit.
        let line_len = REFERENCES_PREFIX.len()
            + references_len
            + if reference_entries.is_empty() {
                0
            } else {
                REFERENCES_SEPARATOR.len()
            }
            + entry.len();
        if used + line_len + 1 > cap {
            break;
        }
        references_len = line_len - REFERENCES_PREFIX.len();
        reference_entries.push(entry);
        reference_ids.push(unit.unit);
    }
    if !reference_entries.is_empty() {
        let line = format!(
            "{REFERENCES_PREFIX}{}",
            reference_entries.join(REFERENCES_SEPARATOR)
        );
        used += line.len() + 1;
        lines.push(line);
    }
    let mut overlap = None;
    {
        // What the preceding text may occupy after its own prefix
        // and newline. Under a paragraph's worth of room (a tail
        // shorter than the ellipsis and a few words says nothing)
        // the kind is skipped outright, never recorded as carrying
        // nothing. A chunk opening the document has no paragraph
        // before it and the loop never runs.
        const MIN_OVERLAP_BYTES: usize = 24;
        let budget = cap.saturating_sub(used + PRECEDING_PREFIX.len() + 1);
        let mut first = chunk_first;
        let mut carried = 0usize;
        let mut pieces: Vec<String> = Vec::new();
        while first > 0 && budget >= MIN_OVERLAP_BYTES {
            let span = spans[(first - 1) as usize];
            let paragraph = one_line(&text[span.start as usize..span.end as usize]);
            if pieces.is_empty() && paragraph.len() > budget {
                // Even the nearest paragraph is too long: keep its
                // tail, the ellipsis charged against the same budget.
                let room = budget - "…".len();
                let mut cut = paragraph.len() - room;
                while cut < paragraph.len() && !paragraph.is_char_boundary(cut) {
                    cut += 1;
                }
                pieces.push(format!("…{}", &paragraph[cut..]));
                first -= 1;
                break;
            }
            let separator = usize::from(!pieces.is_empty());
            if carried + separator + paragraph.len() > budget {
                break;
            }
            carried += separator + paragraph.len();
            pieces.push(paragraph);
            first -= 1;
        }
        if !pieces.is_empty() {
            pieces.reverse();
            lines.push(format!("{PRECEDING_PREFIX}{}", pieces.join(" ")));
            overlap = Some((first, chunk_first - 1));
        }
    }
    if lines.is_empty() {
        return None;
    }
    let body = lines.join("\n");
    debug_assert!(!body.contains("\n\n"));
    debug_assert!(body.len() <= cap, "{} > {cap}", body.len());
    Some(ContextBlock {
        sha256: sha256_hex(body.as_bytes()),
        bytes: body.len(),
        position: position_ids,
        references: reference_ids,
        overlap_paragraphs: overlap,
        text: body,
    })
}

/// How the block's preamble line opens — what
/// `user_message_occurrence_text` recognizes it by.
pub(super) const BLOCK_PREAMBLE_OPENING: &str = "Chunk context (";

/// The block's own preamble line — what tells the model the block is
/// not the part to extract from (ADR 0033 §3.6, rules 1 and 2 in the
/// model's own terms).
pub(super) fn block_preamble(index: usize, total: usize) -> String {
    let part = if total > 1 {
        format!("part {} of {total}", index + 1)
    } else {
        "the document".to_string()
    };
    format!(
        "{BLOCK_PREAMBLE_OPENING}this document's own text and structure, for reading {part} — \
         extract facts from {part} only; nothing below carries a [N] paragraph number, so \
         a fact stated only here is not extracted):"
    )
}

/// ADR 0033 §3.4's `structure` trace record: one per unit.
#[derive(serde::Serialize)]
pub(super) struct TraceStructure<'a> {
    pub(super) kind: &'static str,
    #[serde(flatten)]
    pub(super) unit: &'a Unit,
}

/// ADR 0033 §3.6.4's `chunk_context` trace record: one per chunk that
/// got a block.
#[derive(serde::Serialize)]
pub(super) struct TraceChunkContext<'a> {
    pub(super) kind: &'static str,
    pub(super) chunk_index: usize,
    #[serde(flatten)]
    pub(super) block: &'a ContextBlock,
}
