//! Paragraph splitting: THE one function that decides where a passage's
//! paragraphs begin and end. The BM25 lane indexes paragraphs, the
//! vector lane embeds them, and the passage store carries their spans —
//! all three import this and nothing else, so they can never disagree
//! about what a paragraph is.
//!
//! The split is mechanical and deterministic: blank-line runs separate
//! paragraphs, nothing is normalized, and a span is a byte range into
//! the ORIGINAL text — the store's byte-exact lookup contract never
//! depends on how this function evolves, because reassembly never
//! happens through it. Spans are not persisted anywhere; they are
//! recomputed when a passage becomes resident (the same
//! rebuild-on-load posture as the entry index).

use crate::embedding::fnv1a;

/// One paragraph's place in its source text: `[start, end)` byte
/// offsets (always on `\n` boundaries, so always valid UTF-8 cuts) and
/// the FNV-1a hash of exactly those bytes — the change detector the
/// index and vector lanes key their skip-if-unchanged logic on, the
/// same way gloss refresh does. `index` is the paragraph's position in
/// THIS split; re-storing a source renumbers from zero (retraction is
/// source-wholesale, so nothing outlives that).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParagraphSpan {
    pub(crate) index: u32,
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) hash: u64,
}

/// Splits `text` into paragraph spans. A line is blank when it is
/// empty or all Unicode whitespace (which covers `\r`, so CRLF blank
/// lines count); one or more consecutive blank lines form a separator
/// and belong to no paragraph. A paragraph's span runs from its first
/// line's first byte to its last line's last content byte — the
/// terminating newline (and a final `\r` before it) stays out, while
/// interior line breaks stay in. No blank line at all means one
/// paragraph; nothing but blank lines means none.
pub(crate) fn split(text: &str) -> Vec<ParagraphSpan> {
    // Offsets are u32 in the span (and in every sidecar that will key
    // off them); every entrance caps a passage far below this already.
    assert!(
        text.len() <= u32::MAX as usize,
        "passage text exceeds u32 offsets"
    );
    let mut spans = Vec::new();
    let mut run: Option<(u32, u32)> = None;
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let line_start = offset;
        offset += line.len();
        let content = line.strip_suffix('\n').unwrap_or(line);
        let content = content.strip_suffix('\r').unwrap_or(content);
        if content.chars().all(char::is_whitespace) {
            flush(&mut spans, &mut run, text);
        } else {
            let end = (line_start + content.len()) as u32;
            match &mut run {
                Some((_, run_end)) => *run_end = end,
                None => run = Some((line_start as u32, end)),
            }
        }
    }
    flush(&mut spans, &mut run, text);
    spans
}

fn flush(spans: &mut Vec<ParagraphSpan>, run: &mut Option<(u32, u32)>, text: &str) {
    if let Some((start, end)) = run.take() {
        spans.push(ParagraphSpan {
            index: spans.len() as u32,
            start,
            end,
            hash: fnv1a(&text[start as usize..end as usize]),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(text: &str) -> Vec<&str> {
        split(text)
            .iter()
            .map(|span| &text[span.start as usize..span.end as usize])
            .collect()
    }

    #[test]
    fn split_paragraphs_breaks_on_blank_lines_and_drops_leading_trailing_blanks() {
        let text = "\n最初の段落。\n二行目も同じ段落。\n\n \t \n次の段落。\n\n";
        assert_eq!(
            texts(text),
            vec!["最初の段落。\n二行目も同じ段落。", "次の段落。"],
            "whitespace-only lines separate; edge blanks vanish"
        );
        let spans = split(text);
        assert_eq!(spans[0].index, 0);
        assert_eq!(spans[1].index, 1);
    }

    #[test]
    fn split_paragraphs_treats_crlf_blank_lines_the_same_as_lf() {
        let crlf = "a\r\nb\r\n\r\nc\r\n";
        assert_eq!(
            texts(crlf),
            vec!["a\r\nb", "c"],
            "interior CRLFs stay inside the span; the final one stays out"
        );
    }

    #[test]
    fn split_paragraphs_does_not_treat_information_separators_as_blank() {
        // U+001C-U+001F (FILE/GROUP/RECORD/UNIT SEPARATOR) are whitespace
        // under Python's str.isspace() but not under Unicode's White_Space
        // property, which char::is_whitespace() follows — a line made of
        // only one must stay content here, or the Python LangChain SDK's
        // mirror of this function (sdk/python-langchain, which once used
        // str.isspace()) would split a paragraph the server keeps whole.
        let text = "最初の段落。\n\u{1e}\n続き。\n\n次の段落。";
        assert_eq!(
            texts(text),
            vec!["最初の段落。\n\u{1e}\n続き。", "次の段落。"],
            "a lone information-separator control does not blank its line"
        );
    }

    #[test]
    fn split_paragraphs_of_an_empty_or_all_blank_document_is_empty() {
        assert!(split("").is_empty());
        assert!(split("\n\n\n").is_empty());
        assert!(
            split("  \n\u{3000}\n").is_empty(),
            "ideographic space is blank"
        );
    }

    #[test]
    fn split_paragraphs_of_a_single_block_with_no_blank_line_is_one_paragraph() {
        let text = "一行だけ。";
        let spans = split(text);
        assert_eq!(spans.len(), 1);
        assert_eq!(texts(text), vec!["一行だけ。"]);
        // A trailing newline does not create a second paragraph, and
        // stays outside the span.
        assert_eq!(texts("一行だけ。\n"), vec!["一行だけ。"]);
    }

    #[test]
    fn split_paragraphs_hash_differs_when_and_only_when_the_paragraph_bytes_differ() {
        let a = split("同じ本文。\n\n違う本文。");
        let b = split("同じ本文。\n\nまた違う本文。");
        assert_eq!(a[0].hash, b[0].hash, "identical bytes, identical hash");
        assert_ne!(a[1].hash, b[1].hash);
        assert_ne!(a[0].hash, a[1].hash);
    }

    #[test]
    fn paragraph_spans_are_byte_offsets_into_the_original_text_never_a_copy() {
        // Multi-byte UTF-8 with CRLFs: the offsets must slice cleanly
        // (a wrong boundary would panic) and reproduce the exact bytes.
        let text = "青嶺酒造は1907年創業。\r\n杜氏は高瀬。\r\n\r\nSecond ¶ has ASCII too.";
        let spans = split(text);
        assert_eq!(spans.len(), 2);
        assert_eq!(
            &text[spans[0].start as usize..spans[0].end as usize],
            "青嶺酒造は1907年創業。\r\n杜氏は高瀬。"
        );
        assert_eq!(
            &text[spans[1].start as usize..spans[1].end as usize],
            "Second ¶ has ASCII too."
        );
    }
}
