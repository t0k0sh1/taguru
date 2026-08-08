//! Issue #496 S4 (ADR 0016): coverage verification — the recall-side
//! twin of ADR 0013's precision-side accounting. The mechanical pass
//! removes what the model asserted and the document never said; this
//! pass reports what the document said and the model never asserted.
//! A sentence dense enough to hold a candidate pair (two of ADR
//! 0014's deterministic terms) states something extractable; when no
//! accepted association lands at least two of its three parts in that
//! sentence, the sentence is flagged as uncovered — the systematic
//! recall ceiling (the 2026-08-08 bench's same-fact-dropped-every-run
//! failures) made visible, per document, with the sentence quoted.
//!
//! Report-first on purpose: the batch is never changed and nothing is
//! re-asked, so the check is free of LLM calls, fingerprint-neutral,
//! and equally applicable to a manifest-skipped document's
//! already-written batch. Re-extracting the flagged sentences is ADR
//! 0016 §4's staged upgrade, bought only when measured gap rates
//! justify the extra calls.

use super::*;

/// A sentence owes coverage only when it holds at least this many
/// distinct candidate terms — one name anchors nothing relational,
/// two is the smallest shape a triple can land on ("候補ペア").
pub(super) const COVERAGE_MIN_TERMS: usize = 2;

/// An association covers a sentence when at least this many of its
/// three parts (subject, label, object) occur in the sentence. Two of
/// three on purpose, not all three: the subject of a fact often lives
/// in an earlier sentence (a heading, a list's introducer) while the
/// sentence itself carries the label and object — demanding all three
/// would flag exactly the extractions the discipline's
/// make-implicit-membership-explicit rule produces.
pub(super) const COVERAGE_MIN_PARTS: usize = 2;

/// Byte cap on the sentence quote a gap line carries — enough to find
/// the sentence, never a second copy of the document on stderr.
pub(super) const GAP_QUOTE_MAX_BYTES: usize = 120;

/// One sentence that held a candidate pair and never became a triple.
pub(super) struct CoverageGap {
    /// The canonical paragraph index (the server's own split — the
    /// same numbering the model cites), so the gap is addressable.
    pub(super) paragraph: u32,
    pub(super) quote: String,
}

impl CoverageGap {
    /// The gap as one report line's payload, paragraph first — the
    /// same path-first shape ADR 0013's removal records use.
    pub(super) fn describe(&self) -> String {
        format!("[paragraph {}] {}", self.paragraph, self.quote)
    }
}

/// Every uncovered sentence in `text`, judged against the accepted
/// associations' `[subject, label, object]` triples — deterministic,
/// dictionary-free, no model in the loop. Reuses ADR 0014's segmenter
/// for what counts as a term and ADR 0013's occurrence machinery for
/// what counts as present, so all three controls agree on both
/// questions by construction.
pub(super) fn coverage_gaps(text: &str, triples: &[[&str; 3]]) -> Vec<CoverageGap> {
    let mut gaps = Vec::new();
    for span in crate::paragraph::split(text) {
        let content = &text[span.start as usize..span.end as usize];
        for sentence in sentences(content) {
            if candidate_terms(sentence).len() < COVERAGE_MIN_TERMS {
                continue;
            }
            let haystack = normalize_for_occurrence(sentence);
            let covered = triples.iter().any(|parts| {
                parts
                    .iter()
                    .filter(|part| name_occurs(&haystack, part))
                    .count()
                    >= COVERAGE_MIN_PARTS
            });
            if !covered {
                gaps.push(CoverageGap {
                    paragraph: span.index,
                    quote: quote_sentence(sentence),
                });
            }
        }
    }
    gaps
}

/// Deterministic sentence split within one paragraph: terminators
/// (。!?, full- and half-width) and line breaks end a sentence. An
/// ASCII period deliberately does not — it lives inside too many
/// identifiers (`file.rs`, `qwen2.5:14b`) to be a boundary, and the
/// cost of missing it is only a LARGER unit, which can only make
/// coverage easier to satisfy: the check stays precision-biased,
/// flagging less rather than fabricating gaps.
fn sentences(paragraph: &str) -> impl Iterator<Item = &str> {
    paragraph
        .split(['。', '!', '?', '\u{ff01}', '\u{ff1f}', '\n'])
        .map(str::trim)
        .filter(|sentence| !sentence.is_empty())
}

/// The gap line's sentence quote, capped at a char boundary — a
/// truncated quote still locates the sentence; the paragraph index
/// does the addressing.
fn quote_sentence(sentence: &str) -> String {
    if sentence.len() <= GAP_QUOTE_MAX_BYTES {
        return sentence.to_string();
    }
    format!(
        "{}…",
        &sentence[..floor_char_boundary(sentence, GAP_QUOTE_MAX_BYTES)]
    )
}
