use super::{FNV_OFFSET, FNV_PRIME};

/// Where the term walkers deliver their stream. The index build wants
/// bare hashes; the search-explain path wants the spelling each hash
/// was computed from next to it. One walker feeds both through this
/// sink, so the two views cannot disagree about what a term is —
/// spelling events cost nothing when the sink ignores them.
trait TermSink {
    /// One character of the ASCII word being hashed, in hash order,
    /// case-folded exactly as the hash saw it. A `word` call closes
    /// the sequence.
    fn word_char(&mut self, ch: char);
    /// The word whose characters were just streamed.
    fn word(&mut self, hash: u64);
    /// An adjacent character pair inside a non-ASCII run.
    fn pair(&mut self, hash: u64, first: char, second: char);
    /// A non-ASCII run of exactly one character.
    fn lone(&mut self, hash: u64, ch: char);
}

/// The index build's sink: hashes only.
struct HashSink(Vec<u64>);

impl TermSink for HashSink {
    fn word_char(&mut self, _: char) {}

    fn word(&mut self, hash: u64) {
        self.0.push(hash);
    }

    fn pair(&mut self, hash: u64, _: char, _: char) {
        self.0.push(hash);
    }

    fn lone(&mut self, hash: u64, ch: char) {
        let _ = ch;
        self.0.push(hash);
    }
}

/// The explain path's sink: every hash next to its spelling.
struct SpellingSink {
    word: String,
    terms: Vec<(String, u64)>,
}

impl TermSink for SpellingSink {
    fn word_char(&mut self, ch: char) {
        self.word.push(ch);
    }

    fn word(&mut self, hash: u64) {
        self.terms.push((std::mem::take(&mut self.word), hash));
    }

    fn pair(&mut self, hash: u64, first: char, second: char) {
        let mut spelling = String::with_capacity(first.len_utf8() + second.len_utf8());
        spelling.push(first);
        spelling.push(second);
        self.terms.push((spelling, hash));
    }

    fn lone(&mut self, hash: u64, ch: char) {
        self.terms.push((ch.to_string(), hash));
    }
}

/// The terms of one passage or query: [`text_terms`] over the
/// normalized text, plus a word term per piece of every camelCase run.
/// One function serves both sides of the search, so they cannot
/// disagree about what a term is.
pub(crate) fn passage_terms(raw: &str) -> Vec<u64> {
    let mut sink = HashSink(Vec::new());
    walk_passage_terms(raw, &mut sink);
    sink.0
}

/// [`passage_terms`] with each hash next to the spelling it was
/// computed from — a whole lowercased word, a camelCase piece, the two
/// characters of an adjacent pair, or a lone character. Same walker,
/// same order; only the sink differs, so the rendered stream IS the
/// hashed stream. Recomputed per explain call — the index keeps hashes
/// only, and grows no reverse map for a diagnostic path.
pub(crate) fn spelled_passage_terms(raw: &str) -> Vec<(String, u64)> {
    let mut sink = SpellingSink {
        word: String::new(),
        terms: Vec::new(),
    };
    walk_passage_terms(raw, &mut sink);
    sink.terms
}

/// The walker under [`passage_terms`]. The camelCase split reads an
/// NFKC-folded but NOT lowercased view of the input: lowercasing would
/// erase the very case boundaries that let `state` reach `AppState`,
/// while the width fold keeps a full-width `Ａ` — which the normalized
/// whole-word term already folds to ASCII — in the same run as its
/// ASCII neighbors instead of breaking it (so `ＡpplePie` yields the
/// `apple` piece, matching a plain `apple` cue).
fn walk_passage_terms(raw: &str, sink: &mut impl TermSink) {
    use unicode_normalization::UnicodeNormalization;
    walk_text_terms(&taguru::context::normalize_entry(raw), sink);
    let mut run: Vec<char> = Vec::new();
    for ch in raw.nfkc() {
        if ch.is_ascii_alphanumeric() {
            run.push(ch);
        } else {
            camel_pieces(&run, sink);
            run.clear();
        }
    }
    camel_pieces(&run, sink);
}

/// Emits one lowercased word term per piece of an ASCII run that
/// splits at case boundaries: `aB` → `a|B`, digits stick to their
/// piece (`U64Max` → `u64|max`), and an acronym ends before its last
/// capital (`HTTPServer` → `http|server`). A run with no boundary
/// emits nothing — its whole-word term is already in the stream.
/// Pieces hash exactly like [`text_terms`] words, so a piece matches
/// wherever the same word occurs standalone.
fn camel_pieces(run: &[char], sink: &mut impl TermSink) {
    let mut starts = vec![0];
    for at in 1..run.len() {
        if !run[at].is_ascii_uppercase() {
            continue;
        }
        let after_lower = run[at - 1].is_ascii_lowercase() || run[at - 1].is_ascii_digit();
        let ends_acronym = run[at - 1].is_ascii_uppercase()
            && run.get(at + 1).is_some_and(|ch| ch.is_ascii_lowercase());
        if after_lower || ends_acronym {
            starts.push(at);
        }
    }
    if starts.len() < 2 {
        return;
    }
    starts.push(run.len());
    for window in starts.windows(2) {
        let mut word = FNV_OFFSET;
        for ch in &run[window[0]..window[1]] {
            let ch = ch.to_ascii_lowercase();
            sink.word_char(ch);
            word ^= ch as u64;
            word = word.wrapping_mul(FNV_PRIME);
        }
        sink.word(word | 1 << 63);
    }
}

/// [`walk_text_terms`] collected as bare keys — the tokenization tests'
/// entrance; production goes through [`passage_terms`], which layers
/// the camelCase pieces on top of the same walker.
#[cfg(test)]
pub(super) fn text_terms(text: &str) -> Vec<u64> {
    let mut sink = HashSink(Vec::new());
    walk_text_terms(text, &mut sink);
    sink.0
}

/// The word/bigram layer under [`passage_terms`]. ASCII-alphanumeric
/// runs count as whole words; everything else contributes adjacent
/// character pairs within its run (a run of one contributes the lone
/// character). Space-delimited languages need word terms — character
/// pairs occur in every English document alike, which flattens IDF to
/// nothing — while undelimited Japanese needs the bigrams. Runs break
/// at spaces and punctuation, and a script switch breaks the run too,
/// so terms never straddle "第10篇"-style boundaries.
fn walk_text_terms(text: &str, sink: &mut impl TermSink) {
    let mut word = FNV_OFFSET; // running FNV-1a over the current ASCII word
    let mut in_word = false;
    let mut run: Option<char> = None; // previous char of the current non-ASCII run
    let mut run_len = 0usize;
    fn flush_run(sink: &mut impl TermSink, run: &mut Option<char>, run_len: &mut usize) {
        if let (Some(last), 1) = (*run, *run_len) {
            sink.lone(last as u64, last); // below the pair space: pairs always have bits 32+
        }
        *run = None;
        *run_len = 0;
    }
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            flush_run(sink, &mut run, &mut run_len);
            sink.word_char(ch);
            word ^= ch as u64;
            word = word.wrapping_mul(FNV_PRIME);
            in_word = true;
        } else {
            if in_word {
                sink.word(word | 1 << 63); // disjoint from pair keys (chars < 2^21)
                word = FNV_OFFSET;
                in_word = false;
            }
            if ch.is_alphanumeric() {
                if let Some(prev) = run {
                    sink.pair(((prev as u64) << 32) | ch as u64, prev, ch);
                }
                run = Some(ch);
                run_len += 1;
            } else {
                flush_run(sink, &mut run, &mut run_len);
            }
        }
    }
    if in_word {
        sink.word(word | 1 << 63);
    }
    flush_run(sink, &mut run, &mut run_len);
}
