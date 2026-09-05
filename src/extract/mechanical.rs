//! Issue #496 S1 (ADR 0013): the deterministic validation pass between
//! parsing and the corrective turn. Items the graph could never import
//! as answered — a required field missing or empty, an alias that maps
//! a spelling to itself, a subject/object that never appears in the
//! document text — are removed here, mechanically and with explicit
//! accounting, instead of spending LLM corrective turns that measurably
//! flail on exactly these shapes (the 2026-08-08 bench: five attempts,
//! zero corrections, 53–63 s per document). The corrective turn stays,
//! demoted to the last resort for what removal cannot judge: a present
//! but wrong-typed or out-of-range value is content the model can
//! actually fix, so it keeps the ADR 0001 §8 bucket-2 path.
//!
//! Never-silent-drop (ADR 0001 §8 bucket 3) is preserved as
//! *accounting*: every removal is named path-first in the batch
//! report, on stderr, and in the `--diagnostics-out` sidecar —
//! distinct from `--lossy`, which validates nothing and reports only
//! a count.

use super::*;

/// What the mechanical pass concluded about one parsed answer: the
/// output with removed items already gone, one path-addressed record
/// per removal, and the issues only a corrective turn can still
/// address. `issues` empty means the answer is accepted with zero
/// model calls spent on repair.
pub(super) struct MechanicalEvaluation {
    pub(super) output: ModelOutput,
    pub(super) removed: Vec<Removal>,
    pub(super) issues: Vec<String>,
}

/// One mechanically removed item (ADR 0013), as #786 / ADR 0024
/// record it: the path-addressed `reason` every report line and
/// diagnostics record has always carried, plus the item the model
/// actually wrote (`item`, verbatim JSON — the raw array element for
/// a Stage 1 removal, the parsed alias for a Stage 2 prune), so the
/// loss can be read in the original instead of reconstructed from a
/// message. `Display` is the pre-#786 string, byte for byte, so
/// stderr, the report line, and `removed_items` are unchanged.
#[derive(Clone, serde::Serialize, PartialEq)]
#[cfg_attr(test, derive(Debug))]
pub(super) struct Removal {
    pub(super) path: String,
    pub(super) reason: String,
    pub(super) item: serde_json::Value,
}

impl Removal {
    pub(super) fn new(
        path: impl Into<String>,
        reason: impl Into<String>,
        item: &impl serde::Serialize,
    ) -> Self {
        Self {
            path: path.into(),
            reason: reason.into(),
            // Every item here is a serde_json::Value or a derive-
            // Serialize model struct — never fails; Null rather than
            // a panic on the impossible path.
            item: serde_json::to_value(item).unwrap_or(serde_json::Value::Null),
        }
    }

    /// Which batch-item kind the path names — `associations[..]`,
    /// `aliases[..]`, or `questions[..]` (the last only from lossy
    /// parse drops; the mechanical pass never removes a question).
    pub(super) fn item_kind(&self) -> &'static str {
        if self.path.starts_with("associations[") {
            "association"
        } else if self.path.starts_with("questions[") {
            "question"
        } else {
            "alias"
        }
    }
}

impl std::fmt::Display for Removal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.reason)
    }
}

/// A checkpoint written before #786 recorded each removal as its
/// display string only; it still loads — as a `Removal` with the
/// string split back into path/reason and no item (`Null`) — so a
/// resumed document keeps its accounting.
impl<'de> Deserialize<'de> for Removal {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Either {
            Structured {
                path: String,
                reason: String,
                #[serde(default)]
                item: serde_json::Value,
            },
            Legacy(String),
        }
        Ok(match Either::deserialize(deserializer)? {
            Either::Structured { path, reason, item } => Self { path, reason, item },
            Either::Legacy(text) => {
                let (path, reason) = text.split_once(": ").unwrap_or(("", text.as_str()));
                Self {
                    path: path.to_string(),
                    reason: reason.to_string(),
                    item: serde_json::Value::Null,
                }
            }
        })
    }
}

/// The strict-mode replacement for [`interpret_model_output`]: the
/// same lenient walk and the same issue texts, but items whose
/// departure is mechanically judgeable are removed (recorded in
/// `removed`) instead of flagged, so only genuinely corrective issues
/// survive into `issues`. `document` is the document (chunk) text the
/// model was shown — callers strip `user_message`'s preamble first
/// (`user_message_document`), so occurrence never depends on the
/// source path. `--lossy` never calls this (its contract is
/// byte-for-byte the pre-#199 behavior).
pub(super) fn mechanical_interpret(
    value: &serde_json::Value,
    rules: &ItemRules,
    document: &str,
    vocabulary: &HashSet<String>,
) -> MechanicalEvaluation {
    let haystack = normalize_for_occurrence(document);
    let mut issues = Vec::new();
    let mut removed: Vec<Removal> = Vec::new();
    let empty_map = serde_json::Map::new();
    let obj = value.as_object().unwrap_or(&empty_map);

    let associations = match get_present(obj, "associations") {
        None => Vec::new(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                association_mechanically(
                    index,
                    item,
                    &haystack,
                    vocabulary,
                    &mut issues,
                    &mut removed,
                )
            })
            .collect(),
        Some(other) => {
            issues.push(format!(
                "associations: expected an array, got {}",
                describe_value(other)
            ));
            Vec::new()
        }
    };
    let aliases = match get_present(obj, "aliases") {
        None => Vec::new(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                alias_mechanically(index, item, &haystack, &mut issues, &mut removed)
            })
            .collect(),
        Some(other) => {
            issues.push(format!(
                "aliases: expected an array, got {}",
                describe_value(other)
            ));
            Vec::new()
        }
    };
    // Questions have no mechanical rule: a bad paragraph citation or a
    // malformed question is always the corrective path, unchanged.
    let questions = interpret_questions(obj, rules, &mut issues);

    MechanicalEvaluation {
        output: ModelOutput {
            associations,
            aliases,
            questions,
        },
        removed,
        issues,
    }
}

/// One association element: removal beats correction exactly when the
/// item cannot carry a fact as answered. A non-object element and a
/// missing/empty required field assert nothing (merge would drop them
/// wholesale); a complete, issue-free item whose subject or object
/// never appears in the document is a fabrication the corrective turn
/// cannot un-invent. Everything else — wrong-typed or oversized
/// fields, weight/paragraph business rules — keeps its Stage 1 issue
/// and the corrective path.
fn association_mechanically(
    index: usize,
    item: &serde_json::Value,
    haystack: &str,
    vocabulary: &HashSet<String>,
    issues: &mut Vec<String>,
    removed: &mut Vec<Removal>,
) -> Option<ModelAssociation> {
    let path = format!("associations[{index}]");
    let Some(obj) = item.as_object() else {
        removed.push(Removal::new(
            &path,
            format!("expected an object, got {}", describe_value(item)),
            item,
        ));
        return None;
    };
    let absent: Vec<String> = ["subject", "label", "object"]
        .iter()
        .filter_map(|key| field_absence(obj, key).map(|kind| format!("{key} {kind}")))
        .collect();
    if !absent.is_empty() {
        removed.push(Removal::new(&path, absent.join(", "), item));
        return None;
    }
    let mut item_issues = Vec::new();
    let parsed = interpret_association_item(index, item, &mut item_issues)?;
    if !item_issues.is_empty() {
        issues.append(&mut item_issues);
        return Some(parsed);
    }
    // A single character (a bare Japanese particle, most often) reads
    // fine to `interpret_required_string` but functions as no relation
    // at all: unusable for query/paths/schema, and — because labels
    // accumulate into the run's reuse vocabulary (#759) — a survivor
    // gets suggested back to every later chunk, snowballing into every
    // association sharing one meaningless label. Same anchor-nothing
    // judgment as a single-character candidate name (`candidates.rs`).
    let label = parsed
        .label
        .as_deref()
        .expect("no absence and no issue means the field parsed");
    if label.chars().count() < 2 {
        removed.push(Removal::new(
            &path,
            format!(
                "label {} is a single character — too generic to be a relation",
                quote_for_issue(label)
            ),
            item,
        ));
        return None;
    }
    // ADR 0039: a label or name written with an ideograph from another
    // language's repertoire (`适用法律` for a document that says
    // `適用法律`) is a second spelling of one term that `query`,
    // `resolve`, and the reuse vocabulary (#778) would all treat as a
    // different label — removed before it can be offered back to a
    // later document. Every offending field of the item is named in the
    // one record, as the occurrence check below names both positions.
    // Subject and object keep ADR 0015's allowlist exactly as the
    // occurrence check applies it; a label has no allowlist.
    let subject = parsed
        .subject
        .as_deref()
        .expect("no absence and no issue means the field parsed");
    let object = parsed
        .object
        .as_deref()
        .expect("no absence and no issue means the field parsed");
    let allowed = |name: &str| vocabulary.contains(&normalize_for_occurrence(name));
    let foreign: Vec<(&str, &str, Vec<char>)> = [
        ("label", label, false),
        ("subject", subject, allowed(subject)),
        ("object", object, allowed(object)),
    ]
    .into_iter()
    .filter(|(_, _, allowlisted)| !allowlisted)
    .filter_map(|(field, name, _)| {
        let chars = foreign_ideographs(name, haystack);
        (!chars.is_empty()).then_some((field, name, chars))
    })
    .collect();
    if !foreign.is_empty() {
        removed.push(Removal::new(&path, foreign_script_reason(&foreign), item));
        return None;
    }
    // Both positions are checked before anything is recorded, so an
    // item fabricating subject AND object names them together — the
    // one removal record is the complete diagnosis, not whichever
    // field happened to be checked first (mirroring `absent` above).
    let fabricated: Vec<String> = [("subject", &parsed.subject), ("object", &parsed.object)]
        .into_iter()
        .filter_map(|(field, name)| {
            let name = name
                .as_deref()
                .expect("no absence and no issue means the field parsed");
            // ADR 0015: a spelling the target context already uses is
            // not a fabrication, however the document spells the entity
            // — the vocabulary allowlist is consulted before removal.
            (!name_occurs(haystack, name) && !vocabulary.contains(&normalize_for_occurrence(name)))
                .then(|| {
                    format!(
                        "{field} {} does not appear in the document text",
                        quote_for_issue(name)
                    )
                })
        })
        .collect();
    if !fabricated.is_empty() {
        removed.push(Removal::new(&path, fabricated.join(", "), item));
        return None;
    }
    Some(parsed)
}

/// One alias element: a non-object element, a missing/empty required
/// field, and a self-alias (a mapping that maps nothing) are removed;
/// a present-but-invalid `kind` or an oversized field keeps its issue
/// and the corrective path. The occurrence check deliberately does not
/// apply here — an alias's whole purpose is to record a spelling, and
/// #496 S1 names subject/object only.
fn alias_mechanically(
    index: usize,
    item: &serde_json::Value,
    haystack: &str,
    issues: &mut Vec<String>,
    removed: &mut Vec<Removal>,
) -> Option<ModelAlias> {
    let path = format!("aliases[{index}]");
    let Some(obj) = item.as_object() else {
        removed.push(Removal::new(
            &path,
            format!("expected an object, got {}", describe_value(item)),
            item,
        ));
        return None;
    };
    let absent: Vec<String> = ["alias", "canonical", "kind"]
        .iter()
        .filter_map(|key| field_absence(obj, key).map(|kind| format!("{key} {kind}")))
        .collect();
    if !absent.is_empty() {
        removed.push(Removal::new(&path, absent.join(", "), item));
        return None;
    }
    let mut item_issues = Vec::new();
    let parsed = interpret_alias_item(index, item, &mut item_issues)?;
    if let (Some(spelling), Some(canonical)) = (&parsed.alias, &parsed.canonical)
        && spelling == canonical
    {
        // interpret_alias_item's own self-alias issue is in item_issues;
        // it dies with the item rather than becoming a corrective ask.
        removed.push(Removal::new(&path, "alias equals its canonical", item));
        return None;
    }
    // ADR 0039, the alias-spelling half: a spelling in another
    // language's repertoire records a form the document never uses.
    // The canonical is judged where it is asserted — the association
    // it must match.
    if let Some(spelling) = parsed.alias.as_deref() {
        let chars = foreign_ideographs(spelling, haystack);
        if !chars.is_empty() {
            removed.push(Removal::new(
                &path,
                foreign_script_reason(&[("alias", spelling, chars)]),
                item,
            ));
            return None;
        }
    }
    if !item_issues.is_empty() {
        issues.append(&mut item_issues);
    }
    Some(parsed)
}

/// ADR 0039 §3.1–3.2: the ideographs of `name` that are outside JIS X
/// 0208's repertoire AND do not occur in the document — in first-
/// appearance order, each once. `haystack` is the occurrence check's
/// normal form, which leaves ideographs untouched, so containment is
/// by character. Empty for a name in the document's own script, for a
/// Chinese or Korean document (every character it uses is in the
/// text), and for anything that is not an ideograph at all.
pub(super) fn foreign_ideographs(name: &str, haystack: &str) -> Vec<char> {
    let mut found: Vec<char> = Vec::new();
    for c in name.chars() {
        if jis_x0208::is_ideograph(c)
            && !jis_x0208::in_jis_x0208(c)
            && !haystack.contains(c)
            && !found.contains(&c)
        {
            found.push(c);
        }
    }
    found
}

/// The one removal reason for every field of an item ADR 0039 flags:
/// `label "适用法律" uses 适, object "处分" uses 处 — ideographs outside
/// the document's script` (singular when one character in all).
fn foreign_script_reason(fields: &[(&str, &str, Vec<char>)]) -> String {
    let parts: Vec<String> = fields
        .iter()
        .map(|(field, name, chars)| {
            let listed: Vec<String> = chars.iter().map(char::to_string).collect();
            format!(
                "{field} {} uses {}",
                quote_for_issue(name),
                listed.join(", ")
            )
        })
        .collect();
    let total: usize = fields.iter().map(|(_, _, chars)| chars.len()).sum();
    let noun = if total == 1 {
        "an ideograph"
    } else {
        "ideographs"
    };
    format!(
        "{} — {noun} outside the document's script",
        parts.join(", ")
    )
}

/// A required field that is mechanically absent: missing (or null —
/// [`get_present`]'s ruling) or a string that trims to nothing. A
/// present wrong-typed value is NOT absent — it is content the
/// corrective turn can fix, so it stays an issue.
fn field_absence(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'static str> {
    match get_present(obj, key) {
        None => Some("missing"),
        Some(serde_json::Value::String(text)) if text.trim().is_empty() => Some("empty"),
        Some(_) => None,
    }
}

/// The occurrence check's normal form: every Unicode whitespace
/// character dropped, everything lowercased. Whitespace-blind because
/// spacing is exactly what an extractor legitimately normalizes
/// (`"CI テストランナー"` for a document that says `"CI の テストランナー"`),
/// and language-independent by construction — no tokenizer, no
/// dictionary (that is S2's territory).
pub(super) fn normalize_for_occurrence(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

/// A covering run shorter than this asserts nothing — single shared
/// characters would let any name assemble itself out of an unrelated
/// document's alphabet.
const OCCURRENCE_MIN_RUN: usize = 2;

/// Below this many characters a name must appear verbatim — a short
/// name (`"k6"`, `"S3"`) has no room for partial coverage to mean
/// anything.
const OCCURRENCE_VERBATIM_MAX: usize = 3;

/// Coverage threshold, as a ratio: at least 3/4 of the name's
/// characters must be covered by runs of [`OCCURRENCE_MIN_RUN`]+
/// characters that appear in the document. High enough that a
/// fabricated entity sharing one fragment fails, low enough that a
/// particle dropped from a Japanese compound (`"プール最大接続数"` built
/// from `"プールの最大接続数"`) still passes.
const OCCURRENCE_COVERAGE_NUM: usize = 3;
const OCCURRENCE_COVERAGE_DEN: usize = 4;

/// ADR 0036 (#853): in a name written entirely in a sparse script —
/// Latin, Cyrillic, Greek…: no ideograph, kana, or hangul anywhere in
/// it — a run of letters counts toward coverage only as a whole word
/// of the name at least this long (`std`, `API`, `Copy`, `heap`; a
/// two-letter word is `of`/`at`/`is`, which any text of the language
/// contains) …
const SPARSE_WORD_MIN_RUN: usize = 3;

/// … or as a stem at least this long inside a longer word — the
/// inflected or pluralized form the document has (`selects` for
/// `selection`, `recommendation` for `recommendations`). Two-letter
/// runs are [`OCCURRENCE_MIN_RUN`]'s unit for ideographs, where each
/// character is a morpheme; in a sparse script the morpheme is the
/// word, and two shared letters are a bigram any paragraph of the
/// language supplies (measured on the verification corpus: 90% of
/// fabricated English phrases passed against unrelated English text).
const SPARSE_STEM_MIN_RUN: usize = 5;

/// One character of a name as the cover walks it: lowercased, with
/// whether it opened or closed a word of the ORIGINAL name — a word
/// boundary is the name's edge or a neighboring non-alphanumeric
/// character (whitespace, `-`, `_`, `::`), read before whitespace is
/// dropped so `Copy trait` still knows where `copy` ends.
#[derive(Clone, Copy)]
struct OccurrenceGlyph {
    ch: char,
    word_start: bool,
    word_end: bool,
}

/// [`normalize_for_occurrence`] with word boundaries kept: the
/// glyphs' characters, concatenated, are exactly the normal form.
fn occurrence_glyphs(name: &str) -> Vec<OccurrenceGlyph> {
    let original: Vec<char> = name.chars().collect();
    let mut glyphs = Vec::with_capacity(original.len());
    for (index, &c) in original.iter().enumerate() {
        if c.is_whitespace() {
            continue;
        }
        let word_start = index == 0 || !original[index - 1].is_alphanumeric();
        let word_end = index + 1 == original.len() || !original[index + 1].is_alphanumeric();
        // Lowercasing can expand one character to several (`İ`);
        // the flags stay on the outermost of them.
        let lowered: Vec<char> = c.to_lowercase().collect();
        let last = lowered.len() - 1;
        for (offset, ch) in lowered.into_iter().enumerate() {
            glyphs.push(OccurrenceGlyph {
                ch,
                word_start: word_start && offset == 0,
                word_end: word_end && offset == last,
            });
        }
    }
    glyphs
}

/// A character of a script where one character is a morpheme — CJK
/// ideographs (with kana and the other East Asian scripts in the
/// same block range), hangul (syllables and the conjoining jamo a
/// decomposed spelling uses), halfwidth katakana, the ideograph
/// extension planes — as opposed to a sparse script where the
/// morpheme is the word. Alphabetic only: CJK punctuation in a name
/// is a symbol like any other.
fn is_dense_script(c: char) -> bool {
    c.is_alphabetic()
        && matches!(
            c as u32,
            0x1100..=0x11FF
                | 0x2E80..=0x9FFF
                | 0xA960..=0xA97F
                | 0xAC00..=0xD7AF
                | 0xD7B0..=0xD7FF
                | 0xF900..=0xFAFF
                | 0xFF66..=0xFF9F
                | 0x20000..=0x3134F
        )
}

/// Where a run of a sparse-script name's characters that the document
/// contains must stop (ADR 0036 §3.2): a run of letters never crosses
/// a word boundary of the name, so it is cut after the first word end
/// inside it — the cover then takes the longest run WITHIN the word
/// (`prediction` still counts when the document goes on with `heads`
/// and the name with `head insertion`), and letters of two adjacent
/// words never assemble a stem neither word has (`ab cd ef` against
/// `abcde`). A run holding a digit or symbol is left whole, as under
/// ADR 0013.
fn clip_at_word_end(run: &[OccurrenceGlyph]) -> usize {
    if run.iter().any(|glyph| !glyph.ch.is_alphabetic()) {
        return run.len();
    }
    run.iter()
        .position(|glyph| glyph.word_end)
        .map_or(run.len(), |index| index + 1)
}

/// Whether a run of a sparse-script name's characters that the
/// document contains counts toward coverage (ADR 0036): any run
/// holding a digit or symbol does, as under ADR 0013 (`20→100`,
/// `v1.2`); a run of letters only as a whole word of the name of
/// [`SPARSE_WORD_MIN_RUN`]+ letters or a stem of
/// [`SPARSE_STEM_MIN_RUN`]+.
fn sparse_run_counts(run: &[OccurrenceGlyph]) -> bool {
    if run.iter().any(|glyph| !glyph.ch.is_alphabetic()) {
        return true;
    }
    run.len() >= SPARSE_STEM_MIN_RUN
        || (run.len() >= SPARSE_WORD_MIN_RUN && run[0].word_start && run[run.len() - 1].word_end)
}

/// Whether a subject/object name plausibly appears in the document:
/// verbatim substring after normalization, or — for names long enough
/// to judge — greedy coverage by document substrings. Deterministic,
/// dictionary-free, same answer every attempt; the corrective loop's
/// nondeterminism is exactly what this replaces (#496). A name with
/// no dense-script character is judged by words and stems, not
/// character pairs (ADR 0036, #853): the pair rule was calibrated on
/// ideographs and admits nearly any English phrase against English
/// text.
pub(super) fn name_occurs(haystack: &str, name: &str) -> bool {
    let glyphs = occurrence_glyphs(name);
    let needle: String = glyphs.iter().map(|glyph| glyph.ch).collect();
    if needle.is_empty() {
        return true; // emptiness is field_absence's finding, not this one's
    }
    if haystack.contains(&needle) {
        return true;
    }
    if glyphs.len() <= OCCURRENCE_VERBATIM_MAX {
        return false;
    }
    let sparse_script = !glyphs.iter().any(|glyph| is_dense_script(glyph.ch));
    // Greedy left-to-right cover: at each position take the longest
    // run that appears in the document (containment is monotone in
    // run length, so the longest run binary-searches), count it if it
    // meets the minimum, then continue after it.
    let mut covered = 0usize;
    let mut start = 0usize;
    // `get`, not `start < len()`: the extra iteration a `<=` would add
    // (empty tail, no run) is unobservable — the form leaves no such
    // mutant to test.
    while glyphs.get(start).is_some() {
        let mut low = 0usize;
        let mut high = glyphs.len() - start;
        while low < high {
            let mid = (low + high).div_ceil(2);
            let probe: String = glyphs[start..start + mid]
                .iter()
                .map(|glyph| glyph.ch)
                .collect();
            if haystack.contains(&probe) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        if sparse_script {
            low = clip_at_word_end(&glyphs[start..start + low]);
        }
        if low >= OCCURRENCE_MIN_RUN
            && (!sparse_script || sparse_run_counts(&glyphs[start..start + low]))
        {
            covered += low;
            start += low;
        } else {
            start += 1;
        }
    }
    covered * OCCURRENCE_COVERAGE_DEN >= glyphs.len() * OCCURRENCE_COVERAGE_NUM
}

/// The Stage 2 mechanical half (ADR 0013): an alias whose canonical
/// names nothing ANY output's associations contain cannot import —
/// merge() would drop it — so it is removed with accounting instead of
/// spending the cross-chunk corrective turn issue #199 gave it.
/// Shadowing and conflicting mappings stay corrective: both carry real
/// content whose right resolution the model, not mechanics, must
/// judge. Called AFTER `correct_cross_output_issues` so every
/// corrective message's item indices still match the answer text the
/// model sees replayed; a dangling alias a correction itself
/// introduces lands here too, on the same terms.
pub(super) fn prune_unresolvable_aliases(outputs: &mut [ChunkOutput]) -> usize {
    let (concept_names, label_names) = association_name_sets(outputs);
    let mut count = 0;
    for chunk in outputs.iter_mut() {
        let removed = &mut chunk.removed;
        let mut index = 0usize;
        chunk.output.aliases.retain(|alias| {
            let path = format!("aliases[{index}]");
            index += 1;
            let (Some(spelling), Some(canonical), Some(kind)) =
                (&alias.alias, &alias.canonical, &alias.kind)
            else {
                return true; // Stage 1's finding, not this one's
            };
            if spelling == canonical {
                return true; // likewise
            }
            let names = match kind.as_str() {
                "concept" => &concept_names,
                "label" => &label_names,
                _ => return true,
            };
            if names.contains(spelling) || names.contains(canonical) {
                return true; // resolvable, or shadowing (corrective)
            }
            removed.push(Removal::new(
                path,
                format!(
                    "canonical {} names nothing the associations contain",
                    quote_for_issue(canonical)
                ),
                alias,
            ));
            count += 1;
            false
        });
    }
    count
}

/// Issue #758: the names earlier documents of this run — and the
/// `--vocabulary` context — already settled on, per namespace, each
/// spelling mapped to the record it resolves to: a subject/object or
/// alias canonical to itself, an alias spelling to its canonical.
/// That is exactly the lookup import's `add_alias` consults before
/// refusing a rewire (`AliasError::Conflict`: one spelling is one
/// referent, aliases included), replayed here so the refusal happens
/// at extraction time with accounting instead of stopping an import
/// stream three batches in with the earlier batches already applied.
/// A merge (aliasing two names that both already exist) is refused
/// on the same terms — import has no merge, the consolidation audit
/// (ADR 0012 §4) proposes one later.
#[derive(Default)]
pub(super) struct ClaimedNames {
    pub(super) concepts: BTreeMap<String, String>,
    pub(super) labels: BTreeMap<String, String>,
}

impl ClaimedNames {
    /// Seeds both namespaces from `--vocabulary`'s harvested name
    /// sets. The export's own alias spellings are not harvested
    /// (ADR 0015: only the spellings the graph settles on are offered),
    /// so a target-context alias is known here only through its
    /// canonical — the import refusal still stands for those; this is
    /// the subset extract can see.
    pub(super) fn seeded(concepts: &BTreeSet<String>, labels: &BTreeSet<String>) -> Self {
        Self {
            concepts: concepts
                .iter()
                .map(|name| (name.clone(), name.clone()))
                .collect(),
            labels: labels
                .iter()
                .map(|name| (name.clone(), name.clone()))
                .collect(),
        }
    }

    /// Records what a document this run just wrote will intern.
    pub(super) fn absorb_extraction(&mut self, extraction: &Extraction) {
        for fact in &extraction.associations {
            claim_name(&mut self.concepts, &fact.subject);
            claim_name(&mut self.concepts, &fact.object);
            claim_name(&mut self.labels, &fact.label);
        }
        claim_aliases(&mut self.concepts, &extraction.concepts);
        claim_aliases(&mut self.labels, &extraction.labels);
    }

    /// Records what a manifest-skipped document's already-written
    /// batch interns — the same names `absorb_vocabulary` rereads for
    /// the label prompt, so a skipped document claims exactly what a
    /// freshly written one does.
    pub(super) fn absorb_batch(&mut self, batch: &crate::ingest::Batch) {
        for [subject, label, object] in batch.association_triples() {
            claim_name(&mut self.concepts, subject);
            claim_name(&mut self.concepts, object);
            claim_name(&mut self.labels, label);
        }
        claim_aliases(&mut self.concepts, batch.concept_aliases());
        claim_aliases(&mut self.labels, batch.label_aliases());
    }

    /// What `spelling` resolves to in `namespace`, or `None` when
    /// nothing has claimed it.
    fn resolve<'a>(namespace: &'a BTreeMap<String, String>, spelling: &'a str) -> Option<&'a str> {
        namespace.get(spelling).map(String::as_str)
    }
}

/// A name (subject, object, label, alias canonical) resolves to
/// itself — unless an earlier alias already maps this spelling, in
/// which case import's `add_alias` would keep routing through that
/// alias, and so does this map.
fn claim_name(namespace: &mut BTreeMap<String, String>, name: &str) {
    namespace
        .entry(name.to_string())
        .or_insert_with(|| name.to_string());
}

/// An alias spelling resolves to its canonical's record: the canonical
/// claims itself first (import interns it before any alias lands on
/// it), then the spelling routes to whatever the canonical resolves
/// to — aliasing to an alias lands on the true record, as `add_alias`
/// does.
fn claim_aliases(namespace: &mut BTreeMap<String, String>, aliases: &BTreeMap<String, String>) {
    for (spelling, canonical) in aliases {
        claim_name(namespace, canonical);
        let target = namespace
            .get(canonical)
            .cloned()
            .unwrap_or_else(|| canonical.clone());
        namespace.entry(spelling.clone()).or_insert(target);
    }
}

/// Issue #758: an alias whose spelling an EARLIER document of this run
/// (or the `--vocabulary` context) already interned as a different
/// record cannot import — `add_alias` refuses the rewire, and the 409
/// stops the whole import stream — so it is removed with accounting,
/// alongside [`prune_unresolvable_aliases`]. Mechanical, not
/// corrective: the in-document shadowing check stays corrective
/// because the model can re-judge its OWN associations, but it can
/// never un-claim a name a previous document settled on. The same
/// mapping claimed twice (spelling already resolving to this very
/// canonical) is import's idempotent no-op and survives.
pub(super) fn prune_claimed_aliases(outputs: &mut [ChunkOutput], claimed: &ClaimedNames) -> usize {
    let mut count = 0;
    for chunk in outputs.iter_mut() {
        let removed = &mut chunk.removed;
        let mut index = 0usize;
        chunk.output.aliases.retain(|alias| {
            let path = format!("aliases[{index}]");
            index += 1;
            let (Some(spelling), Some(canonical), Some(kind)) =
                (&alias.alias, &alias.canonical, &alias.kind)
            else {
                return true; // Stage 1's finding, not this one's
            };
            let namespace = match kind.as_str() {
                "concept" => &claimed.concepts,
                "label" => &claimed.labels,
                _ => return true,
            };
            let Some(existing) = ClaimedNames::resolve(namespace, spelling) else {
                return true; // nothing claimed this spelling yet
            };
            let target = ClaimedNames::resolve(namespace, canonical).unwrap_or(canonical);
            if existing == target {
                return true; // the same mapping again — import's no-op
            }
            removed.push(Removal::new(
                path,
                format!(
                    "alias {} already names a {kind} an earlier document or the target \
                     context settled on; an alias cannot rewire it (import would refuse the \
                     batch)",
                    quote_for_issue(spelling)
                ),
                alias,
            ));
            count += 1;
            false
        });
    }
    count
}

/// The alias index an issue path names (`aliases[3].alias: …`,
/// `aliases[3]: conflicts …`, `aliases[3].canonical: …`), or `None`
/// for any issue about something other than an alias item.
pub(super) fn alias_issue_index(issue: &str) -> Option<usize> {
    let rest = issue.strip_prefix("aliases[")?;
    let (digits, _) = rest.split_once(']')?;
    digits.parse().ok()
}

/// ADR 0022 (#763): what the Stage 2 re-check still flags after the
/// one corrective turn. Alias issues — a spelling that still shadows
/// an association name, a mapping that still conflicts, a canonical
/// that still names the reserved type label — are removed here with
/// accounting, the ADR 0013 way: an alias records a spelling variant,
/// never a fact, so losing it loses nothing the consolidation audit
/// cannot propose later (the same ruling #758 made for cross-document
/// shadowing), while failing the whole source over it lost every fact
/// the document held. Anything that is NOT an alias item (a schema
/// domain/range violation on an association) is content, and keeps
/// ADR 0001 §8's ruling: `Err` with that output's issues, the caller
/// fails the source. Indices are removed highest-first within an
/// output so each recorded path still names the alias the issue did.
pub(super) fn prune_uncorrected_aliases(
    outputs: &mut [ChunkOutput],
    issues_by_output: Vec<(usize, Vec<String>)>,
) -> Result<usize, (usize, Vec<String>)> {
    let mut count = 0;
    for (output_index, issues) in issues_by_output {
        let non_alias: Vec<String> = issues
            .iter()
            .filter(|issue| alias_issue_index(issue).is_none())
            .cloned()
            .collect();
        if !non_alias.is_empty() {
            return Err((output_index, non_alias));
        }
        let chunk = &mut outputs[output_index];
        let mut doomed: Vec<(usize, &String)> = issues
            .iter()
            .filter_map(|issue| alias_issue_index(issue).map(|index| (index, issue)))
            .collect();
        doomed.sort_by_key(|(index, _)| std::cmp::Reverse(*index));
        doomed.dedup_by_key(|(index, _)| *index);
        for (index, issue) in doomed {
            if index < chunk.output.aliases.len() {
                let alias = chunk.output.aliases.remove(index);
                // The issue text is `aliases[N]<.field>: <finding>`;
                // the path half is the item's address, the rest the
                // finding, as every other Removal splits them.
                let (path, finding) = issue.split_once(": ").unwrap_or((issue.as_str(), ""));
                chunk.removed.push(Removal::new(
                    path,
                    format!("{finding} — still so after the corrective turn; removed"),
                    &alias,
                ));
                count += 1;
            }
        }
    }
    Ok(count)
}
