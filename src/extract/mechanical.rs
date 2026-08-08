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
    pub(super) removed: Vec<String>,
    pub(super) issues: Vec<String>,
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
) -> MechanicalEvaluation {
    let haystack = normalize_for_occurrence(document);
    let mut issues = Vec::new();
    let mut removed = Vec::new();
    let empty_map = serde_json::Map::new();
    let obj = value.as_object().unwrap_or(&empty_map);

    let associations = match get_present(obj, "associations") {
        None => Vec::new(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                association_mechanically(index, item, &haystack, &mut issues, &mut removed)
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
            .filter_map(|(index, item)| alias_mechanically(index, item, &mut issues, &mut removed))
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
    issues: &mut Vec<String>,
    removed: &mut Vec<String>,
) -> Option<ModelAssociation> {
    let path = format!("associations[{index}]");
    let Some(obj) = item.as_object() else {
        removed.push(format!(
            "{path}: expected an object, got {}",
            describe_value(item)
        ));
        return None;
    };
    let absent: Vec<String> = ["subject", "label", "object"]
        .iter()
        .filter_map(|key| field_absence(obj, key).map(|kind| format!("{key} {kind}")))
        .collect();
    if !absent.is_empty() {
        removed.push(format!("{path}: {}", absent.join(", ")));
        return None;
    }
    let mut item_issues = Vec::new();
    let parsed = interpret_association_item(index, item, &mut item_issues)?;
    if !item_issues.is_empty() {
        issues.append(&mut item_issues);
        return Some(parsed);
    }
    for (field, name) in [("subject", &parsed.subject), ("object", &parsed.object)] {
        let name = name
            .as_deref()
            .expect("no absence and no issue means the field parsed");
        if !name_occurs(haystack, name) {
            removed.push(format!(
                "{path}.{field}: {} does not appear in the document text",
                quote_for_issue(name)
            ));
            return None;
        }
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
    issues: &mut Vec<String>,
    removed: &mut Vec<String>,
) -> Option<ModelAlias> {
    let path = format!("aliases[{index}]");
    let Some(obj) = item.as_object() else {
        removed.push(format!(
            "{path}: expected an object, got {}",
            describe_value(item)
        ));
        return None;
    };
    let absent: Vec<String> = ["alias", "canonical", "kind"]
        .iter()
        .filter_map(|key| field_absence(obj, key).map(|kind| format!("{key} {kind}")))
        .collect();
    if !absent.is_empty() {
        removed.push(format!("{path}: {}", absent.join(", ")));
        return None;
    }
    let mut item_issues = Vec::new();
    let parsed = interpret_alias_item(index, item, &mut item_issues)?;
    if let (Some(spelling), Some(canonical)) = (&parsed.alias, &parsed.canonical)
        && spelling == canonical
    {
        // interpret_alias_item's own self-alias issue is in item_issues;
        // it dies with the item rather than becoming a corrective ask.
        removed.push(format!("{path}: alias equals its canonical"));
        return None;
    }
    if !item_issues.is_empty() {
        issues.append(&mut item_issues);
    }
    Some(parsed)
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

/// Whether a subject/object name plausibly appears in the document:
/// verbatim substring after normalization, or — for names long enough
/// to judge — greedy coverage by document substrings. Deterministic,
/// dictionary-free, same answer every attempt; the corrective loop's
/// nondeterminism is exactly what this replaces (#496).
pub(super) fn name_occurs(haystack: &str, name: &str) -> bool {
    let needle = normalize_for_occurrence(name);
    if needle.is_empty() {
        return true; // emptiness is field_absence's finding, not this one's
    }
    if haystack.contains(&needle) {
        return true;
    }
    let chars: Vec<char> = needle.chars().collect();
    if chars.len() <= OCCURRENCE_VERBATIM_MAX {
        return false;
    }
    // Greedy left-to-right cover: at each position take the longest
    // run that appears in the document (containment is monotone in
    // run length, so the longest run binary-searches), count it if it
    // meets the minimum, then continue after it.
    let mut covered = 0usize;
    let mut start = 0usize;
    while start < chars.len() {
        let mut low = 0usize;
        let mut high = chars.len() - start;
        while low < high {
            let mid = (low + high).div_ceil(2);
            let probe: String = chars[start..start + mid].iter().collect();
            if haystack.contains(&probe) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        if low >= OCCURRENCE_MIN_RUN {
            covered += low;
            start += low;
        } else {
            start += 1;
        }
    }
    covered * OCCURRENCE_COVERAGE_DEN >= chars.len() * OCCURRENCE_COVERAGE_NUM
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
pub(super) fn prune_unresolvable_aliases(
    outputs: &mut [ChunkOutput],
    chunk_total: usize,
) -> Vec<String> {
    let (concept_names, label_names) = association_name_sets(outputs);
    let mut removed = Vec::new();
    for chunk in outputs.iter_mut() {
        let prefix = if chunk_total > 1 {
            format!("chunk {}/{chunk_total} ", chunk.chunk_index + 1)
        } else {
            String::new()
        };
        let mut index = 0usize;
        chunk.output.aliases.retain(|alias| {
            let path = format!("{prefix}aliases[{index}]");
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
            removed.push(format!(
                "{path}: canonical {} names nothing the associations contain",
                quote_for_issue(canonical)
            ));
            false
        });
    }
    removed
}
