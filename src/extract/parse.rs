//! Parsing and interpreting one model answer into a `ModelOutput`.

use super::*;

/// The shape the model is asked for. Lenient on the model's side —
/// unknown fields pass, weight defaults — because [`merge`] validates
/// every item strictly before anything is emitted, and (issue #199)
/// [`interpret_model_output`] names every departure it papers over as a
/// path-addressed issue so the strict path can turn it into a
/// corrective turn instead of a silent drop.
// Clone/Serialize/Deserialize are for the chunk checkpoint file's own
// storage/(de)serialization (issue #179) only — the lenient hand-rolled
// parse (`interpret_model_output`) that actually reads a model's raw
// answer builds these from `serde_json::Value` directly and never
// touches derive-based deserialization.
#[derive(Default, Clone, serde::Serialize, Deserialize)]
#[cfg_attr(test, derive(Debug))]
pub(super) struct ModelOutput {
    pub(super) associations: Vec<ModelAssociation>,
    pub(super) aliases: Vec<ModelAlias>,
    pub(super) questions: Vec<ModelQuestion>,
}

#[derive(Clone, serde::Serialize, Deserialize)]
#[cfg_attr(test, derive(Debug))]
pub(super) struct ModelAssociation {
    pub(super) subject: Option<String>,
    pub(super) label: Option<String>,
    pub(super) object: Option<String>,
    pub(super) weight: Option<f64>,
    pub(super) paragraph: Option<u32>,
}

#[derive(Clone, serde::Serialize, Deserialize)]
#[cfg_attr(test, derive(Debug))]
pub(super) struct ModelAlias {
    pub(super) alias: Option<String>,
    pub(super) canonical: Option<String>,
    pub(super) kind: Option<String>,
}

#[derive(Clone, serde::Serialize, Deserialize)]
#[cfg_attr(test, derive(Debug))]
pub(super) struct ModelQuestion {
    pub(super) paragraph: Option<u32>,
    pub(super) question: Option<String>,
}

/// The rules one document's items are checked against — the two
/// pieces of per-document context [`interpret_model_output`] needs
/// that no single item carries on its own.
#[derive(Clone, Copy)]
pub(super) struct ItemRules {
    /// The document's canonical paragraph count (`--questions`'
    /// `paragraph` citations and, informationally only, associations'
    /// own `paragraph` tag are checked against this).
    pub(super) paragraph_count: usize,
    /// Whether this run asked for questions at all (`--questions N` >
    /// 0). When false, a volunteered `questions` array is `merge()`'s
    /// policy trim, never a validity issue — see [`interpret_questions`].
    pub(super) questions_requested: bool,
}

/// The assistant text must contain one JSON object; code fences and
/// prose around it are tolerated (strip, then widest-braces fallback).
/// Test-only: exercises the lenient Value-walk parse (via
/// [`interpret_model_output`]) in isolation from
/// [`evaluate_answer`]'s strict/lossy distinction — every production
/// corrective loop calls `evaluate_answer` directly.
#[cfg(test)]
pub(super) fn parse_model_output(content: &str) -> Result<ModelOutput, String> {
    let value = candidate_json(content)?;
    let lenient_rules = ItemRules {
        paragraph_count: usize::MAX,
        questions_requested: true,
    };
    let (output, _issues) = interpret_model_output(&value, &lenient_rules);
    Ok(output)
}

/// Trim, strip fences, and parse into a bare `Value` — everything
/// [`parse_model_output`] used to do before handing the result to
/// serde's derived `Deserialize`. A non-object top level (an array, a
/// scalar) is refused here exactly like the derived impl refused it,
/// so every caller downstream keeps seeing "not a JSON object" for the
/// same inputs.
pub(super) fn candidate_json(content: &str) -> Result<serde_json::Value, String> {
    let unfenced = strip_fences(content.trim());
    // Name the real failure: a thinking-mode model can spend its whole
    // budget on reasoning and answer with no text at all, and "EOF at
    // line 1 column 0" diagnoses nothing.
    if unfenced.is_empty() {
        return Err(empty_answer_diagnosis());
    }
    if let Some(value) = parse_top_level_object(unfenced) {
        return Ok(value);
    }
    let first = match serde_json::from_str::<serde_json::Value>(unfenced) {
        Ok(_) => "the top-level value is not a JSON object".to_string(),
        Err(error) => error.to_string(),
    };
    if let (Some(start), Some(end)) = (unfenced.find('{'), unfenced.rfind('}'))
        && start < end
        && let Some(value) = parse_top_level_object(&unfenced[start..=end])
    {
        return Ok(value);
    }
    Err(format!("not a JSON object: {first}"))
}

pub(super) fn parse_top_level_object(text: &str) -> Option<serde_json::Value> {
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(value) if value.is_object() => Some(value),
        _ => None,
    }
}

/// Reads a JSON object into the lenient [`ModelOutput`] shape while
/// collecting a path-addressed issue for every departure the lenient
/// walk papers over: a present-but-wrong-typed field, a non-object
/// array element, a missing/empty/oversized required string, an
/// out-of-range business value. The returned `ModelOutput` is exactly
/// what today's lenient deserializer would have produced — absent and
/// null both read as "not present," a malformed scalar or array
/// element reads as `None`/skipped — so a caller that ignores the
/// issues (lossy mode, [`parse_model_output`]'s golden-test callers)
/// sees byte-for-byte the old behavior. Issue #199/ADR 0001 §8: this is
/// the "lenient parse, strict accounting" split — parsing never gets
/// stricter, accounting does.
pub(super) fn interpret_model_output(
    value: &serde_json::Value,
    rules: &ItemRules,
) -> (ModelOutput, Vec<String>) {
    let mut issues = Vec::new();
    let empty_map = serde_json::Map::new();
    // interpret_model_output tolerates a non-object top level (reads
    // nothing) rather than asserting one; candidate_json is what
    // actually refuses a non-object answer for parse_model_output's
    // callers.
    let obj = value.as_object().unwrap_or(&empty_map);
    let associations = interpret_associations(obj, &mut issues);
    let aliases = interpret_aliases(obj, &mut issues);
    let questions = interpret_questions(obj, rules, &mut issues);
    (
        ModelOutput {
            associations,
            aliases,
            questions,
        },
        issues,
    )
}

pub(super) fn interpret_associations(
    obj: &serde_json::Map<String, serde_json::Value>,
    issues: &mut Vec<String>,
) -> Vec<ModelAssociation> {
    match get_present(obj, "associations") {
        None => Vec::new(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| interpret_association_item(index, item, issues))
            .collect(),
        Some(other) => {
            issues.push(format!(
                "associations: expected an array, got {}",
                describe_value(other)
            ));
            Vec::new()
        }
    }
}

pub(super) fn interpret_association_item(
    index: usize,
    item: &serde_json::Value,
    issues: &mut Vec<String>,
) -> Option<ModelAssociation> {
    let path = format!("associations[{index}]");
    let Some(obj) = item.as_object() else {
        issues.push(format!(
            "{path}: expected an object, got {}",
            describe_value(item)
        ));
        return None;
    };
    let subject = interpret_required_string(obj, "subject", &path, MAX_NAME_BYTES, issues);
    let label = interpret_required_string(obj, "label", &path, MAX_NAME_BYTES, issues);
    let object = interpret_required_string(obj, "object", &path, MAX_NAME_BYTES, issues);
    let weight = interpret_weight(obj, &path, issues);
    let paragraph = interpret_association_paragraph(obj, &path, issues);
    Some(ModelAssociation {
        subject,
        label,
        object,
        weight,
        paragraph,
    })
}

pub(super) fn interpret_aliases(
    obj: &serde_json::Map<String, serde_json::Value>,
    issues: &mut Vec<String>,
) -> Vec<ModelAlias> {
    match get_present(obj, "aliases") {
        None => Vec::new(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| interpret_alias_item(index, item, issues))
            .collect(),
        Some(other) => {
            issues.push(format!(
                "aliases: expected an array, got {}",
                describe_value(other)
            ));
            Vec::new()
        }
    }
}

pub(super) fn interpret_alias_item(
    index: usize,
    item: &serde_json::Value,
    issues: &mut Vec<String>,
) -> Option<ModelAlias> {
    let path = format!("aliases[{index}]");
    let Some(obj) = item.as_object() else {
        issues.push(format!(
            "{path}: expected an object, got {}",
            describe_value(item)
        ));
        return None;
    };
    let alias = interpret_required_string(obj, "alias", &path, MAX_NAME_BYTES, issues);
    let canonical = interpret_canonical(obj, &path, issues);
    let kind = interpret_kind(obj, &path, issues);
    // Self-alias is item-local (both sides come from this one item);
    // dangling-canonical and shadowing need the merged name set and are
    // Stage 2's job (issue #199 §2 cross-chunk validation, cross_output_issues).
    if let (Some(spelling), Some(canonical_name)) = (&alias, &canonical)
        && spelling == canonical_name
    {
        issues.push(format!("{path}.alias: equals its canonical"));
    }
    Some(ModelAlias {
        alias,
        canonical,
        kind,
    })
}

/// `canonical` never fails on emptiness here: an empty (or merely
/// non-matching) canonical is exactly a *dangling* canonical, and
/// dangling-ness can only be judged against the merged association
/// names — Stage 2's `cross_output_issues`, not this item-local pass.
pub(super) fn interpret_canonical(
    obj: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    issues: &mut Vec<String>,
) -> Option<String> {
    match get_present(obj, "canonical") {
        None => {
            issues.push(format!("{path}.canonical: missing"));
            None
        }
        Some(serde_json::Value::String(text)) => {
            let trimmed = text.trim();
            if trimmed.len() > MAX_NAME_BYTES {
                issues.push(format!(
                    "{path}.canonical: {} bytes exceeds the {MAX_NAME_BYTES}-byte cap",
                    trimmed.len()
                ));
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Some(other) => {
            issues.push(format!(
                "{path}.canonical: expected a string, got {}",
                describe_value(other)
            ));
            None
        }
    }
}

pub(super) fn interpret_kind(
    obj: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    issues: &mut Vec<String>,
) -> Option<String> {
    match get_present(obj, "kind") {
        None => {
            issues.push(format!("{path}.kind: missing"));
            None
        }
        Some(serde_json::Value::String(text)) if text == "concept" || text == "label" => {
            Some(text.clone())
        }
        Some(serde_json::Value::String(text)) => {
            issues.push(format!(
                "{path}.kind: expected \"concept\" or \"label\", got {text:?}"
            ));
            None
        }
        Some(other) => {
            issues.push(format!(
                "{path}.kind: expected \"concept\" or \"label\", got {}",
                describe_value(other)
            ));
            None
        }
    }
}

pub(super) fn interpret_questions(
    obj: &serde_json::Map<String, serde_json::Value>,
    rules: &ItemRules,
    issues: &mut Vec<String>,
) -> Vec<ModelQuestion> {
    match get_present(obj, "questions") {
        None => Vec::new(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| interpret_question_item(index, item, rules, issues))
            .collect(),
        Some(other) => {
            // questions_cap == 0 makes any questions array the model
            // volunteers merge()'s policy trim, never a validity issue
            // — see the doc comment on ItemRules::questions_requested.
            if rules.questions_requested {
                issues.push(format!(
                    "questions: expected an array, got {}",
                    describe_value(other)
                ));
            }
            Vec::new()
        }
    }
}

pub(super) fn interpret_question_item(
    index: usize,
    item: &serde_json::Value,
    rules: &ItemRules,
    issues: &mut Vec<String>,
) -> Option<ModelQuestion> {
    let path = format!("questions[{index}]");
    let Some(obj) = item.as_object() else {
        if rules.questions_requested {
            issues.push(format!(
                "{path}: expected an object, got {}",
                describe_value(item)
            ));
        }
        return None;
    };
    if !rules.questions_requested {
        // Not asked for: whatever the model volunteers is merge()'s
        // policy trim (questions_cap == 0), so read it plainly (today's
        // lenient semantics) without spending an issue on it.
        let paragraph = get_present(obj, "paragraph").and_then(interpret_paragraph_index);
        let question = get_present(obj, "question")
            .and_then(|value| value.as_str())
            .map(str::to_string);
        return Some(ModelQuestion {
            paragraph,
            question,
        });
    }
    let paragraph = match get_present(obj, "paragraph") {
        None => {
            issues.push(format!("{path}.paragraph: missing"));
            None
        }
        Some(value) => match interpret_paragraph_index(value) {
            Some(paragraph) if (paragraph as usize) < rules.paragraph_count => Some(paragraph),
            Some(paragraph) => {
                issues.push(format!(
                    "{path}.paragraph: must cite a paragraph below {}, got {paragraph}",
                    rules.paragraph_count
                ));
                None
            }
            None => {
                issues.push(format!(
                    "{path}.paragraph: expected an integer paragraph index, got {}",
                    describe_value(value)
                ));
                None
            }
        },
    };
    let question = interpret_required_string(
        obj,
        "question",
        &path,
        crate::api::MAX_QUESTION_BYTES,
        issues,
    );
    Some(ModelQuestion {
        paragraph,
        question,
    })
}

/// A required string field shared by associations (`subject`/`label`/
/// `object`), aliases (`alias`), and questions (`question`): missing,
/// wrong-typed, empty-after-trim, and oversized are each their own
/// issue text so the model sees exactly which of the four it hit.
pub(super) fn interpret_required_string(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    path: &str,
    max_bytes: usize,
    issues: &mut Vec<String>,
) -> Option<String> {
    match get_present(obj, key) {
        None => {
            issues.push(format!("{path}.{key}: missing"));
            None
        }
        Some(serde_json::Value::String(text)) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                issues.push(format!("{path}.{key}: empty"));
                None
            } else if trimmed.len() > max_bytes {
                issues.push(format!(
                    "{path}.{key}: {} bytes exceeds the {max_bytes}-byte cap",
                    trimmed.len()
                ));
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Some(other) => {
            issues.push(format!(
                "{path}.{key}: expected a string, got {}",
                describe_value(other)
            ));
            None
        }
    }
}

/// `weight` is optional (absent/null is a plain 1.0 assertion, kept as
/// `None` here for `merge()` to default) but a *present* value must be
/// a finite, non-zero number under the magnitude cap — a zero asserts
/// nothing and an infinite/oversized one is not a fact merge() can
/// carry. A well-TYPED business-rule violation (zero, over-cap,
/// non-finite) still returns `Some(weight)`, not `None`: `merge()` —
/// not this parse-level pass — is the sole authority on whether that
/// value survives (its own zero/finite/magnitude checks, unchanged by
/// issue #199), in strict mode via the corrective turn this value's
/// issue triggers and in `--lossy` via its original drop-and-proceed
/// logic. Returning `None` here instead would let a lossy run's
/// `unwrap_or(1.0)` default silently launder an invalid weight into a
/// valid-looking `1.0` — exactly the silent behavior change issue #199
/// forbids for a mode whose entire contract is "byte-for-byte today's
/// behavior." Only a WRONG-TYPED value (never a number at all) returns
/// `None`, matching `lenient_f64`'s original type-only leniency.
pub(super) fn interpret_weight(
    obj: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    issues: &mut Vec<String>,
) -> Option<f64> {
    match get_present(obj, "weight") {
        None => None,
        Some(serde_json::Value::Number(number)) => {
            let weight = number.as_f64().unwrap_or(f64::NAN);
            if !weight.is_finite() {
                issues.push(format!(
                    "{path}.weight: expected finite non-zero number, got {weight}"
                ));
            } else if weight == 0.0 {
                issues.push(format!(
                    "{path}.weight: expected finite non-zero number, got 0"
                ));
            } else if weight.abs() > MAX_ASSOCIATION_WEIGHT {
                issues.push(format!(
                    "{path}.weight: expected finite non-zero number, got {weight} \
                     (over the {MAX_ASSOCIATION_WEIGHT} cap)"
                ));
            }
            Some(weight)
        }
        Some(other) => {
            issues.push(format!(
                "{path}.weight: expected finite non-zero number, got {}",
                describe_value(other)
            ));
            None
        }
    }
}

/// An association's `paragraph` is optional and, unlike a question's,
/// never business-rule-checked here: a well-typed but out-of-range
/// paragraph costs only the tag in `merge()` (the fact survives
/// untagged), so only a wrong-typed value is a validity issue.
pub(super) fn interpret_association_paragraph(
    obj: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    issues: &mut Vec<String>,
) -> Option<u32> {
    match get_present(obj, "paragraph") {
        None => None,
        Some(value) => match interpret_paragraph_index(value) {
            Some(paragraph) => Some(paragraph),
            None => {
                issues.push(format!(
                    "{path}.paragraph: expected an integer paragraph index, got {}",
                    describe_value(value)
                ));
                None
            }
        },
    }
}

/// A non-negative integer that fits `u32` — the same shape
/// `lenient_u32` used to accept, just read from a `Value` already in
/// hand instead of through a deserializer.
pub(super) fn interpret_paragraph_index(value: &serde_json::Value) -> Option<u32> {
    value.as_u64().and_then(|value| u32::try_from(value).ok())
}

/// A present, non-null field read from a JSON object — absent and
/// `null` are the same "not here" for every optional field this
/// module validates (ADR 0001 §8's ruling applies to required fields;
/// an optional field's null and absence are both simply valid-absent).
pub(super) fn get_present<'a>(
    obj: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a serde_json::Value> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => Some(value),
    }
}

/// How many bytes of a string value's own text an issue message
/// embeds before eliding the rest — long enough to recognize the
/// value, short enough that a pathological answer cannot make one
/// issue line balloon.
pub(super) const MAX_ISSUE_VALUE_BYTES: usize = 64;

/// Renders a JSON value's type and, for scalars, its content — for a
/// wrong-typed-field issue's "got …" clause. A `String` is quoted
/// (`string "high"`) so the corrective message can distinguish a
/// wrong-typed value from a business-rule violation on a rightly-typed
/// one (which builds its own "got 0"/"got 2000000" text instead of
/// calling this).
pub(super) fn describe_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(flag) => format!("boolean {flag}"),
        serde_json::Value::Number(number) => format!("number {number}"),
        serde_json::Value::String(text) => format!("string {}", quote_for_issue(text)),
        serde_json::Value::Array(_) => "an array".to_string(),
        serde_json::Value::Object(_) => "an object".to_string(),
    }
}

pub(super) fn quote_for_issue(text: &str) -> String {
    let cut = floor_char_boundary(text, MAX_ISSUE_VALUE_BYTES);
    if cut < text.len() {
        format!("{:?}…", &text[..cut])
    } else {
        format!("{text:?}")
    }
}

/// The canonical JSON Schema for the shape [`parse_model_output`] accepts —
/// mirrored by hand (never derived from `ModelOutput`'s own `Deserialize`
/// impl) into the Python and TypeScript LangChain SDKs as
/// `MODEL_OUTPUT_JSON_SCHEMA`, the same discipline [`PROMPT_VERSION`] and
/// [`system_prompt`]'s wording already follow. A `BaseChatModel` that
/// supports schema-constrained generation can be pointed at this to shape
/// what the model answers with, instead of only checking it afterward.
///
/// Deliberately stricter than `ModelOutput`'s own lenient `Deserialize`:
/// - `additionalProperties: false` everywhere, and every field this schema
///   marks required is one [`merge`] always drops the item over anyway
///   (`subject`/`label`/`object` on an association; `alias`/`canonical`/
///   `kind` on an alias; `paragraph`/`question` on a question) — a
///   schema-constrained model structurally cannot produce the
///   wrong-typed-scalar or extra-property cases [`lenient_string`] and
///   friends exist to tolerate, so there is nothing to be lenient about.
/// - `weight` and an association's `paragraph` stay optional: [`merge`]
///   defaults a missing weight to `1.0` and untags (never drops) a
///   missing or out-of-range paragraph, so omitting either is a valid,
///   intentional shape rather than something merely tolerated.
///
/// What this schema does NOT encode — [`merge`]'s later business-rule
/// validation, applied identically however the answer was produced:
/// - Byte-length caps (`MAX_NAME_BYTES`, `MAX_QUESTION_BYTES`): JSON
///   Schema's `maxLength` counts UTF-16 code units, not UTF-8 bytes, so it
///   cannot mirror these precisely.
/// - An association's weight must be finite, non-zero, and within
///   `MAX_ASSOCIATION_WEIGHT` — a magnitude/business check, not a shape.
/// - A paragraph index must be less than the document's paragraph count —
///   known only per-document at merge time, never at schema-authoring
///   time; this schema only enforces the universal `>= 0` half.
/// - Cross-item rules: deduplication, and an alias's `canonical` naming a
///   subject/object/label the associations actually contain.
/// - A concept's entity type set (ADR 0009 §6.1): known only per-context,
///   at validation time — the same argument the paragraph-count entry
///   above already makes, just for a schema document instead of a
///   document's own paragraph count.
/// - "The object of relation R must be a concept some other item in this
///   answer typed as T" (ADR 0009 §7.2): a cross-item rule exactly like
///   deduplication and dangling-canonical above, checked by
///   [`schema_output_issues`] instead.
/// - Allowed relation labels are deliberately never rendered as an `enum`:
///   a structurally-constrained model could then never propose a new
///   relation, which ADR 0009 (and #218 before it) requires stays
///   possible even in a context with a schema — constraining the
///   model's *shape* is not the same as constraining its *content*.
///
/// `title` is required content, not decoration: LangChain's Python
/// `with_structured_output()` derives the tool/function name a bare JSON
/// Schema is bound under from this key, and raises before ever calling the
/// model when it is absent — confirmed against `langchain_core`'s
/// `convert_to_openai_function`, which every provider's tool-calling
/// integration funnels through.
///
/// [`json_schema_response_format`] puts this schema on this producer's own
/// OpenAI-compatible wire (`--structured-output`, ADR 0001 §4.1).
pub(crate) fn model_output_json_schema() -> serde_json::Value {
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ModelOutput",
        "type": "object",
        "additionalProperties": false,
        "required": ["associations", "aliases"],
        "properties": {
            "associations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["subject", "label", "object"],
                    "properties": {
                        "subject": {"type": "string", "minLength": 1},
                        "label": {"type": "string", "minLength": 1},
                        "object": {"type": "string", "minLength": 1},
                        "weight": {"type": "number"},
                        "paragraph": {"type": "integer", "minimum": 0}
                    }
                }
            },
            "aliases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["alias", "canonical", "kind"],
                    "properties": {
                        "alias": {"type": "string", "minLength": 1},
                        "canonical": {"type": "string", "minLength": 1},
                        "kind": {"type": "string", "enum": ["concept", "label"]}
                    }
                }
            },
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["paragraph", "question"],
                    "properties": {
                        "paragraph": {"type": "integer", "minimum": 0},
                        "question": {"type": "string", "minLength": 1}
                    }
                }
            }
        }
    })
}

/// An answer with no content once fences are stripped — the
/// thinking-budget-burn shape [`parse_model_output`] diagnoses. The
/// ladder's EMPTY state shares this exact definition so a
/// fenced-but-empty answer ("```json\n```") classifies identically on
/// both paths.
pub(super) fn is_empty_answer(content: &str) -> bool {
    strip_fences(content.trim()).is_empty()
}

pub(super) fn empty_answer_diagnosis() -> String {
    "the answer was empty — thinking-mode models can burn their whole budget on \
     reasoning before any text (docs/extract.html: turn thinking off)"
        .to_string()
}

pub(super) fn strip_fences(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    // ```json\n … \n``` — drop the info-string line and the closing fence.
    let body = rest.split_once('\n').map(|(_, body)| body).unwrap_or(rest);
    body.rsplit_once("```")
        .map(|(body, _)| body)
        .unwrap_or(body)
        .trim()
}
