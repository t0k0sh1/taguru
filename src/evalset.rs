//! `eval.jsonl` (ADR 0003 §11, ADR 0004 §6): the dataset `taguru
//! benchmark search` (issue #260) and `taguru evaluate` (issue #215)
//! share. One `taguru_eval` header on line 1 (equality-checked — the
//! same `taguru_batch` reasoning applies, since the producer is a
//! person hand-writing the file, not §10's `IMAGE_VERSION` range
//! acceptance for taguru's own artifacts), then one case record per
//! line.
//!
//! #260 reads a fixed core subset — `case_id`/`query`/`cues`/
//! `expected_sources[]`/`expected_concepts[]`/`options.limit` — and
//! carries #215's own extensions (`expected_labels`/
//! `expected_associations`/`expected_citations`/`options.floor`/
//! `tags`/`since`/`until`) through untouched. Every case still declares
//! those extension fields explicitly and uses
//! `#[serde(deny_unknown_fields)]`: a typo in a hand-written dataset
//! must be a reported error (matching `Header`/`GroupLine`,
//! src/ingest.rs:1037,1062), which only works if the fields #260
//! itself does not interpret are still named — otherwise
//! `deny_unknown_fields` would reject them as typos instead of letting
//! them ride through. Detected extension use warns once per run, never
//! once per case (ADR 0003 §11).
//!
//! **Versioning (ADR 0004 §6).** `taguru_eval` stays `1`. Completing a
//! field that was reserved-but-undefined finishes version 1; it does
//! not revise it — `Option<Value>` on the extension fields was never a
//! schema promise, only what `deny_unknown_fields` forced so a
//! hand-written dataset's typo stays a reported error. `options.tags`
//! and `options.until` are additive (`#[serde(default)]`, so existing
//! v1 files parse unchanged) and close ADR 0004 §2.1's filter gaps;
//! `options.sources` — never implemented, named no server field — is
//! retired outright rather than carried through undefined.
//!
//! **Two modes ([`Extensions`]).** Once the extension fields have
//! concrete types instead of opaque `Value`s, a malformed value turns
//! from a silent pass-through into a hard parse error — which would
//! mean #260 starts rejecting datasets over fields it has no business
//! validating, breaking ADR 0003 §11's "carries through untouched and
//! does not interpret" literally (a parse error is interpretation).
//! [`load_eval_file`] therefore takes a mode: [`Extensions::CarryThrough`]
//! (#260) leaves every extension field as an opaque, unvalidated
//! `Value` and warns once per run if any case uses one; [`Extensions::Interpret`]
//! (#215) types and validates them, reporting a malformed one as an
//! ordinary parse error, and never warns — that mode's own call is the
//! interpreter the extensions were declared for.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

const EVAL_VERSION: u64 = 1;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireHeader {
    taguru_eval: u64,
    #[serde(default)]
    name: Option<String>,
    /// A #215 execution binding (ADR 0003 §11) — #260 always overrides
    /// the target with its own per-model corpus, so this rides through
    /// declared-but-unread rather than being rejected as a typo.
    /// Left untyped in both modes — ADR 0004 does not fix its
    /// semantics, and #215's own CLI takes `--context` as a required
    /// flag instead (ADR 0004 §5).
    #[serde(default)]
    #[allow(dead_code)]
    default_target: Option<Value>,
}

/// One `expected_sources[]` entry (ADR 0003 §11): a source path plus
/// the paragraphs that answer it, and a graded relevance #215's own
/// nDCG extension reads in full — #260 only checks `relevance >= 1`.
/// A core field, typed identically in both modes.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExpectedSource {
    pub(crate) source: String,
    /// Empty means "any paragraph of this source answers the case."
    #[serde(default)]
    pub(crate) paragraphs: Vec<u32>,
    #[serde(default = "default_relevance")]
    pub(crate) relevance: u8,
}

fn default_relevance() -> u8 {
    1
}

/// One `expected_associations[]` entry (ADR 0004 §7 step 2): `/query`
/// pins all three positions, so all three are required here too.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExpectedAssociation {
    pub(crate) subject: String,
    pub(crate) label: String,
    pub(crate) object: String,
}

/// One `expected_citations[]` entry (ADR 0004 §8). `paragraph` matches
/// `CitationRequest.paragraph` (`src/api/sources.rs:68-74`). `evaluate`
/// (#215) is what reads `paragraph`/`section`/`quote` — this loader
/// only types and validates them, so they are unread dead code in this
/// tree until #215 lands.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExpectedCitation {
    pub(crate) source: String,
    #[allow(dead_code)]
    pub(crate) paragraph: u32,
    /// Three-valued: an absent key means "don't check section," an
    /// explicit `null` means "assert this paragraph is outside every
    /// stored section" — a real, checkable claim (ADR 0004 §8), not a
    /// serde artifact. A plain `Option<Option<String>>` field collapses
    /// `null` into "absent" under serde's derive, so this goes through
    /// [`double_option`] instead.
    #[serde(default, deserialize_with = "double_option")]
    #[allow(dead_code)]
    pub(crate) section: Option<Option<String>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) quote: Option<String>,
}

/// Distinguishes an absent key (outer `None`) from an explicit `null`
/// (`Some(None)`) for [`ExpectedCitation::section`] — see its doc
/// comment for why this needs to be three-valued.
fn double_option<'de, D>(deserializer: D) -> Result<Option<Option<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer).map(Some)
}

/// `options` on the wire (ADR 0003 §11, ADR 0004 §6): `limit` is the
/// only field #260 reads and is always typed. `floor`/`tags`/`since`/
/// `until` are #215-only; under [`Extensions::CarryThrough`] they stay
/// opaque `Value`s (their presence folds into the once-per-run warning
/// via [`WireOptions::carries_215_extension`]), under
/// [`Extensions::Interpret`] they are typed by [`EvalOptions`].
/// `options.sources` is retired (ADR 0004 §6/§13.3): omitting it from
/// this struct means `deny_unknown_fields` rejects it in both modes.
#[derive(Debug, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields, default)]
struct WireOptions {
    limit: Option<usize>,
    floor: Option<Value>,
    tags: Option<Value>,
    since: Option<Value>,
    until: Option<Value>,
}

impl WireOptions {
    fn carries_215_extension(&self) -> bool {
        self.floor.is_some() || self.tags.is_some() || self.since.is_some() || self.until.is_some()
    }
}

/// One case record on the wire (ADR 0003 §11). Extension fields stay
/// opaque `Value`s here regardless of mode — [`interpret_case`]
/// converts them into [`EvalCase`]'s typed fields under
/// [`Extensions::Interpret`] only.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct WireCase {
    case_id: String,
    query: String,
    /// #260 does not drive retrieval from cues (its only search entry
    /// point is `POST /contexts/{name}/sources/search`, over `query`
    /// — ADR 0003 §11); this rides along and is only echoed back in
    /// `retrieval.json`'s per-case block for a reader's own reference.
    #[serde(default)]
    cues: Vec<String>,
    #[serde(default)]
    expected_sources: Vec<ExpectedSource>,
    #[serde(default)]
    expected_concepts: Vec<String>,
    #[serde(default)]
    options: WireOptions,
    // #215-only extensions: opaque under `CarryThrough`, typed by
    // `interpret_case` under `Interpret`.
    #[serde(default)]
    expected_labels: Option<Value>,
    #[serde(default)]
    expected_associations: Option<Value>,
    #[serde(default)]
    expected_citations: Option<Value>,
}

impl WireCase {
    fn carries_215_extension(&self) -> bool {
        self.expected_labels.is_some()
            || self.expected_associations.is_some()
            || self.expected_citations.is_some()
            || self.options.carries_215_extension()
    }
}

/// `options`, resolved for a caller (ADR 0003 §11, ADR 0004 §6).
/// `limit` is populated in both modes; `floor`/`tags`/`since`/`until`
/// are populated only under [`Extensions::Interpret`] — under
/// [`Extensions::CarryThrough`] they are always empty/`None`, since
/// #260 has no business reading fields it does not interpret.
#[derive(Debug, Default, Clone)]
pub(crate) struct EvalOptions {
    pub(crate) limit: Option<usize>,
    // `evaluate` (#273) reads these for the passage lane's
    // SearchPassagesRequest.
    pub(crate) floor: Option<f32>,
    pub(crate) tags: Vec<String>,
    pub(crate) since: Option<u64>,
    pub(crate) until: Option<u64>,
}

/// One case record, resolved for a caller (ADR 0003 §11, ADR 0004 §6).
/// `expected_labels`/`expected_associations`/`expected_citations` are
/// populated only under [`Extensions::Interpret`] — under
/// [`Extensions::CarryThrough`] they are always empty, matching
/// `EvalOptions`'s same rule.
#[derive(Debug, Clone)]
pub(crate) struct EvalCase {
    pub(crate) case_id: String,
    pub(crate) query: String,
    pub(crate) cues: Vec<String>,
    pub(crate) expected_sources: Vec<ExpectedSource>,
    pub(crate) expected_concepts: Vec<String>,
    pub(crate) options: EvalOptions,
    // `evaluate` (#273) reads these for the structural lane's coverage
    // cues and expected_associations[] probes.
    pub(crate) expected_labels: Vec<String>,
    pub(crate) expected_associations: Vec<ExpectedAssociation>,
    // `evaluate`'s citation lane (#275) is what reads this; unread dead
    // code in this tree until #275 lands.
    #[allow(dead_code)]
    pub(crate) expected_citations: Vec<ExpectedCitation>,
}

impl EvalCase {
    /// Whether this case carries any expectation at all — the switch
    /// that turns recall@k/MRR on (ADR 0003 §11).
    pub(crate) fn has_expectations(&self) -> bool {
        !self.expected_sources.is_empty() || !self.expected_concepts.is_empty()
    }
}

/// The shared loader's mode (ADR 0004 §6) — see the module doc comment
/// for the rationale.
pub(crate) enum Extensions {
    /// #260: extension fields stay opaque, unvalidated; a case carrying
    /// any of them adds one line to `warnings`, once per run. `consumer`
    /// is the caller's own verb name, substituted into the warning text
    /// so the loader itself never hard-codes one verb's name.
    CarryThrough { consumer: &'static str },
    /// #215: extension fields are typed and validated; a malformed one
    /// is a reported parse error like any other field. No warning is
    /// emitted. `evaluate` (#273) constructs this.
    Interpret,
}

/// A parsed, validated `eval.jsonl`: the header's declared name (for
/// `retrieval.json`'s `inputs` block), every case, and warnings to
/// print once — never once per case (ADR 0003 §11).
#[derive(Debug)]
pub(crate) struct LoadedEvalSet {
    pub(crate) name: Option<String>,
    pub(crate) cases: Vec<EvalCase>,
    pub(crate) warnings: Vec<String>,
}

/// Converts one field's raw `Value` under [`Extensions::Interpret`]:
/// absent (`None`) becomes `T::default()`, present becomes a typed
/// value or a reported parse error in the same style as every other
/// field in this file.
fn interpret_field<T: DeserializeOwned + Default>(
    raw: Option<Value>,
    label: &str,
    number: usize,
    case_id: &str,
    field: &str,
) -> Result<T, String> {
    match raw {
        None => Ok(T::default()),
        Some(value) => serde_json::from_value(value)
            .map_err(|error| format!("{label}: line {number}: case '{case_id}': {field}: {error}")),
    }
}

/// Types and validates one wire case's #215-only extension fields
/// under [`Extensions::Interpret`] (ADR 0004 §6/§7/§8). A malformed or
/// out-of-range value is a reported parse error, matching every other
/// field this loader validates.
/// The typed result of interpreting one wire case's #215-only extension
/// fields — [`interpret_case`]'s return value, named rather than a
/// four-tuple so the call site reads by field, not position.
struct InterpretedExtensions {
    expected_labels: Vec<String>,
    expected_associations: Vec<ExpectedAssociation>,
    expected_citations: Vec<ExpectedCitation>,
    options: EvalOptions,
}

fn interpret_case(
    wire: WireCase,
    label: &str,
    number: usize,
) -> Result<InterpretedExtensions, String> {
    let case_id = wire.case_id.as_str();

    let expected_labels: Vec<String> = interpret_field(
        wire.expected_labels,
        label,
        number,
        case_id,
        "expected_labels",
    )?;
    for name in &expected_labels {
        if name.trim().is_empty() {
            return Err(format!(
                "{label}: line {number}: case '{case_id}': expected_labels: entries must not be \
                 empty"
            ));
        }
    }

    let expected_associations: Vec<ExpectedAssociation> = interpret_field(
        wire.expected_associations,
        label,
        number,
        case_id,
        "expected_associations",
    )?;
    for assoc in &expected_associations {
        if assoc.subject.trim().is_empty()
            || assoc.label.trim().is_empty()
            || assoc.object.trim().is_empty()
        {
            return Err(format!(
                "{label}: line {number}: case '{case_id}': expected_associations: subject/\
                 label/object must not be empty"
            ));
        }
    }

    let expected_citations: Vec<ExpectedCitation> = interpret_field(
        wire.expected_citations,
        label,
        number,
        case_id,
        "expected_citations",
    )?;
    for citation in &expected_citations {
        if citation.source.trim().is_empty() {
            return Err(format!(
                "{label}: line {number}: case '{case_id}': expected_citations: source must not \
                 be empty"
            ));
        }
    }

    let floor: Option<f32> =
        interpret_field(wire.options.floor, label, number, case_id, "options.floor")?;
    if let Some(floor) = floor
        && !(0.0..=1.0).contains(&floor)
    {
        return Err(format!(
            "{label}: line {number}: case '{case_id}': options.floor must be 0.0..=1.0, got \
             {floor}"
        ));
    }
    let tags: Vec<String> =
        interpret_field(wire.options.tags, label, number, case_id, "options.tags")?;
    let since: Option<u64> =
        interpret_field(wire.options.since, label, number, case_id, "options.since")?;
    let until: Option<u64> =
        interpret_field(wire.options.until, label, number, case_id, "options.until")?;

    let options = EvalOptions {
        limit: wire.options.limit,
        floor,
        tags,
        since,
        until,
    };

    Ok(InterpretedExtensions {
        expected_labels,
        expected_associations,
        expected_citations,
        options,
    })
}

/// Parses and validates `path`. Every check that can be caught before
/// any HTTP request runs here — a malformed dataset must never surface
/// mid-run as a confusing per-case failure.
pub(crate) fn load_eval_file(path: &Path, mode: Extensions) -> Result<LoadedEvalSet, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let label = path.display().to_string();

    let mut header: Option<WireHeader> = None;
    let mut case_ids = BTreeSet::new();
    let mut cases = Vec::new();
    let mut saw_215_extension = false;

    for (index, raw_line) in text.lines().enumerate() {
        let number = index + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("{label}: line {number}: not JSON: {error}"))?;

        if header.is_none() {
            let parsed: WireHeader = serde_json::from_value(value).map_err(|error| {
                format!("{label}: line {number}: not a valid eval header: {error}")
            })?;
            if parsed.taguru_eval != EVAL_VERSION {
                return Err(format!(
                    "{label}: line {number}: taguru_eval must be {EVAL_VERSION}, got {}",
                    parsed.taguru_eval
                ));
            }
            header = Some(parsed);
            continue;
        }

        let wire: WireCase = serde_json::from_value(value)
            .map_err(|error| format!("{label}: line {number}: not a valid eval case: {error}"))?;
        if wire.case_id.is_empty() {
            return Err(format!("{label}: line {number}: case_id must not be empty"));
        }
        if wire.query.trim().is_empty() {
            return Err(format!(
                "{label}: line {number}: case '{}': query must not be empty",
                wire.case_id
            ));
        }
        if !case_ids.insert(wire.case_id.clone()) {
            return Err(format!(
                "{label}: line {number}: duplicate case_id '{}'",
                wire.case_id
            ));
        }
        for expected in &wire.expected_sources {
            if expected.relevance > 3 {
                return Err(format!(
                    "{label}: line {number}: case '{}': expected_sources relevance must be \
                     0..=3, got {}",
                    wire.case_id, expected.relevance
                ));
            }
        }
        saw_215_extension |= wire.carries_215_extension();

        let case = match mode {
            Extensions::CarryThrough { .. } => EvalCase {
                case_id: wire.case_id,
                query: wire.query,
                cues: wire.cues,
                expected_sources: wire.expected_sources,
                expected_concepts: wire.expected_concepts,
                options: EvalOptions {
                    limit: wire.options.limit,
                    ..EvalOptions::default()
                },
                expected_labels: Vec::new(),
                expected_associations: Vec::new(),
                expected_citations: Vec::new(),
            },
            Extensions::Interpret => {
                let case_id = wire.case_id.clone();
                let query = wire.query.clone();
                let cues = wire.cues.clone();
                let expected_sources = wire.expected_sources.clone();
                let expected_concepts = wire.expected_concepts.clone();
                let interpreted = interpret_case(wire, &label, number)?;
                EvalCase {
                    case_id,
                    query,
                    cues,
                    expected_sources,
                    expected_concepts,
                    options: interpreted.options,
                    expected_labels: interpreted.expected_labels,
                    expected_associations: interpreted.expected_associations,
                    expected_citations: interpreted.expected_citations,
                }
            }
        };
        cases.push(case);
    }

    let Some(header) = header else {
        return Err(format!(
            "{label}: empty file: expected a taguru_eval header line"
        ));
    };
    if cases.is_empty() {
        return Err(format!(
            "{label}: no cases: expected at least one case line after the header"
        ));
    }

    let mut warnings = Vec::new();
    if saw_215_extension && let Extensions::CarryThrough { consumer } = mode {
        warnings.push(format!(
            "{label}: this eval.jsonl carries #215-only fields (expected_labels / \
             expected_associations / expected_citations / options.floor|tags|since|until) — \
             {consumer} does not interpret them and passes them through untouched"
        ));
    }

    Ok(LoadedEvalSet {
        name: header.name,
        cases,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write_temp(tag: &str, text: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "taguru-evalset-test-{tag}-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::write(&path, text).unwrap();
        path
    }

    const CONSUMER: &str = "taguru benchmark search";
    const HEADER: &str = r#"{"taguru_eval":1,"name":"sake retrieval cases"}"#;
    const CASE: &str = r#"{"case_id":"brand-origin-001","query":"青嶺はどこの蔵の酒か","cues":["青嶺"],"expected_sources":[{"source":"corpus/brewery.md","paragraphs":[0],"relevance":3}],"expected_concepts":["青嶺酒造"],"options":{"limit":10}}"#;

    fn carry_through() -> Extensions {
        Extensions::CarryThrough { consumer: CONSUMER }
    }

    #[test]
    fn a_well_formed_file_loads_its_name_and_one_case() {
        let path = write_temp("ok", &format!("{HEADER}\n{CASE}\n"));
        let loaded = load_eval_file(&path, Extensions::Interpret).unwrap();
        assert_eq!(loaded.name.as_deref(), Some("sake retrieval cases"));
        assert_eq!(loaded.cases.len(), 1);
        assert!(loaded.warnings.is_empty());
        assert!(loaded.cases[0].has_expectations());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_wrong_version_is_refused_by_equality_not_range() {
        let path = write_temp("version", "{\"taguru_eval\":2}\n");
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(error.contains("taguru_eval must be 1"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_typo_field_is_rejected_not_silently_ignored_under_interpret() {
        let path = write_temp(
            "typo-interpret",
            &format!("{HEADER}\n{{\"case_id\":\"c1\",\"query\":\"q\",\"expcted_sources\":[]}}\n"),
        );
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(error.contains("not a valid eval case"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_typo_field_is_rejected_not_silently_ignored_under_carry_through() {
        let path = write_temp(
            "typo-carry",
            &format!("{HEADER}\n{{\"case_id\":\"c1\",\"query\":\"q\",\"expcted_sources\":[]}}\n"),
        );
        let error = load_eval_file(&path, carry_through()).unwrap_err();
        assert!(error.contains("not a valid eval case"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_215_only_field_is_accepted_and_warned_once_per_run_with_the_callers_verb_name() {
        let path = write_temp(
            "215-fields",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"expected_labels\":[\"好き\"]}}\n\
                 {{\"case_id\":\"c2\",\"query\":\"q2\",\"options\":{{\"floor\":0.5}}}}\n"
            ),
        );
        let loaded = load_eval_file(&path, carry_through()).unwrap();
        assert_eq!(loaded.cases.len(), 2);
        assert_eq!(loaded.warnings.len(), 1, "{:?}", loaded.warnings);
        assert!(
            loaded.warnings[0].contains(CONSUMER),
            "{:?}",
            loaded.warnings
        );
        assert!(
            loaded.cases[0].expected_labels.is_empty(),
            "CarryThrough must not populate typed extension fields"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn carry_through_accepts_malformed_extension_values() {
        let path = write_temp(
            "carry-malformed",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"expected_citations\":42,\
                 \"options\":{{\"tags\":\"not-an-array\"}}}}\n"
            ),
        );
        let loaded = load_eval_file(&path, carry_through()).unwrap();
        assert_eq!(loaded.cases.len(), 1);
        assert_eq!(loaded.warnings.len(), 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn interpret_reports_a_malformed_extension_value_as_a_parse_error() {
        let path = write_temp(
            "interpret-malformed",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"expected_citations\":42}}\n"
            ),
        );
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(error.contains("expected_citations"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn interpret_never_warns() {
        let path = write_temp(
            "interpret-no-warn",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"expected_labels\":[\"好き\"]}}\n"
            ),
        );
        let loaded = load_eval_file(&path, Extensions::Interpret).unwrap();
        assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn interpret_types_expected_labels_associations_and_citations() {
        let path = write_temp(
            "interpret-types",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\
                 \"expected_labels\":[\"好き\"],\
                 \"expected_associations\":[{{\"subject\":\"青嶺\",\"label\":\"好き\",\
                 \"object\":\"日本酒\"}}],\
                 \"expected_citations\":[{{\"source\":\"corpus/brewery.md\",\"paragraph\":3}}]}}\n"
            ),
        );
        let loaded = load_eval_file(&path, Extensions::Interpret).unwrap();
        let case = &loaded.cases[0];
        assert_eq!(case.expected_labels, vec!["好き".to_string()]);
        assert_eq!(case.expected_associations.len(), 1);
        assert_eq!(case.expected_associations[0].subject, "青嶺");
        assert_eq!(case.expected_citations.len(), 1);
        assert_eq!(case.expected_citations[0].paragraph, 3);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn interpret_types_options_floor_tags_since_until() {
        let path = write_temp(
            "interpret-options",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"options\":{{\"limit\":5,\"floor\":0.4,\
                 \"tags\":[\"brand\"],\"since\":100,\"until\":200}}}}\n"
            ),
        );
        let loaded = load_eval_file(&path, Extensions::Interpret).unwrap();
        let options = &loaded.cases[0].options;
        assert_eq!(options.limit, Some(5));
        assert_eq!(options.floor, Some(0.4));
        assert_eq!(options.tags, vec!["brand".to_string()]);
        assert_eq!(options.since, Some(100));
        assert_eq!(options.until, Some(200));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn interpret_distinguishes_absent_null_and_present_section() {
        let path = write_temp(
            "interpret-section",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"expected_citations\":[\
                 {{\"source\":\"s\",\"paragraph\":0}},\
                 {{\"source\":\"s\",\"paragraph\":1,\"section\":null}},\
                 {{\"source\":\"s\",\"paragraph\":2,\"section\":\"沿革\"}}]}}\n"
            ),
        );
        let loaded = load_eval_file(&path, Extensions::Interpret).unwrap();
        let citations = &loaded.cases[0].expected_citations;
        assert_eq!(citations[0].section, None);
        assert_eq!(citations[1].section, Some(None));
        assert_eq!(citations[2].section, Some(Some("沿革".to_string())));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn interpret_rejects_an_out_of_range_floor() {
        let path = write_temp(
            "interpret-floor-range",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"options\":{{\"floor\":1.5}}}}\n"
            ),
        );
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(error.contains("options.floor must be 0.0..=1.0"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn interpret_rejects_an_empty_association_subject() {
        let path = write_temp(
            "interpret-assoc-empty",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"expected_associations\":[\
                 {{\"subject\":\"\",\"label\":\"好き\",\"object\":\"日本酒\"}}]}}\n"
            ),
        );
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(
            error.contains("subject/label/object must not be empty"),
            "{error}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn options_sources_is_rejected_under_carry_through() {
        let path = write_temp(
            "sources-carry",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"options\":{{\"sources\":[\"a\"]}}}}\n"
            ),
        );
        let error = load_eval_file(&path, carry_through()).unwrap_err();
        assert!(error.contains("not a valid eval case"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn options_sources_is_rejected_under_interpret() {
        let path = write_temp(
            "sources-interpret",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"options\":{{\"sources\":[\"a\"]}}}}\n"
            ),
        );
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(error.contains("not a valid eval case"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_existing_v1_file_without_tags_or_until_loads_unchanged_under_both_modes() {
        let path = write_temp("v1-unchanged", &format!("{HEADER}\n{CASE}\n"));
        let interpreted = load_eval_file(&path, Extensions::Interpret).unwrap();
        assert_eq!(interpreted.cases.len(), 1);
        assert!(interpreted.cases[0].options.tags.is_empty());
        assert_eq!(interpreted.cases[0].options.until, None);

        let carried = load_eval_file(&path, carry_through()).unwrap();
        assert_eq!(carried.cases.len(), 1);
        assert!(carried.warnings.is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn duplicate_case_ids_are_refused() {
        let path = write_temp(
            "dup",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q\"}}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q2\"}}\n"
            ),
        );
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(error.contains("duplicate case_id 'c1'"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_empty_query_is_refused() {
        let path = write_temp(
            "empty-query",
            &format!("{HEADER}\n{{\"case_id\":\"c1\",\"query\":\"  \"}}\n"),
        );
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(error.contains("query must not be empty"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_out_of_range_relevance_is_refused() {
        let path = write_temp(
            "relevance",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q\",\"expected_sources\":\
                 [{{\"source\":\"s\",\"relevance\":4}}]}}\n"
            ),
        );
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(error.contains("relevance must be 0..=3"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_file_with_no_cases_after_the_header_is_refused() {
        let path = write_temp("no-cases", &format!("{HEADER}\n"));
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(error.contains("no cases"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_empty_file_is_refused() {
        let path = write_temp("empty", "");
        let error = load_eval_file(&path, Extensions::Interpret).unwrap_err();
        assert!(error.contains("empty file"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let path = write_temp("blank", &format!("\n{HEADER}\n\n{CASE}\n\n"));
        let loaded = load_eval_file(&path, Extensions::Interpret).unwrap();
        assert_eq!(loaded.cases.len(), 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_case_without_expectations_does_not_switch_on_recall() {
        let path = write_temp(
            "no-expectations",
            &format!("{HEADER}\n{{\"case_id\":\"c1\",\"query\":\"q\"}}\n"),
        );
        let loaded = load_eval_file(&path, Extensions::Interpret).unwrap();
        assert!(!loaded.cases[0].has_expectations());
        let _ = fs::remove_file(&path);
    }
}
