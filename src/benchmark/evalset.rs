//! `eval.jsonl` (ADR 0003 §11): the dataset `taguru benchmark search`
//! (issue #260) and #215 share. One `taguru_eval` header on line 1
//! (equality-checked — the same `taguru_batch` reasoning applies,
//! since the producer is a person hand-writing the file, not
//! §10's `IMAGE_VERSION` range acceptance for taguru's own artifacts),
//! then one case record per line.
//!
//! #260 reads a fixed core subset — `case_id`/`query`/`cues`/
//! `expected_sources[]`/`expected_concepts[]`/`options.limit` — and
//! carries #215's own extensions (`expected_labels`/
//! `expected_associations`/`expected_citations`/`options.floor`/
//! `sources`/`since`) through untouched. Every `EvalCase` still
//! declares those extension fields explicitly and uses
//! `#[serde(deny_unknown_fields)]`: a typo in a hand-written dataset
//! must be a reported error (matching `Header`/`GroupLine`,
//! src/ingest.rs:1037,1062), which only works if the fields #260
//! itself does not interpret are still named — otherwise
//! `deny_unknown_fields` would reject them as typos instead of letting
//! them ride through. Detected extension use warns once per run, never
//! once per case (ADR 0003 §11).

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

pub(super) const EVAL_VERSION: u64 = 1;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvalHeader {
    taguru_eval: u64,
    #[serde(default)]
    name: Option<String>,
    /// A #215 execution binding (ADR 0003 §11) — #260 always overrides
    /// the target with its own per-model corpus, so this rides through
    /// declared-but-unread rather than being rejected as a typo.
    #[serde(default)]
    #[allow(dead_code)]
    default_target: Option<Value>,
}

/// One `expected_sources[]` entry (ADR 0003 §11): a source path plus
/// the paragraphs that answer it, and a graded relevance #215's own
/// nDCG extension reads in full — #260 only checks `relevance >= 1`.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(super) struct ExpectedSource {
    pub(super) source: String,
    /// Empty means "any paragraph of this source answers the case."
    #[serde(default)]
    pub(super) paragraphs: Vec<u32>,
    #[serde(default = "default_relevance")]
    pub(super) relevance: u8,
}

fn default_relevance() -> u8 {
    1
}

/// `options` (ADR 0003 §11): `limit` is the only field #260 reads;
/// `floor`/`sources`/`since` are #215-only and carried through
/// untouched — their presence folds into the once-per-run warning via
/// [`EvalOptions::carries_215_extension`].
#[derive(Debug, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields, default)]
pub(super) struct EvalOptions {
    pub(super) limit: Option<usize>,
    floor: Option<Value>,
    sources: Option<Value>,
    since: Option<Value>,
}

impl EvalOptions {
    fn carries_215_extension(&self) -> bool {
        self.floor.is_some() || self.sources.is_some() || self.since.is_some()
    }
}

/// One case record (ADR 0003 §11).
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(super) struct EvalCase {
    pub(super) case_id: String,
    pub(super) query: String,
    /// #260 does not drive retrieval from cues (its only search entry
    /// point is `POST /contexts/{name}/sources/search`, over `query`
    /// — ADR 0003 §11); this rides along and is only echoed back in
    /// `retrieval.json`'s per-case block for a reader's own reference.
    #[serde(default)]
    pub(super) cues: Vec<String>,
    #[serde(default)]
    pub(super) expected_sources: Vec<ExpectedSource>,
    #[serde(default)]
    pub(super) expected_concepts: Vec<String>,
    #[serde(default)]
    pub(super) options: EvalOptions,
    // #215-only extensions: never interpreted, only detected for the
    // once-per-run warning.
    #[serde(default)]
    expected_labels: Option<Value>,
    #[serde(default)]
    expected_associations: Option<Value>,
    #[serde(default)]
    expected_citations: Option<Value>,
}

impl EvalCase {
    fn carries_215_extension(&self) -> bool {
        self.expected_labels.is_some()
            || self.expected_associations.is_some()
            || self.expected_citations.is_some()
            || self.options.carries_215_extension()
    }

    /// Whether this case carries any expectation at all — the switch
    /// that turns recall@k/MRR on (ADR 0003 §11).
    pub(super) fn has_expectations(&self) -> bool {
        !self.expected_sources.is_empty() || !self.expected_concepts.is_empty()
    }
}

/// A parsed, validated `eval.jsonl`: the header's declared name (for
/// `retrieval.json`'s `inputs` block), every case, and warnings to
/// print once — never once per case (ADR 0003 §11).
#[derive(Debug)]
pub(super) struct LoadedEvalSet {
    pub(super) name: Option<String>,
    pub(super) cases: Vec<EvalCase>,
    pub(super) warnings: Vec<String>,
}

/// Parses and validates `path`. Every check that can be caught before
/// any HTTP request runs here — a malformed dataset must never surface
/// mid-run as a confusing per-case failure.
pub(super) fn load_eval_file(path: &Path) -> Result<LoadedEvalSet, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let label = path.display().to_string();

    let mut header: Option<EvalHeader> = None;
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
            let parsed: EvalHeader = serde_json::from_value(value).map_err(|error| {
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

        let case: EvalCase = serde_json::from_value(value)
            .map_err(|error| format!("{label}: line {number}: not a valid eval case: {error}"))?;
        if case.case_id.is_empty() {
            return Err(format!("{label}: line {number}: case_id must not be empty"));
        }
        if case.query.trim().is_empty() {
            return Err(format!(
                "{label}: line {number}: case '{}': query must not be empty",
                case.case_id
            ));
        }
        if !case_ids.insert(case.case_id.clone()) {
            return Err(format!(
                "{label}: line {number}: duplicate case_id '{}'",
                case.case_id
            ));
        }
        for expected in &case.expected_sources {
            if expected.relevance > 3 {
                return Err(format!(
                    "{label}: line {number}: case '{}': expected_sources relevance must be \
                     0..=3, got {}",
                    case.case_id, expected.relevance
                ));
            }
        }
        saw_215_extension |= case.carries_215_extension();
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
    if saw_215_extension {
        warnings.push(format!(
            "{label}: this eval.jsonl carries #215-only fields (expected_labels / \
             expected_associations / expected_citations / options.floor|sources|since) — \
             taguru benchmark search does not interpret them and passes them through \
             untouched"
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

    const HEADER: &str = r#"{"taguru_eval":1,"name":"sake retrieval cases"}"#;
    const CASE: &str = r#"{"case_id":"brand-origin-001","query":"青嶺はどこの蔵の酒か","cues":["青嶺"],"expected_sources":[{"source":"corpus/brewery.md","paragraphs":[0],"relevance":3}],"expected_concepts":["青嶺酒造"],"options":{"limit":10}}"#;

    #[test]
    fn a_well_formed_file_loads_its_name_and_one_case() {
        let path = write_temp("ok", &format!("{HEADER}\n{CASE}\n"));
        let loaded = load_eval_file(&path).unwrap();
        assert_eq!(loaded.name.as_deref(), Some("sake retrieval cases"));
        assert_eq!(loaded.cases.len(), 1);
        assert!(loaded.warnings.is_empty());
        assert!(loaded.cases[0].has_expectations());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_wrong_version_is_refused_by_equality_not_range() {
        let path = write_temp("version", "{\"taguru_eval\":2}\n");
        let error = load_eval_file(&path).unwrap_err();
        assert!(error.contains("taguru_eval must be 1"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_typo_field_is_rejected_not_silently_ignored() {
        let path = write_temp(
            "typo",
            &format!("{HEADER}\n{{\"case_id\":\"c1\",\"query\":\"q\",\"expcted_sources\":[]}}\n"),
        );
        let error = load_eval_file(&path).unwrap_err();
        assert!(error.contains("not a valid eval case"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_215_only_field_is_accepted_and_warned_once_per_run_not_per_case() {
        let path = write_temp(
            "215-fields",
            &format!(
                "{HEADER}\n\
                 {{\"case_id\":\"c1\",\"query\":\"q1\",\"expected_labels\":[\"好き\"]}}\n\
                 {{\"case_id\":\"c2\",\"query\":\"q2\",\"options\":{{\"floor\":0.5}}}}\n"
            ),
        );
        let loaded = load_eval_file(&path).unwrap();
        assert_eq!(loaded.cases.len(), 2);
        assert_eq!(loaded.warnings.len(), 1, "{:?}", loaded.warnings);
        assert!(
            loaded.warnings[0].contains("#215-only"),
            "{:?}",
            loaded.warnings
        );
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
        let error = load_eval_file(&path).unwrap_err();
        assert!(error.contains("duplicate case_id 'c1'"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_empty_query_is_refused() {
        let path = write_temp(
            "empty-query",
            &format!("{HEADER}\n{{\"case_id\":\"c1\",\"query\":\"  \"}}\n"),
        );
        let error = load_eval_file(&path).unwrap_err();
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
        let error = load_eval_file(&path).unwrap_err();
        assert!(error.contains("relevance must be 0..=3"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_file_with_no_cases_after_the_header_is_refused() {
        let path = write_temp("no-cases", &format!("{HEADER}\n"));
        let error = load_eval_file(&path).unwrap_err();
        assert!(error.contains("no cases"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_empty_file_is_refused() {
        let path = write_temp("empty", "");
        let error = load_eval_file(&path).unwrap_err();
        assert!(error.contains("empty file"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let path = write_temp("blank", &format!("\n{HEADER}\n\n{CASE}\n\n"));
        let loaded = load_eval_file(&path).unwrap();
        assert_eq!(loaded.cases.len(), 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_case_without_expectations_does_not_switch_on_recall() {
        let path = write_temp(
            "no-expectations",
            &format!("{HEADER}\n{{\"case_id\":\"c1\",\"query\":\"q\"}}\n"),
        );
        let loaded = load_eval_file(&path).unwrap();
        assert!(!loaded.cases[0].has_expectations());
        let _ = fs::remove_file(&path);
    }
}
