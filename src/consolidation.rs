//! `taguru consolidation`: the judging client of the consolidation
//! audit (ADR 0012 §5) — the communities pattern one shelf over. The
//! SERVER detects candidates and fingerprints their evidence; this
//! verb judges each candidate with the extract LLM and writes the
//! judgments back through `POST /import` as an ordinary derived
//! context (`{name}::consolidation`), one source per candidate keyed
//! by its fingerprint. Judgment identity IS the fingerprint: a re-run
//! over an unchanged graph looks every candidate up, finds its stored
//! judgment, and makes zero LLM calls; a candidate whose evidence
//! moved has a new fingerprint and re-judges. Dismissals are
//! first-class judgments for exactly this reason — a candidate the
//! operator judged benign must not re-cost an LLM call every audit.
//!
//! Proposal-only end to end: nothing here applies anything. An
//! accepted judgment names its proposed action (an alias, a
//! retraction, a negative-weight assertion, a re-import), and the
//! operator applies it through the ordinary write APIs (ADR 0012 §6).

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::api::consolidation::{
    CONSOLIDATION_DETECTOR, ConsolidationAudit, ContradictionCandidate,
};
use crate::config::{load_config, subcommand_usage_error};
use crate::remote::default_base_url;
use crate::remote::{Api, ApiFailure};

const CONSOLIDATION_USAGE: &str =
    "usage: taguru consolidation --context NAME [--checks LIST] [--into NAME]
                             [--dry-run] [--config FILE] [--url URL] [URL]

Judges a RUNNING server's consolidation-audit candidates (merge /
contradiction / staleness — ADR 0012) with the extract LLM and stores
the judgments as an ordinary derived context (default
'NAME::consolidation'), one source per candidate keyed by its content
fingerprint. Incremental by that fingerprint: a candidate already
judged — accepted OR dismissed — is reused without an LLM call, until
its evidence changes and its fingerprint moves with it.

Judgments are proposals, never applications: an accepted judgment
names its suggested action (alias / retract / negative weight /
re-import) and the operator applies it through the ordinary write
APIs. --dry-run reports what would be judged without calling the LLM
or writing anything.

--checks LIST     comma-separated sections (default: merge,contradiction,staleness)
--into NAME       the judgment context (default: 'CONTEXT::consolidation')

The LLM rides the extract provider: TAGURU_EXTRACT_URL,
TAGURU_EXTRACT_MODEL, TAGURU_EXTRACT_API_KEY (docs/extract.html) —
required only when something actually needs judging. Auth rides
TAGURU_API_TOKEN / TAGURU_API_TOKENS; --url and the positional URL are
aliases, defaulting to TAGURU_ADDR after --config applies.

exit codes: 0 judgments up to date (or dry-run report) · 1 failed ·
2 usage error
";

/// The judgment artifact's format stamp — mismatch-checked like
/// `taguru_communities` (equality; hand-written artifacts do not
/// exist, so a different number is a different program).
const CONSOLIDATION_FORMAT: u64 = 1;

/// The manifest's reserved source id inside the judgment context.
const MANIFEST_SOURCE: &str = "consolidation:manifest";

/// One candidate's judgment source id: `judgment:{fingerprint}`.
fn judgment_source(fingerprint: &str) -> String {
    format!("judgment:{fingerprint}")
}

/// One candidate as the judge sees it, whatever section it came from.
struct Candidate {
    fingerprint: String,
    kind: &'static str,
    /// One line for the report ("merge: 青嶺酒造 ↔ 青嶺酒蔵").
    headline: String,
    /// The candidate's full wire shape — the evidence the prompt
    /// quotes verbatim, so the judge reads exactly what the audit
    /// reported.
    payload: Value,
}

pub fn run(args: &[String]) -> i32 {
    let usage = |message: &str| subcommand_usage_error("consolidation", message);
    let mut context: Option<String> = None;
    let mut into: Option<String> = None;
    let mut checks = "merge,contradiction,staleness".to_string();
    let mut config: Option<PathBuf> = None;
    let mut dry_run = false;
    let mut explicit_url: Option<String> = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{CONSOLIDATION_USAGE}");
                return 0;
            }
            "--context" => match rest.next() {
                Some(name) if context.is_none() => context = Some(name.clone()),
                Some(_) => return usage("--context given twice"),
                None => return usage("--context needs a name"),
            },
            "--into" => match rest.next() {
                Some(name) if into.is_none() => into = Some(name.clone()),
                Some(_) => return usage("--into given twice"),
                None => return usage("--into needs a name"),
            },
            "--checks" => match rest.next() {
                Some(list) => checks = list.clone(),
                None => return usage("--checks needs a comma-separated list"),
            },
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => return usage("--config given twice"),
                None => return usage("--config needs a file path"),
            },
            "--dry-run" => dry_run = true,
            "--url" => match rest.next() {
                Some(value) if explicit_url.is_none() => {
                    explicit_url = Some(value.trim_end_matches('/').to_string());
                }
                Some(_) => return usage("--url given twice"),
                None => return usage("--url needs a server URL"),
            },
            other if other.starts_with('-') => {
                return usage(&format!("unknown flag '{other}'"));
            }
            url if explicit_url.is_none() => explicit_url = Some(url.trim_end_matches('/').into()),
            _ => return usage("the URL was already given"),
        }
    }
    let Some(context) = context else {
        return usage("--context is required");
    };
    let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    // SAFETY (same contract as serve/communities): applied while the
    // process is still single-threaded.
    if let Some(path) = &config {
        load_config(path);
    }
    let base = match explicit_url.map(Ok).unwrap_or_else(default_base_url) {
        Ok(url) => url,
        Err(error) => {
            eprintln!("taguru: consolidation: {error}");
            return 2;
        }
    };
    match drive(&base, &context, into.as_deref(), &checks, dry_run) {
        Ok(report) => {
            print!("{report}");
            0
        }
        Err(error) => {
            eprintln!("taguru: consolidation: {error}");
            1
        }
    }
}

fn drive(
    base: &str,
    context: &str,
    into: Option<&str>,
    checks: &str,
    dry_run: bool,
) -> Result<String, String> {
    crate::remote::reject_userinfo(base)?;
    let api = Api::new(base.to_string());
    eprintln!("consolidation → {base}");
    api.warn_on_version_skew("consolidation");
    let artifact = into
        .map(str::to_string)
        .unwrap_or_else(|| format!("{context}::consolidation"));

    let checks: Vec<&str> = checks
        .split(',')
        .map(str::trim)
        .filter(|check| !check.is_empty())
        .collect();
    let audit_value = api.post(
        &["contexts", context, "consolidation", "audit"],
        &json!({"checks": checks}),
    )?;
    let audit: ConsolidationAudit = serde_json::from_value(audit_value)
        .map_err(|error| format!("unrecognized audit response: {error}"))?;
    if audit.detector != CONSOLIDATION_DETECTOR {
        return Err(format!(
            "the server's detector ({}) is not this build's ({CONSOLIDATION_DETECTOR}) — \
             fingerprints would be incomparable; upgrade whichever side is behind",
            audit.detector
        ));
    }
    let candidates = flatten(&audit);

    // The stored judgments this run can reuse. A manifest whose
    // detector differs marks every stored judgment incomparable —
    // loudly, the communities behavior for a changed algorithm.
    let (manifest_detector, judged) = stored_judgments(&api, &artifact, &candidates)?;
    let comparable = match &manifest_detector {
        Some(detector) if detector != CONSOLIDATION_DETECTOR => {
            eprintln!(
                "consolidation: detector changed ({detector} → {CONSOLIDATION_DETECTOR}); \
                 previous judgments are incomparable and every candidate re-judges"
            );
            false
        }
        _ => true,
    };
    let (reusable, fresh): (Vec<&Candidate>, Vec<&Candidate>) =
        candidates.iter().partition(|candidate| {
            comparable && judged.contains(&judgment_source(&candidate.fingerprint))
        });

    let mut report = String::new();
    for candidate in &fresh {
        report.push_str(&format!(
            "{} {}: {}\n",
            if dry_run { "would judge" } else { "judging" },
            candidate.kind,
            candidate.headline
        ));
    }
    if dry_run {
        report.push_str(&format!(
            "dry run: {} to judge, {} reused, nothing written\n",
            fresh.len(),
            reusable.len()
        ));
        return Ok(report);
    }
    if fresh.is_empty() {
        report.push_str(&format!(
            "judgments up to date ({} reused, no LLM calls)\n",
            reusable.len()
        ));
        return Ok(report);
    }

    let chat = crate::extract::ChatClient::from_env()
        .map_err(|error| format!("{error} (only --dry-run works without it)"))?;
    let mut batches: Vec<String> = Vec::new();
    let mut applied = 0usize;
    let mut dismissed = 0usize;
    for candidate in &fresh {
        let judgment = judge(&chat, candidate)?;
        if judgment["verdict"] == json!("apply") {
            applied += 1;
        } else {
            dismissed += 1;
        }
        report.push_str(&format!(
            "  → {} ({})\n",
            judgment["verdict"].as_str().unwrap_or("?"),
            judgment["action"].as_str().unwrap_or("-"),
        ));
        batches.push(judgment_batch(&artifact, context, candidate, &judgment));
    }
    // The manifest travels LAST, the communities ordering: its
    // presence at the new stamp means every judgment before it landed.
    batches.push(manifest_batch(&artifact, context));
    for chunk in pack_batches(&batches) {
        api.import(&chunk)?;
    }
    report.push_str(&format!(
        "judged {} ({} apply, {} dismiss), {} reused → '{artifact}'\n",
        fresh.len(),
        applied,
        dismissed,
        reusable.len()
    ));
    Ok(report)
}

/// Flattens every present section into the one candidate list the
/// judge walks — headline for the report, full wire shape for the
/// prompt.
fn flatten(audit: &ConsolidationAudit) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    if let Some(merge) = &audit.merge {
        for pair in &merge.candidates {
            candidates.push(Candidate {
                fingerprint: pair.fingerprint.clone(),
                kind: "merge",
                headline: format!("{} ↔ {}", pair.a, pair.b),
                payload: serde_json::to_value(pair).unwrap_or_default(),
            });
        }
    }
    if let Some(contradiction) = &audit.contradiction {
        for candidate in &contradiction.candidates {
            let (fingerprint, headline) = match candidate {
                ContradictionCandidate::Objects {
                    subject,
                    label,
                    objects,
                    fingerprint,
                    ..
                } => (
                    fingerprint.clone(),
                    format!("({subject}, {label}) × {}", objects.len()),
                ),
                ContradictionCandidate::Contested {
                    subject,
                    label,
                    object,
                    fingerprint,
                    ..
                } => (fingerprint.clone(), format!("{subject} —{label}→ {object}")),
            };
            candidates.push(Candidate {
                fingerprint,
                kind: "contradiction",
                headline,
                payload: serde_json::to_value(candidate).unwrap_or_default(),
            });
        }
    }
    if let Some(staleness) = &audit.staleness {
        for stale in &staleness.candidates {
            candidates.push(Candidate {
                fingerprint: stale.fingerprint.clone(),
                kind: "staleness",
                headline: format!(
                    "{} —{}→ {} (gap {}s)",
                    stale.subject, stale.label, stale.object, stale.gap
                ),
                payload: serde_json::to_value(stale).unwrap_or_default(),
            });
        }
    }
    candidates
}

/// The manifest's detector (None when the artifact context does not
/// exist yet) and the set of judgment sources already stored.
fn stored_judgments(
    api: &Api,
    artifact: &str,
    candidates: &[Candidate],
) -> Result<(Option<String>, BTreeSet<String>), String> {
    let mut wanted: Vec<String> = vec![MANIFEST_SOURCE.to_string()];
    wanted.extend(
        candidates
            .iter()
            .map(|candidate| judgment_source(&candidate.fingerprint)),
    );
    let body = json!({ "sources": wanted });
    let found = match api.post_envelope(&["contexts", artifact, "sources", "lookup"], &body) {
        Ok(result) => result,
        Err(ApiFailure::NotFound { .. }) => return Ok((None, BTreeSet::new())),
        Err(ApiFailure::Other(error)) => return Err(error),
    };
    let passages = found["passages"].as_object().cloned().unwrap_or_default();
    let manifest_detector = match passages.get(MANIFEST_SOURCE).and_then(Value::as_str) {
        Some(text) => {
            let manifest: Value = serde_json::from_str(text)
                .map_err(|error| format!("the stored manifest did not parse: {error}"))?;
            match manifest["taguru_consolidation"].as_u64() {
                Some(CONSOLIDATION_FORMAT) => {}
                Some(other) => {
                    return Err(format!(
                        "judgment artifact format {other} is not this build's \
                         {CONSOLIDATION_FORMAT} — refusing to diff against it"
                    ));
                }
                None => {
                    return Err(
                        "the stored manifest carries no taguru_consolidation stamp".to_string()
                    );
                }
            }
            manifest["detector"].as_str().map(str::to_string)
        }
        None => None,
    };
    let judged = passages
        .keys()
        .filter(|source| source.as_str() != MANIFEST_SOURCE)
        .cloned()
        .collect();
    Ok((manifest_detector, judged))
}

/// One candidate through the LLM: strict-JSON verdict, leniently
/// unwrapped (models decorate), then re-serialized normalized so the
/// artifact stores one canonical shape.
fn judge(chat: &crate::extract::ChatClient, candidate: &Candidate) -> Result<Value, String> {
    let system = "You judge one consolidation candidate of a knowledge graph: is it a \
                  real problem worth acting on, or benign? Answer STRICT JSON only, no \
                  prose around it: {\"verdict\": \"apply\" | \"dismiss\", \"action\": \
                  \"<for apply: the concrete write to make — an alias, a retraction, a \
                  negative-weight assertion, or a re-import under one canonical \
                  spelling; for dismiss: why it is benign>\", \"rationale\": \"<one or \
                  two sentences, in the language of the evidence>\"}. Never propose \
                  automatic changes beyond those write kinds.";
    let user = format!(
        "Candidate kind: {}. Evidence, exactly as the audit reported it:\n{}",
        candidate.kind,
        serde_json::to_string_pretty(&candidate.payload).unwrap_or_default(),
    );
    let response = chat.complete(
        &[
            json!({"role": "system", "content": system}),
            json!({"role": "user", "content": user}),
        ],
        &crate::extract::RequestOptions::default(),
    )?;
    let verdict = parse_judgment(&response.content).ok_or_else(|| {
        format!(
            "the model's judgment for {} was not the required JSON shape: {}",
            candidate.headline,
            response.content.trim()
        )
    })?;
    Ok(verdict)
}

/// Pulls the judgment object out of a model reply that may decorate it
/// (code fences, prose): first `{` to last `}`, parsed, `verdict`
/// validated. `None` when nothing usable is inside.
fn parse_judgment(content: &str) -> Option<Value> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    // A reply like "} judged {" finds both delimiters in the wrong
    // order; the LLM is outside the trust boundary, so this input is
    // real, and slicing it would panic.
    let body = content.get(start..=end)?;
    let parsed: Value = serde_json::from_str(body).ok()?;
    match parsed["verdict"].as_str() {
        Some("apply") | Some("dismiss") => Some(json!({
            "verdict": parsed["verdict"],
            "action": parsed["action"].as_str().unwrap_or(""),
            "rationale": parsed["rationale"].as_str().unwrap_or(""),
        })),
        _ => None,
    }
}

/// One judgment as one import batch: retract-then-apply idempotent,
/// the passage carrying the normalized judgment JSON, one association
/// recording the verdict so the artifact is queryable as a graph too.
fn judgment_batch(
    artifact: &str,
    context: &str,
    candidate: &Candidate,
    judgment: &Value,
) -> String {
    let source = judgment_source(&candidate.fingerprint);
    let text = json!({
        "kind": candidate.kind,
        "candidate": candidate.headline,
        "judgment": judgment,
        "evidence": candidate.payload,
    });
    let header = json!({
        "taguru_batch": 1,
        "context": artifact,
        "source": source,
        "create": {"description": format!("Consolidation judgments for '{context}' (ADR 0012); derived, safe to delete")},
    });
    let association = json!({
        "subject": candidate.headline,
        "label": judgment["verdict"],
        "object": candidate.kind,
        "weight": 1.0,
    });
    let passage = json!({"passage": text.to_string()});
    format!("{header}\n{association}\n{passage}")
}

/// The manifest batch — written LAST so its stamp attests a complete
/// artifact, never a torn one.
fn manifest_batch(artifact: &str, context: &str) -> String {
    let header = json!({
        "taguru_batch": 1,
        "context": artifact,
        "source": MANIFEST_SOURCE,
        "create": {"description": format!("Consolidation judgments for '{context}' (ADR 0012); derived, safe to delete")},
    });
    let manifest = json!({
        "taguru_consolidation": CONSOLIDATION_FORMAT,
        "detector": CONSOLIDATION_DETECTOR,
        "context": context,
    });
    let passage = json!({"passage": manifest.to_string()});
    format!("{header}\n{passage}")
}

/// Packs whole batches into bodies under the same chunk ceiling
/// `taguru communities` uses — a batch is the import format's atom
/// and never splits.
fn pack_batches(batches: &[String]) -> Vec<String> {
    const IMPORT_CHUNK_BYTES: usize = 2 * 1024 * 1024;
    let mut chunks = Vec::new();
    let mut current = String::new();
    for batch in batches {
        if !current.is_empty() && current.len() + batch.len() + 2 > IMPORT_CHUNK_BYTES {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(batch);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_judgment_unwraps_decorated_replies_and_refuses_junk() {
        let fenced = "```json\n{\"verdict\": \"apply\", \"action\": \"alias b → a\", \
                      \"rationale\": \"同一の蔵\"}\n```";
        let parsed = parse_judgment(fenced).unwrap();
        assert_eq!(parsed["verdict"], "apply");
        assert_eq!(parsed["action"], "alias b → a");

        let dismissed = parse_judgment("{\"verdict\": \"dismiss\", \"action\": \"benign\"}");
        assert_eq!(dismissed.unwrap()["verdict"], "dismiss");

        assert!(parse_judgment("no json here").is_none());
        assert!(
            parse_judgment("} 判定できません {").is_none(),
            "reversed delimiters must answer None, never panic"
        );
        assert!(
            parse_judgment("{\"verdict\": \"maybe\"}").is_none(),
            "an unknown verdict is junk, not a judgment"
        );
    }

    #[test]
    fn batches_are_import_streams_with_the_manifest_shape() {
        let candidate = Candidate {
            fingerprint: "00ff".into(),
            kind: "merge",
            headline: "a ↔ b".into(),
            payload: json!({"a": "a", "b": "b"}),
        };
        let batch = judgment_batch(
            "sake::consolidation",
            "sake",
            &candidate,
            &json!({"verdict": "apply", "action": "alias", "rationale": "同一"}),
        );
        let lines: Vec<&str> = batch.lines().collect();
        assert_eq!(lines.len(), 3, "header + association + passage");
        let header: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(header["taguru_batch"], 1);
        assert_eq!(header["source"], "judgment:00ff");
        let manifest = manifest_batch("sake::consolidation", "sake");
        let passage: Value = serde_json::from_str(manifest.lines().nth(1).unwrap()).unwrap();
        let stored: Value = serde_json::from_str(passage["passage"].as_str().unwrap()).unwrap();
        assert_eq!(stored["taguru_consolidation"], 1);
        assert_eq!(stored["detector"], CONSOLIDATION_DETECTOR);
    }

    #[test]
    fn pack_batches_never_splits_a_batch() {
        let big = "x".repeat(3 * 1024 * 1024);
        let chunks = pack_batches(&[big.clone(), "small".to_string()]);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], big);
        assert_eq!(chunks[1], "small");
    }
}
