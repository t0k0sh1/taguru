//! `taguru consolidation`: the judging client of the consolidation
//! audit (ADR 0012 §5) — the communities pattern one shelf over. The
//! SERVER detects candidates and fingerprints their evidence; this
//! verb judges each candidate with the extract LLM and writes the
//! judgments back through `POST /import` as an ordinary derived
//! `context` (`{name}::consolidation`), one source per candidate keyed
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
    "usage: taguru consolidation --context ID [--checks LIST] [--into ID]
                             [--dry-run] [--config FILE] [--url URL] [URL]

Judges a RUNNING server's consolidation-audit candidates (merge /
contradiction / staleness — ADR 0012) with the extract LLM and stores
the judgments as an ordinary derived context, one source per
candidate keyed by its content fingerprint. The derived context's id
comes from the source's id (so renaming the source never detaches it);
its display name defaults to 'NAME::consolidation'. Incremental by that fingerprint: a candidate already
judged — accepted OR dismissed — is reused without an LLM call, until
its evidence changes and its fingerprint moves with it.

Judgments are proposals, never applications: an accepted judgment
names its suggested action (alias / retract / negative weight /
re-import) and the operator applies it through the ordinary write
APIs. --dry-run reports what would be judged without calling the LLM
or writing anything.

--checks LIST     comma-separated sections (default: merge,contradiction,staleness)
--into ID         the judgment context's id (default: derived from --context's id)

The LLM rides the extract provider: TAGURU_EXTRACT_URL,
TAGURU_EXTRACT_MODEL, TAGURU_EXTRACT_API_KEY (docs/extract.html) —
required only when something actually needs judging. Auth rides
TAGURU_API_TOKEN / TAGURU_API_TOKENS; --url and the positional URL are
aliases, defaulting to TAGURU_ADDR after --config applies.

exit codes: 0 judgments up to date (or dry-run report) · 1 failed ·
2 usage error
";

/// The `type` of the judgment artifact's manifest (ADR 0042). Judged
/// with the format `version` beside it — hand-written artifacts do not
/// exist, so another kind or another revision is a different program's
/// record.
const MANIFEST_TYPE: &str = "consolidation_manifest";

/// The manifest's reserved source id inside the judgment `context`.
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
                Some(name) if context.is_none() && !name.starts_with('-') => {
                    context = Some(name.clone());
                }
                Some(name) if name.starts_with('-') => {
                    return usage("--context needs a context id");
                }
                Some(_) => return usage("--context given twice"),
                None => return usage("--context needs a context id"),
            },
            "--into" => match rest.next() {
                Some(name) if into.is_none() && !name.starts_with('-') => {
                    into = Some(name.clone());
                }
                Some(name) if name.starts_with('-') => return usage("--into needs a context id"),
                Some(_) => return usage("--into given twice"),
                None => return usage("--into needs a context id"),
            },
            "--checks" => match rest.next() {
                Some(list) if !list.starts_with('-') => checks = list.clone(),
                _ => return usage("--checks needs a comma-separated list"),
            },
            "--config" => match rest.next() {
                Some(path) if config.is_none() && !path.starts_with('-') => {
                    config = Some(PathBuf::from(path));
                }
                Some(path) if path.starts_with('-') => {
                    return usage("--config needs a file path");
                }
                Some(_) => return usage("--config given twice"),
                None => return usage("--config needs a file path"),
            },
            "--dry-run" => dry_run = true,
            "--url" => match rest.next() {
                Some(value) if explicit_url.is_none() && !value.starts_with('-') => {
                    explicit_url = Some(value.trim_end_matches('/').to_string());
                }
                Some(value) if value.starts_with('-') => {
                    return usage("--url needs a server URL");
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
    if let Some(into) = &into
        && !crate::registry::is_context_id(into)
    {
        return usage(&format!(
            "--into '{into}' is not a context id: it takes a lowercase hyphenated UUID, \
             the id column of GET /contexts"
        ));
    }
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
    // A base no request could leave on is a usage error (exit 2, issue
    // #751), caught before drive() prints its target line — the same
    // upfront refusal every other client verb gives it.
    if let Err(message) = crate::remote::reject_unusable_base(&base) {
        return usage(&message);
    }
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
    // `--context` takes the id (#964); the artifact's display name is
    // built from the source's display name, never from the id.
    let row = api.get(&["contexts", context])?;
    let source_name = row["name"]
        .as_str()
        .ok_or_else(|| format!("context '{context}': the row carries no name"))?
        .to_string();
    let artifact = format!("{source_name}::consolidation");
    // The artifact is addressed by id alone (#966): `--into`'s, or the
    // one derived from the source's id — no name lookup, so a rename of
    // either context, or a second context with the same display name,
    // cannot redirect the run. A first run registers it through the
    // create block under `artifact`.
    let artifact_id = into.map(str::to_string).unwrap_or_else(|| {
        crate::registry::derived_context_id(&format!("{context}::consolidation"))
    });

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
    let (manifest_detector, judged) = stored_judgments(&api, &artifact_id, &candidates)?;
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
        batches.push(judgment_batch(
            &artifact_id,
            &artifact,
            &format!("'{source_name}' ({context})"),
            candidate,
            &judgment,
            batches.is_empty(),
        ));
    }
    // The manifest travels LAST, the communities ordering: its
    // presence at the new stamp means every judgment before it landed.
    batches.push(manifest_batch(&artifact_id, context));
    for chunk in crate::remote::pack_import_chunks(&batches) {
        api.import(&chunk)?;
    }
    report.push_str(&format!(
        "judged {} ({} apply, {} dismiss), {} reused → '{artifact}' ({artifact_id})\n",
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

/// The manifest's detector (None when the artifact `context` does not
/// exist yet) and the set of judgment sources already stored.
fn stored_judgments(
    api: &Api,
    artifact_id: &str,
    candidates: &[Candidate],
) -> Result<(Option<String>, BTreeSet<String>), String> {
    let mut wanted: Vec<String> = vec![MANIFEST_SOURCE.to_string()];
    wanted.extend(
        candidates
            .iter()
            .map(|candidate| judgment_source(&candidate.fingerprint)),
    );
    let body = json!({ "sources": wanted });
    // An id nothing answers to is a first run.
    let found = match api.post_envelope(&["contexts", artifact_id, "sources", "lookup"], &body) {
        Ok(result) => result,
        Err(ApiFailure::NotFound { .. }) => return Ok((None, BTreeSet::new())),
        Err(ApiFailure::Other(error)) => return Err(error),
    };
    let passages = found["passages"].as_object().cloned().unwrap_or_default();
    let manifest_detector = match passages.get(MANIFEST_SOURCE).and_then(Value::as_str) {
        Some(text) => {
            let manifest: Value = serde_json::from_str(text)
                .map_err(|error| format!("the stored manifest did not parse: {error}"))?;
            judge_manifest(&manifest).map_err(|error| {
                format!(
                    "the stored manifest is not one this build reads ({error}) — refusing \
                     to diff against it; delete the judgment context to start over"
                )
            })?;
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
/// Only the run's FIRST batch carries the create block (`create:
/// true`) — the `taguru communities` pattern: create is consumed only
/// when the artifact `context` does not exist yet, so repeating it on
/// every batch was pure payload (issue #752).
fn judgment_batch(
    artifact_id: &str,
    artifact_name: &str,
    context: &str,
    candidate: &Candidate,
    judgment: &Value,
    create: bool,
) -> String {
    let source = judgment_source(&candidate.fingerprint);
    let text = json!({
        "kind": candidate.kind,
        "candidate": candidate.headline,
        "judgment": judgment,
        "evidence": candidate.payload,
    });
    let description = create.then(|| {
        format!("Consolidation judgments for {context} (ADR 0012); derived, safe to delete")
    });
    let header = crate::format::source_header_line(
        &source,
        artifact_id,
        description
            .as_deref()
            .map(|description| crate::format::HeaderCreate {
                name: artifact_name,
                description,
            }),
    );
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
/// artifact, never a torn one. No create block: at least one judgment
/// batch always precedes it in the same run (`fresh.is_empty()`
/// returns before any import), so the artifact `context` exists by the
/// time this lands.
/// Judges a stored manifest's `type` and `version` (ADR 0042). A
/// `version` that is present must be a string — `null` and numbers are
/// values, not the omission the column allows — and must be this
/// build's own; an absent one reads as this build's.
fn judge_manifest(manifest: &Value) -> Result<(), String> {
    crate::format::check_type(manifest.get("type").and_then(Value::as_str), MANIFEST_TYPE)?;
    match manifest.get("version") {
        None => crate::format::check_version(None),
        Some(Value::String(version)) => crate::format::check_version(Some(version)),
        Some(other) => Err(format!("version {other} is not a string")),
    }
}

fn manifest_batch(artifact_id: &str, context_id: &str) -> String {
    let header = crate::format::source_header_line(MANIFEST_SOURCE, artifact_id, None);
    let manifest = json!({
        "type": MANIFEST_TYPE,
        "version": crate::format::FORMAT_VERSION,
        "detector": CONSOLIDATION_DETECTOR,
        "context_id": context_id,
    });
    let passage = json!({"passage": manifest.to_string()});
    format!("{header}\n{passage}")
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
            "9f1d6a52-2b74-4c0e-a1c3-5e8b7d4f6a20",
            "sake::consolidation",
            "'sake' (3b1c6e0a-aaaa-4bbb-8ccc-0000000000aa)",
            &candidate,
            &json!({"verdict": "apply", "action": "alias", "rationale": "同一"}),
            true,
        );
        let lines: Vec<&str> = batch.lines().collect();
        assert_eq!(lines.len(), 3, "header + association + passage");
        let header: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(header["type"], "source");
        assert_eq!(header["id"], "judgment:00ff");
        assert!(
            header["create"]["description"].is_string(),
            "the run's first batch carries the create block: {header}"
        );

        let later = judgment_batch(
            "9f1d6a52-2b74-4c0e-a1c3-5e8b7d4f6a20",
            "sake::consolidation",
            "sake",
            &candidate,
            &json!({"verdict": "apply", "action": "alias", "rationale": "同一"}),
            false,
        );
        let header: Value = serde_json::from_str(later.lines().next().unwrap()).unwrap();
        assert!(
            header.get("create").is_none(),
            "create rides the first batch only (issue #752): {header}"
        );

        let manifest = manifest_batch("sake::consolidation", "sake");
        let header: Value = serde_json::from_str(manifest.lines().next().unwrap()).unwrap();
        assert!(
            header.get("create").is_none(),
            "a judgment batch always precedes the manifest: {header}"
        );
        let passage: Value = serde_json::from_str(manifest.lines().nth(1).unwrap()).unwrap();
        let stored: Value = serde_json::from_str(passage["passage"].as_str().unwrap()).unwrap();
        assert_eq!(stored["type"], "consolidation_manifest");
        assert_eq!(stored["version"], crate::format::FORMAT_VERSION);
        assert_eq!(stored["detector"], CONSOLIDATION_DETECTOR);
    }
}
