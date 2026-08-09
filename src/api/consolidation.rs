//! `POST /contexts/{name}/consolidation/audit` (ADR 0012): merge,
//! contradiction, and staleness candidates in one response, sections
//! selected by the caller's `checks` — every candidate carrying a
//! content fingerprint over its own evidence, because the fingerprint
//! is the judgment artifact's identity and its staleness mechanism at
//! once (ADR 0012 §5). Candidates for review, never verdicts: nothing
//! here applies anything, and the server never judges — the judging
//! client reuses stored judgments for unchanged fingerprints and pays
//! an LLM only for what moved.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;
use serde::{Deserialize, Serialize};

use taguru::context::{Context, MergeEvidence, ObjectRow};
use taguru::deadline::Deadline;

use crate::hash::{FNV1A_OFFSET, fnv1a_fold};
use crate::metrics::ErrorKind;
use crate::registry::{AccessError, AppState};

use super::vocabulary::vocabulary_audit;
use super::{
    AppJson, AppPath, ErrorCode, MAX_MATCH_LIMIT, access_error, clamp, deadline_exceeded, error,
    not_found, ok, overlong,
};

/// The detector stamp (the `louvain-cc/1` precedent): fingerprints are
/// comparable only within one detector version, and a bump invalidates
/// every stored judgment LOUDLY on the judging side rather than
/// silently diffing incomparable evidence.
pub(crate) const CONSOLIDATION_DETECTOR: &str = "consolidation/1";

/// Per-list ceiling on merge evidence facts when the caller does not
/// choose one — totals stay exact either way (no silent caps).
pub(super) const DEFAULT_EVIDENCE_CAP: usize = 20;

#[derive(Debug, Deserialize)]
pub struct ConsolidationAuditRequest {
    /// Which sections to compute — required and non-empty: every
    /// section is O(edges) or worse, so the caller names what it pays
    /// for (ADR 0012 §4).
    pub checks: Vec<String>,
    /// Per-section candidate ceiling. Omitted means 100, capped at
    /// 1000; each section's `total` is exact regardless.
    pub limit: Option<usize>,
    /// Per-list ceiling on merge evidence facts (default 20).
    pub evidence_cap: Option<usize>,
    /// Merge: lexical twin floor (default 0.6, `vocabulary/audit`'s).
    pub dice_floor: Option<f64>,
    /// Merge: semantic twin floor (default 0.6).
    pub cosine_floor: Option<f32>,
    /// Staleness: minimum gap (seconds) before an edge is a candidate.
    /// No default pretends to know the corpus's tempo — omitted means
    /// 0, every positive gap reports, ranked worst first.
    pub floor_secs: Option<u64>,
}

#[derive(Serialize, Deserialize)]
pub struct ConsolidationAudit {
    pub detector: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge: Option<MergeSection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contradiction: Option<ContradictionSection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub staleness: Option<StalenessSection>,
}

#[derive(Serialize, Deserialize)]
pub struct MergeSection {
    /// Candidates found — exact, however many the page below carries.
    pub total: usize,
    pub candidates: Vec<MergeCandidate>,
    /// Why the semantic tier contributed nothing, when it didn't.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_note: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct MergeCandidate {
    pub a: String,
    pub b: String,
    /// Which sweep surfaced the pair: `lexical` (bigram Dice) or
    /// `semantic` (gloss-embedding cosine).
    pub tier: String,
    pub name_score: f64,
    /// Each side's declared types (live `schema:type` objects), when a
    /// schema is installed — a cross-type pair is a strong dismissal
    /// signal the judge should see (ADR 0012 §7).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub types_a: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub types_b: Vec<String>,
    #[serde(flatten)]
    pub evidence: MergeEvidence,
    pub fingerprint: String,
}

#[derive(Serialize, Deserialize)]
pub struct ContradictionSection {
    pub total: usize,
    pub candidates: Vec<ContradictionCandidate>,
}

/// One contradiction candidate — the two kinds the issue names, kept
/// as one ranked list with a discriminator (ADR 0012 §4).
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContradictionCandidate {
    /// ≥ 2 live objects under one `(subject, label)`.
    Objects {
        subject: String,
        label: String,
        functional_tendency: f64,
        objects: Vec<ObjectOut>,
        fingerprint: String,
    },
    /// One edge whose per-source attributions disagree in sign.
    Contested {
        subject: String,
        label: String,
        object: String,
        weight: f64,
        count: u64,
        supporting_sources: Vec<String>,
        disputing_sources: Vec<String>,
        fingerprint: String,
    },
}

/// [`ObjectRow`] plus the registry's half: the row's latest assertion
/// time (`date ?? stored_at`, max over its sources), absent when none
/// of them is datable — ordered rows are what make "supersession or
/// conflict" askable at a glance (ADR 0011 §7).
#[derive(Serialize, Deserialize)]
pub struct ObjectOut {
    pub object: String,
    pub weight: f64,
    pub count: u64,
    pub sources: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest: Option<u64>,
}

#[derive(Serialize, Deserialize)]
pub struct StalenessSection {
    pub total: usize,
    /// Live edges whose sources carry no effective time at all —
    /// reported so "no candidates" and "nothing was datable" never
    /// look alike (ADR 0012 §4).
    pub undatable: usize,
    pub candidates: Vec<StaleCandidate>,
}

#[derive(Serialize, Deserialize)]
pub struct StaleCandidate {
    pub subject: String,
    pub label: String,
    pub object: String,
    /// The edge's own latest attesting effective time.
    pub latest: u64,
    /// The newest effective time across the subject's live edges.
    pub neighborhood_latest: u64,
    pub gap: u64,
    pub sources: Vec<String>,
    pub fingerprint: String,
}

fn fingerprint_hex(digest: u64) -> String {
    format!("{digest:016x}")
}

fn fold_field(digest: u64, field: &str) -> u64 {
    fnv1a_fold(fnv1a_fold(digest, field.bytes()), [0u8])
}

pub async fn audit_consolidation(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ConsolidationAuditRequest>,
) -> Response {
    let started_at = Instant::now();
    // Before the dedup: `dedup` folds only CONSECUTIVE repeats, so an
    // alternating list would otherwise pass both guards below at any
    // length the body cap admits.
    if let Some(refusal) = overlong("checks", request.checks.len(), started_at) {
        return refusal;
    }
    let mut checks: Vec<&str> = request.checks.iter().map(String::as_str).collect();
    checks.dedup();
    if checks.is_empty() {
        return error(
            ErrorCode::InvalidArgument,
            "'checks' must name at least one of merge, contradiction, staleness — \
             every section is a full-graph pass, so the caller names what it pays for",
            started_at,
        );
    }
    if let Some(unknown) = checks
        .iter()
        .find(|check| !matches!(**check, "merge" | "contradiction" | "staleness"))
    {
        return error(
            ErrorCode::InvalidArgument,
            format!("unknown check '{unknown}' — merge, contradiction, and staleness exist"),
            started_at,
        );
    }
    let limit = clamp(request.limit, 100, MAX_MATCH_LIMIT);
    let evidence_cap = clamp(request.evidence_cap, DEFAULT_EVIDENCE_CAP, MAX_MATCH_LIMIT);
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }

    let wants = |check: &str| checks.contains(&check);
    // Pre-`read_context` resolutions, in the standing order: the
    // hidden label (merge evidence exclusion + type attachment), and
    // the effective-time map (contradiction ordering, staleness) from
    // the passage store, whose locks must never nest inside the
    // entry's read path.
    let hidden = tokio::task::block_in_place(|| state.hidden_label(&name));
    let effective: HashMap<String, u64> = if wants("contradiction") || wants("staleness") {
        match tokio::task::block_in_place(|| state.source_effective_times(&name)) {
            None => return not_found(&name, started_at),
            Some(Ok(map)) => map,
            Some(Err(io_error)) => {
                tracing::warn!(context = %name, "source metadata read failed: {io_error}");
                state.metrics().record_error(ErrorKind::Load);
                return error(
                    ErrorCode::Internal,
                    format!("context '{name}' source metadata could not be read — see server logs"),
                    started_at,
                );
            }
        }
    } else {
        HashMap::new()
    };

    let merge = if wants("merge") {
        match tokio::task::block_in_place(|| {
            merge_section(
                &state,
                &name,
                request.dice_floor.unwrap_or(0.6),
                request.cosine_floor.unwrap_or(0.6),
                evidence_cap,
                limit,
                hidden,
                deadline,
            )
        }) {
            Ok(section) => Some(section),
            Err(failure) => return access_error(&state, failure, &name, started_at),
        }
    } else {
        None
    };

    let contradiction = if wants("contradiction") {
        match tokio::task::block_in_place(|| {
            state.read_context(&name, |context| {
                contradiction_section(context, &effective, limit, deadline)
            })
        })
        .and_then(std::convert::identity)
        {
            Ok(section) => Some(section),
            Err(failure) => return access_error(&state, failure, &name, started_at),
        }
    } else {
        None
    };

    let staleness = if wants("staleness") {
        match tokio::task::block_in_place(|| {
            state.read_context(&name, |context| {
                staleness_section(context, &effective, request.floor_secs.unwrap_or(0), limit)
            })
        }) {
            Ok(section) => Some(section),
            Err(failure) => return access_error(&state, failure, &name, started_at),
        }
    } else {
        None
    };

    ok(
        ConsolidationAudit {
            detector: CONSOLIDATION_DETECTOR.to_string(),
            merge,
            contradiction,
            staleness,
        },
        started_at,
    )
}

/// The merge section: the twin sweep (`vocabulary_audit`'s own core —
/// lexical + semantic, type-name concepts already excluded) as the
/// candidate generator, each surviving pair corroborated structurally
/// (ADR 0012 §4). A pair both tiers surface reports once, under the
/// lexical tier (deterministic, and the lexical score is the
/// explainable one).
#[allow(clippy::too_many_arguments)]
pub(super) fn merge_section(
    state: &AppState,
    name: &str,
    dice_floor: f64,
    cosine_floor: f32,
    evidence_cap: usize,
    limit: usize,
    hidden: Option<&'static str>,
    deadline: Deadline,
) -> Result<MergeSection, AccessError> {
    let vocabulary = vocabulary_audit(state, name, dice_floor, cosine_floor, deadline)?;
    let mut pairs: Vec<(String, String, f64, &'static str)> = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for twin in vocabulary.lexical_concepts {
        let key = if twin.a <= twin.b {
            (twin.a.clone(), twin.b.clone())
        } else {
            (twin.b.clone(), twin.a.clone())
        };
        if seen.insert(key) {
            pairs.push((twin.a, twin.b, twin.score, "lexical"));
        }
    }
    for twin in vocabulary.semantic_concepts {
        let key = if twin.a <= twin.b {
            (twin.a.clone(), twin.b.clone())
        } else {
            (twin.b.clone(), twin.a.clone())
        };
        if seen.insert(key) {
            pairs.push((twin.a, twin.b, f64::from(twin.score), "semantic"));
        }
    }

    let excluded: Vec<&str> = hidden.into_iter().collect();
    let mut candidates = state.read_context(name, |context| {
        pairs
            .into_iter()
            .filter_map(|(a, b, name_score, tier)| {
                let evidence = context.merge_evidence(&a, &b, evidence_cap, &excluded)?;
                let (types_a, types_b) = match hidden {
                    Some(label) => (
                        context.concept_types(&a, label),
                        context.concept_types(&b, label),
                    ),
                    None => (Vec::new(), Vec::new()),
                };
                Some(MergeCandidate {
                    fingerprint: fingerprint_hex(evidence.fingerprint),
                    a,
                    b,
                    tier: tier.to_string(),
                    name_score,
                    types_a,
                    types_b,
                    evidence,
                })
            })
            .collect::<Vec<_>>()
    })?;

    // Structure first, name score second: shared facts are the
    // evidence a merge stands on; a bare spelling twin still reports,
    // ranked low (ADR 0012 §4).
    candidates.sort_by(|x, y| {
        y.evidence
            .overlap
            .total_cmp(&x.evidence.overlap)
            .then_with(|| y.name_score.total_cmp(&x.name_score))
            .then_with(|| (&x.a, &x.b).cmp(&(&y.a, &y.b)))
    });
    let total = candidates.len();
    candidates.truncate(limit);
    Ok(MergeSection {
        total,
        candidates,
        semantic_note: vocabulary.semantic_note,
    })
}

/// The contradiction section: the lib's grouped pass and sign scan,
/// each object row joined with its latest assertion time, multi-object
/// groups ranked by measured functional tendency with contested edges
/// after them (a dispute is already ordered evidence; the grouped kind
/// is where ranking earns its keep).
pub(super) fn contradiction_section(
    context: &Context,
    effective: &HashMap<String, u64>,
    limit: usize,
    deadline: Deadline,
) -> Result<ContradictionSection, AccessError> {
    let latest_of = |row: &ObjectRow| -> Option<u64> {
        row.sources
            .iter()
            .filter_map(|source| effective.get(source).copied())
            .max()
    };
    let mut groups = context
        .contradiction_groups(deadline)
        .map_err(|_| AccessError::DeadlineExceeded)?;
    groups.sort_by(|a, b| {
        b.functional_tendency
            .total_cmp(&a.functional_tendency)
            .then_with(|| (&a.label, &a.subject).cmp(&(&b.label, &b.subject)))
    });
    let conflicts = context
        .sign_conflicts(deadline)
        .map_err(|_| AccessError::DeadlineExceeded)?;

    let total = groups.len() + conflicts.len();
    let mut candidates: Vec<ContradictionCandidate> = Vec::with_capacity(limit.min(total));
    for group in groups {
        if candidates.len() == limit {
            break;
        }
        candidates.push(ContradictionCandidate::Objects {
            fingerprint: fingerprint_hex(group.fingerprint),
            subject: group.subject,
            label: group.label,
            functional_tendency: group.functional_tendency,
            objects: group
                .objects
                .into_iter()
                .map(|row| ObjectOut {
                    latest: latest_of(&row),
                    object: row.object,
                    weight: row.weight,
                    count: row.count,
                    sources: row.sources,
                })
                .collect(),
        });
    }
    for conflict in conflicts {
        if candidates.len() == limit {
            break;
        }
        candidates.push(ContradictionCandidate::Contested {
            fingerprint: fingerprint_hex(conflict.fingerprint),
            subject: conflict.subject,
            label: conflict.label,
            object: conflict.object,
            weight: conflict.weight,
            count: conflict.count,
            supporting_sources: conflict.supporting_sources,
            disputing_sources: conflict.disputing_sources,
        });
    }
    Ok(ContradictionSection { total, candidates })
}

/// The staleness section: edges left behind by their own subject's
/// neighborhood, the gap measured in assertion time (ADR 0012 §4).
/// Undated edges are counted, never guessed at.
pub(super) fn staleness_section(
    context: &Context,
    effective: &HashMap<String, u64>,
    floor_secs: u64,
    limit: usize,
) -> StalenessSection {
    let live: Vec<taguru::context::Association> = context
        .query_any(&[], &[], &[])
        .into_iter()
        .filter(|association| association.count > 0)
        .collect();

    // Pass 1: each edge's own latest, and each subject's newest.
    let mut undatable = 0usize;
    let mut edge_latest: Vec<Option<u64>> = Vec::with_capacity(live.len());
    let mut subject_latest: HashMap<&str, u64> = HashMap::new();
    for association in &live {
        let latest = association
            .attributions
            .iter()
            .filter_map(|attribution| effective.get(&attribution.source).copied())
            .max();
        match latest {
            Some(time) => {
                let newest = subject_latest
                    .entry(association.subject.as_str())
                    .or_insert(time);
                *newest = (*newest).max(time);
            }
            None => undatable += 1,
        }
        edge_latest.push(latest);
    }

    // Pass 2: the gap against the subject's newest.
    let mut candidates: Vec<StaleCandidate> = live
        .iter()
        .zip(&edge_latest)
        .filter_map(|(association, latest)| {
            let latest = (*latest)?;
            let neighborhood_latest = *subject_latest.get(association.subject.as_str())?;
            let gap = neighborhood_latest.saturating_sub(latest);
            if gap == 0 || gap < floor_secs {
                return None;
            }
            let mut digest = fold_field(FNV1A_OFFSET, "stale");
            digest = fold_field(digest, &association.subject);
            digest = fold_field(digest, &association.label);
            digest = fold_field(digest, &association.object);
            digest = fnv1a_fold(digest, latest.to_le_bytes());
            digest = fnv1a_fold(digest, neighborhood_latest.to_le_bytes());
            Some(StaleCandidate {
                subject: association.subject.clone(),
                label: association.label.clone(),
                object: association.object.clone(),
                latest,
                neighborhood_latest,
                gap,
                sources: association
                    .attributions
                    .iter()
                    .map(|attribution| attribution.source.clone())
                    .collect(),
                fingerprint: fingerprint_hex(digest),
            })
        })
        .collect();
    candidates.sort_by(|a, b| {
        b.gap
            .cmp(&a.gap)
            .then_with(|| (&a.subject, &a.label, &a.object).cmp(&(&b.subject, &b.label, &b.object)))
    });
    let total = candidates.len();
    candidates.truncate(limit);
    StalenessSection {
        total,
        undatable,
        candidates,
    }
}
