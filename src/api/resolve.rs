use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::context::{Context, MatchKind, Resolution};
use taguru::deadline::Deadline;

use crate::metrics::{ResolveTier, SearchOp};
use crate::registry::AppState;

use super::{
    AppJson, AppPath, ErrorCode, MAX_MATCH_LIMIT, access_error, clamp, deadline_exceeded, error,
    not_found, ok, search_log_enabled,
};

#[derive(Debug, Deserialize)]
pub struct ResolveRequest {
    pub cue: String,
    /// One-call override of the fuzzy-entry floor — the loosen-and-retry
    /// move after a miss. Omitted means the context's setting.
    pub dice_floor: Option<f64>,
    /// One-call override of the semantic-tier floor, same story.
    pub semantic_floor: Option<f32>,
    /// Candidate-count cap, clamped to the shared ceiling like every
    /// other match endpoint. Omitted means the ceiling itself — resolve
    /// historically served every qualifying candidate, so the default
    /// only stops the flood a short cue raises against a large
    /// vocabulary (every name containing it, in one response body).
    pub limit: Option<usize>,
}

/// One resolve candidate plus the tier that produced it. Lexical scores
/// are coverage/Dice, semantic scores are cosine similarities — ordinal
/// within a tier, never comparable across tiers.
#[derive(Serialize)]
pub struct TieredResolution {
    pub name: String,
    pub score: f64,
    pub tier: &'static str,
    /// The lexical string relation behind the score (exact / alias /
    /// containment / fuzzy) — the caller's warning that a high score
    /// may be a lookalike, not the thing (possible inside impossible
    /// scores 0.8). Absent on semantic candidates, whose score is a
    /// cosine, not a string overlap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    /// The candidate's own gloss — its name plus its heaviest facts,
    /// the same text the semantic tier embeds. This is the evidence
    /// that tells lookalike candidates apart: string overlap says
    /// 東京都 and 京都 are near, their facts say they are different
    /// things. Attached to the top candidates only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gloss: Option<String>,
}

fn lexical_tier(resolutions: Vec<Resolution>) -> Vec<TieredResolution> {
    resolutions
        .into_iter()
        .map(|resolution| TieredResolution {
            name: resolution.name,
            score: resolution.score,
            tier: "lexical",
            kind: Some(resolution.kind.as_str()),
            gloss: None,
        })
        .collect()
}

/// Lexical coverage at or above this is a confident entry — the cue is
/// (at least half of) a stored spelling — and the semantic tier is
/// skipped. Below it, a lexical hit is a fragment collision (蔵 inside
/// 祭りを主催する蔵元 scores 0.11) and MUST NOT silence the semantic
/// tier: in a dense vocabulary some substring almost always matches
/// something, and letting it gate the entry buried 0.55-cosine answers
/// behind 0.11-coverage noise.
pub(super) const LEXICAL_CONFIDENCE: f64 = 0.5;

/// Merges the two entry tiers: lexical candidates keep the front (their
/// scores are string evidence, best first), semantic candidates append
/// for names not already present. Scales stay incomparable, which is
/// what the tier field is for.
pub(super) fn merge_tiers(
    lexical: Vec<Resolution>,
    semantic: &[(String, f32)],
) -> Vec<TieredResolution> {
    let mut merged = lexical_tier(lexical);
    for (name, score) in semantic {
        if merged.iter().any(|candidate| &candidate.name == name) {
            continue;
        }
        merged.push(TieredResolution {
            name: name.clone(),
            score: f64::from(*score),
            tier: "semantic",
            kind: None,
            gloss: None,
        });
    }
    merged
}

/// Enforces the resolve cap after the tiers merge. Semantic candidates
/// sit at the tail, appended precisely because the lexical tier was
/// weak — so overflow comes out of the lexical segment's own tail,
/// never out of the semantic candidates the fallback just earned. Only
/// a limit smaller than the semantic tier itself cuts into it (from
/// its tail too; both tiers arrive best-first).
pub(super) fn trim_to_limit(
    mut served: Vec<TieredResolution>,
    limit: usize,
) -> Vec<TieredResolution> {
    let overflow = served.len().saturating_sub(limit);
    if overflow == 0 {
        return served;
    }
    let semantic = served
        .iter()
        .rev()
        .take_while(|candidate| candidate.tier == "semantic")
        .count();
    let lexical = served.len() - semantic;
    served.drain(lexical - overflow.min(lexical)..lexical);
    served.truncate(limit);
    served
}

/// How many resolve candidates carry their gloss. The head of the list
/// is where a caller weighs lookalikes against each other; decorating a
/// long fuzzy tail would only bloat the response it has already
/// stopped reading.
const GLOSSED_CANDIDATES: usize = 8;

/// Attaches each top candidate's gloss — the evidence a caller needs
/// to tell lookalike names apart without a second round trip. One
/// shared read after the tiers settle, covering lexical and semantic
/// candidates alike; if the context vanished in between, the
/// candidates simply keep gloss = None (the entry answer itself
/// already succeeded, and a decoration must not turn it into an
/// error).
fn attach_glosses(
    state: &AppState,
    name: &str,
    labels: bool,
    mut served: Vec<TieredResolution>,
) -> Vec<TieredResolution> {
    let head = served.len().min(GLOSSED_CANDIDATES);
    if head == 0 {
        return served;
    }
    let glosses = state.read_context(name, |context| {
        served[..head]
            .iter()
            .map(|candidate| {
                if labels {
                    context.label_gloss(&candidate.name, Context::GLOSS_EXAMPLES)
                } else {
                    context.concept_gloss(&candidate.name, Context::GLOSS_FACTS)
                }
            })
            .collect::<Vec<_>>()
    });
    if let Ok(glosses) = glosses {
        for (candidate, gloss) in served.iter_mut().zip(glosses) {
            candidate.gloss = gloss;
        }
    }
    served
}

/// The resolve pipeline's data half — the same tiers, floors, merge
/// inputs, and truncation the endpoint serves, shared verbatim with
/// the explain twin so the two cannot disagree about what would be
/// served. `bounded` is the lexical tier after the flood bound;
/// `overflow` is the tail that bound cut (empty in every non-
/// pathological call), kept so explain can locate a candidate past it.
struct ResolvedTiers {
    bounded: Vec<Resolution>,
    overflow: Vec<Resolution>,
    confident: bool,
    /// What the semantic tier contributed: empty when the lexical tier
    /// was confident, the provider is off, nothing is embedded — or
    /// when a provider failure degraded to lexical-only (logged).
    semantic: Vec<(String, f32)>,
}

/// Runs the entry ladder up to (but not through) the final merge:
/// lexical tiers first; the semantic tier joins whenever they came
/// back empty OR merely fragment-weak (best score under
/// [`LEXICAL_CONFIDENCE`]). `Err` is a response to serve immediately:
/// unknown context, or a provider failure with nothing lexical to
/// degrade to.
#[allow(clippy::result_large_err)] // the Err IS the response served next
fn resolve_tiers(
    state: &AppState,
    name: &str,
    request: &ResolveRequest,
    labels: bool,
    deadline: Deadline,
    started_at: Instant,
) -> Result<ResolvedTiers, Response> {
    let limit = clamp(request.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    // The lexical read sweeps the vocabulary (see resolve_with_fallback's
    // own deadline comment above its call site) — the same unconditional
    // whole-table cost as unreachable_from's full scan, so it runs under
    // block_in_place rather than straight on the async task.
    let mut bounded = match tokio::task::block_in_place(|| {
        state.read_context(name, |context| match (labels, request.dice_floor) {
            (false, Some(floor)) => context.resolve_with_floor(&request.cue, floor),
            (false, None) => context.resolve(&request.cue),
            (true, Some(floor)) => context.resolve_label_with_floor(&request.cue, floor),
            (true, None) => context.resolve_label(&request.cue),
        })
    }) {
        Ok(result) => result,
        Err(failure) => return Err(access_error(state, failure, name, started_at)),
    };
    // Resolutions arrive best-first, so bounding the flood right here
    // keeps the strongest candidates and spares everything downstream
    // (the semantic dedup scan, gloss reads, serialization) the
    // pathological tail. The confidence probe below reads only the
    // first entry, which the bound never touches.
    let overflow = bounded.split_off(bounded.len().min(limit));
    let confident = bounded
        .first()
        .is_some_and(|best| best.score >= LEXICAL_CONFIDENCE);
    let semantic = if confident {
        Vec::new()
    } else {
        // The provider round trip can take hundreds of milliseconds;
        // tell the runtime this thread will block so other tasks
        // migrate off it.
        match tokio::task::block_in_place(|| {
            state.semantic_resolve(name, &request.cue, labels, request.semantic_floor, deadline)
        }) {
            None => return Err(not_found(name, started_at)),
            Some(Ok(semantic)) => semantic,
            // The semantic tier is enrichment once ANY lexical
            // candidate exists: degrade to the weak lexical results
            // and log, rather than failing an answerable request over
            // a provider hiccup.
            Some(Err(message)) if !bounded.is_empty() => {
                tracing::warn!("semantic entry failed (serving weak lexical results): {message}");
                Vec::new()
            }
            Some(Err(message)) if deadline.expired() => {
                tracing::warn!(
                    "semantic entry failed with nothing lexical to fall back on: {message}"
                );
                return Err(error(
                    ErrorCode::Timeout,
                    format!("semantic entry failed: {message}"),
                    started_at,
                ));
            }
            Some(Err(message)) => {
                tracing::warn!(
                    "semantic entry failed with nothing lexical to fall back on: {message}"
                );
                return Err(error(
                    ErrorCode::EmbeddingsFailed,
                    format!("semantic entry failed: {message}"),
                    started_at,
                ));
            }
        }
    };
    Ok(ResolvedTiers {
        bounded,
        overflow,
        confident,
        semantic,
    })
}

/// The full entry ladder: [`resolve_tiers`], then the merge, the trim,
/// and the gloss decoration of what survived.
fn resolve_with_fallback(
    state: &AppState,
    name: &str,
    request: &ResolveRequest,
    labels: bool,
    deadline: Deadline,
    started_at: Instant,
) -> Response {
    // Both resolve handlers land here first, and the lexical read alone
    // takes the registry lock and sweeps the vocabulary before the
    // semantic tier's own deadline checks ever run — fail a spent
    // request fast, the way every other handler pre-flights.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let limit = clamp(request.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    let tiers = match resolve_tiers(state, name, request, labels, deadline, started_at) {
        Ok(tiers) => tiers,
        Err(response) => return response,
    };
    let served = trim_to_limit(merge_tiers(tiers.bounded, &tiers.semantic), limit);
    let served = attach_glosses(state, name, labels, served);
    let op = if labels {
        SearchOp::ResolveLabel
    } else {
        SearchOp::Resolve
    };
    let tier = resolve_tier_of(&served);
    state.note_search(op, name, served.is_empty());
    state.metrics().record_resolve_tier(tier);
    if search_log_enabled() {
        tracing::info!(
            target: "taguru::search",
            context = %name,
            op = if labels { "resolve_label" } else { "resolve" },
            cue = %request.cue,
            hits = served.len(),
            tier = tier.as_str(),
            top_score = served.first().map_or(0.0, |best| best.score),
            "search",
        );
    }
    ok(served, started_at)
}

/// Classifies a resolve by what was actually served, so every serve
/// path lands in exactly one bucket: any semantic candidate means the
/// semantic tier answered; otherwise the best (first) lexical score
/// decides confident versus fragment-weak; nothing at all is a miss.
pub(super) fn resolve_tier_of(served: &[TieredResolution]) -> ResolveTier {
    if served.is_empty() {
        ResolveTier::Miss
    } else if served.iter().any(|candidate| candidate.tier == "semantic") {
        ResolveTier::Semantic
    } else if served[0].score >= LEXICAL_CONFIDENCE {
        ResolveTier::Lexical
    } else {
        ResolveTier::WeakLexical
    }
}

pub async fn resolve(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    resolve_with_fallback(&state, &name, &request, false, deadline, started_at)
}

pub async fn resolve_label(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    resolve_with_fallback(&state, &name, &request, true, deadline, started_at)
}

#[derive(Debug, Deserialize)]
pub struct ExplainResolveRequest {
    pub cue: String,
    /// The name the caller expected among the candidates.
    pub expected: String,
    /// The same one-call overrides resolve takes — explain answers for
    /// the exact call being questioned.
    pub dice_floor: Option<f64>,
    pub semantic_floor: Option<f32>,
    pub limit: Option<usize>,
}

/// How many nearest stored spellings a `not_in_vocabulary` verdict
/// attaches per tier — enough to spot the fork, not a listing.
const NEAREST_SPELLINGS: usize = 5;

/// One verdict for "why didn't (or did) this name come back for this
/// cue": machine-readable in `verdict`, human-readable in `summary`,
/// evidence attached. Every verdict is a 200 — a diagnosed miss is
/// this endpoint's success.
#[derive(Serialize)]
pub struct ResolveExplanation {
    /// `not_in_vocabulary` | `served` | `cue_resolved_exactly` |
    /// `below_floor` | `below_cutoff` | `semantic_not_run` |
    /// `semantic_below_floor`, first match wins in that order.
    pub verdict: &'static str,
    pub summary: String,
    pub cue: String,
    pub expected: String,
    pub in_vocabulary: bool,
    /// The canonical name `expected` maps to (differs when `expected`
    /// is an alias), and how it maps (`exact` / `alias`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical: Option<LexicalExplain>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic: Option<SemanticExplain>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking: Option<ResolveRanking>,
    /// Nearest stored spellings, attached to `not_in_vocabulary` —
    /// the fix (register an alias) is one step away.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nearest: Option<NearestSpellings>,
}

/// The lexical tier's account: the Dice/coverage score the resolver
/// actually gave (cue → canonical) next to the floor in effect —
/// "scored 0.42, floor 0.6" answers "what floor would have shown it".
#[derive(Serialize)]
pub struct LexicalExplain {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    pub floor: f64,
    /// Whether the tier's best candidate was confident (≥ 0.5) — the
    /// predicate that decides if the semantic tier joins at all.
    pub confident: bool,
}

/// The semantic tier's account: whether it joined this call, why not
/// when it could not have, and the expected name's own gloss cosine
/// against the floor when the sweep could run.
#[derive(Serialize)]
pub struct SemanticExplain {
    pub entered: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub floor: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cosine: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    /// The tier serves its top `cap` only (a constant, not a request
    /// knob) — present whenever the sweep ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cap: Option<usize>,
}

/// Where the canonical stands against the served list: its rank when
/// present, and a `limit_to_reach` VERIFIED by rerunning the real
/// serve computation.
#[derive(Serialize)]
pub struct ResolveRanking {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub limit: usize,
    pub served: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit_to_reach: Option<usize>,
}

#[derive(Serialize)]
pub struct NearestSpellings {
    pub lexical: Vec<Resolution>,
    pub semantic: Vec<NearestGloss>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_note: Option<String>,
}

#[derive(Serialize)]
pub struct NearestGloss {
    pub name: String,
    pub cosine: f32,
}

/// What one `read_context` pass gathers for an explain call — the
/// membership probe, the floor in effect, and the cue's whole lexical
/// neighborhood, all under the same lock so they describe one moment.
struct VocabularyView {
    /// `expected` at floor 1.0: only normalized-exact spellings, so
    /// `None` means no entry, no alias, no canonical.
    canonical: Option<Resolution>,
    dice_floor: f64,
    /// The cue at floor 0.0: every lexically related name, best-first
    /// — where the canonical's actual score is read off, floor or no
    /// floor.
    sweep: Vec<Resolution>,
    /// `expected` at floor 0.0 (head only) — the nearest stored
    /// spellings when membership failed.
    nearest: Vec<Resolution>,
}

fn explain_resolve_verdict(
    state: &AppState,
    name: &str,
    request: &ExplainResolveRequest,
    labels: bool,
    deadline: Deadline,
    started_at: Instant,
) -> Response {
    // Both explain handlers land here first, and the view alone runs
    // several lexical sweeps inside read_context (membership, the cue
    // sweep, and possibly the nearest-spellings scan) before the
    // semantic tier's own deadline checks would ever run — fail a spent
    // request fast, the way resolve_with_fallback already does for its
    // own lexical read.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Several lexical sweeps in one read_context (see the comment above
    // this function) — the same unconditional whole-table cost as
    // unreachable_from's full scan, so it runs under block_in_place
    // rather than straight on the async task.
    let view = match tokio::task::block_in_place(|| {
        state.read_context(name, |context| {
            let resolve_floor = |cue: &str, floor: f64| {
                if labels {
                    context.resolve_label_with_floor(cue, floor)
                } else {
                    context.resolve_with_floor(cue, floor)
                }
            };
            // Membership means the expected string IS a stored spelling —
            // exact or alias, nothing looser. The floor cannot express
            // that (it gates only the fuzzy tier; containment sails past
            // any floor), so the kind is the test.
            let canonical = resolve_floor(&request.expected, 1.0)
                .into_iter()
                .find(|candidate| matches!(candidate.kind, MatchKind::Exact | MatchKind::Alias));
            let dice_floor = request.dice_floor.map_or_else(
                || context.dice_floor(),
                // NaN falls to the strictest floor, exactly as the scorer
                // treats it (`clamp_unit_or`'s nan_fallback).
                |floor| {
                    if floor.is_nan() {
                        1.0
                    } else {
                        floor.clamp(0.0, 1.0)
                    }
                },
            );
            let sweep = resolve_floor(&request.cue, 0.0);
            let nearest = if canonical.is_none() {
                let mut nearest = resolve_floor(&request.expected, 0.0);
                nearest.truncate(NEAREST_SPELLINGS);
                nearest
            } else {
                Vec::new()
            };
            VocabularyView {
                canonical,
                dice_floor,
                sweep,
                nearest,
            }
        })
    }) {
        Ok(view) => view,
        Err(failure) => return access_error(state, failure, name, started_at),
    };

    let namespace = if labels { "label" } else { "concept" };
    let Some(membership) = view.canonical else {
        // Verdict 1: not in the vocabulary at all. Attach the nearest
        // stored spellings, lexical and semantic alike, so the repair
        // (register an alias) needs no second round trip.
        let (glosses, semantic_note) = match tokio::task::block_in_place(|| {
            state.semantic_resolve(name, &request.expected, labels, Some(0.0), deadline)
        }) {
            None => return not_found(name, started_at),
            Some(Ok(matches)) => (matches, None),
            Some(Err(message)) => (Vec::new(), Some(message)),
        };
        let explanation = ResolveExplanation {
            verdict: "not_in_vocabulary",
            summary: format!(
                "'{}' is not in this context's {namespace} vocabulary — no entry, no \
                 alias, no canonical spelling; the nearest stored spellings are attached",
                request.expected
            ),
            cue: request.cue.clone(),
            expected: request.expected.clone(),
            in_vocabulary: false,
            canonical: None,
            expected_kind: None,
            lexical: None,
            semantic: None,
            ranking: None,
            nearest: Some(NearestSpellings {
                lexical: view.nearest,
                semantic: glosses
                    .into_iter()
                    .map(|(name, cosine)| NearestGloss { name, cosine })
                    .collect(),
                semantic_note,
            }),
        };
        state.note_read(name, true);
        return ok(explanation, started_at);
    };
    let canonical = membership.name.clone();
    let expected_kind = membership.kind.as_str();

    // The pipeline itself, with the request's own overrides — the
    // serve boundary below is exactly the resolve call being explained.
    let resolve_request = ResolveRequest {
        cue: request.cue.clone(),
        dice_floor: request.dice_floor,
        semantic_floor: request.semantic_floor,
        limit: request.limit,
    };
    let limit = clamp(resolve_request.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    let tiers = match resolve_tiers(state, name, &resolve_request, labels, deadline, started_at) {
        Ok(tiers) => tiers,
        Err(response) => return response,
    };
    let served = trim_to_limit(merge_tiers(tiers.bounded.clone(), &tiers.semantic), limit);
    let full: Vec<Resolution> = tiers
        .bounded
        .iter()
        .chain(tiers.overflow.iter())
        .cloned()
        .collect();
    let merged = merge_tiers(full.clone(), &tiers.semantic);

    let served_at = served
        .iter()
        .position(|candidate| candidate.name == canonical);
    let merged_at = merged
        .iter()
        .position(|candidate| candidate.name == canonical);
    // Whether the effective-floor resolve admitted the canonical at
    // all — the floor gates ONLY the fuzzy tier (exact and containment
    // pass it by construction), so "scored under the floor" is
    // precisely "visible at floor 0, absent here".
    let admitted = full.iter().any(|candidate| candidate.name == canonical);
    let target = view
        .sweep
        .iter()
        .find(|candidate| candidate.name == canonical);
    // The exact tier answers alone when the cue IS a stored spelling —
    // nothing else is ever scored, however close. An all-exact sweep
    // is that early return's signature.
    let exact_shortcut = target.is_none()
        && !view.sweep.is_empty()
        && view
            .sweep
            .iter()
            .all(|candidate| matches!(candidate.kind, MatchKind::Exact | MatchKind::Alias));

    // The gloss lane's account — the cue embedding is already cached
    // when the pipeline entered the tier, so this adds no provider
    // round trip beyond the one targeted scoring explain budgets for.
    let lane = match tokio::task::block_in_place(|| {
        state.explain_semantic_resolve(
            name,
            &request.cue,
            &canonical,
            labels,
            request.semantic_floor,
            deadline,
        )
    }) {
        Some(lane) => lane,
        None => return not_found(name, started_at),
    };
    let entered = !tiers.confident;
    let semantic = {
        use crate::registry::GlossLaneReport;
        let mut explain = SemanticExplain {
            entered,
            reason: None,
            floor: None,
            cosine: None,
            rank: None,
            cap: None,
        };
        match &lane {
            GlossLaneReport::Off => {
                explain.reason = Some("no embedding provider is configured".to_string());
            }
            GlossLaneReport::ModelChanged { stored, current } => {
                explain.reason = Some(format!(
                    "gloss vectors belong to model '{stored}' but the provider is \
                     '{current}' — awaiting re-embed"
                ));
            }
            GlossLaneReport::WidthChanged { stored, current } => {
                explain.reason = Some(format!(
                    "gloss vectors are {stored}-dimensional but the model now answers \
                     {current} (a dimensions setting changed behind its name) — \
                     awaiting re-embed"
                ));
            }
            GlossLaneReport::EmptyTable => {
                explain.reason = Some(format!(
                    "no {namespace} gloss vectors exist yet — no embedding refresh has run"
                ));
            }
            GlossLaneReport::QueryEmbeddingFailed(error) => {
                explain.reason = Some(format!("the cue embedding failed: {error}"));
            }
            GlossLaneReport::Ran {
                floor,
                cosine,
                rank,
                cap,
                ..
            } => {
                explain.floor = Some(*floor);
                explain.cosine = *cosine;
                explain.rank = *rank;
                explain.cap = Some(*cap);
                if cosine.is_none() {
                    explain.reason = Some(format!(
                        "'{canonical}' has no gloss vector yet — added after the last refresh"
                    ));
                }
            }
        }
        if !entered && explain.reason.is_none() {
            explain.reason = Some(format!(
                "the lexical tier was confident (best score ≥ {LEXICAL_CONFIDENCE}), \
                 so the semantic tier never joined"
            ));
        }
        explain
    };

    // The smallest VERIFIED limit that serves the canonical: rerun the
    // real serve computation (bound, merge, trim). `serve_at` is NOT
    // monotonic in `limit` — trim_to_limit spends overflow out of the
    // lexical tail first, so a limit small enough that semantic
    // candidates alone can fill it wipes the ENTIRE bounded lexical
    // segment, and a canonical that is also a semantic candidate can
    // already be served far below its unbounded `merged_at` rank.
    // Symmetrically, the instant a growing limit pulls the canonical's
    // own lexical rank into `bounded`, merge_tiers's dedup drops the
    // semantic duplicate that had been carrying it, and the lexical
    // rank alone can take a few more limits to clear the trim — a dip
    // right after `merged_at + 1`. Both swings are bounded by how many
    // semantic candidates there are to dedupe against
    // (`SEMANTIC_RESOLVE_LIMIT`, capped at 5), so two short linear
    // scans — one from 1, one just above `merged_at` — cover every
    // limit where the verdict can flip before the doubling search
    // (valid once neither effect can recur) takes over.
    let serve_at = |limit: usize| -> Vec<TieredResolution> {
        let mut bounded = full.clone();
        bounded.truncate(limit);
        trim_to_limit(merge_tiers(bounded, &tiers.semantic), limit)
    };
    let serves_canonical = |candidate: usize| {
        serve_at(candidate)
            .iter()
            .any(|served| served.name == canonical)
    };
    let semantic_span = tiers.semantic.len();
    let limit_to_reach = if served_at.is_some() {
        Some(limit)
    } else {
        (1..=semantic_span.min(merged.len()))
            .find(|&candidate| serves_canonical(candidate))
            .or_else(|| {
                merged_at.and_then(|at| {
                    let window_end = (at + 1 + semantic_span).min(merged.len());
                    ((at + 1)..=window_end).find(|&candidate| serves_canonical(candidate))
                })
            })
            .or_else(|| {
                merged_at.and_then(|at| {
                    let mut candidate = (at + 1 + semantic_span + 1).min(merged.len());
                    for _ in 0..8 {
                        if candidate > merged.len() {
                            return None;
                        }
                        if serves_canonical(candidate) {
                            return Some(candidate);
                        }
                        candidate = candidate.saturating_mul(2).min(merged.len());
                    }
                    None
                })
            })
    };

    let target_score = target.map(|candidate| candidate.score);
    let target_kind = target.map(|candidate| candidate.kind.as_str());
    use crate::registry::GlossLaneReport;
    let (verdict, summary) = if let Some(at) = served_at {
        (
            "served",
            format!(
                "served: '{canonical}' ranked {} of {} at limit {limit} ({} tier)",
                at + 1,
                served.len(),
                served[at].tier
            ),
        )
    } else if exact_shortcut {
        (
            "cue_resolved_exactly",
            format!(
                "the cue is an exact spelling of '{}' — the exact tier answers alone, \
                 so '{canonical}' was never scored against it; a reworded cue is the \
                 way in",
                view.sweep[0].name
            ),
        )
    } else if let Some(score) = target_score.filter(|_| !admitted) {
        (
            "below_floor",
            format!(
                "the resolver scored '{canonical}' {score:.4} ({}) against dice_floor \
                 {:.4} — a floor at or under {score:.4} would have shown it",
                target_kind.unwrap_or("fuzzy"),
                view.dice_floor
            ),
        )
    } else if target_score.is_some() {
        // At or above the floor yet not served: lost on the limit.
        let reach = match limit_to_reach {
            Some(reach) => format!("limit {reach} reaches it"),
            None => "no limit reaches it".to_string(),
        };
        (
            "below_cutoff",
            format!(
                "'{canonical}' passed the floor but ranked {} of {} — the request \
                 served {limit}; {reach}",
                merged_at.map_or(0, |at| at + 1),
                merged.len()
            ),
        )
    } else if !entered {
        (
            "semantic_not_run",
            format!(
                "'{canonical}' shares no spelling with the cue, and the semantic tier \
                 never joined: the lexical tier was confident (best score ≥ \
                 {LEXICAL_CONFIDENCE})"
            ),
        )
    } else {
        match &lane {
            GlossLaneReport::Ran {
                floor,
                cosine: Some(cosine),
                rank,
                passing,
                cap,
            } => {
                if cosine < floor {
                    (
                        "semantic_below_floor",
                        format!(
                            "'{canonical}' shares no spelling with the cue; its gloss \
                             cosine {cosine:.4} sits under the semantic floor {floor:.4} \
                             — a floor at or under {cosine:.4} would have shown it"
                        ),
                    )
                } else if rank.is_some_and(|rank| rank > *cap) {
                    (
                        "below_cutoff",
                        format!(
                            "'{canonical}' cleared the semantic floor (cosine \
                             {cosine:.4}) but ranked {} of {passing} — the semantic \
                             tier serves its top {cap} only",
                            rank.unwrap_or_default()
                        ),
                    )
                } else {
                    // Cleared the floor, within the cap, yet missing:
                    // the request's own limit trimmed it out.
                    let reach = match limit_to_reach {
                        Some(reach) => format!("limit {reach} reaches it"),
                        None => "no limit reaches it".to_string(),
                    };
                    (
                        "below_cutoff",
                        format!(
                            "'{canonical}' earned a semantic seat (cosine {cosine:.4}) \
                             but the request served only {limit}; {reach}"
                        ),
                    )
                }
            }
            _ => (
                "semantic_not_run",
                format!(
                    "'{canonical}' shares no spelling with the cue, and the semantic \
                     tier could not score it: {}",
                    semantic.reason.as_deref().unwrap_or("it did not run")
                ),
            ),
        }
    };

    let explanation = ResolveExplanation {
        verdict,
        summary,
        cue: request.cue.clone(),
        expected: request.expected.clone(),
        in_vocabulary: true,
        canonical: Some(canonical.clone()),
        expected_kind: Some(expected_kind),
        lexical: Some(LexicalExplain {
            score: target_score,
            kind: target_kind,
            floor: view.dice_floor,
            confident: tiers.confident,
        }),
        semantic: Some(semantic),
        ranking: Some(ResolveRanking {
            rank: merged_at.map(|at| at + 1),
            tier: merged_at.map(|at| merged[at].tier),
            score: merged_at.map(|at| merged[at].score),
            limit,
            served: served_at.is_some(),
            limit_to_reach,
        }),
        nearest: None,
    };
    state.note_read(name, false);
    ok(explanation, started_at)
}

/// `POST /contexts/{name}/resolve/explain` — "why didn't this concept
/// come back for this cue" in one call: the same tiers, floors, and
/// trims the resolve endpoint runs, with the expected name located in
/// (or placed against) each of them. Read-only.
pub async fn explain_resolve(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ExplainResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    explain_resolve_verdict(&state, &name, &request, false, deadline, started_at)
}

/// `POST /contexts/{name}/resolve_label/explain` — explain_resolve,
/// for relation labels.
pub async fn explain_resolve_label(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ExplainResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    explain_resolve_verdict(&state, &name, &request, true, deadline, started_at)
}
