//! Scatter-gather planning shared by `/recall`, `/query`, and
//! `/sources/search`: the pre-checks, shard-set computation, and
//! outcome triage every fan-out search runs before merging.

use super::*;

/// The shared front half of every fan-out search: the single-instance
/// pre-checks (byte for byte), the direct-name dedup, and the
/// shard-set/per-shard-body computation. `direct` preserves first-
/// appearance order — the same order `cross_targets` seats direct
/// names in.
pub(super) struct Scatter {
    pub(super) direct: Vec<String>,
    /// direct contexts per shard, order preserved within each shard.
    pub(super) per_shard: BTreeMap<usize, Vec<String>>,
    pub(super) shards: Vec<usize>,
}

pub(super) fn plan_scatter(
    state: &RouterState,
    contexts: &[String],
    groups: &[String],
    started_at: Instant,
) -> Result<Scatter, Box<Response>> {
    if contexts.is_empty() && groups.is_empty() {
        return Err(Box::new(api::error(
            ErrorCode::InvalidArgument,
            "'contexts' or 'groups' must name at least one target",
            started_at,
        )));
    }
    for (field, count) in [("contexts", contexts.len()), ("groups", groups.len())] {
        if let Some(refusal) = api::overlong(field, count, started_at) {
            return Err(Box::new(refusal));
        }
    }
    let mut seen = BTreeSet::new();
    let direct: Vec<String> = contexts
        .iter()
        .filter(|name| seen.insert((*name).clone()))
        .cloned()
        .collect();
    let mut per_shard: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for name in &direct {
        let Some(shard) = state.map().shard_of(name) else {
            // Unmapped with no fallback cannot exist anywhere the
            // router reaches — the same first-missing-name refusal a
            // single instance gives, in the same list order.
            return Err(Box::new(api::error(
                ErrorCode::NoContext,
                format!("context '{name}' not found"),
                started_at,
            )));
        };
        per_shard.entry(shard).or_default().push(name.clone());
    }
    let shards: Vec<usize> = if groups.is_empty() {
        per_shard.keys().copied().collect()
    } else {
        // Groups live on every shard (the projected-broadcast
        // invariant), so naming one fans out everywhere.
        state.map().all().collect()
    };
    Ok(Scatter {
        direct,
        per_shard,
        shards,
    })
}

/// Sorts multi-shard failures into the single-instance refusal order:
/// scope refusals over direct names come before existence, existence
/// before group resolution — tie-broken by where each shard's first
/// direct target sits in the request's own order.
pub(super) fn abort_rank(code: Option<&str>) -> u8 {
    match code {
        Some("forbidden") => 0,
        Some("no_context") => 1,
        Some("no_group") => 2,
        _ => 3,
    }
}

/// The fan-out outcome, split three ways: HTTP-answered failures abort
/// the whole request (a shard that answered an error is a context that
/// failed, and one failing context fails a single instance's search
/// whole); transport failures become the labeled `unreached` partials;
/// the rest merge.
pub(super) struct Gathered {
    pub(super) answers: Vec<(usize, Bytes)>,
    pub(super) unreached: Vec<Unreached>,
}

pub(super) fn gather(
    state: &RouterState,
    scatter: &Scatter,
    outcomes: Vec<(usize, Result<ShardAnswer, String>)>,
) -> Result<Gathered, Box<Response>> {
    let mut answers = Vec::new();
    let mut unreached = Vec::new();
    let mut aborts: Vec<(u8, usize, usize, ShardAnswer)> = Vec::new();
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => answers.push((shard, answer.body)),
            Ok(answer) => {
                let code = serde_json::from_slice::<Value>(&answer.body)
                    .ok()
                    .and_then(|body| body.get("code").and_then(Value::as_str).map(str::to_string));
                let first_direct = scatter
                    .per_shard
                    .get(&shard)
                    .and_then(|targets| targets.first())
                    .and_then(|name| scatter.direct.iter().position(|direct| direct == name))
                    .unwrap_or(usize::MAX);
                aborts.push((abort_rank(code.as_deref()), first_direct, shard, answer));
            }
            Err(error) => unreached.push(Unreached {
                shard: state.map().url(shard).to_string(),
                contexts: scatter.per_shard.get(&shard).cloned().unwrap_or_default(),
                error,
            }),
        }
    }
    if let Some((_, _, _, answer)) = aborts
        .into_iter()
        .min_by_key(|(rank, position, shard, _)| (*rank, *position, *shard))
    {
        // The shard's own bytes pass through — same code, same
        // message, same status a single instance would have answered.
        return Err(Box::new(
            (
                answer.status,
                [(header::CONTENT_TYPE, "application/json")],
                answer.body,
            )
                .into_response(),
        ));
    }
    if answers.is_empty() && !unreached.is_empty() {
        return Err(Box::new(unreachable_refusal(&unreached, Instant::now())));
    }
    Ok(Gathered { answers, unreached })
}

/// Builds each shard's request body: the caller's own body with the
/// `contexts` list cut down to what that shard owns. Everything else —
/// groups, cue, limit, the verbatim `after` cursor — is forwarded
/// untouched.
pub(super) fn shard_body(base: &Value, targets: Option<&Vec<String>>) -> Bytes {
    let mut body = base.clone();
    body["contexts"] = json!(targets.cloned().unwrap_or_default());
    Bytes::from(body.to_string())
}
