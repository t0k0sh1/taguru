//! Scatter-gather planning shared by `/recall`, `/query`, and
//! `/sources/search`: the pre-checks, shard-set computation, and
//! outcome triage every fan-out search runs before merging.

use super::*;

/// The shared front half of every fan-out search: the single-instance
/// pre-checks (byte for byte), the direct-id dedup, and the
/// shard-set/per-shard-body computation. `direct` preserves first-
/// appearance order — the same order `cross_targets` seats direct
/// ids in.
pub(super) struct Scatter {
    pub(super) direct: Vec<String>,
    /// direct `context_ids` per shard, order preserved within each
    /// shard.
    pub(super) per_shard: BTreeMap<usize, Vec<String>>,
    pub(super) shards: Vec<usize>,
}

/// Plans the fan-out. Targets are context ids (#965) and the route map
/// speaks names, so each distinct direct id's owner is asked of the
/// shards themselves ([`locate_owner`]; a single-shard map pays
/// nothing); an id no shard holds is the single-instance first-missing
/// refusal, in the same list order.
pub(super) async fn plan_scatter(
    state: &RouterState,
    map: &RouteMap,
    context_ids: &[String],
    groups: &[String],
    headers: &HeaderMap,
    deadline: Deadline,
    started_at: Instant,
) -> Result<Scatter, Box<Response>> {
    if context_ids.is_empty() && groups.is_empty() {
        return Err(Box::new(api::error(
            ErrorCode::InvalidArgument,
            "'context_ids' or 'groups' must name at least one target",
            started_at,
        )));
    }
    for (field, count) in [("context_ids", context_ids.len()), ("groups", groups.len())] {
        if let Some(refusal) = api::overlong(field, count, started_at) {
            return Err(Box::new(refusal));
        }
    }
    if let Some(refusal) = api::invalid_context_ids("context_ids", context_ids, started_at) {
        return Err(Box::new(refusal));
    }
    let mut seen = BTreeSet::new();
    let direct: Vec<String> = context_ids
        .iter()
        .filter(|id| seen.insert((*id).clone()))
        .cloned()
        .collect();
    let mut per_shard: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for id in &direct {
        match locate_owner(state, map, id, headers, deadline, started_at).await {
            Located::Shard(shard) => per_shard.entry(shard).or_default().push(id.clone()),
            Located::Missing => {
                return Err(Box::new(api::error(
                    ErrorCode::NoContext,
                    format!("context '{id}' not found"),
                    started_at,
                )));
            }
            Located::Answered(response) => return Err(Box::new(response)),
            // No reachable shard holds it, but some could not be asked:
            // send the id to them, where it surfaces as the labeled
            // `unreached` partial (the shard that can answer for it is
            // down) instead of refusing every other target's results.
            Located::Unreached { shards, .. } => {
                for shard in shards {
                    per_shard.entry(shard).or_default().push(id.clone());
                }
            }
        }
    }
    let shards: Vec<usize> = if groups.is_empty() {
        per_shard.keys().copied().collect()
    } else {
        // Groups live on every shard (the projected-broadcast
        // invariant), so naming one fans out everywhere.
        map.all().collect()
    };
    Ok(Scatter {
        direct,
        per_shard,
        shards,
    })
}

/// Sorts multi-shard failures into the single-instance refusal order:
/// scope refusals over direct names come before existence, existence
/// before `group` resolution — tie-broken by where each shard's first
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
/// the whole request (a shard that answered an error is a `context` that
/// failed, and one failing `context` fails a single instance's search
/// whole); transport failures become the labeled `unreached` partials;
/// the rest merge.
pub(super) struct Gathered {
    pub(super) answers: Vec<(usize, Bytes)>,
    pub(super) unreached: Vec<Unreached>,
}

pub(super) fn gather(
    map: &RouteMap,
    scatter: &Scatter,
    outcomes: Vec<(usize, Result<ShardAnswer, String>)>,
    started_at: Instant,
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
                shard: map.url(shard).to_string(),
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
        return Err(Box::new(passthrough(answer)));
    }
    if answers.is_empty() && !unreached.is_empty() {
        return Err(Box::new(unreachable_refusal(&unreached, started_at)));
    }
    Ok(Gathered { answers, unreached })
}

/// Builds each shard's request body: the caller's own body with the
/// `context_ids` list cut down to what that shard owns. Everything else —
/// `groups`, cue, limit, the verbatim `after` cursor — is forwarded
/// untouched.
pub(super) fn shard_body(base: &Value, targets: Option<&Vec<String>>) -> Bytes {
    let mut body = base.clone();
    body["context_ids"] = json!(targets.cloned().unwrap_or_default());
    Bytes::from(body.to_string())
}
