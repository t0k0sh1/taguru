//! Cross-shard merges for `/recall`, `/query`, `/sources/search`, and
//! `GET /contexts`: combine each shard's already-cursored page with
//! the exact single-instance comparator so the union is the page one
//! instance would have answered.

use super::*;

/// Re-serializes an extracted request into the base JSON each shard's
/// body is projected from. A failure is a router bug (these types
/// serialize by construction), surfaced as `internal`.
fn reserialize<T: Serialize>(request: &T) -> Result<Value, Box<Response>> {
    serde_json::to_value(request).map_err(|error| {
        Box::new(api::error(
            ErrorCode::Internal,
            format!("could not re-serialize the request: {error}"),
            Instant::now(),
        ))
    })
}

pub(super) async fn cross_recall(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppJson(request): api::AppJson<api::CrossRecallRequest>,
) -> Response {
    let limit = request.limit;
    let contexts = request.contexts.clone();
    let groups = request.groups.clone();
    let base = match reserialize(&request) {
        Ok(base) => base,
        Err(refusal) => return *refusal,
    };
    merge_matches(
        state, headers, deadline, "/recall", contexts, groups, limit, base,
    )
    .await
}

pub(super) async fn cross_query(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppJson(request): api::AppJson<api::CrossQueryRequest>,
) -> Response {
    let limit = request.limit;
    let contexts = request.contexts.clone();
    let groups = request.groups.clone();
    let base = match reserialize(&request) {
        Ok(base) => base,
        Err(refusal) => return *refusal,
    };
    merge_matches(
        state, headers, deadline, "/query", contexts, groups, limit, base,
    )
    .await
}

/// The graph-verb merge: concatenate every shard's already-cursored
/// top page, rank with the exact single-instance comparator
/// ([`api::cross_rank`]), cut at the same clamp, sum the totals. Each
/// shard's page is its own top-`limit` past the cursor under the same
/// total order, so the union's top-`limit` is the global page.
#[allow(clippy::too_many_arguments)]
async fn merge_matches(
    state: RouterState,
    headers: HeaderMap,
    deadline: Deadline,
    path: &str,
    contexts: Vec<String>,
    groups: Vec<String>,
    limit: Option<usize>,
    base: Value,
) -> Response {
    let started_at = Instant::now();
    let map = state.map();
    let scatter = match plan_scatter(&map, &contexts, &groups, started_at) {
        Ok(scatter) => scatter,
        Err(refusal) => return *refusal,
    };
    let outcomes = state
        .fan_out(
            &map,
            &scatter.shards,
            Method::POST,
            path,
            &headers,
            |shard| Some(shard_body(&base, scatter.per_shard.get(&shard))),
            deadline,
        )
        .await;
    let gathered = match gather(&map, &scatter, outcomes, started_at) {
        Ok(gathered) => gathered,
        Err(refusal) => return *refusal,
    };
    let mut total = 0usize;
    let mut matches: Vec<api::CrossMatch<api::AssociationOut>> = Vec::new();
    let mut searched: BTreeSet<String> = BTreeSet::new();
    for (shard, body) in gathered.answers {
        match serde_json::from_slice::<ShardEnvelope<api::CrossMatchPage>>(&body) {
            Ok(page) => {
                total += page.result.total;
                matches.extend(page.result.matches);
                if let Some(plan) = page.result.plan {
                    searched.extend(plan.contexts);
                }
            }
            Err(error) => {
                return api::error(
                    ErrorCode::Internal,
                    format!(
                        "shard {} answered an unreadable page: {error}",
                        map.url(shard)
                    ),
                    started_at,
                );
            }
        }
    }
    matches.sort_by(|a, b| {
        api::cross_rank(
            (
                a.inner.weight,
                a.context.as_str(),
                a.inner.subject.as_str(),
                a.inner.label.as_str(),
                a.inner.object.as_str(),
            ),
            (
                b.inner.weight,
                b.context.as_str(),
                b.inner.subject.as_str(),
                b.inner.label.as_str(),
                b.inner.object.as_str(),
            ),
        )
    });
    matches.truncate(api::clamp(
        limit,
        api::DEFAULT_MATCH_LIMIT,
        api::MAX_MATCH_LIMIT,
    ));
    // The merged plan re-seats the union of the shard plans into the
    // single-instance effective order: direct names in request order,
    // group-resolved members after them in name order (the shards
    // resolved the groups — the router only reorders). A dead shard's
    // contexts are honestly absent: they were not searched, and the
    // `unreached` labels beside the plan say why.
    let mut contexts: Vec<String> = scatter
        .direct
        .iter()
        .filter(|name| searched.remove(name.as_str()))
        .cloned()
        .collect();
    contexts.extend(searched);
    router_ok(
        api::CrossMatchPage {
            total,
            matches,
            plan: Some(api::MatchPlan { contexts }),
        },
        gathered.unreached,
        started_at,
    )
}

pub(super) async fn cross_search_passages(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppJson(request): api::AppJson<api::CrossSearchPassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    let base = match reserialize(&request) {
        Ok(base) => base,
        Err(refusal) => return *refusal,
    };
    let map = state.map();
    let scatter = match plan_scatter(&map, &request.contexts, &request.groups, started_at) {
        Ok(scatter) => scatter,
        Err(refusal) => return *refusal,
    };
    let outcomes = state
        .fan_out(
            &map,
            &scatter.shards,
            Method::POST,
            "/sources/search",
            &headers,
            |shard| Some(shard_body(&base, scatter.per_shard.get(&shard))),
            deadline,
        )
        .await;
    let gathered = match gather(&map, &scatter, outcomes, started_at) {
        Ok(gathered) => gathered,
        Err(refusal) => return *refusal,
    };
    // Passage scores don't share a scale across contexts, so the
    // single-instance merge is rank interleaving: (per-context rank,
    // target-list position). Rank is recovered from each shard's page
    // — within it, one context's hits appear in rank order — and the
    // target-list position needs only the RELATIVE order of the
    // searched contexts: direct names first, in request order, then
    // group-resolved members in name order, which is exactly how
    // `cross_targets` seats them. The shard plans name every searched
    // context (hits alone would miss the empty-handed ones), so the
    // seat map and the merged plan both come from them.
    let mut pool: Vec<(usize, api::CrossMatch<api::PassageHit>)> = Vec::new();
    let mut plan_entries: Vec<api::SearchContextPlan> = Vec::new();
    for (shard, body) in gathered.answers {
        match serde_json::from_slice::<ShardEnvelope<api::CrossPassagePage>>(&body) {
            Ok(page) => {
                let mut rank_of: BTreeMap<String, usize> = BTreeMap::new();
                for hit in page.result.hits {
                    let rank = rank_of.entry(hit.context.clone()).or_insert(0);
                    let seat = *rank;
                    *rank += 1;
                    pool.push((seat, hit));
                }
                plan_entries.extend(page.result.plan.contexts);
            }
            Err(error) => {
                return api::error(
                    ErrorCode::Internal,
                    format!(
                        "shard {} answered an unreadable page: {error}",
                        map.url(shard)
                    ),
                    started_at,
                );
            }
        }
    }
    let mut seat: BTreeMap<String, usize> = scatter
        .direct
        .iter()
        .enumerate()
        .map(|(position, name)| (name.clone(), position))
        .collect();
    let resolved: BTreeSet<String> = plan_entries
        .iter()
        .map(|entry| entry.context.clone())
        .filter(|name| !seat.contains_key(name))
        .collect();
    for (position, name) in resolved.into_iter().enumerate() {
        seat.insert(name, scatter.direct.len() + position);
    }
    pool.sort_by_key(|(rank, hit)| (*rank, seat.get(&hit.context).copied().unwrap_or(usize::MAX)));
    pool.truncate(api::clamp(request.limit, 5, api::MAX_MATCH_LIMIT));
    plan_entries.sort_by_key(|entry| seat.get(&entry.context).copied().unwrap_or(usize::MAX));
    router_ok(
        api::CrossPassagePage {
            plan: api::SearchPlan {
                contexts: plan_entries,
            },
            hits: pool.into_iter().map(|(_, hit)| hit).collect(),
        },
        gathered.unreached,
        started_at,
    )
}

pub(super) async fn merge_contexts(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppQuery(query): api::AppQuery<api::ListContextsQuery>,
    request: Request,
) -> Response {
    let started_at = Instant::now();
    let path = full_path(&request);
    let map = state.map();
    let shards: Vec<usize> = map.all().collect();
    let outcomes = state
        .fan_out(
            &map,
            &shards,
            Method::GET,
            &path,
            &headers,
            |_| None,
            deadline,
        )
        .await;
    let mut unreached = Vec::new();
    let mut rows: BTreeMap<String, (usize, crate::registry::DirectoryEntry)> = BTreeMap::new();
    let mut total = 0usize;
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => {
                match serde_json::from_slice::<ShardEnvelope<api::ContextPage>>(&answer.body) {
                    Ok(page) => {
                        total += page.result.total;
                        for entry in page.result.contexts {
                            match rows.get(&entry.name) {
                                // A context answered by two shards is a
                                // mid-move stray: the map's owner wins the
                                // row, and the duplicate leaves the total.
                                Some((held_shard, _))
                                    if map.shard_of(&entry.name) != Some(shard)
                                        || *held_shard == shard =>
                                {
                                    total = total.saturating_sub(1);
                                    warn!(
                                        context = %entry.name,
                                        "context answered by more than one shard — \
                                         mid-move stray? the route map's owner wins"
                                    );
                                }
                                Some(_) => {
                                    total = total.saturating_sub(1);
                                    warn!(
                                        context = %entry.name,
                                        "context answered by more than one shard — \
                                         mid-move stray? the route map's owner wins"
                                    );
                                    rows.insert(entry.name.clone(), (shard, entry));
                                }
                                None => {
                                    rows.insert(entry.name.clone(), (shard, entry));
                                }
                            }
                        }
                    }
                    Err(error) => {
                        return api::error(
                            ErrorCode::Internal,
                            format!(
                                "shard {} answered an unreadable page: {error}",
                                map.url(shard)
                            ),
                            started_at,
                        );
                    }
                }
            }
            Ok(answer) => return passthrough(answer),
            Err(error) => unreached.push(Unreached {
                shard: map.url(shard).to_string(),
                contexts: Vec::new(),
                error,
            }),
        }
    }
    if rows.is_empty() && !unreached.is_empty() && unreached.len() == shards.len() {
        return unreachable_refusal(&unreached, started_at);
    }
    // `clamp_page`, exactly as the single instance's `list_contexts`
    // cuts its own page: `limit=0` floors to one so the keyset
    // listing's empty page keeps meaning "no more pages".
    let contexts: Vec<crate::registry::DirectoryEntry> = rows
        .into_values()
        .map(|(_, entry)| entry)
        .take(api::clamp_page(
            query.limit,
            api::MAX_MATCH_LIMIT,
            api::MAX_MATCH_LIMIT,
        ))
        .collect();
    router_ok(api::ContextPage { total, contexts }, unreached, started_at)
}
