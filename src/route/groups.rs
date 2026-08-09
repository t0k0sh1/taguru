//! Group verbs: every group exists on every shard, so directory reads
//! union the per-shard projections and writes broadcast sequentially,
//! member lists projected per shard by the map.

use super::*;

pub(super) async fn merge_groups(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppQuery(query): api::AppQuery<api::KeysetQuery>,
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
    let mut rows: BTreeMap<String, api::GroupEntry> = BTreeMap::new();
    let mut total = 0usize;
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => {
                match serde_json::from_slice::<ShardEnvelope<api::GroupPage>>(&answer.body) {
                    Ok(page) => {
                        // Every shard holds every group, so any one
                        // shard's directory count is the true count;
                        // max rides out a half-created record.
                        total = total.max(page.result.total);
                        for entry in page.result.groups {
                            merge_group_entry(&mut rows, entry);
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
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => {
                // A partially-unioned group row would look complete;
                // the group surfaces refuse rather than thin out.
                return unreachable_refusal(
                    &[Unreached {
                        shard: map.url(shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                );
            }
        }
    }
    let groups: Vec<api::GroupEntry> = rows
        .into_values()
        .take(api::clamp(
            query.limit,
            api::MAX_MATCH_LIMIT,
            api::MAX_MATCH_LIMIT,
        ))
        .collect();
    router_ok(api::GroupPage { total, groups }, Vec::new(), started_at)
}

/// Unions one shard's row into the merged directory: member contexts
/// are per-shard projections (disjoint by the map), children are
/// broadcast whole, the description is identical everywhere a
/// non-drifted record lives. Fingerprints are folded together — each
/// shard's token covers the members that shard holds, so the union's
/// token must move whenever ANY shard's does; the fold order is the
/// fan-out's shard order (stable across requests), so an unchanged
/// fleet keeps an unchanged token.
fn merge_group_entry(rows: &mut BTreeMap<String, api::GroupEntry>, entry: api::GroupEntry) {
    match rows.get_mut(&entry.name) {
        Some(held) => {
            let mut members: BTreeSet<String> = held.contexts.drain(..).collect();
            members.extend(entry.contexts);
            held.contexts = members.into_iter().collect();
            held.groups.extend(entry.groups);
            let mut digest = crate::hash::fnv1a_fold(
                crate::hash::FNV1A_OFFSET,
                held.fingerprint.bytes().chain([0xff]),
            );
            digest = crate::hash::fnv1a_fold(digest, entry.fingerprint.bytes());
            held.fingerprint = format!("{digest:016x}");
        }
        None => {
            rows.insert(entry.name.clone(), entry);
        }
    }
}

pub(super) fn full_path(request: &Request) -> String {
    request
        .uri()
        .path_and_query()
        .map(|paq| paq.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string())
}

// ---------------------------------------------------------------------------
// Group verbs: projected broadcast

/// Runs one group write against every shard IN ORDER, the request's
/// member lists projected per shard by the map. The first refusal
/// stops the broadcast and passes through as-is — shards before it
/// have applied (deltas converge on retry; documented divergence).
#[allow(clippy::too_many_arguments)]
async fn broadcast_group_write<F>(
    state: &RouterState,
    map: &RouteMap,
    method: Method,
    path: &str,
    headers: &HeaderMap,
    body_for: F,
    deadline: Deadline,
    started_at: Instant,
) -> Result<Vec<Bytes>, Box<Response>>
where
    F: Fn(usize) -> Option<Bytes>,
{
    let mut answers = Vec::new();
    for shard in map.all() {
        match state
            .call_shard(
                map,
                shard,
                method.clone(),
                path,
                headers,
                body_for(shard),
                deadline,
            )
            .await
        {
            Ok(answer) if answer.status.is_success() => answers.push(answer.body),
            Ok(answer) => {
                return Err(Box::new(
                    (
                        answer.status,
                        [(header::CONTENT_TYPE, "application/json")],
                        answer.body,
                    )
                        .into_response(),
                ));
            }
            Err(error) => {
                return Err(Box::new(unreachable_refusal(
                    &[Unreached {
                        shard: map.url(shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                )));
            }
        }
    }
    Ok(answers)
}

/// Projects the named member-list fields of a JSON body per shard —
/// any member the map does not place is refused up front with the
/// single-instance nonexistent-member message, since a context no
/// shard owns cannot exist on any of them.
fn project_body(
    map: &RouteMap,
    base: &Value,
    fields: &[&str],
    started_at: Instant,
) -> Result<impl Fn(usize) -> Option<Bytes> + use<>, Box<Response>> {
    let mut lists: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for field in fields {
        let members: Vec<String> = base
            .get(*field)
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        for member in &members {
            if map.shard_of(member).is_none() {
                return Err(Box::new(api::error(
                    ErrorCode::NoContext,
                    format!("context '{member}' not found; nothing was applied"),
                    started_at,
                )));
            }
        }
        lists.insert((*field).to_string(), members);
    }
    let base = base.clone();
    let shards_projection: Vec<BTreeMap<String, Vec<String>>> = map
        .all()
        .map(|shard| {
            lists
                .iter()
                .map(|(field, members)| {
                    (
                        field.clone(),
                        map.project(members.iter().map(String::as_str), shard),
                    )
                })
                .collect()
        })
        .collect();
    Ok(move |shard: usize| {
        let mut body = base.clone();
        for (field, members) in &shards_projection[shard] {
            body[field.as_str()] = json!(members);
        }
        Some(Bytes::from(body.to_string()))
    })
}

pub(super) async fn create_group_broadcast(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    body: Bytes,
) -> Response {
    let started_at = Instant::now();
    let map = state.map();
    let base: Value = if body.is_empty() {
        json!({})
    } else {
        match serde_json::from_slice(&body) {
            Ok(base) => base,
            Err(_) => {
                // Malformed bodies go to one shard untouched so the
                // refusal is the shard's own extractor shape.
                return forward_group_probe(
                    &state,
                    &map,
                    Method::PUT,
                    &name,
                    headers,
                    body,
                    deadline,
                )
                .await;
            }
        }
    };
    let path = format!("/groups/{}", urlencode(&name));
    let body_for = match project_body(&map, &base, &["contexts"], started_at) {
        Ok(body_for) => body_for,
        Err(refusal) => return *refusal,
    };
    match broadcast_group_write(
        &state,
        &map,
        Method::PUT,
        &path,
        &headers,
        body_for,
        deadline,
        started_at,
    )
    .await
    {
        Ok(_) => api::ok(true, started_at),
        Err(refusal) => *refusal,
    }
}

pub(super) async fn update_group_broadcast(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    body: Bytes,
) -> Response {
    let started_at = Instant::now();
    let map = state.map();
    let base: Value = match serde_json::from_slice(&body) {
        Ok(base) => base,
        Err(_) => {
            return forward_group_probe(
                &state,
                &map,
                Method::PATCH,
                &name,
                headers,
                body,
                deadline,
            )
            .await;
        }
    };
    let path = format!("/groups/{}", urlencode(&name));
    let body_for = match project_body(
        &map,
        &base,
        &["add_contexts", "remove_contexts"],
        started_at,
    ) {
        Ok(body_for) => body_for,
        Err(refusal) => return *refusal,
    };
    match broadcast_group_write(
        &state,
        &map,
        Method::PATCH,
        &path,
        &headers,
        body_for,
        deadline,
        started_at,
    )
    .await
    {
        Ok(answers) => {
            let mut rows: BTreeMap<String, api::GroupEntry> = BTreeMap::new();
            for body in answers {
                if let Ok(envelope) =
                    serde_json::from_slice::<ShardEnvelope<api::GroupEntry>>(&body)
                {
                    merge_group_entry(&mut rows, envelope.result);
                }
            }
            match rows.into_values().next() {
                Some(entry) => api::ok(entry, started_at),
                None => api::error(
                    ErrorCode::Internal,
                    "every shard applied the update but none answered a readable entry",
                    started_at,
                ),
            }
        }
        Err(refusal) => *refusal,
    }
}

/// Sends an unparseable body to the group's first shard verbatim, so
/// the refusal (shape, status, message) is the single-instance
/// extractor's own.
#[allow(clippy::too_many_arguments)]
async fn forward_group_probe(
    state: &RouterState,
    map: &RouteMap,
    method: Method,
    name: &str,
    headers: HeaderMap,
    body: Bytes,
    deadline: Deadline,
) -> Response {
    let started_at = Instant::now();
    let path = format!("/groups/{}", urlencode(name));
    match state
        .call_shard(map, 0, method, &path, &headers, Some(body), deadline)
        .await
    {
        Ok(answer) => (
            answer.status,
            [(header::CONTENT_TYPE, "application/json")],
            answer.body,
        )
            .into_response(),
        Err(error) => unreachable_refusal(
            &[Unreached {
                shard: map.url(0).to_string(),
                contexts: Vec::new(),
                error,
            }],
            started_at,
        ),
    }
}

pub(super) async fn delete_group_broadcast(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    group_broadcast_simple(state, Method::DELETE, name, None, headers, deadline).await
}

pub(super) async fn rename_group_broadcast(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    body: Bytes,
) -> Response {
    group_broadcast_simple(
        state,
        Method::POST,
        format!("{name}/rename"),
        Some(body),
        headers,
        deadline,
    )
    .await
}

/// Delete and rename broadcast the identical request everywhere; a
/// 404 from EVERY shard is the single-instance not-found, while a
/// mixed answer (drift healing) succeeds with the successes.
async fn group_broadcast_simple(
    state: RouterState,
    method: Method,
    name_path: String,
    body: Option<Bytes>,
    headers: HeaderMap,
    deadline: Deadline,
) -> Response {
    let started_at = Instant::now();
    let (encoded_name, suffix) = match name_path.split_once('/') {
        Some((name, suffix)) => (urlencode(name), format!("/{suffix}")),
        None => (urlencode(&name_path), String::new()),
    };
    let path = format!("/groups/{encoded_name}{suffix}");
    let map = state.map();
    let mut not_found: Option<ShardAnswer> = None;
    let mut succeeded = false;
    for shard in map.all() {
        match state
            .call_shard(
                &map,
                shard,
                method.clone(),
                &path,
                &headers,
                body.clone(),
                deadline,
            )
            .await
        {
            Ok(answer) if answer.status.is_success() => succeeded = true,
            Ok(answer) if answer.status == StatusCode::NOT_FOUND => not_found = Some(answer),
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => {
                return unreachable_refusal(
                    &[Unreached {
                        shard: map.url(shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                );
            }
        }
    }
    match (succeeded, not_found) {
        (true, _) => api::ok(true, started_at),
        (false, Some(answer)) => (
            answer.status,
            [(header::CONTENT_TYPE, "application/json")],
            answer.body,
        )
            .into_response(),
        (false, None) => api::ok(true, started_at),
    }
}

pub(super) async fn union_group(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    let path = format!("/groups/{}", urlencode(&name));
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
    let mut rows: BTreeMap<String, api::GroupEntry> = BTreeMap::new();
    let mut not_found: Option<ShardAnswer> = None;
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => {
                match serde_json::from_slice::<ShardEnvelope<api::GroupEntry>>(&answer.body) {
                    Ok(envelope) => merge_group_entry(&mut rows, envelope.result),
                    Err(error) => {
                        return api::error(
                            ErrorCode::Internal,
                            format!(
                                "shard {} answered an unreadable group: {error}",
                                map.url(shard)
                            ),
                            started_at,
                        );
                    }
                }
            }
            Ok(answer) if answer.status == StatusCode::NOT_FOUND => not_found = Some(answer),
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => {
                return unreachable_refusal(
                    &[Unreached {
                        shard: map.url(shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                );
            }
        }
    }
    match (rows.into_values().next(), not_found) {
        (Some(entry), _) => api::ok(entry, started_at),
        (None, Some(answer)) => (
            answer.status,
            [(header::CONTENT_TYPE, "application/json")],
            answer.body,
        )
            .into_response(),
        (None, None) => api::error(
            ErrorCode::NoGroup,
            format!("group '{name}' not found"),
            started_at,
        ),
    }
}

/// `GET /groups/{name}/export`: every shard's record line names its
/// own projection; the union record — one line, importable — is what
/// the group actually is.
pub(super) async fn export_group_union(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    let path = format!("/groups/{}/export", urlencode(&name));
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
    let mut merged: Option<crate::groups::GroupRecord> = None;
    let mut not_found: Option<ShardAnswer> = None;
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => {
                let Some(record) = parse_group_export(&answer.body) else {
                    return api::error(
                        ErrorCode::Internal,
                        format!(
                            "shard {} answered an unreadable group record",
                            map.url(shard)
                        ),
                        started_at,
                    );
                };
                match &mut merged {
                    Some(held) => {
                        held.contexts.extend(record.contexts);
                        held.groups.extend(record.groups);
                    }
                    None => merged = Some(record),
                }
            }
            Ok(answer) if answer.status == StatusCode::NOT_FOUND => not_found = Some(answer),
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => {
                return unreachable_refusal(
                    &[Unreached {
                        shard: map.url(shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                );
            }
        }
    }
    match (merged, not_found) {
        (Some(record), _) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")],
            crate::export::render_group(&name, &record),
        )
            .into_response(),
        (None, Some(answer)) => (
            answer.status,
            [(header::CONTENT_TYPE, "application/json")],
            answer.body,
        )
            .into_response(),
        (None, None) => api::error(
            ErrorCode::NoGroup,
            format!("group '{name}' not found"),
            started_at,
        ),
    }
}

/// One shard's export body back into a record: a single
/// `taguru_group` line, the same shape `parse_group` reads.
fn parse_group_export(body: &Bytes) -> Option<crate::groups::GroupRecord> {
    let text = std::str::from_utf8(body).ok()?;
    let line = text.lines().find(|line| !line.trim().is_empty())?;
    let value: Value = serde_json::from_str(line).ok()?;
    let object = value.as_object()?;
    let string_set = |key: &str| -> BTreeSet<String> {
        object
            .get(key)
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    };
    Some(crate::groups::GroupRecord {
        description: object
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        contexts: string_set("contexts"),
        groups: string_set("groups"),
    })
}
