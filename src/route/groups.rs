//! `group` verbs: every `group` exists on every shard, so directory reads
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
        let answer = match merge_fate(&map, shard, outcome, started_at) {
            Ok(MergeFate::Success(answer)) => answer,
            // A directory read cannot 404 — pass a stray one through
            // like any other error status.
            Ok(MergeFate::NotFound(answer)) => return passthrough(answer),
            Err(refusal) => return *refusal,
        };
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
    // `clamp_page`, exactly as the single instance's `list_groups`
    // cuts its own page: a keyset listing's empty page means "no more
    // pages" to the SDK iterators, so an explicit `limit=0` must floor
    // to one rather than answer a terminal-looking empty page.
    let groups: Vec<api::GroupEntry> = rows
        .into_values()
        .take(api::clamp_page(
            query.limit,
            api::MAX_MATCH_LIMIT,
            api::MAX_MATCH_LIMIT,
        ))
        .collect();
    router_ok(api::GroupPage { total, groups }, Vec::new(), started_at)
}

/// Unions one shard's row into the merged directory: member `context_ids`
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
            let mut members: BTreeSet<String> = held.context_ids.drain(..).collect();
            members.extend(entry.context_ids);
            held.context_ids = members.into_iter().collect();
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
// `group` verbs: projected broadcast

/// Runs one `group` write against every shard IN ORDER, the request's
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
        let outcome = state
            .call_shard(
                map,
                shard,
                method.clone(),
                path,
                headers,
                body_for(shard),
                deadline,
            )
            .await;
        match merge_fate(map, shard, outcome, started_at) {
            Ok(MergeFate::Success(answer)) => answers.push(answer.body),
            // Writes have no drift-healing 404 arm — a not-found stops
            // the broadcast and passes through like any other refusal.
            Ok(MergeFate::NotFound(answer)) => return Err(Box::new(passthrough(answer))),
            Err(refusal) => return Err(refusal),
        }
    }
    Ok(answers)
}

/// Why a body could not be projected: a refusal the router answers
/// itself, or a shape the single-instance extractor would refuse —
/// sent to one shard verbatim (`forward_group_probe`) so the refusal
/// is the shard's own, never a router-invented one.
enum Unprojectable {
    Refusal(Box<Response>),
    Probe,
}

/// Projects the named member-list fields of a JSON body per shard.
/// Members are context ids (#965), and the map speaks names, so the
/// owner of each id in a `checked` field (the create/add lists) is asked
/// of the shards themselves ([`locate_owner`]; a single-shard map pays
/// nothing). An id no shard holds is refused up front with the
/// single-instance nonexistent-member message, since a `context` no
/// shard owns cannot exist on any of them. `unchecked` fields (the
/// remove lists) need no owner: a single instance's `update_group`
/// treats removals as an idempotent set difference and never validates
/// their existence, so the whole list goes to every shard, where
/// whatever is not a member there is the same no-op.
///
/// Three gates before any of that. A body that is not a JSON object
/// is refused here as `invalid_argument`: indexing a non-object
/// `Value` by key would panic into a 500, and forwarding it to one
/// shard is no answer either — serde reads a JSON array as a
/// positional struct, so a single instance accepts `[]` as an empty
/// request, and a probe would create the group on that one shard
/// alone. Refusing is the one outcome that lands nothing anywhere
/// (the single instance's positional reading is a serde artefact no
/// client relies on, and this is the documented divergence). A member
/// list that is not an array of strings is [`Unprojectable::Probe`]:
/// the shard's typed extractor refuses it in its own shape, where the
/// router used to drop a stray non-string member in silence. A list
/// past `MAX_INPUT_ITEMS` is refused with `api::overlong`'s shape on
/// the WHOLE list — split per shard, each shard's slice would pass its
/// own cap and the intended hard limit would scale with the shard
/// count. A member that is not a canonical context id is refused with
/// the shard's own 400, before any probe.
#[allow(clippy::too_many_arguments)]
async fn project_body(
    state: &RouterState,
    map: &RouteMap,
    base: &Value,
    checked: &[&str],
    unchecked: &[&str],
    headers: &HeaderMap,
    deadline: Deadline,
    started_at: Instant,
) -> Result<impl Fn(usize) -> Option<Bytes> + use<>, Unprojectable> {
    if !base.is_object() {
        return Err(Unprojectable::Refusal(Box::new(api::error(
            ErrorCode::InvalidArgument,
            "a group request is a JSON object; nothing was applied",
            started_at,
        ))));
    }
    let mut lists: BTreeMap<String, (bool, Vec<String>)> = BTreeMap::new();
    let fields = checked
        .iter()
        .map(|field| (*field, true))
        .chain(unchecked.iter().map(|field| (*field, false)));
    for (field, check) in fields {
        let members: Vec<String> = match base.get(field) {
            None => Vec::new(),
            Some(Value::Array(values)) => {
                let mut members = Vec::with_capacity(values.len());
                for value in values {
                    match value.as_str() {
                        Some(member) => members.push(member.to_string()),
                        None => return Err(Unprojectable::Probe),
                    }
                }
                members
            }
            Some(_) => return Err(Unprojectable::Probe),
        };
        if let Some(refusal) = api::overlong(field, members.len(), started_at) {
            return Err(Unprojectable::Refusal(Box::new(refusal)));
        }
        if let Some(refusal) = api::invalid_context_ids(field, &members, started_at) {
            return Err(Unprojectable::Refusal(Box::new(refusal)));
        }
        lists.insert(field.to_string(), (check, members));
    }
    // Owners, asked once per distinct id of a checked list.
    let mut owner_of: BTreeMap<String, usize> = BTreeMap::new();
    for (check, members) in lists.values() {
        if !check {
            continue;
        }
        for member in members {
            if owner_of.contains_key(member) {
                continue;
            }
            match locate_owner(state, map, member, headers, deadline, started_at).await {
                Located::Shard(shard) => {
                    owner_of.insert(member.clone(), shard);
                }
                Located::Missing => {
                    return Err(Unprojectable::Refusal(Box::new(api::error(
                        ErrorCode::NoContext,
                        format!("context '{member}' not found; nothing was applied"),
                        started_at,
                    ))));
                }
                Located::Answered(response)
                | Located::Unreached {
                    refusal: response, ..
                } => {
                    return Err(Unprojectable::Refusal(Box::new(response)));
                }
            }
        }
    }
    let base = base.clone();
    let shards_projection: Vec<BTreeMap<String, Vec<String>>> = map
        .all()
        .map(|shard| {
            lists
                .iter()
                .map(|(field, (check, members))| {
                    let projected = if *check {
                        members
                            .iter()
                            .filter(|member| owner_of.get(*member) == Some(&shard))
                            .cloned()
                            .collect()
                    } else {
                        members.clone()
                    };
                    (field.clone(), projected)
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
    let body_for = match project_body(
        &state,
        &map,
        &base,
        &["context_ids"],
        &[],
        &headers,
        deadline,
        started_at,
    )
    .await
    {
        Ok(body_for) => body_for,
        Err(Unprojectable::Refusal(refusal)) => return *refusal,
        Err(Unprojectable::Probe) => {
            return forward_group_probe(&state, &map, Method::PUT, &name, headers, body, deadline)
                .await;
        }
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
        &state,
        &map,
        &base,
        &["add_context_ids"],
        &["remove_context_ids"],
        &headers,
        deadline,
        started_at,
    )
    .await
    {
        Ok(body_for) => body_for,
        Err(Unprojectable::Refusal(refusal)) => return *refusal,
        Err(Unprojectable::Probe) => {
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

/// Sends a body the single-instance extractor refuses — unparseable,
/// not an object, a member list that is not an array of strings — to
/// the `group`'s first shard verbatim, so the refusal (shape, status,
/// message) is that extractor's own. Nothing lands: the shard refuses
/// before it touches state, exactly as it would have alone.
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
        Ok(answer) => passthrough(answer),
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
        let outcome = state
            .call_shard(
                &map,
                shard,
                method.clone(),
                &path,
                &headers,
                body.clone(),
                deadline,
            )
            .await;
        match merge_fate(&map, shard, outcome, started_at) {
            Ok(MergeFate::Success(_)) => succeeded = true,
            Ok(MergeFate::NotFound(answer)) => not_found = Some(answer),
            Err(refusal) => return *refusal,
        }
    }
    match (succeeded, not_found) {
        (true, _) => api::ok(true, started_at),
        (false, Some(answer)) => passthrough(answer),
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
        match merge_fate(&map, shard, outcome, started_at) {
            Ok(MergeFate::Success(answer)) => {
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
            Ok(MergeFate::NotFound(answer)) => not_found = Some(answer),
            Err(refusal) => return *refusal,
        }
    }
    match (rows.into_values().next(), not_found) {
        (Some(entry), _) => api::ok(entry, started_at),
        (None, Some(answer)) => passthrough(answer),
        (None, None) => api::error(
            ErrorCode::NoGroup,
            format!("group '{name}' not found"),
            started_at,
        ),
    }
}

/// `GET /groups/{name}/export`: every shard's record line names its
/// own projection; the union record — one line, importable — is what
/// the `group` actually is.
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
        match merge_fate(&map, shard, outcome, started_at) {
            Ok(MergeFate::Success(answer)) => {
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
                        held.context_ids.extend(record.context_ids);
                        held.groups.extend(record.groups);
                    }
                    None => merged = Some(record),
                }
            }
            Ok(MergeFate::NotFound(answer)) => not_found = Some(answer),
            Err(refusal) => return *refusal,
        }
    }
    match (merged, not_found) {
        (Some(record), _) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")],
            crate::export::render_group(&name, &record),
        )
            .into_response(),
        (None, Some(answer)) => passthrough(answer),
        (None, None) => api::error(
            ErrorCode::NoGroup,
            format!("group '{name}' not found"),
            started_at,
        ),
    }
}

/// One shard's export body back into a record: a single
/// `group` line, the same shape `parse_group` reads.
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
        context_ids: string_set("context_ids"),
        groups: string_set("groups"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_over(map_text: &str) -> RouterState {
        RouterState {
            inner: Arc::new(RouterInner {
                map: parking_lot::RwLock::new(Arc::new(RouteMap::parse(map_text).unwrap())),
                client: reqwest::Client::new(),
                metrics: RouterMetrics::default(),
                instructions: OnceLock::new(),
            }),
        }
    }

    const TWO_SHARDS: &str = "a = http://a:1\nb = http://b:1\n* = http://b:1\n";
    const ONE_SHARD: &str = "* = http://b:1\n";

    /// `project_body` over `map_text`, with the arguments a handler
    /// would have. No test here reaches a network: every shape that
    /// needs a probe uses the one-shard map, where ownership is
    /// answered without asking.
    async fn project(
        map_text: &str,
        base: &Value,
        checked: &[&str],
        unchecked: &[&str],
    ) -> Result<impl Fn(usize) -> Option<Bytes> + use<>, Unprojectable> {
        let state = state_over(map_text);
        let map = state.map();
        project_body(
            &state,
            &map,
            base,
            checked,
            unchecked,
            &HeaderMap::new(),
            Deadline::unbounded(),
            Instant::now(),
        )
        .await
    }

    async fn refusal_body(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn id(n: usize) -> String {
        format!("00000000-0000-4000-8000-{n:012x}")
    }

    /// A non-object body is refused by the router itself — indexing it
    /// by key would panic into a 500, and a probe would let serde's
    /// positional reading create the group on one shard alone.
    #[tokio::test]
    async fn a_non_object_body_is_refused_before_any_shard_sees_it() {
        for base in [
            json!([]),
            json!(["d", ["a"]]),
            json!(42),
            json!("x"),
            json!(true),
        ] {
            let outcome = project(TWO_SHARDS, &base, &["context_ids"], &[]).await;
            let Err(Unprojectable::Refusal(refusal)) = outcome else {
                panic!("{base} must be refused, never probed or projected");
            };
            assert_eq!(refusal.status(), StatusCode::BAD_REQUEST, "{base}");
            let body = refusal_body(*refusal).await;
            assert_eq!(body["code"], "invalid_argument", "{base}: {body}");
        }
    }

    /// A member list the single-instance extractor would refuse is a
    /// probe, never projected: a list that is not an array, a member
    /// that is not a string (it used to be dropped in silence).
    #[tokio::test]
    async fn a_body_the_typed_extractor_refuses_is_a_probe_not_a_projection() {
        for base in [
            json!({"context_ids": "a"}),
            json!({"context_ids": {"a": 1}}),
            json!({"context_ids": ["a", 42]}),
            json!({"context_ids": ["a", null]}),
        ] {
            let outcome = project(TWO_SHARDS, &base, &["context_ids"], &[]).await;
            assert!(
                matches!(outcome, Err(Unprojectable::Probe)),
                "{base} must probe a shard"
            );
        }
        // The unchecked (remove) lists are typed the same way.
        let base = json!({"remove_context_ids": [1]});
        let outcome = project(TWO_SHARDS, &base, &[], &["remove_context_ids"]).await;
        assert!(matches!(outcome, Err(Unprojectable::Probe)));
        // An absent list is empty, a null body field is not an array.
        assert!(
            project(TWO_SHARDS, &json!({}), &["context_ids"], &[])
                .await
                .is_ok()
        );
        let outcome = project(
            TWO_SHARDS,
            &json!({"context_ids": null}),
            &["context_ids"],
            &[],
        )
        .await;
        assert!(matches!(outcome, Err(Unprojectable::Probe)));
    }

    /// The per-request cap is judged on the whole list, before the
    /// split per shard: 1001 members over two shards is refused with
    /// `api::overlong`'s shape even though each shard's slice would
    /// pass its own cap; 1000 projects (here over one shard, where
    /// ownership needs no probe — every member lands on it).
    #[tokio::test]
    async fn the_member_cap_is_the_whole_list_not_the_per_shard_slice() {
        let members: Vec<String> = (0..1001).map(id).collect();
        let base = json!({"context_ids": members});
        let outcome = project(TWO_SHARDS, &base, &["context_ids"], &[]).await;
        let Err(Unprojectable::Refusal(refusal)) = outcome else {
            panic!("1001 members must be refused as overlong");
        };
        assert_eq!(refusal.status(), StatusCode::BAD_REQUEST);
        let body = refusal_body(*refusal).await;
        assert_eq!(body["code"], "over_limit", "{body}");
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("context_ids carries 1001 items"),
            "{body}"
        );

        let members: Vec<String> = (0..1000).map(id).collect();
        let base = json!({"context_ids": members, "description": "d"});
        let body_for = project(ONE_SHARD, &base, &["context_ids"], &[])
            .await
            .unwrap_or_else(|_| panic!("1000 members must project"));
        let only: serde_json::Value = serde_json::from_slice(&body_for(0).unwrap()).unwrap();
        assert_eq!(only["context_ids"].as_array().unwrap().len(), 1000);
        assert_eq!(only["description"], "d");
    }

    /// A member that is not a canonical context id — a name, an
    /// uppercase spelling — is refused with the shard's own 400 before
    /// any probe, in every member list, the remove lists included.
    #[tokio::test]
    async fn a_member_that_is_not_a_context_id_is_refused_before_any_probe() {
        for (base, checked, unchecked) in [
            (json!({"context_ids": ["sake"]}), "context_ids", ""),
            (
                json!({"add_context_ids": [id(1), "SAKE"]}),
                "add_context_ids",
                "",
            ),
            (
                json!({"remove_context_ids": ["sake"]}),
                "",
                "remove_context_ids",
            ),
        ] {
            let checked: Vec<&str> = Some(checked)
                .into_iter()
                .filter(|f| !f.is_empty())
                .collect();
            let unchecked: Vec<&str> = Some(unchecked)
                .into_iter()
                .filter(|f| !f.is_empty())
                .collect();
            let outcome = project(TWO_SHARDS, &base, &checked, &unchecked).await;
            let Err(Unprojectable::Refusal(refusal)) = outcome else {
                panic!("{base} must be refused");
            };
            assert_eq!(refusal.status(), StatusCode::BAD_REQUEST, "{base}");
            let body = refusal_body(*refusal).await;
            assert_eq!(body["code"], "invalid_argument", "{base}: {body}");
            assert!(
                body["error"]
                    .as_str()
                    .unwrap()
                    .contains("is not a context id"),
                "{body}"
            );
        }
    }

    /// A remove list needs no owner: the whole list goes to every
    /// shard, where a non-member removes as the same no-op a single
    /// instance gives — and no probe is made (this two-shard map
    /// would refuse the connection).
    #[tokio::test]
    async fn a_remove_list_goes_whole_to_every_shard_without_a_probe() {
        let base = json!({"remove_context_ids": [id(1), id(2)], "add_groups": ["g"]});
        let body_for = project(TWO_SHARDS, &base, &[], &["remove_context_ids"])
            .await
            .unwrap_or_else(|_| panic!("a remove list must project"));
        for shard in 0..2 {
            let body: serde_json::Value =
                serde_json::from_slice(&body_for(shard).unwrap()).unwrap();
            assert_eq!(body["remove_context_ids"], json!([id(1), id(2)]));
            assert_eq!(body["add_groups"], json!(["g"]));
        }
    }
}
