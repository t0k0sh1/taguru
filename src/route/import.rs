//! `POST /import`: splits the batch stream by each batch's `context`
//! header, dry-run-preflights every chunk against the owning shards,
//! then dispatches in stream order — the same all-or-nothing shape a
//! single instance gives one stream.

use super::*;

pub(super) async fn route_import(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppQuery(query): api::AppQuery<api::ImportQuery>,
    api::AppBytes(body): api::AppBytes,
) -> Response {
    let started_at = Instant::now();
    let map = state.map();
    // The same whole-stream validation a shard would run, so a
    // malformed stream refuses here with the same message and the same
    // stream-level line numbers.
    let stream = match crate::ingest::parse_stream(&body[..]) {
        Ok(stream) => stream,
        Err(message) => return api::error(ErrorCode::MalformedRequest, message, started_at),
    };
    let slices = crate::ingest::split_batches(&body);
    debug_assert_eq!(slices.len(), stream.batches.len());
    // Route every batch before anything ships: a stream naming an
    // unroutable context refuses whole, like any other invalid stream.
    let mut chunks: Vec<(usize, Vec<std::ops::Range<usize>>)> = Vec::new();
    for (batch, range) in stream.batches.iter().zip(slices) {
        let Some(shard) = map.shard_of(&batch.context) else {
            return api::error(
                ErrorCode::NoContext,
                format!(
                    "batch source '{}': context '{}' has no route-map entry and no '*' \
                     fallback (TAGURU_ROUTE_MAP); nothing was applied",
                    batch.source, batch.context
                ),
                started_at,
            );
        };
        match chunks.last_mut() {
            Some((last_shard, ranges)) if *last_shard == shard => ranges.push(range),
            _ => chunks.push((shard, vec![range])),
        }
    }
    let assemble = |ranges: &[std::ops::Range<usize>]| {
        let mut chunk = Vec::new();
        for range in ranges {
            chunk.extend_from_slice(&body[range.clone()]);
        }
        Bytes::from(chunk)
    };
    // Schema records route to exactly the ONE shard owning their
    // context — never broadcast, unlike group records below, which
    // can span several — refused whole up front too, before anything
    // ships, the same posture as the batch loop just above (ADR 0009
    // §13).
    let mut schema_shards: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, (context, _)) in stream.schemas.iter().enumerate() {
        let Some(shard) = map.shard_of(context) else {
            return api::error(
                ErrorCode::NoContext,
                format!(
                    "schema record: context '{context}' has no route-map entry and no '*' \
                     fallback (TAGURU_ROUTE_MAP); nothing was applied"
                ),
                started_at,
            );
        };
        schema_shards.entry(shard).or_default().push(index);
    }
    // Each owning shard's schema lines, rendered once — preflighted
    // below beside the batch chunks, then applied after them.
    let schema_lines_for = |shard: usize| -> String {
        let mut lines = String::new();
        if let Some(indices) = schema_shards.get(&shard) {
            for &index in indices {
                let (context, installed) = &stream.schemas[index];
                lines.push_str(&crate::export::render_schema(context, installed.document()));
            }
        }
        lines
    };
    // Each shard's projected group-record stream, rendered once —
    // preflighted below beside the batch chunks, then applied last.
    let group_lines_for = |shard: usize| -> String {
        let mut lines = String::new();
        for (name, record) in &stream.groups {
            let projected = crate::groups::GroupRecord {
                description: record.description.clone(),
                contexts: record
                    .contexts
                    .iter()
                    .filter(|member| map.shard_of(member) == Some(shard))
                    .cloned()
                    .collect(),
                groups: record.groups.clone(),
            };
            lines.push_str(&crate::export::render_group(name, &projected));
        }
        lines
    };
    // Preflight: every batch chunk AND every projected group stream
    // dry-runs first, so a refusal a single instance would answer with
    // nothing applied — malformed batches, a scoped key beyond its
    // grant on a batch context or on a group record's closure — is
    // answered the same way here, with nothing applied on any shard.
    // (A real dry_run request IS its own preflight. Group VALIDATION
    // against live state still runs after the batches land, exactly
    // where the single instance runs it.)
    let preflight_query = "/import?dry_run=true";
    let real_query = if query.dry_run {
        "/import?dry_run=true"
    } else {
        "/import"
    };
    if !query.dry_run {
        let group_shards: Vec<usize> = if stream.groups.is_empty() {
            Vec::new()
        } else {
            map.all().collect()
        };
        let preflights = chunks
            .iter()
            .map(|(shard, ranges)| (*shard, assemble(ranges)))
            .chain(
                schema_shards
                    .keys()
                    .map(|&shard| (shard, Bytes::from(schema_lines_for(shard)))),
            )
            .chain(
                group_shards
                    .iter()
                    .map(|&shard| (shard, Bytes::from(group_lines_for(shard)))),
            );
        for (shard, chunk) in preflights {
            match state
                .call_shard(
                    &map,
                    shard,
                    Method::POST,
                    preflight_query,
                    &headers,
                    Some(chunk),
                    deadline,
                )
                .await
            {
                Ok(answer) if answer.status.is_success() => {}
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
    }
    // The real run, chunk by chunk in stream order. A refusal that
    // survived preflight (mid-apply IO, a budget spent partway) stops
    // here exactly as it stops a single instance partway: the batches
    // before it landed durably, and re-POSTing the stream is exact.
    let mut batches: Vec<Value> = Vec::new();
    for (index, (shard, ranges)) in chunks.iter().enumerate() {
        match state
            .call_shard(
                &map,
                *shard,
                Method::POST,
                real_query,
                &headers,
                Some(assemble(ranges)),
                deadline,
            )
            .await
        {
            Ok(answer) if answer.status.is_success() => {
                if let Ok(envelope) = serde_json::from_slice::<ShardEnvelope<Value>>(&answer.body)
                    && let Some(outcomes) = envelope.result.get("batches").and_then(Value::as_array)
                {
                    batches.extend(outcomes.iter().cloned());
                }
            }
            Ok(answer) => {
                return rewrap_import_refusal(
                    answer,
                    batches.len(),
                    0,
                    !query.dry_run && index > 0,
                    started_at,
                );
            }
            Err(error) => {
                return unreachable_refusal(
                    &[Unreached {
                        shard: map.url(*shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                );
            }
        }
    }
    // Schema records install after every batch, before groups restore
    // (ADR 0009 §13) — one request per OWNING shard (never broadcast,
    // unlike groups below). A dry run dispatches them too (with
    // dry_run forwarded): the owning shard still parses and
    // scope-checks the record exactly as a single instance's dry run
    // does, while omitting it from the preview (ADR 0009 §13; no
    // read-only twin `put_schema` can preview through).
    // Keyed by the record's ORIGINAL index in `stream.schemas`, not by
    // shard: `schema_shards` iterates in shard-number order, which is
    // not stream order, so accumulating into a plain `Vec` here would
    // silently reorder `result.schemas` relative to what one instance
    // (or a different route-map split) answers for the identical
    // stream — the same equivalence `taguru router`'s own doc promises
    // for every other verb.
    let mut schemas_by_index: BTreeMap<usize, Value> = BTreeMap::new();
    for (&shard, indices) in &schema_shards {
        match state
            .call_shard(
                &map,
                shard,
                Method::POST,
                real_query,
                &headers,
                Some(Bytes::from(schema_lines_for(shard))),
                deadline,
            )
            .await
        {
            Ok(answer) if answer.status.is_success() => {
                if let Ok(envelope) = serde_json::from_slice::<ShardEnvelope<Value>>(&answer.body)
                    && let Some(outcomes) = envelope.result.get("schemas").and_then(Value::as_array)
                {
                    // `schema_lines_for(shard)` rendered `indices` in
                    // this same order, so the shard's own response
                    // order matches it position for position.
                    for (&index, outcome) in indices.iter().zip(outcomes.iter()) {
                        schemas_by_index.insert(index, outcome.clone());
                    }
                }
            }
            Ok(answer) => {
                return rewrap_import_refusal(
                    answer,
                    batches.len(),
                    schemas_by_index.len(),
                    !query.dry_run && (!chunks.is_empty() || !schemas_by_index.is_empty()),
                    started_at,
                );
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
    let schemas: Vec<Value> = schemas_by_index.into_values().collect();
    // Group records apply LAST, after every batch (and every schema) —
    // the same order the stream contract promises — projected per
    // shard and broadcast. A dry run dispatches them too (with
    // dry_run forwarded): the shards still parse and scope-check the
    // records exactly as a single instance's dry run does, while
    // omitting them from the preview.
    let mut groups: Vec<Value> = Vec::new();
    if !stream.groups.is_empty() {
        let mut per_group: BTreeMap<usize, Vec<GroupOutcomeWire>> = BTreeMap::new();
        for shard in map.all() {
            match state
                .call_shard(
                    &map,
                    shard,
                    Method::POST,
                    real_query,
                    &headers,
                    Some(Bytes::from(group_lines_for(shard))),
                    deadline,
                )
                .await
            {
                Ok(answer) if answer.status.is_success() => {
                    if let Ok(envelope) =
                        serde_json::from_slice::<ShardEnvelope<GroupsOnlyWire>>(&answer.body)
                    {
                        for (position, outcome) in envelope.result.groups.into_iter().enumerate() {
                            per_group.entry(position).or_default().push(outcome);
                        }
                    }
                }
                Ok(answer) => {
                    return rewrap_import_refusal(
                        answer,
                        batches.len(),
                        schemas.len(),
                        !query.dry_run && (!chunks.is_empty() || !schemas.is_empty()),
                        started_at,
                    );
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
        for (_, outcomes) in per_group {
            let Some(first) = outcomes.first() else {
                continue;
            };
            let outcome = if outcomes.iter().all(|entry| entry.outcome == "unchanged") {
                "unchanged"
            } else if outcomes.iter().all(|entry| entry.outcome == "created") {
                "created"
            } else {
                "replaced"
            };
            groups.push(json!({
                "name": first.name,
                "outcome": outcome,
                // Projections partition the member set, so the true
                // count is their sum; the child-group list is
                // broadcast whole, so any shard's count is the count.
                "contexts": outcomes.iter().map(|entry| entry.contexts).sum::<usize>(),
                "groups": first.groups,
            }));
        }
    }
    let mut result = json!({"batches": batches});
    if !schemas.is_empty() {
        result["schemas"] = json!(schemas);
    }
    if !groups.is_empty() {
        result["groups"] = json!(groups);
    }
    api::ok(result, started_at)
}

#[derive(Deserialize)]
struct GroupsOnlyWire {
    #[serde(default)]
    groups: Vec<GroupOutcomeWire>,
}

#[derive(Deserialize)]
struct GroupOutcomeWire {
    name: String,
    outcome: String,
    contexts: usize,
    groups: usize,
}

/// A shard refusal from the real (post-preflight) import run, with the
/// cross-chunk truth prepended when earlier chunks OR schema records
/// already landed on a different shard — the failing shard's own note
/// counts only what reached IT, never what an earlier request to a
/// different shard already made durable. `schemas_landed` is 0 for
/// the batch loop's own call site (the schema loop has not started
/// yet at that point), so its wording is unchanged.
///
/// `integrity`/`durable_batches` are RECOMPUTED here from the true
/// cross-shard counts, never copied from the failing shard's own
/// body: that shard only knows what reached IT (0, for a shard whose
/// own request carried nothing before the refusal), which would
/// under-report exactly the way `schema_import_refusal`'s own fix
/// avoids for a single instance. `issues`/`retryable_after_correction`
/// ARE copied verbatim — they describe the failing shard's own
/// refusal (which context, which cap), a claim the cross-shard
/// context does not change.
fn rewrap_import_refusal(
    answer: ShardAnswer,
    batches_landed: usize,
    schemas_landed: usize,
    other_landed: bool,
    started_at: Instant,
) -> Response {
    if !other_landed {
        return (
            answer.status,
            [(header::CONTENT_TYPE, "application/json")],
            answer.body,
        )
            .into_response();
    }
    let (code, message, issues, retryable) = match serde_json::from_slice::<Value>(&answer.body) {
        Ok(body) => (
            body.get("code")
                .and_then(Value::as_str)
                .unwrap_or("internal")
                .to_string(),
            body.get("error")
                .and_then(Value::as_str)
                .unwrap_or("shard refusal with an unreadable body")
                .to_string(),
            body.get("issues").cloned(),
            body.get("retryable_after_correction").cloned(),
        ),
        Err(_) => (
            "internal".to_string(),
            String::from_utf8_lossy(&answer.body).into_owned(),
            None,
            None,
        ),
    };
    let landed = match (batches_landed, schemas_landed) {
        (batches, 0) => format!("{batches} batch(es)"),
        (0, schemas) => format!("{schemas} schema record(s)"),
        (batches, schemas) => format!("{batches} batch(es) and {schemas} schema record(s)"),
    };
    let mut rewrapped = json!({
        "status": "error",
        "code": code,
        "error": format!(
            "{landed} landed durably on earlier shards before this refusal (re-POSTing the \
             whole stream is exact — each batch replaces its own source, each schema install \
             is independent); the refusing shard says: {message}"
        ),
        // `other_landed` is true only when at least one of the two
        // counts is nonzero, so this is always "durable_prefix" here
        // — the `!other_landed` early return above is what answers
        // "nothing_written" (the failing shard's own body, unedited).
        "integrity": "durable_prefix",
        "time": started_at.elapsed().as_secs_f64(),
    });
    if batches_landed > 0 {
        rewrapped["durable_batches"] = json!(batches_landed);
    }
    if let Some(issues) = issues {
        rewrapped["issues"] = issues;
    }
    if let Some(retryable) = retryable {
        rewrapped["retryable_after_correction"] = retryable;
    }
    (
        answer.status,
        [(header::CONTENT_TYPE, "application/json")],
        rewrapped.to_string(),
    )
        .into_response()
}
