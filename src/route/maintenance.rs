//! Broadcast operator verbs: `POST /flush` and
//! `POST /maintenance/compact`, sent to every shard and merged.

use super::*;

pub(super) async fn broadcast_flush(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    let map = state.map();
    let shards: Vec<usize> = map.all().collect();
    let outcomes = state
        .fan_out(
            &map,
            &shards,
            Method::POST,
            "/flush",
            &headers,
            |_| None,
            deadline,
        )
        .await;
    let mut flushed: Vec<String> = Vec::new();
    let mut unreached = Vec::new();
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => {
                if let Ok(envelope) =
                    serde_json::from_slice::<ShardEnvelope<Vec<String>>>(&answer.body)
                {
                    flushed.extend(envelope.result);
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
    if flushed.is_empty() && unreached.len() == shards.len() && !shards.is_empty() {
        return unreachable_refusal(&unreached, started_at);
    }
    router_ok(flushed, unreached, started_at)
}

pub(super) async fn broadcast_maintenance(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    request: Request,
) -> Response {
    let started_at = Instant::now();
    let path = full_path(&request);
    let map = state.map();
    let shards: Vec<usize> = map.all().collect();
    let shard_count = shards.len();
    // Sequential on purpose: each shard's sweep drains its own
    // traffic; running them one at a time keeps the fleet from
    // pausing everywhere at once.
    let mut contexts: Vec<Value> = Vec::new();
    let mut deadline_exceeded = false;
    let mut unreached = Vec::new();
    for shard in shards {
        match state
            .call_shard(&map, shard, Method::POST, &path, &headers, None, deadline)
            .await
        {
            Ok(answer) if answer.status.is_success() => {
                if let Ok(envelope) = serde_json::from_slice::<ShardEnvelope<Value>>(&answer.body) {
                    if let Some(swept) = envelope.result.get("contexts").and_then(Value::as_array) {
                        contexts.extend(swept.iter().cloned());
                    }
                    deadline_exceeded |= envelope
                        .result
                        .get("deadline_exceeded")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
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
    // The same guard as `broadcast_flush` above: a fleet where NO shard
    // could be asked answers a 502 refusal, never an empty-but-200
    // sweep report that reads as "nothing needed compacting".
    if contexts.is_empty() && unreached.len() == shard_count && shard_count > 0 {
        return unreachable_refusal(&unreached, started_at);
    }
    router_ok(
        json!({"contexts": contexts, "deadline_exceeded": deadline_exceeded}),
        unreached,
        started_at,
    )
}
