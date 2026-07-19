use std::collections::HashSet;

use serde_json::{Value, json};

use super::args::{need, optional_bool, pick};
use super::route::route_tool;

/// Extracts `(subject, label, object)` from an `AssociationOut`-shaped
/// value, for `run_retrieve`'s cross-step deduplication. `None` for
/// anything not shaped that way, which the caller treats as "keep it,
/// nothing to dedupe against" rather than dropping it.
pub(super) fn triple_of(association: &Value) -> Option<(String, String, String)> {
    Some((
        association.get("subject")?.as_str()?.to_string(),
        association.get("label")?.as_str()?.to_string(),
        association.get("object")?.as_str()?.to_string(),
    ))
}

/// Ceiling on the `origins` cue list [`run_retrieve`] accepts. Each cue
/// drives its own `resolve` round trip (and, with describe_first, a
/// `describe`), so an unbounded list would amplify one composed call
/// into arbitrarily many requests — slipping past the per-request cap
/// the direct read endpoints put on list inputs, which it reaches one
/// cue at a time. Mirrors `api::MAX_INPUT_ITEMS`; restated here because
/// this module compiles into the stdio bridge too, which carries no
/// `api` module to borrow the constant from.
pub(super) const MAX_ORIGIN_CUES: usize = 1000;

/// [`run_retrieve_bounded`] with no byte budget — every planned call
/// fires unconditionally. What the stdio bridge calls; `taguru-mcp.rs`'s
/// `dispatch_tool` documents why its composition stays uncapped.
#[allow(dead_code)] // consumed by the stdio bridge; the HTTP transport always calls run_retrieve_bounded instead
pub fn run_retrieve(
    arguments: &Value,
    call: impl FnMut(&'static str, String, Option<Value>) -> Result<String, String>,
) -> Result<Value, String> {
    run_retrieve_bounded(arguments, None, call)
}

/// The composed retrieval loop (`Context.retrieve()` in both SDKs),
/// reimplemented here so an MCP-only agent gets it in one call instead
/// of orchestrating five tool calls by hand. `route_tool` stays a pure
/// one-shot `(method, path, body)` mapping — this is deliberately a
/// separate function rather than another `route_tool` arm, since it
/// issues a variable number of requests built from earlier ones'
/// results. Each step still builds its request by calling `route_tool`
/// itself, so this can never drift from the single-call tools it
/// composes. `call` performs one routed request; the two transports
/// supply it (a ureq round trip for the stdio bridge, an in-process
/// dispatch for the HTTP transport, which must bridge onto its own
/// async call itself).
///
/// `budget`, when `Some`, caps the running total of every dispatched
/// call's raw response size: once one call pushes the total past it,
/// the next `call_tool` refuses before firing rather than composing
/// (and paying the round-trip cost for) a result the caller's own size
/// cap would discard anyway. The running total only ever over-counts
/// the true composed size — a step often keeps just one field of a
/// response, e.g. `"result"` — so this can cut off a little early but
/// never late; the caller's own post-hoc check on the final value
/// stays the source of truth either way.
pub fn run_retrieve_bounded(
    arguments: &Value,
    budget: Option<usize>,
    mut call: impl FnMut(&'static str, String, Option<Value>) -> Result<String, String>,
) -> Result<Value, String> {
    let mut spent: usize = 0;
    let mut call_tool = |name: &'static str, args: Value| -> Result<Value, String> {
        let (method, path, body) = route_tool(name, &args)?;
        let text = call(method, path, body)?;
        spent += text.len();
        if let Some(budget) = budget
            && spent > budget
        {
            return Err(format!(
                "retrieve's composed result already exceeds {budget} bytes after the \
                 '{name}' call; narrow it — fewer origins, a smaller resolve_limit or \
                 activate_limit, or fetch_citations: false — rather than paying for calls \
                 whose result would be discarded anyway"
            ));
        }
        serde_json::from_str::<Value>(&text)
            .map_err(|error| format!("tool '{name}' returned invalid JSON: {error}"))
    };

    let context = need(arguments, "context")?.to_string();
    let origins: Vec<String> = match arguments.get("origins") {
        Some(Value::String(text)) => vec![text.clone()],
        Some(Value::Array(items)) => {
            // Each origin cue fans out to its own `resolve` round trip (and,
            // with describe_first, a `describe`), so an unbounded list
            // amplifies one call into arbitrarily many — slipping past the
            // per-request list cap the direct read endpoints enforce, since it
            // reaches them one cue at a time. Refuse an oversized list up
            // front — before cloning every cue into a `String` — at the same
            // ceiling `overlong` applies to `origins` on those endpoints.
            if items.len() > MAX_ORIGIN_CUES {
                return Err(format!(
                    "argument 'origins' carries {} cues, past the per-request limit of {}; \
                     split the retrieval",
                    items.len(),
                    MAX_ORIGIN_CUES
                ));
            }
            items
                .iter()
                .map(|item| {
                    item.as_str().map(str::to_string).ok_or_else(|| {
                        "argument 'origins' must be a string or an array of strings".to_string()
                    })
                })
                .collect::<Result<_, _>>()?
        }
        Some(Value::Null) | None => return Err("missing required argument 'origins'".to_string()),
        Some(_) => {
            return Err("argument 'origins' must be a string or an array of strings".to_string());
        }
    };
    let auto_pick = optional_bool(arguments, "auto_pick", true)?;
    let describe_first = optional_bool(arguments, "describe_first", true)?;
    let fetch_citations = optional_bool(arguments, "fetch_citations", true)?;
    let text_fallback_only_if_empty =
        optional_bool(arguments, "text_fallback_only_if_empty", true)?;

    // Step 1: resolve each origin cue, auto-picking the top candidate
    // (or falling back to the cue itself verbatim when auto_pick is
    // off) into a deduplicated anchor list.
    let mut resolved = serde_json::Map::new();
    let mut anchors: Vec<String> = Vec::new();
    for cue in &origins {
        let mut resolve_args = pick(arguments, &["dice_floor", "semantic_floor"]);
        resolve_args["context"] = json!(context);
        resolve_args["cue"] = json!(cue);
        if let Some(limit) = arguments.get("resolve_limit").filter(|v| !v.is_null()) {
            resolve_args["limit"] = limit.clone();
        }
        let candidates = call_tool("resolve", resolve_args)?
            .get("result")
            .cloned()
            .unwrap_or(Value::Array(Vec::new()));
        let picked = if auto_pick {
            candidates
                .as_array()
                .and_then(|list| list.first())
                .and_then(|first| first.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        } else {
            Some(cue.clone())
        };
        resolved.insert(cue.clone(), candidates);
        if let Some(picked) = picked
            && !anchors.contains(&picked)
        {
            anchors.push(picked);
        }
    }

    // Step 2: describe every anchor — skippable via describe_first: false.
    let mut outline = serde_json::Map::new();
    if describe_first {
        for anchor in &anchors {
            let described =
                call_tool("describe", json!({ "context": context, "concept": anchor }))?
                    .get("result")
                    .cloned()
                    .unwrap_or(Value::Null);
            outline.insert(anchor.clone(), described);
        }
    }

    // Step 3: gather associations — query (only when labels pins the
    // facets) then always activate, deduplicated by
    // (subject, label, object) with query's matches taking priority
    // over activate's (query runs first and wins the dedupe).
    let mut associations: Vec<Value> = Vec::new();
    let mut activations: Vec<Value> = Vec::new();
    let mut seen_triples: HashSet<(String, String, String)> = HashSet::new();
    if !anchors.is_empty() {
        if let Some(labels) = arguments.get("labels").filter(|v| !v.is_null()) {
            let matched = call_tool(
                "query",
                json!({ "context": context, "subject": anchors, "label": labels }),
            )?;
            for entry in matched
                .get("result")
                .and_then(|result| result.get("matches"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                match triple_of(entry) {
                    Some(triple) => {
                        if seen_triples.insert(triple) {
                            associations.push(entry.clone());
                        }
                    }
                    None => associations.push(entry.clone()),
                }
            }
        }
        let mut activate_args = json!({ "context": context, "origins": anchors });
        if let Some(decay) = arguments.get("activate_decay").filter(|v| !v.is_null()) {
            activate_args["decay"] = decay.clone();
        }
        if let Some(limit) = arguments.get("activate_limit").filter(|v| !v.is_null()) {
            activate_args["limit"] = limit.clone();
        }
        let page = call_tool("activate", activate_args)?;
        activations = page
            .get("result")
            .and_then(|result| result.get("matches"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for activation in &activations {
            let association = activation
                .get("association")
                .cloned()
                .unwrap_or(Value::Null);
            match triple_of(&association) {
                Some(triple) => {
                    if seen_triples.insert(triple) {
                        associations.push(association);
                    }
                }
                None => associations.push(association),
            }
        }
    }

    // Step 4: fetch a citation for every located attribution,
    // deduplicated by (source, paragraph). A locator whose passage was
    // never stored (or was retracted) is skipped rather than failing
    // the whole call — the graph fact still stands; any other failure
    // (auth, a downed server) aborts immediately.
    let mut citations: Vec<Value> = Vec::new();
    if fetch_citations {
        let mut wanted: Vec<(String, u64)> = Vec::new();
        let mut seen_keys: HashSet<(String, u64)> = HashSet::new();
        for association in &associations {
            for attribution in association
                .get("attributions")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let (Some(source), Some(paragraph)) = (
                    attribution.get("source").and_then(Value::as_str),
                    attribution.get("paragraph").and_then(Value::as_u64),
                ) else {
                    continue;
                };
                let key = (source.to_string(), paragraph);
                if seen_keys.insert(key.clone()) {
                    wanted.push(key);
                }
            }
        }
        for (source, paragraph) in wanted {
            match call_tool(
                "cite_passage",
                json!({ "context": context, "source": source, "paragraph": paragraph }),
            ) {
                Ok(response) => citations.push(json!({
                    "source": source,
                    "paragraph": paragraph,
                    "citation": response.get("result").cloned().unwrap_or(Value::Null),
                })),
                Err(message) if message.starts_with("HTTP 404") => continue,
                Err(message) => return Err(message),
            }
        }
    }

    // Step 5: text-lane fallback — only when the caller named a
    // fallback query, and (by default) only when no associations were
    // gathered. The search's result is `{plan, hits}` (#151):
    // `passage_hits` keeps its historical array contract, and the plan
    // rides beside it as `search_plan` — null when no fallback ran,
    // so "the search never happened" and "the semantic lane was
    // skipped" stay distinguishable here too.
    let mut passage_hits = Value::Array(Vec::new());
    let mut search_plan = Value::Null;
    if let Some(text_fallback_query) = arguments.get("text_fallback_query").and_then(Value::as_str)
        && (!text_fallback_only_if_empty || associations.is_empty())
    {
        let mut search_args = json!({ "context": context, "query": text_fallback_query });
        if let Some(limit) = arguments.get("search_limit").filter(|v| !v.is_null()) {
            search_args["limit"] = limit.clone();
        }
        let mut page = call_tool("search_passages", search_args)?
            .get("result")
            .cloned()
            .unwrap_or(Value::Null);
        passage_hits = page
            .get_mut("hits")
            .map(Value::take)
            .unwrap_or(Value::Array(Vec::new()));
        search_plan = page.get_mut("plan").map(Value::take).unwrap_or(Value::Null);
    }

    Ok(json!({
        "resolved": resolved,
        "outline": outline,
        "associations": associations,
        "activations": activations,
        "citations": citations,
        "passage_hits": passage_hits,
        "search_plan": search_plan,
    }))
}
