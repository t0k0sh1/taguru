use std::collections::HashSet;

use serde_json::{Value, json};

use super::args::{need, optional_bool, optional_string, pick};
use super::route::route_tool;
use crate::trace::span;

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

/// Which transport composed this retrieval — the one fact
/// [`run_retrieve_bounded`] cannot see for itself, and the only
/// per-caller value the `taguru.retrieve` span carries (ADR 0008 §5).
/// The parent span comes free from `tracing`'s current-span stack —
/// `POST /mcp`'s request span for [`Transport::RemoteMcp`], the
/// bridge's `taguru.tool_call` for [`Transport::StdioMcp`] — so this
/// is the only thing a caller needs to say about itself.
#[derive(Clone, Copy)]
pub enum Transport {
    #[allow(dead_code)]
    // consumed by taguru's remote_mcp.rs; taguru-mcp only ever constructs StdioMcp
    RemoteMcp,
    StdioMcp,
}

impl Transport {
    fn as_str(self) -> &'static str {
        match self {
            Transport::RemoteMcp => "remote_mcp",
            Transport::StdioMcp => "stdio_mcp",
        }
    }
}

// ADR 0008 §7 event codes this composition owns — the `taguru.reason`
// values a `taguru.skip` event on `taguru.retrieve` or one of its
// phases carries. Lane-level codes (`vector_*`, `zero_limit`, ...)
// live beside the lanes that emit them (`src/registry.rs`,
// `src/registry/search.rs`), not here.
const SKIP_DESCRIBE_DISABLED: &str = "describe_disabled";
const SKIP_NO_ANCHORS: &str = "no_anchors";
const SKIP_LABELS_ABSENT: &str = "labels_absent";
const SKIP_CITATIONS_DISABLED: &str = "citations_disabled";
const SKIP_CITATION_PASSAGE_MISSING: &str = "citation_passage_missing";
const SKIP_BUDGET_EXHAUSTED: &str = "budget_exhausted";
// Doubles as `taguru.fallback.reason`'s attribute vocabulary (recorded
// on the root span whether or not the fallback ran) and, for the two
// "did not run" values, as a `taguru.skip` event's `taguru.reason`.
// The two "did run" values are never an event reason — nothing to
// skip — only an attribute; kept in the same closed set anyway so the
// four states of "why did/didn't the fallback run" stay one
// vocabulary instead of two that could drift.
const FALLBACK_NOT_REQUESTED: &str = "fallback_not_requested";
const FALLBACK_SUPPRESSED: &str = "fallback_suppressed";
const FALLBACK_GRAPH_EMPTY: &str = "graph_empty";
const FALLBACK_ALWAYS: &str = "unconditional";

/// [`run_retrieve_bounded`] with no byte budget — every planned call
/// fires unconditionally. What the stdio bridge calls; `taguru-mcp.rs`'s
/// `dispatch_tool` documents why its composition stays uncapped.
#[allow(dead_code)] // consumed by the stdio bridge; the HTTP transport always calls run_retrieve_bounded instead
pub fn run_retrieve(
    arguments: &Value,
    call: impl FnMut(&'static str, String, Option<Value>) -> Result<String, String>,
) -> Result<Value, String> {
    run_retrieve_bounded(arguments, None, Transport::StdioMcp, call)
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
///
/// One `taguru.retrieve` span covers the whole call, with one child
/// span per step (ADR 0008 §5) — the root and status handling live
/// here, in the one place that sees whether the composition as a
/// whole succeeded; the steps themselves are [`retrieve_inner`], kept
/// separate because the many early `?` returns below would make any
/// other shape (a single function owning both the root span and every
/// early exit) unreadable.
/// The `taguru.retrieve` root span's full field set (ADR 0008 §5), factored
/// out so a `taguru.retrieve` span has the same shape everywhere one is
/// created — [`run_retrieve_bounded`] below, and the pre-flight deadline
/// guard in `remote_mcp.rs` that returns before this function is even
/// reached.
pub fn root_span(transport: Transport) -> tracing::Span {
    span!(
        "taguru.retrieve",
        otel.kind = "internal",
        taguru.operation = "retrieve",
        taguru.transport = transport.as_str(),
        taguru.origin.count = tracing::field::Empty,
        taguru.anchor.count = tracing::field::Empty,
        taguru.association.count = tracing::field::Empty,
        taguru.activation.count = tracing::field::Empty,
        taguru.citation.returned = tracing::field::Empty,
        taguru.passage.hit_count = tracing::field::Empty,
        taguru.fallback.ran = tracing::field::Empty,
        taguru.fallback.reason = tracing::field::Empty,
        taguru.dispatch.bytes = tracing::field::Empty,
        taguru.error.kind = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
    )
}

pub fn run_retrieve_bounded(
    arguments: &Value,
    budget: Option<usize>,
    transport: Transport,
    call: impl FnMut(&'static str, String, Option<Value>) -> Result<String, String>,
) -> Result<Value, String> {
    let root = root_span(transport);
    // Synchronous from here down, so a plain guard is correct — the
    // transport's own block_in_place/block_on keeps this thread's
    // tracing span stack intact around the whole call (ADR 0008 §10).
    let _entered = root.enter();
    let outcome = retrieve_inner(arguments, budget, &root, call);
    if let Err(message) = &outcome {
        // Set only on the span whose own operation failed to produce
        // its result — never inferred from a child's status — so a
        // retrieval that degrades but still answers is never marked
        // ERROR here (ADR 0008 §9).
        root.record("otel.status_code", "ERROR");
        root.record("taguru.error.kind", error_kind(message));
    }
    outcome
}

fn retrieve_inner(
    arguments: &Value,
    budget: Option<usize>,
    root: &tracing::Span,
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
            tracing::info!(taguru.reason = SKIP_BUDGET_EXHAUSTED, "taguru.skip");
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
    // Validated here, alongside the other options, not down at Step 5
    // where it's used: a wrong-typed value must be refused before this
    // function pays for any of Steps 1-4's resolve/describe/query/
    // citation calls, not after.
    let text_fallback_query = optional_string(arguments, "text_fallback_query")?;

    // Step 1: resolve each origin cue, auto-picking the top candidate
    // (or falling back to the cue itself verbatim when auto_pick is
    // off) into a deduplicated anchor list.
    //
    // `taguru.op` below is `SearchOp::as_str()`'s own spelling
    // (`src/metrics.rs`), copied rather than called: this file is
    // dual-included into the stdio bridge, which has no `metrics`
    // module to import from (see `src/mcp.rs`'s module doc for the
    // same asymmetry elsewhere). ADR 0008 §6 governs the vocabulary;
    // the metrics enum stays the source of truth for the HTTP-served
    // paths that CAN import it (`src/api/*`).
    let resolve_span = span!(
        "taguru.resolve",
        otel.kind = "internal",
        taguru.op = "resolve",
        taguru.origin.count = origins.len(),
        taguru.anchor.count = tracing::field::Empty,
    );
    let mut resolved = serde_json::Map::new();
    let mut anchors: Vec<String> = Vec::new();
    {
        let _guard = resolve_span.enter();
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
        resolve_span.record("taguru.anchor.count", anchors.len());
    }
    root.record("taguru.origin.count", origins.len());
    root.record("taguru.anchor.count", anchors.len());

    // Step 2: describe every anchor — skippable via describe_first: false.
    let mut outline = serde_json::Map::new();
    if describe_first {
        let describe_span = span!(
            "taguru.describe",
            otel.kind = "internal",
            taguru.anchor.count = anchors.len(),
        );
        let _guard = describe_span.enter();
        for anchor in &anchors {
            let described =
                call_tool("describe", json!({ "context": context, "concept": anchor }))?
                    .get("result")
                    .cloned()
                    .unwrap_or(Value::Null);
            outline.insert(anchor.clone(), described);
        }
    } else {
        tracing::info!(taguru.reason = SKIP_DESCRIBE_DISABLED, "taguru.skip");
    }

    // Step 3: gather associations — query (only when labels pins the
    // facets) then always activate, deduplicated by
    // (subject, label, object) with query's matches taking priority
    // over activate's (query runs first and wins the dedupe).
    let mut associations: Vec<Value> = Vec::new();
    let mut activations: Vec<Value> = Vec::new();
    let mut seen_triples: HashSet<(String, String, String)> = HashSet::new();
    if anchors.is_empty() {
        tracing::info!(taguru.reason = SKIP_NO_ANCHORS, "taguru.skip");
    } else {
        if let Some(labels) = arguments.get("labels").filter(|v| !v.is_null()) {
            let query_span = span!(
                "taguru.query",
                otel.kind = "internal",
                taguru.op = "query",
                taguru.anchor.count = anchors.len(),
                taguru.association.count = tracing::field::Empty,
            );
            let _guard = query_span.enter();
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
            query_span.record("taguru.association.count", associations.len());
        } else {
            tracing::info!(taguru.reason = SKIP_LABELS_ABSENT, "taguru.skip");
        }
        let activate_span = span!(
            "taguru.activate",
            otel.kind = "internal",
            taguru.op = "activate",
            taguru.anchor.count = anchors.len(),
            taguru.activation.count = tracing::field::Empty,
        );
        let _guard = activate_span.enter();
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
        activate_span.record("taguru.activation.count", activations.len());
    }
    root.record("taguru.association.count", associations.len());
    root.record("taguru.activation.count", activations.len());

    // Step 4: fetch a citation for every located attribution,
    // deduplicated by (source, paragraph). A locator whose passage was
    // never stored (or was retracted) is skipped rather than failing
    // the whole call — the graph fact still stands; any other failure
    // (auth, a downed server) aborts immediately.
    let mut citations: Vec<Value> = Vec::new();
    if fetch_citations {
        let citations_span = span!(
            "taguru.citations",
            otel.kind = "internal",
            taguru.citation.requested = tracing::field::Empty,
            taguru.citation.returned = tracing::field::Empty,
            taguru.citation.missing = tracing::field::Empty,
        );
        let _guard = citations_span.enter();
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
        citations_span.record("taguru.citation.requested", wanted.len());
        // Not a per-item event for each 404 — ADR 0008 §8's per-item
        // rule keeps citations aggregate-only; `missing` becomes one
        // count, recorded once below.
        let mut missing: usize = 0;
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
                Err(message) if message.starts_with("HTTP 404") => {
                    missing += 1;
                    continue;
                }
                Err(message) => return Err(message),
            }
        }
        citations_span.record("taguru.citation.returned", citations.len());
        if missing > 0 {
            citations_span.record("taguru.citation.missing", missing);
            tracing::info!(
                taguru.reason = SKIP_CITATION_PASSAGE_MISSING,
                taguru.citation.missing = missing,
                "taguru.skip",
            );
        }
    } else {
        tracing::info!(taguru.reason = SKIP_CITATIONS_DISABLED, "taguru.skip");
    }
    root.record("taguru.citation.returned", citations.len());

    // Step 5: text-lane fallback — only when the caller named a
    // fallback query, and (by default) only when no associations were
    // gathered. The search's result is `{plan, hits}` (#151):
    // `passage_hits` keeps its historical array contract, and the plan
    // rides beside it as `search_plan` — null when no fallback ran,
    // so "the search never happened" and "the semantic lane was
    // skipped" stay distinguishable here too. Any other result shape —
    // a pre-#151 server's bare array through the stdio bridge is the
    // realistic one — is refused loudly: answering `passage_hits: []`
    // for a search that DID find things would be a silent wrong
    // answer, the one failure mode worse than an error.
    let mut passage_hits = Value::Array(Vec::new());
    let mut search_plan = Value::Null;
    // Decided before it runs, so the reason is nameable either way —
    // as a `taguru.skip` event when it doesn't run, as the
    // `taguru.fallback.reason` attribute when it does (ADR 0008 §7).
    let fallback_reason = match (&text_fallback_query, text_fallback_only_if_empty) {
        (None, _) => FALLBACK_NOT_REQUESTED,
        (Some(_), true) if !associations.is_empty() => FALLBACK_SUPPRESSED,
        (Some(_), true) => FALLBACK_GRAPH_EMPTY,
        (Some(_), false) => FALLBACK_ALWAYS,
    };
    root.record("taguru.fallback.reason", fallback_reason);
    let ran = matches!(fallback_reason, FALLBACK_GRAPH_EMPTY | FALLBACK_ALWAYS);
    root.record("taguru.fallback.ran", ran);
    if !ran {
        tracing::info!(taguru.reason = fallback_reason, "taguru.skip");
    } else {
        // `ran` is true only from the `(Some(_), _)` arms above, so a
        // fallback query is guaranteed here.
        let text_fallback_query =
            text_fallback_query.expect("ran implies a fallback query was given");
        let fallback_span = span!(
            "taguru.passage_fallback",
            otel.kind = "internal",
            taguru.op = "search_passages",
            taguru.passage.hit_count = tracing::field::Empty,
        );
        let _guard = fallback_span.enter();
        let mut search_args = json!({ "context": context, "query": text_fallback_query });
        if let Some(limit) = arguments.get("search_limit").filter(|v| !v.is_null()) {
            search_args["limit"] = limit.clone();
        }
        let mut page = call_tool("search_passages", search_args)?
            .get("result")
            .cloned()
            .unwrap_or(Value::Null);
        match page.get_mut("hits").map(Value::take) {
            Some(hits @ Value::Array(_)) => {
                fallback_span.record(
                    "taguru.passage.hit_count",
                    hits.as_array().map(Vec::len).unwrap_or(0),
                );
                passage_hits = hits;
                search_plan = page.get_mut("plan").map(Value::take).unwrap_or(Value::Null);
            }
            _ => {
                return Err(
                    "search_passages answered without a 'hits' array — a server predating \
                     the #151 response shape? upgrade it to match this MCP binary"
                        .to_string(),
                );
            }
        }
    }
    root.record(
        "taguru.passage.hit_count",
        passage_hits.as_array().map(Vec::len).unwrap_or(0),
    );
    // The transport's own cap (`taguru.result.bytes`) is a separate
    // number — the composed JSON's true serialized size, known only
    // after `to_string()` back in the caller — this is the raw sum of
    // every dispatched call's response, before composition.
    root.record("taguru.dispatch.bytes", spent);

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

/// A request that never left the process because the outbound request
/// builder itself refused it (a URI too long, an unencodable header) —
/// before either transport's round trip, so no server-side
/// [`ToolError::structured`](super::protocol::ToolError::structured)
/// body is possible either way. Both `call_inner` (`src/remote_mcp.rs`)
/// and `Bridge::call` (`src/bin/taguru-mcp.rs`) format their `Err` with
/// this same prefix, rather than each spelling out its own wording, so
/// a tool failure reads identically no matter which transport composed
/// it — the invariant `call_inner`'s own doc comment already claims.
pub const REQUEST_BUILD_FAILED_PREFIX: &str = "could not build the outbound request";

/// Classifies a composed retrieval's own error text into ADR 0008
/// §9's closed `taguru.error.kind` vocabulary. String-matched against
/// the two transports' own error text — acceptable here specifically
/// because `call_inner`'s doc comment (`src/remote_mcp.rs`) already
/// makes byte-identical error text across both transports a
/// maintained invariant, and nowhere else in this codebase infers a
/// status from message prose.
pub fn error_kind(message: &str) -> &'static str {
    if message.starts_with("HTTP 404") {
        "not_found"
    } else if message.starts_with("HTTP 401") || message.starts_with("HTTP 403") {
        "unauthorized"
    } else if message.starts_with("HTTP 5") {
        "upstream_error"
    } else if message.contains("exceeded its budget") {
        "deadline_exceeded"
    } else if message.contains("response cap")
        || (message.contains("already exceeds") && message.contains("bytes"))
    {
        "result_too_large"
    } else if message.contains("cancelled") {
        "cancelled"
    } else if message.contains("missing required argument")
        || message.contains("must be a string or an array")
        || message.contains("past the per-request limit")
    {
        "invalid_argument"
    } else if message.contains("failed to reach the taguru server")
        || message.contains(REQUEST_BUILD_FAILED_PREFIX)
    {
        "transport"
    } else {
        "provider_error"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_kind_maps_every_named_transport_constant() {
        assert_eq!(error_kind("HTTP 404: not found"), "not_found");
        assert_eq!(error_kind("HTTP 401: unauthorized"), "unauthorized");
        assert_eq!(error_kind("HTTP 403: forbidden"), "unauthorized");
        assert_eq!(error_kind("HTTP 500: boom"), "upstream_error");
        assert_eq!(error_kind("HTTP 503: unavailable"), "upstream_error");
        assert_eq!(
            error_kind(
                "request exceeded its budget before this operation could start; narrow the \
                 query or raise TAGURU_REQUEST_TIMEOUT_SECS"
            ),
            "deadline_exceeded"
        );
        assert_eq!(
            error_kind(
                "tool result exceeds the MCP response cap (TAGURU_MCP_MAX_RESULT_BYTES); \
                 narrow the call"
            ),
            "result_too_large"
        );
        assert_eq!(
            error_kind(
                "retrieve's composed result already exceeds 1000 bytes after the 'resolve' call"
            ),
            "result_too_large"
        );
        assert_eq!(
            error_kind("missing required argument 'origins'"),
            "invalid_argument"
        );
        assert_eq!(
            error_kind("argument 'origins' must be a string or an array of strings"),
            "invalid_argument"
        );
        assert_eq!(
            error_kind("argument 'origins' carries 2000 cues, past the per-request limit of 1000"),
            "invalid_argument"
        );
        assert_eq!(error_kind("failed to reach the taguru server"), "transport");
        assert_eq!(
            error_kind(&format!("{REQUEST_BUILD_FAILED_PREFIX}: uri too long")),
            "transport"
        );
        assert_eq!(error_kind("the request was cancelled"), "cancelled");
        assert_eq!(
            error_kind("tool 'resolve' returned invalid JSON: EOF"),
            "provider_error"
        );
    }

    /// `call_inner` (`src/remote_mcp.rs`) and `Bridge::call`
    /// (`src/bin/taguru-mcp.rs`) both format a request-build failure by
    /// hand, out of this crate's reach to check at compile time (one
    /// lives in the `taguru` binary, the other in `taguru-mcp`) — this
    /// pins the shared constant itself, the one piece both call sites
    /// actually import rather than each spelling out their own prefix.
    #[test]
    fn request_build_failed_prefix_is_shared_between_transports() {
        assert_eq!(
            REQUEST_BUILD_FAILED_PREFIX,
            "could not build the outbound request"
        );
        assert_eq!(
            error_kind(&format!("{REQUEST_BUILD_FAILED_PREFIX}: bad header")),
            "transport"
        );
    }
}
