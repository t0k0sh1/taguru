use serde_json::{Value, json};

fn object_schema(properties: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": properties, "required": required })
}

/// [`object_schema`] for the search tools, which target one context or
/// several: `context`, `contexts`, and `groups` join the given
/// properties, and an `anyOf` demands at least one (the `cite_passage`
/// precedent). `contexts` and `groups` combine — both are the
/// cross-context form — but `context` beside either is refused by
/// `route_tool`, where the message can say so; a schema can only say
/// "invalid".
pub(super) fn search_target_schema(properties: Value, required: &[&str]) -> Value {
    let mut schema = object_schema(properties, required);
    schema["properties"]["context"] =
        json!({ "type": "string", "description": "Context name (from list_contexts)" });
    schema["properties"]["contexts"] = json!({
        "type": "array",
        "items": { "type": "string" },
        "description": "several contexts at once — full names; every match comes back tagged with its context. Combines with groups; don't pass context beside it."
    });
    schema["properties"]["groups"] = json!({
        "type": "array",
        "items": { "type": "string" },
        "description": "group names (from list_groups) — each resolves to every context it reaches, nested children included, deduped against contexts and each other. Combines with contexts; don't pass context beside it."
    });
    schema["anyOf"] = json!([
        { "required": ["context"] },
        { "required": ["contexts"] },
        { "required": ["groups"] },
    ]);
    schema
}

/// Layers a second `anyOf` onto a [`search_target_schema`] result via
/// `allOf`, so both constraints hold at once — assigning straight into
/// `schema["anyOf"]` would silently replace the target-selection one
/// instead of adding to it.
fn require_any_of(mut schema: Value, alternatives: Value) -> Value {
    let target_any_of = schema.as_object_mut().unwrap().remove("anyOf").unwrap();
    schema["allOf"] = json!([{ "anyOf": target_any_of }, { "anyOf": alternatives }]);
    schema
}

/// Schema property `description` policy (#51): add one when a caller
/// cannot get the fact from the property's name, its own `type`, or
/// the tool's own `description` — a non-obvious default/ceiling
/// applied on omission, an `additionalProperties` map's key → value
/// shape, a deprecated-alias relationship, or a divergence from a
/// same-named property on a sibling tool (e.g. create vs update
/// semantics). Skip it when it would only restate the type or repeat
/// what the tool description already says. The same property, same
/// meaning, on two tools gets the same text; a real behavioral
/// difference gets stated, not silently dropped.
pub(super) fn tool_definitions() -> Vec<Value> {
    let context = json!({ "type": "string", "description": "Context name (from list_contexts)" });
    let match_after = json!({
        "type": "object",
        "description": "resume past the previous page's last match: copy {weight, subject, label, object} verbatim from it, plus context too when targeting several contexts. total stays constant across pages",
        "properties": {
            "weight": { "type": "number" },
            "subject": { "type": "string" },
            "label": { "type": "string" },
            "object": { "type": "string" },
            "context": { "type": "string", "description": "required when targeting several contexts (contexts/groups); omit for a single context" }
        },
        "required": ["weight", "subject", "label", "object"]
    });
    let tools = vec![
        (
            "list_contexts",
            "Routing directory: every context's name, description, stats (counts, top concepts, label sample), and usage counters (reads/empty_reads/writes, last-used times). Pick the search/ingest target here yourself.",
            object_schema(
                json!({
                    "limit": { "type": "integer", "minimum": 0, "description": "page size, keyset-paged by name (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "only contexts whose name sorts strictly after this one" },
                    "pinned": { "type": "boolean", "description": "only contexts with this pinned state" }
                }),
                &[],
            ),
        ),
        (
            "create_context",
            "Create a context. One context = one 文脈: one spelling, one referent — different things sharing a spelling get separate contexts. The description drives routing; say concretely what the context covers.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "pinned": { "type": "boolean", "description": "keep resident (always-hot contexts like glossaries)" },
                    "dice_floor": { "type": "number", "description": "fuzzy-entry floor (default 0.3)" },
                    "semantic_floor": { "type": "number", "description": "semantic-entry floor (default 0.35)" }
                }),
                &["name"],
            ),
        ),
        (
            "update_context",
            "Update description / pinned / dice_floor / semantic_floor.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "pinned": { "type": "boolean", "description": "omit to leave unchanged" },
                    "dice_floor": { "type": "number", "description": "omit to leave unchanged" },
                    "semantic_floor": { "type": "number", "description": "omit to leave unchanged" }
                }),
                &["name"],
            ),
        ),
        (
            "delete_context",
            "Delete a context and its files (irreversible).",
            object_schema(json!({ "name": { "type": "string" } }), &["name"]),
        ),
        (
            "rename_context",
            "Rename a context (admin role): the whole file family moves to the new name and every group naming it is rewritten to match. Fails if the destination name is already taken.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "to": { "type": "string", "description": "the new name" }
                }),
                &["name", "to"],
            ),
        ),
        (
            "list_groups",
            "Group directory: every group's name, description, member context names, and child group names. A group bundles contexts (many-to-many) and may nest child groups up to 3 levels (cycles refused) — organize related contexts under one name. Groups and contexts are separate namespaces.",
            object_schema(
                json!({
                    "limit": { "type": "integer", "minimum": 0, "description": "page size, keyset-paged by name (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "only groups whose name sorts strictly after this one" }
                }),
                &[],
            ),
        ),
        (
            "create_group",
            "Create a group bundling contexts and, optionally, child groups (nesting: at most 3 groups tall, never cyclic; each set holds at most 1000 names — past that, split into nested child groups). Every listed context and child group must already exist; membership never dangles — deleting a context or a group drops it from every group.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "contexts": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "initial member context names (from list_contexts)"
                    },
                    "groups": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "initial child group names (from list_groups)"
                    }
                }),
                &["name"],
            ),
        ),
        (
            "update_group",
            "Update a group's description and/or membership. add_contexts/remove_contexts and add_groups/remove_groups are deltas against the current members, not a replacement list; a name in both ends up a member. Added contexts and child groups must exist; removing a non-member is a no-op; nesting stays at most 3 groups tall and acyclic, and the resulting membership at most 1000 member contexts and 1000 child groups (removals apply first, so one request can trade members within the cap).",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string", "description": "omit to leave unchanged" },
                    "add_contexts": { "type": "array", "items": { "type": "string" } },
                    "remove_contexts": { "type": "array", "items": { "type": "string" } },
                    "add_groups": { "type": "array", "items": { "type": "string" } },
                    "remove_groups": { "type": "array", "items": { "type": "string" } }
                }),
                &["name"],
            ),
        ),
        (
            "delete_group",
            "Delete a group (irreversible). Only the bundling goes; the member contexts, the child groups, and their data are untouched — parents naming the group just drop the child.",
            object_schema(json!({ "name": { "type": "string" } }), &["name"]),
        ),
        (
            "rename_group",
            "Rename a group (admin role): the group's file moves to the new name and every OTHER group naming it as a child is rewritten to match. Fails if the destination name is already taken.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "to": { "type": "string", "description": "the new name" }
                }),
                &["name", "to"],
            ),
        ),
        (
            "add_associations",
            "Write facts as a batch (one document = one call, up to 10,000 associations; split larger documents), a source id on every element; single-fact calls cost a full durable write each, so collect a document's facts first. Discipline: check spellings with resolve/resolve_label and reuse before minting; don't re-assert paraphrases within one document; negation = positive label + negative weight; make implicit membership an explicit edge; weave ordered procedures with the three edges 最初の工程/次の工程/工程 (details in get_protocol). All-or-nothing: a rejected batch writes nothing (`integrity: \"nothing_written\"` in the error), and a rejection lists every offending `associations[i].field` as a path-addressed issue. Correct exactly those fields in your own copy of the batch and resend the COMPLETE batch — never delete an item to work around a rejection, never invent a fact that was not already there, and never call add_associations again for just the fixed items (that would silently omit everything else). If store_passages for the same source also runs, note it is a SEPARATE write — a document's facts can land while its passage store still fails, or vice versa; POST /import is the only all-or-nothing per-source call across both.",
            object_schema(
                json!({
                    "context": context,
                    "associations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": { "type": "string" },
                                "label": { "type": "string" },
                                "object": { "type": "string" },
                                "weight": { "type": "number" },
                                "source": { "type": "string" },
                                "paragraph": { "type": "integer", "description": "zero-based paragraph position" }
                            },
                            "required": ["subject", "label", "object", "weight"]
                        }
                    }
                }),
                &["context", "associations"],
            ),
        ),
        (
            "store_passages",
            "Register the original text behind each source id. Always finish an ingest with this; answers ground in originals looked up from attributions. Optionally attach doc2query questions per source ({source: [{paragraph, question}]}, paragraph = 0-based blank-line-separated position in THAT text): questions a user might type whose answer is that paragraph, phrased away from its wording — they embed beside the paragraph and catch question-shaped queries the text's own vector misses. Optionally attach section markers per source ({source: [{paragraph, section}]}, same paragraph numbering): a marker names where its section starts and the section implicitly governs every paragraph after it until the next marker or the passage's end — citation and every association read label their paragraph with the section that governs it. Optionally attach source metadata: tags ({source: [tag]}) and a document date ({source: epoch seconds} in dates) — search_passages can then pre-filter by tag and time; the server stamps stored_at itself. Storage replaces per source wholesale, metadata included. All-or-nothing: a rejected call writes nothing (`integrity: \"nothing_written\"`), and a rejection lists every offending path — `passages['src']`, `questions['src'][i].question`, `sections['src'][i].section`, `tags['src'][i]`, `dates['src']` — as a path-addressed issue naming the source AND the item index. Correct exactly those fields and resend the COMPLETE call (every source, every question/section/tag) rather than deleting an item or resending only the fixed ones — a partial resend silently drops whatever this call would have replaced wholesale.",
            object_schema(
                json!({
                    "context": context,
                    "passages": { "type": "object", "additionalProperties": { "type": "string" }, "description": "source → text" },
                    "questions": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "paragraph": { "type": "integer" },
                                    "question": { "type": "string" }
                                },
                                "required": ["paragraph", "question"]
                            }
                        }
                    },
                    "sections": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "paragraph": { "type": "integer" },
                                    "section": { "type": "string" }
                                },
                                "required": ["paragraph", "section"]
                            }
                        }
                    },
                    "tags": {
                        "type": "object",
                        "additionalProperties": { "type": "array", "items": { "type": "string" } },
                        "description": "source → tags (≤32 per source, ≤128 bytes each)"
                    },
                    "dates": {
                        "type": "object",
                        "additionalProperties": { "type": "integer" },
                        "description": "source → the document's own date (epoch seconds) — time filters prefer it over the server's stored_at stamp"
                    }
                }),
                &["context", "passages"],
            ),
        ),
        (
            "lookup_passages",
            "Fetch the passages behind attribution source ids — the answer-from-originals half of retrieval.",
            object_schema(
                json!({
                    "context": context,
                    "sources": { "type": "array", "items": { "type": "string" } }
                }),
                &["context", "sources"],
            ),
        ),
        (
            "list_sources",
            "Source ids with registered passages — targets for retract_source / lookup_passages, inventory for diff sync. Keyset-paged by id; total above the returned count means more pages. Beside the bare `sources` list, `entries` carries each source's metadata: stored_at (server-stamped epoch seconds), date (user-supplied), tags — absent keys mean the source has none.",
            object_schema(
                json!({
                    "context": context,
                    "limit": { "type": "integer", "minimum": 0, "description": "page size (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "only ids sorting strictly after this one" },
                    "prefix": { "type": "string", "description": "only ids starting with this text" }
                }),
                &["context"],
            ),
        ),
        (
            "resolve",
            "Resolve free wording to stored concept names (normalized entry, absorbs typos). The retrieval entry: use the canonical names it returns as origins for explore/activate. Each candidate says how it matched (kind: exact/alias = the cue IS a stored spelling; containment/fuzzy = it merely overlaps one) and carries a gloss of its heaviest facts — read the gloss before adopting a lookalike (京都 scores 0.67 against 東京都; the glosses tell them apart). Empty → reword, or lower dice_floor (e.g. 0.2) and retry.",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "dice_floor": { "type": "number", "description": "one-call override of the fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the semantic floor" },
                    "limit": { "type": "integer", "minimum": 0, "description": "max candidates (default/ceiling 1000)" }
                }),
                &["context", "cue"],
            ),
        ),
        (
            "resolve_label",
            "resolve, for relation labels. Use before writes (check before mint) and to pick query labels.",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "dice_floor": { "type": "number", "description": "one-call override of the fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the semantic floor" },
                    "limit": { "type": "integer", "minimum": 0, "description": "max candidates (default/ceiling 1000)" }
                }),
                &["context", "cue"],
            ),
        ),
        (
            "explain_resolve",
            "Why didn't (or did) a concept come back for a cue — one call instead of re-running resolve with varied floors and cross-referencing by hand. Name the cue AND the concept you expected; the answer is the first verdict that applies: not_in_vocabulary (nearest stored spellings attached — register an alias?), cue_resolved_exactly (the cue IS another stored spelling; the exact tier answers alone), below_floor (its actual score vs the dice_floor in effect — the floor that would have shown it), below_cutoff (passed the floor, lost on limit), semantic_not_run / semantic_below_floor (whether the fallback tier joined, and its gloss cosine vs the semantic floor when it did), or served. Pass the same overrides as the resolve call being questioned.",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "expected": { "type": "string", "description": "the concept you expected among the candidates" },
                    "dice_floor": { "type": "number", "description": "one-call override of the fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the semantic floor" },
                    "limit": { "type": "integer", "minimum": 0, "description": "max candidates (default/ceiling 1000)" }
                }),
                &["context", "cue", "expected"],
            ),
        ),
        (
            "explain_resolve_label",
            "explain_resolve, for relation labels.",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "expected": { "type": "string", "description": "the label you expected among the candidates" },
                    "dice_floor": { "type": "number", "description": "one-call override of the fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the semantic floor" },
                    "limit": { "type": "integer", "minimum": 0, "description": "max candidates (default/ceiling 1000)" }
                }),
                &["context", "cue", "expected"],
            ),
        ),
        (
            "describe",
            "A concept's outline: which labels carry how many facts, per role. Check a hub here first, then query just the labels you need — never pull a whole profile blind.",
            object_schema(
                json!({ "context": context, "concept": { "type": "string" } }),
                &["context", "concept"],
            ),
        ),
        (
            "query",
            "Position-pinned search. subject/label/object each take a string or an array (array = match any); at least one of the three must be given — leaving all three out is refused rather than matching everything. Outline with describe, then narrow by label. Targets one context (context) or several at once (contexts and/or groups) — cross-context matches carry their context, and past the limit the strongest |weight| survives (weights share one scale).",
            require_any_of(
                search_target_schema(
                    json!({
                        "subject": { "type": ["string", "array"] },
                        "label": { "type": ["string", "array"] },
                        "object": { "type": ["string", "array"] },
                        "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" },
                        "after": match_after
                    }),
                    &[],
                ),
                json!([
                    { "required": ["subject"] },
                    { "required": ["label"] },
                    { "required": ["object"] },
                ]),
            ),
        ),
        (
            "recall",
            "Every association touching the cue, whatever its position. Use query when the role matters. Targets one context (context) or several at once (contexts and/or groups) — cross-context matches carry their context, and past the limit the strongest |weight| survives (weights share one scale).",
            search_target_schema(
                json!({
                    "cue": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" },
                    "after": match_after
                }),
                &["cue"],
            ),
        ),
        (
            "activate",
            "Spread activation from origins, strongest first (path shows the route). The main tool for gathering related knowledge. strength orders within one call only.",
            object_schema(
                json!({
                    "context": context,
                    "origins": { "type": "array", "items": { "type": "string" } },
                    "decay": { "type": "number", "description": "default 0.5" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 20" }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "explore",
            "Exhaustive structural walk with hop distances, for unranked neighborhood views. Truncation keeps the nearest hops (watch total).",
            object_schema(
                json!({
                    "context": context,
                    "origins": { "type": "array", "items": { "type": "string" } },
                    "max_depth": { "type": "integer", "description": "hop ceiling; default and max 10" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" },
                    "after": {
                        "type": "object",
                        "description": "resume past the previous page's last recollection — copy every field verbatim from it. total stays constant across pages",
                        "properties": {
                            "distance": { "type": "integer" },
                            "subject": { "type": "string" },
                            "label": { "type": "string" },
                            "object": { "type": "string" }
                        },
                        "required": ["distance", "subject", "label", "object"]
                    }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "list_labels",
            "The relation vocabulary (canonical only). Read it before extracting to avoid spelling forks. Keyset-paged by label; total above the returned count means more pages.",
            object_schema(
                json!({
                    "context": context,
                    "limit": { "type": "integer", "minimum": 0, "description": "page size (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "only labels sorting strictly after this one" },
                    "prefix": { "type": "string", "description": "only labels starting with this text" }
                }),
                &["context"],
            ),
        ),
        (
            "get_aliases",
            "Registered aliases (alias → canonical), paged across both namespaces — concepts first, then labels. total above the returned count means more pages; continue with after = 'concept:<alias>' or 'label:<alias>' (the last entry shown).",
            object_schema(
                json!({
                    "context": context,
                    "limit": { "type": "integer", "minimum": 0, "description": "page size (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "'concept:<alias>' or 'label:<alias>' — the last entry of the previous page" },
                    "prefix": { "type": "string", "description": "only aliases (in either namespace) starting with this text" }
                }),
                &["context"],
            ),
        ),
        (
            "add_aliases",
            "Point alternate spellings at canonical names (entry-only; results always return canonicals). The fix when live wording misses. Cannot join two existing concepts — that would be a merge, which is rebuild territory.",
            object_schema(
                json!({
                    "context": context,
                    "concepts": { "type": "object", "additionalProperties": { "type": "string" }, "description": "alias → canonical" },
                    "labels": { "type": "object", "additionalProperties": { "type": "string" }, "description": "alias → canonical" }
                }),
                &["context"],
            ),
        ),
        (
            "remove_aliases",
            "Withdraw mis-registered alias spellings (exact spellings, per namespace). The spelling stops resolving and is free to re-register; canonicals and their knowledge are untouched. Canonical names are refused — removal cannot unname a record.",
            object_schema(
                json!({
                    "context": context,
                    "concepts": { "type": "array", "items": { "type": "string" }, "description": "alias spellings to withdraw" },
                    "labels": { "type": "array", "items": { "type": "string" }, "description": "alias spellings to withdraw" }
                }),
                &["context"],
            ),
        ),
        (
            "retract_source",
            "Withdraw one source's (document's) contributions from graph and passage store. Diff sync for updated documents: retract the old version, then re-ingest the new. Concepts and edges remain; only weights come down.",
            object_schema(
                json!({ "context": context, "source": { "type": "string" } }),
                &["context", "source"],
            ),
        ),
        (
            "retract_association",
            "Withdraw one (subject, label, object) association outright — every source's contribution to that one edge, where retract_source would discard a whole document's. The surgical correction for a fact that should never have been asserted (an extraction error, a merge mistake). A fact that is merely CONTESTED wants a negative-weight assertion instead, which preserves the dispute as evidence. Names resolve through aliases; `retracted: false` means the triple named no live edge and nothing changed. The edge row stays visible at weight 0 until compaction; re-asserting the triple later just works.",
            object_schema(
                json!({
                    "context": context,
                    "subject": { "type": "string" },
                    "label": { "type": "string" },
                    "object": { "type": "string" }
                }),
                &["context", "subject", "label", "object"],
            ),
        ),
        (
            "search_passages",
            "Paragraph search over registered passages: a lexical lane (bigram BM25) fused with a semantic lane (paragraph embeddings) where the server has them. The text lane for knowledge that never fit triples (order, conditions, discourse) — look here too when graph search comes up short. The semantic lane works best on declarative phrasing: rephrase the information need as a plausible ANSWER sentence, not a question (query \"SSO is included in the Enterprise plan\", not \"What plan includes SSO?\") — the guess only has to be shaped like the text you hope to find. Optional source filters run BEFORE the lanes: tags (any-of, on tags stored with the source) and a half-open time window [since, until) in epoch seconds over each source's date ?? stored_at — sources with neither timestamp, or no tags, never match the respective filter kind. The result is {plan, hits}: each hit names its paragraph (source + paragraph) and reports per-lane rank/score in `lanes` (a hit only the vector lane surfaced is exactly the paraphrase case the lexical lane cannot see), and `plan` says per searched context whether each lane actually ran — and why not when it did not (embeddings off, nothing embedded yet, model or vector width changed, provider refused) — plus the vector lane's effective cosine floor and, under a filter, how many sources were eligible of how many stored; check it before reading empty hits as \"not in the corpus\". Targets one context (context) or several at once (contexts and/or groups) — cross-context hits carry their context and interleave by per-context rank; scores compare within one context only.",
            search_target_schema(
                json!({
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 5" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the vector lane's cosine floor (0-1); floors only the semantic lane — BM25-only hits still return" },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "only sources carrying at least one of these tags may answer" },
                    "since": { "type": "integer", "description": "only sources whose date ?? stored_at is at or after this (epoch seconds)" },
                    "until": { "type": "integer", "description": "only sources whose date ?? stored_at is strictly before this (epoch seconds)" }
                }),
                &["query"],
            ),
        ),
        (
            "search_communities",
            "Global search over a context's community-summary artifact (built offline by `taguru communities`): ranked LLM summaries of densely connected concept clusters — for corpus-overview questions (\"what are the main themes?\") that passage and graph search answer poorly. Each hit names its community, the matched summary paragraph, hierarchy level (0 = finest), member concepts with strengths, and concept_count; the response's `stale` flag means the source graph moved since derivation — the summaries describe an older graph, served honestly rather than withheld. The artifact is an ordinary context (default '{context}::communities'; `derived` overrides, for artifacts built with --into); a missing artifact is a refusal naming the build command, never an empty result. One context per call.",
            object_schema(
                json!({
                    "context": context,
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 5" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the artifact's vector-lane cosine floor (0-1)" },
                    "derived": { "type": "string", "description": "the artifact context to search; omitted means '{context}::communities'" }
                }),
                &["context", "query"],
            ),
        ),
        (
            "explain_search",
            "Why didn't (or did) a source appear in search_passages — one call instead of orchestrating search, citations, and lowered limits by hand. Name the query AND the source (optionally which paragraph) you expected; the answer is the first verdict that applies: not_stored (never ingested here, or retracted), filtered_out (the request's tags/since/until filter excludes the source — the search never considered it), no_term_overlap (the query's terms and the paragraph's terms side by side, as strings — the spelling-mismatch case: stored under 酒蔵, you searched 酒造 — register an alias or reword), below_cutoff (its actual rank, the score cutoff at your limit, and a verified limit that reaches it), or served (its rank — it WAS there). Evidence carries per-term tf/df/BM25 contributions and the vector lane's cosine or the reason that lane never ran. Pass the SAME tags/since/until as the search being explained, or the explanation accounts for a call nobody made. One context per call.",
            object_schema(
                json!({
                    "context": context,
                    "query": { "type": "string" },
                    "source": { "type": "string", "description": "the source you expected among the hits" },
                    "paragraph": { "type": "integer", "description": "zero-based paragraph position; omitted picks the source's best showing" },
                    "limit": { "type": "integer", "minimum": 0, "description": "the search call being explained (default 5)" },
                    "semantic_floor": { "type": "number", "description": "the floor override of the search call being explained — pass the same value" },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "the tag filter of the search call being explained — pass the same values" },
                    "since": { "type": "integer", "description": "the time window's inclusive start (epoch seconds) of the search call being explained" },
                    "until": { "type": "integer", "description": "the time window's exclusive end (epoch seconds) of the search call being explained" }
                }),
                &["context", "query", "source"],
            ),
        ),
        (
            "cite_passage",
            "Fetch one located, verbatim excerpt from a registered source by paragraph position: the citation counterpart of lookup_passages' whole-document dereference. Returns the exact paragraph text plus source and section provenance.",
            json!({
                "type": "object",
                "properties": {
                    "context": context,
                    "source": { "type": "string" },
                    "paragraph": { "type": "integer", "description": "zero-based paragraph position" },
                    "index": { "type": "integer", "description": "deprecated alias for `paragraph`; kept for pre-#35 callers, prefer `paragraph`" }
                },
                "required": ["context", "source"],
                "anyOf": [
                    { "required": ["paragraph"] },
                    { "required": ["index"] }
                ]
            }),
        ),
        (
            "retrieve",
            "The composed retrieval loop the SDKs' Context.retrieve() runs, as one call: resolve each origin cue to an anchor (auto-picking the top candidate; every candidate, gloss included, still comes back under resolved so a bad auto-pick is visible), describe each anchor, gather associations (query when labels pins the facets, activate always), fetch a citation for every located attribution, and optionally fall back to passage search. origins must already be extracted entity names, not a question — decomposing a question and phrasing a declarative text_fallback_query are the caller's job. citations rides back as a list of {source, paragraph, citation} (paragraphs missing a stored passage are silently skipped, same as the SDKs). When the fallback search ran, its hits land in passage_hits and its execution plan in search_plan (null when no fallback ran).",
            object_schema(
                json!({
                    "context": context,
                    "origins": {
                        "type": ["string", "array"],
                        "items": { "type": "string" },
                        "description": "cue(s) to resolve into anchors"
                    },
                    "labels": {
                        "type": ["string", "array"],
                        "items": { "type": "string" },
                        "description": "relation labels to query on, alongside the always-run activate"
                    },
                    "dice_floor": { "type": "number", "description": "one-call override of the resolve fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the resolve semantic floor" },
                    "resolve_limit": { "type": "integer", "minimum": 0, "description": "max resolve candidates per origin (default/ceiling 1000)" },
                    "auto_pick": { "type": "boolean", "description": "adopt each origin's top resolve candidate as its anchor; false uses the cue itself verbatim (default true)" },
                    "describe_first": { "type": "boolean", "description": "describe every anchor before gathering associations (default true)" },
                    "activate_decay": { "type": "number", "description": "activate's decay (default 0.5)" },
                    "activate_limit": { "type": "integer", "minimum": 0, "description": "activate's limit (default 20)" },
                    "fetch_citations": { "type": "boolean", "description": "resolve every located attribution into a cited passage (default true)" },
                    "text_fallback_query": { "type": "string", "description": "declarative-phrasing query for a search_passages fallback pass; omitted runs no fallback" },
                    "text_fallback_only_if_empty": { "type": "boolean", "description": "only run the fallback when no associations were gathered (default true)" },
                    "search_limit": { "type": "integer", "minimum": 0, "description": "the fallback search_passages call's limit (default 5)" }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "refresh_embeddings",
            "After ingesting, re-embed what changed (servers with embeddings only): the glosses (name + graph context) of new or changed concepts and labels, and — where the server opted in — the stored paragraphs. Makes paraphrases and question-shaped cues land through resolve's semantic fallback and search_passages' vector lane.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "audit_vocabulary",
            "Vocabulary health check: lexical fork candidates (青嶺酒蔵/青嶺酒造) and semantic ones (創業年/設立年; needs embeddings). Candidates, not verdicts — same referent → alias onto one canonical; different things that will keep colliding → record one ordinary 別物/distinct_from fact (one direction suffices; it lands in both glosses and warns future resolves). Run at ingest milestones.",
            object_schema(
                json!({
                    "context": context,
                    "dice_floor": { "type": "number", "description": "lexical floor (default 0.6)" },
                    "cosine_floor": { "type": "number", "description": "semantic floor (default 0.6)" }
                }),
                &["context"],
            ),
        ),
        (
            "audit_coverage",
            "Post-ingest audit: associations unreachable from origins (the document's main entities). Non-empty = membership edges are missing — add them before finishing.",
            object_schema(
                json!({
                    "context": context,
                    "origins": { "type": "array", "items": { "type": "string" } },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" },
                    "after": {
                        "type": "object",
                        "description": "resume past the previous page's last match — copy every field verbatim from it. total stays constant across pages",
                        "properties": {
                            "weight": { "type": "number" },
                            "subject": { "type": "string" },
                            "label": { "type": "string" },
                            "object": { "type": "string" }
                        },
                        "required": ["weight", "subject", "label", "object"]
                    }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "audit_drift",
            "Graph-vs-archive drift audit: three read-only checks in one call. Unsourced weight — edges carrying weight no named source explains, the residue plain associate() calls or an export/import round trip leave behind — worst-first, paginated, filterable with unsourced_floor. Dead-canonical aliases — alias spellings whose canonical concept or label has zero live edges left. Optionally (include_twins) the same lexical/semantic fork candidates audit_vocabulary finds. Run periodically to catch drift audit_coverage and audit_vocabulary don't: weight nothing ingested explains, and aliases pointing at names nothing uses anymore.",
            object_schema(
                json!({
                    "context": context,
                    "unsourced_floor": { "type": "number", "description": "minimum unsourced weight (by magnitude) to include; default: any amount at all" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" },
                    "after": {
                        "type": "object",
                        "description": "resume past the previous page's last unsourced match — copy every field verbatim from it. total stays constant across pages",
                        "properties": {
                            "weight": { "type": "number" },
                            "subject": { "type": "string" },
                            "label": { "type": "string" },
                            "object": { "type": "string" }
                        },
                        "required": ["weight", "subject", "label", "object"]
                    },
                    "include_twins": { "type": "boolean", "description": "also run the lexical/semantic fork-candidate sweep and include it as `twins` (default false — it's the same CPU-bound pairwise scan audit_vocabulary runs)" },
                    "dice_floor": { "type": "number", "description": "lexical floor (default 0.6); only used when include_twins is set" },
                    "cosine_floor": { "type": "number", "description": "semantic floor (default 0.6); only used when include_twins is set" }
                }),
                &["context"],
            ),
        ),
        (
            "flush",
            "Persist every dirty context to disk now; answers the flushed names (admin role). The backup handshake's first half: flush, then snapshot the data directory — the same discipline the operator docs describe, reachable by an agent tending its own memory.",
            object_schema(json!({}), &[]),
        ),
        (
            "export_context",
            "The whole context as an import batch stream (JSON Lines text) — one batch per source, create block first, aliases last; `taguru import` or POST /import restores it (per-source retract-then-apply, idempotent). The portable, version-independent backup of one context. The stream rides back as one text block: for very large contexts prefer GET /contexts/{name}/export over plain HTTP, or `taguru export` offline.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "export_group",
            "One group as its import-stream record (a single `taguru_group` JSON line — the group's complete truth); importing it restores the group as a whole-record replace. A context-scoped key exports exactly the slice its grant can read.",
            object_schema(
                json!({ "name": { "type": "string", "description": "Group name (from list_groups)" } }),
                &["name"],
            ),
        ),
        (
            "get_context",
            "One directory row by name — the cheap existence-and-stats check, without listing everything else.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "get_group",
            "One group's name, description, member contexts, and child groups.",
            object_schema(
                json!({ "name": { "type": "string", "description": "Group name (from list_groups)" } }),
                &["name"],
            ),
        ),
        (
            "compact",
            "Rebuild one context's on-disk image without the dead weight the append-only format accumulates (retracted edges, unlinked attributions, arena slack); answers what was shed and the resulting footprint (admin role). Content is preserved — this is maintenance, not a knowledge change.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "import",
            "Apply (or, with dry_run: true, preview) an NDJSON import stream — the same format `taguru import`/POST /import accept: a create block, associations, aliases, and passage per source, retract-then-apply and idempotent (admin role). A dry run writes nothing; its `associations`/`aliases` counts are optimistic previews, every other field exact. `taguru_group` records in the stream are not applied through this tool (their outcome is likewise not previewed) — use POST /import directly for a stream that carries any. Bounded by the server's request body cap (TAGURU_MAX_BODY_BYTES, 8 MiB by default) — and smaller over the /mcp HTTP transport, where the stream is escaped into the JSON-RPC envelope that must itself fit that cap — with a hard 32 MiB tool ceiling above it; a larger stream needs POST /import or `taguru import` directly. This is the ONE all-or-nothing call across facts+aliases+passage for a source; a rejection identifies the failing batch/source/line/path and reports write integrity explicitly: `nothing_written` when no batch landed yet, or a `durable_prefix` naming exactly how many earlier batches in this stream already did (never implying any part of the REJECTED batch itself was accepted — each batch is whole-or-none). Correct exactly the named path and resend the COMPLETE remaining stream (every batch from the failure point on, unless already fixed and durable) — never delete the offending line, never resend only a subset.",
            object_schema(
                json!({
                    "stream": { "type": "string", "description": "NDJSON import stream (one taguru_batch/taguru_group/fact/alias/passage line per row)" },
                    "dry_run": { "type": "boolean", "description": "preview without writing anything (default false)" }
                }),
                &["stream"],
            ),
        ),
        (
            "get_protocol",
            "The complete manual: ingest discipline and retrieval loop.",
            object_schema(json!({}), &[]),
        ),
    ];

    tools
        .into_iter()
        .map(|(name, description, schema)| {
            json!({ "name": name, "description": description, "inputSchema": schema })
        })
        .collect()
}
