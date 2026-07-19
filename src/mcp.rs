//! The MCP surface, shared by both transports: the stdio bridge
//! (`taguru-mcp`) and the server's own `POST /mcp` endpoint. Tool
//! definitions, the tool → HTTP request mapping, and the JSON-RPC
//! framing live here exactly once; a transport differs only in how a
//! routed request is executed (a ureq round trip from the bridge, an
//! in-process `Router` call on the server) and in how replies travel
//! back (stdout lines vs HTTP responses).
//!
//! Compiled into both binaries via `#[path]` — deliberately not part
//! of the library surface, which stays [`crate::context`]-only.

// Explicit `#[path]` on every child: `taguru-mcp.rs` loads this file itself
// via `#[path = "../mcp.rs"]`, and a `#[path]`-loaded file's own unpathed
// `mod` children resolve relative to ITS directory directly (no implicit
// `mcp/` subdirectory the way an ordinarily-reached module gets) — so
// unpathed children here would 404 under that binary despite building fine
// under `taguru`'s plain `mod mcp;`. Spelling out the path makes resolution
// identical for both.
#[path = "mcp/args.rs"]
mod args;
#[path = "mcp/protocol.rs"]
mod protocol;
#[path = "mcp/retrieve.rs"]
mod retrieve;
#[path = "mcp/route.rs"]
mod route;
#[path = "mcp/schema.rs"]
mod schema;

// Both binaries compile this re-export, but each calls only part of it
// directly (the other reaches the rest only internally, or not at all) —
// the same asymmetry `cancelled_request_id`'s and `run_retrieve`'s own
// `#[allow(dead_code)]` already document at their definitions, one layer
// further out.
#[allow(unused_imports)]
pub use protocol::{
    Call, FALLBACK_PROTOCOL_VERSION, Message, cancelled_request_id, classify, error_response,
    initialize_result, response, tool_response, tools_result,
};
#[allow(unused_imports)]
pub use retrieve::{run_retrieve, run_retrieve_bounded};
pub use route::route_tool;

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{args::*, protocol::*, retrieve::*, route::*, schema::*};

    /// The wiring invariant a new tool is most likely to break: every
    /// advertised tool definition must route to an HTTP request. An
    /// argument object carrying every required key satisfies whichever
    /// subset each tool needs.
    #[test]
    fn every_advertised_tool_routes_to_a_request() {
        let arguments = json!({
            "name": "ctx", "context": "ctx", "cue": "x", "concept": "x",
            "origins": ["x"], "associations": [], "passages": {},
            "sources": ["s"], "source": "s", "query": "q", "paragraph": 0,
            "stream": "{}", "to": "ctx2", "expected": "x",
            "subject": "s", "label": "l", "object": "o",
        });
        for tool in tool_definitions() {
            let name = tool["name"].as_str().expect("definitions carry names");
            // retrieve is a composed multi-call tool with no single
            // (method, path, body) to map onto — run_retrieve's own
            // tests cover it.
            if name == "retrieve" {
                continue;
            }
            let routed = route_tool(name, &arguments);
            assert!(routed.is_ok(), "tool '{name}' does not route: {routed:?}");
            let (method, path, _) = routed.unwrap();
            assert!(
                matches!(method, "GET" | "PUT" | "PATCH" | "POST" | "DELETE"),
                "tool '{name}' uses unknown method {method}"
            );
            assert!(path.starts_with('/'), "tool '{name}' path: {path}");
        }
    }

    /// The HTTP layer deserializes every `limit` as `Option<usize>`, so a
    /// negative value should be refused at MCP schema validation instead
    /// of surfacing as a later deserialization failure.
    #[test]
    fn every_limit_property_has_a_minimum_of_zero() {
        for tool in tool_definitions() {
            let properties = &tool["inputSchema"]["properties"];
            if let Some(limit) = properties.get("limit") {
                assert_eq!(
                    limit["minimum"],
                    json!(0),
                    "tool '{}' limit lacks minimum: 0",
                    tool["name"]
                );
            }
        }
    }

    /// The search tools target one context or several: `contexts`
    /// and/or `groups` route to the cross-context path with the arrays
    /// in the body; `context` keeps the historical per-context route,
    /// body unchanged.
    #[test]
    fn search_tools_route_to_the_cross_context_paths_on_contexts() {
        let (method, path, body) =
            route_tool("recall", &json!({"contexts": ["a", "b"], "cue": "x"})).unwrap();
        assert_eq!((method, path.as_str()), ("POST", "/recall"));
        assert_eq!(body.unwrap(), json!({"contexts": ["a", "b"], "cue": "x"}));

        let (_, path, body) =
            route_tool("query", &json!({"contexts": ["a"], "subject": "s"})).unwrap();
        assert_eq!(path, "/query");
        assert_eq!(body.unwrap(), json!({"contexts": ["a"], "subject": "s"}));

        let (_, path, body) = route_tool(
            "search_passages",
            &json!({"contexts": ["a"], "query": "q", "semantic_floor": 0.5}),
        )
        .unwrap();
        assert_eq!(path, "/sources/search");
        assert_eq!(
            body.unwrap(),
            json!({"contexts": ["a"], "query": "q", "semantic_floor": 0.5})
        );

        // `groups` alone reaches the same route, and beside `contexts`
        // both ride the body.
        let (_, path, body) = route_tool("recall", &json!({"groups": ["g"], "cue": "x"})).unwrap();
        assert_eq!(path, "/recall");
        assert_eq!(body.unwrap(), json!({"groups": ["g"], "cue": "x"}));
        let (_, path, body) = route_tool(
            "search_passages",
            &json!({"contexts": ["a"], "groups": ["g"], "query": "q"}),
        )
        .unwrap();
        assert_eq!(path, "/sources/search");
        assert_eq!(
            body.unwrap(),
            json!({"contexts": ["a"], "groups": ["g"], "query": "q"})
        );

        // The single-context form is untouched, and the body never
        // carries the path-bound name.
        let (_, path, body) = route_tool("recall", &json!({"context": "a", "cue": "x"})).unwrap();
        assert_eq!(path, "/contexts/a/recall");
        assert_eq!(body.unwrap(), json!({"cue": "x"}));
    }

    /// The one-context form beside the cross-context form is ambiguous
    /// and no target at all is no search — each refusal says which way
    /// to fix the call, and an explicit null counts as an omission (the
    /// `pick` rule).
    #[test]
    fn search_tools_refuse_an_ambiguous_or_absent_target() {
        let ambiguous = "pass either 'context' or 'contexts'/'groups', not both";
        assert_eq!(
            route_tool(
                "recall",
                &json!({"context": "a", "contexts": ["b"], "cue": "x"})
            ),
            Err(ambiguous.to_string())
        );
        assert_eq!(
            route_tool(
                "recall",
                &json!({"context": "a", "groups": ["g"], "cue": "x"})
            ),
            Err(ambiguous.to_string())
        );
        let missing = "missing required argument 'context' (or 'contexts'/'groups', to search several at once)";
        assert_eq!(
            route_tool("search_passages", &json!({"query": "q"})),
            Err(missing.to_string())
        );
        assert_eq!(
            route_tool(
                "recall",
                &json!({"context": null, "contexts": null, "cue": "x"})
            ),
            Err(missing.to_string())
        );
    }

    #[test]
    fn unknown_tools_and_missing_arguments_are_refused() {
        assert_eq!(
            route_tool("no_such_tool", &json!({})),
            Err("unknown tool 'no_such_tool'".to_string())
        );
        // A context-scoped tool without its context argument names what
        // is missing instead of building a broken path.
        assert_eq!(
            route_tool("describe", &json!({"concept": "x"})),
            Err("missing required argument 'context'".to_string())
        );
    }

    /// `associations` is schema-required (unlike, say, `add_aliases`'s
    /// concepts/labels, which default to empty maps server-side). A
    /// caller that omits it made a mistake, not an empty-batch request
    /// — omission must refuse, not silently route as `[]` and come
    /// back a do-nothing 200.
    #[test]
    fn add_associations_without_the_associations_argument_is_refused() {
        assert_eq!(
            route_tool("add_associations", &json!({"context": "ctx"})),
            Err("missing required argument 'associations'".to_string())
        );
        // Explicit null is the same omission, not a value.
        assert_eq!(
            route_tool(
                "add_associations",
                &json!({"context": "ctx", "associations": null})
            ),
            Err("missing required argument 'associations'".to_string())
        );
        // A deliberate empty batch is a value, not an omission — it
        // still routes.
        assert!(
            route_tool(
                "add_associations",
                &json!({"context": "ctx", "associations": []})
            )
            .is_ok()
        );
    }

    /// A required string argument present with the wrong JSON type is a
    /// caller mistake distinct from omission — say so, rather than blame a
    /// "missing" argument that was in fact supplied.
    #[test]
    fn a_required_argument_of_the_wrong_type_names_the_type_error() {
        assert_eq!(
            route_tool("describe", &json!({"context": 7, "concept": "x"})),
            Err("argument 'context' must be a string".to_string())
        );
    }

    /// When both context and its payload are missing, the context — the
    /// path segment resolved first — is the one reported, so the caller
    /// fixes the outer error before the inner one.
    #[test]
    fn add_associations_reports_the_missing_context_before_the_missing_payload() {
        assert_eq!(
            route_tool("add_associations", &json!({})),
            Err("missing required argument 'context'".to_string())
        );
    }

    /// Beyond `add_associations` (covered above), every other tool whose
    /// schema marks a body argument required must refuse routing when
    /// that argument is omitted instead of composing a request with the
    /// key silently absent — `pick` alone would drop it without a word,
    /// pushing the caller's mistake past this layer and into a slower,
    /// vaguer failure downstream.
    #[test]
    fn schema_required_body_arguments_are_refused_when_omitted() {
        let base = json!({
            "name": "ctx", "context": "ctx", "cue": "x", "concept": "x",
            "origins": ["x"], "passages": {}, "sources": ["s"], "source": "s",
            "query": "q", "paragraph": 0, "to": "ctx2", "expected": "x",
            "subject": "s", "label": "l", "object": "o",
        });
        let cases = [
            ("rename_context", "to"),
            ("rename_group", "to"),
            ("store_passages", "passages"),
            ("lookup_passages", "sources"),
            ("resolve", "cue"),
            ("resolve_label", "cue"),
            ("explain_resolve", "cue"),
            ("explain_resolve", "expected"),
            ("explain_resolve_label", "cue"),
            ("explain_resolve_label", "expected"),
            ("describe", "concept"),
            ("recall", "cue"),
            ("activate", "origins"),
            ("explore", "origins"),
            ("retract_source", "source"),
            ("retract_association", "subject"),
            ("retract_association", "label"),
            ("retract_association", "object"),
            ("search_passages", "query"),
            ("explain_search", "query"),
            ("explain_search", "source"),
            ("cite_passage", "source"),
            ("audit_coverage", "origins"),
        ];
        for (tool, key) in cases {
            let mut arguments = base.clone();
            arguments[key] = Value::Null;
            let routed = route_tool(tool, &arguments);
            assert!(
                routed.is_err(),
                "tool '{tool}' should refuse a missing '{key}', got {routed:?}"
            );
            let err = routed.unwrap_err();
            assert!(
                err.contains(key),
                "tool '{tool}' missing '{key}' error should name it, got: {err}"
            );
        }
    }

    /// `cite_passage` accepts either `paragraph` or its deprecated alias
    /// `index` (positive cases covered above); omitting both must refuse
    /// rather than route a citation request with neither name present.
    #[test]
    fn cite_passage_without_paragraph_or_index_is_refused() {
        let routed = route_tool(
            "cite_passage",
            &json!({"context": "sake", "source": "docs/aomine.md"}),
        );
        assert_eq!(
            routed,
            Err(
                "missing required argument 'paragraph' (or its deprecated alias 'index')"
                    .to_string()
            )
        );
    }

    /// Context names arrive as URL path segments; anything outside the
    /// unreserved set must be percent-encoded, byte by byte.
    #[test]
    fn context_names_are_percent_encoded_into_one_segment() {
        let (_, path, _) = route_tool("list_labels", &json!({"context": "日本 語/酒"})).unwrap();
        let segment = path
            .strip_prefix("/contexts/")
            .and_then(|rest| rest.strip_suffix("/labels"))
            .expect("path shape");
        assert!(!segment.contains('/'), "slash must be encoded: {path}");
        assert!(!segment.contains(' '), "space must be encoded: {path}");
        assert_eq!(segment, "%E6%97%A5%E6%9C%AC%20%E8%AA%9E%2F%E9%85%92");
    }

    #[test]
    fn pick_copies_only_present_non_null_keys() {
        let arguments = json!({"cue": "x", "limit": null, "extra": 7});
        assert_eq!(
            pick(&arguments, &["cue", "limit", "absent"]),
            json!({"cue": "x"})
        );
    }

    #[test]
    fn query_string_encodes_present_keys_and_skips_absent_or_null_ones() {
        let arguments = json!({"limit": 50, "after": null, "extra": "x"});
        assert_eq!(
            query_string(&arguments, &["limit", "after", "absent"]).unwrap(),
            "?limit=50"
        );
    }

    #[test]
    fn query_string_percent_encodes_string_values() {
        let arguments = json!({"after": "日本 語"});
        assert_eq!(
            query_string(&arguments, &["after"]).unwrap(),
            "?after=%E6%97%A5%E6%9C%AC%20%E8%AA%9E"
        );
    }

    #[test]
    fn query_string_is_empty_when_no_keys_are_present() {
        assert_eq!(query_string(&json!({}), &["limit", "after"]).unwrap(), "");
    }

    /// A present-but-wrong-typed argument must be refused like `need`
    /// and `optional_bool` refuse theirs, not silently dropped — the
    /// value never reaches a request for anything downstream to
    /// reject, so this is the only place the mistake can be caught.
    #[test]
    fn query_string_rejects_a_present_but_wrong_typed_value_instead_of_dropping_it() {
        let arguments = json!({"limit": [5]});
        assert_eq!(
            query_string(&arguments, &["limit", "after"]).unwrap_err(),
            "argument 'limit' must be a string, number, or boolean"
        );
    }

    /// `create_context`/`update_context` advertise `pinned: boolean`,
    /// and item 6 (#62) added `pinned`/`prefix` boolean/string filters
    /// to list tools — a bool argument must not silently vanish here.
    #[test]
    fn query_string_encodes_bool_values() {
        let arguments = json!({"pinned": true});
        assert_eq!(
            query_string(&arguments, &["pinned"]).unwrap(),
            "?pinned=true"
        );
        let arguments = json!({"pinned": false});
        assert_eq!(
            query_string(&arguments, &["pinned"]).unwrap(),
            "?pinned=false"
        );
    }

    /// list_contexts advertises limit/after and, when the caller
    /// supplies them, routes them onto the GET request's query string
    /// — the wiring the issue tracked was missing entirely.
    #[test]
    fn list_contexts_schema_advertises_limit_and_after() {
        let list_contexts = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "list_contexts")
            .expect("list_contexts is defined");
        let properties = &list_contexts["inputSchema"]["properties"];
        assert_eq!(properties["limit"]["type"], "integer");
        assert_eq!(properties["after"]["type"], "string");
    }

    #[test]
    fn list_contexts_routes_limit_and_after_onto_the_query_string() {
        let (method, path, body) =
            route_tool("list_contexts", &json!({"limit": 50, "after": "sake"})).unwrap();
        assert_eq!(method, "GET");
        assert_eq!(path, "/contexts?limit=50&after=sake");
        assert_eq!(body, None);
    }

    #[test]
    fn list_contexts_without_arguments_has_no_query_string() {
        let (_, path, _) = route_tool("list_contexts", &json!({})).unwrap();
        assert_eq!(path, "/contexts");
    }

    /// A templating slip that sends `limit` as a one-element array must
    /// surface as a routing error, not silently fall back to the
    /// server's default page size with no sign `limit` was ignored.
    #[test]
    fn list_contexts_rejects_a_wrong_typed_limit_instead_of_dropping_it() {
        let error = route_tool("list_contexts", &json!({"limit": [5]})).unwrap_err();
        assert_eq!(
            error,
            "argument 'limit' must be a string, number, or boolean"
        );
    }

    /// #62 item 6: `pinned` filters the directory (population, not a
    /// cursor) — advertised in the schema and routed onto the query
    /// string like `limit`/`after`.
    #[test]
    fn list_contexts_schema_advertises_pinned() {
        let list_contexts = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "list_contexts")
            .expect("list_contexts is defined");
        let properties = &list_contexts["inputSchema"]["properties"];
        assert_eq!(properties["pinned"]["type"], "boolean");
    }

    #[test]
    fn list_contexts_routes_pinned_onto_the_query_string() {
        let (_, path, _) = route_tool("list_contexts", &json!({"pinned": true})).unwrap();
        assert_eq!(path, "/contexts?pinned=true");
    }

    /// #62 item 6: `list_sources`/`list_labels`/`get_aliases` advertise
    /// and route `prefix` the same way — narrows the population, so it
    /// belongs beside `limit`/`after` in both schema and query string.
    #[test]
    fn list_sources_schema_advertises_prefix() {
        let list_sources = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "list_sources")
            .expect("list_sources is defined");
        let properties = &list_sources["inputSchema"]["properties"];
        assert_eq!(properties["prefix"]["type"], "string");
    }

    #[test]
    fn list_sources_routes_prefix_onto_the_query_string() {
        let (_, path, _) = route_tool(
            "list_sources",
            &json!({"context": "sake", "prefix": "doc-"}),
        )
        .unwrap();
        assert_eq!(path, "/contexts/sake/sources?prefix=doc-");
    }

    #[test]
    fn list_labels_schema_advertises_prefix() {
        let list_labels = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "list_labels")
            .expect("list_labels is defined");
        let properties = &list_labels["inputSchema"]["properties"];
        assert_eq!(properties["prefix"]["type"], "string");
    }

    #[test]
    fn list_labels_routes_prefix_onto_the_query_string() {
        let (_, path, _) =
            route_tool("list_labels", &json!({"context": "sake", "prefix": "産地"})).unwrap();
        assert_eq!(path, "/contexts/sake/labels?prefix=%E7%94%A3%E5%9C%B0");
    }

    #[test]
    fn get_aliases_schema_advertises_prefix() {
        let get_aliases = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "get_aliases")
            .expect("get_aliases is defined");
        let properties = &get_aliases["inputSchema"]["properties"];
        assert_eq!(properties["prefix"]["type"], "string");
    }

    #[test]
    fn get_aliases_routes_prefix_onto_the_query_string() {
        let (_, path, _) =
            route_tool("get_aliases", &json!({"context": "sake", "prefix": "a"})).unwrap();
        assert_eq!(path, "/contexts/sake/aliases?prefix=a");
    }

    /// #39: the schema had no `limit` and `route_tool` whitelisted only
    /// `origins`, so there was no way to raise the cap through this tool
    /// at all.
    #[test]
    fn audit_coverage_schema_advertises_limit() {
        let audit_coverage = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "audit_coverage")
            .expect("audit_coverage is defined");
        let properties = &audit_coverage["inputSchema"]["properties"];
        assert_eq!(properties["limit"]["type"], "integer");
    }

    #[test]
    fn audit_coverage_routes_limit_into_the_request_body() {
        let (method, path, body) = route_tool(
            "audit_coverage",
            &json!({"context": "sake", "origins": ["x"], "limit": 500}),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/unreachable_from");
        assert_eq!(body, Some(json!({"origins": ["x"], "limit": 500})));
    }

    #[test]
    fn audit_drift_schema_advertises_limit() {
        let audit_drift = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "audit_drift")
            .expect("audit_drift is defined");
        let properties = &audit_drift["inputSchema"]["properties"];
        assert_eq!(properties["limit"]["type"], "integer");
    }

    #[test]
    fn audit_drift_routes_every_option_into_the_request_body() {
        let (method, path, body) = route_tool(
            "audit_drift",
            &json!({
                "context": "sake",
                "unsourced_floor": 0.5,
                "limit": 25,
                "include_twins": true,
                "dice_floor": 0.7,
                "cosine_floor": 0.8
            }),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/drift/audit");
        assert_eq!(
            body,
            Some(json!({
                "unsourced_floor": 0.5,
                "limit": 25,
                "include_twins": true,
                "dice_floor": 0.7,
                "cosine_floor": 0.8
            }))
        );
    }

    /// #60: query/recall/explore/audit_coverage advertise `after` for
    /// resuming a page past its predecessor's last row — the MCP-layer
    /// wiring for the keyset cursors the HTTP side already accepts.
    /// Every field a client must copy verbatim is declared `required`,
    /// so a caller builds a whole cursor or none — not a partial one
    /// the downstream Rust struct would reject anyway.
    #[test]
    fn search_and_audit_tools_advertise_after() {
        let cases: [(&str, &[&str]); 5] = [
            ("query", &["weight", "subject", "label", "object"]),
            ("recall", &["weight", "subject", "label", "object"]),
            ("explore", &["distance", "subject", "label", "object"]),
            ("audit_coverage", &["weight", "subject", "label", "object"]),
            ("audit_drift", &["weight", "subject", "label", "object"]),
        ];
        for (name, required) in cases {
            let tool = tool_definitions()
                .into_iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("{name} is defined"));
            let after = &tool["inputSchema"]["properties"]["after"];
            assert_eq!(after["type"], "object", "tool '{name}' after");
            for field in required {
                assert!(
                    after["properties"].get(*field).is_some(),
                    "tool '{name}' after.properties.{field}"
                );
            }
            let actual_required: Vec<&str> = after["required"]
                .as_array()
                .unwrap_or_else(|| panic!("tool '{name}' after.required is an array"))
                .iter()
                .map(|value| value.as_str().unwrap())
                .collect();
            assert_eq!(actual_required, required, "tool '{name}' after.required");
        }
    }

    /// `after` rides straight through to the request body, whatever
    /// shape the caller sent — single-context `MatchCursor`,
    /// cross-context `CrossMatchCursor` (an extra `context` field), or
    /// explore's own `{distance, subject, label, object}`. `pick`
    /// forwards it verbatim; the downstream Rust struct is what
    /// actually validates the shape.
    #[test]
    fn search_and_audit_tools_route_after_into_the_request_body() {
        let cursor = json!({"weight": 0.5, "subject": "a", "label": "b", "object": "c"});
        let (_, _, body) = route_tool(
            "recall",
            &json!({"context": "sake", "cue": "x", "after": cursor}),
        )
        .unwrap();
        assert_eq!(body.unwrap()["after"], cursor);

        let cross_cursor = json!({
            "weight": 0.5, "context": "sake", "subject": "a", "label": "b", "object": "c"
        });
        let (_, _, body) = route_tool(
            "query",
            &json!({"contexts": ["sake"], "subject": "s", "after": cross_cursor}),
        )
        .unwrap();
        assert_eq!(body.unwrap()["after"], cross_cursor);

        let explore_cursor = json!({"distance": 2, "subject": "a", "label": "b", "object": "c"});
        let (_, _, body) = route_tool(
            "explore",
            &json!({"context": "sake", "origins": ["a"], "after": explore_cursor}),
        )
        .unwrap();
        assert_eq!(body.unwrap()["after"], explore_cursor);

        let (_, _, body) = route_tool(
            "audit_coverage",
            &json!({"context": "sake", "origins": ["a"], "after": cursor}),
        )
        .unwrap();
        assert_eq!(body.unwrap()["after"], cursor);

        let (_, _, body) =
            route_tool("audit_drift", &json!({"context": "sake", "after": cursor})).unwrap();
        assert_eq!(body.unwrap()["after"], cursor);
    }

    #[test]
    fn pick_with_alias_falls_back_to_the_old_key_name() {
        let arguments = json!({"source": "s", "index": 3});
        assert_eq!(
            pick_with_alias(&arguments, &["source", "paragraph"], "paragraph", "index"),
            json!({"source": "s", "paragraph": 3})
        );
    }

    #[test]
    fn pick_with_alias_prefers_the_canonical_key_when_both_are_present() {
        let arguments = json!({"source": "s", "paragraph": 1, "index": 99});
        assert_eq!(
            pick_with_alias(&arguments, &["source", "paragraph"], "paragraph", "index"),
            json!({"source": "s", "paragraph": 1})
        );
    }

    #[test]
    fn cite_passage_routes_to_the_citations_endpoint() {
        let (method, path, body) = route_tool(
            "cite_passage",
            &json!({"context": "sake", "source": "docs/aomine.md", "paragraph": 1}),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/citations");
        assert_eq!(
            body,
            Some(json!({"source": "docs/aomine.md", "paragraph": 1}))
        );
    }

    /// The three explain mirrors route beside their parents: per-context
    /// POSTs, the addressing key peeled off, every override passed
    /// through — including `expected`, which no parent tool carries.
    #[test]
    fn explain_tools_route_beside_their_parents() {
        let (method, path, body) = route_tool(
            "explain_search",
            &json!({"context": "sake", "query": "酒造", "source": "docs/kura.md",
                    "paragraph": 1, "limit": 5, "semantic_floor": 0.2}),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/sources/search/explain");
        assert_eq!(
            body,
            Some(
                json!({"query": "酒造", "source": "docs/kura.md", "paragraph": 1, "limit": 5,
                        "semantic_floor": 0.2})
            )
        );

        let (method, path, body) = route_tool(
            "explain_resolve",
            &json!({"context": "sake", "cue": "青嶺", "expected": "青嶺酒造", "dice_floor": 0.2}),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/resolve/explain");
        assert_eq!(
            body,
            Some(json!({"cue": "青嶺", "expected": "青嶺酒造", "dice_floor": 0.2}))
        );

        let (method, path, body) = route_tool(
            "explain_resolve_label",
            &json!({"context": "sake", "cue": "醸す", "expected": "杜氏"}),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/resolve_label/explain");
        assert_eq!(body, Some(json!({"cue": "醸す", "expected": "杜氏"})));
    }

    /// MCP clients written against the pre-#35 argument name still work:
    /// `pick` alone would silently drop `index` since it only whitelists
    /// `paragraph`, so this exercises the alias fallback end to end.
    #[test]
    fn cite_passage_accepts_the_pre_35_index_argument_name() {
        let (_, _, body) = route_tool(
            "cite_passage",
            &json!({"context": "sake", "source": "docs/aomine.md", "index": 1}),
        )
        .unwrap();
        assert_eq!(
            body,
            Some(json!({"source": "docs/aomine.md", "paragraph": 1}))
        );
    }

    /// The advertised contract matches what `route_tool` actually accepts:
    /// `index` is a documented deprecated alias, and the schema requires
    /// one of `paragraph`/`index` rather than unconditionally demanding
    /// `paragraph`.
    #[test]
    fn cite_passage_schema_advertises_index_as_a_deprecated_alias() {
        let tool = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "cite_passage")
            .expect("cite_passage is defined");
        let schema = &tool["inputSchema"];
        assert!(
            schema["properties"]["index"]["type"] == "integer",
            "schema should advertise `index` as an integer: {schema}"
        );
        assert_eq!(schema["required"], json!(["context", "source"]));
        assert_eq!(
            schema["anyOf"],
            json!([{ "required": ["paragraph"] }, { "required": ["index"] }])
        );
    }

    /// `query`'s description says subject/label/object need at least
    /// one; the schema must say so too, on top of (not instead of) the
    /// target-selection `anyOf` `search_target_schema` already adds.
    #[test]
    fn query_schema_requires_a_position_alongside_a_target() {
        let tool = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "query")
            .expect("query is defined");
        let schema = &tool["inputSchema"];
        assert!(
            schema.get("anyOf").is_none(),
            "the target-selection anyOf should move under allOf, not stay alongside it: {schema}"
        );
        assert_eq!(
            schema["allOf"],
            json!([
                {
                    "anyOf": [
                        { "required": ["context"] },
                        { "required": ["contexts"] },
                        { "required": ["groups"] },
                    ]
                },
                {
                    "anyOf": [
                        { "required": ["subject"] },
                        { "required": ["label"] },
                        { "required": ["object"] },
                    ]
                },
            ])
        );
    }

    /// The framing rules both transports rely on: requests carry ids,
    /// notifications don't, and non-JSON-RPC input is neither.
    #[test]
    fn classify_separates_requests_notifications_and_garbage() {
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": 1, "method": "ping"})),
            Message::Request {
                call: Call::Ping,
                ..
            }
        ));
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"})),
            Message::Notification
        ));
        // A null id is a notification too, not a request.
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": null, "method": "ping"})),
            Message::Notification
        ));
        // The id survives even though the method is what's missing — the
        // sender is still waiting on a reply it can correlate, not a null.
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": 1})),
            Message::Undecodable { id } if id == json!(1)
        ));
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})),
            Message::Request {
                call: Call::Unknown { .. },
                ..
            }
        ));
    }

    /// JSON-RPC 2.0 allows only a string, a number, or null for id
    /// (§4); an object/array/bool id must be refused up front rather
    /// than echoed into a response that would itself become malformed
    /// — regardless of whether the rest of the message (here, a valid
    /// `method`) would otherwise decode cleanly.
    #[test]
    fn classify_refuses_an_id_that_is_not_a_string_number_or_null() {
        for bad_id in [json!({"a": 1}), json!([1, 2]), json!(true)] {
            assert!(
                matches!(
                    classify(&json!({"jsonrpc": "2.0", "id": bad_id, "method": "ping"})),
                    Message::InvalidId
                ),
                "{bad_id}"
            );
        }
        // A missing method doesn't change the diagnosis: the id's own
        // type is still what's wrong, so this must not be reported as
        // Undecodable (whose "no method" text would misdescribe it).
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": [1]})),
            Message::InvalidId
        ));
    }

    /// The one notification method a transport needs to look inside —
    /// everything else stays opaque behind `Message::Notification`.
    #[test]
    fn cancelled_request_id_reads_only_its_own_notification() {
        assert_eq!(
            cancelled_request_id(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": { "requestId": 7 },
            })),
            Some(json!(7))
        );
        assert_eq!(
            cancelled_request_id(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": { "requestId": "abc" },
            })),
            Some(json!("abc"))
        );
        // A different notification, a request, and a malformed
        // cancellation (no params, no requestId) all read as None.
        assert_eq!(
            cancelled_request_id(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"})),
            None
        );
        assert_eq!(
            cancelled_request_id(&json!({"jsonrpc": "2.0", "id": 1, "method": "ping"})),
            None
        );
        assert_eq!(
            cancelled_request_id(&json!({"jsonrpc": "2.0", "method": "notifications/cancelled"})),
            None
        );
    }

    #[test]
    fn initialize_result_echoes_the_client_version_or_falls_back() {
        let echoed = initialize_result(Some("2025-06-18"), "manual");
        assert_eq!(echoed["protocolVersion"], "2025-06-18");
        assert_eq!(echoed["instructions"], "manual");
        assert_eq!(echoed["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));

        let fallback = initialize_result(None, "manual");
        assert_eq!(fallback["protocolVersion"], FALLBACK_PROTOCOL_VERSION);
    }

    /// A version this build was never written against — a future spec
    /// revision, or a client just making one up — falls back instead
    /// of being echoed back as if the two sides had agreed to it.
    #[test]
    fn initialize_result_falls_back_on_an_unrecognized_client_version() {
        let unrecognized = initialize_result(Some("2099-01-01"), "manual");
        assert_eq!(unrecognized["protocolVersion"], FALLBACK_PROTOCOL_VERSION);

        let garbage = initialize_result(Some("not-a-version"), "manual");
        assert_eq!(garbage["protocolVersion"], FALLBACK_PROTOCOL_VERSION);
    }

    /// The spec requires the fallback to be the NEWEST version the
    /// server supports, not merely one of them — a server that fell
    /// back to its oldest-understood version would pass every other
    /// assertion in this file while still violating the spec.
    #[test]
    fn fallback_protocol_version_is_the_newest_supported_one() {
        assert_eq!(
            Some(&FALLBACK_PROTOCOL_VERSION),
            SUPPORTED_PROTOCOL_VERSIONS.last()
        );
    }

    #[test]
    fn tool_response_marks_errors_without_aborting_the_rpc() {
        let ok = tool_response(Ok("fine".into()));
        assert_eq!(ok["content"][0]["text"], "fine");
        assert!(ok.get("isError").is_none());

        let err = tool_response(Err("HTTP 404: gone".into()));
        assert_eq!(err["isError"], true);
        assert_eq!(err["content"][0]["text"], "HTTP 404: gone");
    }

    #[test]
    fn import_schema_advertises_stream_and_dry_run() {
        let import = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "import")
            .expect("import is defined");
        let properties = &import["inputSchema"]["properties"];
        assert_eq!(properties["stream"]["type"], "string");
        assert_eq!(properties["dry_run"]["type"], "boolean");
        assert_eq!(import["inputSchema"]["required"], json!(["stream"]));
    }

    /// The stream rides the body as a raw string, not the `pick`/JSON
    /// shape every other write tool uses — `call_inner`/`Bridge::call`
    /// special-case `Value::String` so this reaches `import_batch` as
    /// literal NDJSON text, newlines intact, not `\n`-escaped inside a
    /// quoted JSON string.
    #[test]
    fn import_routes_the_stream_as_a_raw_string_body() {
        let stream = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s\"}\n";
        let (method, path, body) = route_tool("import", &json!({"stream": stream})).unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/import");
        assert_eq!(body, Some(Value::String(stream.to_string())));
    }

    #[test]
    fn import_routes_dry_run_onto_the_query_string() {
        let (_, path, _) = route_tool("import", &json!({"stream": "x", "dry_run": true})).unwrap();
        assert_eq!(path, "/import?dry_run=true");

        let (_, path, _) = route_tool("import", &json!({"stream": "x"})).unwrap();
        assert_eq!(
            path, "/import",
            "dry_run absent means no query string at all"
        );
    }

    #[test]
    fn import_refuses_a_stream_over_the_byte_limit() {
        let oversized = "x".repeat(MAX_IMPORT_STREAM_BYTES + 1);
        let routed = route_tool("import", &json!({"stream": oversized}));
        assert!(
            routed.is_err(),
            "a stream past the tool's own byte cap must not route"
        );
    }

    /// A minimal HTTP-response envelope for the given `route_tool` path
    /// suffix, mirroring `ApiResponse<T>`'s `{result, status, time}`
    /// shape — `run_retrieve` decodes exactly that.
    fn envelope(result: Value) -> String {
        json!({ "result": result, "status": "ok", "time": 0.0 }).to_string()
    }

    #[test]
    fn run_retrieve_resolves_describes_activates_and_cites_in_one_call() {
        let arguments = json!({ "context": "sake", "origins": ["tokyo"] });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(
                    json!([{"name": "Tokyo", "score": 1.0, "tier": "exact"}]),
                ))
            } else if path.ends_with("/describe") {
                Ok(envelope(
                    json!({"concept": "Tokyo", "as_subject": [], "as_object": []}),
                ))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{
                        "strength": 1.0,
                        "path": ["Tokyo"],
                        "association": {
                            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
                            "weight": 1.0, "count": 1,
                            "attributions": [
                                {"source": "doc1", "weight": 1.0, "count": 1, "paragraph": 0}
                            ]
                        }
                    }]
                })))
            } else if path.ends_with("/citations") {
                Ok(envelope(
                    json!({"text": "Tokyo is the capital.", "source": "doc1", "section": null}),
                ))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(result["resolved"]["tokyo"][0]["name"], "Tokyo");
        assert_eq!(result["outline"]["Tokyo"]["concept"], "Tokyo");
        assert_eq!(result["associations"].as_array().unwrap().len(), 1);
        assert_eq!(result["associations"][0]["subject"], "Tokyo");
        assert_eq!(result["activations"].as_array().unwrap().len(), 1);
        assert_eq!(
            result["citations"],
            json!([{
                "source": "doc1",
                "paragraph": 0,
                "citation": {"text": "Tokyo is the capital.", "source": "doc1", "section": null},
            }])
        );
        assert_eq!(result["passage_hits"], json!([]));
    }

    /// A budget that the first citation round trip alone pushes past
    /// must stop the loop right there — the second and third citations
    /// (same association, three attributions) must never dispatch, not
    /// merely have their result discarded once the whole call finishes.
    #[test]
    fn run_retrieve_bounded_stops_dispatching_citations_once_the_budget_is_spent() {
        let resolve_body = envelope(json!([{"name": "Tokyo"}]));
        let association = json!({
            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
            "weight": 1.0, "count": 1,
            "attributions": [
                {"source": "doc1", "weight": 1.0, "count": 1, "paragraph": 0},
                {"source": "doc2", "weight": 1.0, "count": 1, "paragraph": 0},
                {"source": "doc3", "weight": 1.0, "count": 1, "paragraph": 0},
            ]
        });
        let activate_body = envelope(json!({
            "total": 1,
            "matches": [{"strength": 1.0, "path": ["Tokyo"], "association": association}]
        }));
        let citation_body = envelope(json!({"text": "x", "source": "doc", "section": null}));
        // One byte short of resolve + activate + one citation: the first
        // citation's response is what tips the scale, not a fourth call.
        let budget = resolve_body.len() + activate_body.len() + citation_body.len() - 1;

        let arguments = json!({ "context": "sake", "origins": ["tokyo"], "describe_first": false });
        let mut citation_calls = 0usize;
        let result = run_retrieve_bounded(&arguments, Some(budget), |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(resolve_body.clone())
            } else if path.ends_with("/activate") {
                Ok(activate_body.clone())
            } else if path.ends_with("/citations") {
                citation_calls += 1;
                Ok(citation_body.clone())
            } else {
                panic!("unexpected call: {path}");
            }
        });

        assert!(
            matches!(&result, Err(message) if message.contains(&format!("already exceeds {budget} bytes"))
                && message.contains("cite_passage")),
            "{result:?}"
        );
        assert_eq!(
            citation_calls, 1,
            "the budget must be spent inside the first citation call, before a second ever fires"
        );
    }

    /// `run_retrieve` (what the uncapped stdio bridge calls) is just
    /// `run_retrieve_bounded` with no budget — a budget so tight even
    /// the first call would trip it must still let a `None` budget
    /// through untouched.
    #[test]
    fn run_retrieve_passes_no_budget_to_run_retrieve_bounded() {
        let arguments = json!({ "context": "sake", "origins": ["tokyo"], "describe_first": false });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({"total": 0, "matches": []})))
            } else {
                panic!("unexpected call: {path}");
            }
        });
        assert!(result.is_ok(), "{result:?}");
    }

    /// query (only run when `labels` is given) and activate can surface
    /// the same edge; the triple-keyed dedupe must collapse them to one
    /// entry, keeping query's copy since it is gathered first.
    #[test]
    fn run_retrieve_dedupes_associations_across_query_and_activate() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"], "labels": ["capital_of"],
            "describe_first": false, "fetch_citations": false
        });
        let association = json!({
            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
            "weight": 1.0, "count": 1, "attributions": []
        });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/query") {
                Ok(envelope(json!({"total": 1, "matches": [association]})))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{"strength": 1.0, "path": ["Tokyo"], "association": association}]
                })))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(
            result["associations"].as_array().unwrap().len(),
            1,
            "{result}"
        );
        assert_eq!(result["activations"].as_array().unwrap().len(), 1);
    }

    /// `triple_of`'s doc comment says a value it can't parse into
    /// `(subject, label, object)` means "keep it, nothing to dedupe
    /// against" — not "drop it". A malformed `query` match and a
    /// malformed `activate` association (both missing `label`) must
    /// both still land in the final `associations` list.
    #[test]
    fn run_retrieve_keeps_an_association_triple_of_cannot_parse() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"], "labels": ["capital_of"],
            "describe_first": false, "fetch_citations": false
        });
        let malformed_from_query = json!({
            "subject": "Tokyo", "object": "Japan", "weight": 1.0, "count": 1, "attributions": []
        });
        let malformed_from_activate = json!({
            "subject": "Osaka", "object": "Japan", "weight": 1.0, "count": 1, "attributions": []
        });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/query") {
                Ok(envelope(
                    json!({"total": 1, "matches": [malformed_from_query]}),
                ))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{
                        "strength": 1.0, "path": ["Tokyo"], "association": malformed_from_activate
                    }]
                })))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(
            result["associations"].as_array().unwrap().len(),
            2,
            "an association triple_of cannot parse must be kept, not dropped: {result}"
        );
    }

    /// A citation attribution pointing at a passage that was never
    /// stored (or was retracted) comes back 404 from `cite_passage` —
    /// that one locator is skipped, not the whole retrieval.
    #[test]
    fn run_retrieve_skips_a_404_citation_without_failing() {
        let arguments = json!({ "context": "sake", "origins": ["tokyo"], "describe_first": false });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{
                        "strength": 1.0,
                        "path": ["Tokyo"],
                        "association": {
                            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
                            "weight": 1.0, "count": 1,
                            "attributions": [
                                {"source": "doc1", "weight": 1.0, "count": 1, "paragraph": 0}
                            ]
                        }
                    }]
                })))
            } else if path.ends_with("/citations") {
                Err("HTTP 404: {\"error\":\"no such paragraph\"}".to_string())
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("a 404 citation must not fail the whole retrieval");

        assert_eq!(result["citations"], json!([]));
    }

    /// Any citation failure other than 404 (auth, a downed server) must
    /// abort the whole call rather than being swallowed like the 404
    /// case above.
    #[test]
    fn run_retrieve_fails_outright_on_a_non_404_citation_error() {
        let arguments = json!({ "context": "sake", "origins": ["tokyo"], "describe_first": false });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{
                        "strength": 1.0,
                        "path": ["Tokyo"],
                        "association": {
                            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
                            "weight": 1.0, "count": 1,
                            "attributions": [
                                {"source": "doc1", "weight": 1.0, "count": 1, "paragraph": 0}
                            ]
                        }
                    }]
                })))
            } else if path.ends_with("/citations") {
                Err("HTTP 500: internal error".to_string())
            } else {
                panic!("unexpected call: {path}");
            }
        });
        assert_eq!(result, Err("HTTP 500: internal error".to_string()));
    }

    /// A wrong-typed boolean knob — `"auto_pick": "false"` (a JSON
    /// string, not a bool) — must be rejected outright rather than
    /// silently falling back to its `true` default: for a knob whose
    /// default is `true`, silently keeping it would run the whole
    /// retrieval against the opposite of what the caller plainly
    /// intended, and no tool call should fire before that check.
    #[test]
    fn run_retrieve_rejects_wrong_typed_boolean_arguments() {
        for key in [
            "auto_pick",
            "describe_first",
            "fetch_citations",
            "text_fallback_only_if_empty",
        ] {
            let mut arguments = json!({ "context": "sake", "origins": ["tokyo"] });
            arguments[key] = json!("false");
            let result = run_retrieve(&arguments, |_method, path, _body| {
                panic!("unexpected call: {path}");
            });
            assert!(
                matches!(&result, Err(message) if message.contains(key) && message.contains("must be a boolean")),
                "{key}: {result:?}"
            );
        }
    }

    /// `auto_pick: false` anchors on each cue verbatim instead of
    /// resolve's top candidate — resolve still runs (so `resolved`
    /// still reports what was found), but an empty result must not
    /// empty out the anchor list too.
    #[test]
    fn run_retrieve_with_auto_pick_off_anchors_on_the_cue_itself() {
        let arguments = json!({
            "context": "sake", "origins": ["Tokyo"], "auto_pick": false,
            "describe_first": false, "fetch_citations": false
        });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({"total": 0, "matches": []})))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(result["resolved"]["Tokyo"], json!([]));
        assert_eq!(result["associations"], json!([]));
    }

    /// The text-lane fallback only fires when a fallback query was
    /// given AND (by default) associations came back empty — here
    /// resolve finds nothing, so anchors stay empty, activate never
    /// runs, and search_passages is the only other call.
    #[test]
    fn run_retrieve_runs_the_text_fallback_when_associations_are_empty() {
        let arguments = json!({
            "context": "sake", "origins": ["nonexistent"],
            "text_fallback_query": "some declarative fact"
        });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([])))
            } else if path.ends_with("/sources/search") {
                Ok(envelope(json!({
                    "plan": {"contexts": [{"context": "sake", "lanes": {
                        "bm25": {"ran": true},
                        "vector": {"ran": false, "reason": "no embedding provider is configured"}
                    }}]},
                    "hits": [
                        {"source": "doc1", "paragraph": 0, "score": 0.9, "text": "...", "lanes": {}}
                    ]
                })))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(result["passage_hits"].as_array().unwrap().len(), 1);
        assert_eq!(
            result["search_plan"]["contexts"][0]["context"], "sake",
            "the fallback search's plan rides beside its hits"
        );
    }

    /// A pre-#151 server's bare-array search result (reachable through
    /// the stdio bridge under version skew) must fail the retrieve
    /// loudly — an empty `passage_hits` for a search that found things
    /// would be a silent wrong answer.
    #[test]
    fn run_retrieve_refuses_a_pre_plan_search_shape() {
        let arguments = json!({
            "context": "sake", "origins": ["nonexistent"],
            "text_fallback_query": "some declarative fact"
        });
        let outcome = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([])))
            } else if path.ends_with("/sources/search") {
                Ok(envelope(json!([
                    {"source": "doc1", "paragraph": 0, "score": 0.9, "text": "...", "lanes": {}}
                ])))
            } else {
                panic!("unexpected call: {path}");
            }
        });
        let error = outcome.expect_err("the legacy shape must refuse");
        assert!(error.contains("without a 'hits' array"), "{error}");
    }

    /// `text_fallback_only_if_empty: false` runs the fallback
    /// unconditionally, even alongside associations already found.
    #[test]
    fn run_retrieve_runs_the_text_fallback_unconditionally_when_the_empty_gate_is_off() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"], "describe_first": false,
            "fetch_citations": false, "text_fallback_query": "some declarative fact",
            "text_fallback_only_if_empty": false
        });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{
                        "strength": 1.0,
                        "path": ["Tokyo"],
                        "association": {
                            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
                            "weight": 1.0, "count": 1, "attributions": []
                        }
                    }]
                })))
            } else if path.ends_with("/sources/search") {
                Ok(envelope(json!({
                    "plan": {"contexts": []},
                    "hits": [
                        {"source": "doc1", "paragraph": 0, "score": 0.9, "text": "...", "lanes": {}}
                    ]
                })))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(result["associations"].as_array().unwrap().len(), 1);
        assert_eq!(result["passage_hits"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn run_retrieve_requires_context_and_origins() {
        assert_eq!(
            run_retrieve(&json!({"origins": ["x"]}), |_, _, _| unreachable!()),
            Err("missing required argument 'context'".to_string())
        );
        assert_eq!(
            run_retrieve(&json!({"context": "sake"}), |_, _, _| unreachable!()),
            Err("missing required argument 'origins'".to_string())
        );
    }

    #[test]
    fn run_retrieve_refuses_an_origins_list_past_the_input_cap() {
        // One resolve round trip per cue makes an oversized list a fanout
        // amplifier, so it is refused before the first call fires — the
        // way the direct endpoints refuse an overlong `origins` batch.
        let origins: Vec<String> = (0..=MAX_ORIGIN_CUES).map(|i| format!("cue{i}")).collect();
        let result = run_retrieve(
            &json!({ "context": "sake", "origins": origins }),
            |_, _, _| unreachable!("no request may fire once the list is refused"),
        );
        assert!(
            matches!(&result, Err(message) if message.contains("past the per-request limit")),
            "{result:?}"
        );
    }

    #[test]
    fn run_retrieve_admits_an_origins_list_at_exactly_the_input_cap() {
        // The cap refuses lists *past* the ceiling, not at it: a list of
        // exactly MAX_ORIGIN_CUES cues clears the guard and reaches the first
        // resolve round trip. Pins the `>` boundary so a `>=` slip — which
        // would refuse the largest admissible list — cannot pass unnoticed.
        let origins: Vec<String> = (0..MAX_ORIGIN_CUES).map(|i| format!("cue{i}")).collect();
        assert_eq!(origins.len(), MAX_ORIGIN_CUES);
        let mut calls = 0usize;
        let result = run_retrieve(
            &json!({ "context": "sake", "origins": origins }),
            |_, path, _| {
                calls += 1;
                assert!(
                    path.ends_with("/resolve"),
                    "first round trip is a resolve: {path}"
                );
                Err("stop past the guard".to_string())
            },
        );
        assert_eq!(
            calls, 1,
            "the admitted list fired exactly one resolve before we bailed"
        );
        assert_eq!(result, Err("stop past the guard".to_string()));
    }

    /// `resolve_limit` rides every resolve round trip's body: a caller's
    /// candidate cap must reach the resolve endpoint, not be dropped
    /// between the composed call and the per-cue request. The non-null
    /// gate that admits it is what makes a supplied cap take effect;
    /// resolve returns nothing, so no anchor forms and it is the only call.
    #[test]
    fn run_retrieve_forwards_resolve_limit_to_each_resolve_call() {
        let arguments = json!({ "context": "sake", "origins": ["tokyo"], "resolve_limit": 7 });
        let mut saw_resolve = false;
        run_retrieve(&arguments, |_method, path, body| {
            assert!(
                path.ends_with("/resolve"),
                "resolve is the only call: {path}"
            );
            let body = body.expect("resolve carries a body");
            assert_eq!(
                body["limit"], 7,
                "resolve_limit must ride the resolve body: {body}"
            );
            saw_resolve = true;
            Ok(envelope(json!([])))
        })
        .expect("run_retrieve succeeds");
        assert!(saw_resolve, "resolve must have fired");
    }

    /// `labels` both gates and rides the query round trip: naming facets
    /// must fire a `query` whose body carries them. Were the non-null gate
    /// to invert, a named facet set would silently skip query altogether.
    #[test]
    fn run_retrieve_forwards_labels_to_the_query_round_trip() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"], "labels": ["capital_of"],
            "describe_first": false, "fetch_citations": false
        });
        let mut saw_query = false;
        run_retrieve(&arguments, |_method, path, body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{ "name": "Tokyo" }])))
            } else if path.ends_with("/query") {
                let body = body.expect("query carries a body");
                assert_eq!(
                    body["label"],
                    json!(["capital_of"]),
                    "labels must ride the query body: {body}"
                );
                saw_query = true;
                Ok(envelope(json!({ "matches": [] })))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({ "matches": [] })))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");
        assert!(saw_query, "labels must trigger a query round trip");
    }

    /// `activate_decay` and `activate_limit` both ride the activate body:
    /// the spreading-activation knobs must reach the activate endpoint
    /// rather than being dropped between the composed call and the request.
    #[test]
    fn run_retrieve_forwards_activate_decay_and_limit_to_the_activate_call() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"],
            "activate_decay": 0.5, "activate_limit": 9,
            "describe_first": false, "fetch_citations": false
        });
        let mut saw_activate = false;
        run_retrieve(&arguments, |_method, path, body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{ "name": "Tokyo" }])))
            } else if path.ends_with("/activate") {
                let body = body.expect("activate carries a body");
                assert_eq!(
                    body["decay"], 0.5,
                    "activate_decay must ride the activate body: {body}"
                );
                assert_eq!(
                    body["limit"], 9,
                    "activate_limit must ride the activate body: {body}"
                );
                saw_activate = true;
                Ok(envelope(json!({ "matches": [] })))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");
        assert!(saw_activate, "activate must have fired");
    }

    /// `search_limit` rides the text-fallback search body, capping the
    /// fallback page as the caller asked. resolve anchors but activate
    /// returns nothing, so associations stay empty and the fallback fires.
    #[test]
    fn run_retrieve_forwards_search_limit_to_the_text_fallback() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"], "describe_first": false,
            "fetch_citations": false, "text_fallback_query": "some declarative fact",
            "search_limit": 4
        });
        let mut saw_search = false;
        run_retrieve(&arguments, |_method, path, body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{ "name": "Tokyo" }])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({ "matches": [] })))
            } else if path.ends_with("/sources/search") {
                let body = body.expect("search carries a body");
                assert_eq!(
                    body["limit"], 4,
                    "search_limit must ride the search body: {body}"
                );
                saw_search = true;
                Ok(envelope(json!({"plan": {"contexts": []}, "hits": []})))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");
        assert!(saw_search, "the text fallback must have fired a search");
    }

    #[test]
    fn retrieve_is_advertised_with_context_and_origins_required() {
        let tool = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "retrieve")
            .expect("retrieve is defined");
        assert_eq!(
            tool["inputSchema"]["required"],
            json!(["context", "origins"])
        );
    }
}
