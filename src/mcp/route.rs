use serde_json::{Value, json};

use super::args::{need, need_present, pick, pick_with_alias, query_string, segment};

/// Hard ceiling on the `import` tool's `stream` argument, checked
/// before the stream ever leaves the process — `taguru-mcp` does not
/// link `ingest.rs` (its only path to the server is HTTP), so
/// `ingest::MAX_LINE_BYTES` is unreachable here and this stands in for
/// it. It is an upper bound, NOT the effective cap: the server's own
/// request body cap (`TAGURU_MAX_BODY_BYTES`, 8 MiB default) binds
/// first at that default. The bridge POSTs the raw stream to `/import`
/// under that cap, and over the `/mcp` HTTP transport the stream is
/// JSON-quoted into the *outer* envelope (every newline escaped to
/// `\n`, close to double its raw size) which must itself fit the body
/// cap — so a stream between the body cap and this ceiling passes here
/// only to be 413'd by the server. This 32 MiB becomes the binding
/// limit solely once an operator raises `TAGURU_MAX_BODY_BYTES` above
/// it. That same doubling is why `taguru-mcp`'s per-line frame cap
/// (`TAGURU_MCP_MAX_LINE_BYTES`) defaults to ~2× this value: a line
/// under the frame cap must still be able to carry a full-size stream.
pub(super) const MAX_IMPORT_STREAM_BYTES: usize = 32 * 1024 * 1024;

/// Maps one tool call onto (method, path, body) — pure, so the mapping
/// from advertised tools to HTTP requests is testable without a server.
pub fn route_tool(
    name: &str,
    arguments: &Value,
) -> Result<(&'static str, String, Option<Value>), String> {
    let context_path = |key: &str| -> Result<String, String> {
        Ok(format!("/contexts/{}", segment(need(arguments, key)?)))
    };
    let group_path = |key: &str| -> Result<String, String> {
        Ok(format!("/groups/{}", segment(need(arguments, key)?)))
    };
    // The search tools target one context or several: `context`
    // prefixes the per-context path; `contexts` and/or `groups`
    // (arrays, riding the body) mean the cross-context route — no
    // prefix. `context` beside either is ambiguous, and none at all
    // names no target.
    let search_base = || -> Result<String, String> {
        let given = |key: &str| arguments.get(key).is_some_and(|value| !value.is_null());
        match (given("context"), given("contexts") || given("groups")) {
            (true, true) => {
                Err("pass either 'context' or 'contexts'/'groups', not both".to_string())
            }
            (false, false) => Err(
                "missing required argument 'context' (or 'contexts'/'groups', to search several at once)"
                    .to_string(),
            ),
            (true, false) => context_path("context"),
            (false, true) => Ok(String::new()),
        }
    };
    Ok(match name {
        "get_protocol" => ("GET", "/protocol".to_string(), None),
        "flush" => ("POST", "/flush".to_string(), None),
        "export_context" => ("GET", format!("{}/export", context_path("context")?), None),
        "export_group" => ("GET", format!("{}/export", group_path("name")?), None),
        "get_context" => ("GET", context_path("context")?, None),
        "get_group" => ("GET", group_path("name")?, None),
        "compact" => (
            "POST",
            format!("{}/compact", context_path("context")?),
            None,
        ),
        "import" => {
            let stream = need(arguments, "stream")?;
            if stream.len() > MAX_IMPORT_STREAM_BYTES {
                return Err(format!(
                    "stream argument is {} bytes, over the {MAX_IMPORT_STREAM_BYTES}-byte \
                     tool limit; split the import or POST the stream to /import directly",
                    stream.len()
                ));
            }
            (
                "POST",
                format!("/import{}", query_string(arguments, &["dry_run"])?),
                Some(Value::String(stream.to_string())),
            )
        }
        "list_contexts" => (
            "GET",
            format!(
                "/contexts{}",
                query_string(arguments, &["limit", "after", "pinned"])?
            ),
            None,
        ),
        "create_context" => (
            "PUT",
            context_path("name")?,
            Some(pick(
                arguments,
                &["description", "pinned", "dice_floor", "semantic_floor"],
            )),
        ),
        "update_context" => (
            "PATCH",
            context_path("name")?,
            Some(pick(
                arguments,
                &["description", "pinned", "dice_floor", "semantic_floor"],
            )),
        ),
        "delete_context" => ("DELETE", context_path("name")?, None),
        "rename_context" => {
            let path = format!("{}/rename", context_path("name")?);
            need(arguments, "to")?;
            ("POST", path, Some(pick(arguments, &["to"])))
        }
        "list_groups" => (
            "GET",
            format!("/groups{}", query_string(arguments, &["limit", "after"])?),
            None,
        ),
        "create_group" => (
            "PUT",
            group_path("name")?,
            Some(pick(arguments, &["description", "contexts", "groups"])),
        ),
        "update_group" => (
            "PATCH",
            group_path("name")?,
            Some(pick(
                arguments,
                &[
                    "description",
                    "add_contexts",
                    "remove_contexts",
                    "add_groups",
                    "remove_groups",
                ],
            )),
        ),
        "delete_group" => ("DELETE", group_path("name")?, None),
        "rename_group" => {
            let path = format!("{}/rename", group_path("name")?);
            need(arguments, "to")?;
            ("POST", path, Some(pick(arguments, &["to"])))
        }
        "add_associations" => {
            // Resolve `context` first so a caller who omitted BOTH hears
            // about the primary argument, not the secondary one, in the
            // order the schema lists them.
            let path = format!("{}/associations", context_path("context")?);
            // Schema-required: an omitted (or null) argument must
            // refuse, not fall back to an empty batch — that would
            // route a caller's mistake into a silent, do-nothing 200.
            let associations = arguments
                .get("associations")
                .filter(|value| !value.is_null())
                .cloned()
                .ok_or_else(|| "missing required argument 'associations'".to_string())?;
            ("POST", path, Some(associations))
        }
        "store_passages" => {
            let path = format!("{}/sources", context_path("context")?);
            need_present(arguments, "passages")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["passages", "questions", "sections"])),
            )
        }
        "lookup_passages" => {
            let path = format!("{}/sources/lookup", context_path("context")?);
            need_present(arguments, "sources")?;
            ("POST", path, Some(pick(arguments, &["sources"])))
        }
        "list_sources" => (
            "GET",
            format!(
                "{}/sources{}",
                context_path("context")?,
                query_string(arguments, &["limit", "after", "prefix"])?
            ),
            None,
        ),
        "resolve" => {
            let path = format!("{}/resolve", context_path("context")?);
            need(arguments, "cue")?;
            (
                "POST",
                path,
                Some(pick(
                    arguments,
                    &["cue", "dice_floor", "semantic_floor", "limit"],
                )),
            )
        }
        "resolve_label" => {
            let path = format!("{}/resolve_label", context_path("context")?);
            need(arguments, "cue")?;
            (
                "POST",
                path,
                Some(pick(
                    arguments,
                    &["cue", "dice_floor", "semantic_floor", "limit"],
                )),
            )
        }
        "explain_resolve" => {
            let path = format!("{}/resolve/explain", context_path("context")?);
            need(arguments, "cue")?;
            need(arguments, "expected")?;
            (
                "POST",
                path,
                Some(pick(
                    arguments,
                    &["cue", "expected", "dice_floor", "semantic_floor", "limit"],
                )),
            )
        }
        "explain_resolve_label" => {
            let path = format!("{}/resolve_label/explain", context_path("context")?);
            need(arguments, "cue")?;
            need(arguments, "expected")?;
            (
                "POST",
                path,
                Some(pick(
                    arguments,
                    &["cue", "expected", "dice_floor", "semantic_floor", "limit"],
                )),
            )
        }
        "describe" => {
            let path = format!("{}/describe", context_path("context")?);
            need(arguments, "concept")?;
            ("POST", path, Some(pick(arguments, &["concept"])))
        }
        "query" => (
            "POST",
            format!("{}/query", search_base()?),
            Some(pick(
                arguments,
                &[
                    "contexts", "groups", "subject", "label", "object", "limit", "after",
                ],
            )),
        ),
        "recall" => {
            let path = format!("{}/recall", search_base()?);
            need(arguments, "cue")?;
            (
                "POST",
                path,
                Some(pick(
                    arguments,
                    &["contexts", "groups", "cue", "limit", "after"],
                )),
            )
        }
        "activate" => {
            let path = format!("{}/activate", context_path("context")?);
            need_present(arguments, "origins")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["origins", "decay", "limit"])),
            )
        }
        "explore" => {
            let path = format!("{}/explore", context_path("context")?);
            need_present(arguments, "origins")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["origins", "max_depth", "limit", "after"])),
            )
        }
        "list_labels" => (
            "GET",
            format!(
                "{}/labels{}",
                context_path("context")?,
                query_string(arguments, &["limit", "after", "prefix"])?
            ),
            None,
        ),
        "get_aliases" => (
            "GET",
            format!(
                "{}/aliases{}",
                context_path("context")?,
                query_string(arguments, &["limit", "after", "prefix"])?
            ),
            None,
        ),
        "add_aliases" => (
            "POST",
            format!("{}/aliases", context_path("context")?),
            Some(pick(arguments, &["concepts", "labels"])),
        ),
        "remove_aliases" => (
            "DELETE",
            format!("{}/aliases", context_path("context")?),
            Some(pick(arguments, &["concepts", "labels"])),
        ),
        "retract_source" => {
            let path = format!("{}/sources/retract", context_path("context")?);
            need(arguments, "source")?;
            ("POST", path, Some(pick(arguments, &["source"])))
        }
        "retract_association" => {
            let path = format!("{}/associations/retract", context_path("context")?);
            need(arguments, "subject")?;
            need(arguments, "label")?;
            need(arguments, "object")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["subject", "label", "object"])),
            )
        }
        "search_passages" => {
            let path = format!("{}/sources/search", search_base()?);
            need(arguments, "query")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["contexts", "groups", "query", "limit"])),
            )
        }
        "explain_search" => {
            let path = format!("{}/sources/search/explain", context_path("context")?);
            need(arguments, "query")?;
            need(arguments, "source")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["query", "source", "paragraph", "limit"])),
            )
        }
        "cite_passage" => {
            let path = format!("{}/citations", context_path("context")?);
            need(arguments, "source")?;
            let has_paragraph = arguments
                .get("paragraph")
                .is_some_and(|value| !value.is_null());
            let has_index = arguments.get("index").is_some_and(|value| !value.is_null());
            if !has_paragraph && !has_index {
                return Err(
                    "missing required argument 'paragraph' (or its deprecated alias 'index')"
                        .to_string(),
                );
            }
            (
                "POST",
                path,
                Some(pick_with_alias(
                    arguments,
                    &["source", "paragraph"],
                    "paragraph",
                    "index",
                )),
            )
        }
        "refresh_embeddings" => (
            "POST",
            format!("{}/embeddings/refresh", context_path("context")?),
            Some(json!({})),
        ),
        "audit_vocabulary" => (
            "POST",
            format!("{}/vocabulary/audit", context_path("context")?),
            Some(pick(arguments, &["dice_floor", "cosine_floor"])),
        ),
        "audit_coverage" => {
            let path = format!("{}/unreachable_from", context_path("context")?);
            need_present(arguments, "origins")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["origins", "limit", "after"])),
            )
        }
        "audit_drift" => (
            "POST",
            format!("{}/drift/audit", context_path("context")?),
            Some(pick(
                arguments,
                &[
                    "unsourced_floor",
                    "limit",
                    "after",
                    "include_twins",
                    "dice_floor",
                    "cosine_floor",
                ],
            )),
        ),
        _ => return Err(format!("unknown tool '{name}'")),
    })
}
