//! Shared API error shape and write-boundary validation across endpoints.

use serde_json::{Value, json};

use crate::support::*;

#[test]
fn an_association_batch_over_the_cap_is_rejected_before_any_write() {
    let server = Server::start("batchcap");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    let batch: Vec<Value> = (0..10_001)
        .map(|i| json!({"subject": format!("s{i}"), "label": "l", "object": "o", "weight": 1.0}))
        .collect();
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(Value::Array(batch)),
    );
    assert_eq!(status, 400, "{body}");

    // The guard ran before the write lock: nothing was applied.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));
}

#[test]
fn an_insane_weight_is_rejected_before_any_write() {
    let server = Server::start("weightcap");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    // Finite but absurd: two of these would saturate an edge to
    // +Infinity, and a later retract would mint Inf − Inf = NaN — a
    // fact nothing can read or reset again.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "a", "label": "l", "object": "b", "weight": 1.0},
            {"subject": "青嶺酒造", "label": "生産量", "object": "無限", "weight": 1.0e300},
        ])),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("associations[1].weight"),
        "the message must point at the offending item: {body}"
    );
    // Refused whole, before the write lock: not even the sane first
    // item landed.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));

    // The documented boundary stays usable, negation included.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "a", "label": "l", "object": "b", "weight": 1.0e6},
            {"subject": "a", "label": "l2", "object": "b", "weight": -1.0e6},
        ])),
    );
}

#[test]
fn a_present_body_is_parsed_whatever_the_content_type_says() {
    let server = Server::start("rawbody");

    // requests.put(url, data=json.dumps(...)) territory: a JSON body
    // with no JSON Content-Type. The description must land — this
    // used to silently drop the body and create with every field
    // defaulted, under a 200.
    let (status, body) = server.call_raw(
        "PUT",
        "/contexts/sake",
        Some(r#"{"description":"青嶺酒造の記憶","pinned":true}"#),
        None,
    );
    assert_eq!(status, 200, "{body}");
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(
        directory["contexts"][0]["description"],
        json!("青嶺酒造の記憶")
    );
    assert_eq!(directory["contexts"][0]["pinned"], json!(true));

    // A present body that is not JSON is an error, never defaults.
    let (status, body) =
        server.call_raw("PUT", "/contexts/beer", Some("definitely not json"), None);
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["status"], json!("error"));

    // An absent body still means defaults — the documented shape.
    let (status, body) = server.call_raw("PUT", "/contexts/beer", None, None);
    assert_eq!(status, 200, "{body}");

    // The other optional-body endpoint follows the same contract.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some("also not json"),
        None,
    );
    assert_eq!(status, 400, "{body}");
    let (status, body) = server.call_raw("POST", "/contexts/sake/vocabulary/audit", None, None);
    assert_eq!(status, 200, "{body}");
}

#[test]
fn off_axis_errors_speak_the_api_error_shape_too() {
    let server = Server::start("errshape");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    // Unknown path → 404 in the error shape.
    let (status, body) = server.call("GET", "/contextz", None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("unknown_path"), "{body}");
    assert!(body["time"].is_number(), "{body}");

    // Known path, wrong method → 405 in the error shape.
    let (status, body) = server.call("DELETE", "/contexts/sake/recall", None);
    assert_eq!(status, 405, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("method_not_allowed"), "{body}");

    // Malformed JSON on a JSON-required endpoint → 400 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some("{not json"),
        Some("application/json"),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");

    // Wrong media type → 415 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some("cue=x"),
        Some("application/x-www-form-urlencoded"),
    );
    assert_eq!(status, 415, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");

    // Well-formed JSON of the wrong type → 422 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some(r#"{"cue": 42}"#),
        Some("application/json"),
    );
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");
}

/// Every JSON error carries the machine-readable `code` the protocol
/// documents — the stable branch key for clients that must not parse
/// message wording. One axis per assertion, domain refusals included.
#[test]
fn errors_carry_the_documented_machine_readable_code() {
    let server = Server::start_with_env(
        "errcodes",
        &[
            ("TAGURU_API_TOKEN", "codekey"),
            // Roomy enough for the over_limit body below (an origins
            // list is ~8 KiB), tight enough to trip on the 413 probe.
            ("TAGURU_MAX_BODY_BYTES", "16384"),
        ],
    );
    let code_of = |status: u16, body: &Value| -> (u16, String) {
        (
            status,
            body["code"].as_str().unwrap_or("<missing>").to_string(),
        )
    };

    // Missing bearer token → unauthorized.
    let (status, body) = server.call("GET", "/contexts", None);
    assert_eq!(
        code_of(status, &body),
        (401, "unauthorized".into()),
        "{body}"
    );

    let key = Some("codekey");
    let (status, body) = server.call_with_token("PUT", "/contexts/sake", Some(json!({})), key);
    assert_eq!(status, 200, "{body}");

    // PUT on an existing context → already_exists.
    let (status, body) = server.call_with_token("PUT", "/contexts/sake", Some(json!({})), key);
    assert_eq!(
        code_of(status, &body),
        (409, "already_exists".into()),
        "{body}"
    );

    // Unknown context → no_context.
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/ghost/recall",
        Some(json!({"cue": "x"})),
        key,
    );
    assert_eq!(code_of(status, &body), (404, "no_context".into()), "{body}");

    // A refused value (a weight the graph must never accumulate) →
    // invalid_argument.
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "s", "label": "l", "object": "o", "weight": 1e300}])),
        key,
    );
    assert_eq!(
        code_of(status, &body),
        (400, "invalid_argument".into()),
        "{body}"
    );

    // A list-shaped input past its cap → over_limit (split and resend).
    let origins: Vec<String> = (0..1001).map(|index| format!("o{index}")).collect();
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/activate",
        Some(json!({"origins": origins})),
        key,
    );
    assert_eq!(code_of(status, &body), (400, "over_limit".into()), "{body}");

    // No embedding provider configured → embeddings_unconfigured.
    let (status, body) =
        server.call_with_token("POST", "/contexts/sake/embeddings/refresh", None, key);
    assert_eq!(
        code_of(status, &body),
        (501, "embeddings_unconfigured".into()),
        "{body}"
    );

    // Unknown source on the citation endpoint → no_source; a stored
    // source with an out-of-range locator → no_paragraph.
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"doc": "本文。"}})),
        key,
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "ghost", "paragraph": 0})),
        key,
    );
    assert_eq!(code_of(status, &body), (404, "no_source".into()), "{body}");
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "doc", "paragraph": 9})),
        key,
    );
    assert_eq!(
        code_of(status, &body),
        (404, "no_paragraph".into()),
        "{body}"
    );

    // The body cap now answers in the SAME JSON shape as every other
    // axis (it used to be axum's plain text) — payload_too_large.
    let (status, body) = server.call_with_token(
        "PUT",
        "/contexts/big",
        Some(json!({"description": "こ".repeat(8000)})),
        key,
    );
    assert_eq!(
        code_of(status, &body),
        (413, "payload_too_large".into()),
        "{body}"
    );
}

#[test]
fn oversized_names_are_rejected_at_every_write_boundary() {
    let server = Server::start("namecap");
    let long = "字".repeat(400); // 1200 bytes, over the 1024-byte name cap

    // A context name becomes a file stem (percent-encoded ×3): 64
    // bytes is the cap.
    let (status, body) = server.call(
        "PUT",
        &format!("/contexts/{}", "n".repeat(65)),
        Some(json!({})),
    );
    assert_eq!(status, 400, "{body}");

    // The description rides in every directory listing.
    let (status, body) = server.call(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "d".repeat(5000)})),
    );
    assert_eq!(status, 400, "{body}");

    server.ok("PUT", "/contexts/sake", Some(json!({})));
    let (status, body) = server.call(
        "PATCH",
        "/contexts/sake",
        Some(json!({"description": "d".repeat(5000)})),
    );
    assert_eq!(status, 400, "{body}");

    // Graph names: the top-concepts snapshot carries them into every
    // GET /contexts response, far outside the cache budget.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": long, "label": "l", "object": "o", "weight": 1.0}])),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("associations[0].subject"),
        "the message must point at the offending field: {body}"
    );

    // Aliases and passage source ids persist names too.
    let mut concepts = serde_json::Map::new();
    concepts.insert(long.clone(), json!("x"));
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": concepts, "labels": {}})),
    );
    assert_eq!(status, 400, "{body}");
    let mut passages = serde_json::Map::new();
    passages.insert(long.clone(), json!("原文"));
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": passages})),
    );
    assert_eq!(status, 400, "{body}");

    // retract_source's source id is a name like any other — refused
    // before the lookup, marker fsync, and WAL fsync it would otherwise
    // pay for on every oversized call.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": long})),
    );
    assert_eq!(status, 400, "{body}");

    // A rename's destination rides in the body, not the path — it
    // needs its own cap, for both the context and the group route.
    let (status, body) = server.call("POST", "/contexts/sake/rename", Some(json!({"to": long})));
    assert_eq!(status, 400, "{body}");
    server.ok("PUT", "/groups/kura", Some(json!({})));
    let (status, body) = server.call("POST", "/groups/kura/rename", Some(json!({"to": long})));
    assert_eq!(status, 400, "{body}");

    // Nothing landed anywhere.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));

    // The boundary itself stays usable.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{
            "subject": "s".repeat(1024), "label": "l", "object": "o", "weight": 1.0
        }])),
    );
}

#[test]
fn empty_names_are_rejected_at_the_write_boundary() {
    let server = Server::start("emptyname");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    // An empty subject/label/object is not a degenerate name, it is no
    // name — each must be refused on its own, naming the offending
    // field.
    for (field, triple) in [
        (
            "subject",
            json!({"subject": "", "label": "l", "object": "o", "weight": 1.0}),
        ),
        (
            "label",
            json!({"subject": "s", "label": "", "object": "o", "weight": 1.0}),
        ),
        (
            "object",
            json!({"subject": "s", "label": "l", "object": "", "weight": 1.0}),
        ),
    ] {
        let (status, body) =
            server.call("POST", "/contexts/sake/associations", Some(json!([triple])));
        assert_eq!(status, 400, "{field}: {body}");
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains(&format!("associations[0].{field}")),
            "the message must point at the offending field: {body}"
        );
    }

    // An empty alias spelling is worse than unaddressable:
    // `str::contains("")` is always true, so once interned it would
    // containment-match every future cue as a phantom hit. Both roles,
    // both namespaces.
    for (name, request) in [
        (
            "empty concept alias",
            json!({"concepts": {"": "x"}, "labels": {}}),
        ),
        (
            "empty concept canonical",
            json!({"concepts": {"a": ""}, "labels": {}}),
        ),
        (
            "empty label alias",
            json!({"concepts": {}, "labels": {"": "x"}}),
        ),
        (
            "empty label canonical",
            json!({"concepts": {}, "labels": {"l": ""}}),
        ),
    ] {
        let (status, body) = server.call("POST", "/contexts/sake/aliases", Some(request));
        assert_eq!(status, 400, "{name}: {body}");
    }

    // A source that is PRESENT is a name like any other: empty would
    // intern a real, permanent source id that unrelated callers'
    // mistakes then silently merge into (and retract together).
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "s", "label": "l", "object": "o", "weight": 1.0, "source": ""}])),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("associations[0].source"),
        "{body}"
    );

    // The passage store keys sources the same way, and question and
    // section text is embedded verbatim on refresh — providers refuse
    // zero-length input, which would stall the refresh pass for good.
    for (name, request) in [
        ("empty passage source id", json!({"passages": {"": "text"}})),
        (
            "empty question",
            json!({
                "passages": {"doc.md": "text"},
                "questions": {"doc.md": [{"paragraph": 0, "question": ""}]},
            }),
        ),
        (
            "empty section",
            json!({
                "passages": {"doc.md": "text"},
                "sections": {"doc.md": [{"paragraph": 0, "section": ""}]},
            }),
        ),
    ] {
        let (status, body) = server.call("POST", "/contexts/sake/sources", Some(request));
        assert_eq!(status, 400, "{name}: {body}");
    }

    // retract_source's source id is no exception either.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": ""})),
    );
    assert_eq!(status, 400, "{body}");

    // An omitted source is the ordinary unsourced-association case,
    // not a missing name — it must NOT be swept up by the same check.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "s", "label": "l", "object": "o", "weight": 1.0}])),
    );

    // Only the one, deliberately unsourced association landed.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(1));
}

/// issue #182: a batch with several distinct bad associations gets one
/// issue per rejected path in a single round trip, not just the first
/// — the collect-all discipline ADR 0001 §8 already gives extraction,
/// now applied to this REST write.
#[test]
fn add_associations_collects_every_issue_in_one_pass() {
    let server = Server::start("collectall");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "", "label": "l", "object": "o", "weight": 1.0},
            {"subject": "s", "label": "l", "object": "o", "weight": 1e300},
            {"subject": "s2", "label": "l", "object": "o", "weight": "strong"},
        ])),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("invalid_argument"), "{body}");
    assert_eq!(body["integrity"], json!("nothing_written"), "{body}");
    assert_eq!(body["retryable_after_correction"], json!(true), "{body}");
    let issues = body["issues"].as_array().expect("issues array");
    assert_eq!(issues.len(), 3, "{body}");
    let paths: Vec<&str> = issues
        .iter()
        .map(|issue| issue["path"].as_str().unwrap())
        .collect();
    assert_eq!(
        paths,
        vec![
            "associations[0].subject",
            "associations[1].weight",
            "associations[2].weight",
        ],
        "{body}"
    );
    assert_eq!(issues[0]["kind"], json!("empty"), "{body}");
    assert_eq!(issues[1]["kind"], json!("range"), "{body}");
    assert_eq!(issues[2]["kind"], json!("type"), "{body}");
    assert_eq!(issues[2]["actual"], json!("string"), "{body}");

    // Refused whole: nothing landed.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));
}

/// issue #182: a business-rule-invalid item never silently disappears
/// into a subset write — the write boundary and the response shape
/// documented for the acceptance criteria stay true even for the
/// smallest one-bad-item batch.
#[test]
fn add_associations_invalid_batch_never_writes_a_valid_subset() {
    let server = Server::start("nosubset");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "good", "label": "l", "object": "o", "weight": 1.0},
            {"subject": "bad", "label": "l", "object": "o", "weight": "nope"},
        ])),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["integrity"], json!("nothing_written"), "{body}");
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));
}

/// issue #182 acceptance criterion: invalid `store_passages`
/// questions/sections return paths that identify both the source and
/// the item index.
#[test]
fn store_passages_issue_paths_name_the_source_and_item_index() {
    let server = Server::start("passagepaths");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"doc.md": "text"},
            "questions": {"doc.md": [
                {"paragraph": 0, "question": "fine?"},
                {"paragraph": 0, "question": 123},
            ]},
        })),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("invalid_argument"), "{body}");
    assert_eq!(body["integrity"], json!("nothing_written"), "{body}");
    let issues = body["issues"].as_array().expect("issues array");
    assert_eq!(issues.len(), 1, "{body}");
    assert_eq!(
        issues[0]["path"],
        json!("questions['doc.md'][1].question"),
        "{body}"
    );
    assert_eq!(issues[0]["kind"], json!("type"), "{body}");

    // Nothing was stored — an orphaned source in `sections` collects
    // alongside a wrong-typed question in one pass too.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"doc.md": "text"},
            "sections": {"ghost.md": [{"paragraph": 0, "section": "intro"}]},
        })),
    );
    assert_eq!(status, 400, "{body}");
    let issues = body["issues"].as_array().expect("issues array");
    assert_eq!(issues.len(), 1, "{body}");
    assert_eq!(issues[0]["path"], json!("sections['ghost.md']"), "{body}");
    assert_eq!(issues[0]["kind"], json!("unknown_reference"), "{body}");

    let lookup = server.ok(
        "POST",
        "/contexts/sake/sources/lookup",
        Some(json!({"sources": ["doc.md"]})),
    );
    assert_eq!(lookup["missing"], json!(["doc.md"]), "{lookup}");
}
