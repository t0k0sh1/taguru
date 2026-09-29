//! `associations.rs`/`aliases.rs` validation gaps (issue #626):
//! `errors.rs` covers subject/label/object shape errors thoroughly but
//! never `paragraph`'s non-integer values, `source`'s over-length
//! side (only its empty side), or a wrong-type (bool/array/object)
//! subject/label/object at the HTTP level. `POST .../associations/
//! retract` has NO validation test at all — every call site in
//! `tests/` uses valid input for setup only. And the schema-load
//! failure fail-closed arm (`Some(Err(message))`) was never driven
//! through the HTTP surface. Aliases: the success envelope's shape
//! (a bare `applied` count, nothing richer) and the write counter's
//! interaction with a non-empty batch were both unconfirmed.

use std::fs;

use serde_json::{Value, json};

use crate::support::*;

// --- interpret_paragraph: every non-integer JSON shape ---------------------

/// `interpret_paragraph` (`associations.rs`) accepts a missing/null
/// paragraph silently but must refuse every other non-non-negative-
/// integer shape with `kind: "type"` — including negative and
/// fractional numbers, which fall into the SAME `wrong_type` arm as
/// bool/array/object/string (`Value::Number::as_u64()` fails for
/// both), so `actual` reads `"number"` there, not a `"range"` kind.
#[test]
fn paragraph_rejects_every_non_integer_shape() {
    let server = Server::start("assoc-paragraph-shapes");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let cases: &[(&str, Value, &str)] = &[
        ("negative", json!(-1), "number"),
        ("fractional", json!(1.5), "number"),
        ("string", json!("0"), "string"),
        ("boolean", json!(true), "boolean"),
        ("array", json!([0]), "array"),
        ("object", json!({"n": 0}), "object"),
        // u32::MAX + 1: still a JSON number, still fails the
        // u32::try_from conversion the same way a negative one does.
        ("overflow", json!(4_294_967_296u64), "number"),
    ];
    for (name, paragraph, expected_actual) in cases {
        let (status, body) = server.call(
            "POST",
            "/contexts/sake/associations",
            Some(json!([
                {"subject": "s", "label": "l", "object": "o", "weight": 1.0,
                 "source": "a.md", "paragraph": paragraph}
            ])),
        );
        assert_eq!(status, 400, "{name}: {body}");
        assert_eq!(body["code"], json!("invalid_argument"), "{name}: {body}");
        let issues = body["issues"].as_array().expect("issues array");
        assert_eq!(issues.len(), 1, "{name}: {body}");
        assert_eq!(
            issues[0]["path"],
            json!("associations[0].paragraph"),
            "{name}: {body}"
        );
        assert_eq!(issues[0]["kind"], json!("type"), "{name}: {body}");
        assert_eq!(
            issues[0]["actual"],
            json!(*expected_actual),
            "{name}: {body}"
        );
        assert_eq!(
            issues[0]["expected"],
            json!("a non-negative integer paragraph index"),
            "{name}: {body}"
        );
    }

    // A missing/null paragraph is not an error at all — the
    // corrective case, so the refusal-shape assertions above are not
    // vacuously true for a handler that rejects everything.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "s", "label": "l", "object": "o", "weight": 1.0}
        ])),
    );
}

// --- interpret_source: the over-length side ---------------------------------

/// `errors.rs` covers an empty `source`; the over-`MAX_NAME_BYTES`
/// side of the same `check_bounded_len` call was never exercised.
#[test]
fn source_over_the_name_byte_cap_is_rejected() {
    let server = Server::start("assoc-source-too-long");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let long_source = "字".repeat(400); // 1200 bytes, over the 1024-byte cap
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "s", "label": "l", "object": "o", "weight": 1.0,
             "source": long_source}
        ])),
    );
    assert_eq!(status, 400, "{body}");
    let issues = body["issues"].as_array().expect("issues array");
    assert_eq!(issues.len(), 1, "{body}");
    assert_eq!(issues[0]["path"], json!("associations[0].source"), "{body}");
    assert_eq!(issues[0]["kind"], json!("too_long"), "{body}");
}

// --- subject/label/object wrong-type, at the HTTP level --------------------

/// `interpret_bounded_text` (`src/api.rs`, shared by subject/label/
/// object) has a unit test for a numeric wrong-type
/// (`interpret_bounded_text_reports_missing_and_wrong_type`); nothing
/// drives bool/array/object through the actual HTTP route.
#[test]
fn subject_label_object_reject_every_wrong_json_type() {
    let server = Server::start("assoc-wrong-types");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    for field in ["subject", "label", "object"] {
        for (name, value) in [
            ("boolean", json!(true)),
            ("array", json!(["x"])),
            ("object", json!({"x": 1})),
        ] {
            let mut item = json!({"subject": "s", "label": "l", "object": "o", "weight": 1.0});
            item[field] = value;
            let (status, body) =
                server.call("POST", "/contexts/sake/associations", Some(json!([item])));
            assert_eq!(status, 400, "{field}/{name}: {body}");
            let issues = body["issues"].as_array().expect("issues array");
            assert_eq!(issues.len(), 1, "{field}/{name}: {body}");
            assert_eq!(
                issues[0]["path"],
                json!(format!("associations[0].{field}")),
                "{field}/{name}: {body}"
            );
            assert_eq!(issues[0]["kind"], json!("type"), "{field}/{name}: {body}");
        }
    }
}

// --- POST .../associations/retract: no validation test existed at all ------

/// `retract_association` validates subject/label/object through
/// `empty`/`oversized` (`src/api.rs`) directly — a fail-FAST loop, not
/// `add_associations`'s collect-all `issues` array — so the wire shape
/// is a plain refusal message, and it stops at the FIRST bad field
/// rather than reporting every one.
#[test]
fn retract_association_rejects_empty_and_oversized_fields_and_stops_at_the_first() {
    let server = Server::start("assoc-retract-validation");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let long = "s".repeat(1025);
    for (name, request, offending_field) in [
        (
            "empty subject",
            json!({"subject": "", "label": "l", "object": "o"}),
            "subject",
        ),
        (
            "empty label",
            json!({"subject": "s", "label": "", "object": "o"}),
            "label",
        ),
        (
            "empty object",
            json!({"subject": "s", "label": "l", "object": ""}),
            "object",
        ),
        (
            "oversized subject",
            json!({"subject": long.clone(), "label": "l", "object": "o"}),
            "subject",
        ),
        (
            "oversized label",
            json!({"subject": "s", "label": long.clone(), "object": "o"}),
            "label",
        ),
        (
            "oversized object",
            json!({"subject": "s", "label": "l", "object": long.clone()}),
            "object",
        ),
    ] {
        let (status, body) =
            server.call("POST", "/contexts/sake/associations/retract", Some(request));
        assert_eq!(status, 400, "{name}: {body}");
        assert_eq!(body["code"], json!("invalid_argument"), "{name}: {body}");
        // Fail-fast, plain-message shape: no `issues` array at all —
        // unlike `add_associations`'s collect-all refusal.
        assert!(body.get("issues").is_none(), "{name}: {body}");
        assert!(
            body["error"].as_str().unwrap().starts_with(offending_field),
            "{name}: the message must name the offending field first: {body}"
        );
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .ends_with("nothing was applied"),
            "{name}: {body}"
        );
    }

    // Both subject AND label are bad — the loop stops at subject
    // (checked first) and never mentions label, proving fail-fast
    // rather than collect-all.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations/retract",
        Some(json!({"subject": "", "label": "", "object": "o"})),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"].as_str().unwrap().starts_with("subject"),
        "{body}"
    );

    // A JSON type mismatch is a different failure mode entirely: the
    // typed `AppJson<RetractAssociationRequest>` extractor rejects it
    // before the handler's own field-by-field walk ever runs.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations/retract",
        Some(json!({"subject": 5, "label": "l", "object": "o"})),
    );
    // axum's `JsonRejection` maps a deserialize failure to 422, not
    // 400 — distinct from every plain-body-parse-failure 400 this file
    // and `errors.rs` otherwise pin.
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");
}

// --- schema-load failure: fail-closed, driven through HTTP ------------------

/// A write into a schema-guarded context whose image will not load
/// must be a hard refusal with nothing written — never a silent
/// fallback (`crate::schema`'s module doc). A rename used to be the
/// one wire-reachable route to the unresolved-schema slow path; under
/// ids a rename re-registers nothing (the entry survives intact), so
/// that state is a replica-only affair covered by the unit tests
/// (`hidden_label_fails_closed_when_schema_resolution_errors`,
/// `ensure_hot_refuses_to_load_when_its_own_schema_check_fails`).
/// What the wire can still reach — and what this pins — is the
/// fail-closed mapping itself: a restart drops the context cold, the
/// corrupted image fails its next load, and the write answers 500
/// with `nothing_written`.
#[test]
fn add_associations_fails_closed_when_the_schema_image_is_corrupt() {
    let mut server = Server::start("assoc-schema-load-failure");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "PUT",
        "/contexts/sake/schema",
        Some(json!({
            "type": "schema", "mode": "off", "closed_labels": false,
            "types": {}, "relations": {}
        })),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "s", "label": "l", "object": "o", "weight": 1.0}
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/rename",
        Some(json!({"to": "shochu"})),
    );

    let image = server
        .data_dir
        .join(format!("{}.ctx", server.context_stem("shochu")));
    // Restart so the context is cold — a rename no longer unloads
    // anything, and a hot context would serve the write from memory
    // without ever reading the corrupted bytes.
    let data_dir = server.stop_gracefully();
    let mut bytes = fs::read(&image).expect("the context image must exist");
    assert!(bytes.len() > 8, "sanity: the version byte must exist");
    bytes[8] = 0xFF;
    fs::write(&image, &bytes).expect("the corrupted image must be writable");
    server = Server::start_on("assoc-schema-load-failure-reboot", data_dir);

    let (status, body) = server.call(
        "POST",
        "/contexts/shochu/associations",
        Some(json!([
            {"subject": "s2", "label": "l", "object": "o2", "weight": 1.0}
        ])),
    );
    assert_eq!(status, 500, "{body}");
    assert_eq!(body["code"], json!("internal"), "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("image corrupt"),
        "{body}"
    );
}

// --- aliases: the success envelope's actual (flat) shape --------------------

/// The success `result` is a bare `applied` count — `ok(applied,
/// started_at)` in `add_aliases`/`remove_aliases` — no per-item
/// results exist to check; this pins that flat shape as behavior
/// rather than leaving it unconfirmed.
#[test]
fn aliases_success_result_is_a_bare_applied_count() {
    let server = Server::start("aliases-envelope-shape");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "l", "object": "o", "weight": 1.0}
        ])),
    );

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"青嶺酒蔵": "青嶺酒造"}, "labels": {}})),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["status"], json!("ok"), "{body}");
    assert_eq!(body["result"], json!(1), "{body}");

    let (status, body) = server.call(
        "DELETE",
        "/contexts/sake/aliases",
        Some(json!({"concepts": ["青嶺酒蔵"], "labels": []})),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"], json!(1), "{body}");
}

/// `metrics.rs`'s own aliases/write-counter test only ever covers the
/// EMPTY batch (`applied == 0` must not bump `usage.writes`); the
/// non-empty side of that same rule was never checked.
#[test]
fn a_non_empty_alias_batch_bumps_the_write_counter() {
    let server = Server::start("aliases-write-counter");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "l", "object": "o", "weight": 1.0}
        ])),
    );

    let before = server.ok("GET", "/contexts/sake", None);
    assert_eq!(before["usage"]["writes"], json!(1), "{before}");

    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"青嶺酒蔵": "青嶺酒造"}, "labels": {}})),
    );

    let after = server.ok("GET", "/contexts/sake", None);
    assert_eq!(after["usage"]["writes"], json!(2), "{after}");
}
