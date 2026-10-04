//! The change feed (#422): `GET /contexts/{name}/changes` — event kinds
//! and per-call aggregation across every write entrance, cursor
//! resume/paging, and the `stale_cursor` contract. The ring's own
//! eviction/epoch arithmetic is unit-tested in
//! `src/registry/changes.rs`; this covers the HTTP surface over real
//! writes.

use serde_json::{Value, json};

use crate::support::*;

fn tail_cursor(server: &Server, context: &str) -> String {
    let tail = server.ok(
        "GET",
        &format!("/contexts/{}/changes", server.cx(context)),
        None,
    );
    assert_eq!(tail["events"], json!([]));
    assert_eq!(tail["more"], json!(false));
    tail["next"].as_str().expect("cursor").to_string()
}

fn kinds(page: &Value) -> Vec<String> {
    page["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|event| event["type"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn every_write_entrance_feeds_the_expected_event() {
    let server = Server::start("changes-events");
    server.create_with_id(
        "cef2e28b-43f0-4b6c-8201-abab0785399f",
        json!({"name": "sake"}),
    );
    let cursor = tail_cursor(&server, "sake");

    // One write call = one aggregated event, however many lines it carried.
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("sake")),
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0},
        ])),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/aliases", server.cx("sake")),
        Some(json!({"concepts": {"Aomine": "青嶺酒造"}})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/sources", server.cx("sake")),
        Some(json!({"passages": {"doc.md": "青嶺酒造の紹介。"}})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/associations/retract", server.cx("sake")),
        Some(json!({"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/sources/retract", server.cx("sake")),
        Some(json!({"source": "doc.md"})),
    );
    server.ok(
        "PUT",
        &format!("/contexts/{}/schema", server.cx("sake")),
        Some(json!({
            "type": "schema", "mode": "warn", "closed_labels": false,
            "types": {}, "relations": {}
        })),
    );

    let page = server.ok(
        "GET",
        &format!("/contexts/{}/changes?since={cursor}", server.cx("sake")),
        None,
    );
    assert_eq!(
        kinds(&page),
        vec![
            "associations_added",
            "aliases_added",
            "source_stored",
            "association_retracted",
            "source_retracted",
            "schema_updated",
        ],
        "{page}"
    );
    let events = page["events"].as_array().unwrap();
    assert_eq!(events[0]["count"], json!(2), "two lines, one event");
    assert_eq!(events[2]["source"], json!("doc.md"));
    assert_eq!(events[3]["subject"], json!("青嶺酒造"));
    assert_eq!(events[4]["source"], json!("doc.md"));
    assert_eq!(events[5]["mode"], json!("warn"));
    assert_eq!(page["more"], json!(false));

    // The page's own cursor resumes past everything it served.
    let next = page["next"].as_str().unwrap();
    let after = server.ok(
        "GET",
        &format!("/contexts/{}/changes?since={next}", server.cx("sake")),
        None,
    );
    assert_eq!(after["events"], json!([]));

    // An idempotent re-PUT of the same schema feeds nothing.
    server.ok(
        "PUT",
        &format!("/contexts/{}/schema", server.cx("sake")),
        Some(json!({
            "type": "schema", "mode": "warn", "closed_labels": false,
            "types": {}, "relations": {}
        })),
    );
    let after = server.ok(
        "GET",
        &format!("/contexts/{}/changes?since={next}", server.cx("sake")),
        None,
    );
    assert_eq!(after["events"], json!([]), "{after}");
}

#[test]
fn an_import_feeds_the_same_events_as_its_component_writes() {
    let server = Server::start("changes-import");
    server.create_with_id(
        "cef2e28b-43f0-4b6c-8201-abab0785399f",
        json!({"name": "sake"}),
    );
    let cursor = tail_cursor(&server, "sake");

    let batch = "{\"type\": \"source\", \"context_id\": \"cef2e28b-43f0-4b6c-8201-abab0785399f\", \"id\": \"doc.md\"}\n\
                 {\"passage\": \"本文。\"}\n\
                 {\"subject\": \"a\", \"label\": \"r\", \"object\": \"b\", \"weight\": 1.0}\n\
                 {\"subject\": \"a\", \"label\": \"r\", \"object\": \"c\", \"weight\": 1.0}\n";
    let (status, body) = post_import(&server, batch, None);
    assert_eq!(status, 200, "{body}");

    let page = server.ok(
        "GET",
        &format!("/contexts/{}/changes?since={cursor}", server.cx("sake")),
        None,
    );
    let observed = kinds(&page);
    // A re-import replaces per source: the batch retracts the source's
    // old contribution, applies the associations, and stores the
    // passage — each a real content change the feed reports.
    assert!(
        observed.contains(&"associations_added".to_string()),
        "{page}"
    );
    assert!(observed.contains(&"source_stored".to_string()), "{page}");
}

#[test]
fn limit_pages_with_more_and_the_cursor_walks_the_gap() {
    let server = Server::start("changes-paging");
    server.create_with_id(
        "cef2e28b-43f0-4b6c-8201-abab0785399f",
        json!({"name": "sake"}),
    );
    let cursor = tail_cursor(&server, "sake");
    for index in 0..3 {
        server.ok(
            "POST",
            &format!("/contexts/{}/associations", server.cx("sake")),
            Some(json!([
                {"subject": format!("s{index}"), "label": "r", "object": "o", "weight": 1.0},
            ])),
        );
    }

    let first = server.ok(
        "GET",
        &format!(
            "/contexts/{}/changes?since={cursor}&limit=2",
            server.cx("sake")
        ),
        None,
    );
    assert_eq!(first["events"].as_array().unwrap().len(), 2);
    assert_eq!(first["more"], json!(true), "{first}");

    let next = first["next"].as_str().unwrap();
    let second = server.ok(
        "GET",
        &format!(
            "/contexts/{}/changes?since={next}&limit=2",
            server.cx("sake")
        ),
        None,
    );
    assert_eq!(second["events"].as_array().unwrap().len(), 1);
    assert_eq!(second["more"], json!(false));
}

/// #676: `?limit=0` must floor to 1 (`clamp_page`), not serve an
/// empty, non-advancing page — a client honoring "more:true means
/// poll again immediately" would busy-loop against the same cursor
/// forever otherwise.
#[test]
fn limit_zero_is_floored_to_one_not_left_as_a_non_advancing_page() {
    let server = Server::start("changes-limit-zero");
    server.create_with_id(
        "cef2e28b-43f0-4b6c-8201-abab0785399f",
        json!({"name": "sake"}),
    );
    let cursor = tail_cursor(&server, "sake");
    for index in 0..2 {
        server.ok(
            "POST",
            &format!("/contexts/{}/associations", server.cx("sake")),
            Some(json!([
                {"subject": format!("s{index}"), "label": "r", "object": "o", "weight": 1.0},
            ])),
        );
    }

    let page = server.ok(
        "GET",
        &format!(
            "/contexts/{}/changes?since={cursor}&limit=0",
            server.cx("sake")
        ),
        None,
    );
    assert_eq!(page["events"].as_array().unwrap().len(), 1, "{page}");
    assert_eq!(page["more"], json!(true), "{page}");
    assert_ne!(
        page["next"].as_str().unwrap(),
        cursor,
        "the cursor must advance past the served event: {page}"
    );
}

#[test]
fn lost_positions_answer_stale_cursor_and_unknown_contexts_404() {
    let server = Server::start("changes-stale");
    server.create_with_id(
        "cef2e28b-43f0-4b6c-8201-abab0785399f",
        json!({"name": "sake"}),
    );

    for cursor in ["garbage", "cf1-00000000000000aa-7"] {
        let (status, body) = server.call(
            "GET",
            &format!("/contexts/{}/changes?since={cursor}", server.cx("sake")),
            None,
        );
        assert_eq!(status, 410, "{body}");
        assert_eq!(body["code"], json!("stale_cursor"), "{body}");
    }
    // issue #621: a stale cursor is a successfully-consulted context
    // whose answer happens to be a 410 — it must still count as a
    // read, the same as citation's UnknownSource/IndexOutOfRange arms
    // (advisory usage row only; eviction uses the separate last_touch
    // field, unaffected either way).
    let entry = server.ok("GET", &format!("/contexts/{}", server.cx("sake")), None);
    assert_eq!(entry["usage"]["reads"], json!(2), "{entry}");

    // Delete-and-recreate mints a new ring: the old cursor is gone even
    // though the name answers again.
    let cursor = tail_cursor(&server, "sake");
    server.ok("DELETE", &format!("/contexts/{}", server.cx("sake")), None);
    server.create_with_id(
        "cef2e28b-43f0-4b6c-8201-abab0785399f",
        json!({"name": "sake"}),
    );
    let (status, body) = server.call(
        "GET",
        &format!("/contexts/{}/changes?since={cursor}", server.cx("sake")),
        None,
    );
    assert_eq!(status, 410, "{body}");

    let (status, _) = server.call(
        "GET",
        &format!(
            "/contexts/{}/changes",
            "00000000-0000-4000-8000-00000000dead"
        ),
        None,
    );
    assert_eq!(status, 404);
}
