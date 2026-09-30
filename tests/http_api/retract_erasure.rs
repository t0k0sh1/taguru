//! The document-erasure lifecycle (#437): `sources/retract?dry_run=true`
//! previews without writing, and `POST /contexts/{name}/compact` rewrites
//! the passage log alongside the graph image so "retract, then compact"
//! removes a document's bytes, not just its truth.

use serde_json::json;

use crate::support::*;

fn seed(server: &Server) {
    server.ok("POST", "/contexts", Some(json!({"name": "sake"})));
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("sake")),
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0,
             "source": "doc.md"},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0,
             "source": "doc.md"},
        ])),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/sources", server.cx("sake")),
        Some(json!({"passages": {"doc.md": "青嶺酒造の紹介。"}})),
    );
}

#[test]
fn dry_run_reports_the_real_footprint_and_writes_nothing() {
    let server = Server::start("retract-dry-run");
    seed(&server);
    let tail = server.ok(
        "GET",
        &format!("/contexts/{}/changes", server.cx("sake")),
        None,
    );
    let cursor = tail["next"].as_str().unwrap().to_string();

    let preview = server.ok(
        "POST",
        &format!(
            "/contexts/{}/sources/retract?dry_run=true",
            server.cx("sake")
        ),
        Some(json!({"source": "doc.md"})),
    );
    assert_eq!(preview["associations_touched"], json!(2), "{preview}");
    assert_eq!(preview["passage_removed"], json!(true), "{preview}");

    // Nothing changed: the graph still answers, the source is still
    // listed, and the change feed heard nothing.
    let recall = server.ok(
        "POST",
        &format!("/contexts/{}/recall", server.cx("sake")),
        Some(json!({"cue": "青嶺酒造"})),
    );
    assert_eq!(recall["total"], json!(2), "{recall}");
    let sources = server.ok(
        "GET",
        &format!("/contexts/{}/sources", server.cx("sake")),
        None,
    );
    assert_eq!(sources["sources"], json!(["doc.md"]), "{sources}");
    let feed = server.ok(
        "GET",
        &format!("/contexts/{}/changes?since={cursor}", server.cx("sake")),
        None,
    );
    assert_eq!(feed["events"], json!([]), "{feed}");

    // The real retraction then reports exactly what the preview said.
    let real = server.ok(
        "POST",
        &format!("/contexts/{}/sources/retract", server.cx("sake")),
        Some(json!({"source": "doc.md"})),
    );
    assert_eq!(
        real["associations_touched"],
        preview["associations_touched"]
    );
    assert_eq!(real["passage_removed"], preview["passage_removed"]);
}

#[test]
fn dry_run_on_an_unknown_source_previews_zero() {
    let server = Server::start("retract-dry-run-unknown");
    seed(&server);
    let preview = server.ok(
        "POST",
        &format!(
            "/contexts/{}/sources/retract?dry_run=true",
            server.cx("sake")
        ),
        Some(json!({"source": "never-stored.md"})),
    );
    assert_eq!(preview["associations_touched"], json!(0));
    assert_eq!(preview["passage_removed"], json!(false));

    let (status, _) = server.call(
        "POST",
        &format!(
            "/contexts/{}/sources/retract?dry_run=true",
            "00000000-0000-4000-8000-00000000dead"
        ),
        Some(json!({"source": "doc.md"})),
    );
    assert_eq!(status, 404);
}

#[test]
fn compact_rewrites_the_passage_log_when_one_exists() {
    let server = Server::start("retract-compact");
    seed(&server);
    server.ok(
        "POST",
        &format!("/contexts/{}/sources/retract", server.cx("sake")),
        Some(json!({"source": "doc.md"})),
    );

    let outcome = server.ok(
        "POST",
        &format!("/contexts/{}/compact", server.cx("sake")),
        None,
    );
    assert_eq!(outcome["passages_compacted"], json!(true), "{outcome}");
    // The passage stays gone after the rewrite.
    let sources = server.ok(
        "GET",
        &format!("/contexts/{}/sources", server.cx("sake")),
        None,
    );
    assert_eq!(sources["total"], json!(0), "{sources}");

    // A context with no passage history has no log to rewrite — the
    // guard must not mint store files just to compact nothing.
    server.ok("POST", "/contexts", Some(json!({"name": "graph-only"})));
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("graph-only")),
        Some(json!([{"subject": "a", "label": "r", "object": "b", "weight": 1.0}])),
    );
    let outcome = server.ok(
        "POST",
        &format!("/contexts/{}/compact", server.cx("graph-only")),
        None,
    );
    assert_eq!(outcome["passages_compacted"], json!(false), "{outcome}");
}
