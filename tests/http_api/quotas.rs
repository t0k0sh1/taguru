//! Per-context quotas over HTTP (issue #136): the storage ceiling
//! refuses growth with the documented 507 `storage_full` contract
//! while the shrink paths stay open, `/import` stops at the capped
//! batch as a resumable prefix, and both the declared ceilings and
//! the refusal count surface on `/metrics` beside the #137 families.

use serde_json::json;

use super::support::{Server, post_import, post_import_dry_run};

#[test]
fn storage_quota_refuses_growth_with_507_and_keeps_the_ways_down_open() {
    let server = Server::start_with_env(
        "storage-quota",
        &[
            (
                "TAGURU_CONTEXT_QUOTAS",
                r#"{"capped": {"storage_bytes": 1, "cache_bytes": 1048576}}"#,
            ),
            ("TAGURU_METRICS_PER_CONTEXT", "1"),
        ],
    );

    // Creation is never gated — the declaration waits for the name.
    server.ok(
        "PUT",
        "/contexts/capped",
        Some(json!({"description": "quota'd tenant"})),
    );
    // The first write lands (nothing on disk yet); its WAL append
    // alone carries the family past the one-byte ceiling, so the
    // second growth write refuses with the documented contract.
    server.ok(
        "POST",
        "/contexts/capped/associations",
        Some(json!([{
            "subject": "蔵", "label": "杜氏", "object": "高瀬",
            "weight": 1.0, "source": "keep.md"
        }])),
    );
    let (status, refused) = server.call(
        "POST",
        "/contexts/capped/associations",
        Some(json!([{
            "subject": "蔵", "label": "銘柄", "object": "青嶺",
            "weight": 1.0, "source": "keep.md"
        }])),
    );
    assert_eq!(status, 507, "{refused}");
    assert_eq!(refused["code"], json!("storage_full"), "{refused}");
    let message = refused["error"].as_str().unwrap();
    assert!(message.contains("storage quota"), "{message}");
    assert!(message.contains("retract or compact"), "{message}");

    // Passages are the other growth entrance — same contract.
    let (status, refused) = server.call(
        "POST",
        "/contexts/capped/sources",
        Some(json!({"passages": {"keep.md": "原文の段落。"}})),
    );
    assert_eq!(status, 507, "{refused}");
    assert_eq!(refused["code"], json!("storage_full"), "{refused}");

    // The ways down the refusal names stay open at the ceiling.
    server.ok(
        "POST",
        "/contexts/capped/sources/retract",
        Some(json!({"source": "keep.md"})),
    );
    server.ok("POST", "/contexts/capped/compact", None);

    // An uncapped sibling writes freely through all of it.
    server.ok(
        "PUT",
        "/contexts/free",
        Some(json!({"description": "no quota"})),
    );
    server.ok(
        "POST",
        "/contexts/free/associations",
        Some(json!([{
            "subject": "蔵", "label": "杜氏", "object": "高瀬",
            "weight": 1.0, "source": "keep.md"
        }])),
    );

    // The declared ceilings ride the per-context families; the
    // refusals count on their own series; an uncapped context renders
    // no quota series at all.
    let (status, body) = server.call("GET", "/metrics", None);
    assert_eq!(status, 200);
    let text = body.as_str().expect("metrics body is text, not JSON");
    assert!(
        text.contains("taguru_context_quota_bytes{context=\"capped\",resource=\"storage\"} 1\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "taguru_context_quota_bytes{context=\"capped\",resource=\"cache\"} 1048576\n"
        ),
        "{text}"
    );
    assert!(
        !text.contains("taguru_context_quota_bytes{context=\"free\""),
        "an uncapped context renders no quota series"
    );
    let refusals = text
        .lines()
        .find_map(|line| {
            line.strip_prefix("taguru_storage_quota_refusals_total ")
                .and_then(|value| value.trim().parse::<u64>().ok())
        })
        .expect("the refusal counter renders");
    assert_eq!(refusals, 2, "one graph refusal, one passages refusal");
}

/// `/import` stops at the first batch whose target context is at its
/// ceiling — a resumable prefix like a spent deadline — after landing
/// every batch before it. A dry run previews straight through: its
/// capacity answers are advisory by documented contract.
#[test]
fn import_stops_at_the_capped_batch_as_a_resumable_prefix() {
    let server = Server::start_with_env(
        "import-quota",
        &[(
            "TAGURU_CONTEXT_QUOTAS",
            r#"{"capped": {"storage_bytes": 1}}"#,
        )],
    );

    // Put the capped context at its ceiling before the import.
    server.ok(
        "PUT",
        "/contexts/capped",
        Some(json!({"description": "quota'd tenant"})),
    );
    server.ok(
        "POST",
        "/contexts/capped/associations",
        Some(json!([{
            "subject": "蔵", "label": "杜氏", "object": "高瀬",
            "weight": 1.0, "source": "keep.md"
        }])),
    );

    let stream = concat!(
        "{\"taguru_batch\": 1, \"context\": \"free\", \"source\": \"a.md\", ",
        "\"create\": {\"description\": \"uncapped\"}}\n",
        "{\"subject\": \"蔵\", \"label\": \"銘柄\", \"object\": \"青嶺\", \"weight\": 1.0}\n",
        "{\"taguru_batch\": 1, \"context\": \"capped\", \"source\": \"b.md\"}\n",
        "{\"subject\": \"蔵\", \"label\": \"産地\", \"object\": \"灘\", \"weight\": 1.0}\n",
    );
    let (status, previewed) = post_import_dry_run(&server, stream, None);
    assert_eq!(status, 200, "{previewed}");

    let (status, refused) = post_import(&server, stream, None);
    assert_eq!(status, 507, "{refused}");
    assert_eq!(refused["code"], json!("storage_full"), "{refused}");
    let message = refused["error"].as_str().unwrap();
    assert!(message.contains("storage quota"), "{message}");
    assert!(
        message.contains("batch 2 of 2"),
        "the refusal names the resumable position: {message}"
    );

    // The uncapped batch before the stop landed and stayed.
    let (status, free) = server.call("GET", "/contexts/free", None);
    assert_eq!(status, 200, "{free}");
}
