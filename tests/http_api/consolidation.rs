//! The consolidation audit (ADR 0012), end to end: caller-selected
//! sections, merge candidates corroborated structurally with types
//! attached, contradiction groups joined with assertion times and
//! ranked by measured functional tendency, contested edges with both
//! sides named, staleness gaps with an honest undatable count — and
//! fingerprints that hold still until the evidence moves.

use serde_json::{Value, json};

use crate::support::*;

/// One corpus exercising every section: spelling twins sharing
/// structure, a dated supersession under a functional-tendency label,
/// a sign-contested edge, and an undatable associations-only source.
fn seed(server: &Server) {
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "整理"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            // The twins: two spellings of one brewery, two shared facts,
            // one distinct fact each.
            {"subject": "青嶺酒造", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "doc-a"},
            {"subject": "青嶺酒蔵", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "doc-b"},
            {"subject": "青嶺酒造", "label": "所在地", "object": "霧沢町", "weight": 1.0, "source": "doc-a"},
            {"subject": "青嶺酒蔵", "label": "所在地", "object": "霧沢町", "weight": 1.0, "source": "doc-b"},
            {"subject": "青嶺酒造", "label": "創業", "object": "1907", "weight": 1.0, "source": "doc-a"},
            {"subject": "青嶺酒蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "doc-b"},
            // The supersession: 杜氏 is one-object for two subjects,
            // two-object for this one — old fact dated 1000, new 2000.
            {"subject": "蔵A", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "doc-2019"},
            {"subject": "蔵A", "label": "杜氏", "object": "青山", "weight": 1.0, "source": "doc-2024"},
            {"subject": "蔵B", "label": "杜氏", "object": "田中", "weight": 1.0, "source": "doc-2019"},
            // The dispute: one edge, two sources, opposite signs.
            {"subject": "蔵A", "label": "行う", "object": "大量生産", "weight": 1.0, "source": "宣伝"},
            {"subject": "蔵A", "label": "行う", "object": "大量生産", "weight": -2.0, "source": "doc-2024"},
            // Staleness: 蔵A's 銘柄 fact is only attested at 1000 while
            // its neighborhood reaches 2000; this one is undatable.
            {"subject": "蔵A", "label": "銘柄", "object": "初霜", "weight": 1.0, "source": "doc-2019"},
            {"subject": "蔵C", "label": "銘柄", "object": "幻", "weight": 1.0, "source": "doc-undated"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {
                "doc-2019": "旧情報。", "doc-2024": "新情報。", "宣伝": "宣伝文。",
                "doc-a": "表記Aの資料。", "doc-b": "表記Bの資料。"
            },
            "dates": {"doc-2019": 1000, "doc-2024": 2000, "宣伝": 1500}
        })),
    );
}

fn audit(server: &Server, body: Value) -> Value {
    server.ok("POST", "/contexts/sake/consolidation/audit", Some(body))
}

#[test]
fn sections_detect_join_and_fingerprint_their_candidates() {
    let server = Server::start("consolidation-audit");
    seed(&server);

    let report = audit(
        &server,
        json!({"checks": ["merge", "contradiction", "staleness"]}),
    );
    assert_eq!(report["detector"], json!("consolidation/1"));

    // Merge: the spelling twins surface with their shared structure.
    let merge = &report["merge"];
    assert_eq!(merge["total"], json!(1), "{merge}");
    let pair = &merge["candidates"][0];
    assert_eq!(pair["tier"], json!("lexical"));
    assert_eq!(pair["shared_total"], json!(2), "銘柄 and 所在地");
    assert_eq!(pair["only_a_total"], json!(1));
    assert_eq!(pair["only_b_total"], json!(1));
    assert_eq!(pair["overlap"], json!(0.5));

    // Contradiction: the grouped kind leads (ranked by measured
    // functional tendency), rows dated; the contested edge follows
    // with both sides named.
    let contradiction = &report["contradiction"];
    assert_eq!(contradiction["total"], json!(2), "{contradiction}");
    let grouped = &contradiction["candidates"][0];
    assert_eq!(grouped["kind"], json!("objects"));
    assert_eq!(grouped["subject"], json!("蔵A"));
    assert_eq!(grouped["label"], json!("杜氏"));
    // Three 杜氏 subjects (蔵A with two objects, 蔵B and 青嶺酒蔵 with
    // one each): tendency 2/3.
    assert_eq!(grouped["functional_tendency"], json!(2.0 / 3.0));
    let objects = grouped["objects"].as_array().unwrap();
    assert_eq!(objects[0]["object"], json!("高瀬"));
    assert_eq!(objects[0]["latest"], json!(1000));
    assert_eq!(objects[1]["object"], json!("青山"));
    assert_eq!(objects[1]["latest"], json!(2000));
    let contested = &contradiction["candidates"][1];
    assert_eq!(contested["kind"], json!("contested"));
    assert_eq!(contested["supporting_sources"], json!(["宣伝"]));
    assert_eq!(contested["disputing_sources"], json!(["doc-2024"]));

    // Staleness: 蔵A's 銘柄 fact (latest 1000) trails its own
    // neighborhood (2000); the undatable edge is counted, not guessed.
    let staleness = &report["staleness"];
    assert_eq!(staleness["undatable"], json!(1), "{staleness}");
    let stale = staleness["candidates"].as_array().unwrap();
    assert!(
        stale
            .iter()
            .any(|candidate| candidate["label"] == json!("銘柄")
                && candidate["gap"] == json!(1000)),
        "{staleness}"
    );

    // Fingerprints hold still across identical audits…
    let again = audit(&server, json!({"checks": ["contradiction"]}));
    assert_eq!(
        again["contradiction"]["candidates"][0]["fingerprint"],
        grouped["fingerprint"]
    );
    // …and move when the evidence moves.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵A", "label": "杜氏", "object": "青山", "weight": 1.0, "source": "doc-2025"}
        ])),
    );
    let moved = audit(&server, json!({"checks": ["contradiction"]}));
    assert_ne!(
        moved["contradiction"]["candidates"][0]["fingerprint"],
        grouped["fingerprint"]
    );

    // The selector is honest: unrequested sections are absent, an
    // empty or unknown selector refuses.
    let only_merge = audit(&server, json!({"checks": ["merge"]}));
    assert!(only_merge.get("contradiction").is_none());
    assert!(only_merge.get("staleness").is_none());
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/consolidation/audit",
        Some(json!({"checks": []})),
    );
    assert_eq!(status, 400, "{body}");
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/consolidation/audit",
        Some(json!({"checks": ["typo"]})),
    );
    assert_eq!(status, 400, "{body}");
    let (status, _) = server.call(
        "POST",
        "/contexts/nope/consolidation/audit",
        Some(json!({"checks": ["staleness"]})),
    );
    assert_eq!(status, 404);
}
