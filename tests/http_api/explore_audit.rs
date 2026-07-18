//! The explore/recall/query/audit graph-inspection endpoints.

use serde_json::{Value, json};

use crate::support::*;

#[test]
fn unreachable_from_pages_like_recall_and_query() {
    let server = Server::start("orphanpage");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0},
            // Three islands no walk from the origin can reach.
            {"subject": "x1", "label": "l", "object": "y1", "weight": 1.0},
            {"subject": "x2", "label": "l", "object": "y2", "weight": 1.0},
            {"subject": "x3", "label": "l", "object": "y3", "weight": 1.0},
        ])),
    );

    let audit = server.ok(
        "POST",
        "/contexts/sake/unreachable_from",
        Some(json!({"origins": ["青嶺酒造"], "limit": 2})),
    );
    assert_eq!(audit["total"], json!(3));
    assert_eq!(audit["matches"].as_array().unwrap().len(), 2);
}

/// A client resumes any of the three match-page endpoints past their
/// limit by building a [`MatchCursor`]-shaped `after` from the last
/// match on the previous page — `total` stays constant across pages,
/// and the walk ends when a page comes back empty.
#[test]
fn recall_query_and_unreachable_from_resume_past_a_match_cursor() {
    let server = Server::start("match-cursor");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "銘柄", "object": "a", "weight": 1.0},
            {"subject": "蔵", "label": "銘柄", "object": "b", "weight": 1.0},
            {"subject": "蔵", "label": "銘柄", "object": "c", "weight": 1.0},
            {"subject": "孤島1", "label": "l", "object": "先1", "weight": 1.0},
            {"subject": "孤島2", "label": "l", "object": "先2", "weight": 1.0},
            {"subject": "孤島3", "label": "l", "object": "先3", "weight": 1.0},
        ])),
    );
    let cursor_from = |m: &Value| {
        json!({
            "weight": m["weight"], "subject": m["subject"],
            "label": m["label"], "object": m["object"],
        })
    };

    // recall: page past the limit, total constant, walk ends on empty.
    let first = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "蔵", "limit": 2})),
    );
    assert_eq!(first["total"], json!(3), "{first}");
    let matches = first["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(
        matches
            .iter()
            .map(|m| m["object"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    let second = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "蔵", "limit": 2, "after": cursor_from(&matches[1])})),
    );
    assert_eq!(second["total"], json!(3), "{second}");
    let matches = second["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0]["object"], json!("c"), "{second}");
    let third = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "蔵", "limit": 2, "after": cursor_from(&matches[0])})),
    );
    assert_eq!(third["matches"], json!([]), "the walk has ended: {third}");

    // query: same cursor shape, position-pinned search.
    let first = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"label": "銘柄", "limit": 2})),
    );
    assert_eq!(first["total"], json!(3), "{first}");
    let matches = first["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 2);
    let second = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"label": "銘柄", "limit": 2, "after": cursor_from(&matches[1])})),
    );
    assert_eq!(second["total"], json!(3), "{second}");
    assert_eq!(second["matches"].as_array().unwrap().len(), 1);

    // unreachable_from: three islands past the reachable "蔵" cluster.
    let first = server.ok(
        "POST",
        "/contexts/sake/unreachable_from",
        Some(json!({"origins": ["蔵"], "limit": 2})),
    );
    assert_eq!(first["total"], json!(3), "{first}");
    let matches = first["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 2);
    let second = server.ok(
        "POST",
        "/contexts/sake/unreachable_from",
        Some(json!({
            "origins": ["蔵"], "limit": 2,
            "after": cursor_from(&matches[1]),
        })),
    );
    assert_eq!(second["total"], json!(3), "{second}");
    assert_eq!(second["matches"].as_array().unwrap().len(), 1);

    // A malformed cursor is a 422 (body fails to deserialize), not an
    // empty page.
    let (status, refusal) = server.call(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "蔵", "after": {"bogus": true}})),
    );
    assert_eq!(status, 422, "{refusal}");
}

/// `drift/audit` answers three things at once: unsourced weight
/// (worst-magnitude first, floor- and cursor-paginated like
/// `unreachable_from`), dead-canonical aliases, and — opt-in — the
/// same vocabulary-audit twins.
#[test]
fn audit_drift_surfaces_unsourced_weight_dead_aliases_and_pages_worst_first() {
    let server = Server::start("driftaudit");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    // Empty context: every field comes back at its zero value.
    let empty = server.ok("POST", "/contexts/sake/drift/audit", None);
    assert_eq!(empty["total"], json!(0), "{empty}");
    assert_eq!(empty["unsourced"], json!([]), "{empty}");
    assert_eq!(empty["dead_concept_aliases"], json!({}), "{empty}");
    assert_eq!(empty["dead_label_aliases"], json!({}), "{empty}");
    assert_eq!(empty["twins"], json!(null), "{empty}");

    // Three sourceless associations of increasing weight (unsourced
    // weight), plus one sourced association retracted afterward to
    // leave its fresh alias pointing at a dead canonical.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "x1", "label": "l", "object": "y1", "weight": 1.0},
            {"subject": "x2", "label": "l", "object": "y2", "weight": 2.0},
            {"subject": "x3", "label": "l", "object": "y3", "weight": 3.0},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "高瀬", "label": "杜氏", "object": "蔵",
                     "weight": 1.0, "source": "a.md"}])),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations/retract",
        Some(json!({"subject": "高瀬", "label": "杜氏", "object": "蔵"})),
    );
    let applied = server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"タカセ": "高瀬"}})),
    );
    assert_eq!(applied, json!(1), "{applied}");

    // No floor, no limit: everything, worst-magnitude first.
    let full = server.ok("POST", "/contexts/sake/drift/audit", None);
    assert_eq!(full["total"], json!(3), "{full}");
    let matches = full["unsourced"].as_array().unwrap();
    assert_eq!(
        matches
            .iter()
            .map(|m| m["unsourced_weight"].as_f64().unwrap())
            .collect::<Vec<_>>(),
        vec![3.0, 2.0, 1.0],
        "{full}"
    );
    assert_eq!(matches[0]["association"]["subject"], json!("x3"));
    assert_eq!(matches[0]["unsourced_count"], json!(1));
    assert_eq!(
        full["dead_concept_aliases"],
        json!({"タカセ": "高瀬"}),
        "{full}"
    );
    assert_eq!(full["dead_label_aliases"], json!({}), "{full}");
    assert_eq!(full["twins"], json!(null), "{full}");

    // A floor past the smallest two drops them, `total` still counts
    // only what clears it.
    let floored = server.ok(
        "POST",
        "/contexts/sake/drift/audit",
        Some(json!({"unsourced_floor": 2.5})),
    );
    assert_eq!(floored["total"], json!(1), "{floored}");
    let floored_matches = floored["unsourced"].as_array().unwrap();
    assert_eq!(floored_matches.len(), 1);
    assert_eq!(floored_matches[0]["association"]["subject"], json!("x3"));

    // limit + after pages the same worst-first order as recall/query.
    let first = server.ok(
        "POST",
        "/contexts/sake/drift/audit",
        Some(json!({"limit": 2})),
    );
    assert_eq!(first["total"], json!(3), "{first}");
    let first_matches = first["unsourced"].as_array().unwrap();
    assert_eq!(first_matches.len(), 2);
    let last = &first_matches[1]["association"];
    let cursor = json!({
        "weight": first_matches[1]["unsourced_weight"],
        "subject": last["subject"], "label": last["label"], "object": last["object"],
    });
    let second = server.ok(
        "POST",
        "/contexts/sake/drift/audit",
        Some(json!({"limit": 2, "after": cursor})),
    );
    assert_eq!(second["total"], json!(3), "{second}");
    let second_matches = second["unsourced"].as_array().unwrap();
    assert_eq!(second_matches.len(), 1);
    assert_eq!(second_matches[0]["association"]["subject"], json!("x1"));

    // include_twins folds in the same vocabulary-audit shape, at the
    // caller's dice_floor.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "株式会社青嶺", "label": "kind", "object": "会社",
             "weight": 1.0, "source": "a.md"},
            {"subject": "青嶺株式会社", "label": "kind", "object": "会社",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
    let with_twins = server.ok(
        "POST",
        "/contexts/sake/drift/audit",
        Some(json!({"include_twins": true, "dice_floor": 0.4})),
    );
    assert!(!with_twins["twins"].is_null(), "{with_twins}");
    assert!(
        with_twins["twins"]["lexical_concepts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|pair| {
                let a = pair["a"].as_str().unwrap();
                let b = pair["b"].as_str().unwrap();
                (a == "株式会社青嶺" || b == "株式会社青嶺") && a != b
            }),
        "{with_twins}"
    );
}

#[test]
fn explore_without_max_depth_stops_at_the_server_ceiling() {
    let server = Server::start("depthcap");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    // A 15-hop chain: c0 → c1 → … → c15.
    let chain: Vec<Value> = (0..15)
        .map(|i| {
            json!({"subject": format!("c{i}"), "label": "next", "object": format!("c{}", i + 1), "weight": 1.0})
        })
        .collect();
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(Value::Array(chain)),
    );

    let walked = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["c0"]})),
    );
    let deepest = walked["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["distance"].as_u64().unwrap())
        .max()
        .unwrap();
    assert_eq!(deepest, 10, "omitted max_depth must stop at the ceiling");
}

#[test]
fn explore_pages_and_keeps_the_closest_past_the_limit() {
    let server = Server::start("explorepage");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    // A hub with four direct neighbours; one leads a hop further to a
    // heavy edge.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "hub", "label": "l", "object": "n1", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "n2", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "n3", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "n4", "weight": 1.0},
            {"subject": "n1", "label": "l", "object": "far", "weight": 9.0},
        ])),
    );

    let walked = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["hub"], "limit": 3})),
    );
    assert_eq!(walked["total"], json!(5));
    let matches = walked["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 3);
    // The cut keeps the closest structure, not the heaviest weight:
    // the distance-2 edge (weight 9.0) is what falls off.
    assert!(
        matches.iter().all(|r| r["distance"] == json!(1)),
        "{walked}"
    );
}

/// A client resumes past `explore`'s limit by building an
/// `ExploreCursor`-shaped `after` from the last match's `distance` and
/// its association's `(subject, label, object)`. The four same-distance
/// neighbours are inserted in a deliberately non-lexicographic order
/// (d, b, a, c) — a retain-only cursor cut (skipping the required
/// re-sort) would keep that exact wrong order instead of a, b, c, d.
#[test]
fn explore_resumes_past_a_cursor_with_same_distance_ties_in_lexicographic_order() {
    let server = Server::start("explore-cursor");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "hub", "label": "l", "object": "d", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "b", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "a", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "c", "weight": 1.0},
        ])),
    );
    let objects_of = |matches: &[Value]| -> Vec<String> {
        matches
            .iter()
            .map(|m| m["association"]["object"].as_str().unwrap().to_string())
            .collect()
    };
    let cursor_from = |m: &Value| {
        json!({
            "distance": m["distance"],
            "subject": m["association"]["subject"],
            "label": m["association"]["label"],
            "object": m["association"]["object"],
        })
    };

    let first = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["hub"], "limit": 2})),
    );
    assert_eq!(first["total"], json!(4), "{first}");
    let matches = first["matches"].as_array().unwrap();
    assert_eq!(objects_of(matches), vec!["a", "b"], "{first}");

    let second = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({
            "origins": ["hub"], "limit": 2,
            "after": cursor_from(&matches[1]),
        })),
    );
    assert_eq!(second["total"], json!(4), "total stays constant: {second}");
    let matches = second["matches"].as_array().unwrap();
    assert_eq!(objects_of(matches), vec!["c", "d"], "{second}");

    let third = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({
            "origins": ["hub"], "limit": 2,
            "after": cursor_from(&matches[1]),
        })),
    );
    assert_eq!(third["matches"], json!([]), "the walk has ended: {third}");

    // A malformed cursor is a 422 (body fails to deserialize), not an
    // empty page.
    let (status, refusal) = server.call(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["hub"], "after": {"bogus": true}})),
    );
    assert_eq!(status, 422, "{refusal}");
}
