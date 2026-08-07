//! Assertion-time windows on the graph lanes (ADR 0011), end to end:
//! `since`/`until` on recall/query/explore/activate serve only what an
//! in-window-dated source attests, with weights and attributions
//! re-derived from the window; an undated source is invisible to every
//! window; the shared window contract refuses `since >= until`; the
//! cross-context lanes refuse a window loudly instead of silently
//! ignoring it; and the retrieval cache keys windowed and unwindowed
//! calls apart.

use serde_json::{Value, json};

use crate::support::*;

/// Two dated regimes of one fact plus an undatable bystander:
/// doc-2019 (date 1000) says the 杜氏 is 高瀬, doc-2024 (date 2000)
/// says it is 青山 and corroborates the 銘柄; doc-undated stores no
/// passage at all, so it has no metadata — not even the `stored_at`
/// the store stamps on passages — and no window can ever see its
/// fact (ADR 0011 §4's rule for associations-only sources).
fn seed(server: &Server) {
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "窓"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "doc-2019"},
            {"subject": "蔵", "label": "銘柄", "object": "青嶺", "weight": 2.0, "source": "doc-2019"},
            {"subject": "蔵", "label": "杜氏", "object": "青山", "weight": 1.0, "source": "doc-2024"},
            {"subject": "蔵", "label": "銘柄", "object": "青嶺", "weight": 4.0, "source": "doc-2024"},
            {"subject": "蔵", "label": "幽霊", "object": "無日付", "weight": 1.0, "source": "doc-undated"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {
                "doc-2019": "杜氏は高瀬。銘柄は青嶺。",
                "doc-2024": "杜氏は青山。銘柄は青嶺。"
            },
            "dates": {"doc-2019": 1000, "doc-2024": 2000}
        })),
    );
}

#[test]
fn windows_filter_reweigh_and_refuse_across_the_graph_lanes() {
    let server = Server::start("temporal-window");
    seed(&server);

    // As-of (until alone): only the 2019 regime existed by t=1500.
    let as_of = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏", "until": 1500})),
    );
    assert_eq!(as_of["total"], json!(1));
    assert_eq!(as_of["matches"][0]["object"], json!("高瀬"));

    // The corroborated fact re-derives weight and citations from the
    // window: in-window sum 2.0 / count 1, and only doc-2019 cited.
    let brand = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "銘柄", "until": 1500})),
    );
    assert_eq!(brand["matches"][0]["weight"], json!(2.0));
    assert_eq!(brand["matches"][0]["count"], json!(1));
    let attributions = brand["matches"][0]["attributions"].as_array().unwrap();
    assert_eq!(attributions.len(), 1);
    assert_eq!(attributions[0]["source"], json!("doc-2019"));
    // Unwindowed, the same edge accumulates both regimes.
    let full = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "銘柄"})),
    );
    assert_eq!(full["matches"][0]["weight"], json!(3.0));
    assert_eq!(full["matches"][0]["count"], json!(2));

    // `since` alone reads "asserted since": the 2024 regime.
    let recent = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "蔵", "since": 1500})),
    );
    assert_eq!(recent["total"], json!(2), "{recent}");
    // The undated source's fact appears in no window at all…
    let ghost = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "幽霊", "since": 0})),
    );
    assert_eq!(ghost["total"], json!(0));
    // …but is served unwindowed, as ever.
    let ghost_full = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "幽霊"})),
    );
    assert_eq!(ghost_full["total"], json!(1));

    // The walks honor the window: activate as-of-1500 ranks the 2019
    // facts and never surfaces the 2024 assertion.
    let activated = server.ok(
        "POST",
        "/contexts/sake/activate",
        Some(json!({"origins": ["蔵"], "decay": 1.0, "until": 1500})),
    );
    assert_eq!(activated["total"], json!(2));
    for hit in activated["matches"].as_array().unwrap() {
        assert_ne!(hit["association"]["object"], json!("青山"), "{activated}");
    }
    let explored = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["蔵"], "until": 1500})),
    );
    assert_eq!(explored["total"], json!(2));

    // The half-open boundary, at the boundary: until == the effective
    // time excludes (upper bound exclusive), since == it includes.
    let at_until = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏", "until": 1000})),
    );
    assert_eq!(at_until["total"], json!(0), "{at_until}");
    let closed = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏", "since": 1000, "until": 1500})),
    );
    assert_eq!(closed["total"], json!(1), "{closed}");
    assert_eq!(closed["matches"][0]["object"], json!("高瀬"));

    // A dateless-but-stored source falls back to stored_at (≈ now):
    // its fact is windowed IN by a since in the past.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵D", "label": "銘柄", "object": "残雪", "weight": 1.0, "source": "doc-memo"}
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"doc-memo": "日付なしのメモ。"}})),
    );
    let by_stored_at = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵D", "since": 2001})),
    );
    assert_eq!(by_stored_at["total"], json!(1), "{by_stored_at}");

    // The shared window contract holds on the graph lanes too.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "since": 2000, "until": 2000})),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("invalid_argument"), "{body}");

    // Cross-context lanes refuse a window loudly — request bodies
    // ignore unknown fields, so silence would mean silently unwindowed
    // results.
    let (status, body) = server.call(
        "POST",
        "/query",
        Some(json!({"contexts": ["sake"], "subject": "蔵", "until": 1500})),
    );
    assert_eq!(status, 400, "{body}");
    let (status, _) = server.call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake"], "cue": "蔵", "since": 1})),
    );
    assert_eq!(status, 400);

    // The MCP lane forwards the window — an advertised parameter the
    // router dropped would serve silently unwindowed results, the
    // exact failure the cross refusal exists to prevent.
    let via_mcp = server.call_tool(
        1,
        "query",
        json!({"context": "sake", "subject": "蔵", "label": "杜氏", "until": 1500}),
    );
    let text: Value =
        serde_json::from_str(via_mcp["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(text["result"]["total"], json!(1), "{text}");
    assert_eq!(text["result"]["matches"][0]["object"], json!("高瀬"));

    // Windowed and unwindowed calls never share a cache entry: the
    // replay of one must not answer the other (both directions).
    let full_again = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏"})),
    );
    assert_eq!(full_again["total"], json!(2));
    let windowed_again = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏", "until": 1500})),
    );
    assert_eq!(windowed_again["total"], json!(1));
}
