//! `POST /recall` (`cross_recall`): fan-out over several contexts at
//! once. Two untested branches from issue #627's audit: `cross_targets`'s
//! upfront existence check aborts naming the LIST-first missing
//! target when several are missing at once (`src/api/recall.rs`'s own
//! `cross_matches_names_a_target_that_does_not_exist` unit test only
//! ever names ONE missing target, via `cross_matches` called
//! directly — it never pins which one is reported when several are
//! missing, nor exercises the HTTP-level `cross_targets` check that
//! runs before `cross_matches` is ever reached), and the mid-loop
//! memory-bound pool cut (`pool.len() >= limit * 2`) still yields the
//! exact global top-`limit` even when it fires before every target has
//! been folded in.

use serde_json::json;

use crate::support::*;

/// `cross_targets`'s own existence check (`src/api/recall.rs`) walks
/// `contexts` with `.iter().find(...)` — a plain, ordered scan, not a
/// concurrent fan-out — so when several named contexts are missing at
/// once, the refusal must always name the FIRST one by request-list
/// order, deterministically. Deleting a context sweeps it out of any
/// group's membership too (`sweep_context_from_groups`, `delete`'s own
/// doc), so a group can never durably name a member that has already
/// vanished; the ordering guarantee this pins is `contexts`'s own
/// list, not a race.
#[test]
fn cross_recall_aborts_naming_the_first_missing_context_by_list_order() {
    let server = Server::start("recall-fanout-abort");
    server.ok("PUT", "/contexts/stays", None);
    server.ok(
        "POST",
        "/contexts/stays/associations",
        Some(json!([
            {"subject": "蔵", "label": "産地", "object": "灘", "weight": 1.0, "source": "a.md"}
        ])),
    );

    let (status, refused) = server.call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["absent-a", "absent-b", "stays"], "cue": "蔵"})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"), "{refused}");
    let message = refused["error"].as_str().unwrap();
    assert!(
        message.contains("absent-a"),
        "must name the list-first target, not the second: {message}"
    );
    assert!(
        !message.contains("absent-b"),
        "must not name the second target instead: {message}"
    );
}

/// The mid-loop memory-bound cut (`src/api/recall.rs`: once the
/// pooled matches reach `limit * 2`, the pool is cut down to the
/// current top-`limit` before the next target's results are even
/// folded in — a streaming top-K, not a single sort at the end).
/// `limit: 5` makes the cut fire after the second of three targets;
/// the third target's weights are deliberately the HIGHEST of all
/// three, so a correct result must show the mid-loop cut did not
/// permanently lock in the first two targets' (weaker) candidates —
/// the final page is exactly that third target's top 5, and `total`
/// still counts every match from all three, cut or not.
#[test]
fn cross_recall_mid_loop_pool_cut_still_yields_the_exact_global_top_limit() {
    let server = Server::start("recall-pool-cut");
    // r1: mid-strength, r2: weak, r3: strongest — target order is the
    // request's own `contexts` list order, not weight order.
    let waves: [(&str, &[f64]); 3] = [
        ("r1", &[10.0, 9.0, 8.0, 7.0, 6.0, 5.0]),
        ("r2", &[4.0, 3.0, 2.0, 1.0, 0.5, 0.4]),
        ("r3", &[20.0, 19.0, 18.0, 17.0, 16.0, 15.0]),
    ];
    for (name, weights) in waves {
        server.ok("PUT", &format!("/contexts/{name}"), None);
        let ops: Vec<serde_json::Value> = weights
            .iter()
            .enumerate()
            .map(|(i, weight)| {
                json!({
                    "subject": "蔵", "label": "産地", "object": format!("{name}-o{i}"),
                    "weight": weight, "source": "a.md",
                })
            })
            .collect();
        server.ok(
            "POST",
            &format!("/contexts/{name}/associations"),
            Some(serde_json::Value::Array(ops)),
        );
    }

    let page = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["r1", "r2", "r3"], "cue": "蔵", "limit": 5})),
    );
    assert_eq!(page["total"], json!(18), "{page}");
    let matches = page["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 5, "{page}");
    for entry in matches {
        assert_eq!(entry["context"], json!("r3"), "{page}");
    }
    let weights: Vec<f64> = matches
        .iter()
        .map(|entry| entry["weight"].as_f64().unwrap())
        .collect();
    assert_eq!(weights, vec![20.0, 19.0, 18.0, 17.0, 16.0], "{page}");
}
