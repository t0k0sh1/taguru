//! The /contexts directory listing: paging, pinning, scoped-key filtering.

use serde_json::json;

use crate::support::*;

#[test]
fn the_directory_pages_by_name_and_serves_single_contexts() {
    let server = Server::start("dirpage");
    for name in ["apple", "banana", "cherry"] {
        server.ok(
            "PUT",
            &format!("/contexts/{name}"),
            Some(json!({"description": name})),
        );
    }

    let page = server.ok("GET", "/contexts?limit=2", None);
    assert_eq!(page["total"], json!(3), "total names the full count");
    let names: Vec<&str> = page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context| context["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["apple", "banana"], "name order, first page");

    let page = server.ok("GET", "/contexts?limit=2&after=banana", None);
    assert_eq!(page["total"], json!(3));
    let names: Vec<&str> = page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context| context["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["cherry"], "keyset picks up after the cursor");

    let single = server.ok("GET", "/contexts/banana", None);
    assert_eq!(single["name"], json!("banana"));
    assert_eq!(single["description"], json!("banana"));
    let (status, body) = server.call("GET", "/contexts/nope", None);
    assert_eq!(status, 404);
    assert_eq!(body["status"], json!("error"));
}

/// #62 item 6: `pinned` defines the population of interest, like a
/// search query — so unlike `after`/`limit`, it counts toward `total`.
#[test]
fn the_directory_filters_by_pinned_and_counts_total_after_filtering() {
    let server = Server::start("dirpinned");
    server.ok(
        "PUT",
        "/contexts/apple",
        Some(json!({"description": "a", "pinned": true})),
    );
    server.ok(
        "PUT",
        "/contexts/banana",
        Some(json!({"description": "b", "pinned": false})),
    );
    server.ok(
        "PUT",
        "/contexts/cherry",
        Some(json!({"description": "c", "pinned": true})),
    );

    let pinned = server.ok("GET", "/contexts?pinned=true", None);
    assert_eq!(pinned["total"], json!(2), "{pinned}");
    let names: Vec<&str> = pinned["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context| context["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["apple", "cherry"]);

    let unpinned = server.ok("GET", "/contexts?pinned=false", None);
    assert_eq!(unpinned["total"], json!(1), "{unpinned}");
    assert_eq!(unpinned["contexts"][0]["name"], json!("banana"));

    let all = server.ok("GET", "/contexts", None);
    assert_eq!(all["total"], json!(3), "no filter means every context");
}

/// A context-scoped key's directory listing pages its own allow-list,
/// not the full registry — the allow-list has no relation to name
/// order, so this exercises a different path from the unscoped case
/// above.
#[test]
fn a_scoped_keys_directory_pages_its_allow_list_not_the_full_registry() {
    let server = Server::start_with_env(
        "http-scoped-dirpage",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,curator:ctok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"curator": {"role": "read", "contexts": ["date", "apple", "cherry"]}}"#,
            ),
        ],
    );
    for name in ["apple", "banana", "cherry", "date"] {
        let (status, _) = server.call_with_token(
            "PUT",
            &format!("/contexts/{name}"),
            Some(json!({"description": name})),
            Some("atok"),
        );
        assert_eq!(status, 200);
    }

    // The unscoped admin sees everything.
    let (status, full) = server.call_with_token("GET", "/contexts", None, Some("atok"));
    assert_eq!(status, 200);
    assert_eq!(full["result"]["total"], json!(4), "{full}");

    // The scoped key's world is its three-context grant — "banana"
    // never appears, and `total` counts only the visible set.
    let (status, first) = server.call_with_token("GET", "/contexts?limit=2", None, Some("ctok"));
    assert_eq!(status, 200, "{first}");
    assert_eq!(first["result"]["total"], json!(3), "{first}");
    let names: Vec<&str> = first["result"]["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context| context["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec!["apple", "cherry"],
        "sorted allow-list, first page"
    );

    let (status, second) =
        server.call_with_token("GET", "/contexts?limit=2&after=cherry", None, Some("ctok"));
    assert_eq!(status, 200, "{second}");
    assert_eq!(second["result"]["total"], json!(3));
    let names: Vec<&str> = second["result"]["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context| context["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["date"], "keyset picks up after the cursor");
}
