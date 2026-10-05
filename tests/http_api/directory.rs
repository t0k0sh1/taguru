//! The /contexts directory listing: paging, pinning, context-scoped key
//! filtering.

use serde_json::json;

use crate::support::*;

#[test]
fn the_directory_pages_by_name_and_serves_single_contexts() {
    let server = Server::start("dirpage");
    for name in ["apple", "banana", "cherry"] {
        server.ok(
            "POST",
            "/contexts",
            Some(json!({"name": name, "description": name})),
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

    let single = server.ok("GET", &format!("/contexts/{}", server.cx("banana")), None);
    assert_eq!(single["id"], json!(server.cx("banana")));
    assert_eq!(single["name"], json!("banana"));
    assert_eq!(single["description"], json!("banana"));
    let (status, body) = server.call(
        "GET",
        "/contexts/00000000-0000-4000-8000-00000000dead",
        None,
    );
    assert_eq!(status, 404);
    assert_eq!(body["status"], json!("error"));
}

/// #585 item 4: `?limit=0` must not read as the end of the directory.
/// A page SIZE of zero, unfloored, would `take(0)` into an empty page
/// while `total` still reports every context — indistinguishable from
/// the real end of the directory to an SDK `iter` loop, which stops on
/// the first empty page. `list_contexts` floors it to one instead.
#[test]
fn a_zero_limit_still_returns_one_context_rather_than_reading_as_the_end() {
    let server = Server::start("dirpage-zero-limit");
    for name in ["apple", "banana", "cherry"] {
        server.ok(
            "POST",
            "/contexts",
            Some(json!({"name": name, "description": name})),
        );
    }

    let page = server.ok("GET", "/contexts?limit=0", None);
    assert_eq!(page["total"], json!(3), "{page}");
    let names: Vec<&str> = page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context| context["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["apple"], "floored to one, not zero");
}

/// #62 item 6: `pinned` defines the population of interest, like a
/// search query — so unlike `after`/`limit`, it counts toward `total`.
#[test]
fn the_directory_filters_by_pinned_and_counts_total_after_filtering() {
    let server = Server::start("dirpinned");
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "apple", "description": "a", "pinned": true})),
    );
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "banana", "description": "b", "pinned": false})),
    );
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "cherry", "description": "c", "pinned": true})),
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
/// order, so this exercises a different path from the no-grant-entry
/// case above.
#[test]
fn a_context_scoped_keys_directory_pages_its_allow_list_not_the_full_registry() {
    let grants = format!(
        r#"{{"curator": {{"role": "read", "contexts": ["{}", "{}", "{}"]}}}}"#,
        fixed_id("date"),
        fixed_id("apple"),
        fixed_id("cherry")
    );
    let server = Server::start_with_env(
        "http-scoped-dirpage",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,curator:ctok"),
            ("TAGURU_KEY_GRANTS", grants.as_str()),
        ],
    );
    for name in ["apple", "banana", "cherry", "date"] {
        server.create_fixed_as(name, name, Some("atok"));
    }

    // The admin key with no grant entry sees everything.
    let (status, full) = server.call_with_token("GET", "/contexts", None, Some("atok"));
    assert_eq!(status, 200);
    assert_eq!(full["result"]["total"], json!(4), "{full}");

    // The context-scoped key's world is its three-context grant —
    // "banana" never appears, and `total` counts only the visible set.
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

/// The `(name, id)` keyset cursor (#964): with `after` AND `after_id`,
/// the page resumes strictly past the cursor pair — the cursor row
/// itself must never repeat, and a same-named sibling after it must.
/// Only duplicate names can tell the pair comparison from a plain
/// name comparison, so the fixture creates two.
#[test]
fn the_after_id_cursor_resumes_inside_a_same_named_group_without_repeating() {
    let server = Server::start("dirpage-after-id");
    let first = server.create_context("dup");
    let second = server.create_context("dup");
    let (low, high) = if first < second {
        (first, second)
    } else {
        (second, first)
    };

    let page = server.ok(
        "GET",
        &format!("/contexts?limit=2&after=dup&after_id={low}"),
        None,
    );
    let ids: Vec<&str> = page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        vec![high.as_str()],
        "the cursor row must be excluded, its same-named sibling served: {page}"
    );
}

/// `taguru contexts` (#967): the CLI bridge from a display name to the
/// id every `--context` takes. Names are not unique, so `--name` lists
/// every exact match, and matching is byte-for-byte.
#[test]
fn the_contexts_verb_finds_ids_by_exact_name_and_lists_twins() {
    let server = Server::start("contexts-verb");
    let sake_a = fixed_id("contexts-verb-sake-a");
    let sake_b = fixed_id("contexts-verb-sake-b");
    let beer = server.create_fixed("beer");
    server.create_with_id(&sake_a, json!({"name": "sake", "description": "first"}));
    server.create_with_id(&sake_b, json!({"name": "sake", "description": "second"}));
    let padded = server.create_fixed("Sake ");

    // The whole directory, `ID<TAB>NAME`, in the directory's (name, id) order.
    let (code, stdout, stderr) = run_cli(&["contexts", "--url", &server.base], &[]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let (low, high) = if sake_a < sake_b {
        (&sake_a, &sake_b)
    } else {
        (&sake_b, &sake_a)
    };
    assert_eq!(
        stdout,
        format!("{padded}\tSake \n{beer}\tbeer\n{low}\tsake\n{high}\tsake\n"),
        "{stdout}"
    );

    // `--name` keeps exact matches only — both twins, never `Sake `.
    let (code, stdout, stderr) = run_cli(&["contexts", "--name", "sake", &server.base], &[]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(stdout, format!("{low}\tsake\n{high}\tsake\n"));

    let (code, stdout, stderr) = run_cli(
        &[
            "contexts",
            "--name",
            "beer",
            "--json",
            "--url",
            &server.base,
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let rows: serde_json::Value = serde_json::from_str(&stdout).expect("--json is one document");
    assert_eq!(
        rows,
        json!([{"id": beer, "name": "beer", "description": ""}])
    );

    // No match is a failure the shell can branch on.
    let (code, stdout, stderr) = run_cli(&["contexts", "--name", "ghost", &server.base], &[]);
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(stdout, "");
    assert!(
        stderr.contains("no context is named 'ghost'"),
        "stderr: {stderr}"
    );

    // A usage error is 2, like every other verb.
    let (code, _, stderr) = run_cli(&["contexts", "--bogus"], &[]);
    assert_eq!(code, 2, "stderr: {stderr}");
    assert!(stderr.contains("unknown argument '--bogus'"), "{stderr}");

    // An unusable base URL is a usage mistake (2), not a network failure (1).
    let (code, _, stderr) = run_cli(&["contexts", "--url", "ftp://h"], &[]);
    assert_eq!(code, 2, "stderr: {stderr}");
    assert!(stderr.contains("only supports http/https"), "{stderr}");
    let (code, _, stderr) = run_cli(&["contexts", "--url", "not a url"], &[]);
    assert_eq!(code, 2, "stderr: {stderr}");
    assert!(stderr.contains("is not a usable base URL"), "{stderr}");
}
