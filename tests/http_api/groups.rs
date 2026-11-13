//! Group CRUD, nesting, membership caps, and restart reconciliation.

use serde_json::{Value, json};

use crate::support::*;

/// The whole group lifecycle over HTTP: create (with and without a
/// body), the keyset-paged directory, single GET, delta PATCH, DELETE —
/// and the namespace split from contexts.
#[test]
fn groups_bundle_contexts_with_crud_paging_and_a_separate_namespace() {
    let server = Server::start("groups-crud");
    for name in ["apple", "banana", "cherry"] {
        server.ok(
            "POST",
            "/contexts",
            Some(json!({"name": name, "description": name})),
        );
    }

    // POST mints the id and answers the new row (#995).
    let created = server.ok(
        "POST",
        "/groups",
        group_body(
            "fruit",
            Some(json!({"description": "果物の文脈", "context_ids": [server.cx("banana"), server.cx("apple")]})),
        ),
    );
    let fruit = created["id"]
        .as_str()
        .expect("the row carries its id")
        .to_string();
    assert_eq!(created["name"], json!("fruit"));
    assert_eq!(
        created["context_ids"],
        server.cx_sorted(&["apple", "banana"])
    );
    assert_eq!(fruit, server.gx("fruit"), "the id is the file's stem");
    // A create is no longer keyed by name: the same name again mints a
    // second group; the caller's own id, repeated with the same
    // record, answers the standing row instead (a retry converges),
    // and with a different record is refused.
    let twin = server.create_group("fruit", None);
    assert_ne!(twin, fruit);
    let pinned = "00000000-0000-4000-9000-0000000000f1";
    let mut body = group_body("pinned", None).unwrap();
    body["id"] = json!(pinned);
    let first = server.ok("POST", "/groups", Some(body.clone()));
    assert_eq!(first["id"], json!(pinned));
    assert_eq!(server.ok("POST", "/groups", Some(body.clone())), first);
    body["description"] = json!("different");
    let (status, taken) = server.call("POST", "/groups", Some(body));
    assert_eq!(status, 409, "{taken}");
    assert_eq!(taken["code"], json!("already_exists"));
    // An id that is not a canonical UUID is a 400 — in the body and in
    // the path alike — never a lookup.
    for id in ["fruit", "00000000-0000-4000-9000-0000000000F1"] {
        let mut body = group_body("x", None).unwrap();
        body["id"] = json!(id);
        let (status, refused) = server.call("POST", "/groups", Some(body));
        assert_eq!(status, 400, "{id}: {refused}");
        let (status, refused) = server.call("GET", &format!("/groups/{id}"), None);
        assert_eq!(status, 400, "{id}: {refused}");
        assert!(
            refused["error"]
                .as_str()
                .unwrap()
                .contains("is not a group id"),
            "{refused}"
        );
    }
    // A create needs a name.
    let (status, nameless) = server.call("POST", "/groups", None);
    assert_eq!(status, 400, "{nameless}");
    for id in [twin.as_str(), pinned] {
        server.ok("DELETE", &format!("/groups/{id}"), None);
    }
    // A body with only the name is a valid create (defaults) — an empty
    // group is how "create first, fill later" starts.
    let empty = server.create_group("empty", None);

    // The directory pages by (name, id), `total` cursor-independent,
    // exactly like /contexts.
    let page = server.ok("GET", "/groups", None);
    assert_eq!(page["total"], json!(2), "{page}");
    let names: Vec<&str> = page["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|group| group["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["empty", "fruit"], "name order");
    assert_eq!(page["groups"][0]["id"], json!(empty));
    assert_eq!(
        page["groups"][1]["context_ids"],
        server.cx_sorted(&["apple", "banana"]),
        "members come back sorted: {page}"
    );
    let page = server.ok("GET", "/groups?limit=1", None);
    assert_eq!(page["total"], json!(2));
    assert_eq!(page["groups"][0]["id"], json!(empty));
    let page = server.ok("GET", "/groups?limit=1&after=empty", None);
    assert_eq!(page["groups"][0]["id"], json!(fruit));
    let page = server.ok(
        "GET",
        &format!("/groups?limit=1&after=empty&after_id={empty}"),
        None,
    );
    assert_eq!(page["groups"][0]["id"], json!(fruit));
    let (status, alone) = server.call("GET", &format!("/groups?after_id={empty}"), None);
    assert_eq!(status, 400, "after_id needs after: {alone}");

    let single = server.ok("GET", &format!("/groups/{fruit}"), None);
    assert_eq!(single["description"], json!("果物の文脈"));
    assert_eq!(single["name"], json!("fruit"));
    let (status, missing) = server.call("GET", &format!("/groups/{GHOST_GROUP_ID}"), None);
    assert_eq!(status, 404);
    assert_eq!(missing["code"], json!("no_group"));

    // PATCH applies deltas — removals first, then adds — and answers
    // the updated row. Removing a non-member is an idempotent no-op.
    let updated = server.ok(
        "PATCH",
        &format!("/groups/{fruit}"),
        Some(
            json!({"description": "更新後", "add_context_ids": [server.cx("cherry")],
                    "remove_context_ids": [server.cx("apple"), GHOST_ID]}),
        ),
    );
    assert_eq!(updated["description"], json!("更新後"));
    assert_eq!(
        updated["context_ids"],
        server.cx_sorted(&["banana", "cherry"])
    );
    let (status, missing) = server.call(
        "PATCH",
        &format!("/groups/{GHOST_GROUP_ID}"),
        Some(json!({"description": "x"})),
    );
    assert_eq!(status, 404, "{missing}");
    assert_eq!(missing["code"], json!("no_group"));

    // Groups and contexts are separate namespaces: one name, both kinds.
    let apple = server.create_group("apple", Some(json!({"description": "同名"})));
    assert_eq!(
        server
            .call("GET", &format!("/contexts/{}", server.cx("apple")), None)
            .0,
        200
    );
    assert_eq!(server.call("GET", &format!("/groups/{apple}"), None).0, 200);

    // DELETE removes the bundling alone; the members live on.
    assert_eq!(
        server.ok("DELETE", &format!("/groups/{fruit}"), None),
        json!(true)
    );
    assert_eq!(server.call("GET", &format!("/groups/{fruit}"), None).0, 404);
    for name in ["banana", "cherry"] {
        assert_eq!(
            server
                .call("GET", &format!("/contexts/{}", server.cx(name)), None)
                .0,
            200
        );
    }
    let (status, gone) = server.call("DELETE", &format!("/groups/{fruit}"), None);
    assert_eq!(status, 404, "{gone}");
    assert_eq!(gone["code"], json!("no_group"));
}

/// Strict referential integrity: an add never dangles (no_context, and
/// NOTHING applies), and deleting a context sweeps it out of every
/// group immediately.
#[test]
fn group_membership_is_strict_and_context_deletion_sweeps() {
    let server = Server::start("groups-strict");
    // A create naming a missing context refuses whole: no group.
    let (status, refused) = server.call(
        "POST",
        "/groups",
        group_body("g", Some(json!({"context_ids": [GHOST_ID]}))),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"));
    assert_eq!(server.ok("GET", "/groups", None)["total"], json!(0));

    server.ok("POST", "/contexts", Some(json!({"name": "a"})));
    server.ok("POST", "/contexts", Some(json!({"name": "b"})));
    server.ok(
        "POST",
        "/groups",
        group_body(
            "g",
            Some(json!({"context_ids": [server.cx("a"), server.cx("b")]})),
        ),
    );

    // An add naming a missing context refuses whole: membership as was.
    let (status, refused) = server.call(
        "PATCH",
        &format!("/groups/{}", server.gx("g")),
        Some(json!({"add_context_ids": [GHOST_ID], "remove_context_ids": [server.cx("a")]})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"));
    assert_eq!(
        server.ok("GET", &format!("/groups/{}", server.gx("g")), None)["context_ids"],
        server.cx_sorted(&["a", "b"]),
        "the refused delta must not half-apply"
    );

    // Deleting a member context drops it from the group, immediately.
    server.ok("DELETE", &format!("/contexts/{}", server.cx("a")), None);
    assert_eq!(
        server.ok("GET", &format!("/groups/{}", server.gx("g")), None)["context_ids"],
        server.cx_sorted(&["b"])
    );

    // The write-boundary caps hold for groups too.
    let long_name = "x".repeat(65);
    let (status, oversized) = server.call("POST", "/groups", group_body(&long_name, None));
    assert_eq!(status, 400, "{oversized}");
    assert_eq!(oversized["code"], json!("invalid_argument"));
    let (status, oversized) = server.call(
        "POST",
        "/groups",
        group_body("big", Some(json!({"description": "x".repeat(5000)}))),
    );
    assert_eq!(status, 400, "{oversized}");
    let over_cap: Vec<String> = (0..1001)
        .map(|i| format!("00000000-0000-4000-8000-{i:012x}"))
        .collect();
    let (status, refused) = server.call(
        "POST",
        "/groups",
        group_body("big", Some(json!({"context_ids": over_cap}))),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"));
}

/// Membership is capped in TOTAL, not just per request: deltas cannot
/// grow a group past 1,000 names per set. The cap is judged on the
/// delta's result, before existence — so these ghosts never need to
/// exist to hear it — and removals apply first, making room in the
/// same request.
#[test]
fn group_membership_cannot_be_grown_past_the_cap_by_deltas() {
    let server = Server::start("groups-total-cap");
    server.ok("POST", "/contexts", Some(json!({"name": "a"})));
    server.ok(
        "POST",
        "/groups",
        group_body("g", Some(json!({"context_ids": [server.cx("a")]}))),
    );
    server.ok("POST", "/groups", group_body("kid", None));
    server.ok(
        "PATCH",
        &format!("/groups/{}", server.gx("g")),
        Some(json!({"add_group_ids": [server.gx("kid")]})),
    );

    // 1 member + 1,000 adds = one past the cap: refused whole.
    let ghosts: Vec<String> = (0..1000)
        .map(|i| format!("00000000-0000-4000-8000-{i:012x}"))
        .collect();
    let (status, refused) = server.call(
        "PATCH",
        &format!("/groups/{}", server.gx("g")),
        Some(json!({"add_context_ids": ghosts})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"), "{refused}");
    // Child groups ride the same cap on their own set.
    let ghost_kids: Vec<String> = (0..1000)
        .map(|i| format!("00000000-0000-4000-9000-{i:012x}"))
        .collect();
    let (status, refused) = server.call(
        "PATCH",
        &format!("/groups/{}", server.gx("g")),
        Some(json!({"add_group_ids": ghost_kids})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"), "{refused}");

    // Removing the member in the same request makes room — the cap
    // passes and the EXISTENCE gate answers next, proving the cap is
    // judged on the result and before existence.
    let ghosts: Vec<String> = (0..1000)
        .map(|i| format!("00000000-0000-4000-8000-{i:012x}"))
        .collect();
    let (status, refused) = server.call(
        "PATCH",
        &format!("/groups/{}", server.gx("g")),
        Some(json!({"add_context_ids": ghosts, "remove_context_ids": [server.cx("a")]})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"), "{refused}");

    // Nothing half-applied anywhere along the way.
    let row = server.ok("GET", &format!("/groups/{}", server.gx("g")), None);
    assert_eq!(row["context_ids"], server.cx_sorted(&["a"]), "{row}");
    assert_eq!(row["group_ids"], json!([server.gx("kid")]), "{row}");
}

/// Nesting: a group may hold child groups — at most three storeys,
/// never a cycle, children must exist — child deltas patch like
/// context deltas, and deleting a child sweeps it out of every parent.
#[test]
fn groups_nest_with_a_depth_cap_and_no_cycles() {
    let server = Server::start("groups-nesting");
    for name in ["a", "b"] {
        server.ok("POST", "/contexts", Some(json!({"name": name})));
    }
    server.ok(
        "POST",
        "/groups",
        group_body("leaf", Some(json!({"context_ids": [server.cx("a")]}))),
    );
    server.ok(
        "POST",
        "/groups",
        group_body(
            "mid",
            Some(json!({"group_ids": [server.gx("leaf")], "context_ids": [server.cx("b")]})),
        ),
    );
    server.ok(
        "POST",
        "/groups",
        group_body("top", Some(json!({"group_ids": [server.gx("mid")]}))),
    );

    // Rows carry their children; members stay the direct ones.
    let row = server.ok("GET", &format!("/groups/{}", server.gx("mid")), None);
    assert_eq!(row["group_ids"], json!([server.gx("leaf")]), "{row}");
    assert_eq!(row["context_ids"], server.cx_sorted(&["b"]));
    let page = server.ok("GET", "/groups", None);
    assert_eq!(page["groups"][2]["name"], json!("top"), "{page}");
    assert_eq!(page["groups"][2]["group_ids"], json!([server.gx("mid")]));

    // A fourth storey refuses as a cap, a cycle (the self-loop
    // included) as a bad argument, an unknown child in the group
    // namespace's own 404 — and nothing half-applies.
    let (status, refused) = server.call(
        "POST",
        "/groups",
        group_body("over", Some(json!({"group_ids": [server.gx("top")]}))),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"));
    assert_eq!(
        server.ok("GET", "/groups", None)["total"],
        json!(3),
        "nothing landed"
    );
    let (status, refused) = server.call(
        "PATCH",
        &format!("/groups/{}", server.gx("leaf")),
        Some(json!({"add_group_ids": [server.gx("top")]})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("invalid_argument"));
    let (status, refused) = server.call(
        "PATCH",
        &format!("/groups/{}", server.gx("leaf")),
        Some(json!({"add_group_ids": [server.gx("leaf")]})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("invalid_argument"));
    let (status, refused) = server.call(
        "PATCH",
        &format!("/groups/{}", server.gx("top")),
        Some(json!({"add_group_ids": ["ead6ef03-d61e-460c-933d-6d450c50a1e5"], "remove_group_ids": [server.gx("mid")]})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_group"));
    assert_eq!(
        server.ok("GET", &format!("/groups/{}", server.gx("top")), None)["group_ids"],
        json!([server.gx("mid")]),
        "the refused delta must not half-apply"
    );

    // Child deltas move like context deltas — and a child may sit
    // under two parents at once: the shape is a DAG, not a tree.
    let updated = server.ok(
        "PATCH",
        &format!("/groups/{}", server.gx("top")),
        Some(json!({"add_group_ids": [server.gx("leaf")], "remove_group_ids": [server.gx("mid")]})),
    );
    assert_eq!(updated["group_ids"], json!([server.gx("leaf")]));
    assert_eq!(
        server.ok("GET", &format!("/groups/{}", server.gx("mid")), None)["group_ids"],
        json!([server.gx("leaf")])
    );

    // Deleting a child sweeps it from every parent; its member
    // contexts live on untouched.
    server.ok("DELETE", &format!("/groups/{}", server.gx("leaf")), None);
    assert_eq!(
        server.ok("GET", &format!("/groups/{}", server.gx("top")), None)["group_ids"],
        json!([])
    );
    assert_eq!(
        server.ok("GET", &format!("/groups/{}", server.gx("mid")), None)["group_ids"],
        json!([])
    );
    assert_eq!(
        server
            .call("GET", &format!("/contexts/{}", server.cx("a")), None)
            .0,
        200
    );

    // The children list rides the same input ceiling as contexts.
    let over_cap: Vec<String> = (0..1001)
        .map(|i| format!("00000000-0000-4000-9000-{i:012x}"))
        .collect();
    let (status, refused) = server.call(
        "POST",
        "/groups",
        group_body("wide", Some(json!({"group_ids": over_cap}))),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"));
}

/// Groups persist across a restart, and boot reconciliation drops any
/// dangling member a crash could have left in a group file.
#[test]
fn groups_survive_restart_and_boot_reconciles_dangling_members() {
    let server = Server::start("groups-restart");
    server.ok("POST", "/contexts", Some(json!({"name": "sake"})));
    server.ok(
        "POST",
        "/groups",
        group_body(
            "drinks",
            Some(json!({"description": "飲料", "context_ids": [server.cx("sake")]})),
        ),
    );

    let sake = server.cx("sake");
    let drinks = server.group_stem("drinks");
    let data_dir = server.stop_gracefully();
    // Plant a dangling member and a dangling child the way a crash
    // between a deletion and the sweep's rewrite would: straight into
    // the file (children are ids on file since #994).
    std::fs::write(
        data_dir.join(format!("{drinks}.group")),
        serde_json::to_vec(
            &json!({"id": drinks, "name": "drinks", "description": "飲料",
                                   "context_ids": [sake, GHOST_ID],
                                   "group_ids": [GHOST_GROUP_ID]}),
        )
        .unwrap(),
    )
    .unwrap();

    let server = Server::start_on("groups-restart2", data_dir);
    let survived = server.ok("GET", &format!("/groups/{}", server.gx("drinks")), None);
    assert_eq!(survived["description"], json!("飲料"));
    assert_eq!(
        survived["context_ids"],
        server.cx_sorted(&["sake"]),
        "boot must reconcile the planted dangling member: {survived}"
    );
    assert_eq!(
        survived["group_ids"],
        json!([]),
        "boot must reconcile the planted dangling child: {survived}"
    );
    // And the fix reached the file, not just memory.
    let on_disk = std::fs::read_to_string(server.data_dir.join(format!("{drinks}.group"))).unwrap();
    assert!(!on_disk.contains(GHOST_ID), "{on_disk}");
    assert!(!on_disk.contains(GHOST_GROUP_ID), "{on_disk}");
}

/// The grant story for groups: every key sees every row but only its
/// granted members; a write touching any context beyond the grant —
/// current members included — refuses whole, and out-of-scope names
/// answer the same 403 whether or not they exist (no existence oracle).
#[test]
fn key_grants_filter_group_members_and_gate_group_writes() {
    let grants = format!(
        r#"{{"reader": "read", "potter": {{"role": "write", "contexts": ["{}"]}}}}"#,
        fixed_id("sake")
    );
    let server = Server::start_with_env(
        "groups-grants",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,reader:rtok,potter:stok"),
            ("TAGURU_KEY_GRANTS", grants.as_str()),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    for context in ["sake", "bunko"] {
        server.create_fixed_as(context, "", Some("atok"));
    }
    assert_eq!(
        call(
            "POST",
            "/groups",
            group_body(
                "mixed",
                Some(json!({"context_ids": [server.cx("sake"), server.cx("bunko")]}))
            ),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "POST",
            "/groups",
            group_body("ours", Some(json!({"context_ids": [server.cx("sake")]}))),
            "atok"
        )
        .0,
        200
    );

    // Reads: every row is visible (groups are labels, not content),
    // but a context-scoped key sees only its granted members.
    let (status, listed) = call("GET", "/groups", None, "stok");
    assert_eq!(status, 200);
    assert_eq!(listed["result"]["total"], json!(2), "{listed}");
    assert_eq!(
        listed["result"]["groups"][0]["name"],
        json!("mixed"),
        "{listed}"
    );
    assert_eq!(
        listed["result"]["groups"][0]["context_ids"],
        server.cx_sorted(&["sake"]),
        "bunko must be filtered from a sake-scoped view: {listed}"
    );
    let (_, single) = call(
        "GET",
        &format!("/groups/{}", server.gx("mixed")),
        None,
        "stok",
    );
    assert_eq!(
        single["result"]["context_ids"],
        server.cx_sorted(&["sake"]),
        "{single}"
    );

    // Role gate: read keys read, nothing more.
    let (status, refused) = call("POST", "/groups", group_body("new", None), "rtok");
    assert_eq!(status, 403, "{refused}");
    assert!(
        refused["error"].as_str().unwrap().contains("needs 'write'"),
        "{refused}"
    );

    // Writes judge every involved context — current members included.
    // Touching a group with an out-of-grant member refuses whole, even
    // for a description-only change.
    let (status, refused) = call(
        "PATCH",
        &format!("/groups/{}", server.gx("mixed")),
        Some(json!({"description": "mine now"})),
        "stok",
    );
    assert_eq!(status, 403, "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains(&format!("no grant on context '{}'", server.cx("bunko"))),
        "{refused}"
    );

    // The oracle stays shut: an out-of-scope name answers the same 403
    // whether it exists (bunko) or not (ghost) — never a revealing 404.
    let (status_real, real) = call(
        "PATCH",
        &format!("/groups/{}", server.gx("ours")),
        Some(json!({"add_context_ids": [server.cx("bunko")]})),
        "stok",
    );
    let (status_ghost, ghost) = call(
        "PATCH",
        &format!("/groups/{}", server.gx("ours")),
        Some(json!({"add_context_ids": [GHOST_ID]})),
        "stok",
    );
    assert_eq!((status_real, status_ghost), (403, 403), "{real} / {ghost}");
    assert_eq!(real["code"], ghost["code"], "{real} / {ghost}");
    let (_, ours) = call(
        "GET",
        &format!("/groups/{}", server.gx("ours")),
        None,
        "atok",
    );
    assert_eq!(
        ours["result"]["context_ids"],
        server.cx_sorted(&["sake"]),
        "the refused adds must not have applied: {ours}"
    );

    // Inside the grant, a context-scoped writer works normally...
    assert_eq!(
        call(
            "PATCH",
            &format!("/groups/{}", server.gx("ours")),
            Some(json!({"description": "陶工の棚"})),
            "stok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "POST",
            "/groups",
            group_body("mine", Some(json!({"context_ids": [server.cx("sake")]}))),
            "stok"
        )
        .0,
        200
    );
    // ...but deletion is an operator verb (admin), like contexts.
    assert_eq!(
        call(
            "DELETE",
            &format!("/groups/{}", server.gx("mine")),
            None,
            "stok"
        )
        .0,
        403
    );
    assert_eq!(
        call(
            "DELETE",
            &format!("/groups/{}", server.gx("mine")),
            None,
            "atok"
        )
        .0,
        200
    );

    // Nesting counts through: a child whose members sit beyond the
    // grant poisons every write on the parent — the child's NAME stays
    // visible (labels, not content), but its contexts are what a grant
    // is about, wherever they hang.
    assert_eq!(
        call(
            "POST",
            "/groups",
            group_body("shelf", Some(json!({"context_ids": [server.cx("bunko")]}))),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PATCH",
            &format!("/groups/{}", server.gx("ours")),
            Some(json!({"add_group_ids": [server.gx("shelf")]})),
            "atok"
        )
        .0,
        200
    );
    let (_, nested) = call(
        "GET",
        &format!("/groups/{}", server.gx("ours")),
        None,
        "stok",
    );
    assert_eq!(
        nested["result"]["group_ids"],
        json!([server.gx("shelf")]),
        "{nested}"
    );
    assert_eq!(
        nested["result"]["context_ids"],
        server.cx_sorted(&["sake"]),
        "{nested}"
    );
    let (status, refused) = call(
        "PATCH",
        &format!("/groups/{}", server.gx("ours")),
        Some(json!({"description": "still mine"})),
        "stok",
    );
    assert_eq!(status, 403, "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains(&format!("no grant on context '{}'", server.cx("bunko"))),
        "{refused}"
    );
    // Naming such a child in a delta refuses the same way — a
    // context-scoped key cannot create a parent over contexts it has
    // no grant on.
    let (status, refused) = call(
        "POST",
        "/groups",
        group_body("annex", Some(json!({"group_ids": [server.gx("shelf")]}))),
        "stok",
    );
    assert_eq!(status, 403, "{refused}");
    assert!(
        server.try_gx("annex").is_none(),
        "the refused create landed nothing"
    );
}

/// The changelog's other stated limitation, pinned: a DELETE whose
/// unlink fails answers 500, says the group will reappear, drops it
/// from the live directory — and the next boot really does resurface
/// it, because the file survived as the on-disk truth.
#[cfg(unix)]
#[test]
fn a_failed_group_unlink_resurfaces_the_group_at_restart() {
    use std::os::unix::fs::PermissionsExt;

    let server = Server::start("group-unlink-fail");
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "sake", "description": "d"})),
    );
    server.ok(
        "POST",
        "/groups",
        group_body(
            "kura",
            Some(json!({"description": "蔵元一式", "context_ids": [server.cx("sake")]})),
        ),
    );
    // Nothing dirty may remain: the flusher must not collide with the
    // read-only window below.
    server.ok("POST", "/flush", None);

    // Unlink needs write permission on the PARENT directory; freezing
    // the data directory makes exactly the unlink fail.
    let kura_stem = server.group_stem("kura");
    let frozen = std::fs::Permissions::from_mode(0o555);
    std::fs::set_permissions(&server.data_dir, frozen).expect("chmod must apply");
    let (status, refusal) = server.call("DELETE", &format!("/groups/{}", server.gx("kura")), None);
    let restored = std::fs::Permissions::from_mode(0o755);
    std::fs::set_permissions(&server.data_dir, restored).expect("chmod must restore");
    assert_eq!(status, 500, "{refusal}");
    assert_eq!(refusal["code"], json!("internal"), "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("reappears at the next restart"),
        "{refusal}"
    );

    // The live directory dropped the record, as the error admits…
    let (status, missing) = server.call("GET", &format!("/groups/{}", server.gx("kura")), None);
    assert_eq!(status, 404, "{missing}");

    // …and the surviving file resurfaces it at the next boot.
    let data_dir = server.stop_gracefully();
    assert!(
        data_dir.join(format!("{kura_stem}.group")).exists(),
        "the unlink must have failed"
    );
    let server = Server::start_on("group-unlink-reboot", data_dir);
    let row = server.ok("GET", &format!("/groups/{}", server.gx("kura")), None);
    assert_eq!(row["context_ids"], server.cx_sorted(&["sake"]), "{row}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// Context rows carry `revision` counters and group rows a
/// `fingerprint` over the transitive members' revisions — the change
/// tokens the retrieval caches key on (#149). The fingerprint moves
/// exactly when a visible member's revision does — direct or through a
/// nested child, on any lane, membership edits included — and reads
/// move nothing.
#[test]
fn revision_and_group_fingerprint_move_exactly_with_member_changes() {
    let server = Server::start("group-fingerprint");
    for name in ["apple", "banana", "outside"] {
        server.ok("POST", "/contexts", Some(json!({"name": name})));
    }
    let row = server.ok("GET", &format!("/contexts/{}", server.cx("apple")), None);
    assert_eq!(
        row["revision"],
        json!({"graph": 0, "passages": 0, "config": 0}),
        "{row}"
    );

    // parent holds apple directly and banana only through a child.
    server.ok(
        "POST",
        "/groups",
        group_body("child", Some(json!({"context_ids": [server.cx("banana")]}))),
    );
    server.ok(
        "POST",
        "/groups",
        group_body(
            "parent",
            Some(json!({"context_ids": [server.cx("apple")], "group_ids": [server.gx("child")]})),
        ),
    );
    let fingerprint = |server: &Server| {
        server.ok("GET", &format!("/groups/{}", server.gx("parent")), None)["fingerprint"]
            .as_str()
            .expect("every group row carries the token")
            .to_string()
    };
    let start = fingerprint(&server);
    assert_eq!(start.len(), 16, "one 64-bit token in hex: {start}");

    // Reads change nothing.
    server.ok("GET", "/contexts", None);
    server.ok(
        "POST",
        &format!("/contexts/{}/query", server.cx("apple")),
        Some(json!({"subject": "蔵"})),
    );
    assert_eq!(fingerprint(&server), start);

    // A graph write to a direct member moves it…
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("apple")),
        Some(json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0}])),
    );
    let after_direct = fingerprint(&server);
    assert_ne!(after_direct, start);

    // …a passage write to a member reached only through the child too…
    server.ok(
        "POST",
        &format!("/contexts/{}/sources", server.cx("banana")),
        Some(json!({"passages": {"doc": "バナナの原文。"}})),
    );
    let after_nested = fingerprint(&server);
    assert_ne!(after_nested, after_direct);

    // …while the same writes to a non-member change nothing.
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("outside")),
        Some(json!([{"subject": "a", "label": "b", "object": "c", "weight": 1.0}])),
    );
    assert_eq!(fingerprint(&server), after_nested);

    // The config lane (a floor edit) counts as a member change too.
    server.ok(
        "PATCH",
        &format!("/contexts/{}", server.cx("apple")),
        Some(json!({"semantic_floor": 0.5})),
    );
    let after_config = fingerprint(&server);
    assert_ne!(after_config, after_nested);

    // A membership edit moves it with no member written at all.
    server.ok(
        "PATCH",
        &format!("/groups/{}", server.gx("parent")),
        Some(json!({"add_context_ids": [server.cx("outside")]})),
    );
    let after_membership = fingerprint(&server);
    assert_ne!(after_membership, after_config);

    // The counters ride the directory listing row for row.
    let listing = server.ok("GET", "/contexts", None);
    let apple = listing["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["name"] == json!("apple"))
        .unwrap();
    assert_eq!(apple["revision"]["graph"], json!(1), "{apple}");
    assert_eq!(apple["revision"]["config"], json!(1), "{apple}");
    let banana = listing["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["name"] == json!("banana"))
        .unwrap();
    assert_eq!(banana["revision"]["passages"], json!(1), "{banana}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// Members are ids (#965): a name — in a create, in either PATCH list, or
/// in any of the three cross-search routes — is a 400 that says so, and
/// refuses before anything is looked up or applied.
#[test]
fn a_context_name_where_an_id_belongs_is_a_400_everywhere_members_are_listed() {
    let server = Server::start("groups-names-refused");
    server.ok("POST", "/contexts", Some(json!({"name": "sake"})));
    server.ok(
        "POST",
        "/groups",
        group_body("g", Some(json!({"context_ids": [server.cx("sake")]}))),
    );

    let refused_with_a_name = |method: &str, path: &str, body: Value, field: &str| {
        let (status, refused) = server.call(method, path, Some(body));
        assert_eq!(status, 400, "{path} {field}: {refused}");
        assert_eq!(refused["code"], json!("invalid_argument"), "{refused}");
        let error = refused["error"].as_str().unwrap();
        assert!(
            error.contains(field) && error.contains("'sake' is not a context id"),
            "{path}: {error}"
        );
    };
    refused_with_a_name(
        "POST",
        "/groups",
        group_body("named", Some(json!({"context_ids": ["sake"]}))).unwrap(),
        "context_ids",
    );
    refused_with_a_name(
        "PATCH",
        &format!("/groups/{}", server.gx("g")),
        json!({"add_context_ids": ["sake"]}),
        "add_context_ids",
    );
    refused_with_a_name(
        "PATCH",
        &format!("/groups/{}", server.gx("g")),
        json!({"remove_context_ids": ["sake"]}),
        "remove_context_ids",
    );
    refused_with_a_name(
        "POST",
        "/recall",
        json!({"context_ids": ["sake"], "cue": "蔵"}),
        "context_ids",
    );
    refused_with_a_name(
        "POST",
        "/query",
        json!({"context_ids": ["sake"], "label": "l"}),
        "context_ids",
    );
    refused_with_a_name(
        "POST",
        "/sources/search",
        json!({"context_ids": ["sake"], "query": "蔵"}),
        "context_ids",
    );

    // Nothing applied: no `named` group, `g` unchanged.
    assert!(server.try_gx("named").is_none());
    assert_eq!(
        server.ok("GET", &format!("/groups/{}", server.gx("g")), None)["context_ids"],
        server.cx_sorted(&["sake"])
    );
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// A rename touches no group: membership names ids, which a rename never
/// changes — and a cross-search served after the rename (and after an
/// identical one that filled the retrieval cache under the old name)
/// answers with the NEW display name, beside the unchanged id.
#[test]
fn renaming_a_context_leaves_its_groups_alone_and_cross_search_shows_the_new_name() {
    let server = Server::start("groups-rename-by-id");
    server.ok("POST", "/contexts", Some(json!({"name": "before"})));
    let id = server.cx("before");
    server.ok(
        "POST",
        &format!("/contexts/{id}/associations"),
        Some(json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬",
                     "weight": 1.0, "source": "a.md"}])),
    );
    server.ok(
        "POST",
        "/groups",
        group_body("g", Some(json!({"context_ids": [id.clone()]}))),
    );
    let fingerprint =
        server.ok("GET", &format!("/groups/{}", server.gx("g")), None)["fingerprint"].clone();

    let recall = || {
        server.ok(
            "POST",
            "/recall",
            Some(json!({"context_ids": [id.clone()], "cue": "蔵"})),
        )
    };
    let before = recall();
    assert_eq!(before["matches"][0]["context_id"], json!(id), "{before}");
    assert_eq!(before["matches"][0]["context_name"], json!("before"));
    assert_eq!(recall(), before, "an identical recall is stable (cached)");

    server.ok(
        "POST",
        &format!("/contexts/{id}/rename"),
        Some(json!({"to": "after"})),
    );

    let group = server.ok("GET", &format!("/groups/{}", server.gx("g")), None);
    assert_eq!(group["context_ids"], json!([id.clone()]), "{group}");
    assert_eq!(
        group["fingerprint"], fingerprint,
        "a rename moves no revision counter, so the group's fingerprint stays"
    );
    let after = recall();
    assert_eq!(after["matches"][0]["context_id"], json!(id), "{after}");
    assert_eq!(
        after["matches"][0]["context_name"],
        json!("after"),
        "the cached page must not serve the old display name: {after}"
    );
    let via_group = server.ok(
        "POST",
        "/recall",
        Some(json!({"group_ids": [server.gx("g")], "cue": "蔵"})),
    );
    assert_eq!(via_group["matches"][0]["context_name"], json!("after"));
    assert_eq!(via_group["plan"]["context_ids"], json!([id]), "{via_group}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// Names are labels (ADR 0050): two `groups` may share one, each
/// addressed by its own id — in the path (#995) and in every list that
/// names a `group` (#996).
#[test]
fn groups_sharing_a_name_are_two_groups_at_every_entrance() {
    let server = Server::start("groups-ambiguous");
    let sake = server.create_context("sake");
    let first = server.create_group("kura", Some(json!({"context_ids": [sake]})));
    let second = server.create_group("kura", Some(json!({"description": "twin"})));
    assert_ne!(
        first, second,
        "a second create under one name mints a second group"
    );
    let page = server.ok("GET", "/groups", None);
    assert_eq!(page["total"], json!(2), "{page}");
    // Name first, then id — the twins sit side by side in id order.
    let mut ids = [first.clone(), second.clone()];
    ids.sort();
    assert_eq!(
        page["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].clone())
            .collect::<Vec<_>>(),
        [json!(ids[0]), json!(ids[1])],
        "{page}"
    );
    // A page boundary between the twins: the (name, id) cursor resumes
    // after the first, never skipping the second.
    let head = server.ok("GET", "/groups?limit=1", None);
    assert_eq!(head["groups"][0]["id"], json!(ids[0]), "{head}");
    let tail = server.ok(
        "GET",
        &format!("/groups?limit=1&after=kura&after_id={}", ids[0]),
        None,
    );
    assert_eq!(tail["groups"][0]["id"], json!(ids[1]), "{tail}");
    let end = server.ok(
        "GET",
        &format!("/groups?limit=1&after=kura&after_id={}", ids[1]),
        None,
    );
    assert_eq!(end["groups"], json!([]), "{end}");
    // By id, each twin answers on its own.
    assert_eq!(
        server.ok("GET", &format!("/groups/{first}"), None)["context_ids"],
        json!([sake])
    );
    assert_eq!(
        server.ok("GET", &format!("/groups/{second}"), None)["description"],
        json!("twin")
    );
    // Every list names a group by id (#996), so the twins are two
    // groups everywhere: nested, searched and removed one at a time, and
    // a name where an id belongs is a 400, never a lookup.
    let parent = server.create_group(
        "parent",
        Some(json!({"group_ids": [first.clone(), second.clone()]})),
    );
    let mut both = vec![first.clone(), second.clone()];
    both.sort();
    assert_eq!(
        server.ok("GET", &format!("/groups/{parent}"), None)["group_ids"],
        json!(both)
    );
    let page = server.ok(
        "POST",
        "/recall",
        Some(json!({"group_ids": [first.clone()], "cue": "蔵"})),
    );
    assert_eq!(page["plan"]["context_ids"], json!([sake]), "{page}");
    let row = server.ok(
        "PATCH",
        &format!("/groups/{parent}"),
        Some(json!({"remove_group_ids": [second.clone()]})),
    );
    assert_eq!(row["group_ids"], json!([first.clone()]));
    for (method, path, body) in [
        (
            "POST",
            "/groups".to_string(),
            group_body("p2", Some(json!({"group_ids": ["kura"]}))),
        ),
        (
            "PATCH",
            format!("/groups/{parent}"),
            Some(json!({"add_group_ids": ["kura"]})),
        ),
        (
            "PATCH",
            format!("/groups/{parent}"),
            Some(json!({"remove_group_ids": ["kura"]})),
        ),
        (
            "POST",
            "/recall".to_string(),
            Some(json!({"group_ids": ["kura"], "cue": "蔵"})),
        ),
    ] {
        let (status, refusal) = server.call(method, &path, body);
        assert_eq!(status, 400, "{method} {path}: {refusal}");
        assert_eq!(
            refusal["code"],
            json!("invalid_argument"),
            "{method} {path}: {refusal}"
        );
        assert!(
            refusal["error"]
                .as_str()
                .unwrap()
                .contains("'kura' is not a group id"),
            "{method} {path}: {refusal}"
        );
    }
    assert_eq!(server.ok("GET", "/groups", None)["total"], json!(3));
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}
