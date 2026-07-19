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
            "PUT",
            &format!("/contexts/{name}"),
            Some(json!({"description": name})),
        );
    }

    server.ok(
        "PUT",
        "/groups/fruit",
        Some(json!({"description": "果物の文脈", "contexts": ["banana", "apple"]})),
    );
    // Create is PUT-once: a second landing answers already_exists.
    let (status, dup) = server.call("PUT", "/groups/fruit", None);
    assert_eq!(status, 409, "{dup}");
    assert_eq!(dup["code"], json!("already_exists"));
    // An absent body is a valid create (defaults) — an empty group is
    // how "create first, fill later" starts.
    server.ok("PUT", "/groups/empty", None);

    // The directory pages by name, `total` cursor-independent, exactly
    // like /contexts.
    let page = server.ok("GET", "/groups", None);
    assert_eq!(page["total"], json!(2), "{page}");
    let names: Vec<&str> = page["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|group| group["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["empty", "fruit"], "name order");
    assert_eq!(
        page["groups"][1]["contexts"],
        json!(["apple", "banana"]),
        "members come back sorted: {page}"
    );
    let page = server.ok("GET", "/groups?limit=1", None);
    assert_eq!(page["total"], json!(2));
    assert_eq!(page["groups"][0]["name"], json!("empty"));
    let page = server.ok("GET", "/groups?limit=1&after=empty", None);
    assert_eq!(page["groups"][0]["name"], json!("fruit"));

    let single = server.ok("GET", "/groups/fruit", None);
    assert_eq!(single["description"], json!("果物の文脈"));
    let (status, missing) = server.call("GET", "/groups/nope", None);
    assert_eq!(status, 404);
    assert_eq!(missing["code"], json!("no_group"));

    // PATCH applies deltas — removals first, then adds — and answers
    // the updated row. Removing a non-member is an idempotent no-op.
    let updated = server.ok(
        "PATCH",
        "/groups/fruit",
        Some(json!({"description": "更新後", "add_contexts": ["cherry"],
                    "remove_contexts": ["apple", "never-was-a-member"]})),
    );
    assert_eq!(updated["description"], json!("更新後"));
    assert_eq!(updated["contexts"], json!(["banana", "cherry"]));
    let (status, missing) = server.call("PATCH", "/groups/nope", Some(json!({"description": "x"})));
    assert_eq!(status, 404, "{missing}");
    assert_eq!(missing["code"], json!("no_group"));

    // Groups and contexts are separate namespaces: one name, both kinds.
    server.ok("PUT", "/groups/apple", Some(json!({"description": "同名"})));
    assert_eq!(server.call("GET", "/contexts/apple", None).0, 200);
    assert_eq!(server.call("GET", "/groups/apple", None).0, 200);

    // DELETE removes the bundling alone; the members live on.
    assert_eq!(server.ok("DELETE", "/groups/fruit", None), json!(true));
    assert_eq!(server.call("GET", "/groups/fruit", None).0, 404);
    for name in ["banana", "cherry"] {
        assert_eq!(
            server.call("GET", &format!("/contexts/{name}"), None).0,
            200
        );
    }
    let (status, gone) = server.call("DELETE", "/groups/fruit", None);
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
    let (status, refused) = server.call("PUT", "/groups/g", Some(json!({"contexts": ["ghost"]})));
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"));
    assert_eq!(server.call("GET", "/groups/g", None).0, 404);

    server.ok("PUT", "/contexts/a", None);
    server.ok("PUT", "/contexts/b", None);
    server.ok("PUT", "/groups/g", Some(json!({"contexts": ["a", "b"]})));

    // An add naming a missing context refuses whole: membership as was.
    let (status, refused) = server.call(
        "PATCH",
        "/groups/g",
        Some(json!({"add_contexts": ["ghost"], "remove_contexts": ["a"]})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"));
    assert_eq!(
        server.ok("GET", "/groups/g", None)["contexts"],
        json!(["a", "b"]),
        "the refused delta must not half-apply"
    );

    // Deleting a member context drops it from the group, immediately.
    server.ok("DELETE", "/contexts/a", None);
    assert_eq!(
        server.ok("GET", "/groups/g", None)["contexts"],
        json!(["b"])
    );

    // The write-boundary caps hold for groups too.
    let long_name = "x".repeat(65);
    let (status, oversized) = server.call("PUT", &format!("/groups/{long_name}"), None);
    assert_eq!(status, 400, "{oversized}");
    assert_eq!(oversized["code"], json!("invalid_argument"));
    let (status, oversized) = server.call(
        "PUT",
        "/groups/big",
        Some(json!({"description": "x".repeat(5000)})),
    );
    assert_eq!(status, 400, "{oversized}");
    let over_cap: Vec<String> = (0..1001).map(|i| format!("c{i}")).collect();
    let (status, refused) = server.call("PUT", "/groups/big", Some(json!({"contexts": over_cap})));
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
    server.ok("PUT", "/contexts/a", None);
    server.ok("PUT", "/groups/g", Some(json!({"contexts": ["a"]})));
    server.ok("PUT", "/groups/kid", None);
    server.ok("PATCH", "/groups/g", Some(json!({"add_groups": ["kid"]})));

    // 1 member + 1,000 adds = one past the cap: refused whole.
    let ghosts: Vec<String> = (0..1000).map(|i| format!("ghost{i:04}")).collect();
    let (status, refused) =
        server.call("PATCH", "/groups/g", Some(json!({"add_contexts": ghosts})));
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"), "{refused}");
    // Child groups ride the same cap on their own set.
    let ghost_kids: Vec<String> = (0..1000).map(|i| format!("gg{i:04}")).collect();
    let (status, refused) = server.call(
        "PATCH",
        "/groups/g",
        Some(json!({"add_groups": ghost_kids})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"), "{refused}");

    // Removing the member in the same request makes room — the cap
    // passes and the EXISTENCE gate answers next, proving the cap is
    // judged on the result and before existence.
    let ghosts: Vec<String> = (0..1000).map(|i| format!("ghost{i:04}")).collect();
    let (status, refused) = server.call(
        "PATCH",
        "/groups/g",
        Some(json!({"add_contexts": ghosts, "remove_contexts": ["a"]})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"), "{refused}");

    // Nothing half-applied anywhere along the way.
    let row = server.ok("GET", "/groups/g", None);
    assert_eq!(row["contexts"], json!(["a"]), "{row}");
    assert_eq!(row["groups"], json!(["kid"]), "{row}");
}

/// Nesting: a group may hold child groups — at most three storeys,
/// never a cycle, children must exist — child deltas patch like
/// context deltas, and deleting a child sweeps it out of every parent.
#[test]
fn groups_nest_with_a_depth_cap_and_no_cycles() {
    let server = Server::start("groups-nesting");
    for name in ["a", "b"] {
        server.ok("PUT", &format!("/contexts/{name}"), None);
    }
    server.ok("PUT", "/groups/leaf", Some(json!({"contexts": ["a"]})));
    server.ok(
        "PUT",
        "/groups/mid",
        Some(json!({"groups": ["leaf"], "contexts": ["b"]})),
    );
    server.ok("PUT", "/groups/top", Some(json!({"groups": ["mid"]})));

    // Rows carry their children; members stay the direct ones.
    let row = server.ok("GET", "/groups/mid", None);
    assert_eq!(row["groups"], json!(["leaf"]), "{row}");
    assert_eq!(row["contexts"], json!(["b"]));
    let page = server.ok("GET", "/groups", None);
    assert_eq!(page["groups"][2]["name"], json!("top"), "{page}");
    assert_eq!(page["groups"][2]["groups"], json!(["mid"]));

    // A fourth storey refuses as a cap, a cycle (the self-loop
    // included) as a bad argument, an unknown child in the group
    // namespace's own 404 — and nothing half-applies.
    let (status, refused) = server.call("PUT", "/groups/over", Some(json!({"groups": ["top"]})));
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"));
    assert_eq!(server.call("GET", "/groups/over", None).0, 404);
    let (status, refused) = server.call(
        "PATCH",
        "/groups/leaf",
        Some(json!({"add_groups": ["top"]})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("invalid_argument"));
    let (status, refused) = server.call(
        "PATCH",
        "/groups/leaf",
        Some(json!({"add_groups": ["leaf"]})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("invalid_argument"));
    let (status, refused) = server.call(
        "PATCH",
        "/groups/top",
        Some(json!({"add_groups": ["ghost"], "remove_groups": ["mid"]})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_group"));
    assert_eq!(
        server.ok("GET", "/groups/top", None)["groups"],
        json!(["mid"]),
        "the refused delta must not half-apply"
    );

    // Child deltas move like context deltas — and a child may sit
    // under two parents at once: the shape is a DAG, not a tree.
    let updated = server.ok(
        "PATCH",
        "/groups/top",
        Some(json!({"add_groups": ["leaf"], "remove_groups": ["mid"]})),
    );
    assert_eq!(updated["groups"], json!(["leaf"]));
    assert_eq!(
        server.ok("GET", "/groups/mid", None)["groups"],
        json!(["leaf"])
    );

    // Deleting a child sweeps it from every parent; its member
    // contexts live on untouched.
    server.ok("DELETE", "/groups/leaf", None);
    assert_eq!(server.ok("GET", "/groups/top", None)["groups"], json!([]));
    assert_eq!(server.ok("GET", "/groups/mid", None)["groups"], json!([]));
    assert_eq!(server.call("GET", "/contexts/a", None).0, 200);

    // The children list rides the same input ceiling as contexts.
    let over_cap: Vec<String> = (0..1001).map(|i| format!("g{i}")).collect();
    let (status, refused) = server.call("PUT", "/groups/wide", Some(json!({"groups": over_cap})));
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"));
}

/// Groups persist across a restart, and boot reconciliation drops any
/// dangling member a crash could have left in a group file.
#[test]
fn groups_survive_restart_and_boot_reconciles_dangling_members() {
    let server = Server::start("groups-restart");
    server.ok("PUT", "/contexts/sake", None);
    server.ok(
        "PUT",
        "/groups/drinks",
        Some(json!({"description": "飲料", "contexts": ["sake"]})),
    );

    let data_dir = server.stop_gracefully();
    // Plant a dangling member and a dangling child the way a crash
    // between a deletion and the sweep's rewrite would: straight into
    // the file.
    std::fs::write(
        data_dir.join("drinks.group"),
        serde_json::to_vec(&json!({"description": "飲料", "contexts": ["sake", "gone"],
                                   "groups": ["nowhere"]}))
        .unwrap(),
    )
    .unwrap();

    let server = Server::start_on("groups-restart2", data_dir);
    let survived = server.ok("GET", "/groups/drinks", None);
    assert_eq!(survived["description"], json!("飲料"));
    assert_eq!(
        survived["contexts"],
        json!(["sake"]),
        "boot must reconcile the planted dangling member: {survived}"
    );
    assert_eq!(
        survived["groups"],
        json!([]),
        "boot must reconcile the planted dangling child: {survived}"
    );
    // And the fix reached the file, not just memory.
    let on_disk = std::fs::read_to_string(server.data_dir.join("drinks.group")).unwrap();
    assert!(!on_disk.contains("gone"), "{on_disk}");
    assert!(!on_disk.contains("nowhere"), "{on_disk}");
}

/// The scope story for groups: every key sees every row but only its
/// granted members; a write touching any context beyond the grant —
/// current members included — refuses whole, and out-of-scope names
/// answer the same 403 whether or not they exist (no existence oracle).
#[test]
fn key_scopes_filter_group_members_and_gate_group_writes() {
    let server = Server::start_with_env(
        "groups-scopes",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,reader:rtok,potter:stok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"reader": "read", "potter": {"role": "write", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    for context in ["sake", "bunko"] {
        assert_eq!(
            call("PUT", &format!("/contexts/{context}"), None, "atok").0,
            200
        );
    }
    assert_eq!(
        call(
            "PUT",
            "/groups/mixed",
            Some(json!({"contexts": ["sake", "bunko"]})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/groups/ours",
            Some(json!({"contexts": ["sake"]})),
            "atok"
        )
        .0,
        200
    );

    // Reads: every row is visible (groups are labels, not content),
    // but a scoped key sees only its granted members.
    let (status, listed) = call("GET", "/groups", None, "stok");
    assert_eq!(status, 200);
    assert_eq!(listed["result"]["total"], json!(2), "{listed}");
    assert_eq!(
        listed["result"]["groups"][0]["name"],
        json!("mixed"),
        "{listed}"
    );
    assert_eq!(
        listed["result"]["groups"][0]["contexts"],
        json!(["sake"]),
        "bunko must be filtered from a sake-scoped view: {listed}"
    );
    let (_, single) = call("GET", "/groups/mixed", None, "stok");
    assert_eq!(single["result"]["contexts"], json!(["sake"]), "{single}");

    // Role gate: read keys read, nothing more.
    let (status, refused) = call("PUT", "/groups/new", None, "rtok");
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
        "/groups/mixed",
        Some(json!({"description": "mine now"})),
        "stok",
    );
    assert_eq!(status, 403, "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{refused}"
    );

    // The oracle stays shut: an out-of-scope name answers the same 403
    // whether it exists (bunko) or not (ghost) — never a revealing 404.
    let (status_real, real) = call(
        "PATCH",
        "/groups/ours",
        Some(json!({"add_contexts": ["bunko"]})),
        "stok",
    );
    let (status_ghost, ghost) = call(
        "PATCH",
        "/groups/ours",
        Some(json!({"add_contexts": ["ghost"]})),
        "stok",
    );
    assert_eq!((status_real, status_ghost), (403, 403), "{real} / {ghost}");
    assert_eq!(real["code"], ghost["code"], "{real} / {ghost}");
    let (_, ours) = call("GET", "/groups/ours", None, "atok");
    assert_eq!(
        ours["result"]["contexts"],
        json!(["sake"]),
        "the refused adds must not have applied: {ours}"
    );

    // Inside the grant, a scoped writer works normally...
    assert_eq!(
        call(
            "PATCH",
            "/groups/ours",
            Some(json!({"description": "陶工の棚"})),
            "stok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/groups/mine",
            Some(json!({"contexts": ["sake"]})),
            "stok"
        )
        .0,
        200
    );
    // ...but deletion is an operator verb (admin), like contexts.
    assert_eq!(call("DELETE", "/groups/mine", None, "stok").0, 403);
    assert_eq!(call("DELETE", "/groups/mine", None, "atok").0, 200);

    // Nesting counts through: a child whose members sit beyond the
    // grant poisons every write on the parent — the child's NAME stays
    // visible (labels, not content), but its contexts are what a grant
    // is about, wherever they hang.
    assert_eq!(
        call(
            "PUT",
            "/groups/shelf",
            Some(json!({"contexts": ["bunko"]})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PATCH",
            "/groups/ours",
            Some(json!({"add_groups": ["shelf"]})),
            "atok"
        )
        .0,
        200
    );
    let (_, nested) = call("GET", "/groups/ours", None, "stok");
    assert_eq!(nested["result"]["groups"], json!(["shelf"]), "{nested}");
    assert_eq!(nested["result"]["contexts"], json!(["sake"]), "{nested}");
    let (status, refused) = call(
        "PATCH",
        "/groups/ours",
        Some(json!({"description": "still mine"})),
        "stok",
    );
    assert_eq!(status, 403, "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{refused}"
    );
    // Naming such a child in a delta refuses the same way — a scoped
    // key cannot create a parent over contexts it has no grant on.
    let (status, refused) = call(
        "PUT",
        "/groups/annex",
        Some(json!({"groups": ["shelf"]})),
        "stok",
    );
    assert_eq!(status, 403, "{refused}");
    assert_eq!(call("GET", "/groups/annex", None, "atok").0, 404);
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
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "PUT",
        "/groups/kura",
        Some(json!({"description": "蔵元一式", "contexts": ["sake"]})),
    );
    // Nothing dirty may remain: the flusher must not collide with the
    // read-only window below.
    server.ok("POST", "/flush", None);

    // Unlink needs write permission on the PARENT directory; freezing
    // the data directory makes exactly the unlink fail.
    let frozen = std::fs::Permissions::from_mode(0o555);
    std::fs::set_permissions(&server.data_dir, frozen).expect("chmod must apply");
    let (status, refusal) = server.call("DELETE", "/groups/kura", None);
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
    let (status, missing) = server.call("GET", "/groups/kura", None);
    assert_eq!(status, 404, "{missing}");

    // …and the surviving file resurfaces it at the next boot.
    let data_dir = server.stop_gracefully();
    assert!(
        data_dir.join("kura.group").exists(),
        "the unlink must have failed"
    );
    let server = Server::start_on("group-unlink-reboot", data_dir);
    let row = server.ok("GET", "/groups/kura", None);
    assert_eq!(row["contexts"], json!(["sake"]), "{row}");
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
        server.ok("PUT", &format!("/contexts/{name}"), None);
    }
    let row = server.ok("GET", "/contexts/apple", None);
    assert_eq!(
        row["revision"],
        json!({"graph": 0, "passages": 0, "config": 0}),
        "{row}"
    );

    // parent holds apple directly and banana only through a child.
    server.ok(
        "PUT",
        "/groups/child",
        Some(json!({"contexts": ["banana"]})),
    );
    server.ok(
        "PUT",
        "/groups/parent",
        Some(json!({"contexts": ["apple"], "groups": ["child"]})),
    );
    let fingerprint = |server: &Server| {
        server.ok("GET", "/groups/parent", None)["fingerprint"]
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
        "/contexts/apple/query",
        Some(json!({"subject": "蔵"})),
    );
    assert_eq!(fingerprint(&server), start);

    // A graph write to a direct member moves it…
    server.ok(
        "POST",
        "/contexts/apple/associations",
        Some(json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0}])),
    );
    let after_direct = fingerprint(&server);
    assert_ne!(after_direct, start);

    // …a passage write to a member reached only through the child too…
    server.ok(
        "POST",
        "/contexts/banana/sources",
        Some(json!({"passages": {"doc": "バナナの原文。"}})),
    );
    let after_nested = fingerprint(&server);
    assert_ne!(after_nested, after_direct);

    // …while the same writes to a non-member change nothing.
    server.ok(
        "POST",
        "/contexts/outside/associations",
        Some(json!([{"subject": "a", "label": "b", "object": "c", "weight": 1.0}])),
    );
    assert_eq!(fingerprint(&server), after_nested);

    // The config lane (a floor edit) counts as a member change too.
    server.ok(
        "PATCH",
        "/contexts/apple",
        Some(json!({"semantic_floor": 0.5})),
    );
    let after_config = fingerprint(&server);
    assert_ne!(after_config, after_nested);

    // A membership edit moves it with no member written at all.
    server.ok(
        "PATCH",
        "/groups/parent",
        Some(json!({"add_contexts": ["outside"]})),
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
