//! Label/alias/source directory paging and the compact endpoints.

use std::process::Command;

use serde_json::json;

use crate::support::*;

/// The three collection listings page like the directory: keyset
/// cursors, a total that tells the whole story, and — for aliases —
/// one cursor spanning both namespaces, concepts first.
#[test]
fn labels_aliases_and_sources_page_with_keyset_cursors() {
    let server = Server::start("http-keyset-pages");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "a.md"},
            {"subject": "蔵", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "a.md"},
            {"subject": "蔵", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "b.md"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"a.md": "本文。", "b.md": "本文。", "c.md": "本文。"}})),
    );
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({
            "concepts": {"Aomine": "青嶺", "Kura": "蔵"},
            "labels": {"establishment": "創業年"},
        })),
    );

    // labels: sorted, paged, total constant across pages.
    let first = server.ok("GET", "/contexts/sake/labels?limit=2", None);
    assert_eq!(first["total"], json!(3), "{first}");
    assert_eq!(first["labels"].as_array().unwrap().len(), 2);
    let last = first["labels"][1].as_str().unwrap();
    let second = server.ok(
        "GET",
        &format!("/contexts/sake/labels?after={}", urlencode(last)),
        None,
    );
    assert_eq!(second["total"], json!(3));
    assert_eq!(second["labels"].as_array().unwrap().len(), 1);

    // sources: keyset by id.
    let first = server.ok("GET", "/contexts/sake/sources?limit=2", None);
    assert_eq!(first["total"], json!(3), "{first}");
    assert_eq!(first["sources"], json!(["a.md", "b.md"]), "{first}");
    let second = server.ok("GET", "/contexts/sake/sources?after=b.md", None);
    assert_eq!(second["sources"], json!(["c.md"]), "{second}");

    // aliases: one cursor across both namespaces, concepts first.
    let first = server.ok("GET", "/contexts/sake/aliases?limit=2", None);
    assert_eq!(first["total"], json!(3), "{first}");
    assert_eq!(
        first["concepts"],
        json!({"Aomine": "青嶺", "Kura": "蔵"}),
        "{first}"
    );
    assert_eq!(first["labels"], json!({}), "{first}");
    let second = server.ok("GET", "/contexts/sake/aliases?after=concept:Kura", None);
    assert_eq!(second["concepts"], json!({}), "{second}");
    assert_eq!(
        second["labels"],
        json!({"establishment": "創業年"}),
        "{second}"
    );
    // A malformed cursor is a 400, not an empty page.
    let (status, refusal) = server.call("GET", "/contexts/sake/aliases?after=bogus", None);
    assert_eq!(status, 400, "{refusal}");
}

/// #62 item 6: `prefix` narrows labels/sources/aliases to a population
/// — like `pinned` on the directory, it counts toward `total` (unlike
/// `after`/`limit`), and for aliases it applies to both namespaces
/// before they're joined into the `concept:`/`label:` cursor.
#[test]
fn labels_aliases_and_sources_filter_by_prefix_and_count_total_after_filtering() {
    let server = Server::start("http-prefix-filter");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "a.md"},
            {"subject": "蔵", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "a.md"},
            {"subject": "蔵", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "b.md"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"a.md": "本文。", "b.md": "本文。", "c.txt": "本文。"}})),
    );
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({
            "concepts": {"Aomine": "青嶺", "Kura": "蔵"},
            "labels": {"establishment": "創業年"},
        })),
    );

    // labels: only those starting with "杜".
    let filtered = server.ok(
        "GET",
        &format!("/contexts/sake/labels?prefix={}", urlencode("杜")),
        None,
    );
    assert_eq!(filtered["total"], json!(1), "{filtered}");
    assert_eq!(filtered["labels"], json!(["杜氏"]), "{filtered}");

    // sources: only the .md files.
    let filtered = server.ok("GET", "/contexts/sake/sources?prefix=a", None);
    assert_eq!(filtered["total"], json!(1), "{filtered}");
    assert_eq!(filtered["sources"], json!(["a.md"]), "{filtered}");

    // aliases: prefix applies within each namespace before the
    // concept:/label: cursor joins them.
    let filtered = server.ok("GET", "/contexts/sake/aliases?prefix=A", None);
    assert_eq!(filtered["total"], json!(1), "{filtered}");
    assert_eq!(
        filtered["concepts"],
        json!({"Aomine": "青嶺"}),
        "{filtered}"
    );
    assert_eq!(filtered["labels"], json!({}), "{filtered}");

    // No matches means total 0, not an error.
    let empty = server.ok("GET", "/contexts/sake/sources?prefix=zzz", None);
    assert_eq!(empty["total"], json!(0), "{empty}");
    assert_eq!(empty["sources"], json!([]), "{empty}");
}

/// Percent-encodes one query value the way ureq will not do for us.
fn urlencode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (byte as char).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

/// POST /contexts/{name}/compact rewrites the image live: smaller
/// footprint, identical answers, and — being an admin verb — refused
/// for write-scoped keys by the fail-closed role table.
#[test]
fn the_compact_endpoint_shrinks_live_and_is_admin_only() {
    let server = Server::start_with_env(
        "http-compact",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,scribe:wtok"),
            ("TAGURU_KEY_SCOPES", r#"{"scribe": "write"}"#),
        ],
    );
    let admin = Some("atok");
    let (status, _) = server.call_with_token(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "d"})),
        admin,
    );
    assert_eq!(status, 200);
    server.call_with_token(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "keep.md"},
            {"subject": "蔵", "label": "廃止銘柄", "object": "旧銘", "weight": 1.0, "source": "gone.md"},
        ])),
        admin,
    );
    server.call_with_token(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "gone.md"})),
        admin,
    );

    let (status, refused) =
        server.call_with_token("POST", "/contexts/sake/compact", None, Some("wtok"));
    assert_eq!(status, 403, "{refused}");

    let (status, outcome) = server.call_with_token("POST", "/contexts/sake/compact", None, admin);
    assert_eq!(status, 200, "{outcome}");
    let shed = &outcome["result"];
    assert_eq!(shed["dead_edges"], json!(1), "{outcome}");
    assert!(
        shed["bytes_after"].as_u64().unwrap() < shed["bytes_before"].as_u64().unwrap(),
        "{outcome}"
    );
    let (status, facts) = server.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵"})),
        admin,
    );
    assert_eq!(status, 200);
    assert_eq!(facts["result"]["total"], json!(1), "{facts}");
    assert_eq!(
        facts["result"]["matches"][0]["label"],
        json!("杜氏"),
        "{facts}"
    );
}

/// `POST /maintenance/compact` sweeps every context whose dead ratio
/// clears `min_dead_ratio`, reopening the server when done. Being
/// server-wide, it is refused for a context-scoped key exactly like
/// `/flush` — the CONTEXT bypass guard, not the role table, and refused
/// even at admin role — and, like every other operator verb, refused
/// outright for a non-admin role.
#[test]
fn the_maintenance_compact_endpoint_sweeps_worst_ratio_first_and_is_admin_only() {
    let server = Server::start_with_env(
        "http-maint-compact",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,scribe:wtok,curator:ctok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"scribe": "write", "curator": {"role": "admin", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let admin = Some("atok");
    server.call_with_token(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "d"})),
        admin,
    );
    server.call_with_token(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "keep.md"},
            {"subject": "蔵", "label": "廃止銘柄", "object": "旧銘", "weight": 1.0, "source": "gone.md"},
        ])),
        admin,
    );
    server.call_with_token(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "gone.md"})),
        admin,
    );

    // Write-scoped, unscoped: refused by the plain role table.
    let (status, refused) =
        server.call_with_token("POST", "/maintenance/compact", None, Some("wtok"));
    assert_eq!(status, 403, "{refused}");

    // Admin role but context-scoped: refused because this sweeps the
    // WHOLE server, not just its granted context.
    let (status, scoped) =
        server.call_with_token("POST", "/maintenance/compact", None, Some("ctok"));
    assert_eq!(status, 403, "{scoped}");
    assert!(
        scoped["error"].as_str().unwrap().contains("server-wide"),
        "{scoped}"
    );

    // A floor above sake's actual ratio (0.5, one dead of two) selects
    // nothing — the sweep still runs and reopens, just with no work.
    let (status, none) = server.call_with_token(
        "POST",
        "/maintenance/compact?min_dead_ratio=0.9",
        None,
        admin,
    );
    assert_eq!(status, 200, "{none}");
    assert_eq!(none["result"]["contexts"], json!([]), "{none}");
    assert_eq!(none["result"]["deadline_exceeded"], json!(false));

    let (status, outcome) = server.call_with_token("POST", "/maintenance/compact", None, admin);
    assert_eq!(status, 200, "{outcome}");
    let contexts = outcome["result"]["contexts"].as_array().unwrap();
    assert_eq!(contexts.len(), 1, "{outcome}");
    assert_eq!(contexts[0]["name"], json!("sake"), "{outcome}");
    assert_eq!(contexts[0]["dead_edges"], json!(1), "{outcome}");
    assert!(
        contexts[0]["bytes_after"].as_u64().unwrap()
            < contexts[0]["bytes_before"].as_u64().unwrap(),
        "{outcome}"
    );
    assert_eq!(outcome["result"]["deadline_exceeded"], json!(false));

    // The sweep reopens the server: an ordinary call succeeds right after.
    assert_eq!(server.call("GET", "/health", None).0, 200);

    // Live content survived the rebuild.
    let (status, facts) = server.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵"})),
        admin,
    );
    assert_eq!(status, 200);
    assert_eq!(facts["result"]["total"], json!(1), "{facts}");
    assert_eq!(
        facts["result"]["matches"][0]["label"],
        json!("杜氏"),
        "{facts}"
    );
}

/// Ratio-triggered auto-compaction, end to end and by DEFAULT — no
/// TAGURU_AUTO_COMPACT in the environment (issue #135): a context at
/// 2 dead of 3 sits past the 0.5 trigger, so the flusher tick itself
/// rebuilds it — no operator call — and the run shows on /metrics
/// (outcome counter, reclaimed bytes, last-success clock) while the
/// live content survives verbatim.
#[test]
fn auto_compaction_fires_from_the_flusher_tick_by_default() {
    let server = Server::start_with_env("auto-compact", &[("TAGURU_FLUSH_SECS", "1")]);
    let (status, _) = server.call("PUT", "/contexts/sake", Some(json!({"pinned": true})));
    assert_eq!(status, 200);
    server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "keep.md"},
            {"subject": "蔵", "label": "廃止銘柄", "object": "旧銘", "weight": 1.0, "source": "gone.md"},
            {"subject": "蔵", "label": "廃止蔵", "object": "旧蔵", "weight": 1.0, "source": "gone.md"},
        ])),
    );
    server.call(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "gone.md"})),
    );

    // The tick fires every second; give a loaded CI box a generous
    // window before calling the loop broken.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let metric = |text: &str, name: &str| -> u64 {
        text.lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|rest| rest.trim().parse().ok())
            .unwrap_or_else(|| panic!("{name} must render: {text}"))
    };
    let text = loop {
        let (status, body) = server.call("GET", "/metrics", None);
        assert_eq!(status, 200);
        let text = body
            .as_str()
            .expect("metrics body is text, not JSON")
            .to_string();
        if metric(&text, "taguru_auto_compactions_total{outcome=\"ok\"}") >= 1 {
            break text;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the flusher never auto-compacted: {text}"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    assert_eq!(
        metric(&text, "taguru_auto_compactions_total{outcome=\"failed\"}"),
        0,
        "{text}"
    );
    assert!(
        metric(&text, "taguru_auto_compact_reclaimed_bytes_total") > 0,
        "{text}"
    );
    assert!(
        metric(&text, "taguru_auto_compact_last_success_timestamp_seconds") > 0,
        "{text}"
    );
    assert_eq!(
        metric(&text, "taguru_dead_edges"),
        0,
        "the rebuild shed both dead edges: {text}"
    );

    // Live content survived the automatic rebuild.
    let (status, facts) = server.call(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵"})),
    );
    assert_eq!(status, 200);
    assert_eq!(facts["result"]["total"], json!(1), "{facts}");
    assert_eq!(
        facts["result"]["matches"][0]["label"],
        json!("杜氏"),
        "{facts}"
    );
}

/// The changelog's stated limitation, pinned: compaction — the live
/// endpoint and the offline CLI both — leaves `.group` files
/// byte-for-byte alone. Groups hold nothing to compact, and a rewrite
/// here would be a regression in disguise.
#[test]
fn compact_leaves_group_files_byte_for_byte() {
    let server = Server::start("compact-groups");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "keep.md"},
            {"subject": "蔵", "label": "廃止銘柄", "object": "旧銘", "weight": 1.0, "source": "gone.md"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "gone.md"})),
    );
    server.ok(
        "PUT",
        "/groups/kura",
        Some(json!({"description": "蔵元一式", "contexts": ["sake"]})),
    );
    let group_file = server.data_dir.join("kura.group");
    let before = std::fs::read(&group_file).expect("the group file must exist");

    // The live endpoint rewrites the context image, not the group file.
    let shed = server.ok("POST", "/contexts/sake/compact", None);
    assert_eq!(shed["dead_edges"], json!(1), "{shed}");
    let after_live = std::fs::read(&group_file).expect("the group file must survive");
    assert_eq!(
        before, after_live,
        "live compact must not touch group files"
    );

    // The offline CLI sweep over the whole directory: same statement.
    let data_dir = server.stop_gracefully();
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .args(["compact"])
        .env("TAGURU_DATA_DIR", &data_dir);
    let compacted = command.output().expect("binary must run");
    assert_eq!(compacted.status.code(), Some(0), "{compacted:?}");
    let after_cli = std::fs::read(&group_file).expect("the group file must survive");
    assert_eq!(
        before, after_cli,
        "offline compact must not touch group files"
    );

    // And the untouched record still boots: the group answers as stored.
    let server = Server::start_on("compact-groups-reboot", data_dir);
    let row = server.ok("GET", "/groups/kura", None);
    assert_eq!(row["contexts"], json!(["sake"]), "{row}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}
