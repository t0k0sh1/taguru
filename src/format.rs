//! The file-format stamp every JSON / JSONL record taguru reads or
//! writes carries (ADR 0042): a `type` column naming what the record
//! is, a `version` column naming the format's revision, and — where the
//! record has an identity — an `id` column. Three columns for three
//! facts, where one key used to stand for all of them.

/// The format revision this build reads and writes, as a date. One
/// value for every record type: a revision is a statement about the
/// whole family of shapes docs/import.html and its siblings describe,
/// not about one of them.
pub(crate) const FORMAT_VERSION: &str = "2026-11-13";

/// The format revisions before [`FORMAT_VERSION`], oldest first. None
/// was released: they stamped records on main between the v0.9.7
/// release and this one. A reader that knows its shape did not change
/// across them may read a record stamped with one as its own (the
/// at-rest `{stem}.schema.json`); a reader whose record did change
/// treats one as written by an earlier build (a stored derivation
/// manifest: incomparable, rebuild). Streams stamped with one are not
/// import input (ADR 0057 — only the v0.9.7 release's export is).
pub(crate) const EARLIER_VERSIONS: [&str; 2] = ["2026-09-17", "2026-10-01"];

/// Whether `version` is one of [`EARLIER_VERSIONS`] — by equality, as
/// [`check_version`] judges the current one.
pub(crate) fn is_earlier_version(version: &str) -> bool {
    EARLIER_VERSIONS.contains(&version)
}

/// How to move a data directory this build refuses to boot: the tail
/// every pre-id refusal (boot and `taguru inspect`) ends with. The
/// v0.9.7 release's export is this build's migration input (ADR 0057):
/// that release reads every directory an earlier release wrote, and
/// this build imports what it exports, minting the ids.
pub(crate) fn migration_hint() -> String {
    "boot it on taguru v0.9.7 and export it there (`taguru export --out DIR`), then point \
     this build at an empty data directory and run `taguru import DIR` before starting it \
     (or start it and POST /import) — this build reads a v0.9.7 export as migration input \
     and mints the ids (ADR 0057; docs/import.html#upgrading-from-0-9-7)"
        .to_string()
}

/// Judges a record's `version` column. Absent means "whatever the
/// running build reads" — the courtesy a hand-written file gets, and
/// the opposite of reading it as the oldest revision. Present means
/// exactly [`FORMAT_VERSION`]; anything else is refused by name, since
/// guessing at a shape this build was not written for is how a file
/// gets half-read.
pub(crate) fn check_version(found: Option<&str>) -> Result<(), String> {
    match found {
        None => Ok(()),
        Some(version) if version == FORMAT_VERSION => Ok(()),
        Some(version) => Err(format!(
            "version '{version}' is not a format this taguru reads (it reads '{FORMAT_VERSION}'; \
             omit `version` to mean the running build's own)"
        )),
    }
}

#[derive(serde::Serialize)]
struct SourceHeader<'a> {
    #[serde(rename = "type")]
    record_type: &'static str,
    version: &'static str,
    id: &'a str,
    context_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    create: Option<HeaderCreate<'a>>,
}

/// The create block a writer outside `taguru export` brings: the
/// display name the `context` gets when its `context_id` is not yet on
/// the server, and the description it starts with (#965).
#[derive(Clone, Copy, serde::Serialize)]
pub(crate) struct HeaderCreate<'a> {
    pub(crate) name: &'a str,
    pub(crate) description: &'a str,
}

/// One source file's header line, columns in reading order — what the
/// record is, which revision, which source, which `context` (by id) —
/// with a create block only when the writer brings a name for a
/// `context` that may not exist yet. Every writer outside
/// `taguru export` (which carries the full create block) renders its
/// header here, so the line is spelled one way.
pub(crate) fn source_header_line(
    id: &str,
    context_id: &str,
    create: Option<HeaderCreate<'_>>,
) -> String {
    serde_json::to_string(&SourceHeader {
        record_type: "source",
        version: FORMAT_VERSION,
        id,
        context_id,
        create,
    })
    .expect("a struct of strings always serializes")
}

/// Judges a record's `type` column against the one kind its reader
/// accepts. `None` — the column is absent, as in every record written
/// before ADR 0042 — is refused like any wrong value: a reader never
/// guesses what an untyped record is.
pub(crate) fn check_type(found: Option<&str>, expected: &str) -> Result<(), String> {
    match found {
        Some(kind) if kind == expected => Ok(()),
        Some(kind) => Err(format!(
            "type '{kind}' is not '{expected}', the only record this reads"
        )),
        None => Err(format!(
            "no `type` column where '{expected}' was expected (a record written before \
             this format names none)"
        )),
    }
}

/// Deserializes a record's `version` column: a string when the column
/// is there, and only an ABSENT column reads as `None`. An explicit
/// `null` is not an omission — it is a value, and the wrong one — so it
/// is refused like any other non-string, where a bare `Option<String>`
/// would fold it into "absent" and let it pass as the running build's
/// revision. Pair with `#[serde(default)]` so absence still parses.
pub(crate) fn version_column<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    <String as serde::Deserialize>::deserialize(deserializer).map(Some)
}

/// The `type` column of a stream-level record, read off the parsed
/// line before its full shape is judged. `None` for a line with no
/// `type` string — an operation line, or something that is not an
/// object at all.
pub(crate) fn record_type(value: &serde_json::Value) -> Option<&str> {
    value.as_object()?.get("type")?.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hint names the one release whose export this build imports,
    /// and where the steps are written down.
    #[test]
    fn the_migration_hint_names_the_v097_export_and_the_docs() {
        let hint = migration_hint();
        assert!(hint.starts_with("boot it on taguru v0.9.7"), "{hint}");
        assert!(hint.contains("`taguru import DIR`"), "{hint}");
        assert!(
            hint.contains("docs/import.html#upgrading-from-0-9-7"),
            "{hint}"
        );
    }

    /// The earlier revisions are published dates too, older than the
    /// current one, and matched exactly.
    #[test]
    fn the_earlier_versions_precede_the_current_one_and_match_exactly() {
        assert_eq!(EARLIER_VERSIONS, ["2026-09-17", "2026-10-01"]);
        assert!(
            EARLIER_VERSIONS
                .iter()
                .all(|version| *version < FORMAT_VERSION)
        );
        assert!(is_earlier_version("2026-10-01"));
        assert!(is_earlier_version("2026-09-17"));
        assert!(!is_earlier_version(FORMAT_VERSION));
        assert!(!is_earlier_version("2026-10-01 "));
        assert!(!is_earlier_version("2026-08-01"));
    }

    #[test]
    fn an_absent_version_reads_as_the_running_builds_own() {
        assert_eq!(check_version(None), Ok(()));
    }

    #[test]
    fn the_current_version_is_accepted_and_any_other_is_refused_by_name() {
        assert_eq!(check_version(Some(FORMAT_VERSION)), Ok(()));
        let error = check_version(Some("2008-10-17")).unwrap_err();
        assert!(error.contains("'2008-10-17'"), "{error}");
        assert!(error.contains(FORMAT_VERSION), "{error}");
        // A near-miss is still a miss: equality, never a prefix or a range.
        assert!(check_version(Some("2026-11-13 ")).is_err());
        assert!(check_version(Some("")).is_err());
    }

    /// Pinned as a literal on purpose: a revision date is a published
    /// fact, and changing it must be a visible edit here too.
    #[test]
    fn the_format_version_is_the_published_date() {
        assert_eq!(FORMAT_VERSION, "2026-11-13");
    }

    #[test]
    fn a_source_header_line_spells_its_columns_in_reading_order() {
        let id = "3f2a9c1e-5b7d-4e86-9a10-0c4d8e2f7a63";
        assert_eq!(
            source_header_line("docs/a.md", id, None),
            r#"{"type":"source","version":"2026-11-13","id":"docs/a.md","context_id":"3f2a9c1e-5b7d-4e86-9a10-0c4d8e2f7a63"}"#
        );
        assert_eq!(
            source_header_line(
                "docs/a.md",
                id,
                Some(HeaderCreate {
                    name: "sake",
                    description: "酒蔵"
                })
            ),
            r#"{"type":"source","version":"2026-11-13","id":"docs/a.md","context_id":"3f2a9c1e-5b7d-4e86-9a10-0c4d8e2f7a63","create":{"name":"sake","description":"酒蔵"}}"#
        );
    }

    #[test]
    fn a_type_is_accepted_only_when_it_is_the_expected_one() {
        assert_eq!(check_type(Some("communities"), "communities"), Ok(()));
        let wrong = check_type(Some("group"), "communities").unwrap_err();
        assert!(
            wrong.contains("'group'") && wrong.contains("'communities'"),
            "{wrong}"
        );
        let absent = check_type(None, "communities").unwrap_err();
        assert!(absent.contains("no `type` column"), "{absent}");
        // Equality, never a prefix.
        assert!(check_type(Some("communities_manifest"), "communities").is_err());
    }

    #[derive(serde::Deserialize)]
    struct Versioned {
        #[serde(default, deserialize_with = "version_column")]
        version: Option<String>,
    }

    #[test]
    fn only_an_absent_version_column_reads_as_none() {
        let parse = |text: &str| serde_json::from_str::<Versioned>(text);
        assert_eq!(parse("{}").unwrap().version, None);
        assert_eq!(
            parse(r#"{"version":"2026-11-13"}"#)
                .unwrap()
                .version
                .as_deref(),
            Some("2026-11-13")
        );
        // An explicit null is a value, not an omission; so is a number.
        assert!(parse(r#"{"version":null}"#).is_err());
        assert!(parse(r#"{"version":1}"#).is_err());
    }

    #[test]
    fn record_type_reads_only_a_string_type_on_an_object() {
        let line = |text: &str| serde_json::from_str::<serde_json::Value>(text).unwrap();
        assert_eq!(
            record_type(&line(r#"{"type":"source","id":"a"}"#)),
            Some("source")
        );
        assert_eq!(record_type(&line(r#"{"subject":"a"}"#)), None);
        assert_eq!(record_type(&line(r#"{"type":1}"#)), None);
        assert_eq!(record_type(&line(r#"["type"]"#)), None);
    }
}
