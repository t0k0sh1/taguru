//! The file-format stamp every JSON / JSONL record taguru reads or
//! writes carries (ADR 0042): a `type` column naming what the record
//! is, a `version` column naming the format's revision, and — where the
//! record has an identity — an `id` column. Three columns for three
//! facts, where one key used to stand for all of them.

/// The format revision this build reads and writes, as a date. One
/// value for every record type: a revision is a statement about the
/// whole family of shapes docs/import.html and its siblings describe,
/// not about one of them.
pub(crate) const FORMAT_VERSION: &str = "2026-09-17";

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
    context: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    create: Option<CreateDescription<'a>>,
}

#[derive(serde::Serialize)]
struct CreateDescription<'a> {
    description: &'a str,
}

/// One source file's header line, columns in reading order — what the
/// record is, which revision, which source, which `context` — with a
/// create block only when the writer brings a description for a
/// `context` that may not exist yet. Every writer outside
/// `taguru export` (which carries the full create block) renders its
/// header here, so the line is spelled one way.
pub(crate) fn source_header_line(
    id: &str,
    context: &str,
    create_description: Option<&str>,
) -> String {
    serde_json::to_string(&SourceHeader {
        record_type: "source",
        version: FORMAT_VERSION,
        id,
        context,
        create: create_description.map(|description| CreateDescription { description }),
    })
    .expect("a struct of strings always serializes")
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
        assert!(check_version(Some("2026-09-17 ")).is_err());
        assert!(check_version(Some("")).is_err());
    }

    /// Pinned as a literal on purpose: a revision date is a published
    /// fact, and changing it must be a visible edit here too.
    #[test]
    fn the_format_version_is_the_published_date() {
        assert_eq!(FORMAT_VERSION, "2026-09-17");
    }

    #[test]
    fn a_source_header_line_spells_its_columns_in_reading_order() {
        assert_eq!(
            source_header_line("docs/a.md", "sake", None),
            r#"{"type":"source","version":"2026-09-17","id":"docs/a.md","context":"sake"}"#
        );
        assert_eq!(
            source_header_line("docs/a.md", "sake", Some("酒蔵")),
            r#"{"type":"source","version":"2026-09-17","id":"docs/a.md","context":"sake","create":{"description":"酒蔵"}}"#
        );
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
