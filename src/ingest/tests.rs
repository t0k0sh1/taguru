use super::*;

#[test]
fn every_usage_variable_is_a_known_key() {
    // This command's own USAGE is invisible to cli.rs's
    // consistency tests: a variable documented here but missing
    // from KNOWN_KEYS would make --config warn "typo?" on a
    // perfectly valid setting.
    crate::config::assert_usage_vars_are_known_keys(USAGE);
}

fn parse(text: &str) -> Result<Batch, String> {
    parse_batch(std::io::Cursor::new(text))
}

const HEADER: &str = r#"{"taguru_batch": 1, "context": "sake", "source": "doc-1"}"#;

#[test]
fn split_batches_slices_exactly_the_bytes_between_stream_level_records() {
    let body = concat!(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s1\"}\n",
        "{\"assoc\": [\"a\", \"likes\", \"b\"]}\n",
        "\n",
        "{\"taguru_group\": 1, \"name\": \"g\", \"contexts\": [\"sake\"]}\n",
        "{\"taguru_batch\": 1, \"context\": \"beer\", \"source\": \"s2\"}\n",
        "{\"assoc\": [\"c\", \"likes\", \"d\"]}",
    )
    .as_bytes();
    let ranges = split_batches(body);
    assert_eq!(ranges.len(), 2);
    let first = std::str::from_utf8(&body[ranges[0].clone()]).unwrap();
    assert!(first.starts_with("{\"taguru_batch\": 1, \"context\": \"sake\""));
    // The batch's ops (and the blank line) ride along; the group
    // record between the batches belongs to neither.
    assert!(first.contains("likes"));
    assert!(!first.contains("taguru_group"));
    let second = std::str::from_utf8(&body[ranges[1].clone()]).unwrap();
    assert!(second.starts_with("{\"taguru_batch\": 1, \"context\": \"beer\""));
    assert!(second.ends_with("\"d\"]}"), "EOF closes the last batch");
}

/// [`split_batches_slices_exactly_the_bytes_between_stream_level_records`]'s
/// `taguru_schema` case: a schema record between two batches
/// belongs to neither, the same as a group record.
#[test]
fn split_batches_excludes_a_schema_record_from_either_adjacent_batch() {
    let body = format!(
        "{{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s1\"}}\n\
         {{\"assoc\": [\"a\", \"likes\", \"b\"]}}\n\
         {SCHEMA_LINE}\n\
         {{\"taguru_batch\": 1, \"context\": \"beer\", \"source\": \"s2\"}}\n\
         {{\"assoc\": [\"c\", \"likes\", \"d\"]}}"
    );
    let body = body.as_bytes();
    let ranges = split_batches(body);
    assert_eq!(ranges.len(), 2);
    let first = std::str::from_utf8(&body[ranges[0].clone()]).unwrap();
    assert!(first.contains("likes"));
    assert!(!first.contains("taguru_schema"));
    let second = std::str::from_utf8(&body[ranges[1].clone()]).unwrap();
    assert!(second.starts_with("{\"taguru_batch\": 1, \"context\": \"beer\""));
}

#[test]
fn a_batch_parses_and_the_header_source_stamps_every_association() {
    let batch = parse(&format!(
        "{HEADER}\n\
         {{\"subject\": \"青嶺酒造\", \"label\": \"創業年\", \"object\": \"1907年\", \"weight\": 2.0}}\n\
         \n\
         {{\"alias\": \"Aomine\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}}\n\
         {{\"alias\": \"設立年\", \"canonical\": \"創業年\", \"kind\": \"label\"}}\n\
         {{\"passage\": \"青嶺酒造は1907年創業。\"}}\n"
    ))
    .unwrap();
    assert_eq!(batch.context, "sake");
    assert_eq!(batch.associations.len(), 1);
    assert_eq!(batch.associations[0].source.as_deref(), Some("doc-1"));
    assert_eq!(batch.concepts.len(), 1);
    assert_eq!(batch.labels.len(), 1);
    assert_eq!(batch.passage.as_deref(), Some("青嶺酒造は1907年創業。"));
    assert_eq!(batch.op_count(), 3);
}

#[test]
fn an_association_carrying_its_own_source_is_refused_by_line_number() {
    let error = parse(&format!(
        "{HEADER}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0, \
          \"source\": \"rogue\"}}\n"
    ))
    .unwrap_err();
    assert!(error.contains("line 2"), "{error}");
    assert!(error.contains("source"), "{error}");
}

#[test]
fn the_first_line_must_be_a_header_of_a_readable_version() {
    let error =
        parse("{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}\n")
            .unwrap_err();
    assert!(error.contains("not a batch header"), "{error}");

    let error =
        parse("{\"taguru_batch\": 2, \"context\": \"c\", \"source\": \"s\"}\n").unwrap_err();
    assert!(error.contains("taguru_batch 2"), "{error}");

    assert!(parse("\n\n").unwrap_err().contains("empty file"));
}

/// Notepad and other Windows editors stamp a UTF-8 BOM onto every
/// file they save; left in place it rides onto '{' as the first
/// byte of the header line and fails to parse as JSON at all, with
/// nothing in the error pointing at what actually went wrong.
#[test]
fn a_leading_bom_does_not_break_the_first_line() {
    let batch = parse(&format!("\u{FEFF}{HEADER}\n")).unwrap();
    assert_eq!(batch.context, "sake");
    assert_eq!(batch.source, "doc-1");
}

#[test]
fn a_stream_of_batches_parses_with_per_batch_state() {
    let batches = parse_stream(std::io::Cursor::new(format!(
        "{HEADER}\n\
         {{\"passage\": \"第1段落。\"}}\n\
         {{\"paragraph\": 0, \"question\": \"何?\"}}\n\
         {{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n"
    )))
    .unwrap()
    .batches;
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].source, "doc-1");
    assert_eq!(batches[0].questions.len(), 1);
    assert_eq!(batches[1].source, "doc-2");
    // Per-batch validation still applies at each boundary: the
    // second batch carries no passage, so its questions would have
    // refused — and doc-1's question must not leak into doc-2.
    assert!(batches[1].questions.is_empty());
    assert_eq!(batches[1].associations[0].source.as_deref(), Some("doc-2"));
}

#[test]
fn a_stream_restating_one_source_is_refused() {
    let error = parse_stream(std::io::Cursor::new(format!(
        "{HEADER}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n\
         {HEADER}\n"
    )))
    .unwrap_err();
    assert!(
        error.contains("line 3") && error.contains("one batch owns one source's truth"),
        "{error}"
    );
}

#[test]
fn a_batch_boundary_runs_the_finish_validations() {
    // The FIRST batch is the broken one (a question with no
    // passage); the boundary — not the end of the stream — must
    // catch it.
    let error = parse_stream(std::io::Cursor::new(format!(
        "{HEADER}\n\
         {{\"paragraph\": 0, \"question\": \"何?\"}}\n\
         {{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}}\n\
         {{\"passage\": \"本文。\"}}\n"
    )))
    .unwrap_err();
    assert!(error.contains("question"), "{error}");
}

#[test]
fn parse_batch_refuses_a_multi_batch_stream() {
    let error = parse(&format!(
        "{HEADER}\n\
         {{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}}\n"
    ))
    .unwrap_err();
    assert!(error.contains("exactly one"), "{error}");
}

#[test]
fn duplicate_aliases_and_second_passages_are_refused() {
    let error = parse(&format!(
        "{HEADER}\n\
         {{\"alias\": \"a\", \"canonical\": \"x\", \"kind\": \"concept\"}}\n\
         {{\"alias\": \"a\", \"canonical\": \"y\", \"kind\": \"concept\"}}\n"
    ))
    .unwrap_err();
    assert!(
        error.contains("line 3") && error.contains("twice"),
        "{error}"
    );

    let error = parse(&format!(
        "{HEADER}\n{{\"passage\": \"one\"}}\n{{\"passage\": \"two\"}}\n"
    ))
    .unwrap_err();
    assert!(
        error.contains("line 3") && error.contains("passage"),
        "{error}"
    );
}

/// An empty alias spelling would containment-match every future
/// cue (`str::contains("")` is always true) — the import surface
/// must refuse it just like the HTTP one does.
#[test]
fn empty_alias_spellings_are_refused() {
    for line in [
        "{\"alias\": \"\", \"canonical\": \"x\", \"kind\": \"concept\"}",
        "{\"alias\": \"a\", \"canonical\": \"\", \"kind\": \"label\"}",
    ] {
        let error = parse(&format!("{HEADER}\n{line}\n")).unwrap_err();
        assert!(
            error.contains("line 2") && error.contains("must not be empty"),
            "{error}"
        );
    }
}

/// Empty question or section text is refused like empty names: a
/// question row is embedded verbatim on the next refresh, and
/// providers refuse zero-length input — one empty row would fail
/// its whole chunk (and abandon the pass) on every refresh.
#[test]
fn empty_question_and_section_text_is_refused() {
    for line in [
        "{\"paragraph\": 0, \"question\": \"\"}",
        "{\"paragraph\": 0, \"section\": \"\"}",
    ] {
        let error = parse(&format!("{HEADER}\n{{\"passage\": \"本文。\"}}\n{line}\n")).unwrap_err();
        assert!(
            error.contains("line 3") && error.contains("must not be empty"),
            "{error}"
        );
    }
}

/// An empty context name would `file_stem` to a bare `.ctx` the
/// server's directory scan never rediscovers; an empty source name
/// has no identity to retract a re-import against. Both are refused
/// at the header, each naming its own field.
#[test]
fn an_empty_context_or_source_name_in_the_header_is_refused() {
    for (field, header) in [
        (
            "context",
            r#"{"taguru_batch": 1, "context": "", "source": "s"}"#,
        ),
        (
            "source",
            r#"{"taguru_batch": 1, "context": "c", "source": ""}"#,
        ),
    ] {
        let error = parse(header).unwrap_err();
        assert!(
            error.contains(field) && error.contains("must not be empty"),
            "{field}: {error}"
        );
    }
}

#[test]
fn group_records_ride_a_stream_and_stand_alone() {
    let stream = parse_stream(std::io::Cursor::new(format!(
        "{HEADER}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n\
         {{\"taguru_group\": 1, \"name\": \"kura\", \"description\": \"蔵\", \
           \"contexts\": [\"sake\", \"sake\"], \"groups\": [\"kid\"]}}\n\
         {{\"taguru_group\": 1, \"name\": \"kid\"}}\n"
    )))
    .unwrap();
    assert_eq!(stream.batches.len(), 1);
    assert_eq!(stream.groups.len(), 2);
    let (name, record) = &stream.groups[0];
    assert_eq!(name, "kura");
    assert_eq!(record.description, "蔵");
    // List duplicates fold into the set — membership IS a set,
    // exactly as over the API.
    assert_eq!(record.contexts.len(), 1);
    assert_eq!(record.groups.len(), 1);
    // Absent fields read as empty, the shape export omits.
    assert_eq!(stream.groups[1].1, GroupRecord::default());

    // A group record closes the batch before it: an op line after
    // one has no batch to join.
    let error = parse_stream(std::io::Cursor::new(format!(
        "{HEADER}\n\
         {{\"taguru_group\": 1, \"name\": \"kura\"}}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n"
    )))
    .unwrap_err();
    assert!(
        error.contains("line 3") && error.contains("not a batch header"),
        "{error}"
    );

    // A groups-only stream is a legitimate restore; an empty one is
    // still a mistake.
    let alone = parse_stream(std::io::Cursor::new(
        "{\"taguru_group\": 1, \"name\": \"kura\"}\n",
    ))
    .unwrap();
    assert!(alone.batches.is_empty());
    assert_eq!(alone.groups.len(), 1);
    assert!(
        parse_stream(std::io::Cursor::new("\n"))
            .unwrap_err()
            .contains("group record")
    );
}

#[test]
fn group_records_validate_their_shape_with_line_numbers() {
    let case = |line: &str| parse_stream(std::io::Cursor::new(format!("{line}\n"))).unwrap_err();
    assert!(case("{\"taguru_group\": 2, \"name\": \"g\"}").contains("taguru_group 2"));
    assert!(case("{\"taguru_group\": 1, \"name\": \"\"}").contains("must not be empty"));
    assert!(case("{\"taguru_group\": 1, \"name\": \"g\", \"nope\": 1}").contains("unknown field"));
    let long = "x".repeat(65);
    assert!(case(&format!("{{\"taguru_group\": 1, \"name\": \"{long}\"}}")).contains("65 bytes"));
    assert!(
        case(&format!(
            "{{\"taguru_group\": 1, \"name\": \"g\", \"contexts\": [\"{long}\"]}}"
        ))
        .contains("65 bytes")
    );

    // Restating one group refuses the whole stream, by line.
    let error = parse_stream(std::io::Cursor::new(
        "{\"taguru_group\": 1, \"name\": \"g\"}\n{\"taguru_group\": 1, \"name\": \"g\"}\n",
    ))
    .unwrap_err();
    assert!(
        error.contains("line 2") && error.contains("one record owns one group's truth"),
        "{error}"
    );

    // The member cap judges the SET: one name past it refuses.
    let over_set: String = (0..=MAX_GROUP_MEMBERS)
        .map(|i| format!("\"c{i:04}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let error = case(&format!(
        "{{\"taguru_group\": 1, \"name\": \"g\", \"contexts\": [{over_set}]}}"
    ));
    assert!(error.contains("split into nested child groups"), "{error}");
}

/// The single-batch entrance (`taguru extract` re-validating its
/// own output) never carries group records.
#[test]
fn parse_batch_refuses_group_records() {
    let error = parse(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\"}\n\
         {\"taguru_group\": 1, \"name\": \"kura\"}\n",
    )
    .unwrap_err();
    assert!(error.contains("exactly one batch was expected"), "{error}");
}

const SCHEMA_LINE: &str = r#"{"taguru_schema": 1, "context": "sake", "mode": "warn", "closed_labels": false, "types": {}, "relations": {}}"#;

/// `taguru_schema` records ride a stream and stand alone — the
/// schema twin of [`group_records_ride_a_stream_and_stand_alone`]
/// (ADR 0009 §13). A schema record closes the batch before it, an
/// op line after one has no batch to join, a schema-only stream is
/// a legitimate restore, and the empty-stream message now names
/// all three record kinds.
#[test]
fn schema_records_ride_a_stream_and_stand_alone() {
    let stream = parse_stream(std::io::Cursor::new(format!(
        "{HEADER}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n\
         {SCHEMA_LINE}\n\
         {{\"taguru_group\": 1, \"name\": \"kid\"}}\n"
    )))
    .unwrap();
    assert_eq!(stream.batches.len(), 1);
    assert_eq!(stream.schemas.len(), 1);
    assert_eq!(stream.groups.len(), 1);
    let (context, installed) = &stream.schemas[0];
    assert_eq!(context, "sake");
    assert_eq!(installed.document().mode, crate::schema::SchemaMode::Warn);

    // A schema record closes the batch before it: an op line after
    // one has no batch to join.
    let error = parse_stream(std::io::Cursor::new(format!(
        "{HEADER}\n\
         {SCHEMA_LINE}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n"
    )))
    .unwrap_err();
    assert!(
        error.contains("line 3") && error.contains("not a batch header"),
        "{error}"
    );

    // A schemas-only stream is a legitimate restore.
    let alone = parse_stream(std::io::Cursor::new(format!("{SCHEMA_LINE}\n"))).unwrap();
    assert!(alone.batches.is_empty());
    assert_eq!(alone.schemas.len(), 1);

    // The empty-stream message now names every record kind.
    let error = parse_stream(std::io::Cursor::new("\n")).unwrap_err();
    assert!(error.contains("schema record"), "{error}");
    assert!(error.contains("group record"), "{error}");
}

/// The version-refusal wording, `deny_unknown_fields`, a missing
/// field, an empty context, a cross-record duplicate, and a
/// `schema::install`-level violation — the schema twin of
/// [`group_records_validate_their_shape_with_line_numbers`].
#[test]
fn schema_records_validate_their_shape_with_line_numbers() {
    let case = |line: &str| parse_stream(std::io::Cursor::new(format!("{line}\n"))).unwrap_err();

    // parse_group's exact wording shape (ADR 0009 §13 bullet 4).
    assert!(
        case(
            r#"{"taguru_schema": 2, "context": "sake", "mode": "off", "closed_labels": false, "types": {}, "relations": {}}"#
        )
        .contains("taguru_schema 2 is not a version this taguru reads (it reads 1)")
    );

    assert!(
        case(
            r#"{"taguru_schema": 1, "context": "", "mode": "off", "closed_labels": false, "types": {}, "relations": {}}"#
        )
        .contains("must not be empty")
    );

    // The context name's own byte cap — mirrors
    // `group_records_validate_their_shape_with_line_numbers`'s
    // `long` case for a group's `name`.
    let long = "x".repeat(65);
    assert!(
        case(&format!(
            r#"{{"taguru_schema": 1, "context": "{long}", "mode": "off", "closed_labels": false, "types": {{}}, "relations": {{}}}}"#
        ))
        .contains("65 bytes")
    );

    // Every field is required — no struct-level default, matching
    // SchemaDocument's own at-rest posture.
    assert!(
        case(r#"{"taguru_schema": 1, "context": "sake", "mode": "off"}"#).contains("missing field")
    );

    assert!(
        case(
            r#"{"taguru_schema": 1, "context": "sake", "mode": "off", "closed_labels": false, "types": {}, "relations": {}, "nope": 1}"#
        )
        .contains("unknown field")
    );

    // A structural violation `schema::install` itself catches
    // (here: the relation named the reserved type label) surfaces
    // with the line number, not just the bare violation text.
    let error = case(
        r#"{"taguru_schema": 1, "context": "sake", "mode": "off", "closed_labels": false, "types": {}, "relations": {"schema:type": {}}}"#,
    );
    assert!(
        error.contains("line 1") && error.contains("reserved"),
        "{error}"
    );

    // Restating one context's schema refuses the whole stream, by
    // line — mirrors a group record's own duplicate refusal.
    let error = parse_stream(std::io::Cursor::new(format!(
        "{SCHEMA_LINE}\n{SCHEMA_LINE}\n"
    )))
    .unwrap_err();
    assert!(
        error.contains("line 2") && error.contains("one record owns one context's schema"),
        "{error}"
    );
}

/// The single-batch entrance (`taguru extract` re-validating its
/// own output) never carries schema records either.
#[test]
fn parse_batch_refuses_schema_records() {
    let error = parse(&format!("{HEADER}\n{SCHEMA_LINE}\n")).unwrap_err();
    assert!(
        error.contains("schema record for context 'sake'")
            && error.contains("exactly one batch was expected"),
        "{error}"
    );
}

/// [`apply_schema_record`]'s own failure path, not just
/// `parse_schema`'s validation: a schema record naming a context
/// neither an earlier batch of the same stream nor a previous
/// request ever created returns [`SchemaApplyError::NoContext`] —
/// the CLI-specific arm `run_local`'s Pass 2 counts into
/// `schema_failures` (exit 1), and the server twin
/// `schema_import_refusal` maps to 404 `no_context`.
#[test]
fn apply_schema_record_refuses_a_context_that_does_not_exist() {
    let dir = std::env::temp_dir().join(format!(
        "taguru-ingest-schema-no-context-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

    let installed = schema::install(schema::SchemaDocument {
        schema: schema::SCHEMA_VERSION,
        mode: schema::SchemaMode::Off,
        closed_labels: false,
        types: BTreeMap::new(),
        relations: BTreeMap::new(),
    })
    .unwrap();
    let error = apply_schema_record(&state, "ghost", installed).unwrap_err();
    assert!(matches!(error, SchemaApplyError::NoContext), "{error:?}");

    let _ = fs::remove_dir_all(&dir);
}

/// A line longer than the cap is refused at the cap, not buffered
/// whole first: the bounded reader stops one byte past the ceiling,
/// so a malicious 100 MiB line cannot force a 100 MiB allocation
/// before the length check runs.
#[test]
fn a_line_past_the_byte_cap_is_refused_without_buffering_it_whole() {
    let giant = "x".repeat(MAX_LINE_BYTES + 1);
    let error = parse(&format!("{HEADER}\n{giant}")).unwrap_err();
    assert!(error.contains("line cap"), "{error}");
}

/// Source metadata (#167) rides the passage line; a pre-metadata
/// line still parses (all three fields default), and the tag
/// vocabulary is the same one the HTTP store enforces.
#[test]
fn passage_line_metadata_parses_validates_and_defaults() {
    let batch = parse(&format!(
        "{HEADER}\n\
         {{\"passage\": \"本文。\", \"stored_at\": 1700000000, \"date\": 1000, \
          \"tags\": [\"酒\", \"蔵\"]}}\n"
    ))
    .unwrap();
    assert_eq!(batch.stored_at, Some(1_700_000_000));
    assert_eq!(batch.date, Some(1_000));
    assert_eq!(batch.tags, vec!["酒".to_string(), "蔵".to_string()]);

    let bare = parse(&format!("{HEADER}\n{{\"passage\": \"本文。\"}}\n")).unwrap();
    assert_eq!(
        (bare.stored_at, bare.date, bare.tags.len()),
        (None, None, 0)
    );

    let empty = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\", \"tags\": [\"\"]}}\n"
    ))
    .unwrap_err();
    assert!(empty.contains("tag"), "{empty}");
    let oversized = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\", \"tags\": [\"{}\"]}}\n",
        "t".repeat(crate::api::MAX_TAG_BYTES + 1)
    ))
    .unwrap_err();
    assert!(oversized.contains("exceeds"), "{oversized}");
    let too_many = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\", \"tags\": [{}]}}\n",
        (0..=crate::api::MAX_TAGS_PER_SOURCE)
            .map(|i| format!("\"t{i}\""))
            .collect::<Vec<_>>()
            .join(", ")
    ))
    .unwrap_err();
    assert!(too_many.contains("at most"), "{too_many}");
}

#[test]
fn a_question_line_rides_the_batch_and_needs_a_passage_to_attach_to() {
    let batch = parse(&format!(
        "{HEADER}\n\
         {{\"passage\": \"一つ目。\\n\\n二つ目。\"}}\n\
         {{\"paragraph\": 1, \"question\": \"二つ目は何?\"}}\n"
    ))
    .unwrap();
    assert_eq!(batch.questions, vec![(1, "二つ目は何?".to_string())]);
    assert!(
        batch.describe().contains("1 question(s)"),
        "{}",
        batch.describe()
    );

    // The same question line without a passage has nothing to name.
    let error = parse(&format!(
        "{HEADER}\n{{\"paragraph\": 1, \"question\": \"二つ目は何?\"}}\n"
    ))
    .unwrap_err();
    assert!(error.contains("no passage line"), "{error}");
}

#[test]
fn more_than_the_per_paragraph_question_cap_in_one_file_is_refused() {
    let questions: String = (0..=crate::api::MAX_QUESTIONS_PER_PARAGRAPH)
        .map(|i| format!("{{\"paragraph\": 0, \"question\": \"言い換え{i}?\"}}\n"))
        .collect();
    let error = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\"}}\n{questions}"
    ))
    .unwrap_err();
    assert!(
        error.contains("already carries") && error.contains("the cap"),
        "{error}"
    );

    let long = "q".repeat(crate::api::MAX_QUESTION_BYTES + 1);
    let error = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\"}}\n{{\"paragraph\": 0, \"question\": \"{long}\"}}\n"
    ))
    .unwrap_err();
    assert!(
        error.contains("question") && error.contains("cap"),
        "{error}"
    );
}

/// A doc2query generator repeating itself, or a batch author pasting
/// the same line twice, must not burn two of the paragraph's capped
/// slots on text that says nothing new — it folds into one entry,
/// matching the group-list dedup elsewhere in this file.
#[test]
fn a_repeated_question_on_the_same_paragraph_folds_into_one_entry() {
    let batch = parse(&format!(
        "{HEADER}\n\
         {{\"passage\": \"本文。\"}}\n\
         {{\"paragraph\": 0, \"question\": \"何?\"}}\n\
         {{\"paragraph\": 0, \"question\": \"何?\"}}\n"
    ))
    .unwrap();
    assert_eq!(batch.questions, vec![(0, "何?".to_string())]);

    // The repeat must not spend one of the paragraph's capped slots
    // either: MAX_QUESTIONS_PER_PARAGRAPH distinct questions plus one
    // repeat of the first must still fit under the cap.
    let distinct: String = (0..crate::api::MAX_QUESTIONS_PER_PARAGRAPH)
        .map(|i| format!("{{\"paragraph\": 0, \"question\": \"言い換え{i}?\"}}\n"))
        .collect();
    let batch = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\"}}\n{distinct}\
         {{\"paragraph\": 0, \"question\": \"言い換え0?\"}}\n"
    ))
    .unwrap();
    assert_eq!(
        batch.questions.len(),
        crate::api::MAX_QUESTIONS_PER_PARAGRAPH
    );
}

#[test]
fn a_section_line_rides_the_batch_and_needs_a_passage_to_attach_to() {
    let batch = parse(&format!(
        "{HEADER}\n\
         {{\"passage\": \"導入。\\n\\n本編。\"}}\n\
         {{\"paragraph\": 1, \"section\": \"本編\"}}\n"
    ))
    .unwrap();
    assert_eq!(batch.sections, vec![(1, "本編".to_string())]);
    assert!(
        batch.describe().contains("1 section(s)"),
        "{}",
        batch.describe()
    );

    // The same section line without a passage has nothing to name.
    let error = parse(&format!(
        "{HEADER}\n{{\"paragraph\": 1, \"section\": \"本編\"}}\n"
    ))
    .unwrap_err();
    assert!(error.contains("no passage line"), "{error}");
}

#[test]
fn a_locator_line_rides_the_batch_and_needs_a_passage_to_attach_to() {
    let batch = parse(&format!(
        "{HEADER}\n\
         {{\"passage\": \"導入。\\n\\n本編。\"}}\n\
         {{\"paragraph\": 1, \"locator\": {{\"kind\": \"page\", \"value\": \"12\"}}}}\n"
    ))
    .unwrap();
    assert_eq!(
        batch.locators,
        vec![(
            1,
            crate::passages::Locator {
                kind: "page".to_string(),
                value: "12".to_string(),
            }
        )]
    );
    assert!(
        batch.describe().contains("1 locator(s)"),
        "{}",
        batch.describe()
    );

    // The same locator line without a passage has nothing to name.
    let error = parse(&format!(
        "{HEADER}\n{{\"paragraph\": 1, \"locator\": {{\"kind\": \"page\", \"value\": \"12\"}}}}\n"
    ))
    .unwrap_err();
    assert!(error.contains("no passage line"), "{error}");
}

#[test]
fn an_association_with_a_paragraph_needs_a_passage_to_attach_to() {
    // A paragraph locator on an association resolves against THIS
    // batch's passage, so it parses fine when the passage is present.
    let batch = parse(&format!(
        "{HEADER}\n\
         {{\"passage\": \"導入。\\n\\n本編。\"}}\n\
         {{\"subject\": \"青嶺酒造\", \"label\": \"創業年\", \"object\": \"1907年\", \"weight\": 1.0, \"paragraph\": 1}}\n"
    ))
    .unwrap();
    assert_eq!(batch.associations[0].paragraph, Some(1));

    // The same locator with no passage line has nothing to name, and
    // apply retracts the source first — so it must be refused rather
    // than persisted into a passage that will not exist.
    let error = parse(&format!(
        "{HEADER}\n\
         {{\"subject\": \"青嶺酒造\", \"label\": \"創業年\", \"object\": \"1907年\", \"weight\": 1.0, \"paragraph\": 1}}\n"
    ))
    .unwrap_err();
    assert!(error.contains("no passage line"), "{error}");

    // A plain association (no locator) still stands on its own.
    parse(&format!(
        "{HEADER}\n\
         {{\"subject\": \"青嶺酒造\", \"label\": \"創業年\", \"object\": \"1907年\", \"weight\": 1.0}}\n"
    ))
    .unwrap();
}

#[test]
fn report_surfaces_a_dropped_passage_that_was_not_replaced() {
    let batch = parse(HEADER).unwrap();

    // A passage was retracted and the batch brought no replacement:
    // the loss must show in the report, never vanish silently.
    let dropped = Applied {
        created: false,
        retracted: 3,
        associations: 0,
        aliases: 0,
        passage_stored: false,
        passage_dropped: true,
        questions_stored: 0,
        questions_dropped: 0,
        sections_stored: 0,
        sections_dropped: 0,
        locators_stored: 0,
        locators_dropped: 0,
        association_paragraphs_dropped: 0,
        schema_violations: 0,
        schema_issues: Vec::new(),
    };
    let line = report(&batch, &dropped);
    assert!(line.contains("previous passage dropped"), "{line}");

    // A batch that carries a replacement reads as a store, not a
    // drop, even though the prior passage was removed to make room.
    let replaced = Applied {
        passage_stored: true,
        ..dropped
    };
    let line = report(&batch, &replaced);
    assert!(line.contains("passage stored"), "{line}");
    assert!(!line.contains("dropped"), "{line}");
}

#[test]
fn a_section_beyond_the_byte_cap_is_refused() {
    let long = "s".repeat(crate::api::MAX_SECTION_BYTES + 1);
    let error = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\"}}\n{{\"paragraph\": 0, \"section\": \"{long}\"}}\n"
    ))
    .unwrap_err();
    assert!(
        error.contains("section") && error.contains("cap"),
        "{error}"
    );
}

#[test]
fn a_malformed_section_line_is_refused_by_line_number() {
    let error = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\"}}\n{{\"paragraph\": \"zero\", \"section\": \"見出し\"}}\n"
    ))
    .unwrap_err();
    assert!(
        error.contains("line 3") && error.contains("section"),
        "{error}"
    );
}

#[test]
fn a_locator_value_beyond_the_byte_cap_is_refused() {
    let long = "s".repeat(crate::api::MAX_LOCATOR_VALUE_BYTES + 1);
    let error = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\"}}\n\
         {{\"paragraph\": 0, \"locator\": {{\"kind\": \"page\", \"value\": \"{long}\"}}}}\n"
    ))
    .unwrap_err();
    assert!(
        error.contains("locator.value") && error.contains("cap"),
        "{error}"
    );
}

#[test]
fn a_locator_kind_beyond_the_byte_cap_is_refused() {
    let long = "k".repeat(crate::api::MAX_LOCATOR_KIND_BYTES + 1);
    let error = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\"}}\n\
         {{\"paragraph\": 0, \"locator\": {{\"kind\": \"{long}\", \"value\": \"1\"}}}}\n"
    ))
    .unwrap_err();
    assert!(
        error.contains("locator.kind") && error.contains("cap"),
        "{error}"
    );
}

#[test]
fn an_empty_locator_kind_or_value_is_refused() {
    for locator in [
        r#"{"kind": "", "value": "1"}"#,
        r#"{"kind": "page", "value": ""}"#,
    ] {
        let error = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\"}}\n\
             {{\"paragraph\": 0, \"locator\": {locator}}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("line 3") && error.contains("must not be empty"),
            "{error}"
        );
    }
}

#[test]
fn a_malformed_locator_line_is_refused_by_line_number() {
    let error = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\"}}\n\
         {{\"paragraph\": \"zero\", \"locator\": {{\"kind\": \"page\", \"value\": \"1\"}}}}\n"
    ))
    .unwrap_err();
    assert!(
        error.contains("line 3") && error.contains("locator"),
        "{error}"
    );
}

#[test]
fn a_locator_line_with_an_unknown_field_is_refused() {
    let error = parse(&format!(
        "{HEADER}\n{{\"passage\": \"本文。\"}}\n\
         {{\"paragraph\": 0, \"locator\": {{\"kind\": \"page\", \"value\": \"1\", \"page\": 1}}}}\n"
    ))
    .unwrap_err();
    assert!(error.contains("line 3"), "{error}");
}

#[test]
fn weights_and_name_sizes_are_capped_like_the_api() {
    let error = parse(&format!(
        "{HEADER}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1e300}}\n"
    ))
    .unwrap_err();
    assert!(
        error.contains("line 2") && error.contains("weight"),
        "{error}"
    );

    let long = "x".repeat(MAX_NAME_BYTES + 1);
    let error = parse(&format!(
        "{HEADER}\n\
         {{\"subject\": \"{long}\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n"
    ))
    .unwrap_err();
    assert!(
        error.contains("line 2") && error.contains("subject"),
        "{error}"
    );

    let error = parse(&format!(
        "{{\"taguru_batch\": 1, \"context\": \"{}\", \"source\": \"s\"}}\n",
        "c".repeat(MAX_CONTEXT_NAME_BYTES + 1)
    ))
    .unwrap_err();
    assert!(error.contains("context"), "{error}");
}

#[test]
fn empty_subject_label_or_object_is_refused() {
    for (field, line) in [
        (
            "subject",
            r#"{"subject": "", "label": "l", "object": "b", "weight": 1.0}"#,
        ),
        (
            "label",
            r#"{"subject": "a", "label": "", "object": "b", "weight": 1.0}"#,
        ),
        (
            "object",
            r#"{"subject": "a", "label": "l", "object": "", "weight": 1.0}"#,
        ),
    ] {
        let error = parse(&format!("{HEADER}\n{line}\n")).unwrap_err();
        assert!(
            error.contains("line 2") && error.contains(field) && error.contains("empty"),
            "{field}: {error}"
        );
    }

    // Every field non-empty still parses fine.
    let batch = parse(&format!(
        "{HEADER}\n{{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n"
    ))
    .unwrap();
    assert_eq!(batch.associations.len(), 1);
}

#[test]
fn a_line_that_is_no_known_shape_names_the_known_shapes() {
    let error = parse(&format!("{HEADER}\n{{\"foo\": 1}}\n")).unwrap_err();
    assert!(
        error.contains("line 2") && error.contains("association"),
        "{error}"
    );
}

/// The batch-open marker around `apply_batch`'s four mutations:
/// absent after success, never opened at all for a batch whose
/// alias step is predicted to fail before anything runs, gone
/// again once the documented repair — re-importing the source —
/// completes. A marker surviving a genuine mid-batch refusal (one
/// prediction cannot catch, e.g. a disk fault) is covered
/// separately by
/// [`apply_batch_refuses_when_an_unreplaced_passage_cannot_be_retracted`].
#[test]
fn apply_batch_brackets_its_steps_with_the_import_marker() {
    let dir = std::env::temp_dir().join(format!("taguru-ingest-marker-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

    // A completed batch leaves no marker: its truth is fully on disk.
    let happy = parse(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \"create\": {}}\n\
         {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n",
    )
    .unwrap();
    apply_batch(&state, &happy).unwrap();
    assert!(
        crate::registry::import_marker_paths(&dir, "sake").is_empty(),
        "a completed batch clears its marker"
    );

    // A batch whose alias step is predicted to fail is refused
    // before anything runs: no marker opens for it to keep. (An
    // alias to a canonical nothing interned — the same rejection
    // `add_alias` would raise for real, just caught here first.)
    let torn = parse(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}\n\
         {\"alias\": \"Aomine\", \"canonical\": \"存在しない\", \"kind\": \"concept\"}\n",
    )
    .unwrap();
    let refusal = apply_batch(&state, &torn).unwrap_err();
    assert!(matches!(refusal, ApplyRefusal::Rejected(_)));
    assert_eq!(
        crate::registry::import_marker_paths(&dir, "sake").len(),
        0,
        "a predicted rejection opens no marker"
    );

    // A corrected batch for the same source applies cleanly —
    // there was never a tear to repair, just a rejected batch
    // that nothing depended on.
    let fixed = parse(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}\n\
         {\"subject\": \"青嶺酒造\", \"label\": \"銘柄\", \"object\": \"青嶺\", \"weight\": 1.0}\n\
         {\"alias\": \"Aomine\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n",
    )
    .unwrap();
    apply_batch(&state, &fixed).unwrap();
    assert!(
        crate::registry::import_marker_paths(&dir, "sake").is_empty(),
        "a normal import leaves no marker"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// An associations-only re-import (no passage line in this batch)
/// for a source that already has one on disk: the differential
/// sync still retracts that old passage first, same as any other
/// batch. If the retraction genuinely fails to remove it — not
/// "there was nothing to remove" — nothing later in this batch
/// will ever overwrite the stale copy, so the batch must refuse
/// and keep its marker rather than clear it over a source whose
/// truth is now half-applied.
#[test]
fn apply_batch_refuses_when_an_unreplaced_passage_cannot_be_retracted() {
    let mut exhausted = false;
    let mut saw_the_refusal = false;
    for failure in 0..24 {
        let dir = std::env::temp_dir().join(format!(
            "taguru-ingest-marker-passage-fault-{failure}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        let seeded = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \"create\": {}}\n\
             {\"passage\": \"杜氏は高瀬。\"}\n\
             {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n",
        )
        .unwrap();
        apply_batch(&state, &seeded).unwrap();

        let reimport = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\"}\n\
             {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬2\", \"weight\": 1.0}\n",
        )
        .unwrap();

        crate::storage::fail_persistence_ops_after(failure);
        let result = apply_batch(&state, &reimport);
        let past_end = crate::storage::clear_persistence_fault();

        if let Err(ApplyRefusal::Io(message)) = &result
            && message.contains("could not be retracted")
        {
            saw_the_refusal = true;
            assert_eq!(
                crate::registry::import_marker_paths(&dir, "sake").len(),
                1,
                "step {failure}: refusing to retract an unreplaced passage still \
                 cleared the marker"
            );
            // The documented repair still converges: retrying the
            // same associations-only batch re-attempts the
            // retraction (idempotent per-source) with the fault
            // now cleared.
            apply_batch(&state, &reimport).unwrap();
            assert!(
                crate::registry::import_marker_paths(&dir, "sake").is_empty(),
                "step {failure}: repair did not clear the marker"
            );
        }

        drop(state);
        let _ = fs::remove_dir_all(&dir);

        if past_end {
            exhausted = true;
            break;
        }
    }
    assert!(
        exhausted,
        "sweep bound too small to reach past every persistence step"
    );
    assert!(
        saw_the_refusal,
        "the sweep never reached the passage-retract fault point"
    );
}

/// `Applied::passage_dropped` is documented as "retracted AND no
/// replacement carried" — `preview_batch` implements exactly that
/// AND, so a routine re-import that supplies a replacement passage
/// must report `passage_dropped: false` from both entrances alike.
#[test]
fn apply_and_preview_agree_that_a_replaced_passage_is_not_dropped() {
    let dir = std::env::temp_dir().join(format!(
        "taguru-ingest-passage-replace-parity-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

    let seeded = parse(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \"create\": {}}\n\
         {\"passage\": \"杜氏は高瀬。\"}\n",
    )
    .unwrap();
    apply_batch(&state, &seeded).unwrap();

    let reimport = parse(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\"}\n\
         {\"passage\": \"杜氏は高瀬二代目。\"}\n",
    )
    .unwrap();

    let previewed = preview_batch(&state, &reimport).unwrap();
    assert!(
        !previewed.passage_dropped,
        "preview: a replacement passage was carried, so nothing was dropped"
    );

    let applied = apply_batch(&state, &reimport).unwrap();
    assert!(
        !applied.passage_dropped,
        "apply: a replacement passage was carried, so nothing was dropped, \
         matching preview_batch's own report for the identical batch"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// A batch that pairs a valid association with a conflicting
/// alias, aimed at a context that does not exist yet, is refused
/// before the association ever lands: predicting the alias step's
/// outcome up front means a batch that would otherwise write the
/// association and only then fail on its alias no longer gets to
/// write anything at all — not even the context it would have
/// created.
#[test]
fn a_predicted_alias_rejection_creates_nothing_and_applies_nothing() {
    let dir = std::env::temp_dir().join(format!(
        "taguru-ingest-predicted-rejection-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

    let torn = parse(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \"create\": {}}\n\
         {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n\
         {\"alias\": \"Aomine\", \"canonical\": \"存在しない\", \"kind\": \"concept\"}\n",
    )
    .unwrap();
    let refusal = apply_batch(&state, &torn).unwrap_err();
    assert!(
        matches!(refusal, ApplyRefusal::Rejected(_)),
        "expected a predicted rejection, got {refusal:?}"
    );
    assert!(!refusal.wrote_anything());
    assert_eq!(refusal.ops_written(), 0);
    assert!(
        state.directory_entry("sake").is_none(),
        "a predicted rejection must not create the context the batch named"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn ops_written_counts_only_the_partial_refusal() {
    assert_eq!(ApplyRefusal::NoContext("sake".to_string()).ops_written(), 0);
    assert_eq!(ApplyRefusal::Io("boom".to_string()).ops_written(), 0);
    assert_eq!(ApplyRefusal::Access(AccessError::NotFound).ops_written(), 0);
    assert_eq!(
        ApplyRefusal::Partial {
            applied: 5,
            message: "boom".to_string(),
            full: false,
        }
        .ops_written(),
        5
    );
    assert_eq!(
        ApplyRefusal::Rejected(AliasRejection {
            namespace: AliasNamespace::Concept,
            alias: "a".to_string(),
            canonical: "c".to_string(),
            error: AliasError::UnknownCanonical,
        })
        .ops_written(),
        0
    );
}

/// Move one deterministic filesystem failure through the complete
/// import: marker, source retraction, passage store, associations,
/// aliases, and marker unlink. A stopped batch keeps its marker;
/// a failure before the marker applies nothing; and any swallowed
/// best-effort failure must still leave a complete, retryable truth.
#[test]
fn every_import_persistence_failure_is_detected_or_fully_repaired() {
    let mut exhausted = false;
    for failure in 0..24 {
        let dir = std::env::temp_dir().join(format!(
            "taguru-ingest-fault-{failure}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let batch = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\"}\n\
             {\"passage\": \"青嶺酒造の杜氏は高瀬。\"}\n\
             {\"subject\": \"青嶺酒造\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n\
             {\"alias\": \"青嶺\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n",
        )
        .unwrap();

        crate::storage::fail_persistence_ops_after(failure);
        let first = apply_batch(&state, &batch);
        let past_end = crate::storage::clear_persistence_fault();
        let marker = crate::registry::import_marker_path(&dir, "sake", "doc-1");

        if past_end {
            assert!(
                first.is_ok(),
                "the past-end attempt must complete: {first:?}"
            );
            assert!(!marker.exists());
        } else {
            if let Err(refusal) = &first {
                let before_marker = refusal.text().contains("marker not persisted");
                assert_eq!(
                    marker.exists(),
                    !before_marker,
                    "a stopped batch at step {failure} lost its tear witness: {refusal:?}"
                );
                if let ApplyRefusal::Partial { applied, .. } = refusal {
                    assert_eq!(
                        refusal.ops_written(),
                        *applied,
                        "step {failure}: ops_written must mirror the partial \
                         refusal's own running total"
                    );
                }
            }
            // Re-import is the documented repair and is exact even
            // when the injected error was swallowed after a fully
            // superseding write or only prevented marker cleanup.
            apply_batch(&state, &batch).unwrap();
            assert!(
                !marker.exists(),
                "repair did not clear failure step {failure}"
            );
        }

        assert_eq!(
            state
                .read_context("sake", |context| context.association_count())
                .unwrap(),
            1,
            "retry at step {failure} was not idempotent"
        );
        assert_eq!(
            state
                .read_context("sake", |context| context.resolve("青嶺")[0].name.clone())
                .unwrap(),
            "青嶺酒造",
            "alias step {failure} did not land"
        );
        assert_eq!(
            state
                .lookup_passages("sake", &["doc-1".to_string()])
                .unwrap()
                .unwrap()
                .0["doc-1"],
            "青嶺酒造の杜氏は高瀬。"
        );
        drop(state);
        let _ = fs::remove_dir_all(&dir);
        if past_end {
            exhausted = true;
            break;
        }
    }
    assert!(exhausted, "import exceeded the persistence sweep bound");
}

#[test]
fn directories_expand_to_their_sorted_jsonl_files() {
    let dir = std::env::temp_dir().join(format!("taguru-ingest-expand-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("b.jsonl"), "x").unwrap();
    fs::write(dir.join("a.jsonl"), "x").unwrap();
    fs::write(dir.join("ignored.txt"), "x").unwrap();
    // A subdirectory that happens to be named like a batch file
    // must never ride along: `fs::File::open` on it would fail
    // with a confusing "Is a directory" error far from here,
    // instead of `expand` just not collecting it in the first
    // place.
    fs::create_dir_all(dir.join("c.jsonl")).unwrap();
    let files = expand(&[dir.to_string_lossy().into_owned()]).unwrap();
    let names: Vec<_> = files
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap())
        .collect();
    assert_eq!(names, ["a.jsonl", "b.jsonl"]);

    let empty = dir.join("empty");
    fs::create_dir_all(&empty).unwrap();
    let error = expand(&[empty.to_string_lossy().into_owned()]).unwrap_err();
    assert!(error.contains("no .jsonl files"), "{error}");
    let _ = fs::remove_dir_all(&dir);
}

fn unit(label: &str, bytes: usize) -> Unit {
    Unit {
        text: "x".repeat(bytes),
        label: label.to_string(),
        kind: UnitKind::Batch,
    }
}

#[test]
fn pack_chunks_fills_each_chunk_up_to_the_budget_without_splitting_a_unit() {
    let units = vec![unit("a", 40), unit("b", 40), unit("c", 40), unit("d", 5)];
    let queue = pack_chunks(units, 100);
    // a+b = 80 (fits); +c = 120 (over 100) → c starts the next
    // chunk; c+d = 45 (fits, and nothing follows).
    let sizes: Vec<usize> = queue.iter().map(Chunk::size).collect();
    assert_eq!(sizes, vec![80, 45], "{sizes:?}");
    assert_eq!(queue[0].units.len(), 2);
    assert_eq!(queue[1].units.len(), 2);
}

#[test]
fn pack_chunks_never_splits_a_single_oversized_unit_across_two_chunks() {
    // A unit alone over budget still rides whole in its own chunk
    // — pack_chunks never refuses; the caller checks this case
    // before packing (run_remote's pre-send hard-error).
    let units = vec![unit("small", 10), unit("huge", 500)];
    let queue = pack_chunks(units, 100);
    assert_eq!(queue.len(), 2);
    assert_eq!(queue[0].units.len(), 1);
    assert_eq!(queue[1].units.len(), 1);
    assert_eq!(queue[1].units[0].label, "huge");
}

#[test]
fn pack_chunks_on_an_empty_input_is_an_empty_queue() {
    assert!(pack_chunks(Vec::new(), 100).is_empty());
}

#[test]
fn chunk_halve_splits_at_the_unit_boundary_closest_to_half_the_bytes() {
    let chunk = Chunk {
        units: vec![unit("one", 10), unit("two", 10), unit("three", 80)],
    };
    let (first, second) = chunk.halve();
    assert_eq!(
        first.units.len(),
        2,
        "{:?}",
        first.units.iter().map(|u| &u.label).collect::<Vec<_>>()
    );
    assert_eq!(second.units.len(), 1);
    assert_eq!(second.units[0].label, "three");
}

#[test]
fn chunk_halve_never_produces_an_empty_half_on_two_units() {
    // Even a lopsided two-unit chunk (one unit far bigger than the
    // other) must still split into one unit per half — the split
    // point is clamped to 1..len-1, never 0 or len.
    let chunk = Chunk {
        units: vec![unit("big", 90), unit("small", 10)],
    };
    let (first, second) = chunk.halve();
    assert_eq!(first.units.len(), 1);
    assert_eq!(second.units.len(), 1);
    assert_eq!(first.units[0].label, "big");
    assert_eq!(second.units[0].label, "small");
}

#[test]
fn chunk_body_guarantees_a_trailing_newline_per_unit() {
    let chunk = Chunk {
        units: vec![
            Unit {
                text: "{\"a\":1}".to_string(),
                label: "no-newline".to_string(),
                kind: UnitKind::Batch,
            },
            Unit {
                text: "{\"b\":2}\n".to_string(),
                label: "has-newline".to_string(),
                kind: UnitKind::Batch,
            },
        ],
    };
    assert_eq!(chunk.body(), "{\"a\":1}\n{\"b\":2}\n");
}

/// [`run_remote`]'s Pass 1 strips a leading BOM before
/// `split_batches` runs — proven here at the byte level, matching
/// the local path's own `a_leading_bom_does_not_break_the_first_line`
/// pin. Without the strip, the BOM's three bytes would ride inside
/// `split_batches`' very first range and be sent to the server as
/// part of the wire chunk.
#[test]
fn a_leading_bom_is_stripped_before_split_batches_runs() {
    let mut bytes = vec![0xEF, 0xBB, 0xBF];
    bytes.extend_from_slice(b"{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s\"}\n");
    assert!(bytes.starts_with(&[0xEF, 0xBB, 0xBF]));
    bytes.drain(0..3);
    let ranges = split_batches(&bytes);
    assert_eq!(ranges.len(), 1);
    let first = std::str::from_utf8(&bytes[ranges[0].clone()]).unwrap();
    assert!(first.starts_with("{\"taguru_batch\""), "{first}");
    assert!(!first.as_bytes().starts_with(&[0xEF, 0xBB, 0xBF]));
}
