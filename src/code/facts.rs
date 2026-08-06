//! Pure fact building: one file's [`SymbolNode`]s in, one batch's
//! worth of lines out — no I/O, no parser dependency, so the whole
//! fact model is unit-testable on synthetic symbols before any real
//! grammar exists.
//!
//! The shape produced here is exactly the batch contract `taguru
//! import` reads (`src/ingest/model.rs`): one file states one
//! source's complete truth, so re-syncing a changed file is the
//! platform's own retract-then-apply differential sync.
//!
//! The passage is NOT the raw file: code bodies contain blank lines,
//! and the paragraph splitter (`src/paragraph.rs`) breaks on them, so
//! "one symbol = one paragraph" cannot survive raw source text.
//! Instead each symbol contributes its one-line signature as one
//! paragraph, in source order — paragraph index i IS symbol index i,
//! every symbol keeps its own `lines` locator and section heading,
//! and BM25 gets the identifier-dense line that matters for lookup.

use crate::code::grammar::SymbolNode;

/// Mirrors of the server's wire caps (`src/api.rs`) — duplicated here
/// so the fact builder stays free of the api module's dependency web;
/// the import path re-validates against the originals, so drift shows
/// up as a refused batch line, never silent corruption.
const MAX_NAME_BYTES: usize = 1024;
const MAX_SECTION_BYTES: usize = 512;

/// `taguru_batch` header version this builder renders
/// (`src/ingest.rs` BATCH_VERSION).
const BATCH_VERSION: u64 = 1;

/// Version of the fact model this builder produces. The sync state
/// records it beside its content fingerprints: a fingerprint says
/// "this file's bytes produced the facts already stored", which stops
/// being true the moment build()/render_batch() change shape — bump
/// this on any such change and every map rebuilds in full instead of
/// serving stale facts behind fresh-looking fingerprints.
pub(crate) const FACTS_VERSION: u32 = 1;

/// One association line's fields, pre-render.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Assoc {
    pub(crate) subject: String,
    pub(crate) label: String,
    pub(crate) object: String,
    pub(crate) weight: f64,
    pub(crate) paragraph: Option<u32>,
}

/// Everything one file contributes: the batch body for its source,
/// plus warnings for symbols the caps forced out (reported, never
/// silently dropped).
#[derive(Debug, Default)]
pub(crate) struct FileFacts {
    /// Repo-relative path — the source id.
    pub(crate) source: String,
    /// One paragraph per kept symbol, blank-line separated; empty
    /// when the file has no symbols (no passage line is rendered).
    pub(crate) passage: String,
    /// (paragraph, section heading) pairs, one per kept symbol.
    pub(crate) sections: Vec<(u32, String)>,
    /// (paragraph, "start-end" line range) pairs, one per kept symbol.
    pub(crate) locators: Vec<(u32, String)>,
    pub(crate) associations: Vec<Assoc>,
    /// Symbols skipped because a rendered name would blow a wire cap.
    pub(crate) warnings: Vec<String>,
}

/// Builds one file's facts. `path` is the repo-relative path (also
/// the source id); `symbols` must be in source order — their order
/// becomes the paragraph numbering.
pub(crate) fn build(path: &str, symbols: &[SymbolNode]) -> FileFacts {
    let mut facts = FileFacts {
        source: path.to_string(),
        ..FileFacts::default()
    };

    // Directory containment: `src contains src/ingest`, `src/ingest
    // contains src/ingest/model.rs`. Every file asserts its own
    // ancestor chain, so the shared edges accumulate one attribution
    // per file and retracting one file only withdraws its share.
    let mut child = path;
    while let Some((parent, _)) = child.rsplit_once('/') {
        facts.associations.push(Assoc {
            subject: parent.to_string(),
            label: "contains".to_string(),
            object: child.to_string(),
            weight: 1.0,
            paragraph: None,
        });
        child = parent;
    }
    facts.associations.reverse(); // root-first reads better in batches

    // An `impl OtherFilesType` scope names a parent this file never
    // defines (`src/context/alias.rs::Context`): without an anchor the
    // methods' container floats free of the tree. The file honestly
    // contains that impl surface, so anchor every such parent once.
    let defined: std::collections::HashSet<&str> = symbols
        .iter()
        .map(|symbol| symbol.qualified_name.as_str())
        .collect();
    let mut anchored = std::collections::HashSet::new();
    for symbol in symbols {
        if let Some(parent) = symbol.parent.as_deref()
            && !defined.contains(parent)
            && parent.len() <= MAX_NAME_BYTES
            && anchored.insert(parent)
        {
            facts.associations.push(Assoc {
                subject: path.to_string(),
                label: "contains".to_string(),
                object: parent.to_string(),
                weight: 1.0,
                paragraph: None,
            });
        }
    }

    let mut paragraphs = Vec::new();
    for symbol in symbols {
        if symbol.qualified_name.len() > MAX_NAME_BYTES {
            facts.warnings.push(format!(
                "{path}: skipped {} — qualified name exceeds {MAX_NAME_BYTES} bytes",
                symbol.name
            ));
            continue;
        }
        let paragraph = paragraphs.len() as u32;

        let mut line = sanitize_line(&symbol.signature);
        if line.is_empty() {
            line = format!("{} {}", symbol.kind.keyword(), symbol.name);
        }
        paragraphs.push(line);

        facts.sections.push((
            paragraph,
            truncate_utf8(
                &format!("{} {}", symbol.kind.keyword(), symbol.name),
                MAX_SECTION_BYTES,
            ),
        ));
        facts.locators.push((
            paragraph,
            format!("{}-{}", symbol.start_line, symbol.end_line),
        ));

        // The container is the enclosing symbol when nested, the file
        // otherwise; `defined_in` always points straight at the file —
        // that direct edge is the "where is X" product. A parent over
        // the wire cap can never be a legal subject (its own symbol
        // was already skipped with a warning above); the file is the
        // honest fallback container, or one pathological name would
        // refuse the whole file's batch.
        let container = symbol
            .parent
            .as_deref()
            .filter(|parent| parent.len() <= MAX_NAME_BYTES)
            .unwrap_or(path);
        facts.associations.push(Assoc {
            subject: container.to_string(),
            label: "contains".to_string(),
            object: symbol.qualified_name.clone(),
            weight: 1.0,
            paragraph: Some(paragraph),
        });
        facts.associations.push(Assoc {
            subject: symbol.qualified_name.clone(),
            label: "defined_in".to_string(),
            object: path.to_string(),
            weight: 1.0,
            paragraph: Some(paragraph),
        });
    }
    facts.passage = paragraphs.join("\n\n");
    facts
}

/// Renders one file's batch as the JSONL `taguru import` reads:
/// header, passage, sections, locators, associations. `create` adds
/// the header's create block so the first sync can mint the context.
pub(crate) fn render_batch(context: &str, facts: &FileFacts, create: Option<&str>) -> String {
    let mut lines = Vec::new();
    let mut header = serde_json::json!({
        "taguru_batch": BATCH_VERSION,
        "context": context,
        "source": facts.source,
    });
    if let Some(description) = create {
        header["create"] = serde_json::json!({ "description": description });
    }
    lines.push(header.to_string());
    if !facts.passage.is_empty() {
        lines.push(serde_json::json!({ "passage": facts.passage }).to_string());
    }
    for (paragraph, section) in &facts.sections {
        lines.push(serde_json::json!({ "paragraph": paragraph, "section": section }).to_string());
    }
    for (paragraph, value) in &facts.locators {
        lines.push(
            serde_json::json!({
                "paragraph": paragraph,
                "locator": { "kind": "lines", "value": value },
            })
            .to_string(),
        );
    }
    for assoc in &facts.associations {
        let mut line = serde_json::json!({
            "subject": assoc.subject,
            "label": assoc.label,
            "object": assoc.object,
            "weight": assoc.weight,
        });
        if let Some(paragraph) = assoc.paragraph {
            line["paragraph"] = serde_json::json!(paragraph);
        }
        lines.push(line.to_string());
    }
    let mut batch = lines.join("\n");
    batch.push('\n');
    batch
}

/// Collapses a signature onto one non-blank line: newlines and runs
/// of whitespace become single spaces. A blank line inside a passage
/// paragraph would split it (`src/paragraph.rs`), silently shifting
/// every later symbol's locator — this is the invariant that keeps
/// paragraph i and symbol i the same thing.
fn sanitize_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Truncates to at most `max` bytes on a char boundary.
fn truncate_utf8(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::grammar::SymbolKind;

    fn symbol(
        kind: SymbolKind,
        name: &str,
        qualified: &str,
        parent: Option<&str>,
        signature: &str,
        lines: (u32, u32),
    ) -> SymbolNode {
        SymbolNode {
            kind,
            name: name.to_string(),
            qualified_name: qualified.to_string(),
            parent: parent.map(str::to_string),
            signature: signature.to_string(),
            start_line: lines.0,
            end_line: lines.1,
        }
    }

    /// The plan's worked example: `pub(crate) fn parse_batch` in
    /// `src/ingest/model.rs`.
    #[test]
    fn build_produces_the_planned_fact_shape() {
        let symbols = [symbol(
            SymbolKind::Function,
            "parse_batch",
            "src/ingest/model.rs::parse_batch",
            None,
            "pub(crate) fn parse_batch(reader: impl BufRead) -> Result<Batch, String>",
            (323, 347),
        )];
        let facts = build("src/ingest/model.rs", &symbols);

        assert_eq!(facts.source, "src/ingest/model.rs");
        assert_eq!(
            facts.passage,
            "pub(crate) fn parse_batch(reader: impl BufRead) -> Result<Batch, String>"
        );
        assert_eq!(facts.sections, vec![(0, "fn parse_batch".to_string())]);
        assert_eq!(facts.locators, vec![(0, "323-347".to_string())]);
        assert_eq!(
            facts.associations,
            vec![
                Assoc {
                    subject: "src".into(),
                    label: "contains".into(),
                    object: "src/ingest".into(),
                    weight: 1.0,
                    paragraph: None,
                },
                Assoc {
                    subject: "src/ingest".into(),
                    label: "contains".into(),
                    object: "src/ingest/model.rs".into(),
                    weight: 1.0,
                    paragraph: None,
                },
                Assoc {
                    subject: "src/ingest/model.rs".into(),
                    label: "contains".into(),
                    object: "src/ingest/model.rs::parse_batch".into(),
                    weight: 1.0,
                    paragraph: Some(0),
                },
                Assoc {
                    subject: "src/ingest/model.rs::parse_batch".into(),
                    label: "defined_in".into(),
                    object: "src/ingest/model.rs".into(),
                    weight: 1.0,
                    paragraph: Some(0),
                },
            ]
        );
        assert!(facts.warnings.is_empty());
    }

    #[test]
    fn nested_symbols_hang_off_their_parent_but_still_point_at_the_file() {
        let symbols = [
            symbol(
                SymbolKind::Struct,
                "Context",
                "src/context.rs::Context",
                None,
                "pub struct Context",
                (10, 40),
            ),
            symbol(
                SymbolKind::Method,
                "resolve",
                "src/context.rs::Context::resolve",
                Some("src/context.rs::Context"),
                "pub fn resolve(&self, cue: &str) -> Vec<Match>",
                (20, 30),
            ),
        ];
        let facts = build("src/context.rs", &symbols);
        let contains: Vec<_> = facts
            .associations
            .iter()
            .filter(|assoc| assoc.label == "contains")
            .map(|assoc| (assoc.subject.as_str(), assoc.object.as_str()))
            .collect();
        assert_eq!(
            contains,
            vec![
                ("src", "src/context.rs"),
                ("src/context.rs", "src/context.rs::Context"),
                (
                    "src/context.rs::Context",
                    "src/context.rs::Context::resolve"
                ),
            ]
        );
        let defined: Vec<_> = facts
            .associations
            .iter()
            .filter(|assoc| assoc.label == "defined_in")
            .map(|assoc| (assoc.subject.as_str(), assoc.object.as_str()))
            .collect();
        assert_eq!(
            defined,
            vec![
                ("src/context.rs::Context", "src/context.rs"),
                ("src/context.rs::Context::resolve", "src/context.rs"),
            ]
        );
    }

    /// The invariant the passage design exists for: whatever a grammar
    /// hands over, no paragraph can ever contain or become a blank
    /// line, so paragraph i and symbol i never drift apart.
    #[test]
    fn passage_paragraphs_can_never_split_on_blank_lines() {
        let symbols = [
            symbol(
                SymbolKind::Function,
                "a",
                "f.rs::a",
                None,
                "fn a(\n\n    x: u32,\n\n) -> u32",
                (1, 5),
            ),
            symbol(SymbolKind::Function, "b", "f.rs::b", None, "   \n ", (6, 7)),
        ];
        let facts = build("f.rs", &symbols);
        let paragraphs: Vec<&str> = facts.passage.split("\n\n").collect();
        assert_eq!(paragraphs, vec!["fn a( x: u32, ) -> u32", "fn b"]);
        for paragraph in paragraphs {
            assert!(!paragraph.trim().is_empty());
            assert!(!paragraph.contains('\n'));
        }
    }

    #[test]
    fn a_file_with_no_symbols_still_asserts_its_directory_chain() {
        let facts = build("docs/site.js", &[]);
        assert!(facts.passage.is_empty());
        assert!(facts.sections.is_empty());
        assert_eq!(
            facts.associations,
            vec![Assoc {
                subject: "docs".into(),
                label: "contains".into(),
                object: "docs/site.js".into(),
                weight: 1.0,
                paragraph: None,
            }]
        );
    }

    #[test]
    fn a_root_level_file_has_no_directory_edges() {
        let facts = build("build.rs", &[]);
        assert!(facts.associations.is_empty());
    }

    #[test]
    fn oversized_qualified_names_are_skipped_with_a_warning_not_silently() {
        let long = "x".repeat(1100);
        let symbols = [
            symbol(
                SymbolKind::Function,
                "long",
                &long,
                None,
                "fn long()",
                (1, 2),
            ),
            symbol(
                SymbolKind::Function,
                "ok",
                "f.rs::ok",
                None,
                "fn ok()",
                (3, 4),
            ),
        ];
        let facts = build("f.rs", &symbols);
        assert_eq!(facts.warnings.len(), 1);
        assert!(facts.warnings[0].contains("long"));
        // The kept symbol renumbers cleanly onto paragraph 0.
        assert_eq!(facts.sections, vec![(0, "fn ok".to_string())]);
        assert_eq!(facts.locators, vec![(0, "3-4".to_string())]);
    }

    /// One pathological parent name must cost only its own symbols,
    /// never the file's whole batch: an over-cap subject would be
    /// refused at the wire and sync aborts on a refused batch.
    #[test]
    fn an_oversized_parent_falls_back_to_the_file_container() {
        let long_parent = format!("f.rs::{}", "x".repeat(1100));
        let symbols = [symbol(
            SymbolKind::Method,
            "child",
            "f.rs::child",
            Some(&long_parent),
            "fn child()",
            (5, 6),
        )];
        let facts = build("f.rs", &symbols);
        let contains: Vec<_> = facts
            .associations
            .iter()
            .filter(|assoc| assoc.label == "contains")
            .map(|assoc| (assoc.subject.as_str(), assoc.object.as_str()))
            .collect();
        assert_eq!(contains, vec![("f.rs", "f.rs::child")]);
        assert!(
            facts
                .associations
                .iter()
                .all(|assoc| assoc.subject.len() <= 1024 && assoc.object.len() <= 1024),
            "no association may carry an over-cap name"
        );
    }

    #[test]
    fn render_batch_emits_the_import_contract_line_shapes() {
        let symbols = [symbol(
            SymbolKind::Function,
            "parse_batch",
            "src/ingest/model.rs::parse_batch",
            None,
            "pub(crate) fn parse_batch(reader: impl BufRead) -> Result<Batch, String>",
            (323, 347),
        )];
        let facts = build("src/ingest/model.rs", &symbols);
        let batch = render_batch("code", &facts, Some("code map"));
        let lines: Vec<serde_json::Value> = batch
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();

        assert_eq!(lines[0]["taguru_batch"], 1);
        assert_eq!(lines[0]["context"], "code");
        assert_eq!(lines[0]["source"], "src/ingest/model.rs");
        assert_eq!(lines[0]["create"]["description"], "code map");
        assert!(lines[1]["passage"].is_string());
        assert_eq!(lines[2]["section"], "fn parse_batch");
        assert_eq!(lines[3]["locator"]["kind"], "lines");
        assert_eq!(lines[3]["locator"]["value"], "323-347");
        assert_eq!(lines[4]["label"], "contains");
        let last = lines.last().unwrap();
        assert_eq!(last["label"], "defined_in");
        assert_eq!(last["paragraph"], 0);
        // Without create, the header carries no create block at all.
        let plain = render_batch("code", &facts, None);
        let header: serde_json::Value =
            serde_json::from_str(plain.lines().next().unwrap()).unwrap();
        assert!(header.get("create").is_none());
    }
}
