//! `taguru anchoring` — the anchoring rate (根拠率) and locator
//! validity of extraction batches, offline (#793).
//!
//! For every association in a batch, is its subject and object
//! actually IN the original text (the cited paragraph when the
//! association cites one, else the whole passage)? "In" is a substring
//! match after [`crate::context::normalize_entry`] — the SAME folding
//! the server's entry index resolves names under (NFKC, case, kana),
//! so the rate measures what a reader of the stored graph would find,
//! not byte luck. Two rates per document:
//!
//! - **strict**: the name's own spelling only. The hallucination
//!   floor — a name that is nowhere in the text under any folding.
//! - **with aliases**: any spelling in the name's alias group counts
//!   (the batch's own concept aliases, plus `--vocabulary`'s context
//!   aliases). Aliases are model output too, so a mis-identification
//!   (#758's shape) inflates this rate — the strict/with-aliases GAP
//!   is the "anchored only through an alias" share, reported on
//!   purpose rather than folded away.
//!
//! The strict rate is a within-type comparison, not a cross-type
//! one (#806): a document whose subject appears only in its title
//! while every fact sits in a table row (a specsheet) cannot satisfy
//! "subject AND object in the cited paragraph" for any association,
//! however faithful — the cited row never repeats the title's name.
//! Prose measured strict 0.68 where the same model and run measured
//! 0.00–0.06 on specsheets with locator validity 1.0. A near-zero
//! strict rate on a table-shaped document is the structure showing
//! through; the USAGE text and docs/extract.html say so.
//!
//! **Locator validity**: of the associations that cite a `paragraph`,
//! how many cite one where the subject or the object (alias group
//! included) actually occurs.
//!
//! Needs only batch files — no server, no trace — so it applies to
//! 0.9.3 output unchanged. `scripts/extract_metrics.py --anchoring`
//! rolls the JSON report up by context and group.
//!
//! Exit codes: 0 = report produced · 1 = nothing to report (no
//! readable batch with a passage) · 2 = usage error.

use std::collections::{BTreeMap, HashMap};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use serde::Serialize;

use taguru::context::normalize_entry;

const USAGE: &str = "\
usage: taguru anchoring BATCH_OR_DIR... [--vocabulary PATH] [--json FILE]

Measures, for extraction batch files (the `taguru extract --out`
output; directories expand to their *.jsonl files):

  anchoring rate    associations whose subject AND object occur in the
                    original text (the cited paragraph, else the whole
                    passage), as a normalize_entry substring match —
                    `strict` (the name's own spelling) and
                    `with_aliases` (any spelling in its alias group);
                    the gap is the alias-dependent share
  locator validity  cited associations whose paragraph actually holds
                    the subject or the object (alias group included)

  --vocabulary PATH   batch stream file(s) (a file, or a directory's
                      *.jsonl — the `taguru export` shape; an extract
                      --out works too) whose concept aliases
                      extend each name's alias group — the spellings
                      the target context already settled on
  --json FILE         write the per-document report as JSON (the shape
                      scripts/extract_metrics.py --anchoring reads)

A batch without a passage (--no-passage) cannot be judged and is
reported as skipped.

Read `strict` within one document type, not across types: a document
whose subject appears only in the title while its facts sit in table
rows (a specsheet) cannot satisfy \"subject AND object in the cited
paragraph\" however faithful the extraction, so a near-zero strict
rate there reflects the document's shape, not hallucination. Compare
runs of the same type; let locator validity and `with_aliases` carry
the cross-type view.
";

pub(crate) fn run(args: &[String]) -> i32 {
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut vocabulary: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return 0;
            }
            "--vocabulary" => match iter.next() {
                Some(value) => vocabulary = Some(PathBuf::from(value)),
                None => return usage_error("--vocabulary needs a path"),
            },
            "--json" => match iter.next() {
                Some(value) => json_out = Some(PathBuf::from(value)),
                None => return usage_error("--json needs a path"),
            },
            other if other.starts_with('-') => {
                return usage_error(&format!("unknown argument '{other}'"));
            }
            path => inputs.push(PathBuf::from(path)),
        }
    }
    if inputs.is_empty() {
        return usage_error("at least one batch file or directory is required");
    }
    let files = match expand(&inputs) {
        Ok(files) => files,
        Err(message) => {
            eprintln!("taguru: anchoring: {message}");
            return 1;
        }
    };
    let context_aliases = match vocabulary.as_deref().map(harvest_aliases).transpose() {
        Ok(aliases) => aliases.unwrap_or_default(),
        Err(message) => {
            eprintln!("taguru: anchoring: {message}");
            return 1;
        }
    };

    let mut documents: BTreeMap<String, DocumentReport> = BTreeMap::new();
    let mut skipped = 0usize;
    for file in &files {
        let batch = match std::fs::File::open(file)
            .map_err(|error| error.to_string())
            .and_then(|handle| crate::ingest::parse_batch(BufReader::new(handle)))
        {
            Ok(batch) => batch,
            Err(message) => {
                eprintln!("taguru: anchoring: {}: {message}", file.display());
                return 1;
            }
        };
        let Some(passage) = batch.passage() else {
            eprintln!(
                "taguru: anchoring: {}: no passage (--no-passage batch) — skipped",
                file.display()
            );
            skipped += 1;
            continue;
        };
        let report = judge(
            passage,
            batch.associations(),
            batch.concept_aliases(),
            &context_aliases,
        );
        // Two inputs can hold the same source (the same document
        // extracted into two out-dirs); totals count both, so the
        // table must list both — disambiguate by the batch file's
        // directory, with a warning, exactly as
        // scripts/extract_metrics.py does for its trace aggregation.
        let mut key = batch.source.clone();
        if documents.contains_key(&key) {
            let parent = file
                .parent()
                .and_then(Path::file_name)
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| file.display().to_string());
            key = format!("{} ({parent})", batch.source);
            eprintln!(
                "taguru: anchoring: {}: source '{}' already judged — listed again as \
                 '{key}'",
                file.display(),
                batch.source
            );
        }
        documents.insert(
            key,
            DocumentReport {
                context: batch.context.clone(),
                counts: report,
            },
        );
    }
    if documents.is_empty() {
        eprintln!("taguru: anchoring: no batch with a passage to judge");
        return 1;
    }

    let mut totals = Counts::default();
    for document in documents.values() {
        totals.add(&document.counts);
    }
    print_table(&documents, &totals, skipped);
    if let Some(path) = json_out {
        let report = Report {
            documents: &documents,
            totals: &totals,
            skipped_no_passage: skipped,
        };
        let body = serde_json::to_string_pretty(&report).expect("plain fields always serialize");
        if let Err(error) = std::fs::write(&path, body + "\n") {
            eprintln!("taguru: anchoring: writing {}: {error}", path.display());
            return 1;
        }
    }
    0
}

fn usage_error(message: &str) -> i32 {
    eprintln!("taguru: anchoring: {message}");
    eprint!("{USAGE}");
    2
}

/// Files as given; directories expand to their *.jsonl files, sorted —
/// the same non-recursive rule `taguru import DIR` uses, so the
/// hidden `.extract-trace/` directory is never read as a batch.
fn expand(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_file() {
            files.push(input.clone());
        } else if input.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(input)
                .map_err(|error| format!("cannot read {}: {error}", input.display()))?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("jsonl")
                })
                .collect();
            if found.is_empty() {
                return Err(format!("no .jsonl files under {}", input.display()));
            }
            found.sort();
            files.append(&mut found);
        } else {
            return Err(format!(
                "{} is neither a file nor a directory",
                input.display()
            ));
        }
    }
    Ok(files)
}

/// `--vocabulary`'s concept aliases: every batch of every stream under
/// the path (a file, or a directory of `*.jsonl` files — the `taguru
/// export` shape, read by [`expand`]'s rule so an extract `--out`
/// directory's `.extract-manifest.json` is never parsed as a stream,
/// #805), as (spelling → canonical) pairs.
fn harvest_aliases(path: &Path) -> Result<Vec<(String, String)>, String> {
    let files: Vec<PathBuf> = if path.is_dir() {
        let mut found: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("jsonl")
            })
            .collect();
        if found.is_empty() {
            return Err(format!("no .jsonl files under {}", path.display()));
        }
        found.sort();
        found
    } else {
        vec![path.to_path_buf()]
    };
    let mut aliases = Vec::new();
    for file in files {
        let handle =
            std::fs::File::open(&file).map_err(|error| format!("{}: {error}", file.display()))?;
        let stream = crate::ingest::parse_stream(BufReader::new(handle))
            .map_err(|message| format!("{}: {message}", file.display()))?;
        for batch in &stream.batches {
            for (spelling, canonical) in batch.concept_aliases() {
                aliases.push((spelling.clone(), canonical.clone()));
            }
        }
    }
    Ok(aliases)
}

/// The judged counts for one document. Rates are derived at
/// serialization so the JSON carries both counts and rates.
#[derive(Default, Serialize)]
struct Counts {
    associations: usize,
    anchored_strict: usize,
    anchored_with_aliases: usize,
    cited: usize,
    locator_valid: usize,
    #[serde(serialize_with = "rate3")]
    rate_strict: (usize, usize),
    #[serde(serialize_with = "rate3")]
    rate_with_aliases: (usize, usize),
    #[serde(serialize_with = "rate3")]
    locator_validity: (usize, usize),
}

fn rate3<S: serde::Serializer>(value: &(usize, usize), serializer: S) -> Result<S::Ok, S::Error> {
    let (numerator, denominator) = *value;
    if denominator == 0 {
        serializer.serialize_none()
    } else {
        serializer.serialize_f64(numerator as f64 / denominator as f64)
    }
}

impl Counts {
    fn add(&mut self, other: &Counts) {
        self.associations += other.associations;
        self.anchored_strict += other.anchored_strict;
        self.anchored_with_aliases += other.anchored_with_aliases;
        self.cited += other.cited;
        self.locator_valid += other.locator_valid;
        self.refresh();
    }

    fn refresh(&mut self) {
        self.rate_strict = (self.anchored_strict, self.associations);
        self.rate_with_aliases = (self.anchored_with_aliases, self.associations);
        self.locator_validity = (self.locator_valid, self.cited);
    }
}

#[derive(Serialize)]
struct DocumentReport {
    context: String,
    #[serde(flatten)]
    counts: Counts,
}

#[derive(Serialize)]
struct Report<'a> {
    documents: &'a BTreeMap<String, DocumentReport>,
    totals: &'a Counts,
    skipped_no_passage: usize,
}

/// Alias groups as a union-find over normalized names: an alias
/// (spelling → canonical) joins the two, and a name's group is every
/// normalized spelling reachable through such joins — the batch's own
/// aliases and the context's alike, exactly the "ごはん/ご飯" folding
/// the issue defines. Keys and members are all `normalize_entry`
/// output.
struct AliasGroups {
    parent: HashMap<String, String>,
    members: HashMap<String, Vec<String>>,
}

impl AliasGroups {
    fn build(pairs: impl Iterator<Item = (String, String)>) -> Self {
        let mut parent: HashMap<String, String> = HashMap::new();
        fn find(parent: &mut HashMap<String, String>, node: &str) -> String {
            let up = parent
                .entry(node.to_string())
                .or_insert_with(|| node.to_string())
                .clone();
            if up == node {
                return up;
            }
            let root = find(parent, &up);
            parent.insert(node.to_string(), root.clone());
            root
        }
        let mut names: Vec<String> = Vec::new();
        for (spelling, canonical) in pairs {
            let spelling = normalize_entry(&spelling);
            let canonical = normalize_entry(&canonical);
            names.push(spelling.clone());
            names.push(canonical.clone());
            let spelling_root = find(&mut parent, &spelling);
            let canonical_root = find(&mut parent, &canonical);
            parent.insert(spelling_root, canonical_root);
        }
        let mut members: HashMap<String, Vec<String>> = HashMap::new();
        for name in names {
            let root = find(&mut parent, &name);
            let group = members.entry(root).or_default();
            if !group.contains(&name) {
                group.push(name);
            }
        }
        Self { parent, members }
    }

    /// Every spelling in `name`'s group, `name` itself first. A name
    /// no alias touches has a group of one.
    fn group<'a>(&'a self, normalized: &'a str) -> Vec<&'a str> {
        let mut root = normalized;
        while let Some(up) = self.parent.get(root) {
            if up == root {
                break;
            }
            root = up;
        }
        let mut group = vec![normalized];
        if let Some(members) = self.members.get(root) {
            group.extend(
                members
                    .iter()
                    .map(String::as_str)
                    .filter(|member| *member != normalized),
            );
        }
        group
    }
}

/// One document's counts (see the module doc for the definitions).
fn judge(
    passage: &str,
    associations: &[crate::registry::AssocOp],
    own_aliases: &BTreeMap<String, String>,
    context_aliases: &[(String, String)],
) -> Counts {
    let groups = AliasGroups::build(
        own_aliases
            .iter()
            .map(|(spelling, canonical)| (spelling.clone(), canonical.clone()))
            .chain(context_aliases.iter().cloned()),
    );
    let spans = crate::paragraph::split(passage);
    let paragraph_haystacks: Vec<String> = spans
        .iter()
        .map(|span| normalize_entry(&passage[span.start as usize..span.end as usize]))
        .collect();
    let whole = normalize_entry(passage);

    let mut counts = Counts::default();
    for association in associations {
        counts.associations += 1;
        let cited = association
            .paragraph
            .and_then(|index| paragraph_haystacks.get(index as usize));
        let haystack = cited.map(String::as_str).unwrap_or(whole.as_str());
        let subject = normalize_entry(&association.subject);
        let object = normalize_entry(&association.object);
        let strict = haystack.contains(&subject) && haystack.contains(&object);
        let subject_hit = groups
            .group(&subject)
            .iter()
            .any(|spelling| haystack.contains(spelling));
        let object_hit = groups
            .group(&object)
            .iter()
            .any(|spelling| haystack.contains(spelling));
        if strict {
            counts.anchored_strict += 1;
        }
        if subject_hit && object_hit {
            counts.anchored_with_aliases += 1;
        }
        if association.paragraph.is_some() {
            counts.cited += 1;
            // An out-of-range citation has no paragraph to hold
            // anything: invalid by construction.
            if cited.is_some() && (subject_hit || object_hit) {
                counts.locator_valid += 1;
            }
        }
    }
    counts.refresh();
    counts
}

fn print_table(documents: &BTreeMap<String, DocumentReport>, totals: &Counts, skipped: usize) {
    let rate = |numerator: usize, denominator: usize| -> String {
        if denominator == 0 {
            "-".to_string()
        } else {
            format!("{:.3}", numerator as f64 / denominator as f64)
        }
    };
    println!("source\tassocs\tstrict\twith_aliases\tlocator_validity");
    for (source, document) in documents {
        let counts = &document.counts;
        println!(
            "{source}\t{}\t{}\t{}\t{}",
            counts.associations,
            rate(counts.anchored_strict, counts.associations),
            rate(counts.anchored_with_aliases, counts.associations),
            rate(counts.locator_valid, counts.cited),
        );
    }
    println!(
        "TOTAL\t{}\t{}\t{}\t{}",
        totals.associations,
        rate(totals.anchored_strict, totals.associations),
        rate(totals.anchored_with_aliases, totals.associations),
        rate(totals.locator_valid, totals.cited),
    );
    if skipped > 0 {
        println!("({skipped} batch(es) without a passage skipped)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assoc(subject: &str, object: &str, paragraph: Option<u32>) -> crate::registry::AssocOp {
        crate::registry::AssocOp {
            subject: subject.to_string(),
            label: "rel".to_string(),
            object: object.to_string(),
            weight: 1.0,
            source: None,
            paragraph,
        }
    }

    /// #805: `--vocabulary DIR` reads the directory's `*.jsonl` only
    /// — [`expand`]'s rule — so an extract `--out` directory's
    /// `.extract-manifest.json` is never parsed as a stream, and a
    /// directory with no `.jsonl` is named as such.
    #[test]
    fn harvest_aliases_reads_only_jsonl_files_from_a_directory() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-anchoring-vocab-jsonl-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("prior.jsonl"),
            "{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"prior.md\"}\n\
             {\"subject\":\"青嶺酒造\",\"label\":\"杜氏\",\"object\":\"高瀬\",\"weight\":1.0}\n\
             {\"alias\":\"あおみね\",\"canonical\":\"青嶺酒造\",\"kind\":\"concept\"}\n",
        )
        .unwrap();
        std::fs::write(dir.join(".extract-manifest.json"), "{}\n").unwrap();
        std::fs::write(dir.join("README.txt"), "not a stream").unwrap();

        let aliases = harvest_aliases(&dir).unwrap();
        assert_eq!(
            aliases,
            vec![("あおみね".to_string(), "青嶺酒造".to_string())]
        );

        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        std::fs::write(empty.join(".extract-manifest.json"), "{}\n").unwrap();
        let error = harvest_aliases(&empty).unwrap_err();
        assert_eq!(error, format!("no .jsonl files under {}", empty.display()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The issue's own example: 「ごはん/ご飯」×「食べる/たべる」 — any
    /// spelling combination anchors with aliases; only the exact
    /// spellings anchor strictly. normalize_entry folds katakana and
    /// width, so ケーキ/けーき need no alias at all.
    #[test]
    fn anchoring_judges_strict_and_alias_group_presence() {
        let passage = "ご飯を食べる。\n\nケーキは別腹。";
        let own: BTreeMap<String, String> = [("ごはん".to_string(), "ご飯".to_string())].into();
        let context = vec![("たべる".to_string(), "食べる".to_string())];
        let associations = [
            assoc("ご飯", "食べる", Some(0)),    // strict
            assoc("ごはん", "たべる", Some(0)),  // aliases only
            assoc("けーき", "別腹", Some(1)),    // strict via kana folding
            assoc("ごはん", "存在しない", None), // object nowhere
        ];
        let counts = judge(passage, &associations, &own, &context);
        assert_eq!(counts.associations, 4);
        assert_eq!(counts.anchored_strict, 2);
        assert_eq!(counts.anchored_with_aliases, 3);
        assert_eq!(counts.cited, 3);
        assert_eq!(counts.locator_valid, 3);
    }

    /// The cited paragraph is the haystack when present — a subject
    /// that exists only in ANOTHER paragraph does not anchor a cited
    /// association, but does anchor an uncited one (whole passage).
    /// An out-of-range citation is invalid and anchors against the
    /// whole passage.
    #[test]
    fn citations_narrow_the_haystack_and_out_of_range_is_invalid() {
        let passage = "alphaの話。\n\nbetaの話。";
        let none = BTreeMap::new();
        let cited_wrong = [assoc("alpha", "beta", Some(0))];
        let counts = judge(passage, &cited_wrong, &none, &[]);
        assert_eq!(counts.anchored_strict, 0, "beta is not in paragraph 0");
        assert_eq!(counts.locator_valid, 1, "alpha IS in paragraph 0");

        let uncited = [assoc("alpha", "beta", None)];
        let counts = judge(passage, &uncited, &none, &[]);
        assert_eq!(counts.anchored_strict, 1, "the whole passage holds both");
        assert_eq!(counts.cited, 0);

        let out_of_range = [assoc("alpha", "beta", Some(9))];
        let counts = judge(passage, &out_of_range, &none, &[]);
        assert_eq!(counts.anchored_strict, 1, "falls back to the passage");
        assert_eq!(counts.cited, 1);
        assert_eq!(counts.locator_valid, 0, "paragraph 9 does not exist");
    }

    /// Alias groups are transitive across the batch's own aliases and
    /// the context's: A→B (own) and B→C (context) put A, B, C in one
    /// group, whichever direction the pairs point.
    #[test]
    fn alias_groups_are_transitive_across_both_sources() {
        let groups = AliasGroups::build(
            [
                ("A".to_string(), "B".to_string()),
                ("C".to_string(), "B".to_string()),
                ("D".to_string(), "E".to_string()),
            ]
            .into_iter(),
        );
        let mut group = groups.group("a");
        group.sort_unstable();
        assert_eq!(group, ["a", "b", "c"]);
        let mut group = groups.group("e");
        group.sort_unstable();
        assert_eq!(group, ["d", "e"]);
        assert_eq!(groups.group("z"), ["z"], "an untouched name stands alone");
    }
}
