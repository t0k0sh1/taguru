//! `taguru-code find` / `tree`: the read path a coding agent drives.
//! Everything runs in-process against the data directory — no server,
//! no network. `find` is resolve → `defined_in` → citation: the cue
//! resolves through the same exact/alias/containment/fuzzy tiers the
//! server serves, each hit's `defined_in` edge names the file, and the
//! attribution's paragraph dereferences to the section heading and the
//! `lines` locator the sync stored — one line per hit, `file:lines`
//! shaped so an agent (or an editor) can jump straight there. `tree`
//! walks `contains` edges instead: what a directory, file, or symbol
//! holds, one level at a time.

use std::path::PathBuf;

use crate::code::repo_walk::RepoWalk;
use crate::code::sync::DEFAULT_CONTEXT;
use crate::registry::{AppState, CitationLookup};

/// Hits printed per `find` unless `--limit` says otherwise.
const DEFAULT_LIMIT: usize = 10;

/// Floor for the typo-fallback tier: below this Dice/coverage score a
/// "match" is bigram noise, and an honest "no match — fall back to
/// grep" serves the agent better than plausible-looking junk.
const FUZZY_FLOOR: f64 = 0.65;

struct QueryArgs {
    cue: Option<String>,
    context: String,
    data_dir: Option<PathBuf>,
    json: bool,
    limit: usize,
}

fn parse_args(verb: &str, args: &[String]) -> Result<QueryArgs, String> {
    let mut parsed = QueryArgs {
        cue: None,
        context: DEFAULT_CONTEXT.to_string(),
        data_dir: None,
        json: false,
        limit: DEFAULT_LIMIT,
    };
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--json" => parsed.json = true,
            "--context" => {
                parsed.context = rest
                    .next()
                    .ok_or_else(|| "--context needs a name".to_string())?
                    .clone();
            }
            "--data-dir" => {
                parsed.data_dir = Some(PathBuf::from(
                    rest.next()
                        .ok_or_else(|| "--data-dir needs a path".to_string())?,
                ));
            }
            "--limit" => {
                parsed.limit = rest
                    .next()
                    .ok_or_else(|| "--limit needs a number".to_string())?
                    .parse()
                    .map_err(|_| "--limit needs a number".to_string())?;
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag '{other}'"));
            }
            cue => {
                if parsed.cue.is_some() {
                    return Err(format!("unexpected second argument '{cue}'"));
                }
                parsed.cue = Some(cue.to_string());
            }
        }
    }
    if verb == "find" && parsed.cue.is_none() {
        return Err("find needs a cue: taguru-code find <symbol>".to_string());
    }
    Ok(parsed)
}

/// Opens the data directory the same way sync writes it: default
/// `$PROJECT_ROOT/.taguru`, no per-op WAL, embeddings not needed for
/// graph reads.
fn open(args: &QueryArgs) -> Result<AppState, String> {
    let data_dir = match &args.data_dir {
        Some(dir) => dir.clone(),
        None => RepoWalk::discover(std::path::Path::new("."))?
            .root()
            .join(".taguru"),
    };
    if !data_dir.is_dir() {
        return Err(format!(
            "{} does not exist — run `taguru-code sync` first",
            data_dir.display()
        ));
    }
    let mut config = crate::registry::BootConfig::from_env();
    config.data_dir = data_dir;
    config.wal_enabled = false;
    config
        .boot(None, None, None, None, None)
        .map_err(|error| error.to_string())
}

/// Dice coefficient over character bigrams, case-insensitive — the
/// typo metric for the fuzzy fallback tier. 1.0 for identical strings,
/// 0.0 for nothing shared; single-char inputs compare by equality.
fn bigram_dice(a: &str, b: &str) -> f64 {
    let bigrams = |s: &str| -> Vec<(char, char)> {
        let chars: Vec<char> = s.chars().flat_map(char::to_lowercase).collect();
        chars.windows(2).map(|pair| (pair[0], pair[1])).collect()
    };
    let (mut left, right) = (bigrams(a), bigrams(b));
    if left.is_empty() || right.is_empty() {
        return if a.eq_ignore_ascii_case(b) { 1.0 } else { 0.0 };
    }
    let total = left.len() + right.len();
    let mut shared = 0usize;
    for bigram in &right {
        if let Some(position) = left.iter().position(|candidate| candidate == bigram) {
            left.swap_remove(position);
            shared += 1;
        }
    }
    (2 * shared) as f64 / total as f64
}

/// One matched symbol before citation dereference:
/// (qualified name, tier, score, [(source file, paragraph)]).
type Matched = (String, &'static str, f64, Vec<(String, Option<u32>)>);

/// One `find` hit, fully dereferenced.
pub(crate) struct Hit {
    pub(crate) name: String,
    pub(crate) tier: &'static str,
    pub(crate) score: f64,
    pub(crate) section: Option<String>,
    pub(crate) file: Option<String>,
    pub(crate) lines: Option<String>,
    /// True when the hit is a file/directory node (its name IS the
    /// location) rather than a symbol with a `defined_in` edge.
    pub(crate) is_path_node: bool,
}

impl Hit {
    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "tier": self.tier,
            "score": self.score,
            "section": self.section,
            "file": self.file,
            "lines": self.lines,
            "is_path_node": self.is_path_node,
        })
    }
}

/// The `find` verb.
pub(crate) fn find(args: &[String]) -> i32 {
    let args = match parse_args("find", args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("taguru-code: find: {message}");
            return 2;
        }
    };
    let cue = args.cue.clone().expect("checked in parse_args");
    let state = match open(&args) {
        Ok(state) => state,
        Err(message) => {
            eprintln!("taguru-code: find: {message}");
            return 1;
        }
    };

    let hits = match find_hits(&state, &args.context, &cue, args.limit) {
        Ok(hits) => hits,
        Err(message) => {
            eprintln!("taguru-code: find: {message}");
            return 1;
        }
    };

    if hits.is_empty() {
        eprintln!("taguru-code: find: no match for '{cue}' — fall back to grep");
        return 1;
    }
    if args.json {
        let rendered: Vec<serde_json::Value> = hits.iter().map(Hit::to_json).collect();
        println!("{}", serde_json::json!(rendered));
        return 0;
    }
    for hit in &hits {
        let kind = hit
            .section
            .as_deref()
            .and_then(|section| section.split(' ').next())
            .unwrap_or(if hit.is_path_node { "path" } else { "?" });
        let location = match (&hit.file, &hit.lines) {
            (Some(file), Some(lines)) => format!("{file}:{lines}"),
            (Some(file), None) => file.clone(),
            (None, _) if hit.is_path_node => hit.name.clone(),
            (None, _) => "?".to_string(),
        };
        println!(
            "{kind:<9} {:<60} {location}  [{} {:.2}]",
            hit.name, hit.tier, hit.score
        );
    }
    0
}

/// The shared core of `find` and `eval`: code-shaped tier matching
/// over the graph, then citation dereference. A cue like
/// `parse_batch` names a symbol's LAST segment, but concept names are
/// whole qualified names (`src/ingest/model.rs::parse_batch`) — the
/// generic resolve() tiers score against the whole string and drown
/// the tail match, so this ranks by code-shaped rules first: exact
/// tail segment, then qualified suffix (`Context::resolve`), then
/// tail prefix/substring, then a file-basename tier for path cues,
/// with a tail-segment bigram-Dice pass as the typo fallback. All of
/// it is one in-memory scan of the defined_in edges — the short-name
/// index the alias table deliberately does not hold (aliases are
/// permanent and first-claimant-wins; a read-time scan is neither).
pub(crate) fn find_hits(
    state: &AppState,
    context_name: &str,
    cue: &str,
    limit: usize,
) -> Result<Vec<Hit>, String> {
    let matched: Result<Vec<Matched>, _> = state.read_context(context_name, |context| {
        // The graph is append-only: retracting a source empties an
        // edge's attributions but the edge record stays. An edge no
        // source attests anymore is a ghost — a renamed file's old
        // symbols — and must not surface.
        let symbols: Vec<taguru::context::Association> = context
            .query(None, Some("defined_in"), None)
            .into_iter()
            .filter(|assoc| !assoc.attributions.is_empty())
            .collect();
        let mut tiers: [Vec<&taguru::context::Association>; 4] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for assoc in &symbols {
            let tail = assoc.subject.rsplit("::").next().unwrap_or(&assoc.subject);
            if tail == cue {
                tiers[0].push(assoc);
            } else if cue.contains("::")
                && (assoc.subject.ends_with(&format!("::{cue}"))
                    || assoc.subject.ends_with(&format!("/{cue}"))
                    || assoc.subject == cue)
            {
                // `::` suffix is `Type::method`; `/` suffix lets a
                // file-qualified cue (`extract.rs::run`) pin a
                // top-level symbol the same way.
                tiers[1].push(assoc);
            } else if tail.starts_with(cue) {
                tiers[2].push(assoc);
            } else if tail.contains(cue) {
                tiers[3].push(assoc);
            }
        }
        let mut picked: Vec<Matched> = Vec::new();
        for (tier_name, tier) in ["exact", "qualified", "prefix", "contains"]
            .into_iter()
            .zip(tiers)
        {
            for assoc in tier {
                if picked.len() >= limit {
                    break;
                }
                let location = assoc
                    .attributions
                    .first()
                    .map(|attribution| (assoc.object.clone(), attribution.paragraph))
                    .into_iter()
                    .collect();
                picked.push((assoc.subject.clone(), tier_name, 1.0, location));
            }
        }
        // A path-shaped cue (`model.rs`, `src/ingest`) matches
        // file/directory nodes by basename or whole path.
        if picked.len() < limit {
            let files = context.query(None, Some("contains"), None);
            let mut seen = std::collections::HashSet::new();
            for assoc in &files {
                if assoc.attributions.is_empty() {
                    continue; // retracted ghost, same as above
                }
                let node = &assoc.object;
                if node.contains("::") || !seen.insert(node.clone()) {
                    continue;
                }
                let basename = node.rsplit('/').next().unwrap_or(node);
                if basename == cue || node.as_str() == cue {
                    picked.push((node.clone(), "path", 1.0, Vec::new()));
                    if picked.len() >= limit {
                        break;
                    }
                }
            }
        }
        // Typo fallback, only when the code-shaped rules found
        // nothing: bigram Dice between the cue and each symbol's
        // TAIL segment. The server's resolve() scores against the
        // whole qualified name, where the path drowns the tail —
        // `parse_bacth` must find `parse_batch`, not whatever long
        // name shares the most path bigrams.
        if picked.is_empty() {
            let mut scored: Vec<(f64, &taguru::context::Association)> = symbols
                .iter()
                .map(|assoc| {
                    let tail = assoc.subject.rsplit("::").next().unwrap_or(&assoc.subject);
                    (bigram_dice(cue, tail), assoc)
                })
                .filter(|(score, _)| *score >= FUZZY_FLOOR)
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (score, assoc) in scored.into_iter().take(limit) {
                let location = assoc
                    .attributions
                    .first()
                    .map(|attribution| (assoc.object.clone(), attribution.paragraph))
                    .into_iter()
                    .collect();
                picked.push((assoc.subject.clone(), "fuzzy", score, location));
            }
        }
        picked
    });
    let matched = matched.map_err(|error| {
        format!("context '{context_name}': {error:?} — run `taguru-code sync` first")
    })?;

    // Citation dereference: (source file, paragraph) → section + lines.
    let mut hits = Vec::new();
    for (name, tier, score, locations) in matched {
        if locations.is_empty() {
            let is_path_node = tier == "path";
            hits.push(Hit {
                name,
                tier,
                score,
                section: None,
                file: None,
                lines: None,
                is_path_node,
            });
            continue;
        }
        for (file, paragraph) in locations {
            let citation = paragraph.and_then(|paragraph| {
                match state.citation(context_name, &file, paragraph) {
                    Some(Ok(CitationLookup::Found {
                        section, locator, ..
                    })) => Some((section, locator)),
                    _ => None,
                }
            });
            let (section, locator) = citation.unwrap_or((None, None));
            hits.push(Hit {
                name: name.clone(),
                tier,
                score,
                section,
                file: Some(file),
                lines: locator
                    .filter(|locator| locator.kind == "lines")
                    .map(|locator| locator.value),
                is_path_node: false,
            });
        }
    }

    Ok(hits)
}

/// The `tree` verb: one level of `contains`. Without an argument,
/// lists the roots (subjects that are contained by nothing).
pub(crate) fn tree(args: &[String]) -> i32 {
    let args = match parse_args("tree", args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("taguru-code: tree: {message}");
            return 2;
        }
    };
    let state = match open(&args) {
        Ok(state) => state,
        Err(message) => {
            eprintln!("taguru-code: tree: {message}");
            return 1;
        }
    };
    let listing: Result<Vec<String>, _> = state.read_context(&args.context, |context| {
        match &args.cue {
            Some(target) => context
                .query(Some(target), Some("contains"), None)
                .into_iter()
                // Retracted ghosts (attribution-less edges) stay out,
                // same as find.
                .filter(|assoc| !assoc.attributions.is_empty())
                .map(|assoc| assoc.object)
                .collect(),
            None => {
                // Roots: contains-subjects that are nobody's object.
                let all: Vec<taguru::context::Association> = context
                    .query(None, Some("contains"), None)
                    .into_iter()
                    .filter(|assoc| !assoc.attributions.is_empty())
                    .collect();
                let contained: std::collections::HashSet<String> =
                    all.iter().map(|assoc| assoc.object.clone()).collect();
                let mut roots: Vec<String> = all
                    .into_iter()
                    .map(|assoc| assoc.subject)
                    .filter(|subject| !contained.contains(subject))
                    .collect();
                roots.sort();
                roots.dedup();
                roots
            }
        }
    });
    match listing {
        Ok(children) if children.is_empty() => {
            match &args.cue {
                Some(target) => eprintln!("taguru-code: tree: nothing under '{target}'"),
                None => eprintln!("taguru-code: tree: context is empty — run `taguru-code sync`"),
            }
            1
        }
        Ok(children) => {
            if args.json {
                println!("{}", serde_json::json!(children));
            } else {
                for child in children {
                    println!("{child}");
                }
            }
            0
        }
        Err(error) => {
            eprintln!(
                "taguru-code: tree: context '{}': {error:?} — run `taguru-code sync` first",
                args.context
            );
            1
        }
    }
}
