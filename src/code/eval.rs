//! `taguru-code evalset` / `eval`: the accuracy gate (evaluation axis
//! 1). Because every fact is deterministic AST output, the ground
//! truth is derivable: `evalset` re-parses the repo at HEAD with the
//! same grammars and samples symbols into labeled cases — cue in,
//! expected qualified name / file / line range out — with no human
//! labeling and no LLM. `eval` replays each cue through the same
//! `find` core the agent uses and scores hit@1 / hit@10 plus locator
//! drift; `--thresholds` turns a completed run into a CI-suitable
//! pass/fail gate that exits 3 on regression, `taguru evaluate`'s own
//! convention (ADR 0004), offline.

use std::path::PathBuf;

use crate::code::grammars;
use crate::code::query::find_hits;
use crate::code::repo_walk::RepoWalk;
use crate::code::sync::DEFAULT_CONTEXT;

/// Cases `evalset` samples unless `--sample` says otherwise.
const DEFAULT_SAMPLE: usize = 200;

/// Hits `eval` requests per cue — the same default list length an
/// agent sees, so hit@10 measures the surface the agent actually gets.
const EVAL_LIMIT: usize = 10;

struct EvalArgs {
    start: PathBuf,
    context: String,
    data_dir: Option<PathBuf>,
    out: Option<PathBuf>,
    eval: Option<PathBuf>,
    thresholds: Option<PathBuf>,
    sample: usize,
}

fn parse_args(args: &[String]) -> Result<EvalArgs, String> {
    let mut parsed = EvalArgs {
        start: PathBuf::from("."),
        context: DEFAULT_CONTEXT.to_string(),
        data_dir: None,
        out: None,
        eval: None,
        thresholds: None,
        sample: DEFAULT_SAMPLE,
    };
    let mut rest = args.iter();
    let mut saw_path = false;
    while let Some(arg) = rest.next() {
        let mut value = |flag: &str| {
            rest.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "--out" => parsed.out = Some(PathBuf::from(value("--out")?)),
            "--eval" => parsed.eval = Some(PathBuf::from(value("--eval")?)),
            "--thresholds" => parsed.thresholds = Some(PathBuf::from(value("--thresholds")?)),
            "--context" => parsed.context = value("--context")?,
            "--data-dir" => parsed.data_dir = Some(PathBuf::from(value("--data-dir")?)),
            "--sample" => {
                parsed.sample = value("--sample")?
                    .parse()
                    .map_err(|_| "--sample needs a number".to_string())?;
            }
            other if other.starts_with('-') => return Err(format!("unknown flag '{other}'")),
            path => {
                if saw_path {
                    return Err(format!("unexpected second path '{path}'"));
                }
                parsed.start = PathBuf::from(path);
                saw_path = true;
            }
        }
    }
    Ok(parsed)
}

/// The `evalset` verb.
pub(crate) fn evalset(args: &[String]) -> i32 {
    let args = match parse_args(args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("taguru-code: evalset: {message}");
            return 2;
        }
    };
    let Some(out) = args.out.clone() else {
        eprintln!("taguru-code: evalset: --out FILE is required");
        return 2;
    };
    match generate(&args) {
        Ok(cases) => {
            let body: String = cases.iter().map(|case| format!("{case}\n")).collect();
            if let Err(error) = std::fs::write(&out, body) {
                eprintln!("taguru-code: evalset: writing {}: {error}", out.display());
                return 1;
            }
            println!("wrote {} case(s) to {}", cases.len(), out.display());
            0
        }
        Err(message) => {
            eprintln!("taguru-code: evalset: {message}");
            1
        }
    }
}

/// Samples symbols evenly across the repo at HEAD into labeled cases.
/// Case kinds alternate deterministically: bare tail cue, qualified
/// suffix cue, and (every sixth case) the file's basename as a path
/// cue — the three shapes an agent actually types.
fn generate(args: &EvalArgs) -> Result<Vec<serde_json::Value>, String> {
    let walk = RepoWalk::discover(&args.start)?;
    walk.head()?; // committed-state contract, same refusal as sync
    let paths: Vec<String> = walk
        .full_listing()?
        .into_iter()
        .filter(|path| {
            std::path::Path::new(path)
                .extension()
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase)
                .is_some_and(|extension| grammars::for_extension(&extension).is_some())
        })
        .collect();
    let contents = walk.read_at_head(&paths)?;
    let mut symbols = Vec::new();
    for (path, content) in contents {
        let Some(content) = content else { continue };
        let extension = std::path::Path::new(&path)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        let Some(grammar) = grammars::for_extension(&extension) else {
            continue;
        };
        for symbol in grammar.symbols(&content, &path) {
            symbols.push((path.clone(), symbol));
        }
    }
    if symbols.is_empty() {
        return Err("no symbols found — is this the right repository?".to_string());
    }
    // Tails shared by many symbols (`tests`, `new`, `run`) cannot be
    // answered from the bare name by ANY tool, and an agent would not
    // ask that way either — those sample as qualified cues instead,
    // so the tail metric measures answerable questions while the
    // ambiguous names still exercise the disambiguation path.
    let mut tail_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (_, symbol) in &symbols {
        *tail_counts.entry(symbol.name.as_str()).or_default() += 1;
    }
    let step = (symbols.len() / args.sample.max(1)).max(1);
    let mut cases = Vec::new();
    for (index, (path, symbol)) in symbols.iter().step_by(step).enumerate() {
        if cases.len() >= args.sample {
            break;
        }
        let lines = format!("{}-{}", symbol.start_line, symbol.end_line);
        let ambiguous = tail_counts.get(symbol.name.as_str()).copied().unwrap_or(0) > 3;
        let case = match index % 6 {
            // The dominant shape: the bare symbol name.
            0..=3 if !ambiguous => serde_json::json!({
                "kind": "tail",
                "cue": symbol.name,
                "expect": symbol.qualified_name,
                "file": path,
                "lines": lines,
            }),
            // A qualified cue: `Parent::name` when nested, else the
            // file's basename as the qualifier — also where the
            // ambiguous-tail cases from above land.
            0..=4 => {
                let cue = symbol
                    .parent
                    .as_deref()
                    .and_then(|parent| parent.rsplit("::").next())
                    .map(|parent_tail| format!("{parent_tail}::{}", symbol.name))
                    .unwrap_or_else(|| {
                        let basename = path.rsplit('/').next().unwrap_or(path);
                        format!("{basename}::{}", symbol.name)
                    });
                serde_json::json!({
                    "kind": "qualified",
                    "cue": cue,
                    "expect": symbol.qualified_name,
                    "file": path,
                    "lines": lines,
                })
            }
            // A path cue: the file's basename must find the file node.
            _ => {
                let basename = path.rsplit('/').next().unwrap_or(path);
                serde_json::json!({
                    "kind": "path",
                    "cue": basename,
                    "expect": path,
                    "file": path,
                    "lines": serde_json::Value::Null,
                })
            }
        };
        cases.push(case);
    }
    Ok(cases)
}

/// The `eval` verb.
pub(crate) fn eval(args: &[String]) -> i32 {
    let args = match parse_args(args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("taguru-code: eval: {message}");
            return 2;
        }
    };
    let Some(eval_path) = args.eval.clone() else {
        eprintln!("taguru-code: eval: --eval FILE is required");
        return 2;
    };
    let raw = match std::fs::read_to_string(&eval_path) {
        Ok(raw) => raw,
        Err(error) => {
            eprintln!(
                "taguru-code: eval: reading {}: {error}",
                eval_path.display()
            );
            return 1;
        }
    };
    let data_dir = args.data_dir.clone().unwrap_or_else(|| {
        RepoWalk::discover(&args.start)
            .map(|walk| walk.root().join(".taguru"))
            .unwrap_or_else(|_| PathBuf::from(".taguru"))
    });
    let mut config = crate::registry::BootConfig::from_env();
    config.data_dir = data_dir;
    config.wal_enabled = false;
    let state = match config.boot(None, None, None, None, None) {
        Ok(state) => state,
        Err(error) => {
            eprintln!("taguru-code: eval: {error}");
            return 1;
        }
    };

    let (mut total, mut hit1, mut hit10, mut line_drift) = (0usize, 0usize, 0usize, 0usize);
    let mut misses = Vec::new();
    for (number, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let case: serde_json::Value = match serde_json::from_str(line) {
            Ok(case) => case,
            Err(error) => {
                eprintln!("taguru-code: eval: line {}: {error}", number + 1);
                return 1;
            }
        };
        let (Some(cue), Some(expect)) = (
            case.get("cue").and_then(|value| value.as_str()),
            case.get("expect").and_then(|value| value.as_str()),
        ) else {
            eprintln!(
                "taguru-code: eval: line {}: needs cue and expect",
                number + 1
            );
            return 1;
        };
        total += 1;
        let hits = match find_hits(&state, &args.context, cue, EVAL_LIMIT) {
            Ok(hits) => hits,
            Err(message) => {
                eprintln!("taguru-code: eval: {message}");
                return 1;
            }
        };
        let position = hits.iter().position(|hit| hit.name == expect);
        match position {
            Some(0) => {
                hit1 += 1;
                hit10 += 1;
            }
            Some(_) => hit10 += 1,
            None => misses.push(format!("{cue} → {expect}")),
        }
        if let (Some(position), Some(expected_lines)) =
            (position, case.get("lines").and_then(|value| value.as_str()))
            && hits[position].lines.as_deref() != Some(expected_lines)
        {
            line_drift += 1;
        }
    }
    if total == 0 {
        eprintln!("taguru-code: eval: no cases in {}", eval_path.display());
        return 1;
    }
    let hit1_rate = hit1 as f64 / total as f64;
    let hit10_rate = hit10 as f64 / total as f64;
    for miss in misses.iter().take(10) {
        eprintln!("taguru-code: eval: miss: {miss}");
    }
    println!(
        "{}",
        serde_json::json!({
            "total": total,
            "hit1": hit1,
            "hit10": hit10,
            "hit1_rate": hit1_rate,
            "hit10_rate": hit10_rate,
            "line_drift": line_drift,
        })
    );

    if let Some(thresholds_path) = &args.thresholds {
        let thresholds: serde_json::Value = match std::fs::read_to_string(thresholds_path)
            .map_err(|error| error.to_string())
            .and_then(|raw| serde_json::from_str(&raw).map_err(|error| error.to_string()))
        {
            Ok(thresholds) => thresholds,
            Err(error) => {
                eprintln!(
                    "taguru-code: eval: thresholds {}: {error}",
                    thresholds_path.display()
                );
                return 1;
            }
        };
        let mut violated = false;
        for (key, actual) in [("hit1_rate", hit1_rate), ("hit10_rate", hit10_rate)] {
            if let Some(floor) = thresholds.get(key).and_then(|value| value.as_f64())
                && actual < floor
            {
                eprintln!("taguru-code: eval: {key} {actual:.3} is below the {floor:.3} floor");
                violated = true;
            }
        }
        if let Some(ceiling) = thresholds
            .get("line_drift")
            .and_then(|value| value.as_u64())
            && line_drift as u64 > ceiling
        {
            eprintln!("taguru-code: eval: line_drift {line_drift} exceeds the {ceiling} ceiling");
            violated = true;
        }
        if violated {
            return 3;
        }
    } else {
        eprintln!("note: no --thresholds file — this run is report-only");
    }
    0
}
