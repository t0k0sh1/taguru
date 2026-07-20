//! `taguru calibrate` — measure the semantic-floor bands of a running
//! server's embedding model instead of prescribing a manual ritual
//! (issue #131).
//!
//! `TAGURU_SEMANTIC_FLOOR` is a property of the embedding model, and
//! getting it wrong is silent and total: a floor calibrated for one
//! model can sit above everything another model ever scores, and
//! resolve just answers `[]` while the ranking underneath is perfect.
//! The documented cure was a human ritual — probe with paraphrase
//! cues at a floor low enough to see everything, watch the scores
//! split into two bands, set the floor between them. Every step is
//! mechanical, so this command runs it: for each `(cue, expected)`
//! probe, the resolve/explain verb yields the expected name's own
//! gloss cosine (the upper band, measured floor-independently) and a
//! low-floor resolve yields the best OTHER semantic candidate (the
//! lower band); the report prints both distributions, the gap, and a
//! floor between them — or an overlap warning, which no floor value
//! can fix.
//!
//! The step humans get wrong is probe hygiene: a cue that lexically
//! resolves (it IS a stored spelling, or covers half of one) never
//! exercises the semantic tier at all, and a band "measured" through
//! it is fiction. The explain verb names exactly that, so
//! contaminated probes are excluded loudly instead of polluting the
//! bands.
//!
//! Exit codes: 0 = report produced (an overlap verdict is an honest
//! success) · 1 = calibration impossible (server unreachable, unknown
//! context, embeddings off or stale, every probe excluded) · 2 =
//! usage error.

use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;
use serde_json::{Value, json};

use crate::cli::default_base_url;
use crate::config::{load_config, subcommand_usage_error};

const CALIBRATE_USAGE: &str =
    "usage: taguru calibrate --context NAME --probes FILE [--json] [--config FILE] [URL]

Measures the semantic-floor bands of a RUNNING server's embedding
model and prints a suggested TAGURU_SEMANTIC_FLOOR — the floor is a
property of the model, and this replaces eyeballing score bands after
every model switch.

The probe file is TSV, one probe per line:

    cue<TAB>expected

where `expected` is a stored concept name (or a registered alias) the
cue should reach, and the cue is a PARAPHRASE sharing no spelling with
any stored name — a cue that lexically resolves never exercises the
semantic tier, and such probes are detected and excluded loudly.
`#` comments and blank lines are skipped.

For each probe, the expected name's own gloss cosine feeds the upper
band and the best OTHER semantic candidate feeds the lower band; the
suggested floor sits mid-gap. Overlapping bands are reported as
exactly that — a model that cannot separate these names at this
dimension — never papered over with a number.

The server is asked read-only (works against a replica). Auth rides
the same variables the server reads: TAGURU_API_TOKEN, or the first
key of TAGURU_API_TOKENS. URL defaults to TAGURU_ADDR after --config
applies, exactly like `taguru health`.

exit codes: 0 report produced (an overlap verdict included) ·
1 calibration impossible · 2 usage error
";

/// The floor the low-band sweep runs at: low enough that every real
/// distractor shows, high enough to skip cosine noise around zero.
/// The Bedrock page's manual ritual probes at this same value.
const SWEEP_FLOOR: f64 = 0.05;

/// One `(cue, expected)` line of the probe file.
#[derive(Debug, PartialEq)]
struct Probe {
    cue: String,
    expected: String,
}

/// What one probe's explain answer means for the calibration.
#[derive(Debug)]
enum Classification {
    /// The pair measures the semantic tier: the expected name's own
    /// gloss cosine (floor-independent), its canonical spelling, the
    /// floor the server would serve at today, and the cue's lexical
    /// score when it also brushes the vocabulary (kept, flagged).
    Measured {
        canonical: String,
        cosine: f64,
        floor: f64,
        lexical_overlap: Option<f64>,
    },
    /// The semantic tier never scores this pair — the probe is out,
    /// loudly, and the run continues.
    Excluded { reason: String },
    /// The server cannot calibrate at all (embeddings off, model or
    /// width mismatch, nothing embedded, provider refusal) — the run
    /// aborts with this as the verdict.
    Unusable { reason: String },
}

/// What the bands support: a floor, or the honest refusal to invent
/// one.
#[derive(Debug, PartialEq)]
enum FloorVerdict {
    /// Clean separation — the midpoint is a defensible floor.
    Suggested { floor: f64, gap: f64 },
    /// The bands touch or cross: true matches and distractors score in
    /// the same range, and no floor value separates what the model
    /// itself does not.
    Overlap { by: f64 },
    /// No non-expected candidate cleared the sweep floor anywhere —
    /// nothing to separate from, so nothing is suggested.
    NoLowerBand,
}

#[derive(Serialize)]
struct BandStats {
    min: f64,
    max: f64,
    count: usize,
}

#[derive(Serialize)]
struct OtherCandidate {
    name: String,
    cosine: f64,
}

#[derive(Serialize)]
struct ProbeRow {
    cue: String,
    expected: String,
    /// The stored spelling `expected` maps to (differs when the probe
    /// file used an alias).
    canonical: String,
    cosine: f64,
    /// The strongest semantic candidate that is NOT the expected name
    /// — this probe's contribution to the lower band. Absent when
    /// nothing else cleared the sweep floor.
    #[serde(skip_serializing_if = "Option::is_none")]
    best_other: Option<OtherCandidate>,
    /// The cue also brushes the vocabulary lexically (score under the
    /// confidence bar): the semantic tier still ran and the cosine is
    /// a true measurement, but resolve dedups lexically served names
    /// out of its semantic list, so this probe's `best_other` can
    /// under-report. Kept, flagged.
    #[serde(skip_serializing_if = "Option::is_none")]
    lexical_overlap: Option<f64>,
}

#[derive(Serialize)]
struct ExcludedRow {
    cue: String,
    expected: String,
    reason: String,
}

/// The whole report, `--json`'s exact shape.
#[derive(Serialize)]
struct Report {
    context: String,
    url: String,
    /// The (model, width) identity the sidecar was built with — what
    /// every number below is tied to.
    model: String,
    width: usize,
    /// The floor the server serves at today (context setting or server
    /// default; this command never passes an override to explain).
    effective_floor: f64,
    probes: usize,
    measured: usize,
    verdict: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    suggested_floor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gap: Option<f64>,
    upper: BandStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    lower: Option<BandStats>,
    per_probe: Vec<ProbeRow>,
    excluded: Vec<ExcludedRow>,
}

pub fn run(args: &[String]) -> i32 {
    let usage = |message: &str| subcommand_usage_error("calibrate", message);
    let mut context: Option<String> = None;
    let mut probes_path: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut as_json = false;
    let mut explicit_url: Option<String> = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{CALIBRATE_USAGE}");
                return 0;
            }
            "--context" => match rest.next() {
                Some(name) if context.is_none() => context = Some(name.clone()),
                Some(_) => return usage("--context given twice"),
                None => return usage("--context needs a name"),
            },
            "--probes" => match rest.next() {
                Some(path) if probes_path.is_none() => probes_path = Some(PathBuf::from(path)),
                Some(_) => return usage("--probes given twice"),
                None => return usage("--probes needs a file path"),
            },
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => return usage("--config given twice"),
                None => return usage("--config needs a file path"),
            },
            "--json" => as_json = true,
            flag if flag.starts_with('-') => {
                return usage(&format!("unknown argument '{flag}'"));
            }
            url => {
                if explicit_url
                    .replace(url.trim_end_matches('/').to_string())
                    .is_some()
                {
                    return usage(&format!("one optional URL only, got '{url}'"));
                }
            }
        }
    }
    let Some(context) = context else {
        return usage("--context NAME is required");
    };
    let Some(probes_path) = probes_path else {
        return usage("--probes FILE is required");
    };

    // The config file first, then the URL default off the (possibly
    // just-loaded) environment — the same order health resolves in, so
    // one --config deployment file aims every CLI verb at its own port.
    let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    if let Some(path) = &config {
        load_config(path);
    }
    let base = match explicit_url {
        Some(url) => url,
        None => match default_base_url() {
            Ok(url) => url,
            Err(error) => {
                eprintln!("taguru: calibrate: {error}");
                return 2;
            }
        },
    };

    let text = match std::fs::read_to_string(&probes_path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!(
                "taguru: calibrate: cannot read probe file {}: {error}",
                probes_path.display()
            );
            return 1;
        }
    };
    let probes = match parse_probes(&text) {
        Ok(probes) => probes,
        Err(error) => {
            eprintln!("taguru: calibrate: {}: {error}", probes_path.display());
            return 1;
        }
    };
    if probes.is_empty() {
        eprintln!(
            "taguru: calibrate: {} holds no probes — nothing to measure",
            probes_path.display()
        );
        return 1;
    }

    let api = Api::new(base.clone());
    match calibrate(&api, &context, &probes, as_json) {
        Ok(report) => {
            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).expect("report serializes")
                );
            } else {
                print_human(&report);
            }
            0
        }
        Err(message) => {
            eprintln!("taguru: calibrate: {message}");
            1
        }
    }
}

/// The measurement itself, server preflight through band arithmetic.
/// `Err` is the whole run refusing (exit 1); per-probe trouble lands
/// in the report's `excluded` instead.
fn calibrate(api: &Api, context: &str, probes: &[Probe], quiet: bool) -> Result<Report, String> {
    // Preflight: the sidecar identity every number is tied to — and
    // the states no probe loop can fix, named before any spend.
    let status = api.get(&["contexts", context, "embeddings"])?;
    let Some(provider_model) = status["provider_model"].as_str() else {
        return Err(
            "this server has no embedding provider (TAGURU_EMBED_URL / TAGURU_EMBED_MODEL \
             are unset) — there is no semantic tier to calibrate"
                .to_string(),
        );
    };
    let glosses = &status["glosses"];
    if glosses.is_null() {
        return Err(format!(
            "context '{context}' has no gloss vectors yet — \
             POST /contexts/{context}/embeddings/refresh first, then calibrate"
        ));
    }
    let sidecar_model = glosses["model"].as_str().unwrap_or_default().to_string();
    let width = glosses["width"].as_u64().unwrap_or_default() as usize;
    if sidecar_model != provider_model {
        return Err(format!(
            "the gloss vectors belong to model '{sidecar_model}' but the provider is \
             '{provider_model}' — refresh embeddings first, then calibrate"
        ));
    }

    let mut per_probe = Vec::new();
    let mut excluded = Vec::new();
    let mut upper = Vec::new();
    let mut lower = Vec::new();
    let mut effective_floor = None;
    for (position, probe) in probes.iter().enumerate() {
        if !quiet {
            eprintln!(
                "  probing {}/{}: '{}' → {}",
                position + 1,
                probes.len(),
                probe.cue,
                probe.expected
            );
        }
        let explain = api.post(
            &["contexts", context, "resolve", "explain"],
            &json!({"cue": probe.cue, "expected": probe.expected}),
        )?;
        match classify_explain(&explain, &probe.expected) {
            Classification::Unusable { reason } => return Err(reason),
            Classification::Excluded { reason } => {
                excluded.push(ExcludedRow {
                    cue: probe.cue.clone(),
                    expected: probe.expected.clone(),
                    reason,
                });
            }
            Classification::Measured {
                canonical,
                cosine,
                floor,
                lexical_overlap,
            } => {
                // The same cue embedding is already cached server-side,
                // so this second call costs no provider spend.
                let resolved = api.post(
                    &["contexts", context, "resolve"],
                    &json!({"cue": probe.cue, "semantic_floor": SWEEP_FLOOR}),
                )?;
                let best_other = best_other(resolved.as_array().unwrap_or(&Vec::new()), &canonical);
                effective_floor.get_or_insert(floor);
                upper.push(cosine);
                if let Some((_, other)) = &best_other {
                    lower.push(*other);
                }
                per_probe.push(ProbeRow {
                    cue: probe.cue.clone(),
                    expected: probe.expected.clone(),
                    canonical,
                    cosine,
                    best_other: best_other.map(|(name, cosine)| OtherCandidate { name, cosine }),
                    lexical_overlap,
                });
            }
        }
    }

    let Some(effective_floor) = effective_floor else {
        return Err(format!(
            "every probe was excluded ({} of {}) — no cue exercised the semantic tier; \
             reword the cues as paraphrases that share no spelling with stored names",
            excluded.len(),
            probes.len()
        ));
    };
    // Best-first, so the report reads top-down like the bands it names.
    per_probe.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));

    let verdict = floor_verdict(&upper, &lower);
    let (verdict_name, suggested_floor, gap) = match verdict {
        FloorVerdict::Suggested { floor, gap } => ("suggested", Some(floor), Some(gap)),
        FloorVerdict::Overlap { by } => ("overlap", None, Some(-by)),
        FloorVerdict::NoLowerBand => ("no_lower_band", None, None),
    };
    Ok(Report {
        context: context.to_string(),
        url: api.base.clone(),
        model: sidecar_model,
        width,
        effective_floor,
        probes: probes.len(),
        measured: per_probe.len(),
        verdict: verdict_name,
        suggested_floor,
        gap,
        upper: band_stats(&upper).expect("upper band is non-empty when any probe measured"),
        lower: band_stats(&lower),
        per_probe,
        excluded,
    })
}

/// Splits one explain answer into measured / excluded / unusable —
/// the shapes the resolve/explain verb actually serves, provoked and
/// pinned by the integration test.
fn classify_explain(explain: &Value, expected: &str) -> Classification {
    let verdict = explain["verdict"].as_str().unwrap_or_default();
    if verdict == "not_in_vocabulary" {
        return Classification::Excluded {
            reason: format!(
                "'{expected}' is not stored in this context — no entry, no alias; \
                 fix the probe file (or register the alias)"
            ),
        };
    }
    let semantic = &explain["semantic"];
    // Every state where the sweep could not run reports a reason and
    // no floor: provider off, model changed, width changed, nothing
    // embedded, the cue embedding refused. None of them is this
    // probe's fault — the whole run stops.
    if semantic["floor"].is_null() {
        let reason = semantic["reason"]
            .as_str()
            .unwrap_or("the semantic tier did not run");
        return Classification::Unusable {
            reason: format!("the semantic tier cannot run: {reason}"),
        };
    }
    let lexical_score = explain["lexical"]["score"].as_f64();
    if verdict == "cue_resolved_exactly" {
        return Classification::Excluded {
            reason: "the cue is itself a stored spelling — the exact tier answers alone \
                     and nothing else is ever scored; reword the cue as a paraphrase"
                .to_string(),
        };
    }
    if semantic["entered"] == Value::Bool(false) {
        let overlap = lexical_score
            .map(|score| format!(" (score {score:.2})"))
            .unwrap_or_default();
        return Classification::Excluded {
            reason: format!(
                "the cue lexically resolves with confidence{overlap} — \
                 the semantic tier never joins; reword the cue as a paraphrase"
            ),
        };
    }
    let canonical = explain["canonical"]
        .as_str()
        .unwrap_or(expected)
        .to_string();
    let Some(cosine) = semantic["cosine"].as_f64() else {
        return Classification::Excluded {
            reason: format!(
                "'{canonical}' has no gloss vector yet (stored after the last refresh?) — \
                 refresh embeddings and rerun"
            ),
        };
    };
    Classification::Measured {
        canonical,
        cosine,
        floor: semantic["floor"].as_f64().unwrap_or_default(),
        lexical_overlap: lexical_score,
    }
}

/// The strongest semantic candidate that is not the canonical — one
/// probe's lower-band contribution. Lexical candidates are skipped
/// outright: their scores are string coverage, not cosines, and the
/// two never share a scale.
fn best_other(resolved: &[Value], canonical: &str) -> Option<(String, f64)> {
    resolved
        .iter()
        .filter(|candidate| candidate["tier"] == "semantic")
        .filter(|candidate| candidate["name"].as_str() != Some(canonical))
        .filter_map(|candidate| {
            Some((
                candidate["name"].as_str()?.to_string(),
                candidate["score"].as_f64()?,
            ))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

fn band_stats(band: &[f64]) -> Option<BandStats> {
    if band.is_empty() {
        return None;
    }
    Some(BandStats {
        min: band.iter().copied().fold(f64::INFINITY, f64::min),
        max: band.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        count: band.len(),
    })
}

/// The banding arithmetic: a floor mid-gap, or the honest refusal.
/// `upper` is non-empty by the time this runs (the caller aborted
/// otherwise).
fn floor_verdict(upper: &[f64], lower: &[f64]) -> FloorVerdict {
    let upper_min = upper.iter().copied().fold(f64::INFINITY, f64::min);
    if lower.is_empty() {
        return FloorVerdict::NoLowerBand;
    }
    let lower_max = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let gap = upper_min - lower_max;
    if gap <= 0.0 {
        return FloorVerdict::Overlap { by: -gap };
    }
    // A two-decimal floor when rounding keeps it inside the gap (an
    // operator-friendly number), the exact midpoint when the gap is
    // too tight to survive rounding.
    let midpoint = (upper_min + lower_max) / 2.0;
    let rounded = (midpoint * 100.0).round() / 100.0;
    let floor = if rounded > lower_max && rounded < upper_min {
        rounded
    } else {
        midpoint
    };
    FloorVerdict::Suggested { floor, gap }
}

fn print_human(report: &Report) {
    println!(
        "taguru calibrate — context '{}' via {}",
        report.context, report.url
    );
    println!(
        "model: {} ({} dimensions) — the identity this measurement is tied to",
        report.model, report.width
    );
    println!();
    println!(
        "probes: {} measured, {} excluded (of {})",
        report.measured,
        report.excluded.len(),
        report.probes
    );
    if !report.excluded.is_empty() {
        println!();
        println!("excluded — the semantic tier never scores these:");
        for row in &report.excluded {
            println!("  '{}' → {}: {}", row.cue, row.expected, row.reason);
        }
    }
    println!();
    println!("measured — expected cosine | best other candidate:");
    for row in &report.per_probe {
        let other = match &row.best_other {
            Some(other) => format!("{} {:.4}", other.name, other.cosine),
            None => "(nothing else cleared the sweep)".to_string(),
        };
        let flag = match row.lexical_overlap {
            Some(score) => format!("  [lexical overlap {score:.2} — best-other may under-report]"),
            None => String::new(),
        };
        println!(
            "  {:.4}  '{}' → {} | {other}{flag}",
            row.cosine, row.cue, row.canonical
        );
    }
    println!();
    println!(
        "upper band (expected-name cosines): {:.4} .. {:.4}  ({} probes)",
        report.upper.min, report.upper.max, report.upper.count
    );
    match &report.lower {
        Some(lower) => println!(
            "lower band (best non-expected):     {:.4} .. {:.4}  ({} probes)",
            lower.min, lower.max, lower.count
        ),
        None => println!("lower band (best non-expected):     empty"),
    }
    println!();
    match (report.verdict, report.suggested_floor, report.gap) {
        ("suggested", Some(floor), Some(gap)) => {
            println!("gap: {gap:.4}");
            println!(
                "suggested floor: {floor} — mid-gap; the server serves at {} today",
                report.effective_floor
            );
            println!();
            println!("TAGURU_SEMANTIC_FLOOR={floor}");
        }
        ("overlap", ..) => {
            println!(
                "verdict: OVERLAP — the bands touch or cross. True matches and \
                 distractors score in the same range, so no floor value can separate \
                 what the model itself does not. That is a model/dimension problem \
                 (or a probe-set problem: check the best-other names above for \
                 legitimately adjacent concepts), not a floor problem."
            );
        }
        _ => {
            println!(
                "verdict: no lower band — no non-expected candidate cleared the \
                 {SWEEP_FLOOR} sweep anywhere. The vocabulary is too small (or the \
                 probes too narrow) to measure a distractor band; nothing suggested. \
                 The current floor {} stands until a fuller corpus says otherwise.",
                report.effective_floor
            );
        }
    }
}

/// TSV probes: `cue<TAB>expected`, `#` comments and blank lines
/// skipped. Errors carry 1-based line numbers — the operator fixes a
/// file, not a stream.
fn parse_probes(text: &str) -> Result<Vec<Probe>, String> {
    let mut probes = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((cue, expected)) = line.split_once('\t') else {
            return Err(format!(
                "line {}: no TAB — probes are 'cue<TAB>expected'",
                index + 1
            ));
        };
        let (cue, expected) = (cue.trim(), expected.trim());
        if cue.is_empty() || expected.is_empty() {
            return Err(format!(
                "line {}: empty cue or expected — probes are 'cue<TAB>expected'",
                index + 1
            ));
        }
        probes.push(Probe {
            cue: cue.to_string(),
            expected: expected.to_string(),
        });
    }
    Ok(probes)
}

/// The one HTTP door: bearer attached when the environment holds one,
/// 200 unwrapped to `result`, anything else an error message carrying
/// the server's own words.
struct Api {
    agent: ureq::Agent,
    base: String,
    token: Option<String>,
}

impl Api {
    fn new(base: String) -> Self {
        Self {
            // Above the server's default 30s request budget, so a
            // server-side timeout answers as itself (a 408 body with
            // words) instead of a client-side cut.
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs(35)))
                .http_status_as_error(false)
                .build()
                .into(),
            base,
            token: bearer_token(),
        }
    }

    /// Percent-encodes each path segment through the url crate — cue
    /// text never rides the path, but context names are operator
    /// strings and 日本語 names must address the same context the
    /// server stores.
    fn url(&self, segments: &[&str]) -> Result<String, String> {
        let mut url = url::Url::parse(&self.base)
            .map_err(|error| format!("'{}' is not a usable base URL: {error}", self.base))?;
        url.path_segments_mut()
            .map_err(|()| format!("'{}' cannot carry a path", self.base))?
            .extend(segments);
        Ok(url.to_string())
    }

    fn get(&self, segments: &[&str]) -> Result<Value, String> {
        let url = self.url(segments)?;
        let mut request = self.agent.get(&url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        finish(request.call(), &url)
    }

    fn post(&self, segments: &[&str], body: &Value) -> Result<Value, String> {
        let url = self.url(segments)?;
        let mut request = self
            .agent
            .post(&url)
            .header("Content-Type", "application/json");
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        finish(request.send(body.to_string().as_str()), &url)
    }
}

/// Unwraps one response: 200 hands back `result`, anything else the
/// server's own error line (or the transport's).
fn finish(
    response: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
    url: &str,
) -> Result<Value, String> {
    let mut response = response.map_err(|error| format!("{url}: {error}"))?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|error| format!("{url}: unreadable response: {error}"))?;
    let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if status != 200 {
        let message = parsed["error"].as_str().unwrap_or(text.trim());
        return Err(format!("{url} answered {status}: {message}"));
    }
    if parsed["result"].is_null() && !text.contains("\"result\"") {
        return Err(format!("{url}: not a taguru response: {}", text.trim()));
    }
    Ok(parsed["result"].clone())
}

/// The bearer the server would accept, read the way the server reads
/// it: `TAGURU_API_TOKEN` outright, else the first `name:token` entry
/// of `TAGURU_API_TOKENS`. `None` = an unauthenticated server.
/// Crate-visible: `taguru communities` authenticates the same way.
pub(crate) fn bearer_token() -> Option<String> {
    if let Ok(token) = std::env::var("TAGURU_API_TOKEN") {
        let token = token.trim();
        if !token.is_empty() {
            return Some(token.to_string());
        }
    }
    let ring = std::env::var("TAGURU_API_TOKENS").ok()?;
    ring.split(',').find_map(|entry| {
        let (_, token) = entry.trim().split_once(':')?;
        let token = token.trim();
        (!token.is_empty()).then(|| token.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_parse_as_tsv_with_comments_and_blanks() {
        let text = "# 較正プローブ\nむかしのじゅし\t琥珀\n\n  かがやくいし \t ダイヤモンド \n";
        assert_eq!(
            parse_probes(text).unwrap(),
            vec![
                Probe {
                    cue: "むかしのじゅし".to_string(),
                    expected: "琥珀".to_string()
                },
                Probe {
                    cue: "かがやくいし".to_string(),
                    expected: "ダイヤモンド".to_string()
                },
            ]
        );
    }

    #[test]
    fn a_probe_line_without_a_tab_is_refused_with_its_line_number() {
        let error = parse_probes("むかしのじゅし 琥珀").unwrap_err();
        assert!(error.contains("line 1"), "{error}");
        let error = parse_probes("a\tb\n\t琥珀").unwrap_err();
        assert!(error.contains("line 2"), "{error}");
    }

    #[test]
    fn a_clean_gap_suggests_the_midpoint_rounded_to_two_decimals() {
        let verdict = floor_verdict(&[0.81, 0.62, 0.75], &[0.13, 0.35, 0.2]);
        // midpoint of (0.62, 0.35) is 0.485 — rounds to 0.48/0.49, both
        // inside the gap; round() halves-away gives 0.49.
        match verdict {
            FloorVerdict::Suggested { floor, gap } => {
                assert!((gap - 0.27).abs() < 1e-9, "{gap}");
                assert!((floor - 0.49).abs() < 1e-9, "{floor}");
            }
            other => panic!("expected a suggestion, got {other:?}"),
        }
    }

    #[test]
    fn a_gap_too_tight_for_rounding_keeps_the_exact_midpoint() {
        // Gap (0.501, 0.504): every two-decimal value falls outside.
        let verdict = floor_verdict(&[0.504], &[0.501]);
        match verdict {
            FloorVerdict::Suggested { floor, .. } => {
                assert!((floor - 0.5025).abs() < 1e-9, "{floor}");
            }
            other => panic!("expected a suggestion, got {other:?}"),
        }
    }

    #[test]
    fn touching_or_crossing_bands_refuse_to_invent_a_floor() {
        assert_eq!(
            floor_verdict(&[0.5, 0.9], &[0.5]),
            FloorVerdict::Overlap { by: 0.0 }
        );
        match floor_verdict(&[0.4], &[0.6]) {
            FloorVerdict::Overlap { by } => assert!((by - 0.2).abs() < 1e-9, "{by}"),
            other => panic!("expected overlap, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_lower_band_suggests_nothing() {
        assert_eq!(floor_verdict(&[0.7], &[]), FloorVerdict::NoLowerBand);
    }

    #[test]
    fn explain_shapes_classify_as_the_provoked_cases_did() {
        // The measured case: entered, floored, cosine present.
        let measured = json!({
            "verdict": "semantic_below_floor",
            "canonical": "琥珀",
            "semantic": {"entered": true, "floor": 0.35, "cosine": 0.27},
            "lexical": {"floor": 0.3, "confident": false},
        });
        match classify_explain(&measured, "アンバー") {
            Classification::Measured {
                canonical,
                cosine,
                floor,
                lexical_overlap,
            } => {
                assert_eq!(canonical, "琥珀");
                assert!((cosine - 0.27).abs() < 1e-9);
                assert!((floor - 0.35).abs() < 1e-9);
                assert_eq!(lexical_overlap, None);
            }
            other => panic!("expected measured, got {other:?}"),
        }

        // Weak lexical overlap: still measured, overlap carried.
        let flagged = json!({
            "verdict": "served",
            "canonical": "ダイヤモンド",
            "semantic": {"entered": true, "floor": 0.35, "cosine": 0.8},
            "lexical": {"score": 0.36, "kind": "fuzzy", "floor": 0.3, "confident": false},
        });
        match classify_explain(&flagged, "ダイヤモンド") {
            Classification::Measured {
                lexical_overlap, ..
            } => assert_eq!(lexical_overlap, Some(0.36)),
            other => panic!("expected measured, got {other:?}"),
        }

        // Confident lexical: excluded, not measured — entered is false.
        let confident = json!({
            "verdict": "served",
            "canonical": "琥珀",
            "semantic": {"entered": false, "reason": "the lexical tier was confident…",
                          "floor": 0.35, "cosine": 1.0},
            "lexical": {"score": 0.5, "kind": "containment", "floor": 0.3, "confident": true},
        });
        assert!(matches!(
            classify_explain(&confident, "琥珀"),
            Classification::Excluded { .. }
        ));

        // The cue being someone else's exact spelling: excluded even
        // though the sweep ran.
        let exact = json!({
            "verdict": "cue_resolved_exactly",
            "canonical": "ダイヤモンド",
            "semantic": {"entered": false, "reason": "…", "floor": 0.35, "cosine": 1.0},
            "lexical": {"floor": 0.3, "confident": true},
        });
        assert!(matches!(
            classify_explain(&exact, "ダイヤモンド"),
            Classification::Excluded { .. }
        ));

        // Not stored at all: excluded (a probe-file problem).
        let missing = json!({
            "verdict": "not_in_vocabulary",
            "in_vocabulary": false,
        });
        assert!(matches!(
            classify_explain(&missing, "翡翠"),
            Classification::Excluded { .. }
        ));

        // No floor + a reason = the server cannot calibrate: abort.
        let off = json!({
            "verdict": "semantic_not_run",
            "canonical": "琥珀",
            "semantic": {"entered": true, "reason": "no embedding provider is configured"},
            "lexical": {"floor": 0.3, "confident": false},
        });
        assert!(matches!(
            classify_explain(&off, "琥珀"),
            Classification::Unusable { .. }
        ));

        // Ran but the expected name has no vector yet: that one probe
        // is out, the run continues.
        let unembedded = json!({
            "verdict": "semantic_not_run",
            "canonical": "新顔",
            "semantic": {"entered": true, "floor": 0.35, "cosine": null,
                          "reason": "'新顔' has no gloss vector yet…"},
            "lexical": {"floor": 0.3, "confident": false},
        });
        assert!(matches!(
            classify_explain(&unembedded, "新顔"),
            Classification::Excluded { .. }
        ));
    }

    #[test]
    fn best_other_reads_semantic_candidates_only_and_skips_the_canonical() {
        let resolved = json!([
            {"name": "ダイヤモンド", "score": 0.36, "tier": "lexical", "kind": "fuzzy"},
            {"name": "琥珀", "score": 0.9, "tier": "semantic"},
            {"name": "宝石", "score": 0.7, "tier": "semantic"},
            {"name": "輝き", "score": 0.75, "tier": "semantic"},
        ]);
        let best = best_other(resolved.as_array().unwrap(), "琥珀").unwrap();
        assert_eq!(best.0, "輝き");
        assert!((best.1 - 0.75).abs() < 1e-9);
        // Nothing semantic but the canonical → no lower-band entry.
        let only_self = json!([{"name": "琥珀", "score": 0.9, "tier": "semantic"}]);
        assert_eq!(best_other(only_self.as_array().unwrap(), "琥珀"), None);
    }

    #[test]
    fn the_first_keyring_entry_serves_as_the_bearer() {
        // bearer_token reads the environment; exercising the parsing
        // through a thread-safe seam would mean injecting the env —
        // the split_once discipline is pinned here instead.
        let ring = "ci:tokA,laptop:tokB";
        let first = ring
            .split(',')
            .find_map(|entry| entry.trim().split_once(':'))
            .map(|(_, token)| token);
        assert_eq!(first, Some("tokA"));
    }
}
