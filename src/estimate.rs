//! `taguru estimate`: capacity planning by measurement, not by chart.
//! The image is fixed-width records, so the honest way to answer
//! "what does a corpus of N associations cost" is to BUILD a context
//! of that shape and measure it — `footprint()` and `to_bytes()` here
//! are the very numbers the running server budgets and reports. Past
//! the synthesis cap the totals extrapolate linearly, which the
//! format makes sound: every table grows linearly in its own count.
//! Latency and CPU are deliberately NOT estimated — the benchmark
//! example measures them.

use taguru::context::Context;

use crate::cli::fmt_bytes;

const USAGE: &str = "\
usage: taguru estimate --associations N [--concepts N] [--labels N]
                       [--sources N] [--name-bytes B] [--embedding-dims D]
                       [--passage-bytes B]

defaults: concepts = associations/2 · labels = 50 · sources = associations/20
          name-bytes = 24 (bytes per interned name, UTF-8)
          embedding-dims = 0 (semantic tier off; 3072 = text-embedding-3-large)
          passage-bytes = 0 (total original text registered via /sources)
";

/// Builds get capped here; a million associations synthesize in
/// seconds and extrapolate linearly beyond.
const SYNTHESIS_CAP: u64 = 1_000_000;

struct Plan {
    associations: u64,
    concepts: u64,
    labels: u64,
    sources: u64,
    name_bytes: usize,
    embedding_dims: u64,
    passage_bytes: u64,
}

pub fn run(args: &[String]) -> i32 {
    if args.first().is_some_and(|a| a == "--help" || a == "-h") {
        print!("{USAGE}");
        return 0;
    }
    let plan = match parse(args) {
        Ok(plan) => plan,
        Err(message) => {
            eprintln!("taguru: estimate: {message}");
            eprint!("{USAGE}");
            return 2;
        }
    };

    // Shrink every pool by the same factor as the associations so the
    // shape stays proportional; the linear extrapolation below then
    // scales every component back together.
    let measured_associations = plan.associations.min(SYNTHESIS_CAP);
    let shrink = measured_associations as f64 / plan.associations as f64;
    let scaled = |count: u64, floor: u64| ((count as f64 * shrink).round() as u64).max(floor);
    let measured_concepts = scaled(plan.concepts, 2);
    let measured_labels = scaled(plan.labels, 1);
    let context = synthesize(
        measured_associations,
        measured_concepts,
        measured_labels,
        scaled(plan.sources, 1),
        plan.name_bytes,
    );
    if vocabulary_too_small(&context, measured_associations) {
        eprintln!(
            "taguru: estimate: warning: {measured_concepts} concepts × {measured_labels} labels \
             top out at {} unique associations without repeating a triple; measured that many \
             instead of the requested {measured_associations} — the estimate below is a lower \
             bound, not a measurement of the requested shape",
            context.association_count()
        );
    }
    let factor = plan.associations as f64 / measured_associations as f64;
    let footprint = (context.footprint() as f64 * factor) as u64;
    let image = (context.to_bytes().len() as f64 * factor) as u64;

    // The vector sidecar is arithmetic, not synthesis: one f32 vector
    // and one gloss hash per canonical concept and label.
    let embedded_names = plan.concepts + plan.labels;
    let vectors = if plan.embedding_dims > 0 {
        embedded_names * (plan.embedding_dims * 4 + plan.name_bytes as u64 + 64)
    } else {
        0
    };

    println!(
        "target shape: {} associations · {} concepts · {} labels · {} sources · {}-byte names",
        plan.associations, plan.concepts, plan.labels, plan.sources, plan.name_bytes
    );
    if factor > 1.0 {
        println!(
            "measured a {}-association synthesis ({} concepts · {} labels · {} sources), scaled ×{factor:.1}",
            context.association_count(),
            context.concept_count(),
            context.label_count(),
            context.source_count(),
        );
    } else {
        println!(
            "measured at full scale: {} associations · {} concepts · {} labels · {} sources",
            context.association_count(),
            context.concept_count(),
            context.label_count(),
            context.source_count(),
        );
    }

    println!();
    println!("memory (per loaded context):");
    println!(
        "  graph footprint    {:>12}   (the number the server budgets and reports)",
        fmt_bytes(footprint)
    );
    if vectors > 0 {
        println!(
            "  vector store       {:>12}   ({} dims × 4 B × {} names, resident after semantic use)",
            fmt_bytes(vectors),
            plan.embedding_dims,
            embedded_names
        );
    }
    println!(
        "  TAGURU_CACHE_BYTES ≥ {} to keep this context hot; footprint is modeled",
        fmt_bytes(footprint + vectors)
    );
    println!("  bytes, not RSS — leave ~20-30% container headroom on top.");

    println!();
    println!("disk (per context):");
    println!(
        "  image              {:>12}   (2× transiently during each atomic save)",
        fmt_bytes(image)
    );
    if vectors > 0 {
        println!("  vectors sidecar    {:>12}", fmt_bytes(vectors));
    }
    if plan.passage_bytes > 0 {
        println!(
            "  passages           {:>12}   (original text, JSON-escaped on disk)",
            fmt_bytes(plan.passage_bytes)
        );
    }
    println!("  WAL                truncates after each successful flush;");
    println!("                     ceiling is TAGURU_WAL_MAX_BYTES (default 256 MiB)");

    println!();
    println!("not estimated: latency and CPU — measure those with");
    println!("  cargo run --release --example benchmark");
    0
}

fn parse(args: &[String]) -> Result<Plan, String> {
    let mut associations = None;
    let mut concepts = None;
    let mut labels = None;
    let mut sources = None;
    let mut name_bytes = None;
    let mut embedding_dims = None;
    let mut passage_bytes = None;

    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        let slot = match flag.as_str() {
            "--associations" => &mut associations,
            "--concepts" => &mut concepts,
            "--labels" => &mut labels,
            "--sources" => &mut sources,
            "--name-bytes" => &mut name_bytes,
            "--embedding-dims" => &mut embedding_dims,
            "--passage-bytes" => &mut passage_bytes,
            other => return Err(format!("unknown flag '{other}'")),
        };
        let Some(value) = rest.next() else {
            return Err(format!("{flag} needs a number"));
        };
        // Underscore grouping accepted: 1_000_000 reads better than
        // counting zeros.
        let parsed: u64 = value
            .replace('_', "")
            .parse()
            .map_err(|_| format!("{flag}: '{value}' is not a number"))?;
        if slot.replace(parsed).is_some() {
            return Err(format!("{flag} given twice"));
        }
    }

    let Some(associations) = associations else {
        return Err("--associations is required".to_string());
    };
    if associations == 0 {
        return Err("--associations must be at least 1".to_string());
    }
    let name_bytes = name_bytes.unwrap_or(24).clamp(2, 1024) as usize;
    Ok(Plan {
        associations,
        concepts: concepts.unwrap_or((associations / 2).max(2)).max(2),
        labels: labels.unwrap_or(50).max(1),
        sources: sources.unwrap_or((associations / 20).max(1)).max(1),
        name_bytes,
        embedding_dims: embedding_dims.unwrap_or(0),
        passage_bytes: passage_bytes.unwrap_or(0),
    })
}

/// Whether `synthesize` fell short of `requested` — the vocabulary was
/// too small to mint that many distinct (subject, label, object)
/// triples, so past some point every call accumulated weight onto an
/// already-minted edge instead of creating one. When that happens the
/// footprint below is measured on fewer associations than asked for.
fn vocabulary_too_small(measured: &Context, requested: u64) -> bool {
    (measured.association_count() as u64) < requested
}

/// Builds a context of the requested shape. Subjects sweep the whole
/// concept pool (`i % concepts`), so the arena and entry index carry
/// every name.
///
/// Within one subject, `round` (how many times the subject pool has
/// cycled) picks the (object, label) pair: `label` walks all `labels`
/// values first, and only once that walk completes does the object
/// offset advance to a fresh value (never 0, so the object is never
/// the subject itself). That combined period is the full
/// `(concepts - 1) * labels` — the true count of non-self-loop
/// (object, label) pairs available to one subject — so triples stay
/// unique that much longer than a scheme that advances both from the
/// same `round` directly, which collides every
/// `lcm(concepts, labels)` rounds instead (far sooner whenever
/// `concepts` and `labels` share a large factor, e.g. small explicit
/// `--concepts`/`--labels`). Past that period the vocabulary is
/// simply too small for the requested count and triples repeat,
/// accumulating weight instead of minting new edges — see the
/// `association_count` shortfall this guards against in
/// `small_vocabulary_does_not_collapse_associations_via_collisions`.
///
/// `concepts - 1` divides below, so this relies on `concepts >= 2`
/// (guaranteed today by `parse`'s `.max(2)` and the `scaled` floor in
/// `run`); a future caller of this function must keep that invariant.
fn synthesize(
    associations: u64,
    concepts: u64,
    labels: u64,
    sources: u64,
    name_bytes: usize,
) -> Context {
    let mut context = Context::default();
    for i in 0..associations {
        let subject_index = i % concepts;
        let round = i / concepts;
        let label_index = round % labels;
        let object_offset = 1 + (round / labels) % (concepts - 1);
        let object_index = (subject_index + object_offset) % concepts;
        let subject = synthetic_name('c', subject_index, name_bytes);
        let object = synthetic_name('c', object_index, name_bytes);
        let label = synthetic_name('l', label_index, name_bytes);
        let source = synthetic_name('s', i % sources, name_bytes);
        context
            .associate_from(&subject, &label, &object, 1.0, &source)
            .expect("a synthetic corpus stays far under u32 capacity");
    }
    context
}

/// A unique name of exactly `width` bytes (digits win over the width
/// when an index needs more room — uniqueness beats exactness).
fn synthetic_name(prefix: char, index: u64, width: usize) -> String {
    let mut name = format!("{prefix}{index}");
    while name.len() < width {
        name.push('x');
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_names_hold_their_width_and_stay_unique() {
        let name = synthetic_name('c', 7, 24);
        assert_eq!(name.len(), 24);
        assert_ne!(synthetic_name('c', 1, 8), synthetic_name('c', 11, 8));
    }

    #[test]
    fn synthesis_delivers_the_requested_shape() {
        let context = synthesize(10_000, 500, 20, 100, 24);
        assert_eq!(context.association_count(), 10_000);
        assert_eq!(context.concept_count(), 500);
        assert_eq!(context.source_count(), 100);
        // Label coverage is bounded by the rounds (associations /
        // concepts); at 20 rounds all 20 labels are touched.
        assert_eq!(context.label_count(), 20);
    }

    /// With a small explicit vocabulary, the old "offset object and
    /// label both directly by round" scheme collided every
    /// `lcm(concepts, labels)` rounds — for concepts=4, labels=2 that's
    /// every 4 rounds, so 24 requested associations collapsed onto 16
    /// unique triples (repeats just accumulate weight onto an existing
    /// edge instead of minting a new one). The true limit here is
    /// `(concepts - 1) * labels` = 6 rounds per subject, i.e. 24
    /// associations across the 4 subjects — exactly what was asked
    /// for.
    #[test]
    fn small_vocabulary_does_not_collapse_associations_via_collisions() {
        let context = synthesize(24, 4, 2, 1, 8);
        assert_eq!(context.association_count(), 24);
    }

    #[test]
    fn vocabulary_too_small_flags_only_the_infeasible_shape() {
        // (concepts - 1) * labels = 3 * 2 = 6 unique triples per
        // subject × 4 subjects = 24 max; asking for 25 cannot be met.
        let short = synthesize(25, 4, 2, 1, 8);
        assert!(vocabulary_too_small(&short, 25));

        let exact = synthesize(24, 4, 2, 1, 8);
        assert!(!vocabulary_too_small(&exact, 24));
    }

    #[test]
    fn doubling_the_associations_scales_the_measurements_roughly_linearly() {
        let small = synthesize(5_000, 250, 10, 50, 24);
        let large = synthesize(10_000, 500, 10, 100, 24);
        let footprint_ratio = large.footprint() as f64 / small.footprint() as f64;
        let image_ratio = large.to_bytes().len() as f64 / small.to_bytes().len() as f64;
        assert!(
            (1.6..=2.4).contains(&footprint_ratio),
            "footprint ratio {footprint_ratio}"
        );
        assert!(
            (1.6..=2.4).contains(&image_ratio),
            "image ratio {image_ratio}"
        );
    }

    #[test]
    fn parse_fills_the_documented_defaults() {
        let plan = parse(&["--associations".into(), "1_000_000".into()]).unwrap();
        assert_eq!(plan.associations, 1_000_000);
        assert_eq!(plan.concepts, 500_000);
        assert_eq!(plan.labels, 50);
        assert_eq!(plan.sources, 50_000);
        assert_eq!(plan.name_bytes, 24);
        assert_eq!(plan.embedding_dims, 0);
    }

    #[test]
    fn parse_rejects_nonsense() {
        assert!(parse(&[]).is_err());
        assert!(parse(&["--associations".into(), "abc".into()]).is_err());
        assert!(parse(&["--frobnicate".into(), "1".into()]).is_err());
        assert!(
            parse(&[
                "--associations".into(),
                "1".into(),
                "--associations".into(),
                "2".into()
            ])
            .is_err()
        );
    }
}
