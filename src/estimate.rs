//! `taguru estimate`: capacity planning by measurement, not by chart.
//! The image is fixed-width records, so the honest way to answer
//! "what does a corpus of N associations cost" is to BUILD a context
//! of that shape and measure it — `footprint()` and `to_bytes()` here
//! are the very numbers the running server budgets and reports. Past
//! the synthesis cap the totals extrapolate linearly, which the
//! format makes sound: every table grows linearly in its own count.
//! Latency and CPU are deliberately NOT estimated — the benchmark
//! example measures them.

use std::sync::Arc;

use taguru::context::Context;

use crate::bm25::Bm25Index;
use crate::cli::fmt_bytes;
use crate::embedding::{PassageKey, PassageVectorStore};
use crate::passages::{PassageRecord, record_bytes};
use crate::registry::DEFAULT_PASSAGE_VECTOR_LIMIT;

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

/// Passage measurement gets its own cap, independent of the graph's:
/// building gigabytes of synthetic text just to measure per-byte
/// overhead would make `estimate` itself slow enough to undermine its
/// own pitch (BUILD and measure, not chart). Every structure measured
/// against the sample — the passage record, BM25's postings — grows
/// linearly in input size, so it extrapolates the same way the graph
/// synthesis above does.
const PASSAGE_SAMPLE_CAP: u64 = 16 * 1024 * 1024;

/// The paragraph size `synthetic_passage_text` always emits at least one
/// of, however few bytes it's asked for. `estimate_passages` caps its
/// source-sample count against this floor: without it, a huge `--sources`
/// derived from a huge `--associations` paired with a tiny
/// `--passage-bytes` would synthesize one such paragraph per sampled
/// source regardless of `PASSAGE_SAMPLE_CAP` — tens of thousands of
/// sources each contributing ~400 bytes no matter how little text
/// `--passage-bytes` actually called for.
const SYNTHETIC_PARAGRAPH_BYTES: u64 = 400;

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
    // Anywhere in the argument list, like every other subcommand: an
    // operator halfway through composing flags asks for the manual
    // without first deleting what they typed.
    if args.iter().any(|a| a == "--help" || a == "-h") {
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
    if (context.label_count() as u64) < measured_labels {
        eprintln!(
            "taguru: estimate: warning: only {} of the requested {measured_labels} labels \
             appeared in the measured synthesis — the estimate below is a lower bound, not a \
             measurement of the requested shape",
            context.label_count()
        );
    }
    let factor = plan.associations as f64 / measured_associations as f64;
    let footprint = (context.footprint() as f64 * factor) as u64;
    let image = (context.to_bytes().len() as f64 * factor) as u64;

    // The vector sidecar is arithmetic, not synthesis: one f32 vector
    // and one gloss hash per canonical concept and label. Routed
    // through f64 like `footprint`/`image` above: --concepts,
    // --labels, and --embedding-dims are each unbounded from the
    // CLI, and u64 multiplication overflows (panicking in debug,
    // silently wrapping to a nonsense small number in release) well
    // before a realistic capacity-planning input does.
    let embedded_names = plan.concepts as f64 + plan.labels as f64;
    let vectors = if plan.embedding_dims > 0 {
        (embedded_names * (plan.embedding_dims as f64 * 4.0 + plan.name_bytes as f64 + 64.0)) as u64
    } else {
        0
    };

    let passages = if plan.passage_bytes > 0 {
        Some(estimate_passages(&plan))
    } else {
        None
    };
    let passage_resident = passages.as_ref().map_or(0, |p| {
        p.store_bytes
            .saturating_add(p.bm25_bytes)
            .saturating_add(p.vector_bytes)
    });

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
            embedded_names as u64
        );
    }
    if let Some(p) = &passages {
        println!(
            "  passage store      {:>12}   (original text + paragraph spans, resident with the context)",
            fmt_bytes(p.store_bytes)
        );
        println!(
            "  BM25 index         {:>12}   (lexical passage search)",
            fmt_bytes(p.bm25_bytes)
        );
        if p.vector_bytes > 0 {
            println!(
                "  passage vectors    {:>12}   ({} of {} paragraph(s) embedded, TAGURU_PASSAGE_VECTOR_LIMIT capped)",
                fmt_bytes(p.vector_bytes),
                p.vectorized_paragraphs,
                p.total_paragraphs
            );
        }
    }
    println!(
        "  TAGURU_CACHE_BYTES ≥ {} to keep this context hot; footprint is modeled",
        fmt_bytes(
            footprint
                .saturating_add(vectors)
                .saturating_add(passage_resident)
        )
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
    println!("maintenance window (per context):");
    println!(
        "  compaction peak    {:>12}   (transient: this context's footprint held twice — \
live + freshly rebuilt — until the old copy drops; `taguru compact` / \
POST /maintenance/compact)",
        fmt_bytes(footprint)
    );

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
        let label_index = (round + subject_index) % labels;
        let object_offset = 1 + (round / labels) % (concepts - 1);
        let object_index = (subject_index + object_offset) % concepts;
        let subject = synthetic_name('c', subject_index, name_bytes);
        let object = synthetic_name('c', object_index, name_bytes);
        let label = synthetic_name('l', label_index, name_bytes);
        let source = synthetic_name('s', i % sources, name_bytes);
        context
            .associate_from(&subject, &label, &object, 1.0, &source, None)
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

/// Passage-related resident bytes for the requested shape. The passage
/// store and the BM25 index are measured for real (`record_bytes`,
/// `Bm25Index::footprint`) on a bounded sample and extrapolated
/// linearly; the per-paragraph vector cost is arithmetic (matching how
/// `vectors` above is computed) but the PARAGRAPH COUNT it multiplies
/// is measured from that same sample, then capped at
/// `DEFAULT_PASSAGE_VECTOR_LIMIT` — the ceiling the server itself
/// enforces per context.
struct PassageEstimate {
    store_bytes: u64,
    bm25_bytes: u64,
    vector_bytes: u64,
    total_paragraphs: u64,
    vectorized_paragraphs: u64,
}

/// A synthetic passage body of exactly `bytes` bytes, `\n\n`-separated
/// into paragraphs so `paragraph::split` (and so BM25 and the vector
/// lane) sees the same shape a real document would rather than one
/// giant paragraph.
fn synthetic_passage_text(bytes: u64) -> String {
    let paragraph_bytes = SYNTHETIC_PARAGRAPH_BYTES as usize;
    let mut text = String::with_capacity(bytes as usize + paragraph_bytes);
    let mut word: u64 = 0;
    while (text.len() as u64) < bytes {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        let paragraph_start = text.len();
        while text.len() - paragraph_start < paragraph_bytes {
            text.push_str("word");
            text.push_str(&word.to_string());
            text.push(' ');
            word += 1;
        }
    }
    // The loop above always finishes whichever paragraph it's mid-way
    // through before re-checking `bytes`, so it can overshoot by up to
    // one SYNTHETIC_PARAGRAPH_BYTES — for a small `bytes`, or one that
    // lands just past a paragraph boundary (500 overshoots to 812), that's
    // not a rounding error, it's up to double. `estimate_passages` measures
    // this sample for real and scales it by `factor` to reach the final
    // estimate, so an oversized sample doesn't stay a sampling quirk — it
    // becomes a systematic overestimate of every passage number `estimate`
    // reports. Truncating back to the request here is what keeps the
    // sample honest.
    text.truncate(bytes as usize);
    text
}

fn estimate_passages(plan: &Plan) -> PassageEstimate {
    let sample_bytes = plan.passage_bytes.min(PASSAGE_SAMPLE_CAP);
    let factor = plan.passage_bytes as f64 / sample_bytes as f64;
    let requested_sample_sources = ((plan.sources as f64 / factor).round() as u64).max(1);
    // Every sampled source costs at least one SYNTHETIC_PARAGRAPH_BYTES
    // paragraph no matter how tiny its share of sample_bytes works out to,
    // so a --sources far larger than sample_bytes can support at that
    // floor (e.g. a huge --associations deriving a huge default --sources
    // alongside a small --passage-bytes) must not be sampled in full —
    // that's exactly the runaway PASSAGE_SAMPLE_CAP exists to prevent.
    let sample_sources =
        requested_sample_sources.min((sample_bytes / SYNTHETIC_PARAGRAPH_BYTES).max(1));
    let per_source_bytes = (sample_bytes / sample_sources).max(1);

    let mut records: Vec<(String, Arc<PassageRecord>)> =
        Vec::with_capacity(sample_sources as usize);
    let mut sample_paragraphs = 0u64;
    for i in 0..sample_sources {
        let text = synthetic_passage_text(per_source_bytes);
        let source = synthetic_name('s', i, plan.name_bytes);
        let (record, _, _) = PassageRecord::new(Arc::from(text.as_str()), Vec::new(), Vec::new());
        sample_paragraphs += record.paragraphs.len() as u64;
        records.push((source, record));
    }

    let store_bytes_sample: usize = records
        .iter()
        .map(|(source, record)| record_bytes(source, record))
        .sum();
    let bm25_bytes_sample = Bm25Index::build(&records).footprint();

    let total_paragraphs = (sample_paragraphs as f64 * factor).round() as u64;
    let vectorized_paragraphs = total_paragraphs.min(DEFAULT_PASSAGE_VECTOR_LIMIT as u64);
    let vector_bytes = if plan.embedding_dims > 0 && sample_paragraphs > 0 {
        let mut store = PassageVectorStore::new("estimate");
        for (source, record) in &records {
            for span in &record.paragraphs {
                store.push(
                    PassageKey {
                        source: source.clone(),
                        index: span.index,
                        hash: span.hash,
                        question_hash: None,
                    },
                    vec![0.0f32; plan.embedding_dims as usize],
                );
            }
        }
        let per_paragraph = store.footprint() as f64 / sample_paragraphs as f64;
        (per_paragraph * vectorized_paragraphs as f64) as u64
    } else {
        0
    };

    PassageEstimate {
        store_bytes: (store_bytes_sample as f64 * factor) as u64,
        bm25_bytes: (bm25_bytes_sample as f64 * factor) as u64,
        vector_bytes,
        total_paragraphs,
        vectorized_paragraphs,
    }
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

    /// Regression test: the paragraph loop always finishes the paragraph
    /// it's mid-way through before checking whether `bytes` is met, so
    /// without the final truncate a target of 500 landed a whole
    /// SYNTHETIC_PARAGRAPH_BYTES paragraph past "roughly 500" — 812 bytes,
    /// a 62% overshoot that `estimate_passages` would carry straight into
    /// its reported store/BM25 estimate via `factor`.
    #[test]
    fn synthetic_passage_text_lands_on_the_exact_requested_length() {
        for bytes in [0, 1, 10, 399, 400, 401, 500, 799, 800, 801, 4096] {
            let text = synthetic_passage_text(bytes);
            assert_eq!(
                text.len() as u64,
                bytes,
                "requested {bytes} bytes, got {}",
                text.len()
            );
        }
    }

    #[test]
    fn synthesis_delivers_the_requested_shape() {
        let context = synthesize(10_000, 500, 20, 100, 24);
        assert_eq!(context.association_count(), 10_000);
        assert_eq!(context.concept_count(), 500);
        assert_eq!(context.source_count(), 100);
        // Label coverage is bounded by min(labels, concepts), not by
        // the round count: the label index is offset by the subject
        // index, so with concepts >= labels every label is already
        // touched within round 0 alone.
        assert_eq!(context.label_count(), 20);
    }

    #[test]
    fn synthesis_covers_every_label_even_with_only_a_couple_of_rounds() {
        // The CLI default shape (concepts = associations / 2, labels =
        // 50) yields only 2 rounds per subject. Label coverage must
        // not depend on round count when concepts >= labels — this is
        // the regression test for issue #74, where the default shape
        // silently measured only 2 of the 50 planned labels.
        let context = synthesize(1_000, 500, 50, 100, 24);
        assert_eq!(context.label_count(), 50);
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

    fn passage_plan(passage_bytes: u64, sources: u64, embedding_dims: u64) -> Plan {
        Plan {
            associations: 100,
            concepts: 50,
            labels: 5,
            sources,
            name_bytes: 24,
            embedding_dims,
            passage_bytes,
        }
    }

    #[test]
    fn estimate_passages_measures_store_and_bm25_bytes_without_embedding_dims() {
        let plan = passage_plan(4096, 4, 0);
        let estimate = estimate_passages(&plan);
        assert!(estimate.store_bytes > 0);
        assert!(estimate.bm25_bytes > 0);
        assert_eq!(
            estimate.vector_bytes, 0,
            "embedding-dims=0 must cost nothing"
        );
        assert!(estimate.total_paragraphs > 0);
        assert_eq!(estimate.vectorized_paragraphs, estimate.total_paragraphs);
    }

    #[test]
    fn estimate_passages_sizes_vectors_when_embedding_dims_are_set() {
        let plan = passage_plan(4096, 4, 8);
        let estimate = estimate_passages(&plan);
        assert!(estimate.vector_bytes > 0);
        assert_eq!(estimate.vectorized_paragraphs, estimate.total_paragraphs);
    }

    /// Regression test for the server-side ceiling: however many
    /// paragraphs the requested shape would produce, only
    /// `DEFAULT_PASSAGE_VECTOR_LIMIT` of them are ever embedded, so the
    /// estimate must cap `vectorized_paragraphs` there too rather than
    /// pricing vectors for paragraphs the server would never embed.
    #[test]
    fn estimate_passages_caps_vectorized_paragraphs_at_the_server_limit() {
        let plan = passage_plan(50_000_000, 5_000, 1536);
        let estimate = estimate_passages(&plan);
        assert!(estimate.total_paragraphs > DEFAULT_PASSAGE_VECTOR_LIMIT as u64);
        assert_eq!(
            estimate.vectorized_paragraphs,
            DEFAULT_PASSAGE_VECTOR_LIMIT as u64
        );
        assert!(estimate.vector_bytes > 0);
    }

    #[test]
    fn estimate_passages_extrapolates_past_the_sample_cap_roughly_linearly() {
        let small = estimate_passages(&passage_plan(PASSAGE_SAMPLE_CAP, 200, 0));
        let large = estimate_passages(&passage_plan(PASSAGE_SAMPLE_CAP * 4, 800, 0));
        let store_ratio = large.store_bytes as f64 / small.store_bytes as f64;
        let bm25_ratio = large.bm25_bytes as f64 / small.bm25_bytes as f64;
        assert!(
            (3.2..=4.8).contains(&store_ratio),
            "store ratio {store_ratio}"
        );
        assert!((3.2..=4.8).contains(&bm25_ratio), "bm25 ratio {bm25_ratio}");
    }

    /// Regression test: a huge `--associations` derives a huge default
    /// `--sources` (associations / 20) even when `--passage-bytes` is
    /// tiny. `sample_sources` used to track that derived `sources` count
    /// uncapped, so every one of the tens of thousands of sampled sources
    /// synthesized its own ~400-byte paragraph regardless of
    /// `PASSAGE_SAMPLE_CAP` — tens of megabytes of synthetic text (and a
    /// matching BM25 build) to measure a 400-byte request.
    #[test]
    fn estimate_passages_caps_the_source_sample_when_sources_vastly_outnumber_bytes() {
        let plan = passage_plan(400, 1_000_000 / 20, 0);
        let estimate = estimate_passages(&plan);
        assert!(
            estimate.store_bytes < 1024 * 1024,
            "measuring 400 requested passage bytes across 50,000 sources must not \
             synthesize megabytes: store_bytes = {}",
            estimate.store_bytes
        );
        assert!(
            estimate.bm25_bytes < 1024 * 1024,
            "bm25_bytes = {}",
            estimate.bm25_bytes
        );
    }
}
