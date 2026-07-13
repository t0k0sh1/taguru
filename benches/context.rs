use std::hint::black_box;

use gungraun::prelude::*;
use gungraun::{Callgrind, EventKind};
use taguru::context::{Activation, Association, Context, CorruptImage, Recollection, Resolution};

// Same hand-rolled xorshift PRNG as examples/benchmark/main.rs:16-31,
// ported rather than shared — the library surface is context+deadline
// only, so benches/ cannot depend on the example binary.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// A tenth to a fiftieth of examples/benchmark's scale (20k/100k,
// 200k/1M): under Callgrind's native-vs-instrumented overhead, this
// keeps one CI run in the low minutes.
const CONCEPT_COUNT: u64 = 2_000;
const EDGE_COUNT: u64 = 10_000;
const LABEL_COUNT: u64 = 20;
const SOURCE_COUNT: u64 = 100;
// 30% of associations carry a source, mirroring examples/benchmark.
const SOURCED_NUMERATOR: u64 = 3;
const SOURCED_DENOMINATOR: u64 = 10;

// Known to exist in the fixture graph below: concept ids run
// 0..CONCEPT_COUNT and label ids 0..LABEL_COUNT.
const CUE_CONCEPT: &str = "概念1234";
const OTHER_CONCEPT: &str = "概念1";
const CUE_LABEL: &str = "関係7";
const OTHER_LABEL: &str = "関係3";

struct Names {
    concepts: Vec<String>,
    labels: Vec<String>,
    sources: Vec<String>,
}

fn prepare_names() -> Names {
    Names {
        concepts: (0..CONCEPT_COUNT).map(|i| format!("概念{i}")).collect(),
        labels: (0..LABEL_COUNT).map(|i| format!("関係{i}")).collect(),
        sources: (0..SOURCE_COUNT).map(|i| format!("文書{i}")).collect(),
    }
}

// The same mixed associate/associate_from ingest loop as
// examples/benchmark's setup phase — reused as-is for the read-only
// fixtures below, and benchmarked directly as bench_associate.
fn ingest(names: &Names) -> Context {
    let mut rng = Rng(0x1234_5678);
    let mut context = Context::default();
    for _ in 0..EDGE_COUNT {
        let subject = names.concepts[rng.below(CONCEPT_COUNT) as usize].as_str();
        let object = names.concepts[rng.below(CONCEPT_COUNT) as usize].as_str();
        let label = names.labels[rng.below(LABEL_COUNT) as usize].as_str();
        if rng.below(SOURCED_DENOMINATOR) < SOURCED_NUMERATOR {
            let source = names.sources[rng.below(SOURCE_COUNT) as usize].as_str();
            context
                .associate_from(subject, label, object, 1.0, source, None)
                .unwrap();
        } else {
            context.associate(subject, label, object, 1.0).unwrap();
        }
    }
    context
}

fn build_fixture() -> Context {
    ingest(&prepare_names())
}

fn build_image() -> Vec<u8> {
    build_fixture().to_bytes()
}

// ---------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------

#[library_benchmark]
#[bench::exact(setup = build_fixture)]
fn bench_resolve(context: Context) -> Vec<Resolution> {
    black_box(context.resolve(black_box(CUE_CONCEPT)))
}

#[library_benchmark]
#[bench::exact(setup = build_fixture)]
fn bench_resolve_label(context: Context) -> Vec<Resolution> {
    black_box(context.resolve_label(black_box(CUE_LABEL)))
}

#[library_benchmark]
#[bench::decay_05_limit_20(setup = build_fixture)]
fn bench_activate(context: Context) -> (usize, Vec<Activation>) {
    black_box(context.activate(black_box(&[CUE_CONCEPT]), black_box(0.5), black_box(20)))
}

#[library_benchmark]
#[bench::pinned(setup = build_fixture)]
fn bench_query_pinned(context: Context) -> Vec<Association> {
    black_box(context.query(
        black_box(Some(CUE_CONCEPT)),
        black_box(Some(CUE_LABEL)),
        black_box(None),
    ))
}

#[library_benchmark]
#[bench::unpinned(setup = build_fixture)]
fn bench_query_unpinned(context: Context) -> Vec<Association> {
    black_box(context.query(black_box(None), black_box(None), black_box(None)))
}

#[library_benchmark]
#[bench::exact(setup = build_fixture)]
fn bench_query_any(context: Context) -> Vec<Association> {
    black_box(context.query_any(
        black_box(&[CUE_CONCEPT, OTHER_CONCEPT]),
        black_box(&[CUE_LABEL, OTHER_LABEL]),
        black_box(&[]),
    ))
}

#[library_benchmark]
#[bench::depth_2(setup = build_fixture)]
fn bench_explore_depth_2(context: Context) -> Vec<Recollection> {
    black_box(context.explore(black_box(&[CUE_CONCEPT]), black_box(2)))
}

#[library_benchmark]
#[bench::unbounded(setup = build_fixture)]
fn bench_explore_unbounded(context: Context) -> Vec<Recollection> {
    black_box(context.explore(black_box(&[CUE_CONCEPT]), black_box(Context::UNBOUNDED)))
}

library_benchmark_group!(
    name = context_reads,
    benchmarks = [
        bench_resolve,
        bench_resolve_label,
        bench_activate,
        bench_query_pinned,
        bench_query_unpinned,
        bench_query_any,
        bench_explore_depth_2,
        bench_explore_unbounded,
    ]
);

// ---------------------------------------------------------------------
// Writes — the ingest loop itself is the measured cost.
// ---------------------------------------------------------------------

#[library_benchmark]
#[bench::exact(setup = prepare_names)]
fn bench_associate(names: Names) -> Context {
    black_box(ingest(black_box(&names)))
}

library_benchmark_group!(name = context_writes, benchmarks = [bench_associate]);

// ---------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------

#[library_benchmark]
#[bench::exact(setup = build_fixture)]
fn bench_to_bytes(context: Context) -> Vec<u8> {
    black_box(context.to_bytes())
}

#[library_benchmark]
#[bench::exact(setup = build_image)]
fn bench_from_bytes(image: Vec<u8>) -> Result<Context, CorruptImage> {
    black_box(Context::from_bytes(black_box(&image)))
}

library_benchmark_group!(
    name = context_persistence,
    benchmarks = [bench_to_bytes, bench_from_bytes]
);

// Ir (total instructions retired) is cache-simulation-independent and
// therefore the most portable signal across CI runners; flagged as the
// "if in doubt" default by gungraun's own docs. The 5% margin is the
// documented default starting point — revisit once CI has measured the
// same-build noise floor from HashMap's per-process random seed.
main!(
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::default().soft_limits([(EventKind::Ir, 5.0)])),
    library_benchmark_groups = [context_reads, context_writes, context_persistence]
);
