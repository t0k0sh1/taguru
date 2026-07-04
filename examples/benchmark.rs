use std::hint::black_box;
use std::time::Instant;

use associative_rag::context::Context;

// Latency probe over every public operation at realistic scale, on a
// uniform random graph (average degree ~10, one giant component — the
// worst case for neighborhood walks). Run in release mode or the numbers
// mean nothing:
//
//   cargo run --release --example benchmark
//
// Timings are wall-clock on whatever machine runs this; treat them as
// relative between operations and between commits, not as absolutes.

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

fn time<T>(label: &str, iters: u32, mut f: impl FnMut() -> T) {
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    let per = start.elapsed().as_secs_f64() / f64::from(iters);
    if per >= 1e-3 {
        println!("  {label}: {:.2} ms/call ({iters} calls)", per * 1e3);
    } else {
        println!("  {label}: {:.2} µs/call ({iters} calls)", per * 1e6);
    }
}

fn main() {
    for &(concept_count, edge_count) in &[(20_000u64, 100_000u64), (200_000, 1_000_000)] {
        println!(
            "=== {concept_count} 概念 / {edge_count} 連想 (ラベル200種, 出典1000種, 30%出典付き) ==="
        );
        let concept_names: Vec<String> = (0..concept_count).map(|i| format!("概念{i}")).collect();
        let labels: Vec<String> = (0..200).map(|i| format!("関係{i}")).collect();
        let sources: Vec<String> = (0..1000).map(|i| format!("文書{i}")).collect();

        let mut rng = Rng(0x1234_5678);
        let mut context = Context::default();
        let start = Instant::now();
        for _ in 0..edge_count {
            let subject = concept_names[rng.below(concept_count) as usize].as_str();
            let object = concept_names[rng.below(concept_count) as usize].as_str();
            let label = labels[rng.below(200) as usize].as_str();
            if rng.below(10) < 3 {
                let source = sources[rng.below(1000) as usize].as_str();
                context
                    .associate_from(subject, label, object, 1.0, source)
                    .unwrap();
            } else {
                context.associate(subject, label, object, 1.0).unwrap();
            }
        }
        let ingest = start.elapsed().as_secs_f64();
        println!(
            "  ingest: {:.2} s / {:.0} 件毎秒 / {:.2} µs毎件",
            ingest,
            edge_count as f64 / ingest,
            ingest / edge_count as f64 * 1e6
        );

        let mut rng = Rng(0xDEAD_BEEF);
        let cue = |rng: &mut Rng| concept_names[rng.below(concept_count) as usize].as_str();

        {
            let mut r = Rng(1);
            time("recall(概念)", 1000, || context.recall(cue(&mut r)));
        }
        {
            let mut r = Rng(2);
            time("query(主語+ラベル固定)", 1000, || {
                context.query(Some(cue(&mut r)), Some("関係7"), None)
            });
        }
        {
            let mut r = Rng(3);
            time("explore(深さ2)", 100, || {
                context.explore(&[cue(&mut r)], 2)
            });
        }
        {
            let mut r = Rng(4);
            time("explore(深さ3)", 20, || {
                context.explore(&[cue(&mut r)], 3)
            });
        }
        {
            let mut r = Rng(5);
            time("activate(decay0.5, limit20)", 100, || {
                context.activate(&[cue(&mut r)], 0.5, 20)
            });
        }
        time("resolve(\"概念1234\")", 100, || {
            context.resolve("概念1234")
        });

        let start = Instant::now();
        let everything = context.query(None, None, None);
        println!(
            "  query(全件) 1回: {:.1} ms ({} 件を materialize)",
            start.elapsed().as_secs_f64() * 1e3,
            everything.len()
        );
        drop(everything);

        let start = Instant::now();
        let component = context.explore(&[cue(&mut rng)], Context::UNBOUNDED);
        println!(
            "  explore(UNBOUNDED) 1回: {:.1} ms ({} 件)",
            start.elapsed().as_secs_f64() * 1e3,
            component.len()
        );
        drop(component);

        let start = Instant::now();
        let image = context.to_bytes();
        println!(
            "  to_bytes: {:.1} ms ({:.1} MB)",
            start.elapsed().as_secs_f64() * 1e3,
            image.len() as f64 / 1e6
        );
        let start = Instant::now();
        let restored = Context::from_bytes(&image).unwrap();
        println!(
            "  from_bytes(検証+インデックス再構築込み): {:.1} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
        black_box(&restored);
        println!();
    }
}
