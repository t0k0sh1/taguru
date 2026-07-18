use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

/// Runs `f` over `items` on up to `workers` threads pulling from one
/// shared queue — the same divide-the-queue-not-the-slice shape
/// `preload_pinned` uses, generalized so a caller only supplies the
/// per-item work. Each worker collects into a local `Vec` and merges
/// into the shared result once at the end, so contention is limited to
/// the queue itself; results come back in arrival order, not input
/// order — callers that need input order carry an index through `T`/`R`
/// and sort afterward.
pub(crate) fn parallel_map<T, R>(items: Vec<T>, workers: usize, f: impl Fn(T) -> R + Sync) -> Vec<R>
where
    T: Send,
    R: Send,
{
    if items.is_empty() {
        return Vec::new();
    }
    let workers = workers.min(items.len()).max(1);
    let queue = Mutex::new(items.into_iter());
    let results: Mutex<Vec<R>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let Some(item) = queue.lock().next() else {
                        break;
                    };
                    local.push(f(item));
                }
                results.lock().extend(local);
            });
        }
    });
    results.into_inner()
}

/// Runs `f` over each of `chunks` on up to `workers` threads, claiming
/// indices in order. Unlike `parallel_map` above — arrival-order
/// results, no notion of failure — this preserves input order and
/// stops claiming new work once a chunk's failure has been recorded.
/// Every caller (`extract_chunks_concurrently` in src/extract.rs, and
/// `embed_stale` / `refresh_passage_embeddings` below) needs both: an
/// input-order-preserving result to fold correctly, and best-effort
/// early termination once a failure surfaces, so a batch that is going
/// to fail stops enlisting new work. Fold-on-failure semantics differ
/// per caller (fail the whole batch vs. keep whatever succeeded), so
/// the fold itself is left to them — this returns the raw, unfolded
/// per-index outcome.
///
/// `next` and `first_failure` are independent atomics; SeqCst on both
/// is required so a worker claiming an index past a just-recorded
/// failure actually observes it (Relaxed would silently reintroduce
/// unbounded over-dispatch past a failure). Every index at or below the
/// true minimum failing index is guaranteed a `Some` slot — a foldable
/// prefix callers can trust. Slots past it are best-effort: `None` if
/// never claimed, `Some` if a worker finished before the failure was
/// recorded. Their count is NOT bounded by `workers` — a failure slow
/// to surface lets the other workers complete arbitrarily many later
/// indices first — so callers fold on the prefix, never on a count of
/// what landed past the failure.
pub(crate) fn dispatch_chunks_concurrently<C: Sync, R: Send + Sync>(
    chunks: &[C],
    workers: usize,
    f: impl Fn(&C) -> Result<R, String> + Sync,
) -> Vec<Option<Result<R, String>>> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let workers = workers.min(chunks.len()).max(1);
    let next = AtomicUsize::new(0);
    let first_failure = AtomicUsize::new(usize::MAX);
    let results: Vec<OnceLock<Result<R, String>>> =
        (0..chunks.len()).map(|_| OnceLock::new()).collect();

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::SeqCst);
                    if index >= chunks.len() || index > first_failure.load(Ordering::SeqCst) {
                        break;
                    }
                    let outcome = f(&chunks[index]);
                    if outcome.is_err() {
                        first_failure.fetch_min(index, Ordering::SeqCst);
                    }
                    let _ = results[index].set(outcome);
                }
            });
        }
    });
    results.into_iter().map(OnceLock::into_inner).collect()
}
