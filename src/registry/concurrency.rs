use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::{Condvar, Mutex};

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

/// A counting semaphore bounding actual concurrent work below however
/// many independent dispatch layers each think they alone own the
/// ceiling. `embed_parallel` sizes both the outer per-context
/// `parallel_map` in the flush tick AND the inner
/// `dispatch_chunks_concurrently` fan-out inside one context's own
/// refresh — nested, those two ceilings would multiply into P × P
/// concurrent provider calls. Every refresh chunk instead acquires a
/// permit here around its provider call, so no matter how many
/// threads across how many contexts attempt one at once, at most
/// `embed_parallel` are ever in flight process-wide.
pub(crate) struct Semaphore {
    permits: Mutex<usize>,
    available: Condvar,
}

impl Semaphore {
    pub(crate) fn new(permits: usize) -> Self {
        Self {
            permits: Mutex::new(permits.max(1)),
            available: Condvar::new(),
        }
    }

    pub(crate) fn acquire(&self) -> SemaphorePermit<'_> {
        let mut permits = self.permits.lock();
        while *permits == 0 {
            self.available.wait(&mut permits);
        }
        *permits -= 1;
        SemaphorePermit { semaphore: self }
    }
}

/// Returns its permit to [`Semaphore`] on drop — held across exactly
/// the provider call, never longer, so a panic mid-call still frees it.
pub(crate) struct SemaphorePermit<'a> {
    semaphore: &'a Semaphore,
}

impl Drop for SemaphorePermit<'_> {
    fn drop(&mut self) {
        *self.semaphore.permits.lock() += 1;
        self.semaphore.available.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use std::fs;

    use crate::registry::test_support::{assoc_op, scratch_dir};
    use crate::registry::{AppState, ContextMeta};
    use taguru::deadline::Deadline;

    /// Pins down the early-stop half of `dispatch_chunks_concurrently`'s
    /// contract on the schedule where it bites: when the failure is
    /// recorded PROMPTLY (here the failing chunk returns instantly while
    /// every success sleeps), no worker claims a new index past it once
    /// the record lands, so only the `workers` chunks already in flight
    /// at that moment can spill past the failure. A failure slow to
    /// surface would let the other workers run far ahead first — which is
    /// why callers fold on the guaranteed prefix (asserted below), never
    /// on a count of what landed past the failure.
    #[test]
    fn dispatch_chunks_concurrently_bounds_spillover_past_a_promptly_recorded_failure() {
        use std::time::Duration;

        const FAILING_INDEX: usize = 20;
        const WORKERS: usize = 4;
        let chunks: Vec<usize> = (0..50).collect();
        let calls: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));

        let outcomes = dispatch_chunks_concurrently(&chunks, WORKERS, |&index| {
            calls.lock().push(index);
            if index == FAILING_INDEX {
                return Err("boom".to_string());
            }
            // Slow enough that a chunk claimed after the failure lands
            // would have ample time to observe it before finishing and
            // going to claim another — if the gate were broken, this
            // sleep is what would let the assertions below catch it.
            std::thread::sleep(Duration::from_millis(20));
            Ok(index)
        });

        let called = calls.lock().clone();
        assert!(
            called.len() < chunks.len(),
            "the gate must stop dispatch well short of all {} chunks; saw {called:?}",
            chunks.len()
        );
        let past_failure = called
            .iter()
            .filter(|&&index| index > FAILING_INDEX)
            .count();
        assert!(
            past_failure <= WORKERS,
            "at most `workers` chunks can already be in flight when the failure \
             lands; saw {past_failure} claimed past index {FAILING_INDEX}: {called:?}"
        );
        for (index, outcome) in outcomes.iter().enumerate().take(FAILING_INDEX) {
            assert!(
                matches!(outcome, Some(Ok(value)) if *value == index),
                "every index below the true minimum failing index must succeed"
            );
        }
        assert!(matches!(&outcomes[FAILING_INDEX], Some(Err(message)) if message == "boom"));
    }

    #[test]
    fn concurrent_reads_of_one_hot_context_do_not_serialize() {
        use std::sync::atomic::AtomicUsize;
        use std::thread;
        use std::time::Duration;

        let dir = scratch_dir("read-parallel");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        let in_read = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut readers = Vec::new();
        for _ in 0..2 {
            let state = state.clone();
            let in_read = Arc::clone(&in_read);
            let peak = Arc::clone(&peak);
            readers.push(thread::spawn(move || {
                state
                    .read_context("sake", |context| {
                        let now = in_read.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        // Long enough that the two readers MUST overlap
                        // unless one lock is excluding the other.
                        thread::sleep(Duration::from_millis(150));
                        in_read.fetch_sub(1, Ordering::SeqCst);
                        context.association_count()
                    })
                    .map_err(|_| "read")
                    .unwrap();
            }));
        }
        for reader in readers {
            reader.join().unwrap();
        }
        assert_eq!(
            peak.load(Ordering::SeqCst),
            2,
            "two readers must be inside one hot context at the same time"
        );

        let _ = fs::remove_dir_all(dir);
    }
}
