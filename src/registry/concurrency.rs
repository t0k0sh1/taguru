use std::collections::VecDeque;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use taguru::deadline::Deadline;

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

/// How long one wait leg blocks before re-checking the deadline — the
/// ceiling on how stale `acquire_until`'s deadline check can get, and
/// (since [`Semaphore::release`] wakes every waiter, never just one)
/// the self-healing floor under a missed wakeup: a notification lost
/// to a race is recovered within one poll, not forever.
const SLOT_POLL: Duration = Duration::from_millis(50);

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
///
/// Two properties `Mutex<usize>` + `Condvar::notify_one` (the
/// original shape) did not have, per issue #563 item 4: a bound on
/// how long a caller waits, and fairness. Both come from the same
/// ticket queue — a waiter blocks only while it holds the earliest
/// outstanding ticket, so `parking_lot::Condvar`'s non-FIFO wakeup
/// order can no longer let a late arrival repeatedly cut ahead of one
/// that has been waiting since before it existed.
pub(crate) struct Semaphore {
    inner: Mutex<SemaphoreInner>,
    available: Condvar,
    /// Mirrors `inner.queue.len()` outside the lock — the scrape-time
    /// gauge (`AppState::embed_slot_waiters`) reads this instead of
    /// taking `inner`, so a slow metrics scrape can never itself
    /// become a reason a refresh thread waits longer.
    waiting: AtomicUsize,
}

struct SemaphoreInner {
    permits: usize,
    /// FIFO order of tickets not yet granted a permit. A waiter is
    /// eligible to take a free permit only once its ticket reaches the
    /// front — see `acquire_until`.
    queue: VecDeque<u64>,
    next_ticket: u64,
}

impl Semaphore {
    /// `permits` is NOT floored here — issue #563 item 5 moved that
    /// floor to the env boundary (`resolve_embed_parallel`) and to the
    /// one call site that turns a raw `BootOptions` into both this and
    /// the `embed_parallel` field (`boot_with`), so the two ceilings
    /// can never read different numbers. A `Semaphore::new(0)` reached
    /// any other way is a caller bug, not a runtime input to guard
    /// against a second time — `acquire_until` on a zero-permit
    /// semaphore just always times out, no hang.
    pub(crate) fn new(permits: usize) -> Self {
        Self {
            inner: Mutex::new(SemaphoreInner {
                permits,
                queue: VecDeque::new(),
                next_ticket: 0,
            }),
            available: Condvar::new(),
            waiting: AtomicUsize::new(0),
        }
    }

    /// Threads currently queued for a permit — read lock-free for the
    /// scrape-time gauge.
    pub(crate) fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }

    /// Blocks for a permit until one is granted or `deadline` passes.
    /// `permit` is `None` on timeout — the caller's provider round
    /// trip never happened, so it should fail the same way a
    /// `Deadline` expiring anywhere else in a refresh chunk does
    /// (`DeadlineExceeded`). `queued` is true whenever the fast
    /// (uncontended) path was missed, whether or not a permit was
    /// eventually granted — the caller's `taguru_embed_slot_waits_total`
    /// signal.
    pub(crate) fn acquire_until(&self, deadline: Deadline) -> Acquisition<'_> {
        let mut inner = self.inner.lock();
        // Fast path: nobody ahead of us and a permit is free right
        // now — skip the ticket queue entirely so the common
        // (uncontended) case never pays for fairness bookkeeping.
        if inner.queue.is_empty() && inner.permits > 0 {
            inner.permits -= 1;
            return Acquisition {
                permit: Some(SemaphorePermit { semaphore: self }),
                queued: false,
            };
        }
        let ticket = inner.next_ticket;
        inner.next_ticket += 1;
        inner.queue.push_back(ticket);
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let granted = loop {
            if inner.permits > 0 && inner.queue.front() == Some(&ticket) {
                inner.permits -= 1;
                inner.queue.pop_front();
                break true;
            }
            if deadline.expired() {
                // Not necessarily still at the front — a slow-to-wake
                // waiter behind us in the queue may have raced this
                // check — so remove by value, not `pop_front`.
                inner.queue.retain(|&queued| queued != ticket);
                break false;
            }
            // Bounded by SLOT_POLL regardless of how far off `deadline`
            // is — an unbounded deadline (Deadline::unbounded, used by
            // the flush ticker's own refresh calls) must still re-check
            // `expired()` on a heartbeat rather than blocking forever
            // on one `wait_for`.
            let slice = deadline.remaining().min(SLOT_POLL);
            self.available.wait_for(&mut inner, slice);
        };
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        let permit = if granted {
            Some(SemaphorePermit { semaphore: self })
        } else {
            // A timed-out waiter leaving the queue can move a
            // different ticket to the front; wake everyone so the new
            // front-of-line notices without waiting out its own poll.
            self.available.notify_all();
            None
        };
        Acquisition {
            permit,
            queued: true,
        }
    }

    fn release(&self) {
        self.inner.lock().permits += 1;
        // notify_all, not notify_one: eligibility is "permits > 0 AND
        // at the front of the queue", which only the front-of-line
        // waiter can act on — but `available` is shared with the
        // timeout path above, which also needs every waiter to
        // re-check after a queue removal. One condvar, so every
        // release wakes every waiter to re-test its own condition.
        self.available.notify_all();
    }
}

/// [`Semaphore::acquire_until`]'s result, split so the caller can
/// record its two independent metrics signals (queued at all vs. gave
/// up) without the semaphore itself depending on [`crate::metrics::Metrics`].
pub(crate) struct Acquisition<'a> {
    pub(crate) permit: Option<SemaphorePermit<'a>>,
    pub(crate) queued: bool,
}

/// Returns its permit to [`Semaphore`] on drop — held across exactly
/// the provider call, never longer, so a panic mid-call still frees it.
pub(crate) struct SemaphorePermit<'a> {
    semaphore: &'a Semaphore,
}

impl Drop for SemaphorePermit<'_> {
    fn drop(&mut self) {
        self.semaphore.release();
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

    #[test]
    fn parallel_map_on_empty_input_returns_empty_without_spawning() {
        let out: Vec<i32> = parallel_map(Vec::<i32>::new(), 4, |item| item * 2);
        assert!(out.is_empty());
    }

    /// `workers = 0` must not mean "no work happens" — the `.max(1)`
    /// clamp (line ~26) guarantees at least one worker runs the queue
    /// to completion.
    #[test]
    fn parallel_map_with_zero_workers_still_processes_everything() {
        let items: Vec<i32> = (0..20).collect();
        let mut out = parallel_map(items, 0, |item| item * 2);
        out.sort_unstable();
        let expected: Vec<i32> = (0..20).map(|item| item * 2).collect();
        assert_eq!(out, expected);
    }

    /// `workers` exceeding the item count must clamp to the item count
    /// (line ~26) rather than spawning idle threads that never claim
    /// anything — a correctness-neutral case in principle, but worth
    /// pinning since nothing else in the repo drives this ratio.
    #[test]
    fn parallel_map_with_more_workers_than_items_processes_everything_once() {
        let items: Vec<i32> = (0..3).collect();
        let mut out = parallel_map(items, 50, |item| item + 1);
        out.sort_unstable();
        assert_eq!(out, vec![1, 2, 3]);
    }

    /// `parallel_map` explicitly does not preserve input order (doc
    /// `~15-17`) — callers that need it sort afterward. What it must
    /// preserve is the multiset: every item processed exactly once, no
    /// duplicates or drops from a worker's local `Vec` failing to merge.
    #[test]
    fn parallel_map_merges_every_worker_with_no_loss_or_duplication() {
        let items: Vec<i32> = (0..200).collect();
        let mut out = parallel_map(items, 8, |item| item);
        out.sort_unstable();
        let expected: Vec<i32> = (0..200).collect();
        assert_eq!(
            out, expected,
            "the merged result must be exactly the input multiset"
        );
    }

    /// Pins down the early-stop half of `dispatch_chunks_concurrently`'s
    /// contract: no matter how many chunks past the failure land (that
    /// count is explicitly NOT bounded by `workers`, per the doc on
    /// `dispatch_chunks_concurrently` above), every index at or below
    /// the true minimum failing index is guaranteed a `Some` slot, and
    /// the gate stops dispatch short of the full chunk count.
    ///
    /// This does not assert an upper bound on spillover past the
    /// failure — that count depends on how promptly the failing
    /// worker's `fetch_min` lands relative to the other workers'
    /// claims, which is a scheduler timing race, not a contract this
    /// function makes (see the doc above, `~66-69`). An earlier version
    /// of this test asserted `past_failure <= WORKERS`, which is only
    /// true when the failure is recorded before any of the `workers`
    /// in-flight chunks finish; nothing prevents a preemption between
    /// `calls.lock().push(index)` and the `first_failure.fetch_min`
    /// call from letting the other workers claim and finish arbitrarily
    /// many more indices first, so that assertion could flake under
    /// load. See `dispatch_chunks_concurrently_workers_1_stops_at_the_failure`
    /// below for a deterministic (non-flaky) bound on spillover.
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
            std::thread::sleep(Duration::from_millis(20));
            Ok(index)
        });

        let called = calls.lock().clone();
        assert!(
            called.len() < chunks.len(),
            "the gate must stop dispatch well short of all {} chunks; saw {called:?}",
            chunks.len()
        );
        for (index, outcome) in outcomes.iter().enumerate().take(FAILING_INDEX) {
            assert!(
                matches!(outcome, Some(Ok(value)) if *value == index),
                "every index below the true minimum failing index must succeed"
            );
        }
        assert!(matches!(&outcomes[FAILING_INDEX], Some(Err(message)) if message == "boom"));
    }

    /// Deterministic companion to the test above: with exactly one
    /// worker, the single thread that claims the failing index is the
    /// same thread that will next check `first_failure` before claiming
    /// another — so the gate at `next.fetch_add`/`first_failure.load`
    /// (lines ~88-90) is guaranteed to see its own just-recorded
    /// failure with no scheduler race involved. This pins the exact
    /// claimed set with no timing dependency at all.
    #[test]
    fn dispatch_chunks_concurrently_workers_1_stops_at_the_failure() {
        const FAILING_INDEX: usize = 5;
        let chunks: Vec<usize> = (0..10).collect();

        let outcomes = dispatch_chunks_concurrently(&chunks, 1, |&index| {
            if index == FAILING_INDEX {
                return Err("boom".to_string());
            }
            Ok(index)
        });

        for (index, outcome) in outcomes.iter().enumerate().take(FAILING_INDEX) {
            assert!(
                matches!(outcome, Some(Ok(value)) if *value == index),
                "index {index} below the failure must succeed"
            );
        }
        assert!(matches!(&outcomes[FAILING_INDEX], Some(Err(message)) if message == "boom"));
        for outcome in &outcomes[FAILING_INDEX + 1..] {
            assert!(
                outcome.is_none(),
                "a single worker must never claim past its own recorded failure: {outcomes:?}"
            );
        }
    }

    /// Issue #563 item 4: the original `Semaphore` had no deadline at
    /// all — a hung provider call could block a waiter (and everyone
    /// behind it) forever, invisibly. `acquire_until` must actually
    /// give up once the deadline passes, report it via `Acquisition`
    /// rather than a bare bool, and leave no trace in `waiting()`
    /// afterward — a lingering ticket would wedge the queue for every
    /// waiter still behind it.
    #[test]
    fn acquire_until_times_out_when_no_permit_frees_up() {
        use std::thread;
        use std::time::Instant;

        let sem = Arc::new(Semaphore::new(1));
        let held = sem
            .acquire_until(Deadline::unbounded())
            .permit
            .expect("the only permit is free at the start");

        // Run the blocked acquire on its own thread so this thread can
        // observe `waiting()` mid-block — sampling it only after the
        // call returns would pass even if `waiting()` were hardcoded
        // to 0.
        let waiter_sem = Arc::clone(&sem);
        let started = Instant::now();
        let waiter = thread::spawn(move || {
            // Extracted to plain values, not the `Acquisition` itself
            // — its `SemaphorePermit` borrows `&waiter_sem`, which
            // does not outlive this closure.
            let acquisition = waiter_sem.acquire_until(Deadline::after(Duration::from_millis(150)));
            (acquisition.permit.is_none(), acquisition.queued)
        });

        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            sem.waiting(),
            1,
            "the blocked thread must show up as queued while it is still waiting"
        );

        let (timed_out, queued) = waiter.join().unwrap();
        assert!(timed_out, "no permit ever freed up, so this must time out");
        assert!(
            queued,
            "it missed the fast path and had to enter the wait queue"
        );
        assert!(
            started.elapsed() >= Duration::from_millis(145),
            "must actually wait out close to the full deadline, not return early"
        );
        assert_eq!(
            sem.waiting(),
            0,
            "a timed-out waiter must remove its own ticket, not linger in the queue"
        );

        drop(held);
    }

    /// The original `Mutex<usize>` + `Condvar::notify_one` shape had no
    /// fairness guarantee — `parking_lot::Condvar` does not wake
    /// waiters FIFO, so a late arrival could repeatedly cut ahead of
    /// one waiting since before it existed (starvation). The ticket
    /// queue fixes this: with exactly one permit held by the test and
    /// three waiters queued strictly in order, permits must be granted
    /// in that same order every time, not just on average.
    #[test]
    fn acquire_until_grants_permits_in_arrival_order() {
        use std::thread;
        use std::time::Instant;

        let sem = Arc::new(Semaphore::new(1));
        let held = sem
            .acquire_until(Deadline::unbounded())
            .permit
            .expect("the only permit is free at the start");

        let order: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for id in 0..3u32 {
            let sem = Arc::clone(&sem);
            let order = Arc::clone(&order);
            handles.push(thread::spawn(move || {
                let acquisition = sem.acquire_until(Deadline::unbounded());
                let permit = acquisition
                    .permit
                    .expect("unbounded deadline never times out");
                order.lock().push(id);
                // Held long enough that a waiter behind this one in
                // the queue, if it woke at all, would still find the
                // permit taken — proof the grant order is the queue
                // order, not a race resolved by whoever wakes first.
                thread::sleep(Duration::from_millis(20));
                drop(permit);
            }));
            // Space out spawns so each thread's ticket lands in `id`
            // order — the arrival order FIFO is required to honor.
            thread::sleep(Duration::from_millis(20));
        }
        // Let all three settle into the wait queue before releasing.
        thread::sleep(Duration::from_millis(20));
        let released_at = Instant::now();
        drop(held);
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(
            *order.lock(),
            vec![0, 1, 2],
            "permits must be granted in the order threads queued, not wakeup order"
        );
        assert!(
            released_at.elapsed() >= Duration::from_millis(60),
            "three permits held 20ms each in sequence must take at least that long \
             serialized — a shorter elapsed time would mean more than one was ever \
             granted at once"
        );
    }

    /// `Semaphore::new(0)`'s doc (`~149-156`) claims it "just always
    /// times out, no hang" rather than blocking forever. Pin that down
    /// directly: a bounded deadline against a zero-permit semaphore
    /// must return within the deadline, report no permit, and leave no
    /// ticket behind.
    #[test]
    fn acquire_until_on_a_zero_permit_semaphore_always_times_out() {
        use std::time::Instant;

        let sem = Semaphore::new(0);
        let started = Instant::now();
        let acquisition = sem.acquire_until(Deadline::after(Duration::from_millis(100)));

        assert!(
            acquisition.permit.is_none(),
            "a zero-permit semaphore must never grant one"
        );
        assert!(acquisition.queued, "it must have entered the wait queue");
        assert!(
            started.elapsed() >= Duration::from_millis(95),
            "must wait out close to the deadline, not return immediately"
        );
        assert_eq!(
            sem.waiting(),
            0,
            "the timed-out waiter must not linger in the queue"
        );
    }

    /// The timeout path's queue removal (`retain`, not `pop_front`,
    /// `~209`) exists for exactly this shape: a waiter behind the one
    /// that times out. `acquire_until_times_out_when_no_permit_frees_up`
    /// above only ever has a single waiter, so it never proves `retain`
    /// finds a non-front ticket — nor that `notify_all` (`~227`) wakes
    /// the waiter still behind it so it does not sit out its own poll
    /// interval unnecessarily. Here two waiters queue behind one held
    /// permit; the first (front) waiter times out early, and the
    /// second (still queued behind it) must still be granted once the
    /// permit is released.
    #[test]
    fn acquire_until_timeout_of_a_non_front_waiter_does_not_wedge_the_queue() {
        use std::thread;
        use std::time::Instant;

        let sem = Arc::new(Semaphore::new(1));
        let held = sem
            .acquire_until(Deadline::unbounded())
            .permit
            .expect("the only permit is free at the start");

        // Front waiter: short deadline, must time out while the permit
        // stays held.
        let front_sem = Arc::clone(&sem);
        let front = thread::spawn(move || {
            front_sem
                .acquire_until(Deadline::after(Duration::from_millis(100)))
                .permit
                .is_none()
        });
        // Give the front waiter time to enter the queue before the
        // second one joins behind it.
        thread::sleep(Duration::from_millis(20));

        // Second waiter: unbounded deadline, queued behind the front
        // one, must still be granted after the front waiter times out
        // and the held permit is released.
        let back_sem = Arc::clone(&sem);
        let back = thread::spawn(move || {
            back_sem
                .acquire_until(Deadline::unbounded())
                .permit
                .is_some()
        });
        thread::sleep(Duration::from_millis(20));
        assert_eq!(
            sem.waiting(),
            2,
            "both waiters must be queued before the front one times out"
        );

        let front_timed_out = front.join().unwrap();
        assert!(front_timed_out, "the front waiter must time out first");

        let started = Instant::now();
        drop(held);
        let back_granted = back.join().unwrap();
        assert!(
            back_granted,
            "the waiter behind the timed-out one must still be granted \
             the permit, not left wedged in the queue"
        );
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "the back waiter must be woken promptly by notify_all, not \
             left waiting out its own poll interval"
        );
    }

    #[test]
    fn concurrent_reads_of_one_hot_context_do_not_serialize() {
        use std::sync::atomic::AtomicUsize;
        use std::thread;
        use std::time::{Duration, Instant};

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
                        // Wait for the other reader to show up (or give
                        // up after a generous deadline) instead of a
                        // fixed sleep — a fixed sleep races the second
                        // thread's spawn under load: if it hasn't
                        // entered `read_context` yet when the first
                        // exits, `peak` stays at 1 even though nothing
                        // is actually serializing the two reads.
                        let deadline = Instant::now() + Duration::from_secs(5);
                        while in_read.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
                            thread::sleep(Duration::from_millis(5));
                        }
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
