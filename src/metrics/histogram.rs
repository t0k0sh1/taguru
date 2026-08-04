//! Latency histogram primitives: the fixed bucket boundaries, the
//! mutex-guarded observe/snapshot state, and the per-route stat
//! wrapper the HTTP middleware and the Prometheus renderer both
//! share.

use super::*;

/// (upper bound in ms, its Prometheus `le` label). Fixed strings keep
/// the rendered form stable — `le="1"`, never a float-formatting
/// surprise like `le="1.0"`.
pub(super) const LATENCY_BUCKETS: [(u64, &str); 8] = [
    (1, "0.001"),
    (5, "0.005"),
    (10, "0.01"),
    (50, "0.05"),
    (100, "0.1"),
    (500, "0.5"),
    (1000, "1"),
    (5000, "5"),
];

/// One latency distribution. `counts[i]` is the exclusive bin ending
/// at `LATENCY_BUCKETS[i]` — NOT cumulative; the `_bucket{le=…}`
/// prefix sums are computed at render time. `count` doubles as the
/// `+Inf` bucket and the `_count` line (the exposition format defines
/// them to be equal), so it also counts observations past the largest
/// finite bound.
///
/// Guarded by one mutex rather than three independent atomics: three
/// separate atomic loads (buckets, then sum, then count) can each land
/// at a different instant, and a render that catches an in-flight
/// `observe()` between its bucket increment and its count increment
/// sees a finite bucket that already includes the new observation but
/// a `+Inf`/`_count` that does not yet — an invalid histogram, since
/// `+Inf` must never be less than a finite bucket. Locking the whole
/// read (and the whole write) makes every render see one consistent
/// instant instead.
#[derive(Default)]
pub(super) struct Histogram {
    state: Mutex<HistogramState>,
}

#[derive(Default, Clone, Copy)]
struct HistogramState {
    counts: [u64; LATENCY_BUCKETS.len()],
    sum_micros: u64,
    count: u64,
}

/// A [`Histogram`] read at one consistent instant: cumulative
/// `_bucket{le=…}` values (ascending, one per finite bound), the
/// running sum, and the total count all agree with each other.
pub(super) struct HistogramSnapshot {
    pub(super) cumulative: [u64; LATENCY_BUCKETS.len()],
    pub(super) sum_micros: u64,
    pub(super) count: u64,
}

impl Histogram {
    pub(super) fn observe(&self, elapsed: Duration) {
        // Bucket at microsecond precision: `as_millis` truncates, so a
        // 1.9 ms observation would land in the `le="0.001"` bucket —
        // every fractional latency slid one bucket optimistic, and the
        // low buckets, where this server's common case lives, are
        // exactly where that skews `histogram_quantile` the most.
        let micros = elapsed.as_micros();
        let mut state = self.state.lock();
        if let Some(bin) = LATENCY_BUCKETS
            .iter()
            .position(|&(bound, _)| micros <= u128::from(bound) * 1000)
        {
            state.counts[bin] += 1;
        }
        state.sum_micros += elapsed.as_micros() as u64;
        state.count += 1;
    }

    pub(super) fn snapshot(&self) -> HistogramSnapshot {
        let state = self.state.lock();
        let mut running = 0u64;
        let mut cumulative = [0u64; LATENCY_BUCKETS.len()];
        for (slot, count) in cumulative.iter_mut().zip(&state.counts) {
            running += count;
            *slot = running;
        }
        HistogramSnapshot {
            cumulative,
            sum_micros: state.sum_micros,
            count: state.count,
        }
    }
}

/// Per-(method, route template) statistics.
#[derive(Default)]
pub(super) struct RouteStat {
    /// Status → count. Bounded per route: a route emits a handful of
    /// distinct statuses over its life.
    pub(super) by_status: RwLock<HashMap<u16, AtomicU64>>,
    pub(super) latency: Histogram,
}
