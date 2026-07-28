//! Dependency-free wall-clock helpers. `pub(crate)`: `benchmark`
//! (`taguru benchmark extract`'s `run_id`/manifest timestamps,
//! `taguru benchmark search`/`compare`'s `generated_at`) and `evaluate`
//! (#273, `evaluation.json`'s `generated_at`) are peer consumers with
//! no owner between them — the same relocation rationale ADR 0004 §10
//! applied to `src/measure.rs` (moved out of `benchmark::compare` so
//! neither `benchmark` nor `evaluate` is a tenant of the other).

use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// RFC3339 UTC, whole seconds (`"2026-07-26T09:14:22Z"`) — no `chrono`/
/// `time` dependency exists in this tree, so the civil-date conversion
/// is Howard Hinnant's `civil_from_days` algorithm
/// (howardhinnant.github.io/date_algorithms.html), the standard
/// closed-form days-since-epoch → (year, month, day) transform.
pub(crate) fn iso8601_utc(unix_secs: u64) -> String {
    let days = (unix_secs / 86400) as i64;
    let secs_of_day = unix_secs % 86400;
    let (y, m, d) = civil_from_days(days);
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day % 3600) / 60;
    let ss = secs_of_day % 60;
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso8601_matches_known_instants() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(1785057262), "2026-07-26T09:14:22Z");
        assert_eq!(iso8601_utc(1709164800), "2024-02-29T00:00:00Z");
        assert_eq!(iso8601_utc(951868800), "2000-03-01T00:00:00Z");
    }
}
