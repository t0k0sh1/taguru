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

/// [`civil_from_days`]'s inverse (Hinnant's `days_from_civil`, same
/// source): (year, month, day) → days since the Unix epoch. Consumed
/// by `extract --date`'s `YYYY-MM-DD` spelling (#466 S1), which
/// round-trips the result through [`iso8601_utc`] to refuse
/// non-existent dates rather than silently normalizing them.
pub(crate) fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = if m > 2 { m - 3 } else { m + 9 } as u64;
    let doy = (153 * mp + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
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

    #[test]
    fn days_from_civil_inverts_the_rendering_direction() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2026, 7, 26), 1785057262 / 86400);
        assert_eq!(days_from_civil(2024, 2, 29), 1709164800 / 86400);
        assert_eq!(days_from_civil(2000, 3, 1), 951868800 / 86400);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        // The negative-year era branch (`y - 399`): year 0 is a leap
        // year (≡ 2000 mod 400), so -1-03-01 → 0-03-01 spans 366 days.
        assert_eq!(days_from_civil(0, 3, 1) - days_from_civil(-1, 3, 1), 366);
    }
}
