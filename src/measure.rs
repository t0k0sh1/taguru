//! Shared metric value shapes (ADR 0003 §9.3, ADR 0004 §10): `Distribution`
//! / `Ratio` / `Count` / `MetricValue` / `MetricDef` and their
//! constructors. `benchmark::compare` (issue #257) and `benchmark::search`
//! (issue #260) are peer consumers — neither owns this module — and
//! `taguru evaluate` (#215/#272-#279) is a third. Moved here from
//! `benchmark::compare` by issue #280 as a pure relocation: no shape, no
//! serialization, no behavior changed.
//!
//! `Distribution`/`Ratio`/`Count` also derive `Deserialize` (#277): the
//! three value shapes are read back out of `evaluation.json` by
//! `evaluate::compare`, following ADR 0004 §9.1's rule that a metric's
//! concrete variant is picked by first reading `definitions[metric]
//! .statistic` and deserializing into the type that name owns — never
//! through `MetricValue`'s own `#[serde(untagged)]`, which is
//! `Serialize`-only and unsound to deserialize through directly (a
//! `Ratio` payload can silently parse as a `Distribution`, since every
//! `Distribution` field but `n` is optional). `#[serde(default)]`
//! matches every other taguru-written-and-reread artifact's range-
//! acceptance posture (ADR 0003 §10): a field a future version adds is
//! read as absent by an older `compare`, never a hard parse error.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// `{n, min, p50, p90, p99, max, mean, sum}` (ADR 0003 §9.3). Every
/// field always serializes — `n: 0` pairs with every statistic `null`,
/// never an omitted key, so a reader tells "measured, zero samples"
/// from "this metric does not apply here" by key presence alone.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Distribution {
    n: u64,
    min: Option<f64>,
    p50: Option<f64>,
    p90: Option<f64>,
    p99: Option<f64>,
    max: Option<f64>,
    mean: Option<f64>,
    sum: Option<f64>,
}

impl Distribution {
    const EMPTY: Self = Self {
        n: 0,
        min: None,
        p50: None,
        p90: None,
        p99: None,
        max: None,
        mean: None,
        sum: None,
    };

    /// Builds from a sample set: drops non-finite values defensively,
    /// sorts once, and applies nearest-rank (ADR 0003 §9.3) with no
    /// interpolation — always an observed value, exactly reproducible
    /// by an external re-aggregator reading the same `runs/*.jsonl`.
    /// `pub(crate)`: `benchmark::compare` (#257) and `benchmark::search`
    /// (#260) both build the same shape for their own per-model
    /// distributions rather than hand-copying this struct.
    pub(crate) fn from_samples(mut samples: Vec<f64>) -> Self {
        samples.retain(|v| v.is_finite());
        let n = samples.len();
        if n == 0 {
            return Self::EMPTY;
        }
        samples.sort_by(f64::total_cmp);
        let sum: f64 = samples.iter().sum();
        Self {
            n: n as u64,
            min: Some(samples[0]),
            p50: Some(nearest_rank(&samples, 50)),
            p90: Some(nearest_rank(&samples, 90)),
            p99: Some(nearest_rank(&samples, 99)),
            max: Some(samples[n - 1]),
            mean: Some(sum / n as f64),
            sum: Some(sum),
        }
    }

    // Test-only: production code reads `n`/`min`/`max`/`sum`/`mean`
    // through `metric_csv_rows`/`MetricValue::scalar` in this module,
    // which has private-field access already; these accessors exist
    // only so `compare/tests.rs`, in a different module, can assert on
    // them.
    #[cfg(test)]
    pub(crate) fn n(&self) -> u64 {
        self.n
    }

    #[cfg(test)]
    pub(crate) fn min(&self) -> Option<f64> {
        self.min
    }

    #[cfg(test)]
    pub(crate) fn max(&self) -> Option<f64> {
        self.max
    }

    #[cfg(test)]
    pub(crate) fn sum(&self) -> Option<f64> {
        self.sum
    }
}

/// `x[ceil(p/100 * n) - 1]` — nearest-rank, no interpolation (ADR 0003
/// §9.3): always an observed value, never a synthesized one between two
/// samples. `sorted` must already be ascending and non-empty.
fn nearest_rank(sorted: &[f64], p: usize) -> f64 {
    let n = sorted.len();
    let rank = (p * n).div_ceil(100).clamp(1, n);
    sorted[rank - 1]
}

/// `{value, n, numerator}` (ADR 0003 §9.3): `numerator` attempts/
/// documents/lines matched some predicate, out of `n` total in scope.
/// `n: 0` pairs with `value: None` and `numerator: None` — never a
/// divide-by-zero `NaN`, never a silently misleading `0.0`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Ratio {
    value: Option<f64>,
    n: u64,
    numerator: Option<u64>,
}

/// `pub(crate)`: `benchmark::compare` (#257) and `benchmark::search`
/// (#260) both reuse this for their own rate metrics (empty-result
/// rate, per-lane hit rate).
pub(crate) fn ratio_metric(numerator: u64, n: u64) -> Ratio {
    if n == 0 {
        Ratio {
            value: None,
            n: 0,
            numerator: None,
        }
    } else {
        Ratio {
            value: Some(numerator as f64 / n as f64),
            n,
            numerator: Some(numerator),
        }
    }
}

impl Ratio {
    // Test-only — see the matching note on `Distribution`'s accessors.
    #[cfg(test)]
    pub(crate) fn value(&self) -> Option<f64> {
        self.value
    }

    #[cfg(test)]
    pub(crate) fn n(&self) -> u64 {
        self.n
    }

    #[cfg(test)]
    pub(crate) fn numerator(&self) -> Option<u64> {
        self.numerator
    }
}

/// `{value, n}`: a pooled total, union size, or derived rate, alongside
/// the number of documents/attempts it was pooled from. Used where
/// `Ratio`'s numerator-over-n-of-the-same-population shape does not
/// fit (a cross-unit rate) or where the value is not itself a share
/// (a union size, a summed count).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Count {
    value: Option<f64>,
    n: u64,
}

/// `pub(crate)`: pairs with `ratio_metric` — `Count`'s fields are
/// private like every other metric shape here, so this is the only way
/// to build one outside this module (ADR 0004 §10).
pub(crate) fn count_metric(value: Option<f64>, n: u64) -> Count {
    Count { value, n }
}

impl Count {
    // Test-only — see the matching note on `Distribution`'s accessors.
    #[cfg(test)]
    pub(crate) fn value(&self) -> Option<f64> {
        self.value
    }

    #[cfg(test)]
    pub(crate) fn n(&self) -> u64 {
        self.n
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum MetricValue {
    Distribution(Distribution),
    Ratio(Ratio),
    Count(Count),
}

impl MetricValue {
    pub(crate) fn sample_size(&self) -> u64 {
        match self {
            MetricValue::Distribution(d) => d.n,
            MetricValue::Ratio(r) => r.n,
            MetricValue::Count(c) => c.n,
        }
    }

    /// A single number to compare against a `{"min"|"max": f64}`
    /// threshold (ADR 0004 §9.3): `Distribution` compares by `mean`,
    /// `Ratio`/`Count` by `value`. `None` whenever the metric has no
    /// samples (`n: 0`, every variant) — a threshold treats that as a
    /// violation, not a silent pass, so this stays `None` rather than
    /// coercing to `0.0`.
    pub(crate) fn scalar(&self) -> Option<f64> {
        match self {
            MetricValue::Distribution(d) => d.mean,
            MetricValue::Ratio(r) => r.value,
            MetricValue::Count(c) => c.value,
        }
    }

    /// A `Ratio`'s raw numerator — the count behind `value` — for a
    /// reader reporting "X of N" rather than the share alone. `None`
    /// for the other variants, which have no numerator.
    pub(crate) fn numerator(&self) -> Option<u64> {
        match self {
            MetricValue::Ratio(r) => r.numerator,
            MetricValue::Distribution(_) | MetricValue::Count(_) => None,
        }
    }
}

/// One `definitions` entry (ADR 0003 §9.3): unit, statistic shape,
/// applicable scopes, description, source pointer, and known caveat —
/// embedded in the artifact itself, not only documented in prose
/// elsewhere.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct MetricDef {
    unit: &'static str,
    statistic: &'static str,
    scopes: &'static [&'static str],
    description: String,
    source: &'static str,
    caveat: Option<String>,
}

pub(crate) fn def(
    unit: &'static str,
    statistic: &'static str,
    scopes: &'static [&'static str],
    description: impl Into<String>,
    source: &'static str,
    caveat: Option<&str>,
) -> MetricDef {
    MetricDef {
        unit,
        statistic,
        scopes,
        description: description.into(),
        source,
        caveat: caveat.map(str::to_string),
    }
}

impl MetricDef {
    /// Used by `benchmark::compare`'s `append_metric_rows` to look up
    /// `measurements.csv`'s `unit` column.
    pub(crate) fn unit(&self) -> &'static str {
        self.unit
    }

    // Test-only — see the matching note on `Distribution`'s accessors.
    #[cfg(test)]
    pub(crate) fn caveat(&self) -> Option<&str> {
        self.caveat.as_deref()
    }
}

pub(crate) type MetricsMap = BTreeMap<String, MetricValue>;

/// `measurements.csv` is a value projection of `measurements.json`
/// (ADR 0003 §9.3): every numeric field of every metric becomes one
/// row. Lives beside the types it projects; the CSV-specific
/// formatting (`csv_field`, row assembly, file writing) stays in
/// `benchmark::compare`.
pub(crate) fn metric_csv_rows(value: &MetricValue) -> Vec<(&'static str, Option<f64>)> {
    match value {
        MetricValue::Distribution(d) => vec![
            ("n", Some(d.n as f64)),
            ("min", d.min),
            ("p50", d.p50),
            ("p90", d.p90),
            ("p99", d.p99),
            ("max", d.max),
            ("mean", d.mean),
            ("sum", d.sum),
        ],
        MetricValue::Ratio(r) => vec![
            ("value", r.value),
            ("n", Some(r.n as f64)),
            ("numerator", r.numerator.map(|v| v as f64)),
        ],
        MetricValue::Count(c) => vec![("value", c.value), ("n", Some(c.n as f64))],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_matches_the_adr_formula() {
        let one = vec![7.0];
        assert_eq!(nearest_rank(&one, 50), 7.0);
        assert_eq!(nearest_rank(&one, 99), 7.0);

        let two = vec![1.0, 2.0];
        assert_eq!(nearest_rank(&two, 50), 1.0, "ceil(50/100*2)=1 -> x[0]");
        assert_eq!(nearest_rank(&two, 90), 2.0, "ceil(90/100*2)=2 -> x[1]");

        let four = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(nearest_rank(&four, 50), 2.0, "ceil(50/100*4)=2 -> x[1]");

        let ten: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(nearest_rank(&ten, 90), 9.0, "ceil(90/100*10)=9 -> x[8]");

        let hundred: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(
            nearest_rank(&hundred, 99),
            99.0,
            "ceil(99/100*100)=99 -> x[98]"
        );

        let all_same = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        assert_eq!(nearest_rank(&all_same, 50), 5.0);
        assert_eq!(nearest_rank(&all_same, 99), 5.0);
    }

    #[test]
    fn distribution_from_samples_sorts_first() {
        let d = Distribution::from_samples(vec![3.0, 1.0, 2.0]);
        assert_eq!(d.min, Some(1.0));
        assert_eq!(d.max, Some(3.0));
        assert_eq!(d.n, 3);
    }

    #[test]
    fn distribution_with_zero_samples_serializes_all_stats_null() {
        let d = Distribution::from_samples(vec![]);
        let value = serde_json::to_value(&d).unwrap();
        let obj = value.as_object().unwrap();
        assert_eq!(obj["n"], 0);
        for key in ["min", "p50", "p90", "p99", "max", "mean", "sum"] {
            assert!(obj.contains_key(key), "{key} must be present");
            assert!(obj[key].is_null(), "{key} must be null, got {:?}", obj[key]);
        }
    }

    #[test]
    fn ratio_with_zero_denominator_serializes_value_and_numerator_null() {
        let r = ratio_metric(0, 0);
        let value = serde_json::to_value(&r).unwrap();
        assert!(value["value"].is_null());
        assert!(value["numerator"].is_null());
        assert_eq!(value["n"], 0);
    }

    #[test]
    fn ratio_with_a_denominator_computes_the_share() {
        let r = ratio_metric(3, 31);
        assert!((r.value.unwrap() - 3.0 / 31.0).abs() < 1e-12);
        assert_eq!(r.n, 31);
        assert_eq!(r.numerator, Some(3));
    }
}
