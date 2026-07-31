//! Budget semantics for evidence selection (ADR 0006 §8, for #304).
//!
//! Three independent hard ceilings — items, bytes, tokens — none a
//! priority over another. Every default/ceiling pair routes through
//! the same [`clamp`](crate::api::clamp) funnel every other numeric
//! limit in `src/api.rs` uses, so an omitted request value takes the
//! default and nothing ever exceeds the ceiling.

use serde::{Deserialize, Serialize};

/// `max_items` default (ADR 0006 §8) — matches `MAX_MATCH_LIMIT`'s own
/// default posture (`src/api.rs:1078`).
pub(crate) const DEFAULT_MAX_ITEMS: usize = 40;
/// `max_items` ceiling (ADR 0006 §8), numerically identical to
/// `MAX_MATCH_LIMIT` (`src/api.rs:1078`) — no requirement the two
/// agree, but reusing the house number avoids a second, unexplained
/// ceiling of the same size.
pub(crate) const CEILING_MAX_ITEMS: usize = 1000;

/// `max_bytes` default (ADR 0006 §8): 64 KiB.
pub(crate) const DEFAULT_MAX_BYTES: usize = 65536;
/// `max_bytes` ceiling (ADR 0006 §8): 1 MiB, chosen well under
/// `DEFAULT_MCP_MAX_RESULT_BYTES` (8 MiB, `src/env.rs:206`) so a
/// package built at this endpoint's own ceiling is never silently
/// truncated a second time by the MCP transport's independent cap.
pub(crate) const CEILING_MAX_BYTES: usize = 1024 * 1024;

/// `max_tokens` default (ADR 0006 §8).
pub(crate) const DEFAULT_MAX_TOKENS: usize = 4000;

/// The request-side `budget` object (ADR 0006 §5.1) — every field
/// optional, resolved through [`BudgetLimits::resolve`]. #305 wires
/// this into the HTTP request body; #304 only needs the shape to
/// exercise [`BudgetLimits::resolve`] against what #305 will send.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct BudgetRequest {
    pub(crate) max_bytes: Option<usize>,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) max_items: Option<usize>,
}

/// The three resolved, independent hard ceilings a call to selection
/// runs under (ADR 0006 §8) — also the wire shape of
/// `BudgetUsage.limits` (ADR 0006 §10), so a caller can always see
/// what ceiling actually applied, not only the request's raw input.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct BudgetLimits {
    pub(crate) max_items: usize,
    pub(crate) max_bytes: usize,
    pub(crate) max_tokens: usize,
}

impl BudgetLimits {
    /// `max_tokens` has no server-enforced ceiling beyond
    /// `max_items`/`max_bytes` implicitly bounding it (ADR 0006 §8) —
    /// an operator wanting a harder cap sets a smaller `max_bytes`.
    /// `usize::MAX` as the ceiling argument to `clamp` makes that
    /// "no ceiling" property literal rather than a special case.
    pub(crate) fn resolve(request: Option<BudgetRequest>) -> Self {
        let request = request.unwrap_or_default();
        Self {
            max_items: super::super::clamp(request.max_items, DEFAULT_MAX_ITEMS, CEILING_MAX_ITEMS),
            max_bytes: super::super::clamp(request.max_bytes, DEFAULT_MAX_BYTES, CEILING_MAX_BYTES),
            max_tokens: super::super::clamp(request.max_tokens, DEFAULT_MAX_TOKENS, usize::MAX),
        }
    }
}

/// The three-budget account after selection finished (ADR 0006 §10).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct BudgetUsage {
    pub(crate) items_used: usize,
    pub(crate) bytes_used: usize,
    pub(crate) tokens_used: usize,
    pub(crate) limits: BudgetLimits,
}

/// The compact-JSON array overhead in bytes (`[`, `]`, and `len - 1`
/// commas, all single-byte ASCII) for an array of `len` elements — the
/// same three characters `serde_json::to_string` emits around a `Vec`
/// with no extra whitespace. Used to fold per-item content lengths
/// into the array-level total ADR 0006 §8 defines `max_bytes`/
/// `max_tokens` over, without re-serializing the whole array on every
/// admission decision.
pub(crate) fn array_overhead(len: usize) -> usize {
    if len == 0 { 2 } else { 2 + (len - 1) }
}

/// ADR 0006 §8's estimated-token formula, accumulated in fixed-point
/// quarter-tokens (1 quarter = 0.25 tokens) so summation order can
/// never change the result — unlike `f64` accumulation, where
/// `(a + b) + c` and `a + (b + c)` can disagree once rounding enters,
/// which would threaten ADR 0006 §9's I3 determinism invariant. Each
/// Basic Latin scalar (`U+0000`-`U+007F`, i.e. `char::is_ascii`)
/// contributes 1 quarter; every other scalar (CJK ideographs, kana,
/// hangul, and everything else) contributes 4 quarters (1.0 token).
pub(crate) fn estimate_token_quarters(text: &str) -> u64 {
    text.chars()
        .map(|ch| if ch.is_ascii() { 1u64 } else { 4u64 })
        .sum()
}

/// The final `max_tokens` comparison: the ceiling of the summed
/// quarters, in whole tokens (ADR 0006 §8: "the estimated total is the
/// ceiling of the sum").
pub(crate) fn tokens_from_quarters(quarters: u64) -> usize {
    quarters.div_ceil(4) as usize
}

/// The exact byte length and estimated-token quarters of `value`'s own
/// compact JSON serialization, with `exclude_keys` removed first (ADR
/// 0006 §8: an `EvidenceItem`'s own `bytes`/`estimated_tokens` fields
/// are excluded from what they measure, or self-reference would make
/// the measured length depend on its own value). Serializes through
/// `serde_json::Value` rather than hand-tracking a mirror struct, so a
/// future field added to the wire type is measured correctly without
/// this function changing.
pub(crate) fn content_metrics<T: Serialize>(value: &T, exclude_keys: &[&str]) -> (usize, u64) {
    let mut json = serde_json::to_value(value).expect("evidence wire types always serialize");
    if let serde_json::Value::Object(map) = &mut json {
        for key in exclude_keys {
            map.remove(*key);
        }
    }
    let text = serde_json::to_string(&json).expect("a serde_json::Value always serializes");
    (text.len(), estimate_token_quarters(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_fills_defaults_and_clamps_to_ceilings() {
        let limits = BudgetLimits::resolve(None);
        assert_eq!(limits.max_items, DEFAULT_MAX_ITEMS);
        assert_eq!(limits.max_bytes, DEFAULT_MAX_BYTES);
        assert_eq!(limits.max_tokens, DEFAULT_MAX_TOKENS);

        let limits = BudgetLimits::resolve(Some(BudgetRequest {
            max_bytes: Some(usize::MAX),
            max_tokens: Some(usize::MAX),
            max_items: Some(usize::MAX),
        }));
        assert_eq!(limits.max_items, CEILING_MAX_ITEMS);
        assert_eq!(limits.max_bytes, CEILING_MAX_BYTES);
        assert_eq!(limits.max_tokens, usize::MAX);
    }

    #[test]
    fn zero_budget_is_a_valid_resolvable_request_not_an_error() {
        let limits = BudgetLimits::resolve(Some(BudgetRequest {
            max_bytes: Some(0),
            max_tokens: Some(0),
            max_items: Some(0),
        }));
        assert_eq!(limits.max_items, 0);
        assert_eq!(limits.max_bytes, 0);
        assert_eq!(limits.max_tokens, 0);
    }

    #[test]
    fn token_estimate_weighs_basic_latin_and_other_scalars_as_documented() {
        // 4 ASCII scalars => 4 quarters => ceil(4/4) = 1 token.
        assert_eq!(tokens_from_quarters(estimate_token_quarters("abcd")), 1);
        // 1 non-Latin scalar => 4 quarters => 1 token.
        assert_eq!(tokens_from_quarters(estimate_token_quarters("日")), 1);
        // 1 ASCII + 1 non-Latin => 1 + 4 = 5 quarters => ceil(5/4) = 2.
        assert_eq!(tokens_from_quarters(estimate_token_quarters("a日")), 2);
    }

    #[test]
    fn array_overhead_matches_real_compact_serialization() {
        assert_eq!(array_overhead(0), "[]".len());
        assert_eq!(array_overhead(1), "[x]".len() - 1); // brackets only, no comma
        let three: Vec<u8> = vec![1, 2, 3];
        let overhead = array_overhead(three.len());
        let rendered = serde_json::to_string(&three).unwrap();
        // rendered = "[1,2,3]": 2 brackets + 2 commas = 4 overhead bytes,
        // plus 3 one-byte digits — overhead alone must equal 4.
        assert_eq!(overhead, rendered.len() - 3);
    }

    #[test]
    fn content_metrics_excludes_named_keys_and_matches_real_json_length() {
        #[derive(Serialize)]
        struct Item {
            text: String,
            bytes: usize,
            estimated_tokens: usize,
        }
        let item = Item {
            text: "hi".to_string(),
            bytes: 999,
            estimated_tokens: 999,
        };
        let (bytes, quarters) = content_metrics(&item, &["bytes", "estimated_tokens"]);
        // Expected: {"text":"hi"} — 13 bytes, all ASCII => 13 quarters.
        let expected = serde_json::json!({"text": "hi"});
        let expected_text = serde_json::to_string(&expected).unwrap();
        assert_eq!(bytes, expected_text.len());
        assert_eq!(quarters, expected_text.len() as u64);
    }
}
