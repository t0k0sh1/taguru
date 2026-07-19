//! Loud environment-variable parsing: every knob here either resolves
//! to a valid value or falls back to a default with a logged `warn!`,
//! never a silent misconfiguration.

use tracing::warn;

pub(crate) fn env_number(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(value) => value.parse().unwrap_or_else(|_| {
            warn!("ignoring {key}={value}: not a number; using {default}");
            default
        }),
        Err(_) => default,
    }
}

/// An optional boolean from the environment: "1"/"true" is true,
/// "0"/"false" is false (both case-insensitive), unset keeps
/// `default`, and any other value is ignored with a warning and also
/// keeps `default` — the same "never silent" contract `env_number`
/// applies to unparseable input, extended to values that parse fine
/// as text but name no recognized boolean (a typo, "yes"/"no",
/// "on"/"off" from another tool's convention). Silently falling back
/// here would let an operator believe a flag they misspelled was
/// applied when it was not.
pub(crate) fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) if value == "1" || value.eq_ignore_ascii_case("true") => true,
        Ok(value) if value == "0" || value.eq_ignore_ascii_case("false") => false,
        Ok(value) => {
            warn!(
                "ignoring {key}={value}: not a recognized boolean (1/true or 0/false); using {default}"
            );
            default
        }
        Err(_) => default,
    }
}

/// An optional 0..=1 fraction from the environment; anything else
/// (including NaN) is ignored with a warning, keeping the built-in
/// calibration.
pub(crate) fn env_floor(key: &str) -> Option<f32> {
    let value = std::env::var(key).ok()?;
    match value.parse::<f32>() {
        Ok(floor) if (0.0..=1.0).contains(&floor) => Some(floor),
        _ => {
            warn!("ignoring {key}={value}: not a number between 0 and 1");
            None
        }
    }
}

/// `TAGURU_METRICS_PER_CONTEXT`: 0/false (and unset) = off, 1/true/all
/// = every context, an integer ≥ 2 = the top-N contexts by total disk
/// bytes. `1` deliberately reads as the boolean "on = all", not top-1:
/// an operator typing `=1` almost certainly means the flag convention
/// every other TAGURU_ boolean uses, and silently truncating the fleet
/// to its single biggest context would be the misconfiguration this
/// module exists to prevent — while "exactly my biggest context" is
/// what `GET /contexts` answers already. Anything else warns and stays
/// off, per the file's contract.
pub(crate) fn env_per_context_metrics(key: &str) -> crate::metrics::PerContextMetrics {
    use crate::metrics::PerContextMetrics;
    let Ok(value) = std::env::var(key) else {
        return PerContextMetrics::Off;
    };
    if value == "0" || value.eq_ignore_ascii_case("false") {
        return PerContextMetrics::Off;
    }
    if value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("all") {
        return PerContextMetrics::All;
    }
    match value.parse::<usize>() {
        // 0 and 1 matched above, so a parsed count is always ≥ 2.
        Ok(top) => PerContextMetrics::Top(top),
        Err(_) => {
            warn!(
                "ignoring {key}={value}: not 0/false, 1/true/all, or a top-N \
                 count; per-context metrics stay off"
            );
            PerContextMetrics::Off
        }
    }
}

/// `tokio::time::interval` panics on a zero period — with
/// `TAGURU_FLUSH_SECS=0` that panic fires inside the spawned flusher
/// task, not the main thread, so the server keeps listening and
/// answering requests while dirty contexts silently stop persisting
/// forever. Floor to 1 instead, loudly, the same "never silent" rule
/// `env_number` already applies to unparseable input.
pub(crate) fn resolve_flush_secs(requested: usize) -> usize {
    if requested == 0 {
        warn!("TAGURU_FLUSH_SECS=0 would never fire (and would panic the flusher task); using 1");
        1
    } else {
        requested
    }
}

/// The limiter holds its budget in a u32; a bigger env value would be
/// silently clamped inside the constructor while the boot line logged
/// the raw number — the logged limit and the enforced limit must be
/// the same number, and the clamp must be loud like every other
/// out-of-range env here.
pub(crate) fn resolve_per_minute(name: &str, requested: usize) -> u32 {
    u32::try_from(requested).unwrap_or_else(|_| {
        warn!(
            "{name}={requested} exceeds the limiter's ceiling; clamping to {}",
            u32::MAX
        );
        u32::MAX
    })
}

/// Whether the boot-time "off-loopback, no rate limit" warning should
/// fire. `0.0.0.0` — the Docker image's default bind — is included:
/// it's still true that *something* beyond this process can reach the
/// port if it's published or routed, the process just can't see how
/// far. A concrete non-loopback address (an explicit LAN/public IP)
/// warrants the same warning for the same reason.
pub(crate) fn needs_off_loopback_warning(ip: std::net::IpAddr, rate_limit_disabled: bool) -> bool {
    !ip.is_loopback() && rate_limit_disabled
}

/// Tokio reserves a few high bits in its semaphore permit counter and
/// panics when constructed above `MAX_PERMITS`. An operator-provided usize
/// must therefore be clamped before it reaches `HeavyOpsLimiter::new`.
pub(crate) fn resolve_heavy_ops(requested: usize) -> usize {
    if requested > tokio::sync::Semaphore::MAX_PERMITS {
        warn!(
            "TAGURU_MAX_CONCURRENT_HEAVY_OPS={requested} exceeds the semaphore's ceiling; \
             clamping to {}",
            tokio::sync::Semaphore::MAX_PERMITS
        );
        tokio::sync::Semaphore::MAX_PERMITS
    } else {
        requested
    }
}

pub(crate) const DEFAULT_MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

/// `TAGURU_REQUEST_TIMEOUT_SECS=0` reads as "no timeout" — its
/// neighbor in the usage text documents 0 = off — but would wrap every
/// request in a zero-length budget: anything that doesn't resolve on
/// its very first poll (most writes, under real network latency)
/// answers 408 while the server boots and logs as if healthy. Floor to
/// 1, loudly; a huge value is how to effectively disable the budget.
pub(crate) fn resolve_timeout_secs(requested: usize) -> usize {
    if requested == 0 {
        warn!(
            "TAGURU_REQUEST_TIMEOUT_SECS=0 would 408 every request; using 1 \
             (set a large value to effectively disable the budget)"
        );
        1
    } else {
        requested
    }
}

/// The same trap for `TAGURU_MAX_BODY_BYTES=0`: by analogy with the
/// WAL ceilings' "0 = off" it reads as "no cap", but a zero
/// DefaultBodyLimit refuses every request that carries a body — all
/// writes 413, no startup complaint. And a truly uncapped body would
/// hand an allocation lever to whoever can reach the port, so 0 gets
/// the default back instead, loudly.
pub(crate) fn resolve_body_bytes(requested: usize) -> usize {
    if requested == 0 {
        warn!(
            "TAGURU_MAX_BODY_BYTES=0 would refuse every write; using the 8 MiB default \
             (set an explicit larger cap for bigger bodies)"
        );
        DEFAULT_MAX_BODY_BYTES
    } else {
        requested
    }
}

pub(crate) const DEFAULT_MCP_MAX_RESULT_BYTES: usize = 8 * 1024 * 1024;

/// Below this a cap is almost certainly a fat-fingered value (a byte
/// count typed where a kibibyte count was meant) rather than a
/// deliberate small budget — still honored, just flagged.
const MCP_MAX_RESULT_BYTES_WARN_FLOOR: usize = 64 * 1024;

/// The same "0 = unbounded is a trap" reasoning as
/// [`resolve_body_bytes`]: an uncapped `POST /mcp` tool result hands
/// an allocation lever to whoever can reach a tool that returns a lot
/// of data (`export_context` on a large context, chiefly), so 0
/// floors to the default instead of disabling the cap. Anything
/// nonzero but under 64 KiB is small enough to likely be a mistake —
/// obeyed regardless, just logged.
pub(crate) fn resolve_mcp_max_result_bytes(requested: usize) -> usize {
    if requested == 0 {
        warn!(
            "TAGURU_MCP_MAX_RESULT_BYTES=0 would refuse every tool result; using the 8 MiB \
             default (set an explicit larger cap for bigger results)"
        );
        return DEFAULT_MCP_MAX_RESULT_BYTES;
    }
    if requested < MCP_MAX_RESULT_BYTES_WARN_FLOOR {
        warn!(
            requested,
            "TAGURU_MCP_MAX_RESULT_BYTES is under 64 KiB — most tool results will fit, but \
             double check this wasn't meant to be a larger unit"
        );
    }
    requested
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_secs_zero_is_floored_to_one_instead_of_panicking_the_flusher() {
        assert_eq!(resolve_flush_secs(0), 1);
        assert_eq!(resolve_flush_secs(5), 5);
    }

    /// The knob's three shapes — and the deliberate reading of `1` as
    /// the boolean "all", never top-1 (see the parser's doc).
    #[test]
    fn per_context_metrics_parses_the_boolean_dialect_and_top_n() {
        use crate::metrics::PerContextMetrics;

        assert_eq!(
            env_per_context_metrics("TAGURU_TEST_PCM_UNSET"),
            PerContextMetrics::Off
        );
        let key = "TAGURU_TEST_PCM_VALUE";
        for (value, expected) in [
            ("0", PerContextMetrics::Off),
            ("false", PerContextMetrics::Off),
            ("1", PerContextMetrics::All),
            ("true", PerContextMetrics::All),
            ("all", PerContextMetrics::All),
            ("ALL", PerContextMetrics::All),
            ("2", PerContextMetrics::Top(2)),
            ("25", PerContextMetrics::Top(25)),
            ("banana", PerContextMetrics::Off),
            ("-3", PerContextMetrics::Off),
        ] {
            unsafe { std::env::set_var(key, value) };
            assert_eq!(env_per_context_metrics(key), expected, "value {value}");
        }
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn zero_timeout_and_body_cap_are_floored_loudly_not_obeyed() {
        assert_eq!(resolve_timeout_secs(0), 1);
        assert_eq!(resolve_timeout_secs(30), 30);
        assert_eq!(resolve_body_bytes(0), DEFAULT_MAX_BODY_BYTES);
        assert_eq!(resolve_body_bytes(1024), 1024);
    }

    #[test]
    fn zero_mcp_result_cap_is_floored_but_a_small_one_is_still_obeyed() {
        assert_eq!(
            resolve_mcp_max_result_bytes(0),
            DEFAULT_MCP_MAX_RESULT_BYTES
        );
        assert_eq!(resolve_mcp_max_result_bytes(1024), 1024);
        assert_eq!(
            resolve_mcp_max_result_bytes(DEFAULT_MCP_MAX_RESULT_BYTES),
            DEFAULT_MCP_MAX_RESULT_BYTES
        );
    }

    #[test]
    fn per_minute_rates_clamp_at_the_limiter_ceiling() {
        assert_eq!(resolve_per_minute("X", 0), 0);
        assert_eq!(resolve_per_minute("X", 600), 600);
        assert_eq!(resolve_per_minute("X", u32::MAX as usize + 1), u32::MAX);
    }

    #[test]
    fn off_loopback_warning_fires_only_without_a_rate_limit() {
        use std::net::{IpAddr, Ipv4Addr};

        let loopback = IpAddr::V4(Ipv4Addr::LOCALHOST);
        let unspecified = IpAddr::V4(Ipv4Addr::UNSPECIFIED); // 0.0.0.0 — the image's default bind
        let lan = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 5));
        let (disabled, configured) = (true, false); // rate_limit_disabled

        // Loopback never warns, rate limit configured or not.
        assert!(!needs_off_loopback_warning(loopback, disabled));
        assert!(!needs_off_loopback_warning(loopback, configured));
        // 0.0.0.0 (the Docker image's default bind) and an explicit LAN
        // address both warn with no rate limit configured...
        assert!(needs_off_loopback_warning(unspecified, disabled));
        assert!(needs_off_loopback_warning(lan, disabled));
        // ...and both stay quiet once one is.
        assert!(!needs_off_loopback_warning(unspecified, configured));
        assert!(!needs_off_loopback_warning(lan, configured));
    }

    #[test]
    fn heavy_operation_limits_clamp_before_semaphore_construction() {
        assert_eq!(resolve_heavy_ops(0), 0);
        assert_eq!(resolve_heavy_ops(2), 2);
        assert_eq!(
            resolve_heavy_ops(usize::MAX),
            tokio::sync::Semaphore::MAX_PERMITS
        );
    }
}
