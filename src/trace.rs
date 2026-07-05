//! Distributed tracing: an opt-in OTLP span pipeline over the existing
//! `tracing` instrumentation, plus inbound trace-context extraction.
//!
//! Setting `OTEL_EXPORTER_OTLP_ENDPOINT` (or the `_TRACES_`-specific
//! variant) turns span export on; all other knobs are the standard
//! `OTEL_*` variables the SDK reads itself (service name, headers,
//! batch cadence). Unset, the server behaves exactly as before —
//! no exporter thread, no request spans, no extra log fields.
//!
//! Inbound requests may carry a parent trace context as W3C
//! `traceparent`/`tracestate` or as the AWS `X-Amzn-Trace-Id` form
//! (ALB / API Gateway); both land in the same request span, so Taguru
//! joins whichever trace its front door started.

use std::sync::OnceLock;

use axum::http::HeaderMap;
use opentelemetry::Context;
use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;

/// Set once by [`provider`]; the request middleware branches on it so
/// the disabled mode stays byte-identical to the pre-tracing server.
static ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether span export was configured at boot.
pub fn enabled() -> bool {
    ENABLED.get().copied().unwrap_or(false)
}

/// Builds the OTLP tracer provider when an endpoint is configured.
/// Returns the provider (its batch worker owns unexported spans, so
/// `shutdown()` must run at exit) and, separately, a build-error
/// message — the caller logs it *after* the subscriber exists, which
/// is why this does not log itself.
pub fn provider() -> (Option<SdkTracerProvider>, Option<String>) {
    let configured = [
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ]
    .iter()
    .any(|key| std::env::var(key).is_ok_and(|value| !value.trim().is_empty()));
    if !configured {
        let _ = ENABLED.set(false);
        return (None, None);
    }
    // The exporter reads endpoint/headers/protocol from the same
    // OTEL_* variables that gated us here.
    let exporter = match opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()
    {
        Ok(exporter) => exporter,
        Err(error) => {
            let _ = ENABLED.set(false);
            return (None, Some(error.to_string()));
        }
    };
    // Resource::builder() already honors OTEL_SERVICE_NAME and
    // OTEL_RESOURCE_ATTRIBUTES; only the fallback name is ours.
    let mut resource = Resource::builder();
    if std::env::var("OTEL_SERVICE_NAME").is_err() {
        resource = resource.with_service_name("taguru");
    }
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource.build())
        .build();
    let _ = ENABLED.set(true);
    (Some(provider), None)
}

/// The parent context for a request span: W3C `traceparent` wins,
/// the AWS `X-Amzn-Trace-Id` form is the fallback, and neither means
/// the span starts a fresh trace. The sampled flag rides along, so the
/// default parent-based sampler respects an upstream "not sampled".
pub fn extract_parent(headers: &HeaderMap) -> Context {
    let header = |name: &str| headers.get(name).and_then(|value| value.to_str().ok());
    let remote = header("traceparent")
        .and_then(|value| parse_traceparent(value, header("tracestate")))
        .or_else(|| header("x-amzn-trace-id").and_then(parse_xray));
    match remote {
        Some(span_context) => Context::new().with_remote_span_context(span_context),
        None => Context::new(),
    }
}

/// `{version}-{trace-id:32}-{parent-id:16}-{flags:2}`, per W3C Trace
/// Context. Version `ff` is forbidden; an unknown version parses
/// leniently (its first three fields keep the same layout) but version
/// `00` must have exactly four fields.
fn parse_traceparent(value: &str, tracestate: Option<&str>) -> Option<SpanContext> {
    let mut parts = value.trim().split('-');
    let version = parts.next()?;
    if version.len() != 2
        || !version.bytes().all(|byte| byte.is_ascii_hexdigit())
        || version.eq_ignore_ascii_case("ff")
    {
        return None;
    }
    let trace_id = TraceId::from(parse_hex(parts.next()?, 32)?);
    let span_id = SpanId::from(parse_hex(parts.next()?, 16)? as u64);
    let flags = parse_hex(parts.next()?, 2)? as u8;
    if version == "00" && parts.next().is_some() {
        return None;
    }
    if trace_id == TraceId::INVALID || span_id == SpanId::INVALID {
        return None;
    }
    // A malformed tracestate is dropped, not fatal — the spec says the
    // trace itself must still be honored.
    let trace_state = tracestate
        .and_then(|value| value.parse::<TraceState>().ok())
        .unwrap_or_default();
    Some(SpanContext::new(
        trace_id,
        span_id,
        TraceFlags::new(flags) & TraceFlags::SAMPLED,
        true,
        trace_state,
    ))
}

/// `Root=1-{epoch:8}-{unique:24};Parent={span:16};Sampled={0|1}`, the
/// header ALB and API Gateway inject. The epoch and unique parts
/// concatenate into the 32-hex trace id (the same mapping the X-Ray
/// exporter reverses). Without a `Parent` there is no span to attach
/// to, so Root-only headers start a fresh trace.
fn parse_xray(value: &str) -> Option<SpanContext> {
    let mut root = None;
    let mut parent = None;
    let mut sampled = false;
    for field in value.split(';') {
        let Some((key, field_value)) = field.trim().split_once('=') else {
            continue;
        };
        if key.eq_ignore_ascii_case("Root") {
            root = Some(field_value);
        } else if key.eq_ignore_ascii_case("Parent") {
            parent = Some(field_value);
        } else if key.eq_ignore_ascii_case("Sampled") {
            sampled = field_value == "1";
        }
    }
    let mut root_parts = root?.splitn(3, '-');
    if root_parts.next()? != "1" {
        return None;
    }
    let epoch = root_parts.next()?;
    let unique = root_parts.next()?;
    if epoch.len() != 8 || unique.len() != 24 {
        return None;
    }
    let trace_id = TraceId::from(parse_hex(&format!("{epoch}{unique}"), 32)?);
    let span_id = SpanId::from(parse_hex(parent?, 16)? as u64);
    if trace_id == TraceId::INVALID || span_id == SpanId::INVALID {
        return None;
    }
    let flags = if sampled {
        TraceFlags::SAMPLED
    } else {
        TraceFlags::default()
    };
    Some(SpanContext::new(
        trace_id,
        span_id,
        flags,
        true,
        TraceState::default(),
    ))
}

/// Exactly `width` hex digits — `from_str_radix` alone would accept a
/// leading `+` and any length, which the wire formats forbid.
fn parse_hex(hex: &str, width: usize) -> Option<u128> {
    if hex.len() != width || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u128::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    const TRACEPARENT: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.append(
                axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    #[test]
    fn a_w3c_traceparent_becomes_the_remote_parent() {
        let context = extract_parent(&headers(&[("traceparent", TRACEPARENT)]));
        let span_context = context.span().span_context().clone();
        assert!(span_context.is_valid());
        assert!(span_context.is_remote());
        assert_eq!(
            span_context.trace_id().to_string(),
            "0af7651916cd43dd8448eb211c80319c"
        );
        assert_eq!(span_context.span_id().to_string(), "b7ad6b7169203331");
        assert!(span_context.is_sampled());
    }

    #[test]
    fn tracestate_rides_along_and_a_malformed_one_is_dropped() {
        let context = extract_parent(&headers(&[
            ("traceparent", TRACEPARENT),
            ("tracestate", "vendor=value,other=thing"),
        ]));
        let state = context.span().span_context().trace_state().clone();
        assert_eq!(state.get("vendor"), Some("value"));

        let context = extract_parent(&headers(&[
            ("traceparent", TRACEPARENT),
            ("tracestate", "not a valid entry"),
        ]));
        assert!(context.span().span_context().is_valid());
    }

    #[test]
    fn malformed_traceparents_are_rejected() {
        for bad in [
            "",
            "00",
            "ff-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "00-00000000000000000000000000000000-b7ad6b7169203331-01",
            "00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01-extra",
            "00-af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "00-+af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b716920333x-01",
        ] {
            assert!(
                parse_traceparent(bad, None).is_none(),
                "must reject {bad:?}"
            );
        }
        // An unknown (non-ff) version parses leniently, extra fields included.
        assert!(
            parse_traceparent(
                "cc-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01-what-ever",
                None
            )
            .is_some()
        );
    }

    #[test]
    fn an_xray_header_maps_onto_the_same_trace_identity() {
        let context = extract_parent(&headers(&[(
            "x-amzn-trace-id",
            "Root=1-5759e988-bd862e3fe1be46a994272793;Parent=53995c3f42cd8ad8;Sampled=1",
        )]));
        let span_context = context.span().span_context().clone();
        assert!(span_context.is_valid());
        assert!(span_context.is_remote());
        assert_eq!(
            span_context.trace_id().to_string(),
            "5759e988bd862e3fe1be46a994272793"
        );
        assert_eq!(span_context.span_id().to_string(), "53995c3f42cd8ad8");
        assert!(span_context.is_sampled());
    }

    #[test]
    fn xray_sampling_and_field_order_are_honored() {
        let sampled_off = parse_xray(
            "Sampled=0;Parent=53995c3f42cd8ad8;Root=1-5759e988-bd862e3fe1be46a994272793",
        )
        .unwrap();
        assert!(!sampled_off.is_sampled());

        // Self= fields (ALB) are ignored, not fatal.
        let with_self = parse_xray(
            "Self=1-00000001-000000000000000000000001;\
             Root=1-5759e988-bd862e3fe1be46a994272793;Parent=53995c3f42cd8ad8;Sampled=1",
        )
        .unwrap();
        assert_eq!(
            with_self.trace_id().to_string(),
            "5759e988bd862e3fe1be46a994272793"
        );
    }

    #[test]
    fn xray_without_a_parent_or_with_garbage_starts_a_fresh_trace() {
        for bad in [
            "Root=1-5759e988-bd862e3fe1be46a994272793",
            "Root=1-5759e988-bd862e3fe1be46a994272793;Sampled=1",
            "Root=2-5759e988-bd862e3fe1be46a994272793;Parent=53995c3f42cd8ad8",
            "Root=1-5759e9-bd862e3fe1be46a994272793;Parent=53995c3f42cd8ad8",
            "Root=1-5759e988-bd862e3fe1be46a994272793;Parent=0000000000000000",
            "just-noise",
            "",
        ] {
            assert!(parse_xray(bad).is_none(), "must reject {bad:?}");
        }
    }

    #[test]
    fn no_inbound_context_means_an_empty_parent() {
        let context = extract_parent(&HeaderMap::new());
        assert!(!context.span().span_context().is_valid());
    }

    #[test]
    fn w3c_wins_over_the_xray_header_when_both_arrive() {
        let context = extract_parent(&headers(&[
            ("traceparent", TRACEPARENT),
            (
                "x-amzn-trace-id",
                "Root=1-5759e988-bd862e3fe1be46a994272793;Parent=53995c3f42cd8ad8;Sampled=1",
            ),
        ]));
        assert_eq!(
            context.span().span_context().trace_id().to_string(),
            "0af7651916cd43dd8448eb211c80319c"
        );
    }
}
