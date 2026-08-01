//! Distributed tracing: an opt-in OTLP span pipeline over the existing
//! `tracing` instrumentation, plus bidirectional trace-context
//! propagation.
//!
//! Setting `OTEL_EXPORTER_OTLP_ENDPOINT` (or the `_TRACES_`-specific
//! variant) turns span export on; all other knobs are the standard
//! `OTEL_*` variables the SDK reads itself (service name, headers,
//! batch cadence). Unset, the server behaves exactly as before —
//! no exporter thread, no request spans, no extra log fields, and
//! every [`span!`] call site expands to `Span::none()`.
//!
//! Inbound requests may carry a parent trace context as W3C
//! `traceparent`/`tracestate` or as the AWS `X-Amzn-Trace-Id` form
//! (ALB / API Gateway); both land in the same request span, so Taguru
//! joins whichever trace its front door started ([`extract_parent`]).
//! Outbound calls — the router's shard fan-out, the stdio bridge's
//! calls to the HTTP server — carry the currently active span forward
//! the same way ([`inject_current`]). Both directions are hand-rolled
//! against the one W3C propagator (below), on purpose: nothing in this
//! tree reads `opentelemetry::global`'s propagator, so there is never a
//! second parser for either direction to quietly disagree with.
//!
//! Dual-included into `taguru-mcp` (`src/bin/taguru-mcp.rs`) as well as
//! compiled into `taguru` (`src/main.rs`), so every helper here works
//! from both binaries without a second copy — see `src/mcp.rs`'s own
//! module doc for the established pattern this follows.

use std::sync::OnceLock;

use http::HeaderMap;
use opentelemetry::Context;
use opentelemetry::propagation::{Injector, TextMapPropagator};
use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;

/// Set once by [`provider`]; the request middleware branches on it so
/// the disabled mode stays byte-identical to the pre-tracing server.
static ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether span export was configured at boot.
pub fn enabled() -> bool {
    ENABLED.get().copied().unwrap_or(false)
}

/// The one way a Taguru-owned span is created (ADR 0008 §5). With
/// export off this expands to `Span::none()` — no `Registry` storage
/// is opened, no field is formatted (the macro arguments are never
/// evaluated on that arm), and `enter`/`record`/`in_scope` on the
/// result are no-ops the optimizer removes. Strictly cheaper than
/// duplicating each call site's body behind `if enabled() { .. } else
/// { .. }`, the pattern this replaces everywhere but the one site
/// (`traced_request`, below) that predates this macro.
///
/// `pub(crate) use`, not `#[macro_export]`: call sites write
/// `use crate::trace::span;`, naming where the enabled-gate discipline
/// lives instead of finding a crate-root macro with no visible origin.
macro_rules! span {
    ($($arg:tt)*) => {
        if $crate::trace::enabled() {
            ::tracing::info_span!($($arg)*)
        } else {
            ::tracing::Span::none()
        }
    };
}
pub(crate) use span;

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

/// The export layer shared by both binaries' `init_telemetry` (ADR
/// 0008 §8/§9), factored here rather than duplicated so the privacy
/// exclusion below can never drift between them. INFO keeps the export
/// layer from re-enabling debug/trace callsites the stderr filter
/// would otherwise leave off; the `taguru::search` exclusion is the
/// privacy firewall ADR 0008 §8 requires — that target carries the raw
/// user question under `TAGURU_LOG_SEARCHES`, and an event on an
/// active span becomes an OTel span event, so it must never reach the
/// export layer regardless of its own level. `error_events_to_status`
/// defaults to true, so a WARN/ERROR event with a field literally
/// named `error` would otherwise color whatever span is active ERROR
/// behind the code's back (ADR 0008 §9) — status is set only where the
/// code explicitly calls `record("otel.status_code", "ERROR")`.
pub fn otel_layer<S>(provider: &SdkTracerProvider) -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    use opentelemetry::trace::TracerProvider as _;
    use tracing_subscriber::Layer as _;
    tracing_opentelemetry::layer()
        .with_tracer(provider.tracer(env!("CARGO_PKG_NAME")))
        .with_error_events_to_status(false)
        .with_filter(
            tracing_subscriber::filter::Targets::new()
                .with_default(tracing::Level::INFO)
                .with_target("taguru::search", tracing::level_filters::LevelFilter::OFF),
        )
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

/// The W3C propagator, process-local. Deliberately not registered
/// through `opentelemetry::global::set_text_map_propagator`: nothing
/// in this tree reads the global (no third-party middleware is
/// installed here), the global costs an `RwLock` read per inject, and
/// a global default would leave two extraction paths — this file's
/// hand-rolled [`parse_traceparent`] and the SDK's own — free to
/// disagree about a malformed header. Both directions stay in this one
/// file, side by side, so they cannot drift.
static W3C: OnceLock<TraceContextPropagator> = OnceLock::new();

/// `HeaderMap` as an OTel [`Injector`] — the write side of
/// [`extract_parent`]. `http::HeaderMap` is the same type in axum,
/// reqwest, and ureq 3, so this one impl serves the router's fan-out
/// and the stdio bridge's outbound calls alike.
struct HeaderInjector<'a>(&'a mut HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        // A propagator only ever emits `traceparent`/`tracestate`; the
        // fallible conversions are here so a future propagator with an
        // exotic key or value cannot panic a request path.
        if let (Ok(name), Ok(value)) = (
            http::HeaderName::from_bytes(key.as_bytes()),
            http::HeaderValue::from_str(&value),
        ) {
            self.0.insert(name, value);
        }
    }
}

/// Writes `traceparent`/`tracestate` for `context` into `headers`,
/// replacing whatever was there. A no-op when export is off — which is
/// what keeps a bare pass-through of the caller's own trace headers
/// (the fallback every inject site also does) meaningful in that mode.
pub fn inject_context(context: &Context, headers: &mut HeaderMap) {
    if !enabled() {
        return;
    }
    W3C.get_or_init(TraceContextPropagator::new)
        .inject_context(context, &mut HeaderInjector(headers));
}

/// [`inject_context`] for the span this thread is currently inside —
/// the outbound call's real parent, whether that is `taguru.retrieve`,
/// one of its phases, `taguru.shard_call`, or `taguru.tool_call`.
pub fn inject_current(headers: &mut HeaderMap) {
    if !enabled() {
        return;
    }
    use tracing_opentelemetry::OpenTelemetrySpanExt as _;
    inject_context(&tracing::Span::current().context(), headers);
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

/// Folds a request method to a fixed set of `&'static str` labels. RFC
/// 9110 leaves the method an open token, so an unauthenticated client
/// can send an unbounded stream of distinct extension methods; keyed
/// straight into the metrics map (or a span name) each would mint a new
/// series. Anything outside the standard set collapses to `<other>`,
/// mirroring how the route collapses to `<unmatched>`.
///
/// Shared by `metrics::track_http` and `route::track_router_http` — one
/// server-span builder, two transports (ADR 0008 §5).
#[allow(dead_code)] // consumed by taguru's HTTP/router middleware; taguru-mcp has no HTTP server of its own
pub(crate) fn normalized_method(method: &http::Method) -> &'static str {
    match *method {
        http::Method::GET => "GET",
        http::Method::POST => "POST",
        http::Method::PUT => "PUT",
        http::Method::DELETE => "DELETE",
        http::Method::PATCH => "PATCH",
        http::Method::HEAD => "HEAD",
        http::Method::OPTIONS => "OPTIONS",
        http::Method::TRACE => "TRACE",
        http::Method::CONNECT => "CONNECT",
        _ => "<other>",
    }
}

/// Runs the request inside an OTel server span. Span name and
/// attributes follow HTTP semconv (`{method} {route}`, method only
/// when unmatched); a 5xx marks the span as an error, a 4xx does not —
/// for a server, a client's mistake is a normal outcome. The span name
/// itself is the one exception to ADR 0008 §5's `taguru.`-prefix rule:
/// `otel.name` already overrides the macro name below, so the literal
/// `"request"` never reaches the wire.
///
/// Shared by `metrics::track_http` and `route::track_router_http`.
#[allow(dead_code)] // consumed by taguru's HTTP/router middleware; taguru-mcp has no HTTP server of its own
pub(crate) async fn traced_request(
    method: &str,
    route: &str,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> (axum::response::Response, Option<String>) {
    use tracing::Instrument as _;
    use tracing_opentelemetry::OpenTelemetrySpanExt as _;

    let span = tracing::info_span!(
        "request",
        otel.name = %if route == "<unmatched>" {
            method.to_string()
        } else {
            format!("{method} {route}")
        },
        otel.kind = "server",
        http.request.method = %method,
        http.route = %route,
        url.path = %request.uri().path(),
        http.response.status_code = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
    );
    // Only fails without an export layer, and we only run when one is
    // installed.
    let _ = span.set_parent(extract_parent(request.headers()));
    let trace_id = {
        use opentelemetry::trace::TraceContextExt as _;
        span.context().span().span_context().trace_id().to_string()
    };

    let response = next.run(request).instrument(span.clone()).await;

    // i64 keeps the attribute an OTLP int — a bare u16 records as text.
    span.record(
        "http.response.status_code",
        i64::from(response.status().as_u16()),
    );
    if response.status().is_server_error() {
        span.record("otel.status_code", "ERROR");
    }
    (response, Some(trace_id))
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
    use http::HeaderValue;

    const TRACEPARENT: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.append(
                http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
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

    // The tests below exercise the injector directly against the
    // private `W3C`/`HeaderInjector` machinery rather than through the
    // public `inject_context`/`inject_current` — those gate on
    // `enabled()`, which is a process-global `OnceLock` only
    // `provider()` can set, and `provider()` is never called from a
    // unit test in this binary: doing so would pin every other test's
    // `span!` call site to the enabled arm for the rest of the process,
    // regardless of test order. Integration tests under `tests/`
    // exercise the enabled path in a spawned, disposable process
    // instead (`tests/http_api/observability.rs`).

    #[test]
    fn the_injector_round_trips_with_extract_parent() {
        // A remote parent, as `extract_parent` would hand a phase span.
        let inbound = extract_parent(&headers(&[
            ("traceparent", TRACEPARENT),
            ("tracestate", "vendor=value"),
        ]));

        let mut outbound = HeaderMap::new();
        TraceContextPropagator::new().inject_context(&inbound, &mut HeaderInjector(&mut outbound));

        let traceparent = outbound
            .get("traceparent")
            .and_then(|value| value.to_str().ok())
            .expect("propagator must write a traceparent header");
        let tracestate = outbound
            .get("tracestate")
            .and_then(|value| value.to_str().ok());

        let round_tripped = extract_parent(&headers(&[
            ("traceparent", traceparent),
            ("tracestate", tracestate.unwrap_or_default()),
        ]));
        assert_eq!(
            round_tripped.span().span_context().trace_id(),
            inbound.span().span_context().trace_id(),
        );
        assert_eq!(
            round_tripped.span().span_context().span_id(),
            inbound.span().span_context().span_id(),
        );
        assert_eq!(
            round_tripped.span().span_context().trace_flags(),
            inbound.span().span_context().trace_flags(),
        );
        assert_eq!(tracestate, Some("vendor=value"));
    }

    #[test]
    fn inject_current_is_a_noop_when_export_is_off() {
        // `enabled()` defaults to false until `provider()` sets it —
        // never called in this test binary (see the comment above).
        assert!(!enabled());
        let mut headers = HeaderMap::new();
        inject_current(&mut headers);
        assert!(headers.is_empty());
    }

    #[test]
    fn inject_context_is_a_noop_when_export_is_off() {
        assert!(!enabled());
        let context = extract_parent(&headers(&[("traceparent", TRACEPARENT)]));
        let mut outbound = HeaderMap::new();
        inject_context(&context, &mut outbound);
        assert!(outbound.is_empty());
    }

    #[test]
    fn nonstandard_methods_fold_to_a_single_label() {
        // Standard methods keep their identity...
        assert_eq!(normalized_method(&http::Method::GET), "GET");
        assert_eq!(normalized_method(&http::Method::DELETE), "DELETE");
        // ...but an extension-method token — which a client can mint
        // without bound, ahead of auth — collapses to one series rather
        // than growing the metrics map (or a span name) per distinct
        // value.
        let weird = http::Method::from_bytes(b"M0001").unwrap();
        assert_eq!(normalized_method(&weird), "<other>");
        let also = http::Method::from_bytes(b"FROBNICATE").unwrap();
        assert_eq!(normalized_method(&also), "<other>");
    }

    #[test]
    fn header_injector_drops_a_malformed_key_without_panicking() {
        let mut headers = HeaderMap::new();
        let mut injector = HeaderInjector(&mut headers);
        // Neither a valid header name (control byte) nor a valid
        // header value (bare CR) — must be dropped silently, not
        // panic a request/response path.
        injector.set("bad\nkey", "value".to_string());
        injector.set("tracestate", "bad\rvalue".to_string());
        assert!(headers.is_empty());
    }
}
