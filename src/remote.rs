//! The one HTTP door every remote verb walks through: `taguru
//! calibrate`, `taguru communities`, and the `--url` verbs that
//! follow them (`export`, `compact`, `import` — ADR 0002 §2.1, §7,
//! issue #243) all speak to the server through this module instead
//! of hand-rolling their own client.
//!
//! Three invariants hold for every request this module sends:
//! - The client's own timeout sits above the server's default 30s
//!   request budget, so a server-side timeout answers as itself (a
//!   408 body with words) instead of a client-side cut.
//! - `http_status_as_error(false)` keeps non-2xx responses readable
//!   as bodies instead of turning them into transport errors, so the
//!   server's own error line survives to the caller.
//! - The bearer comes from the same environment variables the server
//!   itself reads — no separate credential story for the client.

use std::time::Duration;

use serde_json::Value;

/// A failure from the envelope surface, with 404 told apart — "no
/// artifact yet" is a first-run state, not an error.
pub(crate) enum ApiFailure {
    NotFound(String),
    Other(String),
}

impl ApiFailure {
    pub(crate) fn into_message(self) -> String {
        match self {
            ApiFailure::NotFound(message) | ApiFailure::Other(message) => message,
        }
    }
}

/// The one HTTP door: bearer attached when the environment holds one,
/// 200 unwrapped to `result`, anything else an error message carrying
/// the server's own words.
pub(crate) struct Api {
    agent: ureq::Agent,
    // calibrate's report echoes the target it measured, so this stays
    // a field callers can read rather than a private detail.
    pub(crate) base: String,
    token: Option<String>,
}

impl Api {
    pub(crate) fn new(base: String) -> Self {
        Self {
            // Above the server's default 30s request budget, so a
            // server-side timeout answers as itself (a 408 body with
            // words) instead of a client-side cut.
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs(35)))
                .http_status_as_error(false)
                .build()
                .into(),
            base,
            token: bearer_token(),
        }
    }

    /// Percent-encodes each path segment through the url crate —
    /// context names are operator strings and 日本語 names must
    /// address the same context the server stores.
    fn url(&self, segments: &[&str]) -> Result<String, String> {
        let mut url = url::Url::parse(&self.base)
            .map_err(|error| format!("'{}' is not a usable base URL: {error}", self.base))?;
        url.path_segments_mut()
            .map_err(|()| format!("'{}' cannot carry a path", self.base))?
            .extend(segments);
        Ok(url.to_string())
    }

    /// Attaches the bearer header when the environment holds one.
    /// Generic over ureq's request-builder typestate so every verb —
    /// GET without a body, POST with one — can share it.
    fn bearer<B>(&self, request: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
        match &self.token {
            Some(token) => request.header("Authorization", format!("Bearer {token}")),
            None => request,
        }
    }

    /// GET returning the raw body — the analysis stream is JSON
    /// Lines, not the envelope.
    pub(crate) fn get_raw(&self, segments: &[&str]) -> Result<String, String> {
        let url = self.url(segments)?;
        let request = self.bearer(self.agent.get(&url));
        let mut response = request.call().map_err(|error| format!("{url}: {error}"))?;
        let status = response.status().as_u16();
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|error| format!("{url}: unreadable response: {error}"))?;
        if status != 200 {
            let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            return Err(status_error(status, &parsed, &text, &url));
        }
        Ok(text)
    }

    pub(crate) fn get_envelope(&self, segments: &[&str]) -> Result<Value, ApiFailure> {
        let url = self.url(segments).map_err(ApiFailure::Other)?;
        let request = self.bearer(self.agent.get(&url));
        finish(request.call(), &url)
    }

    pub(crate) fn post_envelope(
        &self,
        segments: &[&str],
        body: &Value,
    ) -> Result<Value, ApiFailure> {
        let url = self.url(segments).map_err(ApiFailure::Other)?;
        let request = self.bearer(
            self.agent
                .post(&url)
                .header("Content-Type", "application/json"),
        );
        finish(request.send(body.to_string().as_str()), &url)
    }

    /// Message-only view of [`Api::get_envelope`], for callers that
    /// don't tell 404 apart from any other failure.
    pub(crate) fn get(&self, segments: &[&str]) -> Result<Value, String> {
        self.get_envelope(segments)
            .map_err(ApiFailure::into_message)
    }

    /// Message-only view of [`Api::post_envelope`], for callers that
    /// don't tell 404 apart from any other failure.
    pub(crate) fn post(&self, segments: &[&str], body: &Value) -> Result<Value, String> {
        self.post_envelope(segments, body)
            .map_err(ApiFailure::into_message)
    }

    /// One `POST /import` request carrying a pack of whole batches.
    pub(crate) fn import(&self, stream: &str) -> Result<(), String> {
        let url = self.url(&["import"])?;
        let request = self.bearer(
            self.agent
                .post(&url)
                .header("Content-Type", "application/x-ndjson"),
        );
        finish(request.send(stream), &url)
            .map(|_| ())
            .map_err(|failure| failure.into_message())
    }
}

/// Formats a non-2xx response the way the server would want it read:
/// its own `error` field if the body carries one, otherwise the
/// trimmed body itself. Shared by [`finish`] and [`Api::get_raw`] so
/// the two error surfaces can't drift apart.
fn status_error(status: u16, parsed: &Value, text: &str, url: &str) -> String {
    let message = parsed["error"].as_str().unwrap_or(text.trim());
    format!("{url} answered {status}: {message}")
}

/// Unwraps one envelope response: 200 hands back `result`, 404 comes
/// apart as [`ApiFailure::NotFound`], anything else carries the
/// server's own words.
fn finish(
    response: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
    url: &str,
) -> Result<Value, ApiFailure> {
    let mut response = response.map_err(|error| ApiFailure::Other(format!("{url}: {error}")))?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|error| ApiFailure::Other(format!("{url}: unreadable response: {error}")))?;
    let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if status != 200 {
        let message = status_error(status, &parsed, &text, url);
        return Err(if status == 404 {
            ApiFailure::NotFound(message)
        } else {
            ApiFailure::Other(message)
        });
    }
    // Keys off the parsed object itself — a substring search over the
    // raw text would also match "result" appearing inside a nested
    // value (e.g. `{"data":{"result":"nested"}}`), letting a
    // non-envelope response through as an accidental `Ok(Null)`.
    if !parsed
        .as_object()
        .is_some_and(|body| body.contains_key("result"))
    {
        return Err(ApiFailure::Other(format!(
            "{url}: not a taguru response: {}",
            text.trim()
        )));
    }
    Ok(parsed["result"].clone())
}

/// The bearer the server would accept, read the way the server reads
/// it: `TAGURU_API_TOKEN` outright, else the first `name:token` entry
/// of `TAGURU_API_TOKENS`. `None` = an unauthenticated server.
/// Crate-visible: every remote verb authenticates the same way — the
/// same variables the server reads.
pub(crate) fn bearer_token() -> Option<String> {
    if let Ok(token) = std::env::var("TAGURU_API_TOKEN") {
        let token = token.trim();
        if !token.is_empty() {
            return Some(token.to_string());
        }
    }
    token_from_ring(&std::env::var("TAGURU_API_TOKENS").ok()?)
}

/// The first `name:token` entry of a keyring that carries a token —
/// the parsing half of [`bearer_token`], pulled out so a test can
/// exercise it directly instead of pinning a copy of the logic.
fn token_from_ring(ring: &str) -> Option<String> {
    ring.split(',').find_map(|entry| {
        let (_, token) = entry.trim().split_once(':')?;
        let token = token.trim();
        (!token.is_empty()).then(|| token.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::token_from_ring;

    #[test]
    fn the_first_keyring_entry_serves_as_the_bearer() {
        assert_eq!(
            token_from_ring("ci:tokA,laptop:tokB"),
            Some("tokA".to_string())
        );
        // an entry without a token falls through to the next one
        assert_eq!(
            token_from_ring("ci: ,laptop:tokB"),
            Some("tokB".to_string())
        );
        assert_eq!(token_from_ring("nocolon"), None);
    }
}
