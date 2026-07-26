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

/// The `/health` version-skew preflight's own timeout — short and
/// separate from `Api`'s 35s request budget, since it is a
/// best-effort check that must not delay the real request behind it.
/// Matches `taguru health`'s own precedent for a quick, throwaway
/// probe (src/cli.rs).
const HEALTH_PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(4);

/// The page size the enumeration walk requests explicitly — matches
/// the server's own ceiling (`MAX_MATCH_LIMIT`, src/api.rs), named
/// here so the short-page termination rule has a known number to
/// compare against instead of guessing at the server's default.
const LIST_PAGE_LIMIT: usize = 1000;

/// ADR 0002 §7: a URL carrying userinfo (`https://user:token@host/`)
/// is refused — credentials on the command line are readable from
/// `ps` and shell history for the lifetime of the terminal, the exact
/// leak the existing `TAGURU_API_TOKEN` environment variable already
/// avoids. An unparseable `base` is left alone here: `Api::url`
/// already produces its own "not a usable base URL" message for that
/// case, on the first real request, and duplicating it here would
/// give one fault two different error surfaces.
pub(crate) fn reject_userinfo(base: &str) -> Result<(), String> {
    let Ok(url) = url::Url::parse(base) else {
        return Ok(());
    };
    if !url.username().is_empty() || url.password().is_some() {
        return Err(
            "the URL must not carry credentials (user:password@…) — set TAGURU_API_TOKEN \
             (or TAGURU_API_TOKENS) instead"
                .to_string(),
        );
    }
    Ok(())
}

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
        self.url_with_query(segments, &[])
    }

    /// Same as [`Api::url`], with query pairs appended — the
    /// enumeration walk's `limit`/`after` keyset paging is the only
    /// caller today.
    fn url_with_query(&self, segments: &[&str], query: &[(&str, &str)]) -> Result<String, String> {
        let mut url = url::Url::parse(&self.base)
            .map_err(|error| format!("'{}' is not a usable base URL: {error}", self.base))?;
        url.path_segments_mut()
            .map_err(|()| format!("'{}' cannot carry a path", self.base))?
            .extend(segments);
        if !query.is_empty() {
            url.query_pairs_mut().extend_pairs(query);
        }
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
        let retry_after = retry_after_header(&response);
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|error| format!("{url}: unreadable response: {error}"))?;
        if status != 200 {
            let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            return Err(status_error(
                status,
                &parsed,
                &text,
                &url,
                retry_after.as_deref(),
            ));
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

    /// [`Api::get`] with query pairs appended to the request.
    fn get_with_query(&self, segments: &[&str], query: &[(&str, &str)]) -> Result<Value, String> {
        let url = self.url_with_query(segments, query)?;
        let request = self.bearer(self.agent.get(&url));
        finish(request.call(), &url).map_err(ApiFailure::into_message)
    }

    /// Every name in a listing collection (`"contexts"` or
    /// `"groups"`), keyset-walked with the server's own page size —
    /// the enumeration step `export --url` (and, after it, `compact
    /// --url`) needs before it can call each item's own endpoint.
    pub(crate) fn list_names(&self, collection: &str) -> Result<Vec<String>, String> {
        self.list_names_paged(collection, LIST_PAGE_LIMIT)
    }

    /// [`Api::list_names`] with an explicit page size, split out so a
    /// test can walk several pages (including the short-page
    /// termination and the non-advancing-cursor guard) without
    /// provisioning thousands of contexts to cross the real ceiling.
    ///
    /// The collection name doubles as the envelope's array key — both
    /// `GET /contexts` and `GET /groups` answer `{total, <collection>:
    /// [{name, ...}, ...]}`, keyset-paged by `after`/`limit`.
    fn list_names_paged(&self, collection: &str, page: usize) -> Result<Vec<String>, String> {
        let mut names = Vec::new();
        let mut after: Option<String> = None;
        loop {
            let limit = page.to_string();
            let mut query: Vec<(&str, &str)> = vec![("limit", limit.as_str())];
            if let Some(cursor) = after.as_deref() {
                query.push(("after", cursor));
            }
            let body = self.get_with_query(&[collection], &query)?;
            let items = body[collection]
                .as_array()
                .ok_or_else(|| format!("{}: not a taguru {collection} page", self.base))?;
            let mut page_names = Vec::with_capacity(items.len());
            for item in items {
                let name = item["name"].as_str().ok_or_else(|| {
                    format!("{}: a {collection} entry is missing its name", self.base)
                })?;
                page_names.push(name.to_string());
            }
            let page_len = page_names.len();
            // Checked against the FIRST name, not the last: a page is
            // internally sorted ascending (the server keyset-pages a
            // name-sorted directory), so first > cursor already proves
            // every name in the page cleared it. Checking only the
            // last name would miss a page that partially overlaps the
            // previous one (cursor "b", page ["a", "c"]) — first > cursor
            // fails there, catching the would-be duplicate "a".
            if let (Some(cursor), Some(first)) = (after.as_deref(), page_names.first())
                && first.as_str() <= cursor
            {
                return Err(format!(
                    "the server's {collection} page did not advance past '{cursor}'"
                ));
            }
            after = page_names.last().cloned();
            names.extend(page_names);
            if page_len < page || after.is_none() {
                break;
            }
        }
        Ok(names)
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

    /// One `GET /health`, formatted as a version-skew warning. `None`
    /// on a transport error, a non-200 status, or a body that carries
    /// nothing comparable — the verb's own first request reproduces
    /// any real fault with a better message than a guess made here
    /// would. Runs on its own short timeout rather than `Api`'s 35s
    /// request budget: this is a best-effort preflight, not the real
    /// request, and a stalled or unreachable server should cost it a
    /// few seconds, not most of a minute, before the caller's actual
    /// request gets a turn.
    fn version_skew_line(&self, verb: &str) -> Option<String> {
        let url = self.url(&["health"]).ok()?;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(HEALTH_PREFLIGHT_TIMEOUT))
            .http_status_as_error(false)
            .build()
            .into();
        let mut response = self.bearer(agent.get(&url)).call().ok()?;
        if response.status().as_u16() != 200 {
            return None;
        }
        let body = response.body_mut().read_to_string().ok()?;
        skew_warning(verb, &self.base, env!("CARGO_PKG_VERSION"), &body)
    }

    /// The ADR 0002 §10 preflight for the dual-mode verbs: one
    /// `/health` read, one stderr line when the server's minor version
    /// differs from this CLI's — never a blocker, since a replica
    /// mid-rollout legitimately runs a different minor than its
    /// writer for a while. Wired in by `export --url` (#245) and
    /// `compact --url` (#246); `import --url` (#247) calls the same
    /// method.
    pub(crate) fn warn_on_version_skew(&self, verb: &str) {
        if let Some(line) = self.version_skew_line(verb) {
            eprintln!("{line}");
        }
    }
}

/// The `(major, minor)` prefix of a version string; `None` when it
/// does not lead with two dot-separated integers. A pre-release
/// suffix on the second component (`"0.5.0-rc.1"`) still parses —
/// only the first two dot-separated pieces are read.
fn major_minor(version: &str) -> Option<(u64, u64)> {
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// The one-line warning a version skew earns; `None` when there is
/// nothing trustworthy to compare. `health_body` is `/health`'s raw
/// 200 body: a JSON object carrying `version` compares major.minor
/// against this CLI's own; the literal text `ok` names a pre-0.5
/// server — skewed by definition, since this warning mechanism first
/// ships in 0.5 — and anything else (no `version` field, an
/// unparseable version, a body that isn't JSON or `ok`) is skipped
/// rather than guessed at.
fn skew_warning(verb: &str, base: &str, cli: &str, health_body: &str) -> Option<String> {
    let trimmed = health_body.trim();
    if trimmed == "ok" {
        return Some(format!(
            "taguru: {verb}: warning: this CLI is {cli} but the server at {base} answers \
             /health as a pre-0.5 server — minor versions differ, and pre-1.0 a minor may \
             change response shapes; continuing anyway"
        ));
    }
    let server: Value = serde_json::from_str(trimmed).ok()?;
    let server_version = server["version"].as_str()?;
    if major_minor(cli)? == major_minor(server_version)? {
        return None;
    }
    Some(format!(
        "taguru: {verb}: warning: this CLI is {cli} but the server at {base} runs \
         {server_version} — minor versions differ, and pre-1.0 a minor may change response \
         shapes; continuing anyway"
    ))
}

/// Formats a non-2xx response the way the server would want it read:
/// its own `error` field if the body carries one, otherwise the
/// trimmed body itself, plus `Retry-After` when the response carried
/// one — a heavy-ops or maintenance shed (503) and a rate limit (429)
/// both set it (`src/limits.rs`). ADR 0002 §8: displayed for the
/// operator to act on, never acted on by the client itself, so the
/// value is shown exactly as the server sent it (delta-seconds or an
/// HTTP-date, unparsed either way). Shared by [`finish`] and
/// [`Api::get_raw`] so the two error surfaces can't drift apart.
fn status_error(
    status: u16,
    parsed: &Value,
    text: &str,
    url: &str,
    retry_after: Option<&str>,
) -> String {
    let message = parsed["error"].as_str().unwrap_or(text.trim());
    match retry_after {
        Some(value) => format!("{url} answered {status}: {message} (Retry-After: {value})"),
        None => format!("{url} answered {status}: {message}"),
    }
}

/// The `Retry-After` header's raw value, when the response carried
/// one — read before the body so a caller free to consume the body
/// afterward without losing it.
fn retry_after_header(response: &ureq::http::Response<ureq::Body>) -> Option<String> {
    response
        .headers()
        .get(ureq::http::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
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
    let retry_after = retry_after_header(&response);
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|error| ApiFailure::Other(format!("{url}: unreadable response: {error}")))?;
    let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if status != 200 {
        let message = status_error(status, &parsed, &text, url, retry_after.as_deref());
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
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use serde_json::json;

    use std::sync::{Arc, Mutex};

    use super::{Api, reject_userinfo, skew_warning, token_from_ring};

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

    #[test]
    fn skew_warning_fires_only_on_a_real_minor_or_major_difference() {
        // same minor: nothing to say
        assert_eq!(
            skew_warning("import", "http://h", "0.4.0", r#"{"version":"0.4.0"}"#),
            None
        );
        // patch-only difference is still the same minor
        assert_eq!(
            skew_warning("import", "http://h", "0.4.0", r#"{"version":"0.4.9"}"#),
            None
        );
        // pre-release suffixes still parse down to (major, minor)
        assert_eq!(
            skew_warning("import", "http://h", "0.5.0-rc.1", r#"{"version":"0.5.0"}"#),
            None
        );

        // a minor difference warns, naming both versions, the base, and the verb
        let line = skew_warning("import", "http://h", "0.5.0", r#"{"version":"0.4.0"}"#)
            .expect("0.4 vs 0.5 is a minor skew");
        assert!(line.starts_with("taguru: import: warning:"), "{line}");
        assert!(line.contains("0.5.0") && line.contains("0.4.0") && line.contains("http://h"));

        // a major difference is a fortiori a skew
        assert!(skew_warning("import", "http://h", "1.4.0", r#"{"version":"0.4.0"}"#).is_some());
    }

    #[test]
    fn skew_warning_treats_a_bare_ok_body_as_a_pre_0_5_server() {
        let line = skew_warning("export", "http://h", "0.5.0", "ok")
            .expect("a bare \"ok\" body predates this warning mechanism");
        assert!(line.contains("pre-0.5 server"), "{line}");
        // trimmed the same way whether or not the body carries a newline
        assert_eq!(
            skew_warning("export", "http://h", "0.5.0", "ok\n"),
            Some(line)
        );
    }

    #[test]
    fn skew_warning_stays_silent_when_it_has_nothing_trustworthy_to_compare() {
        // JSON without a `version` field
        assert_eq!(
            skew_warning("compact", "http://h", "0.5.0", r#"{"status":"ok"}"#),
            None
        );
        // not JSON, and not the bare "ok" text either
        assert_eq!(
            skew_warning("compact", "http://h", "0.5.0", "garbage"),
            None
        );
        assert_eq!(skew_warning("compact", "http://h", "0.5.0", ""), None);
        // a version field that doesn't parse as (major, minor)
        assert_eq!(
            skew_warning("compact", "http://h", "0.5.0", r#"{"version":"unknown"}"#),
            None
        );
    }

    /// A minimal one-shot HTTP stub — bind, accept once, answer with a
    /// fixed status and JSON body, close. Enough to exercise
    /// `Api::get_raw`'s plumbing without a real taguru server.
    fn respond_once(status_line: &str, body: serde_json::Value) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let status_line = status_line.to_string();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buffer = [0u8; 1024];
            let _ = stream.read(&mut buffer); // discard the request itself
            let body = body.to_string();
            let response = format!(
                "{status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        });
        format!("http://{addr}")
    }

    /// [`respond_once`], with extra header lines spliced into the
    /// response — the only way to exercise a `Retry-After` reading
    /// without a real taguru server in front of the client.
    fn respond_once_with_headers(
        status_line: &str,
        extra_headers: &str,
        body: serde_json::Value,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let status_line = status_line.to_string();
        let extra_headers = extra_headers.to_string();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buffer = [0u8; 1024];
            let _ = stream.read(&mut buffer); // discard the request itself
            let body = body.to_string();
            let response = format!(
                "{status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n{extra_headers}\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        });
        format!("http://{addr}")
    }

    /// A scripted multi-response HTTP stub — bind, accept one
    /// connection per entry in order, answer each with its own fixed
    /// status and JSON body, close, then move to the next. Lets a test
    /// exercise a client that issues several requests in sequence
    /// (`list_names_paged`'s keyset walk) against one fake server, and
    /// hands back each request's start line so the test can assert on
    /// the query string the client actually sent.
    fn respond_in_order_capturing(
        responses: Vec<(&'static str, serde_json::Value)>,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = Arc::clone(&requests);
        std::thread::spawn(move || {
            for (status_line, body) in responses {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut buffer = [0u8; 2048];
                let read = stream.read(&mut buffer).unwrap_or(0);
                let request = String::from_utf8_lossy(&buffer[..read]);
                let start_line = request.lines().next().unwrap_or("").to_string();
                requests_clone.lock().unwrap().push(start_line);
                let body = body.to_string();
                let response = format!(
                    "{status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (format!("http://{addr}"), requests)
    }

    #[test]
    fn list_names_walks_after_cursors_until_a_short_page() {
        let (base, requests) = respond_in_order_capturing(vec![
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 5, "contexts": [{"name": "a"}, {"name": "b"}]}}),
            ),
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 5, "contexts": [{"name": "c"}, {"name": "d"}]}}),
            ),
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 5, "contexts": [{"name": "e"}]}}),
            ),
        ]);
        let api = Api::new(base);
        let names = api
            .list_names_paged("contexts", 2)
            .expect("three pages should walk cleanly to a short-page end");
        assert_eq!(names, vec!["a", "b", "c", "d", "e"]);

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3, "{requests:?}");
        assert!(
            requests[0].contains("limit=2") && !requests[0].contains("after="),
            "{}",
            requests[0]
        );
        assert!(requests[1].contains("after=b"), "{}", requests[1]);
        assert!(requests[2].contains("after=d"), "{}", requests[2]);
    }

    #[test]
    fn list_names_refuses_a_page_that_does_not_advance() {
        let (base, _requests) = respond_in_order_capturing(vec![
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 4, "contexts": [{"name": "a"}, {"name": "b"}]}}),
            ),
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 4, "contexts": [{"name": "a"}, {"name": "b"}]}}),
            ),
        ]);
        let api = Api::new(base);
        let error = api
            .list_names_paged("contexts", 2)
            .expect_err("a page that repeats the previous cursor must not loop forever");
        assert!(error.contains("did not advance"), "{error}");
    }

    #[test]
    fn list_names_refuses_a_page_without_names() {
        let base = respond_once(
            "HTTP/1.1 200 OK",
            json!({"result": {"total": 1, "contexts": [{"description": "no name field"}]}}),
        );
        let api = Api::new(base);
        let error = api
            .list_names_paged("contexts", 10)
            .expect_err("a listing entry without a name must not be silently skipped");
        assert!(error.contains("missing its name"), "{error}");
    }

    #[test]
    fn an_empty_collection_lists_as_no_names() {
        let base = respond_once(
            "HTTP/1.1 200 OK",
            json!({"result": {"total": 0, "groups": []}}),
        );
        let api = Api::new(base);
        assert_eq!(api.list_names("groups"), Ok(Vec::new()));
    }

    #[test]
    fn a_userinfo_url_is_rejected_and_a_bare_host_is_not() {
        assert!(reject_userinfo("https://user:tok@h").is_err());
        // password-only userinfo (`https://:tok@h`) is still userinfo
        assert!(reject_userinfo("https://:tok@h").is_err());
        assert!(reject_userinfo("https://h").is_ok());
        // left for `Api::url`'s own "not a usable base URL" message
        assert!(reject_userinfo("not a url").is_ok());
    }

    #[test]
    fn version_skew_line_reads_one_health_request_over_http() {
        let base = respond_once(
            "HTTP/1.1 200 OK",
            json!({"status": "ok", "version": "0.1.0"}),
        );
        let api = Api::new(base.clone());
        let line = api
            .version_skew_line("import")
            .expect("0.1.0 differs from this build's own version");
        assert!(line.contains("0.1.0"), "{line}");
        assert!(line.contains(&base), "{line}");
    }

    #[test]
    fn a_shed_response_displays_its_retry_after_header() {
        let base = respond_once_with_headers(
            "HTTP/1.1 503 Service Unavailable",
            "retry-after: 1\r\n",
            json!({"status": "error", "code": "overloaded", "error": "server is at its heavy-operation ceiling"}),
        );
        let api = Api::new(base);
        let error = api
            .get_raw(&["contexts", "sake", "export"])
            .expect_err("503 must not be read as success");
        assert!(error.contains("503"), "{error}");
        assert!(error.contains("(Retry-After: 1)"), "{error}");
    }

    #[test]
    fn a_response_without_the_header_carries_no_retry_after_suffix() {
        let base = respond_once(
            "HTTP/1.1 404 Not Found",
            json!({"status": "error", "code": "no_context", "error": "no such context"}),
        );
        let api = Api::new(base);
        let error = api
            .get_raw(&["contexts", "nope", "export"])
            .expect_err("404 must not be read as success");
        assert!(!error.contains("Retry-After"), "{error}");
    }

    #[test]
    fn version_skew_line_is_none_on_a_non_200_health_response() {
        let base = respond_once(
            "HTTP/1.1 503 Service Unavailable",
            json!({"status": "error", "code": "unhealthy"}),
        );
        let api = Api::new(base);
        assert_eq!(api.version_skew_line("import"), None);
    }
}
