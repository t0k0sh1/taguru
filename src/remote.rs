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
/// here instead of guessing at the server's default.
const LIST_PAGE_LIMIT: usize = 1000;

/// Every response this module reads is capped here, explicitly —
/// `ureq::Body::read_to_string()`'s own default (10 MiB, wired into
/// its `MAX_BODY_SIZE`) is otherwise silent and easy to forget about.
/// An envelope response (an association batch, an error body) is
/// small; `export --url`'s raw stream (`Api::get_raw`) is the one
/// exception — a whole context in one response, with no server-side
/// ceiling on how large a context may grow — so this is sized as a
/// generous safety net against a runaway/malicious server rather than
/// a real business limit. `BodyExceedsLimit` (not a silent truncation)
/// is what a caller sees past it.
const REMOTE_RESPONSE_CAP_BYTES: u64 = 256 * 1024 * 1024;

/// Import batches are packed into request bodies up to this size —
/// comfortably under the server's default 8 MiB body cap.
pub(crate) const IMPORT_CHUNK_BYTES: usize = 2 * 1024 * 1024;

/// Packs whole rendered batches into `POST /import` bodies under
/// [`IMPORT_CHUNK_BYTES`] — a batch is the import format's atom and
/// never splits (ADR 0002 §9), so one batch alone LARGER than the
/// ceiling travels as its own over-sized chunk: the server's body cap
/// (8 MiB default, four times this ceiling) is the authority that
/// refuses it, not a client-side split that would sever the batch's
/// retract-then-apply record set. One packer for `taguru communities`
/// and `taguru consolidation` (issue #752); `import --url`'s own
/// `split_batches` stays separate, slicing request bodies straight
/// out of source-file bytes instead of joining rendered strings.
pub(crate) fn pack_import_chunks(batches: &[String]) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for batch in batches {
        if !current.is_empty() && current.len() + batch.len() + 1 > IMPORT_CHUNK_BYTES {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(batch);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// ADR 0002 §7: a URL carrying userinfo (`https://user:token@host/`)
/// is refused — credentials on the command line are readable from
/// `ps` and shell history for the lifetime of the terminal, the exact
/// leak the existing `TAGURU_API_TOKEN` environment variable already
/// avoids. An unparseable `base` is left alone here: that is
/// [`reject_unusable_base`]'s job, so one fault keeps one error
/// surface.
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

/// Refuses a base URL no request could ever leave on: unparseable, or
/// a scheme `ureq` (the transport underneath [`Api`]) does not speak.
/// `url::Url::parse` alone happily accepts `file://`/`ftp://` and
/// anything else with a well-formed authority, but a non-http(s)
/// scheme would only fail once the request reaches `ureq` — as a
/// transport error (exit 1), exactly the network-problem-shaped
/// message an upfront check exists to avoid for a usage mistake.
/// Every verb entry point calls this before printing anything or
/// touching any file, and maps the message to a usage error (exit 2),
/// the same #248-item-8 pattern `calibrate`/`communities`/`evaluate`/
/// `benchmark search` follow.
pub(crate) fn reject_unusable_base(base: &str) -> Result<(), String> {
    match url::Url::parse(base) {
        Ok(parsed) if matches!(parsed.scheme(), "http" | "https") => Ok(()),
        Ok(parsed) => Err(format!(
            "'{base}' uses '{}', but --url only supports http/https",
            parsed.scheme()
        )),
        Err(_) => Err(format!("'{base}' is not a usable base URL")),
    }
}

/// A failure from the envelope surface, with 404 told apart — "no
/// artifact yet" is a first-run state, not an error.
pub(crate) enum ApiFailure {
    /// `code` is the envelope's own stable `ErrorCode` string
    /// (`src/api.rs`'s wire vocabulary) when the body carried one.
    /// `evaluate`'s citation locator check (#275) needs to tell
    /// `no_source` apart from `no_paragraph` — both answer 404
    /// (`src/api.rs:199-203`), so a caller reading only `message` cannot
    /// distinguish them without fragile prose matching.
    NotFound {
        code: Option<String>,
        message: String,
    },
    Other(String),
}

impl ApiFailure {
    pub(crate) fn into_message(self) -> String {
        match self {
            ApiFailure::NotFound { message, .. } | ApiFailure::Other(message) => message,
        }
    }
}

/// One `POST /import` chunk's failure, with the shapes `import --url`
/// (#247) must tell apart, ADR 0002 §9: a 413 is a pre-application
/// refusal safe to adapt to (halve and resend); a transport failure
/// leaves which chunk landed ambiguous (§8's "connection lost after
/// chunk N/M" message); a malformed base URL means no request was
/// ever attempted, so it must not read like a lost connection either;
/// anything else is the server's own refusal, whose envelope may
/// carry `integrity`/`durable_batches` (issue #182) worth surfacing
/// verbatim.
#[cfg_attr(test, derive(Debug))]
pub(crate) enum ImportFailure {
    /// 413 — refused before applying anything (src/api/import.rs:371),
    /// which is what makes halving the chunk and resending safe.
    TooLarge(String),
    /// The request itself did not complete — which chunk (if any)
    /// landed is unknown, not merely "assume it failed."
    Transport(String),
    /// The URL could never be built (a malformed base) — no request
    /// was ever attempted, so this must not read to the caller as a
    /// lost connection with some ambiguous chunk left to resume.
    InvalidUrl(String),
    /// Any other non-200: the parsed envelope rides along so the
    /// caller can read `code`/`integrity`/`durable_batches` without a
    /// second round trip.
    Refused {
        status: u16,
        message: String,
        body: Value,
    },
}

impl ImportFailure {
    pub(crate) fn into_message(self) -> String {
        match self {
            Self::TooLarge(message) | Self::Transport(message) | Self::InvalidUrl(message) => {
                message
            }
            Self::Refused { message, .. } => message,
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
            .with_config()
            .limit(REMOTE_RESPONSE_CAP_BYTES)
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
    ///
    /// `pub(crate)`: `evaluate` (#273) keyset-walks
    /// `GET /contexts/{name}/sources` itself — its `limit`/`after` shape
    /// does not fit [`Api::list_names`], which assumes each page item is
    /// an object carrying a `name` field; `sources` pages a bare string
    /// array instead. No new HTTP client code, per ADR 0004 §11.
    pub(crate) fn get_with_query(
        &self,
        segments: &[&str],
        query: &[(&str, &str)],
    ) -> Result<Value, String> {
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
    /// test can walk several pages (including a short-but-nonempty
    /// page that must NOT end the walk, only an empty one does — and
    /// the non-advancing-cursor guard) without provisioning thousands
    /// of contexts to cross the real ceiling.
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
            // Only an EMPTY page marks the end — `after.is_none()` is
            // exactly that condition, since a non-empty page always
            // leaves a last name behind. A page shorter than `limit`
            // is not a signal to stop: the server's own contract
            // (`directory_page`) allows a short-but-nonempty page when
            // a delete races the seek, and re-seeks internally only
            // when the whole window comes up empty — so a short page
            // here can still have more names after it. Stopping on
            // `page_len < page` silently truncated the walk on that
            // race; `export --url`'s enumeration must not drop names.
            if page_len == 0 {
                break;
            }
        }
        Ok(names)
    }

    /// Every row of `GET /contexts`, keyset-walked the same way
    /// [`Api::list_names`] walks `"contexts"`/`"groups"` but decoded
    /// whole — `compact --dry-run --url` (issue #371) needs each row's
    /// `stats` (`dead_edges`, `associations`, `arena_slack`, …), which
    /// a bare name throws away. `GET /contexts` already carries them
    /// per row, so no new server surface is required.
    ///
    /// Contexts-only: `GET /groups` rows don't decode as
    /// [`crate::registry::DirectoryEntry`] (no `stats`/`usage`), so
    /// this has no `collection` parameter the way `list_names` does —
    /// a caller enumerating groups keeps using `list_names("groups")`.
    pub(crate) fn list_context_entries(
        &self,
    ) -> Result<Vec<crate::registry::DirectoryEntry>, String> {
        self.list_context_entries_paged(LIST_PAGE_LIMIT)
    }

    /// [`Api::list_context_entries`] with an explicit page size — same
    /// split-out-for-testing reason as [`Api::list_names_paged`].
    fn list_context_entries_paged(
        &self,
        page: usize,
    ) -> Result<Vec<crate::registry::DirectoryEntry>, String> {
        let mut rows = Vec::new();
        let mut after: Option<String> = None;
        loop {
            let limit = page.to_string();
            let mut query: Vec<(&str, &str)> = vec![("limit", limit.as_str())];
            if let Some(cursor) = after.as_deref() {
                query.push(("after", cursor));
            }
            let body = self.get_with_query(&["contexts"], &query)?;
            let items = body["contexts"]
                .as_array()
                .ok_or_else(|| format!("{}: not a taguru contexts page", self.base))?;
            let mut page_rows = Vec::with_capacity(items.len());
            for item in items {
                let entry: crate::registry::DirectoryEntry = serde_json::from_value(item.clone())
                    .map_err(|error| {
                    format!("{}: a contexts entry did not decode: {error}", self.base)
                })?;
                page_rows.push(entry);
            }
            let page_len = page_rows.len();
            // Same first-name-vs-cursor advancement check
            // `list_names_paged` runs — see its comment for why the
            // first name, not the last, is the one compared.
            if let (Some(cursor), Some(first)) = (after.as_deref(), page_rows.first())
                && first.name.as_str() <= cursor
            {
                return Err(format!(
                    "the server's contexts page did not advance past '{cursor}'"
                ));
            }
            after = page_rows.last().map(|entry| entry.name.clone());
            rows.extend(page_rows);
            // Only an EMPTY page marks the end — see
            // `list_names_paged`'s comment for why a short-but-nonempty
            // page must not stop the walk.
            if page_len == 0 {
                break;
            }
        }
        Ok(rows)
    }

    /// One `POST /import` request carrying a pack of whole batches —
    /// the coarse view for callers that don't need to tell a 413 apart
    /// from any other refusal. `import --url` (#247) uses
    /// [`Api::import_chunk`] instead, which does.
    pub(crate) fn import(&self, stream: &str) -> Result<(), String> {
        self.import_chunk(stream, false)
            .map(|_| ())
            .map_err(ImportFailure::into_message)
    }

    /// One `POST /import` request, `?dry_run=true` appended when
    /// `dry_run` is set — the structured view `import --url` (#247)
    /// needs to adapt to a 413 (halve and resend, ADR 0002 §9) and to
    /// report a mid-stream refusal's `integrity`/`durable_batches`
    /// envelope fields, neither of which the message-only [`Api::import`]
    /// can distinguish.
    pub(crate) fn import_chunk(&self, stream: &str, dry_run: bool) -> Result<Value, ImportFailure> {
        let query: &[(&str, &str)] = if dry_run { &[("dry_run", "true")] } else { &[] };
        let url = self
            .url_with_query(&["import"], query)
            .map_err(ImportFailure::InvalidUrl)?;
        // `Expect: 100-continue` (RFC 9110 §10.1.1): ureq waits up to 1s
        // for either "100 Continue" or a final status before writing the
        // body at all. A chunk can be a few MiB, and every real-world
        // "body too large" front door (nginx `client_max_body_size`, an
        // ALB) answers a too-large `Content-Length` with 413 during that
        // wait, closing the connection without ever reading the body —
        // which is exactly what makes the 413 safe to adapt to (ADR 0002
        // §9). Without this header the write races the front door's
        // rejection: the body starts flowing immediately, the 413 lands
        // mid-write, and the client sees a bare connection reset instead
        // of the 413 it was always going to get. A server that ignores
        // the header (or answers 100-continue right away, as this
        // server's own axum stack does) costs at most that 1s wait, once
        // per chunk — `import --url` is a deliberate, infrequent CLI
        // invocation, not a hot path.
        let request = self.bearer(
            self.agent
                .post(&url)
                .header("Content-Type", "application/x-ndjson")
                .header("Expect", "100-continue"),
        );
        let mut response = request
            .send(stream)
            .map_err(|error| ImportFailure::Transport(format!("{url}: {error}")))?;
        let status = response.status().as_u16();
        let retry_after = retry_after_header(&response);
        let text = response
            .body_mut()
            .with_config()
            .limit(REMOTE_RESPONSE_CAP_BYTES)
            .read_to_string()
            .map_err(|error| {
                ImportFailure::Transport(format!("{url}: unreadable response: {error}"))
            })?;
        let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if status == 413 {
            return Err(ImportFailure::TooLarge(status_error(
                status,
                &parsed,
                &text,
                &url,
                retry_after.as_deref(),
            )));
        }
        if status != 200 {
            return Err(ImportFailure::Refused {
                status,
                message: status_error(status, &parsed, &text, &url, retry_after.as_deref()),
                body: parsed,
            });
        }
        if !parsed
            .as_object()
            .is_some_and(|body| body.contains_key("result"))
        {
            return Err(ImportFailure::Refused {
                status,
                message: format!("{url}: not a taguru response: {}", text.trim()),
                body: parsed,
            });
        }
        Ok(parsed["result"].clone())
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
        let body = response
            .body_mut()
            .with_config()
            .limit(REMOTE_RESPONSE_CAP_BYTES)
            .read_to_string()
            .ok()?;
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

    /// One `GET /version`, read for its `schema_formats` array only —
    /// best-effort, same posture as [`Api::version_skew_line`] (its
    /// own short timeout, `None` on any transport/non-200/parse
    /// trouble so the verb's own request produces the real fault, not
    /// a guess made here). `/version` is auth-exempt (`PROBE_EXEMPT`),
    /// but the bearer rides along anyway — matches every other
    /// request this module sends.
    fn schema_formats(&self) -> Option<Vec<u64>> {
        let url = self.url(&["version"]).ok()?;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(HEALTH_PREFLIGHT_TIMEOUT))
            .http_status_as_error(false)
            .build()
            .into();
        let mut response = self.bearer(agent.get(&url)).call().ok()?;
        if response.status().as_u16() != 200 {
            return None;
        }
        let body = response
            .body_mut()
            .with_config()
            .limit(REMOTE_RESPONSE_CAP_BYTES)
            .read_to_string()
            .ok()?;
        let value: Value = serde_json::from_str(&body).ok()?;
        let formats = value.get("schema_formats")?.as_array()?;
        // All-or-nothing: `filter_map` would silently drop a malformed
        // element (e.g. a stray string in the array), which could turn
        // `[2, "oops"]` into a false `[2]` match this CLI reads as
        // "the server carries only format 2" instead of "this
        // capability array is malformed" — the latter must answer
        // `None` (best-effort, same as every other trouble case this
        // method already treats that way) rather than a guess built on
        // a partial read.
        formats.iter().map(serde_json::Value::as_u64).collect()
    }

    /// `import --url`'s ADR 0009 §13 preflight — call only when the
    /// payload actually carries a `taguru_schema` record; a
    /// schema-free import must behave exactly as it did before this
    /// method existed. `Some(message)` refuses before a byte ships.
    /// Unlike [`Api::schema_export_refusal`], an ABSENT
    /// `schema_formats` is fatal here, not safe: it means the peer has
    /// never heard of the key, so shipping the record would either be
    /// silently dropped by a schema-unaware server or fall through to
    /// `parse_stream`'s dispatch as a misleading "not a batch header"
    /// refusal (ADR 0009 §13 bullet 2) — this preflight exists to
    /// replace that with an honest one before the network round trip.
    pub(crate) fn schema_import_refusal(&self) -> Option<String> {
        let mine = crate::schema::SCHEMA_VERSION;
        match self.schema_formats() {
            Some(formats) if formats.contains(&mine) => None,
            Some(formats) => Some(format!(
                "taguru: import: this CLI writes schema format {mine} but the server at {} \
                 reads {formats:?} — nothing was sent; upgrade the server, or import \
                 without the taguru_schema record",
                self.base
            )),
            None => Some(format!(
                "taguru: import: this CLI writes schema format {mine} but the server at {} \
                 does not report a schema_formats — nothing was sent; upgrade the server, \
                 or import without the taguru_schema record",
                self.base
            )),
        }
    }

    /// `export --url`'s ADR 0009 §13 preflight, run unconditionally
    /// (unlike [`Api::schema_import_refusal`], which only runs when the
    /// payload is known to carry a schema record) — a fetch cannot
    /// know in advance whether the context it is about to pull carries
    /// one, and probing each context first would cost a request per
    /// context for no better an answer. An ABSENT `schema_formats` is
    /// SAFE here, not fatal: a server that has never heard of the key
    /// cannot have emitted a `taguru_schema` line, so there is nothing
    /// for this CLI to fail to read — only a format this CLI does not
    /// recognize refuses.
    pub(crate) fn schema_export_refusal(&self) -> Option<String> {
        let mine = crate::schema::SCHEMA_VERSION;
        match self.schema_formats() {
            Some(formats) if !formats.contains(&mine) => Some(format!(
                "taguru: export: this CLI reads schema format {mine} but the server at {} \
                 writes {formats:?} — nothing was fetched; upgrade this CLI",
                self.base
            )),
            _ => None,
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
        .with_config()
        .limit(REMOTE_RESPONSE_CAP_BYTES)
        .read_to_string()
        .map_err(|error| ApiFailure::Other(format!("{url}: unreadable response: {error}")))?;
    let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if status != 200 {
        let message = status_error(status, &parsed, &text, url, retry_after.as_deref());
        return Err(if status == 404 {
            ApiFailure::NotFound {
                code: parsed["code"].as_str().map(str::to_string),
                message,
            }
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

/// The URL a remote verb targets when none is given: TAGURU_ADDR,
/// with an unspecified bind address read as its loopback — 0.0.0.0 is
/// reachable at 127.0.0.1 from inside the same network namespace, and
/// inside the namespace is exactly where a HEALTHCHECK runs.
///
/// Errs when that resolves to port 0: TAGURU_ADDR documents port 0 as
/// "pick free" (the OS assigns an ephemeral port at bind time), but
/// that assignment is invisible to a second, later process — probing
/// port 0 itself always fails, reporting a healthy server unhealthy
/// forever rather than just once.
///
/// One rule for "which server does a CLI verb mean": `health`,
/// `calibrate`, `communities`, `evaluate`, and `benchmark` all resolve
/// their target through this. Lives here rather than in cli.rs — its
/// original home — because these callers are exactly this module's
/// clientele, and cli.rs is the verb dispatcher whose inclusion drags
/// every verb's module with it (the taguru-code dual-inclusion trim,
/// issue #443 item 3, is what moved it).
pub(crate) fn default_base_url() -> Result<String, String> {
    let addr = std::env::var("TAGURU_ADDR").unwrap_or_else(|_| "127.0.0.1:8248".to_string());
    base_url_for(&addr)
}

fn base_url_for(addr: &str) -> Result<String, String> {
    let loopback = loopback_of(addr);
    if loopback.ends_with(":0") {
        return Err(format!(
            "TAGURU_ADDR ({addr}) binds to port 0 (OS-assigned) — the actual port \
             can't be discovered from here; pass the server's real URL explicitly: \
             'taguru health http://host:PORT'"
        ));
    }
    Ok(format!("http://{loopback}"))
}

fn loopback_of(addr: &str) -> String {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    match addr.parse::<SocketAddr>() {
        Ok(mut socket) if socket.ip().is_unspecified() => {
            socket.set_ip(match socket.ip() {
                IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
                IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::LOCALHOST),
            });
            socket.to_string()
        }
        _ => addr.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use serde_json::json;

    use std::sync::{Arc, Mutex};

    use super::{
        Api, IMPORT_CHUNK_BYTES, ImportFailure, REMOTE_RESPONSE_CAP_BYTES, base_url_for,
        loopback_of, pack_import_chunks, reject_unusable_base, reject_userinfo, skew_warning,
        token_from_ring,
    };

    #[test]
    fn an_unspecified_bind_address_probes_via_loopback() {
        assert_eq!(loopback_of("0.0.0.0:8248"), "127.0.0.1:8248");
        assert_eq!(loopback_of("[::]:8248"), "[::1]:8248");
        assert_eq!(loopback_of("127.0.0.1:8248"), "127.0.0.1:8248");
        // A CONCRETE non-loopback address passes through untouched —
        // only the unspecified binds above are rewritten (#753).
        assert_eq!(loopback_of("192.168.1.5:8248"), "192.168.1.5:8248");
        // Hostnames don't parse as socket addresses; pass them through.
        assert_eq!(loopback_of("localhost:8248"), "localhost:8248");
    }

    #[test]
    fn a_port_0_bind_address_refuses_to_guess_the_real_port() {
        let error =
            base_url_for("0.0.0.0:0").expect_err("port 0 cannot resolve to a probeable URL");
        assert!(error.contains("port 0"), "{error}");
        // A concrete port still resolves normally.
        assert_eq!(
            base_url_for("0.0.0.0:8248").unwrap(),
            "http://127.0.0.1:8248"
        );
    }

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

    /// ADR 0009 §13's `import --url` preflight, over all four cases:
    /// a match, a mismatch, and the two directions absence cuts —
    /// fatal for `import` (a server that never heard of the key would
    /// either drop the record or answer `parse_stream`'s misleading
    /// "not a batch header" refusal), safe for `export` (such a
    /// server cannot have emitted a `taguru_schema` line to begin
    /// with).
    #[test]
    fn schema_import_refusal_covers_match_mismatch_and_absence() {
        let base = respond_once("HTTP/1.1 200 OK", json!({"schema_formats": [1]}));
        assert_eq!(Api::new(base).schema_import_refusal(), None);

        let base = respond_once("HTTP/1.1 200 OK", json!({"schema_formats": [2]}));
        let message = Api::new(base)
            .schema_import_refusal()
            .expect("a format this CLI cannot read must refuse");
        assert!(message.contains("reads [2]"), "{message}");
        assert!(message.contains("nothing was sent"), "{message}");

        let base = respond_once("HTTP/1.1 200 OK", json!({"status": "ok"}));
        let message = Api::new(base)
            .schema_import_refusal()
            .expect("an absent schema_formats is fatal for import");
        assert!(
            message.contains("does not report a schema_formats"),
            "{message}"
        );
    }

    /// A mixed-type `schema_formats` array (a malformed capability
    /// announcement — a stray string among the versions) must refuse
    /// as a whole, not silently drop the bad element and match on
    /// whatever numbers happened to parse: `[1, "oops"]` naming this
    /// CLI's own version must still be treated as "cannot trust this
    /// array" (absent-shaped fatal), never as "the server carries
    /// format 1."
    #[test]
    fn schema_formats_refuses_whole_on_a_malformed_element_rather_than_dropping_it() {
        let base = respond_once("HTTP/1.1 200 OK", json!({"schema_formats": [1, "oops"]}));
        let message = Api::new(base)
            .schema_import_refusal()
            .expect("a malformed schema_formats must refuse, not silently match");
        assert!(
            message.contains("does not report a schema_formats"),
            "{message}"
        );
    }

    /// [`schema_import_refusal_covers_match_mismatch_and_absence`]'s
    /// export twin — the one case that differs is absence, which is
    /// safe here rather than fatal.
    #[test]
    fn schema_export_refusal_covers_match_mismatch_and_absence() {
        let base = respond_once("HTTP/1.1 200 OK", json!({"schema_formats": [1]}));
        assert_eq!(Api::new(base).schema_export_refusal(), None);

        let base = respond_once("HTTP/1.1 200 OK", json!({"schema_formats": [2]}));
        let message = Api::new(base)
            .schema_export_refusal()
            .expect("a format this CLI cannot read must refuse");
        assert!(message.contains("writes [2]"), "{message}");
        assert!(message.contains("nothing was fetched"), "{message}");

        let base = respond_once("HTTP/1.1 200 OK", json!({"status": "ok"}));
        assert_eq!(
            Api::new(base).schema_export_refusal(),
            None,
            "an absent schema_formats (a pre-schema server) is safe for export"
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
    /// `extra_headers` must already end each line with its own
    /// `\r\n`; the splice adds none of its own.
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
    fn list_names_walks_after_cursors_until_an_empty_page() {
        let (base, requests) = respond_in_order_capturing(vec![
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 4, "contexts": [{"name": "a"}, {"name": "b"}]}}),
            ),
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 4, "contexts": [{"name": "c"}, {"name": "d"}]}}),
            ),
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 4, "contexts": []}}),
            ),
        ]);
        let api = Api::new(base);
        let names = api
            .list_names_paged("contexts", 2)
            .expect("three pages should walk cleanly to an empty-page end");
        assert_eq!(names, vec!["a", "b", "c", "d"]);

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

    /// The server's own contract (`registry::context_io::directory_page`)
    /// allows a page shorter than `limit` to come back non-empty — a
    /// delete racing the seek — without that meaning the directory is
    /// exhausted. Only an empty page marks the end; a short-but-nonempty
    /// one must not truncate the walk (this is what `page_len < page`
    /// used to do wrong).
    #[test]
    fn a_short_nonempty_page_does_not_end_the_walk() {
        let (base, requests) = respond_in_order_capturing(vec![
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 3, "contexts": [{"name": "a"}, {"name": "b"}]}}),
            ),
            // Short (1 < limit 2) but non-empty: a delete raced the
            // seek on the server side. The walk must keep going.
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 3, "contexts": [{"name": "c"}]}}),
            ),
            // Only this empty page is the real end.
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 3, "contexts": []}}),
            ),
        ]);
        let api = Api::new(base);
        let names = api
            .list_names_paged("contexts", 2)
            .expect("a short-but-nonempty page must not truncate enumeration");
        assert_eq!(names, vec!["a", "b", "c"]);

        let requests = requests.lock().unwrap();
        assert_eq!(
            requests.len(),
            3,
            "the short page at position 2 must not stop the walk: {requests:?}"
        );
        assert!(requests[2].contains("after=c"), "{}", requests[2]);
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
        // left for `reject_unusable_base`'s "not a usable base URL"
        assert!(reject_userinfo("not a url").is_ok());
    }

    /// Issue #753: the safety-net cap, pinned literally — a drifted
    /// constant would silently change what "runaway response" means.
    #[test]
    fn the_remote_response_cap_is_pinned_literally() {
        assert_eq!(REMOTE_RESPONSE_CAP_BYTES, 268_435_456);
    }

    /// Issue #753: the entries walk mirrors `list_names_paged`'s
    /// contract — a short-but-nonempty page continues (only an empty
    /// page ends the walk), and a page that does not advance past the
    /// cursor is refused rather than looping on duplicates.
    #[test]
    fn context_entry_listing_walks_pages_and_refuses_a_non_advancing_one() {
        fn row(name: &str) -> serde_json::Value {
            json!({
                "name": name, "description": "", "pinned": false, "loaded": false,
                "dice_floor": null, "semantic_floor": null, "stats": {}, "usage": {}
            })
        }
        let (base, _) = respond_in_order_capturing(vec![
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 3, "contexts": [row("a"), row("b")]}}),
            ),
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 3, "contexts": [row("c")]}}),
            ),
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 3, "contexts": []}}),
            ),
        ]);
        let api = Api::new(base);
        let names: Vec<String> = api
            .list_context_entries_paged(2)
            .expect("a short page continues the walk")
            .iter()
            .map(|entry| entry.name.clone())
            .collect();
        assert_eq!(names, ["a", "b", "c"]);

        let (base, _) = respond_in_order_capturing(vec![
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 2, "contexts": [row("a"), row("b")]}}),
            ),
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"total": 2, "contexts": [row("b")]}}),
            ),
        ]);
        let api = Api::new(base);
        let error = api
            .list_context_entries_paged(2)
            .expect_err("a stuck cursor must refuse, never loop");
        assert!(error.contains("did not advance past 'b'"), "{error}");
    }

    #[test]
    fn pack_import_chunks_never_splits_a_batch_and_packs_under_the_cap() {
        let small = "a".repeat(100);
        let big = "b".repeat(IMPORT_CHUNK_BYTES);
        let chunks = pack_import_chunks(&[small.clone(), big.clone(), small.clone()]);
        assert_eq!(chunks.len(), 3, "a cap-sized batch travels alone, unsplit");
        assert_eq!(chunks[1], big);

        // One batch alone LARGER than the ceiling still travels whole:
        // its chunk exceeds the ceiling on purpose — the server's body
        // cap is the authority that refuses it, never a client split.
        let over = "c".repeat(IMPORT_CHUNK_BYTES + 1);
        let chunks = pack_import_chunks(&[small.clone(), over.clone(), small.clone()]);
        assert_eq!(chunks, vec![small.clone(), over, small.clone()]);

        let chunks = pack_import_chunks(&[small.clone(), small.clone()]);
        assert_eq!(chunks.len(), 1, "two small batches share one chunk");
        assert_eq!(chunks[0], format!("{small}\n{small}"));

        assert!(pack_import_chunks(&[]).is_empty());
    }

    /// The byte budget accounts for the joining newline exactly: two
    /// batches that fit with it share a chunk, one byte more splits
    /// them — and the cap itself is pinned literally, so a drifted
    /// constant fails here rather than in production against a
    /// server's 8 MiB refusal.
    #[test]
    fn pack_import_chunks_counts_the_separator_against_a_pinned_cap() {
        assert_eq!(IMPORT_CHUNK_BYTES, 2_097_152);
        let first = "x".repeat(IMPORT_CHUNK_BYTES / 2);
        let exactly = "y".repeat(IMPORT_CHUNK_BYTES - first.len() - 1);
        let chunks = pack_import_chunks(&[first.clone(), exactly]);
        assert_eq!(chunks.len(), 1, "len + 1 + len == cap still fits");
        assert_eq!(chunks[0].len(), IMPORT_CHUNK_BYTES);

        let over = "y".repeat(IMPORT_CHUNK_BYTES - first.len());
        let chunks = pack_import_chunks(&[first.clone(), over.clone()]);
        assert_eq!(chunks, vec![first, over], "one byte more splits");
    }

    #[test]
    fn an_unusable_base_is_rejected_with_the_fault_named() {
        assert_eq!(reject_unusable_base("http://h"), Ok(()));
        assert_eq!(reject_unusable_base("https://h:8248/prefix"), Ok(()));
        let error = reject_unusable_base("not a url").expect_err("unparseable");
        assert_eq!(error, "'not a url' is not a usable base URL");
        let error = reject_unusable_base("ftp://h").expect_err("wrong scheme");
        assert_eq!(
            error,
            "'ftp://h' uses 'ftp', but --url only supports http/https"
        );
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

    /// [`a_shed_response_displays_its_retry_after_header`], but through
    /// the POST envelope path (`Api::post` → `finish`) instead of
    /// `get_raw` — the one `compact --url`'s per-context and sweep
    /// calls actually walk, so this is the path this PR's own new
    /// behavior needs proven, not just the pre-existing GET path.
    #[test]
    fn a_shed_envelope_response_displays_its_retry_after_header() {
        let base = respond_once_with_headers(
            "HTTP/1.1 503 Service Unavailable",
            "retry-after: 7\r\n",
            json!({"status": "error", "code": "overloaded", "error": "heavy-operation ceiling"}),
        );
        let api = Api::new(base);
        let error = api
            .post(&["contexts", "sake", "compact"], &json!({}))
            .expect_err("503 must not be read as success");
        assert!(error.contains("503"), "{error}");
        assert!(error.contains("(Retry-After: 7)"), "{error}");
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

    /// `import_chunk(stream, true)` appends `?dry_run=true` to the
    /// request line — `import --url --dry-run` (#247) depends on this
    /// to preview instead of applying.
    #[test]
    fn import_chunk_appends_dry_run_true_only_when_asked() {
        let (base, requests) = respond_in_order_capturing(vec![
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"batches": []}, "status": "ok"}),
            ),
            (
                "HTTP/1.1 200 OK",
                json!({"result": {"batches": []}, "status": "ok"}),
            ),
        ]);
        let api = Api::new(base);
        api.import_chunk("{}\n", false).expect("a 200 must decode");
        api.import_chunk("{}\n", true).expect("a 200 must decode");
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2, "{requests:?}");
        assert!(!requests[0].contains("dry_run"), "{}", requests[0]);
        assert!(requests[1].contains("dry_run=true"), "{}", requests[1]);
    }

    /// A 413 decodes as [`ImportFailure::TooLarge`] — the one status
    /// `import --url` treats as safe to adapt to (halve and resend,
    /// ADR 0002 §9) rather than a refusal to stop on.
    #[test]
    fn import_chunk_classifies_a_413_as_too_large() {
        let base = respond_once(
            "HTTP/1.1 413 Payload Too Large",
            json!({"status": "error", "code": "payload_too_large", "error": "too big"}),
        );
        let api = Api::new(base);
        match api.import_chunk("{}\n", false) {
            Err(ImportFailure::TooLarge(message)) => {
                assert!(message.contains("too big"), "{message}")
            }
            _ => panic!("expected TooLarge, got a different outcome"),
        }
    }

    /// Any other non-200 decodes as [`ImportFailure::Refused`], with
    /// the parsed envelope riding along — `import --url`'s mid-stream
    /// refusal report reads `integrity`/`durable_batches` straight off
    /// this body rather than re-parsing the message text.
    #[test]
    fn import_chunk_carries_the_envelope_body_on_a_refusal() {
        let base = respond_once(
            "HTTP/1.1 404 Not Found",
            json!({
                "status": "error",
                "code": "no_context",
                "error": "no such context",
                "integrity": "durable_prefix",
                "durable_batches": 2,
            }),
        );
        let api = Api::new(base);
        match api.import_chunk("{}\n", false) {
            Err(ImportFailure::Refused {
                status,
                message,
                body,
            }) => {
                assert_eq!(status, 404);
                assert!(message.contains("no such context"), "{message}");
                assert_eq!(body["integrity"], "durable_prefix");
                assert_eq!(body["durable_batches"], 2);
            }
            _ => panic!("expected Refused, got a different outcome"),
        }
    }

    /// [`a_shed_response_displays_its_retry_after_header`]'s twin
    /// through `import_chunk`: a 503 still surfaces `Retry-After` in
    /// the displayed message, whichever [`ImportFailure`] variant that
    /// status maps to.
    #[test]
    fn import_chunk_displays_retry_after_on_a_shed_response() {
        let base = respond_once_with_headers(
            "HTTP/1.1 503 Service Unavailable",
            "retry-after: 3\r\n",
            json!({"status": "error", "code": "overloaded", "error": "heavy-operation ceiling"}),
        );
        let api = Api::new(base);
        let message = api
            .import_chunk("{}\n", false)
            .expect_err("503 must not be read as success")
            .into_message();
        assert!(message.contains("503"), "{message}");
        assert!(message.contains("(Retry-After: 3)"), "{message}");
    }
}
