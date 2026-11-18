//! Bearer-token gate for the whole API. Credentials are NAMED keys —
//! the one-token spelling `TAGURU_API_TOKEN` (key name "default")
//! and/or a `TAGURU_API_TOKENS` keyring ("ci:tokA,laptop:tokB") — so
//! the access log can say WHICH caller, one leaked key dies without
//! rotating the others, and rotation itself is an overlap (add the
//! new key, move callers, drop the old).
//!
//! Authorization rides on top: `TAGURU_KEY_GRANTS` grants each key a
//! ROLE (read ⊂ write ⊂ admin) and optionally a `context` list, and
//! [`enforce_authorization`] holds every request — the in-process MCP
//! dispatch included — to that grant. A key the variable does not
//! name keeps the historical full grant (admin over every `context`),
//! so existing deployments change nothing by upgrading.
//!
//! The whole table hot-reloads ([`reload_keyring`] on SIGHUP,
//! [`reload_keyring_if_changed`] for the `--config` file watch in
//! main): rotation is a table swap, not a restart. Every gate reads
//! the ring through [`SharedKeyring`] and holds ONE snapshot per
//! request, so an in-flight request sees the old table or the new one
//! — never a torn one, and never a removed key re-graded against a
//! table it was not checked into.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{MatchedPath, Request, State};
use axum::http::{HeaderValue, Method, header};
use axum::middleware::Next;
use axum::response::Response;
use serde::Deserialize;
use subtle::ConstantTimeEq;

use crate::api;
use crate::registry::is_context_id;

/// Paths that answer without credentials. `/live` (liveness) and
/// `/health` (readiness — 503 while the write path is degraded) are
/// the orchestrator probes and must be reachable unconfigured.
/// `/metrics` carries only aggregates and route templates (no `context`
/// names, no content), and exempting it keeps scrape configs trivial.
/// `/version` (ADR 0005 §6) must answer before an SDK's compatibility
/// preflight has any credential to send, and carries nothing more
/// sensitive than `/metrics` does. The rate limiter and the in-flight
/// ceiling honor the same list: probes and scrapes must not starve
/// behind a chatty client.
pub(crate) const PROBE_EXEMPT: [&str; 4] = ["/health", "/live", "/metrics", "/version"];

/// What answers without a bearer token. With OAuth enabled, its
/// discovery and grant endpoints join the probes — they exist to
/// CREATE credentials — but unlike the probes they stay rate-limited.
///
/// Listed by exact path, not by a `/oauth/` prefix: this gate lives in
/// a different file from the router it guards (`oauth_http.rs`), so a
/// prefix match would silently exempt any future route added under
/// that path — an admin endpoint, say — unless its author remembered
/// to come look here first.
fn is_auth_exempt(path: &str, oauth_enabled: bool) -> bool {
    const OAUTH_EXEMPT: [&str; 6] = [
        "/oauth/register",
        "/oauth/authorize",
        "/oauth/token",
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-authorization-server",
    ];
    PROBE_EXEMPT.contains(&path) || (oauth_enabled && OAUTH_EXEMPT.contains(&path))
}

/// Strips the `Bearer` auth-scheme, RFC 7235 §2.1's case-insensitively
/// (`bearer`, `BEARER`, `Bearer` alike) — only the scheme, never the
/// token that follows it. The grammar is `auth-scheme 1*SP token68`:
/// one or MORE spaces (issue #731), so a client padding the gap still
/// authenticates; no space at all is not a Bearer credential.
fn strip_bearer_prefix(value: &str) -> Option<&str> {
    let (scheme, rest) = value.split_at_checked("Bearer".len())?;
    if !scheme.eq_ignore_ascii_case("Bearer") {
        return None;
    }
    let rest = rest.strip_prefix(' ')?;
    Some(rest.trim_start_matches(' '))
}

/// The credential a key name is billed and its grant resolved
/// against: itself, or — for an OAuth delegation, `"key@client"` —
/// the name before the last '@'. Every consumer of a key name that
/// isn't an exact keyring lookup (grant resolution, liveness, rate
/// limiting) must fall back
/// through this so a caller cannot dilute a limit meant for one key by
/// minting more delegated clients from it.
pub(crate) fn base_key(key_name: &str) -> &str {
    key_name.rsplit_once('@').map_or(key_name, |(base, _)| base)
}

/// The authenticated key's name, attached to the RESPONSE: the access
/// log middleware sits outside this one, so response extensions are
/// the channel that reaches it.
#[derive(Clone)]
pub struct AuthKey(pub Arc<str>);

/// What a key may do, ordered by inclusion: `Admin` ⊃ `Write` ⊃
/// `Read`. Read is the retrieval loop; Write adds the ingest loop
/// (create `contexts`, assert, store passages, heal aliases, retract
/// and re-sync its sources, refresh embeddings); Admin adds the
/// operator verbs (delete `contexts`, bulk import, flush).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Role {
    Read,
    Write,
    Admin,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::Read => "read",
            Role::Write => "write",
            Role::Admin => "admin",
        }
    }
}

/// One key's grant: its role, and the `contexts` it may touch, listed by
/// context ID (`None` = every `context`; #966 — a display name repeats
/// and changes under rename, an id does not). The default — and the grant of any key
/// `TAGURU_KEY_GRANTS` does not name — is exactly what every key
/// could do before grants existed: admin, everywhere.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyGrant {
    pub role: Role,
    pub contexts: Option<Arc<HashSet<String>>>,
}

impl Default for KeyGrant {
    fn default() -> Self {
        Self {
            role: Role::Admin,
            contexts: None,
        }
    }
}

impl KeyGrant {
    /// Whether the grant covers the `context` with this ID.
    pub fn allows_context(&self, id: &str) -> bool {
        self.contexts
            .as_ref()
            .is_none_or(|allowed| allowed.contains(id))
    }
}

/// One entry of the `TAGURU_KEY_GRANTS` JSON: `"read"` as shorthand,
/// or `{"role": "write", "contexts": ["<context id>"]}` in full.
#[derive(Deserialize)]
#[serde(untagged)]
enum GrantSpec {
    Role(String),
    Full {
        role: String,
        #[serde(default)]
        contexts: Option<Vec<String>>,
        /// Every other key, kept only to be refused by name: an unread
        /// key fails OPEN here — a `contexts` list hiding under a typo
        /// (`context`) or under a key grants never had (`groups`) would
        /// leave the grant on every `context`.
        #[serde(flatten)]
        unknown: std::collections::BTreeMap<String, serde_json::Value>,
    },
}

/// Longest key name `TAGURU_API_TOKENS` accepts, in bytes (ADR 0055).
const MAX_KEY_NAME_BYTES: usize = 64;

/// A key name is its id (ADR 0055): the operator's string, never one the
/// server mints, and it lands verbatim in access logs, audit lines, 403/429
/// texts and `oauth.json`'s `key@client` delegation identities. So it is
/// held to 1..=[`MAX_KEY_NAME_BYTES`] of `[A-Za-z0-9._-]`. The set also
/// keeps '@' out, which OAuth reserves: delegations act as "key@client",
/// and `grant_of` strips at the last '@' to inherit the underlying key's
/// grant, so a raw key named with '@' would silently inherit an unrelated
/// key's grant.
fn valid_key_name(name: &str) -> bool {
    (1..=MAX_KEY_NAME_BYTES).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

/// The configured credentials. Empty = auth disabled (development
/// mode, warned about loudly at boot).
pub struct Keyring {
    keys: Vec<(Arc<str>, String)>,
    grants: HashMap<String, KeyGrant>,
}

impl Keyring {
    /// Builds the ring from the two variables. Malformed input is a
    /// boot REFUSAL, not a skipped entry — a credential that silently
    /// failed to arm reads as an auth hole at the worst possible
    /// moment. Error texts name entries by position, never by content:
    /// a malformed entry is likely a token pasted without its name.
    pub fn parse(single: Option<String>, named: Option<String>) -> Result<Self, String> {
        let mut keys: Vec<(Arc<str>, String)> = Vec::new();
        if let Some(token) = single {
            if token.is_empty() {
                return Err("TAGURU_API_TOKEN is set but empty".to_string());
            }
            keys.push((Arc::from("default"), token));
        }
        if let Some(ring) = named {
            if ring.trim().is_empty() {
                return Err("TAGURU_API_TOKENS is set but holds no entries".to_string());
            }
            for (position, entry) in ring.split(',').enumerate() {
                let entry = entry.trim();
                if entry.is_empty() {
                    continue; // tolerate a trailing comma
                }
                let Some((name, token)) = entry.split_once(':') else {
                    return Err(format!(
                        "TAGURU_API_TOKENS entry #{} is not name:token",
                        position + 1
                    ));
                };
                let name = name.trim();
                let token = token.trim();
                if name.is_empty() || token.is_empty() {
                    return Err(format!(
                        "TAGURU_API_TOKENS entry #{} has an empty name or token",
                        position + 1
                    ));
                }
                // Named by position only: a name outside the set is as
                // likely a token pasted where the name belongs.
                if !valid_key_name(name) {
                    return Err(format!(
                        "TAGURU_API_TOKENS entry #{}: a key name is 1-{MAX_KEY_NAME_BYTES} \
                         characters of A-Z a-z 0-9 . _ - ('@' is reserved for OAuth \
                         delegation, 'key@client')",
                        position + 1
                    ));
                }
                if keys.iter().any(|(existing, _)| existing.as_ref() == name) {
                    return Err(format!("key name '{name}' is configured twice"));
                }
                // A duplicated token VALUE is refused too: `authenticate`
                // scans every key and keeps the last match, so two names
                // sharing one token would silently resolve to whichever
                // was listed last — its grant, its audit name — and
                // rotating the other name would revoke nothing. The
                // `default` key from TAGURU_API_TOKEN sits in `keys`
                // already, so a collision across the two variables is
                // caught by the same scan. Plain `==` is fine: this is
                // boot-time config validation, not a guess from a caller.
                if let Some((existing, _)) = keys.iter().find(|(_, existing)| existing == token) {
                    return Err(format!(
                        "key name '{name}' reuses the token of key name '{existing}'"
                    ));
                }
                keys.push((Arc::from(name), token.to_string()));
            }
            if keys.is_empty() {
                return Err("TAGURU_API_TOKENS is set but holds no entries".to_string());
            }
        }
        Ok(Self {
            keys,
            grants: HashMap::new(),
        })
    }

    /// Applies `TAGURU_KEY_GRANTS` — a JSON object mapping key names to
    /// grants: `{"ci": "read", "bot": {"role": "write", "contexts":
    /// ["<context id>"]}}`. Refusals are boot refusals, keyring-style: a
    /// grant naming no configured key is a typo that would silently guard
    /// nobody, an empty `contexts` list would grant nothing at all —
    /// omitting the field is how "every `context`" is said — an entry
    /// that is not a context id (a display name, say) would match no
    /// `context` ever, silently locking the key out, and a field the
    /// grant does not know would go unread and leave it on every
    /// `context`.
    pub fn apply_grants(&mut self, json: Option<&str>) -> Result<(), String> {
        let Some(json) = json else {
            return Ok(());
        };
        if json.trim().is_empty() {
            return Err("TAGURU_KEY_GRANTS is set but empty".to_string());
        }
        let raw: HashMap<String, GrantSpec> = serde_json::from_str(json).map_err(|error| {
            format!(
                "TAGURU_KEY_GRANTS is not the documented JSON shape \
                 ({{\"name\": \"role\" | {{\"role\": …, \"contexts\": […]}}}}): {error}"
            )
        })?;
        for (name, spec) in raw {
            if !self.keys.iter().any(|(key, _)| key.as_ref() == name) {
                return Err(format!(
                    "TAGURU_KEY_GRANTS names '{name}', which is no configured key"
                ));
            }
            let (role, contexts) = match spec {
                GrantSpec::Role(role) => (role, None),
                GrantSpec::Full {
                    role,
                    contexts,
                    unknown,
                } => {
                    if let Some(field) = unknown.keys().next() {
                        return Err(format!(
                            "TAGURU_KEY_GRANTS key '{name}': unknown field '{field}' (a grant \
                             takes role and contexts) — unread, it would leave the grant on \
                             every context"
                        ));
                    }
                    (role, contexts)
                }
            };
            let role = match role.as_str() {
                "read" => Role::Read,
                "write" => Role::Write,
                "admin" => Role::Admin,
                other => {
                    return Err(format!(
                        "TAGURU_KEY_GRANTS key '{name}': unknown role '{other}' \
                         (read, write, or admin)"
                    ));
                }
            };
            let contexts = match contexts {
                None => None,
                Some(list) if list.is_empty() => {
                    return Err(format!(
                        "TAGURU_KEY_GRANTS key '{name}': an empty contexts list grants \
                         nothing — omit the field to grant every context"
                    ));
                }
                Some(list) => {
                    if let Some(bad) = list.iter().find(|entry| !is_context_id(entry)) {
                        return Err(format!(
                            "TAGURU_KEY_GRANTS key '{name}': '{bad}' is not a context id: \
                             contexts takes lowercase hyphenated UUIDs — the id column of \
                             GET /contexts — not display names"
                        ));
                    }
                    Some(Arc::new(list.into_iter().collect()))
                }
            };
            self.grants.insert(name, KeyGrant { role, contexts });
        }
        Ok(())
    }

    /// The grant behind a key name. OAuth delegations act as
    /// "key@client", so the lookup falls back to the name before the
    /// last '@' — the delegation can never out-rank the key it wraps.
    pub fn grant_of(&self, key_name: &str) -> KeyGrant {
        if let Some(grant) = self.grants.get(key_name) {
            return grant.clone();
        }
        if let Some(grant) = self.grants.get(base_key(key_name)) {
            return grant.clone();
        }
        KeyGrant::default()
    }

    pub fn granted_key_count(&self) -> usize {
        self.grants.len()
    }

    pub fn is_disabled(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Constant-shape scan: EVERY key is compared even after a match,
    /// so response timing says nothing about which key (if any) a
    /// guess grazed. `pub(crate)`: the OAuth consent step validates
    /// the delegated key through the same scan.
    pub(crate) fn authenticate(&self, presented: &str) -> Option<Arc<str>> {
        let mut matched = None;
        for (name, token) in &self.keys {
            if bool::from(presented.as_bytes().ct_eq(token.as_bytes())) {
                matched = Some(Arc::clone(name));
            }
        }
        matched
    }

    /// Whether `key_name` is still a live credential — named directly,
    /// or (for an OAuth delegation, `"key@client"`) via its base key,
    /// the same fallback `grant_of` walks. OAuth mints an access token
    /// once and then persists it independently of the keyring
    /// (`data_dir/oauth.json`), so a key later dropped from
    /// `TAGURU_API_TOKENS` must retire every token delegated from it
    /// too — otherwise the stale delegation keeps authenticating and
    /// `grant_of` falls through to the default grant (admin,
    /// unrestricted) for a caller nobody configured anymore.
    pub(crate) fn recognizes(&self, key_name: &str) -> bool {
        let is_configured = |name: &str| self.keys.iter().any(|(key, _)| key.as_ref() == name);
        is_configured(key_name) || is_configured(base_key(key_name))
    }

    /// The name-level difference `next` introduces over `self` — the
    /// reload audit line's payload. Token bytes never leave this
    /// function: only which names appeared, vanished, changed bytes,
    /// or changed grant. `rotated` is its own list (independent of
    /// `regranted`) because the common Kubernetes rotation changes
    /// bytes under a stable name and would be invisible in a pure
    /// name diff. `regranted` compares the EFFECTIVE grant via
    /// `grant_of`, so a grant entry dropped from `TAGURU_KEY_GRANTS`
    /// counts too — that key falls back to the default admin
    /// grant, a real change worth a line.
    fn diff(&self, next: &Keyring) -> KeyringDiff {
        let mut diff = KeyringDiff::default();
        let old: HashMap<&str, &str> = self
            .keys
            .iter()
            .map(|(name, token)| (name.as_ref(), token.as_str()))
            .collect();
        for (name, token) in &next.keys {
            match old.get(name.as_ref()) {
                None => diff.added.push(name.to_string()),
                Some(previous) => {
                    if *previous != token.as_str() {
                        diff.rotated.push(name.to_string());
                    }
                    if self.grant_of(name) != next.grant_of(name) {
                        diff.regranted.push(name.to_string());
                    }
                }
            }
        }
        for (name, _) in &self.keys {
            if !next.keys.iter().any(|(new, _)| new == name) {
                diff.removed.push(name.to_string());
            }
        }
        for list in [
            &mut diff.added,
            &mut diff.removed,
            &mut diff.rotated,
            &mut diff.regranted,
        ] {
            list.sort_unstable();
        }
        diff
    }
}

/// What a reload changed, by key NAME only — never token bytes.
#[derive(Debug, Default, PartialEq)]
struct KeyringDiff {
    added: Vec<String>,
    removed: Vec<String>,
    rotated: Vec<String>,
    regranted: Vec<String>,
}

impl KeyringDiff {
    fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self.rotated.is_empty()
            && self.regranted.is_empty()
    }
}

/// The swappable handle every auth surface reads the keyring through:
/// a hot reload (SIGHUP, or the `--config` file watch) replaces the
/// inner `Arc` while requests in flight keep the snapshot they loaded
/// — the old table or the new one, never a torn or empty one. A load
/// is one uncontended read lock and an `Arc` clone; the write happens
/// only when an operator rotates keys.
#[derive(Clone)]
pub struct SharedKeyring {
    ring: Arc<parking_lot::RwLock<Arc<Keyring>>>,
    /// Serializes whole reloads — source read → parse → judge → swap
    /// — NOT request loads, and guards the digest of the config-file
    /// content the newest reload judged. Two triggers read the same
    /// file (SIGHUP and the config watch), and each reads it UNDER
    /// this lock: a read taken before it could see file state v1, lose
    /// the lock to a reload that arms the rewritten v2, then swap v1
    /// back in — re-arming every token v2 revoked. Reading under the
    /// lock makes each reload's view of the file at least as new as
    /// every swap ordered before it.
    ///
    /// The digest is the config watch's change detector, shared on
    /// purpose: SIGHUP records what it judged too. A watch-private
    /// digest goes stale the moment a SIGHUP arms other content — a
    /// file then reverted to what the watch last judged would read as
    /// unchanged, leaving the SIGHUP's table armed against a file that
    /// says otherwise, with nothing left to re-trigger. Refusals are
    /// recorded like applies: one loud line per change, not one per
    /// tick. `None` until boot (see
    /// [`SharedKeyring::with_config_digest`]) or a reload has judged
    /// any content.
    ///
    /// Held inside the reload functions so every trigger, future ones
    /// included, is serialized by construction rather than by caller
    /// discipline.
    reload_serial: Arc<parking_lot::Mutex<Option<String>>>,
}

impl SharedKeyring {
    /// A ring with no config file behind it, so nothing judged yet —
    /// the tests' shorthand (here and in other modules). Boot always
    /// goes through [`SharedKeyring::with_config_digest`].
    #[cfg(test)]
    pub fn new(keyring: Keyring) -> Self {
        Self::with_config_digest(keyring, None)
    }

    /// The shared handle over the keyring boot built. `config_digest`,
    /// the digest of the config-file bytes boot APPLIED (`None`
    /// without a config file), starts out as already judged (#942) —
    /// not a fresh read taken later, or a rotation landing between
    /// boot's read and the watch's first one would be recorded as
    /// judged without ever being applied.
    pub fn with_config_digest(keyring: Keyring, config_digest: Option<String>) -> Self {
        Self {
            ring: Arc::new(parking_lot::RwLock::new(Arc::new(keyring))),
            reload_serial: Arc::new(parking_lot::Mutex::new(config_digest)),
        }
    }

    /// The current table. Callers hold the snapshot for their whole
    /// operation: the bearer gate resolves authentication, the OAuth
    /// `recognizes` check, AND the grant from one load — re-loading
    /// between them would let a key a reload just removed fall
    /// through `grant_of` to the default admin grant.
    pub fn load(&self) -> Arc<Keyring> {
        Arc::clone(&self.ring.read())
    }

    fn store(&self, keyring: Arc<Keyring>) {
        *self.ring.write() = keyring;
    }
}

/// `TAGURU_KEY_GRANTS`' name before 0.10.0 (#959) — the one auth,
/// limit, or quota variable renamed or removed since 0.9.7. Its value
/// is never read, but its presence refuses boot and refuses a reload
/// (which keeps the previous table): a deployment that missed the
/// rename would otherwise arm no grant at all, and every key it meant
/// to restrict would fall back to the default grant — admin, on every
/// context.
pub const RETIRED_GRANTS_VAR: &str = "TAGURU_KEY_SCOPES";

/// Boot's and reload's shared refusal of [`RETIRED_GRANTS_VAR`]. `set`
/// is presence alone, an empty value included: a blank-templated
/// secret still says the deployment was written for the old name.
pub fn refuse_retired_grants_var(set: bool) -> Result<(), String> {
    if set {
        return Err(format!(
            "{RETIRED_GRANTS_VAR} is set, but 0.10.0 renamed it TAGURU_KEY_GRANTS (same value \
             shape) and no longer reads the old name: rename it, or unset it — ignoring it \
             would leave every key it restricts with admin rights on every context"
        ));
    }
    Ok(())
}

/// The variables the runtime reload re-reads — the whole auth table
/// and nothing else. The single-token spelling belongs here too: it
/// is the same table (key name "default"), and leaving it out would
/// keep single-token deployments restarting to rotate. The retired
/// grants name rides last, read only so a reload refuses it exactly
/// as boot does.
const AUTH_VARS: [&str; 4] = [
    "TAGURU_API_TOKEN",
    "TAGURU_API_TOKENS",
    "TAGURU_KEY_GRANTS",
    RETIRED_GRANTS_VAR,
];

/// Where each auth variable's CURRENT value comes from on a reload.
/// Captured in `main` BEFORE `config::load_config` folds the file
/// into the process environment — after that, file values and shell
/// values are indistinguishable (`set_var`), and re-reading the
/// environment would resurrect keys the operator already removed
/// from the file. The rule mirrors boot exactly: a variable set in
/// the real (shell) environment keeps winning — its value cannot
/// change while the process lives — and everything else follows the
/// file, absence included. Without a config file nothing can change
/// at runtime, and a reload is an explicit no-op.
pub struct AuthSource {
    config: Option<PathBuf>,
    from_shell: [bool; AUTH_VARS.len()],
}

impl AuthSource {
    /// Must run before the config file is applied to the environment
    /// — this is the only moment "set by the shell" is observable.
    pub fn capture(config: Option<PathBuf>) -> Self {
        Self {
            config,
            from_shell: AUTH_VARS.map(|var| std::env::var_os(var).is_some()),
        }
    }

    pub fn config_path(&self) -> Option<&Path> {
        self.config.as_deref()
    }

    /// The config file's current bytes, `None` without a config file.
    /// `Err` is a reload refusal: an unreadable file must keep the
    /// previous table, unlike boot where the same failure refuses to
    /// start. Raw bytes, not text, so a reload digests exactly what it
    /// judges.
    fn read_config(&self) -> Result<Option<Vec<u8>>, String> {
        let Some(path) = &self.config else {
            return Ok(None);
        };
        std::fs::read(path)
            .map(Some)
            .map_err(|error| format!("cannot read config {}: {error}", path.display()))
    }

    /// The auth values as `file` (the config file's bytes, from
    /// `read_config`) and the shell set them, in [`AUTH_VARS`] order.
    /// `Err` is a reload refusal, as for an unreadable file: a
    /// malformed one must keep the previous table too.
    fn table_from(&self, file: Option<&[u8]>) -> Result<[Option<String>; AUTH_VARS.len()], String> {
        let file: HashMap<String, String> = match file {
            None => HashMap::new(),
            Some(bytes) => {
                let path = self
                    .config
                    .as_deref()
                    .expect("file is Some only when self.config names the file it came from");
                let text = std::str::from_utf8(bytes).map_err(|error| {
                    format!("config {}: not valid UTF-8: {error}", path.display())
                })?;
                // Collecting in file order makes later duplicates win,
                // exactly as `load_config` applies them at boot.
                crate::config::parse_config(text)
                    .map_err(|message| format!("config {}: {message}", path.display()))?
                    .into_iter()
                    .collect()
            }
        };
        Ok(std::array::from_fn(|index| {
            if self.from_shell[index] {
                std::env::var(AUTH_VARS[index]).ok()
            } else {
                file.get(AUTH_VARS[index]).cloned()
            }
        }))
    }
}

/// What a reload attempt did — the caller turns this into counters;
/// the log lines are already written by the time it returns.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ReloadOutcome {
    /// A new table is armed; the audit line carries the diff.
    Applied,
    /// The sources parse clean and match the armed table exactly.
    Unchanged,
    /// The previous table stays armed; the error line says why.
    Refused,
}

/// Rebuilds the keyring from its current sources and swaps it in,
/// fail closed on every path: an unreadable or malformed source, and
/// the one transition that must be impossible — "tokens configured" →
/// "no tokens", which would silently reopen the server as the
/// unauthenticated localhost mode — all keep the previous table armed
/// and say so loudly. Arming keys where none were configured stays
/// allowed: that transition CLOSES the server. (It also preserves the
/// boot invariant "OAuth enabled ⇒ keys configured" across reloads,
/// since armed → empty is the only transition that could break it.)
///
/// Concurrent triggers (a SIGHUP racing the config watch) are
/// serialized end-to-end on [`SharedKeyring::reload_serial`], and the
/// source read happens UNDER that lock, not before it — see that field
/// for the stale swap this rules out. Request loads never touch this
/// lock. Unlike the watch, this never skips content already judged:
/// an operator who sent a SIGHUP gets a judgment either way. Blocking
/// for as long as the read takes, so main runs it off the async worker
/// (`read_off_worker_queued`), never on it: a stalled mount must wedge
/// neither a worker thread nor graceful shutdown.
pub fn reload_keyring(
    shared: &SharedKeyring,
    source: &AuthSource,
    trigger: &'static str,
) -> ReloadOutcome {
    let mut judged = shared.reload_serial.lock();
    let file = match source.read_config() {
        Ok(file) => file,
        Err(error) => return apply_reload(shared, Err(error), trigger),
    };
    if let Some(bytes) = &file {
        *judged = Some(crate::sha256::sha256_hex(bytes));
    }
    apply_reload(shared, source.table_from(file.as_deref()), trigger)
}

/// The `--config` file watch's tick: [`reload_keyring`], but only for
/// content no reload has judged yet — a content digest, not `(mtime,
/// len)` (#309), so a same-length rotation landing on an unchanged
/// mtime still registers — and an unreadable file is not a change:
/// transient volume states must neither trigger a reload nor poison
/// the digest. `None` when nothing was judged; `Some` is a reload that
/// ran, for the caller's counters.
///
/// Blocking, on purpose: the file is read while holding
/// [`SharedKeyring::reload_serial`], exactly as [`reload_keyring`]
/// reads it, so main runs this whole call on the watch's off-worker
/// thread (`read_off_worker`) instead of reading there and judging
/// afterwards. A read finished before the lock is a snapshot a SIGHUP
/// reload may already have superseded, and no check after the fact
/// can order the two: the superseding reload's own read may be older
/// or newer than it.
pub fn reload_keyring_if_changed(
    shared: &SharedKeyring,
    source: &AuthSource,
) -> Option<ReloadOutcome> {
    let mut judged = shared.reload_serial.lock();
    let Ok(Some(file)) = source.read_config() else {
        return None;
    };
    // SHA-256, not the FNV-1a fold `hash.rs` owns for change detection
    // elsewhere: rotation here gates a keyring reload, so the
    // signature needs collision resistance FNV-1a does not offer, not
    // just speed.
    let digest = crate::sha256::sha256_hex(&file);
    if judged.as_ref() == Some(&digest) {
        return None;
    }
    *judged = Some(digest);
    Some(apply_reload(
        shared,
        source.table_from(Some(file.as_slice())),
        "config-watch",
    ))
}

/// Parse → judge → swap. Callers MUST hold
/// [`SharedKeyring::reload_serial`] — and must have read `values`'
/// source under it, never before (see that field).
fn apply_reload(
    shared: &SharedKeyring,
    values: Result<[Option<String>; AUTH_VARS.len()], String>,
    trigger: &'static str,
) -> ReloadOutcome {
    let refused = |error: String| {
        tracing::error!(
            reload_error = %error,
            trigger,
            "keyring reload refused; the previous table stays armed"
        );
        ReloadOutcome::Refused
    };
    let [single, named, grants, retired_grants] = match values {
        Ok(values) => values,
        Err(error) => return refused(error),
    };
    if let Err(error) = refuse_retired_grants_var(retired_grants.is_some()) {
        return refused(error);
    }
    let mut next = match Keyring::parse(single, named) {
        Ok(next) => next,
        Err(error) => return refused(error),
    };
    if let Err(error) = next.apply_grants(grants.as_deref()) {
        return refused(error);
    }
    let current = shared.load();
    if next.is_disabled() && !current.is_disabled() {
        return refused(
            "the reloaded sources configure no key, but authentication is armed — refusing to \
             open the server (unset tokens mean the UNAUTHENTICATED localhost mode); restart if \
             that is really intended"
                .to_string(),
        );
    }
    let diff = current.diff(&next);
    if diff.is_empty() {
        // An operator who sent a SIGHUP deserves an answer either way:
        // silence here would read as a reload that never ran.
        tracing::info!(target: "taguru::audit", trigger, "keyring reload: no change");
        return ReloadOutcome::Unchanged;
    }
    let keys = next.key_count();
    let granted = next.granted_key_count();
    shared.store(Arc::new(next));
    tracing::info!(
        target: "taguru::audit",
        trigger,
        added = ?diff.added,
        removed = ?diff.removed,
        rotated = ?diff.rotated,
        regranted = ?diff.regranted,
        keys,
        granted,
        "keyring reloaded",
    );
    ReloadOutcome::Applied
}

/// Everything the bearer gate consults: the static keyring, the OAuth
/// subsystem when the operator enabled it, and a per-source-IP throttle
/// on FAILED attempts (`TAGURU_AUTH_FAIL_LIMIT_PER_MIN`).
///
/// What that throttle does NOT do: it is consulted only AFTER the
/// constant-time scan has judged the token, so it neither saves the
/// scan's CPU nor stops a guess from being graded — over budget, a
/// wrong token gets 429 + Retry-After instead of 401, but a correct
/// one still passes from the same IP. That ordering is deliberate:
/// checking the budget first would let anyone sharing the client's
/// source IP (a NAT, a proxy) lock the legitimate caller out just by
/// spending it. Brute-force resistance rests on token entropy; the
/// throttle only tells a misconfigured or hostile client to back off.
pub struct Gate {
    pub keyring: SharedKeyring,
    pub oauth: Option<Arc<crate::oauth::Oauth>>,
    pub fail_limiter: Arc<crate::limits::RateLimiter>,
}

/// Refuses any non-exempt request whose `Authorization: Bearer` token
/// matches no configured credential. An empty keyring disables the
/// gate entirely — development mode. Static keys open everything;
/// OAuth access tokens open `/mcp` only — the delegation is scoped to
/// the MCP surface, never the raw API. On success the credential's
/// NAME rides the response out to the access log.
pub async fn require_bearer(
    State(gate): State<Arc<Gate>>,
    request: Request,
    next: Next,
) -> Response {
    // ONE snapshot for the whole request: authentication, the OAuth
    // `recognizes` check, and the grant resolution below must agree
    // on one table. Under a mid-request reload, a second load between
    // them would let a key the reload just removed fall through
    // `grant_of` to the default admin grant — an elevation, not
    // just staleness.
    let keyring = gate.keyring.load();
    if keyring.is_disabled() {
        return next.run(request).await;
    }
    if is_auth_exempt(request.uri().path(), gate.oauth.is_some()) {
        return next.run(request).await;
    }

    let started_at = Instant::now();
    // RFC 7230 §3.2.2: Authorization is single-valued. Two copies are a
    // malformed request — and the shape of a smuggling attempt, where a
    // front proxy authenticates on one copy while this server reads the
    // other. Refuse outright rather than silently honor the first and
    // trust the proxy agreed on which one that was.
    let mut authorizations = request.headers().get_all(header::AUTHORIZATION).iter();
    let presented = match (authorizations.next(), authorizations.next()) {
        (Some(value), None) => value.to_str().ok().and_then(strip_bearer_prefix),
        _ => None,
    };
    let key = presented.and_then(|token| {
        keyring
            .authenticate(token)
            .or_else(|| match (&gate.oauth, request.uri().path()) {
                // Fail closed on a broken clock: treat every token as
                // already expired (u64::MAX can exceed no real
                // `expires_at`) rather than resurrecting the old
                // fail-open `now_secs() == 0` behavior.
                (Some(oauth), "/mcp") => oauth
                    .authenticate(token, crate::oauth::now_secs().unwrap_or(u64::MAX))
                    .filter(|key| keyring.recognizes(key)),
                _ => None,
            })
    });
    if let Some(key) = key {
        let mut request = request;
        // Inner layers (the rate limiter) key their work on WHO; the
        // response copy below feeds the access log on the way out.
        request.extensions_mut().insert(AuthKey(Arc::clone(&key)));
        // The grant, resolved from the SAME snapshot that just
        // authenticated the key, rides the request too:
        // `enforce_authorization` judges from this extension instead
        // of consulting a (possibly newer) keyring of its own, and
        // `remote_mcp` stamps it onto every dispatched tool call
        // alongside the key.
        request.extensions_mut().insert(keyring.grant_of(&key));
        let mut response = next.run(request).await;
        response.extensions_mut().insert(AuthKey(key));
        return response;
    }

    // No credential matched. Throttle repeated failures per source IP:
    // successful auth returned above without touching this limiter, so a
    // legitimate caller presenting the right token is never throttled —
    // not even after a neighbor on the same IP spent the budget. The
    // flip side (see `Gate`): the scan above already ran and already
    // graded this guess, so over budget a wrong token gets a 429 instead
    // of a 401, but the throttle neither saves that CPU nor stops a
    // correct guess from passing. Token entropy is the brute-force bar.
    if !gate.fail_limiter.is_disabled() {
        let peer = crate::limits::peer_ip(&request).unwrap_or_else(|| Arc::from("peer:unknown"));
        if let Err(retry_after) = gate.fail_limiter.admit(&peer, started_at) {
            return crate::limits::shed(
                api::ErrorCode::RateLimited,
                format!("too many failed authentication attempts — retry in {retry_after}s"),
                retry_after,
                started_at,
            );
        }
    }

    let mut response = api::error(
        api::ErrorCode::Unauthorized,
        "missing or invalid bearer token (send Authorization: Bearer <token>)",
        started_at,
    );
    // With OAuth on, the 401 names the discovery document — the hint
    // MCP clients key their authorization flow on (RFC 9728).
    let challenge = match &gate.oauth {
        Some(oauth) => HeaderValue::from_str(&format!(
            "Bearer resource_metadata=\"{}\"",
            oauth.resource_metadata_url()
        ))
        .unwrap_or_else(|_| HeaderValue::from_static("Bearer")),
        None => HeaderValue::from_static("Bearer"),
    };
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, challenge);
    response
}

/// The least role a (method, route template) demands. Fail closed: a
/// route this table does not classify demands Admin, so an endpoint
/// added without a classification locks down for granted keys instead
/// of leaking open. `/mcp` itself needs only Read — every tool call
/// dispatches onto a real route in-process and is judged there.
pub(crate) fn required_role(method: &Method, route: &str) -> Role {
    match (method, route) {
        // The retrieval loop, the directory, and the manual.
        (&Method::GET, "/contexts")
        | (&Method::GET, "/groups")
        | (&Method::GET, "/groups/{id}")
        | (&Method::GET, "/groups/{id}/export")
        | (&Method::GET, "/contexts/{id}")
        | (&Method::GET, "/contexts/{id}/labels")
        | (&Method::GET, "/contexts/{id}/schema")
        | (&Method::GET, "/contexts/{id}/aliases")
        | (&Method::GET, "/contexts/{id}/sources")
        | (&Method::GET, "/contexts/{id}/embeddings")
        | (&Method::GET, "/contexts/{id}/export")
        | (&Method::GET, "/protocol")
        | (&Method::POST, "/recall")
        | (&Method::POST, "/query")
        | (&Method::POST, "/sources/search")
        | (&Method::POST, "/contexts/{id}/recall")
        | (&Method::POST, "/contexts/{id}/query")
        | (&Method::POST, "/contexts/{id}/describe")
        | (&Method::POST, "/contexts/{id}/explore")
        | (&Method::POST, "/contexts/{id}/activate")
        | (&Method::POST, "/contexts/{id}/paths")
        | (&Method::GET, "/contexts/{id}/changes")
        | (&Method::POST, "/contexts/{id}/resolve")
        | (&Method::POST, "/contexts/{id}/resolve/explain")
        | (&Method::POST, "/contexts/{id}/resolve_label")
        | (&Method::POST, "/contexts/{id}/resolve_label/explain")
        | (&Method::POST, "/contexts/{id}/sources/lookup")
        | (&Method::POST, "/contexts/{id}/sources/search")
        | (&Method::POST, "/contexts/{id}/sources/search/explain")
        | (&Method::POST, "/contexts/{id}/citations")
        | (&Method::POST, "/contexts/{id}/unreachable_from")
        | (&Method::POST, "/contexts/{id}/vocabulary/audit")
        | (&Method::POST, "/contexts/{id}/drift/audit")
        | (&Method::POST, "/contexts/{id}/consolidation/audit")
        | (&Method::POST, "/contexts/{id}/schema/audit")
        | (&Method::POST, "/contexts/{id}/schema/validate")
        | (&Method::GET, "/contexts/{id}/communities")
        | (&Method::POST, "/contexts/{id}/communities/search")
        | (&Method::POST, "/contexts/{id}/evidence")
        | (&Method::POST, "/mcp") => Role::Read,
        // The ingest loop — everything the documented agent discipline
        // drives, context creation and per-source re-sync included.
        (&Method::POST, "/contexts")
        | (&Method::PATCH, "/contexts/{id}")
        | (&Method::PUT, "/contexts/{id}/schema")
        | (&Method::POST, "/groups")
        | (&Method::PATCH, "/groups/{id}")
        | (&Method::POST, "/contexts/{id}/associations")
        | (&Method::POST, "/contexts/{id}/associations/retract")
        | (&Method::POST, "/contexts/{id}/aliases")
        | (&Method::DELETE, "/contexts/{id}/aliases")
        | (&Method::POST, "/contexts/{id}/sources")
        | (&Method::POST, "/contexts/{id}/sources/retract")
        // Promotion (ADR 0018) is per-source re-sync between two
        // established contexts — retract_source's classification, not
        // `/import`'s Admin: it cannot create contexts and carries no
        // group or schema records. The destination named in the body
        // is checked against the key's scope in the handler.
        | (&Method::POST, "/contexts/{id}/promote")
        | (&Method::POST, "/contexts/{id}/embeddings/refresh") => Role::Write,
        // Operator verbs — and everything unclassified.
        _ => Role::Admin,
    }
}

/// Holds a request to its key's grant. Sits INSIDE the bearer gate on
/// the HTTP surface (it needs WHO), and directly on the in-process
/// `/mcp` dispatch router — `remote_mcp` stamps the outer request's
/// key AND grant onto every dispatched tool call, so the two surfaces
/// cannot drift. No key at all (auth off, or an exempt path) means no
/// restriction, exactly as before grants existed. The [`KeyGrant`]
/// judged here is the one the bearer gate resolved from the same
/// keyring snapshot that authenticated the key — this layer holds no
/// keyring of its own, so a hot reload landing mid-request can never
/// re-grade the request against a table its key was never checked
/// into. The extension then stays on the request for the handlers
/// that FILTER rather than refuse (`GET /contexts`, the `group`
/// listings) and for those judging `context` names that live in the
/// body or the stored record (`/import`, the `group` writes, the
/// cross-`context` searches).
pub async fn enforce_authorization(
    matched: Option<MatchedPath>,
    request: Request,
    next: Next,
) -> Response {
    let Some(key) = request.extensions().get::<AuthKey>().cloned() else {
        return next.run(request).await;
    };
    // The gate is the only producer of `AuthKey` on a live server and
    // always pairs it with the resolved grant; the default (the
    // historical full grant) only catches a hand-built request in a
    // test.
    let grant = request
        .extensions()
        .get::<KeyGrant>()
        .cloned()
        .unwrap_or_default();
    let started_at = Instant::now();
    let route = matched
        .as_ref()
        .map(|matched| matched.as_str().to_string())
        .unwrap_or_else(|| "<unmatched>".to_string());
    let required = required_role(request.method(), &route);
    if grant.role < required {
        return api::error(
            api::ErrorCode::Forbidden,
            format!(
                "key '{}' has role '{}', but {} {} needs '{}'",
                key.0,
                grant.role.as_str(),
                request.method(),
                route,
                required.as_str()
            ),
            started_at,
        );
    }
    let (mut parts, body) = request.into_parts();
    // `{id}` is a CONTEXT id on every route that has it but the
    // `/groups/{id}` routes below, where it is the GROUP's own id
    // instead (#995): a group's member contexts live in the body and the
    // stored record, out of this middleware's reach, so the group
    // handlers judge them (`api::scope_refusal`) or filter to the
    // grant (the row and the export). The exclusion names its routes
    // exactly — deny-by-default safe: a future route whose parameter
    // is not a context and is not listed here mis-answers 403 for
    // scoped keys, never leaks open (a prefix test would silently
    // swallow future `/groups/...` sub-routes instead of forcing that
    // decision). Grants list ids (#966), so the path's id is judged
    // as is; an id nothing answers to is simply not in the grant — the
    // same 403 an unknown context always drew from a scoped key.
    if grant.contexts.is_some()
        && !matches!(
            route.as_str(),
            "/groups/{id}" | "/groups/{id}/export" | "/groups/{id}/rename"
        )
    {
        let context = api::path_param(&mut parts, "id").await;
        if let Some(context) = context
            && !grant.allows_context(&context)
        {
            // Echoes the id as given and nothing the registry knows
            // about it: a display name here would tell a scoped key
            // both that an out-of-grant id is live and what it is
            // called (an existence oracle).
            return api::error(
                api::ErrorCode::Forbidden,
                format!("key '{}' has no grant on context '{context}'", key.0),
                started_at,
            );
        }
    }
    // The grant extension came in on the request (the gate stamped
    // it) and survives the parts round-trip — nothing to re-insert.
    next.run(Request::from_parts(parts, body)).await
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use axum::routing::get;
    use tower::util::ServiceExt;

    fn ring(single: Option<&str>, named: Option<&str>) -> SharedKeyring {
        SharedKeyring::new(
            Keyring::parse(single.map(String::from), named.map(String::from)).unwrap(),
        )
    }

    fn app(keyring: SharedKeyring) -> Router {
        gate_app(keyring, None)
    }

    fn gate_app(keyring: SharedKeyring, oauth: Option<Arc<crate::oauth::Oauth>>) -> Router {
        // Failed-auth throttle off by default here: these tests assert
        // the 401/200 verdicts, not the throttle. The throttle has its
        // own test with a ConnectInfo layer and a live budget.
        let gate = Arc::new(Gate {
            keyring,
            oauth,
            fail_limiter: Arc::new(crate::limits::RateLimiter::new(0)),
        });
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/live", get(|| async { "ok" }))
            .route("/metrics", get(|| async { "counts" }))
            .route("/contexts", get(|| async { "secret" }))
            .route("/mcp", get(|| async { "tools" }))
            .layer(axum::middleware::from_fn_with_state(gate, require_bearer))
    }

    async fn respond(app: Router, path: &str, authorization: Option<&str>) -> Response {
        let mut request = HttpRequest::builder().uri(path);
        if let Some(authorization) = authorization {
            request = request.header("Authorization", authorization);
        }
        app.oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    async fn status_of(app: Router, path: &str, authorization: Option<&str>) -> u16 {
        respond(app, path, authorization).await.status().as_u16()
    }

    #[tokio::test]
    async fn the_correct_token_passes_through() {
        assert_eq!(
            status_of(
                app(ring(Some("s3cret"), None)),
                "/contexts",
                Some("Bearer s3cret")
            )
            .await,
            200
        );
    }

    /// RFC 7235 §2.1: auth-scheme is case-insensitive — `bearer`,
    /// `BEARER`, and `Bearer` must all introduce the same credentials.
    #[tokio::test]
    async fn the_bearer_scheme_is_case_insensitive() {
        let keyring = ring(Some("s3cret"), None);
        for scheme in ["Bearer", "bearer", "BEARER", "BeArEr"] {
            assert_eq!(
                status_of(
                    app(keyring.clone()),
                    "/contexts",
                    Some(&format!("{scheme} s3cret"))
                )
                .await,
                200,
                "scheme {scheme} must authenticate"
            );
        }
    }

    /// RFC 7235 §2.1's `auth-scheme 1*SP token68`: one or MORE spaces
    /// between scheme and token authenticate alike (issue #731), while
    /// no separator at all — `Bearers3cret` — is not a Bearer
    /// credential.
    #[tokio::test]
    async fn the_bearer_scheme_takes_one_or_more_spaces_never_zero() {
        let keyring = ring(Some("s3cret"), None);
        for padded in ["Bearer s3cret", "Bearer  s3cret", "Bearer    s3cret"] {
            assert_eq!(
                status_of(app(keyring.clone()), "/contexts", Some(padded)).await,
                200,
                "{padded:?} must authenticate"
            );
        }
        assert_eq!(
            status_of(app(keyring), "/contexts", Some("Bearers3cret")).await,
            401,
            "a missing separator is not a Bearer credential"
        );
    }

    /// Every named key opens the gate, and the response carries WHICH —
    /// the fact the access log prints.
    #[tokio::test]
    async fn named_keys_authenticate_and_name_themselves() {
        let keyring = ring(Some("legacy"), Some("ci:tok-a, laptop:tok-b"));
        for (token, expected) in [("tok-a", "ci"), ("tok-b", "laptop"), ("legacy", "default")] {
            let response = respond(
                app(keyring.clone()),
                "/contexts",
                Some(&format!("Bearer {token}")),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            let key = response.extensions().get::<AuthKey>().expect("key name");
            assert_eq!(key.0.as_ref(), expected);
        }
        assert_eq!(
            status_of(app(keyring), "/contexts", Some("Bearer tok-c")).await,
            401
        );
    }

    /// Misconfigured credentials refuse to boot instead of silently
    /// arming a partial keyring — and the errors never echo a token.
    #[test]
    fn malformed_keyrings_are_refused_without_echoing_secrets() {
        let cases = [
            (Some("".to_string()), None),
            (None, Some("".to_string())),
            (None, Some("sekrit-pasted-alone".to_string())),
            (None, Some("ci:".to_string())),
            (None, Some(":sekrit-nameless".to_string())),
            (None, Some("ci:sekrit-a,ci:sekrit-b".to_string())),
            (
                Some("sekrit-x".to_string()),
                Some("default:sekrit-y".to_string()),
            ),
            (
                None,
                Some("ci:sekrit-shared,laptop:sekrit-shared".to_string()),
            ),
            (
                Some("sekrit-shared".to_string()),
                Some("ci:sekrit-shared".to_string()),
            ),
        ];
        for (single, named) in cases {
            let error = Keyring::parse(single.clone(), named.clone())
                .err()
                .unwrap_or_else(|| panic!("{single:?}/{named:?} must be refused"));
            assert!(
                !error.contains("sekrit"),
                "error text must not echo secrets: {error}"
            );
        }
        // A trailing comma is tolerable sloppiness, not a hole.
        assert_eq!(
            Keyring::parse(None, Some("ci:tok-a,".to_string()))
                .unwrap()
                .key_count(),
            1
        );
        // A key name outside 1-64 of [A-Za-z0-9._-] is refused by
        // position (ADR 0055) - '@' included, which would collide with
        // the OAuth-delegation grant fallback - and the text never
        // echoes the name, which may be a misplaced token.
        for name in [
            "bot@internal",
            "sekrit space",
            "日本語",
            "sekrit\u{202e}x",
            &"k".repeat(MAX_KEY_NAME_BYTES + 1),
        ] {
            let error = Keyring::parse(None, Some(format!("ci:tok-a,{name}:sekrit-z")))
                .err()
                .unwrap_or_else(|| panic!("key name {name:?} must be refused"));
            assert!(
                error.contains("entry #2") && !error.contains("sekrit") && !error.contains(name),
                "{error}"
            );
        }
        let longest = "k".repeat(MAX_KEY_NAME_BYTES);
        let keyring = Keyring::parse(
            None,
            Some(format!("ops.prod_1-a:tok-a,{longest}:tok-b,Z:tok-c")),
        )
        .expect("names inside the set and at the length cap are accepted");
        assert_eq!(
            keyring.authenticate("tok-b").as_deref(),
            Some(longest.as_str())
        );
        assert_eq!(
            keyring.authenticate("tok-a").as_deref(),
            Some("ops.prod_1-a")
        );
    }

    /// Two names sharing one token value would make `authenticate`'s
    /// last-match-wins scan resolve the token to whichever name was
    /// listed last — its grant, its audit-log name — while rotating
    /// the first name revoked nothing. Refused at boot, and the error
    /// names both keys without echoing the token itself.
    #[test]
    fn a_reused_token_value_is_refused_naming_both_keys() {
        let error = Keyring::parse(
            None,
            Some("ci:sekrit-shared,laptop:sekrit-shared".to_string()),
        )
        .err()
        .expect("a reused token value must be refused");
        assert!(
            error.contains("'laptop'") && error.contains("'ci'") && !error.contains("sekrit"),
            "{error}"
        );
        // The single-token variable arms the `default` key, so a value
        // collision across the two variables is the same refusal.
        let error = Keyring::parse(
            Some("sekrit-shared".to_string()),
            Some("ci:sekrit-shared".to_string()),
        )
        .err()
        .expect("a value collision across both variables must be refused");
        assert!(
            error.contains("'ci'") && error.contains("'default'") && !error.contains("sekrit"),
            "{error}"
        );
    }

    /// The colon separator may carry operator formatting whitespace on
    /// either side, mirroring the tolerance already given to the comma
    /// between entries — the token half must end up trimmed the same
    /// as the name half, not stored with a stray space that silently
    /// fails every future presentation of the token the operator meant
    /// to configure.
    #[test]
    fn a_token_with_whitespace_around_the_colon_is_trimmed_like_the_name() {
        let keyring =
            Keyring::parse(None, Some(" ci : abc123 , laptop:tok-b".to_string())).unwrap();
        assert_eq!(keyring.authenticate("abc123").as_deref(), Some("ci"));
        assert_eq!(keyring.authenticate(" abc123").as_deref(), None);
        assert_eq!(keyring.authenticate("abc123 ").as_deref(), None);
    }

    #[tokio::test]
    async fn missing_and_wrong_tokens_are_rejected_with_the_api_error_shape() {
        for authorization in [None, Some("Bearer wrong"), Some("Basic s3cret")] {
            let response =
                respond(app(ring(Some("s3cret"), None)), "/contexts", authorization).await;
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            assert_eq!(
                response.headers()[header::WWW_AUTHENTICATE],
                HeaderValue::from_static("Bearer")
            );
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["status"], "error");
            assert!(
                body["error"].is_string() && body["time"].is_number(),
                "{body}"
            );
        }
    }

    /// OAuth access tokens are a scoped delegation: they open /mcp,
    /// nothing else — and anonymous 401s advertise the discovery
    /// document OAuth clients key their flow on.
    #[tokio::test]
    async fn oauth_tokens_open_mcp_only_and_401s_advertise_discovery() {
        let dir = std::env::temp_dir().join(format!("taguru-authoauth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let oauth = Arc::new(crate::oauth::Oauth::open("https://memory.example", &dir));
        let client = oauth
            .register_client("claude", vec!["https://claude.ai/cb".to_string()])
            .unwrap();
        let now = crate::oauth::now_secs().unwrap();
        let verifier = "0123456789012345678901234567890123456789012345";
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &crate::oauth::s256_challenge(verifier),
            "laptop",
            now,
        );
        let grant = oauth
            .exchange_code(
                &client.client_id,
                &code,
                verifier,
                "https://claude.ai/cb",
                now,
            )
            .unwrap();

        // "laptop" must be a live credential of its own — the delegated
        // token inherits its recognition, not a free pass.
        let keyring = ring(Some("s3cret"), Some("laptop:tok-laptop"));
        let bearer = format!("Bearer {}", grant.access_token);
        let app = || gate_app(keyring.clone(), Some(Arc::clone(&oauth)));
        assert_eq!(status_of(app(), "/mcp", Some(&bearer)).await, 200);
        assert_eq!(status_of(app(), "/contexts", Some(&bearer)).await, 401);
        // The static key keeps opening everything.
        assert_eq!(
            status_of(app(), "/contexts", Some("Bearer s3cret")).await,
            200
        );
        // Anonymous callers learn where to start the OAuth dance.
        let refused = respond(app(), "/mcp", None).await;
        let challenge = refused.headers()[header::WWW_AUTHENTICATE]
            .to_str()
            .unwrap();
        assert!(
            challenge.contains(
                "resource_metadata=\"https://memory.example/.well-known/oauth-protected-resource\""
            ),
            "{challenge}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A key retired from `TAGURU_API_TOKENS` must retire every OAuth
    /// token delegated from it too. `oauth.json` outlives the keyring
    /// that issued a grant, so replaying a still-unexpired access
    /// token after the base key is gone must fail closed instead of
    /// falling through `grant_of` to the default admin grant.
    #[tokio::test]
    async fn revoking_a_key_revokes_its_oauth_delegations_too() {
        let dir = std::env::temp_dir().join(format!("taguru-authrevoke-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let oauth = Arc::new(crate::oauth::Oauth::open("https://memory.example", &dir));
        let client = oauth
            .register_client("claude", vec!["https://claude.ai/cb".to_string()])
            .unwrap();
        let now = crate::oauth::now_secs().unwrap();
        let verifier = "0123456789012345678901234567890123456789012345";
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &crate::oauth::s256_challenge(verifier),
            "laptop",
            now,
        );
        let grant = oauth
            .exchange_code(
                &client.client_id,
                &code,
                verifier,
                "https://claude.ai/cb",
                now,
            )
            .unwrap();
        let bearer = format!("Bearer {}", grant.access_token);

        // While "laptop" is still configured, the delegation opens /mcp.
        let live_keyring = ring(Some("s3cret"), Some("laptop:tok-laptop"));
        assert_eq!(
            status_of(
                gate_app(live_keyring, Some(Arc::clone(&oauth))),
                "/mcp",
                Some(&bearer)
            )
            .await,
            200
        );

        // The operator drops "laptop" from TAGURU_API_TOKENS and
        // restarts; the still-unexpired access token must no longer
        // authenticate anywhere, /mcp included.
        let pruned_keyring = ring(Some("s3cret"), None);
        assert_eq!(
            status_of(gate_app(pruned_keyring, Some(oauth)), "/mcp", Some(&bearer)).await,
            401
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The diff a reload logs: names only, sorted, with `rotated`
    /// (same name, new bytes) and `regranted` (same name, new grant)
    /// tracked independently — and never a token byte in the output,
    /// since the whole struct goes onto the audit line via `Debug`.
    #[test]
    fn a_reload_diff_names_keys_without_leaking_token_bytes() {
        let mut old = Keyring::parse(
            Some("sekrit-legacy".into()),
            Some("ci:sekrit-a,laptop:sekrit-b,bot:sekrit-c".into()),
        )
        .unwrap();
        old.apply_grants(Some(r#"{"bot": "read"}"#)).unwrap();
        let mut new = Keyring::parse(
            Some("sekrit-legacy".into()),
            Some("ci:sekrit-a2,bot:sekrit-c,fresh:sekrit-d".into()),
        )
        .unwrap();
        new.apply_grants(Some(r#"{"bot": "write"}"#)).unwrap();
        let diff = old.diff(&new);
        assert_eq!(diff.added, vec!["fresh"]);
        assert_eq!(diff.removed, vec!["laptop"]);
        assert_eq!(diff.rotated, vec!["ci"]);
        assert_eq!(diff.regranted, vec!["bot"]);
        let printed = format!("{diff:?}");
        assert!(!printed.contains("sekrit"), "{printed}");

        // A grant entry DROPPED is a regrant too: the key falls back
        // to the default admin grant, a real change worth a line.
        let no_grant = Keyring::parse(None, Some("bot:sekrit-c".into())).unwrap();
        let mut granted = Keyring::parse(None, Some("bot:sekrit-c".into())).unwrap();
        granted.apply_grants(Some(r#"{"bot": "read"}"#)).unwrap();
        assert_eq!(granted.diff(&no_grant).regranted, vec!["bot"]);
        assert!(granted.diff(&granted).is_empty());
    }

    /// Every reload — SIGHUP's and the config watch's alike — reads its
    /// source UNDER `reload_serial`, not before taking it. A read taken
    /// first could see file state v1, lose the race to another reload
    /// that arms the rewritten v2, then swap v1 back in — re-arming
    /// tokens v2 revoked, and, should the file turn invalid before the
    /// next tick, keeping them armed with nothing left to re-trigger.
    ///
    /// Deterministic via a FIFO as the config file: the reader's
    /// `open` blocks until this test opens the write end, so the
    /// moment our `open` returns the reload thread is provably INSIDE
    /// its read — and at that moment the lock must already be held.
    #[cfg(unix)]
    #[test]
    fn every_reload_reads_its_source_while_holding_the_reload_lock() {
        use std::io::Write;

        let dir = crate::registry::test_support::scratch_dir("auth-reload-read-under-lock");
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("taguru.env");
        let status = std::process::Command::new("mkfifo")
            .arg(&config)
            .status()
            .unwrap();
        assert!(status.success(), "mkfifo failed");
        let source = AuthSource {
            config: Some(config.clone()),
            from_shell: [false; AUTH_VARS.len()],
        };
        let keyring = ring(None, Some("ci:tok-old"));

        let sighup = || Some(reload_keyring(&keyring, &source, "test"));
        let watch = || reload_keyring_if_changed(&keyring, &source);
        let reloads: [(&str, &(dyn Fn() -> Option<ReloadOutcome> + Sync)); 2] =
            [("sighup", &sighup), ("config-watch", &watch)];
        for (trigger, reload) in reloads {
            let token = format!("tok-{trigger}");
            let (locked_during_read, outcome) = std::thread::scope(|scope| {
                let reload = scope.spawn(reload);
                // Returns only once the reload thread has opened the
                // read end, i.e. is mid-read.
                let mut writer = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&config)
                    .unwrap();
                let locked = keyring.reload_serial.is_locked();
                // Finish the read either way, so a failing assertion
                // below reports instead of hanging the reload thread.
                writer
                    .write_all(format!("TAGURU_API_TOKENS=ci:{token}\n").as_bytes())
                    .unwrap();
                drop(writer);
                (locked, reload.join().unwrap())
            });

            assert!(
                locked_during_read,
                "{trigger}: the source read ran outside reload_serial, so it can race another \
                 reload's swap"
            );
            assert_eq!(outcome, Some(ReloadOutcome::Applied), "{trigger}");
            assert_eq!(
                keyring.load().authenticate(&token).as_deref(),
                Some("ci"),
                "{trigger}"
            );
        }
        assert!(keyring.load().authenticate("tok-old").is_none());

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A config file under a fresh scratch dir, and the reload source
    /// that follows it for every auth variable.
    fn watched_config(tag: &str) -> (PathBuf, PathBuf, AuthSource) {
        let dir = crate::registry::test_support::scratch_dir(tag);
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("taguru.env");
        let source = AuthSource {
            config: Some(config.clone()),
            from_shell: [false; AUTH_VARS.len()],
        };
        (dir, config, source)
    }

    /// A ring armed with `named`, booted from config-file bytes
    /// `boot` — the digest main seeds the reload ledger with (#942).
    fn booted_ring(named: &str, boot: &str) -> SharedKeyring {
        SharedKeyring::with_config_digest(
            Keyring::parse(None, Some(named.into())).unwrap(),
            Some(crate::sha256::sha256_hex(boot.as_bytes())),
        )
    }

    /// The watch's change detector is shared with SIGHUP: content a
    /// SIGHUP judged counts as judged, and — what a watch-private
    /// digest gets wrong — content the watch judged EARLIER is a
    /// change again once a SIGHUP has armed something else. Here a
    /// rotation is picked up by SIGHUP and then reverted before the
    /// next tick: a private digest still matching the reverted bytes
    /// would leave the rotation armed against a file that no longer
    /// says it, with no digest change left to re-trigger.
    #[test]
    fn a_file_reverted_after_a_sighup_reload_still_reads_as_a_change() {
        let (dir, config, source) = watched_config("auth-watch-revert");
        let boot = "TAGURU_API_TOKENS=ci:tok-a\n";
        std::fs::write(&config, boot).unwrap();
        let keyring = booted_ring("ci:tok-a", boot);

        std::fs::write(&config, "TAGURU_API_TOKENS=ci:tok-b\n").unwrap();
        assert_eq!(
            reload_keyring(&keyring, &source, "sighup"),
            ReloadOutcome::Applied
        );
        // The SIGHUP's judgment is the watch's too: no second reload.
        assert_eq!(reload_keyring_if_changed(&keyring, &source), None);

        std::fs::write(&config, boot).unwrap();
        assert_eq!(
            reload_keyring_if_changed(&keyring, &source),
            Some(ReloadOutcome::Applied),
            "the reverted file must still read as a change"
        );
        let armed = keyring.load();
        assert_eq!(armed.authenticate("tok-a").as_deref(), Some("ci"));
        assert!(armed.authenticate("tok-b").is_none());

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The watch's change detector: boot's own bytes (#942) and
    /// content already judged are no reload, an unreadable file
    /// neither reloads nor moves the baseline, and each content change
    /// is judged exactly once — a refusal included (malformed or not
    /// UTF-8), one loud line per change rather than per tick — while a
    /// SIGHUP judges whatever the file says, judged or not, and the
    /// last good table stays armed through every refusal.
    #[test]
    fn the_config_watch_judges_each_content_change_once() {
        let (dir, config, source) = watched_config("auth-watch-change-detection");
        let boot = "TAGURU_API_TOKENS=ci:tok-a\n";
        std::fs::write(&config, boot).unwrap();
        let keyring = booted_ring("ci:tok-a", boot);
        assert_eq!(reload_keyring_if_changed(&keyring, &source), None);
        // Unseeded, the very same bytes are a (no-op) change.
        assert_eq!(
            reload_keyring_if_changed(&ring(None, Some("ci:tok-a")), &source),
            Some(ReloadOutcome::Unchanged)
        );

        std::fs::remove_file(&config).unwrap();
        assert_eq!(reload_keyring_if_changed(&keyring, &source), None);
        std::fs::write(&config, boot).unwrap();
        assert_eq!(reload_keyring_if_changed(&keyring, &source), None);

        std::fs::write(&config, "TAGURU_API_TOKENS=ci:tok-b\n").unwrap();
        assert_eq!(
            reload_keyring_if_changed(&keyring, &source),
            Some(ReloadOutcome::Applied)
        );
        assert_eq!(keyring.load().authenticate("tok-b").as_deref(), Some("ci"));
        assert_eq!(reload_keyring_if_changed(&keyring, &source), None);

        for refused in [
            b"TAGURU_API_TOKENS=token-pasted-alone\n".as_slice(),
            b"TAGURU_API_TOKENS=ci:\xff\n".as_slice(),
        ] {
            std::fs::write(&config, refused).unwrap();
            assert_eq!(
                reload_keyring_if_changed(&keyring, &source),
                Some(ReloadOutcome::Refused)
            );
            assert_eq!(reload_keyring_if_changed(&keyring, &source), None);
            assert_eq!(
                reload_keyring(&keyring, &source, "sighup"),
                ReloadOutcome::Refused
            );
            assert_eq!(keyring.load().authenticate("tok-b").as_deref(), Some("ci"));
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    /// One reload against a LIVE gate: the same `SharedKeyring` flips
    /// to the new table for the next request — the rotated name
    /// authenticates with its new bytes, the removed key dies, and so
    /// does the OAuth token delegated from it, with no restart and no
    /// rebuilt router.
    #[tokio::test]
    async fn a_reload_swaps_the_live_gate_and_retires_removed_keys_delegations() {
        let dir = std::env::temp_dir().join(format!("taguru-authreload-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let oauth = Arc::new(crate::oauth::Oauth::open("https://memory.example", &dir));
        let client = oauth
            .register_client("claude", vec!["https://claude.ai/cb".to_string()])
            .unwrap();
        let now = crate::oauth::now_secs().unwrap();
        let verifier = "0123456789012345678901234567890123456789012345";
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &crate::oauth::s256_challenge(verifier),
            "laptop",
            now,
        );
        let grant = oauth
            .exchange_code(
                &client.client_id,
                &code,
                verifier,
                "https://claude.ai/cb",
                now,
            )
            .unwrap();
        let delegated = format!("Bearer {}", grant.access_token);

        let config = dir.join("taguru.env");
        std::fs::write(&config, "TAGURU_API_TOKENS=ci:tok-old,laptop:tok-laptop\n").unwrap();
        let source = AuthSource {
            config: Some(config.clone()),
            from_shell: [false; AUTH_VARS.len()],
        };
        let keyring = ring(None, Some("ci:tok-old,laptop:tok-laptop"));
        let app = || gate_app(keyring.clone(), Some(Arc::clone(&oauth)));

        assert_eq!(status_of(app(), "/mcp", Some(&delegated)).await, 200);
        assert_eq!(
            status_of(app(), "/contexts", Some("Bearer tok-old")).await,
            200
        );

        // Rotate ci, drop laptop, reload.
        std::fs::write(&config, "TAGURU_API_TOKENS=ci:tok-new\n").unwrap();
        assert_eq!(
            reload_keyring(&keyring, &source, "test"),
            ReloadOutcome::Applied
        );

        assert_eq!(
            status_of(app(), "/contexts", Some("Bearer tok-new")).await,
            200
        );
        assert_eq!(
            status_of(app(), "/contexts", Some("Bearer tok-old")).await,
            401
        );
        assert_eq!(
            status_of(app(), "/contexts", Some("Bearer tok-laptop")).await,
            401
        );
        // The delegation minted from "laptop" dies with its key — the
        // per-request `recognizes` check reads the swapped table.
        assert_eq!(status_of(app(), "/mcp", Some(&delegated)).await, 401);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every refusal path keeps the previous table armed: an
    /// unreadable file, a malformed entry, and the one transition that
    /// must never happen by accident — armed → empty, which would
    /// reopen the server unauthenticated. Arming a disabled gate stays
    /// allowed (that transition CLOSES the server), an identical
    /// source is an explicit no-op, and the file's last duplicate wins
    /// exactly as `load_config` applies it at boot.
    #[tokio::test]
    async fn a_reload_fails_closed_and_never_disarms_authentication() {
        let dir =
            std::env::temp_dir().join(format!("taguru-authfailclosed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("taguru.env");
        let source = AuthSource {
            config: Some(config.clone()),
            from_shell: [false; AUTH_VARS.len()],
        };
        let keyring = ring(None, Some("ci:tok-a"));
        let gate = || app(keyring.clone());

        // A malformed source: refused, the old token keeps working.
        std::fs::write(&config, "TAGURU_API_TOKENS=token-pasted-alone\n").unwrap();
        assert_eq!(
            reload_keyring(&keyring, &source, "test"),
            ReloadOutcome::Refused
        );
        assert_eq!(
            status_of(gate(), "/contexts", Some("Bearer tok-a")).await,
            200
        );

        // An unreadable source: the same refusal.
        let missing = AuthSource {
            config: Some(dir.join("gone.env")),
            from_shell: [false; AUTH_VARS.len()],
        };
        assert_eq!(
            reload_keyring(&keyring, &missing, "test"),
            ReloadOutcome::Refused
        );

        // Sources that arm NO key while the gate is armed: refused —
        // a reload must never open the server.
        std::fs::write(&config, "TAGURU_FLUSH_SECS=5\n").unwrap();
        assert_eq!(
            reload_keyring(&keyring, &source, "test"),
            ReloadOutcome::Refused
        );
        assert_eq!(
            status_of(gate(), "/contexts", Some("Bearer tok-a")).await,
            200
        );
        assert_eq!(status_of(gate(), "/contexts", None).await, 401);

        // The same table again — via the file's last duplicate, which
        // wins like load_config's boot application: an explicit no-op.
        std::fs::write(
            &config,
            "TAGURU_API_TOKENS=ci:tok-first\nTAGURU_API_TOKENS=ci:tok-a\n",
        )
        .unwrap();
        assert_eq!(
            reload_keyring(&keyring, &source, "test"),
            ReloadOutcome::Unchanged
        );

        // A grant-only change still applies.
        std::fs::write(
            &config,
            "TAGURU_API_TOKENS=ci:tok-a\nTAGURU_KEY_GRANTS={\"ci\": \"read\"}\n",
        )
        .unwrap();
        assert_eq!(
            reload_keyring(&keyring, &source, "test"),
            ReloadOutcome::Applied
        );
        assert_eq!(keyring.load().grant_of("ci").role, Role::Read);

        // A disabled gate may ARM by reload — that transition closes.
        let disabled = SharedKeyring::new(Keyring::parse(None, None).unwrap());
        std::fs::write(&config, "TAGURU_API_TOKENS=ci:tok-a\n").unwrap();
        assert_eq!(
            reload_keyring(&disabled, &source, "test"),
            ReloadOutcome::Applied
        );
        assert!(!disabled.load().is_disabled());

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The pre-0.10.0 grants name (#959) refuses a reload like any
    /// malformed source, the previous table staying armed: applying the
    /// file without it would arm no grant, and every key it restricts
    /// would fall back to admin everywhere. It refuses beside the new
    /// name too, and with an empty value — presence is the signal.
    #[test]
    fn a_reload_carrying_the_retired_grants_name_keeps_the_previous_table() {
        let dir = std::env::temp_dir().join(format!("taguru-authretired-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("taguru.env");
        let source = AuthSource {
            config: Some(config.clone()),
            from_shell: [false; AUTH_VARS.len()],
        };
        let mut armed = Keyring::parse(None, Some("ci:tok-a".into())).unwrap();
        armed.apply_grants(Some(r#"{"ci": "read"}"#)).unwrap();
        let keyring = SharedKeyring::new(armed);

        for retired in [
            "TAGURU_KEY_SCOPES={\"ci\": \"read\"}\n",
            "TAGURU_KEY_GRANTS={\"ci\": \"read\"}\nTAGURU_KEY_SCOPES={\"ci\": \"read\"}\n",
            "TAGURU_KEY_SCOPES=\n",
        ] {
            std::fs::write(&config, format!("TAGURU_API_TOKENS=ci:tok-b\n{retired}")).unwrap();
            assert_eq!(
                reload_keyring(&keyring, &source, "test"),
                ReloadOutcome::Refused,
                "{retired}"
            );
            let current = keyring.load();
            assert_eq!(current.grant_of("ci").role, Role::Read, "{retired}");
            assert_eq!(
                current.authenticate("tok-a").as_deref(),
                Some("ci"),
                "the old token stays armed: {retired}"
            );
        }

        // Renamed, the same table is an explicit no-op.
        std::fs::write(
            &config,
            "TAGURU_API_TOKENS=ci:tok-a\nTAGURU_KEY_GRANTS={\"ci\": \"read\"}\n",
        )
        .unwrap();
        assert_eq!(
            reload_keyring(&keyring, &source, "test"),
            ReloadOutcome::Unchanged
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_retired_grants_name_refuses_by_presence_and_names_its_replacement() {
        assert_eq!(refuse_retired_grants_var(false), Ok(()));
        let refusal = refuse_retired_grants_var(true).unwrap_err();
        assert!(refusal.contains("TAGURU_KEY_SCOPES"), "{refusal}");
        assert!(refusal.contains("TAGURU_KEY_GRANTS"), "{refusal}");
    }

    #[tokio::test]
    async fn auth_is_disabled_when_no_token_is_configured() {
        let keyring = Keyring::parse(None, None).unwrap();
        assert!(keyring.is_disabled());
        assert_eq!(
            status_of(app(SharedKeyring::new(keyring)), "/contexts", None).await,
            200
        );
    }

    #[tokio::test]
    async fn health_and_metrics_are_exempt_while_everything_else_is_gated() {
        let keyring = ring(Some("s3cret"), None);
        for probe in ["/health", "/live", "/metrics"] {
            assert_eq!(
                status_of(app(keyring.clone()), probe, None).await,
                200,
                "{probe} must answer unconfigured"
            );
        }
        assert_eq!(status_of(app(keyring), "/contexts", None).await, 401);
    }

    /// Repeated failures from one source IP trip a 429 with a
    /// Retry-After, but a valid token from that very IP still passes:
    /// a neighbor spending the shared IP's budget can never lock the
    /// legitimate caller out (the trade `Gate`'s doc spells out).
    #[tokio::test]
    async fn failed_auth_is_throttled_per_source_ip() {
        use axum::extract::connect_info::MockConnectInfo;
        use std::net::SocketAddr;

        let gate = Arc::new(Gate {
            keyring: ring(Some("s3cret"), None),
            oauth: None,
            fail_limiter: Arc::new(crate::limits::RateLimiter::new(3)),
        });
        let app = Router::new()
            .route("/contexts", get(|| async { "secret" }))
            .layer(axum::middleware::from_fn_with_state(gate, require_bearer))
            // Outermost, so every request carries a peer address before
            // the gate reads one — all from the same mocked IP, so they
            // share a bucket and the budget trips.
            .layer(MockConnectInfo(SocketAddr::from(([127, 0, 0, 1], 40000))));
        let send = |authorization: &'static str| {
            let app = app.clone();
            async move {
                app.oneshot(
                    HttpRequest::builder()
                        .uri("/contexts")
                        .header("Authorization", authorization)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        };

        // The 3/min budget spends on failures...
        for _ in 0..3 {
            assert_eq!(
                send("Bearer wrong").await.status(),
                StatusCode::UNAUTHORIZED
            );
        }
        // ...and the next failure is throttled, not merely refused.
        let throttled = send("Bearer wrong").await;
        assert_eq!(throttled.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(throttled.headers().get(header::RETRY_AFTER).is_some());

        // The right token still opens the gate from the same IP: success
        // never reaches the throttle.
        assert_eq!(send("Bearer s3cret").await.status(), StatusCode::OK);
    }

    /// Key grants parse strictly: a grant naming no configured key, a
    /// role typo, an empty `contexts` list, or non-JSON all refuse the
    /// boot instead of arming a partial authorization table.
    #[test]
    fn key_grants_parse_strictly_and_resolve_with_the_oauth_fallback() {
        let mut keyring =
            Keyring::parse(None, Some("boss:tok-a,reader:tok-b,bot:tok-c".to_string())).unwrap();
        keyring
            .apply_grants(Some(
                r#"{"reader": "read", "bot": {"role": "write", "contexts": ["cef2e28b-43f0-4b6c-8201-abab0785399f"]}}"#,
            ))
            .unwrap();
        // The boot and reload audit lines report both counts: every
        // configured key, and the ones a grant names.
        assert_eq!(keyring.key_count(), 3);
        assert_eq!(keyring.granted_key_count(), 2);
        // Unnamed keys keep the historical full grant.
        assert_eq!(keyring.grant_of("boss").role, Role::Admin);
        assert!(keyring.grant_of("boss").allows_context("anything"));
        assert_eq!(keyring.grant_of("reader").role, Role::Read);
        // OAuth delegations ("key@client") inherit the key's grant.
        let delegated = keyring.grant_of("bot@claude-abc123");
        assert_eq!(delegated.role, Role::Write);
        assert!(delegated.allows_context("cef2e28b-43f0-4b6c-8201-abab0785399f"));
        assert!(!delegated.allows_context("ead6ef03-d61e-460c-933d-6d450c50a1e5"));

        for (grants, complaint) in [
            (r#"{"ghost": "read"}"#, "no configured key"),
            (r#"{"reader": "supreme"}"#, "unknown role"),
            (r#"{"reader": {"role": "read", "contexts": []}}"#, "empty"),
            // A display name would match no context ever — a silent
            // lock-out — so it refuses boot (#966).
            (
                r#"{"reader": {"role": "read", "contexts": ["sake"]}}"#,
                "'sake' is not a context id",
            ),
            (
                r#"{"reader": {"role": "read", "contexts": ["CEF2E28B-43F0-4B6C-8201-ABAB0785399F"]}}"#,
                "is not a context id",
            ),
            // An unread key fails OPEN (the grant would cover every
            // context), so a typo or a key grants never had refuses.
            (
                r#"{"reader": {"role": "read", "context": ["cef2e28b-43f0-4b6c-8201-abab0785399f"]}}"#,
                "unknown field 'context'",
            ),
            (
                r#"{"reader": {"role": "read", "groups": ["cef2e28b-43f0-4b6c-8201-abab0785399f"]}}"#,
                "unknown field 'groups'",
            ),
            ("not json", "documented JSON shape"),
            ("", "empty"),
        ] {
            let mut keyring = Keyring::parse(None, Some("reader:tok-b".to_string())).unwrap();
            let error = keyring.apply_grants(Some(grants)).unwrap_err();
            assert!(error.contains(complaint), "{grants} → {error}");
        }
    }

    /// The role table fails closed: an endpoint nobody classified
    /// demands admin, so a granted key locks out of new surface until
    /// someone decides otherwise.
    #[test]
    fn unclassified_routes_demand_admin() {
        assert_eq!(
            required_role(&Method::POST, "/contexts/{id}/future_thing"),
            Role::Admin
        );
        assert_eq!(
            required_role(&Method::GET, "/contexts/{id}/export"),
            Role::Read
        );
        assert_eq!(
            required_role(&Method::GET, "/groups/{id}/export"),
            Role::Read
        );
        assert_eq!(
            required_role(&Method::POST, "/contexts/{id}/sources/retract"),
            Role::Write
        );
        // Promotion (ADR 0018) shares retract_source's classification —
        // per-source re-sync between established contexts, not the
        // context-creating, group-carrying `/import`.
        assert_eq!(
            required_role(&Method::POST, "/contexts/{id}/promote"),
            Role::Write
        );
        assert_eq!(required_role(&Method::POST, "/import"), Role::Admin);
        assert_eq!(
            required_role(&Method::DELETE, "/contexts/{id}"),
            Role::Admin
        );
    }

    /// ADR 0009 §12.5: `GET /schema` sits beside the other `context`
    /// GETs; `PUT` is `Write` (an ingest-loop verb an agent performs),
    /// not `Admin`, the same classification `context` creation itself
    /// already gets.
    #[test]
    fn schema_get_is_read_and_put_is_write() {
        assert_eq!(
            required_role(&Method::GET, "/contexts/{id}/schema"),
            Role::Read
        );
        assert_eq!(
            required_role(&Method::PUT, "/contexts/{id}/schema"),
            Role::Write
        );
    }

    /// ADR 0009 §12.5: both new §10 routes are read-only judgments over
    /// the live graph (`audit`) or a never-persisted proposed document
    /// (`validate`) — neither writes anything, so both classify beside
    /// `vocabulary/audit` and `drift/audit`, not beside `PUT /schema`.
    #[test]
    fn schema_audit_and_validate_are_read() {
        assert_eq!(
            required_role(&Method::POST, "/contexts/{id}/schema/audit"),
            Role::Read
        );
        assert_eq!(
            required_role(&Method::POST, "/contexts/{id}/schema/validate"),
            Role::Read
        );
    }

    /// The three `/explain` endpoints are read-only diagnostics for
    /// their base endpoint — same role, or fail-closed silently
    /// demotes a granted reader to Admin the moment they ask why a
    /// call resolved the way it did.
    #[test]
    fn explain_routes_share_their_base_endpoints_role() {
        for route in [
            "/contexts/{id}/resolve/explain",
            "/contexts/{id}/resolve_label/explain",
            "/contexts/{id}/sources/search/explain",
        ] {
            assert_eq!(required_role(&Method::POST, route), Role::Read, "{route}");
        }
    }

    /// Exemption is by exact path, not by `/oauth/` prefix: a route
    /// merely nested under it — an admin endpoint some future author
    /// adds to `oauth_http.rs` without revisiting this gate — must
    /// stay behind the bearer check.
    #[test]
    fn oauth_exemption_is_exact_not_prefixed() {
        for probe in PROBE_EXEMPT {
            assert!(is_auth_exempt(probe, false), "{probe}");
            assert!(is_auth_exempt(probe, true), "{probe}");
        }

        let oauth_routes = [
            "/oauth/register",
            "/oauth/authorize",
            "/oauth/token",
            "/.well-known/oauth-protected-resource",
            "/.well-known/oauth-protected-resource/mcp",
            "/.well-known/oauth-authorization-server",
        ];
        for route in oauth_routes {
            assert!(!is_auth_exempt(route, false), "{route} needs oauth on");
            assert!(is_auth_exempt(route, true), "{route}");
        }

        for sneaky in [
            "/oauth/admin-panel",
            "/oauth/",
            "/oauth",
            "/.well-known/oauth-evil-metadata",
        ] {
            assert!(
                !is_auth_exempt(sneaky, true),
                "{sneaky} must not ride the prefix match"
            );
        }
    }

    /// The authorization layer end to end: role refusals, `context`
    /// grants, and the untouched full-grant default, all in the
    /// ApiError shape with a 403. Paths carry ids (#964) and so do the
    /// grants (#966): the middleware judges the path's id as is.
    #[tokio::test]
    async fn granted_keys_are_held_to_role_and_context() {
        let dir = crate::registry::test_support::scratch_dir("auth-grants");
        let state = crate::registry::AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let sake = state
            .create("sake", crate::registry::ContextMeta::default())
            .unwrap();
        let bunko = state
            .create("bunko", crate::registry::ContextMeta::default())
            .unwrap();
        let mut keyring =
            Keyring::parse(None, Some("boss:tok-a,reader:tok-b,bot:tok-c".to_string())).unwrap();
        keyring
            .apply_grants(Some(&format!(
                r#"{{"reader": "read", "bot": {{"role": "write", "contexts": ["{sake}"]}}}}"#,
            )))
            .unwrap();
        let keyring = SharedKeyring::new(keyring);
        let app = || {
            let gate = Arc::new(Gate {
                keyring: keyring.clone(),
                oauth: None,
                fail_limiter: Arc::new(crate::limits::RateLimiter::new(0)),
            });
            Router::new()
                .route("/contexts", get(|| async { "rows" }))
                .route(
                    "/contexts/{id}/recall",
                    axum::routing::post(|| async { "hits" }),
                )
                .route(
                    "/contexts/{id}/associations",
                    axum::routing::post(|| async { "landed" }),
                )
                .route("/contexts/{id}", axum::routing::delete(|| async { "gone" }))
                // Authorization innermost, the bearer gate outside it —
                // the same nesting main.rs builds. It judges from the
                // grant extension the gate stamps, keyring-free.
                .layer(axum::middleware::from_fn(enforce_authorization))
                .layer(axum::middleware::from_fn_with_state(gate, require_bearer))
        };
        let send = |method: &'static str, path: String, token: &'static str| {
            let app = app();
            async move {
                app.oneshot(
                    HttpRequest::builder()
                        .method(method)
                        .uri(path)
                        .header("Authorization", format!("Bearer {token}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        };

        // The read key runs the retrieval loop and nothing else.
        assert_eq!(
            send("POST", format!("/contexts/{sake}/recall"), "tok-b")
                .await
                .status(),
            200
        );
        let refused = send("POST", format!("/contexts/{sake}/associations"), "tok-b").await;
        assert_eq!(refused.status(), StatusCode::FORBIDDEN);
        let bytes = axum::body::to_bytes(refused.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "error");
        assert!(
            body["error"].as_str().unwrap().contains("needs 'write'"),
            "{body}"
        );

        // The context-scoped write key writes inside its grant only,
        // and never reaches the admin verbs.
        assert_eq!(
            send("POST", format!("/contexts/{sake}/associations"), "tok-c")
                .await
                .status(),
            200
        );
        let outside = send("POST", format!("/contexts/{bunko}/associations"), "tok-c").await;
        assert_eq!(outside.status(), StatusCode::FORBIDDEN);
        let bytes = axum::body::to_bytes(outside.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let message = body["error"].as_str().unwrap();
        assert!(message.contains(&format!("'{bunko}'")), "{body}");
        // The id as given, never the display name (an existence oracle).
        assert!(!message.contains("bunko'"), "{body}");
        assert_eq!(
            send("DELETE", format!("/contexts/{sake}"), "tok-c")
                .await
                .status(),
            StatusCode::FORBIDDEN
        );

        // A key with no TAGURU_KEY_GRANTS entry keeps the historical full grant.
        assert_eq!(
            send("DELETE", format!("/contexts/{sake}"), "tok-a")
                .await
                .status(),
            200
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The scope check reads the path param through the same decoding
    /// `Path` gives handlers, not the raw percent-encoded segment — an
    /// id split across a percent-encoded byte must still decode to
    /// the granted id, not be refused for comparing unequal to the
    /// still-encoded form.
    #[tokio::test]
    async fn scope_check_matches_a_percent_encoded_context_id() {
        let dir = crate::registry::test_support::scratch_dir("auth-encoded-id");
        let state = crate::registry::AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let sake = state
            .create("sake", crate::registry::ContextMeta::default())
            .unwrap();
        let mut keyring = Keyring::parse(None, Some("bot:tok-c".to_string())).unwrap();
        keyring
            .apply_grants(Some(&format!(
                r#"{{"bot": {{"role": "write", "contexts": ["{sake}"]}}}}"#
            )))
            .unwrap();
        let keyring = SharedKeyring::new(keyring);
        let gate = Arc::new(Gate {
            keyring,
            oauth: None,
            fail_limiter: Arc::new(crate::limits::RateLimiter::new(0)),
        });
        let app = Router::new()
            .route(
                "/contexts/{id}/associations",
                axum::routing::post(|| async { "landed" }),
            )
            .layer(axum::middleware::from_fn(enforce_authorization))
            .layer(axum::middleware::from_fn_with_state(gate, require_bearer));

        // The first byte percent-encoded: decodes to the minted id,
        // exactly the granted one.
        let encoded = format!("%{:02x}{}", sake.as_bytes()[0], &sake[1..]);
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri(format!("/contexts/{encoded}/associations"))
                    .header("Authorization", "Bearer tok-c")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// RFC 7230 §3.2.2: a repeated Authorization header is malformed. Rather
    /// than pick a winner — which a proxy and the gate might disagree on — the
    /// gate fails closed, so smuggling a second credential past the first
    /// never authenticates.
    #[tokio::test]
    async fn duplicate_authorization_headers_are_refused() {
        let request = HttpRequest::builder()
            .uri("/contexts")
            .header("Authorization", "Bearer wrong")
            .header("Authorization", "Bearer s3cret")
            .body(Body::empty())
            .unwrap();
        let response = app(ring(Some("s3cret"), None))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
