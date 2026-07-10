//! OAuth 2.1 for the remote MCP transport: the resource-server side of
//! the MCP authorization spec (RFC 9728 protected-resource metadata,
//! bearer validation, `resource_metadata` hints on 401) plus a
//! deliberately minimal EMBEDDED authorization server, so that clients
//! which insist on OAuth — claude.ai custom connectors above all —
//! connect without the operator standing up an external IdP.
//!
//! The embedded server's only "login" is possession of an
//! already-configured API key: OAuth here formalizes the DELEGATION of
//! a named key to a remote client, not a new identity system. The
//! consent page asks for a key, and every issued token acts as that
//! key with the client's name appended ("laptop@claude"), which is
//! what the access log then shows. OAuth tokens open `/mcp` only —
//! the delegation is scoped to the MCP surface, never the raw API.
//!
//! Mechanics: dynamic client registration (RFC 7591, capped),
//! authorization code + PKCE S256 only, single-use 60-second codes,
//! one-hour opaque access tokens held in memory (hashed), 30-day
//! rotating refresh tokens persisted hashed in `data_dir/oauth.json` —
//! so a server restart costs connected clients one silent refresh,
//! and a stolen store file contains nothing replayable.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use url::{Host, Url};

pub const ACCESS_TTL_SECS: u64 = 3600;
pub const REFRESH_TTL_SECS: u64 = 30 * 24 * 3600;
pub const CODE_TTL_SECS: u64 = 60;
/// Registration is unauthenticated by design (RFC 7591 as MCP clients
/// practice it), so the store must be bounded by us, not by callers.
pub const CLIENT_CAP: usize = 100;
/// Every caller-controlled field of a registration is bounded too, not
/// just the client count: without these, one unauthenticated call could
/// carry a body-sized name or a flood of redirect URIs, and `CLIENT_CAP`
/// such registrations would bloat `oauth.json` without limit. Generous
/// ceilings — a display label and a short list of real callback URLs.
pub const MAX_CLIENT_NAME_BYTES: usize = 256;
pub const MAX_REDIRECT_URIS: usize = 10;
pub const MAX_REDIRECT_URI_BYTES: usize = 2048;

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// 32 bytes of CSPRNG output as base64url — token material, client
/// ids, authorization codes.
pub fn random_token() -> String {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).expect("the OS random source must work");
    base64url(&bytes)
}

/// SHA-256 as lowercase hex — how every secret is stored at rest.
fn digest_hex(secret: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(secret.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Base64url without padding (RFC 4648 §5) — encode-only, which is all
/// PKCE and token minting need.
fn base64url(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[(triple >> 18) as usize & 63] as char);
        out.push(TABLE[(triple >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(triple >> 6) as usize & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(TABLE[triple as usize & 63] as char);
        }
    }
    out
}

/// The S256 code challenge for a verifier (RFC 7636 §4.2).
pub fn s256_challenge(verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    base64url(&hasher.finalize())
}

/// One registered client (RFC 7591). Public client — no secret; PKCE
/// carries the proof instead.
#[derive(Clone, Serialize, Deserialize)]
pub struct Client {
    pub client_id: String,
    pub client_name: String,
    pub redirect_uris: Vec<String>,
    pub created_at: u64,
}

#[derive(Serialize, Deserialize, Default)]
struct StoreFile {
    clients: Vec<Client>,
    refresh: Vec<RefreshToken>,
}

#[derive(Clone, Serialize, Deserialize)]
struct RefreshToken {
    /// SHA-256 hex of the presented token.
    hash: String,
    /// The delegated identity ("laptop@claude") tokens act as.
    key: String,
    client_id: String,
    expires_at: u64,
}

struct CodeGrant {
    code_hash: String,
    client_id: String,
    redirect_uri: String,
    code_challenge: String,
    key: String,
    expires_at: u64,
}

struct AccessToken {
    hash: String,
    key: Arc<str>,
    expires_at: u64,
}

/// What the token endpoint hands out.
pub struct TokenGrant {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: u64,
}

/// A token-endpoint refusal, in RFC 6749 error-code vocabulary.
#[derive(Debug, PartialEq, Eq)]
pub struct OauthError(pub &'static str);

/// The whole OAuth subsystem: registrations, live grants, and the
/// resource-server view over them.
pub struct Oauth {
    /// Canonical public base URL (no trailing slash) — the issuer, and
    /// the base of the canonical `{public_url}/mcp` resource.
    public_url: String,
    store_path: PathBuf,
    // Lock order: a path that needs more than one of these takes them
    // in field order. `register_client` and `mint` both hold `clients`
    // and `refresh` together around `persist`, and the one agreed
    // order is what keeps a concurrent registration and token grant
    // from deadlocking each other.
    clients: Mutex<Vec<Client>>,
    codes: Mutex<Vec<CodeGrant>>,
    access: Mutex<Vec<AccessToken>>,
    refresh: Mutex<Vec<RefreshToken>>,
}

impl Oauth {
    /// Loads persisted registrations and refresh grants; a missing or
    /// unreadable store starts fresh (the safe direction: connected
    /// clients re-authorize, nothing is silently trusted).
    pub fn open(public_url: &str, data_dir: &Path) -> Self {
        let store_path = data_dir.join("oauth.json");
        let persisted: StoreFile = std::fs::read(&store_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        Self {
            public_url: public_url.trim_end_matches('/').to_string(),
            store_path,
            clients: Mutex::new(persisted.clients),
            codes: Mutex::new(Vec::new()),
            access: Mutex::new(Vec::new()),
            refresh: Mutex::new(persisted.refresh),
        }
    }

    pub fn public_url(&self) -> &str {
        &self.public_url
    }

    /// The canonical resource this server protects — the MCP endpoint,
    /// not the API root.
    pub fn resource(&self) -> String {
        format!("{}/mcp", self.public_url)
    }

    pub fn resource_metadata_url(&self) -> String {
        format!("{}/.well-known/oauth-protected-resource", self.public_url)
    }

    /// RFC 9728: who protects what, and where the authorization server
    /// lives (here: same origin).
    pub fn protected_resource_metadata(&self) -> Value {
        json!({
            "resource": self.resource(),
            "authorization_servers": [self.public_url],
            "bearer_methods_supported": ["header"],
        })
    }

    /// RFC 8414: the embedded authorization server's shape. PKCE S256
    /// only, public clients only.
    pub fn authorization_server_metadata(&self) -> Value {
        json!({
            "issuer": self.public_url,
            "authorization_endpoint": format!("{}/oauth/authorize", self.public_url),
            "token_endpoint": format!("{}/oauth/token", self.public_url),
            "registration_endpoint": format!("{}/oauth/register", self.public_url),
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
        })
    }

    /// RFC 7591 registration. Unauthenticated, so validated hard and
    /// capped: https (or loopback) redirect URIs only.
    pub fn register_client(
        &self,
        client_name: &str,
        redirect_uris: Vec<String>,
    ) -> Result<Client, String> {
        if redirect_uris.is_empty() {
            return Err("redirect_uris must not be empty".to_string());
        }
        if redirect_uris.len() > MAX_REDIRECT_URIS {
            return Err(format!(
                "too many redirect_uris ({}, max {MAX_REDIRECT_URIS})",
                redirect_uris.len()
            ));
        }
        for uri in &redirect_uris {
            if uri.len() > MAX_REDIRECT_URI_BYTES {
                return Err(format!(
                    "a redirect uri is too long ({} bytes, max {MAX_REDIRECT_URI_BYTES})",
                    uri.len()
                ));
            }
            if !uri.starts_with("https://") && !is_loopback_redirect(uri) {
                return Err(format!("redirect uri '{uri}' must be https or loopback"));
            }
        }
        let client_name = client_name.trim();
        if client_name.len() > MAX_CLIENT_NAME_BYTES {
            return Err(format!(
                "client_name is too long ({} bytes, max {MAX_CLIENT_NAME_BYTES})",
                client_name.len()
            ));
        }
        let mut clients = self.clients.lock().unwrap();
        if clients.len() >= CLIENT_CAP {
            return Err(format!(
                "registration is full ({CLIENT_CAP} clients); prune data_dir/oauth.json"
            ));
        }
        let client = Client {
            client_id: random_token(),
            client_name: client_name.to_string(),
            redirect_uris,
            created_at: now_secs(),
        };
        clients.push(client.clone());
        let refresh = self.refresh.lock().unwrap();
        self.persist(&clients, &refresh);
        Ok(client)
    }

    pub fn client(&self, client_id: &str) -> Option<Client> {
        self.clients
            .lock()
            .unwrap()
            .iter()
            .find(|client| client.client_id == client_id)
            .cloned()
    }

    /// The consent decision made flesh: a short-lived, single-use code
    /// binding (client, redirect target, PKCE challenge) to the key
    /// the operator delegated.
    pub fn issue_code(
        &self,
        client: &Client,
        redirect_uri: &str,
        code_challenge: &str,
        delegated_key: &str,
        now: u64,
    ) -> String {
        let code = random_token();
        let mut codes = self.codes.lock().unwrap();
        codes.retain(|grant| grant.expires_at > now);
        codes.push(CodeGrant {
            code_hash: digest_hex(&code),
            client_id: client.client_id.clone(),
            redirect_uri: redirect_uri.to_string(),
            code_challenge: code_challenge.to_string(),
            key: format!("{delegated_key}@{}", slug(&client.client_name)),
            expires_at: now + CODE_TTL_SECS,
        });
        code
    }

    /// Authorization-code exchange: single-use, expiry, client,
    /// redirect binding, and the PKCE proof all checked here.
    pub fn exchange_code(
        &self,
        client_id: &str,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
        now: u64,
    ) -> Result<TokenGrant, OauthError> {
        let grant = {
            let mut codes = self.codes.lock().unwrap();
            let hash = digest_hex(code);
            let position = codes
                .iter()
                .position(|grant| bool::from(grant.code_hash.as_bytes().ct_eq(hash.as_bytes())));
            // Single use: the code leaves the store on FIRST presentation,
            // valid or not — a replayed code must find nothing.
            match position {
                Some(position) => codes.swap_remove(position),
                None => return Err(OauthError("invalid_grant")),
            }
        };
        if grant.expires_at <= now
            || grant.client_id != client_id
            || grant.redirect_uri != redirect_uri
        {
            return Err(OauthError("invalid_grant"));
        }
        // RFC 7636 §4.1 bounds the verifier at 43–128 characters. Checking
        // it here (after the code is spent, so single-use still holds) keeps
        // a malformed proof from reaching the digest at all — belt to the
        // constant-time comparison's suspenders.
        if verifier.len() < 43 || verifier.len() > 128 {
            return Err(OauthError("invalid_grant"));
        }
        let challenge = s256_challenge(verifier);
        if !bool::from(grant.code_challenge.as_bytes().ct_eq(challenge.as_bytes())) {
            return Err(OauthError("invalid_grant"));
        }
        Ok(self.mint(grant.key, client_id, now))
    }

    /// Refresh-token exchange with rotation: the presented token dies,
    /// its replacement is returned alongside a fresh access token.
    pub fn exchange_refresh(
        &self,
        client_id: &str,
        refresh_token: &str,
        now: u64,
    ) -> Result<TokenGrant, OauthError> {
        let hash = digest_hex(refresh_token);
        let grant = {
            let mut refresh = self.refresh.lock().unwrap();
            refresh.retain(|token| token.expires_at > now);
            let position = refresh
                .iter()
                .position(|token| bool::from(token.hash.as_bytes().ct_eq(hash.as_bytes())));
            match position {
                Some(position) => refresh.swap_remove(position),
                None => return Err(OauthError("invalid_grant")),
            }
        };
        if grant.client_id != client_id {
            return Err(OauthError("invalid_grant"));
        }
        Ok(self.mint(grant.key, client_id, now))
    }

    fn mint(&self, key: String, client_id: &str, now: u64) -> TokenGrant {
        let access_token = random_token();
        let refresh_token = random_token();
        {
            let mut access = self.access.lock().unwrap();
            access.retain(|token| token.expires_at > now);
            access.push(AccessToken {
                hash: digest_hex(&access_token),
                key: Arc::from(key.as_str()),
                expires_at: now + ACCESS_TTL_SECS,
            });
        }
        {
            // `clients` before `refresh` — the field-order rule on the
            // struct. Taking `refresh` first here while register_client
            // takes `clients` first is an AB-BA deadlock between the
            // (unauthenticated) registration endpoint and every token
            // grant.
            let clients = self.clients.lock().unwrap();
            let mut refresh = self.refresh.lock().unwrap();
            // The same insert-time sweep as `access` above: grants whose
            // refresh token was simply never used again would otherwise
            // sit in the list (and in every oauth.json rewrite) for the
            // full 30-day TTL.
            refresh.retain(|token| token.expires_at > now);
            refresh.push(RefreshToken {
                hash: digest_hex(&refresh_token),
                key,
                client_id: client_id.to_string(),
                expires_at: now + REFRESH_TTL_SECS,
            });
            self.persist(&clients, &refresh);
        }
        TokenGrant {
            access_token,
            refresh_token,
            expires_in: ACCESS_TTL_SECS,
        }
    }

    /// The resource-server check: an unexpired access token resolves to
    /// the delegated identity. Comparison over hashes, constant-shape
    /// like the keyring scan.
    pub fn authenticate(&self, presented: &str, now: u64) -> Option<Arc<str>> {
        let hash = digest_hex(presented);
        let access = self.access.lock().unwrap();
        let mut matched = None;
        for token in access.iter() {
            if token.expires_at > now && bool::from(token.hash.as_bytes().ct_eq(hash.as_bytes())) {
                matched = Some(Arc::clone(&token.key));
            }
        }
        matched
    }

    /// Best-effort persistence via the same atomic-write path images
    /// use; a failed write costs future grants, never correctness.
    fn persist(&self, clients: &[Client], refresh: &[RefreshToken]) {
        let file = StoreFile {
            clients: clients.to_vec(),
            refresh: refresh.to_vec(),
        };
        let bytes = serde_json::to_vec_pretty(&file).unwrap_or_default();
        if let Err(error) = crate::registry::write_atomic(&self.store_path, &bytes) {
            tracing::warn!(%error, "could not persist the OAuth store");
        }
    }
}

/// RFC 8252 loopback interface redirection: `http://` to the loopback
/// address or `localhost`, any port. Parses the URI and checks the
/// decoded host, never a string prefix — `http://127.0.0.1.evil.example`
/// and `http://localhost.evil.example` both start with the loopback
/// text but resolve to an attacker's domain, not the loopback interface.
fn is_loopback_redirect(uri: &str) -> bool {
    let Ok(parsed) = Url::parse(uri) else {
        return false;
    };
    if parsed.scheme() != "http" {
        return false;
    }
    match parsed.host() {
        Some(Host::Domain(domain)) => domain == "localhost",
        Some(Host::Ipv4(addr)) => addr.is_loopback(),
        Some(Host::Ipv6(addr)) => addr.is_loopback(),
        None => false,
    }
}

/// Client names appear inside delegated-key identities and the access
/// log; keep them to a short, safe alphabet.
fn slug(client_name: &str) -> String {
    let cleaned: String = client_name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(24)
        .collect();
    if cleaned.is_empty() {
        "client".to_string()
    } else {
        cleaned
    }
}

/// Minimal HTML escaping for the consent page (attribute and text
/// positions).
pub fn escape_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_oauth(tag: &str) -> (Oauth, PathBuf) {
        let dir = std::env::temp_dir().join(format!("taguru-oauth-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        (Oauth::open("https://memory.example", &dir), dir)
    }

    fn register(oauth: &Oauth) -> Client {
        oauth
            .register_client("claude", vec!["https://claude.ai/cb".to_string()])
            .unwrap()
    }

    /// RFC 4648 §5 (unpadded) and the RFC 7636 appendix B vector: the
    /// two encodings interoperability lives or dies on.
    #[test]
    fn base64url_and_pkce_match_the_rfc_vectors() {
        assert_eq!(base64url(b""), "");
        assert_eq!(base64url(b"f"), "Zg");
        assert_eq!(base64url(b"fo"), "Zm8");
        assert_eq!(base64url(b"foo"), "Zm9v");
        assert_eq!(base64url(b"foob"), "Zm9vYg");
        assert_eq!(base64url(b"fooba"), "Zm9vYmE");
        assert_eq!(base64url(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64url(&[0xfb, 0xff]), "-_8");

        assert_eq!(
            s256_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn the_code_flow_issues_tokens_and_codes_are_single_use() {
        let (oauth, dir) = scratch_oauth("flow");
        let client = register(&oauth);
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &s256_challenge(verifier),
            "laptop",
            1000,
        );

        // Wrong verifier burns the code (single use is on presentation)...
        // (.map drops the grant: TokenGrant carries secrets and gets no Debug.)
        assert_eq!(
            oauth
                .exchange_code(
                    &client.client_id,
                    &code,
                    "wrong-verifier",
                    "https://claude.ai/cb",
                    1001
                )
                .map(|_| ())
                .unwrap_err(),
            OauthError("invalid_grant")
        );
        // ...so even the right verifier finds nothing afterwards.
        assert!(
            oauth
                .exchange_code(
                    &client.client_id,
                    &code,
                    verifier,
                    "https://claude.ai/cb",
                    1002
                )
                .is_err()
        );

        // A fresh code with the right proof mints the delegated identity.
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &s256_challenge(verifier),
            "laptop",
            2000,
        );
        let grant = oauth
            .exchange_code(
                &client.client_id,
                &code,
                verifier,
                "https://claude.ai/cb",
                2001,
            )
            .unwrap();
        assert_eq!(
            oauth
                .authenticate(&grant.access_token, 2002)
                .unwrap()
                .as_ref(),
            "laptop@claude"
        );
        // Expiry is enforced at the resource side.
        assert!(
            oauth
                .authenticate(&grant.access_token, 2001 + ACCESS_TTL_SECS + 1)
                .is_none()
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn expired_or_misbound_codes_are_refused() {
        let (oauth, dir) = scratch_oauth("bind");
        let client = register(&oauth);
        let verifier = "0123456789012345678901234567890123456789012345";
        let challenge = s256_challenge(verifier);

        let expired = oauth.issue_code(&client, "https://claude.ai/cb", &challenge, "k", 1000);
        assert!(
            oauth
                .exchange_code(
                    &client.client_id,
                    &expired,
                    verifier,
                    "https://claude.ai/cb",
                    1000 + CODE_TTL_SECS
                )
                .is_err()
        );

        let misbound = oauth.issue_code(&client, "https://claude.ai/cb", &challenge, "k", 2000);
        assert!(
            oauth
                .exchange_code(
                    &client.client_id,
                    &misbound,
                    verifier,
                    "https://claude.ai/OTHER",
                    2001
                )
                .is_err()
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_verifier_outside_the_rfc_length_bounds_is_refused() {
        let (oauth, dir) = scratch_oauth("verifier-length");
        let client = register(&oauth);
        // The RFC 7636 §4.1 floor is 43 characters.
        let verifier = "0123456789012345678901234567890123456789012";
        assert_eq!(verifier.len(), 43);

        // Too short (42): refused — and the code still burns on
        // presentation, so a length-failed proof cannot be retried.
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &s256_challenge(verifier),
            "k",
            1000,
        );
        assert_eq!(
            oauth
                .exchange_code(
                    &client.client_id,
                    &code,
                    &verifier[..42],
                    "https://claude.ai/cb",
                    1001
                )
                .map(|_| ())
                .unwrap_err(),
            OauthError("invalid_grant")
        );
        assert!(
            oauth
                .exchange_code(
                    &client.client_id,
                    &code,
                    verifier,
                    "https://claude.ai/cb",
                    1002
                )
                .is_err(),
            "the code was spent on first presentation"
        );

        // Too long (129): refused before the digest even though this
        // verifier's own challenge would otherwise match.
        let long = "a".repeat(129);
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &s256_challenge(&long),
            "k",
            2000,
        );
        assert_eq!(
            oauth
                .exchange_code(
                    &client.client_id,
                    &code,
                    &long,
                    "https://claude.ai/cb",
                    2001
                )
                .map(|_| ())
                .unwrap_err(),
            OauthError("invalid_grant")
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Rotation: a refresh token works exactly once, its replacement
    /// carries on, and the whole chain survives a process restart.
    #[test]
    fn refresh_tokens_rotate_and_survive_restart() {
        let (oauth, dir) = scratch_oauth("refresh");
        let client = register(&oauth);
        let verifier = "0123456789012345678901234567890123456789012345";
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &s256_challenge(verifier),
            "laptop",
            1000,
        );
        let first = oauth
            .exchange_code(
                &client.client_id,
                &code,
                verifier,
                "https://claude.ai/cb",
                1001,
            )
            .unwrap();

        // "Restart": a new instance over the same directory.
        let reopened = Oauth::open("https://memory.example", dir.as_path());
        // Access tokens were memory-only; the refresh grant persisted.
        assert!(reopened.authenticate(&first.access_token, 1002).is_none());
        let second = reopened
            .exchange_refresh(&client.client_id, &first.refresh_token, 1003)
            .unwrap();
        assert_eq!(
            reopened
                .authenticate(&second.access_token, 1004)
                .unwrap()
                .as_ref(),
            "laptop@claude"
        );
        // The presented refresh token rotated out.
        assert!(
            reopened
                .exchange_refresh(&client.client_id, &first.refresh_token, 1005)
                .is_err()
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn registration_validates_redirects_and_caps_the_store() {
        let (oauth, dir) = scratch_oauth("register");
        assert!(oauth.register_client("x", vec![]).is_err());
        assert!(
            oauth
                .register_client("x", vec!["http://evil.example/cb".to_string()])
                .is_err()
        );
        assert!(
            oauth
                .register_client("x", vec!["http://127.0.0.1:7777/cb".to_string()])
                .is_ok()
        );
        assert!(
            oauth
                .register_client("x", vec!["http://localhost:7777/cb".to_string()])
                .is_ok()
        );
        // A substring match on the prefix would wrongly accept these:
        // the host is an attacker-controlled domain, not the loopback
        // interface, even though it starts with the loopback text.
        assert!(
            oauth
                .register_client("x", vec!["http://127.0.0.1.evil.example/cb".to_string()])
                .is_err()
        );
        assert!(
            oauth
                .register_client("x", vec!["http://localhost.evil.example/cb".to_string()])
                .is_err()
        );
        for i in 0..CLIENT_CAP {
            let _ = oauth.register_client(&format!("c{i}"), vec!["https://ok.example".to_string()]);
        }
        assert!(
            oauth
                .register_client("one-too-many", vec!["https://ok.example".to_string()])
                .is_err()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn registration_bounds_name_and_redirect_sizes() {
        let (oauth, dir) = scratch_oauth("register-bounds");
        let ok = vec!["https://ok.example/cb".to_string()];

        // Unauthenticated input must not carry a body-sized label into
        // the persisted store.
        assert!(
            oauth
                .register_client(&"n".repeat(MAX_CLIENT_NAME_BYTES + 1), ok.clone())
                .is_err()
        );
        // Too many redirect URIs, even when each is valid.
        let flood: Vec<String> = (0..=MAX_REDIRECT_URIS)
            .map(|i| format!("https://ok.example/cb{i}"))
            .collect();
        assert!(oauth.register_client("x", flood).is_err());
        // A single over-long redirect URI.
        let long_uri = format!("https://ok.example/{}", "p".repeat(MAX_REDIRECT_URI_BYTES));
        assert!(oauth.register_client("x", vec![long_uri]).is_err());
        // The generous ceilings still admit an ordinary registration.
        assert!(
            oauth
                .register_client(&"n".repeat(MAX_CLIENT_NAME_BYTES), ok)
                .is_ok()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn slugs_and_escaping_defang_client_names() {
        assert_eq!(slug("Claude Web (personal)"), "ClaudeWebpersonal");
        assert_eq!(slug("<script>"), "script");
        assert_eq!(slug("日本語のみ"), "client");
        assert_eq!(
            escape_html(r#"<b a="x">&'"#),
            "&lt;b a=&quot;x&quot;&gt;&amp;&#39;"
        );
    }

    /// mint() must sweep expired refresh grants before appending, the
    /// same insert-time pruning every other store here gets — an
    /// unswept list grows (and is re-persisted on every grant) for the
    /// full 30-day TTL.
    #[test]
    fn expired_refresh_tokens_are_swept_on_the_next_mint() {
        let (oauth, dir) = scratch_oauth("refresh-sweep");
        let client = register(&oauth);
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        // One grant at t=0 (its refresh token expires at REFRESH_TTL)…
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &s256_challenge(verifier),
            "laptop",
            0,
        );
        oauth
            .exchange_code(
                &client.client_id,
                &code,
                verifier,
                "https://claude.ai/cb",
                0,
            )
            .unwrap();
        // …and a second minted after the first expired.
        let later = REFRESH_TTL_SECS + 1;
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &s256_challenge(verifier),
            "laptop",
            later,
        );
        oauth
            .exchange_code(
                &client.client_id,
                &code,
                verifier,
                "https://claude.ai/cb",
                later,
            )
            .unwrap();

        let stored: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("oauth.json")).unwrap()).unwrap();
        assert_eq!(
            stored["refresh"].as_array().unwrap().len(),
            1,
            "the expired grant must be swept, not re-persisted"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `register_client` persists under `clients` + `refresh`; `mint`
    /// (here via the refresh grant) persists under the same pair. If
    /// the two sides ever disagree on acquisition order again, this
    /// interleaving deadlocks — the watchdog turns that hang into a
    /// named failure instead of a suite-wide timeout.
    #[test]
    fn concurrent_registration_and_token_grants_do_not_deadlock() {
        let (oauth, dir) = scratch_oauth("lock-order");
        let client = register(&oauth);
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &s256_challenge(verifier),
            "laptop",
            0,
        );
        let seed = oauth
            .exchange_code(
                &client.client_id,
                &code,
                verifier,
                "https://claude.ai/cb",
                0,
            )
            .unwrap();

        let oauth = std::sync::Arc::new(oauth);
        let registrar = {
            let oauth = std::sync::Arc::clone(&oauth);
            std::thread::spawn(move || {
                // Stays under CLIENT_CAP: a capped registration refuses
                // before touching the lock pair this test exercises.
                for _ in 0..80 {
                    oauth
                        .register_client("claude", vec!["https://claude.ai/cb".to_string()])
                        .unwrap();
                }
            })
        };
        let refresher = {
            let oauth = std::sync::Arc::clone(&oauth);
            let client_id = client.client_id.clone();
            std::thread::spawn(move || {
                let mut token = seed.refresh_token;
                for _ in 0..80 {
                    token = oauth
                        .exchange_refresh(&client_id, &token, 0)
                        .unwrap()
                        .refresh_token;
                }
            })
        };
        let (done, watchdog) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let clean = registrar.join().is_ok() && refresher.join().is_ok();
            let _ = done.send(clean);
        });
        match watchdog.recv_timeout(std::time::Duration::from_secs(60)) {
            Ok(clean) => assert!(clean, "a worker thread panicked"),
            Err(_) => panic!("registration deadlocked against a token grant"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}
