# 0055. An API key's name is its id, held to a log-safe character set

- **Status**: Accepted
- **Date**: 2026-11-16
- **Issue**: #1049 (decisions recorded on the issue, 2026-11-16; under #958)
- **Related**: ADR 0045 (an id apart from the name), ADR 0054 (a source's
  id is the string its caller chose), #851 (ID column rule)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

ADR 0045 separated a display `name` from a server-assigned `id` for
contexts, groups and sources (ADR 0054 later kept sources on the caller's
string). It never mentioned API keys, yet a key's name
(`TAGURU_API_TOKENS=name:token,…`, or `default` for `TAGURU_API_TOKEN`)
is also a name doing an id's job:

- It is the join key for `TAGURU_KEY_GRANTS`, the per-key rate-limit
  bucket, and `oauth.json`'s delegation identities (`key@client`).
- Hot reload matches keys across reloads by name: the same name with new
  bytes is a rotation, a new name is a removal plus an addition.
- It is written verbatim into the access log's `key` field, write audit
  lines, and 403/429 message texts.

ADR 0045's problems mostly do not apply. The name never appears in a URL
path or a JSON field (only in error message text), so there is nothing
unreadable to fix. Renaming a key is rare and already visible: the grant
must be renamed in the same edit or the reload is refused, and OAuth
clients consent again. A separate id cannot be minted by the server
either, because the operator writes the keyring in configuration: an id
would need new config syntax (`id:name:token`) or persisted state, for
every deployment, to make a rare rename cheaper.

What does apply is the lack of bounds. The parser accepted any UTF-8
without `@`, `,` or `:` at any length, and that string reaches log lines
and response bodies unescaped.

## 2. Decision

1. **A key's name is its id.** The operator chooses it; the server never
   mints one, and there is no separate display name. Grants, rate limits,
   OAuth delegations, the access log and audit lines keep keying on it.
2. **A key name is 1–64 bytes of `[A-Za-z0-9._-]`.** Anything else is a
   boot (and hot-reload) refusal, naming the entry by its position, never
   by its content, since an invalid name is as likely a token pasted in
   the wrong place. The set excludes `@`, which stays reserved for OAuth
   delegations (`key@client`).
3. **`default` is unchanged.** It is the implicit name of
   `TAGURU_API_TOKEN`; a `TAGURU_API_TOKENS` entry named `default` is
   refused only when both variables are set, by the existing duplicate
   check.
4. **403 and 429 bodies keep naming the key.** The caller holds that key,
   so its name is no secret to them, and naming it is what makes a
   misconfigured grant diagnosable.

## 3. Consequences

- No wire change: `http_contract` and the SDKs are untouched.
- A deployment whose key names fall outside the set refuses to boot (a
  hot reload keeps the previous table) until the names are changed — and
  renaming a key means renaming its `TAGURU_KEY_GRANTS` entry in the same
  edit, while OAuth clients delegated under the old name consent again.
- Log lines and error texts carrying a key name need no escaping.
