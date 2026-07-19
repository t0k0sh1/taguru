use std::collections::BTreeMap;
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::registry::AppState;

use super::{
    AppJson, AppPath, AppQuery, ErrorCode, MAX_ASSOCIATIONS_PER_REQUEST, MAX_MATCH_LIMIT,
    MAX_NAME_BYTES, access_error, clamp, deadline_exceeded, empty, error, key_name, ok, overlong,
    oversized, partial_write_error,
};

/// One name or several: query positions accept `"住所"` and
/// `["住所", "職歴"]` interchangeably.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

pub(super) fn as_refs(position: &Option<OneOrMany>) -> Vec<&str> {
    match position {
        None => Vec::new(),
        Some(OneOrMany::One(name)) => vec![name.as_str()],
        Some(OneOrMany::Many(names)) => names.iter().map(String::as_str).collect(),
    }
}

/// The shared cap on a query's three positions: subject, label, and
/// object are each list-shaped read input, bounded like every other
/// list ([`overlong`]) — `query` and `cross_query` run one check.
fn overlong_positions(
    subject: &Option<OneOrMany>,
    label: &Option<OneOrMany>,
    object: &Option<OneOrMany>,
    started_at: Instant,
) -> Option<Response> {
    [("subject", subject), ("label", label), ("object", object)]
        .into_iter()
        .find_map(|(field, position)| overlong(field, as_refs(position).len(), started_at))
}

/// Refuses a query that pins nothing at all: subject, label, and
/// object empty together would materialize and rank every edge in the
/// context (or, cross-context, every edge in every named target)
/// before the limit ever trims it — treated as a client bug, not a
/// deliberate "give me everything", the same stance [`cross_targets`]
/// takes on an empty `contexts`/`groups` pair.
fn empty_positions(
    subject: &Option<OneOrMany>,
    label: &Option<OneOrMany>,
    object: &Option<OneOrMany>,
    started_at: Instant,
) -> Option<Response> {
    let nothing_pinned =
        as_refs(subject).is_empty() && as_refs(label).is_empty() && as_refs(object).is_empty();
    nothing_pinned.then(|| {
        error(
            ErrorCode::InvalidArgument,
            "'subject', 'label', or 'object' must pin at least one value",
            started_at,
        )
    })
}

/// The full position gate `query` and `cross_query` both run before
/// touching the index: length cap first, then "at least one pinned".
pub(super) fn validate_positions(
    subject: &Option<OneOrMany>,
    label: &Option<OneOrMany>,
    object: &Option<OneOrMany>,
    started_at: Instant,
) -> Option<Response> {
    overlong_positions(subject, label, object, started_at)
        .or_else(|| empty_positions(subject, label, object, started_at))
}

/// Alias registrations, alias → canonical per namespace. Applied in
/// sorted order (BTreeMap), aborting at the first failure with the
/// applied count reported — like association batches, each item is
/// all-or-nothing in the library.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct AliasRequest {
    pub concepts: BTreeMap<String, String>,
    pub labels: BTreeMap<String, String>,
}

/// One page of the alias table: `total` counts BOTH namespaces before
/// paging, the maps carry this page's entries. The page cursor spans
/// the namespaces in order — concepts (sorted by alias), then labels —
/// so the next page starts after the last entry shown:
/// `?after=concept:<alias>` or `?after=label:<alias>`.
#[derive(Serialize)]
pub struct AliasExport {
    pub total: usize,
    pub concepts: BTreeMap<String, String>,
    pub labels: BTreeMap<String, String>,
}

/// `?limit=&after=` — the keyset page every unbounded listing takes
/// (default and ceiling [`MAX_MATCH_LIMIT`], like the directory).
/// `prefix` narrows the population of interest itself (like `total` on
/// the search endpoints, not a cursor) — endpoints that support it
/// apply it before counting `total`; `list_groups` ignores it.
#[derive(Debug, Deserialize)]
pub struct KeysetQuery {
    pub limit: Option<usize>,
    pub after: Option<String>,
    pub prefix: Option<String>,
}

pub async fn add_aliases(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<AliasRequest>,
) -> Response {
    let started_at = Instant::now();
    // Refused before the write lock like an association batch — each
    // pair is one WAL op in a single lock-hold/fsync, the same cost
    // shape MAX_ASSOCIATIONS_PER_REQUEST bounds there.
    let pairs = request.concepts.len() + request.labels.len();
    if pairs > MAX_ASSOCIATIONS_PER_REQUEST {
        return error(
            ErrorCode::OverLimit,
            format!(
                "batch of {pairs} aliases exceeds the per-request limit of \
                 {MAX_ASSOCIATIONS_PER_REQUEST}; split the ingest"
            ),
            started_at,
        );
    }
    // Aliases intern names on both sides; the same cap as every other
    // name-shaped write.
    for (namespace, pairs) in [("concepts", &request.concepts), ("labels", &request.labels)] {
        for (alias, canonical) in pairs {
            for (role, value) in [("alias", alias.as_str()), ("canonical", canonical.as_str())] {
                if let Some(refusal) = oversized(
                    &format!("a {namespace} {role}"),
                    value,
                    MAX_NAME_BYTES,
                    started_at,
                ) {
                    return refusal;
                }
                // An empty spelling is worse than an unaddressable
                // name: `str::contains("")` is always true, so a
                // zero-length alias would containment-match every cue
                // and plant a phantom hit in every resolution from
                // then on.
                if let Some(refusal) = empty(&format!("a {namespace} {role}"), value, started_at) {
                    return refusal;
                }
            }
        }
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Same fsync-bearing WAL write as add_associations; keep it off the
    // async worker.
    match tokio::task::block_in_place(|| {
        state.add_aliases(&name, &request.concepts, &request.labels)
    }) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(Ok(applied)) => {
            // Counts (not the spellings — a batch may run to
            // thousands) reach the audit line, the access log names
            // the context and key. Unconditional, symmetric with
            // remove_aliases: even an empty batch (applied == 0)
            // leaves a line, so an operator reconstructing a bad
            // alias's live window sees every registration attempt.
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                concepts = request.concepts.len(),
                labels = request.labels.len(),
                applied,
                "aliases registered",
            );
            // Same rule as add_associations: an empty batch applies
            // nothing, so it must not bump the write counter either.
            if applied > 0 {
                state.note_write(&name);
            }
            ok(applied, started_at)
        }
        Ok(Err(partial)) => {
            partial_write_error(&state, &name, partial, started_at, |applied, message| {
                format!("applied {applied} aliases, then {message}")
            })
        }
    }
}

/// Alias withdrawals — the exact registered spellings, per namespace.
/// Withdrawal is the undo for a mis-registered alias: the spelling
/// stops resolving and is free to register again; canonicals and
/// edges are untouched, and a canonical name is refused (removal
/// must never unname a record).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct RemoveAliasesRequest {
    pub concepts: Vec<String>,
    pub labels: Vec<String>,
}

pub async fn remove_aliases(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<RemoveAliasesRequest>,
) -> Response {
    let started_at = Instant::now();
    // An empty withdrawal is a malformed request, not a silent no-op.
    if request.concepts.is_empty() && request.labels.is_empty() {
        return error(
            ErrorCode::InvalidArgument,
            "the request names no aliases to remove",
            started_at,
        );
    }
    // Same WAL-op-per-item cost shape as add_aliases; same cap, and the
    // same `OverLimit` code add_aliases uses for the over-cap batch — an
    // oversized request is not a malformed one.
    let names = request.concepts.len() + request.labels.len();
    if names > MAX_ASSOCIATIONS_PER_REQUEST {
        return error(
            ErrorCode::OverLimit,
            format!(
                "batch of {names} alias removals exceeds the per-request limit of \
                 {MAX_ASSOCIATIONS_PER_REQUEST}; split the request"
            ),
            started_at,
        );
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Same fsync-bearing WAL write; keep it off the async worker.
    match tokio::task::block_in_place(|| {
        state.remove_aliases(&name, &request.concepts, &request.labels)
    }) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(Ok(removed)) => {
            // Withdrawn spellings live in the body; counts (not the
            // spellings — a batch may run to thousands) reach the
            // audit line, the access log names the context and key.
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                concepts = request.concepts.len(),
                labels = request.labels.len(),
                removed,
                "aliases removed",
            );
            state.note_write(&name);
            ok(removed, started_at)
        }
        // `full` is unreachable for removals (they free, never fill),
        // but the shared mapping stays uniform.
        Ok(Err(partial)) => {
            partial_write_error(&state, &name, partial, started_at, |applied, message| {
                format!("removed {applied} aliases, then {message}")
            })
        }
    }
}

pub async fn list_aliases(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppQuery(query): AppQuery<KeysetQuery>,
) -> Response {
    let started_at = Instant::now();
    // The cursor names a namespace and an alias; anything else is a
    // malformed request, not an empty page.
    let after = match query.after.as_deref() {
        None => None,
        Some(cursor) => match cursor.split_once(':') {
            Some((kind @ ("concept" | "label"), alias)) => Some((kind == "label", alias)),
            _ => {
                return error(
                    ErrorCode::InvalidArgument,
                    "after must be 'concept:<alias>' or 'label:<alias>' — the last \
                     entry of the previous page",
                    started_at,
                );
            }
        },
    };
    let limit = clamp(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    // Namespace order is concepts-then-labels, same as the wire cursor's
    // `(is_label, alias)` ordering. A cursor already inside the label
    // namespace means every concept alias is behind it, so the concept
    // seek is skipped outright rather than run and immediately discarded.
    let (concept_after, label_after, skip_concepts) = match after {
        None => (None, None, false),
        Some((false, alias)) => (Some(alias), None, false),
        Some((true, alias)) => (None, Some(alias), true),
    };
    // A `prefix` filter forces the whole-namespace scan below; a bare
    // cursor stays on the cheap BTreeMap-seeking path regardless.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // A `prefix` filter defines the population rather than a cursor, so
    // — like `pinned` on `list_contexts` — it forces the whole-namespace
    // path: the BTreeMap-seeking `*_alias_page` fast path has no way to
    // know in advance how many prefix-matching aliases lie within any
    // given range. That path clones and sorts every concept and label
    // alias in the context — the same unconditional whole-table cost as
    // `unreachable_from`'s full scan — so it runs under block_in_place;
    // the bare-cursor path below stays off it, same as
    // `recall`/`describe`, since its BTreeMap seek is already bounded by
    // `limit`.
    let outcome = match query.prefix.as_deref() {
        Some(prefix) => tokio::task::block_in_place(|| {
            state.read_context(&name, |context| {
                let mut concepts: Vec<(String, String)> = context
                    .concept_aliases()
                    .into_iter()
                    .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                    .collect();
                concepts.sort();
                concepts.retain(|(alias, _)| alias.starts_with(prefix));
                let mut labels: Vec<(String, String)> = context
                    .label_aliases()
                    .into_iter()
                    .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                    .collect();
                labels.sort();
                labels.retain(|(alias, _)| alias.starts_with(prefix));
                let total = concepts.len() + labels.len();
                // One ordered sequence — concepts, then labels — filtered
                // past the cursor and cut at the limit, then split back
                // into maps.
                let page: Vec<(bool, (String, String))> = concepts
                    .into_iter()
                    .map(|entry| (false, entry))
                    .chain(labels.into_iter().map(|entry| (true, entry)))
                    .filter(|(is_label, (alias, _))| match after {
                        None => true,
                        Some((after_is_label, after_alias)) => {
                            (*is_label, alias.as_str()) > (after_is_label, after_alias)
                        }
                    })
                    .take(limit)
                    .collect();
                let mut export = AliasExport {
                    total,
                    concepts: BTreeMap::new(),
                    labels: BTreeMap::new(),
                };
                for (is_label, (alias, canonical)) in page {
                    if is_label {
                        export.labels.insert(alias, canonical);
                    } else {
                        export.concepts.insert(alias, canonical);
                    }
                }
                export
            })
        }),
        None => state.read_context(&name, |context| {
            let total = context.concept_alias_count() + context.label_alias_count();
            let mut export = AliasExport {
                total,
                concepts: BTreeMap::new(),
                labels: BTreeMap::new(),
            };
            let mut remaining = limit;
            if !skip_concepts {
                let (_, page) = context.concept_alias_page(concept_after, remaining);
                remaining -= page.len();
                export.concepts.extend(page);
            }
            // A concept page shorter than what was asked for means that
            // namespace ran dry, so the leftover budget spills into labels,
            // started fresh — reached only when `label_after` is `None`,
            // since a label-namespace cursor takes the `skip_concepts`
            // branch above and never sets `remaining` here.
            if remaining > 0 {
                let (_, page) = context.label_alias_page(label_after, remaining);
                export.labels.extend(page);
            }
            export
        }),
    };
    match outcome {
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}
