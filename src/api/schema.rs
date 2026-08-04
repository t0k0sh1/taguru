//! `GET`/`PUT /contexts/{name}/schema` (#380, S2 of #218's ADR 0009
//! split) — the management routes over the schema document S1 (#379)
//! already knows how to validate and persist — plus `POST .../schema/audit`
//! and `POST .../schema/validate` (#385, S7, ADR 0009 §10): the two
//! read-only routes that DO judge the graph, sharing [`schema_audit`]
//! with each other and `schema_issues`/[`SchemaEnv`] with every write
//! entrance (S3, #381), so this module's own reading of "domain/range
//! violation" can never drift from what `strict` would actually refuse.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;
use serde::{Deserialize, Serialize};

use taguru::context::Association;
use taguru::deadline::Deadline;

use crate::metrics::ErrorKind;
use crate::registry::{AccessError, AppState, AssocOp, PutSchemaError};
use crate::schema::{
    self, IssuePath, SCHEMA_TYPE_LABEL, SchemaCheckInput, SchemaDocument, SchemaEnv, schema_issues,
};

use super::{
    AppBytes, AppJson, AppPath, AssociationOut, ErrorCode, Issue, MatchCursor, access_error,
    association_out, deadline_exceeded, error, key_name, locator_keys, not_found, ok,
    optional_body, page_by,
};

/// One directory row by name's schema-document twin — the resident
/// document as `install`ed, or a 404 distinguishing "this context has
/// never installed a schema" ([`ErrorCode::NoSchema`]) from "this
/// context does not exist" ([`ErrorCode::NoContext`]), the boundary ADR
/// 0009 §6.3 requires stay load-bearing rather than collapsing both
/// into one shape.
pub async fn get_schema(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // `schema_of` can load the full context (a cold entry whose schema
    // is recorded but not yet resolved locally) — real disk IO, so it
    // steps off the async worker like every other mutating or
    // load-bearing handler.
    match tokio::task::block_in_place(|| state.schema_of(&name)) {
        None => not_found(&name, started_at),
        Some(Ok(Some(installed))) => ok(installed.document(), started_at),
        Some(Ok(None)) => error(
            ErrorCode::NoSchema,
            format!("context '{name}' has no schema document"),
            started_at,
        ),
        Some(Err(message)) => {
            // The detail (which can name the data directory's own
            // filesystem path — `schema::load_schema`'s digest-mismatch
            // and unreadable-file messages both do) is for the operator,
            // not an authenticated HTTP client: logged here, never
            // forwarded into the response.
            tracing::warn!(context = %name, error = %message, "schema load failed");
            state.metrics().record_error(ErrorKind::Load);
            error(
                ErrorCode::Internal,
                format!("context '{name}' schema could not be loaded — see server logs"),
                started_at,
            )
        }
    }
}

/// Installs `body` as `name`'s schema document wholesale — there is no
/// delta form, so a retry after any failure below is always safe
/// regardless of which side of it the previous attempt reached (ADR
/// 0009 §5.2). `body` is `#[serde(deny_unknown_fields)]` on
/// [`SchemaDocument`] itself, so an unread field or an unknown `mode`
/// spelling is a 400 from [`AppJson`] before this handler even runs;
/// [`schema::install`] catches everything shape-valid-but-refused
/// (version, caps, `is_a` cycles/depth, the reserved relation label).
pub async fn put_schema(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(document): AppJson<SchemaDocument>,
) -> Response {
    let started_at = Instant::now();
    let installed = match schema::install(document) {
        Ok(installed) => installed,
        Err(violation) => {
            return error(
                ErrorCode::InvalidArgument,
                violation.to_string(),
                started_at,
            );
        }
    };
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Writes the sidecar (fsync + rename) then the schema file itself
    // (also fsync + rename), and may load the context first to check
    // its live label-alias table; keep all of that off the async
    // worker like every other mutating endpoint.
    match tokio::task::block_in_place(|| state.put_schema(&name, installed)) {
        None => not_found(&name, started_at),
        Some(Ok(document)) => {
            // Every destructive-ish operator action leaves one
            // self-contained `taguru::audit` line — who, what, to which
            // object — mirroring `delete_context`/`rename_context`.
            // Installing (or replacing) a schema is not destructive to
            // the graph, but it does change what `strict` refuses from
            // this point on, which is exactly the kind of change an
            // incident review wants to find by grepping one target.
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                mode = document.mode.as_str(),
                "context schema installed",
            );
            ok(document, started_at)
        }
        Some(Err(PutSchemaError::ReservedAlias(alias))) => error(
            ErrorCode::InvalidArgument,
            format!(
                "label alias '{alias}' resolves to '{}', the relation label reserved for \
                 type assertions (ADR 0009 §6.3) — rename the alias before installing this \
                 schema",
                schema::SCHEMA_TYPE_LABEL
            ),
            started_at,
        ),
        Some(Err(PutSchemaError::Load(message))) => {
            // Same posture as `get_schema`'s Load arm just above: the
            // detail can name a filesystem path, so it is logged, not
            // returned.
            tracing::warn!(context = %name, error = %message, "schema load failed");
            state.metrics().record_error(ErrorKind::Load);
            error(
                ErrorCode::Internal,
                format!("context '{name}' could not be loaded — see server logs"),
                started_at,
            )
        }
        Some(Err(PutSchemaError::Io(io_error))) => {
            tracing::warn!(context = %name, error = %io_error, "schema write failed");
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
                format!("context '{name}' schema not persisted — see server logs"),
                started_at,
            )
        }
    }
}

/// Ceiling on how many entries each of [`SchemaAudit`]'s four small
/// sections lists — `total` still reports the true count past this cap,
/// the same "count everything, list a bounded prefix" contract
/// [`crate::api::MAX_LISTED_ISSUES`] applies to a validation refusal's
/// issue list, sized for a context whose live vocabulary can be far
/// larger than one audit response should carry.
const MAX_AUDIT_NAMES: usize = 100;

/// One of [`SchemaAudit`]'s three name-list sections: the true count of
/// names this check surfaced, and a name-ordered prefix of them capped
/// at [`MAX_AUDIT_NAMES`].
#[derive(Debug, Serialize)]
pub struct AuditNames {
    pub total: usize,
    pub names: Vec<String>,
}

fn audit_names(names: BTreeSet<String>) -> AuditNames {
    let total = names.len();
    AuditNames {
        total,
        names: names.into_iter().take(MAX_AUDIT_NAMES).collect(),
    }
}

/// [`AuditNames`]'s sibling for `reserved_alias_conflicts` (`alias ->
/// canonical`, not a bare name set) — capped the same way at
/// [`MAX_AUDIT_NAMES`]. An operator can in principle register many
/// aliases resolving to `schema:type` before ever installing a schema
/// (ADR 0009 §6.3 guard 1 leaves the label ordinary until then), so this
/// section is bounded defensively like every other one here, even
/// though a well-behaved server keeps it empty once a schema exists —
/// `PUT /schema`'s own migration-boundary guard already refuses to
/// install over a pre-existing conflict.
#[derive(Debug, Serialize)]
pub struct AuditAliases {
    pub total: usize,
    pub aliases: BTreeMap<String, String>,
}

fn audit_aliases(aliases: BTreeMap<String, String>) -> AuditAliases {
    let total = aliases.len();
    AuditAliases {
        total,
        aliases: aliases.into_iter().take(MAX_AUDIT_NAMES).collect(),
    }
}

/// One live association [`schema_issues`] flagged, alongside every issue
/// it raised. An audit finding travels with the edge it is about,
/// unlike a write entrance's issues, which travel with a request
/// coordinate (see [`IssuePath::Request`]) instead — this route uses
/// [`IssuePath::Edge`].
#[derive(Debug, Serialize)]
pub struct SchemaViolationOut {
    pub association: AssociationOut,
    pub issues: Vec<Issue>,
}

/// ADR 0009 §10's schema audit: five independent read-only checks over
/// the live graph in one response, in `DriftAudit`'s shape
/// (`src/api/vocabulary.rs`) — only `violations`/`total` page (the same
/// `page_by` rank every other match list uses); the other four
/// sections are small enough in practice to return whole, each capped at
/// [`MAX_AUDIT_NAMES`] with its own true count. Framed exactly like
/// `audit_vocabulary`'s own doc (`src/api/vocabulary.rs:38-41`):
/// candidates for review, not verdicts — this audit never auto-applies a
/// fix.
#[derive(Debug, Serialize)]
pub struct SchemaAudit {
    /// `violations`' count before `limit`/`after` — constant across
    /// pages, the same convention every other paged section in this API
    /// uses.
    pub total: usize,
    pub violations: Vec<SchemaViolationOut>,
    /// Live concepts asserting no `schema:type` of their own, excluding
    /// concepts that are themselves asserted type names (ADR 0009 §6.3
    /// exclusion 3's own reasoning: `Brewery` reading as "untyped" would
    /// be noise, not signal) and every concept already declared in
    /// `types` (a declared type with no `schema:type` assertion of its
    /// own is not what this section is about). Reported unconditionally
    /// — untyped concepts are legal in every mode, §6.1 — as a candidate
    /// list only.
    pub untyped_concepts: AuditNames,
    /// Type names actually asserted (the object of some live
    /// `schema:type` edge) but absent from the document's `types` map —
    /// always reported, unconditional on `closed_labels` (§6.2): an
    /// undeclared type is never a violation, only a signal a schema
    /// author may want to see.
    pub undeclared_types: AuditNames,
    /// Live relation labels with no entry in `relations` — populated
    /// only when the document sets `closed_labels` (§6.4), since the
    /// question is meaningless otherwise. `schema:type` itself never
    /// appears here even when `closed_labels` is set (§6.4's own
    /// carve-out for it). The same fact also drives a `violations` entry
    /// on every individual edge carrying such a label; this section
    /// answers a different question — "what vocabulary would `relations`
    /// need to gain" — not "which edges would `strict` refuse."
    pub unknown_labels: AuditNames,
    /// ADR 0009 §6.3 guard 2's install-time bullet, read back: every
    /// already-persisted label alias whose canonical spelling is the
    /// reserved `schema:type` label (`alias → canonical`). Only possible
    /// for an alias created before this context ever had a schema — `PUT
    /// /schema` itself refuses to install over one, so this section
    /// reads empty for any context whose schema installed cleanly;
    /// non-empty here means the next `PUT /schema` (this document or any
    /// other) will refuse until the alias is renamed.
    pub reserved_alias_conflicts: AuditAliases,
}

/// Request body for `POST /contexts/{name}/schema/audit` — an absent
/// body means every default below; paging over `violations` follows the
/// same contract `DriftAuditRequest` (`src/api/vocabulary.rs`) applies to
/// `unsourced`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct SchemaAuditRequest {
    /// Omitted means 100, capped at 1000 — pages exactly like recall,
    /// query, and every other match list.
    pub limit: Option<usize>,
    /// Resume past a previous page's last violation — see [`MatchCursor`].
    pub after: Option<MatchCursor>,
}

/// Request body for `POST /contexts/{name}/schema/validate` — the
/// *proposed* document to dry-run, alongside the same paging fields
/// [`SchemaAuditRequest`] carries. Wrapped rather than reusing `PUT
/// /schema`'s bare-document body: [`SchemaDocument`] is itself
/// `deny_unknown_fields`, so a sibling `limit`/`after` could never sit
/// beside it there without changing that struct's own wire contract.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaValidateRequest {
    pub document: SchemaDocument,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub after: Option<MatchCursor>,
}

/// What [`schema_audit`]'s `read_context` closure computes. Paging and
/// marker resolution both need the registry (`AppState`), unreachable
/// from inside the closure, so this travels back out whole and is
/// finished outside `read_context` — the same shape `audit_drift`
/// (`src/api/vocabulary.rs`) already uses for its own two sweeps.
struct RawAudit {
    violations: Vec<(Association, Vec<Issue>)>,
    untyped_concepts: BTreeSet<String>,
    undeclared_types: BTreeSet<String>,
    unknown_labels: BTreeSet<String>,
    reserved_alias_conflicts: BTreeMap<String, String>,
}

/// The one function [`audit_schema`] and [`validate_schema`] both call —
/// `schema` is already installed either way (a resident document for the
/// former, `schema::install`'s output for the latter, never persisted),
/// so from here on the two routes cannot answer differently for the same
/// document. ADR 0009 §10 fixes this shared shape in its own text: "both
/// routes are O(edges) with no cheap variant."
fn schema_audit(
    state: &AppState,
    name: &str,
    schema: Arc<schema::InstalledSchema>,
    limit: Option<usize>,
    after: Option<&MatchCursor>,
    deadline: Deadline,
) -> Result<SchemaAudit, AccessError> {
    // ADR 0009 §7.1: "pre-existing violations are visible only through
    // the explicitly-invoked, read-only audit" — the whole reason this
    // route exists is to surface what `strict` WOULD refuse, regardless
    // of what mode the document is actually in right now.
    // `schema_issues` only branches on `mode` to skip work when it is
    // `Off`; forcing it to `Strict` here is exactly "judge every op,"
    // nothing more (see [`schema::InstalledSchema::enforcing`]).
    let enforcing = schema::InstalledSchema::enforcing(&schema);

    let raw = state
        .read_context(name, |context| -> Result<RawAudit, AccessError> {
            let live: Vec<Association> = context
                .all_associations(deadline)
                .map_err(|_| AccessError::DeadlineExceeded)?
                .into_iter()
                .filter(|association| association.count > 0)
                .collect();

            let mut live_concepts: BTreeSet<String> = BTreeSet::new();
            let mut live_labels: BTreeSet<String> = BTreeSet::new();
            let mut typed_concepts: BTreeSet<String> = BTreeSet::new();
            let mut asserted_type_names: BTreeSet<String> = BTreeSet::new();
            let mut fact_edges: Vec<Association> = Vec::with_capacity(live.len());
            let mut fact_ops: Vec<AssocOp> = Vec::with_capacity(live.len());

            for association in live {
                live_concepts.insert(association.subject.clone());
                live_concepts.insert(association.object.clone());
                live_labels.insert(association.label.clone());
                if association.label == SCHEMA_TYPE_LABEL {
                    typed_concepts.insert(association.subject.clone());
                    asserted_type_names.insert(association.object.clone());
                } else {
                    fact_ops.push(AssocOp {
                        subject: association.subject.clone(),
                        label: association.label.clone(),
                        object: association.object.clone(),
                        weight: association.weight,
                        source: None,
                        paragraph: None,
                    });
                    fact_edges.push(association);
                }
            }

            // ADR 0009 §6.3 guard 2's install-time bullet, read back: an
            // alias created before this context ever had a schema could
            // resolve to the reserved label without ever having been
            // caught by `add_label_alias`'s or a batch's own guard.
            let reserved_alias_conflicts: BTreeMap<String, String> = context
                .label_aliases()
                .into_iter()
                .filter(|(_, canonical)| *canonical == SCHEMA_TYPE_LABEL)
                .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                .collect();

            // `SchemaEnv::build` runs its own live `query_any` sweep
            // (scoped to the concepts `fact_ops` mentions) on top of the
            // full scan `all_associations` already spent — recheck the
            // deadline before paying for it, the same pre-flight every
            // other heavy step in this closure gets.
            if deadline.expired() {
                return Err(AccessError::DeadlineExceeded);
            }

            // The same union every write entrance builds (S3, #381) —
            // reused verbatim rather than re-deriving type membership by
            // hand, so this audit's domain/range answer can never
            // disagree with what `strict` would actually refuse for the
            // identical set of facts.
            let env = SchemaEnv::build(
                context,
                SchemaCheckInput {
                    schema: Arc::clone(&enforcing),
                    ops: &fact_ops,
                    declared_labels: &BTreeMap::new(),
                    retracted_source: None,
                },
            );

            let mut violations = Vec::new();
            for (edge, op) in fact_edges.into_iter().zip(fact_ops.iter()) {
                if deadline.expired() {
                    return Err(AccessError::DeadlineExceeded);
                }
                let check = schema_issues(&env, std::slice::from_ref(op), IssuePath::Edge);
                if !check.violations.is_empty() {
                    violations.push((edge, check.violations));
                }
            }

            let document = enforcing.document();
            let untyped_concepts = live_concepts
                .into_iter()
                .filter(|concept| {
                    !typed_concepts.contains(concept) && !asserted_type_names.contains(concept)
                })
                .filter(|concept| !document.types.contains_key(concept))
                .collect();
            let undeclared_types = asserted_type_names
                .into_iter()
                .filter(|type_name| !document.types.contains_key(type_name))
                .collect();
            let unknown_labels = if document.closed_labels {
                live_labels
                    .into_iter()
                    .filter(|label| label != SCHEMA_TYPE_LABEL)
                    .filter(|label| !document.relations.contains_key(label))
                    .collect()
            } else {
                BTreeSet::new()
            };

            Ok(RawAudit {
                violations,
                untyped_concepts,
                undeclared_types,
                unknown_labels,
                reserved_alias_conflicts,
            })
        })
        .and_then(std::convert::identity)?;

    // A read like `audit_drift`/`unreachable_from`: zero findings is the
    // audit succeeding, not a miss, so it never counts as an empty read.
    state.note_read(name, false);

    let (total, violations) = page_by(raw.violations, limit, after, |(association, _)| {
        (
            association.weight,
            association.subject.as_str(),
            association.label.as_str(),
            association.object.as_str(),
        )
    });
    let markers = state.resolve_markers(
        name,
        locator_keys(violations.iter().map(|(association, _)| association)),
    );
    let violations = violations
        .into_iter()
        .map(|(association, issues)| SchemaViolationOut {
            association: association_out(association, &markers),
            issues,
        })
        .collect();

    Ok(SchemaAudit {
        total,
        violations,
        untyped_concepts: audit_names(raw.untyped_concepts),
        undeclared_types: audit_names(raw.undeclared_types),
        unknown_labels: audit_names(raw.unknown_labels),
        reserved_alias_conflicts: audit_aliases(raw.reserved_alias_conflicts),
    })
}

/// `POST /contexts/{name}/schema/audit` (#385, S7 of #218's ADR 0009
/// split §10): judges every live association against `name`'s installed
/// document — the pre-existing violations `strict` itself can never
/// surface on its own (§7.1), since a write entrance only ever judges a
/// write as it happens, never what already landed before the schema (or
/// its current mode) existed.
pub async fn audit_schema(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppBytes(body): AppBytes,
) -> Response {
    let started_at = Instant::now();
    let request: SchemaAuditRequest = match optional_body(&body, started_at) {
        Ok(request) => request,
        Err(refusal) => return *refusal,
    };
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    tokio::task::block_in_place(|| {
        // `schema_of`'s slow path takes this entry's write lock, so it
        // must resolve BEFORE `schema_audit`'s own `read_context` call —
        // the same ordering `get_schema`, `vocabulary_audit`, and the
        // associations handler's pre-write arm all already depend on
        // (see `AppState::hidden_label`'s own doc for why the two
        // cannot nest).
        match state.schema_of(&name) {
            None => not_found(&name, started_at),
            Some(Ok(None)) => error(
                ErrorCode::NoSchema,
                format!("context '{name}' has no schema document"),
                started_at,
            ),
            Some(Err(message)) => {
                // Same posture as `get_schema`'s Load arm: the detail can
                // name a filesystem path, so it is logged, never
                // forwarded.
                tracing::warn!(context = %name, error = %message, "schema load failed");
                state.metrics().record_error(ErrorKind::Load);
                error(
                    ErrorCode::Internal,
                    format!("context '{name}' schema could not be loaded — see server logs"),
                    started_at,
                )
            }
            Some(Ok(Some(installed))) => match schema_audit(
                &state,
                &name,
                installed,
                request.limit,
                request.after.as_ref(),
                deadline,
            ) {
                Ok(audit) => ok(audit, started_at),
                Err(failure) => access_error(&state, failure, &name, started_at),
            },
        }
    })
}

/// `POST /contexts/{name}/schema/validate` (#385, S7, §10): the same
/// judgment as [`audit_schema`], but over a PROPOSED document that is
/// validated and evaluated without ever being persisted — the pre-flight
/// §7.1 promises before a `strict` flip. Works identically whether `name`
/// already has an installed schema or none at all; either way this call
/// never reads or writes the resident document.
pub async fn validate_schema(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<SchemaValidateRequest>,
) -> Response {
    let started_at = Instant::now();
    let SchemaValidateRequest {
        document,
        limit,
        after,
    } = request;
    let installed = match schema::install(document) {
        Ok(installed) => Arc::new(installed),
        Err(violation) => {
            return error(
                ErrorCode::InvalidArgument,
                violation.to_string(),
                started_at,
            );
        }
    };
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match tokio::task::block_in_place(|| {
        schema_audit(&state, &name, installed, limit, after.as_ref(), deadline)
    }) {
        Ok(audit) => ok(audit, started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}
