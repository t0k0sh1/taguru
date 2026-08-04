//! Installing a parsed `taguru_schema` record onto a context — the
//! offline CLI's own leg of the two entrances that share
//! [`apply_schema_record`] (the other is `POST /import`'s apply stage,
//! `src/api/import.rs`).

use super::*;

/// Why a `taguru_schema` record's install failed after it already
/// parsed and validated — [`crate::registry::PutSchemaError`]'s
/// entrance-agnostic twin, so both the offline CLI (this module) and
/// `POST /import` (`src/api/import.rs`) can format or map each case
/// without matching on the registry error directly.
#[derive(Debug)]
pub(crate) enum SchemaApplyError {
    /// The record's context does not exist — not yet created by an
    /// earlier batch of the same stream, nor previously.
    NoContext,
    ReservedAlias(String),
    Load(String),
    Io(std::io::Error),
}

impl std::fmt::Display for SchemaApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoContext => write!(
                f,
                "does not exist — a schema record's context must already exist (created \
                 by an earlier batch of the same stream, or previously) before its schema \
                 can install"
            ),
            Self::ReservedAlias(alias) => write!(
                f,
                "label alias '{alias}' resolves to '{}', the relation label reserved for \
                 type assertions (ADR 0009 §6.3) — rename the alias before installing this \
                 schema",
                schema::SCHEMA_TYPE_LABEL
            ),
            Self::Load(message) => write!(f, "schema could not be loaded: {message}"),
            Self::Io(error) => write!(f, "schema not persisted: {error}"),
        }
    }
}

/// Installs one parsed schema record via [`AppState::put_schema`] —
/// shared by the offline CLI's pass 2 (here) and `POST /import`'s
/// apply stage (`src/api/import.rs`) so both entrances judge the same
/// failure the same way. `Ok` carries the installed document back so
/// each caller builds its own report/outcome shape from one source of
/// truth, the way [`apply_batch`]'s `Applied` feeds both `report()`
/// and [`crate::api::import_outcome`].
pub(crate) fn apply_schema_record(
    state: &AppState,
    context: &str,
    installed: schema::InstalledSchema,
) -> Result<schema::SchemaDocument, SchemaApplyError> {
    match state.put_schema(context, installed) {
        None => Err(SchemaApplyError::NoContext),
        Some(Ok(document)) => Ok(document),
        Some(Err(crate::registry::PutSchemaError::ReservedAlias(alias))) => {
            Err(SchemaApplyError::ReservedAlias(alias))
        }
        Some(Err(crate::registry::PutSchemaError::Load(message))) => {
            Err(SchemaApplyError::Load(message))
        }
        Some(Err(crate::registry::PutSchemaError::Io(error))) => Err(SchemaApplyError::Io(error)),
    }
}
