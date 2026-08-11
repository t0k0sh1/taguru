//! Applying WAL ops to a live graph — the write path's batch
//! semantics (`apply_in_order`) and replay's logged-skip/guarded
//! variants (`replay_op`, `replay_wal_guarded`), all built on the one
//! op-to-graph interpreter (`apply_op`) so a replayed op can never
//! mean something different than it did when first applied.

use super::*;

/// Applies ops front to back, stopping at the first rejection — the
/// batch endpoints' historic partial semantics: everything before the
/// failing item stays applied.
pub(super) fn apply_in_order(context: &mut Context, ops: &[WalOp]) -> Result<usize, PartialWrite> {
    let mut applied = 0usize;
    for op in ops {
        if let Err((message, full)) = apply_op(context, op) {
            return Err(PartialWrite {
                applied,
                message,
                full,
            });
        }
        applied += 1;
    }
    Ok(applied)
}

/// How many ops an `apply_in_order` result actually landed — the full
/// count on success, the prefix on a partial write. Feeds
/// `logged_write`'s WAL trim: it never inspects `T` itself, only how
/// far the batch got.
pub(super) fn applied_count(result: &Result<usize, PartialWrite>) -> usize {
    match result {
        Ok(applied) => *applied,
        Err(partial) => partial.applied,
    }
}

/// Re-applies one replayed op. A deterministic library rejection here
/// is the same rejection the original write already reported to its
/// client — replay reruns the op on the exact state the original saw
/// — so it is logged, never fatal.
pub(super) fn replay_op(context: &mut Context, op: &WalOp) {
    if let Err((message, _)) = apply_op(context, op) {
        tracing::warn!("WAL replay skipped an op (same rejection as the original): {message}");
    }
}

/// Runs the whole WAL load — reading and parsing the log, then
/// applying every recovered op — behind one `catch_unwind`. Parsing is
/// as much bug surface as applying: a panic from either half (a bug in
/// `wal::replay`'s parser tripped by adversarial bytes, or a bug in
/// some op's own application logic — a deterministic library
/// rejection is not this, `replay_op` already turns those into a log
/// line) must become the same `Err` shape a corrupt image or
/// unreadable WAL produces. Without this, a poisoned log would panic
/// `ensure_hot` itself on every subsequent access — this context can
/// never come back Hot, so every caller touching it crash-loops
/// forever instead of hitting the existing quarantine-and-retry path
/// ([`LOAD_FAILURE_RETRY`]). Returns the WAL's top seq on success, so
/// the caller can seed `wal_seq`/`graph_revision` from it exactly as
/// `wal::replay` itself would.
pub(super) fn replay_wal_guarded(
    path: &Path,
    watermark: u64,
    context: &mut Context,
) -> Result<u64, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> io::Result<u64> {
        let (ops, top) = wal::replay(path, watermark)?;
        for op in &ops {
            replay_op(context, op);
        }
        Ok(top)
    }))
    .unwrap_or_else(|_| {
        Err(io::Error::other(
            "an op panicked reapplying against a fresh load",
        ))
    })
    .map_err(|error| error.to_string())
}

/// Applies one op to the graph; `Err` carries the human message each
/// op family has always reported through the API, plus whether it was
/// a capacity error.
pub(super) fn apply_op(context: &mut Context, op: &WalOp) -> Result<(), (String, bool)> {
    match op {
        WalOp::Associate(op) => {
            let result = match &op.source {
                Some(source) => context.associate_from(
                    op.subject.as_str(),
                    op.label.as_str(),
                    op.object.as_str(),
                    op.weight,
                    source.as_str(),
                    op.paragraph,
                ),
                None => context.associate(
                    op.subject.as_str(),
                    op.label.as_str(),
                    op.object.as_str(),
                    op.weight,
                ),
            };
            result.map_err(|full| (full.to_string(), true))
        }
        WalOp::AliasConcept { alias, canonical } => context
            .add_concept_alias(alias.as_str(), canonical)
            .map_err(|error| {
                (
                    format!("concept alias '{alias}' → '{canonical}': {error}"),
                    matches!(error, AliasError::Full(_)),
                )
            }),
        WalOp::AliasLabel { alias, canonical } => context
            .add_label_alias(alias.as_str(), canonical)
            .map_err(|error| {
                (
                    format!("label alias '{alias}' → '{canonical}': {error}"),
                    matches!(error, AliasError::Full(_)),
                )
            }),
        // A withdrawal of an absent spelling is a client mistake on
        // the live path (409, like every conflict), and on replay the
        // usual logged skip. Never a capacity error: removal frees.
        WalOp::UnaliasConcept { alias } => match context.remove_concept_alias(alias) {
            Some(_) => Ok(()),
            None => Err((
                format!(
                    "'{alias}' is not a concept alias (canonical names cannot be \
                     removed this way)"
                ),
                false,
            )),
        },
        WalOp::UnaliasLabel { alias } => match context.remove_label_alias(alias) {
            Some(_) => Ok(()),
            None => Err((
                format!(
                    "'{alias}' is not a label alias (canonical names cannot be \
                     removed this way)"
                ),
                false,
            )),
        },
        WalOp::RetractSource { source } => {
            context.retract_source(source);
            Ok(())
        }
        WalOp::RetractAssociation {
            subject,
            label,
            object,
        } => {
            // A triple that names no live edge is a no-op on replay,
            // exactly like an unknown source above.
            context.retract_association(subject, label, object);
            Ok(())
        }
    }
}
