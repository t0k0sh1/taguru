//! Applying WAL ops to a live graph — the write path's batch
//! semantics (`apply_in_order`) and replay's logged-skip/guarded
//! variants (`replay_op`, `replay_wal_guarded`), all built on the one
//! op-to-graph interpreter (`apply_op`) so a replayed op can never
//! mean something different than it did when first applied.

use super::*;
use std::any::Any;
use std::cell::Cell;

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

/// Where a guarded replay currently stands — parsing the log, or
/// partway through applying its recovered ops — so a panic in either
/// half can name what it was doing ([`replay_panic_line`]) instead of
/// the fixed "an op panicked" every prior replay failure reported
/// alike, whatever op or line actually broke.
#[derive(Debug, Clone, Copy)]
pub(super) enum ReplayStage {
    Parsing,
    Applying {
        /// 0-based position within the recovered op slice.
        index: usize,
        total: usize,
        kind: &'static str,
    },
}

/// [`replay_wal_guarded`]'s panic message, split out as a pure
/// function so the downcast logic (identical in shape to
/// `api::panic_response`'s) and the stage-naming can be unit-tested
/// without ever triggering a real panic.
pub(super) fn replay_panic_line(payload: &(dyn Any + Send), stage: ReplayStage) -> String {
    let message = if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        "panicked with a non-string payload".to_string()
    };
    match stage {
        ReplayStage::Parsing => format!("WAL replay panicked parsing the log: {message}"),
        ReplayStage::Applying { index, total, kind } => format!(
            "WAL replay panicked applying op {}/{total} ({kind}): {message}",
            index + 1
        ),
    }
}

/// [`replay_op`]'s rejection line, split out as a pure function so the
/// severity split can be unit-tested without a real `Context`. `full`
/// distinguishes two very different rejections that would otherwise
/// read identically: an ordinary, benign one (the same rejection the
/// original write already reported to its client — replay reruns the
/// op on the exact state the original saw) versus a capacity refusal,
/// where an ALREADY-ACKNOWLEDGED write is being permanently lost
/// because the context is full on replay. Returns whether this was
/// the capacity case, alongside the line to log it at the matching
/// severity.
pub(super) fn replay_rejection_line(op: &WalOp, message: &str, full: bool) -> (bool, String) {
    if full {
        (
            true,
            format!(
                "WAL replay permanently lost an acknowledged write: the context is at \
                 capacity and refused to reapply a replayed {} op: {message}",
                op.kind()
            ),
        )
    } else {
        (
            false,
            format!("WAL replay skipped an op (same rejection as the original): {message}"),
        )
    }
}

/// Re-applies one replayed op. A deterministic library rejection here
/// is either the same rejection the original write already reported
/// to its client (logged, never fatal — see [`replay_rejection_line`])
/// or, when `full` is set, an acknowledged write silently and
/// permanently disappearing — loud enough to stand out from the
/// benign case, but still not fatal to the load: the graph itself
/// stays healthy, only this one op's content is gone. Returns whether
/// this rejection was the capacity case, so [`replay_wal_guarded`] can
/// tally how many acknowledged writes a replay lost.
pub(super) fn replay_op(context: &mut Context, op: &WalOp) -> bool {
    match apply_op(context, op) {
        Ok(()) => false,
        Err((message, full)) => {
            let (is_capacity, line) = replay_rejection_line(op, &message, full);
            if is_capacity {
                tracing::error!("{line}");
            } else {
                tracing::warn!("{line}");
            }
            is_capacity
        }
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
///
/// A panic's payload used to be discarded outright — every replay
/// panic reached `AppState::load_failure` as the same boilerplate
/// string, naming neither the op nor where parsing or applying broke.
/// `stage` (read after `catch_unwind` returns, written from inside the
/// guarded closure) recovers that: [`replay_panic_line`] folds it and
/// the panic payload into one diagnostic line. Likewise, a capacity
/// rejection during replay used to log at the same `warn` level as an
/// ordinary, harmless one; `capacity_losses` tallies how many of this
/// replay's rejections were the acknowledged-write-lost kind
/// ([`replay_rejection_line`]) so a summary can call it out at `error`
/// once the pass completes.
pub(super) fn replay_wal_guarded(
    path: &Path,
    watermark: u64,
    context: &mut Context,
) -> Result<u64, String> {
    let stage = Cell::new(ReplayStage::Parsing);
    let capacity_losses = Cell::new(0usize);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> io::Result<u64> {
        let (ops, top) = wal::replay::<WalOp>(path, watermark)?;
        let total = ops.len();
        for (index, op) in ops.iter().enumerate() {
            stage.set(ReplayStage::Applying {
                index,
                total,
                kind: op.kind(),
            });
            if replay_op(context, op) {
                capacity_losses.set(capacity_losses.get() + 1);
            }
        }
        Ok(top)
    }))
    .unwrap_or_else(|payload| Err(io::Error::other(replay_panic_line(&*payload, stage.get()))));
    let losses = capacity_losses.get();
    if losses > 0 {
        tracing::error!(
            losses,
            "WAL replay permanently lost {losses} acknowledged write(s) to capacity refusals"
        );
    }
    result.map_err(|error| error.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_op() -> WalOp {
        WalOp::AliasConcept {
            alias: "sake".to_string(),
            canonical: "日本酒".to_string(),
        }
    }

    /// A `String` payload — what `panic!("{}", owned)` and most
    /// library panics produce — is recovered verbatim, and the
    /// `Parsing` stage names itself without an op index (there is no
    /// op yet; parsing hasn't handed any back).
    #[test]
    fn replay_panic_line_recovers_a_string_payload_at_the_parsing_stage() {
        let payload: Box<dyn Any + Send> = Box::new("log corrupt beyond repair".to_string());
        let line = replay_panic_line(payload.as_ref(), ReplayStage::Parsing);
        assert!(line.contains("parsing the log"), "{line}");
        assert!(line.contains("log corrupt beyond repair"), "{line}");
    }

    /// A `&'static str` payload — what a bare `panic!("literal")`
    /// produces — is recovered too, at the `Applying` stage this time,
    /// which must name both the 1-based op position and its variant.
    #[test]
    fn replay_panic_line_recovers_a_str_payload_at_the_applying_stage() {
        let payload: Box<dyn Any + Send> = Box::new("boom");
        let stage = ReplayStage::Applying {
            index: 2,
            total: 5,
            kind: "Associate",
        };
        let line = replay_panic_line(payload.as_ref(), stage);
        assert!(line.contains("boom"), "{line}");
        assert!(line.contains("3/5"), "1-based, not the raw index: {line}");
        assert!(line.contains("Associate"), "{line}");
    }

    /// A panic payload that is neither `String` nor `&str` (a custom
    /// panic type, `std::panic::panic_any(42)`) must not be silently
    /// dropped — it still produces a line naming the stage, just
    /// without a recovered message.
    #[test]
    fn replay_panic_line_handles_a_non_string_payload() {
        let payload: Box<dyn Any + Send> = Box::new(42i32);
        let line = replay_panic_line(payload.as_ref(), ReplayStage::Parsing);
        assert!(line.contains("non-string payload"), "{line}");
    }

    /// The ordinary case — a rejection replay expected because it is
    /// the exact rejection the live write already reported — logs at
    /// the benign severity and says nothing about permanent loss.
    #[test]
    fn replay_rejection_line_is_benign_when_not_a_capacity_rejection() {
        let (is_capacity, line) = replay_rejection_line(&sample_op(), "already aliased", false);
        assert!(!is_capacity);
        assert!(line.contains("skipped"), "{line}");
        assert!(
            !line.contains("permanently"),
            "a benign rejection must not claim data loss: {line}"
        );
    }

    /// A capacity rejection during replay means an ALREADY-ACKNOWLEDGED
    /// write just vanished for good — this must read as data loss, not
    /// a routine skip, and must name which op family it was.
    #[test]
    fn replay_rejection_line_calls_out_a_capacity_loss_as_permanent() {
        let (is_capacity, line) = replay_rejection_line(&sample_op(), "alias table full", true);
        assert!(is_capacity);
        assert!(line.contains("permanently lost"), "{line}");
        assert!(line.contains("AliasConcept"), "{line}");
        assert!(line.contains("alias table full"), "{line}");
    }
}
