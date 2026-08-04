//! The CLI's per-batch report line, and import/export's shared logging
//! setup.

use super::*;

/// The CLI's per-file report line.
pub(super) fn report(batch: &Batch, applied: &Applied) -> String {
    format!(
        "context '{}'{} ← source '{}' ({} association(s) retracted): +{} \
         association(s), +{} alias(es){}{}{}{}{}{}",
        batch.context,
        if applied.created { " (created)" } else { "" },
        batch.source,
        applied.retracted,
        applied.associations,
        applied.aliases,
        match (applied.passage_stored, applied.passage_dropped) {
            (true, _) => ", passage stored",
            (false, true) => ", previous passage dropped (batch carried none)",
            (false, false) => "",
        },
        match (applied.questions_stored, applied.questions_dropped) {
            (0, 0) => String::new(),
            (stored, 0) => format!(", +{stored} question(s)"),
            (stored, dropped) => {
                format!(", +{stored} question(s) ({dropped} dropped: no such paragraph)")
            }
        },
        match (applied.sections_stored, applied.sections_dropped) {
            (0, 0) => String::new(),
            (stored, 0) => format!(", +{stored} section(s)"),
            (stored, dropped) => {
                format!(", +{stored} section(s) ({dropped} dropped: no such paragraph)")
            }
        },
        match (applied.locators_stored, applied.locators_dropped) {
            (0, 0) => String::new(),
            (stored, 0) => format!(", +{stored} locator(s)"),
            (stored, dropped) => {
                format!(", +{stored} locator(s) ({dropped} dropped: no such paragraph)")
            }
        },
        match applied.association_paragraphs_dropped {
            0 => String::new(),
            dropped => {
                format!(", {dropped} association paragraph locator(s) dropped: no such paragraph")
            }
        },
        match applied.schema_violations {
            0 => String::new(),
            violations => format!(", schema warnings: {violations}"),
        }
    )
}

/// Import (and export, which shares the need) logs like the server
/// does (RUST_LOG, stderr) so registry warnings — WAL replay notes,
/// load failures — are not dropped on the floor, but stdout stays
/// pure report.
pub(crate) fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_env("RUST_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}
