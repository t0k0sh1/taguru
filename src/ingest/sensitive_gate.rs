//! ADR 0038 §3.4: `import --refuse-sensitive` — the same rule set
//! `extract --redact` masks with (every built-in of both groups) run
//! over each batch's passage, association subject/label/object, alias
//! spellings, and question text before the batch is applied locally
//! or packed for `--url`. A hit refuses the BATCH — the unit import
//! already refuses on; a partial batch is never written — named by
//! path in `schema/check`'s shape (`batches[3].passage`,
//! `batches[3].associations[7].object`), by rule, and for a passage
//! hit by paragraph; never by the matched text. Import never rewrites
//! content: the fix is to re-extract with `--redact`, or to edit the
//! batch.

use super::*;

/// One sensitive match in a batch, addressed for the operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SensitiveHit {
    /// `batches[{b}].passage`, `batches[{b}].associations[{a}].object`,
    /// `batches[{b}].aliases[{k}]` (concept aliases in spelling order,
    /// then label aliases), `batches[{b}].questions[{q}]`.
    pub(super) path: String,
    pub(super) rule: String,
    /// The passage paragraph the match is in — a passage hit only.
    pub(super) paragraph: Option<u32>,
}

impl SensitiveHit {
    /// `batches[3].passage: sensitive: email (paragraph 2)` — the
    /// stderr line's tail and the `--json` error's clause.
    pub(super) fn text(&self) -> String {
        match self.paragraph {
            Some(paragraph) => format!(
                "{}: sensitive: {} (paragraph {paragraph})",
                self.path, self.rule
            ),
            None => format!("{}: sensitive: {}", self.path, self.rule),
        }
    }
}

/// Every sensitive match in `batch` (the `batch_index`-th of its file),
/// path-first. A field is reported once per rule; the passage once per
/// match, with its paragraph. A placeholder already in the text
/// (`«redacted …»`) is what a redacted extract leaves and is not a hit.
pub(super) fn sensitive_hits(
    batch: &Batch,
    batch_index: usize,
    rules: &crate::sensitive::RuleSet,
) -> Vec<SensitiveHit> {
    let mut hits = Vec::new();
    let prefix = format!("batches[{batch_index}]");
    if let Some(passage) = &batch.passage {
        for found in crate::sensitive::scan(passage, rules) {
            if found.preexisting {
                continue;
            }
            hits.push(SensitiveHit {
                path: format!("{prefix}.passage"),
                rule: found.rule,
                paragraph: Some(found.paragraph),
            });
        }
    }
    let mut field = |path: String, text: &str| {
        let mut seen: Vec<String> = Vec::new();
        for found in crate::sensitive::scan(text, rules) {
            if found.preexisting || seen.contains(&found.rule) {
                continue;
            }
            seen.push(found.rule.clone());
            hits.push(SensitiveHit {
                path: path.clone(),
                rule: found.rule,
                paragraph: None,
            });
        }
    };
    for (index, op) in batch.associations.iter().enumerate() {
        field(
            format!("{prefix}.associations[{index}].subject"),
            &op.subject,
        );
        field(format!("{prefix}.associations[{index}].label"), &op.label);
        field(format!("{prefix}.associations[{index}].object"), &op.object);
    }
    for (index, spelling) in batch.concepts.keys().chain(batch.labels.keys()).enumerate() {
        field(format!("{prefix}.aliases[{index}]"), spelling);
    }
    for (index, (_, question)) in batch.questions.iter().enumerate() {
        field(format!("{prefix}.questions[{index}]"), question);
    }
    hits
}

/// The batch-level stderr line after the hits: what was refused and
/// what to do about it — content never rewritten on the way in.
pub(super) fn refused_batch_message(batch_index: usize, batch: &Batch) -> String {
    format!(
        "batches[{batch_index}] (context '{}', source '{}') refused: sensitive content — \
         re-extract with `taguru extract --redact`, or edit the batch; nothing of it was \
         applied",
        batch.context, batch.source
    )
}

/// `--json`'s `error` for a refused batch: `sensitive: ` and the hits.
pub(super) fn refused_batch_error(hits: &[SensitiveHit]) -> String {
    let clauses: Vec<String> = hits.iter().map(SensitiveHit::text).collect();
    format!("sensitive: {}", clauses.join("; "))
}
