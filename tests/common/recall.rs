//! Shared harness for the golden-question recall loop (code_recall.rs,
//! qa_recall.rs): both run the same documented retrieval loop over a
//! hand-built [`Context`] and assert every question's needed facts
//! came back, diverging in exactly one place — how a cue's concept
//! resolutions become origins — so that step is the caller's
//! strategy, injected as a closure.

use taguru::context::Context;

pub struct Question {
    pub ask: &'static str,
    pub cues: &'static [&'static str],
    pub needed: &'static [(&'static str, &'static str, &'static str)],
}

/// The shared tail of the retrieval loop: resolves each cue's label
/// namespace (top hit only, identical in both suites), activates from
/// `origins`, and role-pinned `query_any`s atop them — plus a
/// label-narrowed pass when any cue resolved a label.
fn retrieve(
    context: &Context,
    cues: &[&str],
    origins: Vec<String>,
) -> Vec<(String, String, String)> {
    let mut labels: Vec<String> = Vec::new();
    for cue in cues {
        if let Some(top) = context.resolve_label(cue).into_iter().next()
            && !labels.contains(&top.name)
        {
            labels.push(top.name);
        }
    }
    let origin_refs: Vec<&str> = origins.iter().map(String::as_str).collect();
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

    let triple = |a: taguru::context::Association| (a.subject, a.label, a.object);
    let mut facts: Vec<(String, String, String)> = Vec::new();
    facts.extend(
        context
            .activate(&origin_refs, 0.5, 20)
            .1
            .into_iter()
            .map(|activation| triple(activation.association)),
    );
    facts.extend(
        context
            .query_any(&origin_refs, &[], &[])
            .into_iter()
            .map(triple),
    );
    facts.extend(
        context
            .query_any(&[], &[], &origin_refs)
            .into_iter()
            .map(triple),
    );
    if !label_refs.is_empty() {
        facts.extend(
            context
                .query_any(&origin_refs, &label_refs, &[])
                .into_iter()
                .map(triple),
        );
    }
    facts
}

/// Runs the golden-question regression loop over `questions`, printing
/// a per-question table (`--nocapture` to see it), and panics if any
/// question's needed facts didn't fully come back. `resolve_origins`
/// is the cue -> concept-origin strategy the caller's vocabulary
/// forces — the one place the two golden sets diverge.
pub fn assert_golden_recall(
    label: &str,
    context: &Context,
    questions: &[Question],
    resolve_origins: impl Fn(&Context, &[&str]) -> Vec<String>,
) {
    let mut needed_total = 0usize;
    let mut needed_hit = 0usize;
    let mut unanswered: Vec<&str> = Vec::new();

    println!("\n=== {label} ===");
    for question in questions {
        let origins = resolve_origins(context, question.cues);
        let facts = retrieve(context, question.cues, origins);
        let mut all = true;
        for &(s, l, o) in question.needed {
            needed_total += 1;
            let hit = facts
                .iter()
                .any(|(fs, fl, fo)| fs == s && fl == l && fo == o);
            if hit {
                needed_hit += 1;
            } else {
                all = false;
                println!("  MISS {} — 不足: ({s}, {l}, {o})", question.ask);
            }
        }
        if all {
            println!("  ok   {}", question.ask);
        } else {
            unanswered.push(question.ask);
        }
    }
    println!(
        "  → 必要事実の再現率 {needed_hit}/{needed_total}, 完答 {}/{}",
        questions.len() - unanswered.len(),
        questions.len()
    );

    // The regression floor: every question must stay fully answered.
    // A failure here means an identifier-entry or reachability
    // regression — read the MISS lines above.
    assert!(
        unanswered.is_empty(),
        "unanswered questions: {unanswered:?}"
    );
}
