use std::collections::HashSet;

use super::*;
use crate::context::stats::DICE_FLOOR;
use crate::context_proptest::{
    AliasInput, AssocInput, RetractionInput, config as proptest_config, scenario_strategy,
};
use crate::deadline::Deadline;
use proptest::prelude::*;

/// Applies a scenario the way a caller would: sourced/unsourced
/// associations through the real API, then alias registrations
/// with their `Result` discarded — a `Conflict` from a repeated
/// alias spelling must leave the context unchanged, and that is
/// exactly what the round trip below is checking for, not just
/// the happy path.
fn build_context(
    assoc_ops: &[AssocInput],
    alias_ops: &[AliasInput],
    retractions: &[RetractionInput],
) -> Context {
    let mut context = Context::default();
    for op in assoc_ops {
        match op.source {
            Some(source) => context
                .associate_from(
                    op.subject,
                    op.label,
                    op.object,
                    op.weight,
                    source,
                    op.paragraph,
                )
                .unwrap(),
            None => context
                .associate(op.subject, op.label, op.object, op.weight)
                .unwrap(),
        }
    }
    for alias_op in alias_ops {
        match alias_op {
            AliasInput::Concept { alias, canonical } => {
                let _ = context.add_concept_alias(*alias, canonical);
            }
            AliasInput::Label { alias, canonical } => {
                let _ = context.add_label_alias(*alias, canonical);
            }
        }
    }
    for retraction in retractions {
        match retraction {
            RetractionInput::Source(source) => {
                let _ = context.retract_source(source);
            }
            RetractionInput::Association {
                subject,
                label,
                object,
            } => {
                let _ = context.retract_association(subject, label, object);
            }
        }
    }
    context
}

fn within_reaccumulation_error(left: f64, right: f64, terms: u64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    let tolerance = f64::EPSILON * scale * terms.max(1) as f64 * 8.0;
    (left - right).abs() <= tolerance
}

/// Checks the semantic compaction contract independently of the
/// rebuilt context's record layout or allocation sizes.
fn assert_live_content_is_exact(source: &Context, compacted: &Context) {
    let all_before = source.query_any(&[], &[], &[]);
    let live_before: Vec<_> = all_before
        .iter()
        .filter(|association| association.count > 0)
        .collect();
    let after = compacted.query_any(&[], &[], &[]);
    assert_eq!(after.len(), live_before.len());

    for (before, after) in live_before.iter().zip(&after) {
        assert_eq!(
            (&after.subject, &after.label, &after.object),
            (&before.subject, &before.label, &before.object)
        );
        assert_eq!(after.count, before.count);
        assert!(within_reaccumulation_error(
            after.weight * after.count as f64,
            before.weight * before.count as f64,
            before.count,
        ));
        assert_eq!(after.attributions.len(), before.attributions.len());
        for (before, after) in before.attributions.iter().zip(&after.attributions) {
            assert_eq!(after.source, before.source);
            assert_eq!(after.count, before.count);
            assert_eq!(after.paragraph, before.paragraph);
            assert!(within_reaccumulation_error(
                after.weight,
                before.weight,
                before.count,
            ));
        }
    }

    let live_concepts: HashSet<&str> = live_before
        .iter()
        .flat_map(|association| [association.subject.as_str(), association.object.as_str()])
        .collect();
    let live_labels: HashSet<&str> = live_before
        .iter()
        .map(|association| association.label.as_str())
        .collect();
    let expected_concept_aliases: Vec<_> = source
        .concept_aliases()
        .into_iter()
        .filter(|(_, canonical)| live_concepts.contains(canonical))
        .collect();
    let expected_label_aliases: Vec<_> = source
        .label_aliases()
        .into_iter()
        .filter(|(_, canonical)| live_labels.contains(canonical))
        .collect();
    assert_eq!(compacted.concept_aliases(), expected_concept_aliases);
    assert_eq!(compacted.label_aliases(), expected_label_aliases);
}

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn retract_then_reassert_matches_a_fresh_assert(
        (assoc_ops, _, _) in scenario_strategy(),
        target in any::<prop::sample::Index>(),
    ) {
        let target = &assoc_ops[target.index(assoc_ops.len())];
        let mut reasserted = build_context(&assoc_ops, &[], &[]);
        let _ = reasserted.retract_association(
            target.subject,
            target.label,
            target.object,
        );
        let mut fresh = Context::default();

        for context in [&mut reasserted, &mut fresh] {
            match target.source {
                Some(source) => context.associate_from(
                    target.subject,
                    target.label,
                    target.object,
                    target.weight,
                    source,
                    target.paragraph,
                ).unwrap(),
                None => context.associate(
                    target.subject,
                    target.label,
                    target.object,
                    target.weight,
                ).unwrap(),
            }
        }

        prop_assert_eq!(
            reasserted.query(
                Some(target.subject),
                Some(target.label),
                Some(target.object),
            ),
            fresh.query(
                Some(target.subject),
                Some(target.label),
                Some(target.object),
            ),
        );
    }

    #[test]
    fn resolve_normalization_is_idempotent_and_canonical_hits_are_stable(
        name in "[^\\x00]{1,32}",
    ) {
        let once = normalize_entry(&name);
        prop_assert_eq!(normalize_entry(&once), once.clone());
        prop_assume!(!once.is_empty());
        // The object side of the association below is the fixed
        // literal "object". If `name` also normalizes to "object"
        // (e.g. the full-width "ｏbject"), the two spellings become
        // DIFFERENT concepts that collide under normalization, and
        // resolve's exact-match tie-break (alphabetical, per
        // `sort_resolutions`, not insertion order) can then put
        // "object" ahead of `name` — breaking the single-winner
        // assumption this test relies on below. Excluding that
        // collision keeps `name` the only concept in play.
        prop_assume!(once != "object");

        let mut context = Context::default();
        context.associate(&name, "relation", "object", 1.0).unwrap();
        let first = context.resolve(&name);
        prop_assert!(!first.is_empty());
        prop_assert_eq!(first[0].name.as_str(), name.as_str());
        prop_assert_eq!(first[0].kind, MatchKind::Exact);

        let canonical = first[0].name.clone();
        let second = context.resolve(&canonical);
        prop_assert!(!second.is_empty());
        prop_assert_eq!(second[0].name.as_str(), canonical.as_str());
        prop_assert_eq!(second[0].kind, MatchKind::Exact);
        prop_assert_eq!(second[0].score, 1.0);
    }

    #[test]
    fn compaction_preserves_exactly_the_live_content(
        (assoc_ops, alias_ops, retractions) in scenario_strategy(),
    ) {
        let context = build_context(&assoc_ops, &alias_ops, &retractions);
        let all_before = context.query_any(&[], &[], &[]);
        let aliases_before =
            context.concept_aliases().len() + context.label_aliases().len();

        let (compacted, stats) = context.compacted(Deadline::unbounded()).unwrap();
        assert_live_content_is_exact(&context, &compacted);
        prop_assert_eq!(
            stats.dead_edges,
            all_before.iter().filter(|association| association.count == 0).count()
        );
        prop_assert_eq!(
            stats.aliases_dropped,
            aliases_before
                - compacted.concept_aliases().len()
                - compacted.label_aliases().len()
        );

        let canonical_image = compacted.to_bytes();
        let (again, second_stats) =
            compacted.compacted(Deadline::unbounded()).unwrap();
        prop_assert_eq!(second_stats, CompactionStats::default());
        prop_assert_eq!(again.to_bytes(), canonical_image);
    }

    #[test]
    fn context_round_trips_through_bytes(
        (assoc_ops, alias_ops, retractions) in scenario_strategy(),
        applied_seq in any::<u64>(),
        dice_floor in prop::option::of(-1.0f64..2.0),
    ) {
        let mut context = build_context(&assoc_ops, &alias_ops, &retractions);
        context.set_applied_seq(applied_seq);
        context.set_dice_floor(dice_floor);

        let image = context.to_bytes();
        let restored = Context::from_bytes(&image).unwrap();
        prop_assert_eq!(restored.to_bytes(), image);
        prop_assert_eq!(
            restored.query(None, None, None),
            context.query(None, None, None)
        );
        prop_assert_eq!(restored.concept_aliases(), context.concept_aliases());
        prop_assert_eq!(restored.label_aliases(), context.label_aliases());
        prop_assert_eq!(restored.applied_seq(), context.applied_seq());
        prop_assert_eq!(restored.dice_floor(), DICE_FLOOR);
    }

    #[test]
    fn from_bytes_never_panics_on_arbitrary_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..512),
    ) {
        let _ = Context::from_bytes(&bytes);
    }

    #[test]
    fn changed_v6_bytes_return_a_structured_error(
        (assoc_ops, alias_ops, retractions) in scenario_strategy(),
        byte in any::<prop::sample::Index>(),
        mask in 1u8..=u8::MAX,
    ) {
        let context = build_context(&assoc_ops, &alias_ops, &retractions);
        let mut bytes = context.to_bytes();
        *byte.get_mut(&mut bytes) ^= mask;
        prop_assert!(Context::from_bytes(&bytes).is_err());
    }

    #[test]
    fn truncated_v6_bytes_return_a_structured_error(
        (assoc_ops, alias_ops, retractions) in scenario_strategy(),
        cut in any::<prop::sample::Index>(),
    ) {
        let context = build_context(&assoc_ops, &alias_ops, &retractions);
        let mut bytes = context.to_bytes();
        let keep = cut.index(bytes.len());
        bytes.truncate(keep);
        prop_assert!(Context::from_bytes(&bytes).is_err());
    }

    #[test]
    fn from_bytes_never_panics_on_mutated_legacy_bytes(
        (assoc_ops, alias_ops, retractions) in scenario_strategy(),
        version in 1u32..=5,
        mutations in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 0..24),
    ) {
        let context = build_context(&assoc_ops, &alias_ops, &retractions);
        let mut bytes = context.to_bytes_as_version(version);
        for (pick, value) in mutations {
            *pick.get_mut(&mut bytes) = value;
        }
        let _ = Context::from_bytes(&bytes);
    }

    #[test]
    fn truncated_legacy_bytes_return_a_structured_error(
        (assoc_ops, alias_ops, retractions) in scenario_strategy(),
        version in 1u32..=5,
        cut in any::<prop::sample::Index>(),
    ) {
        let context = build_context(&assoc_ops, &alias_ops, &retractions);
        let mut bytes = context.to_bytes_as_version(version);
        let keep = cut.index(bytes.len());
        bytes.truncate(keep);
        prop_assert!(Context::from_bytes(&bytes).is_err());
    }
}
