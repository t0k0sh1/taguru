//! Shared generated context operations for the library and registry tests.

use proptest::prelude::*;

pub(crate) fn config() -> ProptestConfig {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    ProptestConfig {
        cases,
        ..ProptestConfig::default()
    }
}

/// Generates finite values whose decimal JSON representation parses back to
/// the exact same `f64`. Properties that cross a JSON persistence boundary
/// use this to test their own invariant rather than serde_json's rounding.
pub(crate) fn json_roundtrip_f64_strategy(
    range: std::ops::Range<f64>,
) -> impl Strategy<Value = f64> {
    range.prop_filter("value must survive JSON parsing bit-exactly", |&value| {
        serde_json::from_str::<f64>(&serde_json::to_string(&value).unwrap()).unwrap() == value
    })
}

const CONCEPT_WORDS: &[&str] = &[
    "青嶺酒造",
    "杜氏",
    "南部杜氏",
    "山田錦",
    "高瀬",
    "state",
    "index",
    "boot",
];
const LABEL_WORDS: &[&str] = &["好き", "由来", "所在地", "labelled", "linked"];
const SOURCE_WORDS: &[&str] = &["a.md", "b.md", "文書1", "note.txt"];
const CONCEPT_ALIAS_WORDS: &[&str] = &["蔵元エイリアス", "Aomine", "aliasConceptC"];
const LABEL_ALIAS_WORDS: &[&str] = &["設立年エイリアス", "aliasLabelC"];

#[derive(Clone, Debug)]
pub(crate) struct AssocInput {
    pub(crate) subject: &'static str,
    pub(crate) label: &'static str,
    pub(crate) object: &'static str,
    pub(crate) weight: f64,
    pub(crate) source: Option<&'static str>,
    pub(crate) paragraph: Option<u32>,
}

fn assoc_input_strategy() -> impl Strategy<Value = AssocInput> {
    (
        prop::sample::select(CONCEPT_WORDS),
        prop::sample::select(LABEL_WORDS),
        prop::sample::select(CONCEPT_WORDS),
        -1.0e6f64..1.0e6f64,
        prop::option::of(prop::sample::select(SOURCE_WORDS)),
        prop::option::of(0u32..8),
    )
        .prop_map(
            |(subject, label, object, weight, source, paragraph)| AssocInput {
                subject,
                label,
                object,
                weight,
                source,
                paragraph,
            },
        )
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // consumed by the server target; the library test target shares this module
pub(crate) enum GeneratedWalOp {
    Associate(AssocInput),
    AliasConcept {
        alias: &'static str,
        canonical: &'static str,
    },
    AliasLabel {
        alias: &'static str,
        canonical: &'static str,
    },
    UnaliasConcept(&'static str),
    UnaliasLabel(&'static str),
    RetractSource(&'static str),
    RetractAssociation {
        subject: &'static str,
        label: &'static str,
        object: &'static str,
    },
}

/// Generates the same operation vocabulary the server stages in its WAL,
/// while keeping this shared library-side module independent of server types.
/// Candidates may be rejected at the point where a property applies them (for
/// example, an alias can precede its canonical name); that lets durability
/// properties model the acknowledged subset independently of the generator.
#[allow(dead_code)] // consumed by the server target; the library test target shares this module
pub(crate) fn wal_op_strategy() -> impl Strategy<Value = GeneratedWalOp> {
    let weight = json_roundtrip_f64_strategy(-1.0e6f64..1.0e6f64);

    prop_oneof![
        (
            prop::sample::select(CONCEPT_WORDS),
            prop::sample::select(LABEL_WORDS),
            prop::sample::select(CONCEPT_WORDS),
            weight,
            prop::option::of(prop::sample::select(SOURCE_WORDS)),
            prop::option::of(0u32..8),
        )
            .prop_map(|(subject, label, object, weight, source, paragraph)| {
                GeneratedWalOp::Associate(AssocInput {
                    subject,
                    label,
                    object,
                    weight,
                    source,
                    paragraph,
                })
            }),
        (
            prop::sample::select(CONCEPT_ALIAS_WORDS),
            prop::sample::select(CONCEPT_WORDS),
        )
            .prop_map(|(alias, canonical)| GeneratedWalOp::AliasConcept { alias, canonical }),
        (
            prop::sample::select(LABEL_ALIAS_WORDS),
            prop::sample::select(LABEL_WORDS),
        )
            .prop_map(|(alias, canonical)| GeneratedWalOp::AliasLabel { alias, canonical }),
        prop::sample::select(CONCEPT_ALIAS_WORDS).prop_map(GeneratedWalOp::UnaliasConcept),
        prop::sample::select(LABEL_ALIAS_WORDS).prop_map(GeneratedWalOp::UnaliasLabel),
        prop::sample::select(SOURCE_WORDS).prop_map(GeneratedWalOp::RetractSource),
        (
            prop::sample::select(CONCEPT_WORDS),
            prop::sample::select(LABEL_WORDS),
            prop::sample::select(CONCEPT_WORDS),
        )
            .prop_map(
                |(subject, label, object)| GeneratedWalOp::RetractAssociation {
                    subject,
                    label,
                    object,
                }
            ),
    ]
}

#[derive(Clone, Debug)]
pub(crate) enum AliasInput {
    Concept {
        alias: &'static str,
        canonical: &'static str,
    },
    Label {
        alias: &'static str,
        canonical: &'static str,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum RetractionInput {
    Source(&'static str),
    Association {
        subject: &'static str,
        label: &'static str,
        object: &'static str,
    },
}

/// Generates associations, aliases for names those associations intern,
/// and retractions drawn from the associations and sources just created.
pub(crate) fn scenario_strategy()
-> impl Strategy<Value = (Vec<AssocInput>, Vec<AliasInput>, Vec<RetractionInput>)> {
    prop::collection::vec(assoc_input_strategy(), 1..12).prop_flat_map(|assoc_ops| {
        let mut concept_names: Vec<&'static str> = assoc_ops
            .iter()
            .flat_map(|op| [op.subject, op.object])
            .collect();
        concept_names.sort_unstable();
        concept_names.dedup();
        let mut label_names: Vec<&'static str> = assoc_ops.iter().map(|op| op.label).collect();
        label_names.sort_unstable();
        label_names.dedup();

        let alias_op_strategy = prop_oneof![
            (
                prop::sample::select(CONCEPT_ALIAS_WORDS),
                prop::sample::select(concept_names),
            )
                .prop_map(|(alias, canonical)| AliasInput::Concept { alias, canonical }),
            (
                prop::sample::select(LABEL_ALIAS_WORDS),
                prop::sample::select(label_names),
            )
                .prop_map(|(alias, canonical)| AliasInput::Label { alias, canonical }),
        ];

        let associations: Vec<_> = assoc_ops
            .iter()
            .map(|op| (op.subject, op.label, op.object))
            .collect();
        let association_retraction =
            prop::sample::select(associations).prop_map(|(subject, label, object)| {
                RetractionInput::Association {
                    subject,
                    label,
                    object,
                }
            });
        let sources: Vec<_> = assoc_ops.iter().filter_map(|op| op.source).collect();
        let retraction_strategy = if sources.is_empty() {
            association_retraction.boxed()
        } else {
            prop_oneof![
                association_retraction,
                prop::sample::select(sources).prop_map(RetractionInput::Source),
            ]
            .boxed()
        };

        (
            Just(assoc_ops),
            prop::collection::vec(alias_op_strategy, 0..6),
            prop::collection::vec(retraction_strategy, 0..6),
        )
    })
}
