//! QA golden set over a FICTIONAL Rust codebase ("hikari-cache",
//! ground truth fully known and immune to real-code drift). The
//! counterpart of qa_recall.rs for source-code knowledge: identifiers
//! as concepts, call/definition/type structure as edges, and the entry
//! behaviors code lives or dies by — camelCase cues onto snake_case
//! names, typos, qualified `Type::fn` cues, path fragments, case twins
//! (`Frame` the struct vs `frame` the accessor), and natural-language
//! aliases bridging onto identifiers.
//!
//! Retrieval runs the documented loop mechanically, with one rule the
//! code vocabulary forces: every resolution TIED at the top score
//! becomes an origin, not just the first — case twins both normalize
//! to the same entry form and both come back at 1.0, and an argmax
//! client would silently drop one referent.
//!
//! Run with `--nocapture` for the per-question table.

#[path = "common/recall.rs"]
mod common;

use common::{Question, assert_golden_recall};
use taguru::context::Context;

/// The corpus: one small cache crate as an ingester following the
/// protocol's code discipline would write it — short identifier
/// concepts, files as `src/...` concepts, namespacing via `defined_in`
/// edges, a fixed English label vocabulary, NL aliases registered
/// after tuning.
fn corpus() -> Context {
    let mut context = Context::default();
    let mut fact = |s: &str, l: &str, o: &str| {
        context
            .associate_from(s, l, o, 1.0, "code-walk", None)
            .unwrap();
    };

    // src/store.rs
    fact("CacheStore", "kind", "struct");
    fact("CacheStore", "defined_in", "src/store.rs");
    fact("CacheStore", "field", "byte_budget");
    fact("CacheStore", "field", "hit_count");
    fact("fetch_block", "defined_in", "src/store.rs");
    fact("fetch_block", "calls", "evict_cold");
    fact("fetch_block", "calls", "open_frame");
    fact("fetch_block", "returns", "Option<Block>");
    fact(
        "fetch_block",
        "purpose",
        "look up one block, loading and evicting as needed",
    );
    fact("evict_cold", "defined_in", "src/store.rs");
    fact(
        "evict_cold",
        "purpose",
        "drop least-recently-used blocks until the store fits byte_budget",
    );
    fact("evict_cold", "invariant", "never evicts a pinned block");
    fact("EvictPolicy", "kind", "enum");
    fact("EvictPolicy", "defined_in", "src/store.rs");
    fact("EvictPolicy", "variant", "Lru");
    fact("EvictPolicy", "variant", "Ttl");

    // src/wire.rs
    fact("WireFrame", "kind", "struct");
    fact("WireFrame", "defined_in", "src/wire.rs");
    fact("WireFrame", "field", "frame_len");
    fact("seal_frame", "defined_in", "src/wire.rs");
    fact("seal_frame", "calls", "checksum_of");
    fact("seal_frame", "returns", "SealedFrame");
    fact(
        "seal_frame",
        "invariant",
        "a sealed frame is never mutated again",
    );
    fact("open_frame", "defined_in", "src/wire.rs");
    fact("open_frame", "calls", "checksum_of");
    fact(
        "open_frame",
        "invariant",
        "rejects a frame whose checksum does not match",
    );
    fact("checksum_of", "defined_in", "src/wire.rs");
    fact("checksum_of", "purpose", "CRC32 of the frame body");
    // The case twins: one spelling, one referent — so the type and the
    // accessor are two concepts, and only the entry folds their case.
    fact("Frame", "kind", "struct");
    fact("Frame", "defined_in", "src/wire.rs");
    fact("frame", "kind", "fn");
    fact("frame", "returns", "&Frame");
    fact("frame", "defined_in", "src/wire.rs");

    // Tuning: NL entryways registered after observing missed wordings.
    context
        .add_concept_alias("退避ループ", "evict_cold")
        .unwrap();
    context.add_label_alias("呼ぶ", "calls").unwrap();
    context
}

const QUESTIONS: &[Question] = &[
    Question {
        ask: "fetchBlock は何を呼ぶ? (camelCase cue → snake_case 識別子)",
        cues: &["fetchBlock"],
        needed: &[
            ("fetch_block", "calls", "evict_cold"),
            ("fetch_block", "calls", "open_frame"),
        ],
    },
    Question {
        ask: "エビクションは pin を尊重する? (タイポ cue evict_cld)",
        cues: &["evict_cld"],
        needed: &[("evict_cold", "invariant", "never evicts a pinned block")],
    },
    Question {
        ask: "src/wire.rs には何が定義されている? (モジュールの逆引き目次)",
        cues: &["src/wire.rs"],
        needed: &[
            ("seal_frame", "defined_in", "src/wire.rs"),
            ("WireFrame", "defined_in", "src/wire.rs"),
            ("checksum_of", "defined_in", "src/wire.rs"),
        ],
    },
    Question {
        ask: "CacheStore::fetch_block の返り値は? (修飾名 cue → 短名概念)",
        cues: &["CacheStore::fetch_block"],
        needed: &[("fetch_block", "returns", "Option<Block>")],
    },
    Question {
        ask: "frame は構造体? アクセサ? (ケース双子が両方浮上する)",
        cues: &["frame"],
        needed: &[("Frame", "kind", "struct"), ("frame", "returns", "&Frame")],
    },
    Question {
        ask: "退避ループの目的は? (自然言語エイリアス → 識別子)",
        cues: &["退避ループ"],
        needed: &[(
            "evict_cold",
            "purpose",
            "drop least-recently-used blocks until the store fits byte_budget",
        )],
    },
    Question {
        ask: "seal_frame は何を呼ぶ? (日本語ラベルエイリアス 呼ぶ → calls)",
        cues: &["seal_frame", "呼ぶ"],
        needed: &[("seal_frame", "calls", "checksum_of")],
    },
    Question {
        ask: "checksum_of に依存するのは誰? (呼び出し元の逆引き)",
        cues: &["checksum_of"],
        needed: &[
            ("seal_frame", "calls", "checksum_of"),
            ("open_frame", "calls", "checksum_of"),
        ],
    },
    Question {
        ask: "wire.rs に何がある? (パス断片 cue → フルパス概念)",
        cues: &["wire.rs"],
        needed: &[("WireFrame", "defined_in", "src/wire.rs")],
    },
];

/// Every resolution tied at the top score becomes an origin, not just
/// the first — code vocabularies collide on case (`Frame`/`frame`
/// both normalize to the same entry form and both come back at 1.0),
/// and an argmax strategy would silently drop one referent.
fn resolve_origins(context: &Context, cues: &[&str]) -> Vec<String> {
    let mut origins: Vec<String> = Vec::new();
    for cue in cues {
        let resolutions = context.resolve(cue);
        if let Some(best) = resolutions.first().map(|top| top.score) {
            for resolution in resolutions {
                if resolution.score < best {
                    break;
                }
                if !origins.contains(&resolution.name) {
                    origins.push(resolution.name);
                }
            }
        }
    }
    origins
}

#[test]
fn golden_code_questions_recall_every_needed_fact() {
    let context = corpus();
    assert_golden_recall(
        "code needed-fact recall",
        &context,
        QUESTIONS,
        resolve_origins,
    );
}
