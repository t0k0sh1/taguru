use taguru::context::Context;

// Three additions on the road to "knowledge itself, accurately searchable",
// in the order they were built:
//
// 1. Provenance — `associate_from` tags every assertion with its source, so
//    a retrieved association carries where it came from. Weight 2.0 backed
//    by two independent sources is no longer confusable with one emphatic
//    assertion, and the caller can follow any attribution back to the
//    original text (which lives outside this crate, wherever the caller
//    keeps it — the network is the index, not the storage of record).
// 2. Ranked retrieval — `activate` spreads activation from the origins,
//    attenuated per hop and split by each association's share of |weight|,
//    so retrieval finally *reads* the weights `associate` has been writing:
//    near beats far, heavy beats light, exclusive paths beat hub paths.
// 3. Entry resolution — `resolve` maps free-form wording onto stored
//    concept names lexically (exact / containment). It is the bottom tier
//    of entry handling; LLM query normalization and, at corpus scale, an
//    embedding index over concept names sit above it.
fn main() {
    let mut context = Context::default();

    const IPA: &str = "IPA公式説明";
    const NEWS: &str = "解説記事";

    // --- One 文脈, two sources describing it ---
    context
        .associate_from(
            "情報セキュリティの専門家",
            "構成する",
            "10大脅威選考会",
            1.0,
            IPA,
        )
        .unwrap();
    context
        .associate_from(
            "10大脅威選考会",
            "協力する",
            "情報セキュリティ10大脅威",
            1.0,
            IPA,
        )
        .unwrap();
    context
        .associate_from("10大脅威選考会", "選出する", "脅威候補", 1.0, IPA)
        .unwrap();
    context
        .associate_from(
            "セキュリティの事故や攻撃の状況",
            "選出元になる",
            "脅威候補",
            1.0,
            IPA,
        )
        .unwrap();
    context
        .associate_from(
            "セキュリティの事故や攻撃の状況",
            "発生時期",
            "前年",
            1.0,
            IPA,
        )
        .unwrap();
    context
        .associate_from(
            "セキュリティの事故や攻撃の状況",
            "社会的影響",
            "大きい",
            1.0,
            IPA,
        )
        .unwrap();
    context
        .associate_from("10大脅威選考会", "決定する", "個人向け脅威", 1.0, IPA)
        .unwrap();
    context
        .associate_from("10大脅威選考会", "決定する", "組織向け脅威", 1.0, IPA)
        .unwrap();
    context
        .associate_from("決定", "手段", "投票", 1.0, IPA)
        .unwrap();

    // A second, independent document corroborates two of the facts.
    context
        .associate_from("決定", "手段", "投票", 1.0, NEWS)
        .unwrap();
    context
        .associate_from(
            "10大脅威選考会",
            "協力する",
            "情報セキュリティ10大脅威",
            1.0,
            NEWS,
        )
        .unwrap();

    // (1) Provenance: the same weight 2.0 that used to be ambiguous
    // ("one emphatic assertion, or two independent ones?") now answers
    // that question itself, and each source id points at original text.
    println!("=== recall(\"投票\") — 重み2.0の由来が見える ===");
    for a in context.recall("投票") {
        println!(
            "  {} -({})-> {}  weight {:?}",
            a.subject, a.label, a.object, a.weight
        );
        for attribution in &a.attributions {
            println!(
                "    出典: {} (+{:?})",
                attribution.source, attribution.weight
            );
        }
    }

    // (2) Ranked retrieval: the twice-asserted 協力する edge now outranks
    // the once-asserted ones, and 2-hop facts trail with decayed strength —
    // weights and distance both finally matter at read time.
    println!("\n=== activate([\"10大脅威選考会\"], decay 0.5, limit 6) ===");
    for activation in context.activate(&["10大脅威選考会"], 0.5, 6) {
        println!(
            "  [{:.4}] {} -({})-> {}",
            activation.strength,
            activation.association.subject,
            activation.association.label,
            activation.association.object
        );
    }

    // (3) Entry resolution: a query fragment lands on the stored concept
    // name, which then anchors the ranked retrieval — resolve for the
    // door, activate for the knowledge.
    println!("\n=== resolve(\"選考会\") ===");
    let candidates = context.resolve("選考会");
    for r in &candidates {
        println!("  [{:.3}] {}", r.score, r.name);
    }

    let anchor = candidates[0].name.as_str();
    println!("\n=== resolve→activate: activate([\"{anchor}\"], 0.5, 3) ===");
    for activation in context.activate(&[anchor], 0.5, 3) {
        println!(
            "  [{:.4}] {} -({})-> {}",
            activation.strength,
            activation.association.subject,
            activation.association.label,
            activation.association.object
        );
    }
}
