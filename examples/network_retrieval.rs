use associative_rag::context::Context;

// The consumer of a `Context` is an LLM, not a person: on the way in, an
// LLM distills natural-language knowledge into weighted associations; on
// the way out, it takes whatever a single retrieval returns and rebuilds
// natural prose from it using its own linguistic competence. This example
// shows exactly that middle step — the package one `explore` call hands the
// LLM to write from — so the interesting questions are:
//
//   1. Does one call gather enough *connected* material to rebuild the
//      original passage, or does it stop at directly-adjacent facts?
//   2. What can never be handed over at any depth — i.e. where is the hard
//      ceiling on reconstruction fidelity, before any LLM is involved?
//
// Referent ambiguity is out of scope by design: one `Context` is one 文脈,
// so a spelling shared by two different real-world things (the fruit
// "Apple" vs. the company "Apple") is handled by storing them in two
// different `Context`s, never inside one. The last section demonstrates
// that split.
fn main() {
    // --- One 文脈: IPA's public description of 情報セキュリティ10大脅威 ---
    let mut context = Context::default();
    context
        .associate(
            "情報セキュリティの専門家",
            "構成する",
            "10大脅威選考会",
            1.0,
        )
        .unwrap();
    context
        .associate(
            "10大脅威選考会",
            "協力する",
            "情報セキュリティ10大脅威",
            1.0,
        )
        .unwrap();
    context
        .associate("10大脅威選考会", "選出する", "脅威候補", 1.0)
        .unwrap();
    context
        .associate(
            "セキュリティの事故や攻撃の状況",
            "選出元になる",
            "脅威候補",
            1.0,
        )
        .unwrap();
    context
        .associate("セキュリティの事故や攻撃の状況", "発生時期", "前年", 1.0)
        .unwrap();
    context
        .associate(
            "セキュリティの事故や攻撃の状況",
            "社会的影響",
            "大きい",
            1.0,
        )
        .unwrap();
    context
        .associate("10大脅威選考会", "決定する", "個人向け脅威", 1.0)
        .unwrap();
    context
        .associate("10大脅威選考会", "決定する", "組織向け脅威", 1.0)
        .unwrap();
    context.associate("決定", "手段", "投票", 1.0).unwrap();

    let total = context.query(None, None, None).len();

    // One anchor, widening depth: watch the retrievable neighborhood grow
    // hop by hop. Each line is one association plus its hop distance — the
    // raw material an LLM would turn back into prose.
    for depth in 1..=3 {
        println!("=== explore([\"10大脅威選考会\"], {depth}) ===");
        let recollections = context.explore(&["10大脅威選考会"], depth);
        for r in &recollections {
            println!(
                "  [{}ホップ] {} -({})-> {}  (weight {:?})",
                r.distance,
                r.association.subject,
                r.association.label,
                r.association.object,
                r.association.weight
            );
        }
        println!("  → {}件 / 全{}件\n", recollections.len(), total);
    }

    // Depth 100 changes nothing: 決定/手段/投票 sits in a disconnected
    // component, because "決定" the concept and "決定する" the label were
    // never unified during extraction. No retrieval from this anchor can
    // ever return that fact, so any reconstruction must omit how the
    // committee decides — a hard ceiling on fidelity that is visible
    // mechanically, before any LLM comparison happens.
    let everything = context.explore(&["10大脅威選考会"], 100);
    println!(
        "=== explore([\"10大脅威選考会\"], 100) → {}件 / 全{}件 (「投票」の事実は非連結で到達不能) ===\n",
        everything.len(),
        total
    );

    // --- Two referents, one spelling → two `Context`s ---
    // Within one `Context`, one spelling is one referent, by contract.
    // The fruit and the company therefore live in separate `Context`s, and
    // the same anchor string retrieves only its own 文脈's facts.
    let mut fruit = Context::default();
    fruit.associate("Apple", "分類", "果物", 1.0).unwrap();
    fruit.associate("Apple", "味", "甘い", 1.0).unwrap();

    let mut company = Context::default();
    company
        .associate("Apple", "本社所在地", "クパチーノ", 1.0)
        .unwrap();
    company
        .associate("Apple", "開発する", "iPhone", 1.0)
        .unwrap();

    println!("=== fruit.explore([\"Apple\"], 2) ===");
    for r in fruit.explore(&["Apple"], 2) {
        println!(
            "  {} -({})-> {}",
            r.association.subject, r.association.label, r.association.object
        );
    }
    println!("=== company.explore([\"Apple\"], 2) ===");
    for r in company.explore(&["Apple"], 2) {
        println!(
            "  {} -({})-> {}",
            r.association.subject, r.association.label, r.association.object
        );
    }
}
