use taguru::context::Context;
use taguru::deadline::Deadline;

// A realistic-scale ingestion test: one coherent 文脈 described by five
// paragraphs of several sentences each, with vocabulary deliberately
// recurring across paragraphs. The original text (fictional, so ground
// truth is fully known):
//
// 第1段落 (概要):
//   青嶺酒造は、雲居県霧沢町にある日本酒の蔵元である。創業は1907年で、
//   現在の当主は六代目にあたる。代表銘柄「青嶺」は、地元だけでなく全国
//   にも出荷されている。
// 第2段落 (製法):
//   青嶺酒造は、仕込み水に雲居山の伏流水を使う。原料米には主に山田錦を
//   使い、精米歩合は50パーセントまで磨く。麹造りは杜氏の高瀬が監督し、
//   蔵人が手作業で行う。青嶺酒造は効率を優先した大量生産を行わない。
// 第3段落 (人):
//   杜氏の高瀬は南部杜氏の出身で、経験は30年を超える。蔵人は冬の仕込み
//   の期間だけ蔵に住み込む。六代目当主は経営と販売を担い、造りには口を
//   出さない。
// 第4段落 (製品と評価):
//   代表銘柄「青嶺」は辛口の純米酒である。青嶺は全国新酒鑑評会で金賞を
//   受賞した。金賞の受賞は、雲居山の伏流水と山田錦の品質によるところが
//   大きい、と高瀬は語る。
// 第5段落 (地域):
//   霧沢町は酒蔵観光に力を入れており、青嶺酒造は蔵開きの祭りを毎年春に
//   開く。蔵開きの祭りでは、雲居山の伏流水で仕込んだ新酒がふるまわれる。
//
// Ingestion discipline applied below, as the extracting LLM's job:
// - Concept/label spellings are reused exactly across paragraphs
//   (雲居山の伏流水, 六代目当主, 高瀬, ...), so recurring mentions land on
//   the same nodes instead of fragmenting.
// - Negated statements ("大量生産を行わない", "口を出さない") become the
//   affirmative label with a NEGATIVE weight.
// - Each assertion carries its paragraph as the source, so the same fact
//   stated by two paragraphs (仕込み水, 第2・第5段落) lands as weight 1.0
//   (the corroborated average) with count 2 and two attributions —
//   corroboration, visibly distinct from one emphatic assertion, without
//   weight alone conflating the two.
// - Implicit membership must be made explicit: 「杜氏の高瀬」 appears inside
//   the brewery's own paragraphs, but until 青嶺酒造-杜氏->高瀬 was asserted,
//   the 高瀬・蔵人 cluster was an island unreachable from the brewery. This
//   very test caught the omission (only 24 of 31 associations reachable);
//   the two membership edges in 第2段落 below repair it.
// - Known, accepted losses: the speaker attribution "…と高瀬は語る" cannot
//   be represented (facts about facts); "冬の仕込みの期間だけ" splits into
//   two independent triples; manner ("手作業で") folds into the label.
fn main() {
    let mut context = Context::default();

    const P1: &str = "第1段落";
    const P2: &str = "第2段落";
    const P3: &str = "第3段落";
    const P4: &str = "第4段落";
    const P5: &str = "第5段落";

    // 第1段落 (概要)
    context
        .associate_from("青嶺酒造", "業種", "日本酒の蔵元", 1.0, P1, None)
        .unwrap();
    context
        .associate_from("青嶺酒造", "所在地", "霧沢町", 1.0, P1, None)
        .unwrap();
    context
        .associate_from("霧沢町", "所在する県", "雲居県", 1.0, P1, None)
        .unwrap();
    context
        .associate_from("青嶺酒造", "創業年", "1907年", 1.0, P1, None)
        .unwrap();
    context
        .associate_from("青嶺酒造", "当主", "六代目当主", 1.0, P1, None)
        .unwrap();
    context
        .associate_from("青嶺酒造", "代表銘柄", "青嶺", 1.0, P1, None)
        .unwrap();
    context
        .associate_from("青嶺", "出荷先", "全国", 1.0, P1, None)
        .unwrap();

    // 第2段落 (製法)
    context
        .associate_from("青嶺酒造", "仕込み水", "雲居山の伏流水", 1.0, P2, None)
        .unwrap();
    context
        .associate_from("青嶺酒造", "原料米", "山田錦", 1.0, P2, None)
        .unwrap();
    context
        .associate_from("山田錦", "精米歩合", "50パーセント", 1.0, P2, None)
        .unwrap();
    context
        .associate_from("高瀬", "監督する", "麹造り", 1.0, P2, None)
        .unwrap();
    context
        .associate_from("蔵人", "手作業で行う", "麹造り", 1.0, P2, None)
        .unwrap();
    context
        .associate_from("青嶺酒造", "行う", "大量生産", -1.0, P2, None)
        .unwrap();
    // 「杜氏の高瀬」「蔵人」 are mentions inside the brewery's own 製法
    // paragraph — the membership is implicit in the prose but must become
    // explicit edges, or their cluster is a disconnected island.
    context
        .associate_from("青嶺酒造", "杜氏", "高瀬", 1.0, P2, None)
        .unwrap();
    context
        .associate_from("蔵人", "所属", "青嶺酒造", 1.0, P2, None)
        .unwrap();

    // 第3段落 (人)
    context
        .associate_from("高瀬", "役職", "杜氏", 1.0, P3, None)
        .unwrap();
    context
        .associate_from("高瀬", "出身", "南部杜氏", 1.0, P3, None)
        .unwrap();
    context
        .associate_from("高瀬", "経験年数", "30年以上", 1.0, P3, None)
        .unwrap();
    context
        .associate_from("蔵人", "住み込む場所", "蔵", 1.0, P3, None)
        .unwrap();
    context
        .associate_from("蔵人", "住み込む期間", "冬の仕込み期間", 1.0, P3, None)
        .unwrap();
    context
        .associate_from("六代目当主", "担う", "経営", 1.0, P3, None)
        .unwrap();
    context
        .associate_from("六代目当主", "担う", "販売", 1.0, P3, None)
        .unwrap();
    context
        .associate_from("六代目当主", "口を出す", "造り", -1.0, P3, None)
        .unwrap();

    // 第4段落 (製品と評価)
    context
        .associate_from("青嶺", "分類", "辛口の純米酒", 1.0, P4, None)
        .unwrap();
    context
        .associate_from("青嶺", "受賞する", "金賞", 1.0, P4, None)
        .unwrap();
    context
        .associate_from("金賞", "授与元", "全国新酒鑑評会", 1.0, P4, None)
        .unwrap();
    context
        .associate_from("金賞", "要因", "雲居山の伏流水", 1.0, P4, None)
        .unwrap();
    context
        .associate_from("金賞", "要因", "山田錦", 1.0, P4, None)
        .unwrap();

    // 第5段落 (地域)
    context
        .associate_from("霧沢町", "力を入れる", "酒蔵観光", 1.0, P5, None)
        .unwrap();
    context
        .associate_from("青嶺酒造", "開く", "蔵開きの祭り", 1.0, P5, None)
        .unwrap();
    context
        .associate_from("蔵開きの祭り", "開催時期", "毎年春", 1.0, P5, None)
        .unwrap();
    context
        .associate_from("蔵開きの祭り", "ふるまう", "新酒", 1.0, P5, None)
        .unwrap();
    context
        .associate_from("新酒", "仕込み水", "雲居山の伏流水", 1.0, P5, None)
        .unwrap();
    // 第5段落 re-derives a fact 第2段落 already asserted — corroboration.
    context
        .associate_from("青嶺酒造", "仕込み水", "雲居山の伏流水", 1.0, P5, None)
        .unwrap();

    println!(
        "コーパス: 5段落 → {}連想 (主張は34回、うち仕込み水の1件は2段落から重複主張)\n",
        context.query(None, None, None).len()
    );

    let sources = |attributions: &[taguru::context::Attribution]| {
        attributions
            .iter()
            .map(|a| a.source.as_str())
            .collect::<Vec<_>>()
            .join("・")
    };

    // (1) Entry: fragments of wording land on stored names, ranked. The
    // same lexical check guards the label vocabulary on the write path —
    // check before mint.
    println!("=== resolve(\"青嶺\") / resolve(\"全国\") — 入口の候補解決 ===");
    for r in context.resolve("青嶺") {
        println!("  [{:.3}] {}", r.score, r.name);
    }
    for r in context.resolve("全国") {
        println!("  [{:.3}] {}", r.score, r.name);
    }
    println!("\n=== resolve_label(\"住み込む\") — ラベル語彙の鋳造前チェック ===");
    for r in context.resolve_label("住み込む") {
        println!("  [{:.3}] {}", r.score, r.name);
    }

    // (2) Ranked retrieval around the hub concept: the corroborated fact
    // must rank first, direct facts next (the negated one among them, sign
    // intact), and the strongest 2-hop fact should just make the cut.
    println!("\n=== activate([\"青嶺酒造\"], decay 0.5, limit 12) ===");
    for activation in context.activate(&["青嶺酒造"], 0.5, 12).1 {
        let a = &activation.association;
        println!(
            "  [{:.4}] {} -({})-> {}  weight {:?} (count {})  出典[{}]",
            activation.strength,
            a.subject,
            a.label,
            a.object,
            a.weight,
            a.count,
            sources(&a.attributions)
        );
    }

    // (3) Role-pinned lookup: a person profile assembled across paragraphs.
    println!("\n=== query(Some(\"高瀬\"), None, None) — 高瀬がすること/であること ===");
    for a in context.query(Some("高瀬"), None, None) {
        println!(
            "  {} -({})-> {}  出典[{}]",
            a.subject,
            a.label,
            a.object,
            sources(&a.attributions)
        );
    }

    // (4) Cross-paragraph assembly: everything within 2 hops of the award
    // pulls from 第1・2・4・5段落 — while 第3段落 (the people paragraph,
    // unrelated to the award) must NOT be dragged in.
    println!("\n=== explore([\"金賞\"], 2) — 受賞の周辺知識を段落横断で ===");
    for r in context.explore(&["金賞"], 2) {
        let a = &r.association;
        println!(
            "  [{}ホップ] {} -({})-> {}  出典[{}]",
            r.distance,
            a.subject,
            a.label,
            a.object,
            sources(&a.attributions)
        );
    }

    // (5) Coverage audit, now a first-class API: unreachable_from lists
    // exactly the associations no retrieval from the anchors could ever
    // return. Retrievability is the hard ceiling on everything downstream,
    // so an ingesting pipeline should run this after every document — it
    // is what exposed the 高瀬 island in the first place.
    let total = context.query(None, None, None).len();
    let orphans = context
        .unreachable_from(&["青嶺酒造"], Deadline::unbounded())
        .unwrap();
    println!(
        "\n=== 被覆率監査: unreachable_from([\"青嶺酒造\"]) → 到達不能 {}件 / 全{}件 ===",
        orphans.len(),
        total
    );
    for a in &orphans {
        println!("  {} -({})-> {}", a.subject, a.label, a.object);
    }
}
