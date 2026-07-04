//! QA golden set over the fictional 青嶺酒造 corpus (the same five
//! paragraphs as examples/paragraph_corpus.rs, ground truth fully
//! known). Each question carries the cues an LLM client would extract
//! from its wording and the facts needed to answer it; retrieval runs
//! the documented loop mechanically — resolve every cue (concepts and
//! labels), activate from the resolved origins, and role-pinned
//! query_any on them — and a question counts as answered only when
//! every needed fact came back.
//!
//! This is the regression floor for the retrieval entry: vocabulary
//! discipline, normalization, fuzzy matching, and aliases all change
//! whether these cues land, and this test turns that into a number.
//! Run with `--nocapture` for the per-question table.

use associative_rag::context::Context;

/// The corpus: 34 assertions across five paragraphs, plus the aliases
/// an operator registered while tuning (the 設立年/創業年 fork healed
/// entry-side, an English spelling for the brewery).
fn corpus() -> Context {
    let mut context = Context::default();
    let mut assert_from = |s: &str, l: &str, o: &str, w: f64, p: &str| {
        context.associate_from(s, l, o, w, p).unwrap();
    };

    // 第1段落 (概要)
    assert_from("青嶺酒造", "業種", "日本酒の蔵元", 1.0, "第1段落");
    assert_from("青嶺酒造", "所在地", "霧沢町", 1.0, "第1段落");
    assert_from("霧沢町", "所在する県", "雲居県", 1.0, "第1段落");
    assert_from("青嶺酒造", "創業年", "1907年", 1.0, "第1段落");
    assert_from("青嶺酒造", "当主", "六代目当主", 1.0, "第1段落");
    assert_from("青嶺酒造", "代表銘柄", "青嶺", 1.0, "第1段落");
    assert_from("青嶺", "出荷先", "全国", 1.0, "第1段落");
    // 第2段落 (製法)
    assert_from("青嶺酒造", "仕込み水", "雲居山の伏流水", 1.0, "第2段落");
    assert_from("青嶺酒造", "原料米", "山田錦", 1.0, "第2段落");
    assert_from("山田錦", "精米歩合", "50パーセント", 1.0, "第2段落");
    assert_from("高瀬", "監督する", "麹造り", 1.0, "第2段落");
    assert_from("蔵人", "手作業で行う", "麹造り", 1.0, "第2段落");
    assert_from("青嶺酒造", "行う", "大量生産", -1.0, "第2段落");
    assert_from("青嶺酒造", "杜氏", "高瀬", 1.0, "第2段落");
    assert_from("蔵人", "所属", "青嶺酒造", 1.0, "第2段落");
    // 第3段落 (人)
    assert_from("高瀬", "役職", "杜氏", 1.0, "第3段落");
    assert_from("高瀬", "出身", "南部杜氏", 1.0, "第3段落");
    assert_from("高瀬", "経験年数", "30年以上", 1.0, "第3段落");
    assert_from("蔵人", "住み込む場所", "蔵", 1.0, "第3段落");
    assert_from("蔵人", "住み込む期間", "冬の仕込み期間", 1.0, "第3段落");
    assert_from("六代目当主", "担う", "経営", 1.0, "第3段落");
    assert_from("六代目当主", "担う", "販売", 1.0, "第3段落");
    assert_from("六代目当主", "口を出す", "造り", -1.0, "第3段落");
    // 第4段落 (製品と評価)
    assert_from("青嶺", "分類", "辛口の純米酒", 1.0, "第4段落");
    assert_from("青嶺", "受賞する", "金賞", 1.0, "第4段落");
    assert_from("金賞", "授与元", "全国新酒鑑評会", 1.0, "第4段落");
    assert_from("金賞", "要因", "雲居山の伏流水", 1.0, "第4段落");
    assert_from("金賞", "要因", "山田錦", 1.0, "第4段落");
    // 第5段落 (地域)
    assert_from("霧沢町", "力を入れる", "酒蔵観光", 1.0, "第5段落");
    assert_from("青嶺酒造", "開く", "蔵開きの祭り", 1.0, "第5段落");
    assert_from("蔵開きの祭り", "開催時期", "毎年春", 1.0, "第5段落");
    assert_from("蔵開きの祭り", "ふるまう", "新酒", 1.0, "第5段落");
    assert_from("新酒", "仕込み水", "雲居山の伏流水", 1.0, "第5段落");
    assert_from("青嶺酒造", "仕込み水", "雲居山の伏流水", 1.0, "第5段落");

    // Tuning: aliases registered after observing missed wordings.
    context.add_label_alias("設立年", "創業年").unwrap();
    context
        .add_concept_alias("Aomine Brewery", "青嶺酒造")
        .unwrap();
    context
}

struct Question {
    ask: &'static str,
    cues: &'static [&'static str],
    needed: &'static [(&'static str, &'static str, &'static str)],
}

const QUESTIONS: &[Question] = &[
    Question {
        ask: "青嶺酒造の代表銘柄は?",
        cues: &["青嶺酒造", "代表銘柄"],
        needed: &[("青嶺酒造", "代表銘柄", "青嶺")],
    },
    Question {
        ask: "青嶺はどんな酒?",
        cues: &["青嶺"],
        needed: &[("青嶺", "分類", "辛口の純米酒")],
    },
    Question {
        ask: "杜氏は誰で、経験は?",
        cues: &["青嶺酒造", "杜氏"],
        needed: &[
            ("青嶺酒造", "杜氏", "高瀬"),
            ("高瀬", "経験年数", "30年以上"),
        ],
    },
    Question {
        ask: "金賞の要因になった水は? (何が受賞した?)",
        cues: &["金賞"],
        needed: &[
            ("金賞", "要因", "雲居山の伏流水"),
            ("青嶺", "受賞する", "金賞"),
        ],
    },
    Question {
        ask: "大量生産はしている? (否定重みの事実)",
        cues: &["青嶺酒造", "大量生産"],
        needed: &[("青嶺酒造", "行う", "大量生産")],
    },
    Question {
        ask: "仕込み水は? (2段落が裏付け)",
        cues: &["青嶺酒造", "仕込み水"],
        needed: &[("青嶺酒造", "仕込み水", "雲居山の伏流水")],
    },
    Question {
        ask: "祭りでふるまわれる新酒の仕込み水は? (2ホップ)",
        cues: &["蔵開きの祭り"],
        needed: &[
            ("蔵開きの祭り", "ふるまう", "新酒"),
            ("新酒", "仕込み水", "雲居山の伏流水"),
        ],
    },
    Question {
        ask: "青嶺酒蔵の銘柄は? (蔵/造の誤字 — bigram 入口)",
        cues: &["青嶺酒蔵"],
        needed: &[("青嶺酒造", "代表銘柄", "青嶺")],
    },
    Question {
        ask: "1907年に何が? (全角数字 — NFKC 入口)",
        cues: &["１９０７年"],
        needed: &[("青嶺酒造", "創業年", "1907年")],
    },
    Question {
        ask: "設立はいつ? (設立年→創業年のラベルエイリアス)",
        cues: &["青嶺酒造", "設立年"],
        needed: &[("青嶺酒造", "創業年", "1907年")],
    },
    Question {
        ask: "Aomine Brewery の所在地は? (概念エイリアス)",
        cues: &["Aomine Brewery"],
        needed: &[("青嶺酒造", "所在地", "霧沢町")],
    },
];

/// The mechanical retrieval loop an LLM client runs: resolve every cue
/// against both namespaces, activate from the resolved origins, then
/// role-pinned queries on them (plus the label-narrowed query when a
/// cue resolved as a label).
fn retrieve(context: &Context, cues: &[&str]) -> Vec<(String, String, String)> {
    let mut origins: Vec<String> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    for cue in cues {
        if let Some(top) = context.resolve(cue).into_iter().next()
            && !origins.contains(&top.name)
        {
            origins.push(top.name);
        }
        if let Some(top) = context.resolve_label(cue).into_iter().next()
            && !labels.contains(&top.name)
        {
            labels.push(top.name);
        }
    }
    let origin_refs: Vec<&str> = origins.iter().map(String::as_str).collect();
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

    let triple = |a: associative_rag::context::Association| (a.subject, a.label, a.object);
    let mut facts: Vec<(String, String, String)> = Vec::new();
    facts.extend(
        context
            .activate(&origin_refs, 0.5, 20)
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

#[test]
fn golden_questions_recall_every_needed_fact() {
    let context = corpus();

    let mut needed_total = 0usize;
    let mut needed_hit = 0usize;
    let mut unanswered: Vec<&str> = Vec::new();

    println!("\n=== QA needed-fact recall ===");
    for question in QUESTIONS {
        let facts = retrieve(&context, question.cues);
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
        QUESTIONS.len() - unanswered.len(),
        QUESTIONS.len()
    );

    // The regression floor: every question in the golden set must stay
    // fully answered. A failure here means an entry or reachability
    // regression — read the MISS lines above.
    assert!(
        unanswered.is_empty(),
        "unanswered questions: {unanswered:?}"
    );
}
