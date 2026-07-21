use super::*;

pub(super) fn assoc(subject: &str, label: &str, object: &str, weight: f64) -> Association {
    Association {
        subject: subject.to_string(),
        label: label.to_string(),
        object: object.to_string(),
        weight,
        count: 1,
        attributions: Vec::new(),
    }
}

/// Reads the stored weight of one exact triple through the public API.
pub(super) fn weight_between(context: &Context, subject: &str, label: &str, object: &str) -> f64 {
    let matches = context.query(Some(subject), Some(label), Some(object));
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one association for {subject}/{label}/{object}"
    );
    matches[0].weight
}

pub(super) fn associate_examples(context: &mut Context) {
    // 私はりんごが好きです
    context.associate("私", "好き", "りんご", 1.0).unwrap();
    // 私もみかんは大好きです
    context.associate("私", "好き", "みかん", 2.0).unwrap();
    // 私はバナナが好きではありません
    context.associate("私", "好き", "バナナ", -1.0).unwrap();
}
