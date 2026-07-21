use crate::deadline::{Deadline, DeadlineExceeded};

use super::{
    Context, EntryIndex, MatchKind, Resolution, clamp_unit_or, normalize, sort_resolutions,
};

impl Context {
    /// Maps free-form wording onto stored concept names, returning scored
    /// candidates, best first. This is the entry point for retrieval:
    /// `explore` and `activate` need exact concept names as origins, and a
    /// querying LLM rarely repeats a stored spelling exactly ("選考会" for
    /// "10大脅威選考会").
    ///
    /// The same check belongs on the write path: before minting a new
    /// concept spelling, an ingester should ask what similar spellings
    /// already exist and reuse one — check before mint.
    ///
    /// This tier is lexical, in two layers sharing one score scale: names
    /// are compared in normalized form (Unicode NFKC, case-folded,
    /// katakana folded to hiragana — so full-width romaji and kana
    /// variants land), containment either way scores by how much of the
    /// longer string the shorter one covers (exact match = 1.0), and
    /// near-miss spellings that containment cannot catch (青嶺酒蔵 for
    /// 青嶺酒造) match by character-bigram overlap through a posting
    /// index. It deliberately does NOT guess at semantic similarity —
    /// that belongs to the tiers above it: an LLM normalizing its query
    /// wording, or, at corpus scale, an embedding index over concept
    /// names (only the *entry* needs fuzziness; the knowledge retrieval
    /// itself stays structural). The scan is O(number of concepts) and
    /// allocates nothing per candidate.
    pub fn resolve(&self, cue: &str) -> Vec<Resolution> {
        self.resolve_with_floor(cue, self.dice_floor())
    }

    /// [`Context::resolve`] with an explicit fuzzy floor for this one
    /// call, overriding the context's setting — the loosen-and-retry
    /// move after a miss, or a tightening when a cue is known-exact.
    /// Only the fuzzy tier is affected; exact and containment matches
    /// are never floored.
    pub fn resolve_with_floor(&self, cue: &str, dice_floor: f64) -> Vec<Resolution> {
        self.scored_resolutions(&self.concept_index, Self::concept_name, cue, dice_floor)
    }

    /// [`Context::resolve`] for relation labels instead of concepts — the
    /// two namespaces never mix. This exists chiefly for the write path:
    /// label vocabulary forks silently ("創業年" vs "創業" vs "設立年"),
    /// and once forked, label-pinned queries stop seeing half the facts.
    /// An ingester should call this (or review [`Context::labels`]) before
    /// coining a relation spelling, and reuse a close existing label
    /// instead.
    pub fn resolve_label(&self, cue: &str) -> Vec<Resolution> {
        self.resolve_label_with_floor(cue, self.dice_floor())
    }

    /// [`Context::resolve_label`] with an explicit fuzzy floor for this
    /// one call, as [`Context::resolve_with_floor`].
    pub fn resolve_label_with_floor(&self, cue: &str, dice_floor: f64) -> Vec<Resolution> {
        self.scored_resolutions(&self.label_index, Self::label_name, cue, dice_floor)
    }

    /// The shared scoring behind both resolve entry points: score the
    /// cue against one namespace's entry index, materialize the winning
    /// records as named resolutions, best first. An exact hit whose
    /// record answers to a different canonical spelling was an alias
    /// match — refined here, where the names are known.
    fn scored_resolutions(
        &self,
        index: &EntryIndex,
        name_of: fn(&Self, u32) -> &str,
        cue: &str,
        dice_floor: f64,
    ) -> Vec<Resolution> {
        let needle = normalize(cue);
        let mut resolutions: Vec<Resolution> = index
            .resolve(cue, clamp_unit_or(dice_floor, 1.0))
            .into_iter()
            .map(|(id, (score, kind))| {
                let name = name_of(self, id).to_string();
                let kind = if kind == MatchKind::Exact && normalize(&name) != needle {
                    MatchKind::Alias
                } else {
                    kind
                };
                Resolution { name, score, kind }
            })
            .collect();
        sort_resolutions(&mut resolutions);
        resolutions
    }

    /// Concept pairs whose spellings look like accidental forks of one
    /// referent — the lexical half of a vocabulary audit. Spelling drift
    /// fails silently in this system (two spellings = two referents =
    /// queries see half the facts), so this surfaces (name_a, name_b,
    /// dice) candidates, strongest first, for review. Candidates, not
    /// verdicts: containment pairs are often legitimately distinct
    /// (青嶺 the brand vs 青嶺酒造 the brewery). Aliases pointing at one
    /// record are intentional and never reported.
    pub fn similar_concepts(
        &self,
        dice_floor: f64,
        deadline: Deadline,
    ) -> Result<Vec<(String, String, f64)>, DeadlineExceeded> {
        self.scored_twins(
            &self.concept_index,
            Self::concept_name,
            dice_floor,
            deadline,
        )
    }

    /// [`Context::similar_concepts`] for relation labels — where forks
    /// hurt most, since label-pinned queries silently miss the twin.
    pub fn similar_labels(
        &self,
        dice_floor: f64,
        deadline: Deadline,
    ) -> Result<Vec<(String, String, f64)>, DeadlineExceeded> {
        self.scored_twins(&self.label_index, Self::label_name, dice_floor, deadline)
    }

    /// The shared sweep behind both twin detectors: run one namespace's
    /// entry index name-against-name and materialize the flagged pairs.
    fn scored_twins(
        &self,
        index: &EntryIndex,
        name_of: fn(&Self, u32) -> &str,
        dice_floor: f64,
        deadline: Deadline,
    ) -> Result<Vec<(String, String, f64)>, DeadlineExceeded> {
        Ok(index
            .twins(clamp_unit_or(dice_floor, 1.0), deadline)?
            .into_iter()
            .map(|(a, b, dice)| {
                (
                    name_of(self, a).to_string(),
                    name_of(self, b).to_string(),
                    dice,
                )
            })
            .collect())
    }
}
