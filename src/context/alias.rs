use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::deadline::{Deadline, DeadlineExceeded};

use super::{
    AliasError, AliasRecord, Context, ContextFull, EntryIndex, arena_fits, ids_left, intern_name,
};

/// (alias → canonical) pairs for one namespace — what
/// [`Context::dead_canonical_aliases`] returns per side, the same shape
/// the alias export endpoint already uses.
type AliasMap = BTreeMap<String, String>;

/// One namespace's alias-side storage — the alias table, its entry
/// index, its name-to-id lookup map, and its name-sorted keyset index —
/// bundled so [`add_alias`] takes one namespace handle instead of four
/// separate `&mut` parameters.
struct AliasNamespace<'a> {
    aliases: &'a mut Vec<AliasRecord>,
    index: &'a mut EntryIndex,
    ids: &'a mut HashMap<String, u32>,
    alias_index: &'a mut BTreeMap<String, u32>,
}

/// Registers one alternative spelling in one namespace — the shared
/// mechanics behind [`Context::add_concept_alias`] and
/// [`Context::add_label_alias`], which differ only in which
/// [`AliasNamespace`] they pass. `full_message` names the alias table
/// in the capacity error. All alias semantics live here: resolution of
/// `canonical` through the lookup map (so aliasing to an alias lands on
/// the true canonical record), idempotent re-registration, conflict
/// refusal, and the all-or-nothing capacity checks before anything
/// mutates.
fn add_alias(
    arena: &mut Vec<u8>,
    namespace: AliasNamespace<'_>,
    alias: String,
    canonical: &str,
    full_message: &'static str,
) -> Result<(), AliasError> {
    let AliasNamespace {
        aliases,
        index,
        ids,
        alias_index,
    } = namespace;
    let Some(&target) = ids.get(canonical) else {
        return Err(AliasError::UnknownCanonical);
    };
    if let Some(&existing) = ids.get(&alias) {
        return if existing == target {
            Ok(())
        } else {
            Err(AliasError::Conflict)
        };
    }
    if !ids_left(aliases.len(), 1) {
        return Err(AliasError::Full(ContextFull(full_message)));
    }
    if !arena_fits(arena.len(), alias.len()) {
        return Err(AliasError::Full(ContextFull(
            "the string arena is out of offset space",
        )));
    }
    let (name_offset, name_len) = intern_name(arena, &alias);
    aliases.push(AliasRecord {
        name_offset,
        name_len,
        target,
    });
    index.push(&alias, target);
    alias_index.insert(alias.clone(), target);
    ids.insert(alias, target);
    Ok(())
}

/// A predicted alias-resolution target during pre-flight conflict
/// checking: either a name already interned in the context, or the
/// Nth distinct name this same batch's own associations would freshly
/// intern once applied. The two variants can never compare equal to
/// each other regardless of payload, so a placeholder index can never
/// be mistaken for a real id.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SimTarget {
    Existing(u32),
    Fresh(usize),
}

/// Predicts whether applying `aliases` in key order against `ids` (a
/// namespace's real `concept_ids`/`label_ids` — canonicals and
/// registered aliases share one lookup, exactly as [`add_alias`]
/// consults it) would hit [`AliasError::UnknownCanonical`] or
/// [`AliasError::Conflict`], without writing anything. `fresh` names
/// this same batch's own associations would newly intern once they
/// actually apply; a canonical or an alias may resolve through one of
/// those exactly as it would once the associations land, and an
/// existing name always wins over a same-named fresh one, mirroring
/// `Context::intern_concept`/`intern_label`. An alias that resolves
/// earlier in the same call is visible to later ones, so a same-batch
/// chain (`"aipa"` aliasing `"IPA"`, itself aliased two lines above)
/// predicts correctly instead of a spurious `UnknownCanonical`. Stops
/// at the first predicted rejection, matching the write path's
/// stop-at-first-rejection replay. Capacity (`AliasError::Full`) is
/// never predicted here — it depends on the live table's current
/// size, which a read-only check would have to duplicate for no
/// benefit; the real write path still catches it.
fn check_aliases<'a>(
    ids: &HashMap<String, u32>,
    aliases: &'a BTreeMap<String, String>,
    fresh: impl IntoIterator<Item = &'a str>,
) -> Result<(), (&'a str, &'a str, AliasError)> {
    let mut overlay: HashMap<&'a str, SimTarget> = HashMap::new();
    let mut next_fresh = 0usize;
    for name in fresh {
        if !ids.contains_key(name) && !overlay.contains_key(name) {
            overlay.insert(name, SimTarget::Fresh(next_fresh));
            next_fresh += 1;
        }
    }
    let resolve = |overlay: &HashMap<&'a str, SimTarget>, name: &str| -> Option<SimTarget> {
        overlay
            .get(name)
            .copied()
            .or_else(|| ids.get(name).map(|&id| SimTarget::Existing(id)))
    };
    for (alias, canonical) in aliases {
        let Some(target) = resolve(&overlay, canonical) else {
            return Err((
                alias.as_str(),
                canonical.as_str(),
                AliasError::UnknownCanonical,
            ));
        };
        match resolve(&overlay, alias) {
            Some(existing) if existing != target => {
                return Err((alias.as_str(), canonical.as_str(), AliasError::Conflict));
            }
            Some(_) => {}
            None => {
                overlay.insert(alias.as_str(), target);
            }
        }
    }
    Ok(())
}

impl Context {
    /// Predicts whether [`Context::add_concept_alias`] would reject
    /// any of `aliases` (applied in key order, matching how the
    /// registry's write path applies them) once `fresh` — names this
    /// same batch's own associations have not yet interned — are
    /// accounted for. Lets an import batch be refused before any of
    /// its steps mutate the context, instead of discovering the same
    /// rejection after earlier steps already landed. See
    /// [`check_aliases`] for the prediction rules.
    pub fn check_concept_aliases<'a>(
        &self,
        aliases: &'a BTreeMap<String, String>,
        fresh: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), (&'a str, &'a str, AliasError)> {
        check_aliases(&self.concept_ids, aliases, fresh)
    }

    /// [`Context::check_concept_aliases`] for the label namespace.
    pub fn check_label_aliases<'a>(
        &self,
        aliases: &'a BTreeMap<String, String>,
        fresh: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), (&'a str, &'a str, AliasError)> {
        check_aliases(&self.label_ids, aliases, fresh)
    }

    /// Registers an alternative spelling for an existing concept. Aliases
    /// are entry-only: every lookup — `query`, `recall`, `describe`,
    /// walk origins, `resolve` candidates, and interning on the write
    /// path — resolves the alias to its canonical concept, but results
    /// always carry the canonical spelling and the graph never grows an
    /// alias node. Registering an alias is therefore the post-hoc repair
    /// for "the knowledge exists but this wording misses it": queries
    /// with the new spelling start landing, and future ingests using it
    /// accumulate into the canonical concept instead of forking a new
    /// one.
    ///
    /// `canonical` may itself be an alias — the new alias resolves to
    /// the true canonical record. Re-registering an existing alias of
    /// the same record is a no-op `Ok`, so alias imports are idempotent.
    ///
    /// # Errors
    ///
    /// [`AliasError::UnknownCanonical`] when `canonical` is not
    /// interned; [`AliasError::Conflict`] when `alias` already resolves
    /// to a different record (aliasing two existing concepts together
    /// would be a merge, which does not exist — rebuild instead);
    /// [`AliasError::Full`] when the alias table or arena is out of
    /// space. The context is unchanged on every error.
    pub fn add_concept_alias(
        &mut self,
        alias: impl Into<String>,
        canonical: &str,
    ) -> Result<(), AliasError> {
        add_alias(
            &mut self.arena,
            AliasNamespace {
                aliases: &mut self.concept_aliases,
                index: &mut self.concept_index,
                ids: &mut self.concept_ids,
                alias_index: &mut self.concept_alias_index,
            },
            alias.into(),
            canonical,
            "the concept alias table is out of u32 ids",
        )
    }

    /// [`Context::add_concept_alias`] for relation labels — the label
    /// vocabulary is where spellings fork most often ("創業年" vs
    /// "設立年"), and a label alias heals exactly that: label-pinned
    /// queries and future ingests using either spelling land on one
    /// relation.
    ///
    /// # Errors
    ///
    /// As [`Context::add_concept_alias`], within the label namespace.
    pub fn add_label_alias(
        &mut self,
        alias: impl Into<String>,
        canonical: &str,
    ) -> Result<(), AliasError> {
        add_alias(
            &mut self.arena,
            AliasNamespace {
                aliases: &mut self.label_aliases,
                index: &mut self.label_index,
                ids: &mut self.label_ids,
                alias_index: &mut self.label_alias_index,
            },
            alias.into(),
            canonical,
            "the label alias table is out of u32 ids",
        )
    }

    /// Withdraws one alias spelling from the concept namespace — the
    /// undo for a mis-registered alias. The spelling stops resolving
    /// and becomes free to register again; the canonical record, its
    /// edges, and every other spelling stay untouched. Returns the
    /// canonical name the alias pointed at, or `None` when the exact
    /// spelling is not a concept alias — in particular a CANONICAL
    /// name is refused this way, because removal must never be able
    /// to unname a record. The spelling's arena bytes stay behind as
    /// slack (append-only storage; a few bytes per removal).
    pub fn remove_concept_alias(&mut self, alias: &str) -> Option<String> {
        let position = self
            .concept_aliases
            .iter()
            .position(|record| self.arena_str(record.name_offset, record.name_len) == alias)?;
        let record = self.concept_aliases.remove(position);
        self.arena_slack += record.name_len as usize;
        self.concept_ids.remove(alias);
        self.concept_alias_index.remove(alias);
        self.rebuild_concept_index();
        Some(self.concept_name(record.target).to_string())
    }

    /// [`Context::remove_concept_alias`] for relation labels.
    pub fn remove_label_alias(&mut self, alias: &str) -> Option<String> {
        let position = self
            .label_aliases
            .iter()
            .position(|record| self.arena_str(record.name_offset, record.name_len) == alias)?;
        let record = self.label_aliases.remove(position);
        self.arena_slack += record.name_len as usize;
        self.label_ids.remove(alias);
        self.label_alias_index.remove(alias);
        self.rebuild_label_index();
        Some(self.label_name(record.target).to_string())
    }

    /// Rebuilds the concept entry index from the records. The index is
    /// append-only (arena + bigram postings), so removal is a rebuild
    /// by design: alias curation is rare, a rebuild costs milliseconds,
    /// and resolve keeps a structure with no dead entries to skip.
    fn rebuild_concept_index(&mut self) {
        let mut index = EntryIndex::default();
        for (id, record) in self.concepts.iter().enumerate() {
            index.push(
                self.arena_str(record.name_offset, record.name_len),
                id as u32,
            );
        }
        for record in &self.concept_aliases {
            index.push(
                self.arena_str(record.name_offset, record.name_len),
                record.target,
            );
        }
        self.concept_index = index;
    }

    /// [`Context::rebuild_concept_index`] for the label namespace.
    fn rebuild_label_index(&mut self) {
        let mut index = EntryIndex::default();
        for (id, record) in self.labels.iter().enumerate() {
            index.push(
                self.arena_str(record.name_offset, record.name_len),
                id as u32,
            );
        }
        for record in &self.label_aliases {
            index.push(
                self.arena_str(record.name_offset, record.name_len),
                record.target,
            );
        }
        self.label_index = index;
    }

    /// Every concept alias as (alias, canonical) pairs in registration
    /// order — one coherent table, so workflows that treat the alias
    /// vocabulary as a unit (export names, translate, re-import) never
    /// have to walk the records.
    pub fn concept_aliases(&self) -> Vec<(&str, &str)> {
        self.alias_pairs(&self.concept_aliases, Self::concept_name)
    }

    /// Every label alias as (alias, canonical) pairs in registration
    /// order.
    pub fn label_aliases(&self) -> Vec<(&str, &str)> {
        self.alias_pairs(&self.label_aliases, Self::label_name)
    }

    /// Aliases whose canonical concept/label has gone dead — every live
    /// edge that once used it is retracted, so nothing live resolves
    /// through the alias's real name. Not itself a fault, but a candidate
    /// for a rename that never got a fresh alias. `(alias → canonical)`,
    /// same per-namespace BTreeMap shape `GET .../aliases` already uses.
    pub fn dead_canonical_aliases(
        &self,
        deadline: Deadline,
    ) -> Result<(AliasMap, AliasMap), DeadlineExceeded> {
        let mut live_concepts: BTreeSet<&str> = BTreeSet::new();
        let mut live_labels: BTreeSet<&str> = BTreeSet::new();
        for edge in &self.edges {
            if deadline.expired() {
                return Err(DeadlineExceeded);
            }
            if edge.count > 0 {
                live_concepts.insert(self.concept_name(edge.subject));
                live_concepts.insert(self.concept_name(edge.object));
                live_labels.insert(self.label_name(edge.label));
            }
        }
        Ok((
            Self::dead_aliases(self.concept_aliases(), &live_concepts),
            Self::dead_aliases(self.label_aliases(), &live_labels),
        ))
    }

    /// Filters (alias, canonical) pairs down to the ones whose canonical
    /// spelling `live` does not contain, materializing both sides as
    /// owned strings for the wire shape `dead_canonical_aliases` returns.
    fn dead_aliases<'a>(aliases: Vec<(&'a str, &'a str)>, live: &BTreeSet<&'a str>) -> AliasMap {
        aliases
            .into_iter()
            .filter(|(_, canonical)| !live.contains(canonical))
            .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
            .collect()
    }

    /// Materializes one namespace's alias table as (alias, canonical)
    /// name pairs in registration order.
    fn alias_pairs(
        &self,
        records: &[AliasRecord],
        name_of: fn(&Self, u32) -> &str,
    ) -> Vec<(&str, &str)> {
        records
            .iter()
            .map(|record| {
                (
                    self.arena_str(record.name_offset, record.name_len),
                    name_of(self, record.target),
                )
            })
            .collect()
    }

    /// [`Context::concept_alias_page`]/[`Context::label_alias_page`]'s
    /// shared body: one alias-sorted page of a `BTreeMap<String, Id>`
    /// alias index plus its cursor-independent total, seeked in
    /// O(log n + k) instead of collecting and sorting every alias.
    fn alias_page<Id: Copy>(
        index: &BTreeMap<String, Id>,
        after: Option<&str>,
        limit: usize,
        name_of: impl Fn(Id) -> String,
    ) -> (usize, Vec<(String, String)>) {
        use std::ops::Bound;

        let start = match after {
            Some(after) => Bound::Excluded(after),
            None => Bound::Unbounded,
        };
        let page = index
            .range::<str, _>((start, Bound::Unbounded))
            .take(limit)
            .map(|(alias, &target)| (alias.clone(), name_of(target)))
            .collect();
        (index.len(), page)
    }

    /// One alias-sorted page of the concept-alias namespace plus the
    /// cursor-independent total — the same `group_page`-shaped contract
    /// [`crate::registry::Groups::group_page`] already uses.
    pub fn concept_alias_page(
        &self,
        after: Option<&str>,
        limit: usize,
    ) -> (usize, Vec<(String, String)>) {
        Self::alias_page(&self.concept_alias_index, after, limit, |id| {
            self.concept_name(id).to_string()
        })
    }

    /// [`Context::concept_alias_page`] for the label-alias namespace.
    pub fn label_alias_page(
        &self,
        after: Option<&str>,
        limit: usize,
    ) -> (usize, Vec<(String, String)>) {
        Self::alias_page(&self.label_alias_index, after, limit, |id| {
            self.label_name(id).to_string()
        })
    }

    /// How many concept aliases are registered, independent of any
    /// cursor — lets a paginated caller know the concept-alias
    /// namespace's size without walking it, e.g. to decide whether a
    /// page that came up short spilled into the next namespace.
    pub fn concept_alias_count(&self) -> usize {
        self.concept_alias_index.len()
    }

    /// [`Context::concept_alias_count`] for the label-alias namespace.
    pub fn label_alias_count(&self) -> usize {
        self.label_alias_index.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::test_support::{assoc, weight_between};

    #[test]
    fn aliases_resolve_at_every_entry_point() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "創業年", "1907年", 1.0)
            .unwrap();
        context
            .add_concept_alias("Aomine Brewery", "青嶺酒造")
            .unwrap();
        context.add_label_alias("設立年", "創業年").unwrap();

        // Reads through the alias land on canonical knowledge, and the
        // results carry the canonical spelling.
        assert_eq!(
            context.query(Some("Aomine Brewery"), Some("設立年"), None),
            vec![assoc("青嶺酒造", "創業年", "1907年", 1.0)]
        );
        assert_eq!(context.recall("Aomine Brewery").len(), 1);
        assert_eq!(
            context.describe("Aomine Brewery").unwrap().concept,
            "青嶺酒造"
        );
        assert_eq!(context.explore(&["Aomine Brewery"], 1).len(), 1);

        // resolve surfaces the canonical name for an alias hit.
        assert_eq!(context.resolve("aomine")[0].name, "青嶺酒造");
        assert_eq!(context.resolve_label("設立")[0].name, "創業年");

        // Writes through the alias accumulate into the canonical concept
        // instead of forking a new one — two assertions of 1.0 average
        // to 1.0, not sum to 2.0.
        context
            .associate("Aomine Brewery", "設立年", "1907年", 1.0)
            .unwrap();
        assert_eq!(context.concept_count(), 2);
        assert_eq!(
            weight_between(&context, "青嶺酒造", "創業年", "1907年"),
            1.0
        );

        // Vocabulary views stay canonical-only; aliases live in their
        // own exportable tables.
        assert_eq!(context.labels(), vec!["創業年"]);
        assert_eq!(
            context.concept_aliases(),
            vec![("Aomine Brewery", "青嶺酒造")]
        );
        assert_eq!(context.label_aliases(), vec![("設立年", "創業年")]);
    }

    #[test]
    fn alias_conflicts_and_unknowns_are_rejected() {
        let mut context = Context::default();
        context
            .associate("IPA", "公開する", "10大脅威", 1.0)
            .unwrap();
        context
            .associate("情報処理推進機構", "所在地", "東京", 1.0)
            .unwrap();

        assert_eq!(
            context.add_concept_alias("独法", "存在しない概念"),
            Err(AliasError::UnknownCanonical)
        );
        // Two spellings that both already exist as concepts cannot be
        // aliased together — that would be a merge.
        assert_eq!(
            context.add_concept_alias("IPA", "情報処理推進機構"),
            Err(AliasError::Conflict)
        );
        // Re-registering the same mapping is idempotent; re-pointing the
        // alias elsewhere is a conflict.
        assert_eq!(
            context.add_concept_alias("機構", "情報処理推進機構"),
            Ok(())
        );
        assert_eq!(
            context.add_concept_alias("機構", "情報処理推進機構"),
            Ok(())
        );
        assert_eq!(
            context.add_concept_alias("機構", "IPA"),
            Err(AliasError::Conflict)
        );
        // Aliasing to an alias resolves to the true canonical record.
        assert_eq!(context.add_concept_alias("kikou", "機構"), Ok(()));
        assert_eq!(
            context.concept_aliases(),
            vec![("機構", "情報処理推進機構"), ("kikou", "情報処理推進機構")]
        );
    }

    #[test]
    fn a_removed_alias_stops_resolving_and_frees_its_spelling() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "創業年", "1907年", 1.0)
            .unwrap();
        context.associate("高瀬", "役職", "杜氏", 1.0).unwrap();
        context.add_concept_alias("Aomine", "青嶺酒造").unwrap();
        context.add_label_alias("設立年", "創業年").unwrap();

        // Withdrawal names what the alias pointed at; the spelling
        // stops resolving while the canonical keeps its knowledge.
        assert_eq!(
            context.remove_concept_alias("Aomine").as_deref(),
            Some("青嶺酒造")
        );
        assert!(context.query(Some("Aomine"), None, None).is_empty());
        assert!(context.resolve("aomine").is_empty());
        assert!(context.concept_aliases().is_empty());
        assert_eq!(context.recall("青嶺酒造").len(), 1);
        // The freed spelling's bytes stay behind in the arena as slack.
        assert_eq!(context.arena_slack(), "Aomine".len());

        // Not-an-alias refusals: unknown spellings, and canonical
        // names — removal must never be able to unname a record. Being
        // no-ops, neither adds further slack.
        assert_eq!(context.remove_concept_alias("Aomine"), None);
        assert_eq!(context.remove_concept_alias("青嶺酒造"), None);
        assert_eq!(context.arena_slack(), "Aomine".len());

        // The spelling is free again, pointing elsewhere this time —
        // the un-wedging move a mis-registration needs. This interns a
        // FRESH "Aomine" (append-only arena, no reuse of the freed
        // range), so slack holds at the first registration's bytes —
        // it does not grow just because the spelling was reused, and
        // it does not shrink because the new record is live.
        context.add_concept_alias("Aomine", "高瀬").unwrap();
        assert_eq!(context.describe("Aomine").unwrap().concept, "高瀬");
        assert_eq!(context.arena_slack(), "Aomine".len());

        // Labels mirror, and the removal survives an image roundtrip
        // (the rebuilt entry indexes included).
        assert_eq!(
            context.remove_label_alias("設立年").as_deref(),
            Some("創業年")
        );
        assert_eq!(context.arena_slack(), "Aomine".len() + "設立年".len());
        let reborn = Context::from_bytes(&context.to_bytes()).unwrap();
        assert_eq!(reborn.describe("Aomine").unwrap().concept, "高瀬");
        assert!(reborn.label_aliases().is_empty());
        assert_eq!(reborn.resolve("青嶺")[0].name, "青嶺酒造");
        assert_eq!(reborn.arena_slack(), context.arena_slack());
    }

    #[test]
    fn label_alias_conflicts_and_unknowns_are_rejected() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "創業年", "1907年", 1.0)
            .unwrap();
        context
            .associate("青嶺酒造", "設立地", "霧沢町", 1.0)
            .unwrap();

        assert_eq!(
            context.add_label_alias("操業開始", "存在しないラベル"),
            Err(AliasError::UnknownCanonical)
        );
        // Two spellings that both already exist as labels cannot be
        // aliased together — that would be a merge.
        assert_eq!(
            context.add_label_alias("創業年", "設立地"),
            Err(AliasError::Conflict)
        );
        // Re-registering the same mapping is idempotent; re-pointing the
        // alias elsewhere is a conflict.
        assert_eq!(context.add_label_alias("設立年", "創業年"), Ok(()));
        assert_eq!(context.add_label_alias("設立年", "創業年"), Ok(()));
        assert_eq!(
            context.add_label_alias("設立年", "設立地"),
            Err(AliasError::Conflict)
        );
        // Aliasing to an alias resolves to the true canonical record.
        assert_eq!(context.add_label_alias("founded", "設立年"), Ok(()));
        assert_eq!(
            context.label_aliases(),
            vec![("設立年", "創業年"), ("founded", "創業年")]
        );
        // The namespaces stay separate: a label spelling is not a
        // concept spelling, so it cannot anchor a concept alias.
        assert_eq!(
            context.add_concept_alias("蔵の誕生", "創業年"),
            Err(AliasError::UnknownCanonical)
        );
    }

    #[test]
    fn alias_pages_seek_each_namespace_independently() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "創業年", "1907年", 1.0)
            .unwrap();
        context.associate("高瀬", "役職", "杜氏", 1.0).unwrap();

        // Registered out of alphabetical order in both namespaces.
        context.add_concept_alias("zeta", "青嶺酒造").unwrap();
        context.add_concept_alias("alpha", "高瀬").unwrap();
        context.add_label_alias("yankee", "創業年").unwrap();
        context.add_label_alias("bravo", "役職").unwrap();

        assert_eq!(context.concept_alias_count(), 2);
        assert_eq!(context.label_alias_count(), 2);

        let (total, page) = context.concept_alias_page(None, 1);
        assert_eq!(total, 2);
        assert_eq!(page, vec![("alpha".to_string(), "高瀬".to_string())]);
        let (_, page) = context.concept_alias_page(Some("alpha"), 10);
        assert_eq!(page, vec![("zeta".to_string(), "青嶺酒造".to_string())]);

        let (total, page) = context.label_alias_page(None, 1);
        assert_eq!(total, 2);
        assert_eq!(page, vec![("bravo".to_string(), "役職".to_string())]);
        let (_, page) = context.label_alias_page(Some("bravo"), 10);
        assert_eq!(page, vec![("yankee".to_string(), "創業年".to_string())]);
    }

    #[test]
    fn dead_canonical_aliases_reports_aliases_whose_canonical_has_no_live_edges() {
        let mut context = Context::default();
        context.associate("蔵", "銘柄", "青嶺", 1.0).unwrap();
        context.add_concept_alias("あおね", "青嶺").unwrap();
        context.add_label_alias("ブランド", "銘柄").unwrap();

        // Both canonicals are still live: nothing is reported.
        let (dead_concepts, dead_labels) = context
            .dead_canonical_aliases(Deadline::unbounded())
            .unwrap();
        assert!(dead_concepts.is_empty());
        assert!(dead_labels.is_empty());

        // Retracting the only edge kills both canonicals at once.
        context.retract_association("蔵", "銘柄", "青嶺");
        let (dead_concepts, dead_labels) = context
            .dead_canonical_aliases(Deadline::unbounded())
            .unwrap();
        assert_eq!(
            dead_concepts.get("あおね").map(String::as_str),
            Some("青嶺")
        );
        assert_eq!(
            dead_labels.get("ブランド").map(String::as_str),
            Some("銘柄")
        );
    }

    #[test]
    fn dead_canonical_aliases_matches_what_compaction_would_drop() {
        let mut context = Context::default();
        context.associate("蔵", "銘柄", "青嶺", 1.0).unwrap();
        context.associate("蔵", "杜氏", "高瀬", 1.0).unwrap();
        context.add_concept_alias("あおね", "青嶺").unwrap();
        context.add_label_alias("ブランド", "銘柄").unwrap();
        // 杜氏/高瀬 stay live throughout, so this alias must never be
        // reported dead — a control against a predicate that's too broad.
        context.add_label_alias("肩書", "杜氏").unwrap();
        context.retract_association("蔵", "銘柄", "青嶺");

        let (dead_concepts, dead_labels) = context
            .dead_canonical_aliases(Deadline::unbounded())
            .unwrap();
        assert!(!dead_labels.contains_key("肩書"));

        let (_, stats) = context.compacted(Deadline::unbounded()).unwrap();
        assert_eq!(
            dead_concepts.len() + dead_labels.len(),
            stats.aliases_dropped,
            "dead_canonical_aliases must name exactly what compaction would silently drop"
        );
    }
}
