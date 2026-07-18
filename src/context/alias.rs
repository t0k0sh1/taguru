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

impl Context {
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
