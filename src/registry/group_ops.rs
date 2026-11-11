use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
#[cfg(test)]
use std::fs;
use std::io;
use std::path::Path;

use taguru::deadline::Deadline;

use crate::groups::{self, GroupLookup, GroupRecord, GroupTable, StoredGroup};

use super::{
    AppState, CreateGroupError, GroupRestoreOutcome, GroupView, RenameGroupError,
    RestoreGroupsError, UpdateGroupError,
};

// The name bridge (#994): `groups` are stored and nested by id, but
// every entrance here still takes NAMES — the wire addresses `groups` by
// name until #995 (paths) and #996 (references). Each entrance resolves
// a name through the table's index and refuses one that several `groups`
// share (a hand-assembled data directory; the name-addressed create and
// rename below never mint a duplicate) instead of picking one of them.
impl AppState {
    /// Registers a `group` under a freshly minted id and persists it
    /// immediately — the create twin for `groups`, without the
    /// `pending.creates` choreography: the one fsync happens under the
    /// `groups` lock, which blocks only other `group` writes (see the
    /// field's doc), so nothing here needs the reservation dance.
    ///
    /// Member validation happens under both locks (`groups` before
    /// `registry` — the documented order): `contains_id` already
    /// answers false for a `context` mid-delete, because delete()
    /// removes it and reserves it in `pending.deletes` inside one
    /// critical section. Child names are resolved against the same table
    /// the write lock already holds.
    ///
    /// The name stays unique among `groups` while the wire addresses
    /// them by name: a second create under a standing name is
    /// `AlreadyExists`, exactly as before ids existed.
    pub fn create_group(
        &self,
        name: &str,
        description: String,
        context_ids: BTreeSet<String>,
        children: BTreeSet<String>,
    ) -> Result<(), CreateGroupError> {
        if name.is_empty() {
            return Err(CreateGroupError::InvalidName);
        }
        let mut groups = self.0.groups.write();
        if groups.ids_named(name).next().is_some() {
            return Err(CreateGroupError::AlreadyExists);
        }
        // Unreachable through HTTP while the request-list cap equals
        // [`groups::MAX_GROUP_MEMBERS`] (a create can't say more names
        // than a group may hold), but the invariant belongs to the
        // registry, not to whoever happens to call it.
        check_member_caps(&context_ids, &children).map_err(CreateGroupError::OverCap)?;
        {
            let registry = self.0.registry.read();
            if let Some(missing) =
                first_missing(&context_ids, |context| registry.contains_id(context))
            {
                return Err(CreateGroupError::NoSuchContext(missing.clone()));
            }
        }
        let mut group_ids = BTreeSet::new();
        for child in &children {
            match groups.lookup(child) {
                Ok(id) => group_ids.insert(id.clone()),
                Err(GroupLookup::NotFound) => {
                    return Err(CreateGroupError::NoSuchGroup(child.clone()));
                }
                Err(GroupLookup::Ambiguous(count)) => {
                    return Err(CreateGroupError::AmbiguousGroup {
                        name: child.clone(),
                        count,
                    });
                }
            };
        }
        let id = groups::mint_group_id();
        // The nesting validator wants the prospective map, so insert
        // first and unwind on refusal — nothing escapes the write lock
        // half-done.
        groups.insert(StoredGroup {
            id: id.clone(),
            name: name.to_string(),
            description,
            context_ids,
            group_ids,
        });
        commit_group_mutation(
            &self.0.data_dir,
            &mut groups,
            &id,
            |groups| {
                groups.remove(&id);
            },
            CreateGroupError::Nesting,
            CreateGroupError::Io,
        )?;
        Ok(())
    }

    /// Applies a delta to one `group` — `context` and child removals
    /// first, then additions (a name in both ends up a member), then
    /// the description — validates the nesting that results, and
    /// persists. Nothing applies unless everything does: a refused
    /// nesting or a failed persist restores the previous record, the
    /// update twin of [`rollback_meta`].
    ///
    /// A child to ADD resolves by name across every `group`; a child to
    /// REMOVE resolves among this `group`'s own children only, so a
    /// name nothing here bundles is the same idempotent no-op it always
    /// was.
    pub fn update_group(
        &self,
        name: &str,
        description: Option<String>,
        add_context_ids: BTreeSet<String>,
        remove_context_ids: BTreeSet<String>,
        add_groups: BTreeSet<String>,
        remove_groups: BTreeSet<String>,
    ) -> Result<GroupView, UpdateGroupError> {
        let mut groups = self.0.groups.write();
        let id = match groups.lookup(name) {
            Ok(id) => id.clone(),
            Err(GroupLookup::NotFound) => return Err(UpdateGroupError::NotFound),
            Err(GroupLookup::Ambiguous(count)) => {
                return Err(UpdateGroupError::AmbiguousGroup {
                    name: name.to_string(),
                    count,
                });
            }
        };
        // The cap is judged on the delta's RESULT — removals first,
        // exactly as they will apply — and before the existence
        // lookups: counting needs none, and refusing here lets the
        // membership tests speak in names that need not exist. Children
        // are counted in the request's own terms (names).
        {
            let record = &groups.get(&id).expect("looked up above");
            let child_names = groups.view(record).groups;
            for (field, current, add, remove) in [
                (
                    "member contexts",
                    &record.context_ids,
                    &add_context_ids,
                    &remove_context_ids,
                ),
                ("child groups", &child_names, &add_groups, &remove_groups),
            ] {
                // An empty delta cannot move a set past the cap the
                // invariant already holds it under — skip the rebuild.
                if add.is_empty() && remove.is_empty() {
                    continue;
                }
                let mut prospective: BTreeSet<&str> = current.iter().map(String::as_str).collect();
                for removed in remove {
                    prospective.remove(removed.as_str());
                }
                prospective.extend(add.iter().map(String::as_str));
                if prospective.len() > groups::MAX_GROUP_MEMBERS {
                    return Err(UpdateGroupError::OverCap(field));
                }
            }
        }
        if !add_context_ids.is_empty() {
            let registry = self.0.registry.read();
            if let Some(missing) =
                first_missing(&add_context_ids, |context| registry.contains_id(context))
            {
                return Err(UpdateGroupError::NoSuchContext(missing.clone()));
            }
        }
        // `name` itself IS registered, so a self-add passes this gate —
        // and lands in the validator's lap as the smallest cycle.
        let mut add_ids = BTreeSet::new();
        for child in &add_groups {
            match groups.lookup(child) {
                Ok(child_id) => add_ids.insert(child_id.clone()),
                Err(GroupLookup::NotFound) => {
                    return Err(UpdateGroupError::NoSuchGroup(child.clone()));
                }
                Err(GroupLookup::Ambiguous(count)) => {
                    return Err(UpdateGroupError::AmbiguousGroup {
                        name: child.clone(),
                        count,
                    });
                }
            };
        }
        let mut remove_ids = BTreeSet::new();
        {
            let record = groups.get(&id).expect("looked up above");
            for child in &remove_groups {
                let matching: Vec<&String> = record
                    .group_ids
                    .iter()
                    .filter(|child_id| groups.name_of(child_id) == child)
                    .collect();
                if matching.len() > 1 {
                    return Err(UpdateGroupError::AmbiguousGroup {
                        name: child.clone(),
                        count: matching.len(),
                    });
                }
                remove_ids.extend(matching.into_iter().cloned());
            }
        }
        let record = groups.get_mut(&id).expect("looked up above");
        let previous = record.clone();
        for context in &remove_context_ids {
            record.context_ids.remove(context);
        }
        record.context_ids.extend(add_context_ids);
        for child in &remove_ids {
            record.group_ids.remove(child);
        }
        record.group_ids.extend(add_ids);
        if let Some(description) = description {
            record.description = description;
        }
        commit_group_mutation(
            &self.0.data_dir,
            &mut groups,
            &id,
            |groups| {
                *groups.get_mut(&id).expect("looked up above") = previous;
            },
            UpdateGroupError::Nesting,
            UpdateGroupError::Io,
        )?;
        let record = groups.get(&id).expect("looked up above");
        Ok(GroupView {
            id: id.clone(),
            name: record.name.clone(),
            record: groups.view(record),
        })
    }

    /// Restores a set of `group` records — import's create-or-replace
    /// twin of [`Self::create_group`]/[`Self::update_group`]: each
    /// record replaces its whole row (description and both member
    /// sets), and a `group` absent from the set is untouched, parents
    /// naming a restored `group` included. A record whose name stands
    /// replaces that `group` (its id is kept); any other is created
    /// under a freshly minted id. The SET is judged before the first
    /// write, under the `groups` write lock: every member `context`
    /// registered, every child standing or in the set itself (a child
    /// name the set carries means the set's record), both caps per
    /// record, and the nesting that results — the standing map overlaid
    /// with every record — acyclic and within
    /// [`groups::MAX_GROUP_DEPTH`]. A validation refusal applies
    /// nothing.
    ///
    /// Writes land children-first (depth order over the prospective
    /// map), so a persist failure partway strands no record dangling
    /// on an absent child; memory tracks exactly what persisted. A
    /// record equal to the standing row skips its write, so
    /// re-importing a stream converges to a no-op instead of a
    /// directory-wide fsync storm.
    pub fn restore_groups(
        &self,
        records: &[(String, GroupRecord)],
        deadline: Deadline,
    ) -> Result<Vec<(String, GroupRestoreOutcome)>, RestoreGroupsError> {
        let mut groups = self.0.groups.write();
        // Each record's id: the standing `group` its name addresses, or a
        // fresh one. Minted before the set is judged, so a refused set
        // burns ids — harmless, nothing ever saw them.
        let mut incoming: BTreeMap<&str, String> = BTreeMap::new();
        for (name, record) in records {
            if name.is_empty() {
                return Err(RestoreGroupsError::InvalidName);
            }
            if incoming.contains_key(name.as_str()) {
                return Err(RestoreGroupsError::Duplicate(name.clone()));
            }
            check_member_caps(&record.context_ids, &record.groups).map_err(|field| {
                RestoreGroupsError::OverCap {
                    group: name.clone(),
                    field,
                }
            })?;
            let id = match groups.lookup(name) {
                Ok(id) => id.clone(),
                Err(GroupLookup::NotFound) => groups::mint_group_id(),
                Err(GroupLookup::Ambiguous(count)) => {
                    return Err(RestoreGroupsError::Ambiguous {
                        group: name.clone(),
                        name: name.clone(),
                        count,
                    });
                }
            };
            incoming.insert(name, id);
        }
        let mut stored: Vec<StoredGroup> = Vec::with_capacity(records.len());
        {
            // Lock order: `groups` before `registry`, as documented on
            // the field.
            let registry = self.0.registry.read();
            for (name, record) in records {
                if let Some(missing) =
                    first_missing(&record.context_ids, |context| registry.contains_id(context))
                {
                    return Err(RestoreGroupsError::NoSuchContext {
                        group: name.clone(),
                        context: missing.clone(),
                    });
                }
                let mut group_ids = BTreeSet::new();
                for child in &record.groups {
                    let child_id = match incoming.get(child.as_str()) {
                        Some(child_id) => child_id.clone(),
                        None => match groups.lookup(child) {
                            Ok(child_id) => child_id.clone(),
                            Err(GroupLookup::NotFound) => {
                                return Err(RestoreGroupsError::NoSuchChild {
                                    group: name.clone(),
                                    child: child.clone(),
                                });
                            }
                            Err(GroupLookup::Ambiguous(count)) => {
                                return Err(RestoreGroupsError::Ambiguous {
                                    group: name.clone(),
                                    name: child.clone(),
                                    count,
                                });
                            }
                        },
                    };
                    group_ids.insert(child_id);
                }
                stored.push(StoredGroup {
                    id: incoming[name.as_str()].clone(),
                    name: name.clone(),
                    description: record.description.clone(),
                    context_ids: record.context_ids.clone(),
                    group_ids,
                });
            }
        }
        // The prospective map — what memory becomes if every write
        // lands — is what the validators judge.
        let mut prospective = groups.records().clone();
        for record in &stored {
            prospective.insert(record.id.clone(), record.clone());
        }
        let depths = groups::nesting_depths(&prospective).map_err(|violation| {
            RestoreGroupsError::Nesting(violation.rename(|id| {
                prospective
                    .get(id)
                    .map_or_else(|| id.to_string(), |record| record.name.clone())
            }))
        })?;
        let outcomes: Vec<(String, GroupRestoreOutcome)> = stored
            .iter()
            .map(|record| {
                let outcome = match groups.get(&record.id) {
                    Some(standing) if standing == record => GroupRestoreOutcome::Unchanged,
                    Some(_) => GroupRestoreOutcome::Replaced,
                    None => GroupRestoreOutcome::Created,
                };
                (record.name.clone(), outcome)
            })
            .collect();
        // Depth order over the PROSPECTIVE map alone is not enough: it
        // proves the FINAL state (every record landed) is safe, but says
        // nothing about a partial landing. If record P's STANDING
        // (current, pre-batch) children include record C, and P's own
        // batch update drops that reference, then C landing before P —
        // exactly what plain children-first depth order would do when
        // the two tie — leaves the OLD (still-referencing) P pointing at
        // the NEW (now deeper) C the instant C's write succeeds. A
        // failure on P's own write right after (`Io`/`Timeout`) then
        // strands that combination, which `nesting_depths(&prospective)`
        // above never validated — it assumed P's update, which drops C,
        // also landed. Forcing P before C in exactly this case closes
        // the gap; it never fights the existing children-first rule
        // because that rule reasons over each record's NEW children
        // (dangling-safety), while this reasons over OLD children the
        // batch is dropping — the two only ever disagree on a pair
        // whose edge is unchanged, and an unchanged edge needs no
        // ordering help here (the prospective check already covers it).
        let index_of: BTreeMap<&str, usize> = stored
            .iter()
            .enumerate()
            .map(|(index, record)| (record.id.as_str(), index))
            .collect();
        // `dependents[p]` lists every record that must land AFTER `p` —
        // the successor direction, built directly (rather than as a
        // predecessor list rescanned per placement below) so resolving
        // "what does landing `p` unblock" is a lookup, not a full-batch
        // scan.
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); stored.len()];
        let mut pending = vec![0usize; stored.len()];
        for (parent_index, new_record) in stored.iter().enumerate() {
            let Some(standing_parent) = groups.get(&new_record.id) else {
                continue;
            };
            for old_child in &standing_parent.group_ids {
                // Only a DROPPED reference needs the ordering help: a
                // KEPT one is exactly what `nesting_depths(&prospective)`
                // above already validated (parent's edge, unchanged,
                // combined with the child's new depth) regardless of
                // which of the two lands first.
                if !new_record.group_ids.contains(old_child)
                    && let Some(&child_index) = index_of.get(old_child.as_str())
                {
                    dependents[parent_index].push(child_index);
                    pending[child_index] += 1;
                }
            }
        }
        // Kahn's algorithm: repeatedly take the not-yet-ordered record
        // with the fewest still-pending prerequisites (none) and the
        // lowest prospective depth — the latter recovers the original
        // children-first order whenever no prerequisite applies. The
        // standing map is already acyclic (every prior write validated
        // it), and `dependents` only ever adds edges drawn from that
        // same acyclic graph, so every record eventually becomes ready
        // and this terminates having placed each exactly once. A
        // min-heap of ready records (rather than rescanning all of
        // `records` for the lowest-depth one every placement) keeps
        // this O(n log n) instead of O(n²) on a batch this large — the
        // whole pass runs under the exclusive `groups` lock, blocking
        // every other group operation for as long as it takes.
        let mut ready: BinaryHeap<Reverse<(usize, usize)>> = (0..stored.len())
            .filter(|&index| pending[index] == 0)
            .map(|index| Reverse((depths[stored[index].id.as_str()], index)))
            .collect();
        let mut order: Vec<usize> = Vec::with_capacity(stored.len());
        while let Some(Reverse((_, next))) = ready.pop() {
            order.push(next);
            for &dependent in &dependents[next] {
                pending[dependent] -= 1;
                if pending[dependent] == 0 {
                    ready.push(Reverse((depths[stored[dependent].id.as_str()], dependent)));
                }
            }
        }
        assert_eq!(
            order.len(),
            stored.len(),
            "the standing map's acyclicity bounds `dependents` the same way, \
             so every record eventually becomes ready"
        );
        drop(depths);
        let mut applied = 0usize;
        for &index in &order {
            // Bound the fsync-per-record storm to the request budget. A
            // stream of many tiny group records would otherwise pin
            // `groups.write()` through one fsync each, freezing every
            // group op long past the deadline the batch loop honors.
            // What landed stands (children-first order, so it never
            // dangles); re-POSTing the whole stream is exact.
            if deadline.expired() {
                return Err(RestoreGroupsError::Timeout { applied });
            }
            if outcomes[index].1 == GroupRestoreOutcome::Unchanged {
                // Already standing in the desired state — it counts as
                // landed for the Io/Timeout report below.
                applied += 1;
                continue;
            }
            let record = &stored[index];
            if let Err(error) = groups::write_group(&self.0.data_dir, record) {
                return Err(RestoreGroupsError::Io {
                    group: record.name.clone(),
                    applied,
                    error,
                });
            }
            groups.insert(record.clone());
            applied += 1;
        }
        Ok(outcomes)
    }

    /// Every `context` a scoped key must hold to restore `records`: the
    /// closures of the named `groups` over the STANDING map (what the
    /// replace would release) unioned with their closures over the
    /// prospective one (what it would address). The import gate judges
    /// `group` records with this — [`Self::group_context_closures`]'
    /// twin for the restore path, where children may be names the set
    /// itself brings.
    ///
    /// Judged in the stream's own terms, names: the standing table is
    /// folded into a name-keyed map first, every `group` sharing a name
    /// folded into one entry — a hand-assembled directory's duplicate
    /// names can only make the gate ask for MORE of the grant, never
    /// less. (The restore itself refuses such a name.)
    pub fn group_restore_involves(&self, records: &[(String, GroupRecord)]) -> BTreeSet<String> {
        let groups = self.0.groups.read();
        let mut standing: BTreeMap<String, GroupRecord> = BTreeMap::new();
        for record in groups.records().values() {
            let view = groups.view(record);
            let folded = standing.entry(record.name.clone()).or_default();
            folded.context_ids.extend(view.context_ids);
            folded.groups.extend(view.groups);
        }
        let roots: Vec<&str> = records.iter().map(|(name, _)| name.as_str()).collect();
        let mut involved = groups::context_closure(&standing, roots.iter().copied());
        let mut prospective = standing;
        for (name, record) in records {
            prospective.insert(name.clone(), record.clone());
        }
        involved.extend(groups::context_closure(&prospective, roots.iter().copied()));
        involved
    }

    /// Removes a `group` — the bundling only, never the member `contexts`
    /// nor the child `groups`. Parents bundling the `group` are swept
    /// inside the same critical section, so no reader ever observes a
    /// dangling child.
    ///
    /// One file, so no deletion marker: the memory drop and the unlink
    /// are it. The weaker guarantee is deliberate and priced in — if
    /// the unlink fails, the surviving file re-registers the `group` at
    /// the next boot, and the error message says so.
    pub fn delete_group(&self, name: &str) -> Result<io::Result<()>, GroupLookup> {
        let mut groups = self.0.groups.write();
        let id = groups.lookup(name)?.clone();
        groups.remove(&id);
        // Nesting must not outlive the child — the same sweep a
        // deleted context gets, on the child field, under the write
        // lock already held (best effort past the removal; boot
        // reconciliation heals a sweep that could not persist).
        sweep_membership(&self.0.data_dir, &mut groups, &id, |record| {
            &mut record.group_ids
        });
        Ok(groups::remove_group_file(&self.0.data_dir, &id))
    }

    /// Renames a `group`: its own record's `name` is rewritten, and
    /// nothing else moves (ADR 0045 §2.6) — the file is stemmed by id and
    /// every parent bundles it by id, so there is no file move, no parent
    /// rewrite, and no marker to resume. One staged write, so a crash
    /// leaves either the old name or the new one.
    ///
    /// The destination must not be another `group`'s name while the wire
    /// addresses `groups` by name (the create path's gate, applied to the
    /// other way a name can arrive).
    pub fn rename_group(&self, from: &str, to: &str) -> Result<(), RenameGroupError> {
        if to.is_empty() {
            return Err(RenameGroupError::InvalidName);
        }
        let mut groups = self.0.groups.write();
        let id = match groups.lookup(from) {
            Ok(id) => id.clone(),
            Err(GroupLookup::NotFound) => return Err(RenameGroupError::NotFound),
            Err(GroupLookup::Ambiguous(count)) => return Err(RenameGroupError::Ambiguous(count)),
        };
        // Checked AFTER existence, not before: a self-rename of a name
        // that does not exist is still a `NotFound`, not a silent
        // no-op success.
        if from == to {
            return Ok(());
        }
        if groups.ids_named(to).next().is_some() {
            return Err(RenameGroupError::AlreadyExists);
        }
        let mut record = groups.get(&id).expect("looked up above").clone();
        record.name = to.to_string();
        groups::write_group(&self.0.data_dir, &record).map_err(RenameGroupError::Io)?;
        groups.insert(record);
        Ok(())
    }

    /// One `group` by name, as a name-addressed caller sees it.
    pub fn group(&self, name: &str) -> Result<GroupView, GroupLookup> {
        let groups = self.0.groups.read();
        let id = groups.lookup(name)?;
        let record = groups.get(id).expect("the name index points at records");
        Ok(GroupView {
            id: id.clone(),
            name: record.name.clone(),
            record: groups.view(record),
        })
    }

    /// Union of every `context` reachable from the named `groups` — direct
    /// members plus everything nested children bundle, transitively.
    /// The scoped write gate judges a `group` by what it ADDRESSES, so
    /// this is its view; unknown names contribute nothing, and a name
    /// several `groups` share contributes all of them (the gate errs
    /// toward asking for more of the grant; the write itself refuses the
    /// name).
    pub fn group_context_closures<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> BTreeSet<String> {
        let groups = self.0.groups.read();
        let roots: Vec<&str> = names
            .into_iter()
            .flat_map(|name| groups.ids_named(name))
            .map(String::as_str)
            .collect();
        groups::context_closure(groups.records(), roots)
    }

    /// [`Self::group_context_closures`] for exactly one `group`, by id —
    /// what a directory row's fingerprint covers, so two `groups` that
    /// share a name still fingerprint apart.
    pub fn group_context_closure_of(&self, id: &str) -> BTreeSet<String> {
        groups::context_closure(self.0.groups.read().records(), [id])
    }

    /// [`Self::group_context_closures`] with existence semantics: the
    /// first name that does not resolve to exactly one `group` comes back
    /// as the error, with why, instead of contributing nothing. The
    /// cross-`context` searches resolve their `groups` targets here — a
    /// caller who NAMES a `group` deserves a `no_group` refusal, not a
    /// silently empty search — checked and walked under one lock
    /// acquisition so a concurrent `group` delete cannot slip between
    /// the two.
    pub fn resolve_groups(
        &self,
        names: &[String],
    ) -> Result<BTreeSet<String>, (String, GroupLookup)> {
        let groups = self.0.groups.read();
        let mut roots: Vec<&str> = Vec::with_capacity(names.len());
        for name in names {
            match groups.lookup(name) {
                Ok(id) => roots.push(id),
                Err(why) => return Err((name.clone(), why)),
            }
        }
        Ok(groups::context_closure(groups.records(), roots))
    }

    /// One name-ordered page of `groups` (ties by id) plus the
    /// cursor-independent total. Scope filtering is the API layer's
    /// business, as with [`AppState::directory`] — but unlike the
    /// `context` directory, which clones only `Arc` handles and can hand
    /// over the whole map, a `group`'s record IS its data, so the page is
    /// cut here under the read lock and only the survivors are cloned.
    /// `limit` is assumed at least 1: callers page through
    /// `api::clamp_page`, never the unfloored `api::clamp`, so a `0`
    /// that would otherwise `take(0)` into an empty page —
    /// indistinguishable from "no more `groups`" to a client's paging
    /// loop — never reaches here.
    pub fn group_page(&self, after: Option<&str>, limit: usize) -> (usize, Vec<GroupView>) {
        let groups = self.0.groups.read();
        let page = groups
            .in_name_order(after)
            .take(limit)
            .map(|record| GroupView {
                id: record.id.clone(),
                name: record.name.clone(),
                record: groups.view(record),
            })
            .collect();
        (groups.len(), page)
    }

    /// Drops a deleted `context` out of every `group`, persisting each
    /// touched record. Called from [`AppState::delete`] with the
    /// deletion marker already durable; best effort past that point —
    /// a rewrite that fails leaves memory correct and the file stale,
    /// which the next boot's reconciliation heals.
    pub(super) fn sweep_context_from_groups(&self, context_id: &str) {
        let mut groups = self.0.groups.write();
        sweep_membership(&self.0.data_dir, &mut groups, context_id, |record| {
            &mut record.context_ids
        });
    }
}

/// The strict-membership gate for member `contexts`: the first requested
/// id the registry does not hold, if any. Strict on purpose — an add
/// must never mint a dangling reference.
fn first_missing<'a>(
    names: impl IntoIterator<Item = &'a String>,
    exists: impl Fn(&str) -> bool,
) -> Option<&'a String> {
    names.into_iter().find(|name| !exists(name))
}

/// The per-set membership cap, judged wherever a WHOLE record is
/// stated at once (create, restore): the first over-cap set's label
/// comes back for the refusal. A delta update judges its prospective
/// result instead — see `update_group`.
fn check_member_caps(
    contexts: &BTreeSet<String>,
    children: &BTreeSet<String>,
) -> Result<(), &'static str> {
    for (field, set) in [("member contexts", contexts), ("child groups", children)] {
        if set.len() > groups::MAX_GROUP_MEMBERS {
            return Err(field);
        }
    }
    Ok(())
}

/// The shared validate→persist→rollback tail of [`AppState::create_group`]
/// and [`AppState::update_group`]: validates the nesting the
/// already-mutated table implies, then persists `id`'s record —
/// unwinding the mutation through `rollback` on either refusal, so
/// nothing escapes the caller's write lock half-done. A nesting refusal
/// names its `group` by name (looked up BEFORE the rollback, which may
/// remove the very `group` it blames). The two callers differ only in
/// what "previous" means (create removes the fresh entry, update
/// restores the cloned record) and in their error enums.
/// `restore_groups` does not share this shape: it validates its whole
/// prospective SET before the first write.
fn commit_group_mutation<E>(
    data_dir: &Path,
    groups: &mut GroupTable,
    id: &str,
    rollback: impl FnOnce(&mut GroupTable),
    on_nesting: impl FnOnce(groups::NestingViolation) -> E,
    on_io: impl FnOnce(io::Error) -> E,
) -> Result<(), E> {
    if let Err(violation) = groups::validate_nesting(groups.records()) {
        let violation = violation.rename(|blamed| groups.name_of(blamed).to_string());
        rollback(groups);
        return Err(on_nesting(violation));
    }
    let record = groups
        .get(id)
        .expect("the caller just inserted or edited it");
    if let Err(error) = groups::write_group(data_dir, record) {
        rollback(groups);
        return Err(on_io(error));
    }
    Ok(())
}

/// Shared body of the two membership sweeps — a deleted `context` out of
/// every `group`'s members, a deleted `group` out of every parent's
/// children: removes the id `stale` from the chosen set field of every
/// record and persists each touched one. Best effort by design — a
/// rewrite that fails leaves memory correct and the file stale, which
/// the next boot's reconciliation heals. Lock-free on purpose: every
/// caller already holds the `groups` write lock.
fn sweep_membership(
    data_dir: &Path,
    groups: &mut GroupTable,
    stale: &str,
    field: impl Fn(&mut StoredGroup) -> &mut BTreeSet<String>,
) {
    for record in groups.records_mut() {
        if !field(record).remove(stale) {
            continue;
        }
        if let Err(error) = groups::write_group(data_dir, record) {
            tracing::warn!(
                group = %record.name,
                group_id = %record.id,
                removed = %stale,
                %error,
                "group membership sweep not persisted; the next boot's reconciliation drops it"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::context_proptest::{GeneratedGroupOp, config as proptest_config, group_op_strategy};
    use crate::groups::GroupLookup;
    use crate::registry::ContextMeta;
    use crate::registry::test_support::scratch_dir;
    use crate::storage::{clear_persistence_fault, fail_persistence_ops_after};

    /// A hand-planted `group` id (the shape boot accepts).
    const GROUP_A: &str = "00000000-0000-4000-9000-00000000a000";

    /// A canonical, name-ordered id for a hand-planted `group`: the
    /// name's bytes, hex, in the id's last field — so ids sort exactly
    /// as the names they stand for (for names up to six bytes).
    fn fixed_id(name: &str) -> String {
        let hex: String = name.bytes().map(|byte| format!("{byte:02x}")).collect();
        format!("00000000-0000-4000-9000-{hex:0<12}")
    }

    /// The names recorded in the `.group` files on disk — what a refused
    /// or successful write did (or did not) leave there, without knowing
    /// the minted ids.
    fn group_file_names(dir: &Path) -> BTreeSet<String> {
        fs::read_dir(dir)
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("group"))
            .filter_map(|path| {
                serde_json::from_slice::<StoredGroup>(&fs::read(path).ok()?)
                    .ok()
                    .map(|record| record.name)
            })
            .collect()
    }

    #[test]
    fn group_crud_validates_members_and_survives_a_reboot() {
        let dir = scratch_dir("groups-crud");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("beer", ContextMeta::default()).unwrap();

        state
            .create_group(
                "drinks",
                "beverage knowledge".into(),
                BTreeSet::from([state.id_of("sake")]),
                BTreeSet::new(),
            )
            .unwrap();
        assert!(matches!(
            state.create_group("drinks", String::new(), BTreeSet::new(), BTreeSet::new()),
            Err(CreateGroupError::AlreadyExists)
        ));
        assert!(matches!(
            state.create_group("", String::new(), BTreeSet::new(), BTreeSet::new()),
            Err(CreateGroupError::InvalidName)
        ));
        assert!(matches!(
            state.create_group(
                "ghosts",
                String::new(),
                BTreeSet::from(["nope".to_string()]),
                BTreeSet::new()
            ),
            Err(CreateGroupError::NoSuchContext(missing)) if missing == "nope"
        ));
        // The refused create left nothing behind, in memory or on disk.
        assert!(state.group("ghosts").is_err());
        assert!(!group_file_names(&dir).contains("ghosts"));

        // Deltas: removals first, then adds — and an unknown add
        // refuses the whole update, membership untouched.
        assert!(matches!(
            state.update_group(
                "drinks",
                None,
                BTreeSet::from(["nope".to_string()]),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            ),
            Err(UpdateGroupError::NoSuchContext(missing)) if missing == "nope"
        ));
        assert_eq!(
            state.group("drinks").unwrap().record.context_ids,
            BTreeSet::from([state.id_of("sake")])
        );
        let updated = state
            .update_group(
                "drinks",
                Some("all drinks".into()),
                BTreeSet::from([state.id_of("beer")]),
                BTreeSet::from([state.id_of("sake")]),
                BTreeSet::new(),
                BTreeSet::new(),
            )
            .unwrap();
        assert_eq!(updated.record.description, "all drinks");
        assert_eq!(
            updated.record.context_ids,
            BTreeSet::from([state.id_of("beer")])
        );
        assert!(matches!(
            state.update_group(
                "ghosts",
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::NotFound)
        ));

        // The whole collection survives a reboot from disk alone.
        drop(state);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let survived = state.group("drinks").unwrap();
        assert_eq!(survived.record.description, "all drinks");
        assert_eq!(
            survived.record.context_ids,
            BTreeSet::from([state.id_of("beer")])
        );

        // Deletion removes the record and its file; the members live on.
        state.delete_group("drinks").unwrap().unwrap();
        assert!(state.group("drinks").is_err());
        assert!(state.delete_group("drinks").is_err());
        assert!(!group_file_names(&dir).contains("drinks"));
        assert_eq!(state.group_page(None, usize::MAX), (0, Vec::new()));
        assert!(state.directory().iter().any(|entry| entry.name == "beer"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn every_group_write_persistence_failure_rolls_back_and_retries() {
        let mut exhausted = false;
        for failure in 0..8 {
            let dir = scratch_dir(&format!("group-write-fault-{failure}"));
            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            state
                .create_group(
                    "drinks",
                    "old".to_string(),
                    BTreeSet::new(),
                    BTreeSet::new(),
                )
                .unwrap();

            fail_persistence_ops_after(failure);
            let first = state.update_group(
                "drinks",
                Some("new".to_string()),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            );
            let past_end = clear_persistence_fault();
            if past_end {
                assert!(first.is_ok());
            } else {
                assert!(matches!(first, Err(UpdateGroupError::Io(_))));
                assert_eq!(state.group("drinks").unwrap().record.description, "old");
                state
                    .update_group(
                        "drinks",
                        Some("new".to_string()),
                        BTreeSet::new(),
                        BTreeSet::new(),
                        BTreeSet::new(),
                        BTreeSet::new(),
                    )
                    .unwrap();
            }
            drop(state);

            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            assert_eq!(state.group("drinks").unwrap().record.description, "new");
            drop(state);
            let _ = fs::remove_dir_all(&dir);
            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "group write exceeded the sweep bound");
    }

    #[test]
    fn deleting_a_context_sweeps_it_out_of_every_group() {
        let dir = scratch_dir("groups-sweep");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("beer", ContextMeta::default()).unwrap();
        for group in ["drinks", "fermented"] {
            state
                .create_group(
                    group,
                    String::new(),
                    BTreeSet::from([state.id_of("sake"), state.id_of("beer")]),
                    BTreeSet::new(),
                )
                .unwrap();
        }

        state.delete(&state.id_of("sake")).unwrap().unwrap();

        for group in ["drinks", "fermented"] {
            assert_eq!(
                state.group(group).unwrap().record.context_ids,
                BTreeSet::from([state.id_of("beer")]),
                "'{group}' still names the deleted context"
            );
        }
        // The sweep persisted: a reboot reads the same membership.
        drop(state);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        for group in ["drinks", "fermented"] {
            assert_eq!(
                state.group(group).unwrap().record.context_ids,
                BTreeSet::from([state.id_of("beer")])
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn group_membership_is_capped_per_set_and_judged_before_existence() {
        let dir = scratch_dir("groups-cap");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("real", ContextMeta::default()).unwrap();
        // One name past the cap refuses — before existence, so the
        // names need not exist to hear it — and nothing lands.
        let over: BTreeSet<String> = (0..=groups::MAX_GROUP_MEMBERS)
            .map(|i| format!("c{i:04}"))
            .collect();
        assert!(matches!(
            state.create_group("g", String::new(), over.clone(), BTreeSet::new()),
            Err(CreateGroupError::OverCap("member contexts"))
        ));
        assert!(matches!(
            state.create_group("g", String::new(), BTreeSet::new(), over.clone()),
            Err(CreateGroupError::OverCap("child groups"))
        ));
        assert!(state.group("g").is_err());

        // A delta is judged on its RESULT: 1 member + cap-many adds is
        // one too many, but trading the member out in the same request
        // makes room — the cap passes and the existence gate speaks
        // next, proving the judgement order.
        let one = BTreeSet::from([state.id_of("real")]);
        state
            .create_group("g", String::new(), one.clone(), BTreeSet::new())
            .unwrap();
        let cap_many: BTreeSet<String> = (0..groups::MAX_GROUP_MEMBERS)
            .map(|i| format!("c{i:04}"))
            .collect();
        assert!(matches!(
            state.update_group(
                "g",
                None,
                cap_many.clone(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::OverCap("member contexts"))
        ));
        assert!(matches!(
            state.update_group(
                "g",
                None,
                cap_many.clone(),
                one.clone(),
                BTreeSet::new(),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::NoSuchContext(missing)) if missing == "c0000"
        ));
        // Child groups ride the same cap on their own set.
        assert!(matches!(
            state.update_group(
                "g",
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                over,
                BTreeSet::new()
            ),
            Err(UpdateGroupError::OverCap("child groups"))
        ));
        assert_eq!(state.group("g").unwrap().record.context_ids, one);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_replaces_whole_records_and_reports_what_stood() {
        let dir = scratch_dir("groups-restore");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("bunko", ContextMeta::default()).unwrap();
        state
            .create_group(
                "kura",
                "old".to_string(),
                BTreeSet::from([state.id_of("sake"), state.id_of("bunko")]),
                BTreeSet::new(),
            )
            .unwrap();
        // A standing parent naming the group must survive its replace:
        // a restore rewrites the record, never the references to it.
        state
            .create_group(
                "parent",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from(["kura".to_string()]),
            )
            .unwrap();

        let kura_id = state.group("kura").unwrap().id;
        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: "new".to_string(),
            context_ids: contexts.iter().map(|c| state.id_of(c)).collect(),
            groups: children.iter().map(|g| g.to_string()).collect(),
        };
        // The set references its own newcomer — parent listed first,
        // child later: order inside the set must not matter.
        let records = vec![
            ("kura".to_string(), record(&["sake"], &["kid"])),
            ("kid".to_string(), record(&["bunko"], &[])),
        ];
        let outcomes = state
            .restore_groups(&records, Deadline::unbounded())
            .unwrap();
        assert_eq!(outcomes[0].1.as_str(), "replaced");
        assert_eq!(outcomes[1].1.as_str(), "created");
        // The replace is the WHOLE record — bunko dropped, description
        // replaced — and parent still names kura.
        assert_eq!(state.group("kura").unwrap().record, records[0].1);
        assert_eq!(
            state.group("parent").unwrap().record.groups,
            BTreeSet::from(["kura".to_string()])
        );
        // Disk agrees — in ids: the replaced kura kept its id, and its
        // child reference is kid's freshly minted one.
        let kura = state.group("kura").unwrap();
        assert_eq!(kura.id, kura_id, "a replace keeps the standing id");
        let kid = state.group("kid").unwrap();
        let on_disk: StoredGroup =
            serde_json::from_slice(&fs::read(groups::group_path(&dir, &kura.id)).unwrap()).unwrap();
        assert_eq!(on_disk.group_ids, BTreeSet::from([kid.id.clone()]));
        assert_eq!(on_disk.name, "kura");

        // Restoring the same set again converges to no-ops.
        let again = state
            .restore_groups(&records, Deadline::unbounded())
            .unwrap();
        assert!(
            again
                .iter()
                .all(|(_, outcome)| outcome.as_str() == "unchanged"),
            "{again:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_reports_how_many_landed_before_an_io_failure() {
        let dir = scratch_dir("groups-restore-io-failure");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: String::new(),
            context_ids: contexts.iter().map(|c| state.id_of(c)).collect(),
            groups: children.iter().map(|g| g.to_string()).collect(),
        };

        // "a" and "b" are independent (no nesting), so equal depth keeps
        // them in list order — "a" lands first. "b"'s file is named by
        // an id minted inside the call, so its path cannot be occupied
        // ahead of time; instead the persistence injector is armed at
        // exactly as many steps as one whole `group` write takes
        // (measured here, on a scratch create), isolating an Io failure
        // strictly after "a" already landed.
        let steps_per_write = (0..16)
            .find(|&after| {
                fail_persistence_ops_after(after);
                let landed = state
                    .create_group("probe", String::new(), BTreeSet::new(), BTreeSet::new())
                    .is_ok();
                clear_persistence_fault();
                landed
            })
            .expect("one group write fits in 16 persistence steps");
        state.delete_group("probe").unwrap().unwrap();

        fail_persistence_ops_after(steps_per_write);
        let error = state
            .restore_groups(
                &[
                    ("a".to_string(), record(&[], &[])),
                    ("b".to_string(), record(&[], &[])),
                ],
                Deadline::unbounded(),
            )
            .unwrap_err();
        clear_persistence_fault();
        assert!(matches!(
            &error,
            RestoreGroupsError::Io { group, .. } if group == "b"
        ));
        assert_eq!(error.applied(), 1, "\"a\" already landed when \"b\" failed");
        assert!(
            state.group("a").is_ok(),
            "\"a\"'s successful write must not be rolled back"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// The scenario a plain depth-tie-break order gets wrong: "a"
    /// currently points at "b" (standing, depth 2). This batch deepens
    /// "b" to a fresh depth-3 chain AND, in the same batch, moves "a"
    /// off "b" onto an unrelated depth-3 chain — so both records tie at
    /// prospective depth 3, same as the finding that motivated this
    /// test. Sorting purely by that tied depth (stable, so list order
    /// wins the tie) would write "b" first; if "a"'s own write then
    /// fails, the standing map is left with the OLD "a" (still -> "b")
    /// beside the NEW "b" (now depth 3) — combined depth 4, over
    /// `MAX_GROUP_DEPTH` (3), and never validated because the
    /// prospective check assumed "a" also landed, dropping the "b"
    /// reference entirely. The fix orders "a" before "b" whenever "a"
    /// is dropping a standing reference to it, so failing "a"'s write
    /// (forced here via the persistence fault injector, on the very
    /// first op) leaves the untouched, still-valid standing state
    /// instead — "b" is never even attempted.
    #[test]
    fn restore_groups_never_stitches_an_old_parent_to_a_freshly_deepened_child() {
        let dir = scratch_dir("groups-restore-depth-race");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = |children: &[&str]| GroupRecord {
            description: String::new(),
            context_ids: BTreeSet::new(),
            groups: children.iter().map(|name| name.to_string()).collect(),
        };
        let child_set = |names: &[&str]| -> BTreeSet<String> {
            names.iter().map(|name| name.to_string()).collect()
        };

        // Two independent depth-2 chains standing, plus "a" -> "b" —
        // everything comfortably within the depth cap before the batch.
        state
            .create_group("leaf", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();
        state
            .create_group("mid", String::new(), BTreeSet::new(), child_set(&["leaf"]))
            .unwrap();
        state
            .create_group("z", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();
        state
            .create_group("y", String::new(), BTreeSet::new(), child_set(&["z"]))
            .unwrap();
        state
            .create_group("b", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();
        state
            .create_group("a", String::new(), BTreeSet::new(), child_set(&["b"]))
            .unwrap();

        // "b" first in the list — the order a plain depth sort would
        // keep on a tie, since it matches the finding's own scenario.
        let batch = [
            ("b".to_string(), record(&["mid"])), // b -> mid -> leaf: depth 3
            ("a".to_string(), record(&["y"])),   // a -> y -> z: depth 3, drops "b"
        ];

        fail_persistence_ops_after(0);
        let error = state
            .restore_groups(&batch, Deadline::unbounded())
            .unwrap_err();
        assert!(
            !clear_persistence_fault(),
            "the fault must have fired, not run out of persistence steps first"
        );
        assert!(
            matches!(&error, RestoreGroupsError::Io { group, .. } if group == "a"),
            "the fix orders \"a\" first, so it is \"a\"'s write that hits the fault: {error:?}"
        );
        assert_eq!(error.applied(), 0, "\"a\" failed before anything landed");

        // Nothing changed: "b" must never have been attempted, so the
        // pre-batch (valid, depth <= 3 everywhere) standing state
        // stands exactly as it was.
        assert_eq!(state.group("a").unwrap().record.groups, child_set(&["b"]));
        assert_eq!(state.group("b").unwrap().record.groups, BTreeSet::new());
        let standing = state.0.groups.read().records().clone();
        groups::nesting_depths(&standing)
            .expect("the untouched standing map must still be within the depth cap");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_counts_an_unchanged_record_as_already_landed() {
        let dir = scratch_dir("groups-restore-unchanged-then-io-failure");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: String::new(),
            context_ids: contexts.iter().map(|c| state.id_of(c)).collect(),
            groups: children.iter().map(|g| g.to_string()).collect(),
        };

        // Land "a" first so the call below finds it standing exactly as
        // given and takes the `Unchanged` branch, which counts toward
        // `applied` without writing anything.
        state
            .restore_groups(
                &[("a".to_string(), record(&[], &[]))],
                Deadline::unbounded(),
            )
            .unwrap();

        // "b" is new, so it takes the `write_group` branch; "a" writes
        // nothing, so failing the very first persistence step isolates an
        // Io failure strictly after "a" is judged unchanged.
        fail_persistence_ops_after(0);
        let error = state
            .restore_groups(
                &[
                    ("a".to_string(), record(&[], &[])),
                    ("b".to_string(), record(&[], &[])),
                ],
                Deadline::unbounded(),
            )
            .unwrap_err();
        assert!(
            !clear_persistence_fault(),
            "the injected failure must be what refused \"b\""
        );
        assert!(matches!(
            &error,
            RestoreGroupsError::Io { group, .. } if group == "b"
        ));
        assert_eq!(
            error.applied(),
            1,
            "\"a\" must count as landed via the unchanged branch"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_stops_at_an_expired_deadline_without_writing() {
        use std::time::Duration;

        let dir = scratch_dir("groups-restore-timeout");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = GroupRecord {
            description: String::new(),
            context_ids: BTreeSet::new(),
            groups: BTreeSet::new(),
        };

        // A budget already spent must bound the fsync-per-record loop
        // before its first write, the way the batch loop bounds itself —
        // a stream of many tiny group records must not pin the write
        // lock past the deadline every other group op honors.
        let deadline = Deadline::after(Duration::ZERO);
        std::thread::sleep(Duration::from_millis(1));
        let error = state
            .restore_groups(&[("a".to_string(), record)], deadline)
            .unwrap_err();
        assert!(matches!(&error, RestoreGroupsError::Timeout { applied: 0 }));
        assert_eq!(error.applied(), 0);

        // Nothing landed — not in memory, not on disk.
        assert!(
            state.group("a").is_err(),
            "no record may land past the budget"
        );
        assert!(
            !group_file_names(&dir).contains("a"),
            "no group file may be written past the budget"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_judges_the_whole_set_before_writing_anything() {
        use crate::groups::NestingViolation;

        let dir = scratch_dir("groups-restore-refuse");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: String::new(),
            context_ids: contexts.iter().map(|c| state.id_of(c)).collect(),
            groups: children.iter().map(|g| g.to_string()).collect(),
        };

        // A dangling member refuses the set — the valid record beside
        // it included.
        let refusal = state
            .restore_groups(
                &[
                    ("fine".to_string(), record(&["sake"], &[])),
                    ("broken".to_string(), record(&["ghost"], &[])),
                ],
                Deadline::unbounded(),
            )
            .unwrap_err();
        assert!(matches!(
            &refusal,
            RestoreGroupsError::NoSuchContext { group, context }
                if group == "broken" && context == "ghost"
        ));
        assert_eq!(refusal.applied(), 0);
        assert!(state.group("fine").is_err(), "nothing applies on a refusal");

        // A child neither standing nor in the set.
        assert!(matches!(
            state
                .restore_groups(&[("a".to_string(), record(&[], &["nope"]))], Deadline::unbounded())
                .unwrap_err(),
            RestoreGroupsError::NoSuchChild { group, child } if group == "a" && child == "nope"
        ));

        // A cycle the set closes with itself.
        assert!(matches!(
            state
                .restore_groups(
                    &[
                        ("a".to_string(), record(&[], &["b"])),
                        ("b".to_string(), record(&[], &["a"])),
                    ],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::Nesting(NestingViolation::Cycle(_))
        ));

        // Depth counts the standing groups too: records stacking two
        // more storeys under an existing 2-chain overflow the cap.
        state
            .create_group("mid", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();
        state
            .create_group(
                "top",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from(["mid".to_string()]),
            )
            .unwrap();
        assert!(matches!(
            state
                .restore_groups(
                    &[
                        ("mid".to_string(), record(&[], &["deep"])),
                        ("deep".to_string(), record(&[], &["deeper"])),
                        ("deeper".to_string(), record(&[], &[])),
                    ],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::Nesting(NestingViolation::TooDeep(_))
        ));
        assert!(state.group("deep").is_err());

        // One name twice is two truths for one group.
        assert!(matches!(
            state
                .restore_groups(
                    &[
                        ("dup".to_string(), record(&[], &[])),
                        ("dup".to_string(), record(&["sake"], &[])),
                    ],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::Duplicate(name) if name == "dup"
        ));

        // The cap judges each record's sets.
        let over: BTreeSet<String> = (0..=groups::MAX_GROUP_MEMBERS)
            .map(|i| format!("c{i:04}"))
            .collect();
        assert!(matches!(
            state
                .restore_groups(
                    &[(
                        "wide".to_string(),
                        GroupRecord {
                            description: String::new(),
                            context_ids: over,
                            groups: BTreeSet::new(),
                        },
                    )],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::OverCap {
                field: "member contexts",
                ..
            }
        ));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn groups_nest_to_the_cap_without_cycles_and_sweep_deleted_children() {
        use crate::groups::NestingViolation;

        let dir = scratch_dir("groups-nesting");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        for context in ["sake", "beer", "tea"] {
            state.create(context, ContextMeta::default()).unwrap();
        }
        let one = |name: &str| BTreeSet::from([name.to_string()]);
        let one_ctx = |name: &str| BTreeSet::from([state.id_of(name)]);
        state
            .create_group("leaf", String::new(), one_ctx("tea"), BTreeSet::new())
            .unwrap();
        state
            .create_group("mid", String::new(), one_ctx("beer"), one("leaf"))
            .unwrap();
        state
            .create_group("top", String::new(), one_ctx("sake"), one("mid"))
            .unwrap();

        // The closure reads through the nesting; the record stays flat.
        assert_eq!(
            state.group_context_closures(["top"]),
            ["sake", "beer", "tea"]
                .iter()
                .map(|c| state.id_of(c))
                .collect()
        );
        assert_eq!(
            state.group("top").unwrap().record.context_ids,
            one_ctx("sake")
        );

        // A fourth storey over a full chain refuses, and nothing lands.
        assert!(matches!(
            state.create_group("over", String::new(), BTreeSet::new(), one("top")),
            Err(CreateGroupError::Nesting(NestingViolation::TooDeep(_)))
        ));
        assert!(state.group("over").is_err());
        assert!(!group_file_names(&dir).contains("over"));

        // Closing the chain into a cycle refuses — the self-loop
        // included — and an unknown child answers in the group
        // namespace; the record survives every refusal unchanged.
        assert!(matches!(
            state.update_group(
                "leaf",
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                one("top"),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::Nesting(NestingViolation::Cycle(_)))
        ));
        assert!(matches!(
            state.update_group(
                "leaf",
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                one("leaf"),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::Nesting(NestingViolation::Cycle(_)))
        ));
        assert!(matches!(
            state.update_group(
                "leaf",
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                one("nope"),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::NoSuchGroup(missing)) if missing == "nope"
        ));
        assert_eq!(state.group("leaf").unwrap().record.groups, BTreeSet::new());

        // Deleting a child sweeps it out of every parent, durably.
        state.delete_group("mid").unwrap().unwrap();
        assert_eq!(state.group("top").unwrap().record.groups, BTreeSet::new());
        drop(state);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        assert_eq!(state.group("top").unwrap().record.groups, BTreeSet::new());
        assert_eq!(
            state.group("leaf").unwrap().record.context_ids,
            BTreeSet::from([state.id_of("tea")])
        );

        let _ = fs::remove_dir_all(dir);
    }

    fn assert_live_group_invariants(state: &AppState) {
        let contexts: BTreeSet<String> = state
            .directory()
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        let table = state.0.groups.read();
        let groups = table.records();
        assert_eq!(groups::validate_nesting(groups), Ok(()));
        for (id, record) in groups {
            assert_eq!(&record.id, id, "every record sits under its own id");
            assert_eq!(
                table.lookup(&record.name),
                Ok(id),
                "names stay unique and indexed"
            );
            assert!(record.context_ids.len() <= groups::MAX_GROUP_MEMBERS);
            assert!(record.group_ids.len() <= groups::MAX_GROUP_MEMBERS);
            assert!(
                record
                    .context_ids
                    .iter()
                    .all(|name| contexts.contains(name))
            );
            assert!(
                record
                    .group_ids
                    .iter()
                    .all(|child| groups.contains_key(child))
            );
        }
    }

    proptest! {
        #![proptest_config(proptest_config())]

        #[test]
        fn arbitrary_group_operations_continuously_preserve_store_invariants(
            operations in prop::collection::vec(group_op_strategy(), 1..48),
        ) {
            let dir = scratch_dir("group-properties");
            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();

            for operation in operations {
                match operation {
                    GeneratedGroupOp::CreateContext(name) => {
                        let _ = state.create(name, ContextMeta::default());
                    }
                    GeneratedGroupOp::DeleteContext(name) => {
                        let _ = state.delete(name);
                    }
                    GeneratedGroupOp::CreateGroup { name, contexts, groups } => {
                        let _ = state.create_group(
                            name,
                            String::new(),
                            contexts.into_iter().map(str::to_string).collect(),
                            groups.into_iter().map(str::to_string).collect(),
                        );
                    }
                    GeneratedGroupOp::UpdateGroup {
                        name,
                        add_contexts,
                        remove_contexts,
                        add_groups,
                        remove_groups,
                    } => {
                        let _ = state.update_group(
                            name,
                            None,
                            add_contexts.into_iter().map(str::to_string).collect(),
                            remove_contexts.into_iter().map(str::to_string).collect(),
                            add_groups.into_iter().map(str::to_string).collect(),
                            remove_groups.into_iter().map(str::to_string).collect(),
                        );
                    }
                    GeneratedGroupOp::DeleteGroup(name) => {
                        let _ = state.delete_group(name);
                    }
                    GeneratedGroupOp::RestoreGroups(records) => {
                        let records: Vec<(String, GroupRecord)> = records
                            .into_iter()
                            .map(|(name, contexts, groups)| {
                                (
                                    name.to_string(),
                                    GroupRecord {
                                        description: String::new(),
                                        context_ids: contexts.into_iter().map(str::to_string).collect(),
                                        groups: groups.into_iter().map(str::to_string).collect(),
                                    },
                                )
                            })
                            .collect();
                        let _ = state.restore_groups(&records, Deadline::unbounded());
                    }
                    GeneratedGroupOp::RenameGroup { from, to } => {
                        let _ = state.rename_group(from, to);
                    }
                }
                assert_live_group_invariants(&state);
            }

            drop(state);
            let _ = fs::remove_dir_all(dir);
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_group_persist_rolls_the_update_back() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("groups-rollback");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state
            .create_group(
                "drinks",
                "before".into(),
                BTreeSet::from([state.id_of("sake")]),
                BTreeSet::new(),
            )
            .unwrap();

        // A directory that refuses the staging write fails the persist;
        // nothing may apply in memory either.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let outcome = state.update_group(
            "drinks",
            Some("after".into()),
            BTreeSet::new(),
            BTreeSet::from([state.id_of("sake")]),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(matches!(outcome, Err(UpdateGroupError::Io(_))));
        let record = state.group("drinks").unwrap();
        assert_eq!(record.record.description, "before");
        assert_eq!(
            record.record.context_ids,
            BTreeSet::from([state.id_of("sake")])
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_group_rewrites_only_its_own_record_and_the_parent_follows_by_id() {
        let dir = scratch_dir("rename-group-happy");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state
            .create_group(
                "liquor",
                "d".into(),
                BTreeSet::from([state.id_of("sake")]),
                BTreeSet::new(),
            )
            .unwrap();
        state
            .create_group(
                "drinks",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from(["liquor".to_string()]),
            )
            .unwrap();

        let liquor_id = state.group("liquor").unwrap().id;
        let drinks = state.group("drinks").unwrap();
        let drinks_bytes = fs::read(groups::group_path(&dir, &drinks.id)).unwrap();
        state.rename_group("liquor", "spirits").unwrap();

        assert!(state.group("liquor").is_err());
        let spirits = state.group("spirits").unwrap();
        assert_eq!(spirits.record.description, "d");
        assert_eq!(
            spirits.record.context_ids,
            BTreeSet::from([state.id_of("sake")])
        );
        assert_eq!(
            state.group("drinks").unwrap().record.groups,
            BTreeSet::from(["spirits".to_string()]),
            "the parent's child reference follows the rename"
        );
        assert!(!group_file_names(&dir).contains("liquor"));
        assert!(group_file_names(&dir).contains("spirits"));
        assert_eq!(
            state.group("spirits").unwrap().id,
            liquor_id,
            "the id — and so the file — stays put"
        );
        assert!(groups::group_path(&dir, &liquor_id).exists());
        // The parent was never rewritten: its file still holds the very
        // bytes the create wrote.
        assert_eq!(
            fs::read(groups::group_path(&dir, &drinks.id)).unwrap(),
            drinks_bytes
        );

        // Persisted, not just in memory.
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.group("liquor").is_err());
        assert_eq!(
            state.group("drinks").unwrap().record.groups,
            BTreeSet::from(["spirits".to_string()])
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_group_error_cases() {
        let dir = scratch_dir("rename-group-errors");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create_group("drinks", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();
        state
            .create_group("food", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();

        assert!(matches!(
            state.rename_group("missing", "whatever"),
            Err(RenameGroupError::NotFound)
        ));
        assert!(matches!(
            state.rename_group("drinks", "food"),
            Err(RenameGroupError::AlreadyExists)
        ));
        assert!(matches!(
            state.rename_group("drinks", ""),
            Err(RenameGroupError::InvalidName)
        ));
        assert!(state.rename_group("drinks", "drinks").is_ok());
        assert!(state.group("drinks").is_ok());
        // The `from == to` short-circuit must not mask a NotFound: a
        // self-rename of a name that never existed is still a refusal.
        assert!(matches!(
            state.rename_group("missing", "missing"),
            Err(RenameGroupError::NotFound)
        ));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn every_group_delete_persistence_failure_reconciles_at_boot() {
        let mut exhausted = false;
        for failure in 0..8 {
            let dir = scratch_dir(&format!("group-delete-fault-{failure}"));
            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            state
                .create_group("leaf", String::new(), BTreeSet::new(), BTreeSet::new())
                .unwrap();
            state
                .create_group(
                    "parent",
                    String::new(),
                    BTreeSet::new(),
                    BTreeSet::from(["leaf".to_string()]),
                )
                .unwrap();

            fail_persistence_ops_after(failure);
            let _ = state.delete_group("leaf").unwrap();
            let past_end = clear_persistence_fault();
            drop(state);

            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            assert!(
                !state
                    .group("parent")
                    .unwrap()
                    .record
                    .groups
                    .contains("leaf"),
                "failure at step {failure} left a dangling child after boot"
            );
            drop(state);
            let _ = fs::remove_dir_all(&dir);
            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "group deletion exceeded the sweep bound");
    }

    #[test]
    fn boot_reconciliation_drops_dangling_members_and_rewrites_the_file() {
        let dir = scratch_dir("groups-reconcile");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let sake_id = state.id_of("sake");
        drop(state);

        // A dangling member, planted the way a crash between a
        // context's deletion and the sweep's rewrite would leave one.
        groups::write_group(
            &dir,
            &StoredGroup {
                description: "d".into(),
                context_ids: BTreeSet::from([sake_id, "gone".to_string()]),
                ..StoredGroup::empty(GROUP_A, "drinks")
            },
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        assert_eq!(
            state.group("drinks").unwrap().record.context_ids,
            BTreeSet::from([state.id_of("sake")])
        );
        // Disk is the source of truth: the fix reached the file, not
        // just memory.
        let on_disk = fs::read_to_string(groups::group_path(&dir, GROUP_A)).unwrap();
        assert!(
            !on_disk.contains("gone"),
            "the dangling member survived on disk: {on_disk}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_corrupt_group_file_keeps_its_id_and_sets_the_bytes_aside() {
        let dir = scratch_dir("groups-corrupt");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        drop(state);
        let live = groups::group_path(&dir, GROUP_A);
        fs::write(&live, b"{not json").unwrap();

        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        // The name was lost with the bytes; the id stands in for it.
        let record = state.group(GROUP_A).unwrap();
        assert_eq!(record.id, GROUP_A);
        assert_eq!(record.record, GroupRecord::default());
        // The mangled bytes moved aside for hand recovery, and a fresh
        // empty record took their place — a later write to this group
        // overwrites nothing that was never loaded.
        assert_eq!(
            fs::read(live.with_extension("group.corrupt")).unwrap(),
            b"{not json",
            "the original bytes must survive, set aside"
        );
        let on_disk: StoredGroup = serde_json::from_slice(&fs::read(&live).unwrap()).unwrap();
        assert_eq!(on_disk, StoredGroup::empty(GROUP_A, GROUP_A));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_crash_mid_repair_never_drops_the_corrupt_group_file_itself() {
        let mut exhausted = false;
        for failure in 0..6 {
            let dir = scratch_dir(&format!("groups-corrupt-fault-{failure}"));
            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            drop(state);
            let live = groups::group_path(&dir, GROUP_A);
            fs::write(&live, b"{not json").unwrap();

            fail_persistence_ops_after(failure);
            let result = AppState::boot(dir.clone(), 1 << 20, None);
            let past_end = clear_persistence_fault();

            if past_end {
                let state = result.unwrap();
                assert_eq!(state.group(GROUP_A).unwrap().record, GroupRecord::default());
                drop(state);
            } else {
                // Whatever the write reached — still the mangled bytes,
                // or already the fresh empty record — `path` must
                // resolve to SOMETHING. The write-then-set-aside order
                // guarantees it; the old set-aside-then-write order
                // could lose this file entirely between the two steps.
                assert!(
                    live.exists(),
                    "failure after {failure} successes must not drop the group file"
                );
                let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
                assert_eq!(state.group(GROUP_A).unwrap().record, GroupRecord::default());
                drop(state);
            }

            let _ = fs::remove_dir_all(&dir);
            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "group repair exceeded the sweep bound");
    }

    #[test]
    fn an_unreadable_group_file_refuses_the_boot() {
        let dir = scratch_dir("groups-unreadable");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        drop(state);
        // A directory wearing the extension: fs::read fails on it, on
        // every platform, the way a permission hiccup would — and boot
        // must refuse rather than register 'locked' over an empty
        // record a later write would persist.
        let imposter = groups::group_path(&dir, GROUP_A);
        fs::create_dir(&imposter).unwrap();
        let message = match AppState::boot(dir.clone(), 1 << 20, None) {
            Ok(_) => panic!("boot must refuse while a group file is unreadable"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains(GROUP_A), "names the group file: {message}");
        // Clearing the obstacle heals without further ceremony.
        fs::remove_dir(&imposter).unwrap();
        assert!(AppState::boot(dir.clone(), 1 << 20, None).is_ok());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn boot_reconciliation_untangles_hand_written_nesting() {
        let dir = scratch_dir("groups-reconcile-nesting");
        drop(AppState::boot(dir.clone(), 1 << 20, None).unwrap());
        // Ids sort the way the names do, so the repair's id-order walk
        // drops the same edges a name-order walk would.
        let id = |name: &str| fixed_id(name);
        let write = |name: &str, children: &[&str]| {
            groups::write_group(
                &dir,
                &StoredGroup {
                    group_ids: children.iter().map(|child| id(child)).collect(),
                    ..StoredGroup::empty(&id(name), name)
                },
            )
            .unwrap();
        };
        // A two-cycle, a four-group chain, and a child that exists
        // nowhere — none of which a running server can persist.
        write("cyc-a", &["cyc-b"]);
        write("cyc-b", &["cyc-a"]);
        write("n1", &["ghost", "n2"]);
        write("n2", &["n3"]);
        write("n3", &["n4"]);
        write("n4", &[]);

        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        // The dangling child is gone, the cycle is open, the chain fits
        // the cap — deterministically: edges re-admitted in name order,
        // so (cyc-b, cyc-a) and (n3, n4) are the ones that fall.
        let children =
            |name: &str| -> BTreeSet<String> { state.group(name).unwrap().record.groups.clone() };
        assert_eq!(children("n1"), BTreeSet::from(["n2".to_string()]));
        assert_eq!(children("n2"), BTreeSet::from(["n3".to_string()]));
        assert_eq!(children("n3"), BTreeSet::new());
        assert_eq!(children("cyc-a"), BTreeSet::from(["cyc-b".to_string()]));
        assert_eq!(children("cyc-b"), BTreeSet::new());
        // Disk is the source of truth: the repairs reached the files.
        for (group, dropped) in [("cyc-b", "cyc-a"), ("n3", "n4"), ("n1", "ghost")] {
            let on_disk = fs::read_to_string(groups::group_path(&dir, &id(group))).unwrap();
            assert!(
                !on_disk.contains(&id(dropped)),
                "'{group}' still names '{dropped}' on disk: {on_disk}"
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    /// The member cap admits EXACTLY the cap and refuses one past it —
    /// at-the-line is legal, the refusal names growth only.
    #[test]
    fn the_member_cap_admits_exactly_the_cap() {
        let at_cap: BTreeSet<String> = (0..crate::groups::MAX_GROUP_MEMBERS)
            .map(|at| format!("c{at:05}"))
            .collect();
        assert!(check_member_caps(&at_cap, &BTreeSet::new()).is_ok());
        let mut over = at_cap;
        over.insert("one-more".to_string());
        assert_eq!(
            check_member_caps(&over, &BTreeSet::new()),
            Err("member contexts")
        );
    }

    /// Plants two `groups` named "kura" (ids differ) plus a parent "p"
    /// bundling both, and boots — the hand-assembled shape the name
    /// bridge must refuse rather than resolve.
    fn boot_with_twin_names(tag: &str) -> (std::path::PathBuf, AppState) {
        let dir = scratch_dir(tag);
        drop(AppState::boot(dir.clone(), 1 << 20, None).unwrap());
        for id in [fixed_id("k1"), fixed_id("k2")] {
            groups::write_group(&dir, &StoredGroup::empty(&id, "kura")).unwrap();
        }
        groups::write_group(
            &dir,
            &StoredGroup {
                group_ids: BTreeSet::from([fixed_id("k1"), fixed_id("k2")]),
                ..StoredGroup::empty(&fixed_id("p"), "p")
            },
        )
        .unwrap();
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        (dir, state)
    }

    #[test]
    fn a_name_two_groups_share_is_refused_at_every_name_entrance() {
        let (dir, state) = boot_with_twin_names("groups-twin-names");
        let kura = || BTreeSet::from(["kura".to_string()]);
        let none = BTreeSet::<String>::new;
        assert_eq!(
            state.group("kura").map(|_| ()),
            Err(GroupLookup::Ambiguous(2))
        );
        assert_eq!(
            state.delete_group("kura").map(|_| ()),
            Err(GroupLookup::Ambiguous(2))
        );
        assert!(matches!(
            state.rename_group("kura", "x"),
            Err(RenameGroupError::Ambiguous(2))
        ));
        assert!(matches!(
            state.update_group("kura", None, none(), none(), none(), none()),
            Err(UpdateGroupError::AmbiguousGroup { name, count: 2 }) if name == "kura"
        ));
        assert!(matches!(
            state.update_group("p", None, none(), none(), kura(), none()),
            Err(UpdateGroupError::AmbiguousGroup { name, count: 2 }) if name == "kura"
        ));
        // Removing by name resolves among p's own children — both are
        // called "kura", so that is ambiguous too, and nothing moves.
        assert!(matches!(
            state.update_group("p", None, none(), none(), none(), kura()),
            Err(UpdateGroupError::AmbiguousGroup { name, count: 2 }) if name == "kura"
        ));
        assert_eq!(state.group("p").unwrap().record.groups, kura());
        assert!(matches!(
            state.create_group("q", String::new(), none(), kura()),
            Err(CreateGroupError::AmbiguousGroup { name, count: 2 }) if name == "kura"
        ));
        assert_eq!(
            state.resolve_groups(&["kura".to_string()]),
            Err(("kura".to_string(), GroupLookup::Ambiguous(2)))
        );
        let record = |children: &[&str]| GroupRecord {
            groups: children.iter().map(|child| child.to_string()).collect(),
            ..GroupRecord::default()
        };
        assert!(matches!(
            state
                .restore_groups(&[("kura".to_string(), record(&[]))], Deadline::unbounded())
                .unwrap_err(),
            RestoreGroupsError::Ambiguous { group, name, count: 2 } if group == "kura" && name == "kura"
        ));
        assert!(matches!(
            state
                .restore_groups(&[("r".to_string(), record(&["kura"]))], Deadline::unbounded())
                .unwrap_err(),
            RestoreGroupsError::Ambiguous { group, name, count: 2 } if group == "r" && name == "kura"
        ));
        // The scoped gate's lenient read folds both in; the directory
        // lists both, name first then id.
        let page = state.group_page(None, usize::MAX).1;
        assert_eq!(
            page.iter().map(|view| view.id.clone()).collect::<Vec<_>>(),
            [fixed_id("k1"), fixed_id("k2"), fixed_id("p")]
        );
        assert_eq!(state.group_page(None, usize::MAX).0, 3);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_scoped_gate_folds_every_group_a_shared_name_addresses() {
        let dir = scratch_dir("groups-twin-closure");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("beer", ContextMeta::default()).unwrap();
        let (sake, beer) = (state.id_of("sake"), state.id_of("beer"));
        drop(state);
        for (id, member) in [(fixed_id("k1"), &sake), (fixed_id("k2"), &beer)] {
            groups::write_group(
                &dir,
                &StoredGroup {
                    context_ids: BTreeSet::from([member.clone()]),
                    ..StoredGroup::empty(&id, "kura")
                },
            )
            .unwrap();
        }
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let both = BTreeSet::from([sake.clone(), beer.clone()]);
        assert_eq!(state.group_context_closures(["kura"]), both);
        assert_eq!(
            state.group_context_closure_of(&fixed_id("k1")),
            BTreeSet::from([sake.clone()]),
            "one id, one group"
        );
        assert_eq!(
            state.group_restore_involves(&[("kura".to_string(), GroupRecord::default())]),
            both,
            "the standing twins both count toward what a restore releases"
        );
        // The restore gate walks through nesting in the stream's own
        // terms (names): a standing parent's child bundles sake, so
        // replacing the parent with an empty record still RELEASES sake.
        state
            .create_group(
                "parent",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from(["kid".to_string()]),
            )
            .unwrap_err();
        state
            .create_group(
                "kid",
                String::new(),
                BTreeSet::from([sake.clone()]),
                BTreeSet::new(),
            )
            .unwrap();
        state
            .create_group(
                "parent",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from(["kid".to_string()]),
            )
            .unwrap();
        assert_eq!(
            state.group_restore_involves(&[("parent".to_string(), GroupRecord::default())]),
            BTreeSet::from([sake.clone()]),
            "a standing child's members count toward what the replace releases"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn removing_a_child_by_name_resolves_among_the_parents_own_children() {
        let dir = scratch_dir("groups-remove-by-name");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let one = |name: &str| BTreeSet::from([name.to_string()]);
        let none = BTreeSet::<String>::new;
        for name in ["a", "b"] {
            state
                .create_group(name, String::new(), none(), none())
                .unwrap();
        }
        state
            .create_group("p", String::new(), none(), one("a"))
            .unwrap();
        // A name p does not bundle (b) is the idempotent no-op it always
        // was; one it does (a) leaves; an unknown name is no error.
        let after = state
            .update_group(
                "p",
                None,
                none(),
                none(),
                none(),
                BTreeSet::from(["b".into(), "ghost".into()]),
            )
            .unwrap();
        assert_eq!(after.record.groups, one("a"));
        let after = state
            .update_group("p", None, none(), none(), one("b"), one("a"))
            .unwrap();
        assert_eq!(after.record.groups, one("b"));
        assert_eq!(after.name, "p");
        assert_eq!(after.id, state.group("p").unwrap().id);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_nesting_refusal_names_the_blamed_group_not_its_id() {
        let dir = scratch_dir("groups-nesting-names");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let one = |name: &str| BTreeSet::from([name.to_string()]);
        let none = BTreeSet::<String>::new;
        state
            .create_group("leaf", String::new(), none(), none())
            .unwrap();
        state
            .create_group("mid", String::new(), none(), one("leaf"))
            .unwrap();
        state
            .create_group("top", String::new(), none(), one("mid"))
            .unwrap();
        assert!(matches!(
            state.create_group("over", String::new(), none(), one("top")),
            Err(CreateGroupError::Nesting(groups::NestingViolation::TooDeep(blamed))) if blamed == "over"
        ));
        assert!(matches!(
            state.update_group("leaf", None, none(), none(), one("top"), none()),
            Err(UpdateGroupError::Nesting(groups::NestingViolation::Cycle(blamed)))
                if ["leaf", "mid", "top"].contains(&blamed.as_str())
        ));
        let record = |children: &[&str]| GroupRecord {
            groups: children.iter().map(|child| child.to_string()).collect(),
            ..GroupRecord::default()
        };
        assert!(matches!(
            state
                .restore_groups(&[("fresh".to_string(), record(&["top"]))], Deadline::unbounded())
                .unwrap_err(),
            RestoreGroupsError::Nesting(groups::NestingViolation::TooDeep(blamed)) if blamed == "fresh"
        ));
        let _ = fs::remove_dir_all(dir);
    }

    /// The name stays unique among `groups` while the wire addresses them
    /// by name: a rename onto another `group`'s name is refused, a rename
    /// onto a free one leaves the old name free for a fresh create, and
    /// that create mints a different id.
    #[test]
    fn a_rename_frees_the_old_name_for_a_fresh_group_with_its_own_id() {
        let dir = scratch_dir("groups-rename-free-name");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let none = BTreeSet::<String>::new;
        state
            .create_group("old", String::new(), none(), none())
            .unwrap();
        let old_id = state.group("old").unwrap().id;
        state.rename_group("old", "new").unwrap();
        state
            .create_group("old", String::new(), none(), none())
            .unwrap();
        assert_ne!(state.group("old").unwrap().id, old_id);
        assert_eq!(state.group("new").unwrap().id, old_id);
        assert!(groups::is_group_id(&old_id), "{old_id}");
        let _ = fs::remove_dir_all(dir);
    }
}
