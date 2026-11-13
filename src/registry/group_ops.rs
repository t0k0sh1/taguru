use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
#[cfg(test)]
use std::fs;
use std::io;
use std::path::Path;

use taguru::deadline::Deadline;

use crate::groups::{self, GroupRecord, GroupTable, StoredGroup};

use super::{
    AppState, CreateGroupError, GroupRestoreOutcome, GroupView, RenameGroupError,
    RestoreGroupsError, UpdateGroupError,
};

// `groups` are addressed by id: every entrance that names a `group` —
// the one an operation is about (#995) and every child or search target
// it lists (#996) — takes its id. Names are labels and may repeat
// (ADR 0050), so nothing here resolves one.
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
    /// critical section. Child ids are checked against the same table
    /// the write lock already holds.
    ///
    /// The name is a label, not a key: another `group` may already
    /// carry it. `id` is the caller's (a router hands every shard the
    /// one it minted, #995 decision 1) or, when `None`, minted here; the
    /// caller has checked its shape. A create under an id that already
    /// stands answers that `group` when the standing record is exactly
    /// the one asked for — a retried create converges instead of
    /// failing — and [`CreateGroupError::IdTaken`] when it differs.
    pub fn create_group(
        &self,
        id: Option<&str>,
        name: &str,
        description: String,
        context_ids: BTreeSet<String>,
        children: BTreeSet<String>,
    ) -> Result<GroupView, CreateGroupError> {
        if name.is_empty() {
            return Err(CreateGroupError::InvalidName);
        }
        let mut groups = self.0.groups.write();
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
        if let Some(missing) = first_missing(&children, |child| groups.get(child).is_some()) {
            return Err(CreateGroupError::NoSuchGroup(missing.clone()));
        }
        let requested = StoredGroup {
            id: id.map_or_else(groups::mint_group_id, str::to_string),
            name: name.to_string(),
            description,
            context_ids,
            group_ids: children,
        };
        let id = requested.id.clone();
        if let Some(standing) = groups.get(&id) {
            if *standing != requested {
                return Err(CreateGroupError::IdTaken);
            }
            return Ok(group_view(standing));
        }
        // The nesting validator wants the prospective map, so insert
        // first and unwind on refusal — nothing escapes the write lock
        // half-done.
        groups.insert(requested);
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
        Ok(group_view(groups.get(&id).expect("committed above")))
    }

    /// Applies a delta to one `group` — `context` and child removals
    /// first, then additions (a name in both ends up a member), then
    /// the description — validates the nesting that results, and
    /// persists. Nothing applies unless everything does: a refused
    /// nesting or a failed persist restores the previous record, the
    /// update twin of [`rollback_meta`].
    ///
    /// A child to ADD must be a standing `group`; a child to REMOVE that
    /// this `group` does not bundle is an idempotent no-op — the
    /// `context` rule, one namespace over.
    pub fn update_group(
        &self,
        id: &str,
        description: Option<String>,
        add_context_ids: BTreeSet<String>,
        remove_context_ids: BTreeSet<String>,
        add_group_ids: BTreeSet<String>,
        remove_group_ids: BTreeSet<String>,
    ) -> Result<GroupView, UpdateGroupError> {
        let mut groups = self.0.groups.write();
        if groups.get(id).is_none() {
            return Err(UpdateGroupError::NotFound);
        }
        let id = id.to_string();
        // The cap is judged on the delta's RESULT — removals first,
        // exactly as they will apply — and before the existence
        // lookups: counting needs none, and refusing here lets the
        // membership tests speak in ids that need not exist.
        {
            let record = &groups.get(&id).expect("looked up above");
            for (field, current, add, remove) in [
                (
                    "member contexts",
                    &record.context_ids,
                    &add_context_ids,
                    &remove_context_ids,
                ),
                (
                    "child groups",
                    &record.group_ids,
                    &add_group_ids,
                    &remove_group_ids,
                ),
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
        // The `group` itself IS registered, so adding its own id passes
        // this gate — and lands in the validator's lap as the smallest
        // cycle.
        if let Some(missing) = first_missing(&add_group_ids, |child| groups.get(child).is_some()) {
            return Err(UpdateGroupError::NoSuchGroup(missing.clone()));
        }
        let record = groups.get_mut(&id).expect("looked up above");
        let previous = record.clone();
        for context in &remove_context_ids {
            record.context_ids.remove(context);
        }
        record.context_ids.extend(add_context_ids);
        for child in &remove_group_ids {
            record.group_ids.remove(child);
        }
        record.group_ids.extend(add_group_ids);
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
        Ok(group_view(groups.get(&id).expect("looked up above")))
    }

    /// Restores a set of `group` records — import's create-or-replace
    /// twin of [`Self::create_group`]/[`Self::update_group`]: each
    /// record replaces its whole row (name, description and both member
    /// sets) under its own id, creating the `group` when no `group`
    /// holds that id (#996 — records carry their id, so a restore lands
    /// on the same `group` wherever it runs), and a `group` absent from
    /// the set is untouched, parents bundling a restored `group`
    /// included. The SET is judged before the first write, under the
    /// `groups` write lock: every member `context` registered, every
    /// child standing or in the set itself, both caps per record, and
    /// the nesting that results — the standing map overlaid with every
    /// record — acyclic and within [`groups::MAX_GROUP_DEPTH`]. A
    /// validation refusal applies nothing.
    ///
    /// Writes land children-first (depth order over the prospective
    /// map), so a persist failure partway strands no record dangling
    /// on an absent child; memory tracks exactly what persisted. A
    /// record equal to the standing row skips its write, so
    /// re-importing a stream converges to a no-op instead of a
    /// directory-wide fsync storm.
    pub fn restore_groups(
        &self,
        records: &[StoredGroup],
        deadline: Deadline,
    ) -> Result<Vec<(StoredGroup, GroupRestoreOutcome)>, RestoreGroupsError> {
        let mut groups = self.0.groups.write();
        let mut incoming: BTreeSet<&str> = BTreeSet::new();
        for record in records {
            if record.name.is_empty() {
                return Err(RestoreGroupsError::InvalidName);
            }
            if !incoming.insert(record.id.as_str()) {
                return Err(RestoreGroupsError::Duplicate(record.id.clone()));
            }
            check_member_caps(&record.context_ids, &record.group_ids).map_err(|field| {
                RestoreGroupsError::OverCap {
                    group: record.id.clone(),
                    field,
                }
            })?;
        }
        {
            // Lock order: `groups` before `registry`, as documented on
            // the field.
            let registry = self.0.registry.read();
            for record in records {
                if let Some(missing) =
                    first_missing(&record.context_ids, |context| registry.contains_id(context))
                {
                    return Err(RestoreGroupsError::NoSuchContext {
                        group: record.id.clone(),
                        context: missing.clone(),
                    });
                }
                if let Some(missing) = first_missing(&record.group_ids, |child| {
                    incoming.contains(child) || groups.get(child).is_some()
                }) {
                    return Err(RestoreGroupsError::NoSuchChild {
                        group: record.id.clone(),
                        child: missing.clone(),
                    });
                }
            }
        }
        let stored = records;
        // The prospective map — what memory becomes if every write
        // lands — is what the validators judge.
        let mut prospective = groups.records().clone();
        for record in stored {
            prospective.insert(record.id.clone(), record.clone());
        }
        let depths = groups::nesting_depths(&prospective).map_err(RestoreGroupsError::Nesting)?;
        let outcomes: Vec<(StoredGroup, GroupRestoreOutcome)> = stored
            .iter()
            .map(|record| {
                let outcome = match groups.get(&record.id) {
                    Some(standing) if standing == record => GroupRestoreOutcome::Unchanged,
                    Some(_) => GroupRestoreOutcome::Replaced,
                    None => GroupRestoreOutcome::Created,
                };
                (record.clone(), outcome)
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
            // Only a DROPPED reference needs the ordering help: a KEPT
            // one is exactly what `nesting_depths(&prospective)` above
            // already validated (parent's edge, unchanged, combined
            // with the child's new depth) regardless of which of the
            // two lands first. The whole STANDING branch under the
            // dropped child counts, not just the child: with a > b > c
            // standing and only "a" (dropping "b") and "c" (deepening)
            // in the batch, "c" landing first deepens the old "a"
            // through the untouched "b" just the same. Every edge added
            // runs from a standing ancestor to a standing descendant,
            // so `dependents` stays acyclic.
            let mut stack: Vec<&str> = standing_parent
                .group_ids
                .iter()
                .filter(|old_child| !new_record.group_ids.contains(*old_child))
                .map(String::as_str)
                .collect();
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            while let Some(below) = stack.pop() {
                if !seen.insert(below) {
                    continue;
                }
                if let Some(&child_index) = index_of.get(below) {
                    dependents[parent_index].push(child_index);
                    pending[child_index] += 1;
                }
                if let Some(standing) = groups.get(below) {
                    stack.extend(standing.group_ids.iter().map(String::as_str));
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
                    group: record.id.clone(),
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
    /// closures of the restored `groups` over the STANDING map (what the
    /// replace would release) unioned with their closures over the
    /// prospective one (what it would address). The import gate judges
    /// `group` records with this — [`Self::group_context_closures`]'
    /// twin for the restore path, where children may be ids the set
    /// itself brings.
    pub fn group_restore_involves(&self, records: &[StoredGroup]) -> BTreeSet<String> {
        let groups = self.0.groups.read();
        let roots: Vec<&str> = records.iter().map(|record| record.id.as_str()).collect();
        let mut involved = groups::context_closure(groups.records(), roots.iter().copied());
        let mut prospective = groups.records().clone();
        for record in records {
            prospective.insert(record.id.clone(), record.clone());
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
    ///
    /// `None` when no `group` has `id`; otherwise the removed `group`'s
    /// name (for the audit line) and whether its file went with it.
    pub fn delete_group(&self, id: &str) -> Option<(String, io::Result<()>)> {
        let mut groups = self.0.groups.write();
        let removed = groups.remove(id)?;
        // Nesting must not outlive the child — the same sweep a
        // deleted context gets, on the child field, under the write
        // lock already held (best effort past the removal; boot
        // reconciliation heals a sweep that could not persist).
        sweep_membership(&self.0.data_dir, &mut groups, id, |record| {
            &mut record.group_ids
        });
        Some((
            removed.name,
            groups::remove_group_file(&self.0.data_dir, id),
        ))
    }

    /// Renames a `group`: its own record's `name` is rewritten, and
    /// nothing else moves (ADR 0045 §2.6) — the file is stemmed by id and
    /// every parent bundles it by id, so there is no file move, no parent
    /// rewrite, and no marker to resume. One staged write, so a crash
    /// leaves either the old name or the new one.
    ///
    /// The destination may be another `group`'s name — names are labels
    /// (ADR 0050). Answers the name the `group` had, for the audit line.
    pub fn rename_group(&self, id: &str, to: &str) -> Result<String, RenameGroupError> {
        if to.is_empty() {
            return Err(RenameGroupError::InvalidName);
        }
        let mut groups = self.0.groups.write();
        let Some(standing) = groups.get(id) else {
            return Err(RenameGroupError::NotFound);
        };
        let from = standing.name.clone();
        // Checked AFTER existence, not before: a self-rename of an id
        // that does not exist is still a `NotFound`, not a silent
        // no-op success.
        if from == to {
            return Ok(from);
        }
        let mut record = standing.clone();
        record.name = to.to_string();
        groups::write_group(&self.0.data_dir, &record).map_err(RenameGroupError::Io)?;
        groups.insert(record);
        Ok(from)
    }

    /// One `group` by id.
    pub fn group(&self, id: &str) -> Option<GroupView> {
        let groups = self.0.groups.read();
        groups.get(id).map(group_view)
    }

    /// Every `group` called `name`, in id order — what a test that
    /// planted a `group` by name finds it by.
    #[cfg(test)]
    pub fn groups_named(&self, name: &str) -> Vec<GroupView> {
        let groups = self.0.groups.read();
        groups
            .ids_named(name)
            .map(|id| group_view(groups.get(id).expect("the name index points at records")))
            .collect()
    }

    /// The one `group` called `name` — `None` when there is none or
    /// several (see [`Self::groups_named`]).
    #[cfg(test)]
    pub fn group_named(&self, name: &str) -> Option<GroupView> {
        let mut named = self.groups_named(name);
        (named.len() == 1).then(|| named.remove(0))
    }

    /// The id of the one `group` called `name`; panics otherwise — the
    /// tests' shorthand for building child-id sets by name.
    #[cfg(test)]
    pub fn group_id_named(&self, name: &str) -> String {
        self.group_named(name)
            .unwrap_or_else(|| panic!("exactly one group named '{name}'"))
            .id
    }

    /// Union of every `context` reachable from the `groups` with these
    /// ids — direct members plus everything nested children bundle,
    /// transitively. The scoped write gate judges a `group` by what it
    /// ADDRESSES, so this is its view; an unknown id contributes nothing
    /// (the write itself refuses it).
    pub fn group_context_closures<'a>(
        &self,
        ids: impl IntoIterator<Item = &'a str>,
    ) -> BTreeSet<String> {
        groups::context_closure(self.0.groups.read().records(), ids)
    }

    /// [`Self::group_context_closures`] for exactly one `group`, by id —
    /// what a directory row's fingerprint covers, so two `groups` that
    /// share a name still fingerprint apart.
    pub fn group_context_closure_of(&self, id: &str) -> BTreeSet<String> {
        groups::context_closure(self.0.groups.read().records(), [id])
    }

    /// [`Self::group_context_closures`] with existence semantics: the
    /// first id no `group` holds comes back as the error instead of
    /// contributing nothing. The
    /// cross-`context` searches resolve their `groups` targets here — a
    /// caller who NAMES a `group` deserves a `no_group` refusal, not a
    /// silently empty search — checked and walked under one lock
    /// acquisition so a concurrent `group` delete cannot slip between
    /// the two.
    pub fn resolve_groups(&self, ids: &[String]) -> Result<BTreeSet<String>, String> {
        let groups = self.0.groups.read();
        if let Some(missing) = first_missing(ids, |id| groups.get(id).is_some()) {
            return Err(missing.clone());
        }
        Ok(groups::context_closure(
            groups.records(),
            ids.iter().map(String::as_str),
        ))
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
    pub fn group_page(
        &self,
        after: Option<&str>,
        after_id: Option<&str>,
        limit: usize,
    ) -> (usize, Vec<GroupView>) {
        let groups = self.0.groups.read();
        let page = groups
            .in_name_order(after, after_id)
            .take(limit)
            .map(group_view)
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

/// `record` as a [`GroupView`]: its id, its name, and its membership.
fn group_view(record: &StoredGroup) -> GroupView {
    GroupView {
        id: record.id.clone(),
        name: record.name.clone(),
        record: GroupRecord {
            description: record.description.clone(),
            context_ids: record.context_ids.clone(),
            group_ids: record.group_ids.clone(),
        },
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

    /// The id a test's `group` name stands for: the standing `group`
    /// called `name` when there is exactly one, else [`fixed_id`] — so a
    /// restore record or a child list can be written in names.
    fn id_for(state: &AppState, name: &str) -> String {
        state
            .group_named(name)
            .map_or_else(|| fixed_id(name), |view| view.id)
    }

    /// [`id_for`] over a list — a child-id set written in names.
    fn ids_for(state: &AppState, names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| id_for(state, name)).collect()
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
                None,
                "drinks",
                "beverage knowledge".into(),
                BTreeSet::from([state.id_of("sake")]),
                BTreeSet::new(),
            )
            .unwrap();
        // A name is a label: a second create under it mints another
        // `group` (ADR 0050), which deletes like any other.
        let twin = state
            .create_group(
                None,
                "drinks",
                String::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            )
            .unwrap();
        assert_eq!(twin.name, "drinks");
        assert_ne!(twin.id, GROUP_A);
        assert_eq!(state.groups_named("drinks").len(), 2);
        state.delete_group(&twin.id).unwrap().1.unwrap();
        // A caller's id: the identical record again answers the
        // standing row (a retried create converges), a different
        // record under it is refused, and nothing moved.
        let pinned = state
            .create_group(
                Some(GROUP_A),
                "pinned",
                "d".into(),
                BTreeSet::from([state.id_of("beer")]),
                BTreeSet::new(),
            )
            .unwrap();
        assert_eq!(pinned.id, GROUP_A);
        let again = state
            .create_group(
                Some(GROUP_A),
                "pinned",
                "d".into(),
                BTreeSet::from([state.id_of("beer")]),
                BTreeSet::new(),
            )
            .unwrap();
        assert_eq!(again.id, GROUP_A);
        assert_eq!(again.record, pinned.record);
        for (name, description) in [("other", "d"), ("pinned", "other")] {
            assert!(matches!(
                state.create_group(
                    Some(GROUP_A),
                    name,
                    description.into(),
                    BTreeSet::from([state.id_of("beer")]),
                    BTreeSet::new(),
                ),
                Err(CreateGroupError::IdTaken)
            ));
        }
        assert!(matches!(
            state.create_group(
                Some(GROUP_A),
                "pinned",
                "d".into(),
                BTreeSet::new(),
                BTreeSet::new(),
            ),
            Err(CreateGroupError::IdTaken)
        ));
        assert_eq!(state.group(GROUP_A).unwrap().record, pinned.record);
        state.delete_group(GROUP_A).unwrap().1.unwrap();
        assert!(matches!(
            state.create_group(None, "", String::new(), BTreeSet::new(), BTreeSet::new()),
            Err(CreateGroupError::InvalidName)
        ));
        assert!(matches!(
            state.create_group(
                None, "ghosts",
                String::new(),
                BTreeSet::from(["nope".to_string()]),
                BTreeSet::new()
            ),
            Err(CreateGroupError::NoSuchContext(missing)) if missing == "nope"
        ));
        // The refused create left nothing behind, in memory or on disk.
        assert!(state.group_named("ghosts").is_none());
        assert!(!group_file_names(&dir).contains("ghosts"));

        // Deltas: removals first, then adds — and an unknown add
        // refuses the whole update, membership untouched.
        assert!(matches!(
            state.update_group(
                &state.group_named("drinks").unwrap().id,
                None,
                BTreeSet::from(["nope".to_string()]),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            ),
            Err(UpdateGroupError::NoSuchContext(missing)) if missing == "nope"
        ));
        assert_eq!(
            state.group_named("drinks").unwrap().record.context_ids,
            BTreeSet::from([state.id_of("sake")])
        );
        let updated = state
            .update_group(
                &state.group_named("drinks").unwrap().id,
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
                GROUP_A,
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
        let survived = state.group_named("drinks").unwrap();
        assert_eq!(survived.record.description, "all drinks");
        assert_eq!(
            survived.record.context_ids,
            BTreeSet::from([state.id_of("beer")])
        );

        // Deletion removes the record and its file; the members live on.
        let drinks = state.group_named("drinks").unwrap().id;
        let (removed, file) = state.delete_group(&drinks).unwrap();
        assert_eq!(removed, "drinks");
        file.unwrap();
        assert!(state.group_named("drinks").is_none());
        assert!(state.group(&drinks).is_none());
        assert!(state.delete_group(&drinks).is_none());
        assert!(!group_file_names(&dir).contains("drinks"));
        assert_eq!(state.group_page(None, None, usize::MAX), (0, Vec::new()));
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
                    None,
                    "drinks",
                    "old".to_string(),
                    BTreeSet::new(),
                    BTreeSet::new(),
                )
                .unwrap();

            fail_persistence_ops_after(failure);
            let first = state.update_group(
                &state.group_named("drinks").unwrap().id,
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
                assert_eq!(
                    state.group_named("drinks").unwrap().record.description,
                    "old"
                );
                state
                    .update_group(
                        &state.group_named("drinks").unwrap().id,
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
            assert_eq!(
                state.group_named("drinks").unwrap().record.description,
                "new"
            );
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
                    None,
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
                state.group_named(group).unwrap().record.context_ids,
                BTreeSet::from([state.id_of("beer")]),
                "'{group}' still names the deleted context"
            );
        }
        // The sweep persisted: a reboot reads the same membership.
        drop(state);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        for group in ["drinks", "fermented"] {
            assert_eq!(
                state.group_named(group).unwrap().record.context_ids,
                BTreeSet::from([state.id_of("beer")])
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    /// The child cap counts the ids a record would carry — children
    /// that share a name are still one id each.
    #[test]
    fn the_child_cap_counts_ids_where_children_share_a_name() {
        let dir = scratch_dir("groups-cap-ids");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let children: BTreeSet<String> = (0..groups::MAX_GROUP_MEMBERS)
            .map(|_| {
                state
                    .create_group(
                        None,
                        "twin",
                        String::new(),
                        BTreeSet::new(),
                        BTreeSet::new(),
                    )
                    .unwrap()
                    .id
            })
            .collect();
        let parent = state
            .create_group(None, "parent", String::new(), BTreeSet::new(), children)
            .unwrap()
            .id;
        let extra = state
            .create_group(
                None,
                "twin",
                String::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            )
            .unwrap()
            .id;
        assert!(matches!(
            state.update_group(
                &parent,
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::from([extra]),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::OverCap("child groups"))
        ));
        assert_eq!(
            state.group(&parent).unwrap().record.group_ids.len(),
            groups::MAX_GROUP_MEMBERS,
            "nothing landed"
        );

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
            state.create_group(None, "g", String::new(), over.clone(), BTreeSet::new()),
            Err(CreateGroupError::OverCap("member contexts"))
        ));
        assert!(matches!(
            state.create_group(None, "g", String::new(), BTreeSet::new(), over.clone()),
            Err(CreateGroupError::OverCap("child groups"))
        ));
        assert!(state.group_named("g").is_none());

        // A delta is judged on its RESULT: 1 member + cap-many adds is
        // one too many, but trading the member out in the same request
        // makes room — the cap passes and the existence gate speaks
        // next, proving the judgement order.
        let one = BTreeSet::from([state.id_of("real")]);
        state
            .create_group(None, "g", String::new(), one.clone(), BTreeSet::new())
            .unwrap();
        let cap_many: BTreeSet<String> = (0..groups::MAX_GROUP_MEMBERS)
            .map(|i| format!("c{i:04}"))
            .collect();
        assert!(matches!(
            state.update_group(
                &state.group_named("g").unwrap().id,
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
                &state.group_named("g").unwrap().id,
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
                &state.group_named("g").unwrap().id,
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                over,
                BTreeSet::new()
            ),
            Err(UpdateGroupError::OverCap("child groups"))
        ));
        assert_eq!(state.group_named("g").unwrap().record.context_ids, one);

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
                None,
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
                None,
                "parent",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from([state.group_id_named("kura")]),
            )
            .unwrap();

        let kura_id = state.group_named("kura").unwrap().id;
        let record = |name: &str, contexts: &[&str], children: &[&str]| StoredGroup {
            id: id_for(&state, name),
            name: name.to_string(),
            description: "new".to_string(),
            context_ids: contexts.iter().map(|c| state.id_of(c)).collect(),
            group_ids: ids_for(&state, children),
        };
        // The set references its own newcomer — parent listed first,
        // child later: order inside the set must not matter.
        let records = vec![
            record("kura", &["sake"], &["kid"]),
            record("kid", &["bunko"], &[]),
        ];
        let outcomes = state
            .restore_groups(&records, Deadline::unbounded())
            .unwrap();
        assert_eq!(outcomes[0].1.as_str(), "replaced");
        assert_eq!(outcomes[1].1.as_str(), "created");
        // The replace is the WHOLE record — bunko dropped, description
        // replaced — and parent still names kura.
        let kura = state.group_named("kura").unwrap();
        assert_eq!(kura.record.description, records[0].description);
        assert_eq!(kura.record.context_ids, records[0].context_ids);
        assert_eq!(kura.record.group_ids, records[0].group_ids);
        assert_eq!(
            state.group_named("parent").unwrap().record.group_ids,
            BTreeSet::from([kura_id.clone()])
        );
        // Disk agrees: the replaced kura kept its id, and its child
        // reference is kid's id — the one its record carries.
        let kura = state.group_named("kura").unwrap();
        assert_eq!(kura.id, kura_id, "a replace keeps the standing id");
        let kid = state.group_named("kid").unwrap();
        assert_eq!(
            kid.id,
            fixed_id("kid"),
            "a new record lands under its own id"
        );
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
        let record = |name: &str, contexts: &[&str], children: &[&str]| StoredGroup {
            id: id_for(&state, name),
            name: name.to_string(),
            description: String::new(),
            context_ids: contexts.iter().map(|c| state.id_of(c)).collect(),
            group_ids: ids_for(&state, children),
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
                    .create_group(
                        None,
                        "probe",
                        String::new(),
                        BTreeSet::new(),
                        BTreeSet::new(),
                    )
                    .is_ok();
                clear_persistence_fault();
                landed
            })
            .expect("one group write fits in 16 persistence steps");
        state
            .delete_group(&state.group_named("probe").unwrap().id)
            .unwrap()
            .1
            .unwrap();

        fail_persistence_ops_after(steps_per_write);
        let error = state
            .restore_groups(
                &[record("a", &[], &[]), record("b", &[], &[])],
                Deadline::unbounded(),
            )
            .unwrap_err();
        clear_persistence_fault();
        assert!(matches!(
            &error,
            RestoreGroupsError::Io { group, .. } if *group == id_for(&state, "b")
        ));
        assert_eq!(error.applied(), 1, "\"a\" already landed when \"b\" failed");
        assert!(
            state.group_named("a").is_some(),
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
        let record = |name: &str, children: &[&str]| StoredGroup {
            group_ids: ids_for(&state, children),
            ..StoredGroup::empty(&id_for(&state, name), name)
        };
        let child_set = |names: &[&str]| -> BTreeSet<String> { ids_for(&state, names) };

        // Two independent depth-2 chains standing, plus "a" -> "b" —
        // everything comfortably within the depth cap before the batch.
        state
            .create_group(
                None,
                "leaf",
                String::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            )
            .unwrap();
        state
            .create_group(
                None,
                "mid",
                String::new(),
                BTreeSet::new(),
                child_set(&["leaf"]),
            )
            .unwrap();
        state
            .create_group(None, "z", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();
        state
            .create_group(None, "y", String::new(), BTreeSet::new(), child_set(&["z"]))
            .unwrap();
        state
            .create_group(None, "b", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();
        state
            .create_group(None, "a", String::new(), BTreeSet::new(), child_set(&["b"]))
            .unwrap();

        // "b" first in the list — the order a plain depth sort would
        // keep on a tie, since it matches the finding's own scenario.
        let batch = [
            record("b", &["mid"]), // b -> mid -> leaf: depth 3
            record("a", &["y"]),   // a -> y -> z: depth 3, drops "b"
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
            matches!(&error, RestoreGroupsError::Io { group, .. } if *group == id_for(&state, "a")),
            "the fix orders \"a\" first, so it is \"a\"'s write that hits the fault: {error:?}"
        );
        assert_eq!(error.applied(), 0, "\"a\" failed before anything landed");

        // Nothing changed: "b" must never have been attempted, so the
        // pre-batch (valid, depth <= 3 everywhere) standing state
        // stands exactly as it was.
        assert_eq!(
            state.group_named("a").unwrap().record.group_ids,
            child_set(&["b"])
        );
        assert_eq!(
            state.group_named("b").unwrap().record.group_ids,
            BTreeSet::new()
        );
        let standing = state.0.groups.read().records().clone();
        groups::nesting_depths(&standing)
            .expect("the untouched standing map must still be within the depth cap");

        let _ = fs::remove_dir_all(dir);
    }

    /// The same stranding one level further down: the record that
    /// deepens is not the dropped child itself but a descendant of it,
    /// reached through a group the batch does not carry. Standing
    /// a > b > c and y > z; the batch moves "a" onto "y" (dropping "b")
    /// and hangs "z" under "c". The final state is depth 3 everywhere,
    /// but "c" (prospective depth 2) sorts before "a" (3), and landing
    /// it first makes the old a > b > c > z four deep. "a" must land
    /// first, so failing its write leaves the standing state untouched.
    #[test]
    fn restore_groups_orders_a_parent_before_a_deeper_descendant_of_its_dropped_branch() {
        let dir = scratch_dir("groups-restore-depth-race-transitive");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = |name: &str, children: &[&str]| StoredGroup {
            group_ids: ids_for(&state, children),
            ..StoredGroup::empty(&id_for(&state, name), name)
        };
        let child_set = |names: &[&str]| -> BTreeSet<String> { ids_for(&state, names) };
        for (name, children) in [
            ("c", &[][..]),
            ("b", &["c"][..]),
            ("a", &["b"][..]),
            ("z", &[][..]),
            ("y", &["z"][..]),
        ] {
            state
                .create_group(
                    None,
                    name,
                    String::new(),
                    BTreeSet::new(),
                    child_set(children),
                )
                .unwrap();
        }

        let batch = [record("c", &["z"]), record("a", &["y"])];
        fail_persistence_ops_after(0);
        let error = state
            .restore_groups(&batch, Deadline::unbounded())
            .unwrap_err();
        assert!(
            !clear_persistence_fault(),
            "the fault must have fired, not run out of persistence steps first"
        );
        assert!(
            matches!(&error, RestoreGroupsError::Io { group, .. } if *group == id_for(&state, "a")),
            "\"a\" lands before \"c\", so its write hits the fault: {error:?}"
        );
        assert_eq!(error.applied(), 0);
        assert_eq!(
            state.group_named("c").unwrap().record.group_ids,
            BTreeSet::new()
        );
        let standing = state.0.groups.read().records().clone();
        groups::nesting_depths(&standing)
            .expect("the untouched standing map must still be within the depth cap");

        // Without the fault the whole batch lands, within the cap.
        state.restore_groups(&batch, Deadline::unbounded()).unwrap();
        let landed = state.0.groups.read().records().clone();
        groups::nesting_depths(&landed).unwrap();

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_counts_an_unchanged_record_as_already_landed() {
        let dir = scratch_dir("groups-restore-unchanged-then-io-failure");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = |name: &str, contexts: &[&str], children: &[&str]| StoredGroup {
            id: id_for(&state, name),
            name: name.to_string(),
            description: String::new(),
            context_ids: contexts.iter().map(|c| state.id_of(c)).collect(),
            group_ids: ids_for(&state, children),
        };

        // Land "a" first so the call below finds it standing exactly as
        // given and takes the `Unchanged` branch, which counts toward
        // `applied` without writing anything.
        state
            .restore_groups(&[record("a", &[], &[])], Deadline::unbounded())
            .unwrap();

        // "b" is new, so it takes the `write_group` branch; "a" writes
        // nothing, so failing the very first persistence step isolates an
        // Io failure strictly after "a" is judged unchanged.
        fail_persistence_ops_after(0);
        let error = state
            .restore_groups(
                &[record("a", &[], &[]), record("b", &[], &[])],
                Deadline::unbounded(),
            )
            .unwrap_err();
        assert!(
            !clear_persistence_fault(),
            "the injected failure must be what refused \"b\""
        );
        assert!(matches!(
            &error,
            RestoreGroupsError::Io { group, .. } if *group == id_for(&state, "b")
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
        let record = StoredGroup::empty(&fixed_id("a"), "a");

        // A budget already spent must bound the fsync-per-record loop
        // before its first write, the way the batch loop bounds itself —
        // a stream of many tiny group records must not pin the write
        // lock past the deadline every other group op honors.
        let deadline = Deadline::after(Duration::ZERO);
        std::thread::sleep(Duration::from_millis(1));
        let error = state.restore_groups(&[record], deadline).unwrap_err();
        assert!(matches!(&error, RestoreGroupsError::Timeout { applied: 0 }));
        assert_eq!(error.applied(), 0);

        // Nothing landed — not in memory, not on disk.
        assert!(
            state.group_named("a").is_none(),
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
        let record = |name: &str, contexts: &[&str], children: &[&str]| StoredGroup {
            id: id_for(&state, name),
            name: name.to_string(),
            description: String::new(),
            context_ids: contexts.iter().map(|c| state.id_of(c)).collect(),
            group_ids: ids_for(&state, children),
        };

        // A dangling member refuses the set — the valid record beside
        // it included.
        let refusal = state
            .restore_groups(
                &[
                    record("fine", &["sake"], &[]),
                    record("broken", &["ghost"], &[]),
                ],
                Deadline::unbounded(),
            )
            .unwrap_err();
        assert!(matches!(
            &refusal,
            RestoreGroupsError::NoSuchContext { group, context }
                if *group == fixed_id("broken") && *context == state.id_of("ghost")
        ));
        assert_eq!(refusal.applied(), 0);
        assert!(
            state.group_named("fine").is_none(),
            "nothing applies on a refusal"
        );

        // A child neither standing nor in the set.
        assert!(matches!(
            state
                .restore_groups(&[record("a", &[], &["nope"])], Deadline::unbounded())
                .unwrap_err(),
            RestoreGroupsError::NoSuchChild { group, child }
                if group == fixed_id("a") && child == fixed_id("nope")
        ));

        // A cycle the set closes with itself.
        assert!(matches!(
            state
                .restore_groups(
                    &[record("a", &[], &["b"]), record("b", &[], &["a"]),],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::Nesting(NestingViolation::Cycle(_))
        ));

        // Depth counts the standing groups too: records stacking two
        // more storeys under an existing 2-chain overflow the cap.
        state
            .create_group(None, "mid", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();
        state
            .create_group(
                None,
                "top",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from([state.group_id_named("mid")]),
            )
            .unwrap();
        assert!(matches!(
            state
                .restore_groups(
                    &[
                        record("mid", &[], &["deep"]),
                        record("deep", &[], &["deeper"]),
                        record("deeper", &[], &[]),
                    ],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::Nesting(NestingViolation::TooDeep(_))
        ));
        assert!(state.group_named("deep").is_none());

        // One name twice is two truths for one group.
        assert!(matches!(
            state
                .restore_groups(
                    &[
                        record("dup", &[], &[]),
                        record("dup", &["sake"], &[]),
                    ],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::Duplicate(id) if *id == fixed_id("dup")
        ));

        // The cap judges each record's sets.
        let over: BTreeSet<String> = (0..=groups::MAX_GROUP_MEMBERS)
            .map(|i| format!("c{i:04}"))
            .collect();
        assert!(matches!(
            state
                .restore_groups(
                    &[StoredGroup {
                        context_ids: over,
                        ..StoredGroup::empty(&fixed_id("wide"), "wide")
                    }],
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
        let one = |name: &str| BTreeSet::from([id_for(&state, name)]);
        let one_ctx = |name: &str| BTreeSet::from([state.id_of(name)]);
        state
            .create_group(None, "leaf", String::new(), one_ctx("tea"), BTreeSet::new())
            .unwrap();
        state
            .create_group(None, "mid", String::new(), one_ctx("beer"), one("leaf"))
            .unwrap();
        state
            .create_group(None, "top", String::new(), one_ctx("sake"), one("mid"))
            .unwrap();

        // The closure reads through the nesting; the record stays flat.
        assert_eq!(
            state.group_context_closures([state.group_id_named("top").as_str()]),
            ["sake", "beer", "tea"]
                .iter()
                .map(|c| state.id_of(c))
                .collect()
        );
        assert_eq!(
            state.group_named("top").unwrap().record.context_ids,
            one_ctx("sake")
        );

        // A fourth storey over a full chain refuses, and nothing lands.
        assert!(matches!(
            state.create_group(None, "over", String::new(), BTreeSet::new(), one("top")),
            Err(CreateGroupError::Nesting(NestingViolation::TooDeep(_)))
        ));
        assert!(state.group_named("over").is_none());
        assert!(!group_file_names(&dir).contains("over"));

        // Closing the chain into a cycle refuses — the self-loop
        // included — and an unknown child answers in the group
        // namespace; the record survives every refusal unchanged.
        assert!(matches!(
            state.update_group(
                &state.group_named("leaf").unwrap().id,
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
                &state.group_named("leaf").unwrap().id,
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
                &state.group_named("leaf").unwrap().id,
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                one("nope"),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::NoSuchGroup(missing)) if missing == fixed_id("nope")
        ));
        assert_eq!(
            state.group_named("leaf").unwrap().record.group_ids,
            BTreeSet::new()
        );

        // Deleting a child sweeps it out of every parent, durably.
        state
            .delete_group(&state.group_named("mid").unwrap().id)
            .unwrap()
            .1
            .unwrap();
        assert_eq!(
            state.group_named("top").unwrap().record.group_ids,
            BTreeSet::new()
        );
        drop(state);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        assert_eq!(
            state.group_named("top").unwrap().record.group_ids,
            BTreeSet::new()
        );
        assert_eq!(
            state.group_named("leaf").unwrap().record.context_ids,
            BTreeSet::from([state.id_of("tea")])
        );

        let _ = fs::remove_dir_all(dir);
    }

    fn assert_live_group_invariants(state: &AppState) {
        let contexts: BTreeSet<String> = state
            .directory()
            .into_iter()
            .map(|entry| entry.id)
            .collect();
        let table = state.0.groups.read();
        let groups = table.records();
        assert_eq!(groups::validate_nesting(groups), Ok(()));
        for (id, record) in groups {
            assert_eq!(&record.id, id, "every record sits under its own id");
            // Names are labels (ADR 0050): the index answers every id
            // that carries one.
            let named: BTreeSet<&String> = groups
                .values()
                .filter(|other| other.name == record.name)
                .map(|other| &other.id)
                .collect();
            assert_eq!(
                table.ids_named(&record.name).collect::<BTreeSet<_>>(),
                named,
                "the name index tracks the records"
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

            // The pool speaks names; the API takes ids. A group name
            // resolves to the first id carrying it (names may be shared)
            // and a context name to its id — an unresolved name passes
            // through as an id nothing answers to, the NotFound arm.
            let group_id = |name: &str| -> String {
                state
                    .0
                    .groups
                    .read()
                    .ids_named(name)
                    .next()
                    .cloned()
                    .unwrap_or_else(|| fixed_id(name))
            };
            let group_ids = |names: Vec<&str>| -> BTreeSet<String> {
                names.into_iter().map(group_id).collect()
            };
            let context_ids = |names: Vec<&str>| -> BTreeSet<String> {
                names.into_iter().map(|name| state.id_of(name)).collect()
            };
            for operation in operations {
                match operation {
                    GeneratedGroupOp::CreateContext(name) => {
                        let _ = state.create(name, ContextMeta::default());
                    }
                    GeneratedGroupOp::DeleteContext(name) => {
                        let _ = state.delete(&state.id_of(name));
                    }
                    GeneratedGroupOp::CreateGroup { name, contexts, groups } => {
                        if let Ok(view) = state.create_group(
                            None, name,
                            String::new(),
                            context_ids(contexts),
                            group_ids(groups),
                        ) {
                            prop_assert_eq!(view.name, name);
                        }
                    }
                    GeneratedGroupOp::UpdateGroup {
                        name,
                        add_contexts,
                        remove_contexts,
                        add_groups,
                        remove_groups,
                    } => {
                        let id = group_id(name);
                        let added = context_ids(add_contexts);
                        let removed = context_ids(remove_contexts);
                        if let Ok(view) = state.update_group(
                            &id,
                            None,
                            added.clone(),
                            removed.clone(),
                            group_ids(add_groups),
                            group_ids(remove_groups),
                        ) {
                            // Removals apply first, so every added
                            // member stands afterwards.
                            prop_assert!(added.is_subset(&view.record.context_ids));
                            prop_assert_eq!(view.id, id);
                        }
                    }
                    GeneratedGroupOp::DeleteGroup(name) => {
                        let id = group_id(name);
                        if state.delete_group(&id).is_some() {
                            prop_assert!(state.group(&id).is_none());
                        }
                    }
                    GeneratedGroupOp::RestoreGroups(records) => {
                        let records: Vec<StoredGroup> = records
                            .into_iter()
                            .map(|(name, contexts, groups)| StoredGroup {
                                id: group_id(name),
                                name: name.to_string(),
                                description: String::new(),
                                context_ids: context_ids(contexts),
                                group_ids: group_ids(groups),
                            })
                            .collect();
                        let _ = state.restore_groups(&records, Deadline::unbounded());
                    }
                    GeneratedGroupOp::RenameGroup { from, to } => {
                        let id = group_id(from);
                        if state.rename_group(&id, to).is_ok() {
                            prop_assert_eq!(state.group(&id).unwrap().name, to);
                        }
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
                None,
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
            &state.group_named("drinks").unwrap().id,
            Some("after".into()),
            BTreeSet::new(),
            BTreeSet::from([state.id_of("sake")]),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(matches!(outcome, Err(UpdateGroupError::Io(_))));
        let record = state.group_named("drinks").unwrap();
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
                None,
                "liquor",
                "d".into(),
                BTreeSet::from([state.id_of("sake")]),
                BTreeSet::new(),
            )
            .unwrap();
        state
            .create_group(
                None,
                "drinks",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from([state.group_id_named("liquor")]),
            )
            .unwrap();

        let liquor_id = state.group_named("liquor").unwrap().id;
        let drinks = state.group_named("drinks").unwrap();
        let drinks_bytes = fs::read(groups::group_path(&dir, &drinks.id)).unwrap();
        state
            .rename_group(&state.group_named("liquor").unwrap().id, "spirits")
            .unwrap();

        assert!(state.group_named("liquor").is_none());
        let spirits = state.group_named("spirits").unwrap();
        assert_eq!(spirits.record.description, "d");
        assert_eq!(
            spirits.record.context_ids,
            BTreeSet::from([state.id_of("sake")])
        );
        assert_eq!(
            state.group_named("drinks").unwrap().record.group_ids,
            BTreeSet::from([liquor_id.clone()]),
            "the parent's child reference is the id, which the rename leaves alone"
        );
        assert!(!group_file_names(&dir).contains("liquor"));
        assert!(group_file_names(&dir).contains("spirits"));
        assert_eq!(
            state.group_named("spirits").unwrap().id,
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
        assert!(state.group_named("liquor").is_none());
        assert_eq!(
            state.group_named("drinks").unwrap().record.group_ids,
            BTreeSet::from([liquor_id])
        );
        assert_eq!(
            state.group_named("spirits").unwrap().record.description,
            "d"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_group_error_cases() {
        let dir = scratch_dir("rename-group-errors");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create_group(
                None,
                "drinks",
                String::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            )
            .unwrap();
        state
            .create_group(
                None,
                "food",
                String::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            )
            .unwrap();

        let drinks = state.group_named("drinks").unwrap().id;
        assert!(matches!(
            state.rename_group(GROUP_A, "whatever"),
            Err(RenameGroupError::NotFound)
        ));
        assert!(matches!(
            state.rename_group(&drinks, ""),
            Err(RenameGroupError::InvalidName)
        ));
        assert_eq!(
            state.rename_group(&drinks, "drinks").unwrap(),
            "drinks",
            "a self-rename answers the name it kept"
        );
        assert!(state.group_named("drinks").is_some());
        // The `from == to` short-circuit must not mask a NotFound: a
        // self-rename of an id nothing holds is still a refusal.
        assert!(matches!(
            state.rename_group(GROUP_A, "missing"),
            Err(RenameGroupError::NotFound)
        ));
        // A name is a label (ADR 0050): renaming onto another `group`'s
        // name is allowed, and both stand under it, apart by id.
        assert_eq!(state.rename_group(&drinks, "food").unwrap(), "drinks");
        assert_eq!(state.groups_named("food").len(), 2);
        assert_eq!(state.group(&drinks).unwrap().name, "food");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn every_group_delete_persistence_failure_reconciles_at_boot() {
        let mut exhausted = false;
        for failure in 0..8 {
            let dir = scratch_dir(&format!("group-delete-fault-{failure}"));
            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            state
                .create_group(
                    None,
                    "leaf",
                    String::new(),
                    BTreeSet::new(),
                    BTreeSet::new(),
                )
                .unwrap();
            state
                .create_group(
                    None,
                    "parent",
                    String::new(),
                    BTreeSet::new(),
                    BTreeSet::from([state.group_id_named("leaf")]),
                )
                .unwrap();
            let leaf = state.group_id_named("leaf");

            fail_persistence_ops_after(failure);
            let _ = state
                .delete_group(&state.group_named("leaf").unwrap().id)
                .unwrap();
            let past_end = clear_persistence_fault();
            drop(state);

            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            assert!(
                !state
                    .group_named("parent")
                    .unwrap()
                    .record
                    .group_ids
                    .contains(&leaf),
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
            state.group_named("drinks").unwrap().record.context_ids,
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
        let children = |name: &str| -> BTreeSet<String> {
            state.group_named(name).unwrap().record.group_ids.clone()
        };
        assert_eq!(children("n1"), BTreeSet::from([id("n2")]));
        assert_eq!(children("n2"), BTreeSet::from([id("n3")]));
        assert_eq!(children("n3"), BTreeSet::new());
        assert_eq!(children("cyc-a"), BTreeSet::from([id("cyc-b")]));
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
    /// bundling both, and boots — names are labels (ADR 0050), so this is
    /// an ordinary directory.
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

    /// Every entrance names a `group` by id (#996), so two `groups` that
    /// share a name are simply two `groups`: each is added, removed,
    /// searched and restored on its own, and nothing asks which one a
    /// name means.
    #[test]
    fn groups_that_share_a_name_stay_apart_at_every_entrance() {
        let (dir, state) = boot_with_twin_names("groups-twin-names");
        let (k1, k2, p) = (fixed_id("k1"), fixed_id("k2"), fixed_id("p"));
        let none = BTreeSet::<String>::new;
        let only = |id: &str| BTreeSet::from([id.to_string()]);
        assert_eq!(state.groups_named("kura").len(), 2);
        assert_eq!(
            state.group(&p).unwrap().record.group_ids,
            BTreeSet::from([k1.clone(), k2.clone()]),
            "the parent's row lists both children, apart"
        );
        // Removing one twin removes exactly that one.
        let after = state
            .update_group(&p, None, none(), none(), none(), only(&k1))
            .unwrap();
        assert_eq!(after.record.group_ids, only(&k2));
        let after = state
            .update_group(&p, None, none(), none(), only(&k1), none())
            .unwrap();
        assert_eq!(
            after.record.group_ids,
            BTreeSet::from([k1.clone(), k2.clone()])
        );
        let q = state
            .create_group(None, "q", String::new(), none(), only(&k2))
            .unwrap();
        assert_eq!(q.record.group_ids, only(&k2));
        assert_eq!(
            state.resolve_groups(std::slice::from_ref(&k1)),
            Ok(BTreeSet::new())
        );
        // A restore under one twin's id replaces that twin alone.
        let outcomes = state
            .restore_groups(
                &[StoredGroup {
                    description: "restored".to_string(),
                    ..StoredGroup::empty(&k1, "kura")
                }],
                Deadline::unbounded(),
            )
            .unwrap();
        assert_eq!(outcomes[0].1, GroupRestoreOutcome::Replaced);
        assert_eq!(state.group(&k1).unwrap().record.description, "restored");
        assert_eq!(state.group(&k2).unwrap().record.description, "");
        // The directory lists both, name first then id.
        let page = state.group_page(None, None, usize::MAX).1;
        assert_eq!(
            page.iter().map(|view| view.id.clone()).collect::<Vec<_>>(),
            [k1, k2, p, q.id]
        );
        let _ = fs::remove_dir_all(dir);
    }

    /// The #995 review case (#996): a parent bundles children `a` and
    /// `b`, and `b` is renamed to `a`. The parent's row still lists two
    /// children, and removing one by id removes exactly that one.
    #[test]
    fn a_rename_onto_a_sibling_name_keeps_both_children_apart() {
        let dir = scratch_dir("groups-sibling-rename");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let none = BTreeSet::<String>::new;
        let a = state
            .create_group(None, "a", String::new(), none(), none())
            .unwrap()
            .id;
        let b = state
            .create_group(None, "b", String::new(), none(), none())
            .unwrap()
            .id;
        let parent = state
            .create_group(
                None,
                "parent",
                String::new(),
                none(),
                BTreeSet::from([a.clone(), b.clone()]),
            )
            .unwrap()
            .id;
        assert_eq!(state.rename_group(&b, "a").unwrap(), "b");
        assert_eq!(
            state.group(&parent).unwrap().record.group_ids,
            BTreeSet::from([a.clone(), b.clone()]),
            "two children named alike are still two ids"
        );
        let after = state
            .update_group(
                &parent,
                None,
                none(),
                none(),
                none(),
                BTreeSet::from([b.clone()]),
            )
            .unwrap();
        assert_eq!(after.record.group_ids, BTreeSet::from([a]));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_scoped_gate_walks_each_group_by_its_own_id() {
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
        let (k1, k2) = (fixed_id("k1"), fixed_id("k2"));
        assert_eq!(
            state.group_context_closures([k1.as_str()]),
            BTreeSet::from([sake.clone()]),
            "one id, one group, whatever its name"
        );
        assert_eq!(
            state.group_context_closures([k1.as_str(), k2.as_str()]),
            BTreeSet::from([sake.clone(), beer.clone()])
        );
        assert_eq!(state.group_context_closures(["kura"]), BTreeSet::new());
        assert_eq!(
            state.group_restore_involves(&[StoredGroup::empty(&k1, "kura")]),
            BTreeSet::from([sake.clone()]),
            "the standing record counts toward what a restore releases"
        );
        // The restore gate walks through nesting: a standing parent's
        // child bundles sake, so replacing the parent with an empty
        // record still RELEASES sake.
        let kid = state
            .create_group(
                None,
                "kid",
                String::new(),
                BTreeSet::from([sake.clone()]),
                BTreeSet::new(),
            )
            .unwrap()
            .id;
        let parent = state
            .create_group(
                None,
                "parent",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from([kid]),
            )
            .unwrap()
            .id;
        assert_eq!(
            state.group_restore_involves(&[StoredGroup::empty(&parent, "parent")]),
            BTreeSet::from([sake.clone()]),
            "a standing child's members count toward what the replace releases"
        );
        // And a record bringing a child of its own counts what that
        // child addresses.
        assert_eq!(
            state.group_restore_involves(&[StoredGroup {
                group_ids: BTreeSet::from([k2.clone()]),
                ..StoredGroup::empty(&fixed_id("new"), "new")
            }]),
            BTreeSet::from([beer.clone()])
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn removing_a_child_the_parent_does_not_bundle_is_a_no_op() {
        let dir = scratch_dir("groups-remove-by-id");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let none = BTreeSet::<String>::new;
        let a = state
            .create_group(None, "a", String::new(), none(), none())
            .unwrap()
            .id;
        let b = state
            .create_group(None, "b", String::new(), none(), none())
            .unwrap()
            .id;
        let p = state
            .create_group(
                None,
                "p",
                String::new(),
                none(),
                BTreeSet::from([a.clone()]),
            )
            .unwrap()
            .id;
        // An id p does not bundle (b), or no `group` holds at all, is
        // the idempotent no-op; one it does (a) leaves.
        let after = state
            .update_group(
                &p,
                None,
                none(),
                none(),
                none(),
                BTreeSet::from([b.clone(), fixed_id("ghost")]),
            )
            .unwrap();
        assert_eq!(after.record.group_ids, BTreeSet::from([a.clone()]));
        let after = state
            .update_group(
                &p,
                None,
                none(),
                none(),
                BTreeSet::from([b.clone()]),
                BTreeSet::from([a]),
            )
            .unwrap();
        assert_eq!(after.record.group_ids, BTreeSet::from([b]));
        assert_eq!(after.name, "p");
        assert_eq!(after.id, p);
        // An id no `group` holds is refused on the way IN.
        assert!(matches!(
            state.update_group(&p, None, none(), none(), BTreeSet::from([fixed_id("ghost")]), none()),
            Err(UpdateGroupError::NoSuchGroup(missing)) if missing == fixed_id("ghost")
        ));
        assert!(matches!(
            state.create_group(None, "q", String::new(), none(), BTreeSet::from([fixed_id("ghost")])),
            Err(CreateGroupError::NoSuchGroup(missing)) if missing == fixed_id("ghost")
        ));
        assert_eq!(
            state.resolve_groups(&[p.clone(), fixed_id("ghost")]),
            Err(fixed_id("ghost"))
        );
        let _ = fs::remove_dir_all(dir);
    }

    /// A create or update names the `group` its nesting refusal blames
    /// by name (the request addresses one `group`, so the name reads
    /// best); a restore names it by id, the column its records key on.
    #[test]
    fn a_nesting_refusal_names_the_blamed_group() {
        let dir = scratch_dir("groups-nesting-names");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let none = BTreeSet::<String>::new;
        let leaf = state
            .create_group(None, "leaf", String::new(), none(), none())
            .unwrap()
            .id;
        let mid = state
            .create_group(
                None,
                "mid",
                String::new(),
                none(),
                BTreeSet::from([leaf.clone()]),
            )
            .unwrap()
            .id;
        let top = state
            .create_group(None, "top", String::new(), none(), BTreeSet::from([mid]))
            .unwrap()
            .id;
        assert!(matches!(
            state.create_group(None, "over", String::new(), none(), BTreeSet::from([top.clone()])),
            Err(CreateGroupError::Nesting(groups::NestingViolation::TooDeep(blamed))) if blamed == "over"
        ));
        assert!(matches!(
            state.update_group(&leaf, None, none(), none(), BTreeSet::from([top.clone()]), none()),
            Err(UpdateGroupError::Nesting(groups::NestingViolation::Cycle(blamed)))
                if ["leaf", "mid", "top"].contains(&blamed.as_str())
        ));
        let fresh = fixed_id("fresh");
        assert!(matches!(
            state
                .restore_groups(
                    &[StoredGroup {
                        group_ids: BTreeSet::from([top]),
                        ..StoredGroup::empty(&fresh, "fresh")
                    }],
                    Deadline::unbounded()
                )
                .unwrap_err(),
            RestoreGroupsError::Nesting(groups::NestingViolation::TooDeep(blamed)) if blamed == fresh
        ));
        let _ = fs::remove_dir_all(dir);
    }

    /// A rename leaves the old name to anyone: a fresh create under it
    /// mints a different id, and the renamed `group` keeps its own.
    #[test]
    fn a_rename_frees_the_old_name_for_a_fresh_group_with_its_own_id() {
        let dir = scratch_dir("groups-rename-free-name");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let none = BTreeSet::<String>::new;
        state
            .create_group(None, "old", String::new(), none(), none())
            .unwrap();
        let old_id = state.group_named("old").unwrap().id;
        state
            .rename_group(&state.group_named("old").unwrap().id, "new")
            .unwrap();
        state
            .create_group(None, "old", String::new(), none(), none())
            .unwrap();
        assert_ne!(state.group_named("old").unwrap().id, old_id);
        assert_eq!(state.group_named("new").unwrap().id, old_id);
        assert!(groups::is_group_id(&old_id), "{old_id}");
        let _ = fs::remove_dir_all(dir);
    }
}
