use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

use taguru::deadline::Deadline;

use crate::groups::{self, GroupRecord};
use crate::storage::{commit_staged, remove_persisted_file};

use super::{
    AppState, CreateGroupError, GroupRestoreOutcome, RenameGroupError, RestoreGroupsError,
    UpdateGroupError, file_stem, rename_in_membership, rename_markers_targeting,
    write_rename_marker,
};

impl AppState {
    /// Registers a group and persists it immediately — the create twin
    /// for groups, without the `pending.creates` choreography: the one
    /// fsync happens under the groups lock, which blocks only other
    /// group writes (see the field's doc), so nothing here needs the
    /// reservation dance.
    ///
    /// Member validation happens under both locks (`groups` before
    /// `registry` — the documented order): `contains_key` already
    /// answers false for a name mid-delete, because delete() removes
    /// the name and reserves it in `pending.deletes` inside one
    /// critical section. Child groups are judged against the same map
    /// the write lock already holds.
    pub fn create_group(
        &self,
        name: &str,
        description: String,
        contexts: BTreeSet<String>,
        children: BTreeSet<String>,
    ) -> Result<(), CreateGroupError> {
        if name.is_empty() {
            return Err(CreateGroupError::InvalidName);
        }
        let mut groups = self.0.groups.write();
        if groups.contains_key(name) {
            return Err(CreateGroupError::AlreadyExists);
        }
        // Unreachable through HTTP while the request-list cap equals
        // [`groups::MAX_GROUP_MEMBERS`] (a create can't say more names
        // than a group may hold), but the invariant belongs to the
        // registry, not to whoever happens to call it.
        check_member_caps(&contexts, &children).map_err(CreateGroupError::OverCap)?;
        {
            let registry = self.0.registry.read();
            if let Some(missing) =
                first_missing(&contexts, |context| registry.contains_key(context))
            {
                return Err(CreateGroupError::NoSuchContext(missing.clone()));
            }
        }
        if let Some(missing) = first_missing(&children, |child| groups.contains_key(child)) {
            return Err(CreateGroupError::NoSuchGroup(missing.clone()));
        }
        // A group rename that half-finished under THIS name — as the
        // source (its `.grouprenaming` marker sits at this stem) or the
        // destination (some other stem's marker names it) — would
        // otherwise have boot's resume-sweep move a stale group file over
        // the one we are about to write. A create is a clean start:
        // abandon any such marker. Groups clear their marker on a
        // graceful move failure, so this only bites after a crash or a
        // best-effort boot cleanup that could not remove it — cheap
        // insurance keeping the group path symmetric with create_files.
        let mut stale_markers = rename_markers_targeting(&self.0.data_dir, name, "grouprenaming");
        stale_markers.push(groups::group_renaming_marker_path(
            &self.0.data_dir,
            &file_stem(name),
        ));
        for marker in stale_markers {
            if let Err(error) = remove_persisted_file(&marker)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(CreateGroupError::Io(error));
            }
        }
        let record = GroupRecord {
            description,
            contexts,
            groups: children,
        };
        // The nesting validator wants the prospective map, so insert
        // first and unwind on refusal — nothing escapes the write lock
        // half-done.
        groups.insert(name.to_string(), record);
        if let Err(violation) = groups::validate_nesting(&groups) {
            groups.remove(name);
            return Err(CreateGroupError::Nesting(violation));
        }
        if let Err(error) = groups::write_group(&self.0.data_dir, &file_stem(name), &groups[name]) {
            groups.remove(name);
            return Err(CreateGroupError::Io(error));
        }
        Ok(())
    }

    /// Applies a delta to one group — context and child removals
    /// first, then additions (a name in both ends up a member), then
    /// the description — validates the nesting that results, and
    /// persists. Nothing applies unless everything does: a refused
    /// nesting or a failed persist restores the previous record, the
    /// update twin of [`rollback_meta`].
    pub fn update_group(
        &self,
        name: &str,
        description: Option<String>,
        add_contexts: BTreeSet<String>,
        remove_contexts: BTreeSet<String>,
        add_groups: BTreeSet<String>,
        remove_groups: BTreeSet<String>,
    ) -> Result<GroupRecord, UpdateGroupError> {
        let mut groups = self.0.groups.write();
        if !groups.contains_key(name) {
            return Err(UpdateGroupError::NotFound);
        }
        // The cap is judged on the delta's RESULT — removals first,
        // exactly as they will apply — and before the existence
        // lookups: counting needs none, and refusing here lets the
        // membership tests speak in names that need not exist.
        {
            let record = &groups[name];
            for (field, current, add, remove) in [
                (
                    "member contexts",
                    &record.contexts,
                    &add_contexts,
                    &remove_contexts,
                ),
                ("child groups", &record.groups, &add_groups, &remove_groups),
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
        if !add_contexts.is_empty() {
            let registry = self.0.registry.read();
            if let Some(missing) =
                first_missing(&add_contexts, |context| registry.contains_key(context))
            {
                return Err(UpdateGroupError::NoSuchContext(missing.clone()));
            }
        }
        // `name` itself IS registered, so a self-add passes this gate —
        // and lands in the validator's lap as the smallest cycle.
        if let Some(missing) = first_missing(&add_groups, |child| groups.contains_key(child)) {
            return Err(UpdateGroupError::NoSuchGroup(missing.clone()));
        }
        let record = groups.get_mut(name).unwrap();
        let previous = record.clone();
        for context in &remove_contexts {
            record.contexts.remove(context);
        }
        record.contexts.extend(add_contexts);
        for group in &remove_groups {
            record.groups.remove(group);
        }
        record.groups.extend(add_groups);
        if let Some(description) = description {
            record.description = description;
        }
        if let Err(violation) = groups::validate_nesting(&groups) {
            *groups.get_mut(name).unwrap() = previous;
            return Err(UpdateGroupError::Nesting(violation));
        }
        if let Err(error) = groups::write_group(&self.0.data_dir, &file_stem(name), &groups[name]) {
            *groups.get_mut(name).unwrap() = previous;
            return Err(UpdateGroupError::Io(error));
        }
        Ok(groups[name].clone())
    }

    /// Restores a set of group records — import's create-or-replace
    /// twin of [`Self::create_group`]/[`Self::update_group`]: each
    /// record replaces its whole row (description and both member
    /// sets), and a group absent from the set is untouched, parents
    /// naming a restored group included. The SET is judged before the
    /// first write, under the groups write lock: every member context
    /// registered, every child registered or in the set itself, both
    /// caps per record, and the nesting that results — the standing
    /// map overlaid with every record — acyclic and within
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
        // The prospective map — what memory becomes if every write
        // lands — is what the validators judge.
        let mut prospective = groups.clone();
        let mut incoming: BTreeSet<&str> = BTreeSet::new();
        for (name, record) in records {
            if name.is_empty() {
                return Err(RestoreGroupsError::InvalidName);
            }
            if !incoming.insert(name) {
                return Err(RestoreGroupsError::Duplicate(name.clone()));
            }
            check_member_caps(&record.contexts, &record.groups).map_err(|field| {
                RestoreGroupsError::OverCap {
                    group: name.clone(),
                    field,
                }
            })?;
            prospective.insert(name.clone(), record.clone());
        }
        {
            // Lock order: `groups` before `registry`, as documented on
            // the field.
            let registry = self.0.registry.read();
            for (name, record) in records {
                if let Some(missing) =
                    first_missing(&record.contexts, |context| registry.contains_key(context))
                {
                    return Err(RestoreGroupsError::NoSuchContext {
                        group: name.clone(),
                        context: missing.clone(),
                    });
                }
                if let Some(missing) =
                    first_missing(&record.groups, |child| prospective.contains_key(child))
                {
                    return Err(RestoreGroupsError::NoSuchChild {
                        group: name.clone(),
                        child: missing.clone(),
                    });
                }
            }
        }
        let depths = groups::nesting_depths(&prospective).map_err(RestoreGroupsError::Nesting)?;
        let outcomes: Vec<(String, GroupRestoreOutcome)> = records
            .iter()
            .map(|(name, record)| {
                let outcome = match groups.get(name) {
                    Some(standing) if standing == record => GroupRestoreOutcome::Unchanged,
                    Some(_) => GroupRestoreOutcome::Replaced,
                    None => GroupRestoreOutcome::Created,
                };
                (name.clone(), outcome)
            })
            .collect();
        let mut order: Vec<usize> = (0..records.len()).collect();
        // Every record's name sits in `prospective`, so the settled
        // map has its depth — indexing panics loudly if that invariant
        // ever breaks, where a fallback would silently misorder.
        order.sort_by_key(|&index| depths[records[index].0.as_str()]);
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
            let (name, record) = &records[index];
            if outcomes[index].1 == GroupRestoreOutcome::Unchanged {
                // Already standing in the desired state — it counts as
                // landed for the Io/Timeout report below.
                applied += 1;
                continue;
            }
            if let Err(error) = groups::write_group(&self.0.data_dir, &file_stem(name), record) {
                return Err(RestoreGroupsError::Io {
                    group: name.clone(),
                    applied,
                    error,
                });
            }
            groups.insert(name.clone(), record.clone());
            applied += 1;
        }
        Ok(outcomes)
    }

    /// Every context a scoped key must hold to restore `records`: the
    /// closures of the named groups over the STANDING map (what the
    /// replace would release) unioned with their closures over the
    /// prospective one (what it would address). The import gate judges
    /// group records with this — [`Self::group_context_closures`]'
    /// twin for the restore path, where children may be names the set
    /// itself brings.
    pub fn group_restore_involves(&self, records: &[(String, GroupRecord)]) -> BTreeSet<String> {
        let groups = self.0.groups.read();
        let roots: Vec<&str> = records.iter().map(|(name, _)| name.as_str()).collect();
        let mut involved = groups::context_closure(&groups, roots.iter().copied());
        let mut prospective = groups.clone();
        for (name, record) in records {
            prospective.insert(name.clone(), record.clone());
        }
        involved.extend(groups::context_closure(&prospective, roots.iter().copied()));
        involved
    }

    /// Removes a group — the bundling only, never the member contexts
    /// nor the child groups. `None` for an unknown name, mirroring
    /// [`AppState::delete`]. Parents naming the group are swept inside
    /// the same critical section, so no reader ever observes a
    /// dangling child.
    ///
    /// One file, so no deletion marker: the memory drop and the unlink
    /// are it. The weaker guarantee is deliberate and priced in — if
    /// the unlink fails, the surviving file re-registers the group at
    /// the next boot, and the error message says so.
    pub fn delete_group(&self, name: &str) -> Option<io::Result<()>> {
        let mut groups = self.0.groups.write();
        groups.remove(name)?;
        // Nesting must not outlive the child — the same sweep a
        // deleted context gets, on the child field, under the write
        // lock already held (best effort past the removal; boot
        // reconciliation heals a sweep that could not persist).
        sweep_membership(&self.0.data_dir, &mut groups, name, |record| {
            &mut record.groups
        });
        Some(groups::remove_group_file(
            &self.0.data_dir,
            &file_stem(name),
        ))
    }

    /// Renames a group: its file moves to the new name and every OTHER
    /// group's `groups` field naming it is rewritten to match.
    /// `groups.write()` covers the whole operation, so — unlike a
    /// context rename — no separate reservation is needed: no
    /// concurrent create, update, delete, or rename can observe a
    /// half-renamed state, only wait behind this lock.
    ///
    /// The marker is written and durable BEFORE the file moves, for
    /// the same reason as [`AppState::rename_context`]'s: a crash
    /// between the file move and the membership rewrite must not have
    /// boot's `reconcile_groups` see the old name as a dangling
    /// reference and drop it, rather than resuming the rewrite.
    pub fn rename_group(&self, from: &str, to: &str) -> Result<(), RenameGroupError> {
        if to.is_empty() {
            return Err(RenameGroupError::InvalidName);
        }
        if from == to {
            return Ok(());
        }
        let mut groups = self.0.groups.write();
        if !groups.contains_key(from) {
            return Err(RenameGroupError::NotFound);
        }
        if groups.contains_key(to) {
            return Err(RenameGroupError::AlreadyExists);
        }
        let from_stem = file_stem(from);
        let to_stem = file_stem(to);
        let marker = groups::group_renaming_marker_path(&self.0.data_dir, &from_stem);
        write_rename_marker(&marker, from, to).map_err(RenameGroupError::Io)?;
        if let Err(error) = commit_staged(
            &groups::group_path(&self.0.data_dir, &from_stem),
            &groups::group_path(&self.0.data_dir, &to_stem),
        ) {
            let _ = fs::remove_file(&marker);
            return Err(RenameGroupError::Io(error));
        }
        let record = groups.remove(from).expect("checked contains_key above");
        groups.insert(to.to_string(), record);
        rename_in_membership(&self.0.data_dir, &mut groups, from, to, |record| {
            &mut record.groups
        });
        let _ = fs::remove_file(&marker);
        Ok(())
    }

    /// One group's record by name, or `None` for an unknown group.
    pub fn group(&self, name: &str) -> Option<GroupRecord> {
        self.0.groups.read().get(name).cloned()
    }

    /// Union of every context reachable from the named groups — direct
    /// members plus everything nested children bundle, transitively.
    /// The scoped write gate judges a group by what it ADDRESSES, so
    /// this is its view; unknown names contribute nothing.
    pub fn group_context_closures<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> BTreeSet<String> {
        groups::context_closure(&self.0.groups.read(), names)
    }

    /// [`group_context_closures`] with existence semantics: the first
    /// name that is not a registered group comes back as the error
    /// instead of contributing nothing. The cross-context searches
    /// resolve their `groups` targets here — a caller who NAMES a group
    /// deserves a `no_group` refusal, not a silently empty search —
    /// checked and walked under one lock acquisition so a concurrent
    /// group delete cannot slip between the two.
    pub fn resolve_groups(&self, names: &[String]) -> Result<BTreeSet<String>, String> {
        let groups = self.0.groups.read();
        if let Some(missing) = first_missing(names, |name| groups.contains_key(name)) {
            return Err(missing.clone());
        }
        Ok(groups::context_closure(
            &groups,
            names.iter().map(String::as_str),
        ))
    }

    /// One name-ordered page of groups plus the cursor-independent
    /// total. Scope filtering is the API layer's business, as with
    /// [`AppState::directory`] — but unlike the context directory,
    /// which clones only `Arc` handles and can hand over the whole
    /// map, a group's record IS its data, so the page is cut here
    /// under the read lock and only the survivors are cloned.
    pub fn group_page(
        &self,
        after: Option<&str>,
        limit: usize,
    ) -> (usize, Vec<(String, GroupRecord)>) {
        use std::ops::Bound;

        let groups = self.0.groups.read();
        let start = match after {
            Some(after) => Bound::Excluded(after),
            None => Bound::Unbounded,
        };
        let page = groups
            .range::<str, _>((start, Bound::Unbounded))
            .take(limit)
            .map(|(name, record)| (name.clone(), record.clone()))
            .collect();
        (groups.len(), page)
    }

    /// Drops a deleted context out of every group, persisting each
    /// touched record. Called from [`AppState::delete`] with the
    /// deletion marker already durable; best effort past that point —
    /// a rewrite that fails leaves memory correct and the file stale,
    /// which the next boot's reconciliation heals.
    pub(super) fn sweep_context_from_groups(&self, context_name: &str) {
        let mut groups = self.0.groups.write();
        sweep_membership(&self.0.data_dir, &mut groups, context_name, |record| {
            &mut record.contexts
        });
    }
}

/// The strict-membership gate shared by the group writes: the first
/// requested name the given namespace does not have, if any. Strict on
/// purpose — an add must never mint a dangling reference — and one
/// function for both namespaces (member contexts, child groups): the
/// caller supplies the existence test.
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

/// Shared body of the two membership sweeps — a deleted context out of
/// every group's members, a deleted group out of every parent's
/// children: removes `stale` from the chosen set field of every record
/// and persists each touched one. Best effort by design — a rewrite
/// that fails leaves memory correct and the file stale, which the next
/// boot's reconciliation heals. Lock-free on purpose: every caller
/// already holds the groups write lock.
fn sweep_membership(
    data_dir: &Path,
    groups: &mut BTreeMap<String, GroupRecord>,
    stale: &str,
    field: impl Fn(&mut GroupRecord) -> &mut BTreeSet<String>,
) {
    for (group_name, record) in groups.iter_mut() {
        if !field(record).remove(stale) {
            continue;
        }
        if let Err(error) = groups::write_group(data_dir, &file_stem(group_name), record) {
            tracing::warn!(
                group = %group_name,
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
    use crate::registry::paths::RenameMarker;
    use crate::registry::test_support::scratch_dir;
    use crate::storage::{clear_persistence_fault, fail_persistence_ops_after};

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
                BTreeSet::from(["sake".to_string()]),
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
        assert!(state.group("ghosts").is_none());
        assert!(!groups::group_path(&dir, &file_stem("ghosts")).exists());

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
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["sake".to_string()])
        );
        let updated = state
            .update_group(
                "drinks",
                Some("all drinks".into()),
                BTreeSet::from(["beer".to_string()]),
                BTreeSet::from(["sake".to_string()]),
                BTreeSet::new(),
                BTreeSet::new(),
            )
            .unwrap();
        assert_eq!(updated.description, "all drinks");
        assert_eq!(updated.contexts, BTreeSet::from(["beer".to_string()]));
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
        assert_eq!(survived.description, "all drinks");
        assert_eq!(survived.contexts, BTreeSet::from(["beer".to_string()]));

        // Deletion removes the record and its file; the members live on.
        state.delete_group("drinks").unwrap().unwrap();
        assert!(state.group("drinks").is_none());
        assert!(state.delete_group("drinks").is_none());
        assert!(!groups::group_path(&dir, &file_stem("drinks")).exists());
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
                assert_eq!(state.group("drinks").unwrap().description, "old");
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
            assert_eq!(state.group("drinks").unwrap().description, "new");
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
                    BTreeSet::from(["sake".to_string(), "beer".to_string()]),
                    BTreeSet::new(),
                )
                .unwrap();
        }

        state.delete("sake").unwrap().unwrap();

        for group in ["drinks", "fermented"] {
            assert_eq!(
                state.group(group).unwrap().contexts,
                BTreeSet::from(["beer".to_string()]),
                "'{group}' still names the deleted context"
            );
        }
        // The sweep persisted: a reboot reads the same membership.
        drop(state);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        for group in ["drinks", "fermented"] {
            assert_eq!(
                state.group(group).unwrap().contexts,
                BTreeSet::from(["beer".to_string()])
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
        assert!(state.group("g").is_none());

        // A delta is judged on its RESULT: 1 member + cap-many adds is
        // one too many, but trading the member out in the same request
        // makes room — the cap passes and the existence gate speaks
        // next, proving the judgement order.
        let one = BTreeSet::from(["real".to_string()]);
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
        assert_eq!(state.group("g").unwrap().contexts, one);

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
                BTreeSet::from(["sake".to_string(), "bunko".to_string()]),
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

        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: "new".to_string(),
            contexts: contexts.iter().map(|c| c.to_string()).collect(),
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
        assert_eq!(state.group("kura").unwrap(), records[0].1);
        assert_eq!(
            state.group("parent").unwrap().groups,
            BTreeSet::from(["kura".to_string()])
        );
        // Disk agrees.
        let on_disk: GroupRecord =
            serde_json::from_slice(&fs::read(groups::group_path(&dir, &file_stem("kid"))).unwrap())
                .unwrap();
        assert_eq!(on_disk, records[1].1);

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
        use crate::groups::group_path;

        let dir = scratch_dir("groups-restore-io-failure");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: String::new(),
            contexts: contexts.iter().map(|c| c.to_string()).collect(),
            groups: children.iter().map(|g| g.to_string()).collect(),
        };

        // "a" and "b" are independent (no nesting), so equal depth keeps
        // them in list order — "a" lands first. Occupying "b"'s target
        // path with a non-empty directory fails its rename on every
        // platform this project targets (see
        // `write_atomic_cleans_up_its_staging_file_when_the_commit_fails`),
        // isolating an Io failure strictly after "a" already landed.
        let blocked = group_path(&dir, &file_stem("b"));
        fs::create_dir(&blocked).unwrap();
        fs::write(blocked.join("occupied"), b"x").unwrap();

        let error = state
            .restore_groups(
                &[
                    ("a".to_string(), record(&[], &[])),
                    ("b".to_string(), record(&[], &[])),
                ],
                Deadline::unbounded(),
            )
            .unwrap_err();
        assert!(matches!(
            &error,
            RestoreGroupsError::Io { group, .. } if group == "b"
        ));
        assert_eq!(error.applied(), 1, "\"a\" already landed when \"b\" failed");
        assert!(
            state.group("a").is_some(),
            "\"a\"'s successful write must not be rolled back"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_counts_an_unchanged_record_as_already_landed() {
        use crate::groups::group_path;

        let dir = scratch_dir("groups-restore-unchanged-then-io-failure");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: String::new(),
            contexts: contexts.iter().map(|c| c.to_string()).collect(),
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

        // "b" is new, so it takes the `write_group` branch. Occupying its
        // target path with a non-empty directory fails its rename on
        // every platform this project targets (see
        // `write_atomic_cleans_up_its_staging_file_when_the_commit_fails`),
        // isolating an Io failure strictly after "a" is judged unchanged.
        let blocked = group_path(&dir, &file_stem("b"));
        fs::create_dir(&blocked).unwrap();
        fs::write(blocked.join("occupied"), b"x").unwrap();

        let error = state
            .restore_groups(
                &[
                    ("a".to_string(), record(&[], &[])),
                    ("b".to_string(), record(&[], &[])),
                ],
                Deadline::unbounded(),
            )
            .unwrap_err();
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
            contexts: BTreeSet::new(),
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
            state.group("a").is_none(),
            "no record may land past the budget"
        );
        assert!(
            !groups::group_path(&dir, &file_stem("a")).exists(),
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
            contexts: contexts.iter().map(|c| c.to_string()).collect(),
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
        assert!(
            state.group("fine").is_none(),
            "nothing applies on a refusal"
        );

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
        assert!(state.group("deep").is_none());

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
                            contexts: over,
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
        state
            .create_group("leaf", String::new(), one("tea"), BTreeSet::new())
            .unwrap();
        state
            .create_group("mid", String::new(), one("beer"), one("leaf"))
            .unwrap();
        state
            .create_group("top", String::new(), one("sake"), one("mid"))
            .unwrap();

        // The closure reads through the nesting; the record stays flat.
        assert_eq!(
            state.group_context_closures(["top"]),
            ["sake", "beer", "tea"]
                .iter()
                .map(|c| c.to_string())
                .collect()
        );
        assert_eq!(state.group("top").unwrap().contexts, one("sake"));

        // A fourth storey over a full chain refuses, and nothing lands.
        assert!(matches!(
            state.create_group("over", String::new(), BTreeSet::new(), one("top")),
            Err(CreateGroupError::Nesting(NestingViolation::TooDeep(_)))
        ));
        assert!(state.group("over").is_none());
        assert!(!groups::group_path(&dir, &file_stem("over")).exists());

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
        assert_eq!(state.group("leaf").unwrap().groups, BTreeSet::new());

        // Deleting a child sweeps it out of every parent, durably.
        state.delete_group("mid").unwrap().unwrap();
        assert_eq!(state.group("top").unwrap().groups, BTreeSet::new());
        drop(state);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        assert_eq!(state.group("top").unwrap().groups, BTreeSet::new());
        assert_eq!(state.group("leaf").unwrap().contexts, one("tea"));

        let _ = fs::remove_dir_all(dir);
    }

    fn assert_live_group_invariants(state: &AppState) {
        let contexts: BTreeSet<String> = state
            .directory()
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        let groups = state.0.groups.read();
        assert_eq!(groups::validate_nesting(&groups), Ok(()));
        for record in groups.values() {
            assert!(record.contexts.len() <= groups::MAX_GROUP_MEMBERS);
            assert!(record.groups.len() <= groups::MAX_GROUP_MEMBERS);
            assert!(record.contexts.iter().all(|name| contexts.contains(name)));
            assert!(record.groups.iter().all(|name| groups.contains_key(name)));
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
                BTreeSet::from(["sake".to_string()]),
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
            BTreeSet::from(["sake".to_string()]),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(matches!(outcome, Err(UpdateGroupError::Io(_))));
        let record = state.group("drinks").unwrap();
        assert_eq!(record.description, "before");
        assert_eq!(record.contexts, BTreeSet::from(["sake".to_string()]));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_group_moves_the_file_and_rewrites_parent_membership() {
        let dir = scratch_dir("rename-group-happy");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state
            .create_group(
                "liquor",
                "d".into(),
                BTreeSet::from(["sake".to_string()]),
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

        state.rename_group("liquor", "spirits").unwrap();

        assert!(state.group("liquor").is_none());
        let spirits = state.group("spirits").unwrap();
        assert_eq!(spirits.description, "d");
        assert_eq!(spirits.contexts, BTreeSet::from(["sake".to_string()]));
        assert_eq!(
            state.group("drinks").unwrap().groups,
            BTreeSet::from(["spirits".to_string()]),
            "the parent's child reference follows the rename"
        );
        assert!(!groups::group_path(&dir, &file_stem("liquor")).exists());
        assert!(groups::group_path(&dir, &file_stem("spirits")).exists());
        assert!(!groups::group_renaming_marker_path(&dir, &file_stem("liquor")).exists());

        // Persisted, not just in memory.
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.group("liquor").is_none());
        assert_eq!(
            state.group("drinks").unwrap().groups,
            BTreeSet::from(["spirits".to_string()])
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// The group twin of
    /// `creating_a_context_abandons_a_rename_marker_at_its_own_stem`: a
    /// `.grouprenaming` marker at the created group's own stem must be
    /// abandoned so boot does not resume-move the fresh group onto the
    /// rename's destination.
    #[test]
    fn creating_a_group_abandons_a_rename_marker_at_its_own_stem() {
        let dir = scratch_dir("create-group-clears-source-marker");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            fs::write(
                groups::group_renaming_marker_path(&dir, &file_stem("liquor")),
                serde_json::to_vec(&RenameMarker {
                    from: "liquor".to_string(),
                    to: "spirits".to_string(),
                })
                .unwrap(),
            )
            .unwrap();
            state
                .create_group("liquor", String::new(), BTreeSet::new(), BTreeSet::new())
                .unwrap();
            assert!(
                !groups::group_renaming_marker_path(&dir, &file_stem("liquor")).exists(),
                "create_group must clear a rename marker sitting at its own stem"
            );
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.group("liquor").is_some());
        assert!(state.group("spirits").is_none());
        assert!(groups::group_path(&dir, &file_stem("liquor")).exists());
        assert!(!groups::group_path(&dir, &file_stem("spirits")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// The group twin of
    /// `creating_a_context_abandons_a_rename_marker_naming_it_as_destination`:
    /// creating `spirits` must drop a `.grouprenaming` marker that names
    /// it as `to` (parked at `liquor`'s stem), or boot resume-moves the
    /// stale `liquor` group file over the fresh `spirits` and drops
    /// `liquor`.
    #[test]
    fn creating_a_group_abandons_a_rename_marker_naming_it_as_destination() {
        let dir = scratch_dir("create-group-clears-destination-marker");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create_group("liquor", String::new(), BTreeSet::new(), BTreeSet::new())
                .unwrap();
            fs::write(
                groups::group_renaming_marker_path(&dir, &file_stem("liquor")),
                serde_json::to_vec(&RenameMarker {
                    from: "liquor".to_string(),
                    to: "spirits".to_string(),
                })
                .unwrap(),
            )
            .unwrap();
            state
                .create_group("spirits", String::new(), BTreeSet::new(), BTreeSet::new())
                .unwrap();
            assert!(
                !groups::group_renaming_marker_path(&dir, &file_stem("liquor")).exists(),
                "create_group must clear a rename marker that names it as the destination"
            );
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.group("liquor").is_some(),
            "the abandoned rename must leave the untouched source group intact"
        );
        assert!(
            state.group("spirits").is_some(),
            "the freshly created destination group must survive, not be overwritten by the source"
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
        assert!(state.group("drinks").is_some());

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
                !state.group("parent").unwrap().groups.contains("leaf"),
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
        drop(state);

        // A dangling member, planted the way a crash between a
        // context's deletion and the sweep's rewrite would leave one.
        groups::write_group(
            &dir,
            &file_stem("drinks"),
            &GroupRecord {
                description: "d".into(),
                contexts: BTreeSet::from(["sake".to_string(), "gone".to_string()]),
                groups: BTreeSet::new(),
            },
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["sake".to_string()])
        );
        // Disk is the source of truth: the fix reached the file, not
        // just memory.
        let on_disk = fs::read_to_string(groups::group_path(&dir, &file_stem("drinks"))).unwrap();
        assert!(
            !on_disk.contains("gone"),
            "the dangling member survived on disk: {on_disk}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_corrupt_group_file_keeps_its_name_and_sets_the_bytes_aside() {
        let dir = scratch_dir("groups-corrupt");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        drop(state);
        let live = groups::group_path(&dir, &file_stem("mangled"));
        fs::write(&live, b"{not json").unwrap();

        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = state.group("mangled").unwrap();
        assert_eq!(record, GroupRecord::default());
        // The mangled bytes moved aside for hand recovery, and a fresh
        // empty record took their place — a later write to this group
        // overwrites nothing that was never loaded.
        assert_eq!(
            fs::read(live.with_extension("group.corrupt")).unwrap(),
            b"{not json",
            "the original bytes must survive, set aside"
        );
        let on_disk: GroupRecord = serde_json::from_slice(&fs::read(&live).unwrap()).unwrap();
        assert_eq!(on_disk, GroupRecord::default());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_crash_mid_repair_never_drops_the_corrupt_group_file_itself() {
        let mut exhausted = false;
        for failure in 0..6 {
            let dir = scratch_dir(&format!("groups-corrupt-fault-{failure}"));
            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            drop(state);
            let live = groups::group_path(&dir, &file_stem("mangled"));
            fs::write(&live, b"{not json").unwrap();

            fail_persistence_ops_after(failure);
            let result = AppState::boot(dir.clone(), 1 << 20, None);
            let past_end = clear_persistence_fault();

            if past_end {
                let state = result.unwrap();
                assert_eq!(state.group("mangled").unwrap(), GroupRecord::default());
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
                assert_eq!(state.group("mangled").unwrap(), GroupRecord::default());
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
        let imposter = dir.join("locked.group");
        fs::create_dir(&imposter).unwrap();
        let message = match AppState::boot(dir.clone(), 1 << 20, None) {
            Ok(_) => panic!("boot must refuse while a group file is unreadable"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("locked"), "names the group: {message}");
        // Clearing the obstacle heals without further ceremony.
        fs::remove_dir(&imposter).unwrap();
        assert!(AppState::boot(dir.clone(), 1 << 20, None).is_ok());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn boot_reconciliation_untangles_hand_written_nesting() {
        let dir = scratch_dir("groups-reconcile-nesting");
        drop(AppState::boot(dir.clone(), 1 << 20, None).unwrap());
        let write = |name: &str, children: &[&str]| {
            groups::write_group(
                &dir,
                &file_stem(name),
                &GroupRecord {
                    description: String::new(),
                    contexts: BTreeSet::new(),
                    groups: children.iter().map(|child| child.to_string()).collect(),
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
            |name: &str| -> BTreeSet<String> { state.group(name).unwrap().groups.clone() };
        assert_eq!(children("n1"), BTreeSet::from(["n2".to_string()]));
        assert_eq!(children("n2"), BTreeSet::from(["n3".to_string()]));
        assert_eq!(children("n3"), BTreeSet::new());
        assert_eq!(children("cyc-a"), BTreeSet::from(["cyc-b".to_string()]));
        assert_eq!(children("cyc-b"), BTreeSet::new());
        // Disk is the source of truth: the repairs reached the files.
        for (group, dropped) in [("cyc-b", "cyc-a"), ("n3", "n4"), ("n1", "ghost")] {
            let on_disk = fs::read_to_string(groups::group_path(&dir, &file_stem(group))).unwrap();
            assert!(
                !on_disk.contains(dropped),
                "'{group}' still names '{dropped}' on disk: {on_disk}"
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    /// The group-rename twin of
    /// `an_unfinished_context_rename_is_resumed_at_boot_before_group_reconciliation`:
    /// a surviving `.grouprenaming` marker must resume the file move
    /// AND rewrite the PARENT's `groups` set to the new child name
    /// before `reconcile_groups` runs, or the parent loses the child
    /// as a dangling reference instead of following it to its new name.
    #[test]
    fn an_unfinished_group_rename_is_resumed_at_boot_before_group_reconciliation() {
        let dir = scratch_dir("rename-group-crash");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .create_group(
                    "liquor",
                    String::new(),
                    BTreeSet::from(["sake".to_string()]),
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
        }
        fs::write(
            groups::group_renaming_marker_path(&dir, &file_stem("liquor")),
            serde_json::to_vec(&RenameMarker {
                from: "liquor".to_string(),
                to: "spirits".to_string(),
            })
            .unwrap(),
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.group("liquor").is_none());
        let spirits = state
            .group("spirits")
            .expect("the renamed group must exist");
        assert_eq!(spirits.contexts, BTreeSet::from(["sake".to_string()]));
        assert_eq!(
            state.group("drinks").unwrap().groups,
            BTreeSet::from(["spirits".to_string()]),
            "the parent must be REWRITTEN to the new child name, not pruned as dangling"
        );
        assert!(!groups::group_path(&dir, &file_stem("liquor")).exists());
        assert!(groups::group_path(&dir, &file_stem("spirits")).exists());
        assert!(!groups::group_renaming_marker_path(&dir, &file_stem("liquor")).exists());

        let _ = fs::remove_dir_all(dir);
    }
}
