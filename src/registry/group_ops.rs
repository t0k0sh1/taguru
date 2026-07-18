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
