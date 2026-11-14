//! Group records and their file I/O: one `group` is one `{id}.group`
//! file in the data directory, holding the `group`'s own id, its display
//! name, a routing description, the member `context` ids, and the child
//! `group` ids (nesting). Same philosophy as `contexts` — file existence
//! IS entity existence, no central manifest, discovered by the boot
//! scan — and the same identity rule (ADR 0045): the stem is the
//! `group`'s minted id, the name is a display label recorded inside the
//! file, so a rename rewrites one record and moves nothing.
//!
//! The extension is a SINGLE dot segment on purpose: an id never holds
//! a literal `.`, and `Path::extension()` cuts only the last dot — a
//! `{stem}.group.json` would answer `Some("json")` and the scan would
//! never see it. (The file's content is JSON regardless; the extension
//! names the entity, not the format, exactly as `.ctx` does.)
//!
//! Records are tiny (a description and two id lists), so the whole
//! collection lives resident behind one lock in the registry
//! ([`GroupTable`], keyed by id with a name index beside it); this
//! module owns the record shapes, the nesting rules, and the disk round
//! trip.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::storage::{remove_persisted_file, write_atomic};

/// The nesting ceiling: a chain of nested `groups` may stack at most
/// this many `groups` (a root, a child, a grandchild). Deep taxonomies
/// are filing, not addressing — retrieval wants short paths — and the
/// cap keeps every nesting walk in the codebase trivially bounded.
/// One constant to raise if that judgement changes.
pub(crate) const MAX_GROUP_DEPTH: usize = 3;

/// Ceiling on one `group`'s DIRECT membership — member `contexts` and
/// child `groups` each. The per-request delta lists already cap at the
/// same figure, but deltas accumulate: without a cap on the RESULT, a
/// `group` could be grown without bound patch by patch, and every row
/// serves its full membership (the `group` directory does not page
/// within a row). Matches the request-list cap, so anything a single
/// create can say is exactly what a `group` can hold; past it, split
/// into nested child `groups`.
pub(crate) const MAX_GROUP_MEMBERS: usize = 1000;

/// One `group`'s membership as a row and a merge speak of it: the
/// description, the member `context` ids, and the child `group` ids
/// (#996) — a [`StoredGroup`] without its own id and name, which the
/// row carries beside it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroupRecord {
    pub description: String,
    pub context_ids: BTreeSet<String>,
    /// Child `group` ids.
    pub group_ids: BTreeSet<String>,
}

/// One `group` as it lives on file and in memory: its own id (equal to
/// the file stem), its display name, and every reference by id —
/// member `contexts` (#965, ADR 0046) and child `groups` (#994). A name
/// is a display label that may repeat and may change; a record that
/// referred by name would silently follow the wrong `group` after a
/// rename or a second create, which is exactly why a rename used to
/// rewrite every parent and now rewrites nothing but its own record.
///
/// `deny_unknown_fields` makes a file of any other shape fail to parse
/// instead of loading half-read; the boot scan tells a pre-id file
/// (refused with a migration message) apart from a mangled one (set
/// aside) BEFORE this parse — see [`classify_group`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredGroup {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub context_ids: BTreeSet<String>,
    /// Child `group` ids — nesting, at most [`MAX_GROUP_DEPTH`] `groups`
    /// tall and never cyclic ([`validate_nesting`] guards both). A
    /// child may sit under several parents, exactly as a `context` may:
    /// the shape is a shallow DAG, not a tree.
    #[serde(default)]
    pub group_ids: BTreeSet<String>,
}

impl StoredGroup {
    /// A memberless record — the shape a mangled file is reset to.
    pub(crate) fn empty(id: &str, name: &str) -> Self {
        StoredGroup {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            context_ids: BTreeSet::new(),
            group_ids: BTreeSet::new(),
        }
    }
}

/// Mints a fresh `group` id — the one place every `group` create draws
/// from (today the registry's create and restore; #995's
/// `POST /groups` and the router's broadcast next). A UUID v4 in the
/// canonical spelling, like a `context` id, so it doubles as the file
/// stem with no encoding.
///
/// `TAGURU_TEST_DETERMINISTIC_IDS` swaps in a test-only sequence, as it
/// does for [`crate::registry::mint_context_id`] — but on a counter of
/// its own (variant nibble `9`, where contexts use `8`): sharing the
/// context counter would shift every later context id whenever a
/// `group` is created, and the committed wire fixtures carry context
/// ids.
pub(crate) fn mint_group_id() -> String {
    if std::env::var_os("TAGURU_TEST_DETERMINISTIC_IDS").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        return format!("00000000-0000-4000-9000-{n:012x}");
    }
    uuid::Uuid::new_v4().to_string()
}

/// The resident `group` collection: records by id, plus the name index
/// the directory pages through in name order. Every change of
/// a record's NAME goes through [`GroupTable::insert`], which keeps the
/// index in step; [`GroupTable::get_mut`] is for membership and
/// description edits only.
#[derive(Debug, Clone, Default)]
pub(crate) struct GroupTable {
    records: BTreeMap<String, StoredGroup>,
    by_name: BTreeMap<String, BTreeSet<String>>,
}

impl GroupTable {
    pub(crate) fn new(records: BTreeMap<String, StoredGroup>) -> Self {
        let mut table = GroupTable::default();
        for record in records.into_values() {
            table.insert(record);
        }
        table
    }

    /// Every record, keyed by id — what the nesting walks judge.
    pub(crate) fn records(&self) -> &BTreeMap<String, StoredGroup> {
        &self.records
    }

    pub(crate) fn len(&self) -> usize {
        self.records.len()
    }

    pub(crate) fn get(&self, id: &str) -> Option<&StoredGroup> {
        self.records.get(id)
    }

    /// Membership and description edits only — the name index is not
    /// consulted again, so a name changed through this handle would
    /// strand the index. Renames go through [`Self::insert`].
    pub(crate) fn get_mut(&mut self, id: &str) -> Option<&mut StoredGroup> {
        self.records.get_mut(id)
    }

    /// Mutable access to every record, for the membership sweeps; the
    /// same rule as [`Self::get_mut`] applies to each.
    pub(crate) fn records_mut(&mut self) -> impl Iterator<Item = &mut StoredGroup> {
        self.records.values_mut()
    }

    /// Adds `record`, or replaces the record holding its id, keeping
    /// the name index in step either way.
    pub(crate) fn insert(&mut self, record: StoredGroup) {
        if let Some(previous) = self.records.get(&record.id) {
            let name = previous.name.clone();
            self.unindex(&name, &record.id);
        }
        self.by_name
            .entry(record.name.clone())
            .or_default()
            .insert(record.id.clone());
        self.records.insert(record.id.clone(), record);
    }

    pub(crate) fn remove(&mut self, id: &str) -> Option<StoredGroup> {
        let record = self.records.remove(id)?;
        self.unindex(&record.name, id);
        Some(record)
    }

    fn unindex(&mut self, name: &str, id: &str) {
        if let Some(ids) = self.by_name.get_mut(name) {
            ids.remove(id);
            if ids.is_empty() {
                self.by_name.remove(name);
            }
        }
    }

    /// Every id recorded under `name`, in id order — none, one, or
    /// several (names are labels, ADR 0050): what a test that planted a
    /// `group` by name finds it by.
    #[cfg(test)]
    pub(crate) fn ids_named<'a>(&'a self, name: &str) -> impl Iterator<Item = &'a String> + 'a {
        self.by_name.get(name).into_iter().flatten()
    }

    /// The display name of the `group` holding `id`, falling back to the
    /// id itself for one not in the table (a nesting violation's blame
    /// can name a `group` of a prospective map; callers resolve those
    /// against that map instead).
    pub(crate) fn name_of<'a>(&'a self, id: &'a str) -> &'a str {
        self.records
            .get(id)
            .map_or(id, |record| record.name.as_str())
    }

    /// Records in display order — by name, ties by id — starting after
    /// the `(name, id)` cursor (exclusive). `after_id` alone is
    /// meaningless and the API refuses it; with `after` alone the page
    /// starts past every `group` called `after`, and with both it starts
    /// inside that name's run of same-named `groups`, after `after_id`
    /// — names are not unique, so the id is what lets a page break fall
    /// between duplicates without losing any.
    pub(crate) fn in_name_order<'a>(
        &'a self,
        after: Option<&'a str>,
        after_id: Option<&'a str>,
    ) -> impl Iterator<Item = &'a StoredGroup> + 'a {
        use std::ops::Bound;
        let start = match (after, after_id) {
            (Some(after), Some(_)) => Bound::Included(after),
            (Some(after), None) => Bound::Excluded(after),
            (None, _) => Bound::Unbounded,
        };
        self.by_name
            .range::<str, _>((start, Bound::Unbounded))
            .flat_map(move |(name, ids)| {
                ids.iter()
                    .filter(move |id| match (after, after_id) {
                        (Some(after), Some(after_id)) if name == after => id.as_str() > after_id,
                        _ => true,
                    })
                    .map(|id| &self.records[id])
            })
    }
}

/// Why a proposed nesting cannot stand. Carries a `group` key
/// pinpointing the violation — the id, in the stored map the registry
/// judges (the registry turns it back into a name before it reaches a
/// caller) — deterministic, because the map and each child set iterate
/// in key order. Which `group`, per variant, is documented below.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NestingViolation {
    /// The named `group` reaches itself through its children — a `group` on
    /// the cycle, the one the walk was at when it closed the loop.
    Cycle(String),
    /// A chain of more than [`MAX_GROUP_DEPTH`] `groups`; the key is the
    /// TOP of that chain (the walk's outermost `group`), reported the
    /// same whether the cap is hit descending or on the way back up. A
    /// cycle longer than the cap also lands here — its chains are over
    /// the cap either way, and the walk refuses before following a path
    /// far enough to see the loop close.
    TooDeep(String),
}

impl NestingViolation {
    /// The same violation with its blamed `group` re-keyed — the
    /// registry's id → name step before a refusal leaves it.
    pub(crate) fn rename(self, rename: impl FnOnce(&str) -> String) -> Self {
        match self {
            NestingViolation::Cycle(group) => NestingViolation::Cycle(rename(&group)),
            NestingViolation::TooDeep(group) => NestingViolation::TooDeep(rename(&group)),
        }
    }
}

/// Checks the whole collection for the two nesting invariants — no
/// cycles, no chain of more than [`MAX_GROUP_DEPTH`] `groups` — in one
/// memoized walk, O(groups + edges). Callers hand it the PROSPECTIVE
/// map (under the `groups` write lock, before anything persists) or, at
/// boot, whatever the files claim. A child id with no record counts
/// as depth zero: dangling references are referential integrity's
/// business, healed elsewhere.
pub(crate) fn validate_nesting(
    groups: &BTreeMap<String, StoredGroup>,
) -> Result<(), NestingViolation> {
    nesting_depths(groups).map(|_| ())
}

/// [`validate_nesting`]'s walk with its memo handed back: every
/// `group`'s chain depth (leaves are 1). The `group` restore writes
/// children before the parents that name them, and this map is that
/// order — computable only on a collection the validator accepts,
/// which the `Result` enforces.
pub(crate) fn nesting_depths(
    groups: &BTreeMap<String, StoredGroup>,
) -> Result<BTreeMap<&str, usize>, NestingViolation> {
    let mut settled = BTreeMap::new();
    let mut visiting = Vec::new();
    for id in groups.keys() {
        chain_depth(groups, id, &mut settled, &mut visiting)?;
    }
    Ok(settled)
}

/// The number of `groups` on the longest chain hanging from `id`,
/// itself included — depth-first, with a visiting stack for cycle
/// detection and a settled map so shared children are walked once.
/// Recursion is bounded by the cap, not the input: a path already
/// [`MAX_GROUP_DEPTH`] `groups` long refuses before descending further,
/// so even a hand-written thousand-`group` chain cannot blow the stack.
fn chain_depth<'a>(
    groups: &'a BTreeMap<String, StoredGroup>,
    id: &'a str,
    settled: &mut BTreeMap<&'a str, usize>,
    visiting: &mut Vec<&'a str>,
) -> Result<usize, NestingViolation> {
    if let Some(&depth) = settled.get(id) {
        return Ok(depth);
    }
    let Some(record) = groups.get(id) else {
        return Ok(0);
    };
    if visiting.contains(&id) {
        return Err(NestingViolation::Cycle(id.to_string()));
    }
    if visiting.len() >= MAX_GROUP_DEPTH {
        // The path down to here already stacks MAX + 1 groups counting
        // this one; whatever hangs below cannot make that legal.
        return Err(NestingViolation::TooDeep(visiting[0].to_string()));
    }
    visiting.push(id);
    let mut below = 0;
    for child in &record.group_ids {
        below = below.max(chain_depth(groups, child, settled, visiting)?);
    }
    visiting.pop();
    let depth = below + 1;
    if depth > MAX_GROUP_DEPTH {
        // Name the TOP of the offending chain, like the descent guard
        // above (`visiting[0]`). When the over-tall subtree is reached
        // through a memoized child, `id` is a MIDDLE group, not the
        // chain's root — so the two paths would otherwise blame
        // different groups for the same violation. `visiting` holds
        // `id`'s ancestors here (it was just popped); its first frame
        // is the chain's root, or `id` itself when `id` is a root.
        let blame = visiting.first().copied().unwrap_or(id);
        return Err(NestingViolation::TooDeep(blame.to_string()));
    }
    settled.insert(id, depth);
    Ok(depth)
}

/// Makes an arbitrary collection satisfy the two nesting invariants by
/// dropping edges: on a validator refusal every child set is rebuilt
/// edge by edge — map and set order, so the repair is deterministic —
/// and each edge the validator refuses is dropped, with a warning
/// naming it. A cheap no-op on a valid collection; quadratic on a
/// broken one, which only a hand-edited data directory can produce.
/// Callers drop dangling child references first: a dangling edge is
/// not a shape violation, so the repair would keep it.
pub(crate) fn repair_nesting(groups: &mut BTreeMap<String, StoredGroup>) {
    if validate_nesting(groups).is_ok() {
        return;
    }
    let edges: Vec<(String, String)> = groups
        .iter()
        .flat_map(|(parent, record)| {
            record
                .group_ids
                .iter()
                .map(|child| (parent.clone(), child.clone()))
        })
        .collect();
    for record in groups.values_mut() {
        record.group_ids.clear();
    }
    for (parent, child) in edges {
        groups
            .get_mut(&parent)
            .unwrap()
            .group_ids
            .insert(child.clone());
        if validate_nesting(groups).is_err() {
            groups.get_mut(&parent).unwrap().group_ids.remove(&child);
            tracing::warn!(
                group = %groups[&parent].name,
                group_id = %parent,
                child_id = %child,
                "dropped a nesting edge: it would close a cycle or stack more than {MAX_GROUP_DEPTH} groups"
            );
        }
    }
}

/// Trims every record's direct membership back under `cap` ids per
/// set — member `contexts` and child `groups` each — keeping the FIRST
/// `cap` in id order, so the repair is deterministic. Nothing
/// running can persist an over-cap set (the write paths refuse
/// first); this is boot's counterpart for a hand-edited file, handed
/// [`MAX_GROUP_MEMBERS`] by the reconciler and a small cap by tests.
pub(crate) fn trim_membership(groups: &mut BTreeMap<String, StoredGroup>, cap: usize) {
    for record in groups.values_mut() {
        for (field, set) in [
            ("context_ids", &mut record.context_ids),
            ("group_ids", &mut record.group_ids),
        ] {
            if set.len() <= cap {
                continue;
            }
            let dropped = set.len() - cap;
            while set.len() > cap {
                set.pop_last();
            }
            tracing::warn!(
                group = %record.name,
                group_id = %record.id,
                field,
                dropped,
                "dropped membership past the {cap}-id cap; the first {cap} ids survive"
            );
        }
    }
}

/// Every `context` reachable from the given roots — direct members plus
/// everything nested children bundle, transitively. The scoped write
/// gate judges a `group` by this closure: a grant must cover what the
/// `group` ADDRESSES, not just what it lists directly. Keys without a
/// record contribute nothing; the seen set keeps even a (transient,
/// mid-repair) cycle from looping the walk.
pub(crate) fn context_closure<'map, 'roots: 'map>(
    groups: &'map BTreeMap<String, StoredGroup>,
    roots: impl IntoIterator<Item = &'roots str>,
) -> BTreeSet<String> {
    let mut contexts = BTreeSet::new();
    let mut seen: BTreeSet<&'map str> = BTreeSet::new();
    let mut frontier: Vec<&'map str> = roots
        .into_iter()
        .map(|root| -> &'map str { root })
        .filter(|root| seen.insert(*root))
        .collect();
    while let Some(key) = frontier.pop() {
        let Some(record) = groups.get(key) else {
            continue;
        };
        contexts.extend(record.context_ids.iter().cloned());
        frontier.extend(
            record
                .group_ids
                .iter()
                .map(String::as_str)
                .filter(|child| seen.insert(*child)),
        );
    }
    contexts
}

pub(crate) fn group_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.group"))
}

/// Persists one `group` at its id's stem via the registry's staged write
/// (fsync + rename + parent fsync): a crash mid-write leaves the
/// previous version intact. The staging name is `{stem}.tmp{n}`, which
/// the boot scan's leftover sweep already removes — nothing
/// `group`-specific to clean up.
pub(crate) fn write_group(dir: &Path, record: &StoredGroup) -> io::Result<()> {
    write_atomic(
        &group_path(dir, &record.id),
        &serde_json::to_vec_pretty(record)?,
    )
}

/// Unlinks one `group` file. A file already gone counts as success — the
/// caller's intent (this `group` does not exist on disk) is satisfied.
pub(crate) fn remove_group_file(dir: &Path, id: &str) -> io::Result<()> {
    match remove_persisted_file(group_path(dir, id)) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

/// What one `.group` file's bytes are, judged against the stem it sits
/// under — the one classification the boot scan, the replica tailer,
/// and `taguru inspect` share, so the three cannot disagree about which
/// files a server would refuse.
#[derive(Debug)]
pub(crate) enum ScannedGroup {
    /// Parsed, and its recorded `id` equals its stem — the only healthy
    /// case.
    Current(StoredGroup),
    /// Written before `groups` had ids (ADR 0045): the stem is not an id
    /// (it was a percent-encoded name), or the record is a JSON object
    /// with no `id`. Refused, never half-read — and never reset as
    /// "corrupt", which would overwrite a `group` that only needs the
    /// documented migration ([`crate::format::migration_hint`]; §2.7:
    /// no compatibility).
    PreId,
    /// Parsed, but records an `id` other than its stem: the file was
    /// copied or hand-edited. Refused — trusting either value would
    /// silently rebind the other.
    ForeignId(String),
    /// Does not parse as a current record — the boot scan keeps the
    /// `group` (its id IS the stem) and sets the bytes aside.
    Corrupt(String),
}

/// Whether `stem` can be a `group` id at all: the canonical UUID
/// spelling [`mint_group_id`] produces, the same rule `context` ids
/// follow.
pub(crate) fn is_group_id(stem: &str) -> bool {
    crate::registry::is_context_id(stem)
}

/// Classifies one `.group` file's bytes (see [`ScannedGroup`]). A pre-id
/// or foreign-id shape is recognized BEFORE the strict parse, so neither
/// can fall into the corrupt arm's reset.
pub(crate) fn classify_group(stem: &str, bytes: &[u8]) -> ScannedGroup {
    if !is_group_id(stem) {
        return ScannedGroup::PreId;
    }
    // The identity check reads the raw object before the strict parse,
    // so a file that is BOTH foreign and otherwise odd (an unknown
    // field, say) is still refused as foreign — never reset as corrupt,
    // which would overwrite a record that only sits under the wrong stem.
    if let Ok(serde_json::Value::Object(fields)) = serde_json::from_slice(bytes) {
        match fields.get("id") {
            None => return ScannedGroup::PreId,
            Some(serde_json::Value::String(id)) if id != stem => {
                return ScannedGroup::ForeignId(id.clone());
            }
            Some(_) => {}
        }
    }
    match serde_json::from_slice::<StoredGroup>(bytes) {
        Ok(record) if record.id == stem => ScannedGroup::Current(record),
        Ok(record) => ScannedGroup::ForeignId(record.id),
        Err(error) => ScannedGroup::Corrupt(error.to_string()),
    }
}

/// The boot refusal a [`ScannedGroup::PreId`] file means — shared with
/// `taguru inspect`, which reports the same sentence as a FAILURE.
pub(crate) fn pre_id_refusal(file: &str) -> String {
    format!(
        "group file '{file}' records no group id: this data directory was written \
         before groups had ids (ADR 0050) and this build does not read it — {}",
        crate::format::migration_hint()
    )
}

/// The boot refusal a [`ScannedGroup::ForeignId`] file means.
pub(crate) fn foreign_id_refusal(file: &str, id: &str) -> String {
    format!(
        "group file '{file}' records id '{id}', not its own stem: the file was copied \
         or hand-edited — restore it under its original stem"
    )
}

/// One boot-time pass for `groups`, run after the `context` scan (which
/// also sweeps staging leftovers), returning every record keyed by id.
/// Failures are loud on purpose — the directory scan just listed the
/// file, so trouble reading it is always news (contrast a `context`'s
/// sidecar read, which treats absence as the normal never-written
/// case):
///
/// - An UNREADABLE file refuses the boot. The `group`'s real membership
///   is out of reach, and registering the id over an empty record
///   would hand the next write to that `group` a license to overwrite
///   whatever the file still holds — a transient permission hiccup
///   must not turn into silent membership loss.
/// - A pre-id or foreign-id file refuses the boot (see
///   [`ScannedGroup`]): neither is damage, and neither may be served.
/// - A file that reads but does not PARSE keeps its id and loses its
///   content: the entity must not vanish because its record got
///   mangled. Its name is lost with the content, so the id stands in
///   for it until an operator renames it. The mangled bytes are set
///   aside as `{stem}.group.corrupt` (evidence for hand recovery; every
///   other scan ignores the extension) and a fresh empty record takes
///   their place on disk, so disk and memory agree from the first
///   request and no later write clobbers bytes that were never loaded.
pub(crate) fn scan_groups(dir: &Path) -> io::Result<BTreeMap<String, StoredGroup>> {
    let mut groups = BTreeMap::new();
    for dir_entry in fs::read_dir(dir)? {
        let path = dir_entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("group") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            tracing::warn!("skipping {}: file name is not UTF-8", path.display());
            continue;
        };
        let file = format!("{stem}.group");
        let bytes = fs::read(&path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "group file '{file}' is unreadable ({}): {error} — restore the file \
                     (or move it out of the data directory) and start again",
                    path.display()
                ),
            )
        })?;
        let record = match classify_group(stem, &bytes) {
            ScannedGroup::Current(record) => record,
            ScannedGroup::PreId => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    pre_id_refusal(&file),
                ));
            }
            ScannedGroup::ForeignId(id) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    foreign_id_refusal(&file, &id),
                ));
            }
            ScannedGroup::Corrupt(error) => {
                let set_aside = path.with_extension("group.corrupt");
                tracing::warn!(
                    group_id = %stem,
                    bytes_at = %set_aside.display(),
                    %error,
                    "group file does not parse; keeping the group empty and setting the bytes aside"
                );
                // The empty replacement is written FIRST: `write_group`
                // is a staged write (stage + fsync + rename), so `path`
                // always resolves to either the mangled bytes or the
                // fresh empty record, never to nothing in between.
                // Setting the mangled bytes aside second means a crash
                // between the two steps costs only the hand-recovery
                // copy, never the group's existence — the reverse order
                // (rename `path` away, then write the replacement) has a
                // window where `path` doesn't exist at all, and a crash
                // there drops the group from the very next scan.
                let empty = StoredGroup::empty(stem, stem);
                write_group(dir, &empty)?;
                fs::write(&set_aside, &bytes)?;
                empty
            }
        };
        groups.insert(record.id.clone(), record);
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stored map whose ids double as names — the nesting walks see
    /// only ids, so the tests can speak in short ones.
    fn map(entries: &[(&str, &[&str], &[&str])]) -> BTreeMap<String, StoredGroup> {
        entries
            .iter()
            .map(|(id, contexts, children)| {
                (
                    id.to_string(),
                    StoredGroup {
                        id: id.to_string(),
                        name: id.to_string(),
                        description: String::new(),
                        context_ids: contexts.iter().map(|c| c.to_string()).collect(),
                        group_ids: children.iter().map(|g| g.to_string()).collect(),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn nesting_accepts_a_full_chain_and_names_what_it_refuses() {
        // Exactly MAX_GROUP_DEPTH stacked groups pass; one more is a
        // cap violation reported at the top of the chain.
        let full = map(&[("a", &[], &["b"]), ("b", &[], &["c"]), ("c", &[], &[])]);
        assert_eq!(validate_nesting(&full), Ok(()));
        let over = map(&[
            ("a", &[], &["b"]),
            ("b", &[], &["c"]),
            ("c", &[], &["d"]),
            ("d", &[], &[]),
        ]);
        assert_eq!(
            validate_nesting(&over),
            Err(NestingViolation::TooDeep("a".to_string()))
        );
        // A cycle names a group on it — the self-loop included — and a
        // dangling child is integrity's problem, not nesting's.
        let cycle = map(&[("a", &[], &["b"]), ("b", &[], &["a"])]);
        assert_eq!(
            validate_nesting(&cycle),
            Err(NestingViolation::Cycle("a".to_string()))
        );
        let selfloop = map(&[("a", &[], &["a"])]);
        assert_eq!(
            validate_nesting(&selfloop),
            Err(NestingViolation::Cycle("a".to_string()))
        );
        assert_eq!(validate_nesting(&map(&[("a", &[], &["ghost"])])), Ok(()));
    }

    #[test]
    fn too_deep_blames_the_chain_top_even_through_a_memoized_child() {
        // aaa->bbb->ccc settles at depth 3 (legal) while walked first in
        // name order. Then p_top->q_mid->aaa is discovered later, its
        // over-tall subtree reached through the memoized `aaa`, so the
        // descent guard never fires — the cap trips on the way back up
        // at q_mid. The blame must still be the TOP of the chain
        // (p_top), not the middle group q_mid the walk was at: the two
        // detection paths agree on which group they name.
        let groups = map(&[
            ("aaa", &[], &["bbb"]),
            ("bbb", &[], &["ccc"]),
            ("ccc", &[], &[]),
            ("p_top", &[], &["q_mid"]),
            ("q_mid", &[], &["aaa"]),
        ]);
        assert_eq!(
            validate_nesting(&groups),
            Err(NestingViolation::TooDeep("p_top".to_string())),
            "names the chain's top, not the memoized middle group"
        );
    }

    #[test]
    fn nesting_depths_hand_back_the_validators_memo() {
        let groups = map(&[("a", &[], &["b"]), ("b", &[], &["c"]), ("c", &[], &[])]);
        let depths = nesting_depths(&groups).unwrap();
        assert_eq!(depths["a"], 3);
        assert_eq!(depths["b"], 2);
        assert_eq!(depths["c"], 1);
    }

    #[test]
    fn repair_drops_exactly_the_edges_the_validator_refuses() {
        let mut groups = map(&[
            ("cyc-a", &[], &["cyc-b"]),
            ("cyc-b", &[], &["cyc-a"]),
            ("n1", &[], &["n2"]),
            ("n2", &[], &["n3"]),
            ("n3", &[], &["n4"]),
            ("n4", &[], &[]),
        ]);
        repair_nesting(&mut groups);
        assert_eq!(validate_nesting(&groups), Ok(()));
        // Deterministic: edges re-admitted in name order, so
        // (cyc-b, cyc-a) and (n3, n4) are the ones that fall.
        let children = |name: &str| groups[name].group_ids.iter().cloned().collect::<Vec<_>>();
        assert_eq!(children("cyc-a"), vec!["cyc-b"]);
        assert!(children("cyc-b").is_empty());
        assert_eq!(children("n1"), vec!["n2"]);
        assert_eq!(children("n2"), vec!["n3"]);
        assert!(children("n3").is_empty());
        // A valid collection comes back untouched.
        let valid = map(&[("a", &["x"], &["b"]), ("b", &[], &[])]);
        let mut repaired = valid.clone();
        repair_nesting(&mut repaired);
        assert_eq!(repaired, valid);
    }

    #[test]
    fn trim_keeps_the_first_cap_names_of_each_set_and_leaves_the_rest_alone() {
        let mut groups = map(&[
            ("wide", &["c1", "c2", "c3"], &["g1", "g2", "g3"]),
            ("fits", &["c1", "c2"], &[]),
        ]);
        trim_membership(&mut groups, 2);
        assert_eq!(
            groups["wide"].context_ids,
            ["c1", "c2"].iter().map(|c| c.to_string()).collect()
        );
        assert_eq!(
            groups["wide"].group_ids,
            ["g1", "g2"].iter().map(|g| g.to_string()).collect()
        );
        assert_eq!(
            groups["fits"].context_ids.len(),
            2,
            "within the cap: untouched"
        );
    }

    #[test]
    fn the_context_closure_reads_through_children_and_ignores_unknown_roots() {
        let groups = map(&[
            ("top", &["t"], &["mid", "side"]),
            ("mid", &["m"], &["leaf"]),
            ("side", &[], &["leaf"]),
            ("leaf", &["l1", "l2"], &[]),
        ]);
        assert_eq!(
            context_closure(&groups, ["top", "nope"]),
            ["t", "m", "l1", "l2"]
                .iter()
                .map(|c| c.to_string())
                .collect()
        );
        assert!(context_closure(&groups, ["nope"]).is_empty());
    }

    /// Issue #753: the trim warn reports exactly how many names fell
    /// past the cap — sizes chosen so subtraction, addition, and
    /// integer division all disagree (5-2=3 vs 5/2=2 vs 5+2=7).
    #[test]
    fn trim_membership_drops_past_the_cap_and_counts_the_drop_exactly() {
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct DroppedRecorder(Arc<Mutex<Vec<u64>>>);
        impl tracing::Subscriber for DroppedRecorder {
            fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, event: &tracing::Event<'_>) {
                struct Dropped<'a>(&'a Mutex<Vec<u64>>);
                impl tracing::field::Visit for Dropped<'_> {
                    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                        if field.name() == "dropped" {
                            self.0.lock().unwrap().push(value);
                        }
                    }
                    fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {
                    }
                }
                event.record(&mut Dropped(&self.0));
            }
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }

        let mut groups = map(&[("g", &["a", "b", "c", "d", "e"], &["w", "x", "y", "z"])]);
        let dropped = Arc::new(Mutex::new(Vec::new()));
        tracing::subscriber::with_default(DroppedRecorder(Arc::clone(&dropped)), || {
            trim_membership(&mut groups, 2)
        });
        let record = &groups["g"];
        assert_eq!(
            record
                .context_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["a", "b"],
            "the FIRST names in order survive"
        );
        assert_eq!(
            record
                .group_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["w", "x"]
        );
        assert_eq!(
            dropped.lock().unwrap().as_slice(),
            [3, 2],
            "the warn counts exactly what fell past the cap"
        );

        // At or under the cap nothing moves and nothing warns.
        let mut under = map(&[("g", &["a", "b"], &[])]);
        let quiet = Arc::new(Mutex::new(Vec::new()));
        tracing::subscriber::with_default(DroppedRecorder(Arc::clone(&quiet)), || {
            trim_membership(&mut under, 2)
        });
        assert!(quiet.lock().unwrap().is_empty());
        assert_eq!(under["g"].context_ids.len(), 2);
    }

    /// Issue #753: a `group` file already gone counts as removed;
    /// anything else — a directory squatting on the name, which
    /// `remove_file` refuses even for root — surfaces as the error it
    /// is instead of being swallowed as "already satisfied".
    #[test]
    fn remove_group_file_swallows_only_not_found() {
        let dir = std::env::temp_dir().join(format!("taguru-groups-remove-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        assert!(remove_group_file(&dir, "ghost").is_ok());

        fs::create_dir_all(dir.join("blocked.group").join("x")).unwrap();
        assert!(
            remove_group_file(&dir, "blocked").is_err(),
            "a non-NotFound refusal must surface"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    const ID_A: &str = "00000000-0000-4000-9000-00000000000a";
    const ID_B: &str = "00000000-0000-4000-9000-00000000000b";

    fn stored(id: &str, name: &str, children: &[&str]) -> StoredGroup {
        StoredGroup {
            group_ids: children.iter().map(|g| g.to_string()).collect(),
            ..StoredGroup::empty(id, name)
        }
    }

    /// ADR 0045 §2.7: the classification boot, the replica tailer, and
    /// inspect share. Only an id-stemmed record carrying its own stem is
    /// current; a pre-id shape — by stem or by content — is never
    /// mistaken for mangled bytes (which the boot would reset).
    #[test]
    fn classify_group_tells_pre_id_and_foreign_files_from_corrupt_ones() {
        let current = serde_json::to_vec(&stored(ID_A, "drinks", &[ID_B])).unwrap();
        match classify_group(ID_A, &current) {
            ScannedGroup::Current(record) => assert_eq!(record, stored(ID_A, "drinks", &[ID_B])),
            other => panic!("expected current, got {other:?}"),
        }
        assert!(matches!(
            classify_group(ID_B, &current),
            ScannedGroup::ForeignId(id) if id == ID_A
        ));
        // The pre-#994 record: no id, children by name, under a
        // percent-encoded name stem — refused by either sign alone.
        let pre_id = br#"{"description":"","context_ids":[],"groups":["kid"]}"#;
        assert!(matches!(
            classify_group("drinks", pre_id),
            ScannedGroup::PreId
        ));
        assert!(matches!(classify_group(ID_A, pre_id), ScannedGroup::PreId));
        assert!(matches!(
            classify_group("drinks", &current),
            ScannedGroup::PreId
        ));
        assert!(matches!(
            classify_group(ID_A, b"{not json"),
            ScannedGroup::Corrupt(_)
        ));
        // A non-string id is not an identity claim to compare; the
        // strict parse refuses it as corrupt.
        assert!(matches!(
            classify_group(ID_A, br#"{"id":7,"name":"x"}"#),
            ScannedGroup::Corrupt(_)
        ));
        // Shaped like a record but not one (an id, then a field this
        // build does not know): corrupt, not pre-id.
        assert!(matches!(
            classify_group(
                ID_A,
                format!(r#"{{"id":"{ID_A}","name":"x","groups":[]}}"#).as_bytes()
            ),
            ScannedGroup::Corrupt(_)
        ));
    }

    #[test]
    fn scan_groups_refuses_a_pre_id_directory_without_touching_it() {
        let dir = std::env::temp_dir().join(format!("taguru-groups-pre-id-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let pre_id = br#"{"description":"d","context_ids":[],"groups":[]}"#;
        fs::write(dir.join("drinks.group"), pre_id).unwrap();
        let error = scan_groups(&dir).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("records no group id"), "{error}");
        assert!(
            error
                .to_string()
                .ends_with(&crate::format::migration_hint()),
            "{error}"
        );
        assert_eq!(
            fs::read(dir.join("drinks.group")).unwrap(),
            pre_id,
            "a refused file is left exactly as it was"
        );
        assert!(!dir.join("drinks.group.corrupt").exists());

        fs::remove_file(dir.join("drinks.group")).unwrap();
        fs::write(
            dir.join(format!("{ID_B}.group")),
            serde_json::to_vec(&stored(ID_A, "copied", &[])).unwrap(),
        )
        .unwrap();
        let error = scan_groups(&dir).unwrap_err();
        assert!(
            error.to_string().contains(&format!("records id '{ID_A}'")),
            "{error}"
        );

        // Foreign AND otherwise unparseable (an unknown field): still
        // refused as foreign, and the file is left exactly as it was.
        let odd = format!(r#"{{"id":"{ID_A}","name":"copied","extra":1}}"#);
        fs::write(dir.join(format!("{ID_B}.group")), &odd).unwrap();
        let error = scan_groups(&dir).unwrap_err();
        assert!(
            error.to_string().contains(&format!("records id '{ID_A}'")),
            "{error}"
        );
        assert_eq!(
            fs::read(dir.join(format!("{ID_B}.group"))).unwrap(),
            odd.as_bytes()
        );
        assert!(!dir.join(format!("{ID_B}.group.corrupt")).exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_groups_keys_by_id_and_resets_a_mangled_record_under_its_stem() {
        let dir = std::env::temp_dir().join(format!("taguru-groups-scan-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        write_group(&dir, &stored(ID_A, "drinks", &[ID_B])).unwrap();
        fs::write(dir.join(format!("{ID_B}.group")), b"{not json").unwrap();
        let groups = scan_groups(&dir).unwrap();
        assert_eq!(groups[ID_A], stored(ID_A, "drinks", &[ID_B]));
        assert_eq!(
            groups[ID_B],
            StoredGroup::empty(ID_B, ID_B),
            "the id survives; the lost name falls back to it"
        );
        assert_eq!(
            fs::read(dir.join(format!("{ID_B}.group.corrupt"))).unwrap(),
            b"{not json"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_group_table_indexes_names_through_every_insert_and_remove() {
        let mut table = GroupTable::new(
            [
                (ID_A.to_string(), stored(ID_A, "drinks", &[ID_B])),
                (ID_B.to_string(), stored(ID_B, "beer", &[])),
            ]
            .into(),
        );
        assert_eq!(table.len(), 2);
        assert_eq!(table.ids_named("drinks").collect::<Vec<_>>(), [ID_A]);
        assert_eq!(table.ids_named("nope").count(), 0);
        assert_eq!(table.name_of(ID_B), "beer");
        assert_eq!(table.name_of("unknown"), "unknown");

        // A rename through insert moves the index entry with it.
        table.insert(stored(ID_B, "drinks", &[]));
        assert_eq!(table.ids_named("beer").count(), 0);
        assert_eq!(
            table.ids_named("drinks").collect::<Vec<_>>(),
            [ID_A, ID_B],
            "several ids under one name, in id order"
        );
        assert_eq!(
            table
                .in_name_order(None, None)
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            [ID_A, ID_B],
            "a name tie breaks by id"
        );

        assert_eq!(
            table.remove(ID_A).map(|record| record.name),
            Some("drinks".into())
        );
        assert_eq!(table.remove(ID_A), None);
        assert_eq!(table.ids_named("drinks").collect::<Vec<_>>(), [ID_B]);
        table.remove(ID_B);
        assert_eq!(
            table.ids_named("drinks").count(),
            0,
            "an emptied name leaves the index"
        );
    }

    #[test]
    fn the_name_order_pages_after_a_name_and_id_cursor_exclusively() {
        const ID_C: &str = "00000000-0000-4000-9000-00000000000c";
        let table = GroupTable::new(
            [
                (ID_A.to_string(), stored(ID_A, "b", &[])),
                (ID_B.to_string(), stored(ID_B, "a", &[])),
                (ID_C.to_string(), stored(ID_C, "a", &[])),
            ]
            .into(),
        );
        let ids = |after, after_id| {
            table
                .in_name_order(after, after_id)
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(None, None), [ID_B, ID_C, ID_A]);
        // A name alone skips every `group` under it.
        assert_eq!(ids(Some("a"), None), [ID_A]);
        assert_eq!(ids(Some("b"), None), Vec::<&str>::new());
        // With an id, the page resumes inside the name's run.
        assert_eq!(ids(Some("a"), Some(ID_B)), [ID_C, ID_A]);
        assert_eq!(ids(Some("a"), Some(ID_C)), [ID_A]);
        // The id only narrows the cursor's own name, never a later one.
        assert_eq!(
            ids(Some("a"), Some("ffffffff-ffff-4fff-9fff-ffffffffffff")),
            [ID_A]
        );
        assert_eq!(ids(Some("b"), Some(ID_A)), Vec::<&str>::new());
        // A cursor name nothing carries still lands between its neighbours.
        assert_eq!(ids(Some("a0"), Some(ID_A)), [ID_A]);
    }

    #[test]
    fn a_nesting_violation_renames_its_blame() {
        let rename = |id: &str| format!("<{id}>");
        assert_eq!(
            NestingViolation::Cycle("x".into()).rename(rename),
            NestingViolation::Cycle("<x>".into())
        );
        assert_eq!(
            NestingViolation::TooDeep("y".into()).rename(rename),
            NestingViolation::TooDeep("<y>".into())
        );
    }

    #[test]
    fn minted_group_ids_are_canonical_and_fresh() {
        let first = mint_group_id();
        assert!(is_group_id(&first), "{first}");
        assert_ne!(first, mint_group_id());
    }
}
