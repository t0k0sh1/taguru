//! Group records and their file I/O: one group is one `{stem}.group`
//! file in the data directory, holding a routing description, the
//! member context names, and the child group names (nesting). Same
//! philosophy as contexts — file existence
//! IS entity existence, no central manifest, discovered by the boot
//! scan — and the same name→stem percent-encoding, so any group name a
//! URL can carry stays inside the data directory.
//!
//! The extension is a SINGLE dot segment on purpose: the stem encoding
//! never produces a literal `.`, and `Path::extension()` cuts only the
//! last dot — a `{stem}.group.json` would answer `Some("json")` and the
//! scan would never see it. (The file's content is JSON regardless; the
//! extension names the entity, not the format, exactly as `.ctx` does.)
//!
//! Records are tiny (a description and a name list), so the whole
//! collection lives resident behind one lock in the registry; this
//! module owns only the record shape and the disk round trip.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::registry::{scanned_stem_and_name, write_atomic};

/// The nesting ceiling: a chain of nested groups may stack at most
/// this many groups (a root, a child, a grandchild). Deep taxonomies
/// are filing, not addressing — retrieval wants short paths — and the
/// cap keeps every nesting walk in the codebase trivially bounded.
/// One constant to raise if that judgement changes.
pub(crate) const MAX_GROUP_DEPTH: usize = 3;

/// One group: the prose half of the grouping (same routing role as a
/// context's description) plus the member context names and the child
/// group names. Sorted sets so membership is deduplicated and every
/// listing is deterministic.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GroupRecord {
    pub description: String,
    pub contexts: BTreeSet<String>,
    /// Child group names — nesting, at most [`MAX_GROUP_DEPTH`] groups
    /// tall and never cyclic ([`validate_nesting`] guards both). A
    /// child may sit under several parents, exactly as a context may:
    /// the shape is a shallow DAG, not a tree. The struct-level
    /// `serde(default)` keeps every pre-nesting group file loading
    /// unchanged.
    pub groups: BTreeSet<String>,
}

/// Why a proposed nesting cannot stand. Carries the group name the
/// walk was at when the violation surfaced — deterministic, because
/// the map and each child set iterate in name order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NestingViolation {
    /// The named group reaches itself through its children.
    Cycle(String),
    /// The named group sits on a chain of more than
    /// [`MAX_GROUP_DEPTH`] groups. A cycle longer than the cap also
    /// lands here — its chains are over the cap either way, and the
    /// walk refuses before following a path far enough to see the
    /// loop close.
    TooDeep(String),
}

/// Checks the whole collection for the two nesting invariants — no
/// cycles, no chain of more than [`MAX_GROUP_DEPTH`] groups — in one
/// memoized walk, O(groups + edges). Callers hand it the PROSPECTIVE
/// map (under the groups write lock, before anything persists) or, at
/// boot, whatever the files claim. A child name with no record counts
/// as depth zero: dangling references are referential integrity's
/// business, healed elsewhere.
pub(crate) fn validate_nesting(
    groups: &BTreeMap<String, GroupRecord>,
) -> Result<(), NestingViolation> {
    let mut settled = BTreeMap::new();
    let mut visiting = Vec::new();
    for name in groups.keys() {
        chain_depth(groups, name, &mut settled, &mut visiting)?;
    }
    Ok(())
}

/// The number of groups on the longest chain hanging from `name`,
/// itself included — depth-first, with a visiting stack for cycle
/// detection and a settled map so shared children are walked once.
/// Recursion is bounded by the cap, not the input: a path already
/// [`MAX_GROUP_DEPTH`] groups long refuses before descending further,
/// so even a hand-written thousand-group chain cannot blow the stack.
fn chain_depth<'a>(
    groups: &'a BTreeMap<String, GroupRecord>,
    name: &'a str,
    settled: &mut BTreeMap<&'a str, usize>,
    visiting: &mut Vec<&'a str>,
) -> Result<usize, NestingViolation> {
    if let Some(&depth) = settled.get(name) {
        return Ok(depth);
    }
    let Some(record) = groups.get(name) else {
        return Ok(0);
    };
    if visiting.contains(&name) {
        return Err(NestingViolation::Cycle(name.to_string()));
    }
    if visiting.len() >= MAX_GROUP_DEPTH {
        // The path down to here already stacks MAX + 1 groups counting
        // this one; whatever hangs below cannot make that legal.
        return Err(NestingViolation::TooDeep(visiting[0].to_string()));
    }
    visiting.push(name);
    let mut below = 0;
    for child in &record.groups {
        below = below.max(chain_depth(groups, child, settled, visiting)?);
    }
    visiting.pop();
    let depth = below + 1;
    if depth > MAX_GROUP_DEPTH {
        return Err(NestingViolation::TooDeep(name.to_string()));
    }
    settled.insert(name, depth);
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
pub(crate) fn repair_nesting(groups: &mut BTreeMap<String, GroupRecord>) {
    if validate_nesting(groups).is_ok() {
        return;
    }
    let edges: Vec<(String, String)> = groups
        .iter()
        .flat_map(|(parent, record)| {
            record
                .groups
                .iter()
                .map(|child| (parent.clone(), child.clone()))
        })
        .collect();
    for record in groups.values_mut() {
        record.groups.clear();
    }
    for (parent, child) in edges {
        groups
            .get_mut(&parent)
            .unwrap()
            .groups
            .insert(child.clone());
        if validate_nesting(groups).is_err() {
            groups.get_mut(&parent).unwrap().groups.remove(&child);
            tracing::warn!(
                group = %parent,
                child = %child,
                "dropped a nesting edge: it would close a cycle or stack more than {MAX_GROUP_DEPTH} groups"
            );
        }
    }
}

/// Every context reachable from the named roots — direct members plus
/// everything nested children bundle, transitively. The scoped write
/// gate judges a group by this closure: a grant must cover what the
/// group ADDRESSES, not just what it lists directly. Names without a
/// record contribute nothing; the seen set keeps even a (transient,
/// mid-repair) cycle from looping the walk.
pub(crate) fn context_closure<'map, 'roots: 'map>(
    groups: &'map BTreeMap<String, GroupRecord>,
    roots: impl IntoIterator<Item = &'roots str>,
) -> BTreeSet<String> {
    let mut contexts = BTreeSet::new();
    let mut seen: BTreeSet<&'map str> = BTreeSet::new();
    let mut frontier: Vec<&'map str> = roots
        .into_iter()
        .map(|root| -> &'map str { root })
        .filter(|root| seen.insert(*root))
        .collect();
    while let Some(name) = frontier.pop() {
        let Some(record) = groups.get(name) else {
            continue;
        };
        contexts.extend(record.contexts.iter().cloned());
        frontier.extend(
            record
                .groups
                .iter()
                .map(String::as_str)
                .filter(|child| seen.insert(*child)),
        );
    }
    contexts
}

pub(crate) fn group_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.group"))
}

/// Persists one group via the registry's staged write (fsync + rename +
/// parent fsync): a crash mid-write leaves the previous version intact.
/// The staging name is `{stem}.tmp{n}`, which the boot scan's leftover
/// sweep already removes — nothing group-specific to clean up.
pub(crate) fn write_group(dir: &Path, stem: &str, record: &GroupRecord) -> io::Result<()> {
    write_atomic(&group_path(dir, stem), &serde_json::to_vec_pretty(record)?)
}

/// Unlinks one group file. A file already gone counts as success — the
/// caller's intent (this group does not exist on disk) is satisfied.
pub(crate) fn remove_group_file(dir: &Path, stem: &str) -> io::Result<()> {
    match fs::remove_file(group_path(dir, stem)) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

/// One boot-time pass for groups, run after the context scan (which
/// also sweeps staging leftovers). An unreadable or corrupt file keeps
/// its name and loses its content: the entity must not vanish because
/// one read failed. Louder than a context's sidecar read, which treats
/// absence as the normal never-written case — here the directory scan
/// just listed the file, so a failed read is always news.
pub(crate) fn scan_groups(dir: &Path) -> io::Result<BTreeMap<String, GroupRecord>> {
    let mut groups = BTreeMap::new();
    for dir_entry in fs::read_dir(dir)? {
        let path = dir_entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("group") {
            continue;
        }
        let Some((_, name)) = scanned_stem_and_name(&path) else {
            continue;
        };
        let record = match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|error| {
                tracing::warn!("group '{name}' has a corrupt file, keeping it empty: {error}");
                GroupRecord::default()
            }),
            Err(error) => {
                tracing::warn!("group '{name}' could not be read, keeping it empty: {error}");
                GroupRecord::default()
            }
        };
        groups.insert(name, record);
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(&str, &[&str], &[&str])]) -> BTreeMap<String, GroupRecord> {
        entries
            .iter()
            .map(|(name, contexts, children)| {
                (
                    name.to_string(),
                    GroupRecord {
                        description: String::new(),
                        contexts: contexts.iter().map(|c| c.to_string()).collect(),
                        groups: children.iter().map(|g| g.to_string()).collect(),
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
        let children = |name: &str| groups[name].groups.iter().cloned().collect::<Vec<_>>();
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
}
