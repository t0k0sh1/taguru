//! Group records and their file I/O: one group is one `{stem}.group`
//! file in the data directory, holding a routing description and the
//! member context names. Same philosophy as contexts — file existence
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

/// One group: the prose half of the grouping (same routing role as a
/// context's description) plus the member context names. Sorted set so
/// membership is deduplicated and every listing is deterministic.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GroupRecord {
    pub description: String,
    pub contexts: BTreeSet<String>,
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
