//! Disk-backed context registry: the server-side lifecycle around the
//! library's `Context`. Disk is the source of truth — every context is
//! one image file (`{name}.ctx`, the bytes of `Context::to_bytes`) plus
//! a sidecar `{name}.meta.json` holding the routing description, the
//! cache policy flag, and a stats snapshot. The sidecar stays outside
//! the image on purpose: the image format remains a pure dump of the
//! network, and server metadata can evolve without bumping it.
//!
//! Locking contract: the registry lock guards only the name → entry map
//! and is held just long enough to look up, insert, or remove; every
//! context sits behind its own entry lock. A caller clones the entry's
//! `Arc` and releases the registry immediately, so a slow operation on
//! one context never blocks the others — and a panic poisons only the
//! context it happened in.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use associative_rag::context::Context;
use serde::{Deserialize, Serialize};

/// Server-side metadata for one context: the prose half of the routing
/// directory plus the cache policy flag.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextMeta {
    /// What this 文脈 covers, written by whoever creates the context
    /// (typically the ingesting LLM). Routing quality depends on it, so
    /// the directory serves it next to stats that cannot go stale.
    pub description: String,
    /// Pinned contexts are kept resident in memory regardless of cache
    /// pressure — for small, always-hot contexts like glossaries.
    pub pinned: bool,
}

/// Mechanically derived "what is this context about" numbers. Served by
/// the routing directory so an LLM can pick a context without anything
/// being loaded or scanned per request; unlike the prose description,
/// these are recomputed from the network itself and cannot drift.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextStats {
    pub associations: usize,
    pub concepts: usize,
    pub labels: usize,
    pub sources: usize,
    pub footprint_bytes: usize,
    /// Most connected concepts with their degree, most connected first.
    pub top_concepts: Vec<(String, usize)>,
    /// The first labels of the relation vocabulary (capped; the full
    /// list is at `GET /contexts/{name}/labels`).
    pub label_sample: Vec<String>,
}

impl ContextStats {
    const TOP_CONCEPTS: usize = 10;
    const LABEL_SAMPLE: usize = 50;

    fn of(context: &Context) -> Self {
        Self {
            associations: context.association_count(),
            concepts: context.concept_count(),
            labels: context.label_count(),
            sources: context.source_count(),
            footprint_bytes: context.footprint(),
            top_concepts: context
                .top_concepts(Self::TOP_CONCEPTS)
                .into_iter()
                .map(|(name, degree)| (name.to_string(), degree))
                .collect(),
            label_sample: context
                .labels()
                .into_iter()
                .take(Self::LABEL_SAMPLE)
                .map(String::from)
                .collect(),
        }
    }
}

/// What `{name}.meta.json` holds: the meta inline plus the stats
/// snapshot as of the last save, so a directory listing can describe a
/// context without touching its image.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct MetaFile {
    #[serde(flatten)]
    meta: ContextMeta,
    stats: ContextStats,
}

/// One row of `GET /contexts` — the routing directory an LLM client
/// reads to decide which context to search, skills-style: a name, the
/// prose description, and the mechanical stats that keep it honest.
#[derive(Debug, Clone, Serialize)]
pub struct DirectoryEntry {
    pub name: String,
    pub description: String,
    pub pinned: bool,
    pub stats: ContextStats,
}

pub struct Entry {
    inner: RwLock<EntryInner>,
}

struct EntryInner {
    meta: ContextMeta,
    stats: ContextStats,
    context: Context,
}

pub enum CreateError {
    AlreadyExists,
    Io(io::Error),
}

/// Shared server state: the data directory and the context registry.
#[derive(Clone)]
pub struct AppState(Arc<StateInner>);

struct StateInner {
    data_dir: PathBuf,
    registry: RwLock<HashMap<String, Arc<Entry>>>,
}

impl AppState {
    /// Opens (creating if needed) the data directory and loads every
    /// context image found in it. A file that fails to decode is skipped
    /// with a warning rather than taking the whole server down — the
    /// image stays on disk for inspection.
    pub fn boot(data_dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&data_dir)?;
        let mut registry = HashMap::new();
        for dir_entry in fs::read_dir(&data_dir)? {
            let path = dir_entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("ctx") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(name) = name_from_stem(stem) else {
                eprintln!("skipping {}: file name does not decode", path.display());
                continue;
            };
            let context = match fs::read(&path)
                .and_then(|bytes| Context::from_bytes(&bytes).map_err(io::Error::other))
            {
                Ok(context) => context,
                Err(error) => {
                    eprintln!("skipping context '{name}': {error}");
                    continue;
                }
            };
            let meta = read_meta_file(&data_dir, stem).meta;
            let stats = ContextStats::of(&context);
            registry.insert(
                name,
                Arc::new(Entry {
                    inner: RwLock::new(EntryInner {
                        meta,
                        stats,
                        context,
                    }),
                }),
            );
        }
        Ok(Self(Arc::new(StateInner {
            data_dir,
            registry: RwLock::new(registry),
        })))
    }

    pub fn context_count(&self) -> usize {
        self.0.registry.read().unwrap().len()
    }

    /// Registers an empty context and persists it immediately, so its
    /// existence (and description) survives a crash from the moment the
    /// create call returns. A persistence failure fails the create.
    pub fn create(&self, name: &str, meta: ContextMeta) -> Result<(), CreateError> {
        let mut registry = self.0.registry.write().unwrap();
        if registry.contains_key(name) {
            return Err(CreateError::AlreadyExists);
        }
        let inner = EntryInner {
            meta,
            stats: ContextStats::default(),
            context: Context::default(),
        };
        save_files(&self.0.data_dir, name, &inner).map_err(CreateError::Io)?;
        registry.insert(
            name.to_string(),
            Arc::new(Entry {
                inner: RwLock::new(inner),
            }),
        );
        Ok(())
    }

    /// Removes a context from the registry and deletes its files. Waits
    /// for any in-flight operation on the entry (its lock is taken after
    /// removal), so a concurrent write cannot recreate the files.
    pub fn delete(&self, name: &str) -> Option<io::Result<()>> {
        let entry = self.0.registry.write().unwrap().remove(name)?;
        let _in_flight = entry.inner.write().unwrap();
        let stem = file_stem(name);
        let mut outcome = Ok(());
        for file in [format!("{stem}.ctx"), format!("{stem}.meta.json")] {
            if let Err(error) = fs::remove_file(self.0.data_dir.join(file))
                && error.kind() != io::ErrorKind::NotFound
            {
                outcome = Err(error);
            }
        }
        Some(outcome)
    }

    /// Updates the description and/or pin flag, persisting the sidecar.
    pub fn update_meta(
        &self,
        name: &str,
        description: Option<String>,
        pinned: Option<bool>,
    ) -> Option<io::Result<ContextMeta>> {
        let entry = self.0.registry.read().unwrap().get(name).cloned()?;
        let mut inner = entry.inner.write().unwrap();
        if let Some(description) = description {
            inner.meta.description = description;
        }
        if let Some(pinned) = pinned {
            inner.meta.pinned = pinned;
        }
        let outcome = save_meta_file(&self.0.data_dir, name, &inner).map(|()| inner.meta.clone());
        Some(outcome)
    }

    /// The routing directory: every context's name, description, policy,
    /// and stats snapshot, in name order.
    pub fn directory(&self) -> Vec<DirectoryEntry> {
        let entries: Vec<(String, Arc<Entry>)> = self
            .0
            .registry
            .read()
            .unwrap()
            .iter()
            .map(|(name, entry)| (name.clone(), Arc::clone(entry)))
            .collect();
        let mut directory: Vec<DirectoryEntry> = entries
            .into_iter()
            .map(|(name, entry)| {
                let inner = entry.inner.read().unwrap();
                DirectoryEntry {
                    name,
                    description: inner.meta.description.clone(),
                    pinned: inner.meta.pinned,
                    stats: inner.stats.clone(),
                }
            })
            .collect();
        directory.sort_by(|a, b| a.name.cmp(&b.name));
        directory
    }

    /// Runs a read-only operation on one context, or `None` for an
    /// unknown name.
    pub fn read_context<T>(&self, name: &str, operate: impl FnOnce(&Context) -> T) -> Option<T> {
        let entry = self.0.registry.read().unwrap().get(name).cloned()?;
        let inner = entry.inner.read().unwrap();
        Some(operate(&inner.context))
    }

    /// Runs a mutating operation on one context, then refreshes its
    /// stats and persists image and sidecar. The operation's result is
    /// returned alongside the persistence outcome: on a persistence
    /// failure the write is applied in memory but not yet durable, and
    /// the caller must say so.
    pub fn write_context<T>(
        &self,
        name: &str,
        operate: impl FnOnce(&mut Context) -> T,
    ) -> Option<(T, io::Result<()>)> {
        let entry = self.0.registry.read().unwrap().get(name).cloned()?;
        let mut inner = entry.inner.write().unwrap();
        let result = operate(&mut inner.context);
        inner.stats = ContextStats::of(&inner.context);
        let persisted = save_files(&self.0.data_dir, name, &inner);
        Some((result, persisted))
    }
}

fn image_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.ctx"))
}

fn meta_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.meta.json"))
}

fn save_files(dir: &Path, name: &str, inner: &EntryInner) -> io::Result<()> {
    let stem = file_stem(name);
    write_atomic(&image_path(dir, &stem), &inner.context.to_bytes())?;
    write_meta(dir, &stem, inner)
}

fn save_meta_file(dir: &Path, name: &str, inner: &EntryInner) -> io::Result<()> {
    write_meta(dir, &file_stem(name), inner)
}

fn write_meta(dir: &Path, stem: &str, inner: &EntryInner) -> io::Result<()> {
    let file = MetaFile {
        meta: inner.meta.clone(),
        stats: inner.stats.clone(),
    };
    write_atomic(&meta_path(dir, stem), &serde_json::to_vec_pretty(&file)?)
}

/// Reads the sidecar, falling back to defaults on any problem — a
/// missing or corrupt sidecar must not make the image unreachable.
fn read_meta_file(dir: &Path, stem: &str) -> MetaFile {
    match fs::read(meta_path(dir, stem)) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            eprintln!("ignoring corrupt sidecar for '{stem}': {error}");
            MetaFile::default()
        }),
        Err(_) => MetaFile::default(),
    }
}

/// Writes via a temporary file and rename, so a crash mid-write leaves
/// the previous version intact instead of a torn file.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

/// Encodes a context name as a file stem: bytes outside [A-Za-z0-9_-]
/// become %XX. Context names arrive from URL paths and may contain path
/// separators or dots; encoding them keeps every name inside the data
/// directory (no traversal) and reversible.
fn file_stem(name: &str) -> String {
    let mut stem = String::new();
    for byte in name.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' => stem.push(byte as char),
            _ => stem.push_str(&format!("%{byte:02X}")),
        }
    }
    stem
}

/// Decodes [`file_stem`]'s encoding back into a context name.
fn name_from_stem(stem: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(stem.len());
    let mut cursor = stem.bytes();
    while let Some(byte) = cursor.next() {
        if byte == b'%' {
            let high = cursor.next()?;
            let low = cursor.next()?;
            let hex = [high, low];
            let hex = std::str::from_utf8(&hex).ok()?;
            bytes.push(u8::from_str_radix(hex, 16).ok()?);
        } else {
            bytes.push(byte);
        }
    }
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_stem_roundtrips_any_name() {
        for name in [
            "sake",
            "用語集",
            "a/b\\c..d",
            "MiXed-123_ok",
            "%weird%",
            "空白 と 記号!?",
        ] {
            let stem = file_stem(name);
            assert!(
                stem.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'%'),
                "stem '{stem}' carries raw special bytes"
            );
            assert_eq!(name_from_stem(&stem).as_deref(), Some(name));
        }
    }

    #[test]
    fn name_from_stem_rejects_torn_encodings() {
        assert_eq!(name_from_stem("%"), None);
        assert_eq!(name_from_stem("%4"), None);
        assert_eq!(name_from_stem("%zz"), None);
        // Undecodable UTF-8 is refused rather than lossily replaced.
        assert_eq!(name_from_stem("%FF%FE"), None);
    }
}
