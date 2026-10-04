use std::sync::atomic::AtomicUsize;

use super::*;

pub(crate) fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("taguru-registry-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

/// The stem (id) of the `context` named `name` in `dir`, read back
/// from the sidecar metas on disk — for assertions that outlive the
/// `AppState` that created the context (a reboot boundary), where
/// `AppState::stem_of` has nothing to answer from. Panics on a name
/// no sidecar records.
pub(crate) fn stem_on_disk(dir: &Path, name: &str) -> String {
    for dir_entry in fs::read_dir(dir).expect("data dir must be listable") {
        let path = dir_entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(stem) = file_name.strip_suffix(".meta.json") else {
            continue;
        };
        let Ok(bytes) = fs::read(&path) else { continue };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        if value.get("name").and_then(|v| v.as_str()) == Some(name) {
            return stem.to_string();
        }
    }
    panic!("no sidecar in {} records the name '{name}'", dir.display());
}

pub(crate) fn loaded_map(state: &AppState) -> HashMap<String, bool> {
    state
        .directory()
        .into_iter()
        .map(|entry| (entry.name, entry.loaded))
        .collect()
}

/// The rendered /metrics body — the public read surface every
/// counter assertion goes through.
pub(crate) fn rendered(state: &AppState) -> String {
    state.metrics().render_prometheus(&state.gauge_snapshot())
}

pub(crate) fn assoc_op(
    subject: &str,
    label: &str,
    object: &str,
    weight: f64,
    source: Option<&str>,
) -> AssocOp {
    AssocOp {
        subject: subject.to_string(),
        label: label.to_string(),
        object: object.to_string(),
        weight,
        source: source.map(String::from),
        paragraph: None,
    }
}

/// Wraps a plain source→text map as submissions — the shape almost
/// every passage test wants.
pub(crate) fn plain(
    passages: BTreeMap<String, String>,
) -> BTreeMap<String, crate::passages::PassageSubmission> {
    passages
        .into_iter()
        .map(|(source, text)| (source, crate::passages::PassageSubmission::plain(text)))
        .collect()
}

pub(crate) fn boot_for_passage_embedding(
    dir: &Path,
    embedder: Arc<dyn EmbeddingProvider>,
    limit: usize,
) -> AppState {
    AppState::boot_with(
        dir.to_path_buf(),
        usize::MAX,
        Some(embedder),
        BootOptions {
            embed_passages: true,
            passage_vector_limit: limit,
            ..BootOptions::default()
        },
    )
    .unwrap()
}

/// Deterministic provider: a text starting with a mapped key gets
/// that key's unit vector (glosses start with their name), anything
/// else lands on an axis orthogonal to all of them. Counts provider
/// round trips so cache behavior is observable.
pub(crate) struct MockEmbeddings {
    pub(crate) keys: Vec<(String, Vec<f32>)>,
    pub(crate) calls: Arc<AtomicUsize>,
}

impl MockEmbeddings {
    pub(crate) fn fruity(calls: &Arc<AtomicUsize>) -> Self {
        Self {
            keys: vec![
                ("りんご".to_string(), vec![1.0, 0.0, 0.0]),
                ("アップル".to_string(), vec![0.96, 0.28, 0.0]),
                ("みかん".to_string(), vec![0.28, 0.96, 0.0]),
            ],
            calls: Arc::clone(calls),
        }
    }
}

impl EmbeddingProvider for MockEmbeddings {
    fn model(&self) -> &str {
        "mock"
    }

    fn embed(
        &self,
        texts: &[&str],
        _purpose: EmbedPurpose,
        _deadline: Deadline,
    ) -> Result<Vec<Vec<f32>>, String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(texts
            .iter()
            .map(|text| {
                self.keys
                    .iter()
                    .find(|(key, _)| text.starts_with(key.as_str()))
                    .map(|(_, vector)| vector.clone())
                    .unwrap_or_else(|| vec![0.0, 0.0, 1.0])
            })
            .collect())
    }
}

/// A provider whose `embed` call blocks for 150ms while tracking how
/// many calls are in flight at once (and the peak concurrency
/// observed) — long enough that concurrent calls MUST overlap unless
/// something serializes or gates them. Shared by the refresh
/// concurrency tests below.
pub(crate) struct SlowEmbeddings {
    pub(crate) in_flight: Arc<AtomicUsize>,
    pub(crate) peak: Arc<AtomicUsize>,
}

impl EmbeddingProvider for SlowEmbeddings {
    fn model(&self) -> &str {
        "slow"
    }

    fn embed(
        &self,
        texts: &[&str],
        _purpose: EmbedPurpose,
        _deadline: Deadline,
    ) -> Result<Vec<Vec<f32>>, String> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(150));
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
    }
}

/// A fixed context id for `name` — for tests that must declare an
/// id-keyed setting (`TAGURU_CONTEXT_QUOTAS`) at boot, before the
/// context exists, and then create the context under that id with
/// `create_if_absent`.
pub(crate) fn test_id(name: &str) -> String {
    crate::registry::derived_context_id(name)
}
