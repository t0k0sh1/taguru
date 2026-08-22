//! Per-chunk checkpointing: resuming an interrupted document without
//! re-asking the model for units already answered.

use super::*;

/// The same compatibility inputs [`ManifestEntry`]/[`Manifest::matches`]
/// already check, minus `output` — the per-document gate deciding
/// whether a checkpoint file's cached units are even consulted. Any
/// field mismatch (content edited, model/prompt/`--questions`/
/// `--fact-budget`/`--structured-output`/`--max-output-tokens`/`--lossy`
/// changed) treats the whole file as absent, so a settings change can
/// never silently reuse an incompatible output — the same posture
/// [`Manifest::load`] already takes for an unreadable manifest.
#[derive(Clone, PartialEq, serde::Serialize, Deserialize)]
pub(super) struct CheckpointFingerprint {
    pub(super) sha256: String,
    pub(super) model: String,
    pub(super) prompt_version: u32,
    pub(super) context: String,
    pub(super) questions_n: usize,
    pub(super) no_passage: bool,
    pub(super) description: String,
    pub(super) fact_budget: usize,
    pub(super) structured_output: String,
    pub(super) max_output_tokens: usize,
    /// TAGURU_EXTRACT_ESCALATION_FACTOR as [`escalation_manifest_value`]
    /// encodes it (`""` = default or no budget) — ADR 0019. Same
    /// `default` reasoning as `schema_digest` below: a checkpoint file
    /// predating this field was written under the then-only (uncapped)
    /// rung, and still matches a default rerun; only a non-default
    /// factor invalidates it.
    #[serde(default)]
    pub(super) escalation_factor: String,
    /// `--chunk-bytes` as [`chunk_bytes_manifest_value`] encodes it
    /// (`""` = default) — ADR 0020. A different cap re-cuts every
    /// chunk, and each cached unit is keyed by its chunk's content
    /// hash anyway; the field makes the mismatch explicit rather than
    /// relying on every hash missing.
    #[serde(default)]
    pub(super) chunk_bytes: String,
    pub(super) lossy: bool,
    /// `--schema`'s document digest (`""` = no schema). Same default
    /// as [`ManifestEntry::schema_digest`] and the same reasoning: a
    /// checkpoint file predating this field was necessarily written
    /// under no schema, so it still matches a schema-less rerun and
    /// only invalidates once `--schema` is actually engaged.
    #[serde(default)]
    pub(super) schema_digest: String,
    /// `--candidates`' algorithm fingerprint (`""` = off) — ADR 0014.
    /// Same `default` reasoning as `schema_digest`: a pre-S2 checkpoint
    /// was necessarily written with candidates off, so it still matches
    /// a default rerun and only invalidates once `--candidates` engages.
    #[serde(default)]
    pub(super) candidates: String,
    /// `--vocabulary`'s content digest (`""` = off) — ADR 0015, same
    /// `default` reasoning as `schema_digest`/`candidates`.
    #[serde(default)]
    pub(super) vocabulary_digest: String,
}

/// One durable unit of extraction work: a top-level chunk, or (issue
/// #179's amendment to ADR 0001 §7's split rung) one of the smaller
/// sub-pieces a length-limited chunk was split into. Keyed by the
/// unit's OWN content hash rather than `chunk_index` alone, so a
/// resumed run that splits differently than a prior one never
/// misattributes a sub-piece's output to the wrong text — see
/// `extract_piece`'s reuse guard.
#[derive(Clone, serde::Serialize, Deserialize)]
pub(super) struct CheckpointUnit {
    /// The ORIGINAL chunk's coordinates, kept only for the same "chunk
    /// i/n" reporting `ChunkOutput::chunk_index` already carries.
    pub(super) chunk_index: usize,
    pub(super) output: ModelOutput,
    /// The exact user turn and the model's own final answer text for
    /// this unit — needed so a reused unit can still participate in
    /// Stage 2 cross-chunk correction exactly like a freshly-extracted
    /// one (`correct_cross_output_issues` rebuilds a chunk's own
    /// conversation from these).
    pub(super) user: String,
    pub(super) answer: String,
    /// ADR 0013's mechanical-removal records for this unit, so a
    /// resumed document still reports every removal its reused units
    /// carried. `default` because a pre-0013 checkpoint file simply
    /// had no removals to record — its units validated fully.
    #[serde(default)]
    pub(super) removed: Vec<String>,
}

/// One document's durable checkpoint state: the settings it was
/// extracted under, and every unit completed so far, keyed by content
/// hash. Persisted as one small JSON file, rewritten atomically
/// (`storage::write_atomic`) after every new unit lands — small enough
/// that a read-modify-write-whole-file is simpler and just as durable
/// as an append-only log, the same shape `Manifest` itself already
/// uses for a much larger (per-run, not per-document) equivalent.
#[derive(serde::Serialize, Deserialize)]
pub(super) struct DocumentCheckpoints {
    pub(super) fingerprint: CheckpointFingerprint,
    #[serde(default)]
    pub(super) units: BTreeMap<String, CheckpointUnit>,
}

impl DocumentCheckpoints {
    /// Missing, unreadable, or fingerprint-mismatched checkpoints all
    /// degrade to "nothing cached" — never an error, and never a false
    /// reuse of an incompatible output. Mirrors [`Manifest::load`]'s
    /// posture exactly.
    pub(super) fn load(path: &Path, fingerprint: &CheckpointFingerprint) -> Self {
        let loaded: Option<Self> = fs::read(path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok());
        match loaded {
            Some(checkpoints) if checkpoints.fingerprint == *fingerprint => checkpoints,
            _ => Self {
                fingerprint: fingerprint.clone(),
                units: BTreeMap::new(),
            },
        }
    }

    pub(super) fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let text = serde_json::to_string(self).expect("a checkpoint file serializes");
        crate::storage::write_atomic(path, text.as_bytes())
    }
}

/// The thread-safe handle threaded through one document's extraction —
/// `--parallel` fans a document's own chunks out across threads (see
/// `Run::extract_chunks_concurrently`), so lookups and writes need the
/// same "shared, mutex-guarded, poisoning tolerated" treatment
/// `DiagnosticsSink` already gives its writer, not `DocumentCheckpoints`
/// used bare.
pub(super) struct CheckpointStore {
    pub(super) path: PathBuf,
    pub(super) state: Mutex<DocumentCheckpoints>,
}

impl CheckpointStore {
    pub(super) fn load(path: PathBuf, fingerprint: &CheckpointFingerprint) -> Self {
        let state = DocumentCheckpoints::load(&path, fingerprint);
        Self {
            path,
            state: Mutex::new(state),
        }
    }

    /// `--force`'s "start this document over" extended one level
    /// deeper: an empty store regardless of what a prior run's file
    /// holds, so a forced re-extraction never silently reuses cached
    /// units it was explicitly told to redo.
    pub(super) fn empty(path: PathBuf, fingerprint: CheckpointFingerprint) -> Self {
        Self {
            path,
            state: Mutex::new(DocumentCheckpoints {
                fingerprint,
                units: BTreeMap::new(),
            }),
        }
    }

    /// Look up a unit by its own content hash — the answer to issue
    /// #179's amendment: a split sub-piece's hash differs from its
    /// parent's, so this is a correct cache key regardless of how many
    /// times (or how) a piece has been split. Returns an owned clone so
    /// the lock is never held across the caller's own work.
    pub(super) fn lookup(&self, unit_hash: &str) -> Option<CheckpointUnit> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.units.get(unit_hash).cloned()
    }

    /// Records one freshly-extracted unit and durably persists the
    /// whole (small) file before returning, so a kill immediately after
    /// this call still finds the unit on the next run. A save failure
    /// only warns — the unit still counts for THIS run; a resume that
    /// hits the same failure again would simply re-extract it, the same
    /// "the next run re-extracts" posture a failed manifest save takes.
    pub(super) fn record(&self, source: &str, unit_hash: String, unit: CheckpointUnit) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.units.insert(unit_hash, unit);
        if let Err(error) = state.save(&self.path) {
            eprintln!(
                "taguru: extract: {source}: saving checkpoint {}: {error} — the unit still \
                 counts this run; a resume may repeat it",
                self.path.display()
            );
        }
    }

    /// Best-effort delete once a document's batch has durably landed —
    /// the checkpoint's whole purpose (resuming an INCOMPLETE document)
    /// no longer applies, and clearing it keeps `--dry-run`'s reuse
    /// count honest for the next document that reuses this source path.
    /// A failure here is silently ignored, exactly like `Manifest`'s
    /// "the batch is written; the next run just re-extracts" posture:
    /// nothing correctness-critical depends on this file disappearing
    /// promptly.
    pub(super) fn clear(&self) {
        let _ = fs::remove_file(&self.path);
    }
}
