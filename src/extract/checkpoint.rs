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
    /// TAGURU_EXTRACT_RUNAWAY_RATIO as [`runaway_manifest_value`]
    /// encodes it (`""` = default) — ADR 0035. Same `default`
    /// reasoning: a checkpoint file predating this field was written
    /// under the then-only (unjudged) ladder, and still matches a
    /// default rerun; only a non-default ratio invalidates it.
    #[serde(default)]
    pub(super) runaway_ratio: String,
    /// `--chunk-bytes` as [`chunk_bytes_manifest_value`] encodes it
    /// (`""` = default) — ADR 0020. A different cap re-cuts every
    /// chunk, and each cached unit is keyed by its chunk's content
    /// hash anyway; the field makes the mismatch explicit rather than
    /// relying on every hash missing.
    #[serde(default)]
    pub(super) chunk_bytes: String,
    /// `--chunk-context` as [`ChunkContextMode::manifest_value`]
    /// encodes it (`""` = off) — ADR 0033. Same `default` reasoning as
    /// `chunk_bytes`: a unit's prompt changes with the mode, so a
    /// checkpoint written under another mode is not consulted.
    #[serde(default)]
    pub(super) chunk_context: String,
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
    /// `--redact`'s version (`""` = off) — ADR 0038 §3.5; `default`
    /// for the same reason as `candidates`: a pre-ADR checkpoint was
    /// written with redaction off.
    #[serde(default)]
    pub(super) redaction: String,
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
    /// ADR 0023 §3.5: the completion that produced this unit, so a
    /// reused unit still names its origin. `default` because a
    /// pre-0023 checkpoint has none to name — the trace then says
    /// `null`, never a guess.
    #[serde(default)]
    pub(super) attempt: Option<AttemptRef>,
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
    pub(super) removed: Vec<Removal>,
    /// ADR 0024 §3.6: lossy mode's parse-time drops; `default` for
    /// checkpoints written before the field existed.
    #[serde(default)]
    pub(super) unparsed: Vec<Removal>,
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
    /// ADR 0033 §3.5: the overview pass's answer per chunk (by
    /// `chunk_sha256`), so a resumed document reuses the pass without
    /// a call — an EMPTY answer for a chunk whose ask failed (ADR 0034
    /// §3.3: fixed for the checkpoint's life, never re-asked). Empty
    /// below `--chunk-context overview`.
    #[serde(default)]
    pub(super) overview: BTreeMap<String, OverviewAnswer>,
    /// The digest of the merged overview the cached `units` were
    /// extracted under: every block depends on it, so a unit is only
    /// reused when the overview it saw is the one in force
    /// (`CheckpointStore::bind_overview`). `""` below `overview`.
    #[serde(default)]
    pub(super) overview_digest: String,
}

/// ADR 0037 §3.4: the line for a checkpoint whose fingerprint is not
/// this run's — only when it holds units, since discarding nothing
/// costs nothing (an overview-only or empty file changes settings
/// silently, as a first run would).
pub(super) fn stale_checkpoint_notice(path: &Path, units: usize) -> Option<String> {
    (units > 0).then(|| {
        format!(
            "taguru: extract: checkpoint at {} was written under different settings — \
             {units} unit(s) re-extract",
            path.display()
        )
    })
}

impl DocumentCheckpoints {
    /// Missing, unreadable, or fingerprint-mismatched checkpoints all
    /// degrade to "nothing cached" — never an error, and never a false
    /// reuse of an incompatible output. Mirrors [`Manifest::load`]'s
    /// posture exactly.
    pub(super) fn load(path: &Path, fingerprint: &CheckpointFingerprint) -> Self {
        // ADR 0037 §3.4 (#850): each of the three ways to "nothing
        // cached" is told apart on stderr where it costs something —
        // a damaged file and a settings change both re-bill every
        // unit of the document, and were indistinguishable from a
        // first run. A missing file IS the first run, and stays quiet.
        let loaded: Option<Self> = match fs::read(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                eprintln!(
                    "taguru: extract: ignoring an unreadable checkpoint at {}: {error} — \
                     every unit of this document re-extracts",
                    path.display()
                );
                None
            }
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(checkpoints) => Some(checkpoints),
                Err(error) => {
                    eprintln!(
                        "taguru: extract: ignoring an unreadable checkpoint at {}: {error} — \
                         every unit of this document re-extracts",
                        path.display()
                    );
                    None
                }
            },
        };
        match loaded {
            Some(checkpoints) if checkpoints.fingerprint == *fingerprint => checkpoints,
            Some(stale) => {
                if let Some(line) = stale_checkpoint_notice(path, stale.units.len()) {
                    eprintln!("{line}");
                }
                Self::fresh(fingerprint)
            }
            None => Self::fresh(fingerprint),
        }
    }

    fn fresh(fingerprint: &CheckpointFingerprint) -> Self {
        Self {
            fingerprint: fingerprint.clone(),
            units: BTreeMap::new(),
            overview: BTreeMap::new(),
            overview_digest: String::new(),
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
                overview: BTreeMap::new(),
                overview_digest: String::new(),
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

    /// ADR 0033 §3.5: the overview answer cached for a chunk, if any.
    pub(super) fn overview_answer(&self, chunk_sha256: &str) -> Option<OverviewAnswer> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.overview.get(chunk_sha256).cloned()
    }

    /// Records one chunk's fresh overview answer, durably — the same
    /// posture as [`CheckpointStore::record`].
    pub(super) fn record_overview(
        &self,
        source: &str,
        chunk_sha256: String,
        answer: OverviewAnswer,
    ) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.overview.insert(chunk_sha256, answer);
        if let Err(error) = state.save(&self.path) {
            eprintln!(
                "taguru: extract: {source}: saving checkpoint {}: {error} — the overview \
                 still counts this run; a resume may repeat it",
                self.path.display()
            );
        }
    }

    /// Binds the cached extraction units to the overview in force:
    /// units extracted under a different overview digest saw
    /// different blocks and are discarded (never reused against a
    /// prompt they did not answer). Returns how many were discarded.
    pub(super) fn bind_overview(&self, source: &str, digest: &str) -> usize {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.overview_digest == digest {
            return 0;
        }
        let discarded = state.units.len();
        state.units.clear();
        state.overview_digest = digest.to_string();
        if let Err(error) = state.save(&self.path) {
            eprintln!(
                "taguru: extract: {source}: saving checkpoint {}: {error}",
                self.path.display()
            );
        }
        discarded
    }

    /// How many extracted units the store holds — what a failure
    /// line tells the operator a plain rerun resumes from (#763).
    pub(super) fn unit_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .units
            .len()
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
