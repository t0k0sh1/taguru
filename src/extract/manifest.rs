//! The extraction manifest: what each batch file was computed from,
//! and whether a document can be skipped as unchanged.

use super::*;

/// What each batch file was computed from. Extraction is the
/// expensive step, so unchanged documents skip; any input to the
/// computation changing — document bytes, model, prompt, target
/// context — re-extracts. The context matters even though the model
/// never sees its name: it is baked into the emitted header, and a
/// skip that kept a stale header would send the batch to the wrong
/// context on import.
#[derive(Default, serde::Serialize, Deserialize)]
pub(super) struct Manifest {
    #[serde(default)]
    pub(super) documents: BTreeMap<String, ManifestEntry>,
}

#[derive(serde::Serialize, Deserialize)]
pub(super) struct ManifestEntry {
    pub(super) sha256: String,
    pub(super) model: String,
    pub(super) prompt_version: u32,
    // Default: entries written before this field existed carry no
    // context, mismatch whatever is asked, and simply re-extract once.
    #[serde(default)]
    pub(super) context: String,
    /// --questions N of the run that wrote this batch: changing N is a
    /// computation-input change like any other and re-extracts (there
    /// is no cheaper questions-only pass — generation rides the same
    /// extraction call on purpose, see the design's trade-off note).
    #[serde(default)]
    pub(super) questions_n: usize,
    /// --no-passage of the run that wrote this batch: it decides
    /// whether the emitted batch carries the source passage at all, so
    /// toggling it must re-extract rather than skip with a batch shaped
    /// for the other setting.
    #[serde(default)]
    pub(super) no_passage: bool,
    /// --description of the run that wrote this batch: baked into the
    /// emitted header like `context`, so a change here must re-extract
    /// too rather than skip and leave the old description in place.
    #[serde(default)]
    pub(super) description: String,
    /// --fact-budget of the run that wrote this batch: folded into the
    /// system prompt like --questions, so changing it is a computation-
    /// input change like any other and re-extracts.
    #[serde(default)]
    pub(super) fact_budget: usize,
    /// --structured-output of the run that wrote this batch — the
    /// REQUESTED mode, never the probe's resolution: which rung
    /// carried a run depends on the backend, but the computation input
    /// is what the operator asked for. Empty = off, so entries written
    /// before this field existed keep matching all-defaults runs
    /// instead of forcing a spurious re-extraction of everything.
    #[serde(default)]
    pub(super) structured_output: String,
    /// --max-output-tokens of the run that wrote this batch (0 = none
    /// sent): an explicit output budget changes what the model can
    /// answer, so changing it is a computation-input change like any
    /// other and re-extracts.
    #[serde(default)]
    pub(super) max_output_tokens: usize,
    /// TAGURU_EXTRACT_ESCALATION_FACTOR of the run that wrote this
    /// batch, as [`escalation_manifest_value`] encodes it (`""` = the
    /// default, or no budget for it to apply to — ADR 0019): the
    /// escalated resend's cap changes what the model can answer on
    /// that rung, so a non-default factor re-extracts like
    /// `max_output_tokens`; the empty default keeps entries written
    /// before the field existed matching an all-defaults run.
    #[serde(default)]
    pub(super) escalation_factor: String,
    /// TAGURU_EXTRACT_RUNAWAY_RATIO of the run that wrote this batch,
    /// as [`runaway_manifest_value`] encodes it (`""` = the default —
    /// ADR 0035): the judgment changes which length-limited answers
    /// the ladder keeps pursuing, so a non-default ratio re-extracts
    /// like `escalation_factor`; the empty default keeps entries
    /// written before the field existed matching an all-defaults run.
    #[serde(default)]
    pub(super) runaway_ratio: String,
    /// --chunk-context of the run that wrote this batch, as
    /// [`ChunkContextMode::manifest_value`] encodes it (`""` = off —
    /// ADR 0033): the mode decides what every chunk is prefixed with,
    /// so any other value re-extracts; the empty default keeps entries
    /// written before the field existed matching an all-defaults run.
    #[serde(default)]
    pub(super) chunk_context: String,
    /// --chunk-bytes of the run that wrote this batch, as
    /// [`chunk_bytes_manifest_value`] encodes it (`""` = the default
    /// cap — ADR 0020): the cap decides what every chunk the model is
    /// shown contains, so a different one re-extracts; the empty
    /// default keeps entries written before the field existed
    /// matching a default run.
    #[serde(default)]
    pub(super) chunk_bytes: String,
    /// --lossy of the run that wrote this batch (issue #199): whether
    /// invalid items were dropped-and-counted instead of corrected or
    /// failed changes what the batch's facts even are, so toggling it
    /// re-extracts. `false` (off) for entries written before this
    /// field existed, the same "new field defaults to the value that
    /// changes today's behavior least" precedent `structured_output`/
    /// `max_output_tokens` set: an unforced re-run of an old batch
    /// keeps matching rather than spuriously re-extracting everything;
    /// `--force` re-extracts under the new strict-by-default rules.
    #[serde(default)]
    pub(super) lossy: bool,
    /// The digest of `--schema`'s document (`""` for no schema, the
    /// default for entries written before this field existed — the
    /// same "new field defaults to the value that changes today's
    /// behavior least" precedent `structured_output`/`lossy` set):
    /// swapping in a different schema document changes what the prompt
    /// asks for and what self-validation accepts, so it re-extracts
    /// like any other computation input.
    #[serde(default)]
    pub(super) schema_digest: String,
    /// `--candidates`' algorithm fingerprint (`""` = off) — ADR 0014,
    /// same default-to-off reasoning as `schema_digest` above: the
    /// candidate block changes what the prompt asks for, so toggling
    /// the control (or revising the segmentation algorithm) re-extracts.
    #[serde(default)]
    pub(super) candidates: String,
    /// `--redact`'s version (`""` = off; `redact1`, `redact1:secrets`,
    /// `redact1:pii`) — ADR 0038 §3.5. Entries written before the
    /// control existed default to `""` and keep matching default runs;
    /// the first `--redact` run over the document re-extracts.
    #[serde(default)]
    pub(super) redaction: String,
    /// `--vocabulary`'s content digest (`""` = off) — ADR 0015: the
    /// offered name set changes what the prompt asks for and what the
    /// occurrence check admits, so it re-extracts like any other
    /// computation input.
    #[serde(default)]
    pub(super) vocabulary_digest: String,
    /// `--source-id`'s EFFECTIVE written source for this document
    /// (`""` = off, the path was written) — #466 S1, ADR 0017: baked
    /// into the emitted header like `context`/`description`, so a
    /// change must rewrite the batch rather than skip with the old id
    /// in place. The effective value (suffix included), not the flag,
    /// so a revision of the multi-document suffix scheme re-extracts
    /// too. Prompt-neutral on purpose: the model is still shown the
    /// document path, which is why this field is NOT in the checkpoint
    /// fingerprint — cached chunk answers stay reusable across a
    /// source-id change.
    #[serde(default)]
    pub(super) source_id: String,
    /// `--date` of the run that wrote this batch (0 = no field
    /// emitted): baked into the passage line, same reasoning — and the
    /// same checkpoint-fingerprint exemption — as `source_id`.
    #[serde(default)]
    pub(super) date: u64,
    /// `--tag`s of the run that wrote this batch (empty = no field
    /// emitted): likewise.
    #[serde(default)]
    pub(super) tags: Vec<String>,
    pub(super) output: String,
}

impl Manifest {
    /// Missing or unreadable manifests degrade to re-extraction —
    /// never to an error, and never to a false "unchanged".
    pub(super) fn load(path: &Path) -> Self {
        match fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|_| {
                eprintln!(
                    "taguru: extract: ignoring an unreadable manifest at {} — everything \
                     re-extracts",
                    path.display()
                );
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// Whether `source`'s recorded entry matches today's computation
    /// inputs field for field (plus this build's own `PROMPT_VERSION`)
    /// — any mismatch treats the entry as absent, so a settings change
    /// can never silently reuse an incompatible output.
    pub(super) fn matches(&self, source: &str, inputs: &ComputationInputs) -> bool {
        self.documents.get(source).is_some_and(|entry| {
            entry.sha256 == inputs.sha256
                && entry.model == inputs.model
                && entry.prompt_version == PROMPT_VERSION
                && entry.context == inputs.context
                && entry.questions_n == inputs.questions_n
                && entry.no_passage == inputs.no_passage
                && entry.description == inputs.description
                && entry.fact_budget == inputs.fact_budget
                && entry.structured_output == inputs.structured_output
                && entry.max_output_tokens == inputs.max_output_tokens
                && entry.escalation_factor == inputs.escalation_factor
                && entry.runaway_ratio == inputs.runaway_ratio
                && entry.chunk_bytes == inputs.chunk_bytes
                && entry.chunk_context == inputs.chunk_context
                && entry.lossy == inputs.lossy
                && entry.schema_digest == inputs.schema_digest
                && entry.candidates == inputs.candidates
                && entry.redaction == inputs.redaction
                && entry.vocabulary_digest == inputs.vocabulary_digest
                && entry.source_id == inputs.source_id
                && entry.date == inputs.date
                && entry.tags == inputs.tags
        })
    }

    /// The output file name the last completed run recorded for
    /// `source` — the skip path reads the batch from THERE (issue
    /// #730): identical to `batch_file_name(source)` for anything
    /// written since, while a manifest from before the naming change
    /// names the old un-hashed file, keeping its unchanged documents
    /// skippable (and naming the stale file to remove once a changed
    /// document re-extracts under the new name).
    pub(super) fn output_of(&self, source: &str) -> Option<String> {
        self.documents.get(source).map(|entry| entry.output.clone())
    }

    pub(super) fn record(&mut self, source: &str, inputs: &ComputationInputs, output: &str) {
        self.documents.insert(
            source.to_string(),
            ManifestEntry {
                sha256: inputs.sha256.to_string(),
                model: inputs.model.to_string(),
                prompt_version: PROMPT_VERSION,
                context: inputs.context.to_string(),
                questions_n: inputs.questions_n,
                no_passage: inputs.no_passage,
                description: inputs.description.to_string(),
                fact_budget: inputs.fact_budget,
                structured_output: inputs.structured_output.to_string(),
                max_output_tokens: inputs.max_output_tokens,
                escalation_factor: inputs.escalation_factor.to_string(),
                runaway_ratio: inputs.runaway_ratio.to_string(),
                chunk_bytes: inputs.chunk_bytes.to_string(),
                chunk_context: inputs.chunk_context.to_string(),
                lossy: inputs.lossy,
                schema_digest: inputs.schema_digest.to_string(),
                candidates: inputs.candidates.to_string(),
                redaction: inputs.redaction.to_string(),
                vocabulary_digest: inputs.vocabulary_digest.to_string(),
                source_id: inputs.source_id.to_string(),
                date: inputs.date,
                tags: inputs.tags.to_vec(),
                output: output.to_string(),
            },
        );
    }

    pub(super) fn save(&self, path: &Path) -> std::io::Result<()> {
        let text = serde_json::to_string_pretty(self).expect("a manifest serializes");
        crate::storage::write_atomic(path, text.as_bytes())
    }
}

/// The computation inputs one document's extraction depends on — what
/// [`Manifest::matches`] compares and [`Manifest::record`] stamps
/// (issue #730: seventeen positional arguments, several sharing a
/// type, made those call sites unreviewable — and `Run` passing the
/// list twice let the two drift). Built once per document; field names
/// mirror [`ManifestEntry`]'s, minus `prompt_version` (this build's
/// own constant) and `output` (a result, not an input).
pub(super) struct ComputationInputs<'a> {
    pub(super) sha256: &'a str,
    pub(super) model: &'a str,
    pub(super) context: &'a str,
    pub(super) questions_n: usize,
    pub(super) no_passage: bool,
    pub(super) description: &'a str,
    pub(super) fact_budget: usize,
    pub(super) structured_output: &'a str,
    pub(super) max_output_tokens: usize,
    pub(super) escalation_factor: &'a str,
    pub(super) runaway_ratio: &'a str,
    pub(super) chunk_bytes: &'a str,
    pub(super) chunk_context: &'a str,
    pub(super) lossy: bool,
    pub(super) schema_digest: &'a str,
    pub(super) candidates: &'a str,
    pub(super) redaction: &'a str,
    pub(super) vocabulary_digest: &'a str,
    pub(super) source_id: &'a str,
    pub(super) date: u64,
    pub(super) tags: &'a [String],
}

/// The flatten-then-hash naming scheme [`batch_file_name`] and
/// [`checkpoint_file_name`] share: separators flatten to `__` so the
/// directory stays flat, and the hash suffix is ALWAYS appended —
/// flattening alone is not injective (`"a/b"`, `"a:b"`, and `"a__b"`
/// all flatten to `"a__b"`), so distinct sources could otherwise
/// collide on one file. A flattened name over 120 UTF-8 bytes
/// truncates to a <=96-byte prefix so long paths never blow a
/// filesystem's name-length limit; the prefix is a human-readable
/// label, the hash alone is what keeps names apart.
fn flattened_hashed_name(source: &str, extension: &str) -> String {
    let flattened = source.replace(['/', '\\', ':'], "__");
    let prefix = if flattened.len() > 120 {
        &flattened[..floor_char_boundary(&flattened, 96)]
    } else {
        flattened.as_str()
    };
    format!(
        "{prefix}-{}.{extension}",
        &sha256_hex(source.as_bytes())[..16]
    )
}

/// Output name for a source path ([`flattened_hashed_name`], `.jsonl`)
/// — what `taguru import DIR` reads. The suffix went unconditional in
/// issue #730, the same injectivity fix [`checkpoint_file_name`] got
/// in #227: one run's collisions were already caught by `Run::claimed`,
/// but separate runs into the same `--out` know nothing of each other,
/// so a later run's colliding document silently overwrote the earlier
/// one's batch output. The skip path reads the file the MANIFEST
/// recorded, so batches written under the pre-#730 naming stay
/// skippable — see `Run::extract_document`.
pub(super) fn batch_file_name(source: &str) -> String {
    flattened_hashed_name(source, "jsonl")
}

/// Directory (adjacent to `--out`, hidden like the manifest) holding
/// one per-document chunk checkpoint file — issue #179's durable
/// resume. Never created for `--dry-run` (which calls/writes nothing)
/// or for a document with no checkpointable units yet.
pub(super) const CHECKPOINT_DIR_NAME: &str = ".extract-checkpoints";

/// One checkpoint file per document ([`flattened_hashed_name`],
/// `.json`), named from its source path so a `--out` directory listing
/// stays legible. The unconditional hash suffix arrived here first
/// (issue #227): without it, distinct short source ids could collide
/// on the same file and silently share (and overwrite) each other's
/// checkpoint progress.
pub(super) fn checkpoint_file_name(source: &str) -> String {
    flattened_hashed_name(source, "json")
}
