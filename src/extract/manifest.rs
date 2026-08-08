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
    /// `--vocabulary`'s content digest (`""` = off) — ADR 0015: the
    /// offered name set changes what the prompt asks for and what the
    /// occurrence check admits, so it re-extracts like any other
    /// computation input.
    #[serde(default)]
    pub(super) vocabulary_digest: String,
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matches(
        &self,
        source: &str,
        sha256: &str,
        model: &str,
        context: &str,
        questions_n: usize,
        no_passage: bool,
        description: &str,
        fact_budget: usize,
        structured_output: &str,
        max_output_tokens: usize,
        lossy: bool,
        schema_digest: &str,
        candidates: &str,
        vocabulary_digest: &str,
    ) -> bool {
        self.documents.get(source).is_some_and(|entry| {
            entry.sha256 == sha256
                && entry.model == model
                && entry.prompt_version == PROMPT_VERSION
                && entry.context == context
                && entry.questions_n == questions_n
                && entry.no_passage == no_passage
                && entry.description == description
                && entry.fact_budget == fact_budget
                && entry.structured_output == structured_output
                && entry.max_output_tokens == max_output_tokens
                && entry.lossy == lossy
                && entry.schema_digest == schema_digest
                && entry.candidates == candidates
                && entry.vocabulary_digest == vocabulary_digest
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &mut self,
        source: &str,
        sha256: &str,
        model: &str,
        context: &str,
        questions_n: usize,
        no_passage: bool,
        description: &str,
        fact_budget: usize,
        structured_output: &str,
        max_output_tokens: usize,
        lossy: bool,
        schema_digest: &str,
        candidates: &str,
        vocabulary_digest: &str,
        output: &str,
    ) {
        self.documents.insert(
            source.to_string(),
            ManifestEntry {
                sha256: sha256.to_string(),
                model: model.to_string(),
                prompt_version: PROMPT_VERSION,
                context: context.to_string(),
                questions_n,
                no_passage,
                description: description.to_string(),
                fact_budget,
                structured_output: structured_output.to_string(),
                max_output_tokens,
                lossy,
                schema_digest: schema_digest.to_string(),
                candidates: candidates.to_string(),
                vocabulary_digest: vocabulary_digest.to_string(),
                output: output.to_string(),
            },
        );
    }

    pub(super) fn save(&self, path: &Path) -> std::io::Result<()> {
        let text = serde_json::to_string_pretty(self).expect("a manifest serializes");
        crate::storage::write_atomic(path, text.as_bytes())
    }
}

/// Output name for a source path: separators flatten to `__` so the
/// output directory stays flat — which is what `taguru import DIR`
/// reads. Long paths would blow the OS filename limit, so they keep a
/// head for the human and a hash for uniqueness.
pub(super) fn batch_file_name(source: &str) -> String {
    let mut name = source.replace(['/', '\\', ':'], "__");
    if name.len() > 120 {
        name = format!(
            "{}-{}",
            &name[..floor_char_boundary(&name, 96)],
            &sha256_hex(source.as_bytes())[..16]
        );
    }
    format!("{name}.jsonl")
}

/// Directory (adjacent to `--out`, hidden like the manifest) holding
/// one per-document chunk checkpoint file — issue #179's durable
/// resume. Never created for `--dry-run` (which calls/writes nothing)
/// or for a document with no checkpointable units yet.
pub(super) const CHECKPOINT_DIR_NAME: &str = ".extract-checkpoints";

/// Loosely based on [`batch_file_name`]'s flatten-then-hash scheme,
/// `.json` instead of `.jsonl` — one checkpoint file per document,
/// named from its source path so a `--out` directory listing stays
/// legible. Unlike `batch_file_name`, the hash suffix is ALWAYS
/// appended, not just past the 120-byte threshold: flattening alone is
/// not injective (`"a/b"`, `"a:b"`, and `"a__b"` all flatten to
/// `"a__b"`), so without an unconditional suffix, distinct short
/// source ids could collide on the same file and silently share (and
/// overwrite) each other's checkpoint progress. A flattened name over
/// 120 UTF-8 bytes still truncates to a <=96-byte prefix so long
/// source paths never blow a filesystem's name-length limit, but that
/// truncated prefix is now purely a human-readable label — the hash
/// suffix alone is what keeps names apart.
pub(super) fn checkpoint_file_name(source: &str) -> String {
    let flattened = source.replace(['/', '\\', ':'], "__");
    let prefix = if flattened.len() > 120 {
        &flattened[..floor_char_boundary(&flattened, 96)]
    } else {
        flattened.as_str()
    };
    format!("{prefix}-{}.json", &sha256_hex(source.as_bytes())[..16])
}
