//! `Run`: one extraction run's settled configuration plus everything
//! that accumulates across documents, and the per-document/per-chunk
//! pipeline (`impl Run`) that drives it.

use super::*;

/// [`Run::resolve_system`]'s result: the system text actually used,
/// its `sha256` (always — the trace's `steering` record wants what
/// was actually sent, pinned or not), and the run_id it was pinned
/// from, when it was pinned (ADR 0031 §3.6).
pub(super) struct ResolvedSystem {
    pub(super) text: String,
    pub(super) sha256: String,
    pub(super) pinned_from: Option<String>,
}

/// One extract run: the settled flags, the provider, and everything
/// that accumulates across documents — the manifest, the label
/// vocabulary offered to later prompts, and the output names already
/// claimed. One run targets one context on purpose (docs/extract.html).
pub(super) struct Run {
    pub(super) context: String,
    pub(super) description: Option<String>,
    /// `--source-id` (#466 S1, ADR 0017): the promotion runbook's
    /// session source id, written into the batch header in place of
    /// the document path. `None` = the path, today's batch byte for
    /// byte. The MANIFEST stays keyed by the document path either way
    /// — the path names the input, this names the output.
    pub(super) source_id: Option<String>,
    /// `--date` (#466 S1): epoch seconds for the passage line's
    /// `date` field (`None` = no field).
    pub(super) date: Option<u64>,
    /// `--tag` (#466 S1): tags for the passage line (empty = no field).
    pub(super) tags: Vec<String>,
    /// Whether this run extracts more than one document — under
    /// `--source-id` that appends the runbook's `/{doc}` suffix
    /// (the file stem) so per-source retract-then-apply cannot make
    /// two documents silently replace each other.
    pub(super) multi_document: bool,
    /// Written-source claims (the batch HEADER's source), mirroring
    /// `claimed`'s file-name check one level up: two documents whose
    /// effective source ids collide would clobber each other at import
    /// (retract-then-apply is per source id), so the second one fails
    /// here instead.
    pub(super) claimed_source_ids: BTreeMap<String, String>,
    pub(super) force: bool,
    pub(super) dry_run: bool,
    pub(super) no_passage: bool,
    pub(super) questions: usize,
    /// Resolved from `--fact-budget`/TAGURU_EXTRACT_FACT_BUDGET (0 =
    /// off, the default).
    pub(super) fact_budget: usize,
    /// Resolved from TAGURU_EXTRACT_MAX_ATTEMPTS/
    /// TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES.
    pub(super) correction: CorrectionPolicy,
    /// Resolved from `--structured-output`/
    /// TAGURU_EXTRACT_STRUCTURED_OUTPUT (`Off`, the default). Kept
    /// beside `ladder` because the manifest records the REQUESTED mode
    /// as a computation input even under `--dry-run`, which resolves
    /// no ladder.
    pub(super) structured_output: StructuredOutputMode,
    /// Resolved from `--max-output-tokens`/
    /// TAGURU_EXTRACT_MAX_OUTPUT_TOKENS (`None` = no output-token
    /// parameter is ever sent, today's request).
    pub(super) max_output_tokens: Option<usize>,
    /// TAGURU_EXTRACT_ESCALATION_FACTOR (ADR 0019) — carried here only
    /// for the manifest record; the ladder reads its own copy.
    pub(super) escalation_factor: usize,
    /// `--chunk-bytes`/TAGURU_EXTRACT_CHUNK_BYTES, resolved
    /// ([`CHUNK_BYTES`] by default) — ADR 0020 (#762).
    pub(super) chunk_bytes: usize,
    /// `--chunk-context`/TAGURU_EXTRACT_CHUNK_CONTEXT, resolved
    /// (`Off` by default) — ADR 0033 (#782).
    pub(super) chunk_context: ChunkContextMode,
    /// `Some` exactly when a mechanism or an output budget is engaged
    /// on a live run: the §7 ladder replaces the legacy corrective
    /// loop. `None` under all-defaults — byte-for-byte today's
    /// behavior — and under `--dry-run`.
    pub(super) ladder: Option<LadderConfig>,
    pub(super) out: PathBuf,
    /// `None` exactly under `--dry-run`, which must call nothing.
    pub(super) completions: Option<Completions>,
    pub(super) model_name: String,
    pub(super) manifest: Manifest,
    /// Relation labels this run has settled on, mapped to how many
    /// associations (plus alias canonicals) used each one so far —
    /// issue #759's reuse-frequency signal, offered back to
    /// `system_prompt` so a one-off label isn't indistinguishable from
    /// an established one.
    pub(super) vocabulary: BTreeMap<String, usize>,
    /// Issue #758: every concept/label spelling an earlier document of
    /// this run (or `--vocabulary`'s context) settled on, mapped to
    /// what it resolves to — the set a later document's alias must not
    /// rewire. Grows exactly where `vocabulary` does: when a document
    /// lands, and when a skipped one's batch is absorbed.
    pub(super) claimed_names: ClaimedNames,
    pub(super) claimed: BTreeMap<String, String>,
    /// Chunk completions to run concurrently within one document (1 =
    /// today's sequential loop). Documents themselves always run
    /// sequentially — see [`Run::extract_chunks`].
    pub(super) parallel: usize,
    /// Resolved from `--lossy`/TAGURU_EXTRACT_LOSSY (`false`, the
    /// default). `true` restores merge()'s pre-issue-#199
    /// drop-and-proceed behavior byte for byte: no Stage 1/Stage 2
    /// validation, no corrective turn spent on a business-rule
    /// violation, `report()` marks every drop explicitly as `--lossy`.
    pub(super) lossy: bool,
    /// Resolved from `--candidates`/TAGURU_EXTRACT_CANDIDATES (`false`,
    /// the default — ADR 0001 §12.2's default-off discipline). `true`
    /// appends the document's own candidate names to the system prompt
    /// (ADR 0014, #496 S2) and stamps `candidates_manifest_value` into
    /// the manifest/checkpoint fingerprints.
    pub(super) candidates: bool,
    /// Resolved from `--coverage`/TAGURU_EXTRACT_COVERAGE (`false`,
    /// the default). `true` reports every sentence holding a candidate
    /// pair that no accepted association covers (ADR 0016, #496 S4) —
    /// report-only: the batch is unchanged, so unlike `candidates`
    /// this is never a fingerprint input.
    pub(super) coverage: bool,
    /// `--vocabulary`'s harvested target-context concept names, capped
    /// for the prompt (ADR 0015; empty = the control is off).
    pub(super) vocabulary_names: Vec<String>,
    /// The FULL harvested concept set, occurrence-normalized — the
    /// ADR 0013 check's allowlist, uncapped on purpose: a context
    /// spelling is legitimate whether or not it fit the prompt list.
    pub(super) vocabulary_allowlist: HashSet<String>,
    /// Content digest of the harvested name sets (`""` = off) — a
    /// manifest/checkpoint computation input like `schema_digest`.
    pub(super) vocabulary_digest: String,
    /// ADR 0033 §3.2: the export's relations per concept name,
    /// offered under `--chunk-context ingested` (empty otherwise, or
    /// with no `--vocabulary`).
    pub(super) vocabulary_known: BTreeMap<String, Vec<KnownRelation>>,
    /// The digest of `vocabulary_known` — part of the manifest's
    /// `chunk_context` value under `ingested`.
    pub(super) known_digest: String,
    /// Resolved from `--diagnostics-out`/TAGURU_EXTRACT_DIAGNOSTICS
    /// (`None`, the default: no sidecar, stdout/stderr byte-for-byte
    /// today's). Issue #200.
    pub(super) diagnostics: Option<DiagnosticsSink>,
    /// Resolved from `--schema`/TAGURU_EXTRACT_SCHEMA (`None`, the
    /// default: no schema block in the prompt, no schema
    /// self-validation — today's behavior). ADR 0009 §11.
    pub(super) schema: Option<Arc<crate::schema::InstalledSchema>>,
    /// The canonical digest of `schema`'s document (`""` when `schema`
    /// is `None`) — folded into `ManifestEntry`/`CheckpointFingerprint`
    /// so swapping in a different schema document re-extracts, exactly
    /// as changing any other computation input does.
    pub(super) schema_digest: String,
    /// Issue #179's cooperative stop flag, checked between chunks and
    /// between documents.
    pub(super) stop: StopSignal,
    /// ADR 0025 (#788): whether each document's attempts log (full
    /// prompts and answers) is written — `true` unless
    /// `TAGURU_EXTRACT_TRACE_ATTEMPTS` is `off`.
    pub(super) attempts_log: bool,
    /// Resolved from `--replay`/TAGURU_EXTRACT_REPLAY (`Off`, the
    /// default) — ADR 0031 §3.3.
    pub(super) replay_mode: ReplayMode,
    /// `true` exactly for `Auto`/`Strict` — the manifest skip and the
    /// checkpoint store both go inert under this (ADR 0031 §3.5), and
    /// [`Run::extract_document`] loads a [`ReplayIndex`] for every
    /// document instead of none.
    pub(super) replaying: bool,
    /// `--replay-from`/TAGURU_EXTRACT_REPLAY_FROM, resolved even when
    /// `!replaying` (`out/.extract-trace` by default) — ADR 0031 §3.7.
    pub(super) replay_from: PathBuf,
    /// `--resume-from prompt`'s fold (#822): [`Run::resolve_system`]
    /// never consults [`Completions::pinned_system`] — the whole point
    /// of naming `prompt` as the resume point is that the prompt
    /// itself (system included) is rebuilt fresh, not reused verbatim
    /// (ADR 0031 §3.6). `false` for every other `--resume-from` value
    /// and off `--resume-from` entirely.
    pub(super) disable_system_pin: bool,
    /// `--resume-from` was given at all, whatever step it named (#822)
    /// — `true` even for `read`/`plan`/`steer`, where
    /// [`resume_from_fold`] leaves `replaying` `false` since there is
    /// nothing to replay against. The manifest skip and the checkpoint
    /// store both go inert under this too: the whole point of naming a
    /// resume step is a deliberate redo, and a skip silently answering
    /// "nothing changed, did nothing" would defeat that intent exactly
    /// as much here as under `--replay` (ADR 0031 §3.5's reasoning,
    /// extended).
    pub(super) resume_requested: bool,
}

impl Run {
    /// ADR 0023: the run's id, minted by `Completions` (one per
    /// invocation). Empty under `--dry-run`, which builds no
    /// `Completions` and writes no trace.
    pub(super) fn run_id(&self) -> String {
        self.completions
            .as_ref()
            .map(|completions| completions.run_id().to_string())
            .unwrap_or_default()
    }

    /// ADR 0025: the per-document attempts log, unless
    /// `TAGURU_EXTRACT_TRACE_ATTEMPTS` turned it off (resolved once
    /// into `attempts_log`). Failure to open is reported, not fatal.
    pub(super) fn open_attempt_log(
        &self,
        batch_file_name: &str,
        resuming: bool,
        run_id: &str,
        source: &str,
        document_sha256: &str,
    ) -> Option<AttemptLog> {
        if !self.attempts_log {
            return None;
        }
        let dir = self.out.join(TRACE_DIR_NAME);
        let path = dir.join(attempts_file_name(batch_file_name));
        let opened = fs::create_dir_all(&dir).and_then(|()| {
            AttemptLog::open(path.clone(), resuming, run_id, source, document_sha256)
        });
        match opened {
            Ok(log) => Some(log),
            Err(error) => {
                eprintln!(
                    "taguru: extract: attempts log: opening {}: {error} — this document's \
                     attempts are not recorded",
                    path.display()
                );
                None
            }
        }
    }

    /// ADR 0027 (#789): the trace's `steering` record for one document
    /// — every list computed by the same function that renders its
    /// prompt block, so the record is what the model was actually
    /// shown. Document-scoped (`chunk_index: null`): all of a
    /// document's chunks share one prompt. Exception: when
    /// `resolved_system.pinned_from` is `Some` (ADR 0031 §3.6), the
    /// system prompt actually sent is the pinned, verbatim recorded
    /// text — only `system_sha256` describes it exactly.
    /// `candidates`/`vocabulary`/`context_names`/`schema` still reflect
    /// this run's own current computation, which a pin may have
    /// bypassed; a trace record is advisory (ADR 0023 §3.6), never
    /// authoritative over what the batch itself holds.
    pub(super) fn steering_record<'a>(
        &'a self,
        candidates: &'a [String],
        resolved_system: &'a ResolvedSystem,
    ) -> TraceSteering<'a> {
        TraceSteering {
            kind: "steering",
            chunk_index: None,
            candidates,
            system_sha256: &resolved_system.sha256,
            pinned_from: resolved_system.pinned_from.as_deref(),
            vocabulary: ranked_vocabulary(&self.vocabulary)
                .into_iter()
                .map(|(label, count)| {
                    // The prompt renders these by reference into one
                    // string; the record needs the label text itself.
                    VocabularyEntry {
                        label: self
                            .vocabulary
                            .get_key_value(&label)
                            .map(|(key, _)| key.as_str())
                            .unwrap_or_default(),
                        count,
                    }
                })
                .collect(),
            context_names: &self.vocabulary_names,
            schema: self
                .schema
                .as_deref()
                .map(crate::schema::InstalledSchema::document)
                .filter(|document| document.mode != crate::schema::SchemaMode::Off)
                .and_then(|document| {
                    let types = schema_type_names(document);
                    let constrained_relations: Vec<&str> =
                        schema_constrained_relations(document, &self.vocabulary)
                            .into_iter()
                            .map(|(label, _)| label)
                            .collect();
                    // `schema_block` renders nothing for a schema with
                    // no types and no constrained relations — no block
                    // prompted means `null` here, exactly as for
                    // `mode: off` (ADR 0027's "null exactly when no
                    // schema block was prompted").
                    if types.is_empty() && constrained_relations.is_empty() {
                        return None;
                    }
                    Some(SteeringSchema {
                        types,
                        constrained_relations,
                    })
                }),
        }
    }

    /// ADR 0031 §3.6: this document's system prompt — pinned verbatim
    /// from the attempts log when [`Completions::pinned_system`] names
    /// exactly one distinct recorded system, recomputed from this
    /// run's own settings otherwise. Always recomputes when not
    /// pinning outright and, when pinning, always recomputes anyway
    /// (cheap: no model call) to check the pin against — reporting a
    /// mismatch once on stderr, a hint never a gate; matching itself
    /// is still decided completion by completion, by conversation
    /// content (ADR 0031 §3.2). `--resume-from prompt` (#822) forces
    /// `NoRecord` regardless of what the index would otherwise decide —
    /// silently, since declining a pin the caller never asked for
    /// needs no stderr line of its own.
    pub(super) fn resolve_system(&self, source: &str, candidates: &[String]) -> ResolvedSystem {
        let recompute = || {
            system_prompt(
                &self.vocabulary,
                self.questions,
                self.fact_budget,
                self.schema.as_deref(),
                &self.vocabulary_names,
                candidates,
            )
        };
        let decision = if self.disable_system_pin {
            SystemPinDecision::NoRecord
        } else {
            self.completions
                .as_ref()
                .map(Completions::pinned_system)
                .unwrap_or(SystemPinDecision::NoRecord)
        };
        let (content, pinned_from): (String, Option<String>) = match decision {
            SystemPinDecision::Pin { content, run_id } => {
                (content.to_string(), Some(run_id.to_string()))
            }
            SystemPinDecision::Ambiguous { distinct } => {
                eprintln!(
                    "taguru: extract: {source}: {distinct} distinct system prompts recorded \
                     — not pinning, computing this run's own"
                );
                (recompute(), None)
            }
            SystemPinDecision::NoRecord => (recompute(), None),
        };
        let sha256 = sha256_hex(content.as_bytes());
        if pinned_from.is_some() && sha256_hex(recompute().as_bytes()) != sha256 {
            eprintln!(
                "taguru: extract: {source}: the recorded system prompt differs from this \
                 run's — replaying with the recorded one"
            );
        }
        ResolvedSystem {
            text: content,
            sha256,
            pinned_from,
        }
    }

    /// The Stage 1 item rules for one document, or `None` under
    /// `--lossy` — see [`evaluate_answer`]/[`ItemRules`].
    pub(super) fn item_rules(&self, paragraph_count: usize) -> Option<ItemRules> {
        (!self.lossy).then_some(ItemRules {
            paragraph_count,
            questions_requested: self.questions > 0,
        })
    }

    /// The source id the batch header carries: the document path
    /// (today's behavior), or `--source-id`'s override — verbatim for
    /// a single document, with the runbook's `/{doc}` suffix (the file
    /// stem) when the run extracts several, since one session id
    /// covering two documents would make import's per-source
    /// retract-then-apply fold them into one another.
    pub(super) fn written_source(&self, path: &Path, source: &str) -> String {
        match &self.source_id {
            None => source.to_string(),
            Some(id) if !self.multi_document => id.clone(),
            Some(id) => {
                let stem = path
                    .file_stem()
                    .map(|stem| stem.to_string_lossy())
                    .unwrap_or_default();
                format!("{id}/{stem}")
            }
        }
    }
}

/// [`Run::extract_chunks`]'s result: either every chunk completed, or a
/// cooperative stop request (issue #179) was observed between chunks
/// first. Distinct from `Err` — an interruption isn't a failure, and
/// whatever units already landed are still durably checkpointed.
pub(super) enum ChunkLoopResult {
    Complete(Vec<ChunkOutput>),
    Interrupted,
}

impl Run {
    /// ADR 0033: the manifest/checkpoint/settings record of the chunk
    /// context in force — the mode name, and under `ingested` the
    /// digest of the relations the export offers appended
    /// (`ingested:<digest>`), since those relations are prompt content
    /// and a changed export changes what every chunk is told.
    pub(super) fn chunk_context_value(&self) -> String {
        if self.chunk_context.ingested() {
            format!("ingested:{}", self.known_digest)
        } else {
            self.chunk_context.manifest_value().to_string()
        }
    }

    /// The checkpoint compatibility fingerprint for one document — the
    /// same fields [`Manifest::matches`]/[`Manifest::record`] already
    /// carry, minus `output`. Any mismatch against a loaded file's own
    /// fingerprint treats every cached unit as absent (issue #179).
    pub(super) fn checkpoint_fingerprint(&self, hash: &str) -> CheckpointFingerprint {
        CheckpointFingerprint {
            sha256: hash.to_string(),
            model: self.model_name.clone(),
            prompt_version: PROMPT_VERSION,
            context: self.context.clone(),
            questions_n: self.questions,
            no_passage: self.no_passage,
            description: self.description.as_deref().unwrap_or("").to_string(),
            fact_budget: self.fact_budget,
            structured_output: self.structured_output.manifest_value().to_string(),
            max_output_tokens: self.max_output_tokens.unwrap_or(0),
            escalation_factor: escalation_manifest_value(
                self.max_output_tokens,
                self.escalation_factor,
            ),
            chunk_bytes: chunk_bytes_manifest_value(self.chunk_bytes),
            chunk_context: self.chunk_context_value(),
            lossy: self.lossy,
            schema_digest: self.schema_digest.clone(),
            candidates: candidates_manifest_value(self.candidates).to_string(),
            vocabulary_digest: self.vocabulary_digest.clone(),
        }
    }

    /// Loads one document's checkpoint store. `--force` — already "redo
    /// this document over" at the manifest level — extends the same
    /// intent one level deeper: an empty store, ignoring whatever units
    /// a prior run cached, rather than comparing them against today's
    /// fingerprint (which would often still match and silently defeat
    /// the point of forcing a redo).
    pub(super) fn load_checkpoints(&self, source: &str, hash: &str) -> CheckpointStore {
        let fingerprint = self.checkpoint_fingerprint(hash);
        let path = self
            .out
            .join(CHECKPOINT_DIR_NAME)
            .join(checkpoint_file_name(source));
        // ADR 0031 §3.5: a replay run never reads the checkpoint store
        // — a cached unit was parsed and mechanically validated under
        // whichever code ran the ORIGINAL extraction, which for "change
        // the validation rule, replay the rest" is exactly the answer
        // replay must not silently reuse.
        if self.force || self.replaying || self.resume_requested {
            CheckpointStore::empty(path, fingerprint)
        } else {
            CheckpointStore::load(path, &fingerprint)
        }
    }

    /// The whole per-document pipeline: caps, the manifest skip, the
    /// chunk loop, merge, self-validation, the atomic write, the
    /// report line. `Err` is one document failing — the caller prints
    /// it after `taguru: extract: {source}: ` and the run continues.
    pub(super) fn extract_document(
        &mut self,
        path: &Path,
        source: &str,
    ) -> Result<Outcome, String> {
        if source.len() > MAX_NAME_BYTES {
            return Err(format!(
                "the path is {} bytes, over the {MAX_NAME_BYTES}-byte source cap",
                source.len()
            ));
        }
        let file_name = batch_file_name(source);
        if let Some(other) = self.claimed.get(&file_name) {
            return Err(format!(
                "its batch file name collides with '{other}' — rename one of the documents"
            ));
        }
        self.claimed.insert(file_name.clone(), source.to_string());
        let out_path = self.out.join(&file_name);

        let written_source = self.written_source(path, source);
        if written_source.len() > MAX_NAME_BYTES {
            return Err(format!(
                "its source id '{written_source}' is {} bytes, over the \
                 {MAX_NAME_BYTES}-byte source cap",
                written_source.len()
            ));
        }
        if let Some(other) = self.claimed_source_ids.get(&written_source) {
            return Err(format!(
                "its source id '{written_source}' collides with '{other}' — import's \
                 retract-then-apply is per source id, so one would silently replace the \
                 other; rename one of the documents"
            ));
        }
        self.claimed_source_ids
            .insert(written_source.clone(), source.to_string());

        let text = read_document(path)?;
        let hash = sha256_hex(text.as_bytes());
        // The fingerprint's source-id value is the EFFECTIVE written
        // source, but only under the flag — "" when off, so pre-S1
        // manifest entries (no field) keep matching default runs.
        let source_id_value = if self.source_id.is_some() {
            written_source.as_str()
        } else {
            ""
        };
        // The batch an unchanged document skips FROM is the file the
        // manifest recorded — identical to `out_path` for anything
        // written under the post-#730 naming, but a manifest from
        // before the naming change records the old un-hashed name, and
        // re-extracting an unchanged document just because its file
        // name scheme moved would spend real model calls.
        let recorded_output = self.manifest.output_of(source);
        // Built ONCE for the skip check and the post-write record
        // alike, so the two can never drift field by field.
        let escalation_factor =
            escalation_manifest_value(self.max_output_tokens, self.escalation_factor);
        let chunk_bytes = chunk_bytes_manifest_value(self.chunk_bytes);
        let chunk_context_value = self.chunk_context_value();
        let inputs = ComputationInputs {
            sha256: &hash,
            model: &self.model_name,
            context: &self.context,
            questions_n: self.questions,
            no_passage: self.no_passage,
            description: self.description.as_deref().unwrap_or(""),
            fact_budget: self.fact_budget,
            structured_output: self.structured_output.manifest_value(),
            max_output_tokens: self.max_output_tokens.unwrap_or(0),
            escalation_factor: &escalation_factor,
            chunk_bytes: &chunk_bytes,
            chunk_context: &chunk_context_value,
            lossy: self.lossy,
            schema_digest: &self.schema_digest,
            candidates: candidates_manifest_value(self.candidates),
            vocabulary_digest: &self.vocabulary_digest,
            source_id: source_id_value,
            date: self.date.unwrap_or(0),
            tags: &self.tags,
        };
        // ADR 0031 §3.5: the skip exists to avoid paying for model
        // calls on an unchanged document — under replay a completion is
        // free whether or not it hits, so the skip's reason to exist is
        // gone. #822 extends the same reasoning to `--resume-from`
        // whatever step it names: it is a deliberate ask to redo this
        // document, and "unchanged, skipped" would silently answer
        // that ask with nothing at all — even for `read`/`plan`/`steer`,
        // where nothing is satisfied from a record either.
        if !self.force
            && !self.replaying
            && !self.resume_requested
            && self.manifest.matches(source, &inputs)
            && let Some(recorded) = recorded_output
                .as_deref()
                .map(|name| self.out.join(name))
                .filter(|path| path.is_file())
        {
            let batch = self.absorb_vocabulary(source, &recorded);
            println!("{source}: unchanged, skipped (--force re-extracts)");
            // ADR 0016: coverage is a pure function of (document text,
            // written associations), so a skipped document is judged
            // too, from the batch it already has — no model call, and
            // a past run's recall ceiling stays measurable for free.
            if self.coverage
                && let Some(batch) = batch
            {
                for gap in coverage_gaps(&text, &batch.association_triples()) {
                    eprintln!("taguru: extract: {source}: uncovered: {}", gap.describe());
                }
            }
            return Ok(Outcome::Unchanged);
        }

        // The model sees the server's own paragraph numbering (prompt
        // input only — the passage stays verbatim) so every returned
        // association and question can cite an index the server
        // itself validates against.
        let paragraph_spans = crate::paragraph::split(&text);
        let canonical_paragraphs = paragraph_spans.len();
        // ADR 0014: candidate names come from the WHOLE document, once
        // — every chunk is offered the same list (the vocabulary
        // discipline's reasoning, one level down), and the corrective
        // path rebuilds the identical prompt.
        let candidates = if self.candidates {
            candidate_terms(&text)
        } else {
            Vec::new()
        };
        // ADR 0033 (#782): the `structure` step — the document's
        // headings, articles, and speakers, from its own lines — and
        // the chunk boundaries it makes `plan` prefer. Off: no units,
        // no preferred breaks, today's plan byte for byte.
        let units: Vec<Unit> = if self.chunk_context.is_on() {
            detect_units(&text, &paragraph_spans)
        } else {
            Vec::new()
        };
        let plan = chunk_plan_preferring(&text, self.chunk_bytes, &preferred_breaks(&units));
        if self.dry_run {
            // Read-only: a dry run still calls/writes nothing, but
            // reusable-count reporting is exactly what --dry-run is for
            // (issue #179). Top-level-only — dry-run resolves no ladder
            // and probes nothing (matching its existing contract), so a
            // chunk that would end up split on a real run is honestly
            // reported as pending rather than guessed at.
            let checkpoints = self.load_checkpoints(source, &hash);
            let reusable = plan
                .iter()
                .filter(|descriptor| checkpoints.lookup(&descriptor.sha256).is_some())
                .count();
            if reusable > 0 {
                println!(
                    "{source}: would extract ({} bytes, {} chunk(s), {reusable} reusable from \
                     checkpoint) → {}",
                    text.len(),
                    plan.len(),
                    out_path.display()
                );
            } else {
                println!(
                    "{source}: would extract ({} bytes, {} chunk(s)) → {}",
                    text.len(),
                    plan.len(),
                    out_path.display()
                );
            }
            return Ok(Outcome::Planned);
        }

        let checkpoints = self.load_checkpoints(source, &hash);
        if self.stop.check() {
            return Ok(Outcome::Interrupted);
        }
        if let Some(sink) = self.diagnostics.as_ref() {
            for (index, descriptor) in plan.iter().enumerate() {
                sink.emit_chunk(source, index, plan.len(), descriptor);
            }
        }
        // ADR 0031 §3.4: the whole attempts log is read into memory
        // BEFORE this run opens its own log for writing — a later
        // append (or, pre-replay, a truncate) can then never disturb
        // what was already loaded.
        let replay_index = self
            .replaying
            .then(|| ReplayIndex::load(&self.replay_from.join(attempts_file_name(&file_name))));
        // ADR 0025 (#788): the attempts log — every completion's full
        // prompt and answer — opens here, appending when the checkpoint
        // says this document is being resumed, truncating otherwise,
        // so the file spans exactly the runs that built the batch. A
        // log that cannot open is one stderr line, never a failure.
        // ADR 0031 §3.4: a replay run always appends, even on an
        // otherwise-fresh document (whose checkpoint holds nothing) —
        // truncating the very file replay is about to read from would
        // destroy its own input.
        let run_id = self.run_id();
        let attempt_log = self.open_attempt_log(
            &file_name,
            checkpoints.unit_count() > 0 || self.replaying,
            &run_id,
            source,
            &hash,
        );
        // ADR 0031 §3.2/§3.9: right after the `document` record, once
        // per document — a diagnostic snapshot of this run's settings,
        // never a manifest/checkpoint computation input.
        if let Some(log) = attempt_log.as_ref() {
            log.write_record(&SettingsRecord {
                kind: "settings",
                prompt_version: PROMPT_VERSION,
                model: &self.model_name,
                questions_n: self.questions,
                fact_budget: self.fact_budget,
                structured_output: self.structured_output.manifest_value(),
                max_output_tokens: self.max_output_tokens.unwrap_or(0),
                chunk_bytes: &chunk_bytes,
                chunk_context: &chunk_context_value,
                lossy: self.lossy,
                schema_digest: &self.schema_digest,
                candidates: candidates_manifest_value(self.candidates),
                vocabulary_digest: &self.vocabulary_digest,
                rung: self.ladder.as_ref().map(|ladder| ladder.rung().name()),
            });
            // ADR 0031 §3.4: names the mode and the source directory,
            // right after `settings`.
            if self.replaying {
                log.write_record(&ReplayRecord {
                    kind: "replay",
                    mode: self.replay_mode.name(),
                    replay_from: &self.replay_from.display().to_string(),
                });
            }
        }
        // ADR 0031 §3.2/§3.9: a settings mismatch is a hint, never a
        // gate — matching is still decided completion by completion,
        // by conversation content. Reported once per document, before
        // any completion of it is attempted.
        if let Some(recorded) = replay_index.as_ref().and_then(ReplayIndex::settings) {
            let current = RecordedSettings {
                prompt_version: u64::from(PROMPT_VERSION),
                model: self.model_name.clone(),
                questions_n: self.questions as u64,
                fact_budget: self.fact_budget as u64,
                structured_output: self.structured_output.manifest_value().to_string(),
                max_output_tokens: self.max_output_tokens.unwrap_or(0) as u64,
                chunk_bytes: chunk_bytes.clone(),
                chunk_context: self.chunk_context_value(),
                lossy: self.lossy,
                schema_digest: self.schema_digest.clone(),
                candidates: candidates_manifest_value(self.candidates).to_string(),
                vocabulary_digest: self.vocabulary_digest.clone(),
            };
            let differences = settings_differences(recorded, &current);
            if !differences.is_empty() {
                eprintln!(
                    "taguru: extract: {source}: the records were taken under different \
                     settings ({}) — completions will not match",
                    differences.join(", ")
                );
            }
        }
        if let Some(completions) = self.completions.as_mut() {
            completions.begin_document(replay_index, self.replay_mode);
        }
        // ADR 0031 §3.6: resolved once per document, after the replay
        // index is installed — every chunk (and Stage 2) of a document
        // shares one prompt, pin or not (ADR 0014's same reasoning).
        let resolved_system = self.resolve_system(source, &candidates);
        let observers = Observers {
            sink: self.diagnostics.as_ref(),
            log: attempt_log.as_ref(),
        };
        // ADR 0033 §3.5: the `overview` step — one call per chunk,
        // before any extraction, answers cached in the checkpoint —
        // and the extraction units bound to the overview they saw.
        let overview_answers: Vec<Option<OverviewAnswer>> = if self.chunk_context.overview() {
            let answers = self.overview_pass(source, &plan, &units, &checkpoints, &observers);
            if self.stop.check() {
                return Ok(Outcome::Interrupted);
            }
            answers
        } else {
            vec![None; plan.len()]
        };
        let overview = self
            .chunk_context
            .overview()
            .then(|| Overview::merge(&overview_answers));
        if let Some(overview) = overview.as_ref() {
            let discarded = checkpoints.bind_overview(source, &overview.digest);
            if discarded > 0 {
                eprintln!(
                    "taguru: extract: {source}: chunk context: the overview changed since \
                     {discarded} unit(s) were checkpointed — they are re-extracted"
                );
            }
        }
        // The `annotate` step: one block per chunk, or none under `off`
        // (and none for a chunk nothing applies to).
        let blocks: Vec<Option<ContextBlock>> = if self.chunk_context.is_on() {
            let cap = block_cap(self.chunk_bytes);
            plan.iter()
                .map(|descriptor| {
                    render_block(
                        &text,
                        &paragraph_spans,
                        &units,
                        descriptor.paragraph_first,
                        descriptor.paragraph_last,
                        &descriptor.text,
                        cap,
                        overview.as_ref(),
                        self.chunk_context
                            .ingested()
                            .then_some(&self.vocabulary_known),
                        &candidates,
                    )
                })
                .collect()
        } else {
            vec![None; plan.len()]
        };
        let chunks: Vec<String> = plan
            .iter()
            .map(|descriptor| descriptor.text.clone())
            .collect();
        let context_blocks: Vec<Option<&str>> = blocks
            .iter()
            .map(|block| block.as_ref().map(|block| block.text.as_str()))
            .collect();
        let chunk_result = self
            .extract_chunks(
                source,
                &chunks,
                &context_blocks,
                canonical_paragraphs,
                &resolved_system.text,
                &checkpoints,
                &observers,
            )
            .map_err(|message| with_resume_hint(&checkpoints, message))?;
        let mut outputs = match chunk_result {
            ChunkLoopResult::Complete(outputs) => outputs,
            // Whatever units already landed stay on disk — a rerun
            // resumes from exactly here, never further back.
            ChunkLoopResult::Interrupted => return Ok(Outcome::Interrupted),
        };
        // Issue #199 Stage 2: cross-chunk alias validation (shadowing,
        // conflicting mappings), widened by ADR 0009 §11.2 to the
        // schema's own domain/range judgment when `--schema` installed
        // one — the judgment Stage 1 cannot make per-output, only
        // merge() itself could before this issue, silently. A dangling
        // canonical is no longer corrected but mechanically pruned
        // (ADR 0013) — AFTER the corrective turns, so every corrective
        // message's item indices still match the replayed answers.
        // `--lossy` skips both, matching Stage 1's skip: merge() alone
        // decides what survives.
        if !self.lossy {
            let cross_issues = combined_cross_output_issues(&outputs, self.schema.as_deref());
            if !cross_issues.is_empty() {
                self.correct_cross_output_issues(
                    source,
                    &mut outputs,
                    cross_issues,
                    chunks.len(),
                    canonical_paragraphs,
                    &resolved_system.text,
                    &observers,
                )
                .map_err(|message| with_resume_hint(&checkpoints, message))?;
            }
            prune_unresolvable_aliases(&mut outputs);
            // #758: an alias that would rewire a name an EARLIER
            // document (or the target context) already settled on is
            // import's Conflict refusal — mechanical, on the same
            // terms as the dangling prune: nothing the model could
            // correct, so nothing a corrective turn is spent on.
            prune_claimed_aliases(&mut outputs, &self.claimed_names);
        }
        // Every removal — Stage 1's and Stage 2's — now sits on the
        // output it came from (#786); the report names each one
        // chunk-first, exactly as before.
        let chunk_total = chunks.len();
        let removed: Vec<String> = outputs
            .iter()
            .flat_map(|chunk| {
                chunk.removed.iter().map(move |removal| {
                    if chunk_total > 1 {
                        format!("chunk {}/{chunk_total} {removal}", chunk.chunk_index + 1)
                    } else {
                        removal.to_string()
                    }
                })
            })
            .collect();
        // ADR 0023: what the trace needs of each output, read off
        // before merge() consumes them, in merge()'s input order.
        let pieces: Vec<PieceOrigin> = outputs
            .iter()
            .map(|output| PieceOrigin::of(output, &run_id))
            .collect();
        let extraction = merge(
            outputs.into_iter().map(|chunk| chunk.output).collect(),
            self.questions,
            canonical_paragraphs,
        );
        let body = render_batch(
            &self.context,
            &written_source,
            self.description.as_deref(),
            &extraction,
            (!self.no_passage).then_some(text.as_str()),
            self.date,
            &self.tags,
        );
        // Both failures below keep the checkpoint file too (it is
        // cleared only once the batch has landed), so they carry the
        // same resume hint as a chunk or Stage 2 failure.
        if let Err(message) = crate::ingest::parse_batch(Cursor::new(body.as_bytes())) {
            return Err(with_resume_hint(
                &checkpoints,
                format!(
                    "the emitted batch failed self-validation \
                     ({message}) — a bug in taguru, not in the document"
                ),
            ));
        }
        if let Err(error) = crate::storage::write_atomic(&out_path, body.as_bytes()) {
            return Err(with_resume_hint(
                &checkpoints,
                format!("writing {}: {error}", out_path.display()),
            ));
        }
        // ADR 0016 (#496 S4), the recall-side half — computed here,
        // before the trace, so the trace's `uncovered` records (ADR
        // 0026, #787) are the same gaps stderr names below: every
        // sentence whose candidate pair no accepted association
        // covers. Pure over (document text, accepted associations).
        let uncovered = if self.coverage {
            let triples: Vec<[&str; 3]> = extraction
                .associations
                .iter()
                .map(|fact| {
                    [
                        fact.subject.as_str(),
                        fact.label.as_str(),
                        fact.object.as_str(),
                    ]
                })
                .collect();
            coverage_gaps(&text, &triples)
        } else {
            Vec::new()
        };
        // ADR 0023 §3.3: the trace lands right after the batch and
        // before the manifest records it — a batch the manifest knows
        // has its trace beside it (or a stderr line saying why not).
        write_trace(
            &self.out,
            &file_name,
            &render_trace(
                &run_id,
                source,
                &hash,
                &out_path,
                &plan,
                &pieces,
                &paragraph_spans
                    .iter()
                    .map(|span| &text[span.start as usize..span.end as usize])
                    .collect::<Vec<&str>>(),
                &extraction,
                &uncovered,
                &self.steering_record(&candidates, &resolved_system),
                &units,
                &blocks,
                &overview_answers,
            ),
        );
        self.manifest.record(source, &inputs, &file_name);
        // A manifest from before #730's naming change recorded the
        // un-hashed output name; the replacement just landed durably
        // under the new name, so the old file — positively this
        // source's own record, never a flatten-collision neighbor's —
        // is removed before `taguru import DIR` could read both as a
        // duplicate source.
        if let Some(previous) = &recorded_output
            && *previous != file_name
        {
            let _ = fs::remove_file(self.out.join(previous));
        }
        // The batch is durably written and manifest-recorded — the
        // checkpoint's only purpose (resuming an incomplete document)
        // no longer applies. A document that fails Stage 2/merge/
        // self-validation above instead keeps its checkpoint file: the
        // per-chunk outputs already extracted are still good.
        checkpoints.clear();
        for (label, count) in extraction.label_usage_counts() {
            *self.vocabulary.entry(label).or_insert(0) += count;
        }
        self.claimed_names.absorb_extraction(&extraction);
        // ADR 0013's accounting half: every mechanical removal is
        // named on stderr, path first — the report line below carries
        // only the count. Never-silent-drop survives as visibility.
        for reason in &removed {
            eprintln!("taguru: extract: {source}: removed: {reason}");
        }
        // ADR 0016's stderr half, from the gaps computed above.
        for gap in &uncovered {
            eprintln!("taguru: extract: {source}: uncovered: {}", gap.describe());
        }
        // ADR 0031 §3.4: the `replay_summary` record and its stderr
        // echo — a replaying document's own account of what it did.
        if self.replaying
            && let Some((replayed, live)) =
                self.completions.as_ref().map(Completions::document_counts)
        {
            eprintln!(
                "taguru: extract: {source}: replayed {replayed}/{} completions ({live} live)",
                replayed + live
            );
            if let Some(log) = attempt_log.as_ref() {
                log.write_record(&ReplaySummaryRecord {
                    kind: "replay_summary",
                    replayed,
                    live,
                });
            }
        }
        self.report(
            source,
            &extraction,
            removed.len(),
            uncovered.len(),
            &out_path,
        );
        if let Some(sink) = self.diagnostics.as_ref() {
            sink.emit_document(
                source,
                &extraction,
                removed.len(),
                uncovered.len(),
                &out_path,
            );
        }
        Ok(Outcome::Written)
    }

    /// Every chunk through the model, in order. The system prompt is
    /// fixed for the whole document: the vocabulary grows only when a
    /// document lands, so all of one document's chunks are offered the
    /// same spellings. `--parallel` only ever fans out within this one
    /// document — see [`Run::extract_chunks_concurrently`] — never
    /// across documents, since the vocabulary above accumulates
    /// document-to-document and concurrent documents could diverge on
    /// label spellings.
    ///
    /// ADR 0033 §3.5 as ADR 0034 amends it: the overview pass. One
    /// completion per chunk, in document order, asking for a synopsis
    /// of each unit opening in the chunk and the cast it introduces;
    /// the answer is cached in the checkpoint by `chunk_sha256` and
    /// reused on resume. Its ladder is escalation only (ADR 0019 —
    /// the same capped resend Stage 1 gets; there is no piece to
    /// split): a chunk whose answer the pass cannot land is reported
    /// once and recorded as an EMPTY answer (ADR 0034 §3.3) — the
    /// block carries no synopsis from it, and a resumed document
    /// neither re-asks nor re-binds its extraction units — never a
    /// failed document, since context is advisory. Every completion
    /// is a `stage: "overview"` attempt record.
    pub(super) fn overview_pass(
        &self,
        source: &str,
        plan: &[ChunkDescriptor],
        units: &[Unit],
        checkpoints: &CheckpointStore,
        observers: &Observers,
    ) -> Vec<Option<OverviewAnswer>> {
        let completions = self
            .completions
            .as_ref()
            .expect("a non-dry run built the completions");
        let system = overview_system_prompt();
        // The extraction schema is the wrong shape for this answer, so
        // the json_schema rung steps down to json_object here; the
        // other rungs are what they are.
        let rung = self.ladder.as_ref().map(|ladder| ladder.rung());
        let response_format = match rung {
            Some(Rung::JsonSchema | Rung::JsonObject) => Some(json_object_response_format()),
            Some(Rung::Prompted) | None => None,
        };
        let budget = self
            .ladder
            .as_ref()
            .and_then(|ladder| ladder.max_output_tokens);
        let mut answers = Vec::with_capacity(plan.len());
        for (index, descriptor) in plan.iter().enumerate() {
            if self.stop.check() {
                answers.resize(plan.len(), None);
                return answers;
            }
            if let Some(cached) = checkpoints.overview_answer(&descriptor.sha256) {
                answers.push(Some(cached));
                continue;
            }
            let here =
                units_opening_in(units, descriptor.paragraph_first, descriptor.paragraph_last);
            let offered: Vec<usize> = here.iter().map(|unit| unit.unit).collect();
            let user = overview_user_message(source, index, plan.len(), &descriptor.text, &here);
            let messages = [
                serde_json::json!({"role": "system", "content": &system}),
                serde_json::json!({"role": "user", "content": &user}),
            ];
            let mut options = RequestOptions {
                response_format: response_format.clone(),
                max_tokens: budget,
                fail_fast_on_timeout: false,
            };
            let mut outcome = self.overview_round(
                completions,
                observers,
                source,
                index,
                &descriptor.sha256,
                &messages,
                &options,
                &offered,
            );
            if matches!(outcome, OverviewOutcome::LengthLimited)
                && let Some(ladder) = self.ladder.as_ref()
                && budget.is_some()
            {
                let mut record = MoveRecord::blank(
                    "escalate",
                    completions.run_id(),
                    &descriptor.sha256,
                    index,
                    "the overview answer ended at the output cap; resending once at the \
                     escalated budget",
                );
                record.from_max_tokens = budget;
                record.to_max_tokens = ladder.escalated_budget();
                observers.move_event(&record);
                options.max_tokens = ladder.escalated_budget();
                outcome = self.overview_round(
                    completions,
                    observers,
                    source,
                    index,
                    &descriptor.sha256,
                    &messages,
                    &options,
                    &offered,
                );
            }
            match outcome {
                OverviewOutcome::Answered(answer) => {
                    checkpoints.record_overview(source, descriptor.sha256.clone(), answer.clone());
                    answers.push(Some(answer));
                }
                OverviewOutcome::LengthLimited => {
                    eprintln!(
                        "taguru: extract: {source}: chunk {}/{}: the overview answer was cut \
                         off at the output limit — this chunk contributes no synopsis or cast \
                         (recorded so for this document's resume; --force re-asks)",
                        index + 1,
                        plan.len()
                    );
                    checkpoints.record_overview(
                        source,
                        descriptor.sha256.clone(),
                        OverviewAnswer::default(),
                    );
                    answers.push(None);
                }
                OverviewOutcome::Failed(message) => {
                    eprintln!(
                        "taguru: extract: {source}: chunk {}/{}: overview: {message} — this \
                         chunk contributes no synopsis or cast (recorded so for this \
                         document's resume; --force re-asks)",
                        index + 1,
                        plan.len()
                    );
                    checkpoints.record_overview(
                        source,
                        descriptor.sha256.clone(),
                        OverviewAnswer::default(),
                    );
                    answers.push(None);
                }
            }
        }
        answers
    }

    /// One overview completion, recorded as a `stage: "overview"`
    /// attempt whatever it came back as.
    #[allow(clippy::too_many_arguments)]
    fn overview_round(
        &self,
        completions: &Completions,
        observers: &Observers,
        source: &str,
        chunk_index: usize,
        chunk_sha256: &str,
        messages: &[serde_json::Value],
        options: &RequestOptions,
        offered: &[usize],
    ) -> OverviewOutcome {
        let started = std::time::Instant::now();
        let attempt_ref = completions.next_attempt();
        let rung = self.ladder.as_ref().map(|ladder| ladder.rung().name());
        let record = |state: &'static str,
                      length_limited: bool,
                      response: Option<&ChatCompletion>,
                      parse_error: Option<&str>,
                      transport_retries: usize,
                      replayed_from: Option<&AttemptRef>| {
            observers.emit(
                &DiagnosticsAttempt {
                    source,
                    stage: "overview",
                    chunk_index,
                    attempt: 1,
                    attempt_ref: &attempt_ref,
                    corrects: None,
                    piece_id: chunk_sha256,
                    max_attempts: 1,
                    state,
                    length_limited,
                    elapsed: started.elapsed(),
                    response,
                    replayed_from,
                    transport_retries,
                    parse_error,
                    validation_issues: None,
                    removed_items: None,
                    piece_bytes: None,
                    requested_max_tokens: options.max_tokens,
                    rung,
                },
                messages,
            );
        };
        let response = match completions.complete(chunk_sha256, messages, options) {
            Ok(response) => response,
            Err(error) => {
                let message = error.to_string();
                record(
                    match error.kind {
                        ChatFailure::Timeout => "timeout",
                        ChatFailure::Transport => "transport",
                    },
                    false,
                    None,
                    Some(&message),
                    error.transport_retries,
                    error.replayed_from.as_ref(),
                );
                return OverviewOutcome::Failed(message);
            }
        };
        if indicates_length_limit(response.finish_reason.as_deref()) {
            record(
                "length_limited",
                true,
                Some(&response),
                Some("the overview answer was cut off at the output limit"),
                response.transport_retries,
                response.replayed_from.as_ref(),
            );
            return OverviewOutcome::LengthLimited;
        }
        if let Some(reason) = response.finish_reason.as_deref()
            && indicates_refusal(reason)
        {
            let message = format!("the provider refused the overview (finish_reason {reason})");
            record(
                "refusal",
                false,
                Some(&response),
                Some(&message),
                response.transport_retries,
                response.replayed_from.as_ref(),
            );
            return OverviewOutcome::Failed(message);
        }
        if is_empty_answer(&response.content) {
            let message = empty_answer_diagnosis();
            record(
                "empty",
                false,
                Some(&response),
                Some(&message),
                response.transport_retries,
                response.replayed_from.as_ref(),
            );
            return OverviewOutcome::Failed(message);
        }
        match parse_overview_answer(&response.content, offered) {
            Ok(answer) => {
                record(
                    "stop_valid",
                    false,
                    Some(&response),
                    None,
                    response.transport_retries,
                    response.replayed_from.as_ref(),
                );
                OverviewOutcome::Answered(answer)
            }
            Err(error) => {
                record(
                    "stop_malformed",
                    false,
                    Some(&response),
                    Some(&error),
                    response.transport_retries,
                    response.replayed_from.as_ref(),
                );
                OverviewOutcome::Failed(format!(
                    "the overview answer was not the JSON object asked for ({error})"
                ))
            }
        }
    }

    /// Issue #179: the cooperative stop flag is checked between
    /// top-level chunks here (sequential path only — see
    /// [`Run::extract_chunks_concurrently`]'s doc comment for why
    /// `--parallel` is scoped to between-documents instead). A stop
    /// observed mid-loop returns [`ChunkLoopResult::Interrupted`]
    /// immediately, keeping whatever units already landed.
    #[allow(clippy::too_many_arguments)] // one call site; the blocks ride beside the chunks
    pub(super) fn extract_chunks(
        &self,
        source: &str,
        chunks: &[String],
        context_blocks: &[Option<&str>],
        paragraph_count: usize,
        system: &str,
        checkpoints: &CheckpointStore,
        observers: &Observers,
    ) -> Result<ChunkLoopResult, String> {
        if self.parallel > 1 {
            return self.extract_chunks_concurrently(
                source,
                chunks,
                context_blocks,
                paragraph_count,
                system,
                checkpoints,
                observers,
            );
        }
        let completions = self
            .completions
            .as_ref()
            .expect("a non-dry run built the completions");
        let rules = self.item_rules(paragraph_count);
        let mut outputs = Vec::new();
        for (index, piece) in chunks.iter().enumerate() {
            if self.stop.check() {
                return Ok(ChunkLoopResult::Interrupted);
            }
            match extract_chunk_or_ladder(
                completions,
                system,
                source,
                index,
                chunks.len(),
                piece,
                context_blocks.get(index).copied().flatten(),
                &self.correction,
                self.fact_budget,
                self.ladder.as_ref(),
                rules.as_ref(),
                &self.vocabulary_allowlist,
                observers,
                checkpoints,
            ) {
                Ok(piece_outputs) => outputs.extend(piece_outputs),
                Err(message) => {
                    return Err(format!("chunk {}/{}: {message}", index + 1, chunks.len()));
                }
            }
        }
        Ok(ChunkLoopResult::Complete(outputs))
    }

    /// [`Run::extract_chunks`]'s `--parallel > 1` path: dispatches
    /// through the same claim-indices-with-a-first-failure-gate engine
    /// [`crate::registry::dispatch_chunks_concurrently`] uses for
    /// embedding refresh, so the `SeqCst`-ordering correctness argument
    /// (a worker claiming an index past a just-recorded failure must
    /// actually observe it) lives in exactly one place. This is the
    /// all-or-nothing fold: the lowest-indexed failure fails the whole
    /// document, formatted with its position, and nothing after it is
    /// intentionally dispatched — calls already in flight when the
    /// failure lands simply finish and are discarded.
    ///
    /// Issue #179's cooperative stop is deliberately NOT checked
    /// mid-dispatch here: `dispatch_chunks_concurrently` is shared with
    /// unrelated embedding-refresh code, and threading a stop flag
    /// through it would widen that primitive's contract for one caller.
    /// Under `--parallel`, a stop request only takes effect between
    /// documents — every already-claimed chunk in this document runs to
    /// completion (and gets checkpointed) before the run notices the
    /// request.
    #[allow(clippy::too_many_arguments)] // mirrors `extract_chunks`
    pub(super) fn extract_chunks_concurrently(
        &self,
        source: &str,
        chunks: &[String],
        context_blocks: &[Option<&str>],
        paragraph_count: usize,
        system: &str,
        checkpoints: &CheckpointStore,
        observers: &Observers,
    ) -> Result<ChunkLoopResult, String> {
        let completions = self
            .completions
            .as_ref()
            .expect("a non-dry run built the completions");
        let rules = self.item_rules(paragraph_count);
        let indexed: Vec<(usize, &String)> = chunks.iter().enumerate().collect();
        let outcomes = crate::registry::dispatch_chunks_concurrently(
            &indexed,
            self.parallel,
            |&(index, piece)| {
                extract_chunk_or_ladder(
                    completions,
                    system,
                    source,
                    index,
                    chunks.len(),
                    piece,
                    context_blocks.get(index).copied().flatten(),
                    &self.correction,
                    self.fact_budget,
                    self.ladder.as_ref(),
                    rules.as_ref(),
                    &self.vocabulary_allowlist,
                    observers,
                    checkpoints,
                )
            },
        );

        let mut outputs = Vec::new();
        for (index, outcome) in outcomes.into_iter().enumerate() {
            let outcome = outcome.expect("every index up to the first failure was dispatched");
            match outcome {
                Ok(piece_outputs) => outputs.extend(piece_outputs),
                Err(message) => {
                    return Err(format!("chunk {}/{}: {message}", index + 1, chunks.len()));
                }
            }
        }
        Ok(ChunkLoopResult::Complete(outputs))
    }

    /// Issue #199 Stage 2: one targeted corrective turn per output
    /// `cross_output_issues` flagged, rebuilding THAT output's own
    /// conversation base (never the whole document's) and replaying
    /// its own final answer as the prior bad turn — Stage 1's
    /// rebuild-not-accumulate discipline, at the output level. Bounded
    /// to exactly one extra call per offending output regardless of
    /// `max_attempts` (the issue's "one targeted corrective turn"),
    /// plus ADR 0032's (#811) one escalated resend when that turn
    /// ends at the output cap under a configured budget: a
    /// still-invalid, refused, or empty reply fails the source
    /// outright — Stage 2 never splits and never loops a second
    /// round. An alias issue that the turn leaves standing — or that
    /// a correction cut off at the ladder's top never delivered — is
    /// removed with accounting instead (ADR 0022, #763) — the records
    /// are returned for the report line, stderr, and the sidecar; a
    /// standing issue about anything but an alias still fails the
    /// source.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn correct_cross_output_issues(
        &self,
        source: &str,
        outputs: &mut [ChunkOutput],
        cross_issues: Vec<(usize, Vec<String>)>,
        chunk_total: usize,
        paragraph_count: usize,
        system: &str,
        observers: &Observers,
    ) -> Result<(), String> {
        let completions = self
            .completions
            .as_ref()
            .expect("a non-dry run built the completions");
        let options = RequestOptions {
            response_format: self.ladder.as_ref().and_then(LadderConfig::response_format),
            max_tokens: self
                .ladder
                .as_ref()
                .and_then(|ladder| ladder.max_output_tokens),
            // Stage 2 is a corrective turn, not a piece the ladder could
            // split: a timeout here keeps the ordinary retry discipline.
            fail_fast_on_timeout: false,
        };
        // ADR 0031 §3.2: read fresh per Stage 2 call — a demote (ADR
        // 0021) can move the run-wide rung between Stage 1 and here.
        let rung_name = self.ladder.as_ref().map(|ladder| ladder.rung().name());
        let rules = self.item_rules(paragraph_count);
        let budget = self
            .ladder
            .as_ref()
            .and_then(|ladder| ladder.max_output_tokens);
        for (output_index, issues) in cross_issues {
            let (chunk_index, piece_id, corrected_attempt, user, answer) = {
                let chunk = &outputs[output_index];
                (
                    chunk.chunk_index,
                    chunk.piece_id.clone(),
                    chunk.attempt.clone(),
                    chunk.user.clone(),
                    chunk.answer.clone(),
                )
            };
            let label = format!("chunk {}/{chunk_total}", chunk_index + 1);
            let messages = [
                serde_json::json!({"role": "system", "content": system}),
                serde_json::json!({"role": "user", "content": &user}),
                corrective_assistant_turn(&answer, self.correction.corrective_context_cap),
                serde_json::json!({
                    "role": "user",
                    "content": corrective_validation_message(&issues),
                }),
            ];
            let round = CrossChunkRound {
                completions,
                observers,
                source,
                chunk_index,
                piece_id: &piece_id,
                corrected_attempt: corrected_attempt.as_ref(),
                label: &label,
                messages: &messages,
                user: &user,
                rules: rules.as_ref(),
                vocabulary: &self.vocabulary_allowlist,
                rung: rung_name,
            };
            let mut outcome = round.complete(&options);
            // ADR 0032 (#811): the corrective turn gets ADR 0019's one
            // escalation exactly as a Stage 1 piece does — a neutral
            // resend of the same corrective ask at the escalated
            // budget, the cut-off answer discarded. Only with a budget
            // configured; the ladder escalates nothing otherwise.
            if matches!(outcome, CrossChunkOutcome::LengthLimited) && budget.is_some() {
                let ladder = self
                    .ladder
                    .as_ref()
                    .expect("a configured budget means a ladder");
                let mut record = MoveRecord::blank(
                    "escalate",
                    completions.run_id(),
                    &piece_id,
                    chunk_index,
                    "the cross-chunk correction ended at the output cap; resending once at \
                     the escalated budget",
                );
                record.from_max_tokens = budget;
                record.to_max_tokens = ladder.escalated_budget();
                observers.move_event(&record);
                let escalated = RequestOptions {
                    max_tokens: ladder.escalated_budget(),
                    ..options.clone()
                };
                outcome = round.complete(&escalated);
            }
            match outcome {
                CrossChunkOutcome::Valid(corrected) => outputs[output_index] = *corrected,
                CrossChunkOutcome::Failed(message) => return Err(message),
                CrossChunkOutcome::LengthLimited => {
                    // ADR 0032: a correction the ladder cannot land is
                    // a correction that left its issues standing — the
                    // output keeps its accepted Stage 1 answer and the
                    // re-check below rules on what stands (ADR 0022:
                    // an alias is removed with accounting; anything
                    // else fails the source). Never a truncated
                    // correction imported.
                    eprintln!(
                        "taguru: extract: {source}: {label}: the cross-chunk correction was \
                         cut off at the output limit{} — the truncated answer is discarded \
                         and its issues stand as uncorrected",
                        match budget {
                            Some(_) => " even at the escalated budget",
                            None => "",
                        }
                    );
                }
            }
        }
        // Re-check rather than trust the single corrective turn blindly:
        // a correction can rename an association another output's alias
        // depended on, introducing a FRESH cross-output issue. This is
        // the bounded re-check, not a second round. ADR 0022: an alias
        // issue still standing is removed with accounting (the loop
        // re-checks after each removal pass — removing aliases can only
        // shrink the issue set, so it ends); anything else standing
        // fails the source.
        loop {
            let remaining = combined_cross_output_issues(outputs, self.schema.as_deref());
            if remaining.is_empty() {
                break;
            }
            match prune_uncorrected_aliases(outputs, remaining) {
                Ok(0) => break,
                Ok(_) => {}
                Err((output_index, issues)) => {
                    let chunk_index = outputs[output_index].chunk_index;
                    return Err(format!(
                        "chunk {}/{chunk_total}: still has {} cross-chunk issue(s) after \
                         correction: {}",
                        chunk_index + 1,
                        issues.len(),
                        issues.join("; ")
                    ));
                }
            }
        }
        Ok(())
    }

    /// A skipped document still contributes its labels, so later
    /// documents keep reusing the same vocabulary — and its names to
    /// the claim set (#758), so a later document's alias cannot rewire
    /// what a skipped one already wrote. Its batch file
    /// already exists and the manifest says it matches this source, but
    /// the file itself could still be unreadable or corrupt (truncated
    /// by an interrupted write from an older version, hand-edited,
    /// bit-rotted) — that failure is reported, not swallowed: a silent
    /// miss here would shrink every LATER document's "relation labels
    /// already in use" prompt with no diagnostic at all, degrading
    /// label reuse for the rest of the run without a trace. The parsed
    /// batch is returned (`None` on that failure) so the caller's
    /// coverage check (ADR 0016) reuses this one read instead of
    /// parsing the file twice.
    pub(super) fn absorb_vocabulary(
        &mut self,
        source: &str,
        out_path: &Path,
    ) -> Option<crate::ingest::Batch> {
        match fs::File::open(out_path)
            .map_err(|error| error.to_string())
            .and_then(|file| crate::ingest::parse_batch(std::io::BufReader::new(file)))
        {
            Ok(batch) => {
                for (label, count) in batch.label_usage_counts() {
                    *self.vocabulary.entry(label).or_insert(0) += count;
                }
                self.claimed_names.absorb_batch(&batch);
                Some(batch)
            }
            Err(error) => {
                eprintln!(
                    "taguru: extract: {source}: {}: unreadable, so its labels were not \
                     absorbed into this run's vocabulary: {error}",
                    out_path.display()
                );
                None
            }
        }
    }

    /// The one report line a written document earns.
    pub(super) fn report(
        &self,
        source: &str,
        extraction: &Extraction,
        removed: usize,
        uncovered: usize,
        out_path: &Path,
    ) {
        let mut notes = String::new();
        if extraction.duplicates > 0 {
            notes.push_str(&format!(", {} duplicate(s) folded", extraction.duplicates));
        }
        if removed > 0 {
            // ADR 0013: mechanically removed, each named on stderr —
            // distinct from `--lossy`'s validate-nothing drops and
            // from merge()'s policy trims below.
            notes.push_str(&format!(
                ", {removed} item(s) removed (mechanical validation)"
            ));
        }
        if uncovered > 0 {
            // ADR 0016: sentences whose candidate pair nothing covers,
            // each quoted on stderr — a recall accounting, never a
            // failure: the batch above was still written whole.
            notes.push_str(&format!(", {uncovered} sentence(s) uncovered (coverage)"));
        }
        if extraction.dropped > 0 {
            // Under the default (strict) mode, a surviving `dropped`
            // count is only ever merge()'s policy trim (duplicate
            // overflow, questions_cap == 0 volunteers) — issue #199's
            // validity issues are corrected or fail the source before
            // merge() ever runs. `--lossy` restores the pre-#199
            // drop-and-proceed behavior, so its drops are marked
            // explicitly: a report line must never look identical
            // between a policy trim and a silently discarded fact.
            let marker = if self.lossy { " (--lossy)" } else { "" };
            notes.push_str(&format!(", {} item(s) dropped{marker}", extraction.dropped));
        }
        println!(
            "{source}: {} association(s), {} alias(es){}{}{notes} → {}",
            extraction.associations.len(),
            extraction.concepts.len() + extraction.labels.len(),
            if self.no_passage { "" } else { ", passage" },
            if extraction.questions.is_empty() {
                String::new()
            } else {
                format!(", {} question(s)", extraction.questions.len())
            },
            out_path.display()
        );
    }
}

/// The document re-rendered for question prompts: every canonical
/// paragraph (the server's own split) prefixed with its bracketed
/// number, so the model's `paragraph` references land on exactly the
/// indexes the server validates against. A paragraph too large to fit a
/// single `cap`-byte chunk is pre-split into pieces that EACH repeat the
/// number — otherwise the byte split in [`chunk`] would carry a
/// paragraph's continuation to the model as unlabeled text, and any
/// `paragraph` reference the model drew from it would be a guess. Prompt
/// input only — the passage stays the verbatim document.
pub(super) fn labeled_document(text: &str, cap: usize) -> String {
    let mut blocks = Vec::new();
    for span in crate::paragraph::split(text) {
        let label = format!("[{}] ", span.index);
        let content = &text[span.start as usize..span.end as usize];
        // Reserve the label's room on every piece so a re-labeled
        // continuation still fits the chunk that will carry it, leaving
        // chunk()'s own oversize split with nothing left to cut (and so
        // no piece to strip the label from).
        let piece_cap = cap.saturating_sub(label.len()).max(1);
        for piece in split_oversized(content, piece_cap) {
            // split_oversized cuts just after a newline, so an interior
            // piece ends in one; trim it, or joining blocks with "\n\n"
            // would blur the paragraph boundary into a triple break. A
            // whole (non-oversized) paragraph's span carries no trailing
            // newline, so the common path is untouched.
            blocks.push(format!("{label}{}", piece.trim_end_matches('\n')));
        }
    }
    blocks.join("\n\n")
}

/// A failing document keeps its checkpoint file, so the failure line
/// says what a plain rerun resumes from (#763) — the operator's reflex
/// after a late-chunk failure was `--force`, which is exactly the flag
/// that discards those units.
pub(super) fn with_resume_hint(checkpoints: &CheckpointStore, message: String) -> String {
    match checkpoints.unit_count() {
        0 => message,
        units => format!(
            "{message} ({units} extracted unit(s) are checkpointed — a rerun without \
             --force resumes from them)"
        ),
    }
}

/// What one Stage 2 corrective completion came back as. `Valid` is
/// the corrected output the caller installs in place of the accepted
/// Stage 1 one; `LengthLimited` is the one outcome the ladder can act
/// on (ADR 0032: escalate once, then treat the issues as standing);
/// `Failed` carries the source-failing message for everything else —
/// transport, timeout, refusal, empty, malformed, or still invalid.
pub(super) enum CrossChunkOutcome {
    Valid(Box<ChunkOutput>),
    LengthLimited,
    Failed(String),
}

/// One output's corrective ask, bundled so the escalated resend
/// (ADR 0032) sends exactly what the first round did — same messages,
/// same attribution — with only the request options changed.
pub(super) struct CrossChunkRound<'a> {
    pub(super) completions: &'a Completions,
    pub(super) observers: &'a Observers<'a>,
    pub(super) source: &'a str,
    pub(super) chunk_index: usize,
    pub(super) piece_id: &'a str,
    pub(super) corrected_attempt: Option<&'a AttemptRef>,
    /// `chunk i/n`, the prefix of every failure message.
    pub(super) label: &'a str,
    pub(super) messages: &'a [serde_json::Value],
    /// The output's own user message — carried into the corrected
    /// `ChunkOutput` so a later correction can rebuild the same base.
    pub(super) user: &'a str,
    pub(super) rules: Option<&'a ItemRules>,
    pub(super) vocabulary: &'a HashSet<String>,
    pub(super) rung: Option<&'static str>,
}

impl CrossChunkRound<'_> {
    /// One completion of the corrective ask under `options`, every
    /// outcome recorded as a `stage: "cross_chunk"` attempt (ADR 0025)
    /// exactly as before ADR 0032 — a round is one attempt, so the
    /// escalated resend is a second round with its own record whose
    /// `requested_max_tokens` shows the cap it carried, the `escalate`
    /// move record between them (ADR 0029).
    pub(super) fn complete(&self, options: &RequestOptions) -> CrossChunkOutcome {
        let label = self.label;
        let started = std::time::Instant::now();
        let attempt_ref = self.completions.next_attempt();
        let record = |state: &'static str,
                      length_limited: bool,
                      elapsed: std::time::Duration,
                      response: Option<&ChatCompletion>,
                      parse_error: Option<&str>,
                      validation_issues: Option<&[String]>,
                      removed_items: Option<Vec<String>>,
                      transport_retries: usize,
                      replayed_from: Option<&AttemptRef>| {
            self.observers.emit(
                &DiagnosticsAttempt {
                    source: self.source,
                    stage: "cross_chunk",
                    chunk_index: self.chunk_index,
                    attempt: 1,
                    attempt_ref: &attempt_ref,
                    corrects: self.corrected_attempt,
                    piece_id: self.piece_id,
                    max_attempts: 1,
                    state,
                    length_limited,
                    elapsed,
                    response,
                    replayed_from,
                    transport_retries,
                    parse_error,
                    validation_issues,
                    removed_items,
                    piece_bytes: None,
                    requested_max_tokens: options.max_tokens,
                    rung: self.rung,
                },
                self.messages,
            );
        };
        let response = match self
            .completions
            .complete(self.piece_id, self.messages, options)
        {
            Ok(response) => response,
            Err(error) => {
                let message = error.to_string();
                record(
                    match error.kind {
                        ChatFailure::Timeout => "timeout",
                        ChatFailure::Transport => "transport",
                    },
                    false,
                    started.elapsed(),
                    None,
                    Some(&message),
                    None,
                    None,
                    error.transport_retries,
                    error.replayed_from.as_ref(),
                );
                return CrossChunkOutcome::Failed(format!("{label}: {error}"));
            }
        };
        let elapsed = started.elapsed();
        let transport_retries = response.transport_retries;
        let replayed_from = response.replayed_from.as_ref();
        if indicates_length_limit(response.finish_reason.as_deref()) {
            record(
                "length_limited",
                true,
                elapsed,
                Some(&response),
                Some("the cross-chunk correction was cut off at the output limit"),
                None,
                None,
                transport_retries,
                replayed_from,
            );
            return CrossChunkOutcome::LengthLimited;
        }
        if let Some(reason) = response.finish_reason.as_deref()
            && indicates_refusal(reason)
        {
            let message =
                format!("the provider refused the cross-chunk correction (finish_reason {reason})");
            record(
                "refusal",
                false,
                elapsed,
                Some(&response),
                Some(&message),
                None,
                None,
                transport_retries,
                replayed_from,
            );
            return CrossChunkOutcome::Failed(format!("{label}: {message}"));
        }
        if is_empty_answer(&response.content) {
            let message = empty_answer_diagnosis();
            record(
                "empty",
                false,
                elapsed,
                Some(&response),
                Some(&message),
                None,
                None,
                transport_retries,
                replayed_from,
            );
            return CrossChunkOutcome::Failed(format!("{label}: {message}"));
        }
        match evaluate_answer(
            &response.content,
            self.rules,
            user_message_document(self.user),
            self.vocabulary,
        ) {
            Ok(evaluated) => {
                record(
                    "stop_valid",
                    false,
                    elapsed,
                    Some(&response),
                    None,
                    None,
                    removed_item_texts(&evaluated.removed),
                    transport_retries,
                    replayed_from,
                );
                CrossChunkOutcome::Valid(Box::new(ChunkOutput {
                    output: evaluated.output,
                    chunk_index: self.chunk_index,
                    piece_id: self.piece_id.to_string(),
                    attempt: Some(attempt_ref),
                    user: self.user.to_string(),
                    answer: response.content,
                    removed: evaluated.removed,
                    unparsed: evaluated.unparsed,
                }))
            }
            Err(AnswerFault::Syntax(error)) => {
                record(
                    "stop_malformed",
                    false,
                    elapsed,
                    Some(&response),
                    Some(&error),
                    None,
                    None,
                    transport_retries,
                    replayed_from,
                );
                CrossChunkOutcome::Failed(format!(
                    "{label}: the cross-chunk correction was not the JSON object asked for \
                     ({error})"
                ))
            }
            Err(AnswerFault::Invalid(issues)) => {
                let message = format!(
                    "the cross-chunk correction still left {} invalid item(s) uncorrected: {}",
                    issues.len(),
                    issues.join("; ")
                );
                record(
                    "stop_malformed",
                    false,
                    elapsed,
                    Some(&response),
                    Some(&message),
                    Some(&issues),
                    None,
                    transport_retries,
                    replayed_from,
                );
                CrossChunkOutcome::Failed(format!("{label}: {message}"))
            }
        }
    }
}

/// What one overview completion came back as (ADR 0033 §3.5).
pub(super) enum OverviewOutcome {
    Answered(OverviewAnswer),
    LengthLimited,
    Failed(String),
}
