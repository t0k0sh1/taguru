//! `Run`: one extraction run's settled configuration plus everything
//! that accumulates across documents, and the per-document/per-chunk
//! pipeline (`impl Run`) that drives it.

use super::*;

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
    /// `Some` exactly when a mechanism or an output budget is engaged
    /// on a live run: the §7 ladder replaces the legacy corrective
    /// loop. `None` under all-defaults — byte-for-byte today's
    /// behavior — and under `--dry-run`.
    pub(super) ladder: Option<LadderConfig>,
    pub(super) out: PathBuf,
    /// `None` exactly under `--dry-run`, which must call nothing.
    pub(super) client: Option<ChatClient>,
    pub(super) model_name: String,
    pub(super) manifest: Manifest,
    pub(super) vocabulary: BTreeSet<String>,
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
}

impl Run {
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
        if self.force {
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
            lossy: self.lossy,
            schema_digest: &self.schema_digest,
            candidates: candidates_manifest_value(self.candidates),
            vocabulary_digest: &self.vocabulary_digest,
            source_id: source_id_value,
            date: self.date.unwrap_or(0),
            tags: &self.tags,
        };
        if !self.force
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
        let canonical_paragraphs = crate::paragraph::split(&text).len();
        // ADR 0014: candidate names come from the WHOLE document, once
        // — every chunk is offered the same list (the vocabulary
        // discipline's reasoning, one level down), and the corrective
        // path rebuilds the identical prompt.
        let candidates = if self.candidates {
            candidate_terms(&text)
        } else {
            Vec::new()
        };
        let plan = chunk_plan(&text);
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
        let chunks: Vec<String> = plan.into_iter().map(|descriptor| descriptor.text).collect();
        let chunk_result = self.extract_chunks(
            source,
            &chunks,
            canonical_paragraphs,
            &candidates,
            &checkpoints,
        )?;
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
        let mut removed = Vec::new();
        if !self.lossy {
            let cross_issues = combined_cross_output_issues(&outputs, self.schema.as_deref());
            if !cross_issues.is_empty() {
                self.correct_cross_output_issues(
                    source,
                    &mut outputs,
                    cross_issues,
                    chunks.len(),
                    canonical_paragraphs,
                    &candidates,
                )?;
            }
            let chunk_total = chunks.len();
            for chunk in &outputs {
                for reason in &chunk.removed {
                    removed.push(if chunk_total > 1 {
                        format!("chunk {}/{chunk_total} {reason}", chunk.chunk_index + 1)
                    } else {
                        reason.clone()
                    });
                }
            }
            removed.extend(prune_unresolvable_aliases(&mut outputs, chunk_total));
            // #758: an alias that would rewire a name an EARLIER
            // document (or the target context) already settled on is
            // import's Conflict refusal — mechanical, on the same
            // terms as the dangling prune: nothing the model could
            // correct, so nothing a corrective turn is spent on.
            removed.extend(prune_claimed_aliases(
                &mut outputs,
                chunk_total,
                &self.claimed_names,
            ));
        }
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
        if let Err(message) = crate::ingest::parse_batch(Cursor::new(body.as_bytes())) {
            return Err(format!(
                "the emitted batch failed self-validation \
                 ({message}) — a bug in taguru, not in the document"
            ));
        }
        if let Err(error) = crate::storage::write_atomic(&out_path, body.as_bytes()) {
            return Err(format!("writing {}: {error}", out_path.display()));
        }
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
        self.vocabulary.extend(extraction.label_vocabulary());
        self.claimed_names.absorb_extraction(&extraction);
        // ADR 0013's accounting half: every mechanical removal is
        // named on stderr, path first — the report line below carries
        // only the count. Never-silent-drop survives as visibility.
        for reason in &removed {
            eprintln!("taguru: extract: {source}: removed: {reason}");
        }
        // ADR 0016 (#496 S4), the recall-side half: every sentence
        // whose candidate pair no accepted association covers, named
        // on stderr the same way — the report line carries the count.
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
        for gap in &uncovered {
            eprintln!("taguru: extract: {source}: uncovered: {}", gap.describe());
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
    /// Issue #179: the cooperative stop flag is checked between
    /// top-level chunks here (sequential path only — see
    /// [`Run::extract_chunks_concurrently`]'s doc comment for why
    /// `--parallel` is scoped to between-documents instead). A stop
    /// observed mid-loop returns [`ChunkLoopResult::Interrupted`]
    /// immediately, keeping whatever units already landed.
    pub(super) fn extract_chunks(
        &self,
        source: &str,
        chunks: &[String],
        paragraph_count: usize,
        candidates: &[String],
        checkpoints: &CheckpointStore,
    ) -> Result<ChunkLoopResult, String> {
        if self.parallel > 1 {
            return self.extract_chunks_concurrently(
                source,
                chunks,
                paragraph_count,
                candidates,
                checkpoints,
            );
        }
        let client = self
            .client
            .as_ref()
            .expect("a non-dry run built the client");
        let system = system_prompt(
            &self.vocabulary,
            self.questions,
            self.fact_budget,
            self.schema.as_deref(),
            &self.vocabulary_names,
            candidates,
        );
        let rules = self.item_rules(paragraph_count);
        let mut outputs = Vec::new();
        for (index, piece) in chunks.iter().enumerate() {
            if self.stop.check() {
                return Ok(ChunkLoopResult::Interrupted);
            }
            match extract_chunk_or_ladder(
                client,
                &system,
                source,
                index,
                chunks.len(),
                piece,
                &self.correction,
                self.fact_budget,
                self.ladder.as_ref(),
                rules.as_ref(),
                &self.vocabulary_allowlist,
                self.diagnostics.as_ref(),
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
    pub(super) fn extract_chunks_concurrently(
        &self,
        source: &str,
        chunks: &[String],
        paragraph_count: usize,
        candidates: &[String],
        checkpoints: &CheckpointStore,
    ) -> Result<ChunkLoopResult, String> {
        let client = self
            .client
            .as_ref()
            .expect("a non-dry run built the client");
        let system = system_prompt(
            &self.vocabulary,
            self.questions,
            self.fact_budget,
            self.schema.as_deref(),
            &self.vocabulary_names,
            candidates,
        );
        let rules = self.item_rules(paragraph_count);
        let indexed: Vec<(usize, &String)> = chunks.iter().enumerate().collect();
        let outcomes = crate::registry::dispatch_chunks_concurrently(
            &indexed,
            self.parallel,
            |&(index, piece)| {
                extract_chunk_or_ladder(
                    client,
                    &system,
                    source,
                    index,
                    chunks.len(),
                    piece,
                    &self.correction,
                    self.fact_budget,
                    self.ladder.as_ref(),
                    rules.as_ref(),
                    &self.vocabulary_allowlist,
                    self.diagnostics.as_ref(),
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
    /// `max_attempts` (the issue's "one targeted corrective turn"): a
    /// still-invalid, still-cross-conflicting, length-limited,
    /// refused, or empty reply fails the source outright — Stage 2
    /// never splits and never loops a second round.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn correct_cross_output_issues(
        &self,
        source: &str,
        outputs: &mut [ChunkOutput],
        cross_issues: Vec<(usize, Vec<String>)>,
        chunk_total: usize,
        paragraph_count: usize,
        candidates: &[String],
    ) -> Result<(), String> {
        let client = self
            .client
            .as_ref()
            .expect("a non-dry run built the client");
        let system = system_prompt(
            &self.vocabulary,
            self.questions,
            self.fact_budget,
            self.schema.as_deref(),
            &self.vocabulary_names,
            candidates,
        );
        let options = RequestOptions {
            response_format: self
                .ladder
                .as_ref()
                .and_then(|ladder| ladder.response_format.clone()),
            max_tokens: self
                .ladder
                .as_ref()
                .and_then(|ladder| ladder.max_output_tokens),
        };
        let rules = self.item_rules(paragraph_count);
        let sink = self.diagnostics.as_ref();
        for (output_index, issues) in cross_issues {
            let (chunk_index, user, answer) = {
                let chunk = &outputs[output_index];
                (chunk.chunk_index, chunk.user.clone(), chunk.answer.clone())
            };
            let label = format!("chunk {}/{chunk_total}", chunk_index + 1);
            let messages = [
                serde_json::json!({"role": "system", "content": &system}),
                serde_json::json!({"role": "user", "content": &user}),
                corrective_assistant_turn(&answer, self.correction.corrective_context_cap),
                serde_json::json!({
                    "role": "user",
                    "content": corrective_validation_message(&issues),
                }),
            ];
            let started = std::time::Instant::now();
            let response = match client.complete(&messages, &options) {
                Ok(response) => response,
                Err(error) => {
                    if let Some(sink) = sink {
                        let message = error.to_string();
                        sink.emit(DiagnosticsAttempt {
                            source,
                            stage: "cross_chunk",
                            chunk_index,
                            attempt: 1,
                            max_attempts: 1,
                            state: match error.kind {
                                ChatFailure::Timeout => "timeout",
                                ChatFailure::Transport => "transport",
                            },
                            length_limited: false,
                            elapsed: started.elapsed(),
                            response: None,
                            parse_error: Some(&message),
                            validation_issues: None,
                            removed_items: None,
                            piece_bytes: None,
                            requested_max_tokens: options.max_tokens,
                        });
                    }
                    return Err(format!("{label}: {error}"));
                }
            };
            let elapsed = started.elapsed();
            if indicates_length_limit(response.finish_reason.as_deref()) {
                if let Some(sink) = sink {
                    let message =
                        "the cross-chunk correction was cut off at the output limit".to_string();
                    sink.emit(DiagnosticsAttempt {
                        source,
                        stage: "cross_chunk",
                        chunk_index,
                        attempt: 1,
                        max_attempts: 1,
                        state: "length_limited",
                        length_limited: true,
                        elapsed,
                        response: Some(&response),
                        parse_error: Some(&message),
                        validation_issues: None,
                        removed_items: None,
                        piece_bytes: None,
                        requested_max_tokens: options.max_tokens,
                    });
                }
                return Err(format!(
                    "{label}: the cross-chunk correction was cut off at the output \
                     limit — failing the source rather than importing a truncated correction"
                ));
            }
            if let Some(reason) = response.finish_reason.as_deref()
                && indicates_refusal(reason)
            {
                if let Some(sink) = sink {
                    let message = format!(
                        "the provider refused the cross-chunk correction \
                         (finish_reason {reason})"
                    );
                    sink.emit(DiagnosticsAttempt {
                        source,
                        stage: "cross_chunk",
                        chunk_index,
                        attempt: 1,
                        max_attempts: 1,
                        state: "refusal",
                        length_limited: false,
                        elapsed,
                        response: Some(&response),
                        parse_error: Some(&message),
                        validation_issues: None,
                        removed_items: None,
                        piece_bytes: None,
                        requested_max_tokens: options.max_tokens,
                    });
                }
                return Err(format!(
                    "{label}: the provider refused the cross-chunk correction \
                     (finish_reason {reason})"
                ));
            }
            if is_empty_answer(&response.content) {
                if let Some(sink) = sink {
                    let message = empty_answer_diagnosis();
                    sink.emit(DiagnosticsAttempt {
                        source,
                        stage: "cross_chunk",
                        chunk_index,
                        attempt: 1,
                        max_attempts: 1,
                        state: "empty",
                        length_limited: false,
                        elapsed,
                        response: Some(&response),
                        parse_error: Some(&message),
                        validation_issues: None,
                        removed_items: None,
                        piece_bytes: None,
                        requested_max_tokens: options.max_tokens,
                    });
                }
                return Err(format!("{label}: {}", empty_answer_diagnosis()));
            }
            match evaluate_answer(
                &response.content,
                rules.as_ref(),
                user_message_document(&user),
                &self.vocabulary_allowlist,
            ) {
                Ok(evaluated) => {
                    if let Some(sink) = sink {
                        sink.emit(DiagnosticsAttempt {
                            source,
                            stage: "cross_chunk",
                            chunk_index,
                            attempt: 1,
                            max_attempts: 1,
                            state: "stop_valid",
                            length_limited: false,
                            elapsed,
                            response: Some(&response),
                            parse_error: None,
                            validation_issues: None,
                            removed_items: (!evaluated.removed.is_empty())
                                .then_some(evaluated.removed.as_slice()),
                            piece_bytes: None,
                            requested_max_tokens: options.max_tokens,
                        });
                    }
                    outputs[output_index] = ChunkOutput {
                        output: evaluated.output,
                        chunk_index,
                        user,
                        answer: response.content,
                        removed: evaluated.removed,
                    };
                }
                Err(AnswerFault::Syntax(error)) => {
                    if let Some(sink) = sink {
                        sink.emit(DiagnosticsAttempt {
                            source,
                            stage: "cross_chunk",
                            chunk_index,
                            attempt: 1,
                            max_attempts: 1,
                            state: "stop_malformed",
                            length_limited: false,
                            elapsed,
                            response: Some(&response),
                            parse_error: Some(&error),
                            validation_issues: None,
                            removed_items: None,
                            piece_bytes: None,
                            requested_max_tokens: options.max_tokens,
                        });
                    }
                    return Err(format!(
                        "{label}: the cross-chunk correction was not the JSON object \
                         asked for ({error})"
                    ));
                }
                Err(AnswerFault::Invalid(issues)) => {
                    if let Some(sink) = sink {
                        let message = format!(
                            "the cross-chunk correction still left {} invalid item(s) \
                             uncorrected: {}",
                            issues.len(),
                            issues.join("; ")
                        );
                        sink.emit(DiagnosticsAttempt {
                            source,
                            stage: "cross_chunk",
                            chunk_index,
                            attempt: 1,
                            max_attempts: 1,
                            state: "stop_malformed",
                            length_limited: false,
                            elapsed,
                            response: Some(&response),
                            parse_error: Some(&message),
                            validation_issues: Some(&issues),
                            removed_items: None,
                            piece_bytes: None,
                            requested_max_tokens: options.max_tokens,
                        });
                    }
                    return Err(format!(
                        "{label}: the cross-chunk correction still left {} invalid \
                         item(s) uncorrected: {}",
                        issues.len(),
                        issues.join("; ")
                    ));
                }
            }
        }
        // Re-check rather than trust the single corrective turn blindly:
        // a correction can rename an association another output's alias
        // depended on, introducing a FRESH cross-output issue. This is
        // the bounded re-check, not a second round — any issue here
        // fails the source.
        if let Some((output_index, issues)) =
            combined_cross_output_issues(outputs, self.schema.as_deref())
                .into_iter()
                .next()
        {
            let chunk_index = outputs[output_index].chunk_index;
            return Err(format!(
                "chunk {}/{chunk_total}: still has {} cross-chunk issue(s) after \
                 correction: {}",
                chunk_index + 1,
                issues.len(),
                issues.join("; ")
            ));
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
                self.vocabulary.extend(batch.label_vocabulary());
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
