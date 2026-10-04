use super::*;

impl AppState {
    /// Registers an empty `context` under a freshly minted id and
    /// persists it immediately, so its existence (and description)
    /// survives a crash from the moment the create call returns —
    /// which hands back the minted id, the only address the wire has
    /// for the new `context` (#964). A persistence failure fails the
    /// create.
    ///
    /// No uniqueness check: names are display strings and duplicates
    /// are allowed (issue #961 decision 1) — `POST /contexts` of a
    /// name already in use mints a second, distinct `context`. That is
    /// also why no reservation set is needed here anymore: the stem is
    /// a UUID minted right here, so no concurrent create, delete, or
    /// rename can be working on the same file family. The import
    /// header's `create` block brings its own id and goes through
    /// [`AppState::create_if_absent`] instead.
    ///
    /// The registry lock is NOT held across the disk work
    /// (`save_files`' fsyncs — seconds on slow storage, behind which
    /// every operation on every `context` would otherwise stall): the
    /// files are written unlocked and the entry lands in one short
    /// critical section afterwards.
    ///
    /// No stale-leftover sweep, unlike every release before ids: the
    /// stem is a UUID minted right here, so no earlier generation —
    /// half-deleted, half-restored, or otherwise — can have files,
    /// markers, or a WAL sitting under it. (What CAN linger is the
    /// reverse: a create that crashes between `write_meta` and the
    /// image landing leaves an orphan sidecar under a stem no boot
    /// will ever register, invisible and inert. The old
    /// same-name-same-stem model recycled those on the next create;
    /// the id model just leaves a few bytes behind.)
    pub fn create(&self, name: &str, meta: ContextMeta) -> Result<String, CreateError> {
        let id = mint_context_id();
        self.create_registered(&id, name, meta)?;
        Ok(id)
    }

    /// The body shared by [`Self::create`] (a minted `id`) and
    /// [`Self::create_if_absent`] (the import header's own `id`):
    /// persist the family under `id`, then register it.
    fn create_registered(
        &self,
        id: &str,
        name: &str,
        meta: ContextMeta,
    ) -> Result<(), CreateError> {
        // An empty name would render every listing row and log line
        // blank — refuse it at the lowest boundary, so no entrance
        // (import, direct call) can conjure one.
        if name.is_empty() {
            return Err(CreateError::InvalidName);
        }
        let (stats, usage, context) = self.create_files(id, name, &meta)?;
        self.0.registry.write().insert(
            name,
            Arc::new(Entry::new(
                id.to_string(),
                name.to_string(),
                meta,
                stats,
                Slot::Hot(Box::new(context)),
                0,
                0,
                usage,
                ContextRevision::default(),
                // A brand-new generation never has a schema — and
                // under a freshly minted stem there is no earlier
                // generation's stray file to inherit one from.
                None,
                None,
            )),
        );
        Ok(())
    }

    /// The import header's create semantics (#965 decision 2): exactly
    /// one `context` per `id`, however many batches race. A `context`
    /// already registered under `id` is the answer (`Ok(false)` — the
    /// header's `create` block is ignored, never a rename or a
    /// reconfiguration); otherwise the family is created under THE
    /// HEADER'S id, so an export restored here keeps the id every
    /// other record points at. The id is reserved in `pending_creates`
    /// for the disk work's duration, so a concurrent batch of the same
    /// stream sees "taken" instead of writing the same family twice;
    /// it reports `Ok(false)` exactly as if the winner had already
    /// registered.
    ///
    /// Unlike [`Self::create`], the stem here is NOT freshly minted, so
    /// the clean-slate argument does not hold: an image already on disk
    /// under an id the registry does not know (a context whose load
    /// failed hard enough to be left unregistered) refuses with `Io`
    /// rather than being written over.
    pub fn create_if_absent(
        &self,
        id: &str,
        name: &str,
        meta: ContextMeta,
    ) -> Result<bool, CreateError> {
        if name.is_empty() {
            return Err(CreateError::InvalidName);
        }
        {
            let registry = self.0.registry.read();
            if registry.get_id(id).is_some() {
                return Ok(false);
            }
            // Reserved under the registry read guard, so this
            // check-then-insert cannot interleave with another
            // create_if_absent of the same id (both would need the
            // pending lock inside the same registry guard).
            if !self.0.pending_creates.lock().insert(id.to_string()) {
                return Ok(false);
            }
        }
        let outcome = if image_path(&self.0.data_dir, id).exists() {
            Err(CreateError::Io(io::Error::other(format!(
                "files for context id {id} already exist in the data directory but no \
                 context is registered under it"
            ))))
        } else {
            self.create_registered(id, name, meta).map(|()| true)
        };
        self.0.pending_creates.lock().remove(id);
        outcome
    }

    /// The disk half of [`AppState::create`], run WITHOUT the registry
    /// lock — the `pending.creates` reservation is what keeps the name
    /// taken meanwhile. The freshly minted `id` stem starts from a
    /// clean slate by construction; `save_files` lands the image last,
    /// so a crash anywhere before it commits leaves no image and
    /// nothing registers (the scan keys on `.ctx`).
    fn create_files(
        &self,
        id: &str,
        name: &str,
        meta: &ContextMeta,
    ) -> Result<(ContextStats, ContextUsage, Context), CreateError> {
        let mut context = Context::default();
        context.set_dice_floor(meta.dice_floor);
        let stats = ContextStats::of(&context);
        let usage = ContextUsage::default();
        // A fresh context starts its revision at zeros — which also
        // means a delete-recreate of the same name RESTARTS the
        // counters; a cache keyed on them must treat that as a new
        // lineage (see ContextRevision's doc).
        save_files(
            &self.0.data_dir,
            id,
            name,
            meta,
            &stats,
            &usage,
            ContextRevision::default(),
            None,
            &context,
        )
        .map_err(CreateError::Io)?;
        Ok((stats, usage, context))
    }

    /// Removes a `context` from the registry and deletes its files. The
    /// entry's lock is taken after the removal — waiting out any
    /// in-flight operation — and the slot becomes a tombstone under
    /// it: a flusher, evictor, or writer whose handle predates the
    /// removal finds [`Slot::Deleted`] when it finally locks, and
    /// backs off instead of recreating the files. Any unflushed writes
    /// are discarded — deletion destroys the `context`.
    ///
    /// The name enters `pending.deletes` in the same critical section
    /// that unregisters it and leaves only after the unlink loop: to a
    /// concurrent create() the name stays taken for the delete's whole
    /// run, so no new generation of files can appear under the tail of
    /// this one's removals.
    pub fn delete(&self, id: &str) -> Option<Result<(), DeleteError>> {
        let (entry, name) = {
            let mut registry = self.0.registry.write();
            // The name hint is the index's own reverse scan: `inner`
            // must not be locked while the registry lock is held (the
            // table's locking contract), and `remove_id`'s fallback
            // covers a hint gone stale anyway.
            let name = registry.name_of(id)?;
            let entry = registry.remove_id(id, &name)?;
            (entry, name)
        };
        let mut in_flight = entry.inner.write();
        self.tombstone_locked(&mut in_flight, &entry);
        // The rest of this function is disk I/O (marker, group sweep,
        // unlinks). No reservation set guards it anymore: the entry is
        // out of the registry (nothing new can address it), a create
        // can never re-mint this stem, and a rename that raced this
        // far finds the tombstone under `inner` and backs off.
        drop(in_flight);
        let stem = entry.id.clone();
        // A lazy bucket boot: the bucket's copy of this family must
        // not re-materialize after the unlinks below — veto waits out
        // any in-flight hydration so the two cannot interleave.
        // Nothing needs hydrating FIRST: files that never became local
        // were never shipped into this generation, and the manifest
        // gate (`Hydrator::drained`) keeps this generation
        // un-restorable until every family settles one way or the
        // other, so the deleted family cannot resurrect from either
        // generation.
        if let Some(hydrator) = &self.0.hydrator {
            hydrator.veto(&stem);
        }
        // The durable half of the acknowledgment: while this marker
        // exists, boot resumes the unlinks — so a partial failure here
        // (a held handle, a flaky mount) can leak bytes only until the
        // next start, and a surviving `.ctx` can never resurrect a
        // context the API reported gone. Written before the first
        // unlink; removed only after the last one succeeds.
        let marker = deleted_marker_path(&self.0.data_dir, &stem);
        if let Err(error) = write_atomic(&marker, b"") {
            tracing::warn!(context = %name, %error, "deletion marker not persisted; a partial delete would not resume at boot");
        }
        // Membership must not outlive the member: drop the name from
        // every group now, before the unlink loop's disk time. Best
        // effort — the delete's own durability rides on the marker
        // alone, and a sweep that could not persist is healed by the
        // next boot's reconciliation.
        self.sweep_context_from_groups(&stem);
        let mut outcome = Ok(());
        for file in context_files(&stem) {
            if let Err(error) = remove_persisted_file(self.0.data_dir.join(file))
                && error.kind() != io::ErrorKind::NotFound
            {
                outcome = Err(error);
            }
        }
        // Import markers go with the family: deletion makes any
        // half-applied batch moot, and a survivor would have boot
        // report a tear in a context that no longer exists. Same
        // failure handling as the fixed files — a miss keeps the
        // `.deleted` marker, and boot finishes the job.
        for path in import_marker_paths(&self.0.data_dir, &stem) {
            if let Err(error) = remove_persisted_file(&path)
                && error.kind() != io::ErrorKind::NotFound
            {
                outcome = Err(error);
            }
        }
        if outcome.is_ok() {
            let _ = remove_persisted_file(&marker);
        }
        Some(outcome.map_err(DeleteError::Io))
    }

    /// Renames a `context`: under ids (ADR 0045) a rename is a display-
    /// name change and nothing else — the file family stays where it
    /// is (the stem is the id), so the marker/move/resume machinery
    /// renames used to need is gone, and so is the `group` rewrite
    /// (#965): records hold member ids, which a rename never changes.
    /// What remains is: persist the new name into the sidecar (the
    /// name's one durable home) and move the id between the two names
    /// in the registry's index.
    ///
    /// No availability check and no reservation: the destination name
    /// may already be in use (issue #961 decision 1 — names are not
    /// unique), so there is nothing for a concurrent create or rename
    /// to collide with. The sidecar write happens under the entry
    /// lock and BEFORE the index moves: a success response always
    /// means the new name is durable, and a crash mid-call leaves at
    /// worst a sidecar already renamed whose index entry still says
    /// the old name — the next boot reads the sidecar and registers
    /// the new name, exactly what the caller was about to be told.
    pub fn rename_context(&self, id: &str, to: &str) -> Result<(), RenameContextError> {
        if to.is_empty() {
            return Err(RenameContextError::InvalidName);
        }
        let Some(entry) = self.lookup_id(id) else {
            return Err(RenameContextError::NotFound);
        };
        let from = match self.rename_entry(&entry, to)? {
            // A self-rename: the sidecar was not rewritten and the
            // index has nothing to move.
            None => return Ok(()),
            Some(previous) => previous,
        };
        {
            let mut registry = self.0.registry.write();
            // A delete that raced this rename has already unindexed
            // the entry — reindexing here would resurrect a name row
            // for an id the table no longer holds.
            if registry.get_id(id).is_some() {
                registry.reindex(id, &from, to);
            }
        }
        Ok(())
    }

    /// The entry half of [`AppState::rename_context`]: swaps
    /// `EntryInner::name` and persists the sidecar under the entry's
    /// unchanged id, rolling the in-memory name back if the write
    /// fails — so memory and the sidecar can only disagree over a
    /// crash, never over a reported error. Hands the previous name
    /// back (the index move and the group rewrite need it), or `None`
    /// for a self-rename, which touches nothing. The current name is
    /// read under the same entry lock as the swap, so "self-rename"
    /// is judged against the name the sidecar actually holds.
    fn rename_entry(&self, entry: &Entry, to: &str) -> Result<Option<String>, RenameContextError> {
        let Some(mut guard) = entry.lock_unless_deleted() else {
            // A delete won the race after the resolve above; to its
            // caller the name is simply gone.
            return Err(RenameContextError::NotFound);
        };
        let inner = &mut *guard;
        if inner.name == to {
            return Ok(None);
        }
        let previous = std::mem::replace(&mut inner.name, to.to_string());
        // The revision counters deliberately do NOT move: a rename is
        // the same content under a new name, exactly as before ids —
        // caches keyed by name stop matching on their own, and the
        // group fingerprint still changes (the member NAME is part of
        // its hash).
        if let Err(error) = write_meta(
            &self.0.data_dir,
            &entry.id,
            &inner.name,
            &inner.meta,
            &inner.stats,
            &entry.usage.snapshot(),
            entry.revision_snapshot(inner),
            inner.schema_digest.as_deref(),
        ) {
            inner.name = previous;
            return Err(RenameContextError::Io(error));
        }
        Ok(Some(previous))
    }
}

impl AppState {
    /// Updates the description and/or pin flag, persisting the sidecar
    /// immediately. Pinning loads the `context` now (pinned means
    /// resident); unpinning subjects it to the cache budget again.
    pub fn update_meta(
        &self,
        id: &str,
        description: Option<String>,
        pinned: Option<bool>,
        dice_floor: Option<f64>,
        semantic_floor: Option<f32>,
    ) -> Option<io::Result<ContextMeta>> {
        let entry = self.lookup_id(id)?;
        let outcome = {
            // A `None` means a delete won the lock first: don't
            // recreate the sidecar it just removed.
            let mut guard = entry.lock_unless_deleted()?;
            let inner = &mut *guard;
            // Saved so a load or persist failure below can restore the
            // pre-call state — without it, memory would hold fields
            // that never reached the sidecar, and a later, unrelated
            // successful update would persist them as a side effect.
            let previous = inner.meta.clone();
            if let Some(description) = description {
                inner.meta.description = description;
            }
            if let Some(pinned) = pinned {
                inner.meta.pinned = pinned;
            }
            if let Some(floor) = dice_floor {
                inner.meta.dice_floor = Some(floor.clamp(0.0, 1.0));
                // A loaded context picks the new floor up immediately;
                // a cold one gets it on its next load.
                if let Slot::Hot(context) = &mut inner.slot {
                    context.set_dice_floor(inner.meta.dice_floor);
                }
            }
            if let Some(floor) = semantic_floor {
                // Read at query time from the meta; nothing to push into
                // the loaded context.
                inner.meta.semantic_floor = Some(floor.clamp(0.0, 1.0));
            }
            if inner.meta.pinned
                && let Err(error) = ensure_hot(
                    &self.0.data_dir,
                    &entry.id,
                    inner,
                    &self.0.metrics,
                    self.0.hydrator.as_deref(),
                )
            {
                rollback_meta(inner, previous);
                self.recount_entry(inner);
                return Some(Err(io::Error::other(error)));
            }
            // A pin toggle moves the entry into or out of the budget's
            // world; the estimate must follow.
            self.recount_entry(inner);
            // Bump-and-persist atomically: the config revision rides
            // the same sidecar write as the change it tracks, and both
            // roll back together below — so a served bump always means
            // the new meta is durable. A PATCH that changed nothing
            // bumps nothing: idempotent updates must not churn caches.
            let changed = inner.meta != previous;
            if changed {
                inner.config_revision += 1;
            }
            let result = write_meta(
                &self.0.data_dir,
                &entry.id,
                &inner.name,
                &inner.meta,
                &inner.stats,
                &entry.usage.snapshot(),
                entry.revision_snapshot(inner),
                inner.schema_digest.as_deref(),
            )
            .map(|()| inner.meta.clone());
            if result.is_err() {
                if changed {
                    inner.config_revision -= 1;
                }
                rollback_meta(inner, previous);
                self.recount_entry(inner);
            }
            result
        };
        self.enforce_budget(&entry.id);
        Some(outcome)
    }

    /// The resident schema for `name` — `Ok(None)` for a schema-free
    /// `context` (`GET /contexts/{id}/schema`, #380, turns that into a
    /// 404). Outer `None` means no such `context`.
    ///
    /// The common case is already resolved without touching disk: boot
    /// and every cold-load already ran `load_schema` (ADR 0009 §5.2's
    /// consistency check) into [`EntryInner::schema`], and a
    /// `schema_digest` of `None` — set only by `put_schema` below, under
    /// this same lock — never means anything but "no schema". Only a
    /// digest recorded but not yet checked against its bytes LOCALLY
    /// (a replica mid-hydration, or a rename's freshly registered
    /// entry — see `EntryInner::schema`'s own doc) falls through to the
    /// slow path, which reuses `ensure_hot` rather than calling
    /// `schema::load_schema` directly: `ensure_hot` is the one place
    /// that also runs the hydrator, so a replica whose family has not
    /// been fetched yet resolves correctly here too, not just a purely
    /// local rename. Heavier than strictly needed (it loads the full
    /// graph image to get there), but `GET /schema` is an infrequent
    /// management call, not a retrieval hot path.
    pub fn schema_of(
        &self,
        id: &str,
    ) -> Option<Result<Option<Arc<schema::InstalledSchema>>, String>> {
        let entry = self.lookup_id(id)?;
        {
            let inner = entry.read_unless_deleted()?;
            if inner.schema.is_some() || inner.schema_digest.is_none() {
                return Some(Ok(inner.schema.clone()));
            }
        }
        let outcome = {
            let mut guard = entry.lock_unless_deleted()?;
            let inner = &mut *guard;
            if let Err(error) = ensure_hot(
                &self.0.data_dir,
                &entry.id,
                inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            ) {
                return Some(Err(error));
            }
            self.recount_entry(inner);
            Ok(inner.schema.clone())
        };
        self.enforce_budget(&entry.id);
        Some(outcome)
    }

    /// ADR 0009 §6.3's one gate for the reserved `schema:type` label:
    /// "an installed schema document exists," never "mode != off." An
    /// operator who installs a schema but leaves it in `off` while
    /// drafting types has already committed to the reserved label
    /// meaning something — `off` only means "don't enforce domain/range
    /// yet," not "pretend the label is ordinary." `Some(SCHEMA_TYPE_LABEL)`
    /// whenever this `context` has ever had a schema installed, in any
    /// mode; `None` only for a `context` that never installed one (or an
    /// unknown/deleted name). A schema recorded but currently unreadable
    /// (`schema_of`'s `Err` arm) maps CONSERVATIVELY to "hidden" — per
    /// [`schema`]'s own module doc, every trouble case there is a hard
    /// refusal, never a silent fallback, and this helper must not be the
    /// one place that quietly un-reserves the label because a read
    /// failed.
    ///
    /// ⚠ Never call this from inside a [`AppState::read_context`]
    /// closure: the slow path (through [`AppState::schema_of`]) takes
    /// this entry's write lock, while `read_context` already holds its
    /// read lock for the whole closure — parking_lot's `RwLock` is
    /// neither reentrant nor reader-preferring, so that ordering
    /// deadlocks. Resolve the hidden label first, then pass the
    /// `Option<&str>` into the closure.
    pub fn hidden_label(&self, id: &str) -> Option<&'static str> {
        match self.schema_of(id)? {
            Ok(Some(_)) => Some(schema::SCHEMA_TYPE_LABEL),
            Ok(None) => None,
            Err(_) => Some(schema::SCHEMA_TYPE_LABEL),
        }
    }

    /// [`Self::hidden_label`] as the exclusion slice a `read_context`
    /// call site actually wants (issue #622 finding 4) — off the async
    /// worker, since [`Self::hidden_label`]'s own doc requires it to
    /// run before, never inside, a `read_context` closure. Bundles the
    /// `block_in_place` + `.into_iter().collect()` idiom five HTTP
    /// handlers each wrote out by hand.
    pub fn excluded_hidden_label(&self, id: &str) -> Vec<&'static str> {
        tokio::task::block_in_place(|| self.hidden_label(id))
            .into_iter()
            .collect()
    }

    /// ADR 0009 §6.3 guard 2's `add_label_alias` bullet: the pre-flight
    /// an alias-creating write consults before it runs, mirroring
    /// `predicted_alias_rejection`'s own read-only-prediction shape —
    /// including that shape's own known race: this check and the
    /// write it precedes take two SEPARATE lock acquisitions, not one
    /// held across both, so a `PUT /schema` install landing in the
    /// gap between them is not caught here. That gap is not new to
    /// this guard — `apply_batch` already runs
    /// `predicted_alias_rejection` and its subsequent `add_aliases`/
    /// `add_associations` the same two-lock-acquisitions way, and ADR
    /// 0009 §7.3 explicitly declines to make the write path atomic
    /// against concurrent mutation ("that is #187's scope"). Closing
    /// it here would mean re-running this check under `add_aliases`'
    /// own write lock, which — because `Context` has no schema
    /// knowledge (§7.3's own reasoning for keeping the check a layer
    /// up) — reaches into every other `add_aliases`/`logged_write`
    /// caller too; deferred as the same kind of cross-cutting
    /// atomicity work #187 already owns, not attempted piecemeal here.
    /// Only meaningful once [`AppState::hidden_label`] says a schema
    /// exists — a schema-free `context`'s `schema:type` stays an ordinary
    /// label (guard 1), so nothing here refuses anything for it.
    /// Deliberately does not chase a multi-hop alias chain: once a
    /// schema exists, no *live* alias can ever resolve to the reserved
    /// label (this guard and `PUT /schema`'s migration-boundary check
    /// both stand in its way going forward), so a direct value
    /// comparison against `labels` is the whole check.
    pub fn reserved_alias_conflict(
        &self,
        name: &str,
        labels: &BTreeMap<String, String>,
    ) -> Option<String> {
        self.hidden_label(name)?;
        schema::reserved_aliases(
            labels
                .iter()
                .map(|(alias, canonical)| (alias.as_str(), canonical.as_str())),
        )
        .next()
        .map(str::to_string)
    }

    /// `PUT /contexts/{id}/schema` (#380): installs `installed` as
    /// `name`'s schema document, replacing whatever was there wholesale
    /// — there is no delta form, so a retry after a failure below is
    /// always safe regardless of which side of it the previous attempt
    /// reached (ADR 0009 §5.2). Does exactly what `bump_config_revision`
    /// already does for `dice_floor`, plus `invalidate_cache_identity`:
    /// a schema mutation can change what `query`'s future type filter
    /// (§12.3) returns, so a retrieval-cache key minted before this call
    /// must not keep answering with the old constraints.
    ///
    /// Outer `None` means no such `context`. `Ok` carries the installed
    /// document back (including when the call was a no-op — see below)
    /// so the handler can answer `GET`-shaped without a second lookup.
    pub fn put_schema(
        &self,
        id: &str,
        installed: schema::InstalledSchema,
    ) -> Option<Result<schema::SchemaDocument, PutSchemaError>> {
        let entry = self.lookup_id(id)?;
        let outcome = {
            let mut guard = entry.lock_unless_deleted()?;
            let inner = &mut *guard;
            // Unconditional, unlike `update_meta`'s `pinned`-gated load:
            // the migration-boundary guard just below needs the LIVE
            // label-alias table, which only a hot context has, on every
            // call — aliases can be added between one `PUT` and the
            // next, so a resolution cached from an earlier call would
            // miss one created since.
            if let Err(error) = ensure_hot(
                &self.0.data_dir,
                &entry.id,
                inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            ) {
                self.recount_entry(inner);
                return Some(Err(PutSchemaError::Load(error)));
            }
            self.recount_entry(inner);
            // ADR 0009 §6.3 guard 2's install-time bullet: an
            // already-persisted `label_alias` resolving to the reserved
            // type label. Guard 2's other two bullets — refusing
            // `add_label_alias`/a batch's own `batch.labels` from ever
            // CREATING such an alias once a schema exists — are
            // `AppState::reserved_alias_conflict` (the aliases handler)
            // and `schema_issues`' `SchemaCheck::reserved` (a future
            // write entrance, S4/S5) respectively; neither existed
            // before this schema's own document did, so this call site
            // is the one place a violating alias predating them could
            // still slip through, and it stays on every `PUT` (not only
            // the off-to-installed transition) for exactly that reason.
            if let Some(alias) = schema::reserved_aliases(hot_context(inner).label_aliases()).next()
            {
                return Some(Err(PutSchemaError::ReservedAlias(alias.to_string())));
            }
            let bytes = match schema::document_bytes(installed.document()) {
                Ok(bytes) => bytes,
                Err(error) => return Some(Err(PutSchemaError::Io(error))),
            };
            let digest = crate::sha256::sha256_hex(&bytes);
            // A PUT that changes nothing bumps nothing — the same
            // idempotent-update discipline `update_meta` keeps for a
            // no-op PATCH — so a retried or duplicate `PUT` of the same
            // document never churns the retrieval cache.
            if inner.schema.is_some() && inner.schema_digest.as_deref() == Some(digest.as_str()) {
                Ok(installed.document().clone())
            } else {
                let stem = entry.id.clone();
                let previous_digest = inner.schema_digest.clone();
                inner.config_revision += 1;
                inner.schema_digest = Some(digest);
                // Revision-then-content (ADR 0009 §5.2): this write
                // lands BEFORE the schema file's own `write_atomic`
                // below, both under this entry's write lock, so a
                // crash between the two always fails toward extra
                // invalidation (revision advanced, content unchanged)
                // rather than a served mismatch (content changed,
                // revision stale).
                let meta_result = write_meta(
                    &self.0.data_dir,
                    &stem,
                    &inner.name,
                    &inner.meta,
                    &inner.stats,
                    &entry.usage.snapshot(),
                    entry.revision_snapshot(inner),
                    inner.schema_digest.as_deref(),
                );
                match meta_result {
                    Err(error) => {
                        inner.config_revision -= 1;
                        inner.schema_digest = previous_digest;
                        Err(PutSchemaError::Io(error))
                    }
                    Ok(()) => match schema::write_schema_bytes(&self.0.data_dir, &stem, &bytes) {
                        Ok(()) => {
                            let document = installed.document().clone();
                            inner.schema = Some(Arc::new(installed));
                            inner.invalidate_cache_identity();
                            // The change feed's config-side entrance
                            // (#422): only a PUT that actually changed
                            // the document reaches here — the idempotent
                            // early return above never feeds an event.
                            entry
                                .changes
                                .lock()
                                .push(crate::registry::ChangeKind::SchemaUpdated {
                                    mode: document.mode.as_str().to_string(),
                                });
                            Ok(document)
                        }
                        Err(error) => {
                            inner.config_revision -= 1;
                            inner.schema_digest = previous_digest.clone();
                            // Best-effort restore of the sidecar to the
                            // pre-PUT digest; if this ALSO fails, the
                            // next boot's digest check (§5.2) refuses
                            // rather than silently serving the
                            // mismatch — the same fail-closed posture
                            // `load_schema` already enforces, not a new
                            // mechanism this call adds.
                            let _ = write_meta(
                                &self.0.data_dir,
                                &stem,
                                &inner.name,
                                &inner.meta,
                                &inner.stats,
                                &entry.usage.snapshot(),
                                entry.revision_snapshot(inner),
                                previous_digest.as_deref(),
                            );
                            Err(PutSchemaError::Io(error))
                        }
                    },
                }
            }
        };
        self.enforce_budget(&entry.id);
        Some(outcome)
    }
}

/// Restores `inner.meta` to `previous` after a load or persist failure
/// partway through `update_meta`. Also un-applies the floor from any
/// already-loaded `context`, matching the one place `update_meta` pushes
/// a field straight into the hot `context` instead of just the sidecar.
fn rollback_meta(inner: &mut EntryInner, previous: ContextMeta) {
    if let Slot::Hot(context) = &mut inner.slot {
        context.set_dice_floor(previous.dice_floor);
    }
    inner.meta = previous;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::test_support::{assoc_op, scratch_dir};

    /// An empty `context` name is refused at the registry boundary — the
    /// last guard against a bare `.ctx` file that `scan_data_dir` (which
    /// keys on the file stem) would never rediscover, silently orphaning
    /// every write to it. Parse and API refuse it earlier; this locks
    /// the floor beneath them.
    #[test]
    fn an_empty_context_name_is_refused_by_create() {
        let dir = scratch_dir("empty-name");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(matches!(
            state.create("", ContextMeta::default()),
            Err(CreateError::InvalidName)
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    /// Every stage/commit/unlink position in `context` deletion either
    /// finishes immediately or leaves enough durable state for boot to
    /// finish it. The first index beyond the operation proves the sweep
    /// did not merely sample a few hand-picked failures.
    #[test]
    fn every_context_delete_persistence_failure_recovers_at_boot() {
        let mut exhausted = false;
        for failure in 0..64 {
            let dir = scratch_dir(&format!("delete-fault-{failure}"));
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            let stem = state.stem_of("sake").unwrap();
            state
                .add_associations(
                    &state.id_of("sake"),
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("doc"))],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
            state
                .create_group(
                    "breweries",
                    String::new(),
                    BTreeSet::from([state.id_of("sake")]),
                    BTreeSet::new(),
                )
                .unwrap();

            fail_persistence_ops_after(failure);
            let outcome = state.delete(&state.id_of("sake")).unwrap();
            let past_end = clear_persistence_fault();
            drop(state);

            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            assert!(
                state.directory_entry("sake").is_none(),
                "failure at persistence step {failure} resurrected the context: {outcome:?}"
            );
            assert!(
                state.group("breweries").unwrap().context_ids.is_empty(),
                "boot did not reconcile group membership at step {failure}"
            );
            assert!(
                !deleted_marker_path(&dir, &stem).exists(),
                "boot did not finish the marker at step {failure}"
            );
            drop(state);
            let _ = fs::remove_dir_all(&dir);

            if past_end {
                assert!(outcome.is_ok());
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "context deletion exceeded the sweep bound");
    }

    /// The old dangerous interleaving, defused by ids: a delete
    /// leaves its marker behind (partial failure), the SAME running
    /// server recreates the name — under a FRESH id, so the stale
    /// marker and the new family share nothing. The next boot resumes
    /// the old generation's deletion without touching the recreate.
    #[test]
    fn a_stale_deletion_marker_never_touches_a_recreated_namesake() {
        let dir = scratch_dir("deleted-recreate");
        let old_stem;
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            old_stem = state.stem_of("sake").unwrap();
            state.delete(&state.id_of("sake"));
            // Simulate the failure mode delete() cannot fully guard: its
            // unlink loop errored before removing the marker, so the
            // marker survives on disk while the name is free again.
            fs::write(deleted_marker_path(&dir, &old_stem), b"").unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "recreate")
                .unwrap();
            assert_ne!(
                old_stem,
                state.stem_of("sake").unwrap(),
                "a recreate mints a fresh id"
            );
            state
                .add_associations(
                    &state.id_of("sake"),
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            !deleted_marker_path(&dir, &old_stem).exists(),
            "boot resumed and cleared the old generation's marker"
        );
        assert!(
            state.directory_entry("sake").is_some(),
            "the recreated context must survive the restart"
        );
        let count = state
            .read_context(&state.id_of("sake"), |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 1, "its data must be intact");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The import-marker half of the same defusal: a marker the
    /// delete sweep could not remove names its `context` by DISPLAY
    /// name, which a recreate (under a fresh id) legitimately reuses
    /// — so the next boot's deletion resume must clear it with the
    /// rest of the old family, not blame a half-applied import on
    /// the namesake.
    #[test]
    fn a_stale_import_marker_is_cleared_by_the_deletion_resume_not_blamed_on_a_namesake() {
        let dir = scratch_dir("import-marker-recreate");
        let old_stem;
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            old_stem = state.stem_of("sake").unwrap();
            state.delete(&state.id_of("sake")).unwrap().unwrap();
            // The failure delete() cannot fully guard: its marker sweep
            // missed one (crash, held handle), so the file outlives the
            // name — together with the `.deleted` marker that promises
            // the rest of the teardown to the next boot.
            fs::write(
                import_marker_path(&dir, &old_stem, "doc-1"),
                b"{\"context\":\"sake\",\"source\":\"doc-1\"}",
            )
            .unwrap();
            fs::write(deleted_marker_path(&dir, &old_stem), b"").unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "recreate")
                .unwrap();
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            import_marker_paths(&dir, &old_stem).is_empty(),
            "the deletion resume clears the old generation's import markers"
        );
        assert!(
            state.directory_entry("sake").is_some(),
            "the recreated namesake is untouched"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(unix)]
    fn a_failed_persist_does_not_leave_the_failed_change_in_memory() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("meta-rollback");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        // A clean update lands on disk.
        let meta = state
            .update_meta(
                &state.id_of("sake"),
                Some("A".to_string()),
                None,
                None,
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(meta.description, "A");

        // The disk goes bad: this update must be refused...
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let failed = state
            .update_meta(
                &state.id_of("sake"),
                Some("B".to_string()),
                None,
                None,
                None,
            )
            .unwrap();
        assert!(failed.is_err(), "a persist failure must surface as Err");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

        // ...and must not have left "B" sitting in memory — a later,
        // unrelated successful update must still see and persist "A",
        // not silently resurrect the failed change.
        let meta = state
            .update_meta(&state.id_of("sake"), None, Some(true), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(
            meta.description, "A",
            "the failed update to \"B\" must not have survived in memory"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn every_latecomer_behind_a_delete_finds_the_tombstone() {
        let dir = scratch_dir("delete-tombstone");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("victim", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let stale = state.lookup_named("victim").unwrap();
        state.delete(&state.id_of("victim")).unwrap().unwrap();

        // The gate every post-lookup lock acquisition goes through:
        // a handle that predates the removal must be turned away.
        assert!(
            stale.lock_unless_deleted().is_none(),
            "the tombstone must refuse a stale handle"
        );
        // And the public write path answers NotFound rather than
        // recreating the WAL file the delete just removed.
        assert!(matches!(
            state.add_associations(
                &state.id_of("victim"),
                vec![assoc_op("幽霊", "は", "残らない", 1.0, None)],
                Deadline::unbounded(),
            ),
            Err(AccessError::NotFound)
        ));
        assert!(!wal_path(&dir, &stale.id).exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// A failed create must release its `pending.creates` reservation —
    /// otherwise one disk refusal would leave the name reading as taken
    /// until restart.
    #[test]
    fn a_failed_create_releases_the_name() {
        let dir = scratch_dir("create-release");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        // Fail the create's own sidecar write (its fresh uuid stem
        // has no leftovers to trip over any more — the disk refusal
        // is injected instead), after the name is already reserved.
        fail_persistence_ops_after(0);
        assert!(matches!(
            state.create("sake", ContextMeta::default()),
            Err(CreateError::Io(_))
        ));
        clear_persistence_fault();

        // Fault gone, the same name must create cleanly — the failed
        // attempt's reservation may not linger.
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_create_racing_a_slow_delete_lands_on_its_own_stem_untouched() {
        let dir = scratch_dir("delete-create-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let old_id = state.create("sake", ContextMeta::default()).unwrap();

        // Stall the delete mid-flight: it unregisters the entry, then
        // must wait for this read guard before it may touch files —
        // the window where the OLD name-keyed model had to refuse a
        // same-name create (its new generation shared the doomed
        // stem). Under minted ids the same create simply lands on a
        // fresh stem the unlink loop can never touch.
        let entry = state.lookup_named("sake").unwrap();
        let stall = entry.inner.read();
        let deleter = {
            let state = state.clone();
            let old_id = old_id.clone();
            std::thread::spawn(move || state.delete(&old_id).unwrap().unwrap())
        };
        while state.lookup_named("sake").is_some() {
            std::thread::yield_now();
        }
        let new_id = state
            .create("sake", ContextMeta::default())
            .expect("a mid-delete name is free — ids cannot collide");
        assert_ne!(new_id, old_id, "the recreate mints its own id");

        drop(stall);
        deleter.join().unwrap();
        // The delete finished AFTER the create and removed only its
        // own generation's files.
        assert!(image_path(&dir, &new_id).exists());
        assert!(!image_path(&dir, &old_id).exists());
        assert_eq!(state.stem_of("sake").unwrap(), new_id);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dice_floor_persists_in_the_sidecar_and_reapplies_on_load() {
        let dir = scratch_dir("floor");
        // One shared informative bigram of 4+3: Dice ≈ 0.286 — misses
        // the 0.3 default, lands once the context is tuned to 0.25.
        let fuzzy_cue = "青嶺の純米";
        let lands = |state: &AppState| {
            state
                .read_context(&state.id_of("sake"), |context| {
                    context
                        .resolve(fuzzy_cue)
                        .iter()
                        .any(|hit| hit.name == "青嶺酒造")
                })
                .map_err(|_| "read")
                .unwrap()
        };
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("sake", |context| {
                    context.associate("青嶺酒造", "分類", "酒蔵", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();

            assert!(!lands(&state), "default floor must reject the cue");

            // Tuning applies to the loaded context immediately.
            state
                .update_meta(&state.id_of("sake"), None, None, Some(0.25), None)
                .unwrap()
                .unwrap();
            assert!(lands(&state), "tuned floor must admit the cue");
            // The flusher learns which contexts it persisted — that list
            // feeds the auto embedding refresh.
            assert_eq!(state.flush_dirty(), vec!["sake".to_string()]);
            assert!(state.flush_dirty().is_empty());
        }

        // A cold boot re-applies the floor from the sidecar — the image
        // itself carries no config.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(lands(&state), "floor must survive the restart");
        assert_eq!(state.directory()[0].dice_floor, Some(0.25));

        let _ = fs::remove_dir_all(dir);
    }

    /// `update_meta`'s `dice_floor`/`semantic_floor` clamps
    /// (`floor.clamp(0.0, 1.0)`) have no test: every call site in the
    /// suite already passes an in-range value, so the clamp never
    /// actually clamps anything. It is also the ONLY guard on the PATCH
    /// path — `api/contexts.rs`'s create handler clamps up front, but
    /// its PATCH handler forwards `dice_floor`/`semantic_floor` raw.
    #[test]
    fn update_meta_clamps_out_of_range_floors_into_zero_to_one() {
        let dir = scratch_dir("update-meta-floor-clamp");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        state
            .update_meta(&state.id_of("sake"), None, None, Some(2.5), Some(-1.0))
            .unwrap()
            .unwrap();

        let entry = state.directory_entry("sake").unwrap();
        assert_eq!(
            entry.dice_floor,
            Some(1.0),
            "an over-range dice_floor must clamp to the ceiling"
        );
        assert_eq!(
            entry.semantic_floor,
            Some(0.0),
            "an under-range semantic_floor must clamp to the floor"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// `update_meta`'s pinned-`ensure_hot`-failure rollback
    /// (`rollback_meta` + `recount_entry`, then `Err`) has no test —
    /// the only existing rollback test targets the sibling `write_meta`
    /// failure arm instead, with its `pinned` call made AFTER
    /// permissions are restored so `ensure_hot` there always succeeds.
    /// Here a cold `context` with a corrupted image is pinned: the
    /// attempt must fail closed, `meta.pinned` must roll back to
    /// `false` (not strand the `context` pinned-but-unloadable), and the
    /// budget's `resident_estimate` must stay in sync with that
    /// rollback rather than the failed intermediate state.
    #[test]
    fn update_meta_rolls_back_pinning_when_the_forced_preload_fails() {
        let dir = scratch_dir("update-meta-pin-rollback");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                &state.id_of("sake"),
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.flush_dirty();
        let entry = state.lookup_named("sake").unwrap();
        assert!(
            state.evict_entry("sake", &entry),
            "sanity: an unpinned context must evict cleanly"
        );

        let image = image_path(&dir, &state.stem_of("sake").unwrap());
        let mut bytes = fs::read(&image).unwrap();
        assert!(bytes.len() > 8, "sanity: the version byte must exist");
        bytes[8] = 0xFF;
        fs::write(&image, &bytes).unwrap();

        let error = state
            .update_meta(&state.id_of("sake"), None, Some(true), None, None)
            .expect("the context still exists")
            .expect_err("the forced preload must fail on the corrupt image");
        assert!(!error.to_string().is_empty());

        let after = state.directory_entry("sake").unwrap();
        assert!(
            !after.pinned,
            "a failed forced preload must roll `pinned` back to false, \
             not strand the context pinned yet cold and unloadable"
        );
        assert!(!after.loaded, "it must stay cold, not half-applied");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_context_changes_the_name_in_place_and_leaves_group_membership_alone() {
        let dir = scratch_dir("rename-context-happy");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create(
                "sake",
                ContextMeta {
                    pinned: true,
                    ..ContextMeta::default()
                },
            )
            .unwrap();
        state
            .add_associations(
                &state.id_of("sake"),
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state
            .create_group(
                "drinks",
                String::new(),
                BTreeSet::from([state.id_of("sake")]),
                BTreeSet::new(),
            )
            .unwrap();
        let stem = state.stem_of("sake").unwrap();

        state
            .rename_context(&state.id_of("sake"), "shochu")
            .unwrap();

        assert!(
            state.directory_entry("sake").is_none(),
            "the old name must be gone"
        );
        let entry = state
            .directory_entry("shochu")
            .expect("the new name must answer");
        assert!(entry.pinned, "pinned carries over");
        assert!(
            entry.loaded,
            "a rename unloads nothing: the entry — hot, for a pinned context — is untouched"
        );
        assert_eq!(
            state.stem_of("shochu").unwrap(),
            stem,
            "the id — and so the file family — never moves"
        );
        assert!(
            image_path(&dir, &stem).exists(),
            "the family stays at its stem"
        );
        assert_eq!(
            state.group("drinks").unwrap().context_ids,
            BTreeSet::from([stem.clone()]),
            "members are ids (#965): a rename never touches the record, \
             and the member still names the same context"
        );
        let count = state
            .read_context(&state.id_of("shochu"), |context| {
                context.association_count()
            })
            .unwrap();
        assert_eq!(count, 1, "data is untouched");

        // Persisted, not just in memory: the sidecar is the name's one
        // durable home now.
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_none());
        assert!(state.directory_entry("shochu").is_some());
        assert_eq!(
            state.group("drinks").unwrap().context_ids,
            BTreeSet::from([stem.clone()])
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A rename never touches the file family (the stem is the id),
    /// so the schema file and its recorded digest simply stay where
    /// they are — and stay CONSISTENT: the rename's own sidecar write
    /// re-records the digest it read, and the next boot's §5.2 check
    /// still passes under the new name.
    #[test]
    fn rename_context_keeps_the_schema_file_and_its_recorded_digest() {
        let dir = scratch_dir("rename-context-schema");
        let document =
            br#"{"type":"schema","mode":"off","closed_labels":false,"types":{},"relations":{}}"#;
        let digest = crate::sha256::sha256_hex(document);
        let stem;
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            stem = state.stem_of("sake").unwrap();
            state.flush_dirty();
        }
        fs::write(schema_path(&dir, &stem), document).unwrap();
        let meta_file = meta_path(&dir, &stem);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&meta_file).unwrap()).unwrap();
        value["schema_digest"] = serde_json::json!(digest);
        fs::write(&meta_file, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

        // A fresh boot picks up the hand-planted schema (matching the
        // digest above) before renaming.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .rename_context(&state.id_of("sake"), "shochu")
            .unwrap();

        assert_eq!(
            fs::read(schema_path(&dir, &stem)).unwrap(),
            document,
            "the schema file stays at the unchanged stem"
        );
        assert_eq!(
            read_meta_file(&dir, &stem).schema_digest.as_deref(),
            Some(digest.as_str()),
            "the rename's sidecar write must carry the digest, not drop it"
        );
        drop(state);

        // If the rename's sidecar write had dropped the digest, this
        // boot would refuse (§5.2: file present, nothing recorded).
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("shochu").is_some());
        drop(state);

        let _ = fs::remove_dir_all(dir);
    }

    /// `delete`'s unlink loop walks `context_files`, so the schema file
    /// — its tenth entry since #379 — must go with the rest of the
    /// family, never left as litter a reused name could later collide
    /// with (see `sweep_stale_stem_files`'s own schema-litter guard).
    #[test]
    fn delete_removes_the_schema_file_with_the_rest_of_the_family() {
        let dir = scratch_dir("delete-context-schema");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.flush_dirty();
        let stem = state.stem_of("sake").unwrap();
        fs::write(schema_path(&dir, &stem), b"irrelevant to this test").unwrap();

        state.delete(&state.id_of("sake")).unwrap().unwrap();

        assert!(!schema_path(&dir, &stem).exists());
        drop(state);
        let _ = fs::remove_dir_all(dir);
    }

    /// The rename's sidecar write snapshots the LIVE usage counters —
    /// including everything counted while the `context` sat Cold, which
    /// no flush has seen — so a restart right after the rename reads
    /// them back intact instead of whatever stale snapshot the last
    /// flush happened to leave.
    #[test]
    fn rename_persists_usage_counted_while_the_context_was_cold() {
        let dir = scratch_dir("rename-usage-cold");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state.note_read(&state.id_of("sake"), false);
        state.note_write(&state.id_of("sake"));

        let entry = state.lookup_named("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));

        // Counted while Cold — no flush or eviction will ever see these
        // before the rename runs.
        state.note_read(&state.id_of("sake"), false);
        state.note_read(&state.id_of("sake"), true);
        state.note_write(&state.id_of("sake"));

        state.rename_context(&state.id_of("sake"), "sake2").unwrap();

        let usage = state
            .directory_entry("sake2")
            .expect("the new name must answer")
            .usage;
        assert_eq!((usage.reads, usage.empty_reads, usage.writes), (3, 1, 2));

        // No flush between the rename and this restart: the rename's
        // own sidecar write is the only place these counters could
        // have become durable.
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let usage = state.directory_entry("sake2").unwrap().usage;
        assert_eq!(
            (usage.reads, usage.empty_reads, usage.writes),
            (3, 1, 2),
            "usage counted while Cold must ride the rename's sidecar write"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// The rename fault sweep, under ids: every persistence step of
    /// `rename_context` either rolls the whole call back (the sidecar
    /// write — the name's one durable home — failed, so `from` still
    /// answers) or lands the rename. Group records hold member ids
    /// (#965), so there is no membership rewrite to fail: the member is
    /// the same id live, on disk, and after a reboot, whichever name
    /// the context ends up under.
    #[test]
    fn a_rename_fault_lands_the_context_under_exactly_one_name() {
        let mut exhausted = false;
        for failure in 0..64 {
            let dir = scratch_dir(&format!("rename-membership-fault-{failure}"));
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            let sake_id = state.id_of("sake");
            state
                .create_group(
                    "drinks",
                    String::new(),
                    BTreeSet::from([sake_id.clone()]),
                    BTreeSet::new(),
                )
                .unwrap();

            fail_persistence_ops_after(failure);
            let outcome = state.rename_context(&state.id_of("sake"), "shochu");
            let past_end = clear_persistence_fault();
            let live_members = state.group("drinks").unwrap().context_ids;
            drop(state);

            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            let sake = state.directory_entry("sake");
            let shochu = state.directory_entry("shochu");
            let members = state.group("drinks").unwrap().context_ids;
            match (sake.is_some(), shochu.is_some()) {
                (true, false) => {
                    assert!(
                        outcome.is_err(),
                        "failure at persistence step {failure}: the rename must \
                         not report success while the old name still answers"
                    );
                    assert_eq!(
                        members,
                        BTreeSet::from([sake_id.clone()]),
                        "failure at persistence step {failure} ({outcome:?}): the \
                         rename never landed, so membership must be untouched"
                    );
                }
                (false, true) => {
                    assert_eq!(
                        live_members,
                        BTreeSet::from([sake_id.clone()]),
                        "failure at persistence step {failure} ({outcome:?}): a \
                         landed rename leaves the LIVE record alone"
                    );
                    assert_eq!(
                        members,
                        BTreeSet::from([sake_id.clone()]),
                        "failure at persistence step {failure} ({outcome:?}): after \
                         a reboot the member is still the same id, got {members:?}"
                    );
                }
                other => panic!(
                    "failure at persistence step {failure}: the context must \
                     land under exactly one name, not {other:?}"
                ),
            }
            drop(state);
            let _ = fs::remove_dir_all(&dir);

            if past_end {
                assert!(outcome.is_ok());
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "context rename exceeded the sweep bound");
    }

    #[test]
    fn rename_context_error_cases() {
        let dir = scratch_dir("rename-context-errors");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let sake = state.create("sake", ContextMeta::default()).unwrap();
        state.create("beer", ContextMeta::default()).unwrap();

        assert!(matches!(
            state.rename_context(
                &state.id_of("00000000-0000-4000-8000-000000000000"),
                "whatever"
            ),
            Err(RenameContextError::NotFound)
        ));
        assert!(matches!(
            state.rename_context(&sake, ""),
            Err(RenameContextError::InvalidName)
        ));
        // Names are not unique (issue #961 decision 1): renaming onto
        // a name already in use succeeds, and both contexts answer
        // under it side by side.
        state.rename_context(&sake, "beer").unwrap();
        assert!(
            state.directory_entry("beer").is_none(),
            "two claimants: unique() refuses"
        );
        assert_eq!(
            state
                .directory()
                .iter()
                .filter(|entry| entry.name == "beer")
                .count(),
            2
        );
        state.rename_context(&sake, "sake").unwrap();
        assert!(
            state.rename_context(&sake, "sake").is_ok(),
            "renaming a name to itself is a no-op, not an error"
        );
        assert!(state.directory_entry("sake").is_some());

        let _ = fs::remove_dir_all(dir);
    }

    /// The race the retired name reservations used to guard: a create
    /// of either name while a rename is stalled mid-flight. Under
    /// minted ids nothing collides — each create lands on its own
    /// stem, the rename touches only its own sidecar — so both must
    /// simply succeed, leaving duplicate display names behind (issue
    /// #961 decision 1).
    #[test]
    fn creates_racing_a_stalled_rename_land_beside_it() {
        let dir = scratch_dir("rename-create-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let sake = state.create("sake", ContextMeta::default()).unwrap();

        let entry = state.lookup_named("sake").unwrap();
        let stall = entry.inner.read();
        let renamer = {
            let state = state.clone();
            let sake = sake.clone();
            std::thread::spawn(move || state.rename_context(&sake, "shochu").unwrap())
        };
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("shochu", ContextMeta::default()).unwrap();

        drop(stall);
        renamer.join().unwrap();
        // The renamed entry and the racing create now share "shochu".
        assert_eq!(
            state
                .directory()
                .iter()
                .filter(|entry| entry.name == "shochu")
                .count(),
            2
        );
        assert_eq!(
            state
                .directory()
                .iter()
                .filter(|entry| entry.name == "sake")
                .count(),
            1
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A rename tombstones nothing: the entry IS the same `context`,
    /// so a handle cloned before the rename keeps working — only the
    /// old NAME stops answering.
    #[test]
    fn a_handle_from_before_a_rename_stays_valid_and_the_old_name_stops_answering() {
        let dir = scratch_dir("rename-handle-survives");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let entry = state.lookup_named("sake").unwrap();
        state
            .rename_context(&state.id_of("sake"), "shochu")
            .unwrap();
        assert!(
            entry.read_unless_deleted().is_some(),
            "a rename must not tombstone the entry — it is the same context"
        );
        assert!(
            matches!(
                state.add_associations(
                    &state.id_of("sake"),
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                    Deadline::unbounded(),
                ),
                Err(AccessError::NotFound)
            ),
            "the old name no longer answers"
        );
        state
            .add_associations(
                &state.id_of("shochu"),
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        let _ = fs::remove_dir_all(dir);
    }

    fn valid_schema_document() -> schema::SchemaDocument {
        schema::SchemaDocument {
            record_type: schema::SchemaType::Schema,
            version: Some(crate::format::FORMAT_VERSION.to_string()),
            mode: schema::SchemaMode::Strict,
            closed_labels: false,
            types: BTreeMap::from([("Brewery".to_string(), schema::TypeDef::default())]),
            relations: BTreeMap::new(),
        }
    }

    /// The core #380 contract: a `PUT` bumps the `config` revision,
    /// persists the digest to the sidecar, echoes `schema_mode`, and
    /// re-mints `cache_identity` (ADR 0009 §5.2) — exactly what
    /// `bump_config_revision` already does for `dice_floor`, plus the
    /// identity re-mint.
    #[test]
    fn put_schema_bumps_config_revision_persists_the_digest_and_mints_a_fresh_cache_identity() {
        let dir = scratch_dir("put-schema-basic");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let before_revision = state.directory_entry("sake").unwrap().revision.config;
        let before_identity = state
            .lookup_named("sake")
            .unwrap()
            .inner
            .read()
            .cache_identity;

        let installed = schema::install(valid_schema_document()).unwrap();
        let document = state
            .put_schema(&state.id_of("sake"), installed)
            .unwrap()
            .unwrap();
        assert_eq!(document.mode, schema::SchemaMode::Strict);

        let entry = state.directory_entry("sake").unwrap();
        assert_eq!(entry.revision.config, before_revision + 1);
        assert_eq!(entry.schema_mode.as_deref(), Some("strict"));

        let after_identity = state
            .lookup_named("sake")
            .unwrap()
            .inner
            .read()
            .cache_identity;
        assert_ne!(
            before_identity, after_identity,
            "a schema PUT must re-mint cache_identity so a retrieval-cache key minted \
             before it becomes unreachable (ADR 0009 §5.2)"
        );

        let sidecar = read_meta_file(&dir, &state.stem_of("sake").unwrap());
        let bytes = fs::read(schema_path(&dir, &state.stem_of("sake").unwrap())).unwrap();
        assert_eq!(
            sidecar.schema_digest.as_deref(),
            Some(crate::sha256::sha256_hex(&bytes).as_str()),
            "the recorded digest must match the bytes actually on disk"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// "A PUT that changes nothing bumps nothing" — `update_meta`'s own
    /// idempotent-update discipline, mirrored here so a retried or
    /// duplicate `PUT` of the identical document never churns the
    /// retrieval cache.
    #[test]
    fn a_repeated_put_of_the_same_document_bumps_nothing() {
        let dir = scratch_dir("put-schema-noop");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let installed = schema::install(valid_schema_document()).unwrap();
        state
            .put_schema(&state.id_of("sake"), installed)
            .unwrap()
            .unwrap();
        let revision_after_first = state.directory_entry("sake").unwrap().revision.config;
        let identity_after_first = state
            .lookup_named("sake")
            .unwrap()
            .inner
            .read()
            .cache_identity;

        let installed_again = schema::install(valid_schema_document()).unwrap();
        state
            .put_schema(&state.id_of("sake"), installed_again)
            .unwrap()
            .unwrap();

        let entry = state.directory_entry("sake").unwrap();
        assert_eq!(
            entry.revision.config, revision_after_first,
            "identical content must not bump the revision"
        );
        assert_eq!(
            state
                .lookup_named("sake")
                .unwrap()
                .inner
                .read()
                .cache_identity,
            identity_after_first,
            "identical content must not re-mint cache_identity"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// ADR 0009 §6.3 guard 3's migration-boundary counterpart: an
    /// already-persisted `label_alias` resolving to the reserved type
    /// label refuses the `PUT` outright — nothing written, not even
    /// the sidecar digest.
    #[test]
    fn put_schema_refuses_when_a_persisted_label_alias_resolves_to_the_reserved_type_label() {
        let dir = scratch_dir("put-schema-reserved-alias");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        // Legal today — guard 1: `schema:type` is an ordinary label in
        // a context with no installed schema — and it interns the
        // label id `add_label_alias`'s canonical must resolve against.
        state
            .add_associations(
                &state.id_of("sake"),
                vec![assoc_op(
                    "蔵",
                    schema::SCHEMA_TYPE_LABEL,
                    "Brewery",
                    1.0,
                    Some("a.md"),
                )],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state
            .add_aliases(
                &state.id_of("sake"),
                &BTreeMap::new(),
                &BTreeMap::from([("種別".to_string(), schema::SCHEMA_TYPE_LABEL.to_string())]),
            )
            .unwrap()
            .unwrap();

        let installed = schema::install(valid_schema_document()).unwrap();
        let error = state
            .put_schema(&state.id_of("sake"), installed)
            .unwrap()
            .unwrap_err();
        assert!(
            matches!(&error, PutSchemaError::ReservedAlias(alias) if alias == "種別"),
            "{error:?}"
        );
        assert!(
            !schema_path(&dir, &state.stem_of("sake").unwrap()).exists(),
            "a refused PUT must not write the schema file"
        );
        assert_eq!(
            read_meta_file(&dir, &state.stem_of("sake").unwrap()).schema_digest,
            None,
            "a refused PUT must not touch the sidecar's digest"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// ADR 0009 §5.2's write order, proven rather than merely asserted
    /// in a comment: force the schema file's own write to fail right
    /// after the sidecar's `write_meta` already landed (2 persistence
    /// checkpoints — `write_meta`'s one `write_atomic` call's stage +
    /// commit — succeed, then the schema file's own stage fails). The
    /// sidecar must already be durable by the time the schema file
    /// write is even attempted, and the best-effort restore this
    /// failure triggers must bring the sidecar back to its exact
    /// pre-PUT state (the restore's own write is unfaulted, since the
    /// injector is single-shot).
    #[test]
    fn a_schema_file_write_failure_rolls_back_the_sidecar_after_the_revision_already_landed() {
        let dir = scratch_dir("put-schema-write-order");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let before = read_meta_file(&dir, &state.stem_of("sake").unwrap());

        let installed = schema::install(valid_schema_document()).unwrap();
        fail_persistence_ops_after(2);
        let error = state
            .put_schema(&state.id_of("sake"), installed)
            .unwrap()
            .unwrap_err();
        let exhausted = clear_persistence_fault();
        assert!(!exhausted, "the fault must have fired, not merely run out");
        assert!(matches!(error, PutSchemaError::Io(_)), "{error:?}");

        assert!(
            !schema_path(&dir, &state.stem_of("sake").unwrap()).exists(),
            "the schema file must never land when its own write fails"
        );
        let after = read_meta_file(&dir, &state.stem_of("sake").unwrap());
        assert_eq!(after.schema_digest, before.schema_digest);
        assert_eq!(after.revision.config, before.revision.config);

        let entry = state.lookup_named("sake").unwrap();
        let inner = entry.inner.read();
        assert_eq!(inner.schema_digest, before.schema_digest);
        assert_eq!(inner.config_revision, before.revision.config);
        assert!(inner.schema.is_none());
        drop(inner);

        let _ = fs::remove_dir_all(&dir);
    }

    /// `schema_of`'s two direct cases: a schema-free `context` answers
    /// `Ok(None)` without touching disk, and a missing `context` answers
    /// the outer `None` — both without a `PUT` ever having run.
    #[test]
    fn schema_of_reports_a_schema_free_context_and_a_missing_one() {
        let dir = scratch_dir("schema-of-absent");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        assert!(
            state
                .schema_of(&state.id_of("sake"))
                .unwrap()
                .unwrap()
                .is_none(),
            "a fresh context has no schema"
        );
        assert!(state.schema_of(&state.id_of("nope")).is_none());
        assert!(
            state
                .put_schema(
                    &state.id_of("nope"),
                    schema::install(valid_schema_document()).unwrap()
                )
                .is_none(),
            "a PUT against a context that never existed must answer the outer None, \
             not a PutSchemaError"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// `lookup` — the first step of both `schema_of` and `put_schema` —
    /// already answers `None` for a name `delete` has removed from the
    /// registry, so both report the outer `None` for a deleted `context`
    /// exactly like a never-created one, without either method needing
    /// its own tombstone-detection logic beyond the shared
    /// `read_unless_deleted`/`lock_unless_deleted` gate every other
    /// post-lookup operation in this file already goes through.
    #[test]
    fn schema_of_and_put_schema_report_the_outer_none_for_a_deleted_context() {
        let dir = scratch_dir("schema-of-deleted");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.delete(&state.id_of("sake")).unwrap().unwrap();

        assert!(state.schema_of(&state.id_of("sake")).is_none());
        assert!(
            state
                .put_schema(
                    &state.id_of("sake"),
                    schema::install(valid_schema_document()).unwrap()
                )
                .is_none()
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// `hidden_label`'s `Err` arm (`Err(_) => Some(SCHEMA_TYPE_LABEL)`)
    /// has no test — `hidden_label` is never called from any test in
    /// the suite. The digest-recorded-but-schema-unresolved state a
    /// rename used to produce is now the replica registration's (a
    /// family registered from its meta alone, `cold_from_meta` with
    /// `schema: None`); construct it directly — drop the resident
    /// schema by hand — and a corrupted image then makes the
    /// `ensure_hot` inside `schema_of` fail, so `schema_of` itself
    /// returns `Err`. `hidden_label` must fail CLOSED on that —
    /// report hidden, the same as a schema actually present — rather
    /// than let a resolution failure silently unhide a schema-gated
    /// `context`.
    #[test]
    fn hidden_label_fails_closed_when_schema_resolution_errors() {
        let dir = scratch_dir("hidden-label-schema-err");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let installed = schema::install(valid_schema_document()).unwrap();
        state
            .put_schema(&state.id_of("sake"), installed)
            .unwrap()
            .unwrap();
        state
            .rename_context(&state.id_of("sake"), "shochu")
            .unwrap();
        {
            let entry = state.lookup_named("shochu").unwrap();
            let mut inner = entry.inner.write();
            inner.slot = Slot::Cold;
            inner.schema = None;
        }
        assert!(
            state
                .lookup_named("shochu")
                .unwrap()
                .inner
                .read()
                .schema
                .is_none(),
            "sanity: the digest must be recorded while the schema is unresolved"
        );

        let image = image_path(&dir, &state.stem_of("shochu").unwrap());
        let mut bytes = fs::read(&image).unwrap();
        assert!(bytes.len() > 8, "sanity: the version byte must exist");
        bytes[8] = 0xFF;
        fs::write(&image, &bytes).unwrap();

        assert!(
            matches!(state.schema_of(&state.id_of("shochu")), Some(Err(_))),
            "sanity: the corrupt image must make schema_of itself fail"
        );
        assert_eq!(
            state.hidden_label(&state.id_of("shochu")),
            Some(schema::SCHEMA_TYPE_LABEL),
            "a schema-resolution failure must report hidden, not \
             silently unhide a schema-gated context"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// A delete that cannot clear the `context`'s import marker reports
    /// the failure — the marker survives beside the tombstone and boot
    /// must get the chance to finish the job.
    #[test]
    fn a_delete_that_cannot_clear_its_import_marker_reports_it() {
        let dir = scratch_dir("delete-stuck-marker");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let marker = dir.join(format!(
            "{}.batch.{}",
            state.stem_of("sake").unwrap(),
            crate::registry::paths::IMPORT_MARKER_EXTENSION
        ));
        fs::create_dir_all(&marker).unwrap();
        assert!(
            matches!(
                state.delete(&state.id_of("sake")),
                Some(Err(DeleteError::Io(_)))
            ),
            "an unremovable marker must surface through the delete"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The delete-vs-rename race the retired `pending.renames`
    /// reservation used to guard, exercised at its two seams: a
    /// rename whose entry a delete already tombstoned reports
    /// NotFound (the sidecar the delete unlinked is never rewritten),
    /// and the registry keeps no index row for either name — the
    /// reindex-after-delete guard in `rename_context`.
    #[test]
    fn a_rename_losing_to_a_delete_reports_not_found_and_leaks_no_index_row() {
        let dir = scratch_dir("delete-then-rename");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let id = state.create("sake", ContextMeta::default()).unwrap();
        let entry = state.lookup_named("sake").unwrap();

        state.delete(&id).unwrap().unwrap();
        assert!(
            matches!(
                state.rename_context(&id, "shochu"),
                Err(RenameContextError::NotFound)
            ),
            "an unregistered id must not rename"
        );
        // The narrower window: the rename resolved its entry BEFORE
        // the delete landed — the tombstone under the entry lock is
        // what stops the sidecar write.
        assert!(
            matches!(
                state.rename_entry(&entry, "shochu"),
                Err(RenameContextError::NotFound)
            ),
            "a tombstoned entry must refuse the sidecar rewrite"
        );
        {
            let registry = state.0.registry.read();
            assert!(!registry.contains_name("sake"));
            assert!(!registry.contains_name("shochu"));
        }
        assert!(!meta_path(&dir, &id).exists(), "the delete's unlink stands");

        let _ = fs::remove_dir_all(&dir);
    }

    /// A meta or schema save that fails must roll the config revision
    /// back to exactly where it stood — the served content never
    /// changed, so neither may the revision.
    #[test]
    #[cfg(unix)]
    fn a_failed_meta_or_schema_save_rolls_the_config_revision_back() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("config-rollback");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state.flush_dirty();
        let config = state.context_revision(&state.id_of("sake")).unwrap().config;

        let lock_down = || {
            let mut perms = fs::metadata(&dir).unwrap().permissions();
            perms.set_mode(0o555);
            fs::set_permissions(&dir, perms).unwrap();
        };
        let restore = || {
            let mut perms = fs::metadata(&dir).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&dir, perms).unwrap();
        };

        lock_down();
        let outcome = state.update_meta(&state.id_of("sake"), None, None, None, Some(0.9));
        restore();
        assert!(matches!(outcome, Some(Err(_))));
        assert_eq!(
            state.context_revision(&state.id_of("sake")).unwrap().config,
            config,
            "a failed meta save must leave the revision untouched"
        );

        let installed = schema::install(valid_schema_document()).unwrap();
        lock_down();
        let outcome = state.put_schema(&state.id_of("sake"), installed);
        restore();
        assert!(matches!(outcome, Some(Err(_))));
        assert_eq!(
            state.context_revision(&state.id_of("sake")).unwrap().config,
            config,
            "a failed schema save must leave the revision untouched"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
