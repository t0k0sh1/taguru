use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::{
    AliasRecord, AttributionId, AttributionLocatorRecord, AttributionRecord, ConceptId,
    ConceptRecord, Context, CorruptImage, EdgeId, EdgeRecord, EntryIndex, LabelId, LabelRecord,
    NIL, SourceId, SourceRecord, accumulate_saturating,
};

// The persistence format depends on these exact widths. A field added to
// any record must keep its table fixed-width, naturally aligned, and
// padding-free — and must bump `IMAGE_VERSION`.
const _: () = {
    assert!(size_of::<ConceptRecord>() == ConceptRecord::SIZE && align_of::<ConceptRecord>() == 4);
    assert!(size_of::<LabelRecord>() == LabelRecord::SIZE && align_of::<LabelRecord>() == 4);
    assert!(size_of::<SourceRecord>() == SourceRecord::SIZE && align_of::<SourceRecord>() == 4);
    assert!(size_of::<EdgeRecord>() == EdgeRecord::SIZE && align_of::<EdgeRecord>() == 8);
    assert!(
        size_of::<AttributionRecord>() == AttributionRecord::SIZE
            && align_of::<AttributionRecord>() == 8
    );
    assert!(size_of::<AliasRecord>() == AliasRecord::SIZE && align_of::<AliasRecord>() == 4);
    assert!(
        size_of::<AttributionLocatorRecord>() == AttributionLocatorRecord::SIZE
            && align_of::<AttributionLocatorRecord>() == 4
    );
};

// ---------------------------------------------------------------------------
// Persistence: dumping and restoring the storage buffers as one image.
// ---------------------------------------------------------------------------

/// First 8 bytes of every image.
const IMAGE_MAGIC: [u8; 8] = *b"TAGURUC\0";
/// Format version; bump whenever any record layout or section changes.
/// Version history: 1 = the original six sections; 2 adds the concept
/// and label alias tables between the sources and the arena; 3 adds
/// the u64 durability watermark to the header; 4 adds the sparse
/// attribution-locator table between the label aliases and the arena;
/// 5 splits `EdgeRecord`/`AttributionRecord`'s cumulative `weight` into
/// a `(count, sum)` pair, growing the records to 48 and 24 bytes so the
/// public `weight` can be derived as `sum / count` (see
/// [`super::Association`]); 6 appends a CRC-32C footer over every
/// preceding byte, verified on load — the bit-rot check the structural
/// validation below cannot give, since corruption that happens to keep
/// every invariant would otherwise load as truth and be flushed back
/// as the new canonical bytes.
/// Older images still load (empty alias/locator tables, watermark 0,
/// legacy edge/attribution rows migrated with a synthesized `count`,
/// pre-6 bytes unverifiable and accepted as they always were);
/// writing always produces the current version.
const IMAGE_VERSION: u32 = 6;
/// Magic + version + 4 bytes of padding (so what follows is 8-byte
/// aligned) + the u64 durability watermark.
const IMAGE_HEADER_SIZE: usize = 24;
/// Eight record tables plus the arena, each section prefixed by a u64
/// length.
const IMAGE_SECTIONS: usize = 9;
/// The little-endian CRC-32C of everything before it, closing every
/// v6+ image.
const IMAGE_FOOTER_SIZE: usize = 4;

impl Context {
    /// Serializes the whole network into one contiguous byte image.
    ///
    /// The image is the storage buffers written back to back: a 24-byte
    /// header, then each record table as a u64 count followed by its
    /// fixed-width records field by field in little-endian, then the
    /// string arena, then the checksum footer over everything before
    /// it. Sections are ordered by descending alignment (the
    /// f64-bearing tables first), so every record sits naturally aligned
    /// within the image as well — the layout stays open to zero-copy
    /// mapping later. The derived hash indexes are not written;
    /// [`Context::from_bytes`] rebuilds them.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut image = self.image_body(IMAGE_VERSION);
        image.extend_from_slice(&crate::crc32c::crc32c(&image).to_le_bytes());
        image
    }

    /// Header and sections in the current record shapes, stamped with
    /// `version` — everything but the checksum footer. [`Context::to_bytes`]
    /// seals it; the test-only v5 writer emits it bare, because v5 IS the
    /// current layout minus the footer.
    fn image_body(&self, version: u32) -> Vec<u8> {
        let mut image = Vec::with_capacity(
            IMAGE_HEADER_SIZE
                + IMAGE_SECTIONS * size_of::<u64>()
                + self.edges.len() * EdgeRecord::SIZE
                + self.attributions.len() * AttributionRecord::SIZE
                + self.concepts.len() * ConceptRecord::SIZE
                + self.labels.len() * LabelRecord::SIZE
                + self.sources.len() * SourceRecord::SIZE
                + (self.concept_aliases.len() + self.label_aliases.len()) * AliasRecord::SIZE
                + self.attribution_locators.len() * AttributionLocatorRecord::SIZE
                + self.arena.len()
                + IMAGE_FOOTER_SIZE,
        );
        image.extend_from_slice(&IMAGE_MAGIC);
        image.extend_from_slice(&version.to_le_bytes());
        image.extend_from_slice(&[0; 4]);
        image.extend_from_slice(&self.applied_seq.to_le_bytes());
        store_table(&self.edges, &mut image);
        store_table(&self.attributions, &mut image);
        store_table(&self.concepts, &mut image);
        store_table(&self.labels, &mut image);
        store_table(&self.sources, &mut image);
        store_table(&self.concept_aliases, &mut image);
        store_table(&self.label_aliases, &mut image);
        store_table(&self.attribution_locators, &mut image);
        image.extend_from_slice(&(self.arena.len() as u64).to_le_bytes());
        image.extend_from_slice(&self.arena);
        image
    }

    /// The format version stamped in `image`'s header and whether that
    /// generation carries a verified checksum footer, or `None` when
    /// the bytes do not even begin as an image. Diagnostic sugar for
    /// `taguru inspect`: after a successful load it says whether the
    /// bytes were actually PROVEN intact (v6+) or merely predate
    /// verifiability — without leaking the version cut-over into the
    /// caller.
    pub fn image_generation(image: &[u8]) -> Option<(u32, bool)> {
        let version = image.get(IMAGE_MAGIC.len()..IMAGE_MAGIC.len() + 4)?;
        image.starts_with(&IMAGE_MAGIC).then(|| {
            let version = u32::from_le_bytes(version.try_into().unwrap());
            (version, version >= 6)
        })
    }

    /// Test-only counterpart to [`Context::to_bytes`] that writes a
    /// genuine older-version image byte-for-byte, so the
    /// version-compatibility tests need not slice `to_bytes`'s
    /// (always-current) output. Version 5 is the current sections
    /// without the checksum footer; versions 1–4 additionally write the
    /// pre-v5, `weight`-only edge/attribution shape, whose records a
    /// current slice would misparse.
    #[cfg(test)]
    pub(super) fn to_bytes_as_version(&self, version: u32) -> Vec<u8> {
        debug_assert!(
            version < IMAGE_VERSION,
            "to_bytes_as_version writes legacy images; \
             the current version is what to_bytes is for"
        );
        if version == 5 {
            return self.image_body(5);
        }
        let legacy_edges: Vec<LegacyEdgeRecord> = self
            .edges
            .iter()
            .map(|edge| LegacyEdgeRecord {
                subject: edge.subject,
                label: edge.label,
                object: edge.object,
                next_outgoing: edge.next_outgoing,
                next_incoming: edge.next_incoming,
                next_labeled: edge.next_labeled,
                first_attribution: edge.first_attribution,
                last_attribution: edge.last_attribution,
                weight: edge.sum,
            })
            .collect();
        let legacy_attributions: Vec<LegacyAttributionRecord> = self
            .attributions
            .iter()
            .map(|record| LegacyAttributionRecord {
                source: record.source,
                next: record.next,
                weight: record.sum,
            })
            .collect();

        let mut image = Vec::new();
        image.extend_from_slice(&IMAGE_MAGIC);
        image.extend_from_slice(&version.to_le_bytes());
        image.extend_from_slice(&[0; 4]);
        if version >= 3 {
            image.extend_from_slice(&self.applied_seq.to_le_bytes());
        }
        store_table(&legacy_edges, &mut image);
        store_table(&legacy_attributions, &mut image);
        store_table(&self.concepts, &mut image);
        store_table(&self.labels, &mut image);
        store_table(&self.sources, &mut image);
        if version >= 2 {
            store_table(&self.concept_aliases, &mut image);
            store_table(&self.label_aliases, &mut image);
        }
        if version >= 4 {
            store_table(&self.attribution_locators, &mut image);
        }
        image.extend_from_slice(&(self.arena.len() as u64).to_le_bytes());
        image.extend_from_slice(&self.arena);
        image
    }

    /// Restores a `Context` from an image produced by
    /// [`Context::to_bytes`], rebuilding the derived indexes.
    ///
    /// The image is fully validated before anything is trusted — magic,
    /// version, and section bounds first, then arena ranges and UTF-8,
    /// id ranges, duplicate names and triples, and every adjacency and
    /// attribution chain (ownership, length, tail, cycles) — so a
    /// truncated or tampered image comes back as an error rather than a
    /// `Context` that panics or corrupts itself later. A restored
    /// `Context` is behaviorally identical to the original, including
    /// insertion-order guarantees, and keeps accepting new associations.
    pub fn from_bytes(image: &[u8]) -> Result<Self, CorruptImage> {
        let mut reader = Reader {
            bytes: image,
            pos: 0,
        };
        if reader.take(IMAGE_MAGIC.len())? != IMAGE_MAGIC {
            return Err(CorruptImage("image does not start with the context magic"));
        }
        let version = reader.read_u32()?;
        // A RANGE check: every version from 1 through current loads.
        // (The old two-value check would have started rejecting v2
        // images the moment a third version existed.)
        if !(1..=IMAGE_VERSION).contains(&version) {
            return Err(CorruptImage("image format version is not supported"));
        }
        // The checksum footer comes off — and gets verified — before
        // anything else is trusted: structural validation proves the
        // image consistent, only the checksum proves it the bytes that
        // were written. Pre-6 images have no footer and load exactly as
        // they always did, unverifiable.
        if version >= 6 {
            let body_len = image
                .len()
                .checked_sub(IMAGE_FOOTER_SIZE)
                .ok_or(CorruptImage("image ends inside its checksum footer"))?;
            let (body, footer) = image.split_at(body_len);
            let stored = u32::from_le_bytes(footer.try_into().unwrap());
            if crate::crc32c::crc32c(body) != stored {
                return Err(CorruptImage(
                    "image checksum mismatch — the bytes changed after they were written",
                ));
            }
            reader.bytes = body;
        }
        reader.take(4)?; // header padding
        // Version 3 adds the durability watermark; older images carry
        // none, and 0 is exactly right for them — no WAL can predate
        // the feature, so "replay everything above 0" is a no-op or
        // correct.
        let applied_seq = if version >= 3 { reader.read_u64()? } else { 0 };

        // Version 5 splits cumulative `weight` into a `(count, sum)`
        // pair; older images carry the pre-split shape. `count` is
        // synthesized as each edge's attribution chain length (floored
        // at 1 for unsourced edges) so it can never be lower than the
        // number of sourced contributions a later `retract_source`
        // subtracts one at a time — the floor that keeps its
        // `saturating_sub` from ever needing to save a migrated image.
        let (edges, attributions) = if version >= 5 {
            (
                load_table::<EdgeRecord>(&mut reader)?,
                load_table::<AttributionRecord>(&mut reader)?,
            )
        } else {
            let legacy_edges = load_table::<LegacyEdgeRecord>(&mut reader)?;
            let legacy_attributions = load_table::<LegacyAttributionRecord>(&mut reader)?;
            let attributions = legacy_attributions
                .iter()
                .map(|legacy| AttributionRecord {
                    source: legacy.source,
                    next: legacy.next,
                    count: 1,
                    sum: legacy.weight,
                })
                .collect();
            let mut edges = Vec::with_capacity(legacy_edges.len());
            // A chain is identified by its head: edges sharing a
            // `first_attribution` walk the identical `next` list and get the
            // same length and sum, so memoize by head. Without this a
            // pathological image whose every edge points at one long shared
            // chain walks that chain once per edge — O(edges × chain) —
            // during a migration that holds the write lock and never yields.
            let mut chains: HashMap<AttributionId, (u64, f64)> = HashMap::new();
            for legacy in &legacy_edges {
                let (chain_len, attributed_sum) = match chains.get(&legacy.first_attribution) {
                    Some(&cached) => cached,
                    None => {
                        let computed = legacy_attribution_chain_len(
                            &legacy_attributions,
                            legacy.first_attribution,
                        )?;
                        chains.insert(legacy.first_attribution, computed);
                        computed
                    }
                };
                // An empty chain (first_attribution == NIL) is ambiguous in
                // the legacy format: it means either an edge that was always
                // sourceless or one that was fully retracted (weight zeroed
                // along with the chain). Weight tells the common shapes of
                // each apart — a sourceless edge is usually nonzero, a
                // retraction always zero — but not perfectly: `associate`
                // never rejected weight 0.0, so a live sourceless edge can
                // also land here at weight 0.0, identical on disk to a
                // retracted one (a pre-v5 attribution record keeps no
                // back-pointer to its edge, so no amount of scanning
                // recovers which this was). That case is a known, accepted
                // false negative — see
                // `migrating_a_pre_v5_image_cannot_tell_a_sourceless_zero_weight_edge_from_a_retracted_one`
                // in context.rs — chosen over the alternative of reviving
                // every empty chain, which would undo the fix below and
                // resurrect the far more common case: an actual retraction.
                //
                // A chain that IS non-empty can still under-represent the
                // edge: `upsert` (see `Context::associate`) folds every
                // assertion — sourced or not — into the edge's cumulative
                // weight, but only a sourced one ever links into the
                // attribution chain, so a sourceless call sitting alongside
                // sourced ones on the same edge leaves no chain record at
                // all. `legacy.weight` still carries its contribution,
                // though, so a gap between it and what the chain accounts
                // for (`attributed_sum`) proves at least one sourceless call
                // happened. Exactly how many is unrecoverable — only their
                // combined weight survived — so credit one for the whole
                // gap: enough that `count` still exceeds the chain's
                // combined attribution count after every source retracts,
                // rather than the edge being declared dead out from under a
                // contribution `retract_source` was never told about. See
                // `migrating_a_pre_v5_image_credits_an_undetected_sourceless_call_so_retraction_cannot_revive_it_as_dead`
                // in context.rs.
                let count = if chain_len == 0 && legacy.weight == 0.0 {
                    0
                } else if legacy.weight != attributed_sum {
                    chain_len + 1
                } else {
                    chain_len.max(1)
                };
                edges.push(EdgeRecord {
                    subject: legacy.subject,
                    label: legacy.label,
                    object: legacy.object,
                    next_outgoing: legacy.next_outgoing,
                    next_incoming: legacy.next_incoming,
                    next_labeled: legacy.next_labeled,
                    first_attribution: legacy.first_attribution,
                    last_attribution: legacy.last_attribution,
                    count,
                    sum: legacy.weight,
                });
            }
            (edges, attributions)
        };
        let concepts = load_table::<ConceptRecord>(&mut reader)?;
        let labels = load_table::<LabelRecord>(&mut reader)?;
        let sources = load_table::<SourceRecord>(&mut reader)?;
        // Version 1 predates aliases; its images simply have none.
        let (concept_aliases, label_aliases) = if version >= 2 {
            (
                load_table::<AliasRecord>(&mut reader)?,
                load_table::<AliasRecord>(&mut reader)?,
            )
        } else {
            (Vec::new(), Vec::new())
        };
        // Version 4 adds attribution locators; older images have none, so
        // every attribution's paragraph resolves to `None` — exactly the
        // "no locator" state a same-version write with no paragraph data
        // would also produce.
        let attribution_locators = if version >= 4 {
            load_table::<AttributionLocatorRecord>(&mut reader)?
        } else {
            Vec::new()
        };
        let arena_len = usize::try_from(reader.read_u64()?)
            .map_err(|_| CorruptImage("arena length overflows this platform"))?;
        // Record name_offset/name_len fields are u32 (intern_name
        // asserts this same bound on the write side), so an arena this
        // large could never have been produced by this binary and
        // could not be addressed into correctly if it were — reject it
        // here rather than load a Context that panics on its first
        // intern_name call.
        if arena_len > u32::MAX as usize {
            return Err(CorruptImage("arena length exceeds its 4 GiB offset space"));
        }
        let arena = reader.take(arena_len)?.to_vec();
        // Against the reader's own slice, not `image`: on v6+ the
        // checksum footer was already split off above.
        if reader.pos != reader.bytes.len() {
            return Err(CorruptImage("image carries trailing bytes"));
        }

        let mut context = Context {
            arena,
            concepts,
            labels,
            sources,
            edges,
            attributions,
            concept_aliases,
            label_aliases,
            attribution_locators,
            concept_ids: HashMap::new(),
            label_ids: HashMap::new(),
            source_ids: HashMap::new(),
            concept_alias_index: BTreeMap::new(),
            label_alias_index: BTreeMap::new(),
            label_name_index: BTreeSet::new(),
            edge_ids: HashMap::new(),
            attribution_ids: HashMap::new(),
            source_edges: HashMap::new(),
            concept_index: EntryIndex::default(),
            label_index: EntryIndex::default(),
            dice_floor: None,
            applied_seq,
            // Seeded below by `rebuild_indexes` — none of these are
            // persisted.
            dead_edges: 0,
            dead_attributions: 0,
            arena_slack: 0,
        };
        context.rebuild_indexes()?;
        Ok(context)
    }

    /// Validates a freshly loaded image and rebuilds the derived indexes
    /// from the flat buffers. Called only by `from_bytes`, on a `Context`
    /// whose index maps are still empty. The `index_` phases build maps
    /// as they check; the `validate_` phases only check.
    fn rebuild_indexes(&mut self) -> Result<(), CorruptImage> {
        self.index_names()?;
        self.index_aliases()?;
        self.index_edges()?;
        self.validate_chains()?;
        self.index_attributions()?;
        self.validate_locators()?;
        self.seed_arena_slack();
        Ok(())
    }

    /// Strings: every name range must be a valid arena slice, and names
    /// must be unique per namespace or lookups would be ambiguous.
    fn index_names(&mut self) -> Result<(), CorruptImage> {
        for (id, record) in self.concepts.iter().enumerate() {
            let name = checked_arena_str(&self.arena, record.name_offset, record.name_len)?;
            self.concept_index.push(name, id as u32);
            if self
                .concept_ids
                .insert(name.to_string(), id as u32)
                .is_some()
            {
                return Err(CorruptImage("two concept records share one name"));
            }
        }
        for (id, record) in self.labels.iter().enumerate() {
            let name = checked_arena_str(&self.arena, record.name_offset, record.name_len)?;
            self.label_index.push(name, id as u32);
            if self.label_ids.insert(name.to_string(), id as u32).is_some() {
                return Err(CorruptImage("two label records share one name"));
            }
            self.label_name_index.insert(name.to_string());
        }
        for (id, record) in self.sources.iter().enumerate() {
            let name = checked_arena_str(&self.arena, record.name_offset, record.name_len)?;
            if self
                .source_ids
                .insert(name.to_string(), id as u32)
                .is_some()
            {
                return Err(CorruptImage("two source records share one name"));
            }
        }
        Ok(())
    }

    /// Aliases join the lookup maps after the canonical names, so any
    /// spelling collision — with a name or another alias — surfaces.
    fn index_aliases(&mut self) -> Result<(), CorruptImage> {
        for record in &self.concept_aliases {
            let alias = checked_arena_str(&self.arena, record.name_offset, record.name_len)?;
            if record.target as usize >= self.concepts.len() {
                return Err(CorruptImage("concept alias targets an unknown concept"));
            }
            self.concept_index.push(alias, record.target);
            if self
                .concept_ids
                .insert(alias.to_string(), record.target)
                .is_some()
            {
                return Err(CorruptImage("concept alias collides with another spelling"));
            }
            self.concept_alias_index
                .insert(alias.to_string(), record.target);
        }
        for record in &self.label_aliases {
            let alias = checked_arena_str(&self.arena, record.name_offset, record.name_len)?;
            if record.target as usize >= self.labels.len() {
                return Err(CorruptImage("label alias targets an unknown label"));
            }
            self.label_index.push(alias, record.target);
            if self
                .label_ids
                .insert(alias.to_string(), record.target)
                .is_some()
            {
                return Err(CorruptImage("label alias collides with another spelling"));
            }
            self.label_alias_index
                .insert(alias.to_string(), record.target);
        }
        Ok(())
    }

    /// Edges: endpoints must exist, and triples must be unique — the
    /// accumulate-on-repeat contract depends on it.
    fn index_edges(&mut self) -> Result<(), CorruptImage> {
        for (id, edge) in self.edges.iter().enumerate() {
            if edge.subject as usize >= self.concepts.len()
                || edge.object as usize >= self.concepts.len()
                || edge.label as usize >= self.labels.len()
            {
                return Err(CorruptImage("edge references an unknown concept or label"));
            }
            let key = (edge.subject, edge.label, edge.object);
            if self.edge_ids.insert(key, id as u32).is_some() {
                return Err(CorruptImage("two edge records share one triple"));
            }
        }
        Ok(())
    }

    /// Chains: each must contain exactly its stored count of edges, all
    /// owned by the anchoring record, ending at the stored tail, with
    /// no cycles — and together the chains of a kind must cover every
    /// edge, or some knowledge would be silently unreachable.
    fn validate_chains(&self) -> Result<(), CorruptImage> {
        for (id, record) in self.concepts.iter().enumerate() {
            let id = id as u32;
            validate_edge_chain(
                &self.edges,
                record.first_outgoing,
                record.last_outgoing,
                record.outgoing_count,
                |edge| edge.next_outgoing,
                |edge| edge.subject == id,
            )?;
            validate_edge_chain(
                &self.edges,
                record.first_incoming,
                record.last_incoming,
                record.incoming_count,
                |edge| edge.next_incoming,
                |edge| edge.object == id,
            )?;
        }
        for (id, record) in self.labels.iter().enumerate() {
            let id = id as u32;
            validate_edge_chain(
                &self.edges,
                record.first_edge,
                record.last_edge,
                record.edge_count,
                |edge| edge.next_labeled,
                |edge| edge.label == id,
            )?;
        }
        let edge_total = self.edges.len() as u64;
        let outgoing_total: u64 = self
            .concepts
            .iter()
            .map(|r| u64::from(r.outgoing_count))
            .sum();
        let incoming_total: u64 = self
            .concepts
            .iter()
            .map(|r| u64::from(r.incoming_count))
            .sum();
        let labeled_total: u64 = self.labels.iter().map(|r| u64::from(r.edge_count)).sum();
        if outgoing_total != edge_total
            || incoming_total != edge_total
            || labeled_total != edge_total
        {
            return Err(CorruptImage("edge chains do not cover the edge table"));
        }
        Ok(())
    }

    /// Attribution chains: in range, sources known, acyclic, ending at
    /// the stored tail, and disjoint across edges — a shared record
    /// would let one edge's accumulation corrupt another's. Records
    /// claimed by NO chain are legal: retraction unlinks records in
    /// place and leaves them behind as dead space in the append-only
    /// table.
    ///
    /// While walking each chain — the one pass that already visits every
    /// live attribution — this also builds the derived `(edge, source)`
    /// index the write path relies on. That folds the one-source-per-edge
    /// invariant into validation: the write path never links a source
    /// twice onto one edge, so a duplicate here is a tampered chain and is
    /// rejected rather than silently collapsed in the index.
    ///
    /// Also re-establishes the numeric guarantees the write path
    /// enforces but the flat buffer cannot: every consumed weight `sum`
    /// is finite (a tampered `NaN`/`Inf` would sort as the maximum under
    /// `total_cmp` and permanently occupy every ranked result), and the
    /// combined chain `count` does not overflow `u64` (which would wrap
    /// past the `edge.count` floor and silently defeat the check below).
    fn index_attributions(&mut self) -> Result<(), CorruptImage> {
        let mut claimed = vec![false; self.attributions.len()];
        // Piggybacks on this same walk to seed the two incremental dead-
        // weight counters: `edge.count == 0` — the invariant checked
        // below (`edge.count < chain_count` is corrupt) means a dead
        // edge's chain is always empty — is the only place `dead_edges`
        // can be counted without a second pass, and `claimed` already
        // tells us exactly which attribution records no chain reached.
        let mut dead_edges = 0usize;
        for edge_id in 0..self.edges.len() as u32 {
            let edge = self.edges[edge_id as usize];
            if edge.count == 0 {
                dead_edges += 1;
            }
            if !edge.sum.is_finite() {
                return Err(CorruptImage("edge weight sum is not finite"));
            }
            let mut cursor = edge.first_attribution;
            let mut tail = NIL;
            let mut chain_count: u64 = 0;
            while cursor != NIL {
                let record = *self
                    .attributions
                    .get(cursor as usize)
                    .ok_or(CorruptImage("attribution link is out of range"))?;
                if std::mem::replace(&mut claimed[cursor as usize], true) {
                    return Err(CorruptImage("attribution record belongs to two chains"));
                }
                if record.source as usize >= self.sources.len() {
                    return Err(CorruptImage("attribution references an unknown source"));
                }
                if !record.sum.is_finite() {
                    return Err(CorruptImage("attribution weight sum is not finite"));
                }
                if self
                    .attribution_ids
                    .insert((edge_id, record.source), cursor)
                    .is_some()
                {
                    return Err(CorruptImage("one edge attributes a source twice"));
                }
                self.source_edges
                    .entry(record.source)
                    .or_default()
                    .push(edge_id);
                // A live chained record always carries a positive count:
                // the writer unlinks a record the instant retraction drains
                // it to zero, so a zero-count record never lingers in the
                // chain. Verify it rather than assume it — this is what makes
                // `edge.count == 0` above imply an empty chain (any non-empty
                // chain now sums to >= 1, tripping the count check below), so
                // the dead-edge tally cannot over-count a crafted image.
                if record.count == 0 {
                    return Err(CorruptImage("attribution record carries a zero count"));
                }
                chain_count = chain_count
                    .checked_add(record.count)
                    .ok_or(CorruptImage("attribution chain count overflows u64"))?;
                tail = cursor;
                cursor = record.next;
            }
            if tail != edge.last_attribution {
                return Err(CorruptImage("attribution chain does not end at its tail"));
            }
            if edge.count < chain_count {
                return Err(CorruptImage(
                    "edge count is lower than its attributions' combined count",
                ));
            }
        }
        self.dead_edges = dead_edges;
        self.dead_attributions = claimed.iter().filter(|&&c| !c).count();
        Ok(())
    }

    /// Locators: each must name a real attribution record, and the table
    /// must be strictly increasing by `attribution` — the invariant
    /// `Context`'s binary-search lookup depends on. This crate's own
    /// writer always upholds it (a locator is only ever appended
    /// alongside a brand-new, higher-numbered attribution record), so a
    /// violation here means a tampered or hand-built image.
    fn validate_locators(&self) -> Result<(), CorruptImage> {
        let mut previous: Option<AttributionId> = None;
        for record in &self.attribution_locators {
            if record.attribution as usize >= self.attributions.len() {
                return Err(CorruptImage("locator references an unknown attribution"));
            }
            if previous.is_some_and(|prev| prev >= record.attribution) {
                return Err(CorruptImage("locator table is not sorted by attribution"));
            }
            previous = Some(record.attribution);
        }
        Ok(())
    }

    /// Seeds `arena_slack` on load. Concept, label, and source names are
    /// never deleted — only aliases can be — so every arena byte not
    /// currently spanned by a live record in one of the five name
    /// tables can only be the spelling of a since-removed alias; the
    /// residual (arena length minus every live name's byte span) is
    /// therefore an exact count of that dead weight, not an estimate.
    fn seed_arena_slack(&mut self) {
        fn sum_name_len<T>(records: &[T], name_len: impl Fn(&T) -> u32) -> usize {
            records.iter().map(|record| name_len(record) as usize).sum()
        }
        let live = sum_name_len(&self.concepts, |r| r.name_len)
            + sum_name_len(&self.labels, |r| r.name_len)
            + sum_name_len(&self.sources, |r| r.name_len)
            + sum_name_len(&self.concept_aliases, |r| r.name_len)
            + sum_name_len(&self.label_aliases, |r| r.name_len);
        self.arena_slack = self.arena.len().saturating_sub(live);
    }
}

/// Reads one interned string out of an untrusted arena, validating the
/// range and UTF-8 — the load-time counterpart of [`Context::arena_str`].
fn checked_arena_str(arena: &[u8], offset: u32, len: u32) -> Result<&str, CorruptImage> {
    let start = offset as usize;
    let end = start
        .checked_add(len as usize)
        .ok_or(CorruptImage("name range overflows"))?;
    let bytes = arena
        .get(start..end)
        .ok_or(CorruptImage("name range escapes the arena"))?;
    std::str::from_utf8(bytes).map_err(|_| CorruptImage("name is not valid UTF-8"))
}

/// Counts one legacy edge's attribution chain length and combined weight
/// for the v5 migration in [`Context::from_bytes`]. Defensive rather than
/// trusting: this runs before `index_attributions` has ever looked at the
/// chain, so a hostile or truncated pre-v5 image must not send it out of
/// bounds or looping forever on a cycle.
fn legacy_attribution_chain_len(
    attributions: &[LegacyAttributionRecord],
    mut cursor: AttributionId,
) -> Result<(u64, f64), CorruptImage> {
    let mut len = 0u64;
    let mut sum = 0.0f64;
    let mut steps: usize = 0;
    while cursor != NIL {
        steps += 1;
        if steps > attributions.len() {
            return Err(CorruptImage(
                "legacy attribution chain cycles during migration",
            ));
        }
        let record = attributions
            .get(cursor as usize)
            .ok_or(CorruptImage("legacy attribution link is out of range"))?;
        len += 1;
        accumulate_saturating(&mut sum, record.weight);
        cursor = record.next;
    }
    Ok((len, sum))
}

/// Checks that one linked chain of edges is exactly `count` records long,
/// stays in bounds, contains only edges anchored by its owner, ends at
/// `last`, and cannot cycle (a chain longer than the whole table must
/// repeat a record).
fn validate_edge_chain(
    edges: &[EdgeRecord],
    first: EdgeId,
    last: EdgeId,
    count: u32,
    follow: fn(&EdgeRecord) -> EdgeId,
    owned: impl Fn(&EdgeRecord) -> bool,
) -> Result<(), CorruptImage> {
    let mut cursor = first;
    let mut tail = NIL;
    let mut steps: usize = 0;
    while cursor != NIL {
        steps += 1;
        if steps > count as usize || steps > edges.len() {
            return Err(CorruptImage("edge chain overruns its stored count"));
        }
        let record = edges
            .get(cursor as usize)
            .ok_or(CorruptImage("edge chain link is out of range"))?;
        if !owned(record) {
            return Err(CorruptImage("edge chain contains another record's edge"));
        }
        tail = cursor;
        cursor = follow(record);
    }
    if steps != count as usize {
        return Err(CorruptImage("edge chain is shorter than its stored count"));
    }
    if tail != last {
        return Err(CorruptImage("edge chain does not end at its stored tail"));
    }
    Ok(())
}

/// One fixed-width table row: how many image bytes it spans and how to
/// store/load it, field by field in declaration order, little-endian.
trait Record: Sized {
    const SIZE: usize;
    fn store(&self, image: &mut Vec<u8>);
    fn load(reader: &mut Reader) -> Result<Self, CorruptImage>;
}

/// Writes one table as a u64 record count followed by its records.
fn store_table<T: Record>(records: &[T], image: &mut Vec<u8>) {
    image.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for record in records {
        record.store(image);
    }
}

/// Reads one table written by [`store_table`], bounding the record count
/// by the bytes actually present so a hostile count cannot balloon memory.
fn load_table<T: Record>(reader: &mut Reader) -> Result<Vec<T>, CorruptImage> {
    let count = reader.read_u64()?;
    // Ids run 0..count and `NIL` (u32::MAX) is the reserved sentinel, so a
    // table holds at most `NIL` records — its highest id is then NIL-1, one
    // below the sentinel. Only a count STRICTLY past NIL overflows the id
    // space; `>=` wrongly rejected a legitimately maximal table.
    if count > u64::from(NIL) {
        return Err(CorruptImage("table exceeds the u32 id space"));
    }
    let count = count as usize;
    let bytes_needed = count
        .checked_mul(T::SIZE)
        .ok_or(CorruptImage("table byte size overflows"))?;
    if bytes_needed > reader.remaining() {
        return Err(CorruptImage("table is truncated"));
    }
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        records.push(T::load(reader)?);
    }
    Ok(records)
}

impl Record for ConceptRecord {
    const SIZE: usize = 32;

    fn store(&self, image: &mut Vec<u8>) {
        for field in [
            self.name_offset,
            self.name_len,
            self.first_outgoing,
            self.last_outgoing,
            self.outgoing_count,
            self.first_incoming,
            self.last_incoming,
            self.incoming_count,
        ] {
            image.extend_from_slice(&field.to_le_bytes());
        }
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            name_offset: reader.read_u32()?,
            name_len: reader.read_u32()?,
            first_outgoing: reader.read_u32()?,
            last_outgoing: reader.read_u32()?,
            outgoing_count: reader.read_u32()?,
            first_incoming: reader.read_u32()?,
            last_incoming: reader.read_u32()?,
            incoming_count: reader.read_u32()?,
        })
    }
}

impl Record for LabelRecord {
    const SIZE: usize = 20;

    fn store(&self, image: &mut Vec<u8>) {
        for field in [
            self.name_offset,
            self.name_len,
            self.first_edge,
            self.last_edge,
            self.edge_count,
        ] {
            image.extend_from_slice(&field.to_le_bytes());
        }
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            name_offset: reader.read_u32()?,
            name_len: reader.read_u32()?,
            first_edge: reader.read_u32()?,
            last_edge: reader.read_u32()?,
            edge_count: reader.read_u32()?,
        })
    }
}

impl Record for SourceRecord {
    const SIZE: usize = 8;

    fn store(&self, image: &mut Vec<u8>) {
        image.extend_from_slice(&self.name_offset.to_le_bytes());
        image.extend_from_slice(&self.name_len.to_le_bytes());
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            name_offset: reader.read_u32()?,
            name_len: reader.read_u32()?,
        })
    }
}

impl Record for AliasRecord {
    const SIZE: usize = 12;

    fn store(&self, image: &mut Vec<u8>) {
        image.extend_from_slice(&self.name_offset.to_le_bytes());
        image.extend_from_slice(&self.name_len.to_le_bytes());
        image.extend_from_slice(&self.target.to_le_bytes());
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            name_offset: reader.read_u32()?,
            name_len: reader.read_u32()?,
            target: reader.read_u32()?,
        })
    }
}

impl Record for EdgeRecord {
    const SIZE: usize = 48;

    fn store(&self, image: &mut Vec<u8>) {
        for field in [
            self.subject,
            self.label,
            self.object,
            self.next_outgoing,
            self.next_incoming,
            self.next_labeled,
            self.first_attribution,
            self.last_attribution,
        ] {
            image.extend_from_slice(&field.to_le_bytes());
        }
        image.extend_from_slice(&self.count.to_le_bytes());
        image.extend_from_slice(&self.sum.to_le_bytes());
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            subject: reader.read_u32()?,
            label: reader.read_u32()?,
            object: reader.read_u32()?,
            next_outgoing: reader.read_u32()?,
            next_incoming: reader.read_u32()?,
            next_labeled: reader.read_u32()?,
            first_attribution: reader.read_u32()?,
            last_attribution: reader.read_u32()?,
            count: reader.read_u64()?,
            sum: reader.read_f64()?,
        })
    }
}

impl Record for AttributionRecord {
    const SIZE: usize = 24;

    fn store(&self, image: &mut Vec<u8>) {
        image.extend_from_slice(&self.source.to_le_bytes());
        image.extend_from_slice(&self.next.to_le_bytes());
        image.extend_from_slice(&self.count.to_le_bytes());
        image.extend_from_slice(&self.sum.to_le_bytes());
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            source: reader.read_u32()?,
            next: reader.read_u32()?,
            count: reader.read_u64()?,
            sum: reader.read_f64()?,
        })
    }
}

/// The pre-v5 shape of [`EdgeRecord`] (40 bytes: no `count`, `weight` is
/// the raw cumulative sum). Exists only so [`Context::from_bytes`] can
/// still read images written before version 5; every new image is
/// written in the current [`EdgeRecord`] shape.
///
/// Layout: 8 × u32 + 1 × f64 = 40 bytes, alignment 8, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LegacyEdgeRecord {
    subject: ConceptId,
    label: LabelId,
    object: ConceptId,
    next_outgoing: EdgeId,
    next_incoming: EdgeId,
    next_labeled: EdgeId,
    first_attribution: AttributionId,
    last_attribution: AttributionId,
    weight: f64,
}

impl Record for LegacyEdgeRecord {
    const SIZE: usize = 40;

    fn store(&self, image: &mut Vec<u8>) {
        for field in [
            self.subject,
            self.label,
            self.object,
            self.next_outgoing,
            self.next_incoming,
            self.next_labeled,
            self.first_attribution,
            self.last_attribution,
        ] {
            image.extend_from_slice(&field.to_le_bytes());
        }
        image.extend_from_slice(&self.weight.to_le_bytes());
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            subject: reader.read_u32()?,
            label: reader.read_u32()?,
            object: reader.read_u32()?,
            next_outgoing: reader.read_u32()?,
            next_incoming: reader.read_u32()?,
            next_labeled: reader.read_u32()?,
            first_attribution: reader.read_u32()?,
            last_attribution: reader.read_u32()?,
            weight: reader.read_f64()?,
        })
    }
}

/// The pre-v5 shape of [`AttributionRecord`] (16 bytes: no `count`,
/// `weight` is the raw cumulative sum). See [`LegacyEdgeRecord`].
///
/// Layout: 2 × u32 + 1 × f64 = 16 bytes, alignment 8, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LegacyAttributionRecord {
    source: SourceId,
    next: AttributionId,
    weight: f64,
}

impl Record for LegacyAttributionRecord {
    const SIZE: usize = 16;

    fn store(&self, image: &mut Vec<u8>) {
        image.extend_from_slice(&self.source.to_le_bytes());
        image.extend_from_slice(&self.next.to_le_bytes());
        image.extend_from_slice(&self.weight.to_le_bytes());
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            source: reader.read_u32()?,
            next: reader.read_u32()?,
            weight: reader.read_f64()?,
        })
    }
}

impl Record for AttributionLocatorRecord {
    const SIZE: usize = 8;

    fn store(&self, image: &mut Vec<u8>) {
        image.extend_from_slice(&self.attribution.to_le_bytes());
        image.extend_from_slice(&self.paragraph.to_le_bytes());
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            attribution: reader.read_u32()?,
            paragraph: reader.read_u32()?,
        })
    }
}

/// Cursor over an image's bytes; every read is bounds-checked so a
/// truncated or hostile image fails with [`CorruptImage`], never a panic.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], CorruptImage> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(CorruptImage("section length overflows"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(CorruptImage("image ends mid-section"))?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32, CorruptImage> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, CorruptImage> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, CorruptImage> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Attribution;
    use crate::context::test_support::{associate_examples, weight_between};
    use crate::deadline::Deadline;

    /// Recomputes the checksum footer after a test tampers with an
    /// image's body. The structural validators those tests pin sit
    /// BEHIND the checksum — without resealing, every mutation would
    /// stop at "checksum mismatch" and the validators would go
    /// unexercised (they still matter: pre-v6 images skip the checksum,
    /// and a hand-built image can carry any footer it likes).
    fn resealed(mut image: Vec<u8>) -> Vec<u8> {
        let body = image.len() - 4;
        let crc = crate::crc32c::crc32c(&image[..body]);
        image[body..].copy_from_slice(&crc.to_le_bytes());
        image
    }

    #[test]
    fn extreme_weight_sums_saturate_and_the_image_still_round_trips() {
        // Two individually-finite weights can sum past f64's range. The
        // sum must saturate rather than reach infinity: a non-finite sum
        // would make `from_bytes` refuse the very image `to_bytes` just
        // produced — a context that can be saved but never loaded again.
        let mut context = Context::default();
        context.associate("a", "r", "b", f64::MAX).unwrap();
        context.associate("a", "r", "b", f64::MAX).unwrap();
        // The sourced path accumulates a second sum in the attribution
        // record; push it past the range too.
        context
            .associate_from("a", "r", "c", f64::MAX, "源", None)
            .unwrap();
        context
            .associate_from("a", "r", "c", f64::MAX, "源", None)
            .unwrap();
        // And the negative direction saturates symmetrically.
        context.associate("a", "r", "d", -f64::MAX).unwrap();
        context.associate("a", "r", "d", -f64::MAX).unwrap();

        assert!(weight_between(&context, "a", "r", "b").is_finite());
        assert!(weight_between(&context, "a", "r", "c").is_finite());
        assert!(weight_between(&context, "a", "r", "d") < 0.0);

        let restored = Context::from_bytes(&context.to_bytes())
            .expect("a context built from finite weights must round-trip");
        assert_eq!(
            weight_between(&restored, "a", "r", "b"),
            weight_between(&context, "a", "r", "b"),
        );
    }

    #[test]
    fn aliases_survive_the_image_roundtrip_and_v1_images_still_load() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "創業年", "1907年", 1.0)
            .unwrap();
        context.add_concept_alias("Aomine", "青嶺酒造").unwrap();
        context.add_label_alias("設立年", "創業年").unwrap();

        let restored = Context::from_bytes(&context.to_bytes()).expect("v2 image must load");
        assert_eq!(restored.concept_aliases(), context.concept_aliases());
        assert_eq!(restored.label_aliases(), context.label_aliases());
        assert_eq!(
            restored.query(Some("Aomine"), Some("設立年"), None),
            context.query(Some("Aomine"), Some("設立年"), None)
        );

        // A version-1 image predates the watermark, both alias tables,
        // and the attribution-locator table; it must still load with no
        // aliases. `to_bytes_as_version` writes the genuine pre-v5,
        // weight-only edge/attribution shape a v1 reader expects —
        // slicing `to_bytes`'s (always-current) output no longer works
        // now that those records are wider than they were pre-v5.
        let mut aliasless = Context::default();
        aliasless.associate("私", "好き", "りんご", 1.0).unwrap();
        let v1 = aliasless.to_bytes_as_version(1);
        let loaded = Context::from_bytes(&v1).expect("v1 image must still load");
        assert_eq!(loaded.recall("私").len(), 1);
        assert!(loaded.concept_aliases().is_empty());

        // An alias record pointing at a nonexistent concept is caught.
        // Section math for this (current-version) context (1 unsourced
        // edge, no attributions, 1 concept alias, no label aliases, no
        // locators): header 24, edges 8+48, attributions 8, concepts
        // 8+64, labels 8+20, sources 8 → the concept-alias count sits at
        // 196..204, its one 12-byte record (name_offset, name_len,
        // target) follows at 204..216, so target is 212..216.
        let mut with_alias = Context::default();
        with_alias.associate("私", "好き", "りんご", 1.0).unwrap();
        with_alias.add_concept_alias("わたし", "私").unwrap();
        let mut corrupt = with_alias.to_bytes();
        corrupt[212..216].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Context::from_bytes(&resealed(corrupt)).is_err());
    }

    #[test]
    fn applied_seq_round_trips_through_the_image() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        assert_eq!(context.applied_seq(), 0, "fresh contexts start at 0");
        context.set_applied_seq(42);

        let restored = Context::from_bytes(&context.to_bytes()).unwrap();
        assert_eq!(restored.applied_seq(), 42);
        assert_eq!(restored.recall("私").len(), 1);
    }

    #[test]
    fn v2_images_load_with_a_zero_watermark() {
        // A v2 image predates the watermark and the attribution-locator
        // table — this pins BOTH the backward-compat read and the
        // version RANGE check (a two-value check like `!=1 && !=current`
        // would reject exactly this). `to_bytes_as_version` also proves
        // the watermark never round-trips into a version that predates
        // it: `applied_seq` is set here but must load back as 0.
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.add_concept_alias("わたし", "私").unwrap();
        context.set_applied_seq(7); // must NOT survive into the v2 bytes
        let v2 = context.to_bytes_as_version(2);

        let loaded = Context::from_bytes(&v2).expect("v2 image must load");
        assert_eq!(loaded.applied_seq(), 0);
        assert_eq!(loaded.concept_aliases(), vec![("わたし", "私")]);
    }

    #[test]
    fn v3_images_load_with_no_locators() {
        // A v3 image already has the watermark and both alias tables —
        // it just predates the attribution-locator table, so every
        // attribution's paragraph must resolve to `None` regardless of
        // what locator this context recorded.
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        let v3 = context.to_bytes_as_version(3);

        let loaded = Context::from_bytes(&v3).expect("v3 image must load");
        assert_eq!(
            loaded.recall("私")[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.0,
                count: 1,
                paragraph: None,
            }]
        );
    }

    #[test]
    fn v4_images_synthesize_count_from_the_attribution_chain() {
        // A v4 image predates the count/sum split entirely — its edge
        // and attribution records carry only a flat cumulative
        // `weight`. Two independent sources each asserting once must
        // synthesize `count` as the attribution chain length (2), not
        // as a flat 1, so the migrated weight is the corroborated
        // average (1.5) rather than the old flat sum (3.0).
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        context
            .associate_from("私", "好き", "りんご", 2.0, "文書2", None)
            .unwrap();
        let v4 = context.to_bytes_as_version(4);

        let loaded = Context::from_bytes(&v4).expect("v4 image must load");
        let assoc = &loaded.recall("私")[0];
        assert_eq!(assoc.count, 2);
        assert_eq!(assoc.weight, 1.5);
    }

    #[test]
    fn migrating_a_pre_v5_image_does_not_revive_a_fully_retracted_edge() {
        // A pre-v5 image predates `count`; migration synthesizes it from
        // the attribution chain length, floored at 1 for edges that were
        // always sourceless (first_attribution == NIL but weight != 0).
        // An edge that instead died via retract_source ends up with the
        // very same on-disk shape: first_attribution == NIL. Migration
        // must tell the two apart by weight — retraction always zeroes
        // it — so the synthesized `count` must come back 0 (dead), not
        // 1 (revived).
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "旧版", None)
            .unwrap();
        assert_eq!(context.retract_source("旧版"), Some(1));
        assert_eq!(context.dead_edges(), 1);

        let v4 = context.to_bytes_as_version(4);
        let loaded = Context::from_bytes(&v4).expect("v4 image must load");

        assert_eq!(
            loaded.dead_edges(),
            1,
            "a fully retracted edge must not come back alive after migrating a pre-v5 image"
        );
        assert!(
            loaded.query(Some("a"), None, Some("b"))[0]
                .attributions
                .is_empty()
        );
    }

    #[test]
    fn migrating_a_pre_v5_image_cannot_tell_a_sourceless_zero_weight_edge_from_a_retracted_one() {
        // The flip side of the test above, and a known limitation
        // rather than a bug: `associate` (no source) never rejects
        // weight 0.0 — only NaN/±inf are refused — so a live,
        // always-sourceless edge can legitimately sit at
        // first_attribution == NIL, weight == 0.0. On disk that is
        // bit-for-bit the same shape a fully retracted edge settles
        // into, and a pre-v5 attribution record carries no back-
        // pointer to its edge, so no amount of scanning the legacy
        // image recovers which case this was — migration cannot tell
        // them apart from the bytes alone. Reviving every empty-chain
        // edge would undo the fix above, so migration accepts this
        // false negative in exchange for never reviving a retraction:
        // retractions happen constantly in normal operation, a live
        // edge asserted at weight 0.0 essentially never does.
        let mut context = Context::default();
        context.associate("a", "r", "b", 0.0).unwrap();
        assert_eq!(context.dead_edges(), 0);

        let v4 = context.to_bytes_as_version(4);
        let loaded = Context::from_bytes(&v4).expect("v4 image must load");

        assert_eq!(
            loaded.dead_edges(),
            1,
            "known limitation: a sourceless zero-weight edge is indistinguishable on \
             disk from a fully retracted one, and migration resolves the ambiguity \
             toward the far more common case (retraction) — if this starts failing, \
             the ambiguity became resolvable and this test's premise should be revisited"
        );
    }

    /// The `(edge, source)` index the write path consults is derived, so
    /// it must be rebuilt from the chains on load — and rebuilt to match
    /// what retraction left behind: a source's unlinked record is dead
    /// space and must NOT be indexed, while every live record must be. A
    /// write after the reload proves both directions at once.
    #[test]
    fn the_attribution_index_is_rebuilt_correctly_after_a_reload() {
        let mut context = Context::default();
        context
            .associate_from("x", "r", "y", 2.0, "A", None)
            .unwrap();
        context
            .associate_from("x", "r", "y", 4.0, "B", None)
            .unwrap();
        // Retract A: its record leaves the chain as dead space. The edge
        // keeps only B's 4.0 (count 1).
        assert_eq!(context.retract_source("A"), Some(1));
        assert_eq!(weight_between(&context, "x", "r", "y"), 4.0);

        // Round-trip: the reloaded context rebuilds the index from the
        // live chain alone — B in, A's dead record out.
        let mut restored = Context::from_bytes(&context.to_bytes()).expect("image must load");

        // Re-assert B: the rebuilt index finds its live record, so this
        // folds in (count 2) rather than appending a duplicate.
        restored
            .associate_from("x", "r", "y", 2.0, "B", None)
            .unwrap();
        // Re-assert A: A is absent from the rebuilt index (its old record
        // is dead), so this appends a FRESH record (count 1) instead of
        // resurrecting the retracted 2.0.
        restored
            .associate_from("x", "r", "y", 6.0, "A", None)
            .unwrap();

        // Edge: B's 4.0+2.0 plus A's fresh 6.0 = sum 12.0 over count 3 —
        // an average of 4.0. A resurrected A record would have folded into
        // dead space and left the edge at 3.0; a duplicated B would show B
        // twice below.
        assert_eq!(weight_between(&restored, "x", "r", "y"), 4.0);
        assert_eq!(
            restored.query(Some("x"), None, Some("y"))[0].attributions,
            vec![
                Attribution {
                    source: "B".to_string(),
                    weight: 6.0,
                    count: 2,
                    paragraph: None,
                },
                Attribution {
                    source: "A".to_string(),
                    weight: 6.0,
                    count: 1,
                    paragraph: None,
                },
            ]
        );
    }

    #[test]
    fn migrating_a_pre_v5_image_credits_an_undetected_sourceless_call_so_retraction_cannot_revive_it_as_dead()
     {
        // `upsert` folds every assertion — sourced or not — into the
        // edge's cumulative weight, but only a sourced one ever links
        // into the attribution chain (see `Context::associate` vs
        // `associate_from`). A pre-v5 image that mixes both on the same
        // edge therefore has a chain shorter than the edge's true call
        // count: migration must notice the gap between the edge's total
        // weight and what the chain accounts for, or the sourceless
        // call vanishes from `count` entirely and retracting the last
        // known source wrongly declares the edge dead.
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context
            .associate_from("私", "好き", "りんご", 2.0, "文書1", None)
            .unwrap();
        let native = &context.recall("私")[0];
        assert_eq!(native.count, 2);
        assert_eq!(native.weight, 1.5);

        let v4 = context.to_bytes_as_version(4);
        let mut restored = Context::from_bytes(&v4).expect("v4 image must load");

        assert_eq!(restored.retract_source("文書1"), Some(1));

        assert_eq!(
            restored.dead_edges(),
            0,
            "a sourceless contribution must keep the edge alive after its only \
             source retracts"
        );
        let after = &restored.recall("私")[0];
        assert_eq!(after.count, 1);
        assert_eq!(after.weight, 1.0);
        assert!(after.attributions.is_empty());
    }

    #[test]
    fn image_roundtrip_preserves_every_read_path() {
        let mut context = Context::default();
        associate_examples(&mut context);
        context.associate("りんご", "分類", "果物", 1.5).unwrap();
        context
            .associate_from("決定", "手段", "投票", 1.0, "IPA公式", None)
            .unwrap();
        context
            .associate_from("決定", "手段", "投票", 0.5, "解説記事", None)
            .unwrap();
        context.associate("犬", "好き", "骨", 1.0).unwrap(); // separate component

        let restored = Context::from_bytes(&context.to_bytes()).expect("image must load");

        assert_eq!(restored.recall("私"), context.recall("私"));
        assert_eq!(restored.recall("投票"), context.recall("投票"));
        assert_eq!(
            restored.query(None, None, None),
            context.query(None, None, None)
        );
        assert_eq!(
            restored.query(Some("私"), Some("好き"), None),
            context.query(Some("私"), Some("好き"), None)
        );
        assert_eq!(
            restored.explore(&["私"], Context::UNBOUNDED),
            context.explore(&["私"], Context::UNBOUNDED)
        );
        assert_eq!(
            restored.activate(&["私"], 0.5, 10),
            context.activate(&["私"], 0.5, 10)
        );
        assert_eq!(restored.resolve("りんご"), context.resolve("りんご"));
        assert_eq!(restored.labels(), context.labels());
        assert_eq!(
            restored
                .unreachable_from(&["私"], Deadline::unbounded())
                .unwrap(),
            context
                .unreachable_from(&["私"], Deadline::unbounded())
                .unwrap()
        );
    }

    #[test]
    fn image_roundtrip_keeps_accepting_writes() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "文書1", None)
            .unwrap();

        let mut restored = Context::from_bytes(&context.to_bytes()).expect("image must load");

        // Accumulation must land on the restored edge and its restored
        // attribution — the rebuilt indexes and chain tails must all point
        // at the right records.
        restored
            .associate_from("a", "r", "b", 0.5, "文書1", None)
            .unwrap();
        assert_eq!(weight_between(&restored, "a", "r", "b"), 0.75);
        assert_eq!(
            restored.recall("a")[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.5,
                count: 2,
                paragraph: None,
            }]
        );

        // And chains must keep extending in insertion order.
        restored.associate("a", "r", "c", 1.0).unwrap();
        restored.associate("d", "r2", "a", 1.0).unwrap();
        assert_eq!(restored.recall("a").len(), 3);
        assert_eq!(restored.labels(), vec!["r", "r2"]);
    }

    #[test]
    fn attribution_locators_survive_the_image_roundtrip() {
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", Some(3))
            .unwrap();
        // A second, unlocated source on the same edge must round-trip
        // as `None` alongside the first source's `Some`.
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書2", None)
            .unwrap();

        let restored = Context::from_bytes(&context.to_bytes()).expect("image must load");
        assert_eq!(restored.recall("私"), context.recall("私"));
        assert_eq!(
            restored.recall("私")[0].attributions,
            vec![
                Attribution {
                    source: "文書1".to_string(),
                    weight: 1.0,
                    count: 1,
                    paragraph: Some(3),
                },
                Attribution {
                    source: "文書2".to_string(),
                    weight: 1.0,
                    count: 1,
                    paragraph: None,
                },
            ]
        );
    }

    #[test]
    fn from_bytes_rejects_corrupt_locators() {
        // One sourced, located attribution: header 24, edges 8+48,
        // attributions 8+24, concepts 8+64, labels 8+20, sources 8+8,
        // concept_aliases 8, label_aliases 8 → the locator table starts
        // at 244 (its count, 8 bytes), and the lone record's
        // `attribution` field sits at 252..256. Pointing it at a
        // nonexistent attribution must be caught.
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", Some(3))
            .unwrap();
        let mut dangling = context.to_bytes();
        dangling[252..256].copy_from_slice(&u32::MAX.to_le_bytes());
        let error = Context::from_bytes(&resealed(dangling)).unwrap_err();
        assert!(error.to_string().contains("unknown attribution"), "{error}");

        // Two sourced, located attributions from two distinct sources on
        // the same edge (two attribution records, two locator records):
        // the extra 8-byte source and 24-byte attribution record shift
        // the locator table to 276..300, putting the second record's
        // `attribution` field at 292..296. Setting it to 0 — equal to
        // the first record's — breaks the strictly-increasing invariant
        // without pointing outside the attribution table.
        let mut two_sources = Context::default();
        two_sources
            .associate_from("私", "好き", "りんご", 1.0, "文書1", Some(3))
            .unwrap();
        two_sources
            .associate_from("私", "好き", "りんご", 1.0, "文書2", Some(5))
            .unwrap();
        let mut unsorted = two_sources.to_bytes();
        unsorted[292..296].copy_from_slice(&0u32.to_le_bytes());
        let error = Context::from_bytes(&resealed(unsorted)).unwrap_err();
        assert!(error.to_string().contains("not sorted"), "{error}");
    }

    #[test]
    fn from_bytes_rejects_an_arena_length_past_the_u32_offset_space() {
        // An empty Context: header 24, eight zero-count tables (8 bytes
        // each) → the arena-length u64 sits at 88..96, followed by zero
        // arena bytes and the checksum footer. name_offset/name_len
        // record fields are u32 (intern_name asserts this same bound on
        // the write side), so a declared length past that space must be
        // caught here rather than surfacing as a panic the first time
        // some later write tries to intern a name.
        let context = Context::default();
        let mut oversized = context.to_bytes();
        oversized[88..96].copy_from_slice(&(u32::MAX as u64 + 1).to_le_bytes());
        let error = Context::from_bytes(&resealed(oversized)).unwrap_err();
        assert!(error.to_string().contains("4 GiB"), "{error}");
    }

    #[test]
    fn from_bytes_rejects_an_edge_count_below_its_attribution_chain() {
        // header 24, edge-table count 8 → the lone edge record starts
        // at 32; `count` is its ninth field, after 8 × u32 (32 bytes),
        // so it sits at 32+32=64..72 as a u64. Setting it below the
        // one attribution record's own count (1) must be caught, or a
        // hand-crafted or corrupted image could desynchronize the
        // derived `weight` from the attributions actually backing it.
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        let mut corrupt = context.to_bytes();
        corrupt[64..72].copy_from_slice(&0u64.to_le_bytes());
        let error = Context::from_bytes(&resealed(corrupt)).unwrap_err();
        assert!(error.to_string().contains("combined count"), "{error}");
    }

    #[test]
    fn from_bytes_rejects_non_finite_weights_and_count_overflow() {
        // Same single-edge layout as the test above: the edge's `sum`
        // (its tenth field, an f64) follows `count` at 72..80. A tampered
        // non-finite sum sorts as the maximum under `total_cmp` and would
        // permanently occupy every ranked result, so it must be rejected.
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        let mut nan_edge = context.to_bytes();
        nan_edge[72..80].copy_from_slice(&f64::NAN.to_le_bytes());
        let error = Context::from_bytes(&resealed(nan_edge)).unwrap_err();
        assert!(
            error.to_string().contains("edge weight sum is not finite"),
            "{error}"
        );

        // The lone attribution record follows the edge table (table count
        // at 80..88, record 0 at 88..112); its `sum` — after source u32,
        // next u32, count u64 — sits at 104..112.
        let mut inf_record = context.to_bytes();
        inf_record[104..112].copy_from_slice(&f64::INFINITY.to_le_bytes());
        let error = Context::from_bytes(&resealed(inf_record)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("attribution weight sum is not finite"),
            "{error}"
        );

        // Two sources on one edge give two attribution records (record 0
        // at 88..112, record 1 at 112..136); their `count` u64s sit at
        // 96..104 and 120..128. Two maxed counts overflow the running
        // chain total, which must fail rather than wrap past the floor the
        // `combined count` check downstream relies on.
        let mut two_sources = Context::default();
        two_sources
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        two_sources
            .associate_from("私", "好き", "りんご", 1.0, "文書2", None)
            .unwrap();
        let mut overflow = two_sources.to_bytes();
        overflow[96..104].copy_from_slice(&u64::MAX.to_le_bytes());
        overflow[120..128].copy_from_slice(&u64::MAX.to_le_bytes());
        let error = Context::from_bytes(&resealed(overflow)).unwrap_err();
        assert!(error.to_string().contains("overflows u64"), "{error}");
    }

    #[test]
    fn from_bytes_rejects_one_edge_attributing_a_source_twice() {
        // The write path keeps one attribution record per source per edge,
        // so the derived (edge, source) index built during load can assume
        // it. A tampered chain that links one source twice would collapse
        // silently in that index; catch it instead. Same two-source layout
        // as the overflow test: record 1's `source` u32 sits at 112..116;
        // set it to record 0's source (0) so both claim the same one.
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書2", None)
            .unwrap();
        let mut duplicate = context.to_bytes();
        duplicate[112..116].copy_from_slice(&0u32.to_le_bytes());
        let error = Context::from_bytes(&resealed(duplicate)).unwrap_err();
        assert!(
            error.to_string().contains("attributes a source twice"),
            "{error}"
        );
    }

    #[test]
    fn from_bytes_rejects_a_zero_count_attribution_record() {
        // Same single-edge, single-attribution layout as the tests above:
        // record 0 sits at 88..112, its `count` u64 at 96..104 (after the
        // source and next u32s). The write path unlinks a record the
        // instant retraction drains it to zero, so a live chained record
        // always carries a positive count. A zero here is a crafted or
        // corrupted image: it must be refused, or the `edge.count == 0`
        // dead-edge shortcut — which assumes a dead edge's chain is empty —
        // would over-count a chain that still threads a zero-count record.
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        let mut corrupt = context.to_bytes();
        corrupt[96..104].copy_from_slice(&0u64.to_le_bytes());
        let error = Context::from_bytes(&resealed(corrupt)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("attribution record carries a zero count"),
            "{error}"
        );
    }

    #[test]
    fn empty_context_roundtrips() {
        let restored =
            Context::from_bytes(&Context::default().to_bytes()).expect("image must load");
        assert!(restored.query(None, None, None).is_empty());
    }

    #[test]
    fn from_bytes_rejects_malformed_images() {
        assert!(Context::from_bytes(b"").is_err());
        assert!(Context::from_bytes(b"not an image at all").is_err());

        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        let image = context.to_bytes();

        // Truncation anywhere must be caught by section bounds.
        for len in [image.len() - 1, image.len() / 2, 9] {
            assert!(Context::from_bytes(&image[..len]).is_err());
        }
        // Trailing garbage is not silently ignored.
        let mut padded = image.clone();
        padded.push(0);
        assert!(Context::from_bytes(&padded).is_err());
        // A wrong magic or version is refused outright.
        let mut wrong_magic = image.clone();
        wrong_magic[0] ^= 0xFF;
        assert!(Context::from_bytes(&wrong_magic).is_err());
        let mut wrong_version = image.clone();
        wrong_version[8] = 0xFF;
        assert!(Context::from_bytes(&wrong_version).is_err());
    }

    #[test]
    fn the_checksum_catches_silent_corruption_and_legacy_images_skip_it() {
        let mut context = Context::default();
        context.associate("i", "likes", "apple", 1.0).unwrap();
        let image = context.to_bytes();
        assert_eq!(Context::image_generation(&image), Some((6, true)));

        // Flip the arena's last byte: one character of a stored name.
        // The image stays structurally perfect — ids in range, chains
        // intact, UTF-8 valid — it just says something else. This is
        // the silent-bit-rot shape, and the reseal below proves the
        // structural validators genuinely cannot see it: only the
        // checksum stands between it and being served (and flushed
        // back) as truth.
        let mut flipped = image.clone();
        let last_arena_byte = flipped.len() - 5; // 4-byte footer after it
        flipped[last_arena_byte] ^= 0x01; // the name's final letter shifts
        let error = Context::from_bytes(&flipped).unwrap_err();
        assert!(error.to_string().contains("checksum mismatch"), "{error}");
        let laundered = Context::from_bytes(&resealed(flipped))
            .expect("structural validation alone accepts the corrupted name");
        let assoc = &laundered.recall("i")[0];
        assert_ne!(
            (assoc.label.as_str(), assoc.object.as_str()),
            ("likes", "apple"),
            "the flipped byte changed what the image says, and it loaded anyway"
        );

        // A v5 image is the same section layout minus the footer: it
        // loads, as unverifiable as it always was.
        let v5 = context.to_bytes_as_version(5);
        assert_eq!(Context::image_generation(&v5), Some((5, false)));
        let loaded = Context::from_bytes(&v5).expect("v5 image must load");
        assert_eq!(loaded.recall("i").len(), 1);

        // Bytes that never were an image have no version to report.
        assert_eq!(Context::image_generation(b"junk"), None);
    }

    #[test]
    fn from_bytes_rejects_inconsistent_records() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        let image = context.to_bytes();

        // The first edge record sits right after the 16-byte header and
        // the edge table's u64 count; its first field is `subject`.
        // Pointing it at a nonexistent concept must be caught.
        let mut dangling_subject = image.clone();
        dangling_subject[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Context::from_bytes(&resealed(dangling_subject)).is_err());

        // 私's `outgoing_count` sits at offset 96: header 16, edge table
        // 8 + 40, empty attribution table 8, concept count 8, then the
        // fifth u32 field of the first concept record. A count that
        // disagrees with the actual chain must be caught.
        let mut wrong_count = image.clone();
        wrong_count[96..100].copy_from_slice(&5u32.to_le_bytes());
        assert!(Context::from_bytes(&resealed(wrong_count)).is_err());
    }
}
