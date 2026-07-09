use std::collections::HashMap;

use super::{
    AliasRecord, AttributionId, AttributionLocatorRecord, AttributionRecord, ConceptRecord,
    Context, CorruptImage, EdgeId, EdgeRecord, EntryIndex, LabelRecord, NIL, SourceRecord,
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
/// attribution-locator table between the label aliases and the arena.
/// Older images still load (empty alias/locator tables, watermark 0);
/// writing always produces the current version.
const IMAGE_VERSION: u32 = 4;
/// Magic + version + 4 bytes of padding (so what follows is 8-byte
/// aligned) + the u64 durability watermark.
const IMAGE_HEADER_SIZE: usize = 24;
/// Eight record tables plus the arena, each section prefixed by a u64
/// length.
const IMAGE_SECTIONS: usize = 9;

impl Context {
    /// Serializes the whole network into one contiguous byte image.
    ///
    /// The image is the storage buffers written back to back: a 16-byte
    /// header, then each record table as a u64 count followed by its
    /// fixed-width records field by field in little-endian, then the
    /// string arena. Sections are ordered by descending alignment (the
    /// f64-bearing tables first), so every record sits naturally aligned
    /// within the image as well — the layout stays open to zero-copy
    /// mapping later. The derived hash indexes are not written;
    /// [`Context::from_bytes`] rebuilds them.
    pub fn to_bytes(&self) -> Vec<u8> {
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
                + self.arena.len(),
        );
        image.extend_from_slice(&IMAGE_MAGIC);
        image.extend_from_slice(&IMAGE_VERSION.to_le_bytes());
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
        reader.take(4)?; // header padding
        // Version 3 adds the durability watermark; older images carry
        // none, and 0 is exactly right for them — no WAL can predate
        // the feature, so "replay everything above 0" is a no-op or
        // correct.
        let applied_seq = if version >= 3 { reader.read_u64()? } else { 0 };

        let edges = load_table::<EdgeRecord>(&mut reader)?;
        let attributions = load_table::<AttributionRecord>(&mut reader)?;
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
        let arena = reader.take(arena_len)?.to_vec();
        if reader.pos != image.len() {
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
            edge_ids: HashMap::new(),
            concept_index: EntryIndex::default(),
            label_index: EntryIndex::default(),
            dice_floor: None,
            applied_seq,
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
        self.validate_attributions()?;
        self.validate_locators()
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
    fn validate_attributions(&self) -> Result<(), CorruptImage> {
        let mut claimed = vec![false; self.attributions.len()];
        for edge in &self.edges {
            let mut cursor = edge.first_attribution;
            let mut tail = NIL;
            while cursor != NIL {
                let record = self
                    .attributions
                    .get(cursor as usize)
                    .ok_or(CorruptImage("attribution link is out of range"))?;
                if std::mem::replace(&mut claimed[cursor as usize], true) {
                    return Err(CorruptImage("attribution record belongs to two chains"));
                }
                if record.source as usize >= self.sources.len() {
                    return Err(CorruptImage("attribution references an unknown source"));
                }
                tail = cursor;
                cursor = record.next;
            }
            if tail != edge.last_attribution {
                return Err(CorruptImage("attribution chain does not end at its tail"));
            }
        }
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
    if count >= u64::from(NIL) {
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

impl Record for AttributionRecord {
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
