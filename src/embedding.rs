//! Optional semantic entry tier: embeddings over concept and label
//! GLOSSES — each name plus its heaviest graph facts, because a lone
//! word carries too little signal for sentence-trained embedding
//! models, and the graph already owns the context. Retrieval itself
//! stays structural; only the entry needs meaning. Configured via `TAGURU_EMBED_URL` /
//! `TAGURU_EMBED_MODEL` / `TAGURU_EMBED_API_KEY` against any
//! OpenAI-compatible `/embeddings` endpoint; absent config disables the
//! tier and resolve stays purely lexical.
//!
//! Vectors are a derived cache of (model × name), kept per context in a
//! `{name}.vectors.bin` sidecar — refreshed explicitly (POST
//! /contexts/{name}/embeddings/refresh), loaded on demand by the
//! semantic fallback, and discarded wholesale when the model changes.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::time::Duration;

/// Why a batch is being embedded: glosses entering the store (`Index`)
/// or a live cue looking things up (`Query`). Modern embedding APIs
/// (Cohere, Voyage) encode documents and queries asymmetrically; the
/// OpenAI request shape cannot carry the distinction, so the HTTP
/// provider forwards it as an `X-Taguru-Embed-Purpose` header for
/// bridging proxies to map (e.g. onto Cohere's `input_type`). Plain
/// OpenAI-compatible servers ignore the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedPurpose {
    Index,
    Query,
}

impl EmbedPurpose {
    pub fn as_str(self) -> &'static str {
        match self {
            EmbedPurpose::Index => "index",
            EmbedPurpose::Query => "query",
        }
    }
}

/// Anything that can turn short strings into vectors. The HTTP provider
/// is the real one; tests inject a deterministic mock.
pub trait EmbeddingProvider: Send + Sync {
    fn model(&self) -> &str;
    /// Returns one vector per input text, all the same dimension.
    fn embed(&self, texts: &[&str], purpose: EmbedPurpose) -> Result<Vec<Vec<f32>>, String>;
}

/// OpenAI-compatible `/embeddings` client: `{model, input: [...]}` in,
/// `{data: [{embedding: [...]}]}` out.
pub struct HttpEmbeddings {
    url: String,
    model: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl HttpEmbeddings {
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("TAGURU_EMBED_URL").ok()?;
        let model = std::env::var("TAGURU_EMBED_MODEL").ok()?;
        Some(Self {
            url,
            model,
            api_key: std::env::var("TAGURU_EMBED_API_KEY").ok(),
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(60))
                .build(),
        })
    }
}

impl EmbeddingProvider for HttpEmbeddings {
    fn model(&self) -> &str {
        &self.model
    }

    fn embed(&self, texts: &[&str], purpose: EmbedPurpose) -> Result<Vec<Vec<f32>>, String> {
        // A client span per provider round trip — the one downstream
        // call whose latency Taguru's own timings cannot explain. Runs
        // on the caller's thread (block_in_place), so a request span
        // in scope becomes its parent automatically.
        let span = tracing::info_span!(
            "embed",
            otel.kind = "client",
            embed.model = %self.model,
            embed.inputs = texts.len(),
            embed.purpose = purpose.as_str(),
        );
        let _guard = span.enter();
        let mut request = self
            .agent
            .post(&self.url)
            .set("Content-Type", "application/json")
            .set("X-Taguru-Embed-Purpose", purpose.as_str());
        if let Some(key) = &self.api_key {
            request = request.set("Authorization", &format!("Bearer {key}"));
        }
        let body = serde_json::json!({ "model": self.model, "input": texts });
        let response = request
            .send_string(&body.to_string())
            .map_err(|error| format!("embedding request failed: {error}"))?;
        let parsed: serde_json::Value = response
            .into_json()
            .map_err(|error| format!("embedding response unreadable: {error}"))?;
        let data = parsed
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or("embedding response carries no data array")?;
        if data.len() != texts.len() {
            return Err(format!(
                "embedding response returned {} vectors for {} inputs",
                data.len(),
                texts.len()
            ));
        }
        // A provider or proxy may return `data` out of input order; each
        // entry's `index` names the input it belongs to (the OpenAI
        // embedding contract). Place each vector at its index rather than
        // trust array order, so a reordered response cannot silently pair
        // every embedding with the wrong text. An entry that omits `index`
        // falls back to its array position — the old, order-trusting
        // behavior, so a provider that never sends `index` still works.
        let mut slots: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        for (position, item) in data.iter().enumerate() {
            let index = item
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .unwrap_or(position);
            if index >= slots.len() {
                return Err(format!(
                    "embedding response index {index} out of range for {} inputs",
                    texts.len()
                ));
            }
            let embedding = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or("embedding entry carries no vector")?;
            let mut vector: Vec<f32> = embedding
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            normalize(&mut vector);
            if slots[index].replace(vector).is_some() {
                return Err(format!("embedding response repeated index {index}"));
            }
        }
        // No slot can remain empty here — the count matched and no index
        // repeated — but guard rather than unwrap so a response that both
        // omits `index` and collides two entries onto one position fails
        // cleanly instead of panicking.
        slots
            .into_iter()
            .enumerate()
            .map(|(index, slot)| {
                slot.ok_or_else(|| format!("embedding response missing index {index}"))
            })
            .collect()
    }
}

/// FNV-1a: a tiny content hash for gloss-change detection. Stability
/// across builds matters here (std's DefaultHasher promises none) — a
/// changed hash function would silently re-embed every name.
pub fn fnv1a(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Scales a vector to unit length, so similarity is a plain dot product.
pub fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

/// Cosine similarity of two unit vectors (a dot product).
pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// name → (gloss hash, unit vector). The hash is of the gloss the
/// vector was computed from, so a refresh re-embeds exactly the names
/// whose graph context changed.
pub type VectorTable = HashMap<String, (u64, Vec<f32>)>;

/// One context's name vectors: a derived cache keyed by the model that
/// produced it. Wholesale discarded on model change — vectors from two
/// models must never be compared.
#[derive(Debug, Default)]
pub struct VectorStore {
    pub model: String,
    pub concepts: VectorTable,
    pub labels: VectorTable,
}

const VECTOR_MAGIC: &[u8; 8] = b"TAGURUV2";

impl VectorStore {
    /// Reads a sidecar, returning an empty store on any problem — a
    /// corrupt vector cache costs a re-embed, never an outage.
    pub fn load(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => Self::from_bytes(&bytes).unwrap_or_else(|| {
                tracing::warn!("ignoring corrupt vector store at {}", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        crate::registry::write_atomic(path, &self.to_bytes())
    }

    /// Rough resident bytes when this store is held in memory, so the
    /// cache budget can account for it.
    pub fn footprint(&self) -> usize {
        const ENTRY_OVERHEAD: usize = 64;
        let table = |table: &VectorTable| -> usize {
            table
                .iter()
                .map(|(name, (_, vector))| name.len() + vector.len() * 4 + ENTRY_OVERHEAD)
                .sum()
        };
        table(&self.concepts) + table(&self.labels)
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(VECTOR_MAGIC);
        write_string(&mut out, &self.model);
        write_table(&mut out, &self.concepts);
        write_table(&mut out, &self.labels);
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut pos = 0usize;
        if bytes.get(..8)? != VECTOR_MAGIC {
            return None;
        }
        pos += 8;
        let model = read_string(bytes, &mut pos)?;
        let concepts = read_table(bytes, &mut pos)?;
        let labels = read_table(bytes, &mut pos)?;
        (pos == bytes.len()).then_some(Self {
            model,
            concepts,
            labels,
        })
    }
}

/// One embedded row's identity: the source id attributions carry, the
/// paragraph's position within that source's current split, and the
/// FNV-1a hash of exactly its bytes — the staleness detector, same as
/// gloss vectors. `question_hash` discriminates doc2query rows: a
/// paragraph can carry several vectors (its own text plus each stored
/// question, all pointing AT the paragraph), and without the
/// discriminator they would share one key and collapse into a single
/// carried-forward row on refresh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassageKey {
    pub source: String,
    pub index: u32,
    pub hash: u64,
    /// `None` = the paragraph text's own row; `Some(fnv1a(question))`
    /// = a question row for that paragraph.
    pub question_hash: Option<u64>,
}

const PASSAGE_VECTOR_MAGIC: &[u8; 8] = b"TAGURUP2";
/// The pre-doc2query row format. Discarded politely on load — the same
/// wholesale-discard contract as a model change: the next refresh
/// re-embeds, nothing is lost but provider spend.
const LEGACY_PASSAGE_VECTOR_MAGIC: &[u8; 8] = b"TAGURUP1";

/// Paragraph vectors for one context, `{stem}.pvectors.bin`. Unlike
/// the gloss [`VectorStore`] this is a FLAT table — one contiguous
/// `Vec<f32>` of unit rows — because every query scans all of it (a
/// nearest-neighbor sweep has no point lookups), and at 10⁴–10⁵ rows
/// sequential prefetch is what the scan lives on. Kept separate from
/// the gloss store on purpose: glosses are small and ride every
/// resolve, paragraphs are big and ride only passage search — one
/// file, one Arc, one budget entry each, so neither pays for the
/// other. Same derived-cache contract: corrupt means re-embed, never
/// an outage, and a model change discards it wholesale.
#[derive(Debug, Default)]
pub struct PassageVectorStore {
    pub model: String,
    dim: usize,
    keys: Vec<PassageKey>,
    data: Vec<f32>,
}

impl PassageVectorStore {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            ..Default::default()
        }
    }

    /// Appends one row. The first row fixes the dimension; a
    /// mismatched later row is a provider bug and is dropped loudly
    /// rather than corrupting the flat layout.
    pub fn push(&mut self, key: PassageKey, vector: Vec<f32>) {
        if self.keys.is_empty() {
            self.dim = vector.len();
        }
        if vector.len() != self.dim || self.dim == 0 {
            tracing::warn!(
                "dropping a {}-dim passage vector from a {}-dim store",
                vector.len(),
                self.dim
            );
            return;
        }
        self.keys.push(key);
        self.data.extend_from_slice(&vector);
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Every (key, unit row) pair, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&PassageKey, &[f32])> {
        self.keys
            .iter()
            .zip(self.data.chunks_exact(self.dim.max(1)))
    }

    /// Rough resident bytes when held in memory, for the cache budget.
    pub fn footprint(&self) -> usize {
        const KEY_OVERHEAD: usize = 48;
        self.data.len() * 4
            + self
                .keys
                .iter()
                .map(|key| key.source.len() + KEY_OVERHEAD)
                .sum::<usize>()
    }

    /// Top `limit` rows by cosine against a unit query (a linear sweep
    /// — the flat layout IS the index at this store's scale). Ties
    /// break by (source, index) ascending for deterministic output; a
    /// query of the wrong dimension matches nothing.
    pub fn top_matches(&self, query: &[f32], limit: usize) -> Vec<(&PassageKey, f32)> {
        if self.dim == 0 || query.len() != self.dim {
            return Vec::new();
        }
        let mut scored: Vec<(&PassageKey, f32)> = self
            .iter()
            .map(|(key, row)| (key, similarity(query, row)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.source.cmp(&b.0.source))
                .then_with(|| a.0.index.cmp(&b.0.index))
        });
        scored.truncate(limit);
        scored
    }

    /// Reads the sidecar, returning an empty store on any problem — a
    /// corrupt vector cache costs a re-embed, never an outage. A
    /// pre-doc2query file (TAGURUP1) is discarded the same way, minus
    /// the alarm: it is an upgrade, not corruption.
    pub fn load(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => Self::from_bytes(&bytes).unwrap_or_else(|| {
                if bytes.get(..8) == Some(LEGACY_PASSAGE_VECTOR_MAGIC.as_slice()) {
                    tracing::info!(
                        "passage vectors at {} predate doc2query; re-embedding",
                        path.display()
                    );
                } else {
                    tracing::warn!(
                        "ignoring corrupt passage vector store at {}",
                        path.display()
                    );
                }
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        crate::registry::write_atomic(path, &self.to_bytes())
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(PASSAGE_VECTOR_MAGIC);
        write_string(&mut out, &self.model);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.keys.len() as u32).to_le_bytes());
        for key in &self.keys {
            write_string(&mut out, &key.source);
            out.extend_from_slice(&key.index.to_le_bytes());
            out.extend_from_slice(&key.hash.to_le_bytes());
            match key.question_hash {
                Some(question_hash) => {
                    out.push(1);
                    out.extend_from_slice(&question_hash.to_le_bytes());
                }
                None => out.push(0),
            }
        }
        for value in &self.data {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut pos = 0usize;
        if bytes.get(..8)? != PASSAGE_VECTOR_MAGIC {
            return None;
        }
        pos += 8;
        let model = read_string(bytes, &mut pos)?;
        let dim = read_u32(bytes, &mut pos)? as usize;
        let count = read_u32(bytes, &mut pos)? as usize;
        // Rows without a dimension cannot exist through push(); a file
        // claiming them is corrupt, not merely empty.
        if count > 0 && dim == 0 {
            return None;
        }
        let mut keys = Vec::with_capacity(count.min(1 << 20));
        for _ in 0..count {
            let source = read_string(bytes, &mut pos)?;
            let index = read_u32(bytes, &mut pos)?;
            let hash_bytes = bytes.get(pos..pos + 8)?;
            pos += 8;
            let hash = u64::from_le_bytes(hash_bytes.try_into().ok()?);
            let question_hash = match bytes.get(pos)? {
                0 => {
                    pos += 1;
                    None
                }
                1 => {
                    pos += 1;
                    let question_bytes = bytes.get(pos..pos + 8)?;
                    pos += 8;
                    Some(u64::from_le_bytes(question_bytes.try_into().ok()?))
                }
                _ => return None,
            };
            keys.push(PassageKey {
                source,
                index,
                hash,
                question_hash,
            });
        }
        let mut data = Vec::with_capacity((count * dim).min(1 << 26));
        for _ in 0..count * dim {
            let slice = bytes.get(pos..pos + 4)?;
            pos += 4;
            data.push(f32::from_le_bytes(slice.try_into().ok()?));
        }
        (pos == bytes.len()).then_some(Self {
            model,
            dim,
            keys,
            data,
        })
    }
}

fn write_string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

fn write_table(out: &mut Vec<u8>, table: &VectorTable) {
    out.extend_from_slice(&(table.len() as u32).to_le_bytes());
    // Sorted for byte-stable files under identical content.
    let mut names: Vec<&String> = table.keys().collect();
    names.sort();
    for name in names {
        let (hash, vector) = &table[name];
        write_string(out, name);
        out.extend_from_slice(&hash.to_le_bytes());
        out.extend_from_slice(&(vector.len() as u32).to_le_bytes());
        for value in vector {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn read_string(bytes: &[u8], pos: &mut usize) -> Option<String> {
    let len = read_u32(bytes, pos)? as usize;
    let slice = bytes.get(*pos..*pos + len)?;
    *pos += len;
    String::from_utf8(slice.to_vec()).ok()
}

fn read_table(bytes: &[u8], pos: &mut usize) -> Option<VectorTable> {
    let count = read_u32(bytes, pos)? as usize;
    let mut table = HashMap::with_capacity(count.min(1 << 20));
    for _ in 0..count {
        let name = read_string(bytes, pos)?;
        let hash_bytes = bytes.get(*pos..*pos + 8)?;
        *pos += 8;
        let hash = u64::from_le_bytes(hash_bytes.try_into().ok()?);
        let dim = read_u32(bytes, pos)? as usize;
        let mut vector = Vec::with_capacity(dim.min(1 << 16));
        for _ in 0..dim {
            let slice = bytes.get(*pos..*pos + 4)?;
            *pos += 4;
            vector.push(f32::from_le_bytes(slice.try_into().ok()?));
        }
        table.insert(name, (hash, vector));
    }
    Some(table)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let slice = bytes.get(*pos..*pos + 4)?;
    *pos += 4;
    Some(u32::from_le_bytes(slice.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_store_roundtrips_and_rejects_garbage() {
        let mut store = VectorStore {
            model: "test-model".into(),
            ..Default::default()
        };
        store.concepts.insert("りんご".into(), (7, vec![1.0, 0.0]));
        store.labels.insert("好き".into(), (9, vec![0.0, 1.0]));

        let bytes = store.to_bytes();
        let loaded = VectorStore::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.model, "test-model");
        assert_eq!(loaded.concepts["りんご"], (7, vec![1.0, 0.0]));
        assert_eq!(loaded.labels["好き"], (9, vec![0.0, 1.0]));

        assert!(VectorStore::from_bytes(b"garbage").is_none());
        assert!(VectorStore::from_bytes(&bytes[..bytes.len() - 1]).is_none());
    }

    #[test]
    fn similarity_is_a_dot_product_over_unit_vectors() {
        let mut a = vec![3.0, 4.0];
        normalize(&mut a);
        assert!((similarity(&a, &a) - 1.0).abs() < 1e-6);
        let b = vec![-a[1], a[0]];
        assert!(similarity(&a, &b).abs() < 1e-6);
    }

    /// The HTTP provider must tell a bridging proxy WHY it is embedding
    /// (Cohere-style asymmetric models need `input_type`), and must
    /// normalize whatever comes back. One stub round trip checks both.
    #[test]
    fn http_embeddings_sends_the_purpose_header_and_normalizes() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let served = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 4096];
            // The request is tiny; read until the header block is in.
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
            }
            let body = br#"{"data":[{"embedding":[3.0,4.0]}]}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
            String::from_utf8_lossy(&request).to_string()
        });

        let provider = HttpEmbeddings {
            url: format!("http://{addr}"),
            model: "stub-model".to_string(),
            api_key: None,
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(5))
                .build(),
        };
        let vectors = provider
            .embed(&["こんにちは"], EmbedPurpose::Query)
            .unwrap();
        // 3-4-5 triangle, normalized on receipt.
        assert_eq!(vectors, vec![vec![0.6, 0.8]]);

        let request = served.join().unwrap().to_lowercase();
        assert!(
            request.contains("x-taguru-embed-purpose: query"),
            "{request}"
        );
    }

    /// A provider that returns `data` out of input order is realigned by
    /// each entry's `index`, never trusted in array order — otherwise
    /// every embedding would pair with the wrong text.
    #[test]
    fn http_embeddings_realigns_a_reordered_response_by_index() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let served = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 4096];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
            }
            // Wire order is the REVERSE of the input order: index 1 first.
            let body = br#"{"data":[{"index":1,"embedding":[0.0,2.0]},{"index":0,"embedding":[3.0,4.0]}]}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });

        let provider = HttpEmbeddings {
            url: format!("http://{addr}"),
            model: "stub-model".to_string(),
            api_key: None,
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(5))
                .build(),
        };
        let vectors = provider
            .embed(&["first", "second"], EmbedPurpose::Query)
            .unwrap();
        served.join().unwrap();
        // Input 0 keeps index 0's vector (3-4-5 → 0.6, 0.8); input 1 keeps
        // index 1's (0, 2 → 0, 1). Trusting array order would swap them.
        assert_eq!(vectors, vec![vec![0.6, 0.8], vec![0.0, 1.0]]);
    }

    #[test]
    fn passage_vector_store_round_trips_and_rejects_garbage() {
        let mut store = PassageVectorStore::new("test-model");
        store.push(
            PassageKey {
                source: "docs/aomine.md".into(),
                index: 0,
                hash: 7,
                question_hash: None,
            },
            vec![1.0, 0.0],
        );
        // A doc2query question row: SAME paragraph key, its own
        // discriminator — two rows for one paragraph must both survive
        // the round trip.
        store.push(
            PassageKey {
                source: "docs/aomine.md".into(),
                index: 0,
                hash: 7,
                question_hash: Some(41),
            },
            vec![0.6, 0.8],
        );
        store.push(
            PassageKey {
                source: "docs/aomine.md".into(),
                index: 1,
                hash: 9,
                question_hash: None,
            },
            vec![0.0, 1.0],
        );
        // A wrong-dimension row is dropped, not silently misaligned.
        store.push(
            PassageKey {
                source: "bad".into(),
                index: 0,
                hash: 1,
                question_hash: None,
            },
            vec![1.0, 0.0, 0.0],
        );
        assert_eq!(store.len(), 3);

        let bytes = store.to_bytes();
        let loaded = PassageVectorStore::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.model, "test-model");
        assert_eq!(loaded.len(), 3);
        let rows: Vec<(&PassageKey, &[f32])> = loaded.iter().collect();
        assert_eq!(rows[0].0.index, 0);
        assert_eq!(rows[0].0.question_hash, None);
        assert_eq!(rows[0].1, &[1.0, 0.0]);
        assert_eq!(rows[1].0.question_hash, Some(41));
        assert_eq!(rows[1].1, &[0.6, 0.8]);
        assert_eq!(rows[2].0.hash, 9);
        assert_eq!(rows[2].1, &[0.0, 1.0]);

        assert!(PassageVectorStore::from_bytes(b"garbage").is_none());
        assert!(PassageVectorStore::from_bytes(&bytes[..bytes.len() - 1]).is_none());
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(PassageVectorStore::from_bytes(&padded).is_none());
    }

    #[test]
    fn fnv1a_is_pinned_to_the_published_test_vectors() {
        // Gloss-change detection stores these hashes in the vector
        // sidecar; any drift would silently re-embed every name. Pin
        // the function to the official FNV-1a 64-bit vectors.
        assert_eq!(fnv1a(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a("foobar"), 0x8594_4171_f739_67e8);
    }
}
