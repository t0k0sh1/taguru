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
        let mut vectors = Vec::with_capacity(data.len());
        for item in data {
            let embedding = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or("embedding entry carries no vector")?;
            let mut vector: Vec<f32> = embedding
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            normalize(&mut vector);
            vectors.push(vector);
        }
        if vectors.len() != texts.len() {
            return Err(format!(
                "embedding response returned {} vectors for {} inputs",
                vectors.len(),
                texts.len()
            ));
        }
        Ok(vectors)
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

/// One context's name vectors: a derived cache keyed by the model that
/// produced it. Wholesale discarded on model change — vectors from two
/// models must never be compared.
#[derive(Debug, Default)]
pub struct VectorStore {
    pub model: String,
    /// name → (gloss hash, unit vector). The hash is of the gloss the
    /// vector was computed from, so a refresh re-embeds exactly the
    /// names whose graph context changed.
    pub concepts: HashMap<String, (u64, Vec<f32>)>,
    pub labels: HashMap<String, (u64, Vec<f32>)>,
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
        let table = |table: &HashMap<String, (u64, Vec<f32>)>| -> usize {
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

fn write_string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

fn write_table(out: &mut Vec<u8>, table: &HashMap<String, (u64, Vec<f32>)>) {
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

fn read_table(bytes: &[u8], pos: &mut usize) -> Option<HashMap<String, (u64, Vec<f32>)>> {
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
