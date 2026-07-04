//! Optional semantic entry tier: embeddings over concept and label
//! GLOSSES — each name plus its heaviest graph facts, because a lone
//! word carries too little signal for sentence-trained embedding
//! models, and the graph already owns the context. Retrieval itself
//! stays structural; only the entry needs meaning. Configured via `ARAG_EMBED_URL` /
//! `ARAG_EMBED_MODEL` / `ARAG_EMBED_API_KEY` against any
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

/// Anything that can turn short strings into vectors. The HTTP provider
/// is the real one; tests inject a deterministic mock.
pub trait EmbeddingProvider: Send + Sync {
    fn model(&self) -> &str;
    /// Returns one vector per input text, all the same dimension.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String>;
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
        let url = std::env::var("ARAG_EMBED_URL").ok()?;
        let model = std::env::var("ARAG_EMBED_MODEL").ok()?;
        Some(Self {
            url,
            model,
            api_key: std::env::var("ARAG_EMBED_API_KEY").ok(),
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

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        let mut request = self
            .agent
            .post(&self.url)
            .set("Content-Type", "application/json");
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

const VECTOR_MAGIC: &[u8; 8] = b"ARAGVEC2";

impl VectorStore {
    /// Reads a sidecar, returning an empty store on any problem — a
    /// corrupt vector cache costs a re-embed, never an outage.
    pub fn load(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => Self::from_bytes(&bytes).unwrap_or_else(|| {
                eprintln!("ignoring corrupt vector store at {}", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let bytes = self.to_bytes();
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)
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
}
