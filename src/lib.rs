//! Long-term semantic memory for LLMs. Knowledge accumulates as
//! (subject, relation label, object, signed weight, source)
//! associations, and retrieval is structural rather than
//! embedding-similarity: the cue resolves to a concept and the graph is
//! walked from there.
//!
//! The crate ships two binaries — `taguru`, the HTTP server that owns
//! persistence, auth, and observability, and `taguru-mcp`, an MCP stdio
//! bridge that lets an agent drive a running server. The library
//! surface is [`context`]: one [`context::Context`] is one 文脈 (one
//! context of meaning), a flat-buffer association graph whose whole
//! state round-trips as a single image through `to_bytes` /
//! `from_bytes`.

pub mod context;
pub mod deadline;

#[cfg(test)]
pub(crate) mod context_proptest;

// Shared with the binaries by dual inclusion — see the module docs.
mod crc32c;
mod hash;
