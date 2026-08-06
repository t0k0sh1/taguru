//! Per-language [`Grammar`](crate::code::grammar::Grammar)
//! implementations, one file per language, plus the extension table
//! that picks one for a path. Adding a language is one module here
//! and one row in [`for_extension`] — nothing outside this directory
//! learns about it.

// Explicit `#[path]` on every child, same reason as `code.rs` itself:
// this file is `#[path]`-loaded, so unpathed child mods would resolve
// in the wrong directory.
#[path = "grammars/rust.rs"]
pub(crate) mod rust;

use crate::code::grammar::Grammar;

/// Returns the grammar claiming `extension` (lowercase, no dot), or
/// `None` when the file is not a language this build parses — the
/// sync skips such files and says so once per run.
pub(crate) fn for_extension(extension: &str) -> Option<Box<dyn Grammar>> {
    match extension {
        "rs" => Some(Box::new(rust::RustGrammar)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_extension_table_routes_rs_and_refuses_the_rest() {
        assert_eq!(for_extension("rs").unwrap().language(), "rust");
        assert!(for_extension("py").is_none());
        assert!(for_extension("").is_none());
    }
}
