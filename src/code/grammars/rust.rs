//! Rust grammar: tree-sitter-rust node types → [`SymbolNode`]s. The
//! naming rule is the language-agnostic one — qualified names are
//! `<file path>::<name path>` — with two Rust-specific calls:
//!
//! - An `impl` block is a naming scope, not a symbol: methods inside
//!   `impl Context` become `<file>::Context::method` (the plan's
//!   worked example), and the block itself emits nothing. `impl Trait
//!   for Type` scopes by `Type`. The scope name may denote a type
//!   defined in another file — the `contains` edge then points at a
//!   concept the other file's `defined_in` completes, which is
//!   exactly how a shared graph is supposed to compose.
//! - `mod foo;` declarations emit nothing (the definition is the
//!   other file, which states its own truth); only inline
//!   `mod foo { ... }` is a symbol and a scope.

use crate::code::grammar::{Grammar, SymbolKind, SymbolNode};

#[derive(Default)]
pub(crate) struct RustGrammar;

impl Grammar for RustGrammar {
    fn language(&self) -> &'static str {
        "rust"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["rs"]
    }

    fn symbols(&self, source: &str, repo_relative_path: &str) -> Vec<SymbolNode> {
        let mut parser = tree_sitter::Parser::new();
        if parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .is_err()
        {
            return Vec::new();
        }
        let Some(tree) = parser.parse(source, None) else {
            return Vec::new();
        };
        let mut walker = Walker {
            source,
            path: repo_relative_path,
            symbols: Vec::new(),
        };
        walker.walk(tree.root_node(), &mut Vec::new(), false);
        walker.symbols
    }
}

struct Walker<'a> {
    source: &'a str,
    path: &'a str,
    symbols: Vec<SymbolNode>,
}

impl Walker<'_> {
    /// Depth-first, source order. `scope` is the name path of the
    /// enclosing symbols/impl blocks; `in_impl` marks functions as
    /// methods.
    fn walk(&mut self, node: tree_sitter::Node, scope: &mut Vec<String>, in_impl: bool) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "function_item" | "function_signature_item" => {
                    let kind = if in_impl {
                        SymbolKind::Method
                    } else {
                        SymbolKind::Function
                    };
                    self.emit(child, kind, scope, self.head_before_field(child, "body"));
                }
                "struct_item" | "union_item" => {
                    self.emit(
                        child,
                        SymbolKind::Struct,
                        scope,
                        self.head_before_field(child, "body"),
                    );
                }
                "enum_item" => {
                    self.emit(
                        child,
                        SymbolKind::Enum,
                        scope,
                        self.head_before_field(child, "body"),
                    );
                }
                "trait_item" => {
                    if let Some(name) = self.emit(
                        child,
                        SymbolKind::Trait,
                        scope,
                        self.head_before_field(child, "body"),
                    ) {
                        // Required methods and default bodies inside
                        // the trait nest under its name.
                        scope.push(name);
                        self.walk_field(child, "body", scope, true);
                        scope.pop();
                        continue;
                    }
                }
                "const_item" | "static_item" => {
                    let kind = if child.kind() == "const_item" {
                        SymbolKind::Const
                    } else {
                        SymbolKind::Static
                    };
                    self.emit(child, kind, scope, self.head_before_field(child, "value"));
                }
                "type_item" => {
                    self.emit(
                        child,
                        SymbolKind::TypeAlias,
                        scope,
                        self.head_before_field(child, "type"),
                    );
                }
                "macro_definition" => {
                    self.emit(child, SymbolKind::Macro, scope, None);
                }
                "mod_item" => {
                    // Only inline `mod name { ... }` — a `mod name;`
                    // declaration has no body and emits nothing.
                    if child.child_by_field_name("body").is_some()
                        && let Some(name) = self.emit(
                            child,
                            SymbolKind::Module,
                            scope,
                            self.head_before_field(child, "body"),
                        )
                    {
                        scope.push(name);
                        self.walk_field(child, "body", scope, false);
                        scope.pop();
                    }
                }
                "impl_item" => {
                    // A scope, not a symbol: `impl Context` and
                    // `impl Display for Context` both scope by the
                    // implemented type's last path segment.
                    if let Some(scope_name) = child
                        .child_by_field_name("type")
                        .and_then(|type_node| self.text(type_node))
                        .map(type_scope_name)
                        .filter(|name| !name.is_empty())
                    {
                        scope.push(scope_name);
                        self.walk_field(child, "body", scope, true);
                        scope.pop();
                    }
                }
                // Containers that hide items one level down.
                "foreign_mod_item" => self.walk(child, scope, false),
                _ => {}
            }
        }
    }

    fn walk_field(
        &mut self,
        node: tree_sitter::Node,
        field: &str,
        scope: &mut Vec<String>,
        in_impl: bool,
    ) {
        if let Some(body) = node.child_by_field_name(field) {
            self.walk(body, scope, in_impl);
        }
    }

    /// Emits one symbol if the node has a name; returns that name so
    /// callers can extend the scope with it.
    fn emit(
        &mut self,
        node: tree_sitter::Node,
        kind: SymbolKind,
        scope: &[String],
        signature: Option<String>,
    ) -> Option<String> {
        let name = node
            .child_by_field_name("name")
            .and_then(|name_node| self.text(name_node))?
            .to_string();
        let mut qualified = String::from(self.path);
        for segment in scope {
            qualified.push_str("::");
            qualified.push_str(segment);
        }
        qualified.push_str("::");
        qualified.push_str(&name);
        let parent = if scope.is_empty() {
            None
        } else {
            let mut parent = String::from(self.path);
            for segment in scope {
                parent.push_str("::");
                parent.push_str(segment);
            }
            Some(parent)
        };
        self.symbols.push(SymbolNode {
            kind,
            name: name.clone(),
            qualified_name: qualified,
            parent,
            signature: signature.unwrap_or_else(|| format!("{} {name}", kind.keyword())),
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
        });
        Some(name)
    }

    /// The declaration head: node text up to (not including) `field`,
    /// with trailing `=`/whitespace shed — `pub fn f(x: u32) -> u32`
    /// without the body, `const N: usize` without the value.
    fn head_before_field(&self, node: tree_sitter::Node, field: &str) -> Option<String> {
        let end = node
            .child_by_field_name(field)
            .map(|inner| inner.start_byte())
            .unwrap_or_else(|| node.end_byte());
        let head = self
            .source
            .get(node.start_byte()..end)?
            .trim_end_matches(|c: char| c.is_whitespace() || c == '=' || c == ';');
        Some(head.to_string()).filter(|text| !text.is_empty())
    }

    fn text(&self, node: tree_sitter::Node) -> Option<&str> {
        self.source.get(node.start_byte()..node.end_byte())
    }
}

/// `impl` scope name from the implemented type's text: generics cut,
/// last path segment kept — `crate::ingest::Batch<'a>` scopes as
/// `Batch`, matching what a reader (and a lookup cue) calls it.
fn type_scope_name(type_text: &str) -> String {
    let base = type_text.split('<').next().unwrap_or("").trim();
    base.rsplit("::").next().unwrap_or("").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> Vec<SymbolNode> {
        RustGrammar.symbols(source, "src/x.rs")
    }

    fn names(symbols: &[SymbolNode]) -> Vec<&str> {
        symbols
            .iter()
            .map(|symbol| symbol.qualified_name.as_str())
            .collect()
    }

    #[test]
    fn top_level_items_map_to_their_kinds() {
        let symbols = parse(
            "pub fn run() {}\n\
             struct Config { port: u16 }\n\
             enum Mode { A, B }\n\
             trait Store {}\n\
             const LIMIT: usize = 8;\n\
             static NAME: &str = \"x\";\n\
             type Result2 = Result<(), ()>;\n\
             macro_rules! log { () => {} }\n",
        );
        let kinds: Vec<(SymbolKind, &str)> = symbols
            .iter()
            .map(|symbol| (symbol.kind, symbol.name.as_str()))
            .collect();
        assert_eq!(
            kinds,
            vec![
                (SymbolKind::Function, "run"),
                (SymbolKind::Struct, "Config"),
                (SymbolKind::Enum, "Mode"),
                (SymbolKind::Trait, "Store"),
                (SymbolKind::Const, "LIMIT"),
                (SymbolKind::Static, "NAME"),
                (SymbolKind::TypeAlias, "Result2"),
                (SymbolKind::Macro, "log"),
            ]
        );
        assert_eq!(symbols[0].qualified_name, "src/x.rs::run");
        assert_eq!(symbols[0].signature, "pub fn run()");
        assert_eq!(symbols[4].signature, "const LIMIT: usize");
    }

    /// The plan's worked shape: methods inside `impl` nest under the
    /// type's name, and the impl block itself emits nothing — even
    /// packed tight with no blank lines between methods (the case
    /// that killed the raw-source passage design).
    #[test]
    fn impl_blocks_scope_methods_without_emitting_themselves() {
        let symbols = parse(
            "pub struct Context;\n\
             impl Context {\n\
                 pub fn resolve(&self) {}\n\
                 pub fn recall(&self) {}\n\
             }\n\
             impl std::fmt::Display for Context {\n\
                 fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) }\n\
             }\n",
        );
        assert_eq!(
            names(&symbols),
            vec![
                "src/x.rs::Context",
                "src/x.rs::Context::resolve",
                "src/x.rs::Context::recall",
                "src/x.rs::Context::fmt",
            ]
        );
        let resolve = &symbols[1];
        assert_eq!(resolve.kind, SymbolKind::Method);
        assert_eq!(resolve.parent.as_deref(), Some("src/x.rs::Context"));
        assert_eq!(resolve.signature, "pub fn resolve(&self)");
    }

    #[test]
    fn generic_and_path_qualified_impl_types_scope_by_the_bare_name() {
        let symbols = parse("impl<'a> crate::ingest::Batch<'a> {\n    fn apply(&self) {}\n}\n");
        assert_eq!(names(&symbols), vec!["src/x.rs::Batch::apply"]);
    }

    #[test]
    fn inline_modules_nest_and_declarations_emit_nothing() {
        let symbols = parse(
            "mod other;\n\
             mod inner {\n\
                 pub fn helper() {}\n\
                 pub struct Deep;\n\
                 impl Deep { fn touch(&self) {} }\n\
             }\n",
        );
        assert_eq!(
            names(&symbols),
            vec![
                "src/x.rs::inner",
                "src/x.rs::inner::helper",
                "src/x.rs::inner::Deep",
                "src/x.rs::inner::Deep::touch",
            ]
        );
        assert_eq!(symbols[0].kind, SymbolKind::Module);
        assert_eq!(symbols[3].parent.as_deref(), Some("src/x.rs::inner::Deep"));
    }

    #[test]
    fn trait_methods_nest_under_the_trait() {
        let symbols = parse(
            "trait Store {\n\
                 fn get(&self, key: &str) -> Option<String>;\n\
                 fn put(&self, key: &str, value: String) { let _ = (key, value); }\n\
             }\n",
        );
        assert_eq!(
            names(&symbols),
            vec![
                "src/x.rs::Store",
                "src/x.rs::Store::get",
                "src/x.rs::Store::put",
            ]
        );
        assert_eq!(symbols[1].kind, SymbolKind::Method);
        assert_eq!(
            symbols[1].signature,
            "fn get(&self, key: &str) -> Option<String>"
        );
    }

    #[test]
    fn multi_line_signatures_and_line_ranges_come_out_editor_shaped() {
        let source = "pub(crate) fn parse_batch(\n    reader: impl BufRead,\n) -> Result<Batch, String> {\n    todo!()\n}\n";
        let symbols = parse(source);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].start_line, 1);
        assert_eq!(symbols[0].end_line, 5);
        // The raw head keeps its newlines; facts::build's sanitizer
        // collapses them — this stays the grammar's honest slice.
        assert!(
            symbols[0]
                .signature
                .starts_with("pub(crate) fn parse_batch(")
        );
        assert!(symbols[0].signature.ends_with("-> Result<Batch, String>"));
    }

    #[test]
    fn broken_source_still_yields_the_parseable_symbols() {
        let symbols = parse("fn ok() {}\nfn broken( {\n");
        assert!(names(&symbols).contains(&"src/x.rs::ok"));
    }

    #[test]
    fn an_unparseable_or_empty_file_yields_no_symbols_not_an_error() {
        assert!(parse("").is_empty());
        assert!(parse("こんにちは、これはRustではありません。").is_empty());
    }
}
