//! The language-agnostic contract between `taguru-code` and one
//! language's parser: a [`Grammar`] turns one source file into the
//! flat list of [`SymbolNode`]s the fact builder consumes. Everything
//! above a grammar — fact building, batch rendering, sync, lookup —
//! is language-blind by construction: the grammar alone decides which
//! AST nodes count as definitions, what a symbol's name path looks
//! like (`::` for Rust, `.` for Python), and what its one-line
//! signature is. Adding a language is one file under `grammars/` plus
//! one row in the extension table; nothing else learns about it.

/// What a definition is, as a superset over the languages this tool
/// will grow into. A grammar maps its own node types onto these; a
/// kind with no meaning in some language is simply never produced
/// there. The set is display vocabulary, not semantics — nothing
/// downstream branches on it beyond printing [`SymbolKind::keyword`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SymbolKind {
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    Impl,
    Const,
    Static,
    TypeAlias,
    Macro,
    Module,
    Class,
    Interface,
}

impl SymbolKind {
    /// The short keyword shown in section headings and `find` output
    /// (`fn parse_batch`, `struct Batch`). Methods print as `fn` —
    /// the distinction lives in the qualified name's nesting, not in
    /// the keyword.
    pub(crate) fn keyword(self) -> &'static str {
        match self {
            SymbolKind::Function | SymbolKind::Method => "fn",
            SymbolKind::Struct => "struct",
            SymbolKind::Enum => "enum",
            SymbolKind::Trait => "trait",
            SymbolKind::Impl => "impl",
            SymbolKind::Const => "const",
            SymbolKind::Static => "static",
            SymbolKind::TypeAlias => "type",
            SymbolKind::Macro => "macro",
            SymbolKind::Module => "mod",
            SymbolKind::Class => "class",
            SymbolKind::Interface => "interface",
        }
    }
}

/// One definition a grammar found: the unit the fact builder turns
/// into one passage paragraph plus `defined_in`/`contains` edges.
///
/// `qualified_name` is minted by the grammar as
/// `<repo-relative path><sep><name path>` (Rust: `::`), so it is
/// globally unique without any language-specific module resolution —
/// the file path IS the namespace. `parent` carries the enclosing
/// symbol's qualified name when the definition is nested (a method in
/// an `impl`); `None` means the file itself is the container.
///
/// `signature` must be a single line with no blank-line potential —
/// it becomes one passage paragraph verbatim, and the paragraph
/// splitter (`src/paragraph.rs`) breaks on blank lines. The fact
/// builder sanitizes defensively, but grammars should emit one line.
///
/// Lines are 1-based and inclusive, matching what editors display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SymbolNode {
    pub(crate) kind: SymbolKind,
    pub(crate) name: String,
    pub(crate) qualified_name: String,
    pub(crate) parent: Option<String>,
    pub(crate) signature: String,
    pub(crate) start_line: u32,
    pub(crate) end_line: u32,
}

/// One language's parser. `symbols` returns definitions in source
/// order (the fact builder's paragraph numbering is that order) and
/// never fails: a file the parser cannot make sense of yields an
/// empty list — the sync reports it as skipped, same posture as the
/// SDK connectors' diagnostics-instead-of-raising rule.
pub(crate) trait Grammar {
    /// Language tag, e.g. `"rust"` — used in reports and skip notes.
    fn language(&self) -> &'static str;

    /// File extensions (without the dot) this grammar claims.
    fn extensions(&self) -> &'static [&'static str];

    /// Parses one file into its definitions, in source order.
    fn symbols(&self, source: &str, repo_relative_path: &str) -> Vec<SymbolNode>;
}
