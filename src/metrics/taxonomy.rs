//! The label vocabulary catalog: every fixed enum/type this metrics
//! surface renders as Prometheus labels or gauge rows, plus the two
//! lag types the replication and replica families key by (context,
//! lane).

/// One log lane's shipping lag as the dashboard sees it: records not
/// yet in the bucket, and how long the oldest of them has waited.
#[derive(Clone, Copy, Default)]
pub(super) struct ReplicationLag {
    pub(super) behind_records: u64,
    pub(super) age_secs: u64,
}

/// One log lane's tail lag as a replica sees it: the record seq its
/// local materialization carries vs the newest the manifest ships,
/// and (unix seconds, 0 = caught up) since when it has been behind.
#[derive(Clone, Copy, Default)]
pub(super) struct ReplicaLag {
    pub(super) applied_seq: u64,
    pub(super) shipped_seq: u64,
    pub(super) behind_since_epoch: u64,
}

/// Why a request answered 500. The status code alone cannot separate
/// these — and they demand different responses from an operator: `load`
/// is a corrupt or unreadable image (restore from backup), `wal_refused`
/// is the durability path failing writes (check the disk NOW), `io` is
/// a sidecar or image file operation failing outside the WAL path, and
/// `panic` is a handler unwinding on a bug — not a disk problem, so it
/// warrants a bug report instead of an operator remedy.
#[derive(Clone, Copy)]
pub enum ErrorKind {
    Load,
    WalRefused,
    Io,
    Panic,
}

/// The retrieval operations whose hit/empty split is tracked — the
/// aggregate "is the memory answering" pulse. A fixed set on purpose:
/// ops are the labels, so the family's cardinality is sealed here.
#[derive(Clone, Copy)]
pub enum SearchOp {
    Resolve,
    ResolveLabel,
    Recall,
    Query,
    Activate,
    SearchPassages,
    SearchCommunities,
    Explore,
    Paths,
}

impl SearchOp {
    pub(super) const ALL: [SearchOp; 9] = [
        SearchOp::Resolve,
        SearchOp::ResolveLabel,
        SearchOp::Recall,
        SearchOp::Query,
        SearchOp::Activate,
        SearchOp::SearchPassages,
        SearchOp::SearchCommunities,
        SearchOp::Explore,
        SearchOp::Paths,
    ];

    /// `pub(crate)`: also `taguru.op`'s source of truth on the spans
    /// this vocabulary lines up with (`taguru.passage_search` and its
    /// server-composed siblings, ADR 0008 §6). `src/mcp/retrieve.rs`
    /// cannot call this directly — it is dual-included into the stdio
    /// bridge, which has no `metrics` module — so its own phase spans
    /// (and `src/api/evidence/assemble.rs`'s) copy the same string
    /// literals instead. That copy is NOT compile-time checked against
    /// this match — a renamed variant here would silently drift from
    /// those literals — so `tests::taguru_op_literals_match_search_op`
    /// below pins the two together at test time.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            SearchOp::Resolve => "resolve",
            SearchOp::ResolveLabel => "resolve_label",
            SearchOp::Recall => "recall",
            SearchOp::Query => "query",
            SearchOp::Activate => "activate",
            SearchOp::SearchPassages => "search_passages",
            SearchOp::SearchCommunities => "search_communities",
            SearchOp::Explore => "explore",
            SearchOp::Paths => "paths",
        }
    }
}

/// The retrieval surfaces the exact-match cache fronts — the label
/// vocabulary of `taguru_retrieval_cache_total`, and the cache key's
/// op discriminant (each op reads a different pair of revision lanes,
/// so the same request text under two ops must never collide). Cross
/// variants fold into their base op: the resolved target list already
/// distinguishes them in the key, and a per-variant label would split
/// the hit-rate signal without adding meaning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RetrievalCacheOp {
    Recall,
    Query,
    SearchPassages,
    SearchCommunities,
}

impl RetrievalCacheOp {
    pub(super) const ALL: [RetrievalCacheOp; 4] = [
        RetrievalCacheOp::Recall,
        RetrievalCacheOp::Query,
        RetrievalCacheOp::SearchPassages,
        RetrievalCacheOp::SearchCommunities,
    ];

    pub(super) fn as_str(self) -> &'static str {
        match self {
            RetrievalCacheOp::Recall => "recall",
            RetrievalCacheOp::Query => "query",
            RetrievalCacheOp::SearchPassages => "search_passages",
            RetrievalCacheOp::SearchCommunities => "search_communities",
        }
    }

    /// The `searches` family op a cache hit replays its `note_search`
    /// under — kept here so the two vocabularies cannot drift.
    pub(crate) fn search_op(self) -> SearchOp {
        match self {
            RetrievalCacheOp::Recall => SearchOp::Recall,
            RetrievalCacheOp::Query => SearchOp::Query,
            RetrievalCacheOp::SearchPassages => SearchOp::SearchPassages,
            RetrievalCacheOp::SearchCommunities => SearchOp::SearchCommunities,
        }
    }
}

/// How one semantic-cache probe ended — the label vocabulary of
/// `taguru_semantic_cache_total`. No op label: the tier fronts passage
/// search only. `guarded` and `stale` are first-class rather than
/// folded into `miss` because they are the tuning signals the
/// threshold's operational follow-up reads: `guarded` counts
/// paraphrase-close pairs the text guard split, `stale` counts claims
/// that held while the corpus moved on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SemanticCacheOutcome {
    /// A claim held and its rewritten exact key was live: served.
    Hit,
    /// A claim held but the rewritten key missed (a write bumped a
    /// lane, or the exact tier evicted the canonical): computed fresh.
    Stale,
    /// Cosine cleared the threshold but the negation/numeric/entity
    /// guard refused every candidate: computed fresh.
    Guarded,
    /// No candidate cleared the threshold: computed fresh.
    Miss,
}

impl SemanticCacheOutcome {
    pub(super) const ALL: [SemanticCacheOutcome; 4] = [
        SemanticCacheOutcome::Hit,
        SemanticCacheOutcome::Stale,
        SemanticCacheOutcome::Guarded,
        SemanticCacheOutcome::Miss,
    ];

    /// `pub(crate)`, not private: also the `taguru.cache.semantic` span
    /// attribute's source of truth (ADR 0008 §6) — one vocabulary for
    /// both the Prometheus label and the span, never a parallel
    /// spelling.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            SemanticCacheOutcome::Hit => "hit",
            SemanticCacheOutcome::Stale => "stale",
            SemanticCacheOutcome::Guarded => "guarded",
            SemanticCacheOutcome::Miss => "miss",
        }
    }
}

/// How one `rerank::drive` call ended (#307) — the label vocabulary of
/// `taguru_rerank_outcomes_total`, one closed variant per reason token
/// `src/api/evidence/rerank.rs` itself defines (`REASON_NOT_CONFIGURED`
/// and siblings), plus `Ok` for a successful reorder. Kept as a closed
/// metrics-only enum, the same discipline [`SemanticCacheOutcome`]
/// uses, so every label renders from zero rather than only appearing
/// once its first outcome fires.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RerankOutcomeKind {
    Ok,
    NotConfigured,
    ModelMismatch,
    EmptyPool,
    InvalidPermutation,
    CircuitOpen,
    Timeout,
    ProviderError,
}

impl RerankOutcomeKind {
    pub(super) const ALL: [RerankOutcomeKind; 8] = [
        RerankOutcomeKind::Ok,
        RerankOutcomeKind::NotConfigured,
        RerankOutcomeKind::ModelMismatch,
        RerankOutcomeKind::EmptyPool,
        RerankOutcomeKind::InvalidPermutation,
        RerankOutcomeKind::CircuitOpen,
        RerankOutcomeKind::Timeout,
        RerankOutcomeKind::ProviderError,
    ];

    pub(super) fn as_str(self) -> &'static str {
        match self {
            RerankOutcomeKind::Ok => "ok",
            RerankOutcomeKind::NotConfigured => "not_configured",
            RerankOutcomeKind::ModelMismatch => "model_mismatch",
            RerankOutcomeKind::EmptyPool => "empty_pool",
            RerankOutcomeKind::InvalidPermutation => "invalid_permutation",
            RerankOutcomeKind::CircuitOpen => "circuit_open",
            RerankOutcomeKind::Timeout => "timeout",
            RerankOutcomeKind::ProviderError => "provider_error",
        }
    }

    /// `rerank::drive`'s outcome token, verbatim from
    /// `src/api/evidence/rerank.rs`'s `REASON_*` constants — matched by
    /// value rather than shared as one enum across both modules so
    /// `rerank.rs` (ADR 0006 §12's boundary) never has to import a
    /// metrics-only type. Falls back to `ProviderError`, the closest
    /// "something the provider did" bucket, for any token this list has
    /// not been kept in sync with — a debug assertion catches that drift
    /// in tests instead of silently mislabeling in production.
    pub(super) fn from_token(token: &str) -> Self {
        let found = Self::ALL.into_iter().find(|kind| kind.as_str() == token);
        debug_assert!(
            found.is_some(),
            "unrecognized rerank outcome token: {token}"
        );
        found.unwrap_or(RerankOutcomeKind::ProviderError)
    }
}

/// Which tier ultimately answered a resolve (or resolve_label) —
/// classified from the served payload, so every serve path lands in
/// exactly one bucket. The drift signal lives here: a rising
/// `semantic` share means the cues clients send are pulling away from
/// the stored vocabulary; a rising `miss` share means coverage gaps.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResolveTier {
    /// A confident string match answered alone.
    Lexical,
    /// Embedding candidates were part of the answer.
    Semantic,
    /// Only sub-confidence string fragments survived (the semantic
    /// tier ran but contributed nothing, failed, or is not configured).
    WeakLexical,
    /// Nothing at all.
    Miss,
}

impl ResolveTier {
    pub(super) const ALL: [ResolveTier; 4] = [
        ResolveTier::Lexical,
        ResolveTier::Semantic,
        ResolveTier::WeakLexical,
        ResolveTier::Miss,
    ];

    /// The stable name shared by the metric label and the search event
    /// log, so the two vocabularies can never drift apart.
    pub fn as_str(self) -> &'static str {
        match self {
            ResolveTier::Lexical => "lexical",
            ResolveTier::Semantic => "semantic",
            ResolveTier::WeakLexical => "weak_lexical",
            ResolveTier::Miss => "miss",
        }
    }
}

/// How one schema pre-write check ended (#388, S10 of #218's ADR 0009
/// split §15) — the label vocabulary of `taguru_schema_checks_total`.
/// Counted only at the entrances that actually gate a write
/// (`POST /contexts/{name}/associations`, `POST /import`/`taguru
/// import`) for a context that has an installed schema document
/// (ADR 0009 §6.3's single condition, not `mode`); a schema-free
/// context never touches this family. `?dry_run=true`/`preview_batch`
/// and `POST /schema/validate`/`/schema/audit` are diagnostics, not
/// write gates, and are deliberately excluded — otherwise a
/// validate-then-apply workflow would double-count the same refusal.
/// Kept as a closed metrics-only enum, the same discipline
/// [`RerankOutcomeKind`] uses, so every label renders from zero rather
/// than only appearing once its first outcome fires.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SchemaOutcome {
    /// No reserved-label conflict and no domain/range/closed-label
    /// violation (including every check against a schema-free
    /// context, and any check while `mode == off`).
    Ok,
    /// `mode == warn` and the write proceeded with violations recorded
    /// in the response instead of refusing.
    Warned,
    /// The write refused before anything landed — a reserved-label
    /// conflict (any mode), or `mode == strict` with violations.
    Refused,
}

impl SchemaOutcome {
    pub(super) const ALL: [SchemaOutcome; 3] = [
        SchemaOutcome::Ok,
        SchemaOutcome::Warned,
        SchemaOutcome::Refused,
    ];

    pub(super) fn as_str(self) -> &'static str {
        match self {
            SchemaOutcome::Ok => "ok",
            SchemaOutcome::Warned => "warned",
            SchemaOutcome::Refused => "refused",
        }
    }
}

/// How much per-context detail the scrape carries
/// (`TAGURU_METRICS_PER_CONTEXT`, issue #137). Off by default on
/// purpose: per-context labels × many contexts is exactly the
/// cardinality blow-up the route-template rule at the top of this
/// file exists to prevent, so an operator opts in — and can bound a
/// large fleet's series count with `Top`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PerContextMetrics {
    /// No per-context families on the scrape (the default).
    #[default]
    Off,
    /// Every context gets its rows.
    All,
    /// Only the N largest contexts by total on-disk bytes get rows —
    /// membership shifts as sizes shift, which Prometheus handles as
    /// series going stale, not as an error.
    Top(usize),
}

/// One context's row behind the `taguru_context_*` families — collected
/// by `gauge_snapshot` only while [`PerContextMetrics`] asks for it.
/// Disk sizes come from flush-time bookkeeping, everything else from
/// registry state already in memory: a scrape never walks the data
/// directory (see `AppState::refresh_disk_usage`).
#[derive(Debug, Clone)]
pub struct ContextGaugeRow {
    pub name: String,
    pub pinned: bool,
    /// Graph plus cached gloss/passage/BM25/vector stores — the same
    /// per-entry accounting `taguru_resident_bytes` sums fleet-wide.
    pub resident_bytes: u64,
    pub disk_image_bytes: u64,
    pub disk_wal_bytes: u64,
    pub disk_passages_bytes: u64,
    pub disk_passages_wal_bytes: u64,
    /// Meta + sources + gloss vectors + passage vectors + BM25 +
    /// schema, summed.
    pub disk_sidecar_bytes: u64,
    /// Declared ceilings (`TAGURU_CONTEXT_QUOTAS`, issue #136), when
    /// this context has them — `None` renders no series at all, so an
    /// uncapped fleet's scrape is byte-identical to before quotas
    /// existed.
    pub quota_storage_bytes: Option<u64>,
    pub quota_cache_bytes: Option<u64>,
    pub concepts: u64,
    pub associations: u64,
    pub labels: u64,
    pub sources: u64,
    /// Schema violations recorded by [`AppState::note_schema_check`]
    /// since boot (or since this entry's own creation) — a Prometheus
    /// counter, not a live count of currently-outstanding violations
    /// (those are never swept; ADR 0009 §7.2's write-time check is the
    /// only judge, and it never sweeps the graph either). Zero for
    /// every context that has never failed a schema check, including
    /// every schema-free one.
    pub schema_violations: u64,
}

impl ContextGaugeRow {
    /// Total on-disk bytes across every family — what a restore would
    /// move, and the ranking key for [`PerContextMetrics::Top`].
    pub fn disk_total_bytes(&self) -> u64 {
        self.disk_image_bytes
            + self.disk_wal_bytes
            + self.disk_passages_bytes
            + self.disk_passages_wal_bytes
            + self.disk_sidecar_bytes
    }
}

/// Point-in-time gauges, computed from the registry at scrape time
/// rather than maintained incrementally — they cannot drift.
pub struct GaugeSnapshot {
    pub contexts_registered: u64,
    pub groups_registered: u64,
    pub contexts_resident: u64,
    pub resident_bytes: u64,
    /// Total bytes across every context's write-ahead log. A healthy
    /// server truncates each log every flush interval; sustained
    /// growth here means images are failing to save.
    pub wal_bytes: u64,
    /// Total bytes across every context's PASSAGE log. This one grows
    /// legitimately up to about each context's snapshot size before its
    /// ratio-triggered compaction; growth far past the snapshots means
    /// compactions are failing.
    pub passages_wal_bytes: u64,
    /// Sum, across every context, of edges with `count == 0` — dead
    /// weight `compact` would shed right now. Deliberately NOT broken
    /// down per context here: unlike route templates, a context name is
    /// unbounded, user-chosen data, and this metrics surface only ever
    /// mints fixed-cardinality series (see `http`'s route-template
    /// comment). Per-context detail lives at `GET /contexts` and
    /// `taguru inspect` — or, opted into and bounded via
    /// `TAGURU_METRICS_PER_CONTEXT` (#137), in the `taguru_context_*`
    /// families below.
    pub dead_edges_total: u64,
    /// Sum, across every context, of attribution records unlinked from
    /// every chain but not yet reclaimed by compaction.
    pub dead_attributions_total: u64,
    /// Sum, across every context, of the lower-bound arena bytes behind
    /// removed aliases.
    pub arena_slack_total: u64,
    /// Sum, across every context, of edges carrying weight no named
    /// source explains — see [`taguru::context::Context::unsourced_summary`].
    pub unsourced_edges_total: u64,
    /// Sum, across every context, of unsourced weight (absolute value).
    pub unsourced_weight_total: f64,
    /// The embedding provider's circuit breaker, present exactly when a
    /// provider with one is configured — the family is absent from a
    /// lexical-only server's scrape, like the replica family off a
    /// writer.
    pub embed_breaker: Option<crate::embedding::BreakerSnapshot>,
    /// The reranker provider's circuit breaker (#307), present exactly
    /// when a provider is configured — the same "absent, not
    /// zeroed" gating [`Self::embed_breaker`] uses.
    pub rerank_breaker: Option<crate::breaker::BreakerSnapshot>,
    /// Entries resident in the exact-match retrieval cache, and the
    /// bytes they hold — read from the cache at scrape time like every
    /// other gauge here, so they cannot drift.
    pub retrieval_cache_entries: u64,
    pub retrieval_cache_bytes: u64,
    /// Equivalence claims resident in the semantic cache (slots, not
    /// bytes — payloads live in the exact tier).
    pub semantic_cache_entries: u64,
    /// Per-context rows, empty unless `TAGURU_METRICS_PER_CONTEXT`
    /// asked for them — the one other sanctioned exception (after the
    /// replication lag maps) to this file's no-context-labels rule,
    /// and gated the same way the replica family is: an opted-out
    /// scrape stays byte-identical to what it was.
    pub per_context: Vec<ContextGaugeRow>,
}
