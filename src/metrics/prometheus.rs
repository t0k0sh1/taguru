//! The Prometheus text-exposition renderer (the read side of the
//! `record_*`/`note_*` family in `record.rs`) and its small text-
//! building helpers.

use super::*;

impl Metrics {
    /// The full Prometheus text-exposition body. Deterministic: the
    /// dynamic keys (routes, statuses) are sorted before emission, so
    /// identical state renders byte-identical output.
    pub fn render_prometheus(&self, gauges: &GaugeSnapshot) -> String {
        let mut out = String::new();

        let http = self.http.read();
        let mut routes: Vec<(&(String, String), &Arc<RouteStat>)> = http.iter().collect();
        routes.sort_by(|a, b| a.0.cmp(b.0));

        push_header(
            &mut out,
            "taguru_http_requests_total",
            "counter",
            "Total HTTP requests by method, route template, and status code.",
        );
        for ((method, route), stat) in &routes {
            let statuses = stat.by_status.read();
            let mut by_status: Vec<(u16, u64)> = statuses
                .iter()
                .map(|(&status, count)| (status, count.load(Ordering::Relaxed)))
                .collect();
            by_status.sort_unstable_by_key(|&(status, _)| status);
            for (status, count) in by_status {
                out.push_str(&format!(
                    "taguru_http_requests_total{{method=\"{method}\",route=\"{route}\",status=\"{status}\"}} {count}\n"
                ));
            }
        }

        push_header(
            &mut out,
            "taguru_http_request_duration_seconds",
            "histogram",
            "HTTP request latency by method and route template.",
        );
        for ((method, route), stat) in &routes {
            let snapshot = stat.latency.snapshot();
            push_histogram(
                &mut out,
                "taguru_http_request_duration_seconds",
                &format!("method=\"{method}\",route=\"{route}\""),
                &snapshot,
            );
        }
        drop(http);

        // Fixed-cardinality families always emit every label
        // combination, zeros included, so dashboards never see an
        // absent series.
        push_value(
            &mut out,
            "taguru_cache_hits_total",
            "counter",
            "Context cache hits (already resident, no disk load).",
            self.cache_hits.load(Ordering::Relaxed),
        );
        push_outcomes(
            &mut out,
            "taguru_cache_loads_total",
            "Context loads from disk into the resident cache, by outcome.",
            &self.cache_loads_ok,
            &self.cache_loads_failed,
        );
        push_outcomes(
            &mut out,
            "taguru_cache_evictions_total",
            "Contexts evicted from the resident cache, by outcome (a failed eviction save keeps the context resident).",
            &self.evictions_ok,
            &self.evictions_failed,
        );
        push_outcomes(
            &mut out,
            "taguru_flush_total",
            "Dirty-context persistence attempts, by outcome.",
            &self.flush_ok,
            &self.flush_failed,
        );
        push_outcomes(
            &mut out,
            "taguru_wal_appends_total",
            "Write-ahead log append batches, by outcome (a failed append refuses the write).",
            &self.wal_appends_ok,
            &self.wal_appends_failed,
        );
        push_header(
            &mut out,
            "taguru_embedding_requests_total",
            "counter",
            "Embedding provider round trips, by operation and outcome.",
        );
        for (operation, ok, failed) in [
            (
                "refresh",
                &self.embed_refresh_ok,
                &self.embed_refresh_failed,
            ),
            (
                "resolve",
                &self.embed_resolve_ok,
                &self.embed_resolve_failed,
            ),
        ] {
            out.push_str(&format!(
                "taguru_embedding_requests_total{{operation=\"{operation}\",outcome=\"ok\"}} {}\n",
                ok.load(Ordering::Relaxed)
            ));
            out.push_str(&format!(
                "taguru_embedding_requests_total{{operation=\"{operation}\",outcome=\"failed\"}} {}\n",
                failed.load(Ordering::Relaxed)
            ));
        }
        push_header(
            &mut out,
            "taguru_embedding_width_rebuilds_total",
            "counter",
            "Sidecar wipes forced by the provider changing vector width \
             behind an unchanged model name (each re-embeds the whole \
             store), by store.",
        );
        for (store, counter) in [
            ("gloss", &self.gloss_width_rebuilds),
            ("passages", &self.passage_width_rebuilds),
        ] {
            out.push_str(&format!(
                "taguru_embedding_width_rebuilds_total{{store=\"{store}\"}} {}\n",
                counter.load(Ordering::Relaxed)
            ));
        }
        push_header(
            &mut out,
            "taguru_embedding_duration_seconds",
            "histogram",
            "Embedding provider round-trip latency, retries included \
             (calls past the top bucket land in +Inf). Circuit-breaker \
             short-circuits never touch the provider and are excluded.",
        );
        let snapshot = self.embed_latency.snapshot();
        push_histogram(&mut out, "taguru_embedding_duration_seconds", "", &snapshot);
        if let Some(breaker) = &gauges.embed_breaker {
            push_header(
                &mut out,
                "taguru_embedding_breaker_state",
                "gauge",
                "Embedding provider circuit breaker: 0 = closed, 1 = open \
                 (calls refused until the cooldown lets a probe through), \
                 2 = half-open (probing).",
            );
            out.push_str(&format!(
                "taguru_embedding_breaker_state {}\n",
                breaker.state
            ));
            push_header(
                &mut out,
                "taguru_embedding_breaker_consecutive_failures",
                "gauge",
                "Consecutive provider-attempt failures counted toward the \
                 breaker's opening threshold (pinned at the threshold while \
                 open or half-open).",
            );
            out.push_str(&format!(
                "taguru_embedding_breaker_consecutive_failures {}\n",
                breaker.consecutive_failures
            ));
            push_header(
                &mut out,
                "taguru_embedding_breaker_opened_total",
                "counter",
                "Times the breaker opened (threshold of consecutive attempt \
                 failures reached, or a half-open probe failed).",
            );
            out.push_str(&format!(
                "taguru_embedding_breaker_opened_total {}\n",
                breaker.opened_total
            ));
            push_header(
                &mut out,
                "taguru_embedding_breaker_short_circuits_total",
                "counter",
                "Embedding calls refused without touching the provider \
                 because the breaker was open (or a probe was already in \
                 flight).",
            );
            out.push_str(&format!(
                "taguru_embedding_breaker_short_circuits_total {}\n",
                breaker.short_circuits_total
            ));
        }

        push_header(
            &mut out,
            "taguru_rerank_duration_seconds",
            "histogram",
            "Reranker provider round-trip latency (#307), retries included \
             (calls past the top bucket land in +Inf). A short-circuit that \
             never reached the provider (no reranker configured, a model \
             mismatch, fewer than two candidates, or an open circuit) is \
             still one sample, near zero.",
        );
        let rerank_snapshot = self.rerank_latency.snapshot();
        push_histogram(
            &mut out,
            "taguru_rerank_duration_seconds",
            "",
            &rerank_snapshot,
        );
        push_header(
            &mut out,
            "taguru_rerank_outcomes_total",
            "counter",
            "Requested reranks by how they ended (ADR 0006 §12): `ok` \
             reordered the pool; every other label is a degrade back to \
             the deterministic order, matching `plan.reranker.reason`. \
             Absent entirely from a call that never named `rerank`.",
        );
        for outcome in RerankOutcomeKind::ALL {
            out.push_str(&format!(
                "taguru_rerank_outcomes_total{{outcome=\"{}\"}} {}\n",
                outcome.as_str(),
                self.rerank_outcomes[outcome as usize].load(Ordering::Relaxed)
            ));
        }
        if let Some(breaker) = &gauges.rerank_breaker {
            push_header(
                &mut out,
                "taguru_rerank_breaker_state",
                "gauge",
                "Reranker provider circuit breaker: 0 = closed, 1 = open \
                 (calls refused until the cooldown lets a probe through), \
                 2 = half-open (probing).",
            );
            out.push_str(&format!("taguru_rerank_breaker_state {}\n", breaker.state));
            push_header(
                &mut out,
                "taguru_rerank_breaker_consecutive_failures",
                "gauge",
                "Consecutive provider-attempt failures counted toward the \
                 breaker's opening threshold (pinned at the threshold while \
                 open or half-open).",
            );
            out.push_str(&format!(
                "taguru_rerank_breaker_consecutive_failures {}\n",
                breaker.consecutive_failures
            ));
            push_header(
                &mut out,
                "taguru_rerank_breaker_opened_total",
                "counter",
                "Times the breaker opened (threshold of consecutive attempt \
                 failures reached, or a half-open probe failed).",
            );
            out.push_str(&format!(
                "taguru_rerank_breaker_opened_total {}\n",
                breaker.opened_total
            ));
            push_header(
                &mut out,
                "taguru_rerank_breaker_short_circuits_total",
                "counter",
                "Reranker calls refused without touching the provider \
                 because the breaker was open (or a probe was already in \
                 flight).",
            );
            out.push_str(&format!(
                "taguru_rerank_breaker_short_circuits_total {}\n",
                breaker.short_circuits_total
            ));
        }

        push_header(
            &mut out,
            "taguru_searches_total",
            "counter",
            "Successful retrieval requests by operation and outcome \
             (empty = the operation ran but matched nothing).",
        );
        for op in SearchOp::ALL {
            for (outcome, slot) in [("hit", 0), ("empty", 1)] {
                out.push_str(&format!(
                    "taguru_searches_total{{op=\"{}\",outcome=\"{outcome}\"}} {}\n",
                    op.as_str(),
                    self.searches[op as usize][slot].load(Ordering::Relaxed)
                ));
            }
        }
        push_header(
            &mut out,
            "taguru_retrieval_cache_total",
            "counter",
            "Exact-match retrieval cache consultations by operation and outcome \
             (a hit serves the cached response; a miss computes and fills). \
             Requests served while the cache is disabled land in neither.",
        );
        for op in RetrievalCacheOp::ALL {
            for (outcome, slot) in [("hit", 0), ("miss", 1)] {
                out.push_str(&format!(
                    "taguru_retrieval_cache_total{{op=\"{}\",outcome=\"{outcome}\"}} {}\n",
                    op.as_str(),
                    self.retrieval_cache[op as usize][slot].load(Ordering::Relaxed)
                ));
            }
        }
        push_header(
            &mut out,
            "taguru_semantic_cache_total",
            "counter",
            "Semantic (similarity) cache probes for passage search, by outcome: \
             hit = an equivalent earlier query's cached result served; stale = \
             equivalence held but the corpus moved on; guarded = cosine cleared \
             the threshold but the negation/numeric/entity guard refused; miss = \
             no candidate cleared the threshold. Probes run only on an \
             exact-cache miss with TAGURU_SEMANTIC_CACHE_THRESHOLD set.",
        );
        for outcome in SemanticCacheOutcome::ALL {
            out.push_str(&format!(
                "taguru_semantic_cache_total{{outcome=\"{}\"}} {}\n",
                outcome.as_str(),
                self.semantic_cache[outcome as usize].load(Ordering::Relaxed)
            ));
        }
        push_header(
            &mut out,
            "taguru_resolves_total",
            "counter",
            "Resolve and resolve_label requests by the tier that answered: \
             lexical = confident string match, semantic = embedding candidates \
             included, weak_lexical = only sub-confidence string fragments, \
             miss = nothing. A rising semantic share means cues are drifting \
             from the stored vocabulary.",
        );
        for tier in ResolveTier::ALL {
            out.push_str(&format!(
                "taguru_resolves_total{{tier=\"{}\"}} {}\n",
                tier.as_str(),
                self.resolve_tiers[tier as usize].load(Ordering::Relaxed)
            ));
        }

        push_header(
            &mut out,
            "taguru_schema_checks_total",
            "counter",
            "Schema pre-write checks by outcome, counted only at the \
             entrances that actually gate a write (POST \
             /contexts/{name}/associations, POST /import/taguru import) \
             for a context with an installed schema document — a \
             schema-free context, ?dry_run=true/preview, and POST \
             /schema/validate|/schema/audit never land here. warned = \
             mode warn rode violations out in the response; refused = \
             a reserved-label conflict, or mode strict with violations, \
             refused before anything was written.",
        );
        for outcome in SchemaOutcome::ALL {
            out.push_str(&format!(
                "taguru_schema_checks_total{{outcome=\"{}\"}} {}\n",
                outcome.as_str(),
                self.schema_checks[outcome as usize].load(Ordering::Relaxed)
            ));
        }

        push_header(
            &mut out,
            "taguru_passage_lane_contributions_total",
            "counter",
            "Served passage-search hits by which lane surfaced them: vector_only \
             is what the semantic lane adds beyond BM25; a persistent zero there \
             means the embedding spend buys nothing this corpus needed.",
        );
        for (lane, counter) in [
            ("bm25_only", &self.passage_hits_bm25_only),
            ("both_lanes", &self.passage_hits_both_lanes),
            ("vector_only", &self.passage_hits_vector_only),
        ] {
            out.push_str(&format!(
                "taguru_passage_lane_contributions_total{{lane=\"{lane}\"}} {}\n",
                counter.load(Ordering::Relaxed)
            ));
        }

        push_header(
            &mut out,
            "taguru_errors_total",
            "counter",
            "Requests answered 500, by cause: load = image/WAL unreadable on load, \
             wal_refused = the WAL could not durably record a write (nothing applied), \
             io = a file operation failed outside the WAL path, panic = a handler \
             unwound on a bug.",
        );
        for (kind, counter) in [
            ("io", &self.errors_io),
            ("load", &self.errors_load),
            ("panic", &self.errors_panic),
            ("wal_refused", &self.errors_wal_refused),
        ] {
            out.push_str(&format!(
                "taguru_errors_total{{kind=\"{kind}\"}} {}\n",
                counter.load(Ordering::Relaxed)
            ));
        }

        push_value(
            &mut out,
            "taguru_contexts_registered",
            "gauge",
            "Contexts known to the registry.",
            gauges.contexts_registered,
        );
        push_value(
            &mut out,
            "taguru_groups_registered",
            "gauge",
            "Groups known to the registry.",
            gauges.groups_registered,
        );
        push_value(
            &mut out,
            "taguru_contexts_resident",
            "gauge",
            "Contexts currently resident in memory.",
            gauges.contexts_resident,
        );
        push_value(
            &mut out,
            "taguru_resident_bytes",
            "gauge",
            "Modeled resident estimate of loaded contexts and cached vector stores (graph and vector footprints, NOT process RSS).",
            gauges.resident_bytes,
        );
        push_value(
            &mut out,
            "taguru_retrieval_cache_entries",
            "gauge",
            "Entries resident in the exact-match retrieval cache.",
            gauges.retrieval_cache_entries,
        );
        push_value(
            &mut out,
            "taguru_retrieval_cache_bytes",
            "gauge",
            "Bytes the exact-match retrieval cache holds (serialized responses; bounded by TAGURU_RETRIEVAL_CACHE_BYTES).",
            gauges.retrieval_cache_bytes,
        );
        push_value(
            &mut out,
            "taguru_semantic_cache_entries",
            "gauge",
            "Equivalence claims resident in the semantic cache (slots; payloads live in the exact-match cache).",
            gauges.semantic_cache_entries,
        );
        push_value(
            &mut out,
            "taguru_wal_bytes",
            "gauge",
            "Total bytes across all write-ahead logs; sustained growth means image flushes are failing.",
            gauges.wal_bytes,
        );
        push_value(
            &mut out,
            "taguru_passages_wal_bytes",
            "gauge",
            "Total bytes across all passage logs; these legitimately grow to about each context's snapshot size, so alert on growth far past that.",
            gauges.passages_wal_bytes,
        );
        push_value(
            &mut out,
            "taguru_dead_edges",
            "gauge",
            "Live count of edges with count == 0 across all contexts — dead weight compaction would shed right now.",
            gauges.dead_edges_total,
        );
        push_value(
            &mut out,
            "taguru_dead_attributions",
            "gauge",
            "Live count of attribution records unlinked from every chain but not yet reclaimed, across all contexts.",
            gauges.dead_attributions_total,
        );
        push_value(
            &mut out,
            "taguru_arena_slack_bytes",
            "gauge",
            "Lower-bound arena bytes behind removed aliases across all contexts — bytes compaction would not carry forward.",
            gauges.arena_slack_total,
        );
        push_value(
            &mut out,
            "taguru_unsourced_edges",
            "gauge",
            "Live count of edges carrying weight no named source explains, across all contexts.",
            gauges.unsourced_edges_total,
        );
        push_value(
            &mut out,
            "taguru_unsourced_weight",
            "gauge",
            "Total unsourced weight (absolute value), summed across all contexts.",
            gauges.unsourced_weight_total,
        );
        if !gauges.per_context.is_empty() {
            push_header(
                &mut out,
                "taguru_context_disk_bytes",
                "gauge",
                "On-disk bytes per context and file family: image, graph WAL, \
                 passages snapshot, passages WAL, and the sidecars (meta, \
                 sources, gloss vectors, passage vectors, BM25) summed. The \
                 WAL lanes are live bookkeeping; the rest refresh at each \
                 flush tick and POST /flush, so they lag up to one flush \
                 interval — a scrape never stats the data directory. Present \
                 only with TAGURU_METRICS_PER_CONTEXT.",
            );
            for row in &gauges.per_context {
                let context = escape_label(&row.name);
                for (file, value) in [
                    ("image", row.disk_image_bytes),
                    ("passages", row.disk_passages_bytes),
                    ("passages_wal", row.disk_passages_wal_bytes),
                    ("sidecars", row.disk_sidecar_bytes),
                    ("wal", row.disk_wal_bytes),
                ] {
                    out.push_str(&format!(
                        "taguru_context_disk_bytes{{context=\"{context}\",file=\"{file}\"}} {value}\n"
                    ));
                }
            }
            push_header(
                &mut out,
                "taguru_context_resident_bytes",
                "gauge",
                "Modeled resident bytes per context (graph plus cached \
                 gloss/passage/BM25/vector stores; 0 while cold) — the same \
                 per-entry accounting taguru_resident_bytes sums fleet-wide.",
            );
            for row in &gauges.per_context {
                out.push_str(&format!(
                    "taguru_context_resident_bytes{{context=\"{}\"}} {}\n",
                    escape_label(&row.name),
                    row.resident_bytes
                ));
            }
            push_header(
                &mut out,
                "taguru_context_pinned",
                "gauge",
                "1 when the context is pinned resident (exempt from cache \
                 eviction).",
            );
            for row in &gauges.per_context {
                out.push_str(&format!(
                    "taguru_context_pinned{{context=\"{}\"}} {}\n",
                    escape_label(&row.name),
                    u64::from(row.pinned)
                ));
            }
            type CountPick = fn(&ContextGaugeRow) -> u64;
            let count_families: [(&str, &str, CountPick); 4] = [
                (
                    "taguru_context_concepts",
                    "Concepts stored per context (live for hot contexts, \
                     last-saved snapshot for cold).",
                    |row| row.concepts,
                ),
                (
                    "taguru_context_associations",
                    "Associations stored per context (live for hot contexts, \
                     last-saved snapshot for cold).",
                    |row| row.associations,
                ),
                (
                    "taguru_context_labels",
                    "Distinct relation labels per context (live for hot \
                     contexts, last-saved snapshot for cold).",
                    |row| row.labels,
                ),
                (
                    "taguru_context_sources",
                    "Sources contributing to the graph per context (live for \
                     hot contexts, last-saved snapshot for cold).",
                    |row| row.sources,
                ),
            ];
            for (name, help, pick) in count_families {
                push_header(&mut out, name, "gauge", help);
                for row in &gauges.per_context {
                    out.push_str(&format!(
                        "{name}{{context=\"{}\"}} {}\n",
                        escape_label(&row.name),
                        pick(row)
                    ));
                }
            }
            push_header(
                &mut out,
                "taguru_context_schema_violations_total",
                "counter",
                "Schema violations recorded per context since boot, the \
                 per-context breakdown of taguru_schema_checks_total's \
                 warned/refused rows (present only with \
                 TAGURU_METRICS_PER_CONTEXT). A context with no installed \
                 schema, or one that has never failed a check, renders \
                 zero here rather than being absent.",
            );
            for row in &gauges.per_context {
                out.push_str(&format!(
                    "taguru_context_schema_violations_total{{context=\"{}\"}} {}\n",
                    escape_label(&row.name),
                    row.schema_violations
                ));
            }
            // Declared ceilings only (issue #136) — an uncapped fleet
            // renders no series and no header. Usage lives in the
            // sibling families above, so quota-vs-usage is one
            // division away on the dashboard.
            if gauges
                .per_context
                .iter()
                .any(|row| row.quota_storage_bytes.is_some() || row.quota_cache_bytes.is_some())
            {
                push_header(
                    &mut out,
                    "taguru_context_quota_bytes",
                    "gauge",
                    "Declared per-context ceilings (TAGURU_CONTEXT_QUOTAS): \
                     resource=\"storage\" caps the on-disk family (the sum of \
                     taguru_context_disk_bytes), resource=\"cache\" caps the \
                     resident share (taguru_context_resident_bytes) under \
                     eviction pressure.",
                );
                for row in &gauges.per_context {
                    let context = escape_label(&row.name);
                    for (resource, ceiling) in [
                        ("cache", row.quota_cache_bytes),
                        ("storage", row.quota_storage_bytes),
                    ] {
                        if let Some(ceiling) = ceiling {
                            out.push_str(&format!(
                                "taguru_context_quota_bytes{{context=\"{context}\",resource=\"{resource}\"}} {ceiling}\n"
                            ));
                        }
                    }
                }
            }
        }
        push_value(
            &mut out,
            "taguru_last_flush_success_timestamp_seconds",
            "gauge",
            "Unix time of the last successful image flush (0 = none since boot); alert on time() minus this.",
            self.last_flush_success_epoch.load(Ordering::Relaxed),
        );
        push_outcomes(
            &mut out,
            "taguru_auto_compactions_total",
            "Ratio-triggered auto-compactions run by the flusher tick (TAGURU_AUTO_COMPACT), by outcome; a failed candidate is retried while its dead ratio holds.",
            &self.auto_compact_ok,
            &self.auto_compact_failed,
        );
        push_value(
            &mut out,
            "taguru_auto_compact_reclaimed_bytes_total",
            "counter",
            "Resident bytes shed by successful auto-compactions (bytes before minus after, summed).",
            self.auto_compact_reclaimed_bytes.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_auto_compact_last_success_timestamp_seconds",
            "gauge",
            "Unix time of the last successful auto-compaction (0 = none since boot).",
            self.auto_compact_last_success_epoch.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_storage_quota_refusals_total",
            "counter",
            "Growth writes refused at a declared per-context storage ceiling (TAGURU_CONTEXT_QUOTAS) — graph writes, passage stores, and the import loop's per-batch pre-check all count here.",
            self.storage_quota_refusals.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_keyring_reloads_total",
            "counter",
            "Keyring hot reloads that armed a table (SIGHUP or config-file change; no-change reloads included).",
            self.keyring_reloads.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_keyring_reload_refusals_total",
            "counter",
            "Keyring reloads refused fail-closed (unreadable or malformed sources, or a reload that would disable authentication); the previous table stayed armed.",
            self.keyring_reload_refusals.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_replication_uploads_total",
            "counter",
            "Objects uploaded to the replication bucket (files, WAL segments, markers).",
            self.replication_uploads.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_replication_errors_total",
            "counter",
            "Failed replication operations (uploads, deletes, fence claims); the shipper retries.",
            self.replication_errors.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_replication_fenced",
            "gauge",
            "1 once a newer writer claimed the bucket and shipping fail-stopped (latched; restart to contest).",
            u64::from(self.replication_fenced.load(Ordering::Relaxed)),
        );
        push_value(
            &mut out,
            "taguru_replication_last_success_timestamp_seconds",
            "gauge",
            "Unix time of the last cycle that shipped everything it found (0 = none since boot); time() minus this bounds the DR restore's loss window.",
            self.replication_last_success_epoch.load(Ordering::Relaxed),
        );
        {
            let lag = self.replication_lag.lock();
            push_header(
                &mut out,
                "taguru_replication_lag_records",
                "gauge",
                "Acknowledged log records not yet in the bucket, per context and lane.",
            );
            for ((context, lane), entry) in lag.iter() {
                out.push_str(&format!(
                    "taguru_replication_lag_records{{context=\"{}\",lane=\"{lane}\"}} {}\n",
                    escape_label(context),
                    entry.behind_records
                ));
            }
            push_header(
                &mut out,
                "taguru_replication_lag_seconds",
                "gauge",
                "Age of the oldest unshipped record, per context and lane (0 = caught up).",
            );
            for ((context, lane), entry) in lag.iter() {
                out.push_str(&format!(
                    "taguru_replication_lag_seconds{{context=\"{}\",lane=\"{lane}\"}} {}\n",
                    escape_label(context),
                    entry.age_secs
                ));
            }
        }
        if self.replica_mode.load(Ordering::Relaxed) {
            push_value(
                &mut out,
                "taguru_replica",
                "gauge",
                "1 when this process serves as a read replica (writes refused, the bucket tailed).",
                1,
            );
            push_value(
                &mut out,
                "taguru_replica_generation",
                "gauge",
                "The replication generation this replica follows (0 = none complete in the bucket yet).",
                self.replica_generation.load(Ordering::Relaxed),
            );
            push_value(
                &mut out,
                "taguru_replica_manifest_timestamp_seconds",
                "gauge",
                "Store-clock mtime of the newest complete manifest at the last poll; time() minus this, plus the poll interval, bounds staleness.",
                self.replica_manifest_epoch.load(Ordering::Relaxed),
            );
            push_value(
                &mut out,
                "taguru_replica_last_poll_success_timestamp_seconds",
                "gauge",
                "Unix time of the last successful tailer poll (0 = none since boot); a growing gap means the bucket is unreachable and staleness is unbounded.",
                self.replica_last_poll_epoch.load(Ordering::Relaxed),
            );
            push_value(
                &mut out,
                "taguru_replica_poll_errors_total",
                "counter",
                "Failed tailer polls (bucket errors, un-appliable families); the tailer retries.",
                self.replica_poll_errors.load(Ordering::Relaxed),
            );
            let lag = self.replica_lag.lock();
            push_header(
                &mut out,
                "taguru_replica_applied_seq",
                "gauge",
                "Highest record seq this replica has applied, per context and lane.",
            );
            for ((context, lane), entry) in lag.iter() {
                out.push_str(&format!(
                    "taguru_replica_applied_seq{{context=\"{}\",lane=\"{lane}\"}} {}\n",
                    escape_label(context),
                    entry.applied_seq
                ));
            }
            push_header(
                &mut out,
                "taguru_replica_shipped_seq",
                "gauge",
                "Newest record seq the bucket ships, per context and lane; minus applied_seq = the promotion-time RPO in records.",
            );
            for ((context, lane), entry) in lag.iter() {
                out.push_str(&format!(
                    "taguru_replica_shipped_seq{{context=\"{}\",lane=\"{lane}\"}} {}\n",
                    escape_label(context),
                    entry.shipped_seq
                ));
            }
            push_header(
                &mut out,
                "taguru_replica_behind_seconds",
                "gauge",
                "How long the lane has been behind the shipped stream (0 = caught up).",
            );
            let now = Self::unix_now();
            for ((context, lane), entry) in lag.iter() {
                let behind = match entry.behind_since_epoch {
                    0 => 0,
                    since => now.saturating_sub(since),
                };
                out.push_str(&format!(
                    "taguru_replica_behind_seconds{{context=\"{}\",lane=\"{lane}\"}} {behind}\n",
                    escape_label(context),
                ));
            }
        }
        push_value(
            &mut out,
            "taguru_inflight_requests",
            "gauge",
            "Requests currently being served (probe endpoints exempt).",
            self.inflight.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_requests_shed_total",
            "counter",
            "Requests refused with a 503 at the in-flight ceiling (TAGURU_MAX_CONCURRENT_REQUESTS).",
            self.requests_shed.load(Ordering::Relaxed),
        );
        push_header(
            &mut out,
            "taguru_build_info",
            "gauge",
            "Build metadata as labels; the value is always 1.",
        );
        out.push_str(&format!(
            "taguru_build_info{{version=\"{}\"}} 1\n",
            env!("CARGO_PKG_VERSION")
        ));

        out
    }
}

fn push_header(out: &mut String, name: &str, kind: &str, help: &str) {
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
}

/// Prometheus label-value escaping. Context names are user text — a
/// name carrying a quote or a backslash must not be able to break the
/// exposition line it rides in (label values may hold anything else,
/// UTF-8 included).
fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// One gauge or counter family with a single, label-free value.
fn push_value(out: &mut String, name: &str, kind: &str, help: &str, value: impl std::fmt::Display) {
    push_header(out, name, kind, help);
    out.push_str(&format!("{name} {value}\n"));
}

/// A latency histogram family — bucket counts, the +Inf bucket, sum,
/// and count, the shape both `_duration_seconds` histograms share.
/// `labels` is the label body (no braces) every line in the family
/// carries ahead of `le`, or "" for the label-free embedding histogram.
fn push_histogram(out: &mut String, name: &str, labels: &str, snapshot: &HistogramSnapshot) {
    let le_prefix = if labels.is_empty() {
        String::new()
    } else {
        format!("{labels},")
    };
    let block = if labels.is_empty() {
        String::new()
    } else {
        format!("{{{labels}}}")
    };
    for ((_, le), value) in LATENCY_BUCKETS.iter().zip(snapshot.cumulative) {
        out.push_str(&format!(
            "{name}_bucket{{{le_prefix}le=\"{le}\"}} {value}\n"
        ));
    }
    let count = snapshot.count;
    out.push_str(&format!(
        "{name}_bucket{{{le_prefix}le=\"+Inf\"}} {count}\n"
    ));
    out.push_str(&format!(
        "{name}_sum{block} {}\n",
        snapshot.sum_micros as f64 / 1_000_000.0
    ));
    out.push_str(&format!("{name}_count{block} {count}\n"));
}

/// One counter family with the ok/failed outcome label, zeros included.
fn push_outcomes(out: &mut String, name: &str, help: &str, ok: &AtomicU64, failed: &AtomicU64) {
    push_header(out, name, "counter", help);
    out.push_str(&format!(
        "{name}{{outcome=\"ok\"}} {}\n",
        ok.load(Ordering::Relaxed)
    ));
    out.push_str(&format!(
        "{name}{{outcome=\"failed\"}} {}\n",
        failed.load(Ordering::Relaxed)
    ));
}
