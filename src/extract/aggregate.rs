//! Cross-output validation and merging: `Extraction`/`Fact`, the
//! cross-chunk and schema issue checks, and `merge`.

use super::*;

/// What one document's chunks amounted to, after the contract is
/// enforced: exact-duplicate triples folded (one fact, one line — the
/// in-document paraphrase rule), malformed items dropped, and aliases
/// kept only when their canonical is a name the associations intern —
/// an alias pointing nowhere would fail the whole batch at apply time.
pub(super) struct Extraction {
    pub(super) associations: Vec<Fact>,
    pub(super) concepts: BTreeMap<String, String>,
    pub(super) labels: BTreeMap<String, String>,
    pub(super) questions: Vec<(u32, String)>,
    pub(super) duplicates: usize,
    pub(super) dropped: usize,
    /// ADR 0023 §3.4: which of `merge`'s input outputs each kept item
    /// came from — the output's position in the input list (the same
    /// number [`Fact::origin`] carries), keyed by the item's content
    /// key. A duplicate folded across outputs is attributed to the
    /// output that was kept, i.e. the first.
    pub(super) origins: BTreeMap<ItemKey, usize>,
    /// #786 / ADR 0024: every item `merge` counted in `dropped`
    /// or `duplicates`, with the item as the model wrote it — the
    /// `duplicates`/`dropped` counters are these records' lengths by
    /// reason.
    pub(super) losses: Vec<Loss>,
}

/// One item `merge` did not keep (#786): what it was, why, which
/// output it came from, and — for a duplicate — which output's copy
/// was kept instead. `item` is the model's item verbatim (the parsed
/// struct re-serialized: absent fields are `null`).
#[cfg_attr(test, derive(Debug))]
pub(super) struct Loss {
    /// `association` | `alias` | `question`.
    pub(super) kind: &'static str,
    /// `dropped` (the contract refuses it as written) | `duplicate`
    /// (an identical item was already kept).
    pub(super) reason: &'static str,
    /// The rule, in the report's vocabulary.
    pub(super) rule: String,
    pub(super) item: serde_json::Value,
    pub(super) origin: usize,
    pub(super) kept_origin: Option<usize>,
    /// The paragraph the item cited, when it cited a valid one — the
    /// trace's key to the original text.
    pub(super) paragraph: Option<u32>,
}

/// ADR 0023 §3.1: a batch item's identity is its content, exactly as
/// the batch makes it unique — `merge` folds associations on the
/// triple, aliases on their spelling within a namespace, and questions
/// on (paragraph, text).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ItemKey {
    Association {
        subject: String,
        label: String,
        object: String,
    },
    Concept(String),
    Label(String),
    Question(u32, String),
}

pub(super) struct Fact {
    pub(super) subject: String,
    pub(super) label: String,
    pub(super) object: String,
    pub(super) weight: f64,
    /// The position, in `merge`'s input list, of the output this fact
    /// came from — NOT a chunk index: after the split rung one chunk
    /// yields several outputs (ADR 0023 §2 names the old field's
    /// misnomer). The trace file maps it back to `chunk_index` and
    /// `piece_id`.
    pub(super) origin: usize,
    pub(super) paragraph: Option<u32>,
}

impl Extraction {
    /// The relation spellings this document settled on, with how many
    /// associations (plus alias canonicals) used each one — issue
    /// #759's reuse signal: a label many associations already share is
    /// a safe bet to reuse, one that shows up once might be noise
    /// (a bare particle, a one-off paraphrase) not worth reinforcing.
    /// ADR 0009 §6.3 exclusion 2: `schema:type` never enters this
    /// vocabulary — `extract` has no notion of whether the target
    /// context even has a schema, so unlike the server-side exclusions
    /// (gated on "a schema document exists") this one is unconditional,
    /// the same way a producer never needs to know a server-side
    /// reserved id exists to simply never coin one. Filtering here,
    /// rather than where `system_prompt` emits its vocabulary block,
    /// covers both places this set accumulates — live extraction
    /// output and `absorb_vocabulary`'s reread of past batch files —
    /// with one line instead of two.
    pub(super) fn label_usage_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for label in self
            .associations
            .iter()
            .map(|fact| &fact.label)
            .chain(self.labels.values())
            .filter(|label| label.as_str() != crate::schema::SCHEMA_TYPE_LABEL)
        {
            *counts.entry(label.clone()).or_insert(0) += 1;
        }
        counts
    }
}

/// Issue #199 Stage 2: the alias judgments that can only be made
/// against the FULL merged name set, never one output alone — a
/// chunk-1 alias whose canonical only shows up in chunk 3 is valid
/// (see `merge`'s own comment on this below), so validating aliases
/// output-by-output would reject something `merge` happily accepts.
/// Since ADR 0013 this returns only the CORRECTIVE half of that
/// judgment (shadowing, conflicting mappings); a dangling canonical is
/// the mechanical half, removed by `prune_unresolvable_aliases`.
/// Called only once Stage 1 (`interpret_model_output`'s own issues) is
/// clean for every output, so every alias here already has a
/// well-formed, non-self `alias`/`canonical`/`kind` to judge — items
/// Stage 1 already flagged are skipped rather than re-flagged.
/// Returns one entry per output INDEX (position in `outputs`, matching
/// `ChunkOutput`'s own array position after Stage 1 — not the
/// original document chunk) that contributed at least one issue, in
/// output order, so the caller can address a single targeted
/// corrective turn per offending output.
pub(super) fn cross_output_issues(outputs: &[ChunkOutput]) -> Vec<(usize, Vec<String>)> {
    let (concept_names, label_names) = association_name_sets(outputs);

    // First-registered spelling → canonical wins, exactly like merge()'s
    // Entry::Vacant/Entry::Occupied fold — a later output naming the
    // same spelling with a DIFFERENT canonical is the conflict, not the
    // first one to claim it.
    let mut concept_registry: BTreeMap<String, String> = BTreeMap::new();
    let mut label_registry: BTreeMap<String, String> = BTreeMap::new();
    let mut issues_by_output: Vec<(usize, Vec<String>)> = Vec::new();

    for (output_index, chunk) in outputs.iter().enumerate() {
        let mut issues = Vec::new();
        for (alias_index, alias) in chunk.output.aliases.iter().enumerate() {
            let path = format!("aliases[{alias_index}]");
            let (Some(spelling), Some(canonical), Some(kind)) =
                (&alias.alias, &alias.canonical, &alias.kind)
            else {
                continue; // Stage 1 already has an issue for this alias
            };
            if spelling == canonical {
                continue; // Stage 1's self-alias issue already covers this
            }
            let (names, registry) = match kind.as_str() {
                "concept" => (&concept_names, &mut concept_registry),
                "label" => (&label_names, &mut label_registry),
                _ => continue, // Stage 1's invalid-kind issue already covers this
            };
            if names.contains(spelling) {
                issues.push(format!(
                    "{path}.alias: names something the associations already contain"
                ));
                continue;
            }
            if !names.contains(canonical) {
                // ADR 0013: a dangling canonical is no longer a
                // corrective issue — it cannot import (merge() drops
                // it), so `prune_unresolvable_aliases` removes it with
                // accounting after any corrective turns complete. It
                // still never registers a mapping here.
                continue;
            }
            match registry.get(spelling) {
                None => {
                    registry.insert(spelling.clone(), canonical.clone());
                }
                Some(existing) if existing == canonical => {
                    // A repeated identical mapping is merge()'s
                    // duplicate fold, not a conflict.
                }
                Some(existing) => {
                    issues.push(format!(
                        "{path}: conflicts with an earlier alias mapping {spelling:?} to {existing:?}"
                    ));
                }
            }
        }
        if !issues.is_empty() {
            issues_by_output.push((output_index, issues));
        }
    }
    issues_by_output
}

/// The merged name sets every cross-output judgment is made against:
/// every non-empty subject/object spelling (concepts) and label
/// spelling (labels) across ALL outputs — a chunk-1 alias whose
/// canonical only shows up in chunk 3 resolves. Shared by
/// [`cross_output_issues`] and `prune_unresolvable_aliases` so the two
/// halves of Stage 2 (corrective vs. mechanical, ADR 0013) can never
/// disagree on what "the associations contain."
pub(super) fn association_name_sets(outputs: &[ChunkOutput]) -> (HashSet<String>, HashSet<String>) {
    let mut concept_names: HashSet<String> = HashSet::new();
    let mut label_names: HashSet<String> = HashSet::new();
    for chunk in outputs {
        for item in &chunk.output.associations {
            let subject = item.subject.as_deref().unwrap_or_default().trim();
            let label = item.label.as_deref().unwrap_or_default().trim();
            let object = item.object.as_deref().unwrap_or_default().trim();
            if !subject.is_empty() {
                concept_names.insert(subject.to_string());
            }
            if !object.is_empty() {
                concept_names.insert(object.to_string());
            }
            if !label.is_empty() {
                label_names.insert(label.to_string());
            }
        }
    }
    (concept_names, label_names)
}

/// ADR 0009 §11.2: the schema-side sibling of [`cross_output_issues`]
/// — identical two-pass, per-output-index shape. `schema:type`
/// assertions across every output are unioned FIRST (a type asserted
/// in output 3 licenses a fact in output 1), then every fact
/// association is judged against the completed set — the
/// producer-side mirror of `schema_issues`/`SchemaEnv`
/// (`src/schema/check.rs`), the one function every live write entrance
/// already shares. Producer-side, so there is no live graph to merge
/// into: the union is scoped to what THIS answer set asserts, nothing
/// already stored — and there is no alias resolution to do either,
/// since [`cross_output_issues`] already refuses any alias whose
/// spelling names something the associations already contain, so every
/// subject/object/label spelling reaching here is already this
/// answer's own canonical.
pub(super) fn schema_output_issues(
    outputs: &[ChunkOutput],
    schema: &crate::schema::InstalledSchema,
) -> Vec<(usize, Vec<String>)> {
    let document = schema.document();

    // Pass 1: union every output's own type assertions before judging
    // any one of them.
    let mut asserted: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for chunk in outputs {
        for item in &chunk.output.associations {
            let subject = item.subject.as_deref().unwrap_or_default().trim();
            let label = item.label.as_deref().unwrap_or_default().trim();
            let object = item.object.as_deref().unwrap_or_default().trim();
            if label == crate::schema::SCHEMA_TYPE_LABEL
                && !subject.is_empty()
                && !object.is_empty()
            {
                asserted
                    .entry(subject.to_string())
                    .or_default()
                    .insert(object.to_string());
            }
        }
    }
    let types: BTreeMap<String, SchemaTypeAssertions> = asserted
        .into_iter()
        .map(|(concept, names)| {
            let mut expanded = BTreeSet::new();
            for name in &names {
                expanded.extend(schema.closure_of(name));
            }
            (
                concept,
                SchemaTypeAssertions {
                    asserted: names,
                    expanded,
                },
            )
        })
        .collect();

    let mut issues_by_output: Vec<(usize, Vec<String>)> = Vec::new();
    for (output_index, chunk) in outputs.iter().enumerate() {
        let mut issues = Vec::new();

        // Guard 2 (ADR 0009 §6.3), mode-independent: this answer's own
        // alias declarations must never name the reserved label as a
        // canonical — mirrors `schema_issues`' `reserved` list
        // (`src/schema/check.rs:411`), the one that refuses regardless
        // of mode.
        for (alias_index, alias) in chunk.output.aliases.iter().enumerate() {
            if alias.kind.as_deref() == Some("label")
                && alias.canonical.as_deref() == Some(crate::schema::SCHEMA_TYPE_LABEL)
            {
                issues.push(format!(
                    "aliases[{alias_index}].canonical: '{}' is reserved for type assertions \
                     (ADR 0009 §6.3) — rename the alias",
                    crate::schema::SCHEMA_TYPE_LABEL
                ));
            }
        }

        if document.mode != crate::schema::SchemaMode::Off {
            for (assoc_index, item) in chunk.output.associations.iter().enumerate() {
                let subject = item.subject.as_deref().unwrap_or_default().trim();
                let label = item.label.as_deref().unwrap_or_default().trim();
                let object = item.object.as_deref().unwrap_or_default().trim();
                if label.is_empty() || label == crate::schema::SCHEMA_TYPE_LABEL {
                    continue; // type ops are never judged, §7.2 step 6
                }
                match document.relations.get(label) {
                    Some(relation) => {
                        if schema_side_violates(&relation.domain, types.get(subject)) {
                            issues.push(format!(
                                "associations[{assoc_index}].subject: {}",
                                schema_violation_text(
                                    subject,
                                    &relation.domain,
                                    label,
                                    types.get(subject)
                                ),
                            ));
                        }
                        if schema_side_violates(&relation.range, types.get(object)) {
                            issues.push(format!(
                                "associations[{assoc_index}].object: {}",
                                schema_violation_text(
                                    object,
                                    &relation.range,
                                    label,
                                    types.get(object)
                                ),
                            ));
                        }
                    }
                    None if document.closed_labels => {
                        issues.push(format!(
                            "associations[{assoc_index}].label: '{label}' names no relation \
                             declared in this context's schema (closed_labels)"
                        ));
                    }
                    None => {}
                }
            }
        }

        if !issues.is_empty() {
            issues_by_output.push((output_index, issues));
        }
    }
    issues_by_output
}

/// One concept's asserted type population within one answer set: what
/// was actually asserted (`asserted`) and its `is_a` closure
/// (`expanded`) — [`schema_side_violates`] only ever tests the
/// closure, but the corrective message names what was asserted, never
/// the expansion. Mirrors `src/schema/check.rs`'s own
/// `TypeAssertions`/`actual_types` split and its stated reason: "the
/// expanded set's size is a schema-authoring accident, not information
/// the corrector needs."
pub(super) struct SchemaTypeAssertions {
    pub(super) asserted: BTreeSet<String>,
    pub(super) expanded: BTreeSet<String>,
}

/// A `domain`/`range` violation is `declared` non-empty (an
/// unconstrained side never violates), the concept's expanded type set
/// non-empty (untyped never violates, §6.1), and the two disjoint —
/// the producer-side mirror of `src/schema/check.rs`'s `side_violates`.
pub(super) fn schema_side_violates(
    declared: &BTreeSet<String>,
    types: Option<&SchemaTypeAssertions>,
) -> bool {
    if declared.is_empty() {
        return false;
    }
    let Some(types) = types else {
        return false;
    };
    if types.expanded.is_empty() {
        return false;
    }
    declared.is_disjoint(&types.expanded)
}

/// Cap on how many type names one violation message enumerates — same
/// bound and same reasoning as `src/schema/check.rs`'s own
/// `MAX_ISSUE_TYPE_NAMES`: a relation's declared `domain`/`range` is
/// itself capped, but a concept's asserted type set is not.
pub(super) const MAX_ISSUE_TYPE_NAMES: usize = 8;

pub(super) fn join_capped_names<'a>(names: impl Iterator<Item = &'a String>) -> String {
    let names: Vec<&str> = names.map(String::as_str).collect();
    if names.len() <= MAX_ISSUE_TYPE_NAMES {
        names.join(", ")
    } else {
        format!(
            "{}, … (+{} more)",
            names[..MAX_ISSUE_TYPE_NAMES].join(", "),
            names.len() - MAX_ISSUE_TYPE_NAMES
        )
    }
}

/// One violation's corrective text: what was expected (the declared
/// set as written, never the `is_a` closure) and what the concept was
/// actually asserted as (never the closure either) — same "expected"/
/// "actual" split `src/schema/check.rs`'s `expected_types`/
/// `actual_types` use for the API-facing `Issue`, restated here as one
/// sentence for a corrective LLM turn instead of two structured fields.
pub(super) fn schema_violation_text(
    name: &str,
    declared: &BTreeSet<String>,
    relation: &str,
    types: Option<&SchemaTypeAssertions>,
) -> String {
    let expected = join_capped_names(declared.iter());
    let actual = match types {
        Some(types) => join_capped_names(types.asserted.iter()),
        None => String::new(),
    };
    format!(
        "'{name}' must be typed as one of [{expected}] (or a subtype, via a schema:type \
         assertion) for relation '{relation}', but is typed as [{actual}]"
    )
}

/// Issue #199 Stage 2's cross-output judgment, widened to a schema
/// document when one is installed (ADR 0009 §11): [`cross_output_issues`]'s
/// alias judgment and [`schema_output_issues`]'s domain/range judgment
/// are two independent per-output-index issue lists, merged into one
/// before a single corrective turn is spent per offending output —
/// exactly as if one combined check had produced them. Output order
/// in, output order out.
pub(super) fn combined_cross_output_issues(
    outputs: &[ChunkOutput],
    schema: Option<&crate::schema::InstalledSchema>,
) -> Vec<(usize, Vec<String>)> {
    let mut merged: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (index, issues) in cross_output_issues(outputs) {
        merged.entry(index).or_default().extend(issues);
    }
    if let Some(schema) = schema {
        for (index, issues) in schema_output_issues(outputs, schema) {
            merged.entry(index).or_default().extend(issues);
        }
    }
    merged.into_iter().collect()
}

/// `questions_cap` is this run's --questions N (0 = the model was
/// never asked; whatever it volunteers drops); `paragraph_count` is
/// the document's CANONICAL split size — the numbers the prompt showed
/// and the server will validate against.
pub(super) fn merge(
    outputs: Vec<ModelOutput>,
    questions_cap: usize,
    paragraph_count: usize,
) -> Extraction {
    let mut extraction = Extraction {
        associations: Vec::new(),
        concepts: BTreeMap::new(),
        labels: BTreeMap::new(),
        questions: Vec::new(),
        duplicates: 0,
        dropped: 0,
        origins: BTreeMap::new(),
        losses: Vec::new(),
    };
    let to_value = |item: &dyn erased::Serialize| item.to_value();
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut seen_questions: HashSet<(u32, String)> = HashSet::new();
    let mut per_paragraph: BTreeMap<u32, usize> = BTreeMap::new();
    let mut aliases: Vec<(usize, ModelAlias)> = Vec::new();
    for (origin, output) in outputs.into_iter().enumerate() {
        for item in output.questions {
            let paragraph = item.paragraph;
            let question = item.question.clone().unwrap_or_default();
            let question = question.trim();
            let shape_ok = paragraph
                .is_some_and(|paragraph| (paragraph as usize) < paragraph_count)
                && !question.is_empty()
                && question.len() <= crate::api::MAX_QUESTION_BYTES
                && questions_cap > 0;
            let Some(paragraph) = paragraph.filter(|_| shape_ok) else {
                extraction.dropped += 1;
                extraction.losses.push(Loss {
                    kind: "question",
                    reason: "dropped",
                    rule: if questions_cap == 0 {
                        "questions were not requested (--questions 0)".to_string()
                    } else {
                        "paragraph missing or out of range, or question empty or over \
                         the size cap"
                            .to_string()
                    },
                    item: to_value(&item),
                    origin,
                    kept_origin: None,
                    paragraph: item
                        .paragraph
                        .filter(|&paragraph| (paragraph as usize) < paragraph_count),
                });
                continue;
            };
            let question_key = (paragraph, question.to_string());
            if seen_questions.contains(&question_key) {
                extraction.duplicates += 1;
                extraction.losses.push(Loss {
                    kind: "question",
                    reason: "duplicate",
                    rule: "the same question for the same paragraph was already kept".to_string(),
                    item: to_value(&item),
                    origin,
                    kept_origin: extraction
                        .origins
                        .get(&ItemKey::Question(paragraph, question.to_string()))
                        .copied(),
                    paragraph: Some(paragraph),
                });
                continue;
            }
            let count = per_paragraph.entry(paragraph).or_insert(0);
            if *count >= questions_cap {
                extraction.dropped += 1;
                extraction.losses.push(Loss {
                    kind: "question",
                    reason: "dropped",
                    rule: format!("over the --questions cap of {questions_cap} for the paragraph"),
                    item: to_value(&item),
                    origin,
                    kept_origin: None,
                    paragraph: Some(paragraph),
                });
                continue;
            }
            // Only register with seen_questions once the item is actually
            // kept: inserting it before the cap check would make a
            // cap-dropped question read as a *duplicate* the next time an
            // identical one arrives (from another chunk re-proposing it),
            // permanently mislabeling a paragraph's overflow as
            // deduplication instead of the cap that caused it.
            *count += 1;
            seen_questions.insert(question_key.clone());
            extraction.origins.insert(
                ItemKey::Question(question_key.0, question_key.1.clone()),
                origin,
            );
            extraction.questions.push(question_key);
        }
        for item in output.associations {
            // Absent and null both read as empty; an omitted weight is
            // a plain assertion. Trim before anything else and carry the
            // trimmed form onward — the graph's normalization does NOT
            // fold whitespace, so " apple" and "apple" would split into
            // two concept nodes, and the dedup key below would miss the
            // duplicate. The question path above trims the same way.
            let subject = item.subject.clone().unwrap_or_default();
            let subject = subject.trim();
            let label = item.label.clone().unwrap_or_default();
            let label = label.trim();
            let object = item.object.clone().unwrap_or_default();
            let object = object.trim();
            let weight = item.weight.unwrap_or(1.0);
            let names_ok = [subject, label, object]
                .iter()
                .all(|text| !text.is_empty() && text.len() <= MAX_NAME_BYTES);
            // A zero weight asserts nothing; refusing it here beats
            // shipping a fact the graph treats as absent.
            let cited = item
                .paragraph
                .filter(|&paragraph| (paragraph as usize) < paragraph_count);
            if !names_ok
                || !weight.is_finite()
                || weight == 0.0
                || weight.abs() > MAX_ASSOCIATION_WEIGHT
            {
                extraction.dropped += 1;
                extraction.losses.push(Loss {
                    kind: "association",
                    reason: "dropped",
                    rule: if names_ok {
                        format!(
                            "weight must be finite, non-zero, and at most \
                             {MAX_ASSOCIATION_WEIGHT} in magnitude"
                        )
                    } else {
                        format!(
                            "subject, label, and object must each be non-empty and at most \
                             {MAX_NAME_BYTES} bytes"
                        )
                    },
                    item: to_value(&item),
                    origin,
                    kept_origin: None,
                    paragraph: cited,
                });
                continue;
            }
            let key = (subject.to_string(), label.to_string(), object.to_string());
            if !seen.insert(key) {
                extraction.duplicates += 1;
                extraction.losses.push(Loss {
                    kind: "association",
                    reason: "duplicate",
                    rule: "the same subject/label/object triple was already kept".to_string(),
                    item: to_value(&item),
                    origin,
                    kept_origin: extraction
                        .origins
                        .get(&ItemKey::Association {
                            subject: subject.to_string(),
                            label: label.to_string(),
                            object: object.to_string(),
                        })
                        .copied(),
                    paragraph: cited,
                });
                continue;
            }
            // A missing or out-of-range self-report costs only the
            // paragraph tag, never the fact — the item still carries
            // the model's judgment about subject/label/object/weight.
            let paragraph = cited;
            extraction.origins.insert(
                ItemKey::Association {
                    subject: subject.to_string(),
                    label: label.to_string(),
                    object: object.to_string(),
                },
                origin,
            );
            extraction.associations.push(Fact {
                subject: subject.to_string(),
                label: label.to_string(),
                object: object.to_string(),
                weight,
                origin,
                paragraph,
            });
        }
        aliases.extend(output.aliases.into_iter().map(|alias| (origin, alias)));
    }

    // Aliases check against the MERGED associations, so a chunk-1
    // alias whose canonical only shows up in chunk 3 still lands.
    let mut concept_names: HashSet<&str> = HashSet::new();
    let mut label_names: HashSet<&str> = HashSet::new();
    for fact in &extraction.associations {
        concept_names.insert(&fact.subject);
        concept_names.insert(&fact.object);
        label_names.insert(&fact.label);
    }
    for (origin, alias) in aliases {
        // Trim to match the association names in `concept_names` /
        // `label_names`, which are the trimmed subject/label/object
        // above; an untrimmed spelling or canonical would miss the
        // `names.contains` checks and split the stored alias.
        let spelling = alias.alias.clone().unwrap_or_default();
        let spelling = spelling.trim();
        let canonical = alias.canonical.clone().unwrap_or_default();
        let canonical = canonical.trim();
        let lost = |reason: &'static str, rule: String, kept_origin: Option<usize>| Loss {
            kind: "alias",
            reason,
            rule,
            item: to_value(&alias),
            origin,
            kept_origin,
            paragraph: None,
        };
        let (namespace, names, key) = match alias.kind.as_deref() {
            Some("concept") => (
                &mut extraction.concepts,
                &concept_names,
                ItemKey::Concept(spelling.to_string()),
            ),
            Some("label") => (
                &mut extraction.labels,
                &label_names,
                ItemKey::Label(spelling.to_string()),
            ),
            _ => {
                extraction.dropped += 1;
                extraction.losses.push(lost(
                    "dropped",
                    "kind must be \"concept\" or \"label\"".to_string(),
                    None,
                ));
                continue;
            }
        };
        let shape_ok = !spelling.is_empty()
            && spelling.len() <= MAX_NAME_BYTES
            && canonical.len() <= MAX_NAME_BYTES
            && spelling != canonical;
        // An alias spelling that is itself a name would shadow a real
        // record — the registry refuses that as a conflict, so it
        // never leaves here.
        if !shape_ok || !names.contains(canonical) || names.contains(spelling) {
            extraction.dropped += 1;
            let rule = if !shape_ok {
                format!(
                    "alias must be non-empty, differ from its canonical, and both at most \
                     {MAX_NAME_BYTES} bytes"
                )
            } else if !names.contains(canonical) {
                "canonical names nothing the associations contain".to_string()
            } else {
                "alias is itself a name the associations contain".to_string()
            };
            extraction.losses.push(lost("dropped", rule, None));
            continue;
        }
        match namespace.entry(spelling.to_string()) {
            Entry::Vacant(vacant) => {
                vacant.insert(canonical.to_string());
                extraction.origins.insert(key, origin);
            }
            Entry::Occupied(existing) => {
                let kept_origin = extraction.origins.get(&key).copied();
                if existing.get().as_str() == canonical {
                    extraction.duplicates += 1;
                    extraction.losses.push(lost(
                        "duplicate",
                        "the same alias mapping was already kept".to_string(),
                        kept_origin,
                    ));
                } else {
                    extraction.dropped += 1;
                    extraction.losses.push(lost(
                        "dropped",
                        "alias already maps to a different canonical in this document".to_string(),
                        kept_origin,
                    ));
                }
            }
        }
    }
    extraction
}

/// An object-safe serialize seam so one closure in `merge` turns any
/// of the three model item structs into its JSON value.
mod erased {
    pub(super) trait Serialize {
        fn to_value(&self) -> serde_json::Value;
    }
    impl<T: serde::Serialize> Serialize for T {
        fn to_value(&self) -> serde_json::Value {
            serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
        }
    }
}
