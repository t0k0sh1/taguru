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
}

pub(super) struct Fact {
    pub(super) subject: String,
    pub(super) label: String,
    pub(super) object: String,
    pub(super) weight: f64,
    #[allow(dead_code)] // read by tests only — a follow-up issue surfaces it beyond merge()
    pub(super) chunk_index: usize,
    pub(super) paragraph: Option<u32>,
}

impl Extraction {
    /// The relation spellings this document settled on. ADR 0009 §6.3
    /// exclusion 2: `schema:type` never enters this vocabulary — `extract`
    /// has no notion of whether the target context even has a schema, so
    /// unlike the server-side exclusions (gated on "a schema document
    /// exists") this one is unconditional, the same way a producer never
    /// needs to know a server-side reserved id exists to simply never
    /// coin one. Filtering here, rather than where `system_prompt` emits
    /// its vocabulary block, covers both places this set accumulates —
    /// live extraction output and `absorb_vocabulary`'s reread of past
    /// batch files — with one line instead of two.
    pub(super) fn label_vocabulary(&self) -> BTreeSet<String> {
        self.associations
            .iter()
            .map(|fact| fact.label.clone())
            .chain(self.labels.values().cloned())
            .filter(|label| label != crate::schema::SCHEMA_TYPE_LABEL)
            .collect()
    }
}

/// Issue #199 Stage 2: the alias judgments that can only be made
/// against the FULL merged name set, never one output alone — a
/// chunk-1 alias whose canonical only shows up in chunk 3 is valid
/// (see `merge`'s own comment on this below), so validating aliases
/// output-by-output would reject something `merge` happily accepts.
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
    let mut concept_names: HashSet<String> = HashSet::new();
    let mut label_names: HashSet<String> = HashSet::new();
    for chunk in outputs {
        let output = &chunk.output;
        for item in &output.associations {
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
                issues.push(format!(
                    "{path}.canonical: names nothing the associations contain"
                ));
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
    };
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut seen_questions: HashSet<(u32, String)> = HashSet::new();
    let mut per_paragraph: BTreeMap<u32, usize> = BTreeMap::new();
    let mut aliases: Vec<ModelAlias> = Vec::new();
    for (chunk_index, output) in outputs.into_iter().enumerate() {
        for item in output.questions {
            let paragraph = item.paragraph;
            let question = item.question.unwrap_or_default();
            let question = question.trim();
            let shape_ok = paragraph
                .is_some_and(|paragraph| (paragraph as usize) < paragraph_count)
                && !question.is_empty()
                && question.len() <= crate::api::MAX_QUESTION_BYTES
                && questions_cap > 0;
            let Some(paragraph) = paragraph.filter(|_| shape_ok) else {
                extraction.dropped += 1;
                continue;
            };
            let question_key = (paragraph, question.to_string());
            if seen_questions.contains(&question_key) {
                extraction.duplicates += 1;
                continue;
            }
            let count = per_paragraph.entry(paragraph).or_insert(0);
            if *count >= questions_cap {
                extraction.dropped += 1;
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
            extraction.questions.push(question_key);
        }
        for item in output.associations {
            // Absent and null both read as empty; an omitted weight is
            // a plain assertion. Trim before anything else and carry the
            // trimmed form onward — the graph's normalization does NOT
            // fold whitespace, so " apple" and "apple" would split into
            // two concept nodes, and the dedup key below would miss the
            // duplicate. The question path above trims the same way.
            let subject = item.subject.unwrap_or_default();
            let subject = subject.trim();
            let label = item.label.unwrap_or_default();
            let label = label.trim();
            let object = item.object.unwrap_or_default();
            let object = object.trim();
            let weight = item.weight.unwrap_or(1.0);
            let names_ok = [subject, label, object]
                .iter()
                .all(|text| !text.is_empty() && text.len() <= MAX_NAME_BYTES);
            // A zero weight asserts nothing; refusing it here beats
            // shipping a fact the graph treats as absent.
            if !names_ok
                || !weight.is_finite()
                || weight == 0.0
                || weight.abs() > MAX_ASSOCIATION_WEIGHT
            {
                extraction.dropped += 1;
                continue;
            }
            let key = (subject.to_string(), label.to_string(), object.to_string());
            if !seen.insert(key) {
                extraction.duplicates += 1;
                continue;
            }
            // A missing or out-of-range self-report costs only the
            // paragraph tag, never the fact — the item still carries
            // the model's judgment about subject/label/object/weight.
            let paragraph = item
                .paragraph
                .filter(|&paragraph| (paragraph as usize) < paragraph_count);
            extraction.associations.push(Fact {
                subject: subject.to_string(),
                label: label.to_string(),
                object: object.to_string(),
                weight,
                chunk_index,
                paragraph,
            });
        }
        aliases.extend(output.aliases);
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
    for alias in aliases {
        // Trim to match the association names in `concept_names` /
        // `label_names`, which are the trimmed subject/label/object
        // above; an untrimmed spelling or canonical would miss the
        // `names.contains` checks and split the stored alias.
        let spelling = alias.alias.unwrap_or_default();
        let spelling = spelling.trim();
        let canonical = alias.canonical.unwrap_or_default();
        let canonical = canonical.trim();
        let (namespace, names) = match alias.kind.as_deref() {
            Some("concept") => (&mut extraction.concepts, &concept_names),
            Some("label") => (&mut extraction.labels, &label_names),
            _ => {
                extraction.dropped += 1;
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
            continue;
        }
        match namespace.entry(spelling.to_string()) {
            Entry::Vacant(vacant) => {
                vacant.insert(canonical.to_string());
            }
            Entry::Occupied(existing) => {
                if existing.get().as_str() == canonical {
                    extraction.duplicates += 1;
                } else {
                    extraction.dropped += 1;
                }
            }
        }
    }
    extraction
}
