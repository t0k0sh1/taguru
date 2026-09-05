//! The remote apply path: `taguru import --url` — validates every file
//! the same way [`super::local::run_local`] does, then packs whole
//! batches (and, after them, whole group records) into byte-budgeted
//! chunks POSTed to a running server's `/import` (ADR 0002 §9).

use super::*;

/// The starting byte budget `import --url` packs each chunk up to
/// (ADR 0002 §9) — half the server's default body cap, leaving
/// headroom for a server configured lower. A 413 still hit at this
/// budget halves it further (never below one unit) rather than
/// retrying at the same size.
const REMOTE_IMPORT_BUDGET_BYTES: usize = DEFAULT_MAX_BODY_BYTES / 2;

/// Which record kind a [`Unit`] carries — [`run_remote`]'s
/// "never sent" tallies on a mid-stream refusal count each kind
/// separately, group records included.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum UnitKind {
    Batch,
    Schema,
    Group,
}

/// One batch's, schema's, or group's rendered bytes, packed into a
/// [`Chunk`] for `import --url`'s wire chunking (ADR 0002 §9) —
/// `label` names the source (or context, or group) for the hard-error
/// and progress messages.
pub(super) struct Unit {
    pub(super) text: String,
    pub(super) label: String,
    pub(super) kind: UnitKind,
}

impl Unit {
    fn len(&self) -> usize {
        self.text.len()
    }
}

/// The server's `issues[]` on a refusal, each re-addressed to the unit
/// it names (#863): the server's path is request-relative
/// (`batches[3].associations[7].subject`, `src/api/import.rs`), and
/// `batches[3]` is the chunk's fourth BATCH unit — whose label names the
/// file, context, and source — so the operator reads a file and a
/// line-addressable item, not a request index. A path without a
/// `batches[N]` prefix (a schema record's, say) is printed as sent.
/// `issues_total` past the listed ones (the server caps the list) ends
/// the lines with the remainder, so the count the server knew is never
/// lost between the two sides.
pub(super) fn refusal_issue_lines(body: &Value, units: &[Unit]) -> Vec<String> {
    let issues = body.get("issues").and_then(Value::as_array);
    let Some(issues) = issues else {
        return Vec::new();
    };
    let batch_labels: Vec<&str> = units
        .iter()
        .filter(|unit| unit.kind == UnitKind::Batch)
        .map(|unit| unit.label.as_str())
        .collect();
    let mut lines: Vec<String> = issues
        .iter()
        .map(|issue| {
            let path = issue.get("path").and_then(Value::as_str).unwrap_or("?");
            let expected = issue.get("expected").and_then(Value::as_str).unwrap_or("?");
            let actual = issue.get("actual").and_then(Value::as_str).unwrap_or("?");
            let addressed = batch_index_prefix(path)
                .and_then(|(index, rest)| {
                    batch_labels.get(index).map(|label| {
                        if rest.is_empty() {
                            (*label).to_string()
                        } else {
                            format!("{label}: {rest}")
                        }
                    })
                })
                .unwrap_or_else(|| path.to_string());
            format!("{addressed}: expected {expected}, got {actual}")
        })
        .collect();
    let total = body
        .get("issues_total")
        .and_then(Value::as_u64)
        .map(|total| total as usize)
        .unwrap_or(issues.len());
    let remainder = total.saturating_sub(issues.len());
    if remainder > 0 {
        lines.push(format!(
            "… and {remainder} more issue(s) the server did not list — fix the listed ones \
             and resend; the rest are named on the next refusal"
        ));
    }
    lines
}

/// The never-sent tally after a mid-stream stop — one line per unit
/// kind still queued, with the count and the first such unit's label
/// as the resume point (#863). All three record kinds a chunk can
/// carry, group records included: a refusal or a lost connection
/// mid-stream leaves the queued groups exactly as unsent as the
/// batches and schemas beside them. Empty when nothing is queued.
pub(super) fn never_sent_lines(queue: &VecDeque<Chunk>) -> Vec<String> {
    let mut lines = Vec::new();
    for (kind, what) in [
        (UnitKind::Batch, "batch(es)"),
        (UnitKind::Schema, "schema record(s)"),
        (UnitKind::Group, "group record(s)"),
    ] {
        let mut unsent = queue
            .iter()
            .flat_map(|chunk| &chunk.units)
            .filter(|unit| unit.kind == kind);
        if let Some(first) = unsent.next() {
            let count = 1 + unsent.count();
            lines.push(format!(
                "{count} {what} after this chunk were never sent, from {}",
                first.label
            ));
        }
    }
    lines
}

/// `batches[N]` and what follows its closing bracket (a leading `.`
/// dropped), or `None` for any other path shape.
fn batch_index_prefix(path: &str) -> Option<(usize, &str)> {
    let inner = path.strip_prefix("batches[")?;
    let close = inner.find(']')?;
    let index: usize = inner[..close].parse().ok()?;
    let rest = inner[close + 1..].trim_start_matches('.');
    Some((index, rest))
}

/// One `POST /import` request's worth of units, in stream order — a
/// prefix of whole batch units followed (only in the last chunk that
/// carries any) by whole group units, since groups restore after
/// every batch of the run.
pub(super) struct Chunk {
    pub(super) units: Vec<Unit>,
}

impl Chunk {
    /// What this chunk carries, for a refusal or a lost connection to
    /// name (#863): a chunk is the packer's unit, not the operator's,
    /// so the line says which files' batches it holds — the one unit,
    /// or the first and the last with the count between.
    pub(super) fn carried(&self) -> String {
        match self.units.as_slice() {
            [] => "no unit".to_string(),
            [only] => format!("1 unit: {}", only.label),
            [first, .., last] => format!(
                "{} units, from {} through {}",
                self.units.len(),
                first.label,
                last.label
            ),
        }
    }

    pub(super) fn size(&self) -> usize {
        self.units.iter().map(Unit::len).sum()
    }

    /// The wire body: every unit's text concatenated, each guaranteed
    /// to end in its own newline — a unit sliced off the last batch of
    /// a file (or EOF) may not already carry one.
    pub(super) fn body(&self) -> String {
        let mut body = String::with_capacity(self.size() + self.units.len());
        for unit in &self.units {
            body.push_str(&unit.text);
            if !unit.text.ends_with('\n') {
                body.push('\n');
            }
        }
        body
    }

    /// Splits this chunk in two at the unit boundary closest to half
    /// its accumulated bytes, both halves non-empty — the ADR 0002 §9
    /// 413 adaptation, and the same halving the pre-send budget check
    /// below uses. Only ever called on a chunk carrying more than one
    /// unit: a lone oversized unit is refused before it ever reaches a
    /// chunk (splitting a batch's own record set client-side would
    /// break the retract-then-apply contract's atomicity boundary).
    pub(super) fn halve(mut self) -> (Chunk, Chunk) {
        debug_assert!(self.units.len() > 1, "a lone unit cannot be halved further");
        let total = self.size();
        let mut running = 0usize;
        let mut split_at = 1;
        for (index, unit) in self.units.iter().enumerate() {
            running += unit.len();
            if running * 2 >= total {
                split_at = (index + 1).clamp(1, self.units.len() - 1);
                break;
            }
        }
        let tail = self.units.split_off(split_at);
        (Chunk { units: self.units }, Chunk { units: tail })
    }
}

/// Greedily packs units, in order, into whole-unit chunks under
/// `budget` bytes — a chunk always carries at least one unit even if
/// that unit alone exceeds `budget` (the caller refuses that case
/// before this ever runs; see [`run_remote`]'s pre-send check). Pulled
/// out of [`run_remote`] so the packing rule has a unit test
/// independent of any network call.
pub(super) fn pack_chunks(units: Vec<Unit>, budget: usize) -> VecDeque<Chunk> {
    let mut queue: VecDeque<Chunk> = VecDeque::new();
    let mut pending: Vec<Unit> = Vec::new();
    let mut pending_size = 0usize;
    for unit in units {
        if !pending.is_empty() && pending_size + unit.len() > budget {
            queue.push_back(Chunk {
                units: std::mem::take(&mut pending),
            });
            pending_size = 0;
        }
        pending_size += unit.len();
        pending.push(unit);
    }
    if !pending.is_empty() {
        queue.push_back(Chunk { units: pending });
    }
    queue
}

/// The hard failure for a single batch (or group record) that alone
/// exceeds the byte budget — reported and refused before the network
/// is ever touched, naming the two real fixes (ADR 0002 §9).
fn oversized_unit_message(label: &str, size: usize, budget: usize) -> String {
    format!(
        "{label} alone is {size} byte(s), over the {budget}-byte chunk \
         budget — splitting a batch's own record set client-side would break the \
         retract-then-apply contract's atomicity boundary, so this cannot be packed \
         automatically; reduce what this source's batch carries (split the source \
         upstream of import) — raising the server's TAGURU_MAX_BODY_BYTES alone will \
         not help, since this budget is fixed client-side regardless of the server's cap"
    )
}

/// The hard failure for a single batch (or group record) that already
/// fit under this client's own packing budget — it passed the pre-send
/// check [`oversized_unit_message`] guards — but the SERVER still
/// answered 413 for it. Unlike that client-side refusal, this one IS
/// the server's own body-size cap, so raising `TAGURU_MAX_BODY_BYTES`
/// server-side is a real fix here, not the dead end
/// [`oversized_unit_message`]'s wording correctly rules out for its own
/// (client-fixed-budget) case.
fn server_refused_single_unit_error(label: &str, size: usize) -> i32 {
    eprintln!(
        "taguru: import: {label} ({size} byte(s)) was refused by the server as too \
         large, and cannot be split further — splitting a batch's own record set \
         client-side would break the retract-then-apply contract's atomicity \
         boundary. Raise the server's TAGURU_MAX_BODY_BYTES, or reduce what this \
         source's batch carries (split the source upstream of import)."
    );
    1
}

/// One chunk's landed `ImportOutcome` array (`POST /import`'s response
/// shape, src/api/import.rs), summarized into the same vocabulary
/// [`report`] uses for the local path's per-batch line — one line per
/// chunk instead of one per batch, since a remote chunk can carry many.
fn summarize_chunk_outcomes(outcomes: &[Value]) -> String {
    let created = outcomes
        .iter()
        .filter(|outcome| outcome["created"].as_bool() == Some(true))
        .count();
    let retracted: u64 = outcomes
        .iter()
        .filter_map(|o| o["retracted"].as_u64())
        .sum();
    let associations: u64 = outcomes
        .iter()
        .filter_map(|o| o["associations"].as_u64())
        .sum();
    let aliases: u64 = outcomes.iter().filter_map(|o| o["aliases"].as_u64()).sum();
    let passages = outcomes
        .iter()
        .filter(|outcome| outcome["passage_stored"].as_bool() == Some(true))
        .count();
    // Absent on any server old enough to predate #382, so a missing
    // field (`as_u64()` on `None` → `None`) reads as 0 the same way a
    // real schema-free batch's own count would — a remote run against
    // such a server never prints a warning line it cannot back up.
    let schema_violations: u64 = outcomes
        .iter()
        .filter_map(|o| o["schema_violations"].as_u64())
        .sum();
    format!(
        "{} batch(es){} ({retracted} association(s) retracted): +{associations} \
         association(s), +{aliases} alias(es){}{}",
        outcomes.len(),
        match created {
            0 => String::new(),
            created => format!(", {created} created"),
        },
        match passages {
            0 => String::new(),
            passages => format!(", {passages} passage(s) stored"),
        },
        match schema_violations {
            0 => String::new(),
            violations => format!(", {violations} schema warning(s)"),
        },
    )
}

/// The remote twin of [`super::local::run_local`]: validates every file
/// the same way, then packs whole batches (and, after them, whole group
/// records) into chunks under a byte budget and POSTs each to a
/// running server's `/import`, adapting to a 413 by halving the chunk
/// and resending — never splitting a batch's own record set, never
/// crossing into the next batch (ADR 0002 §9). `--dry-run` sends every
/// chunk as `?dry_run=true` instead of touching anything.
pub(super) fn run_remote(
    base: &str,
    files: &[PathBuf],
    dry_run: bool,
    as_json: bool,
    sensitive_rules: Option<&crate::sensitive::RuleSet>,
) -> i32 {
    // ADR 0002 §7: caught before any request leaves the process.
    if let Err(message) = crate::remote::reject_userinfo(base) {
        return crate::config::subcommand_usage_error("import", &message);
    }
    // A malformed `--url` is a usage problem, not a partial apply — it
    // must exit 2 (`cli.rs`'s own documented usage-error code, and
    // `evaluate --url`'s precedent) rather than fall through to
    // `Api::url`'s per-request `InvalidUrl` failure, which used to
    // surface later as an exit-1 "nothing landed" message that reads
    // like a network problem. Checked up front rather than relying on
    // the chunk loop to hit it: an input with zero batches and zero
    // groups (every batch failed the earlier owner-uniqueness check,
    // say) never enters that loop at all, so a bad URL would otherwise
    // go entirely undetected and exit 0.
    if let Err(message) = crate::remote::reject_unusable_base(base) {
        return crate::config::subcommand_usage_error("import", &message);
    }

    // Pass 1 — every file parses, or nothing applies (same contract as
    // run_local's own Pass 1). File bytes are held past this point
    // (unlike the local path's streaming parse) because split_batches
    // needs them: the chunk packer below slices each batch straight
    // out of its source file rather than re-serializing it.
    let mut units: Vec<Unit> = Vec::new();
    let mut schema_units: Vec<Unit> = Vec::new();
    let mut group_units: Vec<Unit> = Vec::new();
    let mut batch_count = 0usize;
    let mut schema_count = 0usize;
    let mut group_count = 0usize;
    let mut broken = 0usize;
    // ADR 0038 §3.4: batches the gate refused — named on stderr as
    // they are met and never packed, so nothing of them leaves the
    // process; reported under `failed_batches` with the exit code.
    let mut refused: Vec<Value> = Vec::new();
    // Keyed claims → the file that made them (#863): a later file's
    // duplicate is refused naming the earlier file, not just "an
    // earlier file".
    let mut owners: HashMap<(String, String), PathBuf> = HashMap::new();
    let mut schema_owners: HashMap<String, PathBuf> = HashMap::new();
    let mut group_owners: HashMap<String, PathBuf> = HashMap::new();
    for path in files {
        let mut bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("taguru: import: {}: {error}", path.display());
                broken += 1;
                continue;
            }
        };
        // A UTF-8 BOM only ever means anything at byte 0 of the WHOLE
        // stream (parse_stream's own note): stripped here, before
        // split_batches runs, so a batch sliced mid-stream can never
        // carry one riding into a wire chunk that isn't the first.
        if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
            bytes.drain(0..3);
        }
        let mut file_broken = false;
        match parse_stream(&bytes[..]) {
            Ok(stream) => {
                let ranges = split_batches(&bytes);
                // `split_batches` and `parse_stream` are two
                // independent scanners over the same bytes; they agree
                // today, but a `zip` below silently truncates to the
                // shorter side if they ever diverge — dropping trailing
                // batches from this remote import with no error and no
                // non-zero exit (the `--dry-run` summary's batch count
                // is computed inside this same truncated loop, so it
                // would agree with the loss and hide it completely).
                // Checked here, always — not `debug_assert_eq!`, which
                // release builds (what an operator actually runs)
                // compile out — so a divergence refuses this file
                // loudly instead of restoring a silently incomplete
                // backup.
                if ranges.len() != stream.batches.len() {
                    eprintln!(
                        "taguru: import: {}: internal error: split_batches sliced {} \
                         range(s) for {} parsed batch(es) — refusing rather than risk \
                         silently dropping batches",
                        path.display(),
                        ranges.len(),
                        stream.batches.len()
                    );
                    broken += 1;
                    continue;
                }
                for (index, (batch, range)) in stream.batches.iter().zip(ranges).enumerate() {
                    if let Some(earlier) =
                        owners.get(&(batch.context.clone(), batch.source.clone()))
                    {
                        eprintln!(
                            "taguru: import: {}: {}",
                            path.display(),
                            duplicate_source_message(&batch.context, &batch.source, earlier)
                        );
                        file_broken = true;
                        continue;
                    }
                    owners.insert((batch.context.clone(), batch.source.clone()), path.clone());
                    if let Some(rules) = sensitive_rules {
                        let hits = sensitive_hits(batch, index, rules);
                        if !hits.is_empty() {
                            for hit in &hits {
                                eprintln!("taguru: import: {}: {}", path.display(), hit.text());
                            }
                            eprintln!(
                                "taguru: import: {}: {}",
                                path.display(),
                                refused_batch_message(index, batch)
                            );
                            refused.push(serde_json::json!({
                                "context": batch.context,
                                "source": batch.source,
                                "error": refused_batch_error(&hits),
                            }));
                            continue;
                        }
                    }
                    let text = String::from_utf8(bytes[range].to_vec())
                        .expect("parse_stream already proved this range is UTF-8");
                    units.push(Unit {
                        text,
                        label: format!(
                            "{}: context '{}' source '{}'",
                            path.display(),
                            batch.context,
                            batch.source
                        ),
                        kind: UnitKind::Batch,
                    });
                    batch_count += 1;
                }
                for (context, installed) in &stream.schemas {
                    if let Some(earlier) = schema_owners.get(context) {
                        eprintln!(
                            "taguru: import: {}: {}",
                            path.display(),
                            duplicate_schema_message(context, earlier)
                        );
                        file_broken = true;
                        continue;
                    }
                    schema_owners.insert(context.clone(), path.clone());
                    schema_units.push(Unit {
                        text: crate::export::render_schema(context, installed.document()),
                        label: format!("{}: context '{context}' schema", path.display()),
                        kind: UnitKind::Schema,
                    });
                    schema_count += 1;
                }
                for (name, record) in &stream.groups {
                    if let Some(earlier) = group_owners.get(name) {
                        eprintln!(
                            "taguru: import: {}: {}",
                            path.display(),
                            duplicate_group_message(name, earlier)
                        );
                        file_broken = true;
                        continue;
                    }
                    group_owners.insert(name.clone(), path.clone());
                    group_units.push(Unit {
                        text: crate::export::render_group(name, record),
                        label: format!("{}: group '{name}'", path.display()),
                        kind: UnitKind::Group,
                    });
                    group_count += 1;
                }
            }
            Err(message) => {
                eprintln!("taguru: import: {}: {message}", path.display());
                file_broken = true;
            }
        }
        if file_broken {
            broken += 1;
        }
    }
    if broken > 0 {
        let message = format!(
            "{broken} of {} file(s) refused during validation; nothing was applied",
            files.len()
        );
        eprintln!("taguru: import: {message}");
        if as_json {
            // The gate's refusals were judged before this failure and
            // stay in the document (never silence).
            print_import_json_values(
                dry_run,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                refused,
                Some(message),
            );
        }
        return 1;
    }

    // Schemas install after every batch, before groups restore — the
    // same order run_local's Pass 2 keeps (ADR 0009 §13) — so a
    // schema record naming a context an earlier chunk's batch creates
    // must always be sent after it, and a group naming a context an
    // earlier chunk's schema installs onto must always be sent after
    // that.
    units.append(&mut schema_units);
    units.append(&mut group_units);

    // A single unit that alone exceeds the byte budget cannot be
    // packed into any chunk, however the packer below runs — refused
    // before the network is ever touched (ADR 0002 §9).
    if let Some(oversized) = units
        .iter()
        .find(|unit| unit.len() > REMOTE_IMPORT_BUDGET_BYTES)
    {
        let message = oversized_unit_message(
            &oversized.label,
            oversized.len(),
            REMOTE_IMPORT_BUDGET_BYTES,
        );
        eprintln!("taguru: import: {message}");
        if as_json {
            print_import_json_values(
                dry_run,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                refused,
                Some(message),
            );
        }
        return 1;
    }

    let api = Api::new(base.to_string());
    // ADR 0002 §5: every remote, mutating invocation prints its target
    // before sending anything.
    eprintln!("import → {base}");
    api.warn_on_version_skew("import");
    // ADR 0009 §13's explicit compatibility refusal, checked only when
    // the stream actually carries a schema record — a schema-free
    // import must behave exactly as it did before this preflight
    // existed, skew warning included, whatever the peer reports.
    if schema_count > 0
        && let Some(message) = api.schema_import_refusal()
    {
        eprintln!("{message}");
        if as_json {
            print_import_json_values(
                dry_run,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                refused,
                Some(message),
            );
        }
        return 1;
    }

    let mut queue = pack_chunks(units, REMOTE_IMPORT_BUDGET_BYTES);

    // A LIVE estimate, not a fixed denominator: every 413-adaptive
    // halve (below, and at the loop's own proactive check) replaces
    // one queued chunk with two, growing this by one. A chunk printed
    // "N/M" before a later halving can end up describing a since-grown
    // M — the display always reflects "M chunks planned as of THIS
    // line," matching every other adaptive-retry progress display's
    // convention, not a promise that stays fixed for the whole run.
    let mut total = queue.len();
    let mut landed_chunks = 0usize;
    let mut budget = REMOTE_IMPORT_BUDGET_BYTES;
    let mut batches_landed = 0usize;
    let mut schema_records_landed = 0usize;
    let mut group_records_landed = 0usize;
    let mut contexts: BTreeSet<String> = BTreeSet::new();
    let mut json_batches: Vec<Value> = Vec::new();
    let mut json_schemas: Vec<Value> = Vec::new();
    let mut json_groups: Vec<Value> = Vec::new();
    // The last unit the server confirmed (#863): a lost connection
    // names where the confirmed prefix ends, not just its chunk count.
    let mut last_confirmed: Option<String> = None;

    while let Some(chunk) = queue.pop_front() {
        if chunk.size() > budget && chunk.units.len() > 1 {
            let (first, second) = chunk.halve();
            queue.push_front(second);
            queue.push_front(first);
            total += 1;
            continue;
        }
        match api.import_chunk(&chunk.body(), dry_run) {
            Ok(result) => {
                landed_chunks += 1;
                last_confirmed = chunk.units.last().map(|unit| unit.label.clone());
                let outcomes = result
                    .get("batches")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for outcome in &outcomes {
                    if let Some(context) = outcome.get("context").and_then(Value::as_str) {
                        contexts.insert(context.to_string());
                    }
                }
                batches_landed += outcomes.len();
                let schemas = result
                    .get("schemas")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                schema_records_landed += schemas.len();
                let groups = result
                    .get("groups")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                group_records_landed += groups.len();
                if as_json {
                    // The server already answers each chunk in exactly
                    // the shape `--json` reports — accumulated across
                    // chunks and printed once at the end, never
                    // reparsed into a different type that could drift
                    // from what the server actually said.
                    json_batches.extend(outcomes);
                    json_schemas.extend(schemas);
                    json_groups.extend(groups);
                } else {
                    // One line per LANDED chunk, always — every group unit
                    // rides after every batch unit (the append above), so
                    // the tail chunk(s) of a run can carry only group
                    // records and `outcomes` is empty for those. Skipping
                    // the line there (as this used to do) made `chunk N/M`
                    // visibly jump a number, and an operator watching the
                    // log cannot tell that apart from a crash between
                    // prints.
                    println!(
                        "chunk {landed_chunks}/{total}: {}",
                        if outcomes.is_empty() {
                            "schema/group record(s) only".to_string()
                        } else {
                            summarize_chunk_outcomes(&outcomes)
                        }
                    );
                    for schema in &schemas {
                        let context = schema.get("context").and_then(Value::as_str).unwrap_or("?");
                        let mode = schema.get("mode").and_then(Value::as_str).unwrap_or("?");
                        let types = schema.get("types").and_then(Value::as_u64).unwrap_or(0);
                        let relations =
                            schema.get("relations").and_then(Value::as_u64).unwrap_or(0);
                        println!(
                            "context '{context}' schema (mode: {mode}): {types} type(s), \
                             {relations} relation(s) — installed"
                        );
                    }
                    for group in &groups {
                        let name = group.get("name").and_then(Value::as_str).unwrap_or("?");
                        let outcome = group.get("outcome").and_then(Value::as_str).unwrap_or("?");
                        let member_contexts =
                            group.get("contexts").and_then(Value::as_u64).unwrap_or(0);
                        let child_groups = group.get("groups").and_then(Value::as_u64).unwrap_or(0);
                        println!(
                            "group '{name}': {member_contexts} member context(s), \
                             {child_groups} child group(s) — {outcome}"
                        );
                    }
                }
            }
            Err(ImportFailure::TooLarge(message)) => {
                if chunk.units.len() == 1 {
                    eprintln!("taguru: import: {message}");
                    if as_json {
                        print_import_json_values(
                            dry_run,
                            json_batches,
                            json_schemas,
                            json_groups,
                            refused.clone(),
                            Some(message),
                        );
                    }
                    return server_refused_single_unit_error(
                        &chunk.units[0].label,
                        chunk.units[0].len(),
                    );
                }
                // Pre-application rejection (ADR 0002 §8): safe to
                // adapt to and resend automatically — repeatedly, if
                // a further 413 keeps arriving, until this chunk is
                // a single batch (server_refused_single_unit_error) or
                // lands.
                budget = (chunk.size() / 2).max(1);
                let (first, second) = chunk.halve();
                queue.push_front(second);
                queue.push_front(first);
                total += 1;
            }
            Err(ImportFailure::InvalidUrl(message)) => {
                // No request was ever sent — nothing landed, nothing
                // to resume; this is a usage problem, not a partial
                // apply, so it exits like one (2, not 1) — the same
                // fix as the upfront `--url` check above, for the
                // narrower "parses but cannot carry a path" shape that
                // check does not itself catch (e.g. a `data:`/`mailto:`
                // URL).
                return crate::config::subcommand_usage_error("import", &message);
            }
            Err(ImportFailure::Transport(message)) => {
                eprintln!("taguru: import: {message}");
                if dry_run {
                    eprintln!(
                        "taguru: import: connection lost after chunk \
                         {landed_chunks}/{total} of the dry run — nothing was applied"
                    );
                } else {
                    eprintln!(
                        "taguru: import: connection lost after chunk {landed_chunks}/{total} \
                         — re-run `--dry-run` to confirm what would change, then resume"
                    );
                }
                // Where the confirmed prefix ends and what the unconfirmed
                // chunk carried, by file and source (#863) — a chunk
                // number alone maps to nothing the operator holds.
                if let Some(confirmed) = &last_confirmed {
                    eprintln!("taguru: import: last confirmed: {confirmed}");
                }
                eprintln!(
                    "taguru: import: not confirmed — this chunk carried {}",
                    chunk.carried()
                );
                // The chunks still queued behind the dropped one were
                // never sent either — tallied exactly as after a refusal.
                for line in never_sent_lines(&queue) {
                    eprintln!("taguru: import: {line}");
                }
                if as_json {
                    print_import_json_values(
                        dry_run,
                        json_batches,
                        json_schemas,
                        json_groups,
                        refused.clone(),
                        Some(message),
                    );
                }
                return 1;
            }
            Err(ImportFailure::Refused {
                status,
                message,
                body,
            }) => {
                eprintln!(
                    "taguru: import: chunk {}/{total} refused: {message}",
                    landed_chunks + 1
                );
                if matches!(status, 401 | 403) {
                    eprintln!(
                        "taguru: import: /import requires the admin role — check that \
                         TAGURU_API_TOKEN (or TAGURU_API_TOKENS) names a key with it"
                    );
                }
                if let Some(integrity) = body.get("integrity").and_then(Value::as_str) {
                    eprintln!("taguru: import: integrity: {integrity}");
                }
                // The refused chunk by file and source, then the server's
                // own `issues[]` re-addressed to those units (#863) — the
                // server names `batches[3].associations[7].subject`; the
                // operator holds files.
                eprintln!("taguru: import: this chunk carried {}", chunk.carried());
                for line in refusal_issue_lines(&body, &chunk.units) {
                    eprintln!("taguru: import: {line}");
                }
                for line in never_sent_lines(&queue) {
                    eprintln!("taguru: import: {line}");
                }
                if dry_run {
                    eprintln!(
                        "taguru: import: {landed_chunks} chunk(s) of the dry run \
                         previewed cleanly before this refusal; nothing was applied"
                    );
                } else {
                    eprintln!(
                        "taguru: import: {landed_chunks} chunk(s) already landed durably; \
                         re-running the corrected stream is exact (each batch replaces its \
                         own source)"
                    );
                }
                if as_json {
                    print_import_json_values(
                        dry_run,
                        json_batches,
                        json_schemas,
                        json_groups,
                        refused.clone(),
                        Some(message),
                    );
                }
                return 1;
            }
        }
    }

    let refused_count = refused.len();
    if as_json {
        print_import_json_values(
            dry_run,
            json_batches,
            json_schemas,
            json_groups,
            refused,
            None,
        );
    } else if dry_run {
        let mut summary = format!("dry run: {batch_count} batch(es)");
        if schema_count > 0 {
            summary.push_str(&format!(", {schema_count} schema record(s)"));
        }
        if group_count > 0 {
            summary.push_str(&format!(" and {group_count} group record(s)"));
        }
        summary.push_str(" valid, nothing applied");
        println!("{summary}");
        if refused_count > 0 {
            println!("import: {refused_count} batch(es) refused (sensitive)");
        }
    } else {
        println!(
            "import: {batches_landed} batch(es) applied across {} context(s) in {total} \
             chunk(s)",
            contexts.len()
        );
        if schema_count > 0 {
            println!(
                "import: {schema_records_landed} of {schema_count} schema record(s) installed"
            );
        }
        if group_count > 0 {
            println!("import: {group_records_landed} of {group_count} group record(s) restored");
        }
        if refused_count > 0 {
            println!("import: {refused_count} batch(es) refused (sensitive)");
        }
    }
    if refused_count > 0 { 1 } else { 0 }
}

/// [`super::local`]'s `print_import_json` remote twin: the server
/// already answers each chunk in exactly the shape `ImportStreamOutcome`
/// describes, so this builds the same `{dry_run, batches, schemas,
/// groups}` envelope directly from the accumulated `Value`s instead of
/// round-tripping them through the typed structs (which would risk
/// silently dropping a field the server sent that this build's types
/// don't know about). Same `{dry_run, error, batches, schemas, groups}`
/// envelope (no `failed_batches`: a remote refusal fails its whole
/// chunk, never one batch within it, so there is nothing per-batch to
/// name — the chunk's `error` text already says what happened). Called
/// on every remote `--json` exit path, including failures, with
/// whatever `batches`/`schemas`/`groups` landed before the failure —
/// never silence.
fn print_import_json_values(
    dry_run: bool,
    batches: Vec<Value>,
    schemas: Vec<Value>,
    groups: Vec<Value>,
    failed_batches: Vec<Value>,
    error: Option<String>,
) {
    let mut report = serde_json::Map::new();
    report.insert("dry_run".to_string(), Value::Bool(dry_run));
    if let Some(message) = error {
        report.insert("error".to_string(), Value::String(message));
    }
    // `--refuse-sensitive`'s refusals (ADR 0038 §3.4), the one
    // per-batch failure the remote path has — the same key and shape
    // the local path's `FailedBatch` prints, omitted when empty.
    if !failed_batches.is_empty() {
        report.insert("failed_batches".to_string(), Value::Array(failed_batches));
    }
    report.insert("batches".to_string(), Value::Array(batches));
    // Matches ImportStreamOutcome's own `skip_serializing_if` on
    // `schemas`/`groups` — omitted entirely when empty, not printed as
    // `[]`, so local and remote --json agree byte for byte on a
    // schema-/group-less run.
    if !schemas.is_empty() {
        report.insert("schemas".to_string(), Value::Array(schemas));
    }
    if !groups.is_empty() {
        report.insert("groups".to_string(), Value::Array(groups));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Object(report))
            .expect("a Map of already-valid JSON values always serializes")
    );
}

/// Explicit files are taken as given; a directory contributes its
/// `*.jsonl` files in name order. An empty directory is an error — a
/// place the operator pointed at with nothing to do is a mistake, not
/// a success.
pub(super) fn expand(paths: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for raw in paths {
        let path = Path::new(raw);
        if path.is_file() {
            files.push(path.to_path_buf());
        } else if path.is_dir() {
            let mut found: Vec<PathBuf> = fs::read_dir(path)
                .map_err(|error| format!("cannot read {raw}: {error}"))?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
                .collect();
            if found.is_empty() {
                return Err(format!("no .jsonl files under {raw}"));
            }
            found.sort();
            files.append(&mut found);
        } else {
            return Err(format!("{raw} is neither a file nor a directory"));
        }
    }
    Ok(files)
}
