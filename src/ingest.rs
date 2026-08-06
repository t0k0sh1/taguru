//! `taguru import`: offline batch ingestion — the bulk/initial-load
//! path that the REST API is the wrong tool for. A batch file is JSON
//! Lines: one header naming the context and the source, then
//! association / alias / passage lines (the same shapes the HTTP
//! endpoints accept, minus per-line sources — the header's source is
//! stamped on every line, see below).
//!
//! One file states one source's COMPLETE truth: applying it first
//! retracts the source, then adds the file's facts, so re-importing a
//! file is idempotent and importing a revised one is the same
//! differential sync agents do live (`retract_source` → re-ingest).
//! That contract is why association lines may not carry their own
//! source: a source the header does not name would survive the
//! retraction and double on every re-import.
//!
//! Validation is a separate pass: every file parses completely (and
//! the set of files is checked for two files claiming one source)
//! before anything applies. Apply-stage failures cannot all be
//! pre-checked (capacity, disk); those are reported per file and the
//! remaining files still run — every file is one source, independent
//! by construction, and a partially applied one heals on re-import.
//! Until that re-import (or a retraction), the batch-open marker
//! written around every apply keeps the tear visible: boot and
//! `taguru inspect` name the source, however the batch stopped short
//! (see [`apply_batch`]).
//!
//! The writes go through the same registry every server write goes
//! through — WAL-staged, budget-enforced, flushed — and the data
//! directory lock makes the server/import conflict a refusal instead
//! of a corruption.
//!
//! The same contract is served live as `POST /import` (one request =
//! one batch file), so bulk loads reach a running server without a
//! downtime window; [`parse_batch`] and [`apply_batch`] are that
//! endpoint's core too, which is what keeps the two entrances from
//! drifting apart.
//!
//! Beside batches, a stream may carry GROUP records: one
//! `taguru_group` line states one group's complete truth (name,
//! description, member contexts, child groups) the way one batch
//! states one source's. Applying one is a create-or-replace of the
//! whole record — never a delta — so re-importing stays idempotent.
//! Groups apply AFTER every batch of the run (one CLI invocation, one
//! `POST /import` body), whatever file or position carried them, so a
//! group and the member contexts it names can travel together in any
//! order; a member that still does not exist at that point refuses
//! the whole group set, with every batch already durably landed.
//!
//! Split into submodules by concern: `local` runs the offline
//! (non-`--url`) apply path, `remote` runs `import --url`'s chunked
//! wire path, `model` is the batch/stream data model and JSONL line
//! parser, `rejection` predicts and applies pre-write refusals
//! ([`apply_batch`]/[`preview_batch`]), `schema_apply` installs a
//! parsed `taguru_schema` record, and `report` formats the CLI's
//! per-batch line and sets up logging. This hub keeps the `run`
//! dispatcher, the format-version constants, and the shared surface
//! `src/api/import.rs`, `extract.rs`, `compact.rs`, `export.rs`, and
//! the router consume via `crate::ingest::`, re-exported unchanged.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fs;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;
use taguru::context::{AliasError, Context};
use taguru::deadline::Deadline;

use crate::api::{
    MAX_ASSOCIATION_WEIGHT, MAX_ASSOCIATIONS_PER_REQUEST, MAX_CONTEXT_NAME_BYTES,
    MAX_DESCRIPTION_BYTES, MAX_NAME_BYTES,
};
use crate::env::DEFAULT_MAX_BODY_BYTES;
use crate::groups::{GroupRecord, MAX_GROUP_MEMBERS};
use crate::registry::{AccessError, AppState, AssocOp, ContextMeta, CreateError};
use crate::remote::{Api, ImportFailure};
use crate::schema;

#[path = "ingest/local.rs"]
mod local;
#[path = "ingest/model.rs"]
mod model;
#[path = "ingest/rejection.rs"]
mod rejection;
#[path = "ingest/remote.rs"]
mod remote;
#[path = "ingest/report.rs"]
mod report;
#[path = "ingest/schema_apply.rs"]
mod schema_apply;
#[path = "ingest/tests.rs"]
#[cfg(test)]
mod tests;

use local::{
    duplicate_group_message, duplicate_schema_message, duplicate_source_message, run_local,
};
use remote::{expand, run_remote};
use report::report;

pub(crate) use model::{Batch, parse_batch, parse_stream, split_batches};
pub(crate) use rejection::{AliasRejection, Applied, ApplyRefusal, apply_batch, preview_batch};
pub(crate) use report::init_logging;
pub(crate) use schema_apply::{SchemaApplyError, apply_schema_record};

// Test-only cross-submodule access: production code never names these
// at the hub level (each is private to the one submodule that both
// defines and calls it), but the unified test module (`tests.rs`,
// `use super::*;`) exercises them across the split the same way the
// single pre-split file's inline tests did.
#[cfg(test)]
use model::MAX_LINE_BYTES;
#[cfg(test)]
use rejection::AliasNamespace;
#[cfg(test)]
use remote::{Chunk, Unit, UnitKind, pack_chunks};

const USAGE: &str = "\
usage: taguru import [--dry-run] [--no-embed] [--json] [--config FILE]
                      [--url URL] FILE|DIR...

Applies JSONL batch files to TAGURU_DATA_DIR offline (the server must
not be running — the directory lock enforces it), or to a RUNNING
server with --url. One batch = one source's complete truth: import
retracts the source, then applies the batch, so re-importing is
idempotent. A file carries one batch or a whole stream of them (each
`taguru_batch` header line starts the next) — `taguru export` writes
such streams. A `taguru_schema` line states one context's whole
schema document (ADR 0009 §13); it installs AFTER every batch, BEFORE
any group, so a schema record can name a context a batch of the same
stream just created. A `taguru_group` line states one group's complete
truth the same way; groups restore AFTER every batch and schema of
the run (create-or-replace of the whole record), so group files
re-apply in any order. A directory expands to its *.jsonl files,
sorted by name. Format: docs/import.html.

  --dry-run    validate every file and report; touch nothing (with
               --url, POSTs every chunk as ?dry_run=true instead)
  --no-embed   skip the embedding refresh TAGURU_EMBED_URL would enable
               (offline only — combined with --url this is a usage
               error, since the server's own configuration decides
               once the request lands there)
  --config F   read KEY=VALUE environment from F (same dialect as serve)

  --url URL    import into a RUNNING server instead of TAGURU_DATA_DIR
               directly: POST /import, one request per chunk. The
               input is split on batch boundaries only — never
               mid-batch — into chunks under the server's body cap
               (TAGURU_MAX_BODY_BYTES, 8 MiB by default), starting at
               a 4 MiB budget and halved (never crossing a batch
               boundary) and resent on a 413. A single batch that
               alone exceeds the cap is a hard error naming the
               source: raise TAGURU_MAX_BODY_BYTES on the server, or
               split that source's content upstream of import. A lost
               connection names which chunk landed and points at
               --dry-run to confirm before resuming — nothing past
               that is retried automatically (import's
               retract-then-apply contract makes any resend exact).
               Auth rides TAGURU_API_TOKEN (or the first name:token
               entry of TAGURU_API_TOKENS), the admin role the server
               requires. In CI, name the target per invocation:
                 taguru import --url \"$TAGURU_URL\" backups/
  --json       one JSON document instead of per-file lines: {dry_run,
               batches: [...], schemas: [...], groups: [...]} — the
               same shape POST /import answers with (schemas/groups
               omitted when empty). With --url, every field is exact
               (the server previews with the same code path a real
               apply runs). Offline, a real (non-dry-run) run is exact
               the same way; offline --dry-run cannot open the data
               directory without the lock a running import would need,
               so its batch counts are read straight from each file
               (created/retracted and the *_dropped fields all report
               0/false) and its schemas/groups arrays are always empty
               — a preview, not the server's exact one. Every exit path
               prints exactly one document, failures included: validation
               refusing every file, the
               registry refusing to boot, and a remote transport error
               or a server-refused chunk each add a top-level 'error'
               string beside whatever batches/groups already landed; a
               batch refused mid-run (offline only) is named under a
               failed_batches array instead of batches, since there is
               no successful outcome to report for it.
";

/// The one format version this build reads and docs/import.html
/// describes. `pub(crate)` so `GET /version` (ADR 0005 §3, §6) can
/// report it under `batch_formats`.
pub(crate) const BATCH_VERSION: u64 = 1;

/// The `taguru_group` record's own version stamp — separate from
/// [`BATCH_VERSION`] so either shape can rev without dragging the
/// other along. Export serializes it; parse refuses any other value.
pub(crate) const GROUP_VERSION: u64 = 1;

/// Passage cap, mirroring the HTTP default: over the API a passage
/// rides under `TAGURU_MAX_BODY_BYTES` (8 MiB), and a file must not
/// smuggle in what a request could not. Extract caps whole documents
/// here too — a document over it could not ride as a passage.
pub(crate) const MAX_PASSAGE_BYTES: usize = 8 * 1024 * 1024;

pub fn run(args: &[String]) -> i32 {
    let mut dry_run = false;
    let mut no_embed = false;
    let mut as_json = false;
    let mut config: Option<PathBuf> = None;
    let mut url: Option<String> = None;
    let mut paths: Vec<String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return 0;
            }
            "--dry-run" => dry_run = true,
            "--no-embed" => no_embed = true,
            "--json" => as_json = true,
            "--config" => match rest.next() {
                Some(path) => config = Some(PathBuf::from(path)),
                None => {
                    return crate::config::subcommand_usage_error(
                        "import",
                        "--config needs a file path",
                    );
                }
            },
            // Trailing '/' trimmed the same way export/compact/calibrate/
            // communities already do — a joined path segment must not
            // double up.
            "--url" => match rest.next() {
                Some(value) => url = Some(value.trim_end_matches('/').to_string()),
                None => {
                    return crate::config::subcommand_usage_error(
                        "import",
                        "--url needs a server URL",
                    );
                }
            },
            other if other.starts_with('-') => {
                return crate::config::subcommand_usage_error(
                    "import",
                    &format!("unknown flag '{other}'"),
                );
            }
            path => paths.push(path.to_string()),
        }
    }
    if paths.is_empty() {
        eprint!("{USAGE}");
        return 2;
    }
    // ADR 0002 §5: a flag that only makes sense offline, combined with
    // --url, is a usage error rather than a silent no-op — the
    // server's own embedding configuration decides once the request
    // lands there, so --no-embed has nothing left to control remotely.
    if no_embed && url.is_some() {
        return crate::config::subcommand_usage_error(
            "import",
            "--no-embed cannot be combined with --url — the server's own embedding \
             configuration decides",
        );
    }
    // TAGURU_CONFIG fallback (issue #248 item 2): --config wins, but a
    // deployment file baked in via the environment still applies when
    // it's absent — the same priority serve/health/calibrate/
    // communities/evaluate/restore already give it.
    let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    // SAFETY (same contract as serve): applied while the process is
    // still single-threaded — import never starts a runtime at all.
    // Loaded before the --url dispatch below, too: a config file is
    // the usual place a deployment's TAGURU_API_TOKEN lives, and
    // Api::new reads the bearer from the environment at construction.
    if let Some(path) = &config {
        crate::config::load_config(path);
    }

    let files = match expand(&paths) {
        Ok(files) => files,
        Err(message) => return crate::config::subcommand_usage_error("import", &message),
    };

    match url {
        // ADR 0002 §5/§6: `--url` is the only way `import` goes
        // remote — no positional URL argument, no TAGURU_URL or
        // default_base_url() fallback the way `health`/`calibrate`/
        // `communities` have one. Absent, this is exactly the local
        // path that ran before this flag existed.
        Some(base) => run_remote(&base, &files, dry_run, as_json),
        None => run_local(&files, dry_run, no_embed, as_json),
    }
}
