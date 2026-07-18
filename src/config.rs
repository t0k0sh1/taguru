//! `--config FILE` loading and the byte-count formatting used in every
//! size report — pulled out of the CLI dispatcher because the offline
//! subcommands (compact, export, import, extract, estimate, inspect)
//! and `main`'s own `serve` all need them without depending on the
//! dispatcher itself. [`usage_error`] moves with them since
//! [`load_config`] is one of its callers; `cli.rs` pulls it back in
//! for its own argument-parsing errors.

use std::path::Path;
use std::process::exit;

/// Prints a `taguru: MESSAGE — try 'taguru --help'` line and exits
/// with the CLI's usage-error status (2) — the shared choke point for
/// every subcommand's argument-parsing and config-loading failures.
pub(crate) fn usage_error(message: &str) -> ! {
    eprintln!("taguru: {message} — try 'taguru --help'");
    exit(2)
}

/// [`usage_error`] for a subcommand's own flag parsing. Returns the
/// code as a plain `i32` instead of exiting outright, since each
/// subcommand's `run` is meant as a pure `args -> exit code` function —
/// `cli.rs`'s dispatch line for it is the one place that actually
/// calls `exit`.
pub(crate) fn subcommand_usage_error(subcommand: &str, message: &str) -> i32 {
    eprintln!("taguru: {subcommand}: {message} — try 'taguru {subcommand} --help'");
    2
}

/// Human-readable byte count: exact under a KiB, one decimal above —
/// these sit in report lines, not accounting ledgers.
pub(crate) fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 3] = [("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)];
    for (unit, size) in UNITS {
        let value = bytes as f64 / size as f64;
        // One byte under a GiB is 1023.99… MiB, which "{:.1}" prints
        // as "1024.0 MiB" — this unit's own boundary in the smaller
        // unit's spelling. A value that ROUNDS to that boundary wears
        // this unit instead ("1.0 GiB"); anything below it keeps the
        // smaller unit's precision.
        if bytes >= size || value * 1024.0 >= 1023.95 {
            return format!("{value:.1} {unit}");
        }
    }
    format!("{bytes} B")
}

/// Every variable the server reads, for typo detection: a config file
/// is where a misspelled knob silently becomes a no-op, and unlike the
/// shell it is worth linting.
pub(crate) const KNOWN_KEYS: [&str; 41] = [
    "TAGURU_ADDR",
    "TAGURU_DATA_DIR",
    "TAGURU_CACHE_BYTES",
    "TAGURU_FLUSH_SECS",
    "TAGURU_WAL",
    "TAGURU_WAL_MAX_BYTES",
    "TAGURU_PASSAGES_WAL_MAX_BYTES",
    "TAGURU_REPLICATE_URL",
    "TAGURU_REPLICATE_INTERVAL_MS",
    "TAGURU_TAKEOVER",
    "TAGURU_REPLICA",
    "TAGURU_WRITER_URL",
    "TAGURU_API_TOKEN",
    "TAGURU_API_TOKENS",
    "TAGURU_KEY_SCOPES",
    "TAGURU_PUBLIC_URL",
    "TAGURU_MAX_BODY_BYTES",
    "TAGURU_MCP_MAX_RESULT_BYTES",
    "TAGURU_REQUEST_TIMEOUT_SECS",
    "TAGURU_RATE_LIMIT_PER_MIN",
    "TAGURU_AUTH_FAIL_LIMIT_PER_MIN",
    "TAGURU_MAX_CONCURRENT_REQUESTS",
    "TAGURU_MAX_CONCURRENT_HEAVY_OPS",
    "TAGURU_CROSS_SEARCH_CONCURRENCY",
    "TAGURU_EMBED_URL",
    "TAGURU_EMBED_MODEL",
    "TAGURU_EMBED_API_KEY",
    "TAGURU_EMBED_TIMEOUT_SECS",
    "TAGURU_EMBED_AUTO",
    "TAGURU_EMBED_PARALLEL",
    "TAGURU_EMBED_PASSAGES",
    "TAGURU_PASSAGE_VECTOR_LIMIT",
    "TAGURU_SEMANTIC_FLOOR",
    "TAGURU_EXTRACT_URL",
    "TAGURU_EXTRACT_MODEL",
    "TAGURU_EXTRACT_API_KEY",
    "TAGURU_EXTRACT_TIMEOUT_SECS",
    "TAGURU_EXTRACT_PARALLEL",
    "TAGURU_LOG_FORMAT",
    "TAGURU_LOG_SEARCHES",
    "TAGURU_CONFIG",
];

/// Reads a configuration file into the process environment. Exits with
/// a usage error on an unreadable file or a malformed line — a config
/// the operator pointed at explicitly must never be half-applied.
///
/// Call this before the async runtime exists: applying the file means
/// `std::env::set_var`, which is only sound while the process is
/// single-threaded.
pub(crate) fn load_config(path: &Path) {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => usage_error(&format!("cannot read config {}: {error}", path.display())),
    };
    let pairs = match parse_config(&text) {
        Ok(pairs) => pairs,
        Err(message) => usage_error(&format!("config {}: {message}", path.display())),
    };
    // Snapshotted before any file value is applied: a key repeated in
    // the file would otherwise see its own earlier `set_var` on the
    // second pass and be misreported as a real environment override,
    // when it was this same file that set it moments ago.
    let preexisting: std::collections::HashSet<&str> = pairs
        .iter()
        .map(|(key, _)| key.as_str())
        .filter(|key| std::env::var_os(key).is_some())
        .collect();
    let mut seen = std::collections::HashSet::new();
    for (key, value) in &pairs {
        if key.starts_with("TAGURU_") && !KNOWN_KEYS.contains(&key.as_str()) {
            eprintln!("taguru: config: {key} is not a variable taguru reads (typo?)");
        }
        // The real environment wins: a `docker run -e` or shell export
        // must override the file, exactly as it overrides an image's
        // baked-in defaults.
        if preexisting.contains(key.as_str()) {
            eprintln!("taguru: config: {key} set in the environment; the file value is ignored");
            continue;
        }
        if !seen.insert(key.as_str()) {
            eprintln!(
                "taguru: config: {key} appears more than once in the file; the last value wins"
            );
        }
        // SAFETY: the caller runs this before the tokio runtime (or
        // any other thread) starts, so no concurrent environment
        // access exists.
        unsafe { std::env::set_var(key, value) };
    }
}

/// The `docker run --env-file` dialect: one KEY=VALUE per line, `#`
/// comments and blank lines ignored, values taken verbatim (no quoting
/// or expansion). Returned in file order.
fn parse_config(text: &str) -> Result<Vec<(String, String)>, String> {
    // A leading BOM survives `str::trim` (U+FEFF is not `White_Space`),
    // so left in place it rides onto the first key of the file — e.g.
    // "\u{FEFF}TAGURU_ADDR" — which fails the `TAGURU_` typo check
    // silently (it doesn't start with "TAGURU_" either) and then just
    // vanishes as an unrecognized env var, no warning printed at all.
    let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
    let mut pairs = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!(
                "line {}: expected KEY=VALUE, got '{raw}'",
                index + 1
            ));
        };
        let key = key.trim();
        if key.is_empty() || key.contains(char::is_whitespace) {
            return Err(format!(
                "line {}: '{key}' is not a variable name",
                index + 1
            ));
        }
        // Only the whitespace `split_once` leaves stranded around the
        // `=` itself — `KEY = VALUE` or `KEY= VALUE` are both common
        // spellings. The line's own leading/trailing whitespace is
        // already gone from the `line.trim()` above; verbatim applies
        // to the value's content, not to accidental padding around
        // the separator.
        let value = value.trim();
        pairs.push((key.to_string(), value.to_string()));
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_lines_parse_the_env_file_dialect() {
        let text =
            "# comment\n\nTAGURU_ADDR=127.0.0.1:0\nTAGURU_API_TOKEN=a=b=c\n  TAGURU_WAL=1  \n";
        let pairs = parse_config(text).unwrap();
        assert_eq!(
            pairs,
            vec![
                ("TAGURU_ADDR".to_string(), "127.0.0.1:0".to_string()),
                // The first '=' splits; the value keeps the rest verbatim.
                ("TAGURU_API_TOKEN".to_string(), "a=b=c".to_string()),
                ("TAGURU_WAL".to_string(), "1".to_string()),
            ]
        );
    }

    /// A BOM survives `str::trim` (U+FEFF is not `White_Space`), so left
    /// in place it would ride onto the first key as "\u{FEFF}TAGURU_ADDR"
    /// — which is not `TAGURU_ADDR`, so the value never reaches the
    /// server, and it doesn't start with "TAGURU_" either, so not even
    /// the typo warning fires. Notepad and other Windows editors stamp a
    /// BOM onto every UTF-8 file they save, so this is not exotic input.
    #[test]
    fn a_leading_bom_does_not_mangle_the_first_key() {
        let text = "\u{FEFF}TAGURU_ADDR=127.0.0.1:0\nTAGURU_WAL=1\n";
        let pairs = parse_config(text).unwrap();
        assert_eq!(
            pairs,
            vec![
                ("TAGURU_ADDR".to_string(), "127.0.0.1:0".to_string()),
                ("TAGURU_WAL".to_string(), "1".to_string()),
            ]
        );
    }

    #[test]
    fn whitespace_padding_the_separator_is_trimmed_from_key_and_value() {
        let text = "TAGURU_ADDR = 127.0.0.1:0\nTAGURU_WAL= 1\n";
        let pairs = parse_config(text).unwrap();
        assert_eq!(
            pairs,
            vec![
                ("TAGURU_ADDR".to_string(), "127.0.0.1:0".to_string()),
                ("TAGURU_WAL".to_string(), "1".to_string()),
            ]
        );
    }

    #[test]
    fn config_lines_without_a_separator_are_refused_with_the_line_number() {
        let error = parse_config("TAGURU_WAL=1\nnot a pair\n").unwrap_err();
        assert!(error.contains("line 2"), "{error}");
    }

    #[test]
    fn config_keys_with_spaces_are_refused() {
        let error = parse_config("TAGURU WAL=1\n").unwrap_err();
        assert!(error.contains("line 1"), "{error}");
    }

    /// A key repeated in the file must not see its own earlier
    /// `set_var` and mistake it for a real environment override — that
    /// bug both applied the file's *first* occurrence instead of its
    /// last (env-file dialects agree the last wins) and printed a
    /// "set in the environment" message that named this same file as
    /// the culprit.
    #[test]
    fn load_config_lets_the_files_last_duplicate_key_win() {
        let path = std::env::temp_dir().join(format!(
            "taguru-cli-test-dup-key-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::write(
            &path,
            "TAGURU_CLI_TEST_DUP=first\nTAGURU_CLI_TEST_DUP=second\n",
        )
        .unwrap();
        // SAFETY: no other thread touches the environment during this test.
        unsafe { std::env::remove_var("TAGURU_CLI_TEST_DUP") };

        load_config(&path);

        assert_eq!(
            std::env::var("TAGURU_CLI_TEST_DUP").as_deref(),
            Ok("second"),
            "the file's last occurrence must win, not its first"
        );

        // SAFETY: no other thread touches the environment during this test.
        unsafe { std::env::remove_var("TAGURU_CLI_TEST_DUP") };
        let _ = std::fs::remove_file(&path);
    }

    /// A real pre-existing environment variable must still beat the
    /// file — the fix for the duplicate-key bug above must not widen
    /// which keys the file is allowed to override.
    #[test]
    fn load_config_still_defers_to_a_real_pre_existing_environment_variable() {
        let path = std::env::temp_dir().join(format!(
            "taguru-cli-test-real-env-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::write(&path, "TAGURU_CLI_TEST_REAL=from_file\n").unwrap();
        // SAFETY: no other thread touches the environment during this test.
        unsafe { std::env::set_var("TAGURU_CLI_TEST_REAL", "from_shell") };

        load_config(&path);

        assert_eq!(
            std::env::var("TAGURU_CLI_TEST_REAL").as_deref(),
            Ok("from_shell"),
            "a real environment variable must still win over the file"
        );

        // SAFETY: no other thread touches the environment during this test.
        unsafe { std::env::remove_var("TAGURU_CLI_TEST_REAL") };
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fmt_bytes_never_prints_a_full_unit_in_the_smaller_unit() {
        // Straight cases keep their unit and precision.
        assert_eq!(fmt_bytes(1 << 30), "1.0 GiB");
        assert_eq!(fmt_bytes(900 * (1 << 20)), "900.0 MiB");
        assert_eq!(fmt_bytes(1023), "1023 B");
        // One byte under a unit rounds to that unit's own boundary —
        // "1024.0 MiB" is a GiB wearing the wrong spelling.
        assert_eq!(fmt_bytes((1 << 30) - 1), "1.0 GiB");
        assert_eq!(fmt_bytes((1 << 20) - 1), "1.0 MiB");
        // Just below the rounding boundary the smaller unit stays.
        assert_eq!(fmt_bytes(1073635738), "1023.9 MiB");
    }
}
