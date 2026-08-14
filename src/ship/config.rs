//! Replication configuration: the `TAGURU_REPLICATE_URL`/`_INTERVAL_MS`
//! env knobs, and opening the object store a replicate URL names.

use super::*;

/// How the operator turns shipping on: a bucket URL, nothing else
/// required. Credentials ride each cloud's default chain (`AWS_*`,
/// `GOOGLE_*`, `AZURE_*` — whatever `object_store`'s builders read
/// from the environment), so the one variable is the whole feature.
pub(crate) struct ReplicateConfig {
    pub(crate) url: String,
    pub(crate) interval: Duration,
}

impl ReplicateConfig {
    /// `TAGURU_REPLICATE_URL` (unset = shipping off), plus the poll
    /// cadence `TAGURU_REPLICATE_INTERVAL_MS` (default 1000 — the
    /// steady-state RPO knob). A zero interval would spin the poll
    /// loop; floor to 100ms, loudly, like every other env knob.
    pub(crate) fn from_env() -> Option<Self> {
        let url = std::env::var("TAGURU_REPLICATE_URL").ok()?;
        let url = url.trim().to_string();
        if url.is_empty() {
            // The same present-but-blank trap TAGURU_PUBLIC_URL guards
            // against: almost always a templating accident, never a
            // deliberate opt-out spelled as an empty string.
            tracing::warn!(
                "TAGURU_REPLICATE_URL is set but empty: treating replication as disabled — \
                 unset the variable entirely if that's intended"
            );
            return None;
        }
        let requested = crate::env::env_number("TAGURU_REPLICATE_INTERVAL_MS", 1000);
        // `<` vs `<=` is unobservable at the boundary itself (issue
        // #618): at `requested == 100`, both arms compute the same
        // `Duration::from_millis(100)` — the floor branch explicitly,
        // the else branch because `requested` already IS 100 — so a
        // mutant swapping this for `<=` produces the identical
        // `interval` for every possible input; only the warn log
        // (uncaptured by any test here) would differ.
        let interval = if requested < 100 {
            tracing::warn!(
                "TAGURU_REPLICATE_INTERVAL_MS={requested} would busy-poll the data \
                 directory; using 100"
            );
            Duration::from_millis(100)
        } else {
            Duration::from_millis(requested as u64)
        };
        Some(Self { url, interval })
    }
}

/// Opens the store a replicate URL names, with each cloud's default
/// credential chain. `parse_url` alone constructs builders WITHOUT
/// environment credentials — fine for `file://`, wrong for every
/// cloud — so the cloud schemes go through their builders' `from_env`
/// explicitly. `file://` is first-class, not a test crutch: it is how
/// the round trip is verified without cloud spend, and how an
/// air-gapped deployment ships to a mounted remote volume.
///
/// The `ErrorKind` a failure carries is load-bearing, not incidental:
/// `taguru restore`'s exit-code contract (`restore::run`) reads it
/// back to tell a usage mistake (`InvalidInput`/`NotFound` — a
/// malformed URL, an unrecognized scheme, a local path that does not
/// exist) from the store itself refusing to open
/// (`io::Error::other` — a rejected credential, a cloud builder
/// config error, an inaccessible local path). Keep new failure arms
/// on the right side of that split.
pub(crate) fn open_store(url: &str) -> io::Result<(Arc<dyn ObjectStore>, StorePath)> {
    use object_store::ObjectStoreScheme;

    let parsed = url::Url::parse(url)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, format!("{url}: {error}")))?;
    let (scheme, path) = ObjectStoreScheme::parse(&parsed)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, format!("{url}: {error}")))?;
    let root = StorePath::parse(path)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, format!("{url}: {error}")))?;
    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => {
            // The bucket must exist before S3/GCS/Azure accept a write;
            // hold file:// to the same contract instead of silently
            // mkdir-ing a typo into a fresh empty "bucket".
            let dir = parsed.to_file_path().map_err(|()| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{url}: not a local path"),
                )
            })?;
            if !dir.is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "{}: replication target directory does not exist",
                        dir.display()
                    ),
                ));
            }
            // Prefix handling differs local vs cloud: the URL's path IS
            // the directory, so the store roots there and the in-store
            // prefix is empty.
            let local = object_store::local::LocalFileSystem::new_with_prefix(&dir)
                .map_err(|error| io::Error::other(format!("{}: {error}", dir.display())))?;
            return Ok((Arc::new(local), StorePath::default()));
        }
        ObjectStoreScheme::AmazonS3 => Arc::new(
            object_store::aws::AmazonS3Builder::from_env()
                .with_url(url)
                .build()
                .map_err(|error| io::Error::other(format!("{url}: {error}")))?,
        ),
        ObjectStoreScheme::GoogleCloudStorage => Arc::new(
            object_store::gcp::GoogleCloudStorageBuilder::from_env()
                .with_url(url)
                .build()
                .map_err(|error| io::Error::other(format!("{url}: {error}")))?,
        ),
        ObjectStoreScheme::MicrosoftAzure => Arc::new(
            object_store::azure::MicrosoftAzureBuilder::from_env()
                .with_url(url)
                .build()
                .map_err(|error| io::Error::other(format!("{url}: {error}")))?,
        ),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{url}: unsupported replication scheme — use s3://, gs://, az://, or file://"
                ),
            ));
        }
    };
    Ok((store, root))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A malformed/unrecognized URL is a usage mistake: `InvalidInput`.
    #[test]
    fn a_bad_scheme_is_invalid_input() {
        let error = open_store("ftp://wherever").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput, "{error}");
    }

    /// A syntactically fine `file://` URL naming a directory that does
    /// not exist is also a usage mistake, not the store refusing to
    /// open: `NotFound`.
    #[test]
    fn a_missing_local_directory_is_not_found() {
        let missing =
            std::env::temp_dir().join(format!("taguru-open-store-missing-{}", std::process::id()));
        let error = open_store(&format!("file://{}", missing.display())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound, "{error}");
    }

    /// A well-formed cloud URL whose builder refuses to construct
    /// (missing/rejected credentials, a config error) is the STORE
    /// itself being unusable, not a usage mistake — `open_store` must
    /// carry that as `Other`, the kind `restore::run` maps to exit 1
    /// instead of exit 2. Azure's builder is the one of the three
    /// cloud backends that fails synchronously (no network round trip)
    /// on a missing account name, which is what makes this
    /// deterministic without real credentials or a live endpoint.
    #[test]
    fn a_rejected_cloud_builder_config_is_other() {
        let _env = crate::ship::test_support::ScrubbedAzureEnv::new();
        let error = open_store("az://some-bucket").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other, "{error}");
    }

    /// #618: `s3://`/`gs://` must reach their own cloud builders, not
    /// fall through to the catch-all "unsupported scheme" refusal —
    /// whatever the builder does with ambient credentials (succeed,
    /// fail synchronously) is not this test's concern; only that it is
    /// even ATTEMPTED, not skipped.
    #[test]
    fn amazon_s3_and_google_cloud_storage_are_recognized_schemes() {
        for url in ["s3://some-bucket", "gs://some-bucket"] {
            if let Err(error) = open_store(url) {
                assert!(
                    !error.to_string().contains("unsupported replication scheme"),
                    "{url} must reach its own cloud builder, not the catch-all: {error}"
                );
            }
        }
    }

    /// #618: `TAGURU_REPLICATE_URL` set but blank is the same
    /// templating-accident trap `TAGURU_PUBLIC_URL` guards against —
    /// treated as disabled, not as a URL to parse.
    #[test]
    fn from_env_treats_a_blank_url_as_disabled() {
        let _env = crate::ship::test_support::ScopedReplicateEnv::new([Some("   "), None]);
        assert!(ReplicateConfig::from_env().is_none());
    }

    /// #618: unset entirely is the ordinary "shipping off" case, not
    /// the blank-but-present trap above — both must return `None`, but
    /// only one of them logs a warning about it.
    #[test]
    fn from_env_returns_none_when_unset() {
        let _env = crate::ship::test_support::ScopedReplicateEnv::new([None, None]);
        assert!(ReplicateConfig::from_env().is_none());
    }

    /// #618: an interval below the busy-poll floor is raised to it,
    /// loudly — never silently honored.
    #[test]
    fn from_env_floors_a_tiny_interval() {
        let _env = crate::ship::test_support::ScopedReplicateEnv::new([
            Some("file:///tmp/wherever"),
            Some("1"),
        ]);
        let config = ReplicateConfig::from_env().expect("a non-blank URL enables shipping");
        assert_eq!(config.interval, Duration::from_millis(100));
    }

    /// #618: an unparseable interval falls back to the documented
    /// default (1000ms) — `env_number`'s own contract, pinned here at
    /// the call site that actually matters for replication.
    #[test]
    fn from_env_falls_back_to_the_default_interval_on_a_bad_value() {
        let _env = crate::ship::test_support::ScopedReplicateEnv::new([
            Some("file:///tmp/wherever"),
            Some("not-a-number"),
        ]);
        let config = ReplicateConfig::from_env().expect("a non-blank URL enables shipping");
        assert_eq!(config.interval, Duration::from_millis(1000));
    }
}
