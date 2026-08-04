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
