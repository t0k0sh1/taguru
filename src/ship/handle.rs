//! The serve-side handle onto the background shipping task: [`spawn`]
//! boots it, [`ShipperHandle::shutdown`] stops it and waits for its
//! final cycle.

use super::*;

/// The serve-side handle: signals the shipper to stop and waits for
/// its final cycle, so the post-drain flush (`main`'s shutdown runs
/// `flush_dirty` first) reaches the bucket before the process exits.
pub(crate) struct ShipperHandle {
    stop: tokio::sync::watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl ShipperHandle {
    pub(crate) async fn shutdown(self) {
        let _ = self.stop.send(true);
        if let Err(error) = self.task.await {
            tracing::warn!(%error, "replication task did not shut down cleanly");
        }
    }
}

/// Boots the shipper as one background task: claim a generation, then
/// poll until told to stop (one final cycle after the signal drains
/// the shutdown flush) — or until fenced, which stops it for good.
///
/// Claiming inside the task keeps a slow or unreachable bucket off
/// the serve path: the server binds and answers while the claim
/// retries; every failure is surfaced through the metric and the log
/// rather than a refused boot. The ONE thing a boot refuses on is a
/// URL that does not even parse (`open_store` in the caller) — a typo
/// should fail loudly at start, an outage should not.
pub(crate) fn spawn(
    store: Arc<dyn ObjectStore>,
    root: StorePath,
    replicate: ReplicateConfig,
    data_dir: PathBuf,
    progress: Arc<ShipProgress>,
    state: AppState,
    hydration: Option<Arc<crate::hydrate::Hydrator>>,
) -> ShipperHandle {
    let ReplicateConfig { url, interval } = replicate;
    let (stop, mut stopped) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(async move {
        let mut shipper = loop {
            match Shipper::claim(
                Arc::clone(&store),
                root.clone(),
                url.clone(),
                data_dir.clone(),
                Arc::clone(&progress),
                state.clone(),
                hydration.clone(),
            )
            .await
            {
                Ok(shipper) => break shipper,
                Err(error) => {
                    state.metrics().record_replication_error();
                    tracing::warn!(%error, "replication fence claim failed; retrying");
                    tokio::select! {
                        _ = tokio::time::sleep(interval.max(Duration::from_secs(1))) => {}
                        _ = stopped.changed() => return,
                    }
                }
            }
        };
        loop {
            let stopping = *stopped.borrow();
            match shipper.cycle().await {
                Ok(_) => {}
                Err(ShipError::Fenced { newer_generation }) => {
                    // Fail-stop, permanently: the successor owns the
                    // bucket. The serve path is untouched — it keeps
                    // answering from its local truth — but nothing
                    // more leaves this process, and both the metric
                    // and the audit line say so.
                    state.metrics().record_replication_fenced();
                    tracing::error!(
                        target: "taguru::audit",
                        generation = shipper.generation,
                        newer_generation,
                        "replication FENCED: a newer writer claimed the bucket — shipping \
                         stopped for good; this server keeps serving its local data (restart \
                         it to contest the claim, after making sure only one writer should \
                         exist)",
                    );
                    return;
                }
                Err(ShipError::Io(error)) => {
                    // Transient by assumption: the cursors did not
                    // advance past anything unshipped, so the next
                    // cycle retries exactly where this one failed.
                    tracing::warn!(%error, "replication cycle failed; will retry");
                }
            }
            if stopping {
                // The final cycle above drained the post-flush state;
                // saying "cleanly stopped" is now true, and it is what
                // lets the next writer against this bucket start
                // without a takeover handshake.
                shipper.retire_generation().await;
                return;
            }
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                _ = stopped.changed() => {}
            }
        }
    });
    ShipperHandle { stop, task }
}
