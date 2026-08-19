//! Cooperative stop handling (issue #179): the SIGINT/SIGTERM listener
//! thread (`StopSignal`) and the platform-specific thread-local signal
//! blocking helpers that keep tokio's runtime from racing the main
//! thread for a delivered signal.

use super::*;

/// Blocks SIGINT/SIGTERM on the calling thread (issue #213). Signal
/// masks are inherited by every thread spawned afterward, so calling
/// this once at the very top of [`run`] — before the chat client or
/// any other thread exists — keeps the delivery-eligible signal mask
/// clear everywhere except [`StopSignal`]'s own dedicated listener
/// thread, which unblocks it for itself alone.
///
/// Without this, a signal delivered while the main thread is inside a
/// blocking socket read can interrupt that read with `EINTR` — on
/// Linux, a read on a socket with `SO_RCVTIMEO` set (which is how
/// `ureq` implements its own timeout) is never auto-restarted by
/// `SA_RESTART`, regardless of how the handler is installed (see
/// signal(7)). `ureq` then treats the interrupted read as a transport
/// failure and silently reconnects, resending the in-flight chunk's
/// whole request on a brand new connection — invisible to the
/// cooperative-stop design, which only ever meant to check the flag
/// between chunks, never abandon one mid-call. Confirmed via `strace`
/// against a stub that holds the retried connection open: without this
/// fix the process hangs on the retry indefinitely instead of finishing
/// the original chunk and stopping cleanly (issue #213).
/// `pub(crate)` so `benchmark`'s own main thread — which spawns
/// [`StopSignal`]'s listener exactly as `extract` does — clears the
/// same signal mask before any child process or other thread exists,
/// instead of reimplementing this call.
#[cfg(unix)]
pub(crate) fn block_stop_signals_on_this_thread() {
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
    }
}

#[cfg(not(unix))]
pub(crate) fn block_stop_signals_on_this_thread() {}

/// Undoes [`block_stop_signals_on_this_thread`] for the calling thread
/// only — [`StopSignal::install`]'s dedicated listener thread calls
/// this so it alone remains eligible to receive the raw signal (some
/// thread must be, or the kernel has nowhere to deliver it and
/// `signal-hook`/`tokio::signal`'s handler never runs).
#[cfg(unix)]
pub(super) fn unblock_stop_signals_on_this_thread() {
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
    }
}

#[cfg(not(unix))]
pub(super) fn unblock_stop_signals_on_this_thread() {}

/// Issue #179's cooperative stop, checked between chunks/documents
/// rather than interrupting a call in flight. `extract` otherwise never
/// starts a runtime at all (`run`'s own SAFETY comment); this is the one
/// exception, confined to a dedicated background thread that does
/// nothing but wait on a signal — reusing `tokio`'s already-a-workspace-
/// dependency "signal" feature (the server's `shutdown_signal` depends
/// on the same feature) instead of adding a new crate.
///
/// Mirrors `main.rs`'s `shutdown_signal`/`TerminateSignals` two-stage
/// semantics exactly: the first SIGINT/SIGTERM sets `requested` and
/// prints a message; the second exits immediately with code 130 (128 +
/// SIGINT, the same shell convention). A registration failure (rare;
/// e.g. the signal is otherwise blocked) leaves `requested` permanently
/// false and the process's default signal disposition untouched — a
/// bare Ctrl+C still terminates the process the old way, just without
/// the cooperative message or a chance to checkpoint the in-flight
/// chunk.
///
/// `pub(crate)` so `benchmark::run` — a second, longer-running command
/// that spawns `extract` children of its own and needs the identical
/// checked-between-units stop discipline (ADR 0003 §6) — installs the
/// same listener instead of a second implementation. `prefix` (e.g.
/// `"extract"`, `"benchmark"`) names the caller in every message this
/// prints, so a mixed-process log tells the two apart.
pub(crate) struct StopSignal {
    pub(super) requested: Arc<AtomicBool>,
}

impl StopSignal {
    pub(crate) fn install(prefix: &'static str) -> Self {
        let requested = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&requested);
        std::thread::spawn(move || {
            unblock_stop_signals_on_this_thread();
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
            else {
                return;
            };
            runtime.block_on(async {
                let mut signals = StopSignals::install();
                // Printed once registration completes, so a test (or an
                // operator scripting around this) can wait for it
                // instead of guessing a startup margin before sending a
                // signal — a signal sent earlier hits the process's
                // default disposition rather than this cooperative path.
                eprintln!("taguru: {prefix}: stop signal handlers installed");
                signals.recv().await;
                flag.store(true, Ordering::SeqCst);
                eprintln!(
                    "taguru: {prefix}: stopping after the current chunk — press Ctrl+C again \
                     to stop immediately"
                );
                signals.recv().await;
                eprintln!("taguru: {prefix}: second stop signal received — exiting immediately");
                // 128 + SIGINT(2), the shell convention for
                // signal-terminated — same as shutdown_signal's.
                std::process::exit(130);
            });
        });
        Self { requested }
    }

    /// Never blocks: a plain atomic load, safe to call between every
    /// chunk and every document without measurable overhead.
    pub(crate) fn check(&self) -> bool {
        self.requested.load(Ordering::SeqCst)
    }
}

/// The extract-local twin of `main.rs`'s `TerminateSignals`: the
/// signal streams held for [`StopSignal`]'s whole background thread so
/// a second signal can be awaited without a re-registration gap that
/// could lose it — on Windows too (issue #730): `tokio::signal::
/// ctrl_c()` only delivers to a live listener, so awaiting a FRESH
/// future per recv would drop a Ctrl+C landing between the first
/// delivery and the second await.
pub(super) struct StopSignals {
    #[cfg(unix)]
    pub(super) interrupt: Option<tokio::signal::unix::Signal>,
    #[cfg(unix)]
    pub(super) terminate: Option<tokio::signal::unix::Signal>,
    #[cfg(windows)]
    pub(super) ctrl_c: Option<tokio::signal::windows::CtrlC>,
}

impl StopSignals {
    pub(super) fn install() -> Self {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            Self {
                interrupt: signal(SignalKind::interrupt()).ok(),
                terminate: signal(SignalKind::terminate()).ok(),
            }
        }
        #[cfg(windows)]
        {
            Self {
                ctrl_c: tokio::signal::windows::ctrl_c().ok(),
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            Self {}
        }
    }

    async fn recv(&mut self) {
        #[cfg(unix)]
        {
            async fn wait(stream: &mut Option<tokio::signal::unix::Signal>) {
                match stream {
                    Some(stream) => {
                        stream.recv().await;
                    }
                    None => std::future::pending().await,
                }
            }
            let Self {
                interrupt,
                terminate,
            } = self;
            tokio::select! {
                () = wait(interrupt) => {}
                () = wait(terminate) => {}
            }
        }
        #[cfg(windows)]
        {
            // Held for the thread's life, like the unix pair: the OS
            // handler is global, but delivery needs a LIVE listener —
            // a fresh `ctrl_c()` future per recv would drop a signal
            // landing in the gap between two awaits.
            match &mut self.ctrl_c {
                Some(stream) => {
                    stream.recv().await;
                }
                None => std::future::pending().await,
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
}
