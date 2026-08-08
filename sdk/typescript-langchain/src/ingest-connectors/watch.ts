/**
 * The local directory watcher (ADR 0007 §13's deferred third connector,
 * issue #414): repeated, checkpoint-idempotent passes of the existing
 * object-storage sync over a `file://` directory tree. The mechanical
 * mirror of the Python `taguru_langchain.ingest_connectors.watch` module
 * (issue #415).
 *
 * Deliberately a POLLING loop over `syncObjectStorage` (s3.ts) with a
 * `FileObjectStore` (objectstore.ts), not a native filesystem-event
 * subscription: inotify/FSEvents/ReadDirectoryChangesW each behave
 * differently (and all of them miss events across network mounts and
 * container bind mounts), a dependency like `chokidar` would be this
 * package's first non-parser runtime dependency, and the sync it would
 * trigger is already idempotent — an event's only value here would be
 * shaving latency off a poll interval. The two-layer checkpoint makes the
 * poll cheap in the steady state: an unchanged file is skipped on its
 * `(size, mtime)` listing fingerprint without ever being opened, so a pass
 * over a quiet tree costs one directory walk. Each pass IS a
 * `syncObjectStorage` run, verbatim — same source ids
 * (`file:///abs/path/…`), same events and `RunReport` (observability.ts),
 * same deletion policy (default `"report"`: a deleted file is counted,
 * never retracted, #217's no-destructive-sync-by-default requirement), same
 * `dryRun`.
 *
 * One deliberate deviation from the Python original, forced by this SDK's
 * own, already-established async idiom rather than chosen here:
 *
 * - **A plain `Callable[[], bool] | threading.Event` becomes `ShouldStop`
 *   (ingest.ts) — `(() => boolean) | AbortSignal`.** `AbortSignal` is this
 *   SDK's `threading.Event` analogue: `signal.aborted` wakes the moment
 *   `abort()` is called, exactly like `Event.wait` needs no polling at all,
 *   so the inter-pass wait races a `setTimeout` against the signal's own
 *   `"abort"` event (cleaning the listener/timer up either way) instead of
 *   polling. A plain predicate (or no `shouldStop` at all) is polled every
 *   `STOP_POLL_MS`, same as the Python original's plain-callable branch.
 * - **`watchDirectory` is `async`, returning `Promise<AsyncGenerator<
 *   RunReport>>`, not itself an async generator.** Python's eager
 *   validation (interval check, `FileObjectStore(directory)` — a
 *   synchronous constructor) runs in a plain (non-generator) function body
 *   before `_passes()` (the actual generator) is even constructed, so a
 *   missing directory or a negative interval fails at the call site, not on
 *   the caller's first `next()`. This port's `FileObjectStore.open`
 *   (objectstore.ts) is an async factory — a constructor cannot `await` —
 *   so the equivalent eager checks live in `watchDirectory`'s own `async`
 *   body, ahead of the nested `async function* passes()` it returns: a
 *   caller who never awaits the returned promise never learns of a bad
 *   directory, but one who does (as every test here does) sees the same
 *   fail-before-any-pass behavior the Python original guarantees
 *   unconditionally.
 */

import type { CheckpointStore } from "../checkpoints.js";
import type { ShouldStop, TaguruIngester } from "../ingest.js";
import { normalizeStop } from "./bridge.js";
import { FileObjectStore } from "./objectstore.js";
import { RunReport, type SourceEventSinkTarget } from "./observability.js";
import { type DeletionPolicy, S3Connector, syncObjectStorage } from "./s3.js";

// Re-exported so a caller need not reach into observability.ts just to name
// this generator's own element type.
export { RunReport };

// How often a plain (non-`AbortSignal`) `shouldStop` is re-checked while
// waiting out the interval between passes. An `AbortSignal` needs no
// polling at all — its own `"abort"` event wakes the wait the moment it
// fires — so this granularity only bounds the stop latency of the plain-
// predicate (or no-`shouldStop`-at-all) form.
const STOP_POLL_MS = 200;

/**
 * Watches `directory` by running one `syncObjectStorage` (s3.ts) pass,
 * yielding its `RunReport` (observability.ts), waiting `intervalSecs`, and
 * repeating — until `shouldStop` says stop or the caller simply stops
 * consuming the iterator.
 *
 * An async generator on purpose: the caller owns the loop. `for await (const
 * report of await watchDirectory(...))` observes every pass as it
 * completes, `break` ends the watch, and nothing accumulates — a watcher
 * that returned `RunReport[]` would grow without bound over a long-running
 * watch. `shouldStop` exists for the cooperative case (another part of the
 * program deciding to stop a consumer awaiting `next()`): it is checked
 * before each pass, DURING each pass (`syncObjectStorage` checks it between
 * objects), and continuously through the inter-pass wait — an `AbortSignal`
 * ends the wait the moment it is aborted.
 *
 * The directory must exist when this is called (the same no-silent-`mkdir`
 * contract `FileObjectStore` holds) and for the watch's lifetime — a tree
 * that vanishes mid-watch raises out of the generator rather than being
 * quietly treated as "everything was deleted."
 *
 * `eventsOut` is handed to every pass verbatim; note a *path* is opened
 * fresh (truncated) per pass, so over a multi-pass watch only the last
 * pass's sidecar survives — pass an open handle instead to accumulate every
 * pass's events in one stream.
 *
 * Example:
 * ```ts
 * const controller = new AbortController();
 * for await (const report of await watchDirectory("manuals/", {
 *   ingester,
 *   checkpoints,
 *   intervalSecs: 30,
 *   shouldStop: controller.signal,
 * })) {
 *   log.info(`pass done: imported=${report.imported} unchanged=${report.unchanged}`);
 * }
 * ```
 */
export async function watchDirectory(
  directory: string,
  options: {
    ingester: TaguruIngester;
    checkpoints: CheckpointStore;
    intervalSecs?: number;
    connector?: S3Connector;
    deletionPolicy?: DeletionPolicy;
    dryRun?: boolean;
    shouldStop?: ShouldStop;
    eventsOut?: SourceEventSinkTarget;
  },
): Promise<AsyncGenerator<RunReport, void, unknown>> {
  const intervalSecs = options.intervalSecs ?? 30.0;
  if (intervalSecs < 0) {
    throw new Error("intervalSecs must be >= 0");
  }
  // Eager, not inside the generator body: a missing directory should fail
  // at the call site (once this promise is awaited), not on the first
  // `next()` — see this module's own doc comment for why an `async`
  // function, rather than an async generator function itself, is what
  // makes that possible here.
  const store = await FileObjectStore.open(directory);
  const stop = normalizeStop(options.shouldStop);
  const wait = stopAwareWait(options.shouldStop, stop);

  async function* passes(): AsyncGenerator<RunReport, void, unknown> {
    while (!stop()) {
      yield await syncObjectStorage(store, "", {
        ingester: options.ingester,
        checkpoints: options.checkpoints,
        connector: options.connector,
        deletionPolicy: options.deletionPolicy,
        dryRun: options.dryRun,
        shouldStop: options.shouldStop,
        eventsOut: options.eventsOut,
      });
      if (!(await wait(intervalSecs))) {
        return;
      }
    }
  }

  return passes();
}

/**
 * Returns `wait(seconds) -> Promise<keepGoing>`: waits out the interval but
 * resolves to `false` the moment the stop condition fires. An
 * `AbortSignal` wakes natively (no polling); a plain predicate (or no
 * `shouldStop` at all) is polled every `STOP_POLL_MS`.
 */
function stopAwareWait(
  shouldStop: ShouldStop | undefined,
  stop: () => boolean,
): (seconds: number) => Promise<boolean> {
  if (shouldStop instanceof AbortSignal) {
    return (seconds: number) => waitForAbort(shouldStop, seconds);
  }

  return (seconds: number) => pollingWait(stop, seconds);
}

/** Races a `setTimeout` against `signal`'s own `"abort"` event, cleaning up
 * whichever one did NOT fire so neither a dangling timer nor a dangling
 * listener outlives this one wait. */
function waitForAbort(signal: AbortSignal, seconds: number): Promise<boolean> {
  if (signal.aborted) {
    return Promise.resolve(false);
  }
  if (seconds <= 0) {
    return Promise.resolve(true);
  }
  return new Promise<boolean>((resolve) => {
    const onAbort = (): void => {
      clearTimeout(timer);
      resolve(false);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve(true);
    }, seconds * 1000);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

async function pollingWait(
  stop: () => boolean,
  seconds: number,
): Promise<boolean> {
  const deadline = performance.now() + seconds * 1000;
  for (;;) {
    if (stop()) {
      return false;
    }
    const remaining = deadline - performance.now();
    if (remaining <= 0) {
      return true;
    }
    await new Promise<void>((resolve) =>
      setTimeout(resolve, Math.min(STOP_POLL_MS, remaining)),
    );
  }
}
