/**
 * `backoffDelay` itself, unmocked.
 *
 * `retry.test.ts` mocks `backoffDelay` to `() => 0` for the whole file (so
 * its retry-policy tests don't sleep for real) — that mock makes the real
 * implementation's own body permanently uncovered from that file's point of
 * view. This file imports the module fresh, without the mock, so the real
 * `Math.random() * Math.min(BACKOFF_CAP_SECS, BACKOFF_BASE_SECS * 2 ** attempt)`
 * body actually runs and gets checked.
 */
import { describe, expect, it, vi } from "vitest";

import { BACKOFF_BASE_SECS, BACKOFF_CAP_SECS, backoffDelay } from "../../src/retry.js";

describe("backoffDelay", () => {
  it("stays within [0, BACKOFF_BASE_SECS) for attempt 0", () => {
    // Deterministic across the full [0, 1) domain `Math.random()` can
    // draw from, rather than a probabilistic sample of it.
    for (const draw of [0, 0.25, 0.5, 0.75, 0.999999999]) {
      const spy = vi.spyOn(Math, "random").mockReturnValue(draw);
      try {
        const delay = backoffDelay(0);
        expect(delay).toBeGreaterThanOrEqual(0);
        expect(delay).toBeLessThan(BACKOFF_BASE_SECS);
      } finally {
        spy.mockRestore();
      }
    }
  });

  it("multiplies the random draw by the jitter ceiling, not divides by it", () => {
    const spy = vi.spyOn(Math, "random").mockReturnValue(0.4);
    try {
      // ceiling at attempt 0 is min(cap, base) = base = 0.5; 0.4 * 0.5 =
      // 0.2 — a `/` in place of `*` here would give 0.4 / 0.5 = 0.8, well
      // outside attempt 0's [0, base) range.
      expect(backoffDelay(0)).toBeCloseTo(0.2, 10);
    } finally {
      spy.mockRestore();
    }
  });

  it("grows with the attempt number, doubling the exponential base", () => {
    // Full jitter means any single draw can land anywhere in [0, cap), so
    // pin the DETERMINISTIC ceiling — Math.random() is stubbed to its
    // maximum-representable draw (just under 1) to read the exact upper
    // bound `Math.min(cap, base * 2**attempt)` back out.
    const spy = vi.spyOn(Math, "random").mockReturnValue(0.999999999);
    try {
      const attempt1 = backoffDelay(1);
      const attempt2 = backoffDelay(2);
      expect(attempt1).toBeCloseTo(BACKOFF_BASE_SECS * 2, 5);
      expect(attempt2).toBeCloseTo(BACKOFF_BASE_SECS * 4, 5);
      expect(attempt2).toBeGreaterThan(attempt1);
    } finally {
      spy.mockRestore();
    }
  });

  it("clamps to BACKOFF_CAP_SECS once the exponential base would exceed it", () => {
    const spy = vi.spyOn(Math, "random").mockReturnValue(0.999999999);
    try {
      // At a high enough attempt, base * 2**attempt dwarfs the cap — the
      // draw must clamp to (just under) BACKOFF_CAP_SECS, not keep growing
      // unboundedly with the attempt number.
      const delay = backoffDelay(20);
      expect(delay).toBeCloseTo(BACKOFF_CAP_SECS, 5);
      expect(delay).toBeLessThanOrEqual(BACKOFF_CAP_SECS);
    } finally {
      spy.mockRestore();
    }
  });
});
