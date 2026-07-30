/**
 * ADR 0005 §5/§9.1: every enum-like wire field must be open, except
 * `AliasEntry.namespace` (never wire-decoded — deliberately kept
 * closed, see its doc comment in `models.ts`). These are erased at
 * runtime, so a compile-time check is the only kind that can prove
 * the widening landed — `tsconfig.json`'s `"include": ["src", "tests"]`
 * means `npm run typecheck` enforces both the widening and the
 * deliberate exception below.
 */

import { describe, expect, it } from "vitest";

import type {
  AliasEntry,
  GroupImportOutcome,
  LexicalExplain,
  NearestResolution,
  ResolveExplanation,
  ResolveRanking,
  SearchExplanation,
  TieredResolution,
} from "../../src/models.js";

describe("open enum-like unions", () => {
  it("accept a value a newer server may add, at compile time and at runtime", () => {
    // Each assignment below only needs to compile; TypeScript would
    // reject a value outside the known literals if the field were
    // still closed. `satisfies` keeps the object shape checked too.
    const tier: TieredResolution["tier"] = "future_tier";
    const kind: TieredResolution["kind"] = "future_kind";
    const nearestKind: NearestResolution["kind"] = "future_kind";
    const lexicalKind: LexicalExplain["kind"] = "future_kind";
    const rankingTier: ResolveRanking["tier"] = "future_tier";
    const resolveVerdict: ResolveExplanation["verdict"] = "future_verdict";
    const expectedKind: ResolveExplanation["expected_kind"] = "future_kind";
    const searchVerdict: SearchExplanation["verdict"] = "future_verdict";
    const outcome: GroupImportOutcome["outcome"] = "future_outcome";

    for (const value of [
      tier,
      kind,
      nearestKind,
      lexicalKind,
      rankingTier,
      resolveVerdict,
      expectedKind,
      searchVerdict,
      outcome,
    ]) {
      // No runtime validation exists anywhere in this SDK (ADR 0005
      // §2.5) — the point of Open<T> is that nothing throws here.
      expect(typeof value).toBe("string");
    }
  });

  it("AliasEntry.namespace stays deliberately closed — never wire-decoded", () => {
    const concept: AliasEntry["namespace"] = "concept";
    const label: AliasEntry["namespace"] = "label";
    // @ts-expect-error AliasEntry.namespace is SDK-constructed
    // (iterAliases synthesizes it from AliasPage's two fixed keys),
    // so a server can never surprise this SDK with a value it didn't
    // itself mint — widening it would forfeit real type safety for
    // zero compatibility benefit.
    const bad: AliasEntry["namespace"] = "group";
    expect([concept, label, bad]).toEqual(["concept", "label", "group"]);
  });
});
