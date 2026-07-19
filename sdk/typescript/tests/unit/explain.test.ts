import { describe, expect, it } from "vitest";

import type { ResolveExplanation, SearchExplanation } from "../../src/models.js";
import { okBody, stubClient, type StubRequest } from "./stub.js";

const RESOLVE_EXPLANATION = {
  verdict: "below_floor",
  summary: "青嶺酒造 scored 0.42, below the 0.6 floor in effect.",
  cue: "青嶺",
  expected: "青嶺酒造",
  in_vocabulary: true,
  canonical: "青嶺酒造",
  expected_kind: "exact",
  lexical: { score: 0.42, kind: "containment", floor: 0.6, confident: false },
  semantic: { entered: false, reason: "lexical tier not confident" },
  ranking: { rank: 3, tier: "lexical", score: 0.42, limit: 5, served: false, limit_to_reach: 3 },
};

const SEARCH_EXPLANATION = {
  verdict: "no_term_overlap",
  summary: "query says 酒造, the paragraph spells 酒蔵.",
  source: "docs/aomine.md",
  paragraph: 1,
  paragraphs: 3,
  paragraph_named: true,
  query_terms: ["酒造"],
  paragraph_terms: ["酒蔵"],
  bm25: { score: 0, terms: [{ term: "酒造", tf: 0, df: 5, idf: 1.2, contribution: 0 }] },
  vector: { ran: false, reason: "no embedding provider configured" },
  ranking: { fused: false, ranked: 0, limit: 5, served: false },
};

describe("resolve/search explain endpoints", () => {
  it("explainResolve posts cue and expected, and decodes the nested evidence", async () => {
    let seen: StubRequest | undefined;
    const client = stubClient((req) => {
      seen = req;
      return okBody(RESOLVE_EXPLANATION);
    });
    const verdict: ResolveExplanation = await client
      .context("aomine")
      .explainResolve("青嶺", "青嶺酒造", { dice_floor: 0.6 });

    expect(seen!.method).toBe("POST");
    expect(seen!.path).toBe("/contexts/aomine/resolve/explain");
    expect(JSON.parse(seen!.body!)).toEqual({ cue: "青嶺", expected: "青嶺酒造", dice_floor: 0.6 });

    expect(verdict.verdict).toBe("below_floor");
    expect(verdict.in_vocabulary).toBe(true);
    expect(verdict.lexical?.score).toBe(0.42);
    expect(verdict.lexical?.confident).toBe(false);
    expect(verdict.semantic?.entered).toBe(false);
    expect(verdict.ranking?.limit_to_reach).toBe(3);
  });

  it("explainResolveLabel routes to the label endpoint with only the required args", async () => {
    let seen: StubRequest | undefined;
    const client = stubClient((req) => {
      seen = req;
      return okBody(RESOLVE_EXPLANATION);
    });
    await client.context("aomine").explainResolveLabel("杜氏", "杜氏長");

    expect(seen!.path).toBe("/contexts/aomine/resolve_label/explain");
    // No overrides passed → dropUndefined leaves only the two required args.
    expect(JSON.parse(seen!.body!)).toEqual({ cue: "杜氏", expected: "杜氏長" });
  });

  it("explainSearchPassages posts query and source, and decodes the lane evidence", async () => {
    let seen: StubRequest | undefined;
    const client = stubClient((req) => {
      seen = req;
      return okBody(SEARCH_EXPLANATION);
    });
    const verdict: SearchExplanation = await client
      .context("aomine")
      .explainSearchPassages("青嶺酒造の酒造", "docs/aomine.md", {
        paragraph: 1,
        semantic_floor: 0.2,
      });

    expect(seen!.path).toBe("/contexts/aomine/sources/search/explain");
    expect(JSON.parse(seen!.body!)).toEqual({
      query: "青嶺酒造の酒造",
      source: "docs/aomine.md",
      paragraph: 1,
      semantic_floor: 0.2,
    });

    expect(verdict.verdict).toBe("no_term_overlap");
    expect(verdict.query_terms).toEqual(["酒造"]);
    expect(verdict.bm25?.terms[0]!.df).toBe(5);
    expect(verdict.vector?.ran).toBe(false);
  });
});
