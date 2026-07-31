// assembleEvidence (#216, #305, #306): request construction and the
// return shape, against the hand-rolled fetch stub. The golden-fixture
// unwrap itself is covered by wire-contract.test.ts; this file covers
// what that one cannot — option normalization and omission.

import { describe, expect, it } from "vitest";

import type { EvidencePackage } from "../../src/models.js";
import { okBody, stubClient, type StubRequest } from "./stub.js";

const MIXED_LANES_RESULT: EvidencePackage = {
  budget: {
    bytes_used: 1082,
    items_used: 2,
    limits: { max_bytes: 65536, max_items: 40, max_tokens: 4000 },
    tokens_used: 288,
  },
  citations: [
    {
      citation: { section: null, source: "docs/kura.md", text: "青嶺酒造は雲居県霧沢町の蔵元である。" },
      paragraph: 0,
      source: "docs/kura.md",
    },
  ],
  items: [
    {
      association: {
        attributions: [
          { count: 1, paragraph: 0, section: null, source: "docs/kura.md", weight: 1.0 },
        ],
        count: 1,
        label: "杜氏",
        object: "高瀬",
        subject: "青嶺酒造",
        weight: 1.0,
      },
      bytes: 512,
      candidate_id: "association-1",
      citation_refs: [{ paragraph: 0, source: "docs/kura.md" }],
      corroboration: {
        attributions: [{ paragraph: 0, source: "docs/kura.md" }],
        sources: ["docs/kura.md"],
      },
      estimated_tokens: 132,
      fused_rank: 1,
      kind: "association",
      lane_ranks: [{ lane: "graph_activate", rank: 1 }],
    },
    {
      bytes: 383,
      candidate_id: "passage-1",
      citation_refs: [],
      estimated_tokens: 103,
      fused_rank: 2,
      kind: "passage",
      lane_ranks: [{ lane: "passage_bm25", rank: 1 }],
      passage: {
        lanes: { bm25: { rank: 1, score: 0.86304635 } },
        paragraph: 0,
        score: 0.86304635,
        source: "docs/kura.md",
        text: "青嶺酒造は雲居県霧沢町の蔵元である。",
      },
    },
  ],
  omitted: [],
  omitted_by_reason: {},
  omitted_total: 0,
  plan: {
    lanes: {
      activate: { ran: true },
      citations: { ran: true },
      communities: { ran: false, reason: "include_communities was false" },
      passages: { ran: true },
      query: { ran: false, reason: "no 'labels' given" },
      resolve: { ran: true },
    },
    reranker: { configured: false, ran: false },
    selection: { contradiction_groups: 0, dedup_dropped: 0, diversity_tier_width: 10 },
  },
};

describe("assembleEvidence", () => {
  it("posts to the evidence route with a bare origins call", async () => {
    const calls: StubRequest[] = [];
    const client = stubClient((req) => {
      calls.push(req);
      return okBody(MIXED_LANES_RESULT);
    });

    const pkg = await client.context("sake").assembleEvidence("青嶺酒造");

    expect(calls).toHaveLength(1);
    expect(calls[0]!.path).toBe("/contexts/sake/evidence");
    expect(JSON.parse(calls[0]!.body ?? "{}")).toEqual({ origins: "青嶺酒造" });
    expect(pkg.items).toHaveLength(2);
    expect(pkg.items[0]!.kind).toBe("association");
    expect(pkg.items[0]!.corroboration?.sources).toEqual(["docs/kura.md"]);
    expect(pkg.items[1]!.kind).toBe("passage");
    expect(pkg.budget.limits.max_items).toBe(40);
    expect(pkg.plan.lanes.communities.ran).toBe(false);
    expect(pkg.plan.reranker.configured).toBe(false);
  });

  it("omits unset options from the body", async () => {
    const calls: StubRequest[] = [];
    const client = stubClient((req) => {
      calls.push(req);
      return okBody(MIXED_LANES_RESULT);
    });

    await client.context("sake").assembleEvidence(["青嶺酒造", "高瀬"]);

    expect(JSON.parse(calls[0]!.body ?? "{}")).toEqual({ origins: ["青嶺酒造", "高瀬"] });
  });

  it("passes budget and rerank through as nested objects", async () => {
    const calls: StubRequest[] = [];
    const client = stubClient((req) => {
      calls.push(req);
      return okBody(MIXED_LANES_RESULT);
    });

    await client.context("sake").assembleEvidence("青嶺酒造", {
      include_communities: true,
      budget: { max_items: 10, max_bytes: 8192 },
      rerank: { model: "bge-reranker" },
    });

    const body = JSON.parse(calls[0]!.body ?? "{}");
    expect(body.include_communities).toBe(true);
    expect(body.budget).toEqual({ max_items: 10, max_bytes: 8192 });
    expect(body.rerank).toEqual({ model: "bge-reranker" });
  });

  it("decodes a minimal response with every optional field omitted", async () => {
    const minimalResult: EvidencePackage = {
      budget: {
        bytes_used: 10,
        items_used: 1,
        limits: { max_bytes: 65536, max_items: 40, max_tokens: 4000 },
        tokens_used: 3,
      },
      citations: [],
      items: [
        {
          bytes: 10,
          candidate_id: "passage-1",
          citation_refs: [],
          estimated_tokens: 3,
          fused_rank: 1,
          kind: "passage",
          lane_ranks: [{ lane: "passage_bm25", rank: 1 }],
          passage: {
            lanes: { bm25: { rank: 1, score: 1.0 } },
            paragraph: 0,
            score: 1.0,
            source: "s",
            text: "hello",
          },
          // no corroboration, no contradicts — both omitted on the wire.
        },
      ],
      omitted: [],
      omitted_by_reason: {},
      omitted_total: 0,
      plan: {
        lanes: {
          activate: { ran: true },
          citations: { ran: true },
          communities: { ran: false, reason: "include_communities was false" },
          passages: { ran: true },
          query: { ran: false, reason: "no 'labels' given" },
          resolve: { ran: true },
        },
        reranker: { configured: false, ran: false },
        // no model, no reason on reranker — both omitted.
        selection: { contradiction_groups: 0, dedup_dropped: 0, diversity_tier_width: 10 },
      },
    };
    const client = stubClient(() => okBody(minimalResult));

    const pkg = await client.context("sake").assembleEvidence("青嶺酒造");

    expect(pkg.items[0]!.corroboration).toBeUndefined();
    expect(pkg.items[0]!.contradicts).toBeUndefined();
    expect(pkg.items[0]!.association).toBeUndefined();
    expect(pkg.items[0]!.community).toBeUndefined();
    expect(pkg.plan.reranker.model).toBeUndefined();
    expect(pkg.plan.reranker.reason).toBeUndefined();
  });
});
