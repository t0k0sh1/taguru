/** A routed fake Taguru server via an injected fetch — mirrors the Python conftest. */

import { Taguru } from "taguru";

export const AOMINE_ASSOCIATION = {
  subject: "青嶺酒造",
  label: "杜氏",
  object: "高瀬",
  weight: 1.0,
  count: 1,
  attributions: [
    { source: "docs/aomine.md", weight: 1.0, count: 1, paragraph: 1, section: null },
  ],
};

export const FACT_ONLY_ASSOCIATION = {
  subject: "青嶺酒造",
  label: "創業年",
  object: "1907年",
  weight: 1.0,
  count: 1,
  attributions: [{ source: "口伝", weight: 1.0, count: 1, paragraph: null, section: null }],
};

const GROUP_ROWS: Record<string, unknown> = {
  brewery: { name: "brewery", description: "蔵元一式", contexts: ["sake", "tea"], groups: [] },
  parent: { name: "parent", description: "", contexts: ["sake"], groups: ["childg"] },
  childg: { name: "childg", description: "", contexts: ["tea"], groups: [] },
};

const ok = (result: unknown): Response =>
  new Response(JSON.stringify({ result, status: "ok", time: 0.001 }), { status: 200 });

export class FakeServer {
  calls: Array<[string, unknown]> = [];
  imported: string[] = [];
  /** Context names whose every request should fail with a 500, to
   * exercise cross-context partial-failure handling. */
  failContexts = new Set<string>();

  fetch: typeof fetch = async (input, init) => {
    const url = new URL(String(input));
    const path = url.pathname;
    let body: unknown = null;
    if (typeof init?.body === "string") {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = init.body;
      }
    }
    this.calls.push([path, body]);
    const contextMatch = /^\/contexts\/([^/]+)\//.exec(path);
    if (contextMatch && this.failContexts.has(contextMatch[1]!)) {
      return new Response(
        JSON.stringify({ status: "error", code: "internal", error: "simulated failure", time: 0.001 }),
        { status: 500 },
      );
    }
    if (path.startsWith("/groups/")) {
      const row = GROUP_ROWS[path.slice("/groups/".length)];
      if (row === undefined) {
        return new Response(
          JSON.stringify({ status: "error", code: "no_group", error: "group not found", time: 0.001 }),
          { status: 404 },
        );
      }
      return ok(row);
    }
    if (path === "/sources/search") {
      // The cross-context search: one tagged hit per named context,
      // already rank-interleaved the way the server merges.
      const { contexts } = body as { contexts: string[] };
      return ok(
        contexts.map((name) => ({
          context: name,
          source: `docs/${name}.md`,
          paragraph: 0,
          score: 2.0,
          text: `${name} の段落。`,
          lanes: { bm25: { rank: 0, score: 2.0 } },
        })),
      );
    }
    if (path.endsWith("/resolve")) {
      const cue = (body as { cue: string }).cue;
      return ok(
        cue.includes("青嶺")
          ? [{ name: "青嶺酒造", score: 1.0, tier: "lexical", kind: "exact" }]
          : [],
      );
    }
    if (path.endsWith("/activate")) {
      return ok({
        total: 2,
        matches: [
          { strength: 0.9, path: ["青嶺酒造"], association: AOMINE_ASSOCIATION },
          { strength: 0.5, path: ["青嶺酒造"], association: FACT_ONLY_ASSOCIATION },
        ],
      });
    }
    if (path.endsWith("/citations")) {
      const { source } = body as { source: string };
      return ok({ text: "杜氏は高瀬である。", source, section: "人物" });
    }
    if (path.endsWith("/sources/search")) {
      return ok([
        {
          source: "docs/aomine.md",
          paragraph: 1,
          score: 3.0,
          text: "杜氏は高瀬である。",
          lanes: { bm25: { rank: 0, score: 3.0 } },
        },
        {
          source: "docs/other.md",
          paragraph: 0,
          score: 1.0,
          text: "別の文書の段落。",
          lanes: { bm25: { rank: 1, score: 1.0 } },
        },
      ]);
    }
    if (path.endsWith("/labels")) {
      return ok({ total: 2, labels: ["代表銘柄", "杜氏"] });
    }
    if (path === "/import") {
      this.imported.push(typeof init?.body === "string" ? init.body : "");
      return ok({
        batches: [
          {
            context: "sake",
            source: "docs/aomine.md",
            created: false,
            retracted: 0,
            associations: 2,
            aliases: 1,
            passage_stored: true,
            passage_dropped: false,
            questions_stored: 1,
            questions_dropped: 0,
            sections_stored: 0,
            sections_dropped: 0,
            association_paragraphs_dropped: 0,
          },
        ],
      });
    }
    if (path.endsWith("/embeddings/refresh")) {
      return new Response(
        JSON.stringify({ status: "error", error: "no provider", time: 0.001 }),
        { status: 501 },
      );
    }
    throw new Error(`unrouted path: ${path}`);
  };

  client(): Taguru {
    return new Taguru({ base_url: "http://test", api_key: "", fetch: this.fetch });
  }
}
