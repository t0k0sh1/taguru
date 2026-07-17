/** TaguruRetriever against the routed fake server — mirrors the Python suite. */

import { describe, expect, it } from "vitest";

import { TaguruRetriever } from "../../src/retrievers.js";
import { FakeServer } from "./stub.js";

const make = (server: FakeServer, fields: Record<string, unknown> = {}) =>
  new TaguruRetriever({ context: "sake", client: server.client(), ...fields });

describe("TaguruRetriever", () => {
  it("merges both lanes and dedups on (source, paragraph)", async () => {
    const server = new FakeServer();
    const documents = await make(server).invoke("青嶺酒造");

    const merged = documents.filter((d) => d.metadata["lane"] === "graph+text");
    expect(merged).toHaveLength(1);
    const top = merged[0]!;
    expect(top.pageContent).toBe("杜氏は高瀬である。");
    expect(top.metadata["source"]).toBe("docs/aomine.md");
    expect(top.metadata["paragraph"]).toBe(1);
    expect(top.metadata["section"]).toBe("人物");
    expect((top.metadata["associations"] as Array<{ label: string }>)[0]!.label).toBe("杜氏");
    expect(documents[0]).toBe(top); // dual-lane evidence outranks single-lane

    const facts = documents.filter((d) => d.metadata["paragraph"] === null);
    expect(facts).toHaveLength(1);
    expect(facts[0]!.pageContent).toBe("青嶺酒造 創業年 1907年");
    expect(facts[0]!.metadata["source"]).toBe("口伝");

    const textOnly = documents.filter((d) => d.metadata["lane"] === "text");
    expect(textOnly.map((d) => d.metadata["source"])).toEqual(["docs/other.md"]);
  });

  it("honors lane toggles", async () => {
    const graphOnly = new FakeServer();
    const graphDocs = await make(graphOnly, { include_text: false }).invoke("青嶺酒造");
    expect(graphDocs.every((d) => String(d.metadata["lane"]).includes("graph"))).toBe(true);
    expect(graphOnly.calls.some(([path]) => path.endsWith("/sources/search"))).toBe(false);

    const textOnly = new FakeServer();
    const textDocs = await make(textOnly, { include_graph: false }).invoke("青嶺酒造");
    expect(textDocs.every((d) => d.metadata["lane"] === "text")).toBe(true);
    expect(textOnly.calls.some(([path]) => path.endsWith("/resolve"))).toBe(false);
  });

  it("serves the text lane when the cue does not resolve", async () => {
    const server = new FakeServer();
    const documents = await make(server).invoke("無関係な話題");
    expect(documents.length).toBeGreaterThan(0);
    expect(documents.every((d) => d.metadata["lane"] === "text")).toBe(true);
  });

  it("truncates to k and can switch graph-only facts off", async () => {
    const server = new FakeServer();
    expect(await make(server, { k: 1 }).invoke("青嶺酒造")).toHaveLength(1);

    const noFacts = await make(new FakeServer(), { include_graph_only_facts: false }).invoke(
      "青嶺酒造",
    );
    expect(noFacts.every((d) => d.metadata["paragraph"] !== null)).toBe(true);
  });
});

describe("TaguruRetriever cross-context", () => {
  it("requires a target", () => {
    const server = new FakeServer();
    expect(() => new TaguruRetriever({ client: server.client() })).toThrow(/name a target/);
  });

  it("tags documents and rides one cross-context text call", async () => {
    const server = new FakeServer();
    const retriever = new TaguruRetriever({
      contexts: ["sake", "tea"],
      client: server.client(),
    });
    const documents = await retriever.invoke("青嶺酒造");

    // Every Document names the context it came from.
    const contexts = new Set(documents.map((d) => d.metadata["context"]));
    expect(contexts).toEqual(new Set(["sake", "tea"]));

    // The graph lane ran per context; the text lane rode the server's own
    // cross-context search — one top-level call naming both targets.
    const resolves = server.calls.filter(([path]) => path.endsWith("/resolve")).map(([p]) => p);
    expect(resolves).toEqual(["/contexts/sake/resolve", "/contexts/tea/resolve"]);
    const crossSearches = server.calls
      .filter(([path]) => path === "/sources/search")
      .map(([, body]) => body);
    expect(crossSearches).toEqual([{ contexts: ["sake", "tea"], query: "青嶺酒造", limit: 5 }]);
  });

  it("resolves groups to members, nested children included", async () => {
    const server = new FakeServer();
    const retriever = new TaguruRetriever({ groups: ["parent"], client: server.client() });
    const documents = await retriever.invoke("青嶺酒造");

    // parent reaches sake directly and tea through its child group.
    const fetched = server.calls.filter(([path]) => path.startsWith("/groups/")).map(([p]) => p);
    expect(new Set(fetched)).toEqual(new Set(["/groups/parent", "/groups/childg"]));
    const crossSearches = server.calls
      .filter(([path]) => path === "/sources/search")
      .map(([, body]) => body);
    expect(crossSearches).toEqual([{ contexts: ["sake", "tea"], query: "青嶺酒造", limit: 5 }]);
    expect(new Set(documents.map((d) => d.metadata["context"]))).toEqual(new Set(["sake", "tea"]));
  });

  it("still resolves the groups it can when one group fails to fetch", async () => {
    const server = new FakeServer();
    const retriever = new TaguruRetriever({
      groups: ["parent", "no-such-group"],
      client: server.client(),
    });
    const documents = await retriever.invoke("青嶺酒造");

    // parent's members (sake, tea) still come back even though the
    // sibling group 404s.
    expect(new Set(documents.map((d) => d.metadata["context"]))).toEqual(new Set(["sake", "tea"]));
  });

  it("keeps a healthy target's graph docs when another target's graph lane errors", async () => {
    const server = new FakeServer();
    server.failContexts.add("tea");
    const retriever = new TaguruRetriever({
      contexts: ["sake", "tea"],
      client: server.client(),
    });
    const documents = await retriever.invoke("青嶺酒造");

    // sake's graph lane never touched the failing context, so its docs
    // still show up.
    const graphDocs = documents.filter((d) => String(d.metadata["lane"]).includes("graph"));
    expect(graphDocs.length).toBeGreaterThan(0);
    expect(graphDocs.every((d) => d.metadata["context"] === "sake")).toBe(true);

    // tea's cross-context text hit isn't a per-context call, so it
    // still shows up despite tea's graph lane failing.
    expect(documents.some((d) => d.metadata["context"] === "tea")).toBe(true);
  });
});
