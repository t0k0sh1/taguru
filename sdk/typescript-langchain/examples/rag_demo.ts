/**
 * End-to-end RAG over Taguru with LangChain.js: ingest → retrieve → cite.
 * The TypeScript twin of ../python-langchain/examples/rag_demo.py.
 *
 * Runs self-contained: with no TAGURU_URL set it spawns a real server binary
 * (builds it with cargo on first run), and with no OPENAI_API_KEY it drives
 * the ingester with a deterministic fake chat model.
 *
 *     cd sdk/typescript-langchain && npx tsx examples/rag_demo.ts
 */

import { FakeListChatModel } from "@langchain/core/utils/testing";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";

import { Taguru } from "taguru";
import { serverBinary, spawnServer, type SpawnedServer } from "taguru/testing";

import { TaguruIngester, TaguruRetriever } from "../src/index.js";

const DOCS: Record<string, string> = {
  "docs/aomine.md": `青嶺酒造は1907年創業の架空の酒蔵である。代表銘柄は「青嶺」。

杜氏は高瀬である。高瀬は寒仕込みを重視する。

青嶺酒造は大量生産を行わない。`,
};

const FAKE_EXTRACTION = JSON.stringify({
  associations: [
    { subject: "青嶺酒造", label: "創業年", object: "1907年", weight: 1.0, paragraph: 0 },
    { subject: "青嶺酒造", label: "代表銘柄", object: "青嶺", weight: 1.0, paragraph: 0 },
    { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0, paragraph: 1 },
    { subject: "高瀬", label: "重視する", object: "寒仕込み", weight: 1.0, paragraph: 1 },
    { subject: "青嶺酒造", label: "行う", object: "大量生産", weight: -1.0, paragraph: 2 },
  ],
  aliases: [{ alias: "Aomine Brewery", canonical: "青嶺酒造", kind: "concept" }],
  questions: [{ paragraph: 1, question: "誰が酒を仕込んでいるの?" }],
});

async function makeLlm(): Promise<BaseChatModel> {
  if (process.env["OPENAI_API_KEY"]) {
    try {
      const { ChatOpenAI } = await import("@langchain/openai" as string);
      return new ChatOpenAI({ model: "gpt-4.1", temperature: 0 });
    } catch {
      console.log("(OPENAI_API_KEY set but @langchain/openai not installed — using the fake model)");
    }
  } else {
    console.log("(no OPENAI_API_KEY — driving the ingester with a canned fake model)");
  }
  return new FakeListChatModel({
    responses: Array(Object.keys(DOCS).length).fill(FAKE_EXTRACTION),
  });
}

async function main(): Promise<void> {
  let spawned: SpawnedServer | undefined;
  if (!process.env["TAGURU_URL"]) {
    console.log("(no TAGURU_URL — spawning a local server)");
    spawned = await spawnServer(serverBinary(), {});
    process.env["TAGURU_URL"] = spawned.baseUrl;
    delete process.env["TAGURU_API_TOKEN"];
  }

  try {
    const client = new Taguru();
    await client.waitUntilReady();

    // -- write: LLM-driven decomposition, per-source replace (idempotent) --
    const ingester = new TaguruIngester({
      context: "sake-demo",
      llm: await makeLlm(),
      client,
      create_context: true,
      context_description: "青嶺酒造という架空の酒蔵の知識",
      questions: 2,
    });
    const documents = Object.entries(DOCS).map(([source, text]) => ({
      pageContent: text,
      metadata: { source },
    }));
    for (const outcome of await ingester.ingestDocuments(documents)) {
      console.log(
        `ingested ${outcome.source}: ${outcome.associations} facts, ` +
          `${outcome.aliases} aliases, ${outcome.questions_stored} questions ` +
          `(model duplicates folded: ${outcome.duplicates_dropped}, ` +
          `invalid dropped: ${outcome.invalid_dropped})`,
      );
    }

    // -- read: graph lane + text lane, RRF-merged ---------------------------
    const retriever = new TaguruRetriever({ context: "sake-demo", client, k: 5 });

    console.log("\n== graph-shaped cue: 青嶺酒造 ==");
    for (const document of await retriever.invoke("青嶺酒造")) {
      const meta = document.metadata;
      const where =
        meta["paragraph"] !== null && meta["paragraph"] !== undefined
          ? `${meta["source"]}¶${meta["paragraph"]}`
          : "graph fact";
      console.log(`  [${String(meta["lane"]).padStart(10)}] (${where}) ${document.pageContent}`);
    }

    console.log("\n== answer-shaped text query: 1907年に創業した ==");
    for (const document of await retriever.invoke("1907年に創業した")) {
      console.log(`  [${String(document.metadata["lane"]).padStart(10)}] ${document.pageContent}`);
    }
  } finally {
    spawned?.stop();
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
