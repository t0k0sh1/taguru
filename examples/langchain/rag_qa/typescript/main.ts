/**
 * RAG question answering over Taguru with LangChain.js: ingest, retrieve,
 * answer with citations. The TypeScript twin of ../python/main.py — see its
 * docstring for what to look for in the output.
 *
 * Runs self-contained: with no TAGURU_URL set it spawns a real server binary
 * (builds it with cargo on first run), and with no OPENAI_API_KEY it drives
 * both LLM roles (extraction and answering) with deterministic fake models.
 *
 *     cd examples/langchain && npm install && npm start --workspace=rag_qa/typescript
 */

import { resolve } from "node:path";

import type { DocumentInterface } from "@langchain/core/documents";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { Taguru } from "taguru";
import { serverBinary, spawnServer, type SpawnedServer } from "taguru/testing";
import { TaguruIngester, TaguruRetriever } from "langchain-taguru";

const REPO_ROOT = resolve(import.meta.dirname, "../../../..");

const DOCS: Record<string, string> = {
  "docs/aomine/brewery.md": `青嶺酒造は1907年創業の架空の酒蔵である。蔵は岩手県遠野市にある。

杜氏は高瀬である。高瀬は寒仕込みを重視する。

青嶺酒造は大量生産を行わない。`,
  "docs/aomine/lineup.md": `「青嶺 大吟醸」は精米歩合40%の山田錦で仕込まれる。

「青嶺 純米」は地元米の遠野錦を使う。冬季限定の「冬青嶺」は12月から2月にのみ出荷される。`,
  "docs/aomine/visits.md": `蔵見学は要予約で、所要時間はおよそ60分である。

蔵見学の定休日は水曜日である。試飲は見学コースの最後に含まれる。`,
};

// The decomposition a real chat model would produce under the extract
// discipline — canned per source so the demo runs without an API key.
const FAKE_EXTRACTIONS: Record<string, unknown> = {
  "docs/aomine/brewery.md": {
    associations: [
      { subject: "青嶺酒造", label: "創業年", object: "1907年", weight: 1.0, paragraph: 0 },
      { subject: "青嶺酒造", label: "所在地", object: "岩手県遠野市", weight: 1.0, paragraph: 0 },
      { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0, paragraph: 1 },
      { subject: "高瀬", label: "重視する", object: "寒仕込み", weight: 1.0, paragraph: 1 },
      { subject: "青嶺酒造", label: "行う", object: "大量生産", weight: -1.0, paragraph: 2 },
    ],
    aliases: [{ alias: "Aomine Brewery", canonical: "青嶺酒造", kind: "concept" }],
    questions: [{ paragraph: 1, question: "青嶺酒造の杜氏は誰?" }],
  },
  "docs/aomine/lineup.md": {
    associations: [
      { subject: "青嶺 大吟醸", label: "精米歩合", object: "40%", weight: 1.0, paragraph: 0 },
      { subject: "青嶺 大吟醸", label: "原料米", object: "山田錦", weight: 1.0, paragraph: 0 },
      { subject: "青嶺 純米", label: "原料米", object: "遠野錦", weight: 1.0, paragraph: 1 },
      { subject: "冬青嶺", label: "出荷時期", object: "12月から2月", weight: 1.0, paragraph: 1 },
    ],
    aliases: [],
    questions: [{ paragraph: 0, question: "大吟醸の精米歩合はいくつ?" }],
  },
  "docs/aomine/visits.md": {
    associations: [
      { subject: "蔵見学", label: "予約", object: "必要", weight: 1.0, paragraph: 0 },
      { subject: "蔵見学", label: "所要時間", object: "60分", weight: 1.0, paragraph: 0 },
      { subject: "蔵見学", label: "定休日", object: "水曜日", weight: 1.0, paragraph: 1 },
      { subject: "試飲", label: "含まれる", object: "見学コース", weight: 1.0, paragraph: 1 },
    ],
    aliases: [{ alias: "見学", canonical: "蔵見学", kind: "concept" }],
    questions: [{ paragraph: 1, question: "見学が休みなのは何曜日?" }],
  },
};

const QUESTIONS = [
  "青嶺酒造の杜氏は誰ですか?",
  "青嶺 大吟醸はどんな米で造られていますか?",
  "青嶺酒造はお酒を大量生産していますか?",
];

// What a grounded model would answer from the retrieved context — canned in
// the same order as QUESTIONS.
const FAKE_ANSWERS = [
  "杜氏は高瀬です [docs/aomine/brewery.md ¶1]。高瀬は寒仕込みを重視しています [docs/aomine/brewery.md ¶1]。",
  "「青嶺 大吟醸」は精米歩合40%まで磨いた山田錦で仕込まれます [docs/aomine/lineup.md ¶0]。",
  "いいえ、青嶺酒造は大量生産を行っていません [docs/aomine/brewery.md ¶2]。",
];

const PROMPT = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You answer questions about 青嶺酒造 in Japanese, using ONLY the facts in the " +
      "context below. Cite every claim with its bracketed locator, e.g. " +
      "[docs/aomine/brewery.md ¶1]. If the context does not answer the question, say so.\n\n" +
      "{context}",
  ],
  ["human", "{question}"],
]);

/** A real model when OPENAI_API_KEY is available, else the canned fake. */
async function makeLlm(fakeResponses: string[]): Promise<BaseChatModel> {
  if (process.env["OPENAI_API_KEY"]) {
    try {
      const { ChatOpenAI } = await import("@langchain/openai" as string);
      return new ChatOpenAI({ model: "gpt-4.1", temperature: 0 });
    } catch {
      console.log("(OPENAI_API_KEY set but @langchain/openai not installed — using the fake model)");
    }
  } else {
    console.log("(no OPENAI_API_KEY — using a canned fake model)");
  }
  return new FakeListChatModel({ responses: fakeResponses });
}

/**
 * Retrieved Documents → the context block, each line locator-first so the
 * model has something concrete to cite.
 */
function formatDocs(documents: DocumentInterface[]): string {
  return documents
    .map((document) => {
      const meta = document.metadata;
      let where: string;
      if (meta["paragraph"] !== null && meta["paragraph"] !== undefined) {
        where = `${meta["source"]} ¶${meta["paragraph"]}`;
      } else {
        where = meta["source"] ? `${meta["source"]} (graph fact)` : "graph fact";
      }
      return `[${where}] ${document.pageContent}`;
    })
    .join("\n");
}

async function main(): Promise<void> {
  let spawned: SpawnedServer | undefined;
  if (!process.env["TAGURU_URL"]) {
    console.log("(no TAGURU_URL — spawning a local server)");
    spawned = await spawnServer(serverBinary(REPO_ROOT), {});
    process.env["TAGURU_URL"] = spawned.baseUrl;
    delete process.env["TAGURU_API_TOKEN"];
  }

  try {
    const client = new Taguru();
    await client.waitUntilReady();

    // -- write: LLM-driven decomposition, one idempotent batch per source --
    const ingester = new TaguruIngester({
      context: "aomine-qa",
      llm: await makeLlm(Object.keys(DOCS).map((source) => JSON.stringify(FAKE_EXTRACTIONS[source]))),
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
      console.log(`ingested ${outcome.source}: ${outcome.associations} facts, ${outcome.aliases} aliases`);
    }

    // -- read: the retriever composes like any other LCEL component --------
    const retriever = new TaguruRetriever({ context: "aomine-qa", client, k: 6 });
    const llm = await makeLlm(FAKE_ANSWERS);
    const chain = RunnableSequence.from([
      {
        context: retriever.pipe(formatDocs),
        question: new RunnablePassthrough<string>(),
      },
      PROMPT,
      llm,
      new StringOutputParser(),
    ]);

    for (const question of QUESTIONS) {
      console.log(`\n== ${question} ==`);
      // Shown for the demo; the chain runs the same retrieval internally.
      for (const document of await retriever.invoke(question)) {
        const meta = document.metadata;
        const where =
          meta["paragraph"] !== null && meta["paragraph"] !== undefined
            ? `${meta["source"]} ¶${meta["paragraph"]}`
            : "graph fact";
        console.log(`  [${String(meta["lane"]).padStart(10)}] (${where}) ${document.pageContent}`);
      }
      console.log(`  answer: ${await chain.invoke(question)}`);
    }
  } finally {
    spawned?.stop();
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
