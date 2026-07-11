/**
 * Taguru as an assistant's long-term memory: remember across sessions,
 * correct on update. The TypeScript twin of ../python/main.py — see its
 * docstring for the session flow and what to look for in the output.
 *
 * Runs self-contained: with no TAGURU_URL set it spawns a real server binary
 * (builds it with cargo on first run), and with no OPENAI_API_KEY it drives
 * both LLM roles (memorizing and chatting) with deterministic fake models.
 *
 *     cd examples/langchain && npm install && npm start --workspace=conversational_memory/typescript
 */

import { resolve } from "node:path";

import type { DocumentInterface } from "@langchain/core/documents";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { Taguru } from "taguru";
import { serverBinary, spawnServer, type SpawnedServer } from "taguru/testing";
import { TaguruIngester, TaguruRetriever } from "langchain-taguru";

const REPO_ROOT = resolve(import.meta.dirname, "../../../..");

const SESSION_1 = `ユーザー: そばアレルギーがあるので、そば粉を使った料理は食べられません。

ユーザー: いま「たぐる導入」というプロジェクトを進めていて、締切は8月末です。

ユーザー: 犬を飼っています。名前はハナです。`;

const SESSION_2 = `ユーザー: 「たぐる導入」の締切は9月15日に延期になった。8月末ではなくなったよ。`;

// What a real chat model would extract from each transcript — canned so the
// demo runs without an API key. Session 2 both asserts the new deadline and
// NEGATES the outdated one: correction is a first-class write.
const FAKE_MEMORY_SESSION_1 = {
  associations: [
    { subject: "ユーザー", label: "アレルギー", object: "そば", weight: 1.0, paragraph: 0 },
    { subject: "ユーザー", label: "食べられる", object: "そば粉を使った料理", weight: -1.0, paragraph: 0 },
    { subject: "ユーザー", label: "進めている", object: "たぐる導入", weight: 1.0, paragraph: 1 },
    { subject: "たぐる導入", label: "締切", object: "8月末", weight: 1.0, paragraph: 1 },
    { subject: "ユーザー", label: "飼っている", object: "犬", weight: 1.0, paragraph: 2 },
    { subject: "犬", label: "名前", object: "ハナ", weight: 1.0, paragraph: 2 },
  ],
  aliases: [],
  questions: [],
};
const FAKE_MEMORY_SESSION_2 = {
  associations: [
    { subject: "たぐる導入", label: "締切", object: "9月15日", weight: 1.0, paragraph: 0 },
    { subject: "たぐる導入", label: "締切", object: "8月末", weight: -1.0, paragraph: 0 },
  ],
  aliases: [],
  questions: [],
};

const TURNS = ["今夜の夕食、そば屋さんはどうかな?", "「たぐる導入」の締切っていつだったっけ?"];
const RE_ASKED = "もう一度確認だけど、「たぐる導入」の締切はいつ?";

// What a grounded model would answer from the retrieved memories — canned in
// call order: the two session-2 turns, then the re-asked question.
const FAKE_REPLIES = [
  "そばアレルギーをお持ちなので、そば屋は避けたほうがよさそうです(2026-07-05の会話より)。" +
    "別のお店を探しましょうか。",
  "「たぐる導入」の締切は8月末です(2026-07-05の会話より)。",
  "「たぐる導入」の締切は9月15日です。8月末から延期になりました(2026-07-12の会話より)。",
];

const PROMPT = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are the user's personal assistant. Answer in Japanese. Ground yourself in " +
      "the long-term memories below (each line notes which conversation it came from). " +
      "A negative-weight fact is something that is NOT true.\n\n{memory}",
  ],
  ["human", "{message}"],
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

interface FactMeta {
  subject: string;
  label: string;
  object: string;
  weight: number;
}

/**
 * Retrieved Documents → the memory block: provenance, plus each backing
 * graph fact with its weight — negations (weight < 0) stay visible.
 */
function formatMemory(documents: DocumentInterface[]): string {
  if (documents.length === 0) {
    return "- (no relevant memories)";
  }
  const lines: string[] = [];
  for (const document of documents) {
    const meta = document.metadata;
    lines.push(`- (${meta["source"] ?? "graph"}) ${document.pageContent}`);
    for (const fact of (meta["associations"] as FactMeta[] | undefined) ?? []) {
      const sign = fact.weight >= 0 ? "+" : "";
      lines.push(`    fact: ${fact.subject} –${fact.label}→ ${fact.object} (weight ${sign}${fact.weight})`);
    }
  }
  return lines.join("\n");
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

    const ingester = new TaguruIngester({
      context: "assistant-memory",
      llm: await makeLlm([
        JSON.stringify(FAKE_MEMORY_SESSION_1),
        JSON.stringify(FAKE_MEMORY_SESSION_2),
      ]),
      client,
      create_context: true,
      context_description: "アシスタントがユーザーについて記憶していること",
    });
    const retriever = new TaguruRetriever({ context: "assistant-memory", client, k: 4 });
    const chat = await makeLlm(FAKE_REPLIES);
    const respond = PROMPT.pipe(chat).pipe(new StringOutputParser());

    console.log("== session 1 (2026-07-05): the user talks; the transcript is memorized after ==");
    console.log(SESSION_1);
    const [memorized] = await ingester.ingestDocuments([
      { pageContent: SESSION_1, metadata: { source: "conversations/2026-07-05" } },
    ]);
    console.log(`memorized ${memorized!.source}: ${memorized!.associations} facts`);

    console.log("\n== session 2 (2026-07-12): every turn first recalls, then answers ==");
    for (const turn of TURNS) {
      console.log(`\nuser: ${turn}`);
      const memories = await retriever.invoke(turn);
      console.log(formatMemory(memories));
      const reply = await respond.invoke({ memory: formatMemory(memories), message: turn });
      console.log(`assistant: ${reply}`);
    }

    console.log("\n== the user corrects a fact; the correction is memorized ==");
    console.log(SESSION_2);
    const corrected = await ingester.ingestText(SESSION_2, { source: "conversations/2026-07-12" });
    console.log(`memorized ${corrected.source}: ${corrected.associations} facts`);
    const ctx = client.context("assistant-memory");
    for (const match of (await ctx.query({ subject: "たぐる導入", label: "締切" })).matches) {
      const sign = match.weight >= 0 ? "+" : "";
      console.log(`  たぐる導入 –締切→ ${match.object}: weight ${sign}${match.weight}`);
    }

    console.log(`\nuser: ${RE_ASKED}`);
    const memories = await retriever.invoke(RE_ASKED);
    console.log(formatMemory(memories));
    const reply = await respond.invoke({ memory: formatMemory(memories), message: RE_ASKED });
    console.log(`assistant: ${reply}`);
  } finally {
    spawned?.stop();
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
