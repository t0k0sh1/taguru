/**
 * Governed ingestion with TaguruIngester: dry-run review, apply, replace,
 * retract. The TypeScript twin of ../python/main.py — see its docstring for
 * the staged flow and what to look for in the output.
 *
 * Runs self-contained: with no TAGURU_URL set it spawns a real server binary
 * (builds it with cargo on first run), and with no OPENAI_API_KEY it drives
 * the extraction with a deterministic fake model.
 *
 *     cd examples/langchain && npm install && npm start --workspace=document_ingestion/typescript
 */

import { resolve } from "node:path";

import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { Taguru } from "taguru";
import { serverBinary, spawnServer, type SpawnedServer } from "taguru/testing";
import { TaguruIngester } from "langchain-taguru";

const REPO_ROOT = resolve(import.meta.dirname, "../../../..");

const ABOUT_V1 = `蒼月堂は1892年に京都で創業した架空の茶舗である。

看板商品は玉露の「朝霧」である。

当主は蒼井である。`;

const TEA_GUIDE = `玉露は収穫前の数週間、茶園を覆って育てる被覆栽培の茶である。

「朝霧」の茶葉は宇治産である。蒼月堂は「朝霧」の水出しを推奨しない。`;

// The revision: the last paragraph gained a sentence about the new branch.
const ABOUT_V2 = `蒼月堂は1892年に京都で創業した架空の茶舗である。

看板商品は玉露の「朝霧」である。

当主は蒼井である。2026年、東京・銀座に支店を開いた。`;

// The decompositions a real chat model would produce — canned so the demo
// runs without an API key, in call order: about v1, tea guide, about v2.
const FAKE_ABOUT_V1 = {
  associations: [
    { subject: "蒼月堂", label: "創業年", object: "1892年", weight: 1.0, paragraph: 0 },
    { subject: "蒼月堂", label: "創業地", object: "京都", weight: 1.0, paragraph: 0 },
    { subject: "蒼月堂", label: "看板商品", object: "朝霧", weight: 1.0, paragraph: 1 },
    { subject: "朝霧", label: "種類", object: "玉露", weight: 1.0, paragraph: 1 },
    { subject: "蒼月堂", label: "当主", object: "蒼井", weight: 1.0, paragraph: 2 },
  ],
  aliases: [{ alias: "Sougetsudo", canonical: "蒼月堂", kind: "concept" }],
  questions: [{ paragraph: 0, question: "蒼月堂はいつ創業した?" }],
};
const FAKE_TEA_GUIDE = {
  associations: [
    { subject: "玉露", label: "栽培方法", object: "被覆栽培", weight: 1.0, paragraph: 0 },
    { subject: "朝霧", label: "産地", object: "宇治", weight: 1.0, paragraph: 1 },
    { subject: "蒼月堂", label: "推奨する", object: "朝霧の水出し", weight: -1.0, paragraph: 1 },
  ],
  aliases: [],
  questions: [{ paragraph: 0, question: "玉露はどのように育てられる?" }],
};
const FAKE_ABOUT_V2 = {
  associations: [
    { subject: "蒼月堂", label: "創業年", object: "1892年", weight: 1.0, paragraph: 0 },
    { subject: "蒼月堂", label: "創業地", object: "京都", weight: 1.0, paragraph: 0 },
    // A repeat of the first triple — models do this; merge folds it and
    // reports it as duplicates_dropped instead of double-writing.
    { subject: "蒼月堂", label: "創業年", object: "1892年", weight: 1.0, paragraph: 0 },
    { subject: "蒼月堂", label: "看板商品", object: "朝霧", weight: 1.0, paragraph: 1 },
    { subject: "朝霧", label: "種類", object: "玉露", weight: 1.0, paragraph: 1 },
    { subject: "蒼月堂", label: "当主", object: "蒼井", weight: 1.0, paragraph: 2 },
    { subject: "蒼月堂", label: "支店", object: "東京・銀座", weight: 1.0, paragraph: 2 },
  ],
  aliases: [{ alias: "Sougetsudo", canonical: "蒼月堂", kind: "concept" }],
  questions: [{ paragraph: 0, question: "蒼月堂はいつ創業した?" }],
};

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

    const fakes = [FAKE_ABOUT_V1, FAKE_TEA_GUIDE, FAKE_ABOUT_V2];
    const ingester = new TaguruIngester({
      context: "sougetsu-kb",
      llm: await makeLlm(fakes.map((fake) => JSON.stringify(fake))),
      client,
      create_context: true,
      context_description: "蒼月堂という架空の茶舗の知識",
      questions: 1,
    });
    const documents = [
      { pageContent: ABOUT_V1, metadata: { source: "docs/sougetsu/about.md" } },
      { pageContent: TEA_GUIDE, metadata: { source: "docs/sougetsu/tea-guide.md" } },
    ];

    console.log("== 1. dry run: review the exact batch before anything is written ==");
    const reviewed = await ingester.ingestDocuments(documents, { dry_run: true });
    const failed = reviewed.filter((outcome) => !outcome.ok);
    if (failed.length > 0) {
      throw new Error(
        `failed to ingest: ${failed.map((outcome) => `${outcome.source} (${outcome.error})`).join(", ")}`,
      );
    }
    for (const outcome of reviewed) {
      console.log(`--- NDJSON for ${outcome.source} ---`);
      for (const line of outcome.ndjson!.trim().split("\n")) {
        console.log(`  ${line}`);
      }
    }
    console.log(`context created yet? ${await client.contexts.exists("sougetsu-kb")}`);

    console.log("\n== 2. apply the reviewed batches — the NDJSON *is* the import payload ==");
    for (const outcome of reviewed) {
      const applied = (await client.importBatches(outcome.ndjson!)).batches[0]!;
      console.log(
        `${outcome.source}: created=${applied.created} ` +
          `associations=${applied.associations} aliases=${applied.aliases} ` +
          `questions=${applied.questions_stored} passage=${applied.passage_stored}`,
      );
    }

    const ctx = client.context("sougetsu-kb");
    console.log("\n== 3. read the graph back with the core SDK ==");
    console.log(`sources: ${JSON.stringify((await ctx.listSources()).sources)}`);
    const description = await ctx.describe("蒼月堂");
    if (description !== null) {
      const usages = description.as_subject.map((usage) => `${usage.label}×${usage.count}`);
      console.log(`蒼月堂 as subject: ${usages.join(", ")}`);
    }
    let founded = (await ctx.query({ subject: "蒼月堂", label: "創業年" })).matches[0]!;
    console.log(`創業年: ${founded.object} (weight ${founded.weight})`);
    const labels = (await ctx.listLabels()).labels;
    console.log(`label vocabulary (seeds the next ingest prompt): ${JSON.stringify(labels)}`);

    console.log("\n== 4. re-ingest a REVISED document under the same source id ==");
    const outcome = await ingester.ingestText(ABOUT_V2, { source: "docs/sougetsu/about.md" });
    console.log(
      `retracted ${outcome.retracted} old associations, applied ${outcome.associations} ` +
        `(model repeated a triple ${outcome.duplicates_dropped}× — merge folded it)`,
    );
    founded = (await ctx.query({ subject: "蒼月堂", label: "創業年" })).matches[0]!;
    console.log(`創業年 after re-ingest: ${founded.object} (weight ${founded.weight} — replaced, not doubled)`);
    const branches = (await ctx.query({ subject: "蒼月堂", label: "支店" })).matches;
    console.log(`支店 (new in the revision): ${JSON.stringify(branches.map((match) => match.object))}`);

    console.log("\n== 5. withdraw a whole document ==");
    const gone = await ctx.retractSource("docs/sougetsu/tea-guide.md");
    console.log(
      `associations touched: ${gone.associations_touched}, passage removed: ${gone.passage_removed}`,
    );
    console.log(`sources now: ${JSON.stringify((await ctx.listSources()).sources)}`);
  } finally {
    spawned?.stop();
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
