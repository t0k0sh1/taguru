/**
 * Budgeted evidence assembly with Taguru: rank, budget, then hand off to an
 * answer model. The TypeScript twin of ../python/main.py — see its
 * docstring for what to look for in the output.
 *
 * Runs self-contained: with no TAGURU_URL set it spawns a real server binary
 * (builds it with cargo on first run), and with no OPENAI_API_KEY it drives
 * both LLM roles (extraction and answering) with deterministic fake models.
 *
 *     cd examples/langchain && npm install && npm start --workspace=evidence_assembly/typescript
 */

import { resolve } from "node:path";

import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { Taguru } from "taguru";
import type { EvidenceItem, EvidencePackage } from "taguru";
import { serverBinary, spawnServer, type SpawnedServer } from "taguru/testing";
import { TaguruIngester } from "langchain-taguru";

const REPO_ROOT = resolve(import.meta.dirname, "../../../..");

// The same fictional-brewery corpus rag_qa/ uses — already a proven
// fixture, and reusing it keeps this example's own code focused on the
// read side, which is what's new here.
const DOCS: Record<string, string> = {
  "docs/aomine/brewery.md": `青嶺酒造は1907年創業の架空の酒蔵である。蔵は岩手県遠野市にある。

杜氏は高瀬である。高瀬は寒仕込みを重視する。

青嶺酒造は大量生産を行わない。`,
  "docs/aomine/lineup.md": `「青嶺 大吟醸」は精米歩合40%の山田錦で仕込まれる。

「青嶺 純米」は地元米の遠野錦を使う。`,
};

// The decomposition a real chat model would produce under the extract
// discipline — canned per source so the demo runs without an API key. The
// negatively-weighted 大量生産 fact (weight -1.0) is what
// EvidenceItem.contradicts/corroboration would report a disagreement over
// if a second source asserted the opposite; here it demonstrates that a
// denial is stored and surfaced like any other fact, never silently
// dropped.
const FAKE_EXTRACTIONS: Record<string, unknown> = {
  "docs/aomine/brewery.md": {
    associations: [
      { subject: "青嶺酒造", label: "創業年", object: "1907年", weight: 1.0, paragraph: 0 },
      { subject: "青嶺酒造", label: "所在地", object: "岩手県遠野市", weight: 1.0, paragraph: 0 },
      { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0, paragraph: 1 },
      { subject: "青嶺酒造", label: "行う", object: "大量生産", weight: -1.0, paragraph: 2 },
    ],
    aliases: [],
    questions: [{ paragraph: 1, question: "青嶺酒造の杜氏は誰?" }],
  },
  "docs/aomine/lineup.md": {
    associations: [
      { subject: "青嶺 大吟醸", label: "精米歩合", object: "40%", weight: 1.0, paragraph: 0 },
      { subject: "青嶺 大吟醸", label: "原料米", object: "山田錦", weight: 1.0, paragraph: 0 },
    ],
    aliases: [],
    questions: [{ paragraph: 0, question: "大吟醸の精米歩合はいくつ?" }],
  },
};

const QUERY = "青嶺酒造の杜氏は誰ですか?";

const FAKE_ANSWER =
  "杜氏は高瀬です [graph fact]。青嶺酒造は大量生産を行っていません [docs/aomine/brewery.md ¶2]。";

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
 * One admitted EvidenceItem, kind-first — the same locator vocabulary
 * citation_refs/HitLocator use elsewhere in this tree, never the raw
 * fused_score (ADR 0006 §7 never serializes it at all).
 */
function describeItem(item: EvidenceItem): string {
  let where: string;
  let snippet: string;
  if (item.passage) {
    where = `${item.passage.source} ¶${item.passage.paragraph}`;
    snippet = item.passage.text;
  } else if (item.association) {
    const association = item.association;
    where = "graph fact";
    snippet = `${association.subject} —${association.label}→ ${association.object} (weight ${association.weight >= 0 ? "+" : ""}${association.weight.toFixed(1)})`;
  } else if (item.community) {
    where = `${item.community.community} ¶${item.community.paragraph}`;
    snippet = item.community.text;
  } else {
    where = "?";
    snippet = "?";
  }
  let line = `  [${item.kind.padStart(11)} rank=${item.fused_rank}] (${where}) ${snippet}`;
  if (item.corroboration) {
    line += `  corroborated by: ${JSON.stringify([...item.corroboration.sources].sort())}`;
  }
  if (item.contradicts && item.contradicts.length > 0) {
    line += `  contradicts: ${JSON.stringify(item.contradicts)}`;
  }
  return line;
}

function printPackage(label: string, pkg: EvidencePackage): void {
  const budget = pkg.budget;
  console.log(`-- ${label} --`);
  console.log(
    `  budget: ${budget.items_used}/${budget.limits.max_items} items, ` +
      `${budget.bytes_used}/${budget.limits.max_bytes} bytes, ` +
      `${budget.tokens_used}/${budget.limits.max_tokens} tokens (estimated)`,
  );
  console.log(
    `  selection: dedup_dropped=${pkg.plan.selection.dedup_dropped} ` +
      `contradiction_groups=${pkg.plan.selection.contradiction_groups} ` +
      `diversity_tier_width=${pkg.plan.selection.diversity_tier_width}`,
  );
  console.log(`  reranker: configured=${pkg.plan.reranker.configured} ran=${pkg.plan.reranker.ran}`);
  console.log(`  omitted: ${pkg.omitted_total} total, by reason: ${JSON.stringify(pkg.omitted_by_reason)}`);
  for (const item of pkg.items) {
    console.log(describeItem(item));
  }
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
      context: "aomine-evidence",
      llm: await makeLlm(Object.keys(DOCS).map((source) => JSON.stringify(FAKE_EXTRACTIONS[source]))),
      client,
      create_context: true,
      context_description: "青嶺酒造という架空の酒蔵の知識（evidence assembly デモ）",
      questions: 1,
    });
    const documents = Object.entries(DOCS).map(([source, text]) => ({
      pageContent: text,
      metadata: { source },
    }));
    const outcomes = await ingester.ingestDocuments(documents);
    const failed = outcomes.filter((outcome) => !outcome.ok);
    if (failed.length > 0) {
      throw new Error(
        `failed to ingest: ${failed.map((outcome) => `${outcome.source} (${outcome.error})`).join(", ")}`,
      );
    }
    for (const outcome of outcomes) {
      console.log(`ingested ${outcome.source}: ${outcome.associations} facts, ${outcome.aliases} aliases`);
    }

    // -- read: POST /contexts/{name}/evidence directly, no assembly-lane
    //    intermediary — this is the same call `taguru evaluate --assembly`
    //    drives for the equal-budget comparison documented on
    //    docs/evidence.html. Context.assembleEvidence(), not a bare
    //    client-level call — every read/write method is bound to one
    //    context, named after the server's own vocabulary.
    const context = client.context("aomine-evidence");
    console.log(`\n== ${QUERY} ==`);
    const generous = await context.assembleEvidence(["青嶺酒造"], { text_fallback_query: QUERY });
    printPackage("generous budget (server defaults)", generous);

    const tight = await context.assembleEvidence(["青嶺酒造"], {
      text_fallback_query: QUERY,
      budget: { max_items: 1 },
    });
    printPackage("tight budget (max_items=1)", tight);

    // -- handing the generous package to an (fake) answer model --------
    const llm = await makeLlm([FAKE_ANSWER]);
    const answer = (await llm.invoke(QUERY)).content;
    console.log(`\nanswer: ${answer}`);
    console.log(
      "(everything above 'answer:' is evidence Taguru assembled and budgeted; " +
        "everything after it is the answer model's own prose — Taguru itself " +
        "never generates it.)",
    );
  } finally {
    spawned?.stop();
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
