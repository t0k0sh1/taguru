/**
 * Local RAG over PDFs: TaguruIngester (extract) -> Taguru server (embed) ->
 * TaguruRetriever + an LCEL chain (answer), fully local end to end. The
 * TypeScript twin of ../python/main.py — see its docstring for what to look
 * for in the output.
 *
 * Two short fictional papers (tanaka2024, sato2023) go in as PDFs, get split
 * into numbered sections by a plain regex, and each section becomes its own
 * Taguru context — TaguruIngester({context: ...}) switched per section —
 * bundled under its paper's group. TaguruRetriever then searches both
 * papers' groups at once, and a separate LCEL chain answers the question;
 * retrieval and generation are printed as two distinct phases on purpose.
 *
 * Runs self-contained: with no TAGURU_URL set it spawns a real server binary
 * (builds it with cargo on first run), and with no OLLAMA_MODEL set it drives
 * every LLM role with a deterministic fake model — no Ollama required to see
 * the wiring work. Point OLLAMA_MODEL at a model already pulled locally
 * (check with `ollama list`) for the real thing; this script never pulls one
 * for you. Set TAGURU_EMBED_URL/TAGURU_EMBED_MODEL on the server (see the
 * walkthrough) to exercise the semantic lane too — retrieval and citations
 * below work the same either way, on BM25 and the graph alone if not.
 *
 *     cd examples/langchain && npm install && npm start --workspace=local_rag/typescript
 *
 * Full walkthrough, prerequisites, and the modeling behind the context/group
 * choice: https://t0k0sh1.github.io/taguru/local-rag-walkthrough.html
 */

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import type { DocumentInterface } from "@langchain/core/documents";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { PDFParse } from "pdf-parse";
import { ConflictError, Taguru } from "taguru";
import { serverBinary, spawnServer, type SpawnedServer } from "taguru/testing";
import { TaguruIngester, TaguruRetriever } from "langchain-taguru";

const REPO_ROOT = resolve(import.meta.dirname, "../../../..");
const PAPERS_DIR = resolve(import.meta.dirname, "../papers");

const SECTION_RE = /^(\d+)\.\s+(.+)$/gm;

// paper id -> a human-readable byline, used only for group descriptions and
// building CITATION_LABELS below — Taguru itself never sees the byline.
const PAPERS = {
  tanaka2024: "Tanaka et al. 2024",
  sato2023: "Sato et al. 2023",
} as const;

type PaperId = keyof typeof PAPERS;

// The decomposition a real chat model would produce under the extract
// discipline, canned per section so the demo runs without a local LLM.
// Keys are the source id each section is ingested under: "{paper}/{n}".
const FAKE_EXTRACTIONS: Record<string, unknown> = {
  "tanaka2024/1": {
    associations: [
      {
        subject: "ginjo aroma",
        label: "main component",
        object: "isoamyl acetate",
        weight: 1.0,
        paragraph: 0,
      },
    ],
    aliases: [],
  },
  "tanaka2024/2": {
    associations: [
      {
        subject: "tanaka2024 experiment",
        label: "yeast strain used",
        object: "Kyokai No. 901",
        weight: 1.0,
        paragraph: 0,
      },
    ],
    aliases: [],
  },
  "tanaka2024/3": {
    associations: [
      {
        subject: "low-temperature long fermentation",
        label: "effect on isoamyl acetate",
        object: "about 1.8x higher",
        weight: 1.0,
        paragraph: 0,
      },
      {
        subject: "low-temperature long fermentation",
        label: "effect on yield",
        object: "somewhat lower",
        weight: 1.0,
        paragraph: 0,
      },
    ],
    aliases: [],
  },
  "sato2023/1": {
    associations: [
      {
        subject: "koji-mold protease activity",
        label: "directly affects",
        object: "amino acid content",
        weight: 1.0,
        paragraph: 0,
      },
    ],
    aliases: [],
  },
  "sato2023/2": {
    associations: [
      {
        subject: "sato2023 strain comparison",
        label: "strain count",
        object: "three",
        weight: 1.0,
        paragraph: 0,
      },
    ],
    aliases: [],
  },
};

const QUESTION = "How does low-temperature fermentation affect ginjo aroma yield?";

const FAKE_ANSWER =
  "Low-temperature long fermentation raises isoamyl acetate concentration by " +
  "about 1.8x [tanaka2024/3], though the trade-off is a somewhat lower yield " +
  "[tanaka2024/3].";

// Taguru's API only ever deals in the source id (a machine key, chosen at
// ingest time). A human-readable citation label like "Tanaka et al. 2024,
// §3" is not an API concept at all — this mapping is entirely this script's
// own bookkeeping, built once from whatever a paper's front matter carries.
const CITATION_LABELS: Record<string, string> = Object.fromEntries(
  Object.keys(FAKE_EXTRACTIONS).map((source) => {
    const [paper, n] = source.split("/") as [PaperId, string];
    return [source, `${PAPERS[paper]}, §${n}`];
  }),
);

const PROMPT = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer using ONLY the facts in the context below. Cite every claim " +
      "with its bracketed source id, e.g. [tanaka2024/3].\n\n{context}",
  ],
  ["human", "{question}"],
]);

interface Section {
  paper: PaperId;
  n: number;
  title: string;
  text: string;
}

/** One entry per numbered section: paper, section number, title, text. */
async function pdfToSections(pdfPath: string, paperId: PaperId): Promise<Section[]> {
  const parser = new PDFParse({ data: await readFile(pdfPath) });
  let text: string;
  try {
    text = (await parser.getText()).text;
  } finally {
    await parser.destroy();
  }

  const marks = [...text.matchAll(SECTION_RE)];
  const sections: Section[] = [];
  for (let i = 0; i < marks.length; i++) {
    const m = marks[i];
    if (m === undefined || m.index === undefined) continue;
    const numberPart = m[1];
    const titlePart = m[2];
    if (numberPart === undefined || titlePart === undefined) continue;
    const start = m.index + (m[0]?.length ?? 0);
    const end = marks[i + 1]?.index ?? text.length;
    sections.push({
      paper: paperId,
      n: Number(numberPart),
      title: titlePart.trim(),
      text: text.slice(start, end).trim(),
    });
  }
  return sections;
}

/**
 * Create the group, or fold new contexts into an existing one — group
 * creation 409s on a rerun, so the idempotence has to be handled here.
 */
async function ensureGroup(client: Taguru, name: string, description: string, contexts: string[]): Promise<void> {
  try {
    await client.groups.create(name, { description, contexts });
  } catch (error) {
    if (error instanceof ConflictError) {
      await client.groups.update(name, { add_contexts: contexts });
    } else {
      throw error;
    }
  }
}

/** A real local Ollama model when OLLAMA_MODEL is set, else a canned fake. */
async function makeLlm(fakeResponses: string[]): Promise<BaseChatModel> {
  const model = process.env["OLLAMA_MODEL"];
  if (model) {
    try {
      const { ChatOllama } = await import("@langchain/ollama" as string);
      return new ChatOllama({ model, temperature: 0 });
    } catch {
      console.log("(OLLAMA_MODEL set but @langchain/ollama not installed — using the fake model)");
    }
  } else {
    console.log("(no OLLAMA_MODEL — using a canned fake model; this script never pulls one)");
  }
  return new FakeListChatModel({ responses: fakeResponses });
}

/**
 * Retrieved Documents -> the context block, source-id-first so the model
 * has something concrete to cite.
 */
function formatDocs(documents: DocumentInterface[]): string {
  return documents.map((document) => `[${document.metadata["source"]}] ${document.pageContent}`).join("\n");
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

    // -- write: one PDF -> N sections -> N contexts, one paper -> one group --
    const paperIds = Object.keys(PAPERS) as PaperId[];
    const allSections: Section[] = [];
    for (const paperId of paperIds) {
      allSections.push(...(await pdfToSections(resolve(PAPERS_DIR, `${paperId}.pdf`), paperId)));
    }
    const extractLlm = await makeLlm(allSections.map((s) => JSON.stringify(FAKE_EXTRACTIONS[`${s.paper}/${s.n}`])));

    const paperContexts: Record<PaperId, string[]> = { tanaka2024: [], sato2023: [] };
    for (const section of allSections) {
      const context = `section/${section.paper}/${section.n}`;
      const ingester = new TaguruIngester({
        context, // switches every section
        llm: extractLlm,
        client,
        create_context: true,
        context_description: `${section.paper} §${section.n} — ${section.title}`,
      });
      const doc = {
        pageContent: section.text,
        metadata: { source: `${section.paper}/${section.n}` },
      };
      const outcome = (await ingester.ingestDocuments([doc]))[0];
      if (outcome === undefined || !outcome.ok) {
        throw new Error(`FAILED to ingest ${outcome?.source ?? context}: ${outcome?.error}`);
      }
      console.log(`ingested ${context}: ${outcome.associations} facts, ${outcome.aliases} aliases`);
      paperContexts[section.paper].push(context);
    }

    for (const paperId of paperIds) {
      await ensureGroup(client, `paper/${paperId}`, `${PAPERS[paperId]}, full paper`, paperContexts[paperId]);
    }

    // -- read: one retriever across both papers' groups, an independent answer model --
    const retriever = new TaguruRetriever({ client, groups: paperIds.map((p) => `paper/${p}`), k: 8 });
    const answerLlm = await makeLlm([FAKE_ANSWER]);
    const chain = RunnableSequence.from([
      {
        context: retriever.pipe(formatDocs),
        question: new RunnablePassthrough<string>(),
      },
      PROMPT,
      answerLlm,
      new StringOutputParser(),
    ]);

    console.log(`\n== ${QUESTION} ==`);
    // Phase 1: what retrieval brought back — shown before any generation runs.
    for (const document of await retriever.invoke(QUESTION)) {
      const meta = document.metadata;
      const source = String(meta["source"]);
      const label = CITATION_LABELS[source] ?? source;
      const where =
        meta["paragraph"] !== null && meta["paragraph"] !== undefined ? `${label} ¶${meta["paragraph"]}` : label;
      console.log(`  [${String(meta["lane"]).padStart(10)}] ${where}: ${document.pageContent.slice(0, 70)}...`);
    }

    // Phase 2: the answer, generated from exactly those retrieved documents.
    console.log(`  answer: ${await chain.invoke(QUESTION)}`);

    // Trace one claim in the answer back to its original PDF paragraph.
    const citation = await client.context("section/tanaka2024/3").citePassage("tanaka2024/3", 0);
    console.log(`\n  cited passage (tanaka2024 §3, ¶0): ${citation.text.slice(0, 120)}...`);
  } finally {
    spawned?.stop();
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
