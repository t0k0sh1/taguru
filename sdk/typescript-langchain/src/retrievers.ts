/**
 * TaguruRetriever: the documented retrieval loop as a LangChain.js retriever.
 * The mechanical mirror of the Python `taguru_langchain.TaguruRetriever` —
 * see that module for the full design rationale (opaque cue, both lanes by
 * default, RRF merge, graph-only facts as Documents).
 */

import type { CallbackManagerForRetrieverRun } from "@langchain/core/callbacks/manager";
import { Document } from "@langchain/core/documents";
import { BaseRetriever, type BaseRetrieverInput } from "@langchain/core/retrievers";
import {
  NotFoundError,
  Taguru,
  citationKey,
  type Activation,
  type Citation,
  type PassageHit,
} from "taguru";

const RRF_K = 60;

const rrf = (rank: number): number => 1.0 / (RRF_K + rank);

export interface TaguruRetrieverFields extends BaseRetrieverInput {
  context: string;
  client?: Taguru;
  base_url?: string;
  api_key?: string;
  timeout?: number;
  k?: number;
  include_graph?: boolean;
  include_text?: boolean;
  include_graph_only_facts?: boolean;
  activate_decay?: number;
  activate_limit?: number;
  text_limit?: number;
  resolve_limit?: number;
  dice_floor?: number;
  semantic_floor?: number;
}

/**
 * Retrieve Documents from one Taguru context, graph lane + text lane.
 *
 * Passage-backed hits carry `page_content` = the verbatim paragraph and
 * metadata `{source, paragraph, section, lane, associations?, score?,
 * lanes?}`. Graph facts with no stored passage still become Documents
 * (`page_content` = "subject label object") when `include_graph_only_facts`
 * is on.
 */
export class TaguruRetriever extends BaseRetriever {
  static lc_name(): string {
    return "TaguruRetriever";
  }

  lc_namespace = ["taguru"];

  readonly context: string;
  readonly k: number;
  readonly include_graph: boolean;
  readonly include_text: boolean;
  readonly include_graph_only_facts: boolean;
  readonly activate_decay: number | undefined;
  readonly activate_limit: number;
  readonly text_limit: number;
  readonly resolve_limit: number;
  readonly dice_floor: number | undefined;
  readonly semantic_floor: number | undefined;

  private readonly client: Taguru;

  constructor(fields: TaguruRetrieverFields) {
    super(fields);
    this.client =
      fields.client ??
      new Taguru({
        base_url: fields.base_url,
        api_key: fields.api_key,
        timeout: fields.timeout,
      });
    this.context = fields.context;
    this.k = fields.k ?? 8;
    this.include_graph = fields.include_graph ?? true;
    this.include_text = fields.include_text ?? true;
    this.include_graph_only_facts = fields.include_graph_only_facts ?? true;
    this.activate_decay = fields.activate_decay;
    this.activate_limit = fields.activate_limit ?? 20;
    this.text_limit = fields.text_limit ?? 5;
    this.resolve_limit = fields.resolve_limit ?? 5;
    this.dice_floor = fields.dice_floor;
    this.semantic_floor = fields.semantic_floor;
  }

  async _getRelevantDocuments(
    query: string,
    _runManager?: CallbackManagerForRetrieverRun,
  ): Promise<Document[]> {
    const ctx = this.client.context(this.context);

    let graphDocs: Document[] = [];
    if (this.include_graph) {
      const candidates = await ctx.resolve(query, {
        dice_floor: this.dice_floor,
        semantic_floor: this.semantic_floor,
        limit: this.resolve_limit,
      });
      const origins: string[] = [];
      for (const candidate of candidates) {
        if (!origins.includes(candidate.name)) {
          origins.push(candidate.name);
        }
      }
      if (origins.length > 0) {
        const page = await ctx.activate(origins, {
          decay: this.activate_decay,
          limit: this.activate_limit,
        });
        const citations = new Map<string, Citation | null>();
        for (const [source, paragraph] of wantedCitations(page.matches)) {
          try {
            citations.set(citationKey(source, paragraph), await ctx.citePassage(source, paragraph));
          } catch (error) {
            if (!(error instanceof NotFoundError)) {
              throw error;
            }
            citations.set(citationKey(source, paragraph), null);
          }
        }
        graphDocs = graphDocuments(page.matches, citations, this.include_graph_only_facts);
      }
    }

    let textHits: PassageHit[] = [];
    if (this.include_text) {
      textHits = await ctx.searchPassages(query, { limit: this.text_limit });
    }

    return mergeLanes(graphDocs, textHits, this.k);
  }
}

// -- lane assembly (mirrors the Python module's pure functions) -------------------

function wantedCitations(activations: Activation[]): Array<[string, number]> {
  const wanted: Array<[string, number]> = [];
  const seen = new Set<string>();
  for (const activation of activations) {
    for (const attribution of activation.association.attributions) {
      if (attribution.paragraph === null) {
        continue;
      }
      const key = citationKey(attribution.source, attribution.paragraph);
      if (!seen.has(key)) {
        seen.add(key);
        wanted.push([attribution.source, attribution.paragraph]);
      }
    }
  }
  return wanted;
}

interface AssociationMeta {
  subject: string;
  label: string;
  object: string;
  weight: number;
  strength: number;
  path: string[];
}

function associationMeta(activation: Activation): AssociationMeta {
  const association = activation.association;
  return {
    subject: association.subject,
    label: association.label,
    object: association.object,
    weight: association.weight,
    strength: activation.strength,
    path: [...activation.path],
  };
}

function graphDocuments(
  activations: Activation[],
  citations: Map<string, Citation | null>,
  includeGraphOnlyFacts: boolean,
): Document[] {
  const passageDocs = new Map<string, Document>();
  const ordered: Document[] = [];
  for (const activation of activations) {
    const association = activation.association;
    const located: Array<[string, number]> = [];
    for (const attribution of association.attributions) {
      if (attribution.paragraph === null) {
        continue;
      }
      const key = citationKey(attribution.source, attribution.paragraph);
      if (citations.get(key)) {
        located.push([attribution.source, attribution.paragraph]);
      }
    }
    if (located.length > 0) {
      for (const [source, paragraph] of located) {
        const key = citationKey(source, paragraph);
        const citation = citations.get(key)!;
        const existing = passageDocs.get(key);
        if (existing === undefined) {
          const document = new Document({
            pageContent: citation.text,
            metadata: {
              source,
              paragraph,
              section: citation.section,
              lane: "graph",
              associations: [associationMeta(activation)],
            },
          });
          passageDocs.set(key, document);
          ordered.push(document);
        } else {
          (existing.metadata["associations"] as AssociationMeta[]).push(
            associationMeta(activation),
          );
        }
      }
    } else if (includeGraphOnlyFacts) {
      // Real, retrievable knowledge with no verbatim excerpt to ground it in.
      const bestSource = association.attributions[0]?.source ?? null;
      ordered.push(
        new Document({
          pageContent: `${association.subject} ${association.label} ${association.object}`,
          metadata: {
            source: bestSource,
            paragraph: null,
            section: null,
            lane: "graph",
            associations: [associationMeta(activation)],
          },
        }),
      );
    }
  }
  return ordered;
}

function textDocument(hit: PassageHit): Document {
  const lanes: Record<string, { rank: number; score: number }> = {};
  if (hit.lanes.bm25 !== undefined) {
    lanes["bm25"] = { rank: hit.lanes.bm25.rank, score: hit.lanes.bm25.score };
  }
  if (hit.lanes.vector !== undefined) {
    lanes["vector"] = { rank: hit.lanes.vector.rank, score: hit.lanes.vector.score };
  }
  return new Document({
    pageContent: hit.text,
    metadata: {
      source: hit.source,
      paragraph: hit.paragraph,
      section: null,
      lane: "text",
      score: hit.score,
      lanes,
    },
  });
}

function mergeLanes(graphDocs: Document[], textHits: PassageHit[], k: number): Document[] {
  const scored = new Map<string, { score: number; document: Document }>();

  const keyOf = (document: Document): string => {
    if (document.metadata["paragraph"] !== null && document.metadata["paragraph"] !== undefined) {
      return citationKey(document.metadata["source"] as string, document.metadata["paragraph"] as number);
    }
    const associations = (document.metadata["associations"] as AssociationMeta[] | undefined) ?? [];
    const first = associations[0];
    return `fact\u0000${first?.subject}\u0000${first?.label}\u0000${first?.object}`;
  };

  graphDocs.forEach((document, rank) => {
    scored.set(keyOf(document), { score: rrf(rank), document });
  });

  textHits.forEach((hit, rank) => {
    const key = citationKey(hit.source, hit.paragraph);
    const existing = scored.get(key);
    if (existing === undefined) {
      scored.set(key, { score: rrf(rank), document: textDocument(hit) });
    } else {
      existing.document.metadata["lane"] = "graph+text";
      existing.document.metadata["score"] = hit.score;
      existing.document.metadata["lanes"] = textDocument(hit).metadata["lanes"];
      existing.score += rrf(rank);
    }
  });

  return [...scored.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((entry) => entry.document);
}
