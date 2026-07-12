/**
 * TaguruRetriever: the documented retrieval loop as a LangChain.js retriever.
 * The mechanical mirror of the Python `taguru_langchain.TaguruRetriever` —
 * see that module for the full design rationale (opaque cue, both lanes by
 * default, RRF merge, graph-only facts as Documents).
 *
 * The retriever addresses one `context`, several `contexts`, or `groups`
 * (each group reaches every member context, nested children included).
 * Across several contexts the graph lane runs per context and interleaves by
 * per-context rank — the posture the server itself takes for passage scores —
 * and the text lane rides the server's own cross-context search.
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
  /** One target context; name at least one of context/contexts/groups. */
  context?: string;
  /** Several target contexts (full names). */
  contexts?: string[];
  /** Group names — each searches every context it reaches, nested children included. */
  groups?: string[];
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
 * Retrieve Documents from Taguru contexts, graph lane + text lane.
 *
 * Passage-backed hits carry `page_content` = the verbatim paragraph and
 * metadata `{context, source, paragraph, section, lane, associations?,
 * score?, lanes?}`. Graph facts with no stored passage still become Documents
 * (`page_content` = "subject label object") when `include_graph_only_facts`
 * is on.
 */
export class TaguruRetriever extends BaseRetriever {
  static lc_name(): string {
    return "TaguruRetriever";
  }

  lc_namespace = ["taguru"];

  readonly context: string | undefined;
  readonly contexts: string[] | undefined;
  readonly groups: string[] | undefined;
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
    this.contexts = fields.contexts;
    this.groups = fields.groups;
    if (
      this.context === undefined &&
      (this.contexts?.length ?? 0) === 0 &&
      (this.groups?.length ?? 0) === 0
    ) {
      throw new Error("name a target: context, contexts, or groups");
    }
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

  /** Whether retrieval spans several contexts (or a group's worth). */
  private isCross(): boolean {
    return (this.contexts?.length ?? 0) > 0 || (this.groups?.length ?? 0) > 0;
  }

  /**
   * Direct contexts lead in declaration order; group-resolved members follow
   * in name order, overlaps deduped — the server's own cross-search tie
   * order.
   */
  private async resolveTargets(): Promise<string[]> {
    const targets: string[] = [];
    if (this.context !== undefined) {
      targets.push(this.context);
    }
    for (const name of this.contexts ?? []) {
      if (!targets.includes(name)) {
        targets.push(name);
      }
    }
    const members = new Set<string>();
    const seen = new Set<string>();
    // BFS by frontier: the groups of one level are independent fetches,
    // so each level resolves concurrently (nesting is at most 3 levels
    // deep server-side).
    let frontier = [...(this.groups ?? [])];
    while (frontier.length > 0) {
      const fetch = [...new Set(frontier)].filter((name) => !seen.has(name));
      for (const name of fetch) {
        seen.add(name);
      }
      const entries = await Promise.all(fetch.map((name) => this.client.groups.get(name)));
      frontier = [];
      for (const entry of entries) {
        for (const member of entry.contexts) {
          members.add(member);
        }
        frontier.push(...entry.groups);
      }
    }
    for (const name of [...members].sort()) {
      if (!targets.includes(name)) {
        targets.push(name);
      }
    }
    return targets;
  }

  private async graphLane(target: string, query: string): Promise<Document[]> {
    const ctx = this.client.context(target);
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
    if (origins.length === 0) {
      return [];
    }
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
    return graphDocuments(page.matches, citations, this.include_graph_only_facts, target);
  }

  async _getRelevantDocuments(
    query: string,
    _runManager?: CallbackManagerForRetrieverRun,
  ): Promise<Document[]> {
    if (!this.isCross()) {
      const target = this.context!;
      const graphDocs = this.include_graph ? await this.graphLane(target, query) : [];
      let textHits: PassageHit[] = [];
      if (this.include_text) {
        textHits = await this.client.context(target).searchPassages(query, {
          limit: this.text_limit,
        });
      }
      return mergeLanes(graphDocs, textHits, this.k, target);
    }

    const targets = await this.resolveTargets();
    let graphDocs: Document[] = [];
    if (this.include_graph) {
      // Each target's lane is an independent chain of round trips, and
      // completion order is irrelevant (interleave sorts by rank, then
      // target order) — run them concurrently.
      graphDocs = interleave(
        await Promise.all(targets.map((target) => this.graphLane(target, query))),
      );
    }
    let textHits: PassageHit[] = [];
    if (this.include_text) {
      textHits = await this.client.searchPassages(query, {
        contexts: targets,
        limit: this.text_limit,
      });
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
  context: string,
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
              context,
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
            context,
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

/**
 * Per-context rank interleaving — activation strengths are ordinal within
 * one call only, so ranks are the currency across contexts (the posture the
 * server's own cross-context passage merge takes).
 */
function interleave(perTarget: Document[][]): Document[] {
  const indexed: Array<[number, number, Document]> = [];
  perTarget.forEach((documents, index) => {
    documents.forEach((document, rank) => {
      indexed.push([rank, index, document]);
    });
  });
  indexed.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return indexed.map(([, , document]) => document);
}

/**
 * A cross-context hit names its context; a per-context hit inherits the
 * retriever's own target.
 */
function hitContext(hit: PassageHit, fallback: string | undefined): string | undefined {
  const tagged = (hit as { context?: string }).context;
  return tagged || fallback;
}

function textDocument(hit: PassageHit, context: string | undefined): Document {
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
      context: context ?? null,
      source: hit.source,
      paragraph: hit.paragraph,
      section: null,
      lane: "text",
      score: hit.score,
      lanes,
    },
  });
}

function mergeLanes(
  graphDocs: Document[],
  textHits: PassageHit[],
  k: number,
  fallbackContext?: string,
): Document[] {
  const scored = new Map<string, { score: number; document: Document }>();

  const keyOf = (document: Document): string => {
    const context = (document.metadata["context"] as string | null | undefined) ?? "";
    if (document.metadata["paragraph"] !== null && document.metadata["paragraph"] !== undefined) {
      return `${context}\u0000${citationKey(document.metadata["source"] as string, document.metadata["paragraph"] as number)}`;
    }
    const associations = (document.metadata["associations"] as AssociationMeta[] | undefined) ?? [];
    const first = associations[0];
    return `fact\u0000${context}\u0000${first?.subject}\u0000${first?.label}\u0000${first?.object}`;
  };

  graphDocs.forEach((document, rank) => {
    scored.set(keyOf(document), { score: rrf(rank), document });
  });

  textHits.forEach((hit, rank) => {
    const context = hitContext(hit, fallbackContext);
    const key = `${context ?? ""}\u0000${citationKey(hit.source, hit.paragraph)}`;
    const existing = scored.get(key);
    if (existing === undefined) {
      scored.set(key, { score: rrf(rank), document: textDocument(hit, context) });
    } else {
      existing.document.metadata["lane"] = "graph+text";
      existing.document.metadata["score"] = hit.score;
      existing.document.metadata["lanes"] = textDocument(hit, context).metadata["lanes"];
      existing.score += rrf(rank);
    }
  });

  return [...scored.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((entry) => entry.document);
}
