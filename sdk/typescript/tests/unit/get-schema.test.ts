/**
 * `getSchema` returns the same `SchemaDocument` shape `{stem}.schema.json`/
 * `GET /contexts/{name}/schema` persist and serve (ADR 0009 §5) — the same
 * document `taguru extract --schema`/both LangChain ingesters consume.
 */

import { describe, expect, it } from "vitest";

import { NotFoundError } from "../../src/errors.js";
import type { SchemaAudit, SchemaDocument } from "../../src/models.js";
import { type StubRequest, errBody, okBody, stubClient } from "./stub.js";

const SCHEMA_DOCUMENT: SchemaDocument = {
  schema: 1,
  mode: "strict",
  closed_labels: false,
  types: {
    Brewery: { is_a: ["Organization"] },
    Organization: { is_a: [] },
    Person: { is_a: [] },
  },
  relations: {
    杜氏: { domain: ["Brewery"], range: ["Person"] },
  },
};

describe("getSchema", () => {
  it("decodes the schema document", async () => {
    const client = stubClient(() => okBody(SCHEMA_DOCUMENT));
    const document = await client.context("aomine").getSchema();

    expect(document.schema).toBe(1);
    expect(document.mode).toBe("strict");
    expect(document.closed_labels).toBe(false);
    expect(Object.keys(document.types).sort()).toEqual(["Brewery", "Organization", "Person"]);
    expect(document.types["Brewery"]?.is_a).toEqual(["Organization"]);
    expect(document.relations["杜氏"]).toEqual({ domain: ["Brewery"], range: ["Person"] });
  });

  it("rejects with NotFoundError when the context has no schema", async () => {
    const client = stubClient(() =>
      errBody(404, "context 'aomine' has no schema document", undefined, "no_schema"),
    );
    const notFound = await client
      .context("aomine")
      .getSchema()
      .catch((caught: unknown) => caught);
    expect(notFound).toBeInstanceOf(NotFoundError);
    expect((notFound as NotFoundError).code).toBe("no_schema");
  });
});

const SCHEMA_AUDIT: SchemaAudit = {
  total: 1,
  violations: [
    {
      association: {
        subject: "青嶺酒造",
        label: "杜氏",
        object: "広島",
        weight: 1.0,
        count: 1,
        attributions: [],
      },
      issues: [
        {
          path: "edge(青嶺酒造, 杜氏, 広島)",
          kind: "range",
          expected: "one of [Person]",
          actual: "Prefecture",
        },
      ],
    },
  ],
  untyped_concepts: { total: 2, names: ["広島", "青嶺"] },
  undeclared_types: { total: 0, names: [] },
  unknown_labels: { total: 0, names: [] },
  reserved_alias_conflicts: { total: 1, aliases: { 種類: "schema:type" } },
};

describe("putSchema / auditSchema / validateSchema", () => {
  it("putSchema PUTs the document and returns it as installed", async () => {
    const requests: StubRequest[] = [];
    const client = stubClient((req) => {
      requests.push(req);
      return okBody(SCHEMA_DOCUMENT);
    });
    const installed = await client.context("aomine").putSchema(SCHEMA_DOCUMENT);
    expect(requests[0]?.method).toBe("PUT");
    expect(requests[0]?.path).toBe("/contexts/aomine/schema");
    expect(JSON.parse(requests[0]?.body ?? "")).toEqual(SCHEMA_DOCUMENT);
    expect(installed.mode).toBe("strict");
  });

  it("auditSchema and validateSchema decode the shared audit shape", async () => {
    const requests: StubRequest[] = [];
    const client = stubClient((req) => {
      requests.push(req);
      return okBody(SCHEMA_AUDIT);
    });

    const audit = await client.context("aomine").auditSchema({ limit: 10 });
    expect(requests[0]?.path).toBe("/contexts/aomine/schema/audit");
    expect(JSON.parse(requests[0]?.body ?? "")).toEqual({ limit: 10 });
    expect(audit.total).toBe(1);
    expect(audit.violations[0]?.association.object).toBe("広島");
    expect(audit.violations[0]?.issues[0]?.kind).toBe("range");
    expect(audit.reserved_alias_conflicts.aliases).toEqual({ 種類: "schema:type" });

    const validated = await client
      .context("aomine")
      .validateSchema(SCHEMA_DOCUMENT, { limit: 10 });
    expect(requests[1]?.path).toBe("/contexts/aomine/schema/validate");
    expect(JSON.parse(requests[1]?.body ?? "")).toEqual({ document: SCHEMA_DOCUMENT, limit: 10 });
    expect(validated.total).toBe(1);
  });
});
