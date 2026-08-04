/**
 * `getSchema` returns the same `SchemaDocument` shape `{stem}.schema.json`/
 * `GET /contexts/{name}/schema` persist and serve (ADR 0009 §5) — the same
 * document `taguru extract --schema`/both LangChain ingesters consume.
 */

import { describe, expect, it } from "vitest";

import { NotFoundError } from "../../src/errors.js";
import type { SchemaDocument } from "../../src/models.js";
import { errBody, okBody, stubClient } from "./stub.js";

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
