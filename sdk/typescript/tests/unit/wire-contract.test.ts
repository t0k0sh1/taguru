// Golden wire-contract fixtures (#301): the TypeScript SDK's own read
// of tests/fixtures/wire/, alongside Rust's generator/verifier
// (tests/http_api/contract.rs) and Python's
// (sdk/python/tests/unit/test_wire_contract.py). Two checks:
//
// - every fixture whose response this SDK already has a typed
//   interface for (MatchPage, PassagePage, ContextPage, ExplorePage,
//   ActivationPage, CommunityPage) unwraps through the real
//   `unwrapEnvelope` — the same function every live call uses —
//   without throwing;
// - every fixture's declared enum-like fields only carry values
//   shapes.json knows about.
//
// #216's evidence-assembly package and the MCP-specific envelope have
// no SDK model yet (#306 adds them); those fixtures are covered here
// only by the enum check below, not a typed unwrap — matching this
// SDK's own documented posture (ADR 0005 §2.5): no runtime schema
// validation beyond the envelope, so there is nothing stronger to run
// those fixtures through yet.

import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { unwrapEnvelope } from "../../src/transport.js";

// tests/unit/wire-contract.test.ts -> repo root: same depth as the
// Rust twin's CARGO_MANIFEST_DIR-relative path and the Python twin's
// parents[4] (unit, tests, typescript, sdk).
const WIRE_DIR = join(dirname(fileURLToPath(import.meta.url)), "../../../../tests/fixtures/wire");

interface Fixture {
  operation: string;
  contract: string;
  route: string;
  status: number;
  request: unknown;
  response: unknown;
}

function loadFixtures(): Array<[string, Fixture]> {
  const paths: string[] = [];
  for (const dir of ["http", "mcp"]) {
    for (const name of readdirSync(join(WIRE_DIR, dir)).filter((n) => n.endsWith(".json"))) {
      paths.push(join(WIRE_DIR, dir, name));
    }
  }
  paths.sort();
  return paths.map((path) => [path, JSON.parse(readFileSync(path, "utf-8")) as Fixture]);
}

const FIXTURES = loadFixtures();
const SHAPES = JSON.parse(readFileSync(join(WIRE_DIR, "shapes.json"), "utf-8")) as {
  enums: Record<string, string[]>;
};

/**
 * "foo.bar[].baz" — "[]" means "every array element" — the one small
 * path language shapes.json's `enums` keys use. Mirrored (not shared)
 * in sdk/spec/check_contract.py and tests/http_api/contract.rs: each
 * language keeps its own ~15-line copy rather than a cross-language
 * import.
 */
function collectByPath(value: unknown, segments: string[]): unknown[] {
  if (segments.length === 0) return [value];
  const [head, ...rest] = segments;
  const isArray = head!.endsWith("[]");
  const key = isArray ? head!.slice(0, -2) : head!;
  if (typeof value !== "object" || value === null || !(key in value)) return [];
  const next = (value as Record<string, unknown>)[key];
  if (isArray) {
    if (!Array.isArray(next)) return [];
    return next.flatMap((item) => collectByPath(item, rest));
  }
  return collectByPath(next, rest);
}

// operation -> the fixture this SDK already has a typed interface for.
const TYPED_OPERATIONS = [
  "recall",
  "contexts_list",
  "sources_search",
  "explore",
  "activate",
  "communities_search",
];

describe("golden wire-contract fixtures (#301)", () => {
  it("has a non-empty fixture corpus", () => {
    expect(FIXTURES.length).toBeGreaterThan(0);
    const operations = new Set(FIXTURES.map(([, fixture]) => fixture.operation));
    for (const operation of TYPED_OPERATIONS) {
      expect(operations.has(operation)).toBe(true);
    }
  });

  const typedFixtures = FIXTURES.filter(([, fixture]) =>
    TYPED_OPERATIONS.includes(fixture.operation),
  );

  it.each(typedFixtures)("%s unwraps through the real transport envelope", (_path, fixture) => {
    const result = unwrapEnvelope(fixture.status, JSON.stringify(fixture.response));
    expect(result).toBeTruthy();
  });

  it.each(FIXTURES)("%s: every declared enum only carries known values", (path, fixture) => {
    for (const [pathExpr, allowed] of Object.entries(SHAPES.enums)) {
      const allowedSet = new Set(allowed);
      for (const value of collectByPath(fixture, pathExpr.split("."))) {
        if (typeof value === "string" && !allowedSet.has(value)) {
          throw new Error(
            `${path}: ${pathExpr} carries ${JSON.stringify(value)}, which is not ` +
              "declared in shapes.json's enums",
          );
        }
      }
    }
  });
});
