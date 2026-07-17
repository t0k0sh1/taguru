import { describe, expect, it } from "vitest";

import { crossMatchCursor, matchCursor } from "../../src/models.js";
import type { Association, CrossAssociation } from "../../src/models.js";

describe("matchCursor", () => {
  it("narrows a full match down to the four MatchCursor fields", () => {
    const match: Association = {
      subject: "青嶺酒造",
      label: "杜氏",
      object: "高瀬",
      weight: 1.0,
      count: 2,
      attributions: [
        { source: "docs/aomine.md", weight: 1.0, count: 2, paragraph: null, section: null },
      ],
    };
    // `Association` structurally satisfies `MatchCursor`, so passing it
    // straight through would compile — but the server's `MatchCursor`
    // rejects `count`/`attributions` as unrecognized fields.
    expect(matchCursor(match)).toEqual({
      weight: 1.0,
      subject: "青嶺酒造",
      label: "杜氏",
      object: "高瀬",
    });
  });
});

describe("crossMatchCursor", () => {
  it("narrows a full cross-context match down to the five CrossMatchCursor fields", () => {
    const match: CrossAssociation = {
      subject: "青嶺酒造",
      label: "杜氏",
      object: "高瀬",
      weight: 1.0,
      count: 2,
      attributions: [],
      context: "sake",
    };
    expect(crossMatchCursor(match)).toEqual({
      weight: 1.0,
      context: "sake",
      subject: "青嶺酒造",
      label: "杜氏",
      object: "高瀬",
    });
  });
});
