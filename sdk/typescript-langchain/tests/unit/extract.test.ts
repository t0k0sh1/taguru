/** Ports of src/extract.rs's golden tests — same as the Python twin's suite. */

import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { Ajv2020 } from "ajv/dist/2020.js";
import { describe, expect, it } from "vitest";

import {
  candidateJson,
  chunk,
  correctiveAssistantTurnContent,
  correctiveMessage,
  correctiveValidationMessage,
  crossOutputIssues,
  evaluateAnswer,
  indicatesLengthLimit,
  indicatesRefusal,
  interpretModelOutput,
  InvalidFault,
  labeledDocument,
  MAX_ASSOCIATION_WEIGHT,
  MAX_LISTED_ISSUES,
  MAX_NAME_BYTES,
  merge,
  MODEL_OUTPUT_JSON_SCHEMA,
  parseModelOutput,
  renderBatch,
  splitParagraphs,
  SyntaxFault,
  systemPrompt,
  labelVocabulary,
  type Extraction,
  type ItemRules,
  type ModelAlias,
  type ModelAssociation,
  type ModelOutput,
} from "../../src/extract.js";

const association = (
  subject: string,
  label: string,
  object: string,
  weight: number,
  paragraph?: number,
): ModelAssociation => ({ subject, label, object, weight, paragraph: paragraph ?? null });

const alias = (spelling: string, canonical: string, kind: string): ModelAlias => ({
  alias: spelling,
  canonical,
  kind,
});

const output = (partial: Partial<ModelOutput>): ModelOutput => ({
  associations: [],
  aliases: [],
  questions: [],
  ...partial,
});

describe("merge (extract.rs golden ports)", () => {
  it("folds duplicates and drops what the contract refuses", () => {
    const merged = merge(
      [
        output({
          associations: [
            association("青嶺酒造", "杜氏", "高瀬", 1.0, 0),
            association("", "杜氏", "高瀬", 1.0), // empty name
            association("蔵", "重い", "石", 1e300), // over the weight cap
            association("蔵", "無", "石", 0.0), // zero asserts nothing
          ],
          aliases: [alias("Aomine", "青嶺酒造", "concept")],
        }),
        output({
          associations: [
            association("青嶺酒造", "杜氏", "高瀬", 2.0), // exact triple again
            association("青嶺酒造", "創業年", "1907年", 1.0, 99), // out of range
          ],
          aliases: [
            alias("Aomine", "青嶺酒造", "concept"), // same pair again
            alias("蔵元", "存在しない", "concept"), // canonical unknown
            alias("高瀬", "青嶺酒造", "concept"), // shadows a real name
            alias("青嶺酒造", "青嶺酒造", "concept"), // self
            alias("x", "青嶺酒造", "banana"), // unknown kind
            alias("設立年", "創業年", "label"), // canonical among labels
          ],
        }),
      ],
      0,
      2,
    );
    expect(merged.associations).toHaveLength(2);
    expect(merged.associations[0]!.weight).toBe(1.0); // chunk 0's copy survives
    expect(merged.associations[0]!.paragraph).toBe(0);
    expect(merged.associations[1]!.paragraph).toBeNull(); // tag dropped, fact kept
    expect([...merged.concepts.entries()]).toEqual([["Aomine", "青嶺酒造"]]);
    expect(merged.labels.get("設立年")).toBe("創業年");
    expect(merged.duplicates).toBe(2);
    expect(merged.dropped).toBe(7);
    expect(labelVocabulary(merged)).toContain("杜氏");
    expect(labelVocabulary(merged)).toContain("創業年");
  });

  it("trims names so whitespace variants fold", () => {
    const merged = merge(
      [
        output({
          associations: [
            association("  青嶺酒造  ", "杜氏", "高瀬", 1.0),
            association("青嶺酒造", "杜氏", "高瀬", 2.0),
          ],
          aliases: [alias("  Aomine  ", "  青嶺酒造  ", "concept")],
        }),
      ],
      0,
      0,
    );
    expect(merged.associations).toHaveLength(1);
    expect(merged.associations[0]!.subject).toBe("青嶺酒造");
    expect(merged.associations[0]!.weight).toBe(1.0);
    expect(merged.duplicates).toBe(1);
    expect(merged.concepts.get("Aomine")).toBe("青嶺酒造");
  });

  it("validates questions against the canonical paragraph count", () => {
    const merged = merge(
      [
        output({
          questions: [
            { paragraph: 0, question: "最初の質問?" },
            { paragraph: 0, question: "最初の質問?" }, // duplicate
            { paragraph: 0, question: "二つ目の質問?" }, // over cap 1
            { paragraph: 9, question: "範囲外?" },
            { paragraph: 1, question: "" }, // empty
          ],
        }),
      ],
      1,
      2,
    );
    expect(merged.questions).toEqual([[0, "最初の質問?"]]);
    expect(merged.duplicates).toBe(1);
    expect(merged.dropped).toBe(3);
  });

  it("does not mistake a cap-dropped question for a duplicate on repeat", () => {
    // Every document chunk sees the same paragraph list and independently
    // proposes questions for it, so an identical question re-proposed by a
    // later chunk is a realistic occurrence, not an edge case. Before the
    // fix it read as a *duplicate* on the repeat, mislabeling the
    // paragraph's overflow as deduplication instead of the cap that
    // actually caused it.
    const merged = merge(
      [
        output({
          questions: [
            { paragraph: 0, question: "質問A" },
            { paragraph: 0, question: "質問B" }, // over this run's N=1
          ],
        }),
        output({
          questions: [
            { paragraph: 0, question: "質問B" }, // re-proposed, still over the cap
          ],
        }),
      ],
      1,
      1,
    );
    expect(merged.questions).toEqual([[0, "質問A"]]);
    expect(merged.duplicates).toBe(0); // the repeat is still a cap drop, not a duplicate
    expect(merged.dropped).toBe(2);
  });
});

describe("chunking and paragraph split", () => {
  it("splits at paragraph boundaries and survives multibyte walls", () => {
    const text = "第一段落。\n\n第二段落。\n\n第三段落。";
    expect(chunk(text, 1000)).toEqual([text]);
    const split = chunk(text, 20);
    expect(split).toHaveLength(3);
    expect(split.every((piece) => Buffer.byteLength(piece, "utf-8") <= 20)).toBe(true);

    const wall = "あ".repeat(30);
    const pieces = chunk(wall, 32);
    expect(pieces.length).toBeGreaterThan(1);
    expect(pieces.every((piece) => Buffer.byteLength(piece, "utf-8") <= 32)).toBe(true);
    expect(pieces.join("")).toBe(wall);

    expect(chunk("   \n\n  ", 100)).toEqual([]);
  });

  it("never splits a surrogate pair across pieces, even under a cap smaller than one codepoint", () => {
    const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;
    const text = "😀".repeat(20); // U+1F600: 4 UTF-8 bytes, a UTF-16 surrogate pair
    const pieces = chunk(text, 3);
    expect(pieces.some((piece) => LONE_SURROGATE.test(piece))).toBe(false);
    expect(pieces.join("")).toBe(text);
  });

  it("mirrors the server's paragraph split", () => {
    const text = "\n最初の段落。\n二行目も同じ段落。\n\n \t \n次の段落。\n\n";
    expect(splitParagraphs(text)).toEqual(["最初の段落。\n二行目も同じ段落。", "次の段落。"]);
    expect(splitParagraphs("a\r\nb\r\n\r\nc\r\n")).toEqual(["a\r\nb", "c"]);
    expect(splitParagraphs("")).toEqual([]);
    expect(splitParagraphs("\n\n\n")).toEqual([]);
    expect(splitParagraphs("  \n　\n")).toEqual([]); // ideographic space is blank
    expect(splitParagraphs("一行だけ。")).toEqual(["一行だけ。"]);
    expect(splitParagraphs("一行だけ。\n")).toEqual(["一行だけ。"]);
  });

  it("numbers the canonical paragraphs in the prompt copy", () => {
    // A cap that dwarfs the paragraphs leaves the numbering untouched.
    expect(labeledDocument("一段落目。\n\n二段落目。", 10_000)).toBe(
      "[0] 一段落目。\n\n[1] 二段落目。",
    );
  });

  it("repeats an oversized paragraph's number on every continuation", () => {
    // One paragraph far larger than the cap: split at its interior line
    // breaks, every piece must still name paragraph 0 so the model can
    // attribute a question drawn from any of them. The old label-then-
    // byte-split left every piece past the first unlabeled.
    const body = "あ\n".repeat(40);
    const cap = Math.floor((Buffer.byteLength("[0] ", "utf-8") + Buffer.byteLength(body, "utf-8")) / 3);
    const labeled = labeledDocument(body, cap);
    const blocks = labeled.split("\n\n");
    expect(blocks.length).toBeGreaterThan(1);
    expect(blocks.every((block) => block.startsWith("[0] "))).toBe(true);

    // chunk() packs the pre-sized blocks without re-splitting, so the
    // label survives to what the model sees: every \n\n-delimited block in
    // every chunk still opens with the paragraph number.
    const chunks = chunk(labeled, cap);
    expect(
      chunks.flatMap((piece) => piece.split("\n\n")).every((block) => block.startsWith("[0] ")),
    ).toBe(true);
  });
});

describe("the system prompt's fact budget", () => {
  it("omits the fact-budget clause by default", () => {
    expect(systemPrompt([], 0)).not.toContain("association(s) total");
  });

  it("states the fact budget when set", () => {
    expect(systemPrompt([], 0, 5)).toContain("at most 5 association(s) total");
  });
});

describe("corrective-turn replay", () => {
  it("replays the prior answer in full by default", () => {
    expect(correctiveAssistantTurnContent("not json at all", undefined)).toBe("not json at all");
  });

  it("omits the prior answer at a zero cap", () => {
    expect(correctiveAssistantTurnContent("not json at all", 0)).toBe(
      "[omitted: not the requested JSON object]",
    );
  });

  it("truncates at a char boundary under a cap", () => {
    // The cap (3) lands one byte into the 3-byte "…" that starts at byte
    // 2; the cut must back off to the boundary, not split the character.
    expect(correctiveAssistantTurnContent("ab…cd", 3)).toBe("ab… [truncated to 3 bytes]");
  });

  it("leaves content under the cap untouched", () => {
    expect(correctiveAssistantTurnContent("short", 1000)).toBe("short");
  });
});

describe("indicatesLengthLimit", () => {
  it("is true only for output-cap finish reasons", () => {
    expect(indicatesLengthLimit("length")).toBe(true);
    expect(indicatesLengthLimit("max_tokens")).toBe(true);
    expect(indicatesLengthLimit("stop")).toBe(false);
    expect(indicatesLengthLimit("content_filter")).toBe(false);
    expect(indicatesLengthLimit(undefined)).toBe(false);
  });
});

describe("correctiveMessage", () => {
  it("matches today's fixed text when not length-limited", () => {
    const expected =
      "That was not the single JSON object asked for (bad json). " +
      "Answer again with only the JSON object.";
    expect(correctiveMessage("bad json", false, 0)).toBe(expected);
    // A set fact budget changes nothing here — nothing was cut off, so
    // there is nothing to shorten.
    expect(correctiveMessage("bad json", false, 5)).toBe(expected);
  });

  it("asks for SHORTER when length-limited", () => {
    const message = correctiveMessage("bad json", true, 0);
    expect(message).toContain("SHORTER");
    expect(message).toContain("bad json");
    expect(message).not.toContain("association(s) total");
  });

  it("names the fact budget when length-limited and set", () => {
    expect(correctiveMessage("bad json", true, 5)).toContain(
      "Keep it to at most 5 association(s) total.",
    );
  });
});

describe("model-answer parsing", () => {
  const payload = '{"associations": [{"subject": "a", "label": "b", "object": "c", "weight": 1.0}]}';

  it("tolerates fences and prose", () => {
    expect(parseModelOutput(payload).associations).toHaveLength(1);
    expect(parseModelOutput("```json\n" + payload + "\n```").associations).toHaveLength(1);
    expect(
      parseModelOutput(`Here you go:\n${payload}\nHope that helps!`).associations,
    ).toHaveLength(1);
  });

  it("names empty and non-JSON failures", () => {
    expect(() => parseModelOutput("")).toThrow(/empty/);
    expect(() => parseModelOutput("no json here")).toThrow(/not a JSON object/);
  });

  it("no longer coerces numeric strings or bools — issue #180/#181 drops that pydantic-lax-mode parity in favor of the Rust producer's stricter parsing", () => {
    // ADR 0001 §8/§11 unify parsing on the Rust producer's lenient walk
    // (interpretModelOutput), which never coerces a string or bool into a
    // number for weight/paragraph — a present-but-wrong-typed scalar just
    // reads as null in lossy mode (parseModelOutput), the same as an
    // absent or JSON-null one. The path-addressed issue this now raises
    // in strict mode is covered by the interpretModelOutput suite below.
    const strings = parseModelOutput(
      '{"associations": [{"subject": "a", "label": "b", "object": "c",' +
        ' "weight": "1.5", "paragraph": "2"}]}',
    );
    expect(strings.associations[0]!.weight).toBeNull();
    expect(strings.associations[0]!.paragraph).toBeNull();

    const bools = parseModelOutput(
      '{"associations": [{"subject": "a", "label": "b", "object": "c",' +
        ' "weight": true, "paragraph": false}]}',
    );
    expect(bools.associations[0]!.weight).toBeNull();
    expect(bools.associations[0]!.paragraph).toBeNull();
  });

  it("never throws on a wrong-typed or fractional scalar in lossy mode — it silently nulls out", () => {
    // Same ADR 0001 §8 unification: what used to be a hard parse-level
    // rejection (coerceFloat/coerceInt/coerceString throwing) is now an
    // item-level issue that lossy-mode parseModelOutput discards, never a
    // whole-document parse failure.
    const output = parseModelOutput(
      '{"associations": [' +
        '{"subject":"a","label":"b","object":"c","weight":"abc"},' +
        '{"subject":"a","label":"b","object":"c","paragraph":3.5},' +
        '{"subject":"a","label":"b","object":"c","paragraph":"1e2"},' +
        '{"subject":42,"label":"b","object":"c"}' +
        "]}",
    );
    expect(output.associations[0]!.weight).toBeNull();
    expect(output.associations[1]!.paragraph).toBeNull();
    expect(output.associations[2]!.paragraph).toBeNull();
    expect(output.associations[3]!.subject).toBeNull();
  });
});

// -- issue #181: lenient validation walk / cross-chunk alias validation ---------
// Ports of the Rust/Python twins' own tests for interpretModelOutput,
// crossOutputIssues, evaluateAnswer, and correctiveValidationMessage — the
// same cross-language regression floor the rest of this file follows.

const permissiveRules = (): ItemRules => ({ paragraphCount: 100, questionsRequested: true });

const interpret = (text: string, rules: ItemRules): { output: ModelOutput; issues: string[] } => {
  const { value } = candidateJson(text);
  return interpretModelOutput(value, rules);
};

describe("interpretModelOutput (extract.rs/Python golden ports)", () => {
  it("a null or wrong-typed array field reads as empty, not a parse failure", () => {
    const nulled = parseModelOutput(
      '{"associations": null, "questions": [{"paragraph": 0, "question": "何?"}]}',
    );
    expect(nulled.associations).toEqual([]);

    const objectShaped = parseModelOutput(
      '{"associations": [{"subject": "a", "label": "l", "object": "b"}], "aliases": {}}',
    );
    expect(objectShaped.associations).toHaveLength(1);
    expect(objectShaped.aliases).toEqual([]);

    expect(
      parseModelOutput('{"associations": {"subject": "a", "label": "l", "object": "b"}}')
        .associations,
    ).toEqual([]);
    expect(parseModelOutput('{"associations": "none"}').associations).toEqual([]);
  });

  it("missing, wrong-typed, empty, and oversized are four distinct issues", () => {
    const oversized = "x".repeat(MAX_NAME_BYTES + 1);
    const text = JSON.stringify({
      associations: [
        { label: "l", object: "b" },
        { subject: 42, label: "l", object: "b" },
        { subject: "  ", label: "l", object: "b" },
        { subject: oversized, label: "l", object: "b" },
      ],
    });
    const { issues } = interpret(text, permissiveRules());
    expect(issues).toEqual([
      "associations[0].subject: missing",
      "associations[1].subject: expected a string, got number 42",
      "associations[2].subject: empty",
      `associations[3].subject: ${Buffer.byteLength(oversized, "utf-8")} bytes exceeds the 1024-byte cap`,
    ]);
  });

  it("an absent or null weight is valid but a wrong-typed one is an issue", () => {
    const text = JSON.stringify({
      associations: [
        { subject: "a", label: "l", object: "b" },
        { subject: "a", label: "l", object: "c", weight: null },
        { subject: "a", label: "l", object: "d", weight: "strong" },
      ],
    });
    const { output, issues } = interpret(text, permissiveRules());
    expect(output.associations[0]!.weight).toBeNull();
    expect(output.associations[1]!.weight).toBeNull();
    expect(output.associations[2]!.weight).toBeNull();
    expect(issues).toEqual([
      'associations[2].weight: expected finite non-zero number, got string "strong"',
    ]);
  });

  it("zero and over-cap weights report the offending value, not a type mismatch", () => {
    const text = JSON.stringify({
      associations: [
        { subject: "a", label: "l", object: "b", weight: 0 },
        { subject: "a", label: "l", object: "c", weight: MAX_ASSOCIATION_WEIGHT * 2 },
      ],
    });
    const { issues } = interpret(text, permissiveRules());
    expect(issues).toHaveLength(2);
    expect(issues[0]).toBe("associations[0].weight: expected finite non-zero number, got 0");
    expect(issues[1]).toMatch(/^associations\[1\]\.weight: expected finite non-zero number, got/);
    expect(issues[1]).toContain(`over the ${MAX_ASSOCIATION_WEIGHT} cap`);
  });

  it("an out-of-range association paragraph is untagged without an issue", () => {
    const rules: ItemRules = { paragraphCount: 1, questionsRequested: true };
    const inRange = interpret(
      '{"associations": [{"subject": "a", "label": "l", "object": "b", "paragraph": 99}]}',
      rules,
    );
    expect(inRange.output.associations[0]!.paragraph).toBe(99);
    expect(inRange.issues).toEqual([]);

    const wrongTyped = interpret(
      '{"associations": [{"subject": "a", "label": "l", "object": "b", "paragraph": "two"}]}',
      rules,
    );
    expect(wrongTyped.output.associations[0]!.paragraph).toBeNull();
    expect(wrongTyped.issues).toEqual([
      'associations[0].paragraph: expected an integer paragraph index, got string "two"',
    ]);
  });

  it("alias item issues cover missing, wrong kind, and self-alias", () => {
    const text = JSON.stringify({
      aliases: [
        { canonical: "b", kind: "concept" },
        { alias: "x", canonical: "b", kind: "person" },
        { alias: "y", canonical: "y", kind: "concept" },
      ],
    });
    const { issues } = interpret(text, permissiveRules());
    expect(issues).toEqual([
      "aliases[0].alias: missing",
      'aliases[1].kind: expected "concept" or "label", got "person"',
      "aliases[2].alias: equals its canonical",
    ]);
  });

  it("question issues cover missing, out-of-range, and oversized", () => {
    const text = JSON.stringify({
      questions: [
        { question: "何?" },
        { paragraph: 9, question: "何?" },
        { paragraph: 0, question: "  " },
      ],
    });
    const rules: ItemRules = { paragraphCount: 2, questionsRequested: true };
    const { issues } = interpret(text, rules);
    expect(issues).toEqual([
      "questions[0].paragraph: missing",
      "questions[1].paragraph: must cite a paragraph below 2, got 9",
      "questions[2].question: empty",
    ]);
  });

  it("a volunteered question when none was requested is a policy trim, not an issue", () => {
    const rules: ItemRules = { paragraphCount: 2, questionsRequested: false };
    const { output, issues } = interpret('{"questions": [{"question": "何?"}]}', rules);
    expect(output.questions).toHaveLength(1);
    expect(issues).toEqual([]);
  });
});

describe("crossOutputIssues (extract.rs/Python golden ports)", () => {
  it("lets a canonical resolved in a later output through", () => {
    const outputs = [
      output({ aliases: [alias("Aomine", "青嶺酒造", "concept")] }),
      output({ associations: [association("青嶺酒造", "杜氏", "高瀬", 1.0)] }),
    ];
    expect(crossOutputIssues(outputs)).toEqual([]);
  });

  it("names dangling and shadowing aliases by output", () => {
    const outputs = [
      output({
        associations: [association("青嶺酒造", "杜氏", "高瀬", 1.0)],
        aliases: [
          alias("蔵元", "存在しない", "concept"), // dangling: no such name
          alias("高瀬", "青嶺酒造", "concept"), // shadows a real name
        ],
      }),
    ];
    expect(crossOutputIssues(outputs)).toEqual([
      [
        0,
        [
          "aliases[0].canonical: names nothing the associations contain",
          "aliases[1].alias: names something the associations already contain",
        ],
      ],
    ]);
  });

  it("blames the later output for a conflicting canonical", () => {
    const outputs = [
      output({
        associations: [
          association("青嶺酒造", "杜氏", "高瀬", 1.0),
          association("蔵元本店", "支店", "青嶺酒造", 1.0),
        ],
        aliases: [alias("Aomine", "青嶺酒造", "concept")],
      }),
      output({ aliases: [alias("Aomine", "蔵元本店", "concept")] }), // same spelling, different canonical
    ];
    expect(crossOutputIssues(outputs)).toEqual([
      [1, ['aliases[0]: conflicts with an earlier alias mapping "Aomine" to "青嶺酒造"']],
    ]);
  });

  it("folds an identical repeated mapping without a conflict", () => {
    const outputs = [
      output({
        associations: [association("青嶺酒造", "杜氏", "高瀬", 1.0)],
        aliases: [alias("Aomine", "青嶺酒造", "concept")],
      }),
      output({ aliases: [alias("Aomine", "青嶺酒造", "concept")] }), // identical repeat
    ];
    expect(crossOutputIssues(outputs)).toEqual([]);
  });

  it("skips aliases Stage 1 already flagged", () => {
    const outputs = [
      output({
        associations: [association("青嶺酒造", "杜氏", "高瀬", 1.0)],
        aliases: [
          alias("青嶺酒造", "青嶺酒造", "concept"), // self-alias, Stage 1's issue
          { alias: null, canonical: "青嶺酒造", kind: "concept" },
        ],
      }),
    ];
    expect(crossOutputIssues(outputs)).toEqual([]);
  });
});

describe("indicatesRefusal", () => {
  it("is true only for refusal reasons", () => {
    expect(indicatesRefusal("content_filter")).toBe(true);
    expect(indicatesRefusal("refusal")).toBe(true);
    expect(indicatesRefusal("stop")).toBe(false);
    expect(indicatesRefusal("length")).toBe(false);
    expect(indicatesRefusal("tool_calls")).toBe(false);
  });
});

describe("correctiveValidationMessage", () => {
  it("lists every issue and states the five-part contract", () => {
    const issues = [
      'associations[1].weight: expected finite non-zero number, got string "strong"',
      "aliases[0].canonical: names nothing the associations contain",
    ];
    const message = correctiveValidationMessage(issues);
    expect(message.startsWith("That was valid JSON but not a valid extraction (2 issue(s)):")).toBe(
      true,
    );
    expect(message).toContain(issues[0]);
    expect(message).toContain(issues[1]);
    expect(message).toContain("complete corrected JSON object");
    expect(message).toContain("keep every item");
    expect(message).toContain("correct the fields listed above instead of deleting");
    expect(message).toContain("add nothing that was not already there");
    expect(message).toContain("only the JSON object");
  });

  it("caps the listed issues", () => {
    const issues = Array.from(
      { length: MAX_LISTED_ISSUES + 3 },
      (_, i) => `associations[${i}].weight: expected finite non-zero number, got 0`,
    );
    const message = correctiveValidationMessage(issues);
    expect(message).toContain(`(${issues.length} issue(s))`);
    expect(message).toContain("… and 3 more issue(s)");
    expect(message).not.toContain(issues[MAX_LISTED_ISSUES]!);
  });
});

describe("evaluateAnswer", () => {
  it("in strict mode surfaces validity issues; lossy mode ignores them", () => {
    const content = JSON.stringify({
      associations: [{ subject: "a", label: "l", object: "b", weight: "strong" }],
    });
    const strictRules: ItemRules = { paragraphCount: 1, questionsRequested: false };
    try {
      evaluateAnswer(content, strictRules);
      expect.unreachable("expected InvalidFault");
    } catch (error) {
      expect(error).toBeInstanceOf(InvalidFault);
      expect((error as InvalidFault).issues).toEqual([
        'associations[0].weight: expected finite non-zero number, got string "strong"',
      ]);
    }

    // Lossy mode (rules: null) ignores the same issue and hands back the
    // parsed output, byte-for-byte parseModelOutput's behavior.
    const { output } = evaluateAnswer(content, null);
    expect(output.associations).toHaveLength(1);
    expect(output.associations[0]!.weight).toBeNull();
  });

  it("reports a syntax fault before any validation", () => {
    const strictRules: ItemRules = { paragraphCount: 1, questionsRequested: false };
    try {
      evaluateAnswer("not json at all", strictRules);
      expect.unreachable("expected SyntaxFault");
    } catch (error) {
      expect(error).toBeInstanceOf(SyntaxFault);
      expect((error as Error).message).toMatch(/not a JSON object/);
    }
  });
});

describe("candidateJson lossless repairs (issue #181-only additions)", () => {
  it("strips a BOM and reports it", () => {
    const { value, repairs } = candidateJson("﻿" + '{"associations": []}');
    expect(value).toEqual({ associations: [] });
    expect(repairs).toEqual(["bom"]);
  });

  it("removes unambiguous trailing commas", () => {
    const { value, repairs } = candidateJson(
      '{"associations": [{"subject": "a", "label": "l", "object": "b",},],}',
    );
    expect(value).toEqual({ associations: [{ subject: "a", label: "l", object: "b" }] });
    expect(repairs).toEqual(["trailing_comma"]);

    // A comma inside a string value must never be touched.
    const untouched = candidateJson(
      '{"associations": [{"subject": "a, b", "label": "l", "object": "c"}]}',
    );
    expect((untouched.value as { associations: [{ subject: string }] }).associations[0]!.subject).toBe(
      "a, b",
    );
    expect(untouched.repairs).toEqual([]);
  });

  it("labels fence and braces repairs", () => {
    const payload = '{"associations": []}';
    expect(candidateJson("```json\n" + payload + "\n```").repairs).toEqual(["code_fence"]);
    expect(candidateJson(`Here you go:\n${payload}\nHope that helps!`).repairs).toEqual([
      "braces_slice",
    ]);
  });
});

describe("batch rendering", () => {
  it("carries the import line shapes", () => {
    const extraction = merge(
      [
        output({
          associations: [association("青嶺酒造", "杜氏", "高瀬", 2.0)],
          aliases: [alias("Aomine", "青嶺酒造", "concept")],
          questions: [{ paragraph: 1, question: "二行目には何が書いてある?" }],
        }),
      ],
      2,
      2,
    );
    const body = renderBatch("sake", "docs/aomine.md", "酒蔵の記憶", extraction, "一段落目。\n\n二段落目。");
    const lines = body.trim().split("\n").map((line) => JSON.parse(line));
    expect(lines).toHaveLength(5);
    expect(lines[0]).toEqual({
      taguru_batch: 1,
      context: "sake",
      source: "docs/aomine.md",
      create: { description: "酒蔵の記憶" },
    });
    expect(lines[1]).toEqual({ passage: "一段落目。\n\n二段落目。" });
    expect(lines[2]).toEqual({ paragraph: 1, question: "二行目には何が書いてある?" });
    expect(lines[3]).toEqual({ subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 2.0 });
    expect(lines[4]).toEqual({ alias: "Aomine", canonical: "青嶺酒造", kind: "concept" });
  });

  it("strips paragraph locators without a passage", () => {
    const extraction = merge(
      [output({ associations: [association("a", "b", "c", 1.0, 0)] })],
      0,
      1,
    );
    const body = renderBatch("ctx", "src", null, extraction, null);
    const lines = body.trim().split("\n").map((line) => JSON.parse(line));
    expect(lines).toHaveLength(2);
    expect(lines[1]).not.toHaveProperty("paragraph");
  });

  it("orders aliases by alias then canonical, not by their comma-joined string form", () => {
    // "a,b" -> "c" and "a" -> "b,c" both join to the identical string "a,b,c",
    // so a bare `.sort()` (which stringifies each tuple) would leave them in
    // insertion order instead of ordering by alias.
    const extraction: Extraction = {
      associations: [],
      concepts: new Map([
        ["a,b", "c"],
        ["a", "b,c"],
      ]),
      labels: new Map(),
      questions: [],
      duplicates: 0,
      dropped: 0,
    };
    const body = renderBatch("ctx", "src", null, extraction, null);
    const lines = body.trim().split("\n").map((line) => JSON.parse(line));
    expect(lines).toHaveLength(3);
    expect(lines[1]).toEqual({ alias: "a", canonical: "b,c", kind: "concept" });
    expect(lines[2]).toEqual({ alias: "a,b", canonical: "c", kind: "concept" });
  });
});

describe("JSON Schema", () => {
  // tests/unit/extract.test.ts -> repo root: same depth as the Rust twin's
  // CARGO_MANIFEST_DIR-relative path and the Python twin's parents[4].
  const fixturesRoot = join(
    dirname(fileURLToPath(import.meta.url)),
    "../../../../tests/fixtures/model_output",
  );
  const listFixtures = (kind: "accepted" | "rejected"): string[] =>
    readdirSync(join(fixturesRoot, kind))
      .filter((name) => name.endsWith(".json"))
      .map((name) => join(fixturesRoot, kind, name));
  const acceptedFixtures = listFixtures("accepted");
  const rejectedFixtures = listFixtures("rejected");

  const ajv = new Ajv2020({ allErrors: true });
  const validate = ajv.compile(MODEL_OUTPUT_JSON_SCHEMA);

  it("has a non-empty shared fixture corpus", () => {
    expect(acceptedFixtures.length).toBeGreaterThan(0);
    expect(rejectedFixtures.length).toBeGreaterThan(0);
  });

  // MODEL_OUTPUT_JSON_SCHEMA against tests/fixtures/model_output — the same
  // corpus the Rust and Python copies validate against, so the three
  // mirrored schemas cannot silently drift apart.
  it.each(acceptedFixtures)("accepts %s", (path) => {
    const text = readFileSync(path, "utf-8");
    expect(validate(JSON.parse(text))).toBe(true);
    expect(() => parseModelOutput(text)).not.toThrow();
  });

  it.each(rejectedFixtures)("rejects %s", (path) => {
    expect(validate(JSON.parse(readFileSync(path, "utf-8")))).toBe(false);
  });
});

describe("repaired fixtures (shared with the Rust/Python twins, issue #180/#181/#199)", () => {
  // Each tests/fixtures/model_output/repaired/*.json names one (rules,
  // answer, issues, corrected) tuple so all three producers can
  // mechanically check validate(answer) === issues and
  // validate(corrected) === [] against the SAME payloads.
  const fixturesRoot = join(
    dirname(fileURLToPath(import.meta.url)),
    "../../../../tests/fixtures/model_output/repaired",
  );
  const fixturePaths = readdirSync(fixturesRoot)
    .filter((name) => name.endsWith(".json"))
    .map((name) => join(fixturesRoot, name));

  interface RepairedFixture {
    rules: { paragraph_count: number; questions_cap: number };
    answer: Record<string, unknown>;
    issues: string[];
    corrected: Record<string, unknown>;
  }

  const validate = (payload: unknown, rules: ItemRules): string[] => {
    const { output, issues } = interpretModelOutput(payload, rules);
    for (const [, crossIssues] of crossOutputIssues([output])) {
      issues.push(...crossIssues);
    }
    return issues;
  };

  it("has a non-empty repaired fixture corpus", () => {
    expect(fixturePaths.length).toBeGreaterThan(0);
  });

  it.each(fixturePaths)("validates %s identically to the answer/corrected pair", (path) => {
    const fixture = JSON.parse(readFileSync(path, "utf-8")) as RepairedFixture;
    const rules: ItemRules = {
      paragraphCount: fixture.rules.paragraph_count,
      questionsRequested: fixture.rules.questions_cap > 0,
    };
    expect(fixture.issues.length).toBeGreaterThan(0);

    expect(validate(fixture.answer, rules)).toEqual(fixture.issues);
    expect(validate(fixture.corrected, rules)).toEqual([]);

    // Preserve-every-item (ADR 0001 §8 bucket 2's "correct-not-delete, add
    // nothing"): a whole-array field that WAS shaped as an array in the
    // answer must keep the same item count in corrected.
    for (const field of ["associations", "aliases", "questions"] as const) {
      const answerItems = fixture.answer[field];
      if (Array.isArray(answerItems)) {
        const correctedItems = fixture.corrected[field];
        const correctedLength = Array.isArray(correctedItems) ? correctedItems.length : 0;
        expect(correctedLength).toBe(answerItems.length);
      }
    }
  });
});
