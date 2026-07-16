/** Ports of src/extract.rs's golden tests — same as the Python twin's suite. */

import { describe, expect, it } from "vitest";

import {
  chunk,
  labeledDocument,
  merge,
  parseModelOutput,
  renderBatch,
  splitParagraphs,
  labelVocabulary,
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
    expect(labeledDocument("一段落目。\n\n二段落目。")).toBe("[0] 一段落目。\n\n[1] 二段落目。");
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

  it("coerces model numbers as leniently as the pydantic twin", () => {
    // Numeric strings and bools ride through, matching pydantic's lax mode
    // (verified against ModelAssociation in the Python twin).
    const strings = parseModelOutput(
      '{"associations": [{"subject": "a", "label": "b", "object": "c",' +
        ' "weight": "1.5", "paragraph": "2"}]}',
    );
    expect(strings.associations[0]!.weight).toBe(1.5);
    expect(strings.associations[0]!.paragraph).toBe(2);

    const bools = parseModelOutput(
      '{"associations": [{"subject": "a", "label": "b", "object": "c",' +
        ' "weight": true, "paragraph": false}]}',
    );
    expect(bools.associations[0]!.weight).toBe(1);
    expect(bools.associations[0]!.paragraph).toBe(0);
  });

  it("rejects the numbers pydantic's lax mode rejects", () => {
    const parse = (assoc: string) => (): unknown =>
      parseModelOutput(`{"associations": [${assoc}]}`);
    // A non-numeric weight string.
    expect(parse('{"subject":"a","label":"b","object":"c","weight":"abc"}')).toThrow(
      /weight is not a number/,
    );
    // A fractional or exponent paragraph, whether a number or a string.
    expect(parse('{"subject":"a","label":"b","object":"c","paragraph":3.5}')).toThrow(
      /paragraph is not an integer/,
    );
    expect(parse('{"subject":"a","label":"b","object":"c","paragraph":"3.5"}')).toThrow(
      /paragraph is not an integer/,
    );
    expect(parse('{"subject":"a","label":"b","object":"c","paragraph":"1e2"}')).toThrow(
      /paragraph is not an integer/,
    );
    // A number where a name is expected stays strict, exactly like pydantic.
    expect(parse('{"subject":42,"label":"b","object":"c"}')).toThrow(/subject is not a string/);
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
});
