#!/usr/bin/env node
/**
 * Mutation-testing gate for the core TypeScript SDK (`sdk/typescript`) —
 * the Stryker twin of `check_mutants.py` (sdk/python's mutmut gate).
 *
 * Stryker seeds deliberate faults into the source and reports which ones
 * the test suite fails to catch. A survivor is a hole in the tests: a line
 * whose behavior nothing actually asserts. This script runs Stryker, then
 * fails CI when a survivor appears that is not on the reviewed allowlist
 * of genuinely-equivalent mutants (`sdk/typescript/mutation-baseline.txt`).
 *
 *     node sdk/spec/check_mutants_ts.mjs            # run stryker, then verify
 *     node sdk/spec/check_mutants_ts.mjs --verify   # verify an existing report only
 *
 * Scope note: only the core SDK is gated, and only via the hermetic unit
 * suite (`vitest.stryker.config.ts`) — `src/client.ts`/`src/testing.ts`
 * lean on the server-backed integration suite exactly as sdk/python's
 * `_async`/`_sync`/`testing.py` do, and `sdk/typescript-langchain` stays a
 * manual tool for the same reason `sdk/python-langchain` does.
 *
 * Determinism: a baseline entry names one exact mutant as
 * `file:line:col mutator -> replacement` (newlines escaped). Editing a
 * function shifts its mutants' positions; regenerate that file's baseline
 * entries when that happens — the same renumbering caveat mutmut has.
 */

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { exit } from "node:process";

const SDK = join(dirname(fileURLToPath(import.meta.url)), "..", "typescript");
const BASELINE = join(SDK, "mutation-baseline.txt");
const REPORT = join(SDK, "reports", "mutation", "mutation.json");

/** The statuses that mean "the suite did not cleanly kill this" — the same
 * posture check_mutants.py takes with survived/timeout/suspicious. Stryker's
 * NoCoverage (no test even executes the mutated line) is included: within
 * the gated scope an uncovered line is as much a hole as an unasserted one. */
const ESCAPED_STATUSES = new Set(["Survived", "Timeout", "NoCoverage"]);

function loadBaseline() {
  let text;
  try {
    text = readFileSync(BASELINE, "utf-8");
  } catch {
    return new Set();
  }
  return new Set(
    text
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#")),
  );
}

/** One stable-ish name per mutant. Location shifts when the file is edited
 * (the documented renumbering caveat); the replacement text disambiguates
 * mutants that share a location. */
function mutantName(file, mutant) {
  const { line, column } = mutant.location.start;
  const replacement = (mutant.replacement ?? "").replace(/\\/g, "\\\\").replace(/\n/g, "\\n");
  return `${file}:${line}:${column} ${mutant.mutatorName} -> ${replacement}`;
}

function survivors() {
  const report = JSON.parse(readFileSync(REPORT, "utf-8"));
  const escaped = new Map();
  for (const [file, data] of Object.entries(report.files)) {
    for (const mutant of data.mutants) {
      if (ESCAPED_STATUSES.has(mutant.status)) {
        escaped.set(mutantName(file, mutant), mutant);
      }
    }
  }
  return escaped;
}

function main() {
  const verifyOnly = process.argv.includes("--verify");
  if (!verifyOnly) {
    const result = spawnSync("npx", ["stryker", "run"], {
      cwd: SDK,
      stdio: "inherit",
    });
    if (result.error) {
      console.error(`failed to run stryker: ${result.error.message}`);
      return 1;
    }
    // Stryker's own exit code is threshold-based; the baseline comparison
    // below is the actual gate, so a non-zero here is not itself fatal.
  }

  const baseline = loadBaseline();
  const escaped = survivors();

  const fresh = [...escaped.keys()].filter((name) => !baseline.has(name)).sort();
  const stale = [...baseline].filter((name) => !escaped.has(name)).sort();

  if (stale.length > 0) {
    console.error(
      "Baseline entries that no longer escape (regenerate them — a file's " +
        "mutant locations shift when it is edited):",
    );
    for (const name of stale) {
      console.error(`  - ${name}`);
    }
  }

  if (fresh.length > 0) {
    console.error(
      `\n${fresh.length} mutant(s) survived the test suite and are NOT in the ` +
        "reviewed baseline — add a test that kills each, or, if it is a genuine " +
        "equivalent mutant, document it in sdk/typescript/mutation-baseline.txt:",
    );
    for (const name of fresh) {
      const mutant = escaped.get(name);
      console.error(`  - ${name}`);
      console.error(`      status: ${mutant.status}`);
    }
    return 1;
  }

  // A stale baseline is worth surfacing but not worth failing a merge over:
  // it means the tests got STRICTER, never weaker.
  console.log(`OK: every escaped mutant (${escaped.size}) is on the reviewed baseline.`);
  return 0;
}

exit(main());
