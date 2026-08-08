import { defineConfig } from "vitest/config";

// Mutation-testing twin of vitest.config.ts: the hermetic unit suite only.
// The integration suite spawns the real server binary (TAGURU_TEST_BIN),
// which a per-mutant run can neither afford nor needs — the same scope
// decision sdk/python's `[tool.mutmut]` makes with
// `pytest_add_cli_args_test_selection = ["tests/unit/"]`.
export default defineConfig({
  test: {
    include: ["tests/unit/**/*.test.ts"],
    testTimeout: 30_000,
    hookTimeout: 120_000,
  },
});
