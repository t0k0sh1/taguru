import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    testTimeout: 30_000,
    hookTimeout: 120_000,
    // Worker processes exit cleanly even with undici keep-alive sockets open.
    pool: "forks",
  },
});
