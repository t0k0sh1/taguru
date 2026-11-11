# 0051. TypeScript SDK toolchain: TypeScript 7, oxlint instead of ESLint, no mutation gate

- **Status**: Accepted
- **Date**: 2026-11-12
- **Issue**: #1021
- **Related**: ADR 0049 (its two upstream waits, #1017 and #1018),
  Dependabot alert #162 (GHSA-g3hv-75wg-m249)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

- The SDK workspace's lockfile carried minimatch 10.2.6, inside
  GHSA-g3hv-75wg-m249's range (`>=7.0.0 <11.0.1`). Four dev
  dependencies pulled it in, and even their latest releases pin
  minimatch to 10.x, so no lockfile update could reach 11.0.1:
  - `eslint` 10.14.0 (`^10.2.6`)
  - `@eslint/config-array` 0.23.5 (`^10.2.4`)
  - `@typescript-eslint/typescript-estree` 8.72.2 (`^10.2.2`)
  - `@stryker-mutator/core` 10.1.1 (`~10.2.4`)

  ADR 0049 listed this as an upstream wait (#1017).
- TypeScript was 6.0.3; 7.0.2 is current. typescript-eslint 8.72.2
  declares `typescript >=4.8.4 <6.1.0`, so ESLint held TypeScript back
  too.
- Stryker also blocked vitest 5 (#1018).

## 2. Decision

1. **TypeScript 7** for both SDK packages and the LangChain examples.
2. **oxlint replaces ESLint** (eslint and typescript-eslint). The
   oxlint configuration enables typescript-eslint's `recommended` rules
   that the old config inherited, keeps its `no-unused-vars` exception
   for `_`-prefixed names, and turns oxlint's `correctness` category on
   as errors. `npm run lint` runs with `--deny-warnings`.
3. **The TypeScript SDK has no mutation gate.** Stryker is removed,
   together with the `typescript-mutation` CI job and its baseline,
   because it keeps minimatch in the tree after ESLint is gone. The
   Python SDK's `python-mutation` gate stays.

## 3. Consequences

- `npm audit` over the SDK workspace finds nothing; minimatch is no
  longer installed. #1017 is resolved.
- TypeScript 7 ships no compiler API, which two places used:
  - tsup's `--dts` (rollup-plugin-dts). The build now has tsup emit
    the JavaScript only, and `tsc -p tsconfig.build.json`
    (`emitDeclarationOnly`) emit the declarations, one file per
    module instead of one bundled file per entry point. The `types`
    paths in `package.json` are unchanged. The exported names and
    their types match the previous build's.
  - `scripts/check-surface.ts`, which now imports the API from the
    compatibility package `@typescript/typescript6`.
- oxlint's `correctness` rules, which ESLint never ran, found seven
  spots in `langchain-taguru`; all are fixed.
- A logic change in the TypeScript SDK is now covered by its unit and
  integration suites alone, with no mutation score behind them. If a
  mutation tool that does not pull in the advisory becomes available,
  bringing a gate back takes a new ADR.
- vitest 5 (#1018) is no longer blocked by Stryker; moving to it stays
  a separate change.
