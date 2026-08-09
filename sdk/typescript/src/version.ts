/**
 * This build's package version — the runtime twin of Python's
 * `taguru.__version__` (ADR 0005 §9.2). Nothing in `src/` read the
 * package version at runtime before this: `package.json`'s `version`
 * is build-time only, so a TypeScript-side compatibility check had
 * nothing local to compare `GET /version` against. One symbol, one
 * file, on purpose — `sdk/spec/check_versions.py`'s lockstep checker
 * needs an unambiguous regex target, and this value is locked to the
 * server's own version exactly the way `taguru.__version__` is.
 */
export const VERSION = "0.9.0";
