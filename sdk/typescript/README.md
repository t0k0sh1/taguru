# taguru (TypeScript/JavaScript SDK)

Official TypeScript/JavaScript client SDK for the
[Taguru](https://github.com/t0k0sh1/taguru) long-term semantic memory server.
The Python SDK (`taguru` on PyPI) exposes the identical surface — method names
differ only by casing convention (`searchPassages` ↔ `search_passages`); data
fields are the wire's own snake_case in both. Zero runtime dependencies
(built-in `fetch`), Node 20+.

```sh
npm install taguru
```

```typescript
import { Taguru } from "taguru";

const client = new Taguru(); // defaults: $TAGURU_URL / $TAGURU_API_TOKEN, else http://127.0.0.1:8248
await client.contexts.create("sake", { description: "青嶺酒造という架空の酒蔵の知識" });

const ctx = client.context("sake");
await ctx.addAssociations([
  { subject: "青嶺酒造", label: "代表銘柄", object: "青嶺", weight: 1.0, source: "docs/aomine.md" },
]);
await ctx.storePassages({ "docs/aomine.md": "青嶺酒造は1907年創業。代表銘柄は「青嶺」。" });

const result = await ctx.retrieve("青嶺酒造");            // resolve → describe → activate → citations
const hits = await ctx.searchPassages("1907年に創業した"); // text lane (phrase as an answer)
const pkg = await ctx.assembleEvidence("青嶺酒造", { budget: { max_items: 10 } }); // server-side budgeted package
```

Prefer one `addAssociations` call per document. Above the 10,000-association
request limit, `addAssociationsBatched` auto-chunks it; for corpus-scale
ingestion, use `POST /import` or `taguru import`. Each call pays for a full
durable write.

The behavioral contract is the server's own protocol document — read it from
the deployment you target: `await client.protocol()` (`GET /protocol`).

`taguru/testing` (Node-only) spawns a real server binary for integration
tests — the twin of Python's `taguru.testing`.

## Tracing

`retrieve()` composes a client-side loop the same way the server's own
`retrieve` MCP tool does — resolve → describe → query → activate → citations
→ passage fallback. Optionally trace it as one OpenTelemetry span tree,
`taguru.retrieve` with a child span per phase, joined to the server's own
request span through injected `traceparent`/`tracestate` headers:

```sh
npm install @opentelemetry/api  # optional peer dependency — core install stays dependency-free otherwise
```

```typescript
import { trace } from "@opentelemetry/api";
// wire up whichever OpenTelemetry SDK/exporter you use — this package just emits spans into it
```

With no `TracerProvider` configured (or `@opentelemetry/api` not installed at
all), every tracing call in the SDK is a silent no-op — nothing about
`retrieve()`'s behavior or return value changes either way; the package is
loaded through a lazily cached dynamic `import()`, never a static one, so a
plain `npm install taguru` never even attempts to resolve it. See
[Tracing](https://t0k0sh1.github.io/taguru/tracing.html) for the full span
tree, the attribute/event vocabulary (shared with the Python SDK via
`sdk/spec/tracing.yaml`), and the privacy rules (spans carry counts, flags,
and closed reason codes — never cue text, labels, or passage content).

See the repository's `sdk/` directory for the full documentation, the
LangChain integration (`langchain-taguru`), and the cross-language surface
spec.
