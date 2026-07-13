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
```

Prefer one `addAssociations` call per document; for a batch above the server
limit, `addAssociationsBatched` auto-chunks it. Each call pays for a full
durable write.

The behavioral contract is the server's own protocol document — read it from
the deployment you target: `await client.protocol()` (`GET /protocol`).

`taguru/testing` (Node-only) spawns a real server binary for integration
tests — the twin of Python's `taguru.testing`.

See the repository's `sdk/` directory for the full documentation, the
LangChain integration (`langchain-taguru`), and the cross-language surface
spec.
