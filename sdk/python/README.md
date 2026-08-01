# taguru (Python SDK)

Official Python client SDK for the [Taguru](https://github.com/t0k0sh1/taguru)
long-term semantic memory server. The TypeScript SDK (`taguru` on npm) exposes
the identical surface — method names differ only by casing convention
(`search_passages` ↔ `searchPassages`); data fields are the wire's own
snake_case in both.

```sh
pip install taguru
```

```python
from taguru import Taguru

client = Taguru()  # defaults: $TAGURU_URL / $TAGURU_API_TOKEN, else http://127.0.0.1:8248
client.contexts.create("sake", description="青嶺酒造という架空の酒蔵の知識")

ctx = client.context("sake")
ctx.add_associations([
    {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "docs/aomine.md"},
])
ctx.store_passages({"docs/aomine.md": "青嶺酒造は1907年創業。代表銘柄は「青嶺」。"})

result = ctx.retrieve("青嶺酒造")           # resolve → describe → activate → citations
hits = ctx.search_passages("1907年に創業した")  # text lane (phrase as an answer)
package = ctx.assemble_evidence("青嶺酒造", budget={"max_items": 10})  # server-side budgeted package
```

Prefer one `add_associations` call per document. Above the 10,000-association
request limit, `add_associations_batched` auto-chunks it; for corpus-scale
ingestion, use `POST /import` or `taguru import`. Each call pays for a full
durable write.

`AsyncTaguru` is the same surface with `async`/`await`. The behavioral
contract is the server's own protocol document — read it from the deployment
you target: `client.protocol()` (`GET /protocol`).

## Tracing

`retrieve()` composes a client-side loop the same way the server's own
`retrieve` MCP tool does — resolve → describe → query → activate → citations
→ passage fallback. Optionally trace it as one OpenTelemetry span tree,
`taguru.retrieve` with a child span per phase, joined to the server's own
request span through injected `traceparent`/`tracestate` headers:

```sh
pip install "taguru[otel]"  # pulls in opentelemetry-api; core install stays dependency-free otherwise
pip install opentelemetry-sdk  # an actual TracerProvider — pick whichever OTel SDK/exporter you use instead if not this one
```

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import trace

trace.set_tracer_provider(TracerProvider())  # wire up whichever exporter you use — the SDK just emits spans
```

With no `TracerProvider` configured (or `opentelemetry-api` not installed at
all), every tracing call in the SDK is a silent no-op — nothing about
`retrieve()`'s behavior or return value changes either way. See
[Tracing](https://t0k0sh1.github.io/taguru/tracing.html) for the full span
tree, the attribute/event vocabulary (shared with the TypeScript SDK via
`sdk/spec/tracing.yaml`), and the privacy rules (spans carry counts, flags,
and closed reason codes — never cue text, labels, or passage content).

See the repository's `sdk/` directory for the full documentation, the
LangChain integration (`langchain-taguru`), and the cross-language surface
spec.
