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
```

Prefer one `add_associations` call per document; for a batch above the server
limit, `add_associations_batched` auto-chunks it. Each call pays for a full
durable write.

`AsyncTaguru` is the same surface with `async`/`await`. The behavioral
contract is the server's own protocol document — read it from the deployment
you target: `client.protocol()` (`GET /protocol`).

See the repository's `sdk/` directory for the full documentation, the
LangChain integration (`langchain-taguru`), and the cross-language surface
spec.
