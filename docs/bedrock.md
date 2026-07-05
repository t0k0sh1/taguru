# Taguru on Amazon Bedrock

Taguru is model-agnostic: the server speaks HTTP, the bridge speaks MCP
over stdio, and neither cares which model drives the agent. This page
records the three Bedrock integration points and the sharp edges found
running all of them for real (agent: Claude Sonnet 4.5 via the `us.`
inference profile; embeddings: `amazon.titan-embed-text-v2:0`; both in
us-east-1).

## The agent side: Converse + taguru-mcp

Claude Code configured for Bedrock (`CLAUDE_CODE_USE_BEDROCK=1`) needs
nothing special — MCP servers are local child processes, so the
`claude mcp add taguru …` setup from the README works unchanged.

For your own agent loop on the Converse API, host `taguru-mcp` as an
MCP stdio child and translate mechanically:

- `initialize` returns the full protocol manual as `instructions`
  (served live from `GET /protocol`, so it carries the server's
  live-configuration trailer). Use it as the Converse `system` text —
  ideally behind a `cachePoint` block, since it is resent every turn.
- Each entry from `tools/list` maps 1:1 onto a Converse tool:
  `{"toolSpec": {"name", "description", "inputSchema": {"json": <MCP
  inputSchema>}}}` — the schemas are plain JSON Schema and pass
  through unmodified.
- On `stopReason == "tool_use"`, forward each block to `tools/call`
  and return the text as a `toolResult`; loop until `end_turn`.

Measured with Sonnet 4.5: a nine-fact Japanese memo ingested in eight
tool calls following the protocol discipline (create → one batched
`add_associations` → `store_passages` → `audit_coverage` → verify),
and a re-run over the same document produced the retract-then-reingest
diff sync with no double weighting — the manual alone carried the
discipline across a non-Anthropic host API.

Managed agent runtimes (Bedrock AgentCore and friends) want *remote*
MCP endpoints; `taguru-mcp` is stdio-only. Co-locate it in the agent's
container (`cargo install taguru` ships it), or skip MCP and put the
HTTP API behind an OpenAPI action group / Gateway target — it is a
small JSON API with bearer auth.

## Embeddings: Bedrock models behind a bridging proxy

Taguru speaks one embedding protocol: OpenAI-compatible
`POST {model, input}` → `{data: [{embedding}]}`. Bedrock's embedding
models sit behind `InvokeModel` + SigV4 with per-family body shapes.
Bridge with LiteLLM or AWS's Bedrock Access Gateway sample (both speak
OpenAI already) — or with a proxy this small (error handling elided;
bind loopback only):

```python
#!/usr/bin/env python3
# OpenAI-compatible /embeddings → Bedrock InvokeModel (Titan & Cohere).
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import boto3

bedrock = boto3.Session(region_name="us-east-1").client("bedrock-runtime")

def invoke(model_id, body):
    response = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
    return json.loads(response["body"].read())

def embed(model_id, texts, purpose):
    if model_id.startswith("cohere.embed"):
        # X-Taguru-Embed-Purpose is exactly the asymmetry Cohere wants.
        input_type = "search_document" if purpose == "index" else "search_query"
        out = []
        for i in range(0, len(texts), 96):  # Cohere caps texts at 96
            out += invoke(model_id, {"texts": texts[i : i + 96],
                                     "input_type": input_type})["embeddings"]
        return out
    shape = (lambda t: {"inputText": t, "dimensions": 512}) if "v2" in model_id \
        else (lambda t: {"inputText": t})  # Titan: one text per call
    return [invoke(model_id, shape(t))["embedding"] for t in texts]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        texts = request["input"]
        texts = [texts] if isinstance(texts, str) else texts
        purpose = self.headers.get("X-Taguru-Embed-Purpose", "query")
        vectors = embed(request["model"], texts, purpose)
        body = json.dumps({"data":
            [{"embedding": v, "index": i} for i, v in enumerate(vectors)]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

HTTPServer(("127.0.0.1", 8257), Handler).serve_forever()
```

Point Taguru at it:

```sh
TAGURU_EMBED_URL=http://127.0.0.1:8257/v1/embeddings
TAGURU_EMBED_MODEL=amazon.titan-embed-text-v2:0
TAGURU_SEMANTIC_FLOOR=0.2   # calibrated below — do not skip this
TAGURU_EMBED_AUTO=1         # agents cannot be counted on to refresh
```

Taguru normalizes incoming vectors, so raw (unnormalized) Titan output
is fine. Note the fixed `dimensions: 512` — vectors from different
dimension settings must never share a sidecar, and Taguru keys the
vector cache by model *name* only, so pick the dimension once.

## Calibrating `TAGURU_SEMANTIC_FLOOR`

The built-in floor (0.35) is calibrated for `text-embedding-3-large`
glosses. Titan V2 (512d) compresses Japanese cosines: measured true
matches landed at 0.2–0.3 with unrelated names at ~0.15, so the
default silently discarded every correct answer — resolve returned
`[]` while the ranking underneath was perfect.

Calibrate once per embedding model:

1. Ingest something representative; refresh embeddings.
2. Probe with paraphrase cues that share no spelling with any stored
   name: `POST /contexts/{name}/resolve
   {"cue": "…", "semantic_floor": 0.05}`.
3. Read the two score bands and set `TAGURU_SEMANTIC_FLOOR` between
   them.

| model | floor |
| --- | --- |
| `text-embedding-3-large` | 0.35 (the default) |
| `amazon.titan-embed-text-v2:0` (512d, Japanese) | ~0.2 |

## When Bedrock refuses to serve the model

Third-party model access can *flap* — succeed, then deny the identical
request minutes later — while console-side state propagates. Skip the
guessing; one call names every gate:

```sh
aws bedrock get-foundation-model-availability \
  --model-id anthropic.claude-sonnet-4-5-20250929-v1:0
```

- `regionAvailability` — offered in this region at all
- `entitlementAvailability`
- `authorizationStatus` — the Anthropic use-case form (console, once
  per account; individuals: your own name and a GitHub URL are fine)
- `agreementAvailability` — the AWS Marketplace agreement

`agreementAvailability: NOT_AVAILABLE` is the one that resists console
fixes (invocation errors keep blaming IAM while the console shows
nothing pending). Create the agreement directly:

```sh
aws bedrock list-foundation-model-agreement-offers --model-id <id>   # → offerToken
aws bedrock create-foundation-model-agreement --model-id <id> --offer-token <token>
```

Bedrock model offers are $0-upfront, usage-based. Creating the
agreement requires `aws-marketplace:Subscribe` +
`aws-marketplace:ViewSubscriptions` on the *calling* identity, once;
plain invocation afterwards needs only `bedrock:InvokeModel` (drop the
marketplace grant again after it succeeds).

Two more traps:

- Newer Claude models refuse bare model ids ("on-demand throughput
  isn't supported"): invoke the cross-region inference profile
  (`us.` / `apac.` / `global.` prefix) instead, and grant IAM on BOTH
  the inference-profile ARN and the foundation-model ARNs of every
  region it can route to — wildcard the region.
- Amazon first-party models (Titan) skip the marketplace entirely. If
  Titan works while Anthropic/Cohere fail, it is the agreement gate,
  not your policy.

Minimal invocation policy (agreement creation excluded — grant that
separately and briefly):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/anthropic.*",
        "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-*",
        "arn:aws:bedrock:*::foundation-model/cohere.embed-*",
        "arn:aws:bedrock:*:<ACCOUNT_ID>:inference-profile/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:ListFoundationModels",
        "bedrock:GetFoundationModel",
        "bedrock:GetFoundationModelAvailability",
        "bedrock:ListInferenceProfiles",
        "bedrock:GetInferenceProfile"
      ],
      "Resource": "*"
    }
  ]
}
```
