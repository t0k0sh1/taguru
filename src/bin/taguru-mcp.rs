//! taguru-mcp: an MCP (Model Context Protocol) stdio server that bridges an
//! LLM agent to a running Taguru HTTP server (`TAGURU_URL`, default
//! http://127.0.0.1:3000).
//!
//! This is the reference client the retrieval service is designed around:
//! the agent on the other side of stdio is the extractor on the write
//! path and the composer on the read path, and this bridge hands it the
//! structural tools plus the discipline. The full playbook (ingest
//! discipline, retrieval loop) is served as the MCP `instructions`,
//! fetched live from the server's /protocol (falling back to the copy
//! embedded at build time), so the agent learns the protocol the moment
//! it connects.
//!
//! Run one writer per data directory: this bridge talks to the HTTP
//! server rather than opening the data directory itself, so any number
//! of agents can share one running server.

use std::io::{BufRead, Write};
use std::time::Duration;

use serde_json::{Value, json};

const FALLBACK_PROTOCOL_VERSION: &str = "2024-11-05";
const FALLBACK_INSTRUCTIONS: &str = include_str!("../../docs/llm-protocol.md");

fn main() {
    let base = std::env::var("TAGURU_URL").unwrap_or_else(|_| "http://127.0.0.1:3000".to_string());
    let bridge = Bridge {
        base: base.trim_end_matches('/').to_string(),
        agent: ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(30))
            .build(),
    };

    let instructions = bridge
        .call("GET", "/protocol", None)
        .unwrap_or_else(|_| FALLBACK_INSTRUCTIONS.to_string());

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            eprintln!("taguru-mcp: ignoring undecodable line");
            continue;
        };
        if let Some(response) = handle(&bridge, &instructions, &message) {
            let mut out = stdout.lock();
            let _ = writeln!(out, "{response}");
            let _ = out.flush();
        }
    }
}

/// Dispatches one JSON-RPC message; notifications (no id) get no reply.
fn handle(bridge: &Bridge, instructions: &str, message: &Value) -> Option<Value> {
    let method = message.get("method").and_then(Value::as_str)?;
    let id = message.get("id")?.clone();
    if id.is_null() {
        return None;
    }
    let params = message.get("params").cloned().unwrap_or(Value::Null);

    let outcome = match method {
        "initialize" => {
            let version = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(FALLBACK_PROTOCOL_VERSION);
            Ok(json!({
                "protocolVersion": version,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "taguru", "version": env!("CARGO_PKG_VERSION") },
                "instructions": instructions,
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
            match call_tool(bridge, name, &arguments) {
                Ok(text) => Ok(json!({ "content": [{ "type": "text", "text": text }] })),
                Err(text) => Ok(json!({
                    "content": [{ "type": "text", "text": text }],
                    "isError": true,
                })),
            }
        }
        _ => Err(json!({ "code": -32601, "message": format!("unknown method '{method}'") })),
    };

    Some(match outcome {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err(error) => json!({ "jsonrpc": "2.0", "id": id, "error": error }),
    })
}

struct Bridge {
    base: String,
    agent: ureq::Agent,
}

impl Bridge {
    /// One HTTP round trip; the API's JSON error body becomes the Err
    /// text so the agent reads the server's own explanation.
    fn call(&self, method: &str, path: &str, body: Option<Value>) -> Result<String, String> {
        let request = self.agent.request(method, &format!("{}{path}", self.base));
        let response = match body {
            Some(body) => request
                .set("Content-Type", "application/json")
                .send_string(&body.to_string()),
            None => request.call(),
        };
        match response {
            Ok(response) => response
                .into_string()
                .map_err(|error| format!("response unreadable: {error}")),
            Err(ureq::Error::Status(code, response)) => {
                let detail = response.into_string().unwrap_or_default();
                Err(format!("HTTP {code}: {detail}"))
            }
            Err(error) => Err(format!("server unreachable at {}: {error}", self.base)),
        }
    }
}

/// Percent-encodes a context name for use as one URL path segment.
fn segment(name: &str) -> String {
    name.bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (byte as char).to_string()
            }
            _ => format!("%{byte:02X}"),
        })
        .collect()
}

fn object_schema(properties: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": properties, "required": required })
}

/// Pulls a required string argument or explains what is missing.
fn need<'a>(arguments: &'a Value, key: &str) -> Result<&'a str, String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing required argument '{key}'"))
}

/// Copies the listed keys into a request body, skipping absent ones.
fn pick(arguments: &Value, keys: &[&str]) -> Value {
    let mut body = serde_json::Map::new();
    for &key in keys {
        if let Some(value) = arguments.get(key)
            && !value.is_null()
        {
            body.insert(key.to_string(), value.clone());
        }
    }
    Value::Object(body)
}

fn tool_definitions() -> Vec<Value> {
    let context =
        json!({ "type": "string", "description": "コンテキスト名 (GET list_contexts の name)" });
    let tools = vec![
        (
            "list_contexts",
            "ルーティング目録: 全コンテキストの名前・説明・統計(件数、次数上位概念、ラベル見本)。どの文脈で検索・取り込みするかはこの目録を見て自分で判断する。",
            object_schema(json!({}), &[]),
        ),
        (
            "create_context",
            "コンテキストを作る。1コンテキスト=1文脈: 1つの綴りは1つの指示対象。同じ綴りで別物を扱うならコンテキストを分ける。description はルーティングの根拠になるので、何の文脈かを具体的に書く。",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "pinned": { "type": "boolean", "description": "常駐させる (用語集など常に熱い文脈)" },
                    "dice_floor": { "type": "number", "description": "ファジー入口の許容 (既定0.3)" },
                    "semantic_floor": { "type": "number", "description": "意味検索の許容 (既定0.35)" }
                }),
                &["name"],
            ),
        ),
        (
            "update_context",
            "説明・pinned・dice_floor を更新する。",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "pinned": { "type": "boolean" },
                    "dice_floor": { "type": "number" },
                    "semantic_floor": { "type": "number" }
                }),
                &["name"],
            ),
        ),
        (
            "delete_context",
            "コンテキストをファイルごと削除する (取り消し不能)。",
            object_schema(json!({ "name": { "type": "string" } }), &["name"]),
        ),
        (
            "add_associations",
            "事実をバッチで書き込む (1文書=1回)。規律: 綴りは resolve/resolve_label で既存を確認してから再利用 (check before mint)。1文書内の言い換えは再主張しない。否定は肯定ラベル+負の weight。暗黙の所属関係は明示的なエッジにする。順序のある手順は 最初の工程/次の工程/工程 の3種エッジで編む (詳細は get_protocol)。各要素に source (出典id) を付ける。",
            object_schema(
                json!({
                    "context": context,
                    "associations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": { "type": "string" },
                                "label": { "type": "string" },
                                "object": { "type": "string" },
                                "weight": { "type": "number" },
                                "source": { "type": "string" }
                            },
                            "required": ["subject", "label", "object", "weight"]
                        }
                    }
                }),
                &["context", "associations"],
            ),
        ),
        (
            "store_passages",
            "出典idの背後にある原文を登録する (source id → パッセージ)。取り込みの最後に必ず登録し、回答時は attribution の source をここから逆引きして原文に基づいて答える。",
            object_schema(
                json!({
                    "context": context,
                    "passages": { "type": "object", "additionalProperties": { "type": "string" } }
                }),
                &["context", "passages"],
            ),
        ),
        (
            "lookup_passages",
            "attribution が示す source id 群を原文に逆引きする。「グラフで見つけて原文で答える」の後半。",
            object_schema(
                json!({
                    "context": context,
                    "sources": { "type": "array", "items": { "type": "string" } }
                }),
                &["context", "sources"],
            ),
        ),
        (
            "resolve",
            "自由な言い回しを格納済みの概念名に解決する (正規化+誤字も吸収)。検索の入口: explore/activate の origins はここで得た正準名を使う。空なら言い換えるか dice_floor を下げて (例 0.2) 再試行。",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "dice_floor": { "type": "number", "description": "この1回だけのファジー許容の上書き" },
                    "semantic_floor": { "type": "number", "description": "この1回だけの意味検索許容の上書き" }
                }),
                &["context", "cue"],
            ),
        ),
        (
            "resolve_label",
            "関係ラベル名への解決。書き込み前の check-before-mint と、query のラベル選びに使う。",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "dice_floor": { "type": "number" },
                    "semantic_floor": { "type": "number" }
                }),
                &["context", "cue"],
            ),
        ),
        (
            "describe",
            "概念の見出し: どのラベルの事実が何件あるか (役割別)。ハブ概念はまずこれで全体像を掴み、必要なラベルだけ query で取る — 全プロフィールをいきなり取らない。",
            object_schema(
                json!({ "context": context, "concept": { "type": "string" } }),
                &["context", "concept"],
            ),
        ),
        (
            "query",
            "位置固定の検索。subject/label/object は文字列または配列 (配列=いずれかに一致)。describe で見出しを確認してからラベルを絞るのが定石。",
            object_schema(
                json!({
                    "context": context,
                    "subject": { "description": "文字列または配列" },
                    "label": { "description": "文字列または配列" },
                    "object": { "description": "文字列または配列" },
                    "limit": { "type": "integer" }
                }),
                &["context"],
            ),
        ),
        (
            "recall",
            "cue に触れる全連想 (主語・ラベル・目的語のどこに現れても)。役割を区別したいときは query を使う。",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "limit": { "type": "integer" }
                }),
                &["context", "cue"],
            ),
        ),
        (
            "activate",
            "起点から活性化を広げ、関連の強い順に返す (path に経由概念)。関連知識の収集はこれが主役。strength は同一呼び出し内の順序値。",
            object_schema(
                json!({
                    "context": context,
                    "origins": { "type": "array", "items": { "type": "string" } },
                    "decay": { "type": "number", "description": "既定0.5" },
                    "limit": { "type": "integer", "description": "既定20" }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "explore",
            "構造の網羅走査 (ホップ距離注釈付き)。ランキング不要で近傍を全部見たいときに。",
            object_schema(
                json!({
                    "context": context,
                    "origins": { "type": "array", "items": { "type": "string" } },
                    "max_depth": { "type": "integer" }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "list_labels",
            "関係ラベルの全語彙 (正準のみ)。抽出前に眺めて綴りの分岐を防ぐ。",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "get_aliases",
            "登録済みエイリアスのエクスポート (別綴り→正準)。",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "add_aliases",
            "別綴りを正準名に張る (入口専用; 結果は常に正準綴り)。ヒットしない言い回しを見つけたときの修復。既存の2概念を繋ぐことはできない (それはマージ=作り直しの領分)。",
            object_schema(
                json!({
                    "context": context,
                    "concepts": { "type": "object", "additionalProperties": { "type": "string" }, "description": "別綴り→正準の対応" },
                    "labels": { "type": "object", "additionalProperties": { "type": "string" } }
                }),
                &["context"],
            ),
        ),
        (
            "retract_source",
            "1つの出典 (文書) の寄与をグラフと原文ストアから撤回する。文書が更新されたときの差分同期: 旧版を retract してから新版を再取り込みする。概念やエッジ自体は残る (重みが差し引かれるだけ)。",
            object_schema(
                json!({ "context": context, "source": { "type": "string" } }),
                &["context", "source"],
            ),
        ),
        (
            "search_passages",
            "登録済み原文への全文検索 (bigram BM25)。トリプルに落ちない知識 (手続きの順序・条件・談話) のための第2レーン: グラフ検索で見つからない質問はこちらでも探す。",
            object_schema(
                json!({
                    "context": context,
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "description": "既定5" }
                }),
                &["context", "query"],
            ),
        ),
        (
            "refresh_embeddings",
            "取り込み後に概念・ラベルのグロス (名前+グラフ文脈) の埋め込みを差分更新する (埋め込み設定済みのサーバーのみ)。グラフが成長して文脈が変わった名前は自動で再埋め込みされる。これを実行しておくと、専門語の言い換えや質問形の cue が resolve の意味フォールバックで着地する。",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "audit_vocabulary",
            "語彙の健全性監査: 綴りの分岐候補 (字面: 青嶺酒蔵/青嶺酒造) と同義の分岐候補 (意味: 創業年/設立年、要 embeddings) を列挙する。候補であって断定ではない — 本当に同一指示対象なら aliases で綴りを寄せ、別物なら放置する。定期的に、また取り込みの節目に実行する。",
            object_schema(
                json!({
                    "context": context,
                    "dice_floor": { "type": "number", "description": "字面検出の下限 (既定0.6)" },
                    "cosine_floor": { "type": "number", "description": "意味検出の下限 (既定0.6)" }
                }),
                &["context"],
            ),
        ),
        (
            "audit_coverage",
            "取り込み監査: origins (文書の主要エンティティ) からどの走査でも到達できない連想を列挙する。非空なら所属エッジの不足 — 補ってから終える。",
            object_schema(
                json!({
                    "context": context,
                    "origins": { "type": "array", "items": { "type": "string" } }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "get_protocol",
            "取り込み規律と検索ループの完全な手順書 (このサーバーのマニュアル)。",
            object_schema(json!({}), &[]),
        ),
    ];

    tools
        .into_iter()
        .map(|(name, description, schema)| {
            json!({ "name": name, "description": description, "inputSchema": schema })
        })
        .collect()
}

/// Maps one tool call onto its HTTP request.
fn call_tool(bridge: &Bridge, name: &str, arguments: &Value) -> Result<String, String> {
    let context_path = |key: &str| -> Result<String, String> {
        Ok(format!("/contexts/{}", segment(need(arguments, key)?)))
    };
    match name {
        "get_protocol" => bridge.call("GET", "/protocol", None),
        "list_contexts" => bridge.call("GET", "/contexts", None),
        "create_context" => bridge.call(
            "PUT",
            &context_path("name")?,
            Some(pick(
                arguments,
                &["description", "pinned", "dice_floor", "semantic_floor"],
            )),
        ),
        "update_context" => bridge.call(
            "PATCH",
            &context_path("name")?,
            Some(pick(
                arguments,
                &["description", "pinned", "dice_floor", "semantic_floor"],
            )),
        ),
        "delete_context" => bridge.call("DELETE", &context_path("name")?, None),
        "add_associations" => bridge.call(
            "POST",
            &format!("{}/associations", context_path("context")?),
            Some(arguments.get("associations").cloned().unwrap_or(json!([]))),
        ),
        "store_passages" => bridge.call(
            "POST",
            &format!("{}/sources", context_path("context")?),
            Some(pick(arguments, &["passages"])),
        ),
        "lookup_passages" => bridge.call(
            "POST",
            &format!("{}/sources/lookup", context_path("context")?),
            Some(pick(arguments, &["sources"])),
        ),
        "resolve" => bridge.call(
            "POST",
            &format!("{}/resolve", context_path("context")?),
            Some(pick(arguments, &["cue", "dice_floor", "semantic_floor"])),
        ),
        "resolve_label" => bridge.call(
            "POST",
            &format!("{}/resolve_label", context_path("context")?),
            Some(pick(arguments, &["cue", "dice_floor", "semantic_floor"])),
        ),
        "describe" => bridge.call(
            "POST",
            &format!("{}/describe", context_path("context")?),
            Some(pick(arguments, &["concept"])),
        ),
        "query" => bridge.call(
            "POST",
            &format!("{}/query", context_path("context")?),
            Some(pick(arguments, &["subject", "label", "object", "limit"])),
        ),
        "recall" => bridge.call(
            "POST",
            &format!("{}/recall", context_path("context")?),
            Some(pick(arguments, &["cue", "limit"])),
        ),
        "activate" => bridge.call(
            "POST",
            &format!("{}/activate", context_path("context")?),
            Some(pick(arguments, &["origins", "decay", "limit"])),
        ),
        "explore" => bridge.call(
            "POST",
            &format!("{}/explore", context_path("context")?),
            Some(pick(arguments, &["origins", "max_depth"])),
        ),
        "list_labels" => bridge.call("GET", &format!("{}/labels", context_path("context")?), None),
        "get_aliases" => bridge.call(
            "GET",
            &format!("{}/aliases", context_path("context")?),
            None,
        ),
        "add_aliases" => bridge.call(
            "POST",
            &format!("{}/aliases", context_path("context")?),
            Some(pick(arguments, &["concepts", "labels"])),
        ),
        "retract_source" => bridge.call(
            "POST",
            &format!("{}/sources/retract", context_path("context")?),
            Some(pick(arguments, &["source"])),
        ),
        "search_passages" => bridge.call(
            "POST",
            &format!("{}/sources/search", context_path("context")?),
            Some(pick(arguments, &["query", "limit"])),
        ),
        "refresh_embeddings" => bridge.call(
            "POST",
            &format!("{}/embeddings/refresh", context_path("context")?),
            Some(json!({})),
        ),
        "audit_vocabulary" => bridge.call(
            "POST",
            &format!("{}/vocabulary/audit", context_path("context")?),
            Some(pick(arguments, &["dice_floor", "cosine_floor"])),
        ),
        "audit_coverage" => bridge.call(
            "POST",
            &format!("{}/unreachable_from", context_path("context")?),
            Some(pick(arguments, &["origins"])),
        ),
        _ => Err(format!("unknown tool '{name}'")),
    }
}
