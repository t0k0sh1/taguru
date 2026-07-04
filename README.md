# AssociativeRAG

連想ネットワークによる RAG。知識を (主語, 関係ラベル, 目的語, 符号付き重み, 出典) の
連想として蓄積し、検索は埋め込み類似度ではなく**構造**で行います — 質問の見た目では
なく、質問が「何について」かを錨にして、グラフを歩いて知識を取り出します。

クライアントは LLM を想定しています。言語の理解(文書→事実への分解、文脈の選択、
検索結果→文章への再構成)はすべて LLM 側の仕事で、このサーバーは構造の格納と走査
だけを担います。クライアント向けの完全な手順書はサーバー自身が配布します:
`GET /protocol`(中身は [docs/llm-protocol.md](docs/llm-protocol.md))。

## 構成

- **ライブラリ** (`src/context.rs`) — 1つの `Context` = 1つの文脈。全状態は
  「UTF-8 文字列アリーナ + 固定長 `#[repr(C)]` レコード7テーブル」の平坦なバッファに
  収まり、隣接リストはエッジレコードに埋め込んだ侵入型チェーンです。あらゆる変更は
  追記かフィールド更新なので、`to_bytes` / `from_bytes` で全体が1つのイメージとして
  往復します(リトルエンディアン、ロード時全検証)。容量は u32 空間(テーブルあたり
  約42.9億件、文字列4GiB)で、超過は panic ではなく `ContextFull` エラーです。
  - 読み: `recall` / `query` / `query_any` / `describe` / `explore` / `activate` /
    `resolve` / `resolve_label` / `unreachable_from`
  - 書き: `associate` / `associate_from` / `add_concept_alias` / `add_label_alias`
  - 検索の入口は正規化(NFKC・大小・カナ折り畳み)+ bigram 転置索引で、表記ゆれと
    軽微な誤字を吸収します。エイリアスは入口専用の別綴りです(結果は常に正準綴り)。
- **サーバー** (`src/main.rs`, `src/registry.rs`, `src/api.rs`) — ディスクが真実、
  メモリはコンテキスト丸ごと単位のキャッシュです。各コンテキストは
  `{名前}.ctx`(イメージ)+ `{名前}.meta.json`(説明・ピン留め・統計)+
  `{名前}.sources.json`(出典原文)として保存され、起動時はコールド登録(ピン留めは
  先読み)、初アクセスで透過ロード、予算超過で LRU 退避、書き込みは dirty マーク後に
  定期フラッシュ・退避時・シャットダウン時に永続化されます。

## 起動

```sh
cargo run --release
# 環境変数:
#   ARAG_DATA_DIR     データディレクトリ (既定 ./data)
#   ARAG_CACHE_BYTES  非ピン常駐予算 (既定 512 MiB)
#   ARAG_FLUSH_SECS   フラッシュ間隔 = クラッシュ時の消失窓 (既定 5)
```

```sh
curl -X PUT localhost:3000/contexts/sake -H 'Content-Type: application/json' \
  -d '{"description":"青嶺酒造という架空の酒蔵の知識"}'
curl -X POST localhost:3000/contexts/sake/associations -H 'Content-Type: application/json' \
  -d '[{"subject":"青嶺酒造","label":"代表銘柄","object":"青嶺","weight":1.0,"source":"第1段落"}]'
curl -X POST localhost:3000/contexts/sake/activate -H 'Content-Type: application/json' \
  -d '{"origins":["青嶺酒造"]}'
```

エンドポイント一覧と取り込み・検索の規律は `GET /protocol` を参照してください。

## LLM エージェントから使う (MCP)

`arag-mcp` は稼働中の HTTP サーバーへの MCP stdio ブリッジです。エージェント
(Claude Code / Claude Desktop など)がこれを通じて取り込みと検索を行います —
文書→事実の分解と、検索結果→回答の合成はエージェント側の仕事で、規律は
ツール定義と MCP instructions(`/protocol` の内容)として自動的に渡ります。

```sh
cargo build --release                       # target/release/arag-mcp ができる
claude mcp add arag -e ARAG_URL=http://127.0.0.1:3000 -- /path/to/target/release/arag-mcp
```

これで「このフォルダの文書を sake コンテキストに取り込んで」「青嶺酒造について
知っていることを出典付きで教えて」のような依頼が、目録選択 → resolve →
describe/query/activate → 原文逆引き → 引用付き回答、のループとしてそのまま
動きます。取り込み時のチャンク分割・事実抽出・被覆監査(audit_coverage)も
エージェントがツール越しに実行します。

## 検証

```sh
cargo test                                    # ライブラリ + レジストリ + QAゴールデン
cargo test --test qa_recall -- --nocapture    # 質問ごとの再現率テーブル
cargo run --release --example benchmark       # 各操作のレイテンシ (10万/100万連想)
```

`tests/qa_recall.rs` が検索品質の回帰フロアです: 架空コーパスへの11問(誤字入口・
全角入口・エイリアス・2ホップ合成・否定・裏付けを含む)が、文書化された検索ループを
機械的に回して全問完答であり続けることを検査します。
