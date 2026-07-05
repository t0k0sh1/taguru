# Taguru クライアントプロトコル

あなた(LLM)がこのサーバーの唯一の想定クライアントです。ここには連想ネットワークで
知識を「取り込む」「検索する」ための規律と手順が書かれています。サーバーは構造だけを
扱います — 自然言語の理解、文脈の選択、事実への分解、文章への再構成はすべてあなたの
仕事です。

## モデル

- 知識は **(subject, label, object, weight, source)** の連想として格納されます。
  weight は符号付きで、負は「そうではない」という強い知識です(例: 「大量生産を
  行わない」→ `{"subject":"青嶺酒造","label":"行う","object":"大量生産","weight":-1.0}`)。
- 同じトリプルへの再主張は weight を加算し、source ごとの内訳(attributions)が
  残ります。2出典×1.0(独立の裏付け)と1出典×2.0(強調)は区別できます。
- **1コンテキスト=1文脈**。1つの綴りは1つの指示対象を指します。同じ綴りで別物を
  扱うとき(果物のAppleと会社のApple)はコンテキストを分けます。
- グラフは索引であり、原文の保管庫ではありません。原文は sources API に登録し、
  attribution の source id から逆引きします。

## 検索ループ

1. **文脈を選ぶ**: `GET /contexts` の目録(名前・説明・統計)から選択。説明は人が
   書いた要約、統計(連想数・次数上位概念・ラベル見本)は機械的で古びません。
2. **手がかり語を解決する**: 質問からエンティティ候補と関係候補を抜き出し、
   `resolve`(概念)/ `resolve_label`(関係)へ。入口は正規化済みです(全角半角・
   大文字小文字・カタカナ/ひらがな・軽微な誤字は吸収されます)。空か低スコアなら
   言い換えて再試行するか、`dice_floor` を下げて(既定 0.3 → 例えば 0.2)ファジー
   一致の許容を一時的に広げます。埋め込みが設定済みのサーバーでは、字面が
   外れたか弱い(最高スコア 0.5 未満の断片一致)ときに意味検索が併走し、字面候補の
   後ろに `tier: "semantic"` で併記されます(score はコサイン類似度 — 字面スコアと
   は別の尺度。tier をまたいで比較しないこと)。名前は
   グラフ文脈付きのグロスとして埋め込まれているため、専門語の言い換え
   (醸造責任者→杜氏)や質問形の cue も届きます。それでも空なら
   文脈選択が誤りの可能性があります — 次候補のコンテキストへ。
3. **見出しから絞る**: ハブ概念は `describe` で「どのラベルが何件あるか」だけを先に
   確認し、`query` に label 配列(`"label": ["住所","職歴"]`)を渡して必要な面だけを
   取得します。全プロフィールをいきなり取らないでください。
4. **広げる・ランクする**: 関連知識の収集は `activate`(起点からの活性化伝播、
   強い順、`path` に経由概念)、構造の網羅は `explore`(ホップ距離注釈付き)。
   strength は同一呼び出し内での順序値です。呼び出しをまたいで比較しないでください。
5. **原文で答える**: 結果の attributions が示す source を
   `POST /contexts/{name}/sources/lookup` で原文化し、原文の言い回しに基づいて
   回答を組み立てます。負の weight は否定として、attributions の数は裏付けの強さ
   として文章に反映してください。
6. **テキストレーンに切り替える**: 手続きの順序・条件・談話のようにトリプルに
   落ちない知識は、グラフには最初から入っていません。グラフ検索で答えの素材が
   揃わないときは `POST /contexts/{name}/sources/search`(原文への全文検索)でも
   探してください。グラフが主、テキストが安全網です。

## 取り込みループ

1. 文書を読み、事実を (subject, label, object, weight) に分解します。
   - **check before mint**: 新しい綴りを作る前に `resolve` / `resolve_label` で
     既存の綴りを探し、あれば再利用します。関係語彙は `GET /labels` で一覧できます。
   - 1文書内の言い換えは再主張しない(weight が水増しされます)。文書をまたぐ
     再主張はする(それが裏付けです)。
   - 否定文は肯定ラベル+負の weight で。
   - 暗黙の所属(「杜氏の高瀬」が誰の蔵の話か)は明示的なエッジにします。
2. `POST /contexts/{name}/associations` にバッチで書き込みます(1文書=1リクエスト、
   各要素に `source` を付けてください)。
3. 原文を `POST /contexts/{name}/sources` に登録します(source id → パッセージ)。
   粒度は段落〜規則単位に保ってください — セクション丸ごとの長大なパッセージは
   全文検索(BM25)の文書長正規化で不利になり、細部の質問が別の短い
   パッセージに負けます。
4. `POST /contexts/{name}/unreachable_from` を文書の主要エンティティで実行し、
   到達不能の事実(取りこぼす島)がないか監査します。非空なら所属エッジの不足です。
   埋め込みが設定済みなら、最後に `POST /contexts/{name}/embeddings/refresh` で
   新出の名前をベクトル化します(差分のみ、冪等)。サーバーが自動更新
   (`TAGURU_EMBED_AUTO`)で動いている場合はフラッシュ間隔内に自動反映されるため
   手動 refresh は不要です。
5. 節目ごとに `POST /contexts/{name}/vocabulary/audit` で語彙の分岐候補を点検
   します(字面の双子=綴りゆれ、意味の双子=同義フォーク)。候補であって断定では
   ありません — 同一指示対象なら片方を正準に選んで aliases で寄せ、以後の取り込みは
   正準綴りを使います(既に分岐して蓄積した分のマージはできず、作り直しの領分)。
6. 運用中にヒットしない言い回しが見つかったら `POST /contexts/{name}/aliases` で
   別綴りを登録します。エイリアスは入口専用で、結果は常に正準綴りで返ります。
   既存の2概念を後から繋ぐことはできません(それはマージであり、作り直しの領分)。
7. **文書が更新されたら差分同期**: `POST /contexts/{name}/sources/retract` で
   旧版の寄与(重みと attribution、原文)を撤回してから、新版を通常どおり
   取り込みます。概念やエッジ自体は残り、重みだけが差し引かれます。

## 手順(順序のある知識)

順序に意味がある手順は、ステップを概念ノードにして3種のエッジで編みます。
新しい仕組みは不要で、これは所属エッジや負の重みと同格の取り込み規律です。

```json
[{"subject":"日本酒の醸造","label":"最初の工程","object":"洗米","weight":1.0,"source":"工程書"},
 {"subject":"洗米","label":"次の工程","object":"浸漬","weight":1.0,"source":"工程書"},
 {"subject":"日本酒の醸造","label":"工程","object":"洗米","weight":1.0,"source":"工程書"}]
```

- **順序**は `次の工程` の連鎖で(正準ラベルを1つに固定。分岐は複数の
  `次の工程` = DAG としてそのまま表現できます)。**起点**は `最初の工程`。
  **所属**は `工程` で全ステップをハブに繋ぎます(被覆監査のため)。
- **復元**: `query {subject: 手順名, label: "最初の工程"}` で起点を取り、
  `query {label: "次の工程"}` でペアを一括取得して並べます(必要なら
  `query {subject: 現在の工程, label: "次の工程"}` で1歩ずつ)。
  **`explore` の distance を順序に使わないこと** — 所属エッジがハブ経由の
  近道を作るため、鎖上の位置と一致しません。
- **名前の衝突**: 複数の手順が同名の工程を共有する場合は修飾名にします
  (「醸造の蒸米」)。1つの綴りは1つの指示対象、はステップにも適用されます。
- 順序への裏付け・矛盾は通常どおり重みに畳まれます — 出典によって順序が
  食い違う箇所は `次の工程` エッジの低重み化として現れます。
- ステップの細部(加減・条件分岐・コツ)はトリプルに分解せず、原文を
  sources に登録して `sources/search` で引くのが正しい分担です。

## 因果関係

因果は有向エッジそのものです。原因→結果の向きで、正準ラベル
(`引き起こす` / `高める` / `防ぐ` / `要因` など。鋳造前に `resolve_label`)を
使います。

```json
[{"subject":"ストレス","label":"引き起こす","object":"不眠","weight":1.0,"source":"論文A"},
 {"subject":"カフェイン","label":"引き起こす","object":"不眠","weight":-0.8,"source":"論文C"},
 {"subject":"運動","label":"防ぐ","object":"不眠","weight":1.0,"source":"論文D"}]
```

- **検索**: 「なぜXか」= `query {label: ["引き起こす","高める","要因"], object: "X"}`。
  「Xは何をもたらすか」= subject 側を固定。因果の**連鎖**は `activate` の
  `path` から復元できます(推移が成り立つかはあなたが判断します — 系は
  A→B→C を見せるだけで、A→C を自動では主張しません)。
- **争われる因果はネット重みと attributions に現れます**: 賛否の主張が
  同じエッジに畳まれ、ネットが小さく attributions が割れているエッジは
  「係争中」です。回答にはその旨を反映してください。
- **否定と予防を混同しない**: 「引き起こさない」(因果の否認)は
  `引き起こす` への負の weight、「防ぐ」(能動的な抑制)は独立した
  正のラベルです。
- **weight に効果量を入れない**: weight は主張への証拠の量であり、
  「リスク2倍」の 2 ではありません。効果量は目的語で
  (`喫煙 →リスク倍率→ 2倍`)、または原文で表現します。
- **相関と因果はラベルで区別**: 出典が相関しか主張していないなら
  `相関する` を使い、`引き起こす` に格上げしないでください。
- 条件付き因果(「空腹時のみ」)や複合原因(「AとBが揃って初めてC」)は
  事象ノード・複合要因ノードへの具体化で表すか、原文レーンに任せます。

## API

| Method | Path | Body / 戻り |
|---|---|---|
| GET | `/contexts` | `?limit=1000&after=名前` → `{total, contexts:[{name, description, pinned, loaded, dice_floor, semantic_floor, stats}]}`(名前順のキーセットページング) |
| GET | `/contexts/{name}` | 単一コンテキストの目録行 / 404 |
| PUT | `/contexts/{name}` | `{description?, pinned?, dice_floor?, semantic_floor?}` → 作成 |
| PATCH | `/contexts/{name}` | `{description?, pinned?, dice_floor?, semantic_floor?}` → メタ更新 |
| DELETE | `/contexts/{name}` | 削除(ファイルごと) |
| POST | `/contexts/{name}/associations` | `[{subject,label,object,weight,source?}]` → 適用数 |
| POST | `/contexts/{name}/recall` | `{cue, limit?}` → `{total, matches}` |
| POST | `/contexts/{name}/query` | `{subject?, label?, object?, limit?}` 各位置は文字列or配列 → `{total, matches}` |
| POST | `/contexts/{name}/describe` | `{concept}` → ラベル見出し(件数・役割別)/ null |
| POST | `/contexts/{name}/explore` | `{origins, max_depth?, limit?}` → `{total, matches:[{distance, path, association}]}`(ホップ上限10、省略時も適用。切り詰めは近い順に残る) |
| POST | `/contexts/{name}/activate` | `{origins, decay?=0.5, limit?=20}` → `[{strength, path, association}]` |
| POST | `/contexts/{name}/resolve` | `{cue, dice_floor?, semantic_floor?}` → `[{name, score, tier}]` 概念名候補 |
| POST | `/contexts/{name}/resolve_label` | `{cue, dice_floor?, semantic_floor?}` → `[{name, score, tier}]` 関係名候補 |
| POST | `/contexts/{name}/embeddings/refresh` | 概念・ラベルのグロス埋め込みを差分更新(取り込み後に実行。文脈が変わった名前は自動再埋め込み) |
| GET | `/contexts/{name}/labels` | 関係語彙(正準のみ) |
| GET/POST | `/contexts/{name}/aliases` | エクスポート / `{concepts:{別綴:正準}, labels:{...}}` |
| GET/POST | `/contexts/{name}/sources` | 登録済み source 一覧 / `{passages:{source:原文}}` |
| POST | `/contexts/{name}/sources/lookup` | `{sources:[...]}` → `{passages, missing}` |
| POST | `/contexts/{name}/sources/search` | `{query, limit?=5}` → `[{source, score, text}]` 原文全文検索 |
| POST | `/contexts/{name}/sources/retract` | `{source}` → 出典の寄与を撤回(差分同期) |
| POST | `/contexts/{name}/unreachable_from` | `{origins, limit?}` → `{total, matches}` 到達不能な連想 |
| POST | `/contexts/{name}/vocabulary/audit` | `{dice_floor?=0.6, cosine_floor?=0.6}` → 綴り・同義の分岐候補 |

## 認証

- サーバーに `TAGURU_API_TOKEN` が設定されている場合、`/health` と `/metrics` を除く
  すべてのリクエストに `Authorization: Bearer <token>` が必要です。欠落・不一致は
  `401`(本文は下記のエラー形)で拒否されます。
- MCP ブリッジ(taguru-mcp)は自身の環境変数 `TAGURU_API_TOKEN` を読んで全リクエスト
  に自動で付与します — サーバー側で認証を有効にしたら、ブリッジにも同じ値を設定して
  ください。
- 未設定なら認証は無効です(開発モード。localhost 以外に公開してはいけません)。

## エラーと制約

- `401` 認証エラー(上記)。`404` 未知のコンテキスト。`409` 重複作成・エイリアス衝突。
- `507` コンテキスト満杯(`ContextFull`)。書き込みは適用されていません。それ以上の
  知識は新しいコンテキストへ。
- `501` 埋め込みプロバイダー未設定で `/embeddings/refresh` を呼んだ(サーバー側の
  TAGURU_EMBED_* が必要)。`502` 埋め込みプロバイダー障害(refresh、または resolve の
  意味フォールバック中)— 時間をおいて再試行してください。
- `400` associations バッチ上限超過(1リクエスト10,000件まで。何も適用されていません
  — 分割して再送)/ weight が範囲外(有限かつ |weight| ≤ 1,000,000。バッチごと拒否)/
  名前が長すぎる(subject・label・object・source・エイリアスは1024バイトまで —
  名前は見出しであって本文ではありません。原文は sources に、長い知識は分解して。
  コンテキスト名は64バイト、description は4096バイトまで)。
  `408` タイムアウト(既定30秒。クエリを絞って再試行)。`413`
  リクエストボディ超過(既定8MiB。本文はJSONではないプレーンテキストです)。
- 軸外のエラーも同じ形で返ります: 未知のパスは `404`、パスは合っているがメソッド違いは
  `405`、壊れたJSONは `400`、Content-Type 違いは `415`、形は合うが型が違うJSONは `422`。
- recall / query / explore / unreachable_from の既定 limit は 100。`total` が matches
  数を超えていたら切り詰めが起きています(recall/query/unreachable_from は強い
  |weight| 順、explore は近いホップ順で残ります)。絞り込むか limit を上げてください —
  ただしどのエンドポイントも limit の上限は 1000 です。
- 200 が返った書き込みは WAL により永続です(クラッシュしても再起動時に復元されます。
  サーバーが `TAGURU_WAL=0` で運用されている場合のみ、フラッシュ間隔(既定5秒)以内の
  書き込みがクラッシュで失われ得ます)。
