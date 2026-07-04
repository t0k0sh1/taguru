# AssociativeRAG クライアントプロトコル

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
   一致の許容を一時的に広げます。それでも空なら文脈選択が誤りの可能性があります —
   次候補のコンテキストへ。
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
4. `POST /contexts/{name}/unreachable_from` を文書の主要エンティティで実行し、
   到達不能の事実(取りこぼす島)がないか監査します。非空なら所属エッジの不足です。
5. 運用中にヒットしない言い回しが見つかったら `POST /contexts/{name}/aliases` で
   別綴りを登録します。エイリアスは入口専用で、結果は常に正準綴りで返ります。
   既存の2概念を後から繋ぐことはできません(それはマージであり、作り直しの領分)。

## API

| Method | Path | Body / 戻り |
|---|---|---|
| GET | `/contexts` | 目録: name, description, pinned, loaded, dice_floor, stats |
| PUT | `/contexts/{name}` | `{description?, pinned?, dice_floor?}` → 作成 |
| PATCH | `/contexts/{name}` | `{description?, pinned?, dice_floor?}` → メタ更新 |
| DELETE | `/contexts/{name}` | 削除(ファイルごと) |
| POST | `/contexts/{name}/associations` | `[{subject,label,object,weight,source?}]` → 適用数 |
| POST | `/contexts/{name}/recall` | `{cue, limit?}` → `{total, matches}` |
| POST | `/contexts/{name}/query` | `{subject?, label?, object?, limit?}` 各位置は文字列or配列 → `{total, matches}` |
| POST | `/contexts/{name}/describe` | `{concept}` → ラベル見出し(件数・役割別)/ null |
| POST | `/contexts/{name}/explore` | `{origins, max_depth?}` → `[{distance, path, association}]` |
| POST | `/contexts/{name}/activate` | `{origins, decay?=0.5, limit?=20}` → `[{strength, path, association}]` |
| POST | `/contexts/{name}/resolve` | `{cue, dice_floor?}` → `[{name, score}]` 概念名候補 |
| POST | `/contexts/{name}/resolve_label` | `{cue, dice_floor?}` → `[{name, score}]` 関係名候補 |
| GET | `/contexts/{name}/labels` | 関係語彙(正準のみ) |
| GET/POST | `/contexts/{name}/aliases` | エクスポート / `{concepts:{別綴:正準}, labels:{...}}` |
| GET/POST | `/contexts/{name}/sources` | 登録済み source 一覧 / `{passages:{source:原文}}` |
| POST | `/contexts/{name}/sources/lookup` | `{sources:[...]}` → `{passages, missing}` |
| POST | `/contexts/{name}/unreachable_from` | `{origins}` → 到達不能な連想 |

## エラーと制約

- `404` 未知のコンテキスト。`409` 重複作成・エイリアス衝突。
- `507` コンテキスト満杯(`ContextFull`)。書き込みは適用されていません。それ以上の
  知識は新しいコンテキストへ。
- recall / query の既定 limit は 100。`total` が matches 数を超えていたら切り詰めが
  起きています(強い |weight| 順で残ります)。絞り込むか limit を上げてください。
- 書き込みの永続化はフラッシュ間隔(既定5秒)以内に行われます。
