# NeuroQ API Endpoints

NeuroQ は2つのAPIサービスを提供しています:

1. **RunPod Serverless API** (`neuroq-runpod/handler.py`) — アクションベースの単一ハンドラー
2. **Modal Web API** (`neuroq-runpod/modal_app.py`) — FastAPI ベースの RESTful エンドポイント

---

## 1. RunPod Serverless API

RunPod Serverless では、すべてのリクエストが単一の `handler()` 関数で処理されます。
`action` フィールドでアクションを指定します。

**Base URL:** `https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

| Action | 説明 | 主なパラメータ |
|--------|------|---------------|
| `health` | ヘルスチェック（即座に返す） | なし |
| `status` | 詳細ステータス確認 | なし |
| `generate` | テキスト生成 | `prompt`, `model_size`, `max_length`, `temperature`, `temp_min`, `temp_max`, `session_id`, `system_prompt`, `use_translation` |
| `train` | モデル学習（チェックポイント保存付き） | `model_size`, `dataset_id`, `epochs`, `batch_size`, `lr`, `seq_length`, `texts` |
| `train_tokenizer` | BPEトークナイザー学習 | `vocab_size`, `min_frequency`, `texts`, `model_size` |
| `pretrain_openai` | OpenAIデータセット事前学習（バックグラウンド） | なし |
| `pretrain_status` | 事前学習ステータス確認 | なし |
| `clear_session` | 会話履歴クリア | `session_id`, `clear_system_prompt` |
| `set_system_prompt` | システムプロンプト設定 | `session_id`, `system_prompt` |
| `get_system_prompt` | システムプロンプト取得 | `session_id` |
| `time_limit_status` | 日次時間制限ステータス確認 | なし |
| `set_time_limit` | 日次時間制限設定 | `daily_limit_seconds` or `daily_limit_hours` |
| `get_trained_data_ids` | 蓄積された学習データのID一覧取得 | なし |
| `reset_train_data` | 蓄積された学習データの削除 | なし |

### リクエスト形式

```json
{
  "input": {
    "action": "<action_name>",
    ...パラメータ
  }
}
```

### アクション詳細

#### `health`
APIの死活確認。起動直後でも即座にレスポンスを返します。

```json
// Request
{ "input": { "action": "health" } }

// Response
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "model_initialized": false
}
```

#### `status`
モデルの初期化状態、利用可能なモデルサイズ、翻訳機能、データセット一覧などの詳細情報を返します。

```json
// Request
{ "input": { "action": "status" } }

// Response
{
  "status": "ok",
  "initialized": true,
  "device": "cuda",
  "vocab_size": 32000,
  "current_model_size": "micro",
  "available_model_sizes": ["micro", "small", "large"],
  "model_configs_available": true,
  "translation_available": true,
  "dataset_configs_available": true,
  "datasets": [...],
  "daily_time_limit": {...}
}
```

#### `generate`
テキスト生成。会話履歴管理やシステムプロンプト、翻訳パイプライン（日本語→英語→日本語）をサポートします。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `prompt` | string | `"こんにちは"` | 入力プロンプト |
| `model_size` | string | `"micro"` | `micro`, `small`, `large` |
| `max_length` | int | `50` | 最大生成トークン数 |
| `temperature` | float | `0.5` | 生成のランダム性 |
| `temp_min` | float | `0.4` | 温度の下限 |
| `temp_max` | float | `0.8` | 温度の上限 |
| `session_id` | string | `"default"` | セッションID |
| `system_prompt` | string | — | カスタムシステムプロンプト |
| `use_translation` | bool | `true` | 翻訳パイプライン使用 |

```json
// Request
{
  "input": {
    "action": "generate",
    "prompt": "人工知能とは",
    "model_size": "micro",
    "max_length": 100,
    "temperature": 0.8
  }
}

// Response
{
  "status": "success",
  "prompt": "人工知能とは",
  "generated": "...",
  "session_id": "default",
  "model_size": "micro",
  "translation_used": true,
  "processing_time_seconds": 1.23
}
```

#### `train`
モデルの学習。`dataset_id` によるデータセット指定またはカスタムテキストでの学習が可能です。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `model_size` | string | `"micro"` | `micro`, `small`, `large` |
| `dataset_id` | string | — | データセットID（`oasst1_ja`, `combined_clean` 等） |
| `epochs` | int | `10` | 学習エポック数 |
| `batch_size` | int | — | バッチサイズ |
| `lr` | float | — | 学習率 |
| `seq_length` | int | — | シーケンス長 |
| `texts` | string[] | — | カスタム学習テキスト |

利用可能な `dataset_id`:
- `oasst1_ja` — kunishou/oasst1-89k-ja 日本語会話データセット
- `oasst1_ja_cleaned` — クリーニング済み日本語会話データ
- `training_data` — 汎用トレーニングデータ
- `combined_clean` — 結合・クリーニング済みデータセット
- `high_quality` — キュレーション済み高品質会話データ
- `japanese_corpus` — 日本語トレーニングコーパス

#### `train_tokenizer`
BPEトークナイザーの学習。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `vocab_size` | int | `32000` | 語彙サイズ |
| `min_frequency` | int | `2` | 最小出現頻度 |
| `texts` | string[] | — | 学習テキスト（省略時はデフォルト） |
| `model_size` | string | `"default"` | 保存先モデルサイズ |

#### `pretrain_openai`
OpenAIデータセットによる事前学習をバックグラウンドで開始します。

#### `pretrain_status`
事前学習のステータスとログの末尾20行を返します。

#### `set_system_prompt` / `get_system_prompt`
セッションごとのシステムプロンプトを設定・取得します。

#### `clear_session`
指定セッションの会話履歴とシステムプロンプトをクリアします。

#### `time_limit_status` / `set_time_limit`
日次使用時間制限のステータス確認・設定を行います。

#### `get_trained_data_ids` / `reset_train_data`
蓄積された学習データのID一覧取得・リセットを行います。

---

## 2. Modal Web API

Modal.com 上の FastAPI ベースの REST API です。

**Base URL:** `https://<modal-app-url>.modal.run`

| メソッド | パス | 説明 |
|---------|------|------|
| `GET` | `/` | API情報・エンドポイント一覧 |
| `GET` | `/health` | ヘルスチェック |
| `GET` | `/status` | ステータス確認 |
| `POST` | `/generate` | テキスト生成 |
| `POST` | `/embeddings` | テキスト埋め込みベクトル取得 |
| `POST` | `/decode_embeddings` | 埋め込みベクトルからテキスト復元 |
| `POST` | `/train` | モデル学習（バックグラウンド実行） |
| `GET` | `/train/status/{call_id}` | 学習ジョブのステータス確認 |

### エンドポイント詳細

#### `GET /`
API情報と利用可能なエンドポイント一覧を返します。

#### `GET /health`
```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "model_initialized": false
}
```

#### `GET /status`
モデルの初期化状態や利用可能な設定の詳細を返します。

#### `POST /generate`
テキスト生成。会話履歴、Web RAG（検索結果をコンテキストに追加）をサポートします。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `prompt` | string | **必須** | 入力プロンプト |
| `max_length` | int | `50` | 最大生成トークン数 |
| `temperature` | float | `0.5` | 温度 |
| `temp_min` | float | — | 温度の下限 |
| `temp_max` | float | — | 温度の上限 |
| `history` | array | — | 会話履歴 `[{role, content}, ...]` |
| `model_size` | string | `"micro"` | `micro`, `small`, `large` |
| `system_prompt` | string | — | カスタムシステムプロンプト |
| `session_id` | string | — | セッションID |
| `use_rag` | bool | `false` | Web RAG使用 |
| `force_search` | bool | `false` | 検索を強制（`use_rag=true` 時のみ） |

```bash
curl -X POST https://<url>/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "人工知能とは", "max_length": 100}'
```

#### `POST /embeddings`
テキストをトークン化し、各トークンの埋め込みベクトルを取得します。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `text` | string | **必須** | 入力テキスト |
| `model_size` | string | `"micro"` | モデルサイズ |

#### `POST /decode_embeddings`
埋め込みベクトルから最近傍トークン検索でテキストに復元します。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `embeddings` | float[][] | **必須** | 埋め込みベクトル配列 |
| `model_size` | string | `"micro"` | モデルサイズ |
| `top_k` | int | `5` | 各ベクトルの上位K個の候補を返す |

#### `POST /train`
バックグラウンドでモデル学習を開始します。`call_id` を返すので、ステータスは `/train/status/{call_id}` で確認できます。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `model_size` | string | `"micro"` | モデルサイズ |
| `epochs` | int | `10` | エポック数 |
| `batch_size` | int | — | バッチサイズ |
| `learning_rate` | float | — | 学習率 |
| `dataset_ids` | string[] | — | データセットIDリスト（後方互換） |
| `datasets` | object[] | — | `[{id, config}]` 形式のデータセットリスト |
| `text_column` | string | `"text"` | テキストカラム名 |
| `split` | string | `"train"` | データセット分割 |
| `max_samples` | int | — | 最大サンプル数 |
| `epoch_mode` | string | `"fixed"` | `fixed` or `early_stop` |
| `hf_token` | string | — | HuggingFace アクセストークン |

#### `GET /train/status/{call_id}`
学習ジョブのステータスを確認します。

```json
// Running
{ "status": "running", "call_id": "..." }

// Completed
{ "status": "completed", "call_id": "...", "result": {...} }
```
