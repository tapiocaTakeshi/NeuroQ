# NeuroQ Modal API デプロイガイド

## 概要

[Modal.com](https://modal.com) を使用してNeuroQ言語モデルをクラウドGPUでホストするAPIサーバーです。

### 特徴

- ⚡ **高速起動**: コールドスタートでも数秒でGPU推論が開始
- 🎯 **自動スケーリング**: リクエストに応じて自動的にスケール
- 💰 **従量課金**: 使用した分だけ課金（アイドル時は無料）
- 🔒 **HTTPS対応**: 自動的にHTTPS URLが発行される

## セットアップ

### 1. Modalアカウント作成

[modal.com](https://modal.com) でアカウントを作成してください。

### 2. Modal CLIインストール

```bash
pip install modal
```

### 3. 認証

```bash
modal setup
```

ブラウザが開き、認証を求められます。認証後、ターミナルに戻ります。

### 4. ローカルテスト

```bash
# 開発モード（ホットリロード）
modal serve modal_app.py
```

これにより、ローカルでテスト可能なURLが発行されます。

### 5. デプロイ

```bash
modal deploy modal_app.py
```

デプロイ後、永続的なURLが発行されます：

```
https://higuchiyuya-riddle--neuroq-api-fastapi-app.modal.run
```

## API エンドポイント

### ルート情報

```
GET /
```

### ヘルスチェック

```
GET /health
```

レスポンス:

```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "model_initialized": true
}
```

### ステータス確認

```
GET /status
```

レスポンス:

```json
{
  "status": "ok",
  "initialized": true,
  "device": "cuda",
  "vocab_size": 8000,
  "current_model_size": "micro",
  "available_model_sizes": ["micro", "small", "large"]
}
```

### テキスト生成

```
POST /generate
Content-Type: application/json

{
    "prompt": "Hello, how are you?",
    "max_length": 50,
    "temperature": 0.5,
    "session_id": "user123",
    "model_size": "micro"
}
```

レスポンス:

```json
{
  "status": "success",
  "prompt": "Hello, how are you?",
  "generated": "I'm doing well, thank you!",
  "session_id": "user123",
  "model_size": "micro"
}
```

### パラメータ

| パラメータ      | 型       | デフォルト | 説明                                                         |
| --------------- | -------- | ---------- | ------------------------------------------------------------ |
| `prompt`        | string   | 必須       | 入力プロンプト                                               |
| `max_length`    | int      | 50         | 最大生成トークン数                                           |
| `temperature`   | float    | 0.5        | 生成の多様性（0.0-1.0）                                      |
| `temp_min`      | float    | null       | 最低温度（指定時は範囲を使用）                               |
| `temp_max`      | float    | null       | 最高温度                                                     |
| `history`       | array    | null       | 会話履歴 `[{"role":"user","content":"..."},...]`              |
| `model_size`    | string   | "micro"    | モデルサイズ: micro, small, large                            |
| `system_prompt` | string   | null       | カスタムシステムプロンプト（省略時はデフォルト）              |
| `session_id`    | string   | null       | セッションID（システムプロンプトの永続化用）                  |
| `use_rag`       | bool     | false      | Web RAGを使用（検索結果をコンテキストに追加）                 |
| `force_search`  | bool     | false      | 検索を強制（`use_rag=true` の場合のみ有効）                  |

## フルリクエスト例（全パラメータ指定）

### POST /generate（全パラメータ）

```bash
curl -X POST https://api.neuroq.he-ro.jp/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "量子コンピュータの仕組みを教えて",
    "max_length": 100,
    "temperature": 0.7,
    "temp_min": 0.5,
    "temp_max": 0.9,
    "model_size": "small",
    "session_id": "session_abc123",
    "system_prompt": "あなたは量子物理学の教授です。専門用語を避け、中学生にもわかるように説明してください。",
    "history": [
      {"role": "user", "content": "量子ビットって何？"},
      {"role": "assistant", "content": "0と1を同時に持てる特殊なビットです。"},
      {"role": "user", "content": "普通のビットとどう違うの？"},
      {"role": "assistant", "content": "普通のビットは0か1のどちらかですが、量子ビットは両方の状態を重ね合わせられます。"}
    ],
    "use_rag": true,
    "force_search": true
  }'
```

レスポンス:

```json
{
  "status": "success",
  "prompt": "量子コンピュータの仕組みを教えて",
  "generated": "量子コンピュータは...",
  "model_size": "small",
  "system_prompt": "あなたは量子物理学の教授です。...",
  "session_id": "session_abc123",
  "rag_used": true,
  "rag_sources": ["https://example.com/quantum"],
  "rag_query": "量子コンピュータ 仕組み"
}
```

### POST /embeddings（全パラメータ）

```bash
curl -X POST https://api.neuroq.he-ro.jp/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは世界",
    "model_size": "small"
  }'
```

レスポンス:

```json
{
  "status": "success",
  "text": "こんにちは世界",
  "num_tokens": 3,
  "embed_dim": 256,
  "tokens": [
    {
      "index": 0,
      "token": "こんにちは",
      "token_id": 12345,
      "embedding_preview": [0.01, 0.02, 0.03, 0.04, 0.05, "...", -0.01, -0.02, -0.03, -0.04, -0.05],
      "embedding_full": [0.01, 0.02, "...（256次元）"]
    }
  ]
}
```

### POST /decode_embeddings（全パラメータ）

```bash
# ※ embedding_full の値は /embeddings のレスポンスからコピーして使用
curl -X POST https://api.neuroq.he-ro.jp/decode_embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [
      [0.01, 0.02, 0.03, 0.04, 0.05],
      [-0.01, 0.03, -0.02, 0.01, 0.04]
    ],
    "model_size": "small",
    "top_k": 10
  }'
```

レスポンス:

```json
{
  "status": "success",
  "num_input_vectors": 2,
  "embed_dim": 5,
  "decoded_tokens": [
    {
      "index": 0,
      "best_token": "こんにちは",
      "best_token_id": 12345,
      "similarity": 0.9821,
      "candidates": [
        {"rank": 1, "token": "こんにちは", "token_id": 12345, "similarity": 0.9821},
        {"rank": 2, "token": "おはよう", "token_id": 12346, "similarity": 0.8734}
      ]
    }
  ],
  "reconstructed_text": "こんにちは世界"
}
```

### POST /train（全パラメータ）

```bash
curl -X POST https://api.neuroq.he-ro.jp/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_size": "small",
    "epochs": 20,
    "batch_size": 4,
    "learning_rate": 0.00005,
    "dataset_ids": ["OpenAssistant/oasst1", "OpenAssistant/oasst2"],
    "text_column": "text",
    "split": "train",
    "max_samples": 50000
  }'
```

レスポンス:

```json
{
  "status": "started",
  "message": "Training SMALL model started",
  "model_size": "small",
  "epochs": 20,
  "batch_size": 4,
  "learning_rate": 0.00005,
  "dataset_ids": ["OpenAssistant/oasst1", "OpenAssistant/oasst2"],
  "text_column": "text",
  "split": "train",
  "max_samples": 50000,
  "call_id": "fc-xxxxxxxx"
}
```

### GET /train/status/{call_id}

```bash
curl https://api.neuroq.he-ro.jp/train/status/fc-xxxxxxxx
```

レスポンス（実行中）:

```json
{"status": "running", "message": "Training is still in progress"}
```

レスポンス（完了）:

```json
{"status": "completed", "result": {"status": "success", "best_val_loss": 3.1234}}
```

## curlでのテスト（簡易版）

```bash
BASE_URL="https://api.neuroq.he-ro.jp"

# ヘルスチェック
curl $BASE_URL/health

# ステータス確認
curl $BASE_URL/status

# シンプルな生成
curl -X POST $BASE_URL/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "こんにちは", "max_length": 50}'

# 埋め込みベクトル取得
curl -X POST $BASE_URL/embeddings \
  -H "Content-Type: application/json" \
  -d '{"text": "テスト", "model_size": "micro"}'

# 学習開始
curl -X POST $BASE_URL/train \
  -H "Content-Type: application/json" \
  -d '{"model_size": "micro", "epochs": 10}'
```

## Pythonクライアント例

```python
import requests

BASE_URL = "https://api.neuroq.he-ro.jp"

# --- テキスト生成 ---
response = requests.post(
    f"{BASE_URL}/generate",
    json={
        "prompt": "What is quantum computing?",
        "max_length": 100,
        "temperature": 0.7,
        "model_size": "micro"
    }
)
result = response.json()
print(f"Generated: {result['generated']}")

# --- 会話履歴付き生成 ---
response = requests.post(
    f"{BASE_URL}/generate",
    json={
        "prompt": "続きを教えて",
        "max_length": 80,
        "session_id": "session_001",
        "history": [
            {"role": "user", "content": "Pythonの特徴は？"},
            {"role": "assistant", "content": "読みやすい文法が特徴です。"},
        ]
    }
)
print(response.json()["generated"])

# --- Web RAG付き生成 ---
response = requests.post(
    f"{BASE_URL}/generate",
    json={
        "prompt": "東京の天気は？",
        "use_rag": True,
        "force_search": True
    }
)
result = response.json()
print(f"Generated: {result['generated']}")
print(f"Sources: {result.get('rag_sources')}")

# --- 埋め込みベクトル取得 ---
response = requests.post(
    f"{BASE_URL}/embeddings",
    json={"text": "こんにちは世界", "model_size": "micro"}
)
tokens = response.json()["tokens"]
for t in tokens:
    print(f"  {t['token']} (id={t['token_id']}): {t['embedding_preview']}")

# --- 学習開始 & ステータス確認 ---
response = requests.post(
    f"{BASE_URL}/train",
    json={"model_size": "micro", "epochs": 5}
)
call_id = response.json()["call_id"]
print(f"Training started: {call_id}")

# ステータス確認
status = requests.get(f"{BASE_URL}/train/status/{call_id}").json()
print(f"Status: {status['status']}")
```

## GPU設定

`modal_app.py` 内の `gpu` パラメータを変更することで、使用するGPUを選択できます：

```python
@app.cls(
    gpu="T4",  # 変更オプション: "T4", "A10G", "A100", "H100"
    ...
)
```

### GPU料金目安（2024年現在）

| GPU       | メモリ | 料金/時間 |
| --------- | ------ | --------- |
| T4        | 16GB   | $0.59     |
| A10G      | 24GB   | $1.10     |
| A100-40GB | 40GB   | $3.00     |
| A100-80GB | 80GB   | $4.73     |
| H100      | 80GB   | $6.00     |

## トラブルシューティング

### チェックポイントが見つからない

```
⚠️ チェックポイントが見つかりません: checkpoints/...
```

→ ローカルの `checkpoints/` ディレクトリにモデルファイルがあることを確認してください。

### CUDAが利用できない

```
Device: cpu
```

→ `gpu` パラメータが正しく設定されているか確認してください。

### モジュールインポートエラー

```
❌ neuroquantum_layered.py インポートエラー
```

→ 必要なPythonファイルがすべて含まれているか確認してください。

## ファイル構成

```
neuroq-runpod/
├── modal_app.py           # Modalアプリケーション
├── neuroquantum_layered.py # NeuroQuantumモデル
├── neuroquantum_brain.py   # NeuroQuantumBrainAI
├── model_configs.py        # モデル設定
├── qbnn_layered.py         # QBNN層
├── tiktoken_tokenizer.py   # TikTokenトークナイザー
├── neuroq_tokenizer.model  # SentencePieceモデル
├── neuroq_tokenizer.vocab  # 語彙ファイル
└── checkpoints/
    ├── neuroq_tiktoken_english_checkpoint.pt  # Microモデル
    ├── neuroq_small_checkpoint.pt             # Smallモデル
    └── neuroq_large_checkpoint.pt             # Largeモデル
```

## コスト最適化

1. **container_idle_timeout**: アイドル状態のコンテナが自動停止されるまでの時間
2. **allow_concurrent_inputs**: 1つのコンテナで処理できる同時リクエスト数

```python
@app.cls(
    container_idle_timeout=120,  # 2分後に停止
    allow_concurrent_inputs=10,  # 同時10リクエスト
)
```

## ログ確認

```bash
# Modal CLIでログを確認
modal logs neuroq-api
```

## 関連リンク

- [Modal Documentation](https://modal.com/docs/guide)
- [Modal Pricing](https://modal.com/pricing)
- [NeuroQ GitHub](https://github.com/your-repo/neuroq)
