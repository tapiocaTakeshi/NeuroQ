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

## curlでのテスト

```bash
BASE_URL="https://api.neuroq.he-ro.jp"

# ========================================
# 基本
# ========================================

# ヘルスチェック
curl $BASE_URL/health

# ステータス確認
curl $BASE_URL/status

# ========================================
# テキスト生成 (POST /generate)
# ========================================

# シンプルな生成
curl -X POST $BASE_URL/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "こんにちは", "max_length": 50}'

# モデルサイズ・温度を指定
curl -X POST $BASE_URL/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is quantum computing?",
    "max_length": 100,
    "temperature": 0.7,
    "model_size": "small"
  }'

# 会話履歴付き（マルチターン）
curl -X POST $BASE_URL/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "それについてもう少し教えて",
    "max_length": 80,
    "temperature": 0.5,
    "session_id": "user123",
    "history": [
      {"role": "user", "content": "量子コンピュータって何？"},
      {"role": "assistant", "content": "量子ビットを使った計算機です。"}
    ]
  }'

# カスタムシステムプロンプト
curl -X POST $BASE_URL/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "自己紹介して",
    "max_length": 100,
    "session_id": "user123",
    "system_prompt": "あなたは猫です。語尾に「にゃ」を付けてください。"
  }'

# Web RAG（検索結果をコンテキストに追加）
curl -X POST $BASE_URL/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "最新のAIニュースを教えて",
    "max_length": 100,
    "use_rag": true,
    "force_search": true
  }'

# ========================================
# 埋め込みベクトル (POST /embeddings)
# ========================================

# テキストの埋め込みベクトルを取得
curl -X POST $BASE_URL/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは世界",
    "model_size": "micro"
  }'

# ========================================
# 埋め込みデコード (POST /decode_embeddings)
# ========================================

# 埋め込みベクトルからテキストに復元（top-5候補）
# ※ embeddings の値は /embeddings で取得した embedding_full を使用
curl -X POST $BASE_URL/decode_embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [[0.1, 0.2, 0.3, ...]],
    "model_size": "micro",
    "top_k": 5
  }'

# ========================================
# 学習 (POST /train)
# ========================================

# デフォルト設定で学習開始（バックグラウンド実行）
curl -X POST $BASE_URL/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_size": "micro",
    "epochs": 10
  }'

# 詳細設定で学習
curl -X POST $BASE_URL/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_size": "small",
    "epochs": 5,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "dataset_ids": ["OpenAssistant/oasst1"],
    "text_column": "text",
    "split": "train",
    "max_samples": 10000
  }'

# 学習ステータス確認（call_id は /train のレスポンスに含まれる）
curl $BASE_URL/train/status/{call_id}
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
