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

| パラメータ    | 型     | デフォルト | 説明                              |
| ------------- | ------ | ---------- | --------------------------------- |
| `prompt`      | string | 必須       | 入力プロンプト                    |
| `max_length`  | int    | 50         | 最大生成長                        |
| `temperature` | float  | 0.5        | 生成の多様性（0.0-1.0）           |
| `temp_min`    | float  | null       | 最低温度（指定時は範囲を使用）    |
| `temp_max`    | float  | null       | 最高温度                          |
| `session_id`  | string | "default"  | 会話セッションID                  |
| `model_size`  | string | "micro"    | モデルサイズ: micro, small, large |

## curlでのテスト

```bash
# ヘルスチェック
curl https://higuchiyuya-riddle--neuroq-api-fastapi-app.modal.run/health

# ステータス確認
curl https://higuchiyuya-riddle--neuroq-api-fastapi-app.modal.run/status

# テキスト生成
curl -X POST https://higuchiyuya-riddle--neuroq-api-fastapi-app.modal.run/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "こんにちは", "max_length": 50}'
```

## Pythonクライアント例

```python
import requests

BASE_URL = "https://higuchiyuya-riddle--neuroq-api-fastapi-app.modal.run"

# テキスト生成
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
