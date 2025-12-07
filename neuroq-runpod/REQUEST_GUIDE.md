# RunPod Serverless Handler - リクエストガイド

## クイックスタート

### 環境変数の設定

```bash
export RUNPOD_API_KEY='your_api_key_here'
export RUNPOD_ENDPOINT_ID='your_endpoint_id_here'
```

### 基本的なリクエスト

```python
import requests
import json
import os

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RUNPOD_API_KEY}"
}

# リクエスト送信
payload = {
    "input": {
        "action": "generate",
        "mode": "layered",
        "prompt": "ChatGPTについて教えて",
        "max_length": 100,
        "temperature": 0.7,
        "repetition_penalty": 2.5
    }
}

response = requests.post(RUNPOD_URL, headers=headers, json=payload)
result = response.json()

print(result["output"]["generated"])
```

## 推奨パラメータ設定

### 1. 短い回答（会話形式）

```python
{
    "action": "generate",
    "mode": "layered",
    "prompt": "質問文",
    "max_length": 80,
    "temperature": 0.6,
    "top_k": 30,
    "top_p": 0.85,
    "repetition_penalty": 2.5
}
```

### 2. 標準的な回答

```python
{
    "action": "generate",
    "mode": "layered",
    "prompt": "質問文",
    "max_length": 100,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 2.5
}
```

### 3. 長い回答（説明形式）

```python
{
    "action": "generate",
    "mode": "layered",
    "prompt": "質問文",
    "max_length": 150,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 2.0
}
```

## パラメータ説明

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `max_length` | 80-150 | 最大生成長。短いと早く終了、長いと詳細になるが繰り返しが増える可能性 |
| `temperature` | 0.5-0.9 | 温度。低いと一貫性が高く、高いと多様性が高い |
| `top_k` | 20-60 | Top-K サンプリング。低いと保守的、高いと多様 |
| `top_p` | 0.8-0.95 | Top-P (Nucleus) サンプリング。確率分布の累積確率 |
| `repetition_penalty` | 2.0-2.5 | 繰り返しペナルティ。高いほど繰り返しを抑制 |

## アクション一覧

### 1. ヘルスチェック

```python
payload = {
    "input": {
        "action": "health"
    }
}
```

### 2. モデル初期化

```python
payload = {
    "input": {
        "action": "init",
        "mode": "layered",  # または "brain"
        "embed_dim": 64,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 2
    }
}
```

### 3. テキスト生成

```python
payload = {
    "input": {
        "action": "generate",
        "mode": "layered",
        "prompt": "質問文",
        "max_length": 100,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 2.5
    }
}
```

### 4. 学習してから生成

```python
payload = {
    "input": {
        "action": "generate",
        "mode": "layered",
        "prompt": "質問文",
        "max_length": 100,
        "temperature": 0.7,
        "repetition_penalty": 2.5,
        "train_before_generate": True,
        "data_sources": ["huggingface"],
        "common_crawl_config": {
            "max_records": 100
        },
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 0.001,
        "seq_length": 64
    }
}
```

**JSONファイルでの詳細な例**: `train_request_examples.json` を参照してください。

**シンプルなJSON例**:

```json
{
  "input": {
    "action": "generate",
    "mode": "layered",
    "prompt": "ChatGPTについて教えて",
    "max_length": 80,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 2.5,
    "train_before_generate": true,
    "data_sources": ["huggingface"],
    "common_crawl_config": {
      "max_records": 100
    },
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.001,
    "seq_length": 64
  }
}
```

## 使用例ファイル

- `quick_start.py` - シンプルな使用例
- `example_request.py` - 詳細なヘルパー関数
- `request_example.py` - 改善されたパラメータの例
- `train_and_generate.py` - 学習→生成を一度に実行
- `train_request_examples.json` - 学習を含む詳細なJSONリクエスト例
- `train_request_example_simple.json` - 学習を含むシンプルなJSON例
- `train_request_examples_improved.json` - **改善された学習リクエスト例（推奨）**
- `IMPROVEMENT_ANALYSIS.md` - 生成テキスト品質の改善分析

## 実行方法

```bash
# シンプルな例
python neuroq-runpod/quick_start.py

# 改善されたパラメータの例
python neuroq-runpod/request_example.py

# 詳細な例
python neuroq-runpod/example_request.py
```

## トラブルシューティング

### 生成テキストが長すぎる

- `max_length`を減らす（例: 100 → 80）
- `repetition_penalty`を増やす（例: 2.0 → 2.5）

### 繰り返しが多い

- `repetition_penalty`を増やす（2.5以上を推奨）
- `temperature`を下げる（0.6-0.7を推奨）
- `top_k`を減らす（30-40を推奨）

### 生成が停止しない

- `max_length`を適切に設定（80-150が推奨）
- EOSトークンの検出を確認

