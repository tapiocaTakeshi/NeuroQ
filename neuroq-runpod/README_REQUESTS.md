# RunPod Serverless Handler - リクエスト例

RunPod Serverless Handlerへのリクエスト送信方法

## セットアップ

```bash
export RUNPOD_API_KEY='your_api_key_here'
export RUNPOD_ENDPOINT_ID='your_endpoint_id_here'
```

## クイックスタート

### シンプルな例

```python
import requests
import json
import os

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

# ヘッダー
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RUNPOD_API_KEY}"
}

# リクエスト
payload = {
    "input": {
        "action": "generate",
        "mode": "layered",
        "prompt": "ChatGPTについて教えて",
        "max_length": 100,
        "temperature": 0.7,
        "repetition_penalty": 2.0
    }
}

response = requests.post(RUNPOD_URL, headers=headers, json=payload)
result = response.json()

print(result["output"]["generated"])
```

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
# Layeredモード
payload = {
    "input": {
        "action": "init",
        "mode": "layered",
        "embed_dim": 64,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 2
    }
}

# Brainモード
payload = {
    "input": {
        "action": "init",
        "mode": "brain",
        "embed_dim": 128,
        "num_neurons": 100,
        "max_vocab": 16000
    }
}
```

### 3. テキスト生成

```python
# 基本的な生成
payload = {
    "input": {
        "action": "generate",
        "mode": "layered",
        "prompt": "ChatGPTについて教えて",
        "max_length": 100,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 2.0
    }
}
```

### 4. 学習してから生成

```python
payload = {
    "input": {
        "action": "generate",
        "mode": "layered",
        "prompt": "量子コンピュータとは何ですか？",
        "max_length": 150,
        "temperature": 0.7,
        "repetition_penalty": 2.0,
        "train_before_generate": True,
        "data_sources": ["huggingface"],
        "max_records": 100,
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 0.001
    }
}
```

### 5. 学習のみ

```python
payload = {
    "input": {
        "action": "train",
        "mode": "layered",
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 0.001,
        "seq_length": 64,
        "data_sources": ["huggingface"],
        "max_records": 100
    }
}
```

## パラメータ説明

### 生成パラメータ

- `prompt`: 入力テキスト（必須）
- `max_length`: 最大生成長（デフォルト: 100）
- `temperature`: 温度パラメータ（0.1-2.0、デフォルト: 0.7）
- `top_k`: Top-K サンプリング（デフォルト: 40）
- `top_p`: Top-P (Nucleus) サンプリング（0.0-1.0、デフォルト: 0.9）
- `repetition_penalty`: 繰り返しペナルティ（1.0-3.0、デフォルト: 2.0）

### 学習パラメータ

- `epochs`: エポック数（デフォルト: 20）
- `batch_size`: バッチサイズ（デフォルト: 16）
- `learning_rate`: 学習率（デフォルト: 0.001）
- `seq_length`: シーケンス長（デフォルト: 64）
- `vocab_size`: 語彙サイズ（Layeredのみ、デフォルト: 16000）

### モデルパラメータ（Layered）

- `embed_dim`: 埋め込み次元（デフォルト: 64）
- `hidden_dim`: 隠れ層次元（デフォルト: 128）
- `num_heads`: アテンションヘッド数（デフォルト: 4）
- `num_layers`: レイヤー数（デフォルト: 2）
- `lambda_entangle`: もつれ強度（デフォルト: 0.35）

### モデルパラメータ（Brain）

- `embed_dim`: 埋め込み次元（デフォルト: 128）
- `num_neurons`: ニューロン数（デフォルト: 75）
- `num_heads`: アテンションヘッド数（デフォルト: 4）
- `num_layers`: レイヤー数（デフォルト: 3）
- `max_vocab`: 最大語彙サイズ（デフォルト: 16000）

## 使用例ファイル

- `quick_start.py`: シンプルな使用例
- `example_request.py`: 詳細な使用例とヘルパー関数

## 実行方法

```bash
# シンプルな例
python quick_start.py

# 詳細な例
python example_request.py
```

