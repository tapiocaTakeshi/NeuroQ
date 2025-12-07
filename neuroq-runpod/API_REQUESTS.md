# NeuroQ API リクエスト例

## 1. ヘルスチェック

```json
{
  "input": {
    "action": "health"
  }
}
```

**レスポンス例:**
```json
{
  "status": "healthy",
  "layered_available": true,
  "brain_available": true,
  "common_crawl_available": true,
  "cuda_available": true,
  "is_pretrained": false
}
```

---

## 2. テキスト生成 (Layeredモード) - 基本

```json
{
  "input": {
    "action": "generate",
    "mode": "layered",
    "prompt": "人工知能とは",
    "max_length": 100,
    "temperature": 0.8,
    "pretrain": true
  }
}
```

---

## 3. テキスト生成 (Layeredモード) - 詳細パラメータ

```json
{
  "input": {
    "action": "generate",
    "mode": "layered",
    "prompt": "量子コンピュータについて教えて",
    "max_length": 150,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "pretrain": true
  }
}
```

---

## 4. テキスト生成 (Brainモード)

```json
{
  "input": {
    "action": "generate",
    "mode": "brain",
    "prompt": "ニューラルネットワークとは",
    "max_length": 120,
    "temperature": 0.8,
    "pretrain": true
  }
}
```

---

## 5. 追加学習 - カスタムデータのみ

```json
{
  "input": {
    "action": "train",
    "mode": "layered",
    "training_data": [
      "人工知能は、人間の知能を模倣するコンピュータシステムです。",
      "機械学習は、データからパターンを学習する技術です。",
      "ディープラーニングは、多層ニューラルネットワークを使用します。"
    ],
    "epochs": 10,
    "use_common_crawl": false
  }
}
```

---

## 6. 追加学習 - Common Crawl併用

```json
{
  "input": {
    "action": "train",
    "mode": "layered",
    "training_data": [
      "APQBは調整可能擬似量子ビットの略称です。",
      "APQBは量子状態と統計的相関を統一的に記述します。"
    ],
    "epochs": 20,
    "use_common_crawl": true,
    "max_records": 100
  }
}
```

---

## 7. Common Crawlのみで学習

```json
{
  "input": {
    "action": "train",
    "mode": "brain",
    "training_data": [],
    "epochs": 15,
    "use_common_crawl": true,
    "max_records": 50
  }
}
```

---

## パラメータ一覧

### action (必須)
- `health` - ヘルスチェック
- `generate` - テキスト生成
- `train` - 追加学習

### mode
- `layered` (デフォルト) - レイヤード型モデル
- `brain` - 脳型散在ネットワークモデル

### 生成パラメータ (generateアクション)
- `prompt` (必須) - 生成の開始プロンプト
- `max_length` - 最大トークン数 (デフォルト: 100)
- `temperature` - ランダム性 0.0-2.0 (デフォルト: 0.8)
- `top_k` - 上位K個から選択 (デフォルト: 50, layeredのみ)
- `top_p` - 累積確率p (デフォルト: 0.9, layeredのみ)
- `pretrain` - 事前学習 (デフォルト: true)

### 学習パラメータ (trainアクション)
- `training_data` (必須) - 学習データの配列
- `epochs` - エポック数 (デフォルト: 10)
- `use_common_crawl` - Common Crawl使用 (デフォルト: false)
- `max_records` - Common Crawl取得数 (デフォルト: 50)

---

## cURL使用例

### ヘルスチェック
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
    "input": {
      "action": "health"
    }
  }'
```

### テキスト生成
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
    "input": {
      "action": "generate",
      "mode": "layered",
      "prompt": "人工知能とは",
      "max_length": 100,
      "temperature": 0.8,
      "pretrain": true
    }
  }'
```

### 追加学習
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
    "input": {
      "action": "train",
      "mode": "layered",
      "training_data": [
        "学習データ1",
        "学習データ2"
      ],
      "epochs": 10
    }
  }'
```

---

## Python使用例

```python
import requests

url = 'https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
}

# テキスト生成
payload = {
    'input': {
        'action': 'generate',
        'mode': 'layered',
        'prompt': '人工知能とは',
        'max_length': 100,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
        'pretrain': True
    }
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(result)
```

```python
# 追加学習
payload = {
    'input': {
        'action': 'train',
        'mode': 'layered',
        'training_data': [
            'APQBは調整可能擬似量子ビットです。',
            'ニューラルネットワークと量子論を統一します。'
        ],
        'epochs': 20,
        'use_common_crawl': True,
        'max_records': 100
    }
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(result)
```

---

## JavaScript (Node.js) 使用例

```javascript
const axios = require('axios');

const url = 'https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync';
const headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
};

// テキスト生成
const payload = {
    input: {
        action: 'generate',
        mode: 'layered',
        prompt: '人工知能とは',
        max_length: 100,
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        pretrain: true
    }
};

axios.post(url, payload, { headers })
    .then(response => {
        console.log(response.data);
    })
    .catch(error => {
        console.error(error);
    });
```
