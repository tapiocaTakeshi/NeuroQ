# RunPod デプロイメントガイド

## 修正内容

`repetition_penalty`パラメータの重複エラーを修正しました。

### 問題

`generate_text()`関数に`repetition_penalty`が複数回渡されるエラーが発生していました。

### 修正

`handler.py`の`kwargs`作成時に、`repetition_penalty`を除外リストに追加しました。

## デプロイ手順

### 1. コードを確認

修正が正しく適用されているか確認：

```bash
grep -A 10 "生成パラメータ" neuroq-runpod/handler.py
```

以下のように`repetition_penalty`が除外リストに含まれていることを確認：

```python
exclude_keys = [
    "action", "prompt", "mode", "max_length", "max_tokens",
    "temperature", "top_k", "top_p", "repetition_penalty",  # これが含まれている
    "train_before_generate", "data_sources", "common_crawl_config",
    "epochs", "batch_size", "learning_rate", "seq_length"
]
```

### 2. RunPodにデプロイ

RunPodサーバーレスエンドポイントに最新のコードをデプロイしてください。

#### 方法1: RunPod CLIを使用

```bash
# RunPod CLIでデプロイ
runpod deploy
```

#### 方法2: RunPod Web UIを使用

1. RunPodのダッシュボードにログイン
2. Serverless Endpoints セクションを開く
3. 該当するエンドポイントを選択
4. "Update Code" または "Deploy" をクリック
5. 最新の`handler.py`をアップロード

### 3. デプロイ後の確認

デプロイが完了したら、テストスクリプトを実行して確認：

```bash
python neuroq-runpod/test_generation.py
```

エラーが解消されていることを確認してください。

## トラブルシューティング

### エラーが続く場合

1. **デプロイが完了していない**
   - RunPodダッシュボードでデプロイステータスを確認
   - デプロイが完了するまで数分かかる場合があります

2. **キャッシュの問題**
   - RunPodサーバー側のキャッシュをクリア
   - エンドポイントを再起動

3. **コードが正しくアップロードされていない**
   - `handler.py`の内容を再確認
   - ファイルが正しくアップロードされているか確認

## 確認方法

デプロイ後、以下のコマンドで確認：

```bash
# ヘルスチェック
python -c "
import requests
import os
import json

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
RUNPOD_URL = f'https://api.runpod.ai/v2/{ENDPOINT_ID}/run'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {RUNPOD_API_KEY}'
}

response = requests.post(
    RUNPOD_URL,
    headers=headers,
    json={
        'input': {
            'action': 'generate',
            'mode': 'layered',
            'prompt': 'テスト',
            'max_length': 20,
            'temperature': 0.7,
            'repetition_penalty': 2.5
        }
    },
    timeout=30
)

result = response.json()
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

エラーが出ないことを確認してください。

