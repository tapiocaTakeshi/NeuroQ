# RunPod Serverless - 非同期実行ガイド

RunPod Serverlessは非同期でジョブを実行します。リクエストを送信すると、ジョブIDが返され、そのステータスをポーリングして完了を待つ必要があります。

## 基本的な流れ

1. **リクエスト送信** → ジョブIDを取得
2. **ステータスポーリング** → 完了まで待機
3. **結果取得** → 出力を表示

## シンプルな実装例

```python
import requests
import time

RUNPOD_API_KEY = "your_api_key"
ENDPOINT_ID = "your_endpoint_id"
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"

def send_and_wait(input_data: dict, timeout: int = 600):
    """リクエストを送信して完了を待つ"""
    
    # 1. リクエスト送信
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    response = requests.post(
        RUNPOD_URL,
        headers=headers,
        json={"input": input_data},
        timeout=30
    )
    response.raise_for_status()
    job_data = response.json()
    
    job_id = job_data.get("id")
    if not job_id:
        return {"error": "ジョブIDが取得できませんでした"}
    
    # 2. ステータスポーリング
    status_headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    status_url = f"{STATUS_URL}/{job_id}"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = requests.get(status_url, headers=status_headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        status = result.get("status", "UNKNOWN")
        
        if status == "COMPLETED":
            return {"output": result.get("output", {})}
        elif status == "FAILED":
            return {"error": result.get("error", "ジョブが失敗しました")}
        
        time.sleep(2)  # 2秒待機
    
    return {"error": "タイムアウト"}
```

## ステータスの種類

- `IN_QUEUE` - キューに追加され、待機中
- `IN_PROGRESS` - 実行中
- `COMPLETED` - 完了
- `FAILED` - 失敗

## 使用例

```python
# テキスト生成
result = send_and_wait({
    "action": "generate",
    "mode": "layered",
    "prompt": "ChatGPTについて教えて",
    "max_length": 80,
    "temperature": 0.7,
    "repetition_penalty": 2.5
}, timeout=600)

if "output" in result:
    print(result["output"]["generated"])
elif "error" in result:
    print(f"エラー: {result['error']}")
```

## 実装済みファイル

- `simple_request.py` - `send_and_wait()`関数が実装済み
- `example_request.py` - 詳細な実装例

## 実行方法

```bash
python neuroq-runpod/simple_request.py
```

このスクリプトは自動的にジョブの完了を待ち、結果を表示します。

