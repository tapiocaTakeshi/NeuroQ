#!/usr/bin/env python3
"""
RunPod Serverless Handler - クイックスタート例

シンプルなリクエスト例
"""

import requests
import json
import os

# ========================================
# 設定（環境変数から取得）
# ========================================
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

if not RUNPOD_API_KEY or not ENDPOINT_ID:
    print("⚠️ 環境変数を設定してください:")
    print("   export RUNPOD_API_KEY='your_api_key'")
    print("   export RUNPOD_ENDPOINT_ID='your_endpoint_id'")
    exit(1)


# ========================================
# シンプルなリクエスト関数
# ========================================

def send_request(input_data: dict) -> dict:
    """RunPodにリクエストを送信"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    payload = {"input": input_data}
    
    response = requests.post(RUNPOD_URL, headers=headers, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


# ========================================
# 使用例
# ========================================

if __name__ == "__main__":
    print("🧠⚛️ ニューロQ RunPod - クイックスタート\n")
    
    # 例1: ヘルスチェック
    print("1️⃣ ヘルスチェック")
    print("-" * 50)
    result = send_request({"action": "health"})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    
    # 例2: テキスト生成（既存モデルを使用）
    print("2️⃣ テキスト生成")
    print("-" * 50)
    result = send_request({
        "action": "generate",
        "mode": "layered",
        "prompt": "ChatGPTについて教えて",
        "max_length": 80,  # 短めに設定して繰り返しを防ぐ
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 2.5  # 繰り返しを強く抑制
    })
    
    if "generated" in result.get("output", {}):
        print(f"プロンプト: {result['output'].get('prompt', '')}")
        print(f"\n生成テキスト:\n{result['output']['generated']}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    
    # 例3: 学習してから生成
    print("3️⃣ 学習してから生成")
    print("-" * 50)
    print("⚠️ これは時間がかかります（数分〜数十分）")
    print()
    
    result = send_request({
        "action": "generate",
        "mode": "layered",
        "prompt": "量子コンピュータとは何ですか？",
        "max_length": 150,
        "temperature": 0.7,
        "repetition_penalty": 2.0,
        "train_before_generate": True,
        "data_sources": ["huggingface"],
        "max_records": 50,  # 少なめに設定
        "epochs": 10  # 少なめに設定
    })
    
    if "generated" in result.get("output", {}):
        print(f"プロンプト: {result['output'].get('prompt', '')}")
        print(f"\n生成テキスト:\n{result['output']['generated']}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

