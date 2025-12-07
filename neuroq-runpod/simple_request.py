#!/usr/bin/env python3
"""
RunPod Serverless Handler - シンプルなリクエストコマンド

生成テキストの品質を改善するための最適化されたパラメータを使用
"""

import requests
import json
import os

# ========================================
# 環境変数から設定を取得
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
# リクエスト送信関数
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
# 改善されたリクエスト例
# ========================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠⚛️ ニューロQ RunPod - シンプルなリクエスト")
    print("=" * 70)
    print()
    
    # 例1: ヘルスチェック
    print("1️⃣ ヘルスチェック")
    print("-" * 70)
    result = send_request({
        "action": "health"
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    
    # 例2: テキスト生成（改善されたパラメータ）
    print("2️⃣ テキスト生成（改善されたパラメータ）")
    print("-" * 70)
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
    
    if "output" in result and "generated" in result["output"]:
        print(f"プロンプト: {result['output'].get('prompt', '')}")
        print(f"\n生成テキスト:\n{result['output']['generated']}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    
    # 例3: 複数の質問を試す
    print("3️⃣ 複数の質問を試す")
    print("-" * 70)
    
    questions = [
        "こんにちは",
        "あなたは誰ですか",
        "AIとは何ですか"
    ]
    
    for question in questions:
        print(f"\n質問: {question}")
        print("-" * 50)
        
        result = send_request({
            "action": "generate",
            "mode": "layered",
            "prompt": question,
            "max_length": 80,  # 短めに設定
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 2.5  # 繰り返しを強く抑制
        })
        
        if "output" in result and "generated" in result["output"]:
            generated = result["output"]["generated"]
            # 長すぎる場合は最初の部分だけ表示
            if len(generated) > 200:
                print(f"生成テキスト（最初の200文字）:\n{generated[:200]}...")
            else:
                print(f"生成テキスト:\n{generated}")
        print()

