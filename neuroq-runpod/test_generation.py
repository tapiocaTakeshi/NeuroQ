#!/usr/bin/env python3
"""
RunPod Serverless Handler - 生成テスト用スクリプト

様々なパラメータで生成をテストし、最適な設定を探す
"""

import requests
import json
import os
import time

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"

if not RUNPOD_API_KEY or not ENDPOINT_ID:
    print("⚠️ 環境変数を設定してください")
    exit(1)


def send_and_wait(input_data: dict, timeout: int = 600) -> dict:
    """リクエストを送信して完了を待つ"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    # リクエスト送信
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
    
    # ステータスポーリング
    status_headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    status_url = f"{STATUS_URL}/{job_id}"
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < timeout:
        response = requests.get(status_url, headers=status_headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        status = result.get("status", "UNKNOWN")
        
        if status != last_status:
            elapsed = int(time.time() - start_time)
            print(f"   📊 {status} ({elapsed}秒)")
            last_status = status
        
        if status == "COMPLETED":
            return {"output": result.get("output", {})}
        elif status == "FAILED":
            return {"error": result.get("error", "ジョブが失敗しました")}
        
        time.sleep(2)
    
    return {"error": "タイムアウト"}


# ========================================
# テストケース
# ========================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 ニューロQ RunPod - 生成パラメータテスト")
    print("=" * 70)
    print()
    
    prompt = "ChatGPTについて教えて"
    
    # テストケース1: 短い回答（温度低め）
    print("📝 テスト1: 短い回答（温度低め、保守的）")
    print("-" * 70)
    print(f"プロンプト: {prompt}")
    print()
    
    result = send_and_wait({
        "action": "generate",
        "mode": "layered",
        "prompt": prompt,
        "max_length": 50,  # 短め
        "temperature": 0.5,  # 低めで一貫性を保つ
        "top_k": 20,
        "top_p": 0.8,
        "repetition_penalty": 2.5
    }, timeout=600)
    
    if "output" in result and "generated" in result["output"]:
        print(f"生成テキスト:\n{result['output']['generated']}")
    elif "error" in result:
        print(f"❌ エラー: {result['error']}")
    print("\n")
    
    # テストケース2: 標準的な回答
    print("📝 テスト2: 標準的な回答")
    print("-" * 70)
    print(f"プロンプト: {prompt}")
    print()
    
    result = send_and_wait({
        "action": "generate",
        "mode": "layered",
        "prompt": prompt,
        "max_length": 80,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 2.5
    }, timeout=600)
    
    if "output" in result and "generated" in result["output"]:
        print(f"生成テキスト:\n{result['output']['generated']}")
    elif "error" in result:
        print(f"❌ エラー: {result['error']}")
    print("\n")
    
    # テストケース3: より具体的なプロンプト
    print("📝 テスト3: より具体的なプロンプト")
    print("-" * 70)
    specific_prompt = "ChatGPTとは何ですか？どのような特徴がありますか？"
    print(f"プロンプト: {specific_prompt}")
    print()
    
    result = send_and_wait({
        "action": "generate",
        "mode": "layered",
        "prompt": specific_prompt,
        "max_length": 80,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 2.5
    }, timeout=600)
    
    if "output" in result and "generated" in result["output"]:
        print(f"生成テキスト:\n{result['output']['generated']}")
    elif "error" in result:
        print(f"❌ エラー: {result['error']}")
    print("\n")
    
    # テストケース4: シンプルな質問
    print("📝 テスト4: シンプルな質問")
    print("-" * 70)
    simple_prompt = "こんにちは"
    print(f"プロンプト: {simple_prompt}")
    print()
    
    result = send_and_wait({
        "action": "generate",
        "mode": "layered",
        "prompt": simple_prompt,
        "max_length": 50,
        "temperature": 0.6,
        "top_k": 30,
        "top_p": 0.85,
        "repetition_penalty": 2.5
    }, timeout=600)
    
    if "output" in result and "generated" in result["output"]:
        print(f"生成テキスト:\n{result['output']['generated']}")
    elif "error" in result:
        print(f"❌ エラー: {result['error']}")
    print("\n")
    
    print("=" * 70)
    print("✅ テスト完了")
    print("=" * 70)

