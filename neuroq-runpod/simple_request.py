#!/usr/bin/env python3
"""
RunPod Serverless Handler - シンプルなリクエストコマンド

生成テキストの品質を改善するための最適化されたパラメータを使用
"""

import requests
import json
import os
import time

# ========================================
# 環境変数から設定を取得
# ========================================
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"

if not RUNPOD_API_KEY or not ENDPOINT_ID:
    print("⚠️ 環境変数を設定してください:")
    print("   export RUNPOD_API_KEY='your_api_key'")
    print("   export RUNPOD_ENDPOINT_ID='your_endpoint_id'")
    exit(1)


# ========================================
# リクエスト送信関数
# ========================================
def send_request(input_data: dict) -> dict:
    """RunPodにリクエストを送信（非同期実行）"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    payload = {"input": input_data}
    
    try:
        response = requests.post(RUNPOD_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"リクエストエラー: {str(e)}"}


def wait_for_completion(job_id: str, timeout: int = 600, poll_interval: int = 2) -> dict:
    """
    ジョブの完了を待つ
    
    Args:
        job_id: ジョブID
        timeout: タイムアウト（秒）
        poll_interval: ポーリング間隔（秒）
    
    Returns:
        結果（outputフィールドを含む）
    """
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    status_url = f"{STATUS_URL}/{job_id}"
    start_time = time.time()
    last_status = None
    
    print(f"⏳ ジョブの完了を待っています... (ジョブID: {job_id})")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(status_url, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status", "UNKNOWN")
            
            # ステータスが変更された場合のみ表示
            if status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"   📊 ステータス: {status} (経過時間: {elapsed}秒)")
                last_status = status
            
            if status == "COMPLETED":
                print("✅ ジョブが完了しました")
                # outputフィールドを返す
                return {"output": result.get("output", {})}
            elif status == "FAILED":
                error_msg = result.get("error", "ジョブが失敗しました")
                print(f"❌ ジョブが失敗しました: {error_msg}")
                return {"error": error_msg}
            
            time.sleep(poll_interval)
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ ステータス確認エラー: {str(e)}")
            time.sleep(poll_interval)
    
    print(f"⏰ タイムアウト: {timeout}秒以内にジョブが完了しませんでした")
    return {"error": f"タイムアウト: {timeout}秒以内にジョブが完了しませんでした"}


def send_and_wait(input_data: dict, timeout: int = 600) -> dict:
    """リクエストを送信して完了を待つ"""
    # リクエスト送信
    response = send_request(input_data)
    
    # エラーチェック
    if "error" in response:
        return response
    
    # ジョブIDを取得
    job_id = response.get("id")
    if not job_id:
        return {"error": "ジョブIDが取得できませんでした"}
    
    # 完了を待つ
    return wait_for_completion(job_id, timeout=timeout)


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
    result = send_and_wait({
        "action": "health"
    }, timeout=60)
    
    if "output" in result:
        print(json.dumps(result["output"], indent=2, ensure_ascii=False))
    elif "error" in result:
        print(f"❌ エラー: {result['error']}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    
    # 例2: テキスト生成（改善されたパラメータ）
    print("2️⃣ テキスト生成（改善されたパラメータ）")
    print("-" * 70)
    result = send_and_wait({
        "action": "generate",
        "mode": "layered",
        "prompt": "ChatGPTについて教えて",
        "max_length": 80,  # 短めに設定して繰り返しを防ぐ
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 2.5  # 繰り返しを強く抑制
    }, timeout=600)
    
    if "output" in result and "generated" in result["output"]:
        print(f"プロンプト: {result['output'].get('prompt', '')}")
        print(f"\n生成テキスト:\n{result['output']['generated']}")
    elif "error" in result:
        print(f"❌ エラー: {result['error']}")
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
        
        result = send_and_wait({
            "action": "generate",
            "mode": "layered",
            "prompt": question,
            "max_length": 80,  # 短めに設定
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 2.5  # 繰り返しを強く抑制
        }, timeout=600)
        
        if "output" in result and "generated" in result["output"]:
            generated = result["output"]["generated"]
            # 長すぎる場合は最初の部分だけ表示
            if len(generated) > 200:
                print(f"生成テキスト（最初の200文字）:\n{generated[:200]}...")
            else:
                print(f"生成テキスト:\n{generated}")
        elif "error" in result:
            print(f"❌ エラー: {result['error']}")
        print()

