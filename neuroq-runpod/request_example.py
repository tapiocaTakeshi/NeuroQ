#!/usr/bin/env python3
"""
RunPod Serverless Handler - 改善されたリクエスト例

生成テキストの品質を向上させるための最適化されたパラメータを使用
"""

import requests
import json
import os
import time

# ========================================
# 設定
# ========================================
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

if not RUNPOD_API_KEY or not ENDPOINT_ID:
    print("⚠️ 環境変数を設定してください:")
    print("   export RUNPOD_API_KEY='your_api_key'")
    print("   export RUNPOD_ENDPOINT_ID='your_endpoint_id'")
    exit(1)


def send_request(input_data: dict, timeout: int = 600) -> dict:
    """RunPodにリクエストを送信"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    payload = {"input": input_data}
    
    print(f"📤 リクエスト送信: {input_data.get('action', 'unknown')}")
    print(f"   タイムアウト: {timeout}秒")
    
    try:
        response = requests.post(
            RUNPOD_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": f"タイムアウト: {timeout}秒以内に完了しませんでした"}
    except requests.exceptions.RequestException as e:
        return {"error": f"リクエストエラー: {str(e)}"}


# ========================================
# 改善された生成パラメータの例
# ========================================

def generate_text_optimized(prompt: str, mode: str = "layered") -> dict:
    """
    最適化されたパラメータでテキスト生成
    
    改善点:
    - max_lengthを適切に制限 (100-150)
    - repetition_penaltyを強化 (2.0-2.5)
    - temperatureを適切に設定 (0.6-0.8)
    - top_kとtop_pを調整
    """
    return send_request({
        "action": "generate",
        "mode": mode,
        "prompt": prompt,
        "max_length": 100,  # 適切な長さに制限
        "temperature": 0.7,  # バランスの取れた温度
        "top_k": 40,  # Top-K サンプリング
        "top_p": 0.9,  # Top-P (Nucleus) サンプリング
        "repetition_penalty": 2.5,  # 繰り返しを強く抑制
    }, timeout=300)


def generate_with_short_response(prompt: str, mode: str = "layered") -> dict:
    """短い回答を生成（会話形式向け）"""
    return send_request({
        "action": "generate",
        "mode": mode,
        "prompt": prompt,
        "max_length": 80,  # 短めに設定
        "temperature": 0.6,  # 少し低めで一貫性を保つ
        "top_k": 30,
        "top_p": 0.85,
        "repetition_penalty": 2.5,  # 繰り返しを強く抑制
    }, timeout=300)


def generate_with_long_response(prompt: str, mode: str = "layered") -> dict:
    """長い回答を生成（説明向け）"""
    return send_request({
        "action": "generate",
        "mode": mode,
        "prompt": prompt,
        "max_length": 150,  # 長めに設定
        "temperature": 0.8,  # 少し高めで多様性を持たせる
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 2.0,  # 長文の場合は少し緩める
    }, timeout=300)


def generate_creative(prompt: str, mode: str = "layered") -> dict:
    """創造的な回答を生成"""
    return send_request({
        "action": "generate",
        "mode": mode,
        "prompt": prompt,
        "max_length": 120,
        "temperature": 0.9,  # 高めで多様性を持たせる
        "top_k": 60,
        "top_p": 0.95,
        "repetition_penalty": 2.0,
    }, timeout=300)


def generate_precise(prompt: str, mode: str = "layered") -> dict:
    """正確な回答を生成（事実ベース）"""
    return send_request({
        "action": "generate",
        "mode": mode,
        "prompt": prompt,
        "max_length": 100,
        "temperature": 0.5,  # 低めで一貫性を保つ
        "top_k": 20,
        "top_p": 0.8,
        "repetition_penalty": 2.5,
    }, timeout=300)


# ========================================
# 使用例
# ========================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠⚛️ ニューロQ RunPod - 改善されたリクエスト例")
    print("=" * 70)
    print()
    
    # 例1: ヘルスチェック
    print("1️⃣ ヘルスチェック")
    print("-" * 70)
    result = send_request({"action": "health"}, timeout=30)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    
    # 例2: 最適化されたパラメータで生成
    print("2️⃣ 最適化されたパラメータで生成")
    print("-" * 70)
    result = generate_text_optimized("ChatGPTについて教えて", mode="layered")
    
    if "output" in result:
        output = result["output"]
        if "generated" in output:
            print(f"プロンプト: {output.get('prompt', '')}")
            print(f"\n生成テキスト:\n{output['generated']}")
            print(f"\n語彙サイズ: {output.get('vocab_size', 'N/A')}")
        elif "error" in output:
            print(f"エラー: {output['error']}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    
    # 例3: 短い回答を生成
    print("3️⃣ 短い回答を生成（会話形式）")
    print("-" * 70)
    result = generate_with_short_response("量子コンピュータとは何ですか？", mode="layered")
    
    if "output" in result and "generated" in result["output"]:
        print(f"プロンプト: {result['output'].get('prompt', '')}")
        print(f"\n生成テキスト:\n{result['output']['generated']}")
    print()
    
    # 例4: 長い回答を生成
    print("4️⃣ 長い回答を生成（説明形式）")
    print("-" * 70)
    result = generate_with_long_response("ニューラルネットワークの仕組みを説明してください", mode="layered")
    
    if "output" in result and "generated" in result["output"]:
        print(f"プロンプト: {result['output'].get('prompt', '')}")
        print(f"\n生成テキスト:\n{result['output']['generated']}")
    print()
    
    # 例5: 複数の質問を試す
    print("5️⃣ 複数の質問を試す")
    print("-" * 70)
    
    questions = [
        "こんにちは",
        "あなたは誰ですか",
        "AIとは何ですか",
        "機械学習について教えて",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n質問 {i}: {question}")
        print("-" * 50)
        
        result = generate_text_optimized(question, mode="layered")
        
        if "output" in result and "generated" in result["output"]:
            generated = result["output"]["generated"]
            # 最初の200文字だけ表示（長すぎる場合）
            if len(generated) > 200:
                print(f"生成テキスト（最初の200文字）:\n{generated[:200]}...")
            else:
                print(f"生成テキスト:\n{generated}")
        
        time.sleep(1)  # リクエスト間隔を空ける
    
    print("\n" + "=" * 70)
    print("✅ 完了")
    print("=" * 70)

