#!/usr/bin/env python3
"""
RunPod Serverless Handler - 学習してから生成するスクリプト

train_before_generateフラグを使用して学習→生成を実行
"""

import requests
import json
import os
import time
from typing import Optional, Dict, Any

# ========================================
# 環境変数から設定を取得
# ========================================
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not RUNPOD_API_KEY or not ENDPOINT_ID:
    print("=" * 60)
    print("⚠️ 環境変数を設定してください")
    print("=" * 60)
    exit(1)

RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"


def send_and_wait(input_data: dict, timeout: int = 3600) -> dict:
    """リクエストを送信して完了を待つ"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    # リクエスト送信
    print("📤 RunPodにリクエストを送信中...")
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
    
    print(f"✅ ジョブID: {job_id}")
    print("⏳ 学習と生成の完了を待っています...")
    
    # ステータスポーリング
    status_headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    status_url = f"{STATUS_URL}/{job_id}"
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < timeout:
        try:
            status_response = requests.get(status_url, headers=status_headers, timeout=30)
            status_response.raise_for_status()
            result = status_response.json()
            
            current_status = result.get("status", "UNKNOWN")
            
            # ステータスが変わったら表示
            if current_status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"   📊 {current_status} ({elapsed}秒)")
                last_status = current_status
            
            if current_status == "COMPLETED":
                output = result.get("output", {})
                if "error" in output:
                    return {"error": output["error"]}
                return {"output": output, "status": "completed"}
            
            elif current_status == "FAILED":
                output = result.get("output", {})
                error_msg = output.get("error", "不明なエラー") if isinstance(output, dict) else str(output)
                print(f"\n   ❌ エラー詳細:")
                print(f"      エラーメッセージ: {error_msg}")
                print(f"\n   完全なレスポンス:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return {"error": f"失敗: {error_msg}", "status": "failed"}
            
            # ポーリング間隔
            time.sleep(3)
            
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ ステータス取得エラー: {e}")
            time.sleep(5)
    
    return {"error": f"タイムアウト: {timeout}秒以内に完了しませんでした", "status": "timeout"}


def train_and_generate(
    prompt: str,
    mode: str = "layered",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seq_length: int = 64,
    max_records: int = 100,
    data_sources: list = ["huggingface"],
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 2.0,
    **model_kwargs
) -> Dict[str, Any]:
    """
    学習してからテキスト生成（train_before_generateフラグを使用）
    """
    payload = {
        "action": "generate",
        "mode": mode,
        "prompt": prompt,
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "train_before_generate": True,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seq_length": seq_length,
        "data_sources": data_sources,
        "common_crawl_config": {"max_records": max_records},
        **model_kwargs
    }
    
    timeout = 3600  # 1時間
    return send_and_wait(payload, timeout=timeout)


def main():
    """メイン関数"""
    print("=" * 60)
    print("🧠⚛️ ニューロQ RunPod - 学習してから生成")
    print("=" * 60)
    print()
    
    # 学習パラメータ（改善版 - 特殊トークンの生成を抑制）
    mode = "layered"
    epochs = 30  # 30エポックに増加（より良い学習のため）
    batch_size = 16
    learning_rate = 0.0005  # より細かい学習率
    max_records = 300  # より多くのデータで学習（100→300）
    
    # 生成パラメータ（改善版 - 特殊トークンを抑制）
    prompt = "ChatGPTについて教えて"
    max_length = 80
    temperature = 0.6  # 0.7→0.6に下げて一貫性を高める
    top_k = 30  # 40→30に下げて特殊トークンを抑制
    top_p = 0.85  # 0.9→0.85に下げて多様性を抑える
    repetition_penalty = 3.0  # 2.5→3.0に上げて繰り返しと特殊トークンを強く抑制
    
    print(f"📋 学習設定:")
    print(f"   モード: {mode}")
    print(f"   エポック数: {epochs}")
    print(f"   バッチサイズ: {batch_size}")
    print(f"   学習率: {learning_rate}")
    print(f"   最大データレコード数: {max_records}")
    print()
    print(f"📝 生成設定:")
    print(f"   プロンプト: {prompt}")
    print(f"   最大長: {max_length}")
    print(f"   温度: {temperature}")
    print(f"   Top-K: {top_k}")
    print(f"   Top-P: {top_p}")
    print(f"   繰り返しペナルティ: {repetition_penalty}")
    print()
    
    # 学習→生成実行
    print("🚀 学習と生成を開始します...")
    print("   ⏳ この処理には数分かかる場合があります...")
    print()
    
    result = train_and_generate(
        prompt=prompt,
        mode=mode,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_records=max_records,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    
    print()
    print("=" * 60)
    
    if "error" in result:
        print(f"❌ エラー: {result['error']}")
    elif "output" in result:
        output = result["output"]
        
        # 生成結果を表示
        if "generated" in output:
            print(f"✅ 生成完了！")
            print()
            print(f"プロンプト: {output.get('prompt', prompt)}")
            print()
            print(f"生成テキスト:")
            print(f"{output['generated']}")
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("=" * 60)


if __name__ == "__main__":
    main()

