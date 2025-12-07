#!/usr/bin/env python3
"""
RunPod Serverless Handler - モデル学習スクリプト

モデルを学習させるためのスクリプト
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
    print()
    print("以下のコマンドで環境変数を設定してください：")
    print()
    if not RUNPOD_API_KEY:
        print("  export RUNPOD_API_KEY='your_api_key_here'")
    if not ENDPOINT_ID:
        print("  export RUNPOD_ENDPOINT_ID='your_endpoint_id_here'")
    print()
    print("=" * 60)
    exit(1)

RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"


# ========================================
# リクエスト送信と完了待機
# ========================================

def send_request(input_data: dict, timeout: int = 3600) -> dict:
    """RunPodにリクエストを送信して完了を待つ"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    # リクエスト送信
    print("📤 RunPodに学習リクエストを送信中...")
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
    print("⏳ 学習の完了を待っています...")
    
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
                error_msg = "不明なエラー"
                
                # エラーの詳細を表示
                print(f"\n   ❌ エラー詳細:")
                if isinstance(output, dict):
                    error_msg = output.get("error", "不明なエラー")
                    print(f"      エラーメッセージ: {error_msg}")
                    
                    # 全ての出力を表示（デバッグ用）
                    print(f"      出力全体: {json.dumps(output, indent=2, ensure_ascii=False)}")
                else:
                    error_msg = str(output)
                    print(f"      エラー出力: {error_msg}")
                
                # 完全な結果を表示（デバッグ用）
                print(f"\n   完全なレスポンス:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                return {
                    "error": f"学習失敗: {error_msg}",
                    "status": "failed",
                    "full_result": result
                }
            
            # ポーリング間隔
            time.sleep(3)
            
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ ステータス取得エラー: {e}")
            time.sleep(5)
    
    return {"error": f"タイムアウト: {timeout}秒以内に完了しませんでした", "status": "timeout"}


# ========================================
# 学習関数
# ========================================

def train_model(
    mode: str = "layered",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seq_length: int = 64,
    max_records: int = 100,
    data_sources: list = ["huggingface"],
    common_crawl_config: Optional[Dict] = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    モデルを学習させる
    
    Args:
        mode: "layered" または "brain"
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        seq_length: シーケンス長
        max_records: 最大データレコード数
        data_sources: データソースのリスト
        common_crawl_config: Common Crawl設定
        **model_kwargs: モデル設定パラメータ
    
    Returns:
        学習結果
    """
    payload = {
        "action": "train",
        "mode": mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seq_length": seq_length,
        "data_sources": data_sources,
        "common_crawl_config": common_crawl_config or {"max_records": max_records},
        **model_kwargs
    }
    
    timeout = 3600  # 1時間
    return send_request(payload, timeout=timeout)


def train_and_generate(
    prompt: str,
    mode: str = "layered",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seq_length: int = 64,
    max_records: int = 100,
    data_sources: list = ["huggingface"],
    common_crawl_config: Optional[Dict] = None,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 2.0,
    **model_kwargs
) -> Dict[str, Any]:
    """
    学習してからテキスト生成
    
    Args:
        prompt: 生成プロンプト
        mode: "layered" または "brain"
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        seq_length: シーケンス長
        max_records: 最大データレコード数
        data_sources: データソースのリスト
        common_crawl_config: Common Crawl設定
        max_length: 最大生成長
        temperature: 温度パラメータ
        top_k: Top-K サンプリング
        top_p: Top-P サンプリング
        repetition_penalty: 繰り返しペナルティ
        **model_kwargs: モデル設定パラメータ
    
    Returns:
        生成結果
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
        "common_crawl_config": common_crawl_config or {"max_records": max_records},
        **model_kwargs
    }
    
    timeout = 3600  # 1時間
    return send_request(payload, timeout=timeout)


# ========================================
# メイン関数
# ========================================

def main():
    """メイン関数"""
    print("=" * 60)
    print("🧠⚛️ ニューロQ RunPod - モデル学習")
    print("=" * 60)
    print()
    
    # 学習パラメータ
    mode = "layered"  # "layered" または "brain"
    epochs = 20
    batch_size = 16
    learning_rate = 0.001
    max_records = 100
    
    print(f"📋 学習設定:")
    print(f"   モード: {mode}")
    print(f"   エポック数: {epochs}")
    print(f"   バッチサイズ: {batch_size}")
    print(f"   学習率: {learning_rate}")
    print(f"   最大データレコード数: {max_records}")
    print()
    
    # 学習実行
    print("🚀 学習を開始します...")
    result = train_model(
        mode=mode,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_records=max_records
    )
    
    print()
    print("=" * 60)
    
    if "error" in result:
        print(f"❌ エラー: {result['error']}")
    elif "output" in result:
        output = result["output"]
        if "message" in output:
            print(f"✅ {output['message']}")
        if "status" in output:
            print(f"   ステータス: {output['status']}")
        
        # 学習結果の詳細を表示
        if "final_loss" in output:
            print(f"   最終損失: {output['final_loss']:.4f}")
        if "epochs_completed" in output:
            print(f"   完了エポック数: {output['epochs_completed']}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("=" * 60)


if __name__ == "__main__":
    main()

