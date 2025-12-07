#!/usr/bin/env python3
"""
RunPod Serverless Handler - 改善版学習→生成スクリプト

生成テキストの品質を向上させるための改善されたパラメータを使用
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


def train_and_generate_improved(
    prompt: str,
    mode: str = "layered",
    preset: str = "short",  # "short", "standard", "high_quality"
    **custom_params
) -> Dict[str, Any]:
    """
    改善されたパラメータで学習→生成を実行
    
    Args:
        prompt: 生成プロンプト
        mode: "layered" または "brain"
        preset: プリセット設定 ("short", "standard", "high_quality")
        **custom_params: カスタムパラメータ（プリセットを上書き）
    """
    
    # プリセット設定
    presets = {
        "short": {
            "max_records": 300,
            "epochs": 30,
            "batch_size": 16,
            "learning_rate": 0.0005,
            "seq_length": 64,
            "max_length": 80,
            "temperature": 0.6,
            "top_k": 30,
            "top_p": 0.85,
            "repetition_penalty": 3.0
        },
        "standard": {
            "max_records": 500,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "seq_length": 128,
            "max_length": 100,
            "temperature": 0.6,
            "top_k": 30,
            "top_p": 0.85,
            "repetition_penalty": 3.0,
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 4
        },
        "high_quality": {
            "max_records": 1000,
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.00005,
            "seq_length": 256,
            "max_length": 100,
            "temperature": 0.6,
            "top_k": 30,
            "top_p": 0.85,
            "repetition_penalty": 3.0,
            "embed_dim": 512,
            "hidden_dim": 1024,
            "num_heads": 16,
            "num_layers": 6
        }
    }
    
    # プリセットを取得
    params = presets.get(preset, presets["short"]).copy()
    
    # カスタムパラメータで上書き
    params.update(custom_params)
    
    # ペイロードを作成
    payload = {
        "action": "generate",
        "mode": mode,
        "prompt": prompt,
        "max_length": params.pop("max_length"),
        "temperature": params.pop("temperature"),
        "top_k": params.pop("top_k"),
        "top_p": params.pop("top_p"),
        "repetition_penalty": params.pop("repetition_penalty"),
        "train_before_generate": True,
        "epochs": params.pop("epochs"),
        "batch_size": params.pop("batch_size"),
        "learning_rate": params.pop("learning_rate"),
        "seq_length": params.pop("seq_length"),
        "data_sources": ["huggingface"],
        "common_crawl_config": {
            "max_records": params.pop("max_records")
        },
        **params  # 残りのパラメータ（モデルパラメータなど）
    }
    
    timeout = 3600 if preset == "high_quality" else 1800 if preset == "standard" else 600
    return send_and_wait(payload, timeout=timeout)


def main():
    """メイン関数"""
    print("=" * 60)
    print("🧠⚛️ ニューロQ RunPod - 改善版学習→生成")
    print("=" * 60)
    print()
    
    # 改善されたパラメータ
    mode = "layered"
    preset = "short"  # "short", "standard", "high_quality"
    prompt = "ChatGPTについて教えて"
    
    print(f"📋 設定:")
    print(f"   モード: {mode}")
    print(f"   プリセット: {preset}")
    print(f"   プロンプト: {prompt}")
    print()
    
    if preset == "short":
        print("📊 学習設定 (短時間改善版):")
        print("   エポック数: 30")
        print("   データレコード数: 300")
        print("   学習率: 0.0005")
        print()
        print("📝 生成設定 (改善版):")
        print("   温度: 0.6 (特殊トークンを抑制)")
        print("   Top-K: 30 (より保守的)")
        print("   Top-P: 0.85 (多様性を抑える)")
        print("   繰り返しペナルティ: 3.0 (より強力)")
    elif preset == "standard":
        print("📊 学習設定 (標準改善版):")
        print("   エポック数: 50")
        print("   データレコード数: 500")
        print("   学習率: 0.0001")
        print("   モデルサイズ: より大きなモデル")
        print()
        print("📝 生成設定 (改善版):")
        print("   温度: 0.6")
        print("   Top-K: 30")
        print("   繰り返しペナルティ: 3.0")
    elif preset == "high_quality":
        print("📊 学習設定 (高品質版):")
        print("   エポック数: 100")
        print("   データレコード数: 1000")
        print("   学習率: 0.00005")
        print("   モデルサイズ: 最大")
        print()
        print("📝 生成設定 (改善版):")
        print("   温度: 0.6")
        print("   Top-K: 30")
        print("   繰り返しペナルティ: 3.0")
    
    print()
    print("🚀 学習と生成を開始します...")
    print("   ⏳ この処理には数分かかる場合があります...")
    print()
    
    result = train_and_generate_improved(
        prompt=prompt,
        mode=mode,
        preset=preset
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
            print()
            
            # 特殊トークンの検出
            special_tokens = ["PARAGRAPH", "SECTION", "START", "NEWLINE", "ARTICLE"]
            found_tokens = [token for token in special_tokens if token in output['generated']]
            if found_tokens:
                print(f"⚠️ 注意: 特殊トークンが検出されました: {', '.join(found_tokens)}")
                print(f"   より多くのデータとエポック数で学習すると改善されます。")
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("=" * 60)


if __name__ == "__main__":
    main()

