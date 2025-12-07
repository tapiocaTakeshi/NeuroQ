#!/usr/bin/env python3
"""
RunPod Serverless Handler へのリクエスト例

使用方法:
    python example_request.py
"""

import requests
import json
import os
from typing import Dict, Any, Optional

# ========================================
# 設定
# ========================================

# RunPod API設定
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "YOUR_API_KEY_HERE")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "YOUR_ENDPOINT_ID_HERE")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

# ========================================
# ヘルパー関数
# ========================================

def send_request(payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    """
    RunPod Serverless Handlerにリクエストを送信
    
    Args:
        payload: リクエストペイロード
        timeout: タイムアウト（秒）
    
    Returns:
        レスポンス
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    try:
        print(f"📤 リクエスト送信: {payload.get('input', {}).get('action', 'unknown')}")
        response = requests.post(
            RUNPOD_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"リクエストエラー: {str(e)}"}


def wait_for_completion(job_id: str, timeout: int = 600) -> Dict[str, Any]:
    """
    ジョブの完了を待つ
    
    Args:
        job_id: ジョブID
        timeout: タイムアウト（秒）
    
    Returns:
        結果
    """
    import time
    
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(status_url, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status", "UNKNOWN")
            print(f"   ステータス: {status}")
            
            if status == "COMPLETED":
                return result.get("output", {})
            elif status == "FAILED":
                return {"error": result.get("error", "ジョブが失敗しました")}
            
            time.sleep(2)  # 2秒待機
        except requests.exceptions.RequestException as e:
            return {"error": f"ステータス確認エラー: {str(e)}"}
    
    return {"error": "タイムアウト: ジョブが完了しませんでした"}


# ========================================
# リクエスト例
# ========================================

def health_check() -> Dict[str, Any]:
    """ヘルスチェック"""
    payload = {
        "input": {
            "action": "health"
        }
    }
    return send_request(payload, timeout=30)


def init_model_layered(
    embed_dim: int = 64,
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    max_seq_len: int = 128,
    dropout: float = 0.1,
    lambda_entangle: float = 0.35,
    use_openai_embedding: bool = False,
    openai_api_key: Optional[str] = None,
    openai_model: str = "text-embedding-3-large"
) -> Dict[str, Any]:
    """Layeredモードでモデルを初期化"""
    payload = {
        "input": {
            "action": "init",
            "mode": "layered",
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
            "lambda_entangle": lambda_entangle,
            "use_openai_embedding": use_openai_embedding,
            "openai_api_key": openai_api_key,
            "openai_model": openai_model,
        }
    }
    return send_request(payload, timeout=60)


def init_model_brain(
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 3,
    num_neurons: int = 75,
    max_vocab: int = 16000,
    use_openai_embedding: bool = False,
    openai_api_key: Optional[str] = None,
    openai_model: str = "text-embedding-3-large"
) -> Dict[str, Any]:
    """Brainモードでモデルを初期化"""
    payload = {
        "input": {
            "action": "init",
            "mode": "brain",
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "num_neurons": num_neurons,
            "max_vocab": max_vocab,
            "use_openai_embedding": use_openai_embedding,
            "openai_api_key": openai_api_key,
            "openai_model": openai_model,
        }
    }
    return send_request(payload, timeout=60)


def train_model_layered(
    texts: Optional[list] = None,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seq_length: int = 64,
    vocab_size: int = 16000,
    data_sources: list = ["huggingface"],
    common_crawl_config: Optional[Dict] = None,
    max_records: int = 100,
    **model_kwargs
) -> Dict[str, Any]:
    """Layeredモードでモデルを学習"""
    payload = {
        "input": {
            "action": "train",
            "mode": "layered",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seq_length": seq_length,
            "vocab_size": vocab_size,
            "data_sources": data_sources,
            "common_crawl_config": common_crawl_config or {},
            "max_records": max_records,
            **model_kwargs
        }
    }
    
    if texts:
        payload["input"]["texts"] = texts
    
    return send_request(payload, timeout=3600)  # 1時間


def train_model_brain(
    texts: Optional[list] = None,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seq_length: int = 64,
    max_records: int = 100,
    data_sources: list = ["huggingface"],
    common_crawl_config: Optional[Dict] = None,
    **model_kwargs
) -> Dict[str, Any]:
    """Brainモードでモデルを学習"""
    payload = {
        "input": {
            "action": "train",
            "mode": "brain",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seq_length": seq_length,
            "data_sources": data_sources,
            "common_crawl_config": common_crawl_config or {},
            "max_records": max_records,
            **model_kwargs
        }
    }
    
    if texts:
        payload["input"]["texts"] = texts
    
    return send_request(payload, timeout=3600)  # 1時間


def generate_text_layered(
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 2.0,
    train_before_generate: bool = False,
    data_sources: list = ["huggingface"],
    common_crawl_config: Optional[Dict] = None,
    max_records: int = 100,
    epochs: int = 20,
    **model_kwargs
) -> Dict[str, Any]:
    """Layeredモードでテキスト生成"""
    payload = {
        "input": {
            "action": "generate",
            "mode": "layered",
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "train_before_generate": train_before_generate,
            "data_sources": data_sources,
            "common_crawl_config": common_crawl_config or {},
            "max_records": max_records,
            "epochs": epochs,
            **model_kwargs
        }
    }
    
    timeout = 3600 if train_before_generate else 300
    return send_request(payload, timeout=timeout)


def generate_text_brain(
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    train_before_generate: bool = False,
    data_sources: list = ["huggingface"],
    common_crawl_config: Optional[Dict] = None,
    max_records: int = 100,
    epochs: int = 20,
    **model_kwargs
) -> Dict[str, Any]:
    """Brainモードでテキスト生成"""
    payload = {
        "input": {
            "action": "generate",
            "mode": "brain",
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "train_before_generate": train_before_generate,
            "data_sources": data_sources,
            "common_crawl_config": common_crawl_config or {},
            "max_records": max_records,
            "epochs": epochs,
            **model_kwargs
        }
    }
    
    timeout = 3600 if train_before_generate else 300
    return send_request(payload, timeout=timeout)


# ========================================
# 使用例
# ========================================

def main():
    """メイン関数 - 使用例"""
    
    print("=" * 70)
    print("🧠⚛️ ニューロQ RunPod リクエスト例")
    print("=" * 70)
    
    # 1. ヘルスチェック
    print("\n1. ヘルスチェック")
    print("-" * 70)
    health_result = health_check()
    print(json.dumps(health_result, indent=2, ensure_ascii=False))
    
    # 2. モデル初期化（Layered）
    print("\n2. モデル初期化（Layered）")
    print("-" * 70)
    init_result = init_model_layered(
        embed_dim=64,
        hidden_dim=128,
        num_heads=4,
        num_layers=2
    )
    print(json.dumps(init_result, indent=2, ensure_ascii=False))
    
    # 3. テキスト生成（Layered）- 学習済みモデルがある場合
    print("\n3. テキスト生成（Layered）")
    print("-" * 70)
    generate_result = generate_text_layered(
        prompt="ChatGPTについて教えて",
        max_length=100,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=2.0
    )
    print(json.dumps(generate_result, indent=2, ensure_ascii=False))
    
    # 4. テキスト生成（学習も同時に実行）
    print("\n4. テキスト生成（学習も同時に実行）")
    print("-" * 70)
    generate_with_train_result = generate_text_layered(
        prompt="量子コンピュータとは何ですか？",
        max_length=150,
        temperature=0.7,
        repetition_penalty=2.0,
        train_before_generate=True,
        data_sources=["huggingface"],
        max_records=100,
        epochs=20
    )
    print(json.dumps(generate_with_train_result, indent=2, ensure_ascii=False))
    
    # 5. モデル初期化（Brain）
    print("\n5. モデル初期化（Brain）")
    print("-" * 70)
    init_brain_result = init_model_brain(
        embed_dim=128,
        num_neurons=100,
        max_vocab=16000
    )
    print(json.dumps(init_brain_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # 環境変数の確認
    if RUNPOD_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️ 警告: RUNPOD_API_KEY環境変数を設定してください")
        print("   export RUNPOD_API_KEY='your_api_key'")
    
    if ENDPOINT_ID == "YOUR_ENDPOINT_ID_HERE":
        print("⚠️ 警告: RUNPOD_ENDPOINT_ID環境変数を設定してください")
        print("   export RUNPOD_ENDPOINT_ID='your_endpoint_id'")
        print("\n使用例のみを表示します...")
        print("\n" + "=" * 70)
        print("使用例:")
        print("=" * 70)
        print("\n1. ヘルスチェック:")
        print("""
result = health_check()
print(json.dumps(result, indent=2, ensure_ascii=False))
        """)
        print("\n2. テキスト生成:")
        print("""
result = generate_text_layered(
    prompt="ChatGPTについて教えて",
    max_length=100,
    temperature=0.7,
    repetition_penalty=2.0
)
print(result.get("generated", ""))
        """)
        print("\n3. 学習してから生成:")
        print("""
result = generate_text_layered(
    prompt="量子コンピュータとは何ですか？",
    max_length=150,
    train_before_generate=True,
    data_sources=["huggingface"],
    max_records=100,
    epochs=20
)
print(result.get("generated", ""))
        """)
    else:
        main()

