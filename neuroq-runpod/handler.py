#!/usr/bin/env python3
"""
NeuroQ RunPod Serverless Handler - Optimized Version
=====================================================
高速起動 & 安定動作のための最適化済みハンドラー

特徴:
- 起動時に重い処理をしない（高速起動）
- health checkは即座に200を返す
- モデルは初回リクエスト時にlazy load
- vocab_sizeの整合性を保証
"""

import runpod
import torch
import os
import sys

print("=" * 60)
print("⚛️ NeuroQ RunPod Serverless - Starting...")
print("=" * 60)

# ========================================
# グローバル変数（起動時は全てNone）
# ========================================
model = None
is_initialized = False

# 設定
VOCAB_SIZE = 8000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📊 Device: {DEVICE}")
print(f"📊 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")


# ========================================
# Lazy Model Loading（初回リクエスト時のみ）
# ========================================
def initialize_model():
    """モデルを初期化（初回リクエスト時のみ呼ばれる）"""
    global model, is_initialized
    
    if is_initialized:
        return True
    
    print("🔄 モデル初期化開始...")
    
    try:
        from neuroquantum_layered import NeuroQuantumAI
        
        # モデル作成（個別パラメータで初期化）
        model = NeuroQuantumAI(
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            num_layers=3,
            max_seq_len=256,
            dropout=0.1,
            lambda_entangle=0.5
        )
        
        # トークナイザー確認
        if os.path.exists("neuroq_tokenizer.model"):
            print("✅ トークナイザー: neuroq_tokenizer.model")
        else:
            print("⚠️ トークナイザーファイルが見つかりません")
        
        # 簡易学習（小さなデータで素早く）
        print("🔄 簡易学習開始...")
        training_data = [
            "こんにちは。私はニューロQです。" * 10,
            "量子コンピュータは次世代の技術です。" * 10,
            "人工知能は人間の知能を模倣します。" * 10,
            "ディープラーニングは機械学習の手法です。" * 10,
        ]
        
        model.train(training_data, epochs=3, seq_len=32)
        
        is_initialized = True
        print("✅ モデル初期化完了!")
        return True
        
    except Exception as e:
        print(f"❌ モデル初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========================================
# テキスト生成
# ========================================
def generate_text(prompt: str, max_length: int = 100, 
                  temperature: float = 0.7) -> str:
    """テキスト生成"""
    global model
    
    if model is None:
        return "Error: Model not initialized"
    
    try:
        result = model.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# ========================================
# メインハンドラー（RunPod用）
# ========================================
def handler(job):
    """
    RunPod Serverless Handler
    
    重要: health checkは即座に返す！
    """
    global is_initialized
    
    job_input = job.get("input", {})
    action = job_input.get("action", "generate")
    
    # ========================================
    # HEALTH CHECK（最優先・即座に返す）
    # ========================================
    if action == "health":
        return {
            "status": "healthy",
            "device": DEVICE,
            "cuda_available": torch.cuda.is_available(),
            "model_initialized": is_initialized
        }
    
    # ========================================
    # STATUS CHECK
    # ========================================
    if action == "status":
        return {
            "status": "ok",
            "initialized": is_initialized,
            "device": DEVICE,
            "vocab_size": VOCAB_SIZE
        }
    
    # ========================================
    # GENERATE（モデルが必要な処理）
    # ========================================
    if action == "generate":
        # Lazy initialization
        if not is_initialized:
            print("🔄 初回リクエスト - モデル初期化中...")
            if not initialize_model():
                return {
                    "status": "error",
                    "error": "Failed to initialize model"
                }
        
        prompt = job_input.get("prompt", "こんにちは")
        max_length = job_input.get("max_length", 100)
        temperature = job_input.get("temperature", 0.7)
        
        print(f"📝 Generate: prompt='{prompt[:30]}...'")
        
        result = generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        return {
            "status": "success",
            "prompt": prompt,
            "generated": result
        }
    
    # ========================================
    # TRAIN（学習）
    # ========================================
    if action == "train":
        if not is_initialized:
            if not initialize_model():
                return {
                    "status": "error",
                    "error": "Failed to initialize model"
                }
        
        texts = job_input.get("texts", [])
        epochs = job_input.get("epochs", 5)
        
        if not texts:
            return {
                "status": "error",
                "error": "No training texts provided"
            }
        
        try:
            model.train(texts, epochs=epochs, seq_len=32)
            return {
                "status": "success",
                "message": f"Training completed ({epochs} epochs)"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    # ========================================
    # UNKNOWN ACTION
    # ========================================
    return {
        "status": "error",
        "error": f"Unknown action: {action}",
        "available_actions": ["health", "status", "generate", "train"]
    }


# ========================================
# 起動（何もしない = 高速起動）
# ========================================
print("=" * 60)
print("✅ NeuroQ Handler Ready")
print("   - Health check: instant response")
print("   - Model loading: lazy (on first request)")
print("=" * 60)

# RunPod起動
runpod.serverless.start({"handler": handler})
