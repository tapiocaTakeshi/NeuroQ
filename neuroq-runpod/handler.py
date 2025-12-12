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
import subprocess
import threading
import time
from pathlib import Path

print("=" * 60)
print("⚛️ NeuroQ RunPod Serverless - Starting...")
print("=" * 60)

# ========================================
# グローバル変数（起動時は全てNone）
# ========================================
model = None
is_initialized = False

# 学習状態管理
pretrain_process = None
pretrain_status = "idle"  # idle, running, completed, error
pretrain_log_file = "training_openai.log"

# 設定
VOCAB_SIZE = 8000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📊 Device: {DEVICE}")
print(f"📊 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")


# ========================================
# 事前学習済みモデルのパス
# ========================================
PRETRAINED_MODEL_PATH = "neuroq_pretrained.pt"


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
        from neuroquantum_layered import NeuroQuantumAI, NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer
        
        # ========================================
        # 方法1: 事前学習済みモデルをロード（推奨）
        # ========================================
        # ファイルが存在し、サイズが1KB以上の場合のみロード
        if os.path.exists(PRETRAINED_MODEL_PATH) and os.path.getsize(PRETRAINED_MODEL_PATH) > 1024:
            print(f"📦 事前学習済みモデルをロード: {PRETRAINED_MODEL_PATH}")
            
            checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
            config_dict = checkpoint['config']
            
            # Configを復元
            config = NeuroQuantumConfig(
                vocab_size=config_dict['vocab_size'],
                embed_dim=config_dict['embed_dim'],
                hidden_dim=config_dict['hidden_dim'],
                num_heads=config_dict['num_heads'],
                num_layers=config_dict['num_layers'],
                max_seq_len=config_dict['max_seq_len'],
                dropout=config_dict['dropout'],
                lambda_entangle=config_dict['lambda_entangle'],
            )
            
            # トークナイザーをロード
            tokenizer = NeuroQuantumTokenizer(
                vocab_size=config_dict['vocab_size'],
                model_file="neuroq_tokenizer.model"
            )
            
            # モデルを構築してウェイトをロード
            nn_model = NeuroQuantum(config).to(DEVICE)
            nn_model.load_state_dict(checkpoint['model_state_dict'])
            nn_model.eval()
            
            # NeuroQuantumAI のラッパーを作成
            model = NeuroQuantumAI(
                embed_dim=config_dict['embed_dim'],
                hidden_dim=config_dict['hidden_dim'],
                num_heads=config_dict['num_heads'],
                num_layers=config_dict['num_layers'],
                max_seq_len=config_dict['max_seq_len'],
                dropout=config_dict['dropout'],
                lambda_entangle=config_dict['lambda_entangle'],
            )
            model.model = nn_model
            model.config = config
            model.tokenizer = tokenizer
            
            print(f"✅ 事前学習済みモデルロード完了!")
            print(f"   vocab_size: {config_dict['vocab_size']}")
            print(f"   embed_dim: {config_dict['embed_dim']}")
            print(f"   パラメータ数: {nn_model.num_params:,}")
            
            is_initialized = True
            return True

        # ========================================
        # 方法2: 簡易学習（事前学習済みモデルがない場合）
        # ========================================
        print("⚠️ 事前学習済みモデルが見つかりません。簡易学習を実行...")
        
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
        
        # 簡易学習データ
        print("🔄 簡易学習開始...")
        training_data = [
            "こんにちは。私はニューロQです。量子ビットニューラルネットワークを使った人工知能です。",
            "ニューロQは量子コンピュータの原理を活用した次世代のAIシステムです。",
            "量子コンピュータは量子力学の原理を利用した次世代の計算機です。",
            "人工知能は人間の知能を模倣するコンピュータシステムです。",
            "機械学習はデータからパターンを学習するAIの手法です。",
            "ディープラーニングはニューラルネットワークを多層化した技術です。",
        ] * 10
        
        model.train(training_data, epochs=5, seq_len=64)
        
        is_initialized = True
        print("✅ 簡易学習完了!")
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
    # PRETRAIN_OPENAI（OpenAIデータセット事前学習）
    # ========================================
    if action == "pretrain_openai":
        global pretrain_process, pretrain_status

        # 既に実行中の場合
        if pretrain_status == "running":
            return {
                "status": "error",
                "error": "Pretraining is already running",
                "pretrain_status": pretrain_status
            }

        # ログファイルのパスを確認
        log_path = Path(pretrain_log_file)

        try:
            # バックグラウンドでpretrain_openai.pyを実行
            print("🚀 Starting OpenAI pretraining...")
            pretrain_status = "running"

            # python -u で unbuffered output
            cmd = [
                sys.executable, "-u",
                "pretrain_openai_quick.py"
            ]

            # ログファイルを開いてsubprocessを起動
            with open(log_path, 'w') as log_file:
                pretrain_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )

            # 非同期でプロセスを監視
            def monitor_pretrain():
                global pretrain_status, pretrain_process
                pretrain_process.wait()
                if pretrain_process.returncode == 0:
                    pretrain_status = "completed"
                    print("✅ Pretraining completed successfully")
                else:
                    pretrain_status = "error"
                    print(f"❌ Pretraining failed with code {pretrain_process.returncode}")

            monitor_thread = threading.Thread(target=monitor_pretrain, daemon=True)
            monitor_thread.start()

            return {
                "status": "success",
                "message": "Pretraining started",
                "pretrain_status": pretrain_status,
                "log_file": str(log_path),
                "pid": pretrain_process.pid
            }

        except Exception as e:
            pretrain_status = "error"
            return {
                "status": "error",
                "error": str(e),
                "pretrain_status": pretrain_status
            }

    # ========================================
    # PRETRAIN_STATUS（事前学習ステータス確認）
    # ========================================
    if action == "pretrain_status":
        log_path = Path(pretrain_log_file)

        # ログファイルの最後の数行を読む
        log_tail = ""
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    log_tail = ''.join(lines[-20:])  # 最後の20行
            except Exception as e:
                log_tail = f"Error reading log: {e}"

        return {
            "status": "success",
            "pretrain_status": pretrain_status,
            "log_file": str(log_path),
            "log_exists": log_path.exists(),
            "log_tail": log_tail,
            "process_running": pretrain_process is not None and pretrain_process.poll() is None
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
        "available_actions": ["health", "status", "generate", "train", "pretrain_openai", "pretrain_status"]
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
