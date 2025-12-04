#!/usr/bin/env python3
"""
NeuroQ RunPod Serverless Handler
================================
RunPod Serverless Endpoint用のハンドラー

サポートモード:
- Brain Mode: 脳型散在QBNN
- Layered Mode: 層状QBNN-Transformer

使用方法:
1. このファイルと neuroq_model.py をRunPodにデプロイ
2. モデルファイル (neuroq_model.pt, neuroq_tokenizer.json) を配置
3. RunPod Endpoint から呼び出し
"""

import runpod
import torch
import os
import traceback

from neuroq_model import (
    NeuroQGenerator, 
    NeuroQModel, 
    NeuroQTokenizer, 
    NeuroQConfig,
    create_neuroq_brain,
    create_neuroq_layered,
)

# ========================================
# グローバル設定
# ========================================

MODEL_PATH = os.environ.get("NEUROQ_MODEL_PATH", "neuroq_model.pt")
TOKENIZER_PATH = os.environ.get("NEUROQ_TOKENIZER_PATH", "neuroq_tokenizer.json")
DEFAULT_MODE = os.environ.get("NEUROQ_MODE", "layered")  # 'brain' or 'layered'

# デバイス選択
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🎮 CUDA GPU を使用: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🍎 Apple Silicon GPU (MPS) を使用")
else:
    DEVICE = "cpu"
    print("💻 CPU を使用")

# ========================================
# モデルのロード
# ========================================

generator = None

def load_model(mode: str = None):
    """
    モデルをロード（遅延ロード）
    
    Args:
        mode: 'brain' or 'layered' (Noneの場合はDEFAULT_MODEを使用)
    """
    global generator
    
    if generator is not None:
        return generator
    
    if mode is None:
        mode = DEFAULT_MODE
    
    print(f"📥 NeuroQ モデルをロード中...")
    print(f"   Mode: {mode}")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Tokenizer: {TOKENIZER_PATH}")
    print(f"   Device: {DEVICE}")
    
    # ファイルが存在するか確認
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ モデルファイルが見つかりません: {MODEL_PATH}")
        print("   デモモードで起動します（学習済みモデルなし）")
        
        # デモ用の小さなモデルを作成
        if mode == 'brain':
            config = NeuroQConfig(
                mode='brain',
                vocab_size=2000,
                embed_dim=64,
                num_neurons=32,
                num_heads=4,
                num_layers=2,
                max_seq_len=128,
                connection_density=0.25,
            )
        else:  # layered
            config = NeuroQConfig(
                mode='layered',
                vocab_size=2000,
                embed_dim=64,
                hidden_dim=128,
                num_heads=4,
                num_layers=2,
                max_seq_len=128,
            )
        
        model = NeuroQModel(config)
        tokenizer = NeuroQTokenizer(vocab_size=2000)
        
        # 基本的な語彙を構築
        basic_texts = [
            "こんにちは、私はNeuroQです。量子ビットニューラルネットワークベースの生成AIです。",
            "Hello, I am NeuroQ. A generative AI based on Quantum-Bit Neural Network.",
            "量子コンピュータは次世代の計算技術です。",
            "人工知能は私たちの生活を変革しています。",
            "QBNNは量子もつれを模倣したニューラルネットワークです。",
        ]
        tokenizer.build_vocab(basic_texts)
        
        generator = NeuroQGenerator(model, tokenizer, DEVICE)
        print(f"✅ デモモデル作成完了 (Mode: {mode})")
    else:
        # 学習済みモデルをロード
        generator = NeuroQGenerator.load(MODEL_PATH, TOKENIZER_PATH, DEVICE)
        print(f"✅ モデルロード完了")
    
    # モデル情報を表示
    info = generator.get_model_info()
    print(f"   モード: {info.get('mode', 'unknown')}")
    print(f"   パラメータ数: {info['num_params']:,}")
    print(f"   埋め込み次元: {info['embed_dim']}")
    print(f"   隠れ層次元: {info['hidden_dim']}")
    print(f"   ニューロン数: {info.get('num_neurons', 'N/A')}")
    print(f"   レイヤー数: {info['num_layers']}")
    
    return generator


# ========================================
# RunPod Handler
# ========================================

def handler(job):
    """
    RunPod Serverless ハンドラー
    
    入力JSON形式:
    {
        "input": {
            "prompt": "生成したいテキストのプロンプト",
            "max_tokens": 128,        # オプション（デフォルト: 128）
            "temperature": 0.7,       # オプション（デフォルト: 0.7）
            "top_k": 40,              # オプション（デフォルト: 40）
            "top_p": 0.9,             # オプション（デフォルト: 0.9）
            "repetition_penalty": 1.2 # オプション（デフォルト: 1.2）
        }
    }
    
    出力JSON形式:
    {
        "prompt": "入力プロンプト",
        "output": "生成されたテキスト",
        "model_info": {
            "mode": "brain" or "layered",
            ...
        }
    }
    """
    try:
        # モデルをロード（初回のみ）
        gen = load_model()
        
        # 入力を取得
        job_input = job.get("input", {})
        
        # パラメータを取得
        prompt = job_input.get("prompt", "")
        max_tokens = int(job_input.get("max_tokens", 128))
        temperature = float(job_input.get("temperature", 0.7))
        top_k = int(job_input.get("top_k", 40))
        top_p = float(job_input.get("top_p", 0.9))
        repetition_penalty = float(job_input.get("repetition_penalty", 1.2))
        
        # バリデーション
        if not prompt:
            return {"error": "prompt is required"}
        
        if max_tokens < 1 or max_tokens > 1024:
            max_tokens = min(max(1, max_tokens), 1024)
        
        if temperature < 0.1 or temperature > 2.0:
            temperature = min(max(0.1, temperature), 2.0)
        
        print(f"📝 生成リクエスト:")
        print(f"   Prompt: {prompt[:50]}...")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Temperature: {temperature}")
        
        # テキスト生成
        output_text = gen.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        print(f"✅ 生成完了: {len(output_text)} 文字")
        
        # レスポンス
        return {
            "prompt": prompt,
            "output": output_text,
            "model_info": gen.get_model_info(),
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}


# ========================================
# 量子情報エンドポイント
# ========================================

def quantum_info(job):
    """量子もつれ情報を取得"""
    try:
        gen = load_model()
        model_info = gen.get_model_info()
        quantum_info = gen.model.get_quantum_info()
        
        return {
            "status": "success",
            "mode": model_info.get('mode', 'unknown'),
            "quantum_info": quantum_info,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ========================================
# ヘルスチェック用エンドポイント
# ========================================

def health_check(job):
    """ヘルスチェック"""
    try:
        gen = load_model()
        return {
            "status": "healthy",
            "model_loaded": gen is not None,
            "device": DEVICE,
            "model_info": gen.get_model_info() if gen else None,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


# ========================================
# メイン
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠⚛️ NeuroQ RunPod Serverless Worker")
    print("   Brain Mode: 脳型散在QBNN")
    print("   Layered Mode: 層状QBNN-Transformer")
    print("=" * 60)
    
    # 起動時にモデルをプリロード
    load_model()
    
    # RunPod Serverless を開始
    runpod.serverless.start({
        "handler": handler,
    })
