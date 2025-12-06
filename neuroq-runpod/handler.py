#!/usr/bin/env python3
"""
NeuroQ RunPod Serverless Handler
=================================
RunPod Serverless API用のハンドラーファイル

参照元:
- neuroquantum_layered.py: 層状QBNN-Transformer
- neuroquantum_brain.py: 脳型散在QBNN

エンドポイント:
- /generate: テキスト生成
- /health: ヘルスチェック
"""

import os
import sys
import json
import torch
from typing import Dict, Any, Optional

# 親ディレクトリをパスに追加（neuroquantum_*.py を参照するため）
# Dockerコンテナ内では同じディレクトリに配置されるので、親ディレクトリ参照は不要
# ただし、ローカル開発環境での互換性のため残す
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# まず現在のディレクトリを追加（Dockerコンテナ内ではこれで十分）
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
# 親ディレクトリも追加（ローカル開発用）
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# neuroquantum_layered.py からインポート
try:
    from neuroquantum_layered import (
        NeuroQuantumAI,
        NeuroQuantumTokenizer,
        NeuroQuantumConfig,
        NeuroQuantum,
    )
    NEUROQUANTUM_LAYERED_AVAILABLE = True
    print("✅ neuroquantum_layered.py からコンポーネントをインポートしました")
except ImportError as e:
    NEUROQUANTUM_LAYERED_AVAILABLE = False
    print(f"⚠️ neuroquantum_layered.py が見つかりません: {e}")

# neuroquantum_brain.py からインポート
try:
    from neuroquantum_brain import (
        NeuroQuantumBrainAI,
        BrainTokenizer,
        NeuroQuantumBrain,
    )
    NEUROQUANTUM_BRAIN_AVAILABLE = True
    print("✅ neuroquantum_brain.py からコンポーネントをインポートしました")
except ImportError as e:
    NEUROQUANTUM_BRAIN_AVAILABLE = False
    print(f"⚠️ neuroquantum_brain.py が見つかりません: {e}")

# RunPod SDK
try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False
    print("⚠️ runpodライブラリがインストールされていません。pip install runpod を実行してください。")

# OpenAI API（ChatGPTエンベディング用）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAIライブラリがインストールされていません。pip install openai を実行してください。")

# ========================================
# グローバル変数
# ========================================

# モデルインスタンス（グローバルに保持）
model_layered: Optional[NeuroQuantumAI] = None
model_brain: Optional[NeuroQuantumBrainAI] = None

# デバイス選択
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🍎 Apple Silicon GPU (MPS) を使用")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🎮 NVIDIA GPU (CUDA) を使用")
else:
    DEVICE = torch.device("cpu")
    print("💻 CPU を使用")


# ========================================
# モデル初期化
# ========================================

def init_model(mode: str = "layered", **kwargs) -> Dict[str, Any]:
    """
    モデルを初期化
    
    Args:
        mode: 'layered' または 'brain'
        **kwargs: モデル設定パラメータ
    
    Returns:
        初期化結果
    """
    global model_layered, model_brain
    
    try:
        if mode == "layered":
            if not NEUROQUANTUM_LAYERED_AVAILABLE:
                return {"error": "neuroquantum_layered.py が利用できません"}
            
            # デフォルト設定
            embed_dim = kwargs.get("embed_dim", 64)
            hidden_dim = kwargs.get("hidden_dim", 128)
            num_heads = kwargs.get("num_heads", 4)
            num_layers = kwargs.get("num_layers", 2)
            max_seq_len = kwargs.get("max_seq_len", 128)
            dropout = kwargs.get("dropout", 0.1)
            lambda_entangle = kwargs.get("lambda_entangle", 0.35)
            
            model_layered = NeuroQuantumAI(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                dropout=dropout,
                lambda_entangle=lambda_entangle,
            )
            model_layered.device = DEVICE
            
            return {
                "status": "success",
                "mode": "layered",
                "message": "Layered mode モデルを初期化しました"
            }
        
        elif mode == "brain":
            if not NEUROQUANTUM_BRAIN_AVAILABLE:
                return {"error": "neuroquantum_brain.py が利用できません"}
            
            # デフォルト設定
            embed_dim = kwargs.get("embed_dim", 128)
            num_heads = kwargs.get("num_heads", 4)
            num_layers = kwargs.get("num_layers", 3)
            num_neurons = kwargs.get("num_neurons", 75)
            max_vocab = kwargs.get("max_vocab", 50000)
            
            model_brain = NeuroQuantumBrainAI(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_neurons=num_neurons,
                max_vocab=max_vocab,
            )
            model_brain.device = DEVICE
            
            return {
                "status": "success",
                "mode": "brain",
                "message": "Brain mode モデルを初期化しました"
            }
        
        else:
            return {"error": f"不明なモード: {mode}"}
    
    except Exception as e:
        return {"error": f"モデル初期化エラー: {str(e)}"}


# ========================================
# テキスト生成
# ========================================

def generate_text(
    prompt: str,
    mode: str = "layered",
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    **kwargs
) -> Dict[str, Any]:
    """
    テキスト生成
    
    Args:
        prompt: 入力プロンプト
        mode: 'layered' または 'brain'
        max_length: 最大生成長
        temperature: 温度パラメータ
        top_k: Top-K サンプリング
        top_p: Top-P サンプリング
        **kwargs: その他のパラメータ
    
    Returns:
        生成結果
    """
    global model_layered, model_brain
    
    try:
        if mode == "layered":
            if model_layered is None:
                # モデルが初期化されていない場合は初期化
                init_result = init_model(mode="layered", **kwargs)
                if "error" in init_result:
                    return init_result
            
            if model_layered.model is None:
                return {"error": "モデルが学習されていません"}
            
            # テキスト生成
            generated = model_layered.generate(
                prompt=prompt,
                max_length=max_length,
                temp_min=temperature * 0.8,
                temp_max=temperature * 1.2,
                top_k=top_k,
                top_p=top_p,
            )
            
            return {
                "status": "success",
                "mode": "layered",
                "prompt": prompt,
                "generated": generated,
            }
        
        elif mode == "brain":
            if model_brain is None:
                # モデルが初期化されていない場合は初期化
                init_result = init_model(mode="brain", **kwargs)
                if "error" in init_result:
                    return init_result
            
            if model_brain.model is None:
                return {"error": "モデルが学習されていません"}
            
            # テキスト生成
            generated = model_brain.generate(
                prompt=prompt,
                max_length=max_length,
                temperature_min=temperature * 0.8,
                temperature_max=temperature * 1.2,
                top_k=top_k,
                top_p=top_p,
            )
            
            return {
                "status": "success",
                "mode": "brain",
                "prompt": prompt,
                "generated": generated,
            }
        
        else:
            return {"error": f"不明なモード: {mode}"}
    
    except Exception as e:
        return {"error": f"生成エラー: {str(e)}"}


# ========================================
# RunPod Handler
# ========================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless Handler
    
    リクエスト形式:
    {
        "input": {
            "action": "generate" | "init" | "health",
            "mode": "layered" | "brain",
            "prompt": "テキストプロンプト",
            "max_length": 100,
            "temperature": 0.7,
            ...
        }
    }
    """
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "generate")
        
        if action == "health":
            return {
                "status": "healthy",
                "layered_available": NEUROQUANTUM_LAYERED_AVAILABLE,
                "brain_available": NEUROQUANTUM_BRAIN_AVAILABLE,
                "openai_available": OPENAI_AVAILABLE,
                "device": str(DEVICE),
            }
        
        elif action == "init":
            mode = input_data.get("mode", "layered")
            kwargs = {k: v for k, v in input_data.items() if k != "action" and k != "mode"}
            return init_model(mode=mode, **kwargs)
        
        elif action == "generate":
            prompt = input_data.get("prompt", "")
            if not prompt:
                return {"error": "promptが必要です"}
            
            mode = input_data.get("mode", "layered")
            max_length = input_data.get("max_length", 100)
            temperature = input_data.get("temperature", 0.7)
            top_k = input_data.get("top_k", 40)
            top_p = input_data.get("top_p", 0.9)
            
            kwargs = {
                k: v for k, v in input_data.items()
                if k not in ["action", "prompt", "mode", "max_length", "temperature", "top_k", "top_p"]
            }
            
            return generate_text(
                prompt=prompt,
                mode=mode,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs
            )
        
        else:
            return {"error": f"不明なアクション: {action}"}
    
    except Exception as e:
        return {"error": f"ハンドラーエラー: {str(e)}"}


# ========================================
# RunPod Serverless 起動
# ========================================

if __name__ == "__main__":
    if RUNPOD_AVAILABLE:
        print("🚀 RunPod Serverless Handler を起動します...")
        runpod.serverless.start({"handler": handler})
    else:
        print("⚠️ RunPod SDKが利用できません。ローカルテストモードで実行します。")
        print("\nテストリクエスト例:")
        print(json.dumps({
            "input": {
                "action": "health"
            }
        }, indent=2))

