#!/usr/bin/env python3
"""
NeuroQ モデル設定ファイル（neuroq-runpod用）

OpenAI text-embedding-3 の次元数に対応したモデル設定

text-embedding-3-small: 1536次元
text-embedding-3-large: 3072次元

使用例:
    from model_configs import NEUROQ_SMALL, NEUROQ_LARGE, create_model, get_model_config
    
    # 設定を取得
    config = get_model_config('small')
    
    # モデルを作成
    model = create_model(NEUROQ_LARGE)
"""

import os
import sys
from pathlib import Path

# パス設定
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ===============================
# NeuroQ モデル設定
# ===============================

# 既存の小さいモデル（高速・軽量）
NEUROQ_MICRO = {
    'name': 'NeuroQ-Micro',
    'description': '軽量高速モデル（開発・テスト用）',
    'vocab_size': 8000,        # SentencePiece BPE トークナイザー
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 2,
    'max_seq_len': 256,
    'dropout': 0.1,
}

# OpenAI text-embedding-3-small 対応
NEUROQ_SMALL = {
    'name': 'NeuroQ-Small',
    'description': 'OpenAI text-embedding-3-small (1536次元) 対応',
    'vocab_size': 8000,        # SentencePiece BPE トークナイザー
    'embed_dim': 1536,         # text-embedding-3-small と同じ
    'num_heads': 4,
    'num_layers': 2,
    'max_seq_len': 512,
    'dropout': 0.1,
}

# OpenAI text-embedding-3-large 対応
NEUROQ_LARGE = {
    'name': 'NeuroQ-Large',
    'description': 'OpenAI text-embedding-3-large (3072次元) 対応',
    'vocab_size': 8000,        # SentencePiece BPE トークナイザー
    'embed_dim': 3072,         # text-embedding-3-large と同じ
    'num_heads': 4,
    'num_layers': 2,
    'max_seq_len': 512,
    'dropout': 0.1,
}

# 利用可能なモデル一覧
AVAILABLE_MODELS = {
    'micro': NEUROQ_MICRO,
    'small': NEUROQ_SMALL,
    'large': NEUROQ_LARGE,
}

# チェックポイントパス
CHECKPOINT_PATHS = {
    'micro': '/model_checkpoints/neuroq_micro_checkpoint.pt',
    'small': '/model_checkpoints/neuroq_small_checkpoint.pt',
    'large': '/model_checkpoints/neuroq_large_checkpoint.pt',
}


def get_model_config(model_size: str = 'micro'):
    """
    モデルサイズに応じた設定を取得
    
    Args:
        model_size: 'micro', 'small', 'large'
    
    Returns:
        モデル設定辞書
    """
    model_size = model_size.lower()
    if model_size not in AVAILABLE_MODELS:
        print(f"⚠️ 不明なモデルサイズ: {model_size}、'micro'を使用します")
        model_size = 'micro'
    return AVAILABLE_MODELS[model_size]


def get_checkpoint_path(model_size: str = 'micro'):
    """
    モデルサイズに応じたチェックポイントパスを取得
    """
    model_size = model_size.lower()
    if model_size not in CHECKPOINT_PATHS:
        model_size = 'micro'
    return CHECKPOINT_PATHS[model_size]


def create_model(config, device=None):
    """
    設定からNeuroQuantumモデルを作成
    
    Args:
        config: モデル設定辞書
        device: PyTorchデバイス (None の場合は自動選択)
    
    Returns:
        model: NeuroQuantumモデル
        device: 使用デバイス
    """
    import torch
    
    try:
        from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    except ImportError:
        print("❌ neuroquantum_layered.py のインポートに失敗")
        raise
    
    # デバイス選択
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    # モデル設定
    model_config = NeuroQuantumConfig()
    model_config.vocab_size = config['vocab_size']
    model_config.embed_dim = config['embed_dim']
    model_config.num_heads = config['num_heads']
    model_config.num_layers = config['num_layers']
    model_config.max_seq_len = config.get('max_seq_len', 256)
    model_config.dropout = config.get('dropout', 0.1)
    
    # モデル作成
    model = NeuroQuantum(model_config).to(device)
    
    return model, device


def get_model_info(config):
    """モデル情報を取得（メモリを使わない）"""
    # パラメータ数の概算
    vocab_size = config['vocab_size']
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    
    # 概算（embedding + transformer layers）
    embedding_params = vocab_size * embed_dim
    attention_params = 4 * embed_dim * embed_dim * num_layers  # Q, K, V, O
    ffn_params = 8 * embed_dim * embed_dim * num_layers  # 2層FFN
    
    total_params = embedding_params + attention_params + ffn_params
    memory_mb = total_params * 4 / (1024 * 1024)  # float32
    
    return {
        'name': config['name'],
        'description': config['description'],
        'embed_dim': config['embed_dim'],
        'num_heads': config['num_heads'],
        'num_layers': config['num_layers'],
        'estimated_params': total_params,
        'estimated_memory_mb': memory_mb,
    }


def print_available_models():
    """利用可能なモデル一覧を表示"""
    print("=" * 60)
    print("📦 NeuroQ 利用可能なモデル")
    print("=" * 60)
    
    for model_name, config in AVAILABLE_MODELS.items():
        info = get_model_info(config)
        print(f"\n🧠 {model_name.upper()}: {info['name']}")
        print(f"   {info['description']}")
        print(f"   embed_dim={info['embed_dim']}, heads={info['num_heads']}, layers={info['num_layers']}")
        print(f"   推定パラメータ数: ~{info['estimated_params']/1e6:.0f}M")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_available_models()
