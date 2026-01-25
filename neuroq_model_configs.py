#!/usr/bin/env python3
"""
NeuroQ モデル設定ファイル

OpenAI text-embedding-3 の次元数に対応したモデル設定

text-embedding-3-small: 1536次元
text-embedding-3-large: 3072次元

使用例:
    from neuroq_model_configs import NEUROQ_SMALL, NEUROQ_LARGE, create_model
    model = create_model(NEUROQ_LARGE)
"""

import os
import sys
from pathlib import Path

# パス設定
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

runpod_dir = os.path.join(current_dir, 'neuroq-runpod')
if runpod_dir not in sys.path:
    sys.path.insert(0, runpod_dir)

# ===============================
# NeuroQ モデル設定
# ===============================

# NeuroQ モデル設定
NEUROQ_SMALL = {
    'name': 'NeuroQ-Small',
    'description': '軽量モデル (256次元)',
    'vocab_size': 8000,        # SentencePiece BPE トークナイザー
    'embed_dim': 256,
    'num_heads': 4,            # 固定
    'num_layers': 2,           # 固定
    'max_seq_len': 512,
    'dropout': 0.1,
}

NEUROQ_LARGE = {
    'name': 'NeuroQ-Large',
    'description': '標準モデル (384次元)',
    'vocab_size': 8000,        # SentencePiece BPE トークナイザー
    'embed_dim': 384,
    'num_heads': 4,            # 固定
    'num_layers': 2,           # 固定
    'max_seq_len': 512,
    'dropout': 0.1,
}

# 利用可能なモデル一覧
AVAILABLE_MODELS = {
    'small': NEUROQ_SMALL,
    'large': NEUROQ_LARGE,
}


def create_model(config, device=None):
    """
    設定からNeuroQuantumモデルを作成
    
    Args:
        config: モデル設定辞書 (NEUROQ_SMALL or NEUROQ_LARGE)
        device: PyTorchデバイス (None の場合は自動選択)
    
    Returns:
        model: NeuroQuantumモデル
        device: 使用デバイス
    """
    import torch
    
    try:
        from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    except ImportError:
        from neuroq_runpod.neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    
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
    model_config.max_seq_len = config['max_seq_len']
    model_config.dropout = config['dropout']
    
    # モデル作成
    model = NeuroQuantum(model_config).to(device)
    
    return model, device


def get_model_info(config):
    """モデル情報を取得"""
    import torch
    
    model, device = create_model(config, device=torch.device("cpu"))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # メモリ推定 (float32)
    memory_mb = total_params * 4 / (1024 * 1024)
    
    info = {
        'name': config['name'],
        'description': config['description'],
        'embed_dim': config['embed_dim'],
        'num_heads': config['num_heads'],
        'num_layers': config['num_layers'],
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_mb': memory_mb,
    }
    
    del model
    return info


def print_model_comparison():
    """モデル比較を表示"""
    print("=" * 70)
    print("📊 NeuroQ モデル比較")
    print("   OpenAI text-embedding-3 次元数対応")
    print("=" * 70)
    
    for model_name, config in AVAILABLE_MODELS.items():
        info = get_model_info(config)
        
        print(f"\n🧠 {info['name']}")
        print(f"   説明: {info['description']}")
        print(f"   ────────────────────────────────────")
        print(f"   embed_dim:    {info['embed_dim']:,}")
        print(f"   num_heads:    {info['num_heads']}")
        print(f"   num_layers:   {info['num_layers']}")
        print(f"   ────────────────────────────────────")
        print(f"   総パラメータ数: {info['total_params']:,} ({info['total_params']/1e6:.1f}M)")
        print(f"   メモリ使用量:  {info['memory_mb']:.1f} MB (float32)")
    
    print("\n" + "=" * 70)
    print("📋 OpenAI text-embedding-3 との対応")
    print("=" * 70)
    print(f"   text-embedding-3-small → NeuroQ-Small (embed_dim=1536)")
    print(f"   text-embedding-3-large → NeuroQ-Large (embed_dim=3072)")
    print("=" * 70)


if __name__ == "__main__":
    print_model_comparison()
