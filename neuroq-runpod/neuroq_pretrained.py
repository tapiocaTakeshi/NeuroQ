#!/usr/bin/env python3
"""
NeuroQ 事前学習済みモデルモジュール
====================================
事前学習済みモデルのロードと初期化を提供します。

使い方:
    from neuroq_pretrained import load_pretrained_model

    model = load_pretrained_model(device='cuda')
"""

import torch
import os
import sys
from pathlib import Path


# デフォルトのモデル設定
DEFAULT_CONFIG = {
    'vocab_size': 8000,
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_heads': 8,
    'num_layers': 6,
    'max_seq_len': 256,
    'dropout': 0.1,
    'lambda_entangle': 0.5,
}


def get_model_path():
    """
    事前学習済みモデルのパスを取得

    Returns:
        str: モデルファイルのパス
    """
    # このファイルの親ディレクトリ（/home/user/）から探す
    current_dir = Path(__file__).parent

    # 候補パスをチェック
    candidate_paths = [
        current_dir / 'NeuroQ' / 'neuroq_pretrained.pt',
        current_dir / 'neuroq_pretrained.pt',
    ]

    for path in candidate_paths:
        if path.exists():
            return str(path)

    # 見つからない場合はデフォルトパスを返す
    return str(current_dir / 'NeuroQ' / 'neuroq_pretrained.pt')


def is_lfs_pointer_file(file_path):
    """
    Check if a file is a Git LFS pointer file.

    Args:
        file_path: Path to the file to check

    Returns:
        bool: True if the file is an LFS pointer, False otherwise
    """
    try:
        # LFS pointer files are typically very small (< 200 bytes)
        file_size = os.path.getsize(file_path)
        if file_size > 1024:
            return False

        # Read the first few bytes to check for LFS marker
        with open(file_path, 'rb') as f:
            header = f.read(50).decode('utf-8', errors='ignore')
            return header.startswith('version https://git-lfs.github.com/spec/')
    except:
        return False


def validate_checkpoint(checkpoint):
    """
    チェックポイントの妥当性を検証

    Args:
        checkpoint: ロードされたチェックポイント

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(checkpoint, dict):
        return False, f"Invalid checkpoint format. Expected dict, got {type(checkpoint).__name__}"

    required_keys = ['config', 'model_state_dict']
    missing_keys = [key for key in required_keys if key not in checkpoint]

    if missing_keys:
        return False, f"Missing required keys: {missing_keys}"

    config = checkpoint['config']
    if not isinstance(config, dict):
        return False, f"Invalid config format. Expected dict, got {type(config).__name__}"

    required_config_keys = ['vocab_size', 'embed_dim', 'hidden_dim', 'num_heads', 'num_layers']
    missing_config_keys = [key for key in required_config_keys if key not in config]

    if missing_config_keys:
        return False, f"Missing required config keys: {missing_config_keys}"

    return True, None


def load_pretrained_model(model_path=None, device='cpu', verbose=True):
    """
    事前学習済みモデルをロード

    Args:
        model_path: モデルファイルのパス（Noneの場合は自動検出）
        device: デバイス（'cpu', 'cuda', 'mps'）
        verbose: 詳細ログを表示するか

    Returns:
        tuple: (model, config, checkpoint) または (None, None, None) if failed
    """
    try:
        # パスの決定
        if model_path is None:
            model_path = get_model_path()

        if verbose:
            print(f"📦 Loading pretrained model from: {model_path}")

        # ファイルの存在確認
        if not os.path.exists(model_path):
            if verbose:
                print(f"❌ Model file not found: {model_path}")
            return None, None, None

        # ファイルサイズ確認
        file_size = os.path.getsize(model_path)
        if verbose:
            print(f"📊 File size: {file_size / (1024 * 1024):.2f} MB")

        # Check for Git LFS pointer file
        if file_size < 1024:  # 1KB未満
            if is_lfs_pointer_file(model_path):
                if verbose:
                    print(f"❌ Git LFS pointer file detected ({file_size} bytes)")
                    print(f"")
                    print(f"   The model file has not been downloaded from Git LFS.")
                    print(f"   Please run one of the following:")
                    print(f"")
                    print(f"   1. Use the helper script:")
                    print(f"      $ python download_model.py")
                    print(f"")
                    print(f"   2. Pull from Git LFS manually:")
                    print(f"      $ git lfs install")
                    print(f"      $ git lfs pull")
                    print(f"")
                    print(f"   3. Use Docker with automatic LFS pull:")
                    print(f"      $ docker build \\")
                    print(f"          --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \\")
                    print(f"          -t neuroq:latest .")
                    print(f"")
                return None, None, None
            else:
                if verbose:
                    print(f"❌ File too small ({file_size} bytes). Possibly corrupted.")
                return None, None, None

        # チェックポイントをロード
        checkpoint = torch.load(model_path, map_location=device)

        # 検証
        is_valid, error_msg = validate_checkpoint(checkpoint)
        if not is_valid:
            if verbose:
                print(f"❌ Checkpoint validation failed: {error_msg}")
            return None, None, None

        # 設定を取得
        config = checkpoint['config']

        if verbose:
            print(f"✅ Model loaded successfully")
            print(f"   vocab_size: {config['vocab_size']}")
            print(f"   embed_dim: {config['embed_dim']}")
            print(f"   hidden_dim: {config['hidden_dim']}")
            print(f"   num_heads: {config['num_heads']}")
            print(f"   num_layers: {config['num_layers']}")

        # neuroquantum_layered からモデルクラスをインポート
        # Note: このインポートはここで行う（遅延インポート）
        neuroq_module_path = Path(__file__).parent / 'NeuroQ' / 'neuroq-runpod'
        if str(neuroq_module_path) not in sys.path:
            sys.path.insert(0, str(neuroq_module_path))

        from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig

        # Configオブジェクトを作成
        model_config = NeuroQuantumConfig(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            max_seq_len=config['max_seq_len'],
            dropout=config.get('dropout', 0.1),
            lambda_entangle=config.get('lambda_entangle', 0.5),
        )

        # モデルを構築
        model = NeuroQuantum(model_config).to(device)

        # ウェイトをロード
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if verbose:
            print(f"✅ Model initialized on device: {device}")

        return model, config, checkpoint

    except Exception as e:
        if verbose:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
        return None, None, None


def create_model_from_config(config=None, device='cpu'):
    """
    設定から新しいモデルを作成（事前学習なし）

    Args:
        config: モデル設定（Noneの場合はデフォルト設定を使用）
        device: デバイス

    Returns:
        model: 初期化されたモデル
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # neuroquantum_layered からモデルクラスをインポート
    neuroq_module_path = Path(__file__).parent / 'NeuroQ' / 'neuroq-runpod'
    if str(neuroq_module_path) not in sys.path:
        sys.path.insert(0, str(neuroq_module_path))

    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig

    model_config = NeuroQuantumConfig(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_len=config.get('max_seq_len', 256),
        dropout=config.get('dropout', 0.1),
        lambda_entangle=config.get('lambda_entangle', 0.5),
    )

    model = NeuroQuantum(model_config).to(device)
    model.eval()

    print(f"✅ New model created with config: {config}")

    return model


if __name__ == "__main__":
    """テスト用のエントリポイント"""
    print("=" * 60)
    print("🧠 NeuroQ Pretrained Model Loader - Test")
    print("=" * 60)

    # デバイスの決定
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("💻 Using CPU")

    # モデルをロード
    model, config, checkpoint = load_pretrained_model(device=device)

    if model is not None:
        print("\n✅ Test successful!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("\n❌ Test failed - model could not be loaded")
        print("   Falling back to default config...")
        model = create_model_from_config(device=device)

    print("=" * 60)
