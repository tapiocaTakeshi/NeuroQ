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


def get_tokenizer_path():
    """
    学習済みfugashiトークナイザーのパスを取得

    Returns:
        str: トークナイザーモデルファイルのパス
    """
    # このファイルの親ディレクトリ（/home/user/）から探す
    current_dir = Path(__file__).parent

    # 候補パスをチェック（SentencePiece .model 形式を優先）
    candidate_paths = [
        current_dir / 'neuroq_tokenizer.model',
        current_dir / 'neuroq_tokenizer_8k.model',
        current_dir / 'neuroq_tokenizer_ja.model',
        current_dir / 'neuroq_tokenizer_oasst1_ja.model',
        current_dir / 'NeuroQ' / 'neuroq_tokenizer.model',
        current_dir / 'NeuroQ' / 'neuroq_tokenizer_8k.model',
        # JSON フォールバック
        current_dir / 'neuroq_tokenizer.json',
        current_dir / 'NeuroQ' / 'neuroq_tokenizer.json',
    ]

    for path in candidate_paths:
        if path.exists():
            return str(path)

    # 見つからない場合はデフォルトパスを返す
    return str(current_dir / 'neuroq_tokenizer.model')


def load_pretrained_tokenizer(tokenizer_path=None, verbose=True):
    """
    学習済みfugashiトークナイザーをロード

    Args:
        tokenizer_path: トークナイザーファイルのパス（Noneの場合は自動検出）
        verbose: 詳細ログを表示するか

    Returns:
        tokenizer: NeuroQuantumTokenizer または None if failed
    """
    try:
        # パスの決定
        if tokenizer_path is None:
            tokenizer_path = get_tokenizer_path()

        if verbose:
            print(f"📦 Loading pretrained tokenizer from: {tokenizer_path}")

        # ファイルの存在確認
        if not os.path.exists(tokenizer_path):
            if verbose:
                print(f"❌ Tokenizer file not found: {tokenizer_path}")
                print(f"")
                print(f"   学習済みトークナイザーが見つかりません。")
                print(f"   以下のいずれかを実行してください:")
                print(f"")
                print(f"   1. トークナイザーを学習:")
                print(f"      $ python train_tokenizer_32k.py")
                print(f"")
                print(f"   2. 既存のトークナイザーファイルを配置:")
                print(f"      neuroq_tokenizer.json を {Path(__file__).parent} に配置")
                print(f"")
            return None

        # ファイルサイズ確認
        file_size = os.path.getsize(tokenizer_path)
        if verbose:
            print(f"📊 File size: {file_size / 1024:.2f} KB")

        # neuroquantum_layered からトークナイザークラスをインポート
        neuroq_module_path = Path(__file__).parent / 'NeuroQ' / 'neuroq-runpod'
        if str(neuroq_module_path) not in sys.path:
            sys.path.insert(0, str(neuroq_module_path))

        from neuroquantum_layered import NeuroQuantumTokenizer

        # トークナイザーをロード
        tokenizer = NeuroQuantumTokenizer(model_file=tokenizer_path)

        if verbose:
            print(f"✅ Tokenizer loaded successfully")
            print(f"   Vocabulary size: {tokenizer.vocab_size}")

        return tokenizer

    except Exception as e:
        if verbose:
            print(f"❌ Error loading tokenizer: {e}")
            import traceback
            traceback.print_exc()
        return None


def create_model_from_config(config=None, tokenizer=None, device='cpu'):
    """
    設定から新しいモデルを作成（重みは初期化のみ）

    Args:
        config: モデル設定（Noneの場合はデフォルト設定を使用）
        tokenizer: NeuroQuantumTokenizer（Noneの場合は自動ロード）
        device: デバイス

    Returns:
        tuple: (model, tokenizer)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # トークナイザーをロードまたは使用
    if tokenizer is None:
        tokenizer = load_pretrained_tokenizer(verbose=True)
        if tokenizer is None:
            print("⚠️ トークナイザーをロードできませんでした。デフォルトのトークナイザーを使用します。")
            # フォールバック
            from neuroquantum_layered import NeuroQuantumTokenizer
            tokenizer = NeuroQuantumTokenizer(vocab_size=config['vocab_size'])

    # トークナイザーの語彙サイズに合わせて設定を更新
    config['vocab_size'] = tokenizer.vocab_size

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

    print(f"✅ 新しいモデルを作成しました")
    print(f"   vocab_size: {config['vocab_size']}")
    print(f"   embed_dim: {config['embed_dim']}")
    print(f"   hidden_dim: {config['hidden_dim']}")

    return model, tokenizer


if __name__ == "__main__":
    """テスト用のエントリポイント"""
    print("=" * 60)
    print("🧠 NeuroQ Tokenizer Loader - Test")
    print("=" * 60)

    # デバイスの決定
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("💻 Using CPU")

    # トークナイザーをロード
    print("\n📦 トークナイザーをロード中...")
    tokenizer = load_pretrained_tokenizer()

    if tokenizer is not None:
        print("\n✅ Tokenizer test successful!")
        print(f"   Vocabulary size: {tokenizer.vocab_size}")

        # テストエンコード/デコード
        test_text = "こんにちは、NeuroQです。"
        encoded = tokenizer.encode(test_text, add_special=False)
        decoded = tokenizer.decode(encoded, skip_special=True)
        print(f"\n   テストテキスト: {test_text}")
        print(f"   エンコード: {encoded[:10]}... ({len(encoded)} tokens)")
        print(f"   デコード: {decoded}")

        # モデルを作成
        print("\n🧠 モデルを作成中...")
        model, tokenizer = create_model_from_config(tokenizer=tokenizer, device=device)
        print(f"   Model type: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("\n❌ Tokenizer test failed")
        print("   デフォルト設定でモデルを作成...")
        model, tokenizer = create_model_from_config(device=device)

    print("=" * 60)
