#!/usr/bin/env python3
"""
会話データで学習
==================
conversation_training_data.txt を使って会話能力を向上させる
"""

import sys
import torch
from pathlib import Path

# パスを追加
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from neuroquantum_layered import NeuroQuantumAI

def load_conversation_data(file_path: str = "conversation_training_data.txt"):
    """
    会話データを読み込む

    Returns:
        List[str]: 会話データのリスト（各要素が1つの会話ターン）
    """
    print(f"📖 会話データ読み込み: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1行ずつが1つの会話ペア
    # 改行で分割して空行を除外
    conversations = [line.strip() for line in content.split('\n') if line.strip()]

    print(f"✅ {len(conversations)} 個の会話ペアを読み込みました")

    # サンプルを表示
    print(f"\n📝 サンプル（最初の3個）:")
    for i, conv in enumerate(conversations[:3], 1):
        # 最初の100文字のみ表示
        preview = conv[:100] + "..." if len(conv) > 100 else conv
        print(f"   {i}. {preview}")

    return conversations


def train_conversation_model(
    conversation_data,
    tokenizer_save_path: str = "neuroq",
    epochs: int = 10,
    batch_size: int = 16,
    seq_len: int = 128
):
    """
    会話データでモデルを学習

    Args:
        conversation_data: 会話データのリスト
        tokenizer_save_path: トークナイザー保存パス（プレフィックス）
        epochs: エポック数
        batch_size: バッチサイズ
        seq_len: シーケンス長
    """
    print("\n" + "=" * 60)
    print("🚀 会話モデル学習開始")
    print("=" * 60)

    # モデル初期化
    print("\n📦 モデル初期化...")
    print(f"   新規モデルを作成")
    model = NeuroQuantumAI(
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=3,
        max_seq_len=256,
        dropout=0.1,
        lambda_entangle=0.5
    )

    print(f"\n📚 学習設定:")
    print(f"   エポック数: {epochs}")
    print(f"   バッチサイズ: {batch_size}")
    print(f"   シーケンス長: {seq_len}")
    print(f"   学習データ数: {len(conversation_data)}")

    # 学習実行
    print(f"\n🔄 学習開始...\n")
    model.train(
        texts=conversation_data,
        epochs=epochs,
        batch_size=batch_size,
        seq_len=seq_len
    )

    # トークナイザー保存
    print(f"\n💾 トークナイザー保存...")
    model.save_tokenizer(tokenizer_save_path)

    print("\n" + "=" * 60)
    print("✅ 学習完了！")
    print("=" * 60)
    print(f"\n注意: モデルの重みは保存していません。")
    print(f"      トークナイザーのみを保存しています。")

    # 簡易テスト
    print(f"\n🧪 簡易テスト:")
    test_prompts = [
        "<USER>こんにちは<ASSISTANT>",
        "<USER>人工知能とは？<ASSISTANT>",
        "<USER>ありがとう<ASSISTANT>"
    ]

    for prompt in test_prompts:
        print(f"\n   入力: {prompt}")
        response = model.generate(
            prompt=prompt,
            max_length=50,
            temp_min=0.4,
            temp_max=0.7
        )
        print(f"   出力: {response}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("💬 会話データ学習スクリプト")
    print("=" * 60)

    # 会話データ読み込み
    conversation_data = load_conversation_data("conversation_training_data.txt")

    # 学習実行
    train_conversation_model(
        conversation_data=conversation_data,
        epochs=10,           # エポック数（必要に応じて調整）
        batch_size=16,       # バッチサイズ
        seq_len=128          # シーケンス長
    )

    print("\n🎉 すべて完了！")
