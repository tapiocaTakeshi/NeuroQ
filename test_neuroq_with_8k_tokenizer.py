#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQ Brain を vocab_size=8000 トークナイザーでテスト

改善を確認します：
- 元の vocab ~300 → 新しい vocab 8000（27倍改善）
- トークナイザーの品質向上
- テキスト生成の改善
"""

import torch
import sys
import os

# neuroquantum_brain.py から必要なクラスをインポート
from neuroquantum_brain import (
    NeuroQuantumBrainAI,
    BrainTokenizer,
)


def test_tokenizer():
    """トークナイザー単体テスト"""
    print("=" * 70)
    print("🔤 トークナイザー テスト")
    print("=" * 70)

    # 学習済みトークナイザーをロード
    tokenizer = BrainTokenizer(
        vocab_size=8000,
        model_file="neuroq_tokenizer_8k.model"
    )

    print(f"✅ トークナイザー読み込み完了")
    print(f"   語彙サイズ: {tokenizer.vocab_size:,}")
    print(f"   実際の語彙サイズ: {tokenizer.actual_vocab_size:,}")

    # サンプルテキストでテスト
    print("\n📝 サンプルテキストテスト:")
    print("-" * 70)

    test_texts = [
        "量子コンピュータは革新的な技術です。",
        "人工知能が未来を変えていきます。",
        "こんにちは、今日も良い天気ですね。",
        "プログラミングは創造的な活動です。",
        "深層学習は複雑な問題を解決します。",
    ]

    for text in test_texts:
        # エンコード
        tokens = tokenizer.encode(text)

        # デコード
        decoded = tokenizer.decode(tokens)

        print(f"\n原文: {text}")
        print(f"トークン数: {len(tokens)}")
        print(f"トークンID: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        print(f"デコード: {decoded}")

    print("\n" + "=" * 70)


def test_neuroq_brain_small():
    """NeuroQ Brain で小規模テスト（軽量版）"""
    print("\n" + "=" * 70)
    print("🧠 NeuroQ Brain 小規模テスト")
    print("=" * 70)

    # 学習データ（少量でテスト）
    texts = [
        "量子コンピュータは革新的な技術です。",
        "人工知能が未来を変えていきます。",
        "こんにちは、今日も良い天気ですね。",
        "プログラミングは創造的な活動です。",
        "深層学習は複雑な問題を解決します。",
        "ニューラルネットワークは脳を模倣します。",
        "機械学習はデータから学習します。",
        "自然言語処理はテキストを理解します。",
    ] * 50  # 400サンプル

    print(f"📚 学習データ: {len(texts)} サンプル")

    # AI構築（軽量設定）
    ai = NeuroQuantumBrainAI(
        embed_dim=64,      # 小さく設定
        num_heads=2,       # 小さく設定
        num_layers=2,      # 小さく設定
        num_neurons=30,    # 小さく設定
        max_vocab=8000,    # トークナイザーに合わせる
    )

    # トークナイザーを手動設定（学習済みモデルを使用）
    ai.tokenizer = BrainTokenizer(
        vocab_size=8000,
        model_file="neuroq_tokenizer_8k.model"
    )

    print(f"✅ AI構築完了")
    print(f"   語彙サイズ: {ai.tokenizer.vocab_size:,}")

    # 軽量学習（エポック数を減らす）
    print("\n🎓 学習開始...")
    ai.train(
        texts=texts,
        epochs=10,          # 少なく設定
        batch_size=8,       # 小さく設定
        lr=0.001,
        seq_length=32,      # 短く設定
    )

    print("\n✅ 学習完了！")

    # テキスト生成テスト
    print("\n📝 テキスト生成テスト:")
    print("-" * 70)

    prompts = [
        "量子コンピュータ",
        "人工知能",
        "プログラミング",
        "こんにちは",
    ]

    for prompt in prompts:
        generated = ai.generate(
            prompt,
            max_length=30,
            temperature_min=0.5,
            temperature_max=0.8
        )
        print(f"\nプロンプト: {prompt}")
        print(f"生成結果: {generated}")

    print("\n" + "=" * 70)
    print("✅ テスト完了！")


def main():
    """メイン処理"""
    print("\n" + "=" * 70)
    print("🚀 NeuroQ Brain with vocab_size=8000 テスト")
    print("=" * 70)

    # トークナイザーファイルの存在確認
    if not os.path.exists("neuroq_tokenizer_8k.model"):
        print("❌ エラー: neuroq_tokenizer_8k.model が見つかりません")
        print("   python train_tokenizer_32k.py --vocab-size 8000 --prefix neuroq_tokenizer_8k を実行してください")
        sys.exit(1)

    # 1. トークナイザー単体テスト
    test_tokenizer()

    # 2. NeuroQ Brain 小規模テスト
    response = input("\n\nNeuroQ Brain の学習テストを実行しますか？ (y/n): ").strip().lower()
    if response == 'y':
        test_neuroq_brain_small()
    else:
        print("\n👋 テスト終了")


if __name__ == '__main__':
    main()
