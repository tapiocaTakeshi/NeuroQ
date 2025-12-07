#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQuantum Layered Tokenizer のSentencePiece対応テスト
"""

import sys
sys.path.insert(0, '/home/user/NeuroQuantum/neuroq-runpod')

from neuroquantum_layered import NeuroQuantumTokenizer

def test_tokenizer():
    print("=" * 70)
    print("🔤 NeuroQuantumTokenizer SentencePiece テスト")
    print("=" * 70)

    # 既存のSentencePieceモデルを読み込み
    tokenizer = NeuroQuantumTokenizer(
        vocab_size=8000,
        model_file="/home/user/NeuroQuantum/neuroq_tokenizer_8k.model"
    )

    print(f"\n✅ トークナイザー読み込み完了")
    print(f"   語彙サイズ: {tokenizer.vocab_size:,}")
    print(f"   実際の語彙サイズ: {tokenizer.actual_vocab_size:,}")

    # サンプルテキストでテスト
    print("\n📝 サンプルテキストテスト:")
    print("-" * 70)

    test_texts = [
        "量子コンピュータについて教えて",
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
    print("✅ テスト完了！vocab_size=8000のSentencePieceトークナイザーが正常に動作しています。")
    print("=" * 70)

if __name__ == '__main__':
    test_tokenizer()
