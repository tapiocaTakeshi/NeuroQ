#!/usr/bin/env python3
"""
NeuroQ vocab_size 整合性チェッカー
================================
モデルの各コンポーネントのvocab_sizeが一致しているかチェック
"""

import torch
import os
import sys

# neuroquantum_layered をインポート
sys.path.insert(0, os.path.dirname(__file__))

try:
    from neuroquantum_layered import NeuroQuantumAI, NeuroQuantumTokenizer
    import sentencepiece as spm

    print("=" * 70)
    print("🔍 NeuroQ vocab_size 整合性チェック")
    print("=" * 70)

    # 1. トークナイザーのvocab_sizeを確認
    print("\n1️⃣ トークナイザーのvocab_size:")
    print("-" * 70)

    tokenizer_paths = [
        "neuroq_tokenizer.model",
        "../neuroq_tokenizer.model",
        "neuroq_tokenizer_8k.model",
        "../neuroq_tokenizer_8k.model",
    ]

    tokenizer_vocab_size = None
    tokenizer_path = None

    for path in tokenizer_paths:
        if os.path.exists(path):
            try:
                sp = spm.SentencePieceProcessor()
                sp.load(path)
                tokenizer_vocab_size = sp.get_piece_size()
                tokenizer_path = path
                print(f"   ✅ トークナイザーファイル: {path}")
                print(f"   📊 語彙サイズ: {tokenizer_vocab_size:,}")
                break
            except Exception as e:
                print(f"   ❌ {path}: {e}")

    if tokenizer_vocab_size is None:
        print("   ❌ 有効なトークナイザーが見つかりません")
        sys.exit(1)

    # 2. NeuroQuantumAIを初期化して確認
    print("\n2️⃣ モデルのvocab_size:")
    print("-" * 70)

    ai = NeuroQuantumAI(embed_dim=64, num_heads=2, num_layers=2)

    # サンプルデータで学習（軽量）
    sample_texts = [
        "量子コンピュータは革新的です。",
        "人工知能が未来を変えます。",
    ] * 10

    ai.train(sample_texts, epochs=1, seq_len=16)

    # vocab_sizeを確認
    print(f"   📊 トークナイザーの実際のvocab_size: {ai.tokenizer.actual_vocab_size:,}")
    print(f"   📊 トークナイザーのvocab_size: {ai.tokenizer.vocab_size:,}")
    print(f"   📊 モデルconfig.vocab_size: {ai.config.vocab_size:,}")
    print(f"   📊 Embedding層のnum_embeddings: {ai.model.text_embedding.num_embeddings:,}")
    print(f"   📊 LM Head出力次元: {ai.model.output_head.out_features:,}")

    # 3. 整合性チェック
    print("\n3️⃣ 整合性チェック:")
    print("-" * 70)

    vocab_sizes = {
        "トークナイザー(actual)": ai.tokenizer.actual_vocab_size,
        "トークナイザー(設定)": ai.tokenizer.vocab_size,
        "モデルConfig": ai.config.vocab_size,
        "Embedding層": ai.model.text_embedding.num_embeddings,
        "LM Head": ai.model.output_head.out_features,
    }

    all_match = len(set(vocab_sizes.values())) == 1

    if all_match:
        print(f"   ✅ すべてのvocab_sizeが一致しています: {list(vocab_sizes.values())[0]:,}")
    else:
        print("   ❌ vocab_sizeに不一致があります:")
        for name, size in vocab_sizes.items():
            print(f"      {name}: {size:,}")

    print("\n" + "=" * 70)
    if all_match:
        print("✅ 整合性チェック: 合格")
    else:
        print("❌ 整合性チェック: 不合格")
    print("=" * 70)

except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
