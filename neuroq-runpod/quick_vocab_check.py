#!/usr/bin/env python3
"""
NeuroQ vocab_size 簡易チェッカー
================================
トークナイザーのvocab_sizeのみをチェック（学習なし）
"""

import os
import sys

try:
    import sentencepiece as spm

    print("=" * 70)
    print("🔍 NeuroQ vocab_size 簡易チェック")
    print("=" * 70)

    # トークナイザーファイルを探す
    tokenizer_paths = [
        "neuroq_tokenizer.model",
        "../neuroq_tokenizer.model",
        "neuroq_tokenizer_8k.model",
        "../neuroq_tokenizer_8k.model",
    ]

    print("\n📊 トークナイザーファイル:")
    print("-" * 70)

    found_tokenizers = []
    for path in tokenizer_paths:
        if os.path.exists(path):
            try:
                sp = spm.SentencePieceProcessor()
                sp.load(path)
                vocab_size = sp.get_piece_size()
                found_tokenizers.append((path, vocab_size))
                print(f"✅ {path}")
                print(f"   語彙サイズ: {vocab_size:,}")
                print(f"   特殊トークン:")
                print(f"   - <pad>: {sp.pad_id()}")
                print(f"   - <unk>: {sp.unk_id()}")
                print(f"   - <s>: {sp.bos_id()}")
                print(f"   - </s>: {sp.eos_id()}")
                print()
            except Exception as e:
                print(f"❌ {path}: {e}")

    if not found_tokenizers:
        print("❌ 有効なトークナイザーが見つかりません")
        sys.exit(1)

    # テストエンコード
    print("\n🧪 トークナイズテスト:")
    print("-" * 70)

    test_path, test_vocab_size = found_tokenizers[0]
    sp = spm.SentencePieceProcessor()
    sp.load(test_path)

    test_texts = [
        "量子コンピュータについて教えて",
        "人工知能が未来を変える",
        "こんにちは、今日も良い天気ですね",
    ]

    for text in test_texts:
        tokens = sp.encode(text, out_type=str)
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)

        print(f"\n入力: {text}")
        print(f"トークン: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"ID数: {len(ids)}")
        print(f"デコード: {decoded}")

    print("\n" + "=" * 70)
    print("✅ チェック完了")
    print("=" * 70)
    print(f"\n推奨vocab_size: {test_vocab_size:,}")
    print("モデルの設定で以下を使用してください:")
    print(f"  - NeuroQuantumConfig(vocab_size={test_vocab_size})")
    print(f"  - NeuroQuantumTokenizer(vocab_size={test_vocab_size}, model_file='{test_path}')")

except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
