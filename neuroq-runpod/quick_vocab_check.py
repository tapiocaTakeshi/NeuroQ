#!/usr/bin/env python3
"""
NeuroQ vocab_size 簡易チェッカー
================================
トークナイザーのvocab_sizeのみをチェック（学習なし）
"""

import os
import sys

try:
    import json

    print("=" * 70)
    print("🔍 NeuroQ vocab_size 簡易チェック")
    print("=" * 70)

    # トークナイザーファイルを探す
    tokenizer_paths = [
        "neuroq_tokenizer.json",
        "../neuroq_tokenizer.json",
        "neuroq_tokenizer_8k.json",
        "../neuroq_tokenizer_8k.json",
    ]

    print("\n📊 トークナイザーファイル:")
    print("-" * 70)

    found_tokenizers = []
    for path in tokenizer_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                vocab = vocab_data.get('vocab', vocab_data)
                vocab_size = len(vocab)
                found_tokenizers.append((path, vocab_size))
                print(f"✅ {path}")
                print(f"   語彙サイズ: {vocab_size:,}")
                special_tokens = vocab_data.get('special_tokens', {})
                if special_tokens:
                    print(f"   特殊トークン:")
                    for name, tid in special_tokens.items():
                        print(f"   - {name}: {tid}")
                print()
            except Exception as e:
                print(f"❌ {path}: {e}")

    if not found_tokenizers:
        print("❌ 有効なトークナイザーが見つかりません")
        sys.exit(1)

    # テスト情報表示
    print("\n🧪 語彙ファイル情報:")
    print("-" * 70)

    test_path, test_vocab_size = found_tokenizers[0]
    with open(test_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    vocab = vocab_data.get('vocab', vocab_data)

    # 語彙の先頭を表示
    vocab_items = list(vocab.items()) if isinstance(vocab, dict) else list(enumerate(vocab))
    print(f"\n語彙の先頭10件:")
    for key, val in vocab_items[:10]:
        print(f"   {key}: {val}")

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
