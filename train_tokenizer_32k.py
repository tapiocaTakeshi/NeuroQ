#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQ 用 fugashi 日本語形態素解析トークナイザー学習スクリプト

vocab_size = 32000 の高品質トークナイザーを学習します。

使用方法:
    python train_tokenizer_32k.py
"""

import os
import sys
import json
from collections import Counter

try:
    import fugashi
    FUGASHI_AVAILABLE = True
except ImportError:
    FUGASHI_AVAILABLE = False
    print("❌ fugashi がインストールされていません！")
    print("   pip install fugashi unidic-lite を実行してください。")
    sys.exit(1)


def train_japanese_tokenizer(
    input_file: str = "data/training_data.txt",
    model_prefix: str = "neuroq_tokenizer_32k",
    vocab_size: int = 32000,
):
    """
    fugashi 日本語形態素解析トークナイザーを学習

    Args:
        input_file: 学習データファイル
        model_prefix: モデルファイルのプレフィックス
        vocab_size: 語彙サイズ（32000推奨）
    """

    print("=" * 70)
    print("🔤 NeuroQ fugashi 日本語形態素解析トークナイザー学習")
    print("=" * 70)
    print(f"📂 入力ファイル: {input_file}")
    print(f"📊 語彙サイズ: {vocab_size:,}")
    print("-" * 70)

    # ファイルの存在確認
    if not os.path.exists(input_file):
        print(f"❌ エラー: {input_file} が見つかりません")
        sys.exit(1)

    # ファイルサイズを確認
    file_size = os.path.getsize(input_file)
    print(f"📦 データサイズ: {file_size:,} バイト ({file_size / 1024 / 1024:.2f} MB)")

    # fugashi で形態素解析し語彙を構築
    print("\n🚀 学習開始...")

    try:
        tagger = fugashi.Tagger()

        # 形態素の出現頻度をカウント
        morpheme_freq = Counter()

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = tagger(line)
                for word in words:
                    morpheme_freq[word.surface] += 1

        # 特殊トークン
        special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3,
        }

        # 頻度順にソートして語彙を構築
        sorted_morphemes = morpheme_freq.most_common(vocab_size - len(special_tokens))

        # 語彙辞書を作成
        vocab = dict(special_tokens)
        for idx, (morpheme, freq) in enumerate(sorted_morphemes):
            vocab[morpheme] = idx + len(special_tokens)

        # JSONファイルに保存
        model_file = f"{model_prefix}.json"
        vocab_data = {
            'vocab': vocab,
            'special_tokens': special_tokens,
            'vocab_size': len(vocab),
        }

        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        print("✅ 学習完了！")

    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)

    # モデルファイル確認
    if os.path.exists(model_file):
        model_size = os.path.getsize(model_file)
        print(f"\n📦 モデルファイル: {model_file} ({model_size:,} バイト)")
    else:
        print(f"\n❌ モデルファイル {model_file} が生成されませんでした")
        return

    actual_vocab_size = len(vocab)
    print(f"📖 語彙サイズ: {actual_vocab_size:,} エントリ")

    # モデルをテスト
    print("\n🧪 モデルテスト中...")
    tagger = fugashi.Tagger()

    # 保存した語彙を読み込み
    with open(model_file, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    loaded_vocab = loaded_data['vocab']

    print(f"   語彙サイズ: {len(loaded_vocab):,}")
    print(f"   PAD ID: {loaded_vocab.get('<pad>', 'N/A')}")
    print(f"   UNK ID: {loaded_vocab.get('<unk>', 'N/A')}")
    print(f"   BOS ID: {loaded_vocab.get('<s>', 'N/A')}")
    print(f"   EOS ID: {loaded_vocab.get('</s>', 'N/A')}")

    # サンプルテキストでテスト
    print("\n📝 サンプルテキストテスト:")
    print("-" * 70)

    # 逆引き辞書を作成
    id_to_token = {v: k for k, v in loaded_vocab.items()}
    unk_id = loaded_vocab.get('<unk>', 1)

    test_texts = [
        "量子コンピュータは革新的な技術です。",
        "人工知能が未来を変えていきます。",
        "こんにちは、今日も良い天気ですね。",
        "プログラミングは創造的な活動です。",
    ]

    for text in test_texts:
        # エンコード（形態素解析してID変換）
        words = tagger(text)
        pieces = [word.surface for word in words]
        encoded = [loaded_vocab.get(piece, unk_id) for piece in pieces]

        # デコード（IDからトークンに逆変換）
        decoded = ''.join([id_to_token.get(tid, '<unk>') for tid in encoded])

        print(f"\n原文: {text}")
        print(f"トークン数: {len(encoded)}")
        print(f"トークンID: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"デコード: {decoded}")
        print(f"トークン: {pieces}")

    print("\n" + "=" * 70)
    print("✅ トークナイザー学習完了！")
    print(f"   モデルファイル: {model_file}")
    print(f"   語彙サイズ: {actual_vocab_size:,}")
    print("=" * 70)

    return model_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NeuroQ fugashi 日本語形態素解析トークナイザー学習')
    parser.add_argument(
        '--input',
        type=str,
        default='data/training_data.txt',
        help='学習データファイル (デフォルト: data/training_data.txt)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='neuroq_tokenizer_32k',
        help='モデルプレフィックス (デフォルト: neuroq_tokenizer_32k)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=32000,
        help='語彙サイズ (デフォルト: 32000)'
    )

    args = parser.parse_args()

    train_japanese_tokenizer(
        input_file=args.input,
        model_prefix=args.prefix,
        vocab_size=args.vocab_size,
    )
