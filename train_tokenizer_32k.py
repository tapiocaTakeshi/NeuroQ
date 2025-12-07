#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQ 用 SentencePiece トークナイザー学習スクリプト

vocab_size = 32000 の高品質トークナイザーを学習します。

使用方法:
    python train_tokenizer_32k.py
"""

import os
import sys

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("❌ sentencepiece がインストールされていません！")
    print("   pip install sentencepiece を実行してください。")
    sys.exit(1)


def train_sentencepiece_tokenizer(
    input_file: str = "data/training_data.txt",
    model_prefix: str = "neuroq_tokenizer_32k",
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
):
    """
    SentencePiece トークナイザーを学習

    Args:
        input_file: 学習データファイル
        model_prefix: モデルファイルのプレフィックス
        vocab_size: 語彙サイズ（32000推奨）
        character_coverage: 文字カバレッジ（日本語では0.9995推奨）
        model_type: モデルタイプ（bpe or unigram）
    """

    print("=" * 70)
    print("🔤 NeuroQ SentencePiece トークナイザー学習")
    print("=" * 70)
    print(f"📂 入力ファイル: {input_file}")
    print(f"📊 語彙サイズ: {vocab_size:,}")
    print(f"🔧 モデルタイプ: {model_type}")
    print(f"📈 文字カバレッジ: {character_coverage}")
    print("-" * 70)

    # ファイルの存在確認
    if not os.path.exists(input_file):
        print(f"❌ エラー: {input_file} が見つかりません")
        sys.exit(1)

    # ファイルサイズを確認
    file_size = os.path.getsize(input_file)
    print(f"📦 データサイズ: {file_size:,} バイト ({file_size / 1024 / 1024:.2f} MB)")

    # SentencePiece 学習
    print("\n🚀 学習開始...")

    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            # 特殊トークン設定
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            # 追加設定
            user_defined_symbols=[],
            max_sentence_length=4192,  # 長い文に対応
            num_threads=os.cpu_count(),  # 並列化
            train_extremely_large_corpus=False,
            # ログ設定
            minloglevel=1,  # INFO レベル
        )
        print("✅ 学習完了！")

    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)

    # モデルファイル確認
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"

    if os.path.exists(model_file):
        model_size = os.path.getsize(model_file)
        print(f"\n📦 モデルファイル: {model_file} ({model_size:,} バイト)")
    else:
        print(f"\n❌ モデルファイル {model_file} が生成されませんでした")
        return

    if os.path.exists(vocab_file):
        vocab_size_actual = sum(1 for _ in open(vocab_file, 'r', encoding='utf-8'))
        print(f"📖 語彙ファイル: {vocab_file} ({vocab_size_actual:,} エントリ)")

    # モデルをテスト
    print("\n🧪 モデルテスト中...")
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)

    actual_vocab_size = sp.get_piece_size()
    print(f"   語彙サイズ: {actual_vocab_size:,}")
    print(f"   PAD ID: {sp.pad_id()}")
    print(f"   UNK ID: {sp.unk_id()}")
    print(f"   BOS ID: {sp.bos_id()}")
    print(f"   EOS ID: {sp.eos_id()}")

    # サンプルテキストでテスト
    print("\n📝 サンプルテキストテスト:")
    print("-" * 70)

    test_texts = [
        "量子コンピュータは革新的な技術です。",
        "人工知能が未来を変えていきます。",
        "こんにちは、今日も良い天気ですね。",
        "プログラミングは創造的な活動です。",
    ]

    for text in test_texts:
        # エンコード
        encoded = sp.encode(text, out_type=int)
        # デコード
        decoded = sp.decode(encoded)

        print(f"\n原文: {text}")
        print(f"トークン数: {len(encoded)}")
        print(f"トークンID: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"デコード: {decoded}")

        # トークン分割を表示
        pieces = sp.encode(text, out_type=str)
        print(f"トークン: {pieces}")

    print("\n" + "=" * 70)
    print("✅ トークナイザー学習完了！")
    print(f"   モデルファイル: {model_file}")
    print(f"   語彙サイズ: {actual_vocab_size:,}")
    print("=" * 70)

    return model_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NeuroQ SentencePiece トークナイザー学習')
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
    parser.add_argument(
        '--coverage',
        type=float,
        default=0.9995,
        help='文字カバレッジ (デフォルト: 0.9995)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='bpe',
        choices=['bpe', 'unigram'],
        help='モデルタイプ (デフォルト: bpe)'
    )

    args = parser.parse_args()

    train_sentencepiece_tokenizer(
        input_file=args.input,
        model_prefix=args.prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.coverage,
        model_type=args.model_type,
    )
