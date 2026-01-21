#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQ 日本語トークナイザー学習スクリプト

日本語に最適化された SentencePiece トークナイザーを学習します。

特徴:
- 日本語テキストに最適化された設定
- 高い文字カバレッジ（0.9995）で漢字をカバー
- BPE (Byte Pair Encoding) による効率的なトークン化
- USER/ASSISTANT 特殊トークンをサポート

使用方法:
    # 基本（8K語彙）
    python train_japanese_tokenizer.py

    # カスタム設定
    python train_japanese_tokenizer.py --vocab-size 16000 --input data/japanese_training_data.txt

    # 大規模（32K語彙）
    python train_japanese_tokenizer.py --vocab-size 32000 --prefix neuroq_tokenizer_32k_ja

必要なライブラリ:
    pip install sentencepiece
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# SentencePiece
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("❌ sentencepiece がインストールされていません！")
    print("   pip install sentencepiece を実行してください。")
    sys.exit(1)


def train_japanese_tokenizer(
    input_file: str,
    model_prefix: str = "neuroq_tokenizer_ja",
    vocab_size: int = 8000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
    user_defined_symbols: Optional[List[str]] = None,
) -> str:
    """
    日本語に最適化された SentencePiece トークナイザーを学習

    Args:
        input_file: 学習データファイル
        model_prefix: モデルファイルのプレフィックス
        vocab_size: 語彙サイズ（推奨: 8000〜32000）
        character_coverage: 文字カバレッジ（日本語: 0.9995推奨）
        model_type: モデルタイプ（bpe or unigram）
        user_defined_symbols: ユーザー定義の特殊トークン

    Returns:
        生成されたモデルファイルのパス
    """
    print("=" * 70)
    print("🔤 NeuroQ 日本語トークナイザー学習")
    print("=" * 70)
    print(f"📂 入力ファイル: {input_file}")
    print(f"📊 語彙サイズ: {vocab_size:,}")
    print(f"🔧 モデルタイプ: {model_type}")
    print(f"📈 文字カバレッジ: {character_coverage}")
    print("-" * 70)

    # ファイルの存在確認
    if not os.path.exists(input_file):
        print(f"❌ エラー: {input_file} が見つかりません")
        print("   まず prepare_japanese_data.py を実行してください。")
        sys.exit(1)

    # ファイルサイズを確認
    file_size = os.path.getsize(input_file)
    print(f"📦 データサイズ: {file_size:,} バイト ({file_size / 1024 / 1024:.2f} MB)")

    # デフォルトの特殊トークン
    if user_defined_symbols is None:
        user_defined_symbols = ['<USER>', '<ASSISTANT>']

    print(f"🏷️  特殊トークン: {user_defined_symbols}")

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
            # ユーザー定義トークン
            user_defined_symbols=user_defined_symbols,
            # 日本語向け設定
            max_sentence_length=4192,  # 長い文に対応
            num_threads=os.cpu_count() or 4,  # 並列化
            # 正規化設定（日本語向け）
            normalization_rule_name='nmt_nfkc_cf',  # NFKC正規化 + Case folding
            remove_extra_whitespaces=True,
            add_dummy_prefix=True,  # 日本語では有効
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

    if not os.path.exists(model_file):
        print(f"\n❌ モデルファイル {model_file} が生成されませんでした")
        return ""

    model_size = os.path.getsize(model_file)
    print(f"\n📦 モデルファイル: {model_file} ({model_size:,} バイト)")

    if os.path.exists(vocab_file):
        vocab_size_actual = sum(1 for _ in open(vocab_file, 'r', encoding='utf-8'))
        print(f"📖 語彙ファイル: {vocab_file} ({vocab_size_actual:,} エントリ)")

    # モデルをテスト
    print("\n🧪 モデルテスト中...")
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)

    actual_vocab_size = sp.get_piece_size()
    print(f"   実際の語彙サイズ: {actual_vocab_size:,}")
    print(f"   PAD ID: {sp.pad_id()}")
    print(f"   UNK ID: {sp.unk_id()}")
    print(f"   BOS ID: {sp.bos_id()}")
    print(f"   EOS ID: {sp.eos_id()}")

    # 特殊トークンの確認
    print(f"\n🏷️  特殊トークンID:")
    for symbol in user_defined_symbols:
        token_id = sp.piece_to_id(symbol)
        print(f"   {symbol}: {token_id}")

    # サンプルテキストでテスト
    print("\n📝 サンプルテキストテスト:")
    print("-" * 70)

    test_texts = [
        "量子コンピュータは革新的な技術です。",
        "人工知能が未来を変えていきます。",
        "こんにちは、今日も良い天気ですね。",
        "プログラミングは創造的な活動です。",
        "<USER>量子コンピュータとは何ですか<ASSISTANT>量子力学を利用した計算機です。",
        "ニューロQは量子ビットニューラルネットワークを採用しています。",
        "深層学習とは、多層のニューラルネットワークを用いた機械学習手法です。",
    ]

    for text in test_texts:
        # エンコード
        encoded = sp.encode(text, out_type=int)
        # デコード
        decoded = sp.decode(encoded)

        print(f"\n原文: {text}")
        print(f"トークン数: {len(encoded)}")
        print(f"トークンID: {encoded[:15]}{'...' if len(encoded) > 15 else ''}")
        print(f"デコード: {decoded}")

        # トークン分割を表示
        pieces = sp.encode(text, out_type=str)
        print(f"トークン: {pieces[:15]}{'...' if len(pieces) > 15 else ''}")

    # 圧縮率の計算
    print("\n📊 圧縮率分析:")
    total_chars = sum(len(t) for t in test_texts)
    total_tokens = sum(len(sp.encode(t, out_type=int)) for t in test_texts)
    compression_ratio = total_chars / total_tokens
    print(f"   総文字数: {total_chars}")
    print(f"   総トークン数: {total_tokens}")
    print(f"   文字/トークン比: {compression_ratio:.2f}")

    print("\n" + "=" * 70)
    print("✅ 日本語トークナイザー学習完了！")
    print(f"   モデルファイル: {model_file}")
    print(f"   語彙ファイル: {vocab_file}")
    print(f"   語彙サイズ: {actual_vocab_size:,}")
    print("=" * 70)

    return model_file


def test_tokenizer(model_file: str) -> None:
    """
    学習済みトークナイザーの詳細テスト
    """
    if not os.path.exists(model_file):
        print(f"❌ モデルファイルが見つかりません: {model_file}")
        return

    print(f"\n🧪 トークナイザー詳細テスト: {model_file}")
    print("=" * 70)

    sp = spm.SentencePieceProcessor()
    sp.load(model_file)

    # 様々なテストケース
    test_cases = {
        "基本的な文": [
            "今日は良い天気です。",
            "明日は雨が降るかもしれません。",
            "東京タワーは高さ333メートルです。",
        ],
        "技術用語": [
            "ニューラルネットワークの学習率を調整する。",
            "GPUを使用して並列処理を行う。",
            "トランスフォーマーモデルのアテンション機構。",
        ],
        "対話形式": [
            "<USER>プログラミングを教えてください<ASSISTANT>はい、どの言語から始めますか？",
            "<USER>Pythonとは<ASSISTANT>Pythonは汎用プログラミング言語です。",
        ],
        "長い文": [
            "人工知能の発展は目覚ましく、特に深層学習の分野では、画像認識、音声認識、自然言語処理など、多くの応用が実現されています。",
        ],
        "混合テキスト": [
            "ChatGPTはOpenAIが開発したAIです。",
            "Python3.9以降が推奨されています。",
            "APIキーを環境変数に設定してください。",
        ],
    }

    for category, texts in test_cases.items():
        print(f"\n📁 {category}:")
        print("-" * 50)
        for text in texts:
            tokens = sp.encode(text, out_type=str)
            ids = sp.encode(text, out_type=int)
            decoded = sp.decode(ids)

            print(f"入力: {text}")
            print(f"トークン数: {len(tokens)}")
            print(f"トークン: {tokens}")
            print(f"復元: {decoded}")
            print(f"完全復元: {'✅' if decoded == text else '⚠️ ' + repr(decoded)}")
            print()

    # UNKトークンのテスト
    print("\n🔍 UNKトークンテスト:")
    unk_test = "🚀💻🎉"  # 絵文字
    unk_tokens = sp.encode(unk_test, out_type=str)
    print(f"入力 (絵文字): {unk_test}")
    print(f"トークン: {unk_tokens}")


def main():
    parser = argparse.ArgumentParser(
        description='NeuroQ 日本語トークナイザー学習'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/japanese_training_data.txt',
        help='学習データファイル (デフォルト: data/japanese_training_data.txt)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='neuroq_tokenizer_ja',
        help='モデルプレフィックス (デフォルト: neuroq_tokenizer_ja)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=8000,
        help='語彙サイズ (デフォルト: 8000)'
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
    parser.add_argument(
        '--test-only',
        type=str,
        default=None,
        help='既存モデルをテストのみ（モデルファイルパス指定）'
    )

    args = parser.parse_args()

    # テストのみモード
    if args.test_only:
        test_tokenizer(args.test_only)
        return

    # 入力ファイルの確認
    if not os.path.exists(args.input):
        # フォールバック: 既存の training_data.txt を使用
        fallback_inputs = [
            'data/japanese_training_corpus.txt',
            'data/training_data.txt',
        ]
        for fallback in fallback_inputs:
            if os.path.exists(fallback):
                print(f"⚠️  {args.input} が見つかりません。{fallback} を使用します。")
                args.input = fallback
                break
        else:
            print(f"❌ 学習データファイルが見つかりません。")
            print("   以下のコマンドでデータを準備してください:")
            print("   python prepare_japanese_data.py")
            sys.exit(1)

    # トークナイザー学習
    model_file = train_japanese_tokenizer(
        input_file=args.input,
        model_prefix=args.prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.coverage,
        model_type=args.model_type,
    )

    # 詳細テスト
    if model_file:
        test_tokenizer(model_file)


if __name__ == '__main__':
    main()
