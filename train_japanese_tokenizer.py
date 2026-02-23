#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQ 日本語トークナイザー学習スクリプト

fugashi（MeCab）を使用した日本語形態素解析トークナイザーを学習します。

特徴:
- MeCab形態素解析による高精度な日本語トークン化
- 頻度ベースの語彙構築
- USER/ASSISTANT 特殊トークンをサポート

使用方法:
    # 基本（8K語彙）
    python train_japanese_tokenizer.py

    # カスタム設定
    python train_japanese_tokenizer.py --vocab-size 16000 --input data/japanese_training_data.txt

    # 大規模（32K語彙）
    python train_japanese_tokenizer.py --vocab-size 32000 --prefix neuroq_tokenizer_32k_ja

必要なライブラリ:
    pip install fugashi unidic-lite
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional
from collections import Counter

# fugashi
try:
    import fugashi
    FUGASHI_AVAILABLE = True
except ImportError:
    FUGASHI_AVAILABLE = False
    print("❌ fugashi がインストールされていません！")
    print("   pip install fugashi unidic-lite を実行してください。")
    sys.exit(1)


def train_japanese_tokenizer(
    input_file: str,
    model_prefix: str = "neuroq_tokenizer_ja",
    vocab_size: int = 8000,
    min_freq: int = 2,
    user_defined_symbols: Optional[List[str]] = None,
) -> str:
    """
    fugashi（MeCab）を使用した日本語トークナイザーを学習

    Args:
        input_file: 学習データファイル
        model_prefix: モデルファイルのプレフィックス
        vocab_size: 語彙サイズ（推奨: 8000〜32000）
        min_freq: 最小頻度
        user_defined_symbols: ユーザー定義の特殊トークン

    Returns:
        生成されたモデルファイルのパス
    """
    print("=" * 70)
    print("🔤 NeuroQ 日本語トークナイザー学習（fugashi/MeCab）")
    print("=" * 70)
    print(f"📂 入力ファイル: {input_file}")
    print(f"📊 語彙サイズ: {vocab_size:,}")
    print(f"🔧 最小頻度: {min_freq}")
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

    # テキストを読み込み
    print("\n📖 テキスト読み込み中...")
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"   {len(texts):,} 行読み込み完了")

    # fugashi Tagger初期化
    print("\n🚀 形態素解析開始...")
    tagger = fugashi.Tagger()

    # 形態素の頻度をカウント
    token_freq = Counter()
    for text in texts:
        words = tagger(text)
        token_freq.update(word.surface for word in words)

    print(f"   ユニーク形態素数: {len(token_freq):,}")

    # 語彙を構築
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>'] + user_defined_symbols
    token_to_idx = {}
    for i, token in enumerate(special_tokens):
        token_to_idx[token] = i

    # 頻度順にトークンを追加
    for token, freq in token_freq.most_common():
        if len(token_to_idx) >= vocab_size:
            break
        if freq >= min_freq and token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)

    actual_vocab_size = len(token_to_idx)

    # JSON形式で保存
    vocab_file = f"{model_prefix}.json"
    vocab_data = {
        'tokenizer_type': 'fugashi',
        'token_to_idx': token_to_idx,
        'vocab_size': actual_vocab_size,
        'actual_vocab_size': actual_vocab_size,
    }
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f"\n📦 語彙ファイル: {vocab_file}")
    print(f"📊 実際の語彙サイズ: {actual_vocab_size:,}")
    print(f"   PAD ID: 0")
    print(f"   UNK ID: 1")
    print(f"   BOS ID: 2")
    print(f"   EOS ID: 3")

    # 特殊トークンの確認
    print(f"\n🏷️  特殊トークンID:")
    for symbol in user_defined_symbols:
        token_id = token_to_idx.get(symbol, -1)
        print(f"   {symbol}: {token_id}")

    # サンプルテキストでテスト
    print("\n📝 サンプルテキストテスト:")
    print("-" * 70)

    idx_to_token = {i: t for t, i in token_to_idx.items()}

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
        words = tagger(text)
        morphemes = [w.surface for w in words]
        encoded = [token_to_idx.get(m, 1) for m in morphemes]  # 1 = UNK

        # デコード
        decoded = ''.join(idx_to_token.get(i, '<unk>') for i in encoded)

        print(f"\n原文: {text}")
        print(f"トークン数: {len(encoded)}")
        print(f"トークンID: {encoded[:15]}{'...' if len(encoded) > 15 else ''}")
        print(f"デコード: {decoded}")
        print(f"形態素: {morphemes[:15]}{'...' if len(morphemes) > 15 else ''}")

    # 圧縮率の計算
    print("\n📊 圧縮率分析:")
    total_chars = sum(len(t) for t in test_texts)
    total_tokens = 0
    for t in test_texts:
        words = tagger(t)
        total_tokens += len(words)
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    print(f"   総文字数: {total_chars}")
    print(f"   総トークン数: {total_tokens}")
    print(f"   文字/トークン比: {compression_ratio:.2f}")

    print("\n" + "=" * 70)
    print("✅ 日本語トークナイザー学習完了！")
    print(f"   語彙ファイル: {vocab_file}")
    print(f"   語彙サイズ: {actual_vocab_size:,}")
    print("=" * 70)

    return vocab_file


def test_tokenizer(model_file: str) -> None:
    """
    学習済みトークナイザーの詳細テスト
    """
    if not os.path.exists(model_file):
        print(f"❌ モデルファイルが見つかりません: {model_file}")
        return

    print(f"\n🧪 トークナイザー詳細テスト: {model_file}")
    print("=" * 70)

    # 語彙を読み込み
    with open(model_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    token_to_idx = data['token_to_idx']
    idx_to_token = {int(i): t for t, i in token_to_idx.items()}

    tagger = fugashi.Tagger()

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
            words = tagger(text)
            morphemes = [w.surface for w in words]
            ids = [token_to_idx.get(m, 1) for m in morphemes]
            decoded = ''.join(idx_to_token.get(i, '<unk>') for i in ids)

            print(f"入力: {text}")
            print(f"トークン数: {len(morphemes)}")
            print(f"形態素: {morphemes}")
            print(f"復元: {decoded}")
            print(f"完全復元: {'✅' if decoded == text else '⚠️ ' + repr(decoded)}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='NeuroQ 日本語トークナイザー学習（fugashi/MeCab）'
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
        '--min-freq',
        type=int,
        default=2,
        help='最小頻度 (デフォルト: 2)'
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
        min_freq=args.min_freq,
    )

    # 詳細テスト
    if model_file:
        test_tokenizer(model_file)


if __name__ == '__main__':
    main()
