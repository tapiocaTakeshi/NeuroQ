#!/usr/bin/env python3
"""
日本語トークナイザー学習スクリプト（fugashi/MeCab）
====================================================

vocab_size=8000 の日本語形態素解析トークナイザーを学習します。

使い方:
    python train_sentencepiece_tokenizer.py

生成されるファイル:
    - neuroq_tokenizer.json: 語彙ファイル（このファイルを使う）
"""

import os
import sys
import json
from typing import List
from collections import Counter

# fugashi
try:
    import fugashi
except ImportError:
    print("❌ fugashi がインストールされていません")
    print("   pip install fugashi unidic-lite を実行してください")
    sys.exit(1)

# データセット取得用
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️ datasets ライブラリがありません。組み込みデータを使用します。")
    print("   より良いトークナイザーを作るには: pip install datasets")


def fetch_training_corpus(max_samples: int = 10000) -> List[str]:
    """
    学習用コーパスを取得

    データソース:
    1. 日本語Wikipedia
    2. 日本語対話データ
    3. 英語データ（少量）
    4. 組み込みテキスト（フォールバック）

    Args:
        max_samples: 最大サンプル数

    Returns:
        テキストのリスト
    """
    texts = []

    if DATASETS_AVAILABLE:
        print("📥 Hugging Face からデータ取得中...")

        # 1. 日本語Wikipedia（メイン）
        try:
            print("   📚 日本語Wikipedia...")
            ds = load_dataset("range3/wiki40b-ja", split="train", streaming=True)
            count = 0
            for item in ds:
                if 'text' in item and len(item['text']) > 50:
                    texts.append(item['text'][:2000])  # 長さ制限
                    count += 1
                    if count >= max_samples // 2:
                        break
            print(f"      ✅ {count} サンプル取得")
        except Exception as e:
            print(f"      ⚠️ 失敗: {e}")

        # 2. 日本語対話データ
        try:
            print("   💬 日本語対話データ...")
            ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
            count = 0
            for item in ds[:max_samples // 4]:
                if 'output' in item and len(item['output']) > 20:
                    texts.append(item['output'])
                    count += 1
            print(f"      ✅ {count} サンプル取得")
        except Exception as e:
            print(f"      ⚠️ 失敗: {e}")

        # 3. 英語データ（少量）
        try:
            print("   📖 英語データ（少量）...")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            count = 0
            for item in ds[:max_samples // 4]:
                if 'text' in item and len(item['text']) > 30:
                    texts.append(item['text'])
                    count += 1
            print(f"      ✅ {count} サンプル取得")
        except Exception as e:
            print(f"      ⚠️ 失敗: {e}")

    # 4. 組み込みテキスト（フォールバック or 補強）
    builtin_texts = [
        # 量子コンピュータ
        "量子コンピュータは、量子力学の原理を利用して情報を処理する革新的な計算機です。",
        "量子ビット（キュービット）は、0と1の重ね合わせ状態を取ることができます。",
        "量子もつれ（エンタングルメント）により、複数の量子ビット間で相関が生じます。",
        "量子ゲートを用いて量子状態を操作し、並列計算を実現します。",

        # AI・機械学習
        "人工知能（AI）は、人間の知的活動をコンピュータで実現しようとする技術です。",
        "機械学習は、データから学習して改善するシステムを実現します。",
        "深層学習（ディープラーニング）は、多層のニューラルネットワークを用いた機械学習手法です。",
        "ニューラルネットワークは、人間の脳の神経回路を模倣した計算システムです。",
        "Transformerは、自然言語処理で広く使われるニューラルネットワークアーキテクチャです。",

        # ChatGPT・大規模言語モデル
        "ChatGPTは、OpenAIが開発した大規模言語モデルベースの対話型AIです。",
        "GPT（Generative Pre-trained Transformer）は、トランスフォーマーアーキテクチャを基盤とした言語モデルです。",
        "大規模言語モデル（LLM）は、膨大なテキストデータで事前学習されたニューラルネットワークです。",
        "自然言語処理（NLP）は、人間の言語をコンピュータで処理する技術分野です。",
        "テキスト生成、質問応答、要約、翻訳など、多様なタスクをこなせます。",

        # NeuroQuantum 関連
        "ニューロQは、量子ビットニューラルネットワークと大規模言語モデルを融合した革新的なAIです。",
        "QBNN（Quantum-Bit Neural Network）は、量子もつれの概念を取り入れたニューラルネットワークです。",
        "APQB理論（Adjustable Pseudo Quantum Bit）により、古典コンピュータで量子的な振る舞いを模倣します。",
        "エンタングル演算子により、層間の相関を制御し、表現力を高めます。",
    ]

    if len(texts) == 0:
        print("   📝 組み込みデータを使用...")
        texts = builtin_texts * (max_samples // len(builtin_texts))
    else:
        # 既存データに追加
        texts.extend(builtin_texts * 10)  # 重要なトピックなので繰り返す

    print(f"\n✅ 合計 {len(texts)} サンプル取得完了")
    return texts[:max_samples]


def train_japanese_tokenizer(
    texts: List[str],
    model_prefix: str = "neuroq_tokenizer",
    vocab_size: int = 8000,
    min_freq: int = 2,
):
    """
    fugashi（MeCab）日本語トークナイザーを学習

    Args:
        texts: 学習用テキストのリスト
        model_prefix: 出力モデルのプレフィックス（.json が生成される）
        vocab_size: 語彙サイズ（推奨: 8000〜32000）
        min_freq: 最小頻度
    """
    print(f"\n🔤 fugashi（MeCab）日本語トークナイザー学習開始...")
    print(f"   語彙サイズ: {vocab_size}")
    print(f"   最小頻度: {min_freq}")

    # 形態素解析
    tagger = fugashi.Tagger()
    token_freq = Counter()

    for text in texts:
        words = tagger(text.strip())
        token_freq.update(word.surface for word in words)

    print(f"   ユニーク形態素数: {len(token_freq):,}")

    # 語彙を構築
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>', '<USER>', '<ASSISTANT>']
    token_to_idx = {}
    for i, token in enumerate(special_tokens):
        token_to_idx[token] = i

    for token, freq in token_freq.most_common():
        if len(token_to_idx) >= vocab_size:
            break
        if freq >= min_freq and token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)

    actual_vocab_size = len(token_to_idx)

    # JSON保存
    vocab_file = f"{model_prefix}.json"
    vocab_data = {
        'tokenizer_type': 'fugashi',
        'token_to_idx': token_to_idx,
        'vocab_size': actual_vocab_size,
        'actual_vocab_size': actual_vocab_size,
    }
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 日本語トークナイザー学習完了！")
    print(f"   生成ファイル:")
    print(f"   - {vocab_file}  (語彙ファイル)")

    print(f"\n📊 トークナイザー情報:")
    print(f"   実際の語彙サイズ: {actual_vocab_size}")
    print(f"   特殊トークン:")
    print(f"   - <pad>: 0")
    print(f"   - <unk>: 1")
    print(f"   - <s>: 2")
    print(f"   - </s>: 3")

    # テスト
    idx_to_token = {i: t for t, i in token_to_idx.items()}
    test_texts = [
        "量子コンピュータについて教えて",
        "ChatGPTとは何ですか？",
        "ニューロQの特徴を説明してください",
    ]

    print(f"\n🧪 トークナイズテスト:")
    for text in test_texts:
        words = tagger(text)
        morphemes = [w.surface for w in words]
        ids = [token_to_idx.get(m, 1) for m in morphemes]
        print(f"   入力: {text}")
        print(f"   形態素: {morphemes}")
        print(f"   ID数: {len(ids)}")
        print()


def main():
    """メイン処理"""
    print("=" * 70)
    print("🚀 NeuroQ 日本語トークナイザー学習（fugashi/MeCab）")
    print("=" * 70)

    # 1. コーパス取得
    texts = fetch_training_corpus(max_samples=10000)

    # 2. トークナイザー学習
    train_japanese_tokenizer(
        texts=texts,
        model_prefix="neuroq_tokenizer",
        vocab_size=8000,
        min_freq=2,
    )

    print("\n" + "=" * 70)
    print("✅ 完了！")
    print("=" * 70)
    print("\n次のステップ:")
    print("1. neuroq_tokenizer.json を確認")
    print("2. handler.py を更新して、このモデルを使用するように設定")
    print("3. モデルを再学習して、新しい語彙サイズで動作確認")
    print()


if __name__ == "__main__":
    main()
