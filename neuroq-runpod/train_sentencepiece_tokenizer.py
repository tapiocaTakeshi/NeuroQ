#!/usr/bin/env python3
"""
日本語トークナイザー学習スクリプト（SentencePiece）
====================================================

vocab_size=8000 の日本語サブワードトークナイザーを学習します。

使い方:
    python train_sentencepiece_tokenizer.py

生成されるファイル:
    - neuroq_tokenizer.model: SentencePieceモデル（このファイルを使う）
    - neuroq_tokenizer.vocab: 語彙ファイル（参照用）
"""

import os
import sys
import tempfile
from typing import List

# sentencepiece
try:
    import sentencepiece as spm
except ImportError:
    print("❌ sentencepiece がインストールされていません")
    print("   pip install sentencepiece を実行してください")
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
    character_coverage: float = 0.9995,
):
    """
    SentencePiece 日本語トークナイザーを学習

    Args:
        texts: 学習用テキストのリスト
        model_prefix: 出力モデルのプレフィックス（.model と .vocab が生成される）
        vocab_size: 語彙サイズ（推奨: 8000〜32000）
        character_coverage: 文字カバレッジ（日本語は0.9995推奨）
    """
    print(f"\n🔤 SentencePiece 日本語トークナイザー学習開始...")
    print(f"   語彙サイズ: {vocab_size}")
    print(f"   文字カバレッジ: {character_coverage}")

    # 学習テキストを一時ファイルに書き出し
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for text in texts:
            text = text.strip()
            if text:
                f.write(text + '\n')
        tmp_corpus_path = f.name

    try:
        # SentencePieceモデルを学習
        spm.SentencePieceTrainer.Train(
            input=tmp_corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='unigram',
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            user_defined_symbols=['<USER>', '<ASSISTANT>', '<SYSTEM>'],
            train_extremely_large_corpus=False,
        )

        print(f"\n✅ SentencePieceトークナイザー学習完了！")
        print(f"   生成ファイル:")
        print(f"   - {model_prefix}.model  (モデルファイル)")
        print(f"   - {model_prefix}.vocab  (語彙ファイル)")

        # モデルを読み込んでテスト
        sp = spm.SentencePieceProcessor()
        sp.Load(f"{model_prefix}.model")

        actual_vocab_size = sp.GetPieceSize()

        print(f"\n📊 トークナイザー情報:")
        print(f"   実際の語彙サイズ: {actual_vocab_size}")
        print(f"   特殊トークン:")
        print(f"   - <pad>: {sp.PieceToId('<pad>')}")
        print(f"   - <unk>: {sp.PieceToId('<unk>')}")
        print(f"   - <s>: {sp.PieceToId('<s>')}")
        print(f"   - </s>: {sp.PieceToId('</s>')}")

        # テスト
        test_texts = [
            "量子コンピュータについて教えて",
            "ChatGPTとは何ですか？",
            "ニューロQの特徴を説明してください",
        ]

        print(f"\n🧪 トークナイズテスト:")
        for text in test_texts:
            pieces = sp.EncodeAsPieces(text)
            ids = sp.EncodeAsIds(text)
            decoded = sp.DecodeIds(ids)
            print(f"   入力: {text}")
            print(f"   サブワード: {pieces}")
            print(f"   ID数: {len(ids)}")
            print(f"   復元: {decoded}")
            print()

    finally:
        os.unlink(tmp_corpus_path)


def main():
    """メイン処理"""
    print("=" * 70)
    print("🚀 NeuroQ 日本語トークナイザー学習（SentencePiece）")
    print("=" * 70)

    # 1. コーパス取得
    texts = fetch_training_corpus(max_samples=10000)

    # 2. トークナイザー学習
    train_japanese_tokenizer(
        texts=texts,
        model_prefix="neuroq_tokenizer",
        vocab_size=8000,
        character_coverage=0.9995,
    )

    print("\n" + "=" * 70)
    print("✅ 完了！")
    print("=" * 70)
    print("\n次のステップ:")
    print("1. neuroq_tokenizer.model を確認")
    print("2. handler.py の TOKENIZER_MODEL_PATH が 'neuroq_tokenizer.model' になっていることを確認")
    print("3. モデルを再学習して、新しい語彙サイズで動作確認")
    print()


if __name__ == "__main__":
    main()
