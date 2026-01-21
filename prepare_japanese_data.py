#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQ 日本語学習データ準備スクリプト（Hugging Face版）

Hugging Face Datasets から日本語コーパスを取得し、
トークナイザーとモデル学習用のデータを準備します。

機能:
1. 日本語Wikipedia データ取得
2. 日本語対話データ取得
3. 日本語指示データ取得
4. データクリーニングと前処理
5. 統合データセット作成

使用方法:
    python prepare_japanese_data.py --output data/japanese_training_data.txt --max-samples 50000

必要なライブラリ:
    pip install datasets tqdm
"""

import os
import sys
import re
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict

# tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm がインストールされていません。pip install tqdm を実行してください。")
    def tqdm(x, **kwargs):
        return x

# Hugging Face datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("❌ datasets がインストールされていません！")
    print("   pip install datasets を実行してください。")
    sys.exit(1)


def clean_text(text: str) -> str:
    """
    テキストクリーニング

    - 余分な空白を削除
    - 制御文字を除去
    - URLを除去
    - 連続する改行を調整
    """
    if not text or not isinstance(text, str):
        return ""

    # URL除去
    text = re.sub(r'https?://[^\s]+', '', text)

    # 制御文字除去（改行、タブ以外）
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # HTML タグ除去
    text = re.sub(r'<[^>]+>', '', text)

    # 余分な空白を整理
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 行頭・行末の空白を削除
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join([line for line in lines if line])

    return text.strip()


def fetch_wikipedia_japanese(max_samples: int = 10000) -> List[str]:
    """
    日本語Wikipedia データを Hugging Face から取得

    データソース: wikipedia (ja)
    """
    texts = []

    print(f"\n📖 日本語Wikipediaデータ取得中... (目標: {max_samples})")

    try:
        # Wikipedia 日本語
        dataset = load_dataset(
            "wikipedia",
            "20220301.ja",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        count = 0
        for item in tqdm(dataset, desc="   Wikipedia", total=max_samples):
            if count >= max_samples:
                break

            text = item.get('text', '')
            if text and len(text) > 100:
                # 長いテキストは分割
                cleaned = clean_text(text)
                if len(cleaned) > 50:
                    # 適切な長さに分割（最大2000文字）
                    if len(cleaned) > 2000:
                        # 段落で分割
                        paragraphs = cleaned.split('\n\n')
                        current_chunk = ""
                        for para in paragraphs:
                            if len(current_chunk) + len(para) < 2000:
                                current_chunk += para + "\n\n"
                            else:
                                if current_chunk.strip():
                                    texts.append(current_chunk.strip())
                                current_chunk = para + "\n\n"
                        if current_chunk.strip():
                            texts.append(current_chunk.strip())
                    else:
                        texts.append(cleaned)
                    count += 1

        print(f"   ✅ {len(texts)} テキストを取得")

    except Exception as e:
        print(f"   ⚠️  Wikipedia取得エラー: {e}")
        print("   代替データソースを試みます...")

        try:
            # 代替: wiki40b-ja
            dataset = load_dataset(
                "range3/wiki40b-ja",
                split="train",
                streaming=True
            )

            count = 0
            for item in tqdm(dataset, desc="   Wiki40b-ja", total=max_samples):
                if count >= max_samples:
                    break

                text = item.get('text', '')
                if text and len(text) > 100:
                    cleaned = clean_text(text[:2000])
                    if len(cleaned) > 50:
                        texts.append(cleaned)
                        count += 1

            print(f"   ✅ {len(texts)} テキストを取得 (wiki40b-ja)")

        except Exception as e2:
            print(f"   ❌ 代替ソースも失敗: {e2}")

    return texts


def fetch_japanese_dialogue(max_samples: int = 5000) -> List[str]:
    """
    日本語対話データを Hugging Face から取得

    データソース:
    - kunishou/databricks-dolly-15k-ja
    - izumi-lab/llm-japanese-dataset
    """
    texts = []

    print(f"\n💬 日本語対話データ取得中... (目標: {max_samples})")

    # 1. Dolly 日本語版
    try:
        print("   📥 kunishou/databricks-dolly-15k-ja を取得...")
        dataset = load_dataset(
            "kunishou/databricks-dolly-15k-ja",
            split="train"
        )

        count = 0
        for item in tqdm(dataset, desc="   Dolly-ja"):
            if count >= max_samples // 2:
                break

            instruction = item.get('instruction', '')
            output = item.get('output', '')

            if instruction and output:
                # USER/ASSISTANT フォーマットで追加
                user_text = clean_text(instruction)
                assistant_text = clean_text(output)

                if len(user_text) > 10 and len(assistant_text) > 10:
                    texts.append(f"<USER>{user_text}<ASSISTANT>{assistant_text}")
                    count += 1

        print(f"   ✅ Dolly-ja から {count} 対話を取得")

    except Exception as e:
        print(f"   ⚠️  Dolly-ja 取得エラー: {e}")

    # 2. LLM Japanese Dataset
    try:
        print("   📥 izumi-lab/llm-japanese-dataset を取得...")
        dataset = load_dataset(
            "izumi-lab/llm-japanese-dataset",
            split="train",
            streaming=True
        )

        count = 0
        for item in tqdm(dataset, desc="   LLM-ja", total=max_samples // 2):
            if count >= max_samples // 2:
                break

            text = item.get('text', '')
            if text and len(text) > 50:
                cleaned = clean_text(text[:1500])
                if len(cleaned) > 30:
                    texts.append(cleaned)
                    count += 1

        print(f"   ✅ LLM-ja から {count} テキストを取得")

    except Exception as e:
        print(f"   ⚠️  LLM-ja 取得エラー: {e}")

    # 3. JGLUE (JCommonsenseQA)
    try:
        print("   📥 shunk031/JGLUE を取得...")
        dataset = load_dataset(
            "shunk031/JGLUE",
            "JCommonsenseQA",
            split="train",
            trust_remote_code=True
        )

        count = 0
        for item in tqdm(dataset, desc="   JGLUE"):
            if count >= max_samples // 4:
                break

            question = item.get('question', '')
            choices = item.get('choice0', ''), item.get('choice1', ''), item.get('choice2', ''), item.get('choice3', ''), item.get('choice4', '')
            label = item.get('label', 0)

            if question:
                answer = choices[label] if 0 <= label < len(choices) else ""
                if answer:
                    q_text = clean_text(question)
                    a_text = clean_text(answer)
                    if q_text and a_text:
                        texts.append(f"<USER>{q_text}<ASSISTANT>{a_text}")
                        count += 1

        print(f"   ✅ JGLUE から {count} QAを取得")

    except Exception as e:
        print(f"   ⚠️  JGLUE 取得エラー: {e}")

    return texts


def fetch_japanese_instruction(max_samples: int = 5000) -> List[str]:
    """
    日本語指示データを Hugging Face から取得

    データソース:
    - llm-jp/databricks-dolly-15k-ja
    - kunishou/oasst1-89k-ja
    """
    texts = []

    print(f"\n📝 日本語指示データ取得中... (目標: {max_samples})")

    # 1. OASST1 日本語版
    try:
        print("   📥 kunishou/oasst1-89k-ja を取得...")
        dataset = load_dataset(
            "kunishou/oasst1-89k-ja",
            split="train",
            streaming=True
        )

        count = 0
        for item in tqdm(dataset, desc="   OASST1-ja", total=max_samples):
            if count >= max_samples:
                break

            text = item.get('text', '')
            if text and len(text) > 50:
                cleaned = clean_text(text[:2000])
                if len(cleaned) > 30:
                    texts.append(cleaned)
                    count += 1

        print(f"   ✅ OASST1-ja から {count} テキストを取得")

    except Exception as e:
        print(f"   ⚠️  OASST1-ja 取得エラー: {e}")

    # 2. ichikara-instruction
    try:
        print("   📥 p1atdev/ichikara-instruction を取得...")
        dataset = load_dataset(
            "p1atdev/ichikara-instruction",
            split="train",
            streaming=True
        )

        count = 0
        for item in tqdm(dataset, desc="   Ichikara", total=max_samples // 2):
            if count >= max_samples // 2:
                break

            instruction = item.get('instruction', '') or item.get('text', '')
            output = item.get('output', '') or item.get('response', '')

            if instruction:
                i_text = clean_text(instruction)
                if output:
                    o_text = clean_text(output)
                    if i_text and o_text:
                        texts.append(f"<USER>{i_text}<ASSISTANT>{o_text}")
                        count += 1
                elif len(i_text) > 30:
                    texts.append(i_text)
                    count += 1

        print(f"   ✅ Ichikara から {count} テキストを取得")

    except Exception as e:
        print(f"   ⚠️  Ichikara 取得エラー: {e}")

    return texts


def fetch_japanese_cc100(max_samples: int = 10000) -> List[str]:
    """
    CC100 日本語データを取得

    データソース: cc100 (ja)
    """
    texts = []

    print(f"\n📰 CC100日本語データ取得中... (目標: {max_samples})")

    try:
        dataset = load_dataset(
            "cc100",
            lang="ja",
            split="train",
            streaming=True
        )

        count = 0
        for item in tqdm(dataset, desc="   CC100-ja", total=max_samples):
            if count >= max_samples:
                break

            text = item.get('text', '')
            if text and len(text) > 100:
                cleaned = clean_text(text[:1500])
                if len(cleaned) > 50:
                    texts.append(cleaned)
                    count += 1

        print(f"   ✅ CC100 から {len(texts)} テキストを取得")

    except Exception as e:
        print(f"   ⚠️  CC100 取得エラー: {e}")

    return texts


def fetch_japanese_oscar(max_samples: int = 10000) -> List[str]:
    """
    OSCAR 日本語データを取得

    データソース: oscar
    """
    texts = []

    print(f"\n🌐 OSCAR日本語データ取得中... (目標: {max_samples})")

    try:
        dataset = load_dataset(
            "oscar",
            "unshuffled_deduplicated_ja",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        count = 0
        for item in tqdm(dataset, desc="   OSCAR-ja", total=max_samples):
            if count >= max_samples:
                break

            text = item.get('text', '')
            if text and len(text) > 100:
                cleaned = clean_text(text[:1500])
                if len(cleaned) > 50:
                    texts.append(cleaned)
                    count += 1

        print(f"   ✅ OSCAR から {len(texts)} テキストを取得")

    except Exception as e:
        print(f"   ⚠️  OSCAR 取得エラー: {e}")

    return texts


def create_builtin_dialogue_data() -> List[str]:
    """
    組み込みの対話データを生成（Hugging Face データのフォールバック/補強用）
    """
    texts = []

    print("\n📝 組み込み対話データを追加...")

    # 基本的なQA
    qa_pairs = [
        ("こんにちは", "こんにちは！何かお手伝いできることはありますか？"),
        ("あなたは誰ですか", "私はニューロQという名前のAIアシスタントです。量子ビットニューラルネットワーク技術を基盤にしています。"),
        ("ありがとう", "どういたしまして！お役に立てて嬉しいです。"),
        ("さようなら", "さようなら！また何かあればいつでもお声がけください。"),
        ("プログラミングを教えて", "プログラミングは、コンピュータに指示を与えるための言語を使ってソフトウェアを作成することです。Pythonは初心者におすすめの言語で、文法がシンプルで学びやすいです。"),
        ("機械学習とは何ですか", "機械学習は、データからパターンを学習するAI技術です。明示的にプログラムしなくても、経験から学んで改善できます。"),
        ("量子コンピュータについて教えて", "量子コンピュータは量子力学の原理を利用した計算機です。量子ビットは0と1の重ね合わせ状態を取ることができ、特定の問題で従来のコンピュータより高速に計算できる可能性があります。"),
        ("深層学習とは", "深層学習は、多層のニューラルネットワークを使った機械学習手法です。画像認識、自然言語処理、音声認識など様々な分野で優れた性能を発揮します。"),
        ("自然言語処理とは", "自然言語処理（NLP）は、コンピュータが人間の言語を理解し生成するための技術分野です。機械翻訳、チャットボット、文章要約、感情分析などに応用されています。"),
        ("ニューラルネットワークとは", "ニューラルネットワークは、人間の脳の神経回路を模倣した計算モデルです。入力層、隠れ層、出力層から構成され、重みの調整によって学習を行います。"),
    ]

    for user, assistant in qa_pairs:
        texts.append(f"<USER>{user}<ASSISTANT>{assistant}")

    # 技術的な説明
    tech_texts = [
        "Transformerは自己注意機構を用いた深層学習アーキテクチャで、自然言語処理に革命をもたらしました。",
        "GPT（Generative Pre-trained Transformer）は大規模なテキストデータで事前学習された言語モデルです。",
        "BERT（Bidirectional Encoder Representations from Transformers）は双方向の文脈を考慮した言語モデルです。",
        "アテンション機構は、入力の特定の部分に注目することで、長い文脈も効率的に処理できます。",
        "勾配降下法は、損失関数を最小化するために重みを更新する最適化手法です。",
        "バッチ正規化は、各層の入力を正規化することで学習を安定させる技術です。",
        "ドロップアウトは、学習時にランダムにニューロンを無効化することで過学習を防ぎます。",
        "転移学習は、ある問題で学習した知識を別の問題に活用する手法です。",
    ]
    texts.extend(tech_texts)

    print(f"   ✅ {len(texts)} 件の組み込みデータを追加")

    return texts


def prepare_japanese_training_data(
    output_path: str,
    max_samples: int = 50000,
    include_wikipedia: bool = True,
    include_dialogue: bool = True,
    include_instruction: bool = True,
    include_cc100: bool = True,
    include_oscar: bool = False,  # OSCAR は大きいのでデフォルトはオフ
) -> None:
    """
    日本語学習データを準備してファイルに保存

    Args:
        output_path: 出力ファイルパス
        max_samples: 最大サンプル数
        include_wikipedia: Wikipedia を含める
        include_dialogue: 対話データを含める
        include_instruction: 指示データを含める
        include_cc100: CC100 を含める
        include_oscar: OSCAR を含める
    """
    print("=" * 70)
    print("🔧 NeuroQ 日本語学習データ準備 (Hugging Face版)")
    print("=" * 70)
    print(f"📊 目標サンプル数: {max_samples:,}")
    print(f"📂 出力ファイル: {output_path}")
    print("-" * 70)

    all_texts = []
    samples_per_source = max_samples // 4  # 各ソースに均等に割り当て

    # 1. Wikipedia
    if include_wikipedia:
        wiki_texts = fetch_wikipedia_japanese(samples_per_source)
        all_texts.extend(wiki_texts)

    # 2. 対話データ
    if include_dialogue:
        dialogue_texts = fetch_japanese_dialogue(samples_per_source)
        all_texts.extend(dialogue_texts)

    # 3. 指示データ
    if include_instruction:
        instruction_texts = fetch_japanese_instruction(samples_per_source)
        all_texts.extend(instruction_texts)

    # 4. CC100
    if include_cc100:
        cc100_texts = fetch_japanese_cc100(samples_per_source)
        all_texts.extend(cc100_texts)

    # 5. OSCAR（オプション）
    if include_oscar:
        oscar_texts = fetch_japanese_oscar(samples_per_source)
        all_texts.extend(oscar_texts)

    # 6. 組み込み対話データ（補強）
    builtin_texts = create_builtin_dialogue_data()
    all_texts.extend(builtin_texts * 10)  # 重要なので繰り返す

    print(f"\n📊 データ統計:")
    print(f"   取得テキスト数: {len(all_texts):,}")

    # 重複除去
    print("\n🧹 重複除去中...")
    unique_texts = list(set(all_texts))
    print(f"   重複除去後: {len(unique_texts):,} テキスト")

    # 短すぎるテキストを除外
    filtered_texts = [t for t in unique_texts if len(t) >= 20]
    print(f"   フィルタ後: {len(filtered_texts):,} テキスト")

    total_chars = sum(len(text) for text in filtered_texts)
    print(f"   総文字数: {total_chars:,}")

    # ファイルに保存
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 データ保存中: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in filtered_texts:
            f.write(text + '\n\n')

    file_size = os.path.getsize(output_path)
    print(f"\n✅ 完了!")
    print(f"   出力ファイル: {output_path}")
    print(f"   ファイルサイズ: {file_size:,} バイト ({file_size / 1024 / 1024:.2f} MB)")
    print(f"   総テキスト数: {len(filtered_texts):,}")
    print(f"   総文字数: {total_chars:,}")
    print(f"   平均テキスト長: {total_chars / len(filtered_texts):.1f} 文字")

    # 統計情報も保存
    stats_path = output_path.replace('.txt', '_stats.json')
    stats = {
        'total_texts': len(filtered_texts),
        'total_chars': total_chars,
        'avg_text_length': total_chars / len(filtered_texts) if filtered_texts else 0,
        'file_size_bytes': file_size,
        'sources': {
            'wikipedia': include_wikipedia,
            'dialogue': include_dialogue,
            'instruction': include_instruction,
            'cc100': include_cc100,
            'oscar': include_oscar,
        }
    }

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"   統計情報: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description='NeuroQ 日本語学習データ準備 (Hugging Face版)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/japanese_training_data.txt',
        help='出力ファイルパス (デフォルト: data/japanese_training_data.txt)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50000,
        help='最大サンプル数 (デフォルト: 50000)'
    )
    parser.add_argument(
        '--no-wikipedia',
        action='store_true',
        help='Wikipediaを除外'
    )
    parser.add_argument(
        '--no-dialogue',
        action='store_true',
        help='対話データを除外'
    )
    parser.add_argument(
        '--no-instruction',
        action='store_true',
        help='指示データを除外'
    )
    parser.add_argument(
        '--no-cc100',
        action='store_true',
        help='CC100を除外'
    )
    parser.add_argument(
        '--include-oscar',
        action='store_true',
        help='OSCARを含める（大容量）'
    )

    args = parser.parse_args()

    prepare_japanese_training_data(
        output_path=args.output,
        max_samples=args.max_samples,
        include_wikipedia=not args.no_wikipedia,
        include_dialogue=not args.no_dialogue,
        include_instruction=not args.no_instruction,
        include_cc100=not args.no_cc100,
        include_oscar=args.include_oscar,
    )


if __name__ == '__main__':
    main()
