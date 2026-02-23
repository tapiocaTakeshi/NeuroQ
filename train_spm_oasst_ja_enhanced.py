#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQ SentencePiece トークナイザー学習（日本語 Open Assistant 強化版）
=====================================================================

日本語 Open Assistant (OASST1/OASST2) データを使用して
SentencePiece Unigram トークナイザーを学習します。

既存コーパス（data/combined_oasst_ja_corpus.txt）が存在する場合はそれを利用し、
存在しない場合は Hugging Face から自動取得します。

機能:
  - OASST1 日本語 + OASST2 日本語フィルタリング
  - 対話形式の特殊トークン対応（<USER>, <ASSISTANT>, <SYSTEM>）
  - 学習品質のバリデーション
  - 既存トークナイザーとの比較レポート

使い方:
    python train_spm_oasst_ja_enhanced.py
    python train_spm_oasst_ja_enhanced.py --vocab-size 16000
    python train_spm_oasst_ja_enhanced.py --corpus data/combined_oasst_ja_corpus.txt
"""

import os
import sys
import re
import argparse
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import sentencepiece as spm
except ImportError:
    print("sentencepiece がインストールされていません。")
    print("  pip install sentencepiece")
    sys.exit(1)

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ---------------------------------------------------------------------------
# テキスト前処理
# ---------------------------------------------------------------------------

_RE_URL = re.compile(r'https?://\S+')
_RE_HTML = re.compile(r'<[^>]+>')
_RE_CTRL = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]')

def _contains_japanese(text: str) -> bool:
    """テキストに日本語文字が含まれているかチェック"""
    for c in text:
        if ('\u3040' <= c <= '\u30FF' or   # ひらがな・カタカナ
            '\u4E00' <= c <= '\u9FFF' or   # 漢字 (CJK統合漢字)
            '\u31F0' <= c <= '\u31FF' or   # カタカナ拡張
            '\uFF65' <= c <= '\uFF9F'):    # 半角カタカナ
            return True
    return False


def clean_text(text: str) -> str:
    """テキストのクリーニング"""
    if not text:
        return ""
    text = _RE_URL.sub('', text)
    text = _RE_HTML.sub('', text)
    text = _RE_CTRL.sub('', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# コーパス取得
# ---------------------------------------------------------------------------

def load_existing_corpus(path: str) -> List[str]:
    """既存のコーパスファイルを読み込む"""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line) > 10:
                texts.append(line)
    return texts


def fetch_oasst_japanese(max_samples: int = 50000) -> List[str]:
    """Hugging Face から OASST 日本語データを取得"""
    if not DATASETS_AVAILABLE:
        print("datasets ライブラリが必要です: pip install datasets")
        return []

    texts = []

    # OASST1 日本語版
    print("  OASST1 日本語 (kunishou/oasst1-89k-ja) を取得中...")
    try:
        ds = load_dataset("kunishou/oasst1-89k-ja", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="  OASST1-ja", total=max_samples):
            if count >= max_samples:
                break
            text = item.get('text', '') or item.get('text_ja', '')
            if text and _contains_japanese(text):
                cleaned = clean_text(text)
                if len(cleaned) > 20:
                    texts.append(cleaned)
                    count += 1
        print(f"    {count} サンプル取得")
    except Exception as e:
        print(f"    OASST1 取得失敗: {e}")

    # OASST2 (日本語フィルタリング)
    print("  OASST2 (OpenAssistant/oasst2) から日本語を取得中...")
    try:
        ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="  OASST2-ja", total=max_samples):
            if count >= max_samples // 2:
                break
            text = item.get('text', '')
            if text and _contains_japanese(text):
                cleaned = clean_text(text)
                if len(cleaned) > 20:
                    texts.append(cleaned)
                    count += 1
        print(f"    {count} サンプル取得")
    except Exception as e:
        print(f"    OASST2 取得失敗: {e}")

    return texts


def prepare_corpus(corpus_path: Optional[str] = None,
                   max_samples: int = 50000) -> Tuple[str, int]:
    """
    学習用コーパスを準備する。

    1. corpus_path が指定されていればそれを使用
    2. data/combined_oasst_ja_corpus.txt が存在すればそれを使用
    3. いずれもなければ Hugging Face から取得

    Returns:
        (コーパスファイルパス, テキスト行数)
    """
    default_corpus = "data/combined_oasst_ja_corpus.txt"

    # 既存ファイルの利用
    if corpus_path and os.path.exists(corpus_path):
        lines = sum(1 for _ in open(corpus_path, encoding='utf-8'))
        print(f"既存コーパスを使用: {corpus_path} ({lines} 行)")
        return corpus_path, lines

    if os.path.exists(default_corpus):
        lines = sum(1 for _ in open(default_corpus, encoding='utf-8'))
        print(f"既存コーパスを使用: {default_corpus} ({lines} 行)")
        return default_corpus, lines

    # Hugging Face から取得
    print("Open Assistant 日本語データを取得します...")
    texts = fetch_oasst_japanese(max_samples)

    if not texts:
        print("コーパスが取得できませんでした。")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)
    out_path = default_corpus
    with open(out_path, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(t.replace('\n', ' ') + '\n')

    print(f"コーパスを保存: {out_path} ({len(texts)} 行)")
    return out_path, len(texts)


# ---------------------------------------------------------------------------
# SentencePiece 学習
# ---------------------------------------------------------------------------

def train_tokenizer(corpus_path: str,
                    model_prefix: str = "neuroq_tokenizer_oasst1_ja",
                    vocab_size: int = 8000,
                    character_coverage: float = 0.9995) -> str:
    """
    SentencePiece Unigram トークナイザーを学習する。

    Returns:
        学習済みモデルのパス (.model)
    """
    print(f"\nSentencePiece 学習開始")
    print(f"  モデル種別    : unigram")
    print(f"  語彙サイズ    : {vocab_size}")
    print(f"  文字カバレッジ: {character_coverage}")
    print(f"  コーパス      : {corpus_path}")
    print(f"  出力先        : {model_prefix}.model / .vocab")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        # 特殊トークンID (NeuroQ 統一仕様)
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        # 対話用特殊トークン
        user_defined_symbols=["<USER>", "<ASSISTANT>", "<SYSTEM>"],
        # 学習パラメータ
        max_sentence_length=16384,
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=False,
        # サブワード正則化 (学習の頑健性向上)
        input_sentence_size=200000,
        shuffle_input_sentence=True,
    )

    model_path = f"{model_prefix}.model"
    print(f"\n学習完了: {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------

def validate_tokenizer(model_path: str) -> dict:
    """学習済みトークナイザーの品質を検証する"""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    actual_vocab = sp.GetPieceSize()
    print(f"\n--- トークナイザー検証 ---")
    print(f"語彙サイズ: {actual_vocab}")

    # 特殊トークン確認
    special = {
        '<pad>': sp.PieceToId('<pad>'),
        '<unk>': sp.PieceToId('<unk>'),
        '<s>':   sp.PieceToId('<s>'),
        '</s>':  sp.PieceToId('</s>'),
        '<USER>': sp.PieceToId('<USER>'),
        '<ASSISTANT>': sp.PieceToId('<ASSISTANT>'),
        '<SYSTEM>': sp.PieceToId('<SYSTEM>'),
    }
    print("特殊トークン:")
    for name, tid in special.items():
        status = "OK" if tid >= 0 else "MISSING"
        print(f"  {name:15s} -> ID {tid:5d}  [{status}]")

    # テストテキストでの動作確認
    test_texts = [
        "こんにちは、量子コンピュータについて教えてください。",
        "日本語の自然言語処理は、形態素解析が重要です。",
        "SentencePieceはサブワード分割を行うトークナイザーです。",
        "NeuroQは量子ビットニューラルネットワーク言語モデルです。",
        "<USER>AIについて教えて<ASSISTANT>AIは人工知能の略称です。",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print("\nトークナイズテスト:")
    total_tokens = 0
    total_chars = 0
    for text in test_texts:
        pieces = sp.EncodeAsPieces(text)
        ids = sp.EncodeAsIds(text)
        decoded = sp.DecodeIds(ids)
        roundtrip_ok = decoded == text

        total_tokens += len(ids)
        total_chars += len(text)

        print(f"  入力 : {text[:60]}{'...' if len(text)>60 else ''}")
        print(f"  分割 : {' '.join(pieces[:15])}{'...' if len(pieces)>15 else ''}")
        print(f"  ID数 : {len(ids)}")
        print(f"  復元 : {'OK' if roundtrip_ok else 'NG (' + decoded[:60] + ')'}")
        print()

    compression = total_chars / total_tokens if total_tokens > 0 else 0
    print(f"平均圧縮率: {compression:.2f} 文字/トークン")

    return {
        'vocab_size': actual_vocab,
        'special_tokens': special,
        'compression_ratio': compression,
    }


def compare_with_existing(new_model: str, existing_models: List[str]):
    """既存トークナイザーとの比較"""
    test_text = "量子コンピュータの基本原理について説明してください。人工知能との関係も教えてください。"

    sp_new = spm.SentencePieceProcessor()
    sp_new.Load(new_model)

    print(f"\n--- 既存トークナイザーとの比較 ---")
    print(f"テスト文: {test_text}\n")

    new_ids = sp_new.EncodeAsIds(test_text)
    print(f"  [NEW] {new_model}")
    print(f"    語彙: {sp_new.GetPieceSize()}, トークン数: {len(new_ids)}")
    print(f"    分割: {' '.join(sp_new.EncodeAsPieces(test_text))}")
    print()

    for path in existing_models:
        if os.path.exists(path) and path != new_model:
            try:
                sp_old = spm.SentencePieceProcessor()
                sp_old.Load(path)
                old_ids = sp_old.EncodeAsIds(test_text)
                print(f"  [OLD] {path}")
                print(f"    語彙: {sp_old.GetPieceSize()}, トークン数: {len(old_ids)}")
                print(f"    分割: {' '.join(sp_old.EncodeAsPieces(test_text))}")
                print()
            except Exception as e:
                print(f"  [OLD] {path}: 読み込み失敗 ({e})")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SentencePiece 日本語トークナイザー学習 (Open Assistant 強化版)')
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='語彙サイズ (デフォルト: 8000)')
    parser.add_argument('--corpus', type=str, default=None,
                        help='コーパスファイルパス (省略時は自動検出/取得)')
    parser.add_argument('--model-prefix', type=str, default='neuroq_tokenizer_oasst1_ja',
                        help='出力モデルのプレフィックス')
    parser.add_argument('--coverage', type=float, default=0.9995,
                        help='文字カバレッジ (デフォルト: 0.9995)')
    parser.add_argument('--max-samples', type=int, default=50000,
                        help='Hugging Face から取得する最大サンプル数')
    parser.add_argument('--skip-compare', action='store_true',
                        help='既存トークナイザーとの比較をスキップ')

    args = parser.parse_args()

    print("=" * 70)
    print("  NeuroQ SentencePiece 学習 (日本語 Open Assistant)")
    print("=" * 70)

    # 1. コーパス準備
    corpus_path, n_lines = prepare_corpus(args.corpus, args.max_samples)

    # 2. トークナイザー学習
    model_path = train_tokenizer(
        corpus_path=corpus_path,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.coverage,
    )

    # 3. バリデーション
    stats = validate_tokenizer(model_path)

    # 4. 既存モデルとの比較
    if not args.skip_compare:
        existing = [
            "neuroq_tokenizer.model",
            "neuroq_tokenizer_8k.model",
            "neuroq_tokenizer_ja.model",
            "neuroq-runpod/neuroq_tokenizer.model",
        ]
        compare_with_existing(model_path, existing)

    # 5. neuroq-runpod ディレクトリへコピー
    import shutil
    runpod_dir = Path("neuroq-runpod")
    if runpod_dir.exists():
        dest = runpod_dir / f"{args.model_prefix}.model"
        shutil.copy2(model_path, dest)
        vocab_src = f"{args.model_prefix}.vocab"
        if os.path.exists(vocab_src):
            shutil.copy2(vocab_src, runpod_dir / f"{args.model_prefix}.vocab")
        print(f"\nneuroq-runpod/ にもコピーしました: {dest}")

    print("\n" + "=" * 70)
    print("  完了")
    print("=" * 70)
    print(f"\n生成ファイル:")
    print(f"  {args.model_prefix}.model  - SentencePiece モデル")
    print(f"  {args.model_prefix}.vocab  - 語彙一覧")
    print(f"\n次のステップ:")
    print(f"  1. japanese_learning_assistant.py を使って対話型学習")
    print(f"  2. handler.py の TOKENIZER_MODEL_PATH を更新")
    print(f"  3. モデルの再学習: train_and_save_checkpoint.py")


if __name__ == "__main__":
    main()
