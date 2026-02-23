#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQ 日本語学習アシスタント
==============================

SentencePiece (Open Assistant 日本語コーパスで学習済み) と
NeuroQuantum QBNN モデルを組み合わせた対話型 日本語学習ツール。

機能:
  - サブワード分割の可視化（漢字・ひらがな・カタカナの構造理解）
  - 語彙分析（頻出サブワード、文字種別の統計）
  - 対話形式の日本語Q&A（NeuroQ モデルによるテキスト生成）
  - Open Assistant 日本語データに基づく自然な応答

使い方:
    python japanese_learning_assistant.py
    python japanese_learning_assistant.py --tokenizer neuroq_tokenizer_oasst1_ja.model
    python japanese_learning_assistant.py --mode analyze --text "量子コンピュータ"
"""

import os
import sys
import re
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

try:
    import sentencepiece as spm
except ImportError:
    print("sentencepiece が必要です: pip install sentencepiece")
    sys.exit(1)

# NeuroQ モデル（オプション）
_NEUROQ_AVAILABLE = False
try:
    _current = Path(__file__).parent
    if str(_current) not in sys.path:
        sys.path.insert(0, str(_current))
    _runpod = _current / 'neuroq-runpod'
    if str(_runpod) not in sys.path:
        sys.path.insert(0, str(_runpod))

    import torch
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer
    _NEUROQ_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 文字種別の判定
# ---------------------------------------------------------------------------

def char_type(c: str) -> str:
    """文字の種別を返す"""
    cp = ord(c)
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    if 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
        return "katakana"
    if 0xFF65 <= cp <= 0xFF9F:
        return "katakana_hw"  # 半角カタカナ
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "kanji"
    if c.isascii() and c.isalpha():
        return "ascii"
    if c.isdigit() or 0xFF10 <= cp <= 0xFF19:
        return "digit"
    if unicodedata.category(c).startswith('P') or unicodedata.category(c).startswith('S'):
        return "symbol"
    return "other"


def char_type_label(t: str) -> str:
    labels = {
        "hiragana":    "ひらがな",
        "katakana":    "カタカナ",
        "katakana_hw": "半角カタカナ",
        "kanji":       "漢字",
        "ascii":       "英字",
        "digit":       "数字",
        "symbol":      "記号",
        "other":       "その他",
    }
    return labels.get(t, t)


# ---------------------------------------------------------------------------
# トークナイザー解析ツール
# ---------------------------------------------------------------------------

class JapaneseLearningAssistant:
    """日本語学習アシスタント"""

    def __init__(self, tokenizer_path: str, model_checkpoint: Optional[str] = None):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        self.vocab_size = self.sp.GetPieceSize()
        self.tokenizer_path = tokenizer_path

        # NeuroQ モデル（オプション）
        self.model = None
        self.device = None
        self.config = None
        if model_checkpoint and _NEUROQ_AVAILABLE:
            self._load_model(model_checkpoint)

    def _load_model(self, checkpoint_path: str):
        """NeuroQ モデルを読み込む"""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            else "cpu"
        )
        self.config = NeuroQuantumConfig(
            vocab_size=self.vocab_size,
            embed_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_layers=4,
            max_seq_len=256,
            lambda_entangle=0.5,
        )
        self.model = NeuroQuantum(self.config).to(self.device)

        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(state, strict=False)
            print(f"モデル読み込み完了: {checkpoint_path}")
        else:
            print(f"チェックポイントなし。ランダム初期化で動作します。")

        self.model.eval()

    # ----- サブワード分割の可視化 -----

    def tokenize_visual(self, text: str) -> str:
        """サブワード分割を見やすく表示する"""
        pieces = self.sp.EncodeAsPieces(text)
        ids = self.sp.EncodeAsIds(text)

        lines = []
        lines.append(f"入力: {text}")
        lines.append(f"トークン数: {len(pieces)}")
        lines.append("")
        lines.append("  ID   | サブワード       | 文字種別")
        lines.append("-------+------------------+-----------")

        for pid, piece in zip(ids, pieces):
            # SentencePiece の先頭マーカー '▁' を除去して文字種を判定
            display = piece.replace('▁', '_')
            raw = piece.replace('▁', '')
            if raw:
                types = Counter(char_type(c) for c in raw)
                dominant = types.most_common(1)[0][0]
                type_str = char_type_label(dominant)
            else:
                type_str = "空白"
            lines.append(f"  {pid:4d} | {display:16s} | {type_str}")

        lines.append("")
        lines.append(f"復元: {self.sp.DecodeIds(ids)}")
        return '\n'.join(lines)

    # ----- 文字種別分析 -----

    def analyze_char_types(self, text: str) -> str:
        """テキストの文字種別統計を表示"""
        counts = Counter()
        for c in text:
            if c.strip():
                counts[char_type(c)] += 1

        total = sum(counts.values())
        lines = []
        lines.append(f"テキスト: {text[:80]}{'...' if len(text)>80 else ''}")
        lines.append(f"文字数: {total}")
        lines.append("")

        for ct, n in counts.most_common():
            pct = n / total * 100 if total else 0
            bar = '#' * int(pct / 2)
            lines.append(f"  {char_type_label(ct):10s}  {n:4d} ({pct:5.1f}%) {bar}")

        return '\n'.join(lines)

    # ----- 語彙統計 -----

    def vocab_stats(self, sample_text: Optional[str] = None) -> str:
        """トークナイザーの語彙統計"""
        lines = []
        lines.append(f"トークナイザー: {self.tokenizer_path}")
        lines.append(f"語彙サイズ: {self.vocab_size}")
        lines.append("")

        # 文字種別ごとの語彙数をサンプリング
        type_counts = Counter()
        for i in range(min(self.vocab_size, 8000)):
            piece = self.sp.IdToPiece(i)
            raw = piece.replace('▁', '')
            if raw:
                types = Counter(char_type(c) for c in raw)
                dominant = types.most_common(1)[0][0]
                type_counts[dominant] += 1

        lines.append("語彙の文字種別分布:")
        for ct, n in type_counts.most_common():
            pct = n / self.vocab_size * 100
            lines.append(f"  {char_type_label(ct):10s}  {n:4d} ({pct:5.1f}%)")

        # サンプルテキストの分析
        if sample_text:
            pieces = self.sp.EncodeAsPieces(sample_text)
            ids = self.sp.EncodeAsIds(sample_text)
            compression = len(sample_text) / len(ids) if ids else 0
            lines.append("")
            lines.append(f"サンプル: {sample_text[:60]}")
            lines.append(f"  圧縮率: {compression:.2f} 文字/トークン")
            lines.append(f"  トークン: {len(ids)}")

        return '\n'.join(lines)

    # ----- テキスト生成 (NeuroQ モデル) -----

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.6, top_k: int = 40,
                 repetition_penalty: float = 2.0,
                 no_repeat_ngram: int = 3) -> str:
        """NeuroQ モデルでテキスト生成"""
        if self.model is None:
            return "(モデルが読み込まれていません。--checkpoint オプションでチェックポイントを指定してください。)"

        input_ids = self.sp.EncodeAsIds(prompt)
        # BOS 付与
        bos_id = self.sp.PieceToId('<s>')
        if bos_id >= 0:
            input_ids = [bos_id] + input_ids

        current_ids = list(input_ids)
        eos_id = self.sp.PieceToId('</s>')

        with torch.no_grad():
            for _ in range(max_tokens):
                seq = current_ids[-self.config.max_seq_len:]
                tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
                logits = self.model(tensor)
                next_logits = logits[0, -1, :].float()

                # 繰り返しペナルティ
                if repetition_penalty > 1.0:
                    for tid in set(current_ids[-50:]):
                        if 0 <= tid < next_logits.size(0):
                            next_logits[tid] /= repetition_penalty

                # N-gram ブロック
                if no_repeat_ngram > 0 and len(current_ids) >= no_repeat_ngram:
                    ngram = tuple(current_ids[-(no_repeat_ngram - 1):])
                    for i in range(len(current_ids) - no_repeat_ngram):
                        if tuple(current_ids[i:i + no_repeat_ngram - 1]) == ngram:
                            blocked = current_ids[i + no_repeat_ngram - 1]
                            if 0 <= blocked < next_logits.size(0):
                                next_logits[blocked] = float('-inf')

                # Temperature + Top-k
                next_logits /= max(temperature, 1e-8)
                values, indices = torch.topk(next_logits, min(top_k, next_logits.size(0)))
                probs = torch.softmax(values, dim=-1)
                chosen_idx = indices[torch.multinomial(probs, 1).item()].item()

                if chosen_idx == eos_id:
                    break
                current_ids.append(chosen_idx)

        # 入力部分を除いてデコード
        generated_ids = current_ids[len(input_ids):]
        return self.sp.DecodeIds(generated_ids)

    # ----- 対話ループ -----

    def interactive(self):
        """対話モード"""
        print("=" * 60)
        print("  NeuroQ 日本語学習アシスタント")
        print("=" * 60)
        print(f"トークナイザー: {self.tokenizer_path} (語彙: {self.vocab_size})")
        if self.model:
            print(f"モデル: NeuroQuantum (デバイス: {self.device})")
        else:
            print("モデル: なし (分析機能のみ)")
        print()
        print("コマンド:")
        print("  /tokenize <テキスト>   - サブワード分割を表示")
        print("  /analyze  <テキスト>   - 文字種別を分析")
        print("  /vocab                 - 語彙統計を表示")
        print("  /compare  <テキスト>   - 複数トークナイザーで比較")
        print("  /help                  - ヘルプを表示")
        print("  /quit                  - 終了")
        print()
        print("テキストをそのまま入力すると、")
        if self.model:
            print("NeuroQ モデルが応答を生成します。")
        else:
            print("トークナイズ結果を表示します。")
        print("-" * 60)

        while True:
            try:
                user = input("\nあなた> ").strip()
                if not user:
                    continue

                if user.lower() in ('/quit', '/exit', 'quit', 'exit'):
                    print("終了します。")
                    break

                if user.startswith('/tokenize'):
                    text = user[len('/tokenize'):].strip()
                    if text:
                        print(self.tokenize_visual(text))
                    else:
                        print("使い方: /tokenize <テキスト>")
                    continue

                if user.startswith('/analyze'):
                    text = user[len('/analyze'):].strip()
                    if text:
                        print(self.analyze_char_types(text))
                    else:
                        print("使い方: /analyze <テキスト>")
                    continue

                if user.startswith('/vocab'):
                    sample = user[len('/vocab'):].strip() or None
                    print(self.vocab_stats(sample))
                    continue

                if user.startswith('/compare'):
                    text = user[len('/compare'):].strip()
                    if text:
                        self._compare_tokenizers(text)
                    else:
                        print("使い方: /compare <テキスト>")
                    continue

                if user.startswith('/help'):
                    self._show_help()
                    continue

                # 通常入力 → 生成 or トークナイズ
                if self.model:
                    print("NeuroQ> ", end="", flush=True)
                    response = self.generate(user)
                    print(response)
                else:
                    print(self.tokenize_visual(user))

            except KeyboardInterrupt:
                print("\n終了します。")
                break
            except Exception as e:
                print(f"エラー: {e}")

    def _compare_tokenizers(self, text: str):
        """プロジェクト内の全トークナイザーでテキストを分割して比較"""
        base = Path(__file__).parent
        models = sorted(base.glob("*.model"))
        models += sorted((base / "neuroq-runpod").glob("*.model"))

        seen = set()
        print(f"テキスト: {text}\n")
        for mp in models:
            name = mp.name
            if name in seen:
                continue
            seen.add(name)
            try:
                sp_tmp = spm.SentencePieceProcessor()
                sp_tmp.Load(str(mp))
                pieces = sp_tmp.EncodeAsPieces(text)
                ids = sp_tmp.EncodeAsIds(text)
                print(f"  [{name}] 語彙={sp_tmp.GetPieceSize()}, トークン数={len(ids)}")
                print(f"    {' '.join(pieces[:20])}{'...' if len(pieces)>20 else ''}")
                print()
            except Exception as e:
                print(f"  [{name}] 読み込み失敗: {e}")

    def _show_help(self):
        print("""
日本語学習アシスタント - ヘルプ
================================

このツールは、SentencePiece トークナイザー（Open Assistant 日本語データで学習）
を使って、日本語テキストの構造を理解するための学習支援ツールです。

コマンド一覧:
  /tokenize <テキスト>   サブワード分割を可視化
                          各サブワードの ID、文字種別を表示
  /analyze  <テキスト>   文字種別（ひらがな/カタカナ/漢字/英字等）の統計
  /vocab [テキスト]      トークナイザーの語彙統計
                          テキスト指定で圧縮率も表示
  /compare  <テキスト>   プロジェクト内の全トークナイザーで比較分割
  /help                  このヘルプを表示
  /quit                  終了

テキストをそのまま入力すると:
  - モデルあり: NeuroQ が応答を生成
  - モデルなし: トークナイズ結果を表示
""")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='NeuroQ 日本語学習アシスタント')
    parser.add_argument('--tokenizer', type=str,
                        default='neuroq_tokenizer_oasst1_ja.model',
                        help='SentencePiece モデルファイル')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='NeuroQ モデルのチェックポイント (.pt)')
    parser.add_argument('--mode', choices=['interactive', 'analyze', 'tokenize', 'vocab'],
                        default='interactive',
                        help='動作モード')
    parser.add_argument('--text', type=str, default=None,
                        help='分析対象テキスト (非対話モード用)')

    args = parser.parse_args()

    # トークナイザーの存在確認
    if not os.path.exists(args.tokenizer):
        # neuroq-runpod 配下も確認
        alt = os.path.join('neuroq-runpod', args.tokenizer)
        if os.path.exists(alt):
            args.tokenizer = alt
        else:
            print(f"トークナイザーが見つかりません: {args.tokenizer}")
            print("先に train_spm_oasst_ja_enhanced.py を実行してください。")
            sys.exit(1)

    assistant = JapaneseLearningAssistant(
        tokenizer_path=args.tokenizer,
        model_checkpoint=args.checkpoint,
    )

    if args.mode == 'interactive':
        assistant.interactive()
    elif args.mode == 'tokenize':
        text = args.text or "量子コンピュータについて教えてください。"
        print(assistant.tokenize_visual(text))
    elif args.mode == 'analyze':
        text = args.text or "NeuroQは量子ビットニューラルネットワーク言語モデルです。"
        print(assistant.analyze_char_types(text))
    elif args.mode == 'vocab':
        print(assistant.vocab_stats(args.text))


if __name__ == "__main__":
    main()
