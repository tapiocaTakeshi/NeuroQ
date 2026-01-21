#!/usr/bin/env python3
"""
BPE トークナイザー学習モジュール

tiktoken スタイルの BPE (Byte Pair Encoding) トークナイザーを
学習データから学習します。

Hugging Face tokenizers ライブラリを使用し、
tiktoken 互換のトークナイザーを生成します。
"""

import os
import json
import tempfile
from typing import List, Dict, Optional, Union
from pathlib import Path


def check_dependencies():
    """依存関係のチェック"""
    missing = []

    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
        from tokenizers.normalizers import NFKC, Sequence as NormSeq
    except ImportError:
        missing.append("tokenizers")

    if missing:
        raise ImportError(
            f"必要なライブラリがありません: {', '.join(missing)}\n"
            f"pip install {' '.join(missing)} を実行してください"
        )

    return True


class BPETokenizerTrainer:
    """
    BPE トークナイザー学習クラス

    tiktoken 互換の BPE トークナイザーを学習します。
    """

    # デフォルト特殊トークン
    DEFAULT_SPECIAL_TOKENS = [
        "<pad>",      # パディング
        "<unk>",      # 未知語
        "<s>",        # 文頭
        "</s>",       # 文末
        "<USER>",     # ユーザー発話
        "<ASSISTANT>",  # アシスタント発話
    ]

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        初期化

        Args:
            vocab_size: 語彙サイズ（デフォルト: 32000）
            min_frequency: 最小出現頻度（デフォルト: 2）
            special_tokens: 特殊トークンリスト
        """
        check_dependencies()

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or self.DEFAULT_SPECIAL_TOKENS

        self.tokenizer = None
        self.is_trained = False

        print(f"BPE トークナイザー学習器を初期化")
        print(f"   語彙サイズ: {vocab_size:,}")
        print(f"   最小出現頻度: {min_frequency}")
        print(f"   特殊トークン数: {len(self.special_tokens)}")

    def train(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> 'BPETokenizerTrainer':
        """
        テキストデータからトークナイザーを学習

        Args:
            texts: 学習用テキストのリスト
            show_progress: 進捗を表示するか

        Returns:
            self
        """
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
        from tokenizers.normalizers import NFKC

        print(f"\n学習データ: {len(texts):,} サンプル")

        # 総文字数を計算
        total_chars = sum(len(t) for t in texts)
        print(f"総文字数: {total_chars:,}")

        # BPE モデルを初期化
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # 正規化: NFKC (Unicode正規化)
        tokenizer.normalizer = NFKC()

        # Pre-tokenizer: ByteLevel (tiktoken と同様のバイトレベル)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # デコーダー: ByteLevel
        tokenizer.decoder = decoders.ByteLevel()

        # Post-processor: ByteLevel
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # BPE トレーナーを設定
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        print(f"\n学習開始...")

        # 一時ファイルに書き出してから学習（メモリ効率）
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        ) as f:
            temp_file = f.name
            for text in texts:
                f.write(text.strip() + '\n')

        try:
            # ファイルから学習
            tokenizer.train([temp_file], trainer)
        finally:
            # 一時ファイル削除
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        self.tokenizer = tokenizer
        self.is_trained = True

        actual_vocab_size = tokenizer.get_vocab_size()
        print(f"\n学習完了！")
        print(f"   実際の語彙サイズ: {actual_vocab_size:,}")

        return self

    def train_from_iterator(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> 'BPETokenizerTrainer':
        """
        イテレーターからトークナイザーを学習（メモリ効率版）

        Args:
            texts: 学習用テキストのリスト/イテレーター
            show_progress: 進捗を表示するか

        Returns:
            self
        """
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
        from tokenizers.normalizers import NFKC

        print(f"\n学習データからイテレーター学習開始...")

        # BPE モデルを初期化
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # BPE トレーナーを設定
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # イテレーターから学習
        tokenizer.train_from_iterator(texts, trainer)

        self.tokenizer = tokenizer
        self.is_trained = True

        actual_vocab_size = tokenizer.get_vocab_size()
        print(f"学習完了！ 語彙サイズ: {actual_vocab_size:,}")

        return self

    def save(self, path: str) -> str:
        """
        トークナイザーを保存

        Args:
            path: 保存パス（.json ファイル）

        Returns:
            保存したファイルパス
        """
        if not self.is_trained:
            raise ValueError("トークナイザーが学習されていません")

        # ディレクトリを作成
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # 保存
        self.tokenizer.save(path)
        print(f"トークナイザーを保存: {path}")

        return path

    def get_vocab_size(self) -> int:
        """語彙サイズを取得"""
        if not self.is_trained:
            return self.vocab_size
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        """テキストをトークンIDにエンコード"""
        if not self.is_trained:
            raise ValueError("トークナイザーが学習されていません")
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        """トークンIDをテキストにデコード"""
        if not self.is_trained:
            raise ValueError("トークナイザーが学習されていません")
        return self.tokenizer.decode(token_ids)

    def get_tokens(self, text: str) -> List[str]:
        """テキストをトークン文字列のリストに変換"""
        if not self.is_trained:
            raise ValueError("トークナイザーが学習されていません")
        return self.tokenizer.encode(text).tokens


class TrainedBPETokenizer:
    """
    学習済み BPE トークナイザー

    保存されたトークナイザーを読み込んで使用するクラス。
    TikTokenTokenizer と同じインターフェースを提供。
    """

    def __init__(self, model_path: str):
        """
        初期化

        Args:
            model_path: トークナイザーモデルファイル（.json）
        """
        check_dependencies()
        from tokenizers import Tokenizer

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"トークナイザーファイルが見つかりません: {model_path}")

        self.model_path = model_path
        self.tokenizer = Tokenizer.from_file(model_path)

        # 語彙サイズ
        self.vocab_size = self.tokenizer.get_vocab_size()

        # 特殊トークンID
        self.pad_id = self._get_token_id("<pad>", default=0)
        self.unk_id = self._get_token_id("<unk>", default=1)
        self.bos_id = self._get_token_id("<s>", default=2)
        self.eos_id = self._get_token_id("</s>", default=3)

        print(f"学習済み BPE トークナイザーを読み込み")
        print(f"   モデル: {model_path}")
        print(f"   語彙サイズ: {self.vocab_size:,}")

    def _get_token_id(self, token: str, default: int = 0) -> int:
        """トークンIDを取得"""
        vocab = self.tokenizer.get_vocab()
        return vocab.get(token, default)

    def encode(self, text: str, add_special: bool = True, verbose: bool = False) -> List[int]:
        """
        テキストをトークンIDのリストに変換

        Args:
            text: 入力テキスト
            add_special: 特殊トークンを追加するか（互換性のため）
            verbose: 詳細ログを出力するか

        Returns:
            トークンIDのリスト
        """
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids

        if verbose:
            print(f"[Encode] 入力: '{text[:50]}...' -> {len(tokens)} トークン")

        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True, verbose: bool = False) -> str:
        """
        トークンIDのリストをテキストに変換

        Args:
            token_ids: トークンIDのリスト
            skip_special: 特殊トークンをスキップするか
            verbose: 詳細ログを出力するか

        Returns:
            デコードされたテキスト
        """
        # 特殊トークンをフィルタ
        if skip_special:
            special_ids = {self.pad_id, self.bos_id, self.eos_id}
            token_ids = [t for t in token_ids if t not in special_ids and t >= 0]

        text = self.tokenizer.decode(token_ids)

        if verbose:
            print(f"[Decode] {len(token_ids)} トークン -> '{text[:100]}...'")

        return text

    def get_tokens(self, text: str) -> List[str]:
        """テキストをトークン文字列のリストに変換"""
        return self.tokenizer.encode(text).tokens

    def get_tokenization_info(self, text: str) -> Dict:
        """トークン化の詳細情報を取得"""
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids
        token_strings = encoding.tokens

        return {
            'input_text': text,
            'input_length': len(text),
            'token_count': len(tokens),
            'token_ids': tokens,
            'tokens': token_strings,
            'compression_ratio': len(text) / len(tokens) if tokens else 0,
            'vocab_size': self.vocab_size,
        }


def train_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = 32000,
    min_frequency: int = 2,
    save_path: Optional[str] = None,
    special_tokens: Optional[List[str]] = None,
) -> Union[BPETokenizerTrainer, str]:
    """
    BPE トークナイザーを学習する便利関数

    Args:
        texts: 学習用テキストのリスト
        vocab_size: 語彙サイズ
        min_frequency: 最小出現頻度
        save_path: 保存パス（指定時は保存してパスを返す）
        special_tokens: 特殊トークンリスト

    Returns:
        BPETokenizerTrainer または 保存パス
    """
    trainer = BPETokenizerTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    trainer.train(texts)

    if save_path:
        return trainer.save(save_path)

    return trainer


# テスト用
if __name__ == "__main__":
    print("=" * 70)
    print("BPE トークナイザー学習テスト")
    print("=" * 70)

    # テストデータ
    test_texts = [
        "量子コンピュータは、量子力学の原理を利用して情報を処理する革新的な計算機です。",
        "量子ビット（キュービット）は、0と1の重ね合わせ状態を取ることができます。",
        "人工知能（AI）は、人間の知的活動をコンピュータで実現しようとする技術です。",
        "機械学習は、データから学習して改善するシステムを実現します。",
        "深層学習（ディープラーニング）は、多層のニューラルネットワークを用いた機械学習手法です。",
        "こんにちは、今日も良い天気ですね。何かお手伝いできることはありますか？",
        "<USER>こんにちは<ASSISTANT>こんにちは！今日も良い天気ですね。",
        "The quick brown fox jumps over the lazy dog.",
        "Hello, World! This is a test message.",
        "Python is a popular programming language.",
    ] * 100  # データを増やす

    # 学習
    trainer = BPETokenizerTrainer(vocab_size=1000, min_frequency=2)
    trainer.train(test_texts)

    # 保存
    save_path = "/tmp/test_bpe_tokenizer.json"
    trainer.save(save_path)

    # 読み込みテスト
    print("\n" + "=" * 70)
    print("学習済みトークナイザーテスト")
    print("=" * 70)

    tokenizer = TrainedBPETokenizer(save_path)

    # テスト
    test_samples = [
        "量子コンピュータについて教えて",
        "こんにちは！",
        "Hello, World!",
        "<USER>質問です<ASSISTANT>",
    ]

    for text in test_samples:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"\n入力: {text}")
        print(f"トークン数: {len(tokens)}")
        print(f"デコード: {decoded}")

    print("\n" + "=" * 70)
    print("テスト完了！")
    print("=" * 70)
