#!/usr/bin/env python3
"""
TikToken ベースのトークナイザー

OpenAI の tiktoken ライブラリを使用したトークナイザー実装。
GPT-4/GPT-3.5 と同じトークン化方式を使用。
"""

import tiktoken
from typing import List, Dict, Optional


class TikTokenTokenizer:
    """
    TikToken ベースのトークナイザー
    
    OpenAI の tiktoken ライブラリを使用。
    日本語を含む多言語に対応。
    """
    
    # 利用可能なエンコーディング
    ENCODINGS = {
        'gpt-4': 'cl100k_base',      # GPT-4, GPT-3.5-turbo
        'gpt-4o': 'o200k_base',       # GPT-4o
        'gpt-3': 'p50k_base',         # GPT-3
        'codex': 'p50k_edit',         # Codex
    }
    
    def __init__(self, encoding_name: str = 'cl100k_base', max_vocab: int = None):
        """
        初期化
        
        Args:
            encoding_name: エンコーディング名 ('cl100k_base', 'o200k_base', 'p50k_base', 等)
                          または 'gpt-4', 'gpt-4o', 'gpt-3' などのモデル名
            max_vocab: 最大語彙サイズ（制限する場合）
        """
        # モデル名が指定された場合、対応するエンコーディングに変換
        if encoding_name in self.ENCODINGS:
            encoding_name = self.ENCODINGS[encoding_name]
        
        self.encoding_name = encoding_name
        self.encoder = tiktoken.get_encoding(encoding_name)
        
        # 語彙サイズ
        self.vocab_size = self.encoder.n_vocab
        if max_vocab and max_vocab < self.vocab_size:
            self.max_vocab = max_vocab
        else:
            self.max_vocab = self.vocab_size
        
        # 特殊トークンID
        self.pad_id = 0
        self.unk_id = 0  # tiktokenにはUNKがないので0を使用
        self.bos_id = self.encoder.eot_token  # End of Text を BOS として使用
        self.eos_id = self.encoder.eot_token  # End of Text
        
        print(f"📚 TikToken トークナイザー初期化")
        print(f"   エンコーディング: {self.encoding_name}")
        print(f"   語彙サイズ: {self.vocab_size:,}")
    
    def encode(self, text: str, add_special: bool = True, verbose: bool = False) -> List[int]:
        """
        テキストをトークンIDのリストに変換

        Args:
            text: 入力テキスト
            add_special: 特殊トークンを追加するか（現在は使用していない）
            verbose: 詳細ログを出力するか

        Returns:
            トークンIDのリスト
        """
        # Allow all special tokens to be encoded as special tokens, and disable
        # the check for disallowed special tokens. This prevents errors when
        # training data contains literal special token strings like <|endoftext|>
        tokens = self.encoder.encode(
            text,
            allowed_special="all",
            disallowed_special=()
        )
        
        # max_vocabを超えるトークンは0に置き換え
        if self.max_vocab < self.vocab_size:
            tokens = [t if t < self.max_vocab else 0 for t in tokens]
        
        if verbose:
            print(f"[Encode] 入力: '{text[:50]}...' -> {len(tokens)} トークン")
            print(f"[Encode] トークンID (先頭10): {tokens[:10]}")
        
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
            token_ids = [t for t in token_ids if t != self.eos_id and t >= 0]
        
        try:
            text = self.encoder.decode(token_ids)
        except Exception as e:
            # デコードエラー時は個別にデコード
            text = ""
            for t in token_ids:
                try:
                    text += self.encoder.decode([t])
                except:
                    pass
        
        if verbose:
            print(f"[Decode] {len(token_ids)} トークン -> '{text[:100]}...'")
        
        return text
    
    def get_tokens(self, text: str) -> List[str]:
        """
        テキストをトークン文字列のリストに変換
        
        Args:
            text: 入力テキスト
            
        Returns:
            トークン文字列のリスト
        """
        token_ids = self.encode(text, add_special=False)
        tokens = []
        for tid in token_ids:
            try:
                tokens.append(self.encoder.decode([tid]))
            except:
                tokens.append(f"[{tid}]")
        return tokens
    
    def get_tokenization_info(self, text: str) -> Dict:
        """
        トークン化の詳細情報を取得
        
        Args:
            text: 入力テキスト
            
        Returns:
            トークン化情報の辞書
        """
        tokens = self.encode(text, add_special=False)
        token_strings = self.get_tokens(text)
        
        return {
            'input_text': text,
            'input_length': len(text),
            'token_count': len(tokens),
            'token_ids': tokens,
            'tokens': token_strings,
            'compression_ratio': len(text) / len(tokens) if tokens else 0,
            'encoding': self.encoding_name,
            'vocab_size': self.vocab_size,
        }
    
    def print_tokenization(self, text: str):
        """
        トークン化の詳細を表示
        
        Args:
            text: 入力テキスト
        """
        info = self.get_tokenization_info(text)
        
        print("=" * 60)
        print("🔤 TikToken トークン化詳細")
        print("=" * 60)
        print(f"入力テキスト: '{info['input_text']}'")
        print(f"入力文字数: {info['input_length']}")
        print(f"トークン数: {info['token_count']}")
        print(f"圧縮率: {info['compression_ratio']:.2f}")
        print(f"エンコーディング: {info['encoding']}")
        print("-" * 60)
        print("トークン分割:")
        for i, (tid, token) in enumerate(zip(info['token_ids'], info['tokens'])):
            # 見やすくするため、スペースや改行を可視化
            display_token = token.replace('\n', '\\n').replace('\t', '\\t')
            if display_token == ' ':
                display_token = '␣'
            print(f"  [{i:3d}] ID={tid:6d} -> '{display_token}'")
        print("=" * 60)


# テスト用
if __name__ == "__main__":
    print("TikToken トークナイザーテスト\n")
    
    # 初期化
    tokenizer = TikTokenTokenizer(encoding_name='o200k_base')
    
    # テストテキスト
    test_texts = [
        "こんにちは世界",
        "量子コンピュータは革新的な技術です",
        "人工知能とは何か",
        "Hello, World!",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    for text in test_texts:
        print()
        tokenizer.print_tokenization(text)
