#!/usr/bin/env python3
"""
DeepL翻訳クライアント

DeepL APIを使用して日本語⇔英語の翻訳を行います。

環境変数:
    DEEPL_API_KEY: DeepL APIキー
"""

import os
import deepl
from typing import Optional


class DeepLTranslator:
    """
    DeepL翻訳クライアント
    
    日本語と英語間の翻訳を行います。
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初期化
        
        Args:
            api_key: DeepL APIキー（省略時は環境変数DEEPL_API_KEYを使用）
        """
        self.api_key = api_key or os.environ.get('DEEPL_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "DeepL APIキーが設定されていません。\n"
                "環境変数 DEEPL_API_KEY を設定するか、api_keyパラメータを指定してください。\n"
                "APIキーは https://www.deepl.com/pro-api で取得できます。"
            )
        
        self.translator = deepl.Translator(self.api_key)
        
        # 使用量を確認
        try:
            usage = self.translator.get_usage()
            if usage.character.count is not None:
                print(f"📊 DeepL API使用状況: {usage.character.count:,} / {usage.character.limit:,} 文字")
                remaining = usage.character.limit - usage.character.count
                print(f"   残り: {remaining:,} 文字")
        except Exception as e:
            print(f"⚠️ DeepL API使用状況の取得に失敗: {e}")
    
    def translate_to_english(self, text: str) -> str:
        """
        日本語を英語に翻訳
        
        Args:
            text: 日本語テキスト
            
        Returns:
            英語テキスト
        """
        try:
            result = self.translator.translate_text(
                text,
                source_lang="JA",
                target_lang="EN-US"
            )
            return result.text
        except Exception as e:
            print(f"⚠️ 翻訳エラー (JA→EN): {e}")
            return text  # エラー時は元のテキストを返す
    
    def translate_to_japanese(self, text: str) -> str:
        """
        英語を日本語に翻訳
        
        Args:
            text: 英語テキスト
            
        Returns:
            日本語テキスト
        """
        try:
            result = self.translator.translate_text(
                text,
                source_lang="EN",
                target_lang="JA"
            )
            return result.text
        except Exception as e:
            print(f"⚠️ 翻訳エラー (EN→JA): {e}")
            return text  # エラー時は元のテキストを返す
    
    def detect_language(self, text: str) -> str:
        """
        テキストの言語を検出
        
        Args:
            text: テキスト
            
        Returns:
            言語コード ('JA', 'EN', etc.)
        """
        # 簡易的な日本語検出
        # 日本語文字（ひらがな、カタカナ、漢字）が含まれているか確認
        japanese_chars = 0
        for char in text:
            if '\u3040' <= char <= '\u309F':  # ひらがな
                japanese_chars += 1
            elif '\u30A0' <= char <= '\u30FF':  # カタカナ
                japanese_chars += 1
            elif '\u4E00' <= char <= '\u9FFF':  # 漢字
                japanese_chars += 1
        
        # 日本語文字が全体の10%以上なら日本語と判定
        if japanese_chars / max(len(text), 1) > 0.1:
            return "JA"
        else:
            return "EN"
    
    def translate_auto(self, text: str, target_lang: str = "JA") -> str:
        """
        自動言語検出して翻訳
        
        Args:
            text: テキスト
            target_lang: 出力言語 ('JA' or 'EN')
            
        Returns:
            翻訳されたテキスト
        """
        source_lang = self.detect_language(text)
        
        if source_lang == target_lang:
            return text  # 同じ言語なら翻訳不要
        
        if target_lang == "JA":
            return self.translate_to_japanese(text)
        else:
            return self.translate_to_english(text)


# テスト用
if __name__ == "__main__":
    print("DeepL翻訳テスト\n")
    
    try:
        translator = DeepLTranslator()
        
        # 日本語 → 英語
        ja_texts = [
            "こんにちは、今日の調子はどうですか？",
            "量子コンピュータは革新的な技術です。",
            "人工知能について教えてください。",
        ]
        
        print("=" * 60)
        print("🔄 日本語 → 英語")
        print("=" * 60)
        for text in ja_texts:
            en_text = translator.translate_to_english(text)
            print(f"JA: {text}")
            print(f"EN: {en_text}")
            print()
        
        # 英語 → 日本語
        en_texts = [
            "Hello, how are you today?",
            "Quantum computers are innovative technology.",
            "Please tell me about artificial intelligence.",
        ]
        
        print("=" * 60)
        print("🔄 英語 → 日本語")
        print("=" * 60)
        for text in en_texts:
            ja_text = translator.translate_to_japanese(text)
            print(f"EN: {text}")
            print(f"JA: {ja_text}")
            print()
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        print("\nDEEPL_API_KEY環境変数を設定してください:")
        print("  export DEEPL_API_KEY='your-api-key'")
