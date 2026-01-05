#!/usr/bin/env python3
"""
NLLB-200 翻訳クラス

Meta (Facebook) の NLLB-200 モデルを使用したオフライン翻訳。
200以上の言語間で翻訳が可能。APIキー不要。

モデルサイズ:
- distilled-600M: 軽量版（推奨、約2.4GB）
- distilled-1.3B: 中量版（約5.2GB）
- 3.3B: 完全版（約13GB）
"""

import torch
from typing import Optional
import warnings

# transformersの警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)


class NLLB200Translator:
    """
    NLLB-200 翻訳クラス
    
    Meta (Facebook) の No Language Left Behind (NLLB) モデルを使用。
    オフラインで動作し、APIキー不要。
    """
    
    # 言語コード（NLLB-200形式）
    LANG_CODES = {
        'ja': 'jpn_Jpan',   # 日本語
        'en': 'eng_Latn',   # 英語
        'zh': 'zho_Hans',   # 中国語（簡体字）
        'ko': 'kor_Hang',   # 韓国語
        'fr': 'fra_Latn',   # フランス語
        'de': 'deu_Latn',   # ドイツ語
        'es': 'spa_Latn',   # スペイン語
        'it': 'ita_Latn',   # イタリア語
        'pt': 'por_Latn',   # ポルトガル語
        'ru': 'rus_Cyrl',   # ロシア語
    }
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", 
                 device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: 使用するモデル名
                - "facebook/nllb-200-distilled-600M" (推奨、軽量)
                - "facebook/nllb-200-distilled-1.3B" (中量)
                - "facebook/nllb-200-3.3B" (高精度)
            device: デバイス ('cuda', 'mps', 'cpu'、自動選択の場合None)
        """
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        print(f"📚 NLLB-200 翻訳モデルをロード中...")
        print(f"   モデル: {model_name}")
        
        # デバイス選択
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
                print("   🍎 Apple Silicon GPU (MPS) を使用")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("   🎮 NVIDIA GPU (CUDA) を使用")
            else:
                self.device = "cpu"
                print("   💻 CPU を使用")
        else:
            self.device = device
        
        # モデルとトークナイザーをロード
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # MPSではfloat16が使えないのでfloat32を使用
        if self.device == "mps":
            self.model = self.model.to(self.device)
        elif self.device == "cuda":
            self.model = self.model.half().to(self.device)  # CUDAはfloat16で高速化
        else:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        print("   ✅ モデルロード完了！")
    
    def _get_lang_code(self, lang: str) -> str:
        """言語コードを取得"""
        lang = lang.lower()
        if lang in self.LANG_CODES:
            return self.LANG_CODES[lang]
        # 既にNLLB形式の場合はそのまま返す
        if '_' in lang:
            return lang
        raise ValueError(f"不明な言語コード: {lang}")
    
    def translate(self, text: str, source_lang: str, target_lang: str, 
                  max_length: int = 256) -> str:
        """
        テキストを翻訳
        
        Args:
            text: 翻訳するテキスト
            source_lang: 元の言語コード ('ja', 'en', etc.)
            target_lang: 翻訳先の言語コード
            max_length: 最大出力長
            
        Returns:
            翻訳されたテキスト
        """
        src_code = self._get_lang_code(source_lang)
        tgt_code = self._get_lang_code(target_lang)
        
        # ソース言語を設定
        self.tokenizer.src_lang = src_code
        
        # エンコード
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                                truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 翻訳
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
        
        # デコード
        translated = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return translated
    
    def translate_to_english(self, text: str) -> str:
        """日本語を英語に翻訳"""
        return self.translate(text, source_lang='ja', target_lang='en')
    
    def translate_to_japanese(self, text: str) -> str:
        """英語を日本語に翻訳"""
        return self.translate(text, source_lang='en', target_lang='ja')
    
    def detect_language(self, text: str) -> str:
        """
        テキストの言語を検出（簡易版）
        
        Args:
            text: テキスト
            
        Returns:
            言語コード ('ja', 'en', etc.)
        """
        # 日本語文字の検出
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
            return "ja"
        else:
            return "en"


# テスト用
if __name__ == "__main__":
    print("=" * 60)
    print("🌐 NLLB-200 翻訳テスト")
    print("=" * 60)
    
    try:
        translator = NLLB200Translator()
        
        # 日本語 → 英語
        ja_texts = [
            "こんにちは、今日の調子はどうですか？",
            "量子コンピュータは革新的な技術です。",
            "人工知能について教えてください。",
        ]
        
        print("\n" + "=" * 60)
        print("🔄 日本語 → 英語")
        print("=" * 60)
        for text in ja_texts:
            en_text = translator.translate_to_english(text)
            print(f"\nJA: {text}")
            print(f"EN: {en_text}")
        
        # 英語 → 日本語
        en_texts = [
            "Hello, how are you today?",
            "Quantum computers are innovative technology.",
            "Please tell me about artificial intelligence.",
        ]
        
        print("\n" + "=" * 60)
        print("🔄 英語 → 日本語")
        print("=" * 60)
        for text in en_texts:
            ja_text = translator.translate_to_japanese(text)
            print(f"\nEN: {text}")
            print(f"JA: {ja_text}")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
