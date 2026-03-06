"""
Translation Pipeline for NeuroQ

日本語入力を英語に翻訳 → AI生成（英語） → 日本語に翻訳して出力

翻訳モデル: facebook/nllb-200-distilled-600M
トークナイザー: tiktoken (英語用)

Usage:
    from translation_pipeline import TranslationPipeline, TiktokenWrapper

    # 翻訳パイプライン初期化
    pipeline = TranslationPipeline()

    # 日本語→英語翻訳
    english_text = pipeline.ja_to_en("こんにちは、世界！")

    # 英語→日本語翻訳
    japanese_text = pipeline.en_to_ja("Hello, world!")
"""

import os
from typing import Optional, List, Union

import torch

# Lazy imports for optional dependencies
_transformers_available = None
_tiktoken_available = None


def _check_transformers():
    global _transformers_available
    if _transformers_available is None:
        try:
            import transformers
            _transformers_available = True
        except ImportError:
            _transformers_available = False
    return _transformers_available


def _check_tiktoken():
    global _tiktoken_available
    if _tiktoken_available is None:
        try:
            import tiktoken
            _tiktoken_available = True
        except ImportError:
            _tiktoken_available = False
    return _tiktoken_available


class TiktokenWrapper:
    """
    Tiktoken トークナイザーラッパー

    英語テキスト用の高速トークナイザー
    OpenAI GPT系モデルと互換性のあるトークン化を提供
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Args:
            encoding_name: tiktoken エンコーディング名
                - "cl100k_base": GPT-4, ChatGPT用 (推奨)
                - "p50k_base": GPT-3用
                - "r50k_base": Codex用
        """
        if not _check_tiktoken():
            raise ImportError(
                "tiktoken is required for TiktokenWrapper. "
                "Install it with: pip install tiktoken"
            )

        import tiktoken
        self.encoding_name = encoding_name
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoder.n_vocab

        # 特殊トークン
        self.bos_token_id = self.encoder.encode("<|begin|>", allowed_special={"<|begin|>"})[0] if "<|begin|>" in self.encoder._special_tokens else 0
        self.eos_token_id = self.encoder.encode("<|end|>", allowed_special={"<|end|>"})[0] if "<|end|>" in self.encoder._special_tokens else 0
        self.pad_token_id = 0

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], torch.Tensor]:
        """
        テキストをトークンIDに変換

        Args:
            text: 入力テキスト
            add_special_tokens: 特殊トークンを追加するか
            max_length: 最大長（truncation=Trueの場合に使用）
            truncation: 切り詰めを行うか
            return_tensors: "pt"でPyTorchテンソルを返す

        Returns:
            トークンIDのリストまたはテンソル
        """
        tokens = self.encoder.encode(text)

        if truncation and max_length is not None:
            tokens = tokens[:max_length]

        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        トークンIDをテキストに変換

        Args:
            token_ids: トークンIDのリストまたはテンソル
            skip_special_tokens: 特殊トークンをスキップするか

        Returns:
            デコードされたテキスト
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.squeeze().tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        return self.encoder.decode(token_ids)

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> dict:
        """
        HuggingFace トークナイザー互換のインターフェース
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        all_input_ids = []
        all_attention_mask = []

        for t in texts:
            tokens = self.encode(t, truncation=truncation, max_length=max_length)
            all_input_ids.append(tokens)
            all_attention_mask.append([1] * len(tokens))

        # パディング
        if padding and len(all_input_ids) > 1:
            max_len = max(len(ids) for ids in all_input_ids)
            for i in range(len(all_input_ids)):
                pad_len = max_len - len(all_input_ids[i])
                all_input_ids[i] = all_input_ids[i] + [self.pad_token_id] * pad_len
                all_attention_mask[i] = all_attention_mask[i] + [0] * pad_len

        result = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
        }

        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(all_input_ids, dtype=torch.long)
            result["attention_mask"] = torch.tensor(all_attention_mask, dtype=torch.long)

        return result


class TranslationPipeline:
    """
    NLLB-200を使用した翻訳パイプライン

    日本語 ↔ 英語 の双方向翻訳をサポート

    パイプライン:
        入力(日本語) → 翻訳(日本語→英語) → AI生成 → 翻訳(英語→日本語) → 出力
    """

    # NLLB-200 言語コード
    LANG_JA = "jpn_Jpan"  # 日本語
    LANG_EN = "eng_Latn"  # 英語

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: Optional[str] = None,
        max_length: int = 512,
        use_tiktoken: bool = True,
        tiktoken_encoding: str = "cl100k_base",
    ):
        """
        Args:
            model_name: NLLB翻訳モデル名
                - "facebook/nllb-200-distilled-600M" (軽量版、推奨)
                - "facebook/nllb-200-1.3B"
                - "facebook/nllb-200-3.3B"
            device: 使用するデバイス ("cuda", "mps", "cpu", またはNoneで自動選択)
            max_length: 翻訳の最大長
            use_tiktoken: tiktokenトークナイザーを使用するか
            tiktoken_encoding: tiktokenのエンコーディング名
        """
        if not _check_transformers():
            raise ImportError(
                "transformers is required for TranslationPipeline. "
                "Install it with: pip install transformers"
            )

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length

        # デバイス選択
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[TranslationPipeline] Loading model: {model_name}")
        print(f"[TranslationPipeline] Device: {self.device}")

        # NLLB翻訳モデルとトークナイザーをロード
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # tiktokenラッパー（英語トークン化用）
        self.tiktoken_wrapper = None
        if use_tiktoken and _check_tiktoken():
            self.tiktoken_wrapper = TiktokenWrapper(encoding_name=tiktoken_encoding)
            print(f"[TranslationPipeline] Tiktoken encoding: {tiktoken_encoding}")

        print("[TranslationPipeline] Ready!")

    def _translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        num_beams: int = 5,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        """
        内部翻訳メソッド

        Args:
            text: 入力テキスト
            src_lang: ソース言語コード
            tgt_lang: ターゲット言語コード
            num_beams: ビームサーチのビーム数
            do_sample: サンプリングを行うか
            temperature: サンプリング温度

        Returns:
            翻訳されたテキスト
        """
        # ソース言語を設定
        self.tokenizer.src_lang = src_lang

        # テキストをエンコード
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        # ターゲット言語のトークンIDを取得
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        # 翻訳を生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
            )

        # デコード
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

    def ja_to_en(
        self,
        text: str,
        num_beams: int = 5,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        """
        日本語から英語に翻訳

        Args:
            text: 日本語テキスト
            num_beams: ビームサーチのビーム数
            do_sample: サンプリングを行うか
            temperature: サンプリング温度

        Returns:
            英語テキスト
        """
        return self._translate(
            text=text,
            src_lang=self.LANG_JA,
            tgt_lang=self.LANG_EN,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
        )

    def en_to_ja(
        self,
        text: str,
        num_beams: int = 5,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        """
        英語から日本語に翻訳

        Args:
            text: 英語テキスト
            num_beams: ビームサーチのビーム数
            do_sample: サンプリングを行うか
            temperature: サンプリング温度

        Returns:
            日本語テキスト
        """
        return self._translate(
            text=text,
            src_lang=self.LANG_EN,
            tgt_lang=self.LANG_JA,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
        )

    def tokenize_english(self, text: str) -> List[int]:
        """
        英語テキストをtiktokenでトークン化

        Args:
            text: 英語テキスト

        Returns:
            トークンIDのリスト
        """
        if self.tiktoken_wrapper is None:
            raise RuntimeError("Tiktoken is not available. Initialize with use_tiktoken=True")
        return self.tiktoken_wrapper.encode(text)

    def detokenize_english(self, token_ids: List[int]) -> str:
        """
        トークンIDを英語テキストに変換

        Args:
            token_ids: トークンIDのリスト

        Returns:
            英語テキスト
        """
        if self.tiktoken_wrapper is None:
            raise RuntimeError("Tiktoken is not available. Initialize with use_tiktoken=True")
        return self.tiktoken_wrapper.decode(token_ids)


class TranslatedNeuroQuantumAI:
    """
    翻訳パイプライン統合版 NeuroQuantumAI

    入力(日本語) → 翻訳(日本語→英語) → AI生成(英語) → 翻訳(英語→日本語) → 出力

    英語で学習したモデルを日本語インターフェースで使用可能にする
    """

    def __init__(
        self,
        neuroq_model,  # NeuroQuantumAI インスタンス
        translation_model: str = "facebook/nllb-200-distilled-600M",
        device: Optional[str] = None,
    ):
        """
        Args:
            neuroq_model: 学習済みNeuroQuantumAIインスタンス
            translation_model: NLLB翻訳モデル名
            device: 使用するデバイス
        """
        self.neuroq = neuroq_model
        self.pipeline = TranslationPipeline(
            model_name=translation_model,
            device=device,
            use_tiktoken=True,
        )

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        translate_input: bool = True,
        translate_output: bool = True,
        **kwargs,
    ) -> str:
        """
        日本語プロンプトから日本語レスポンスを生成

        Args:
            prompt: 日本語プロンプト
            max_length: 生成最大長
            temperature: 生成温度
            translate_input: 入力を日→英翻訳するか
            translate_output: 出力を英→日翻訳するか
            **kwargs: NeuroQuantumAI.generate()に渡す追加引数

        Returns:
            生成されたテキスト（日本語）
        """
        # 1. 入力を英語に翻訳
        if translate_input:
            english_prompt = self.pipeline.ja_to_en(prompt)
            print(f"[Translation] JA→EN: {prompt[:50]}... → {english_prompt[:50]}...")
        else:
            english_prompt = prompt

        # 2. 英語でAI生成
        english_response = self.neuroq.generate(
            prompt=english_prompt,
            max_length=max_length,
            temperature=temperature,
            **kwargs,
        )

        # 3. 出力を日本語に翻訳
        if translate_output:
            japanese_response = self.pipeline.en_to_ja(english_response)
            print(f"[Translation] EN→JA: {english_response[:50]}... → {japanese_response[:50]}...")
        else:
            japanese_response = english_response

        return japanese_response

    def chat(
        self,
        message: str,
        history: Optional[List[dict]] = None,
        **kwargs,
    ) -> str:
        """
        チャット形式の対話

        Args:
            message: ユーザーメッセージ（日本語）
            history: 会話履歴 [{"role": "user/assistant", "content": "..."}]
            **kwargs: generate()に渡す追加引数

        Returns:
            アシスタントの応答（日本語）
        """
        # 会話履歴を構築
        context = ""
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    context += f"User: {content}\n"
                else:
                    context += f"Assistant: {content}\n"

        # 現在のメッセージを追加
        full_prompt = context + f"User: {message}\nAssistant:"

        # 生成
        response = self.generate(full_prompt, **kwargs)

        # "Assistant:" 以降の部分を抽出
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response


def create_english_training_data(
    japanese_texts: List[str],
    translation_pipeline: Optional[TranslationPipeline] = None,
    model_name: str = "facebook/nllb-200-distilled-600M",
) -> List[str]:
    """
    日本語テキストを英語に翻訳して学習データを作成

    Args:
        japanese_texts: 日本語テキストのリスト
        translation_pipeline: 既存のTranslationPipelineインスタンス（None時は新規作成）
        model_name: NLLB翻訳モデル名

    Returns:
        英語テキストのリスト
    """
    if translation_pipeline is None:
        translation_pipeline = TranslationPipeline(model_name=model_name)

    english_texts = []
    total = len(japanese_texts)

    for i, ja_text in enumerate(japanese_texts):
        try:
            en_text = translation_pipeline.ja_to_en(ja_text)
            english_texts.append(en_text)
            if (i + 1) % 100 == 0:
                print(f"[Translation] Progress: {i+1}/{total}")
        except Exception as e:
            print(f"[Translation] Error at index {i}: {e}")
            english_texts.append(ja_text)  # フォールバック

    return english_texts


# テスト用
if __name__ == "__main__":
    print("=" * 60)
    print("Translation Pipeline Test")
    print("=" * 60)

    # Tiktoken テスト
    print("\n[Test 1] Tiktoken Wrapper")
    if _check_tiktoken():
        tokenizer = TiktokenWrapper()
        test_text = "Hello, world! How are you today?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"  Input: {test_text}")
        print(f"  Tokens: {tokens[:10]}... (total: {len(tokens)})")
        print(f"  Decoded: {decoded}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
    else:
        print("  Tiktoken not available")

    # 翻訳パイプラインテスト
    print("\n[Test 2] Translation Pipeline")
    try:
        pipeline = TranslationPipeline()

        # 日本語→英語
        ja_text = "人工知能は私たちの生活を大きく変えています。"
        en_text = pipeline.ja_to_en(ja_text)
        print(f"  JA: {ja_text}")
        print(f"  EN: {en_text}")

        # 英語→日本語
        en_text2 = "Machine learning is a subset of artificial intelligence."
        ja_text2 = pipeline.en_to_ja(en_text2)
        print(f"  EN: {en_text2}")
        print(f"  JA: {ja_text2}")

    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
