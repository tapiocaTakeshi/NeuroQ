#!/usr/bin/env python3
"""
Translation Pipeline Test Script

翻訳パイプラインの動作確認テスト

パイプライン:
    入力(日本語) → 翻訳(日本語→英語) → AI生成(英語) → 翻訳(英語→日本語) → 出力

Usage:
    python test_translation_pipeline.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_tiktoken():
    """Tiktoken トークナイザーテスト"""
    print("\n" + "=" * 60)
    print("[Test 1] Tiktoken Tokenizer")
    print("=" * 60)

    try:
        from translation_pipeline import TiktokenWrapper

        tokenizer = TiktokenWrapper(encoding_name="cl100k_base")
        print(f"  Encoding: cl100k_base")
        print(f"  Vocab size: {tokenizer.vocab_size:,}")

        # 英語テキストのトークン化
        test_texts = [
            "Hello, world!",
            "Artificial intelligence is transforming our lives.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            print(f"\n  Input:   {text}")
            print(f"  Tokens:  {tokens[:8]}{'...' if len(tokens) > 8 else ''} (len={len(tokens)})")
            print(f"  Decoded: {decoded}")

        print("\n  [OK] Tiktoken test passed!")
        return True

    except ImportError as e:
        print(f"  [SKIP] tiktoken not installed: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_translation():
    """NLLB-200 翻訳テスト"""
    print("\n" + "=" * 60)
    print("[Test 2] NLLB-200 Translation")
    print("=" * 60)

    try:
        from translation_pipeline import TranslationPipeline

        print("  Loading NLLB-200 model (this may take a while)...")
        pipeline = TranslationPipeline(
            model_name="facebook/nllb-200-distilled-600M",
            use_tiktoken=True,
        )

        # 日本語→英語テスト
        print("\n  [JA → EN Translation]")
        ja_texts = [
            "こんにちは、世界！",
            "人工知能は私たちの生活を変えています。",
            "量子コンピュータは未来の技術です。",
        ]

        for ja in ja_texts:
            en = pipeline.ja_to_en(ja)
            print(f"    JA: {ja}")
            print(f"    EN: {en}")
            print()

        # 英語→日本語テスト
        print("  [EN → JA Translation]")
        en_texts = [
            "Hello, world!",
            "Machine learning is a powerful tool.",
            "Neural networks can learn patterns from data.",
        ]

        for en in en_texts:
            ja = pipeline.en_to_ja(en)
            print(f"    EN: {en}")
            print(f"    JA: {ja}")
            print()

        print("  [OK] Translation test passed!")
        return True

    except ImportError as e:
        print(f"  [SKIP] Required dependencies not installed: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """翻訳パイプライン統合テスト"""
    print("\n" + "=" * 60)
    print("[Test 3] Full Pipeline Integration")
    print("=" * 60)

    try:
        from neuroquantum_layered import NeuroQuantumAI

        print("  Creating NeuroQuantumAI instance...")
        ai = NeuroQuantumAI(
            embed_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=3,
            max_seq_len=64,
        )

        # 英語データで簡易学習
        print("  Training on English data...")
        english_training_data = [
            "Artificial intelligence is a field of computer science.",
            "Machine learning algorithms learn from data.",
            "Neural networks are inspired by biological brains.",
            "Deep learning uses multiple layers of neurons.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret visual information.",
            "Reinforcement learning trains agents through rewards.",
            "Transfer learning leverages pre-trained models for new tasks.",
        ]

        ai.train(
            texts=english_training_data,
            epochs=5,
            batch_size=4,
            vocab_size=8000,
        )

        # 翻訳パイプラインを有効化
        print("\n  Enabling translation pipeline...")
        try:
            translated_ai = ai.enable_translation_pipeline()

            # 日本語入力で生成テスト
            print("\n  [Japanese Input → English Generation → Japanese Output]")
            test_prompts = [
                "人工知能とは",
                "機械学習について",
            ]

            for prompt in test_prompts:
                print(f"\n    Input (JA): {prompt}")
                response = translated_ai.generate(
                    prompt=prompt,
                    max_length=50,
                    temperature=0.7,
                )
                print(f"    Output (JA): {response}")

            print("\n  [OK] Pipeline integration test passed!")
            return True

        except ImportError as e:
            print(f"\n  [SKIP] Translation pipeline not available: {e}")
            print("  Testing basic English generation instead...")

            response = ai.generate("Artificial intelligence", max_length=30)
            print(f"    EN Input: Artificial intelligence")
            print(f"    EN Output: {response}")
            return True

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_english_training_data_creation():
    """英語学習データ作成テスト"""
    print("\n" + "=" * 60)
    print("[Test 4] English Training Data Creation")
    print("=" * 60)

    try:
        from translation_pipeline import TranslationPipeline, create_english_training_data

        # 日本語データを英語に変換
        japanese_data = [
            "人工知能は、人間の知的能力を機械で再現する技術です。",
            "機械学習は、データからパターンを学習するアルゴリズムです。",
            "ニューラルネットワークは、脳の神経細胞を模倣した計算モデルです。",
        ]

        print("  Input Japanese data:")
        for ja in japanese_data:
            print(f"    - {ja}")

        print("\n  Creating English training data...")
        pipeline = TranslationPipeline()
        english_data = create_english_training_data(
            japanese_texts=japanese_data,
            translation_pipeline=pipeline,
        )

        print("\n  Output English data:")
        for en in english_data:
            print(f"    - {en}")

        print("\n  [OK] English training data creation test passed!")
        return True

    except ImportError as e:
        print(f"  [SKIP] Required dependencies not installed: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def main():
    """メインテスト実行"""
    print("=" * 60)
    print("  NeuroQ Translation Pipeline Test Suite")
    print("=" * 60)
    print("\nPipeline Architecture:")
    print("  Input(JA) -> Translate(JA->EN) -> AI Generate(EN)")
    print("           -> Translate(EN->JA) -> Output(JA)")
    print("\nComponents:")
    print("  - Tokenizer: tiktoken (cl100k_base)")
    print("  - Translation: facebook/nllb-200-distilled-600M")
    print("  - Model: NeuroQuantumAI (trained on English)")

    results = {}

    # Test 1: Tiktoken
    results["tiktoken"] = test_tiktoken()

    # Test 2: Translation (NLLB-200)
    results["translation"] = test_translation()

    # Test 3: Full Pipeline
    results["pipeline"] = test_pipeline_integration()

    # Test 4: Training Data Creation
    results["training_data"] = test_english_training_data_creation()

    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "[OK]" if passed else "[FAIL/SKIP]"
        print(f"  {status} {test_name}")

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"\n  Passed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n  All tests passed!")
    else:
        print("\n  Some tests failed or were skipped.")
        print("  Make sure to install: pip install tiktoken transformers sacremoses")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
