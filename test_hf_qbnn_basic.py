#!/usr/bin/env python3
"""
HF-OpenAI-QBNN Pipeline の基本テスト
===================================
トークナイザー、エンベディング、アテンションの動作確認
"""

import torch
from hf_openai_qbnn_pipeline import (
    HFTokenizerWrapper,
    HFQBNNConfig,
    HFOpenAIQBNNPipeline
)


def test_tokenizer():
    """トークナイザーのテスト"""
    print("\n" + "=" * 60)
    print("📝 Test 1: GPT-2 Tokenizer")
    print("=" * 60)

    tokenizer = HFTokenizerWrapper('gpt2')

    # テストテキスト
    texts = [
        "量子コンピュータは革新的です",
        "The transformer model is powerful",
        "人工知能とディープラーニング"
    ]

    for text in texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"\n  入力: {text}")
        print(f"  トークンID: {tokens[:10]}... (len={len(tokens)})")
        print(f"  デコード: {decoded}")

    print("\n✅ Tokenizer test passed!")


def test_pipeline_basic():
    """パイプラインの基本テスト"""
    print("\n" + "=" * 60)
    print("🧠 Test 2: Pipeline Basic Functionality")
    print("=" * 60)

    # Config
    config = HFQBNNConfig(
        tokenizer_name='gpt2',
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        max_seq_len=128,
        use_qbnn_attention=True
    )

    # Pipeline
    pipeline = HFOpenAIQBNNPipeline(
        config=config,
        use_openai_embedding=False
    )

    # テスト入力
    test_text = "Quantum computing is"
    tokens = pipeline.tokenizer.encode(test_text, add_special_tokens=True)
    token_tensor = torch.tensor([tokens], dtype=torch.long)

    print(f"\n  入力テキスト: '{test_text}'")
    print(f"  トークン数: {len(tokens)}")

    # Forward pass
    with torch.no_grad():
        logits = pipeline(token_tensor)

    print(f"  出力shape: {logits.shape}")
    print(f"  期待shape: (1, {len(tokens)}, {config.vocab_size})")

    assert logits.shape == (1, len(tokens), config.vocab_size), "Shape mismatch!"

    print("\n✅ Pipeline basic test passed!")


def test_generation():
    """テキスト生成のテスト"""
    print("\n" + "=" * 60)
    print("📝 Test 3: Text Generation")
    print("=" * 60)

    # Small config for quick test
    config = HFQBNNConfig(
        tokenizer_name='gpt2',
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        max_seq_len=128,
        use_qbnn_attention=True
    )

    pipeline = HFOpenAIQBNNPipeline(config=config)

    prompts = [
        "The future of",
        "Quantum",
        "AI is"
    ]

    for prompt in prompts:
        output = pipeline.generate(
            prompt,
            max_length=10,
            temperature=1.0
        )
        print(f"\n  入力: '{prompt}'")
        print(f"  出力: '{output}'")

    print("\n✅ Generation test passed!")


def test_mini_training():
    """ミニ学習のテスト"""
    print("\n" + "=" * 60)
    print("🎓 Test 4: Mini Training")
    print("=" * 60)

    config = HFQBNNConfig(
        tokenizer_name='gpt2',
        embed_dim=64,
        num_heads=2,
        num_layers=1,
        hidden_dim=128,
        max_seq_len=64,
        use_qbnn_attention=False  # Faster for testing
    )

    pipeline = HFOpenAIQBNNPipeline(config=config)

    # Minimal training data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Quantum computers use quantum mechanics.",
        "Deep learning is a subset of machine learning.",
    ] * 5

    print(f"\n  学習データ: {len(texts)} samples")
    print(f"  パラメータ数: {pipeline.num_params:,}")

    # Quick training
    pipeline.train_model(
        texts,
        epochs=3,
        batch_size=2,
        lr=1e-3,
        seq_length=32
    )

    # Test generation after training
    output = pipeline.generate("The", max_length=15, temperature=0.8)
    print(f"\n  学習後の生成: '{output}'")

    print("\n✅ Mini training test passed!")


def main():
    """メインテスト"""
    print("=" * 70)
    print("🧪 HF-OpenAI-QBNN Pipeline - Basic Tests")
    print("=" * 70)

    try:
        # Test 1: Tokenizer
        test_tokenizer()

        # Test 2: Pipeline Basic
        test_pipeline_basic()

        # Test 3: Generation
        test_generation()

        # Test 4: Mini Training
        test_mini_training()

        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
