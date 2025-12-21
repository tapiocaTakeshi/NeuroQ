#!/usr/bin/env python3
"""
日本語テキスト生成テストスクリプト
Test script for Japanese text generation improvements
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from neuroquantum_layered import NeuroQuantumAI
    print("✅ Successfully imported NeuroQuantumAI")
except ImportError as e:
    print(f"❌ Failed to import NeuroQuantumAI: {e}")
    sys.exit(1)


def test_japanese_generation():
    """日本語生成テスト"""
    print("\n" + "=" * 70)
    print("🧪 日本語テキスト生成テスト")
    print("=" * 70)

    # テストプロンプト
    test_prompts = [
        "量子コンピュータについて教えて",
        "人工知能とは",
        "ニューラルネットワークの仕組み",
    ]

    try:
        # モデル初期化（事前学習済みモデルを使用）
        print("\n🔄 モデルを初期化中...")
        model = NeuroQuantumAI(
            embed_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            max_seq_len=256,
            dropout=0.1,
            lambda_entangle=0.5,
        )

        # 簡易学習を実行（モデルは毎回初期化）
        print("⚠️ 簡易学習を実行します。")
        # 簡易学習データ
        sample_data = [
            "量子コンピュータは、量子力学の原理を利用した次世代のコンピュータです。",
            "人工知能は人間の知能を模倣するコンピュータシステムです。",
            "ニューラルネットワークは、人間の脳の神経細胞の働きを模倣した計算モデルです。",
            "機械学習はデータからパターンを学習するAIの手法です。",
        ] * 5
        model.train(sample_data, epochs=3)

        print("✅ モデル初期化完了\n")

        # 各プロンプトでテスト
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'─' * 70}")
            print(f"📝 テスト {i}/{len(test_prompts)}")
            print(f"{'─' * 70}")
            print(f"プロンプト: {prompt}")
            print()

            # 改善されたパラメータで生成
            print("🔄 生成中（改善版パラメータ）...")
            result = model.generate(
                prompt=prompt,
                max_length=100,
                temp_min=0.5,           # より保守的な温度
                temp_max=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=2.0,  # 強力な繰り返しペナルティ
                no_repeat_ngram_size=3,  # 3-gramの繰り返し防止
            )

            print("生成結果:")
            print("─" * 70)
            print(result)
            print("─" * 70)

            # 繰り返しチェック
            words = result.split()
            if len(words) > 0:
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1

                repeated_words = [(word, count) for word, count in word_counts.items() if count > 3]
                if repeated_words:
                    print(f"\n⚠️ 繰り返しが検出されました:")
                    for word, count in repeated_words:
                        print(f"   '{word}': {count}回")
                else:
                    print(f"\n✅ 過度な繰り返しは検出されませんでした")

        print("\n" + "=" * 70)
        print("✅ テスト完了")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_japanese_generation()
    sys.exit(0 if success else 1)
