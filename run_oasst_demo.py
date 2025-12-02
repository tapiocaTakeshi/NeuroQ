#!/usr/bin/env python3
"""
OpenAssistant/oasst1 でAPQB生成AIを学習・テスト
"""

import sys
sys.path.insert(0, '/Users/yuyahiguchi/Program/Qubit')

from apqb_generative_ai_v2 import (
    APQBGenerativeAI, 
    load_openassistant_data,
    SubwordTokenizer,
    CharTokenizer
)

def main():
    print("=" * 70)
    print("🧠⚛️ APQB Generative AI - OpenAssistant/oasst1 学習")
    print("=" * 70)
    
    # OpenAssistantデータセットを読み込み（少なめのサンプル数で高速化）
    print("\n📥 データセット読み込み中...")
    training_text = load_openassistant_data(max_samples=1000, lang='all')
    
    if training_text is None:
        print("❌ データセット読み込み失敗")
        return
    
    print(f"📊 学習テキストサイズ: {len(training_text):,} 文字")
    
    # モデル作成と学習
    print("\n🏗️ モデル構築中...")
    ai = APQBGenerativeAI(
        embed_dim=96,       # 少し小さめで高速化
        num_heads=6,
        num_layers=3,
        dropout_r=-0.3,
        max_vocab_size=2000,
        use_subword=True
    )
    
    print("\n📚 学習開始...")
    losses = ai.train(
        training_text,
        epochs=25,          # 高速化のため少なめ
        batch_size=24,
        seq_len=96
    )
    
    # 統計情報
    stats = ai.get_stats()
    print(f"\n📈 モデル統計:")
    print(f"   語彙サイズ: {stats['vocab_size']:,} トークン")
    print(f"   パラメータ数: {stats['total_params']:,}")
    
    # テキスト生成テスト
    prompts = [
        "Hello",
        "How can I",
        "The best way to",
        "Programming is",
        "AI will"
    ]
    
    print("\n" + "=" * 70)
    print("📝 テキスト生成結果")
    print("=" * 70)
    
    for prompt in prompts:
        print(f"\n🔮 プロンプト: 「{prompt}」")
        print("-" * 50)
        
        text = ai.generate(
            prompt, 
            max_length=100, 
            temperature=0.8, 
            top_k=40,
            quantum_sampling=True
        )
        print(f"{text}")
    
    # 量子サンプリング比較
    print("\n" + "=" * 70)
    print("⚛️ 量子サンプリング vs 通常サンプリング")
    print("=" * 70)
    
    test_prompt = "The future"
    
    print(f"\nプロンプト: 「{test_prompt}」")
    print("\n[量子サンプリング ON]")
    for i in range(3):
        text = ai.generate(test_prompt, max_length=60, temperature=0.9, quantum_sampling=True)
        print(f"  {i+1}. {text}")
    
    print("\n[量子サンプリング OFF]")
    for i in range(3):
        text = ai.generate(test_prompt, max_length=60, temperature=0.9, quantum_sampling=False)
        print(f"  {i+1}. {text}")
    
    print("\n✅ 完了！")
    
    return ai


if __name__ == "__main__":
    main()

