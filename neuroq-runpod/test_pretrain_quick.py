#!/usr/bin/env python3
"""
簡易テスト: pretrain_openai.pyの動作確認
"""

import sys
import torch
from neuroquantum_layered import NeuroQuantumAI

print("=" * 60)
print("🧪 Quick Pretrain Test")
print("=" * 60)

# 簡易データ
test_texts = [
    "こんにちは。ニューロQです。",
    "量子コンピュータは次世代の計算機です。",
    "機械学習でデータから学習します。",
] * 10  # 30テキスト

print(f"\n📊 テストデータ: {len(test_texts)} テキスト")

# モデル作成
print("\n🧠 モデル作成...")
model = NeuroQuantumAI(
    embed_dim=128,
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    max_seq_len=64,
    dropout=0.1,
    lambda_entangle=0.5
)

print("\n🚀 学習開始（2エポック）...")
try:
    model.train(
        test_texts,
        epochs=2,
        seq_len=32,
        batch_size=4,
        lr=0.001
    )
    print("\n✅ 学習完了！")

    # テスト生成
    result = model.generate("こんにちは", max_length=20)
    print(f"\n📝 生成テスト: {result}")

    print("\n✅ すべてのテスト成功！")
    sys.exit(0)

except Exception as e:
    print(f"\n❌ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
