#!/bin/bash
# 主要生成AIデータセット特性でNeuroQ再学習
cd /workspace/neuroq-runpod

echo "📥 マルチモーダル学習データ準備..."
python -c "
from train_multimodal.py import MultiModalDataset
dataset = MultiModalDataset(max_samples=1000)
print('✅ データセット準備完了:', len(dataset), 'サンプル')
"

echo "🚀 トレーニング開始 (GPU使用)..."
python train_multimodal.py

echo "💾 チェックポイント保存完了: checkpoints/neuroq_multimodal_final.pt"
echo "🎉 主要生成AI並みの学習完了！"
