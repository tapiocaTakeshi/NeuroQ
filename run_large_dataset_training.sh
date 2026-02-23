#!/bin/bash
# neuroQ 大規模データセット学習実行

echo "🚀 Starting neuroQ Large Dataset Training..."

# 依存インストール
pip install datasets fugashi unidic-lite accelerate

# データ準備
python neuroq_large_datasets_loader.py

# トレーニング実行（GPU必須）
torchrun --nproc_per_node=8 train_neuroq_large_datasets.py

echo "✅ Large dataset training completed!"
echo "Checkpoints: checkpoints/neuroq_large_datasets_epoch*.pt"
