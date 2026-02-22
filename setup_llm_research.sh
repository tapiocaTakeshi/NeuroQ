#!/bin/bash
# neuroQ LLMデータセットリサーチ統合セットアップ

cd "/Users/yuyahiguchi/Program/neuroQ"

echo "🔍 LLM学習データセット詳細調査をneuroQに統合中..."

# リサーチスクリプト実行
python neuroq_llm_datasets_research.py

# 既存データと統合確認
echo "📁 現在のデータセット:"
ls -lh data/

echo "✅ neuroQ LLMデータセットリサーチ完了!"
echo "📈 次ステップ: 高品質データセットをneuroQでファインチューニング"
