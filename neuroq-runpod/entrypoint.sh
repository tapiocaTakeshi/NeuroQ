#!/bin/bash
# NeuroQ Docker Entrypoint Script
# ================================
# Docker起動時の初期化スクリプト

set -e

echo "========================================"
echo "🚀 NeuroQ コンテナ起動中..."
echo "========================================"
echo ""

# トークナイザーファイルの確認
TOKENIZER_MODEL="/app/neuroq_tokenizer.model"
TOKENIZER_VOCAB="/app/neuroq_tokenizer.vocab"

if [ -f "$TOKENIZER_MODEL" ]; then
    echo "✅ トークナイザーモデル: $TOKENIZER_MODEL"
else
    echo "⚠️  トークナイザーモデルが見つかりません: $TOKENIZER_MODEL"
fi

if [ -f "$TOKENIZER_VOCAB" ]; then
    echo "✅ トークナイザー語彙: $TOKENIZER_VOCAB"
else
    echo "⚠️  トークナイザー語彙が見つかりません: $TOKENIZER_VOCAB"
fi

echo ""
echo "========================================"
echo "🚀 NeuroQ ハンドラーを起動中..."
echo "========================================"
echo ""

# メインのハンドラーを実行
exec python -u handler.py
