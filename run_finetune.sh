#!/bin/bash
# NeuroQ LoRAファインチューニング即実行

# 依存関係インストール
pip install -r requirements_finetune.txt -q

# データ確認
ls -la data/*.txt | head -5

# 学習実行 (30分)
python finetune_neuroq_lora.py

# テスト実行
python test_neuroq_lora.py
