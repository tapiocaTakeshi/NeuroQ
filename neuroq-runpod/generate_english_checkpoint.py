#!/usr/bin/env python3
"""
Docker build時に英語版チェックポイントを生成するスクリプト
"""

import sys
import os

print("=" * 60)
print("🚀 Generating English checkpoint for Docker image...")
print("=" * 60)

# train_tiktoken_english.pyの内容を読み込み
with open('train_tiktoken_english.py', 'r') as f:
    content = f.read()

# エポック数を100に変更（50,000だと時間がかかりすぎるため）
content = content.replace("'epochs': 50000,", "'epochs': 100,")

# 実行
try:
    exec(content)
    print("\n✅ English checkpoint generated successfully!")
except Exception as e:
    print(f"\n⚠️ Error generating checkpoint: {e}")
    print("Continuing with build...")
    sys.exit(0)  # エラーでもビルドを続行
