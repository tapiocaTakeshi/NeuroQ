#!/usr/bin/env python3
"""
NeuroQ モデル学習 & チェックポイント保存スクリプト
================================================

このスクリプトを RunPod デプロイ前にローカルで実行して、
学習済みのチェックポイントを作成します。

使い方:
    python train_and_save_checkpoint.py

生成されるファイル:
    - checkpoints/neuroq_checkpoint.pt: 学習済みモデルのチェックポイント
"""

import os
import sys
import torch
from pathlib import Path

# 現在のディレクトリをパスに追加
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=" * 70)
print("🚀 NeuroQ モデル学習 & チェックポイント保存")
print("=" * 70)

# NeuroQuantumBrainAI をインポート
try:
    from neuroquantum_brain import NeuroQuantumBrainAI, NeuroQuantumBrain, get_training_data
    from neuroquantum_layered import NeuroQuantumTokenizer
    print("✅ neuroquantum_brain.py をインポートしました")
except ImportError as e:
    print(f"❌ インポートに失敗: {e}")
    sys.exit(1)

# 設定
TOKENIZER_MODEL_PATH = "neuroq_tokenizer.model"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "neuroq_checkpoint.pt")

# デバイス選択
if torch.backends.mps.is_available():
    device = "mps"
    print("🍎 Apple Silicon GPU (MPS) を使用")
elif torch.cuda.is_available():
    device = "cuda"
    print("🎮 NVIDIA GPU (CUDA) を使用")
else:
    device = "cpu"
    print("💻 CPU を使用")


def save_checkpoint(model_instance, checkpoint_path: str):
    """チェックポイントを保存"""
    try:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model_instance.model.state_dict(),
            'config': {
                'embed_dim': model_instance.embed_dim,
                'num_heads': model_instance.num_heads,
                'num_layers': model_instance.num_layers,
                'num_neurons': model_instance.num_neurons,
                'max_vocab': model_instance.max_vocab,
            },
            'tokenizer_path': TOKENIZER_MODEL_PATH,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ チェックポイント保存完了: {checkpoint_path}")
        return True
        
    except Exception as e:
        print(f"❌ チェックポイント保存エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン処理"""
    
    # 1. トークナイザーの確認
    if not os.path.exists(TOKENIZER_MODEL_PATH):
        print(f"❌ トークナイザーモデルが見つかりません: {TOKENIZER_MODEL_PATH}")
        print("   先に train_sentencepiece_tokenizer.py を実行してください")
        sys.exit(1)
    print(f"✅ トークナイザーモデル確認: {TOKENIZER_MODEL_PATH}")
    
    # 2. 学習データ取得
    print("\n📚 学習データ取得中...")
    texts = get_training_data(use_huggingface=False)
    print(f"   データ数: {len(texts)} サンプル")
    
    # 3. モデル作成
    print("\n🔧 モデル作成中...")
    model = NeuroQuantumBrainAI(
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        num_neurons=100,
        max_vocab=8000,
        use_sentencepiece=True
    )
    
    # トークナイザーを明示的にロード
    print(f"   トークナイザーをロード: {TOKENIZER_MODEL_PATH}")
    model.tokenizer = NeuroQuantumTokenizer(
        vocab_size=8000,
        model_file=TOKENIZER_MODEL_PATH
    )
    
    # 4. 学習
    print("\n🎓 学習開始...")
    model.train(
        texts,
        epochs=25,
        batch_size=16,
        lr=0.002,
        seq_length=48
    )
    
    # 5. チェックポイント保存
    print("\n💾 チェックポイント保存中...")
    if save_checkpoint(model, CHECKPOINT_PATH):
        print("\n" + "=" * 70)
        print("✅ 完了！")
        print("=" * 70)
        print(f"\n次のステップ:")
        print(f"1. チェックポイントを確認: {CHECKPOINT_PATH}")
        print(f"2. Docker イメージをビルド: docker build -t neuroq-runpod .")
        print(f"3. RunPod にデプロイ")
    else:
        print("\n❌ チェックポイント保存に失敗しました")
        sys.exit(1)
    
    # 6. テスト生成
    print("\n🧪 テスト生成...")
    try:
        response = model.generate("こんにちは", max_length=30)
        print(f"   入力: こんにちは")
        print(f"   出力: {response}")
    except Exception as e:
        print(f"   テスト生成エラー: {e}")


if __name__ == "__main__":
    main()
