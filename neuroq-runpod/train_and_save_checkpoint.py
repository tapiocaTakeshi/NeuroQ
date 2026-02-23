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
import logging
from pathlib import Path

# ========================================
# ログ設定
# ========================================
# ログレベル: DEBUG=詳細, INFO=通常, WARNING=警告のみ
LOG_LEVEL = logging.DEBUG  # 詳細ログを有効化

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
TOKENIZER_MODEL_PATH = "neuroq_tokenizer.json"
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
    logger.info("========== Step 1: トークナイザー確認 ==========")
    if not os.path.exists(TOKENIZER_MODEL_PATH):
        logger.error(f"トークナイザーモデルが見つかりません: {TOKENIZER_MODEL_PATH}")
        print(f"❌ トークナイザーモデルが見つかりません: {TOKENIZER_MODEL_PATH}")
        print("   先にfugashiトークナイザーファイルを準備してください")
        sys.exit(1)
    file_size = os.path.getsize(TOKENIZER_MODEL_PATH)
    logger.info(f"トークナイザーモデル: {TOKENIZER_MODEL_PATH}")
    logger.info(f"ファイルサイズ: {file_size / 1024:.2f} KB")
    print(f"✅ トークナイザーモデル確認: {TOKENIZER_MODEL_PATH}")
    
    # 2. 学習データ取得
    logger.info("========== Step 2: 学習データ取得 ==========")
    print("\n📚 学習データ取得中...")
    texts = get_training_data(use_huggingface=False)
    print(f"   データ数: {len(texts)} サンプル")
    
    # 学習データの統計情報
    total_chars = sum(len(t) for t in texts)
    avg_len = total_chars / len(texts) if texts else 0
    logger.info(f"学習データ統計:")
    logger.info(f"  - サンプル数: {len(texts)}")
    logger.info(f"  - 総文字数: {total_chars:,}")
    logger.info(f"  - 平均文字数/サンプル: {avg_len:.1f}")
    logger.debug(f"  - 最初のサンプル (50文字): {texts[0][:50] if texts else 'N/A'}...")
    
    # 3. モデル作成
    logger.info("========== Step 3: モデル作成 ==========")
    print("\n🔧 モデル作成中...")
    
    # ========================================
    # 図4-15: GPTモデルのアーキテクチャ準拠
    # ========================================
    # 本番用は num_layers=12 を推奨（学習時間が長くなる）
    # テスト用は num_layers=3 で高速に学習可能
    model_config = {
        'embed_dim': 128,      # 埋め込み次元
        'num_heads': 4,        # アテンションヘッド数
        'num_layers': 3,       # Transformerブロック数（図では12、テスト用は3）
        'num_neurons': 100,    # 脳型量子層のニューロン数
        'max_vocab': 8000,     # 語彙サイズ
    }
    
    print("\n📋 モデル設定（図4-15: GPTモデルのアーキテクチャ準拠）:")
    print(f"   - embed_dim (埋め込み次元): {model_config['embed_dim']}")
    print(f"   - num_heads (アテンションヘッド数): {model_config['num_heads']}")
    print(f"   - num_layers (Transformerブロック数): {model_config['num_layers']}")
    print(f"   - num_neurons (脳型量子ニューロン数): {model_config['num_neurons']}")
    print(f"   - max_vocab (語彙サイズ): {model_config['max_vocab']}")
    
    logger.info(f"モデル設定: {model_config}")
    
    model = NeuroQuantumBrainAI(
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        num_neurons=model_config['num_neurons'],
        max_vocab=model_config['max_vocab'],
        use_fugashi=True
    )
    
    # トークナイザーを明示的にロード
    logger.info(f"トークナイザーをロード: {TOKENIZER_MODEL_PATH}")
    print(f"   トークナイザーをロード: {TOKENIZER_MODEL_PATH}")
    model.tokenizer = NeuroQuantumTokenizer(
        vocab_size=8000,
        model_file=TOKENIZER_MODEL_PATH
    )
    logger.info(f"トークナイザー語彙サイズ: {model.tokenizer.vocab_size}")
    
    # トークン化詳細表示
    print("\n🔤 トークン化テスト:")
    test_texts = ["こんにちは世界", "量子コンピュータは革新的な技術です", "人工知能とは何か"]
    for test_text in test_texts:
        model.tokenizer.print_tokenization(test_text)
    
    # 4. 学習
    logger.info("========== Step 4: 学習開始 ==========")
    print("\n🎓 学習開始...")
    
    train_config = {
        'epochs': 25,
        'batch_size': 16,
        'lr': 0.002,
        'seq_length': 48,
    }
    logger.info(f"学習設定: {train_config}")
    
    model.train(
        texts,
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        lr=train_config['lr'],
        seq_length=train_config['seq_length'],
        verbose=True  # 詳細ログを有効化
    )
    
    # モデルアーキテクチャ表示
    print("\n📐 モデルアーキテクチャ（図4-15準拠）:")
    model.model.print_architecture()
    
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
    logger.info("========== Step 6: テスト生成 ==========")
    print("\n🧪 テスト生成...")
    try:
        test_prompts = ["こんにちは", "量子コンピュータ", "AIとは"]
        for prompt in test_prompts:
            logger.info(f"テスト生成 - 入力: '{prompt}'")
            response = model.generate(prompt, max_length=30)
            logger.info(f"テスト生成 - 出力: '{response}'")
            print(f"   入力: {prompt}")
            print(f"   出力: {response}")
    except Exception as e:
        logger.error(f"テスト生成エラー: {e}")
        print(f"   テスト生成エラー: {e}")


if __name__ == "__main__":
    main()
