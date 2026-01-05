#!/usr/bin/env python3
"""
TikToken トークナイザーを使用したNeuroQモデル学習スクリプト

このスクリプトを実行すると、tiktokenトークナイザーを使った
モデルのチェックポイントを生成します。

使い方:
    python train_tiktoken.py
"""

import os
import sys
import torch
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 現在のディレクトリをパスに追加
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=" * 70)
print("🚀 TikToken ベースの NeuroQ モデル学習")
print("=" * 70)

# インポート
try:
    from neuroquantum_brain import NeuroQuantumBrain, get_training_data
    from tiktoken_tokenizer import TikTokenTokenizer
    print("✅ モジュールをインポートしました")
except ImportError as e:
    print(f"❌ インポートに失敗: {e}")
    sys.exit(1)

# 設定
CHECKPOINT_PATH = "checkpoints/neuroq_tiktoken_checkpoint.pt"
ENCODING_NAME = "cl100k_base"  # GPT-4と同じエンコーディング

# モデル設定（tiktokenの語彙サイズに対応）
MODEL_CONFIG = {
    'vocab_size': 100277,  # cl100k_baseの語彙サイズ
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 3,
    'num_neurons': 100,
}

# 学習設定
TRAIN_CONFIG = {
    'epochs': 10,      # tiktokenは語彙が大きいので少なめで開始
    'batch_size': 16,
    'lr': 0.001,
    'seq_length': 48,
}


def train_with_tiktoken():
    print("\n" + "=" * 50)
    print("📚 Step 1: トークナイザー初期化")
    print("=" * 50)
    
    tokenizer = TikTokenTokenizer(encoding_name=ENCODING_NAME)
    
    # テスト
    test_texts = ["こんにちは", "量子コンピュータ", "人工知能"]
    print("\n🔤 トークン化テスト:")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        print(f"   '{text}' -> {len(tokens)} トークン")
    
    print("\n" + "=" * 50)
    print("📊 Step 2: 学習データ準備")
    print("=" * 50)
    
    texts = get_training_data()
    logger.info(f"学習データ: {len(texts)} サンプル")
    
    # データをトークン化
    print("\n🔄 データトークン化中...")
    all_tokens = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special=False)
        all_tokens.extend(tokens)
        if i < 3 or i == len(texts) - 1:
            logger.debug(f"テキスト[{i}]: {len(tokens)} トークン")
    
    logger.info(f"総トークン数: {len(all_tokens):,}")
    
    print("\n" + "=" * 50)
    print("🧠 Step 3: モデル構築")
    print("=" * 50)
    
    # デバイス選択
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) を使用")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 NVIDIA GPU (CUDA) を使用")
    else:
        device = torch.device("cpu")
        print("💻 CPU を使用")
    
    print(f"\n📋 モデル設定:")
    for key, value in MODEL_CONFIG.items():
        print(f"   - {key}: {value:,}" if isinstance(value, int) else f"   - {key}: {value}")
    
    model = NeuroQuantumBrain(
        vocab_size=MODEL_CONFIG['vocab_size'],
        embed_dim=MODEL_CONFIG['embed_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_neurons=MODEL_CONFIG['num_neurons'],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"総パラメータ数: {total_params:,}")
    
    print("\n" + "=" * 50)
    print("🎓 Step 4: 学習開始")
    print("=" * 50)
    
    print(f"\n📋 学習設定:")
    for key, value in TRAIN_CONFIG.items():
        print(f"   - {key}: {value}")
    
    # シーケンスを作成
    seq_length = TRAIN_CONFIG['seq_length']
    sequences = []
    for i in range(0, len(all_tokens) - seq_length, seq_length):
        sequences.append(all_tokens[i:i + seq_length])
    
    logger.info(f"シーケンス数: {len(sequences):,}")
    
    # 学習ループ
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    
    batch_size = TRAIN_CONFIG['batch_size']
    epochs = TRAIN_CONFIG['epochs']
    
    print("\n🚀 学習ループ開始...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        # シーケンスをシャッフル
        import random
        random.shuffle(sequences)
        
        for i in range(0, len(sequences) - batch_size, batch_size):
            batch = sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch, device=device)
            
            # 入力と正解を作成
            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]
            
            # フォワードパス
            optimizer.zero_grad()
            logits = model(input_ids)
            
            # ロス計算
            loss = criterion(
                logits.reshape(-1, MODEL_CONFIG['vocab_size']),
                target_ids.reshape(-1)
            )
            
            # バックワードパス
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / max(batch_count, 1)
        print(f"   Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}")
        logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")
    
    print("\n   学習完了！")
    
    print("\n" + "=" * 50)
    print("💾 Step 5: チェックポイント保存")
    print("=" * 50)
    
    # ディレクトリを作成
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    
    # チェックポイントを保存
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': MODEL_CONFIG,
        'tokenizer': {
            'type': 'tiktoken',
            'encoding': ENCODING_NAME,
            'vocab_size': tokenizer.vocab_size,
        },
    }
    
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"✅ チェックポイント保存完了: {CHECKPOINT_PATH}")
    
    print("\n" + "=" * 50)
    print("🧪 Step 6: テスト生成")
    print("=" * 50)
    
    model.eval()
    test_prompts = ["こんにちは", "量子コンピュータ", "AIとは"]
    
    for prompt in test_prompts:
        try:
            input_ids = tokenizer.encode(prompt)
            generated = input_ids.copy()
            
            with torch.no_grad():
                for _ in range(30):
                    seq_tensor = torch.tensor([generated[-256:]], device=device)
                    logits = model(seq_tensor)
                    
                    probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    if next_token == tokenizer.eos_id:
                        break
                    generated.append(next_token)
            
            output = tokenizer.decode(generated, skip_special=True)
            print(f"\n   入力: '{prompt}'")
            print(f"   出力: '{output}'")
        except Exception as e:
            print(f"   エラー: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 完了！")
    print("=" * 70)
    print(f"""
次のステップ:
1. チェックポイントを確認: {CHECKPOINT_PATH}
2. チャットで使用: python chat.py --tokenizer tiktoken
""")


if __name__ == "__main__":
    train_with_tiktoken()
