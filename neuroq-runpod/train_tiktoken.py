#!/usr/bin/env python3
"""
TikToken トークナイザーを使用したNeuroQモデル学習スクリプト

このスクリプトを実行すると、TikTokenトークナイザー（OpenAI GPT-4o と同じ）を使った
モデルのチェックポイントを生成します。

使い方:
    python train_tiktoken.py                    # Microモデル（デフォルト）
    python train_tiktoken.py --model-size small # Smallモデル（embed_dim=1536）
    python train_tiktoken.py -m large           # Largeモデル（embed_dim=3072）
    python train_tiktoken.py -m small --epochs 20
"""

import os
import sys
import torch
import logging
import argparse
import random
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
print("TikToken ベースの NeuroQ モデル学習")
print("=" * 70)

# インポート
try:
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    from neuroquantum_brain import get_training_data
    from model_configs import AVAILABLE_MODELS, get_model_config, get_checkpoint_path
    from tiktoken_tokenizer import TikTokenTokenizer
    print("モジュールをインポートしました")
    MODEL_CONFIGS_AVAILABLE = True
except ImportError as e:
    print(f"一部のインポートに失敗: {e}")
    MODEL_CONFIGS_AVAILABLE = False
    try:
        from neuroquantum_brain import NeuroQuantumBrain, get_training_data
        from tiktoken_tokenizer import TikTokenTokenizer
        print("基本モジュールをインポートしました（Microモード）")
    except ImportError as e2:
        print(f"インポートに失敗: {e2}")
        sys.exit(1)

# TikToken設定
TIKTOKEN_ENCODING = "o200k_base"  # GPT-4o と同じエンコーディング
VOCAB_SIZE = 200019  # o200k_base の語彙サイズ（制限する場合は MAX_VOCAB を設定）
MAX_VOCAB = 32000  # 実際に使用する語彙サイズ（メモリ節約のため制限）

# デフォルト学習設定
DEFAULT_TRAIN_CONFIG = {
    'epochs': 30,
    'batch_size': 8,
    'lr': 0.0005,
    'seq_length': 64,
}


def train_with_tiktoken(model_size='micro', epochs=None, batch_size=None, lr=None):
    """
    TikTokenトークナイザーでモデルを学習

    Args:
        model_size: 'micro', 'small', 'large'
        epochs: エポック数（Noneならデフォルト）
        batch_size: バッチサイズ
        lr: 学習率
    """

    # 学習設定
    train_config = DEFAULT_TRAIN_CONFIG.copy()
    if epochs is not None:
        train_config['epochs'] = epochs
    if batch_size is not None:
        train_config['batch_size'] = batch_size
    if lr is not None:
        train_config['lr'] = lr

    # モデルサイズに応じて調整
    if model_size in ['small', 'large']:
        # 大きいモデルは学習に時間がかかるのでバッチサイズを小さく
        if batch_size is None:
            train_config['batch_size'] = 4 if model_size == 'small' else 2
        if lr is None:
            train_config['lr'] = 0.0003  # 大きいモデルは学習率を下げる

    print(f"\nモデルサイズ: {model_size.upper()}")

    print("\n" + "=" * 50)
    print("Step 1: TikTokenトークナイザー初期化")
    print("=" * 50)

    # TikTokenトークナイザーを初期化
    tokenizer = TikTokenTokenizer(encoding_name=TIKTOKEN_ENCODING, max_vocab=MAX_VOCAB)
    logger.info(f"TikTokenトークナイザー: {TIKTOKEN_ENCODING}, 語彙サイズ: {tokenizer.vocab_size:,} (制限: {MAX_VOCAB:,})")

    # テスト
    test_texts = ["Hello", "こんにちは", "AI"]
    print("\nトークン化テスト:")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        print(f"   '{text}' -> {len(tokens)} トークン")

    print("\n" + "=" * 50)
    print("Step 2: 学習データ準備")
    print("=" * 50)

    texts = get_training_data()
    logger.info(f"学習データ: {len(texts)} サンプル")

    # データをトークン化
    print("\nデータトークン化中...")
    all_tokens = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special=False)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
        if i < 3:
            logger.debug(f"テキスト[{i}]: {len(tokens)} トークン")

    logger.info(f"総トークン数: {len(all_tokens):,}")

    print("\n" + "=" * 50)
    print("Step 3: モデル構築")
    print("=" * 50)

    # デバイス選択
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon GPU (MPS) を使用")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("NVIDIA GPU (CUDA) を使用")
    else:
        device = torch.device("cpu")
        print("CPU を使用")

    # モデル設定を取得
    if MODEL_CONFIGS_AVAILABLE and model_size in ['small', 'large']:
        config = get_model_config(model_size)
        checkpoint_path = get_checkpoint_path(model_size)

        # vocab_sizeをTikTokenに合わせる
        config['vocab_size'] = MAX_VOCAB

        print(f"\nモデル設定 ({config['name']}):")
        print(f"   - vocab_size: {config['vocab_size']:,}")
        print(f"   - embed_dim: {config['embed_dim']:,}")
        print(f"   - num_heads: {config['num_heads']}")
        print(f"   - num_layers: {config['num_layers']}")

        # NeuroQuantumモデルを構築
        nq_config = NeuroQuantumConfig()
        nq_config.vocab_size = config['vocab_size']
        nq_config.embed_dim = config['embed_dim']
        nq_config.num_heads = config['num_heads']
        nq_config.num_layers = config['num_layers']
        nq_config.max_seq_len = config.get('max_seq_len', 512)
        nq_config.dropout = config.get('dropout', 0.1)

        model = NeuroQuantum(nq_config).to(device)
        model_config = config

    else:
        # Microモデル（従来）
        model_config = {
            'vocab_size': MAX_VOCAB,
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 5,
        }
        checkpoint_path = "checkpoints/neuroq_tiktoken_checkpoint.pt"

        print(f"\nモデル設定 (Micro):")
        for key, value in model_config.items():
            print(f"   - {key}: {value:,}" if isinstance(value, int) else f"   - {key}: {value}")

        nq_config = NeuroQuantumConfig()
        nq_config.vocab_size = model_config['vocab_size']
        nq_config.embed_dim = model_config['embed_dim']
        nq_config.num_heads = model_config['num_heads']
        nq_config.num_layers = model_config['num_layers']
        nq_config.max_seq_len = 256
        nq_config.dropout = 0.1

        model = NeuroQuantum(nq_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"総パラメータ数: {total_params:,} ({total_params/1e6:.1f}M)")

    print("\n" + "=" * 50)
    print("Step 4: 学習開始")
    print("=" * 50)

    print(f"\n学習設定:")
    for key, value in train_config.items():
        print(f"   - {key}: {value}")

    # シーケンスを作成
    seq_length = train_config['seq_length']
    sequences = []
    for i in range(0, len(all_tokens) - seq_length, seq_length // 2):
        sequences.append(all_tokens[i:i + seq_length])

    logger.info(f"シーケンス数: {len(sequences):,}")

    # 学習ループ
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = train_config['batch_size']
    epochs = train_config['epochs']

    print("\n学習ループ開始...")

    # 総バッチ数を計算
    total_batches = (len(sequences) - batch_size) // batch_size
    print(f"   エポックあたりのバッチ数: {total_batches:,}")
    print(f"   大きなモデルでは1バッチに数秒かかることがあります\n")

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        # シーケンスをシャッフル
        random.shuffle(sequences)

        # 進捗表示間隔（10%ごと、最低10バッチごと）
        progress_interval = max(10, total_batches // 10)

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
                logits.reshape(-1, model_config['vocab_size']),
                target_ids.reshape(-1)
            )

            # バックワードパス
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # バッチ単位の進捗表示
            if batch_count % progress_interval == 0 or batch_count == 1:
                progress = batch_count / total_batches * 100
                current_avg_loss = total_loss / batch_count
                print(f"   Epoch {epoch + 1}/{epochs} - Batch {batch_count:,}/{total_batches:,} ({progress:.0f}%) - Loss: {current_avg_loss:.4f}")

        avg_loss = total_loss / max(batch_count, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(f"   Epoch {epoch + 1}/{epochs} 完了: Loss={avg_loss:.4f}")
        logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")

    print(f"\n   学習完了！ベストLoss: {best_loss:.4f}")

    print("\n" + "=" * 50)
    print("Step 5: チェックポイント保存")
    print("=" * 50)

    # ディレクトリを作成
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # チェックポイントを保存
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'tokenizer': {
            'type': 'tiktoken',
            'encoding': TIKTOKEN_ENCODING,
            'vocab_size': tokenizer.vocab_size,
            'max_vocab': MAX_VOCAB,
        },
        'model_size': model_size,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"チェックポイント保存完了: {checkpoint_path}")

    print("\n" + "=" * 50)
    print("Step 6: テスト生成")
    print("=" * 50)

    model.eval()
    test_prompts = ["<USER> こんにちは！ <ASSISTANT>", "<USER> AIとは何ですか？ <ASSISTANT>"]

    for prompt in test_prompts:
        try:
            input_ids = tokenizer.encode(prompt)
            generated = input_ids.copy()

            with torch.no_grad():
                for _ in range(50):
                    seq_tensor = torch.tensor([generated[-256:]], device=device)
                    logits = model(seq_tensor)

                    probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)

                    # Top-k sampling
                    top_k = 50
                    top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                    top_probs = top_probs / top_probs.sum()
                    idx = torch.multinomial(top_probs, num_samples=1)
                    next_token = top_indices[idx].item()

                    if next_token == tokenizer.eos_id:
                        break
                    generated.append(next_token)

            output = tokenizer.decode(generated, skip_special=True)
            # <ASSISTANT>以降を抽出
            if '<ASSISTANT>' in output:
                response = output.split('<ASSISTANT>')[-1].strip()
            else:
                response = output

            q = prompt.replace("<USER> ", "").replace(" <ASSISTANT>", "")
            print(f"\n   Q: '{q}'")
            print(f"   A: '{response[:100]}'")
        except Exception as e:
            print(f"   エラー: {e}")

    print("\n" + "=" * 70)
    print("完了！")
    print("=" * 70)
    print(f"""
次のステップ:
1. チェックポイントを確認: {checkpoint_path}
2. チャットで使用: python chat.py --model-size {model_size} --tokenizer tiktoken
""")


def main():
    parser = argparse.ArgumentParser(
        description='TikToken ベースの NeuroQ モデル学習'
    )
    parser.add_argument(
        '--model-size', '-m',
        choices=['micro', 'small', 'large'],
        default='micro',
        help='モデルサイズ: micro(128), small(1536), large(3072)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='学習エポック数'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=None,
        help='バッチサイズ'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='学習率'
    )

    args = parser.parse_args()

    train_with_tiktoken(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == "__main__":
    main()
