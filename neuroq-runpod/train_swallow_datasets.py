#!/usr/bin/env python3
"""
Swallow データセットを使った NeuroQ モデルのローカル学習スクリプト

dataset_configs.py を使用して HuggingFace からデータをロードし、
200エポックで学習を実行する。

対象データセット:
- swallow_nemotron: tokyotech-llm/Swallow-Nemotron-Post-Training-Dataset-v1
- swallow_code_v2:  tokyotech-llm/swallow-code-v2 (stage1-auto-format)
- swallow_math_v2:  tokyotech-llm/swallow-math-v2 (swallow-math-v2-qa)
"""

import os
import sys
import torch
import logging
import random
import argparse
import time
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# パス設定
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    from tiktoken_tokenizer import TikTokenTokenizer
    from model_configs import get_model_config
    from dataset_configs import (
        load_dataset_texts, get_dataset_config, get_training_params
    )
    logger.info("✅ コンポーネントのインポートに成功しました")
except ImportError as e:
    logger.error(f"❌ インポート失敗: {e}")
    sys.exit(1)

ENCODING_NAME = "o200k_base"


def train_with_dataset(dataset_id, model_size="micro", epoch_override=None):
    """
    dataset_configs のデータセットIDを使って学習を実行

    Args:
        dataset_id: dataset_configs.py で定義されたデータセットキー
        model_size: モデルサイズ ('micro', 'small', 'large')
        epoch_override: エポック数の上書き (Noneならデフォルト値を使用)
    """
    ds_config = get_dataset_config(dataset_id)
    params = get_training_params(dataset_id, overrides={
        'epochs': epoch_override,
    })

    epochs = params['epochs']
    batch_size = params['batch_size']
    lr = params['lr']
    seq_length = params['seq_length']

    logger.info("=" * 70)
    logger.info(f"📚 データセット: {ds_config['name']} ({dataset_id})")
    logger.info(f"   {ds_config['description']}")
    if ds_config['id']:
        config_str = f", '{ds_config['config']}'" if ds_config['config'] else ""
        logger.info(f"   load_dataset('{ds_config['id']}'{config_str})")
    logger.info(f"   Epochs: {epochs}, Batch: {batch_size}, LR: {lr}, SeqLen: {seq_length}")
    logger.info("=" * 70)

    # 1. データロード
    logger.info("📥 データセットをロード中...")
    texts = load_dataset_texts(dataset_id)
    logger.info(f"✅ {len(texts):,} 個のテキストを取得しました")

    if not texts:
        logger.error("❌ テキストデータがありません")
        return None

    # 2. トークナイザー
    tokenizer = TikTokenTokenizer(encoding_name=ENCODING_NAME)

    # テキストをトークン化
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special=False)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)

    logger.info(f"✅ 総トークン数: {len(all_tokens):,}")

    # 3. モデル構築
    config_dict = get_model_config(model_size)
    config = NeuroQuantumConfig()
    config.vocab_size = config_dict['vocab_size']
    config.embed_dim = config_dict['embed_dim']
    config.num_heads = config_dict['num_heads']
    config.num_layers = config_dict['num_layers']
    config.max_seq_len = config_dict.get('max_seq_len', 512)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = NeuroQuantum(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ モデル構築完了 ({model_size}): {param_count:,} パラメータ on {device}")

    # 4. データをシーケンスに分割
    all_sequences = []
    for i in range(0, len(all_tokens) - seq_length, seq_length):
        seq = all_tokens[i:i + seq_length]
        if len(seq) == seq_length:
            all_sequences.append(seq)

    random.shuffle(all_sequences)

    val_size = max(1, int(len(all_sequences) * 0.05))
    train_sequences = all_sequences[:-val_size]
    val_sequences = all_sequences[-val_size:]

    logger.info(f"✅ 学習シーケンス数: {len(train_sequences):,}, 検証: {len(val_sequences):,}")

    # 5. 学習ループ
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    min_delta = 0.005
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0
        random.shuffle(train_sequences)

        for i in range(0, len(train_sequences) - batch_size, batch_size):
            batch = train_sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch, device=device)

            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, config.vocab_size), target_ids.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0:
                logger.info(f"   Epoch {epoch+1}/{epochs} | Batch {num_batches} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / max(1, num_batches)

        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for i in range(0, len(val_sequences) - batch_size, batch_size):
                batch = val_sequences[i:i + batch_size]
                batch_tensor = torch.tensor(batch, device=device)

                input_ids = batch_tensor[:, :-1]
                target_ids = batch_tensor[:, 1:]

                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, config.vocab_size), target_ids.reshape(-1))
                total_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / max(1, val_batches)
        elapsed = time.time() - start_time
        logger.info(
            f"⭐ Epoch {epoch+1}/{epochs} 完了 | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"経過: {elapsed:.0f}s"
        )

        # テスト生成（10エポックごと）
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            test_prompts = ["What is", "def ", "1 + 1"]
            logger.info("📝 テスト生成:")
            for prompt in test_prompts:
                with torch.no_grad():
                    input_ids = tokenizer.encode(prompt, add_special=False)
                    generated = input_ids.copy()
                    for _ in range(30):
                        seq = torch.tensor([generated[-seq_length:]], device=device)
                        logits = model(seq)
                        next_token = torch.argmax(logits[0, -1, :]).item()
                        if next_token == tokenizer.eos_id:
                            break
                        generated.append(next_token)
                    response = tokenizer.decode(generated[len(input_ids):], skip_special=True)
                    logger.info(f"   Q: {prompt} -> {response.strip()[:80]}")

        # ベストモデル判定
        is_best = False
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            is_best = True
            logger.info(f"   ✨ Best Loss 更新: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        # チェックポイント保存
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)

        save_checkpoint(model, config_dict, model_size, dataset_id,
                        ckpt_dir / f"neuroq_{model_size}_{dataset_id}_last.pt")

        if is_best:
            save_checkpoint(model, config_dict, model_size, dataset_id,
                            ckpt_dir / f"neuroq_{model_size}_{dataset_id}_best.pt")

    total_time = time.time() - start_time
    logger.info(f"🎉 学習完了: {dataset_id} | 合計時間: {total_time:.0f}s | Best Loss: {best_val_loss:.4f}")

    return ckpt_dir / f"neuroq_{model_size}_{dataset_id}_best.pt"


def save_checkpoint(model, config_dict, model_size, dataset_id, path):
    """チェックポイントを保存"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config_dict,
        'model_size': model_size,
        'dataset_id': dataset_id,
        'tokenizer': {'encoding': ENCODING_NAME},
    }
    torch.save(checkpoint, path)
    logger.info(f"   💾 保存: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swallow データセットで NeuroQ を学習")
    parser.add_argument("--datasets", nargs="+",
                        default=["swallow_nemotron", "swallow_code_v2", "swallow_math_v2"],
                        help="学習するデータセットID")
    parser.add_argument("--model-size", default="micro", choices=["micro", "small", "large"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="エポック数（未指定時はデータセットのデフォルト値）")
    args = parser.parse_args()

    logger.info("🚀 Swallow データセット学習を開始します")
    logger.info(f"   データセット: {args.datasets}")
    logger.info(f"   モデルサイズ: {args.model_size}")
    logger.info(f"   エポック数: {args.epochs or 'データセットデフォルト'}")

    for ds_id in args.datasets:
        try:
            path = train_with_dataset(ds_id, args.model_size, args.epochs)
            if path:
                logger.info(f"✅ {ds_id} 学習完了: {path}")
        except Exception as e:
            logger.error(f"❌ {ds_id} 学習エラー: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n🎉 全データセットの学習が完了しました！")
