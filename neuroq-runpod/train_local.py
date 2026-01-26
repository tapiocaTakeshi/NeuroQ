#!/usr/bin/env python3
"""
ローカル学習スクリプト - Hugging Face会話データセットを使用
Modal認証なしでローカルで学習を実行
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import random
from torch.cuda.amp import autocast, GradScaler

# データセットローダーをインポート
from modal_app import (
    load_ultrachat_data,
    load_dolly_ja_data,
    load_lima_data,
    load_oasst1_data,
    load_alpaca_data,
    load_openchat_sharegpt_data,
)


def train_local(
    model_size: str = "micro",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 0.0005,
    max_samples: int = 1000,
):
    """ローカルで会話データセットを使って学習"""
    print("=" * 70)
    print(f"🚀 NeuroQ {model_size.upper()} ローカル学習")
    print("=" * 70)

    # デバイス
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 Device: {device}")
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()

    # モジュールインポート
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    from tiktoken_tokenizer import TikTokenTokenizer
    from model_configs import get_model_config

    # トークナイザー
    config = get_model_config(model_size)
    model_vocab_size = config["vocab_size"]
    tokenizer = TikTokenTokenizer(encoding_name="o200k_base", max_vocab=model_vocab_size)
    print(f"📚 語彙サイズ: {tokenizer.vocab_size:,}")

    # 会話データセットをロード
    print("\n📊 会話データセットをロード中...")
    texts = []

    # UltraChat - マルチターン会話
    print("\n--- UltraChat ---")
    ultrachat_texts = load_ultrachat_data(max_samples=max_samples)
    texts.extend(ultrachat_texts * 2)
    print(f"   ✅ ultrachat: {len(ultrachat_texts)} × 2 = {len(ultrachat_texts) * 2}")

    # 日本語Dolly
    print("\n--- Dolly Japanese ---")
    dolly_ja_texts = load_dolly_ja_data(max_samples=max_samples)
    texts.extend(dolly_ja_texts * 3)
    print(f"   ✅ dolly-ja: {len(dolly_ja_texts)} × 3 = {len(dolly_ja_texts) * 3}")

    # LIMA - 高品質会話
    print("\n--- LIMA ---")
    lima_texts = load_lima_data(max_samples=min(max_samples, 1000))
    texts.extend(lima_texts * 5)
    print(f"   ✅ lima: {len(lima_texts)} × 5 = {len(lima_texts) * 5}")

    # OASST1 - 会話データ
    print("\n--- OASST1 ---")
    oasst1_texts = load_oasst1_data(max_samples=max_samples)
    texts.extend(oasst1_texts * 2)
    print(f"   ✅ oasst1: {len(oasst1_texts)} × 2 = {len(oasst1_texts) * 2}")

    print(f"\n📊 総テキスト数: {len(texts)}")

    if len(texts) == 0:
        print("❌ データセットのロードに失敗しました")
        return

    # トークン化
    seq_length = 64
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special=False)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    print(f"📊 総トークン数: {len(all_tokens):,}")

    # モデル構築
    print("\n🧠 モデル構築...")
    nq_config = NeuroQuantumConfig()
    nq_config.vocab_size = config["vocab_size"]
    nq_config.embed_dim = config["embed_dim"]
    nq_config.num_heads = config["num_heads"]
    nq_config.num_layers = config["num_layers"]
    nq_config.max_seq_len = config.get("max_seq_len", 512)
    nq_config.dropout = config.get("dropout", 0.1)

    model = NeuroQuantum(nq_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   {config['name']}: {total_params:,} パラメータ ({total_params/1e6:.1f}M)")

    # シーケンス作成
    sequences = []
    for i in range(0, len(all_tokens) - seq_length, seq_length // 2):
        sequences.append(all_tokens[i : i + seq_length])
    print(f"📊 シーケンス数: {len(sequences):,}")

    # 学習ループ
    print("\n🎓 学習開始...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    use_amp = device == "cuda"
    scaler = GradScaler() if use_amp else None

    total_batches = (len(sequences) - batch_size) // batch_size
    print(f"   バッチ数/エポック: {total_batches:,}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        random.shuffle(sequences)

        for i in range(0, len(sequences) - batch_size, batch_size):
            batch = sequences[i : i + batch_size]
            batch_tensor = torch.tensor(batch, device=device)

            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    logits = model(input_ids)
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_count % 50 == 0:
                avg_loss = total_loss / batch_count
                print(f"   Epoch {epoch+1}/{epochs} | Batch {batch_count}/{total_batches} | Loss: {avg_loss:.4f}")

        avg_loss = total_loss / max(batch_count, 1)
        print(f"✅ Epoch {epoch+1}/{epochs} 完了 | 平均Loss: {avg_loss:.4f}")

    # チェックポイント保存
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"neuroq_{model_size}_conversation.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "tokenizer": {
            "type": "tiktoken",
            "encoding": "o200k_base",
            "vocab_size": tokenizer.vocab_size,
        },
        "model_size": model_size,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\n💾 チェックポイント保存: {checkpoint_path}")

    print("\n🎉 学習完了！")
    return checkpoint_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuroQローカル学習")
    parser.add_argument("--model-size", default="micro", choices=["micro", "small", "large"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--max-samples", type=int, default=1000)

    args = parser.parse_args()

    train_local(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
    )
