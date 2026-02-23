#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oasst1-89k-ja User/Assistant 形式データで学習

kunishou/oasst1-89k-ja を変換したデータを使用してモデルを学習します。
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional

# パスを追加
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

neuroq_runpod_dir = str(Path(__file__).parent / "neuroq-runpod")
if neuroq_runpod_dir not in sys.path:
    sys.path.insert(0, neuroq_runpod_dir)

# fugashi
try:
    import fugashi
except ImportError:
    print("fugashi がインストールされていません。pip install fugashi unidic-lite を実行してください。")
    sys.exit(1)


def load_conversation_data(file_path: str) -> List[str]:
    """
    User/Assistant 形式の会話データを読み込む

    形式:
    User: こんにちは
    Assistant: こんにちは！今日はどうしましたか？

    User: 量子コンピュータって何？
    Assistant: 量子力学を使って計算するコンピュータで…
    """
    print(f"会話データ読み込み: {file_path}")

    if not os.path.exists(file_path):
        print(f"ファイルが見つかりません: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 空行で区切られた会話ブロックに分割
    conversations = []
    blocks = content.split('\n\n')

    for block in blocks:
        block = block.strip()
        if block and 'User:' in block and 'Assistant:' in block:
            conversations.append(block)

    print(f"{len(conversations)} 個の会話を読み込みました")

    # サンプルを表示
    print("\nサンプル（最初の2個）:")
    for i, conv in enumerate(conversations[:2], 1):
        preview = conv[:200] + "..." if len(conv) > 200 else conv
        print(f"--- 例 {i} ---")
        print(preview)
        print()

    return conversations


class SimpleTokenizer:
    """fugashi を使用した日本語形態素解析トークナイザー"""

    def __init__(self, model_path: str):
        import json
        self.tagger = fugashi.Tagger()

        # 語彙ファイル（JSON）を読み込み
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab = data['vocab']
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab.get('<pad>', 0)
        self.unk_id = self.vocab.get('<unk>', 1)
        self.bos_id = self.vocab.get('<s>', 2)
        self.eos_id = self.vocab.get('</s>', 3)

        print(f"トークナイザー読み込み完了")
        print(f"  語彙サイズ: {self.vocab_size}")
        print(f"  PAD ID: {self.pad_id}, UNK ID: {self.unk_id}")
        print(f"  BOS ID: {self.bos_id}, EOS ID: {self.eos_id}")

    def encode(self, text: str) -> List[int]:
        words = self.tagger(text)
        return [self.vocab.get(word.surface, self.unk_id) for word in words]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.id_to_token.get(tid, '<unk>') for tid in tokens])


class SimpleTransformer(nn.Module):
    """シンプルなTransformerモデル"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # 埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer エンコーダ層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 出力層
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

        # 因果マスク用
        self.register_buffer('causal_mask', None)

        # パラメータ初期化
        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)

    def _get_causal_mask(self, seq_len: int, device):
        """因果マスクを生成"""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) のトークンID

        Returns:
            (batch_size, seq_len, vocab_size) のロジット
        """
        batch_size, seq_len = x.shape
        device = x.device

        # 位置エンベディング
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # 埋め込み
        x = self.token_embedding(x) + self.position_embedding(positions)

        # 因果マスク
        causal_mask = self._get_causal_mask(seq_len, device)

        # Transformer
        x = self.transformer(x, mask=causal_mask)

        # 出力
        x = self.layer_norm(x)
        logits = self.output_layer(x)

        return logits

    def generate(
        self,
        prompt_tokens: List[int],
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_id: int = 3,
    ) -> List[int]:
        """テキスト生成"""
        self.eval()
        device = next(self.parameters()).device

        tokens = list(prompt_tokens)

        with torch.no_grad():
            for _ in range(max_length):
                if len(tokens) >= self.max_seq_len:
                    break

                x = torch.tensor([tokens[-self.max_seq_len:]], dtype=torch.long, device=device)
                logits = self(x)
                next_logits = logits[0, -1, :] / temperature

                # Top-k サンプリング
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) サンプリング
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                tokens.append(next_token)

                if next_token == eos_id:
                    break

        return tokens


def train_model(
    conversations: List[str],
    tokenizer: SimpleTokenizer,
    epochs: int = 5,
    batch_size: int = 8,
    seq_len: int = 128,
    lr: float = 0.0003,
    checkpoint_path: str = "checkpoints/oasst1_ja_model.pt"
):
    """
    モデルを学習

    Args:
        conversations: 会話データのリスト
        tokenizer: トークナイザー
        epochs: エポック数
        batch_size: バッチサイズ
        seq_len: シーケンス長
        lr: 学習率
        checkpoint_path: チェックポイント保存先
    """
    print("\n" + "=" * 70)
    print("モデル学習開始")
    print("=" * 70)

    # デバイス選択
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("NVIDIA GPU (CUDA) を使用")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon GPU (MPS) を使用")
    else:
        device = torch.device("cpu")
        print("CPU を使用")

    # データ準備
    print("\nデータ準備...")
    all_tokens = []

    for conv in conversations:
        tokens = tokenizer.encode(conv)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)  # 会話の終わり

    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"  総トークン数: {len(all_tokens):,}")

    # シーケンス作成（50%オーバーラップ）
    sequences = []
    step = max(1, seq_len // 2)
    for i in range(0, len(all_tokens) - seq_len - 1, step):
        x = all_tokens[i:i + seq_len]
        y = all_tokens[i + 1:i + seq_len + 1]
        if len(x) == seq_len and len(y) == seq_len:
            sequences.append((x, y))

    print(f"  シーケンス数: {len(sequences):,}")

    # モデル作成
    print("\nモデル構築...")
    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        max_seq_len=seq_len + 32,
        dropout=0.1
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  総パラメータ数: {num_params:,}")

    # 学習設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # チェックポイントディレクトリ作成
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # 学習ループ
    print("\n学習ループ開始...")
    print(f"  エポック数: {epochs}")
    print(f"  バッチサイズ: {batch_size}")
    print(f"  シーケンス長: {seq_len}")
    print(f"  学習率: {lr}")
    print()

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # シャッフル
        indices = np.random.permutation(len(sequences))

        for i in range(0, len(sequences), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) == 0:
                continue

            x_batch = torch.stack([sequences[idx][0] for idx in batch_indices]).to(device)
            y_batch = torch.stack([sequences[idx][1] for idx in batch_indices]).to(device)

            optimizer.zero_grad()
            logits = model(x_batch)

            loss = criterion(
                logits.view(-1, tokenizer.vocab_size),
                y_batch.view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)

        # 進捗表示
        print(f"  Epoch {epoch + 1:3d}/{epochs}: Loss = {avg_loss:.4f}")

        # ベストモデルを保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': tokenizer.vocab_size,
            }, checkpoint_path)
            print(f"       -> チェックポイント保存: {checkpoint_path}")

    print("\n" + "=" * 70)
    print("学習完了!")
    print(f"  最終損失: {avg_loss:.4f}")
    print(f"  最良損失: {best_loss:.4f}")
    print(f"  チェックポイント: {checkpoint_path}")
    print("=" * 70)

    return model, tokenizer


def test_generation(model, tokenizer, temperature: float = 0.5):
    """
    生成テスト

    Args:
        model: 学習済みモデル
        tokenizer: トークナイザー
        temperature: 生成温度（低いほど安定、推奨: 0.3-0.7）
    """
    print("\n" + "=" * 70)
    print("生成テスト")
    print(f"生成パラメータ: temperature={temperature}, top_k=40, top_p=0.9")
    print("=" * 70)

    test_prompts = [
        "User: こんにちは\nAssistant:",
        "User: 量子コンピュータって何？\nAssistant:",
        "User: プログラミングを教えて\nAssistant:",
        "User: 機械学習とは？\nAssistant:",
    ]

    model.eval()
    device = next(model.parameters()).device

    for prompt in test_prompts:
        print(f"\n入力: {prompt}")
        tokens = tokenizer.encode(prompt)

        # 安定した生成のための推奨パラメータ
        generated = model.generate(
            prompt_tokens=tokens,
            max_length=80,  # 短めに制限
            temperature=temperature,  # 低温度で安定性向上
            top_k=40,
            top_p=0.9,
            eos_id=tokenizer.eos_id
        )

        output = tokenizer.decode(generated)
        print(f"出力: {output}")

    print("\n" + "-" * 70)
    print("ヒント: 文字化けが発生する場合は temperature を 0.3 程度に下げてください")
    print("        python train_oasst1_ja.py --use-cleaned で再学習も効果的です")
    print("-" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="oasst1-89k-ja User/Assistant 形式データ学習")
    parser.add_argument("--data", type=str, default=None, help="学習データファイルパス")
    parser.add_argument("--use-cleaned", action="store_true", help="クリーニング済みデータを使用")
    parser.add_argument("--epochs", type=int, default=5, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=8, help="バッチサイズ")
    parser.add_argument("--seq-len", type=int, default=128, help="シーケンス長")
    parser.add_argument("--lr", type=float, default=0.0003, help="学習率")
    args = parser.parse_args()

    print("=" * 70)
    print("oasst1-89k-ja User/Assistant 形式データ学習")
    print("=" * 70)

    # 設定
    if args.data:
        data_path = args.data
    elif args.use_cleaned:
        # クリーニング済みデータを優先
        data_path = "data/combined_clean_data.txt"
        if not os.path.exists(data_path):
            data_path = "data/oasst1_ja_cleaned.txt"
    else:
        data_path = "data/oasst1_ja_conversations.txt"

    tokenizer_path = "neuroq_tokenizer_oasst1_ja.json"
    checkpoint_path = "checkpoints/oasst1_ja_model.pt"

    print(f"学習データ: {data_path}")

    # データ読み込み
    conversations = load_conversation_data(data_path)
    if not conversations:
        print("会話データがありません。")
        return

    # トークナイザー読み込み
    if not os.path.exists(tokenizer_path):
        print(f"トークナイザーが見つかりません: {tokenizer_path}")
        return

    tokenizer = SimpleTokenizer(tokenizer_path)

    # 学習
    model, tokenizer = train_model(
        conversations=conversations,
        tokenizer=tokenizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        checkpoint_path=checkpoint_path
    )

    # 生成テスト（低温度で安定性確認）
    test_generation(model, tokenizer)

    print("\n完了!")


if __name__ == "__main__":
    main()
