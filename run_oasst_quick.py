#!/usr/bin/env python3
"""
OpenAssistant/oasst1 でAPQB生成AIを高速学習・テスト
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter

print("=" * 70)
print("🧠⚛️ APQB Generative AI - OpenAssistant/oasst1 高速デモ")
print("=" * 70)

# データセット読み込み
print("\n📥 OpenAssistant/oasst1 読み込み中...")
from datasets import load_dataset
dataset = load_dataset("OpenAssistant/oasst1", split="train")

# 最初の500サンプルだけ使用（高速化）
texts = []
for i, item in enumerate(dataset):
    if i >= 500:
        break
    text = item.get('text', '')
    if len(text) > 20:
        texts.append(text)

training_text = '\n\n'.join(texts)
print(f"✅ {len(texts)} サンプル読み込み完了 ({len(training_text):,} 文字)")

# ========== シンプルなAPQBモデル ==========
class APQBDropout(nn.Module):
    def __init__(self, r: float = 0.0):
        super().__init__()
        self.r = r
    
    def forward(self, x):
        if not self.training:
            return x
        theta = np.pi * (1 - self.r) / 2
        p_keep = np.sin(theta / 2) ** 2
        mask = (torch.rand_like(x) < p_keep).float()
        if p_keep > 0:
            return x * mask / p_keep
        return x * 0


class APQBModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=64, dropout_r=-0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = APQBDropout(r=dropout_r)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.sampling_r = dropout_r
    
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        
        x = self.token_embed(x) + self.pos_embed(pos)
        x = self.dropout(x)
        
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask, is_causal=True)
        
        x = self.norm(x)
        return self.output(x)
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=40):
        self.eval()
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            context = generated[:, -self.max_seq_len:]
            logits = self(context)[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            
            # 量子サンプリング
            quantum_rand = 0.0
            for i in range(8):
                r = torch.rand(1).item() * 2 - 1 + self.sampling_r
                r = np.clip(r, -1, 1)
                theta = np.pi * (1 - r) / 2
                p_one = np.sin(theta / 2) ** 2
                bit = 1 if np.random.random() < p_one else 0
                quantum_rand += bit * (2 ** -(i + 1))
            
            cumsum = probs.cumsum(dim=-1)
            next_token = (cumsum < quantum_rand).sum(dim=-1, keepdim=True)
            next_token = next_token.clamp(0, self.vocab_size - 1)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# ========== トークナイザー ==========
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.special = ['<PAD>', '<UNK>']
        vocab = self.special + chars
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(vocab)
    
    def encode(self, text):
        return [self.char_to_idx.get(c, 1) for c in text]
    
    def decode(self, ids):
        return ''.join([self.idx_to_char.get(i, '') for i in ids if i >= len(self.special)])


# データセット
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=64):
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+1:idx+self.seq_len+1]


# 学習
print("\n🏗️ モデル構築中...")
tokenizer = CharTokenizer(training_text)
print(f"   語彙サイズ: {tokenizer.vocab_size}")

dataset = TextDataset(training_text, tokenizer, seq_len=64)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = APQBModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    max_seq_len=64,
    dropout_r=-0.3
)

print(f"   パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("\n📚 学習中...")
model.train()
for epoch in range(15):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1}/15: Loss = {total_loss/len(dataloader):.4f}")

print("✅ 学習完了！")

# テキスト生成
print("\n" + "=" * 70)
print("📝 テキスト生成結果")
print("=" * 70)

prompts = ["Hello", "How can", "The best", "I think"]

for prompt in prompts:
    print(f"\n🔮 「{prompt}」")
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = model.generate(prompt_ids, max_new_tokens=80, temperature=0.8)
    text = tokenizer.decode(generated[0].tolist())
    print(f"   → {text[:120]}...")

print("\n✅ 完了！")

