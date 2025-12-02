#!/usr/bin/env python3
"""APQB生成AI - 軽量版（組み込みテキストのみ）"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

print("=" * 70)
print("🧠⚛️ APQB Generative AI - 論文フレームワーク実装")
print("=" * 70)

# ========== APQBコア ==========
class APQBCore:
    @staticmethod
    def theta_to_r(theta):
        return torch.cos(2 * theta)
    
    @staticmethod
    def theta_to_T(theta):
        return torch.abs(torch.sin(2 * theta))
    
    @staticmethod
    def Q_k(theta, k):
        if k % 2 == 0:
            return torch.cos(2 * k * theta)
        else:
            return torch.sin(2 * k * theta)

# ========== APQBトランスフォーマー ==========
class APQBBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, max_order=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_order = max_order
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.theta = nn.Parameter(torch.rand(embed_dim * 4) * np.pi / 4)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x, mask=None):
        # 因果マスク生成
        T = x.shape[1]
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        
        # アテンション
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + attn_out
        x = self.norm1(x)
        
        # APQB変調付きFF
        h = self.ff[0](x)
        theta = torch.sigmoid(self.theta) * np.pi / 2
        T = APQBCore.theta_to_T(theta)
        if self.training:
            h = h * (1 + torch.randn_like(h) * T.mean() * 0.05)
        h = self.ff[1](h)
        h = self.ff[2](h)
        x = x + h
        x = self.norm2(x)
        
        return x
    
    def get_constraint(self):
        theta = torch.sigmoid(self.theta) * np.pi / 2
        r = APQBCore.theta_to_r(theta)
        T = APQBCore.theta_to_T(theta)
        return (r**2 + T**2).mean()

class APQBModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=96, num_heads=4, num_layers=3, max_seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        
        self.blocks = nn.ModuleList([APQBBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        
        self.sampling_theta = nn.Parameter(torch.tensor(np.pi / 4))
    
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(x) + self.pos_embed(pos)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output(self.norm(x))
    
    def get_constraint_loss(self):
        return sum(b.get_constraint() for b in self.blocks) / len(self.blocks)
    
    @torch.no_grad()
    def generate(self, prompt, max_new=100, temp=1.0, top_k=40):
        self.eval()
        gen = prompt.clone()
        for _ in range(max_new):
            ctx = gen[:, -self.max_seq_len:]
            logits = self(ctx)[:, -1, :] / temp
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            
            # APQB量子サンプリング
            theta = torch.sigmoid(self.sampling_theta) * np.pi / 2
            q_rand = 0.0
            for k in range(1, 5):
                Q_k = APQBCore.Q_k(theta, k)
                phase = torch.rand(1, device=probs.device) * 2 * np.pi
                q_rand += torch.sin(phase + k * theta.item()) / k
            q_rand = torch.sigmoid(q_rand)
            
            cumsum = probs.cumsum(dim=-1)
            next_tok = (cumsum < q_rand).sum(dim=-1, keepdim=True).clamp(0, self.vocab_size-1)
            gen = torch.cat([gen, next_tok], dim=1)
        return gen

# ========== トークナイザー ==========
class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.special = ['<P>', '<U>']
        vocab = self.special + chars
        self.c2i = {c: i for i, c in enumerate(vocab)}
        self.i2c = {i: c for c, i in self.c2i.items()}
        self.vocab_size = len(vocab)
    
    def encode(self, t):
        return [self.c2i.get(c, 1) for c in t]
    
    def decode(self, ids):
        return ''.join(self.i2c.get(i, '') for i in ids if i >= len(self.special))

# ========== データセット ==========
class TextDS(Dataset):
    def __init__(self, text, tok, seq_len=64):
        self.data = torch.tensor(tok.encode(text), dtype=torch.long)
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
    def __getitem__(self, i):
        return self.data[i:i+self.seq_len], self.data[i+1:i+self.seq_len+1]

# ========== 学習テキスト ==========
text = """
The field of artificial intelligence has seen remarkable progress in recent years.
Machine learning algorithms can now process vast amounts of data and identify complex patterns.
Neural networks have revolutionized many areas of computer science and technology.
Deep learning models have achieved superhuman performance in various tasks.
Natural language processing enables computers to understand and generate human language.
Quantum computing represents a fundamental shift in computational paradigms.
Quantum bits or qubits can exist in superposition states unlike classical bits.
Entanglement allows quantum systems to exhibit correlations impossible in classical physics.
The APQB framework bridges neural networks and quantum many-body systems.
Mathematics provides the foundation for both artificial intelligence and physics.
Linear algebra describes transformations in high-dimensional vector spaces.
Complex numbers and their geometry connect algebra with trigonometry.
The exponential map links the real line to the unit circle in the complex plane.
Science advances through observation, hypothesis, and experimentation.
Technology continues to transform human society in profound ways.
The internet connects billions of people across the globe.
Renewable energy technologies offer hope for sustainable development.
""" * 15

print(f"\n📊 学習テキスト: {len(text):,} 文字")

# ========== 学習 ==========
tok = Tokenizer(text)
ds = TextDS(text, tok, 64)
dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)

print(f"📚 語彙サイズ: {tok.vocab_size}")
print(f"📚 データサイズ: {len(ds):,}")

model = APQBModel(tok.vocab_size, embed_dim=96, num_heads=4, num_layers=3, max_seq_len=64)
print(f"📚 パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
crit = nn.CrossEntropyLoss()

print("\n🏋️ 学習中...")
model.train()
for epoch in range(20):
    total_loss = 0
    for x, y in dl:
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits.view(-1, tok.vocab_size), y.view(-1))
        constraint = model.get_constraint_loss()
        (loss + 0.1 * (constraint - 1).abs()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        c = model.get_constraint_loss().item()
        print(f"   Epoch {epoch+1}/20: Loss={total_loss/len(dl):.4f}, r²+T²={c:.4f}")

print("✅ 学習完了！")

# ========== 量子統計 ==========
print("\n⚛️ 量子統計 (論文: r² + T² = 1):")
for i, block in enumerate(model.blocks):
    theta = torch.sigmoid(block.theta) * np.pi / 2
    r = APQBCore.theta_to_r(theta)
    T = APQBCore.theta_to_T(theta)
    print(f"   Layer {i}: r={r.mean():.3f}±{r.std():.3f}, T={T.mean():.3f}±{T.std():.3f}, r²+T²={block.get_constraint():.4f}")

# ========== 生成 ==========
print("\n" + "=" * 70)
print("📝 テキスト生成結果")
print("=" * 70)

prompts = ["The ", "Quantum ", "Neural ", "Science ", "Technology "]
for p in prompts:
    ids = torch.tensor([tok.encode(p)], dtype=torch.long)
    gen = model.generate(ids, max_new=80, temp=0.8)
    print(f"\n🔮 「{p}」")
    print(f"   → {tok.decode(gen[0].tolist())}")

print("\n" + "=" * 70)
print("🌡️ 温度による違い")
print("=" * 70)
for temp in [0.5, 0.8, 1.0, 1.3]:
    ids = torch.tensor([tok.encode("The future ")], dtype=torch.long)
    gen = model.generate(ids, max_new=60, temp=temp)
    print(f"\n温度 {temp}: {tok.decode(gen[0].tolist())}")

print("\n✅ 完了！")

