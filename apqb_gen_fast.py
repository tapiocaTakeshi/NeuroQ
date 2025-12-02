#!/usr/bin/env python3
"""APQB生成AI - 高速版"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

print("=" * 60)
print("🧠⚛️ APQB Generative AI - 高速版")
print("=" * 60)

# ========== APQBコア（論文の数学的定義） ==========
def theta_to_r(theta): return torch.cos(2 * theta)
def theta_to_T(theta): return torch.abs(torch.sin(2 * theta))

# ========== APQBモデル ==========
class APQBModel(nn.Module):
    def __init__(self, vocab_size, dim=64, layers=2, seq_len=48):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(seq_len, dim)
        
        # APQB角度パラメータ
        self.theta = nn.ParameterList([nn.Parameter(torch.rand(dim*4) * np.pi/4) for _ in range(layers)])
        
        # トランスフォーマー層
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, 4, dim*4, 0.1, batch_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for i, layer in enumerate(self.layers):
            # APQB温度による正則化
            if self.training:
                th = torch.sigmoid(self.theta[i]) * np.pi/2
                T_val = theta_to_T(th).mean()
                x = x + torch.randn_like(x) * T_val * 0.02
            x = layer(x, src_mask=mask, is_causal=True)
        
        return self.out(self.norm(x))
    
    def get_constraint(self):
        """r² + T² = 1 の検証"""
        c = 0.0
        for th in self.theta:
            th = torch.sigmoid(th) * np.pi/2
            c += (theta_to_r(th)**2 + theta_to_T(th)**2).mean()
        return c / len(self.theta)
    
    @torch.no_grad()
    def generate(self, ids, n=80, temp=0.8):
        self.eval()
        for _ in range(n):
            ctx = ids[:, -self.seq_len:]
            logits = self(ctx)[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            
            # APQB量子サンプリング
            th = torch.sigmoid(self.theta[0].mean()) * np.pi/2
            q = torch.sin(torch.rand(1)*2*np.pi + th) * 0.5 + 0.5
            idx = (probs.cumsum(-1) < q).sum(-1, keepdim=True).clamp(0, self.vocab_size-1)
            ids = torch.cat([ids, idx], 1)
        return ids

# ========== トークナイザー ==========
class Tok:
    def __init__(self, t):
        c = sorted(set(t))
        self.c2i = {c:i for i,c in enumerate(['<P>']+c)}
        self.i2c = {i:c for c,i in self.c2i.items()}
        self.vs = len(self.c2i)
    def enc(self, t): return [self.c2i.get(c,0) for c in t]
    def dec(self, ids): return ''.join(self.i2c.get(i,'') for i in ids if i>0)

# ========== データ ==========
class DS(Dataset):
    def __init__(self, t, tok, sl=48):
        self.d = torch.tensor(tok.enc(t), dtype=torch.long)
        self.sl = sl
    def __len__(self): return max(0, len(self.d)-self.sl-1)
    def __getitem__(self, i): return self.d[i:i+self.sl], self.d[i+1:i+self.sl+1]

# ========== 学習テキスト（小規模） ==========
text = """
Artificial intelligence transforms technology and society in profound ways.
Machine learning enables computers to learn from data and improve over time.
Neural networks model complex patterns through layers of interconnected nodes.
Deep learning has achieved remarkable success in image and speech recognition.
Natural language processing allows machines to understand human communication.
Quantum computing offers new paradigms for solving complex computational problems.
Quantum bits exist in superposition states enabling parallel computation.
Entanglement creates correlations that have no classical analog.
The APQB framework unifies neural networks with quantum many-body physics.
Complex numbers describe rotations and oscillations in mathematics and physics.
The exponential map connects linear algebra with trigonometry and geometry.
Science progresses through careful observation and rigorous experimentation.
Technology continues to reshape human civilization at an accelerating pace.
""" * 20

print(f"\n📊 学習データ: {len(text):,} 文字")

tok = Tok(text)
ds = DS(text, tok, 48)
dl = DataLoader(ds, 32, shuffle=True, drop_last=True)

print(f"📚 語彙: {tok.vs}, データ: {len(ds):,}")

model = APQBModel(tok.vs, 64, 2, 48)
print(f"📚 パラメータ: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), 0.002)
crit = nn.CrossEntropyLoss()

print("\n🏋️ 学習中...")
model.train()
for ep in range(15):
    loss_sum = 0
    for x, y in dl:
        opt.zero_grad()
        loss = crit(model(x).view(-1, tok.vs), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item()
    if (ep+1) % 3 == 0:
        c = model.get_constraint().item()
        print(f"   Epoch {ep+1}: Loss={loss_sum/len(dl):.4f}, r²+T²={c:.4f}")

print("✅ 学習完了！")

# ========== 量子統計 ==========
print("\n⚛️ 量子統計 (論文: r² + T² = 1):")
for i, th in enumerate(model.theta):
    th = torch.sigmoid(th) * np.pi/2
    r = theta_to_r(th)
    T = theta_to_T(th)
    print(f"   Layer {i}: r={r.mean():.3f}, T={T.mean():.3f}, r²+T²={(r**2+T**2).mean():.4f}")

# ========== 生成 ==========
print("\n" + "=" * 60)
print("📝 テキスト生成")
print("=" * 60)

for p in ["The ", "Quantum ", "Neural ", "Science "]:
    ids = torch.tensor([tok.enc(p)], dtype=torch.long)
    gen = model.generate(ids, 70, 0.8)
    print(f"\n🔮 「{p}」\n   → {tok.dec(gen[0].tolist())}")

print("\n🌡️ 温度による違い:")
for t in [0.5, 1.0, 1.5]:
    ids = torch.tensor([tok.enc("Technology ")], dtype=torch.long)
    gen = model.generate(ids, 50, t)
    print(f"   T={t}: {tok.dec(gen[0].tolist())}")

print("\n✅ 完了！")

