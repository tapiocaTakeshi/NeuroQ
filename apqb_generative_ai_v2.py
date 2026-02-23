#!/usr/bin/env python3
"""
APQB Generative AI v2 - APQBドロップアウトを使用した生成AI

APQBドロップアウト層を組み込んだニューラルネットワークで
テキストを生成する量子確率的生成AI

大規模トークン対応版
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, Counter
import re
import os

# ========== 設定 ==========
DEFAULT_CONFIG = {
    'embed_dim': 128,        # 埋め込み次元
    'num_heads': 8,          # アテンションヘッド数
    'num_layers': 4,         # トランスフォーマー層数
    'seq_len': 128,          # シーケンス長
    'max_tokens': 10000,     # 最大トークン数
    'batch_size': 32,        # バッチサイズ
    'epochs': 50,            # エポック数
    'dropout_r': -0.3,       # APQBドロップアウト r
}

# ========== APQB (Adjustable Pseudo Quantum Bit) ==========
class APQB:
    """調整可能な擬似量子ビット"""
    
    def __init__(self, r: float = 0.0):
        self.r = np.clip(r, -1, 1)
    
    @property
    def theta(self) -> float:
        return np.pi * (1 - self.r) / 2
    
    @property
    def p_keep(self) -> float:
        return np.sin(self.theta / 2) ** 2
    
    def measure(self) -> int:
        return 1 if np.random.random() < self.p_keep else 0


# ========== APQBドロップアウト層 ==========
class APQBDropout(nn.Module):
    """APQBベースのドロップアウト層"""
    
    def __init__(self, r: float = 0.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self._r = nn.Parameter(torch.tensor(float(r)))
        else:
            self.register_buffer('_r', torch.tensor(float(r)))
        self.learnable = learnable
    
    @property
    def r(self) -> float:
        return float(self._r.clamp(-1, 1))
    
    @r.setter
    def r(self, value: float):
        with torch.no_grad():
            self._r.fill_(np.clip(value, -1, 1))
    
    def get_keep_probability(self) -> float:
        r = self.r
        theta = np.pi * (1 - r) / 2
        return np.sin(theta / 2) ** 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        
        r = self._r.clamp(-1, 1)
        theta = np.pi * (1 - r.item()) / 2
        p_keep = np.sin(theta / 2) ** 2
        
        # 量子ゆらぎを含むマスク生成
        rand = torch.rand_like(x)
        quantum_noise = torch.zeros_like(x)
        for i in range(4):
            phase = torch.rand_like(x) * 2 * np.pi
            quantum_noise += torch.sin(phase) * 0.05
        
        mask = ((rand + quantum_noise) < p_keep).float()
        
        if p_keep > 0:
            return x * mask / p_keep
        return x * 0


# ========== APQBアテンション層 ==========
class APQBAttention(nn.Module):
    """APQBベースのアテンション機構"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout_r: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.apqb_dropout = APQBDropout(r=dropout_r)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V を計算
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # アテンションスコア
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.apqb_dropout(attn)  # APQBドロップアウト適用
        
        # 出力
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


# ========== APQBトランスフォーマーブロック ==========
class APQBTransformerBlock(nn.Module):
    """APQBドロップアウトを使用したトランスフォーマーブロック"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, 
                 ff_dim: int = None, dropout_r: float = 0.0):
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4
        
        self.attention = APQBAttention(embed_dim, num_heads, dropout_r)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            APQBDropout(r=dropout_r),
            nn.Linear(ff_dim, embed_dim),
            APQBDropout(r=dropout_r)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), mask)
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x


# ========== APQB生成AIモデル ==========
class APQBGenerativeModel(nn.Module):
    """APQBドロップアウトを使用した生成AIモデル"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 num_heads: int = 4, num_layers: int = 3,
                 max_seq_len: int = 128, dropout_r: float = 0.0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embed_dropout = APQBDropout(r=dropout_r)
        
        # トランスフォーマーブロック
        self.blocks = nn.ModuleList([
            APQBTransformerBlock(embed_dim, num_heads, dropout_r=dropout_r)
            for _ in range(num_layers)
        ])
        
        # 出力層
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        
        # 量子サンプリング用のAPQB
        self.sampling_r = 0.0
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = x.shape
        
        # 位置情報
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # 埋め込み
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.embed_dropout(x)
        
        # 因果マスク（未来のトークンを見ない）
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        # トランスフォーマーブロック
        for block in self.blocks:
            x = block(x, mask)
        
        # 出力
        x = self.norm(x)
        logits = self.output(x)
        
        return logits
    
    def set_dropout_r(self, r: float):
        """全APQBドロップアウト層のrを設定"""
        for module in self.modules():
            if isinstance(module, APQBDropout):
                module.r = r
    
    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50,
                 quantum_sampling: bool = True) -> torch.Tensor:
        """テキストを生成"""
        self.eval()
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # 最後のmax_seq_lenトークンのみ使用
            context = generated[:, -self.max_seq_len:]
            
            # 次のトークンの確率を計算
            logits = self(context)
            logits = logits[:, -1, :] / temperature
            
            # Top-k サンプリング
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            
            # 量子サンプリング or 通常サンプリング
            if quantum_sampling:
                next_token = self._quantum_sample(probs)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def _quantum_sample(self, probs: torch.Tensor) -> torch.Tensor:
        """APQBベースの量子サンプリング"""
        batch_size, vocab_size = probs.shape
        
        # 複数のAPQBで量子乱数を生成
        quantum_random = torch.zeros(batch_size, device=probs.device)
        for i in range(12):  # 12ビット精度
            r = torch.rand(batch_size, device=probs.device) * 2 - 1 + self.sampling_r
            r = r.clamp(-1, 1)
            theta = np.pi * (1 - r) / 2
            p_one = torch.sin(theta / 2) ** 2
            bit = (torch.rand(batch_size, device=probs.device) < p_one).float()
            quantum_random += bit * (2 ** -(i + 1))
        
        # 累積確率でサンプリング
        cumsum = probs.cumsum(dim=-1)
        next_token = (cumsum < quantum_random.unsqueeze(1)).sum(dim=-1, keepdim=True)
        next_token = next_token.clamp(0, vocab_size - 1)
        
        return next_token


# ========== サブワードトークナイザー ==========
class SubwordTokenizer:
    """BPEベースのサブワードトークナイザー"""
    
    def __init__(self, max_vocab_size: int = 5000):
        self.max_vocab_size = max_vocab_size
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.merges = []
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
    def train(self, texts: list, min_freq: int = 2):
        """テキストからトークナイザーを学習"""
        # 文字レベルの語彙を構築
        char_vocab = set()
        for text in texts:
            char_vocab.update(text)
        
        # 特殊トークン + 文字語彙
        vocab = list(self.special_tokens) + sorted(char_vocab)
        self.token_to_idx = {t: i for i, t in enumerate(vocab)}
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        
        # 単語を文字のタプルに分割
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            for word in words:
                word_freqs[tuple(word)] += 1
        
        # BPEマージを学習
        while len(self.token_to_idx) < self.max_vocab_size:
            # ペアの頻度をカウント
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # 最頻出ペアを取得
            best_pair = pair_freqs.most_common(1)[0]
            if best_pair[1] < min_freq:
                break
            
            pair = best_pair[0]
            new_token = ''.join(pair)
            
            # 語彙に追加
            new_idx = len(self.token_to_idx)
            self.token_to_idx[new_token] = new_idx
            self.idx_to_token[new_idx] = new_token
            self.merges.append(pair)
            
            # 単語を更新
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs
        
        print(f"📚 トークナイザー学習完了: {len(self.token_to_idx)} トークン")
        return self
    
    def encode(self, text: str) -> list:
        """テキストをトークンIDに変換"""
        tokens = []
        words = text.split()
        
        for word in words:
            # 文字に分割
            chars = list(word)
            
            # マージを適用
            for merge in self.merges:
                i = 0
                new_chars = []
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == merge[0] and chars[i+1] == merge[1]:
                        new_chars.append(''.join(merge))
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            
            # トークンIDに変換
            for c in chars:
                tokens.append(self.token_to_idx.get(c, self.token_to_idx['<UNK>']))
            
            # スペースを追加
            tokens.append(self.token_to_idx.get(' ', self.token_to_idx['<UNK>']))
        
        return tokens
    
    def decode(self, ids: list) -> str:
        """トークンIDをテキストに変換"""
        tokens = []
        for idx in ids:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            token = self.idx_to_token.get(idx, '<UNK>')
            if token not in self.special_tokens:
                tokens.append(token)
        return ''.join(tokens)
    
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)


# ========== 文字レベルトークナイザー ==========
class CharTokenizer:
    """シンプルな文字レベルトークナイザー"""
    
    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    
    def train(self, texts: list):
        """テキストからトークナイザーを学習"""
        char_vocab = set()
        for text in texts:
            char_vocab.update(text)
        
        vocab = list(self.special_tokens) + sorted(char_vocab)
        self.token_to_idx = {t: i for i, t in enumerate(vocab)}
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        
        print(f"📚 文字トークナイザー学習完了: {len(self.token_to_idx)} トークン")
        return self
    
    def encode(self, text: str) -> list:
        return [self.token_to_idx.get(c, self.token_to_idx['<UNK>']) for c in text]
    
    def decode(self, ids: list) -> str:
        tokens = []
        for idx in ids:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            token = self.idx_to_token.get(idx, '')
            if token not in self.special_tokens:
                tokens.append(token)
        return ''.join(tokens)
    
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)


# ========== テキストデータセット ==========
class TextDataset(Dataset):
    """テキストデータセット"""
    
    def __init__(self, text: str, tokenizer, seq_len: int = 128):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # テキストをトークンIDに変換
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.vocab_size = tokenizer.vocab_size
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y
    
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
    
    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids.tolist() if isinstance(ids, torch.Tensor) else ids)


# ========== APQB生成AI ==========
class APQBGenerativeAI:
    """APQBドロップアウトを使用した生成AI（大規模トークン対応）"""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8,
                 num_layers: int = 4, dropout_r: float = -0.3,
                 max_vocab_size: int = 5000, use_subword: bool = True):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_r = dropout_r
        self.max_vocab_size = max_vocab_size
        self.use_subword = use_subword
        
        self.model = None
        self.dataset = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, text: str, epochs: int = 50, batch_size: int = 32,
              lr: float = 0.001, seq_len: int = 128, verbose: bool = True):
        """テキストで学習"""
        if verbose:
            print("🧠⚛️ APQB Generative AI v2 学習開始...")
            print(f"   デバイス: {self.device}")
            print(f"   サブワードトークン: {'有効' if self.use_subword else '無効'}")
        
        # トークナイザー作成
        if self.use_subword:
            self.tokenizer = SubwordTokenizer(max_vocab_size=self.max_vocab_size)
            self.tokenizer.train([text])
        else:
            self.tokenizer = CharTokenizer()
            self.tokenizer.train([text])
        
        # データセット作成
        self.dataset = TextDataset(text, self.tokenizer, seq_len)
        if verbose:
            print(f"   語彙サイズ: {self.dataset.vocab_size} トークン")
            print(f"   シーケンス長: {seq_len}")
            print(f"   データサイズ: {len(self.dataset)} サンプル")
            print(f"   総トークン数: {len(self.dataset.data):,}")
        
        # モデル作成
        self.model = APQBGenerativeModel(
            vocab_size=self.dataset.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=seq_len,
            dropout_r=self.dropout_r
        ).to(self.device)
        
        if verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   パラメータ数: {total_params:,}")
            print(f"   APQBドロップアウト r: {self.dropout_r}")
        
        # データローダー
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, 
                                drop_last=True, num_workers=0)
        
        # オプティマイザと損失関数
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()
        
        # 学習
        self.model.train()
        losses = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits.view(-1, self.dataset.vocab_size), y.view(-1))
                loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            
            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if verbose and (epoch + 1) % 5 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"   Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, LR = {lr_now:.6f}")
        
        if verbose:
            print(f"✅ 学習完了！ 最良損失: {best_loss:.4f}")
        
        return losses
    
    def generate(self, prompt: str = "", max_length: int = 200,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                 dropout_r: float = None, quantum_sampling: bool = True,
                 repetition_penalty: float = 1.1) -> str:
        """テキストを生成"""
        if self.model is None or self.dataset is None:
            raise ValueError("先にtrain()で学習してください")
        
        # ドロップアウト率を設定
        if dropout_r is not None:
            self.model.set_dropout_r(dropout_r)
        
        # プロンプトをエンコード
        if prompt:
            prompt_ids = self.dataset.encode(prompt).unsqueeze(0).to(self.device)
        else:
            # BOSトークンで開始
            bos_id = self.tokenizer.token_to_idx.get('<BOS>', 0)
            prompt_ids = torch.tensor([[bos_id]], dtype=torch.long).to(self.device)
        
        # 生成
        self.model.sampling_r = dropout_r if dropout_r is not None else self.dropout_r
        generated_ids = self.model.generate(
            prompt_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            quantum_sampling=quantum_sampling
        )
        
        # デコード
        return self.dataset.decode(generated_ids[0])
    
    def generate_stream(self, prompt: str = "", max_length: int = 200,
                        temperature: float = 1.0):
        """ストリーミング生成"""
        for char in prompt:
            yield char
        
        generated = self.generate(prompt, max_length, temperature)
        for char in generated[len(prompt):]:
            yield char
    
    def get_stats(self) -> dict:
        """モデル統計を取得"""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'vocab_size': self.tokenizer.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'dropout_r': self.dropout_r,
            'device': str(self.device)
        }
    
    def save(self, path: str):
        """モデルを保存"""
        torch.save({
            'model_state': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'config': {
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'dropout_r': self.dropout_r,
            }
        }, path)
        print(f"💾 モデル保存完了: {path}")
    
    def load(self, path: str):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.tokenizer = checkpoint['tokenizer']
        config = checkpoint['config']
        
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.dropout_r = config['dropout_r']
        
        # モデル再構築
        self.model = APQBGenerativeModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_r=self.dropout_r
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"📂 モデル読み込み完了: {path}")


# ========== OpenAssistantデータセット ==========
def load_openassistant_data(max_samples: int = 5000, lang: str = 'ja'):
    """OpenAssistant/oasst1データセットを読み込み"""
    try:
        from datasets import load_dataset
        print("📥 OpenAssistant/oasst1 データセットをダウンロード中...")
        
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        
        # 日本語または指定言語のデータをフィルタリング
        texts = []
        count = 0
        
        for item in dataset:
            if count >= max_samples:
                break
            
            text = item.get('text', '')
            item_lang = item.get('lang', '')
            
            # 言語フィルタ（'ja'なら日本語、'all'なら全言語）
            if lang == 'all' or item_lang == lang:
                if len(text) > 20:  # 短すぎるテキストは除外
                    texts.append(text)
                    count += 1
            elif lang == 'ja' and any(ord(c) > 0x3000 for c in text):
                # 日本語文字を含むテキストも含める
                if len(text) > 20:
                    texts.append(text)
                    count += 1
        
        if len(texts) < 100:
            # 日本語データが少ない場合は英語も含める
            print(f"⚠️ {lang}データが少ないため、英語データも追加します...")
            for item in dataset:
                if count >= max_samples:
                    break
                text = item.get('text', '')
                if len(text) > 20 and text not in texts:
                    texts.append(text)
                    count += 1
        
        combined_text = '\n\n'.join(texts)
        print(f"✅ {len(texts)} サンプル読み込み完了 ({len(combined_text):,} 文字)")
        return combined_text
        
    except ImportError:
        print("❌ datasetsライブラリが必要です。インストールしてください:")
        print("   pip install datasets")
        return None
    except Exception as e:
        print(f"❌ データセット読み込みエラー: {e}")
        return None


def load_local_or_oasst(use_oasst: bool = True, max_samples: int = 3000):
    """OpenAssistantまたはローカルテキストを読み込み"""
    if use_oasst:
        text = load_openassistant_data(max_samples=max_samples, lang='all')
        if text:
            return text
        print("⚠️ OpenAssistantデータ取得失敗。ローカルテキストを使用します。")
    
    return get_large_training_text()


# ========== 大規模学習テキスト ==========
def get_large_training_text():
    """大規模学習テキストを取得"""
    base_texts = [
        """
        人工知能は私たちの生活を大きく変えています。
        機械学習やディープラーニングの発展により、コンピュータは人間のように考え、学習することができるようになりました。
        画像認識、音声認識、自然言語処理など、AIの応用分野は急速に拡大しています。
        自動運転車、医療診断支援、金融予測など、様々な産業でAIが活用されています。
        AIの倫理的な問題についても、社会全体で議論が進んでいます。
        """,
        """
        量子コンピュータは、この革命をさらに加速させる可能性を秘めています。
        量子ビットは重ね合わせ状態を利用して、従来のコンピュータでは不可能だった計算を可能にします。
        量子もつれや量子干渉など、量子力学の原理を利用した新しい計算パラダイムです。
        暗号解読、分子シミュレーション、最適化問題など、量子コンピュータの応用は多岐にわたります。
        世界中の研究機関や企業が、量子コンピュータの実用化に向けて研究を進めています。
        """,
        """
        APQBは、この量子の性質を擬似的に再現し、ニューラルネットワークに確率的な振る舞いをもたらします。
        擬似量子ビットは、相関係数から量子状態を構築し、確率的な測定を行います。
        この技術により、従来のニューラルネットワークに量子インスパイアドな要素を追加できます。
        APQBドロップアウトは、学習中の正則化に量子的なゆらぎを導入します。
        量子サンプリングにより、テキスト生成に創造的な多様性が生まれます。
        """,
        """
        未来のAIは、量子と古典の融合によって、より創造的で柔軟な思考ができるようになるでしょう。
        技術の進歩は止まることを知りません。私たちは常に新しい可能性を探求し続けています。
        人間とAIが協力して、より良い社会を築いていくことが期待されています。
        環境問題、エネルギー問題、医療問題など、人類が直面する課題にAIが貢献できます。
        持続可能な発展のために、技術は人間の幸福に貢献する方向に進化していくべきです。
        """,
        """
        日本の四季は本当に美しいものです。
        桜の花が咲く春の季節、私たちは新しい始まりを迎えます。
        入学式、入社式、新生活のスタート。春は希望に満ちた季節です。
        花見の文化は日本独特のもので、桜の下で人々が集い、楽しい時間を過ごします。
        """,
        """
        夏には太陽が輝き、海や山で楽しい時間を過ごします。
        花火大会、夏祭り、海水浴。夏は活気に満ちた季節です。
        日本の夏は暑いですが、冷たいかき氷やそうめんで涼をとります。
        お盆には先祖の霊を迎え、家族で過ごす大切な時間があります。
        """,
        """
        秋になると紅葉が美しく色づき、穏やかな風が吹きます。
        読書の秋、芸術の秋、食欲の秋。秋は実りの季節です。
        新米、さんま、栗、柿。秋の味覚を楽しむことができます。
        紅葉狩りは日本の美しい伝統文化の一つです。
        """,
        """
        冬は雪が降り、温かい家の中で家族と過ごす時間が増えます。
        クリスマス、大晦日、お正月。冬は特別なイベントが続きます。
        こたつでみかんを食べながら、のんびりと過ごす時間は格別です。
        新年の抱負を立て、新たな一年への希望を胸に歩み始めます。
        """,
        """
        プログラミングは現代社会で重要なスキルです。
        Python、JavaScript、Java、C++など、様々なプログラミング言語があります。
        アルゴリズムとデータ構造は、効率的なプログラムを書くために必要な知識です。
        ソフトウェア開発では、チームワークとコミュニケーションが重要です。
        バージョン管理、テスト、コードレビューは品質を保つための基本的なプラクティスです。
        """,
        """
        ニューラルネットワークは人間の脳を模倣した計算モデルです。
        入力層、隠れ層、出力層から構成され、重みとバイアスで学習します。
        活性化関数、損失関数、最適化アルゴリズムは深層学習の重要な要素です。
        畳み込みニューラルネットワークは画像認識に、リカレントニューラルネットワークは時系列データに適しています。
        トランスフォーマーアーキテクチャは自然言語処理に革命をもたらしました。
        """,
        """
        宇宙は広大で神秘的な空間です。
        太陽系には8つの惑星があり、地球は唯一生命が確認されている惑星です。
        星座は古代から人々の想像力を刺激してきました。
        ブラックホール、ダークマター、ダークエネルギーなど、宇宙にはまだ多くの謎があります。
        宇宙探査技術の発展により、人類は火星への有人飛行を目指しています。
        """,
        """
        音楽は人間の感情に深く影響を与えます。
        クラシック、ジャズ、ロック、ポップ、電子音楽など、様々なジャンルがあります。
        音楽療法は心身の健康に効果があるとされています。
        AIによる作曲技術も発展し、新しい音楽表現の可能性が広がっています。
        ライブ演奏の臨場感は、録音では得られない特別な体験です。
        """,
    ]
    
    # テキストを繰り返して大規模化
    return '\n'.join(base_texts * 5)


# ========== デモ ==========
def demo(use_oasst: bool = False):
    """デモンストレーション"""
    print("=" * 70)
    print("🧠⚛️ APQB Generative AI v2 - 大規模トークン対応版")
    if use_oasst:
        print("   📚 OpenAssistant/oasst1 データセット使用")
    print("=" * 70)
    
    # 学習テキスト取得
    if use_oasst:
        training_text = load_local_or_oasst(use_oasst=True, max_samples=2000)
    else:
        training_text = get_large_training_text()
    print(f"\n📊 学習テキストサイズ: {len(training_text):,} 文字")
    
    print("\n📚 学習中...")
    
    # APQBドロップアウト r=-0.3 (最適設定)
    print("\n--- 大規模モデル学習 (サブワードトークン) ---")
    ai = APQBGenerativeAI(
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout_r=-0.3,
        max_vocab_size=3000,
        use_subword=True
    )
    losses = ai.train(
        training_text, 
        epochs=40, 
        batch_size=32, 
        seq_len=128
    )
    
    # 統計情報
    stats = ai.get_stats()
    print("\n📈 モデル統計:")
    print(f"   語彙サイズ: {stats['vocab_size']:,} トークン")
    print(f"   埋め込み次元: {stats['embed_dim']}")
    print(f"   パラメータ数: {stats['total_params']:,}")
    
    # テキスト生成テスト
    prompts = ["人工知能", "量子コンピュータ", "日本の", "プログラミング", "未来"]
    
    print("\n" + "=" * 70)
    print("📝 テキスト生成結果 (max_length=150)")
    print("=" * 70)
    
    for prompt in prompts:
        print(f"\n🔮 プロンプト: 「{prompt}」")
        print("-" * 50)
        
        text = ai.generate(prompt, max_length=150, temperature=0.8, top_k=50)
        print(f"{text}")
    
    # 温度による違い
    print("\n" + "=" * 70)
    print("🌡️ 温度パラメータによる違い")
    print("=" * 70)
    
    for temp in [0.5, 0.8, 1.0, 1.2]:
        text = ai.generate("技術", max_length=80, temperature=temp)
        print(f"\n温度 {temp}: {text}")
    
    # 量子サンプリングの比較
    print("\n" + "=" * 70)
    print("⚛️ 量子サンプリング vs 通常サンプリング")
    print("=" * 70)
    
    print("\n[量子サンプリング ON]")
    for i in range(3):
        text = ai.generate("宇宙", max_length=60, temperature=0.9, quantum_sampling=True)
        print(f"  {i+1}. {text}")
    
    print("\n[量子サンプリング OFF]")
    for i in range(3):
        text = ai.generate("宇宙", max_length=60, temperature=0.9, quantum_sampling=False)
        print(f"  {i+1}. {text}")
    
    print("\n✅ デモ完了！")
    
    return ai


# ========== 対話モード ==========
def interactive_mode(use_oasst: bool = False):
    """対話モード"""
    print("=" * 70)
    print("🧠⚛️ APQB Generative AI v2 - 大規模トークン対話モード")
    if use_oasst:
        print("   📚 OpenAssistant/oasst1 データセット使用")
    print("=" * 70)
    
    # 学習テキスト取得
    if use_oasst:
        training_text = load_local_or_oasst(use_oasst=True, max_samples=3000)
    else:
        training_text = get_large_training_text()
    print(f"\n📊 学習テキストサイズ: {len(training_text):,} 文字")
    
    print("\n📚 学習中...")
    ai = APQBGenerativeAI(
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout_r=-0.3,
        max_vocab_size=3000,
        use_subword=True
    )
    ai.train(training_text, epochs=40, batch_size=32, seq_len=128)
    
    # 統計情報
    stats = ai.get_stats()
    print(f"\n📈 モデル統計:")
    print(f"   語彙サイズ: {stats['vocab_size']:,} トークン")
    print(f"   パラメータ数: {stats['total_params']:,}")
    
    print("\n" + "=" * 70)
    print("コマンド:")
    print("  <テキスト>    : プロンプトからテキストを生成")
    print("  /temp <値>    : 温度を設定 (0.1-2.0)")
    print("  /r <値>       : APQBドロップアウトrを設定 (-1.0-1.0)")
    print("  /len <値>     : 生成長さを設定 (10-1000)")
    print("  /topk <値>    : Top-K を設定 (1-100)")
    print("  /quantum      : 量子サンプリング ON/OFF 切替")
    print("  /stats        : モデル統計を表示")
    print("  /save <path>  : モデルを保存")
    print("  /load <path>  : モデルを読み込み")
    print("  /quit         : 終了")
    print("=" * 70)
    
    temperature = 0.8
    dropout_r = -0.3
    max_length = 200
    top_k = 50
    quantum_sampling = True
    
    while True:
        try:
            user_input = input("\n🔮 > ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith("/quit"):
                print("👋 終了します")
                break
            
            elif user_input.startswith("/temp"):
                try:
                    temp = float(user_input.split()[1])
                    temperature = max(0.1, min(2.0, temp))
                    print(f"🌡️ 温度を {temperature} に設定しました")
                except (ValueError, IndexError):
                    print("❌ 使用法: /temp <0.1-2.0>")

            elif user_input.startswith("/r "):
                try:
                    r = float(user_input.split()[1])
                    dropout_r = max(-1.0, min(1.0, r))
                    print(f"⚛️ APQBドロップアウト r を {dropout_r} に設定しました")
                except (ValueError, IndexError):
                    print("❌ 使用法: /r <-1.0-1.0>")

            elif user_input.startswith("/len"):
                try:
                    length = int(user_input.split()[1])
                    max_length = max(10, min(1000, length))
                    print(f"📏 生成長さを {max_length} トークンに設定しました")
                except (ValueError, IndexError):
                    print("❌ 使用法: /len <10-1000>")

            elif user_input.startswith("/topk"):
                try:
                    k = int(user_input.split()[1])
                    top_k = max(1, min(100, k))
                    print(f"🎯 Top-K を {top_k} に設定しました")
                except (ValueError, IndexError):
                    print("❌ 使用法: /topk <1-100>")

            elif user_input.startswith("/quantum"):
                quantum_sampling = not quantum_sampling
                status = "ON ⚛️" if quantum_sampling else "OFF 📊"
                print(f"量子サンプリング: {status}")

            elif user_input.startswith("/stats"):
                stats = ai.get_stats()
                print("\n📈 モデル統計:")
                for key, value in stats.items():
                    if isinstance(value, int) and value > 1000:
                        print(f"   {key}: {value:,}")
                    else:
                        print(f"   {key}: {value}")

            elif user_input.startswith("/save"):
                try:
                    path = user_input.split()[1]
                    ai.save(path)
                except (IndexError, Exception) as e:
                    print(f"❌ 使用法: /save <ファイルパス> ({e})")
            
            elif user_input.startswith("/load"):
                try:
                    path = user_input.split()[1]
                    ai.load(path)
                except Exception as e:
                    print(f"❌ 読み込みエラー: {e}")
            
            else:
                print("\n📜 生成中... ", end="", flush=True)
                
                generated = ai.generate(
                    prompt=user_input,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    dropout_r=dropout_r,
                    quantum_sampling=quantum_sampling
                )
                
                print("\n" + "-" * 50)
                print(generated)
                print("-" * 50)
                q_status = "⚛️" if quantum_sampling else "📊"
                print(f"[温度: {temperature}, r: {dropout_r}, 長さ: {max_length}, TopK: {top_k}, {q_status}]")
        
        except KeyboardInterrupt:
            print("\n👋 終了します")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")
            import traceback
            traceback.print_exc()


# ========== メイン ==========
if __name__ == "__main__":
    import sys
    
    use_oasst = "--oasst" in sys.argv or "--openassistant" in sys.argv
    
    if "--demo" in sys.argv:
        demo(use_oasst=use_oasst)
    else:
        interactive_mode(use_oasst=use_oasst)
    
    # 使い方:
    # python apqb_generative_ai_v2.py              # ローカルテキストで対話モード
    # python apqb_generative_ai_v2.py --oasst     # OpenAssistantで対話モード
    # python apqb_generative_ai_v2.py --demo      # ローカルテキストでデモ
    # python apqb_generative_ai_v2.py --demo --oasst  # OpenAssistantでデモ

