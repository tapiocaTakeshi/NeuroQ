#!/usr/bin/env python3
"""
テキストエンベディング → QBNN パイプライン
==========================================

明確な処理フロー:
  入力(テキスト) → テキストエンベディング → QBNN処理 → 出力

構成:
1. TextEmbedding: テキスト → ベクトル表現
2. QBNN: 量子ビットニューラルネットワーク処理
3. OutputHead: ベクトル → テキスト出力

使用例:
    pipeline = TextEmbeddingQBNNPipeline(vocab_size=5000)
    pipeline.train(texts, epochs=20)
    output = pipeline.generate("量子コンピュータとは")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter


# ============================================================
# STAGE 1: テキストエンベディング (Text Embedding)
# ============================================================

class TextEmbedding(nn.Module):
    """
    テキストエンベディング層
    
    入力: トークンID列 (batch, seq_len)
    出力: エンベディングベクトル (batch, seq_len, embed_dim)
    
    機能:
    - トークン埋め込み: 各トークンをベクトルに変換
    - 位置埋め込み: 位置情報を付加
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 128, 
        max_seq_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # トークン埋め込み: 語彙 → ベクトル
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置埋め込み: 位置 → ベクトル
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # 正規化とドロップアウト
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 重み初期化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        テキストをエンベディングに変換
        
        Args:
            token_ids: (batch, seq_len) トークンID
        
        Returns:
            (batch, seq_len, embed_dim) エンベディング
        """
        batch_size, seq_len = token_ids.shape
        
        # トークン埋め込み
        token_embeds = self.token_embedding(token_ids)
        
        # 位置埋め込み
        positions = torch.arange(seq_len, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # 結合
        embeddings = token_embeds + pos_embeds
        
        # 正規化とドロップアウト
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


# ============================================================
# STAGE 2: QBNN (Quantum-Bit Neural Network)
# ============================================================

class APQB:
    """
    APQB理論 (Adjustable Pseudo Quantum Bit)
    
    量子ビットの擬似的な振る舞いを古典コンピュータで実現:
    - θ: 内部角度パラメータ
    - r = cos(2θ): 相関係数 (決定性)
    - T = |sin(2θ)|: 温度 (ゆらぎ)
    - 制約: r² + T² = 1
    """
    
    @staticmethod
    def theta_to_r(theta: torch.Tensor) -> torch.Tensor:
        """θ → 相関係数 r = cos(2θ)"""
        return torch.cos(2 * theta)
    
    @staticmethod
    def theta_to_T(theta: torch.Tensor) -> torch.Tensor:
        """θ → 温度 T = |sin(2θ)|"""
        return torch.abs(torch.sin(2 * theta))
    
    @staticmethod
    def measure(theta: torch.Tensor) -> torch.Tensor:
        """量子測定 (確率的に 0 or 1)"""
        prob_1 = torch.sin(theta) ** 2
        return (torch.rand_like(prob_1) < prob_1).float()


class QBNNLayer(nn.Module):
    """
    QBNN層 - 量子もつれを模倣した処理
    
    数式:
    1. s = tanh(h) ∈ [-1, 1] (正規化)
    2. h̃ = W·h + b (線形変換)
    3. Δ = Σ J_ij · s_i · s_j (量子もつれ項)
    4. ĥ = h̃ + λ·Δ (もつれ補正)
    5. output = activation(ĥ)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        lambda_entangle: float = 0.35
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 線形変換
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 量子もつれテンソル J
        self.J = nn.Parameter(torch.randn(input_dim, output_dim) * 0.02)
        
        # もつれ強度 λ (学習可能)
        self.lambda_entangle = nn.Parameter(torch.tensor(lambda_entangle))
        
        # 正規化
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        QBNN層の処理
        
        Args:
            h: (batch, seq, input_dim) 入力
        
        Returns:
            (batch, seq, output_dim) 出力
        """
        # 1. 正規化 (疑似量子状態)
        s = torch.tanh(h)
        
        # 2. 線形変換
        h_tilde = self.linear(h)
        
        # 3. 量子もつれ項の計算
        s_out = torch.tanh(h_tilde)
        delta = torch.einsum('...i,ij,...j->...j', s, self.J, s_out)
        
        # 4. もつれ補正を加算
        h_hat = h_tilde + self.lambda_entangle * delta
        
        # 5. 正規化と活性化
        output = self.layer_norm(h_hat)
        output = F.gelu(output)
        
        return output


class QBNNAttention(nn.Module):
    """
    QBNN拡張 Self-Attention
    
    通常のAttentionに量子もつれ補正を追加
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 4, 
        dropout: float = 0.1,
        lambda_entangle: float = 0.35
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V 射影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 量子もつれテンソル (ヘッドごと)
        self.J_attn = nn.Parameter(
            torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02
        )
        self.lambda_attn = nn.Parameter(torch.tensor(lambda_entangle))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        QBNN Attention
        
        Args:
            x: (batch, seq, embed_dim)
            mask: オプションのマスク
        
        Returns:
            (batch, seq, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V を計算
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # アテンションスコア
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 量子もつれ補正
        Q_norm = torch.tanh(Q)
        K_norm = torch.tanh(K)
        delta = torch.einsum('bhid,hde,bhje->bhij', Q_norm, self.J_attn, K_norm)
        attn_scores = attn_scores + self.lambda_attn * delta
        
        # Causal マスク (言語モデル用)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), 
            diagonal=1
        ).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # ソフトマックス
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 出力
        output = torch.matmul(attn_probs, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output


class QBNNBlock(nn.Module):
    """
    QBNNブロック = QBNN Attention + QBNN FFN
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        hidden_dim: int,
        num_heads: int = 4, 
        dropout: float = 0.1,
        lambda_entangle: float = 0.35
    ):
        super().__init__()
        
        # QBNN Attention
        self.attention = QBNNAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            lambda_entangle=lambda_entangle
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        # QBNN FFN
        self.ffn = nn.Sequential(
            QBNNLayer(embed_dim, hidden_dim, lambda_entangle),
            nn.Dropout(dropout),
            QBNNLayer(hidden_dim, embed_dim, lambda_entangle),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        QBNNブロックの処理
        """
        # Attention + 残差接続
        attn_out = self.attention(self.attn_norm(x))
        x = x + self.dropout(attn_out)
        
        # FFN + 残差接続
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)
        
        return x


class QBNN(nn.Module):
    """
    QBNN (Quantum-Bit Neural Network) 本体
    
    入力: エンベディング (batch, seq, embed_dim)
    出力: 処理済みエンベディング (batch, seq, embed_dim)
    
    複数のQBNNブロックを積み重ねて深い処理を行う
    """
    
    def __init__(
        self, 
        embed_dim: int = 128, 
        hidden_dim: int = 256,
        num_heads: int = 4, 
        num_layers: int = 3,
        dropout: float = 0.1,
        lambda_entangle: float = 0.35
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # QBNNブロック
        self.blocks = nn.ModuleList([
            QBNNBlock(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                lambda_entangle=lambda_entangle
            )
            for _ in range(num_layers)
        ])
        
        # 最終正規化
        self.final_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        QBNN処理
        
        Args:
            embeddings: (batch, seq, embed_dim) テキストエンベディング
        
        Returns:
            (batch, seq, embed_dim) 処理済みエンベディング
        """
        h = embeddings
        
        # 各QBNNブロックを通過
        for block in self.blocks:
            h = block(h)
        
        # 最終正規化
        h = self.final_norm(h)
        
        return h
    
    def get_quantum_info(self) -> List[Dict]:
        """量子もつれ情報を取得"""
        info = []
        for i, block in enumerate(self.blocks):
            info.append({
                'block': i,
                'attn_lambda': block.attention.lambda_attn.item(),
            })
        return info


# ============================================================
# STAGE 3: 出力ヘッド (Output Head)
# ============================================================

class OutputHead(nn.Module):
    """
    出力ヘッド
    
    入力: 処理済みエンベディング (batch, seq, embed_dim)
    出力: ロジット (batch, seq, vocab_size)
    
    次のトークンの確率分布を出力
    """
    
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # 線形射影 (embed_dim → vocab_size)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # 重み初期化
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        ロジットを計算
        
        Args:
            hidden_states: (batch, seq, embed_dim)
        
        Returns:
            (batch, seq, vocab_size) ロジット
        """
        logits = self.lm_head(hidden_states)
        return logits


# ============================================================
# トークナイザー
# ============================================================

class SimpleTokenizer:
    """
    シンプルな文字レベルトークナイザー
    """
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        
        # 特殊トークン
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        self.actual_vocab_size = 4
    
    def build_vocab(self, texts: List[str]):
        """語彙を構築"""
        char_freq = Counter()
        for text in texts:
            char_freq.update(text)
        
        special = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        sorted_chars = [c for c, _ in char_freq.most_common(self.vocab_size - len(special))]
        
        all_tokens = special + sorted_chars
        self.char_to_idx = {c: i for i, c in enumerate(all_tokens)}
        self.idx_to_char = {i: c for i, c in enumerate(all_tokens)}
        self.actual_vocab_size = len(all_tokens)
    
    def encode(self, text: str) -> List[int]:
        """テキスト → トークンID"""
        tokens = [self.bos_id]
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.unk_id))
        tokens.append(self.eos_id)
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """トークンID → テキスト"""
        chars = []
        special = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        for t in token_ids:
            if t not in special:
                chars.append(self.idx_to_char.get(t, ''))
        return ''.join(chars)


# ============================================================
# 統合パイプライン: 入力 → エンベディング → QBNN → 出力
# ============================================================

class TextEmbeddingQBNNPipeline(nn.Module):
    """
    テキストエンベディング → QBNN パイプライン
    
    処理フロー:
    ┌─────────┐    ┌─────────────────┐    ┌──────┐    ┌────────────┐
    │  入力   │ → │テキストエンベディング│ → │ QBNN │ → │   出力    │
    │(テキスト)│    │   (ベクトル化)    │    │ 処理 │    │(次トークン)│
    └─────────┘    └─────────────────┘    └──────┘    └────────────┘
    
    使用例:
        pipeline = TextEmbeddingQBNNPipeline(vocab_size=5000)
        pipeline.train_model(texts, epochs=20)
        output = pipeline.generate("量子コンピュータとは")
    """
    
    def __init__(
        self, 
        vocab_size: int = 5000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        lambda_entangle: float = 0.35
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # ========================================
        # STAGE 1: テキストエンベディング
        # ========================================
        self.text_embedding = TextEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # ========================================
        # STAGE 2: QBNN処理
        # ========================================
        self.qbnn = QBNN(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            lambda_entangle=lambda_entangle
        )
        
        # ========================================
        # STAGE 3: 出力ヘッド
        # ========================================
        self.output_head = OutputHead(
            embed_dim=embed_dim,
            vocab_size=vocab_size
        )
        
        # トークナイザー
        self.tokenizer = SimpleTokenizer(vocab_size)
        
        # デバイス
        self.device = torch.device('cpu')
        
        # パラメータ数
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def to(self, device):
        """デバイスを設定"""
        super().to(device)
        self.device = device
        return self
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        パイプライン全体の処理
        
        入力 → テキストエンベディング → QBNN → 出力
        
        Args:
            token_ids: (batch, seq) トークンID
        
        Returns:
            (batch, seq, vocab_size) ロジット
        """
        # STAGE 1: テキストエンベディング
        embeddings = self.text_embedding(token_ids)
        
        # STAGE 2: QBNN処理
        hidden_states = self.qbnn(embeddings)
        
        # STAGE 3: 出力
        logits = self.output_head(hidden_states)
        
        return logits
    
    def train_model(
        self, 
        texts: List[str], 
        epochs: int = 20, 
        batch_size: int = 16,
        lr: float = 0.001,
        seq_length: int = 64
    ):
        """
        モデルを学習
        
        Args:
            texts: 学習テキストのリスト
            epochs: エポック数
            batch_size: バッチサイズ
            lr: 学習率
            seq_length: シーケンス長
        """
        print("=" * 60)
        print("🎓 学習開始: 入力 → エンベディング → QBNN → 出力")
        print("=" * 60)
        
        # トークナイザー構築
        print("\n📚 トークナイザー構築中...")
        self.tokenizer.build_vocab(texts)
        print(f"   語彙サイズ: {self.tokenizer.actual_vocab_size}")
        
        # vocab_sizeを更新
        if self.tokenizer.actual_vocab_size != self.vocab_size:
            print(f"   語彙サイズを更新: {self.vocab_size} → {self.tokenizer.actual_vocab_size}")
            self._rebuild_for_vocab(self.tokenizer.actual_vocab_size)
        
        # データ準備
        print("\n📊 データ準備中...")
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 2:
                all_tokens.extend(tokens)
        
        print(f"   総トークン数: {len(all_tokens):,}")
        
        # シーケンス作成
        sequences = []
        for i in range(0, len(all_tokens) - seq_length - 1, seq_length // 2):
            x = all_tokens[i:i + seq_length]
            y = all_tokens[i + 1:i + seq_length + 1]
            if len(x) == seq_length:
                sequences.append((x, y))
        
        print(f"   シーケンス数: {len(sequences):,}")
        print(f"   パラメータ数: {self.num_params:,}")
        
        # オプティマイザ
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        
        # 学習ループ
        print("\n🔄 学習中...")
        self.train()
        
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(sequences)
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                
                x = torch.tensor([s[0] for s in batch], dtype=torch.long).to(self.device)
                y = torch.tensor([s[1] for s in batch], dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                
                # 前向き伝播: エンベディング → QBNN → 出力
                logits = self(x)
                
                # 損失計算
                loss = criterion(
                    logits.view(-1, self.tokenizer.actual_vocab_size),
                    y.view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / max(len(sequences) // batch_size, 1)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch + 1:3d}/{epochs}: Loss = {avg_loss:.4f}")
        
        print("\n✅ 学習完了！")
    
    def _rebuild_for_vocab(self, new_vocab_size: int):
        """語彙サイズに合わせてモデルを再構築"""
        self.vocab_size = new_vocab_size
        self.text_embedding = TextEmbedding(
            vocab_size=new_vocab_size,
            embed_dim=self.embed_dim,
            max_seq_len=self.max_seq_len,
        ).to(self.device)
        self.output_head = OutputHead(
            embed_dim=self.embed_dim,
            vocab_size=new_vocab_size
        ).to(self.device)
        self.num_params = sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        テキスト生成
        
        処理フロー:
        プロンプト → エンベディング → QBNN → 次トークン予測 → 繰り返し
        
        Args:
            prompt: 入力プロンプト
            max_length: 最大生成長
            temperature: サンプリング温度
            top_k: Top-Kサンプリング
            top_p: Top-Pサンプリング
            repetition_penalty: 繰り返しペナルティ
        
        Returns:
            生成されたテキスト
        """
        self.eval()
        
        # プロンプトをエンコード (EOSを除く)
        tokens = self.tokenizer.encode(prompt)[:-1]
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        generated = tokens[0].tolist()
        
        for _ in range(max_length):
            # 最大シーケンス長で切り詰め
            input_tokens = tokens[:, -self.max_seq_len:]
            
            # 前向き伝播: エンベディング → QBNN → 出力
            logits = self(input_tokens)
            
            # 次トークンのロジット
            next_logits = logits[0, -1, :] / temperature
            
            # 繰り返しペナルティ
            for token_id in set(generated[-20:]):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty
            
            # Top-K フィルタリング
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_vals[-1]] = float('-inf')
            
            # Top-P フィルタリング
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')
            
            # サンプリング
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # EOSで終了
            if next_token.item() == self.tokenizer.eos_id:
                break
            
            generated.append(next_token.item())
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        
        return self.tokenizer.decode(generated)
    
    def get_pipeline_info(self) -> Dict:
        """パイプライン情報を取得"""
        return {
            "pipeline": "入力 → テキストエンベディング → QBNN → 出力",
            "stages": {
                "1_text_embedding": {
                    "vocab_size": self.vocab_size,
                    "embed_dim": self.embed_dim,
                    "max_seq_len": self.max_seq_len,
                },
                "2_qbnn": {
                    "num_layers": self.qbnn.num_layers,
                    "embed_dim": self.qbnn.embed_dim,
                    "quantum_info": self.qbnn.get_quantum_info(),
                },
                "3_output": {
                    "vocab_size": self.output_head.vocab_size,
                },
            },
            "total_params": self.num_params,
        }


# ============================================================
# 学習データ
# ============================================================

def get_training_data() -> List[str]:
    """サンプル学習データ"""
    return [
        # 量子コンピューティング
        "量子コンピュータは、量子力学の原理を利用して情報を処理する革新的な計算機です。",
        "量子ビットは0と1の重ね合わせ状態を取ることができます。",
        "量子エンタングルメントは、複数の量子ビットが強く相関した状態です。",
        "量子コンピューティングは暗号解読や最適化問題で注目されています。",
        
        # ニューラルネットワーク
        "ニューラルネットワークは、人間の脳の神経回路を模倣した計算システムです。",
        "深層学習は、多層のニューラルネットワークを用いた機械学習手法です。",
        "注意機構は、入力の重要な部分に焦点を当てる技術です。",
        "トランスフォーマーは、自己注意機構を用いた革新的なアーキテクチャです。",
        
        # 人工知能
        "人工知能は、人間の知的活動をコンピュータで実現しようとする技術です。",
        "機械学習は、データから学習して改善するシステムを実現します。",
        "自然言語処理は、コンピュータが人間の言語を理解する技術です。",
        "生成AIは、テキストや画像を生成できる人工知能です。",
        
        # 対話
        "こんにちは、今日も良い天気ですね。",
        "人工知能について教えてください。",
        "量子コンピュータとは何ですか？",
        "プログラミングは創造的な活動です。",
        "未来の技術について考えてみましょう。",
        "質問があればお気軽にどうぞ。",
        "ありがとうございます。",
    ] * 50  # データを増幅


# ============================================================
# メイン実行
# ============================================================

def main():
    """
    メイン関数
    
    入力 → テキストエンベディング → QBNN → 出力 のパイプラインをデモ
    """
    print("=" * 70)
    print("🧠⚛️ テキストエンベディング → QBNN パイプライン")
    print("=" * 70)
    print()
    print("処理フロー:")
    print("┌─────────┐    ┌─────────────────┐    ┌──────┐    ┌────────────┐")
    print("│  入力   │ → │テキストエンベディング│ → │ QBNN │ → │   出力    │")
    print("│(テキスト)│    │   (ベクトル化)    │    │ 処理 │    │(次トークン)│")
    print("└─────────┘    └─────────────────┘    └──────┘    └────────────┘")
    print()
    
    # デバイス選択
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎮 CUDA GPU を使用: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) を使用")
    else:
        device = torch.device("cpu")
        print("💻 CPU を使用")
    
    # パイプライン作成
    print("\n🔧 パイプライン構築中...")
    pipeline = TextEmbeddingQBNNPipeline(
        vocab_size=3000,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=3,
        lambda_entangle=0.35
    ).to(device)
    
    # パイプライン情報表示
    info = pipeline.get_pipeline_info()
    print(f"\n📊 パイプライン情報:")
    print(f"   STAGE 1 - テキストエンベディング:")
    print(f"      語彙サイズ: {info['stages']['1_text_embedding']['vocab_size']}")
    print(f"      埋め込み次元: {info['stages']['1_text_embedding']['embed_dim']}")
    print(f"   STAGE 2 - QBNN:")
    print(f"      レイヤー数: {info['stages']['2_qbnn']['num_layers']}")
    for q_info in info['stages']['2_qbnn']['quantum_info']:
        print(f"      Block {q_info['block']}: λ = {q_info['attn_lambda']:.4f}")
    print(f"   STAGE 3 - 出力:")
    print(f"      語彙サイズ: {info['stages']['3_output']['vocab_size']}")
    print(f"   総パラメータ数: {info['total_params']:,}")
    
    # 学習データ取得
    texts = get_training_data()
    print(f"\n📚 学習データ: {len(texts)} サンプル")
    
    # 学習
    pipeline.train_model(texts, epochs=25, batch_size=16, lr=0.002)
    
    # 生成テスト
    print("\n" + "=" * 60)
    print("📝 テキスト生成テスト")
    print("=" * 60)
    
    prompts = [
        "量子コンピュータは",
        "人工知能とは",
        "こんにちは",
        "未来の技術",
    ]
    
    for prompt in prompts:
        output = pipeline.generate(
            prompt, 
            max_length=40, 
            temperature=0.7
        )
        print(f"\n入力: '{prompt}'")
        print(f"出力: {output}")
    
    print("\n✅ 完了！")
    print("\nパイプライン構造:")
    print("  入力(テキスト) → テキストエンベディング → QBNN → 出力(テキスト)")


if __name__ == '__main__':
    main()
