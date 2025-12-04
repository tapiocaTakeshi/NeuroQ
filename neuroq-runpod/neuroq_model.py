#!/usr/bin/env python3
"""
NeuroQ Model - RunPod Serverless用モデル定義
============================================
QBNN（量子ビットニューラルネットワーク）ベースの生成AIモデル

2つのモードをサポート:
- Brain Mode: 脳型散在QBNN（qbnn_brain.py / neuroquantum_brain.py ベース）
- Layered Mode: 層状QBNN-Transformer（qbnn_layered.py / neuroquantum＿layered.py ベース）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re


# ========================================
# APQB（調整可能擬似量子ビット）- 共通
# ========================================

class APQB:
    """
    APQB理論のコア
    
    - θ: 内部角度パラメータ
    - r = cos(2θ): 相関係数
    - T = |sin(2θ)|: 温度（ゆらぎ）
    - r² + T² = 1 (幾何学的制約)
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
    def theta_to_state(theta: torch.Tensor) -> torch.Tensor:
        """θ → 量子状態 [cos(θ), sin(θ)]"""
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    
    @staticmethod
    def constraint(theta: torch.Tensor) -> torch.Tensor:
        """r² + T² = 1 の検証"""
        r = APQB.theta_to_r(theta)
        T = APQB.theta_to_T(theta)
        return r**2 + T**2


# ========================================
# NeuroQ Config（両モード対応）
# ========================================

class NeuroQConfig:
    """
    NeuroQ設定
    
    mode: 'brain' または 'layered'
    """
    def __init__(
        self,
        mode: str = 'layered',  # 'brain' or 'layered'
        vocab_size: int = 8000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        num_neurons: int = 64,  # Brain mode用
        max_seq_len: int = 256,
        dropout: float = 0.1,
        lambda_entangle: float = 0.35,
        connection_density: float = 0.25,  # Brain mode用
        time_steps: int = 3,  # Brain mode用
    ):
        self.mode = mode
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.lambda_entangle = lambda_entangle
        self.connection_density = connection_density
        self.time_steps = time_steps
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, d: dict) -> 'NeuroQConfig':
        return cls(**d)


# ========================================================================
# Part 1: Layered Mode（層状QBNN-Transformer）
# ========================================================================

class QBNNLayerLayered(nn.Module):
    """
    Quantum-Bit Neural Network Layer（層状モード）
    
    数式モデル:
    1. s^(l) = tanh(h^(l)) ∈ [-1, 1]
    2. h̃^(l+1) = W^(l) h^(l) + b^(l)
    3. Δ^(l+1)_j = Σ_i J^(l)_{ij} s^(l)_i s^(l+1)_{raw,j}
    4. ĥ^(l+1) = h̃^(l+1) + λ_eff Δ^(l+1)
    5. h^(l+1) = activation(ĥ^(l+1))
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 lambda_min: float = 0.2, lambda_max: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        
        self.W = nn.Linear(input_dim, output_dim)
        self.J = nn.Parameter(torch.randn(input_dim, output_dim) * 0.02)
        self.lambda_base = nn.Parameter(torch.tensor(0.5))
        self.layer_norm = nn.LayerNorm(output_dim)
        self.register_buffer('call_count', torch.tensor(0))
    
    def forward(self, h_prev: torch.Tensor) -> torch.Tensor:
        s_prev = torch.tanh(h_prev)
        h_tilde = self.W(h_prev)
        s_raw = torch.tanh(h_tilde)
        delta = torch.einsum('...i,ij,...j->...j', s_prev, self.J, s_raw)
        
        lambda_normalized = torch.sigmoid(self.lambda_base)
        if not self.training:
            phase = float(self.call_count) * 0.2
            dynamic_factor = 0.5 + 0.5 * math.sin(phase)
            self.call_count += 1
        else:
            dynamic_factor = 0.5
        
        lambda_range = self.lambda_max - self.lambda_min
        lambda_eff = self.lambda_min + lambda_range * (lambda_normalized * 0.7 + dynamic_factor * 0.3)
        
        h_hat = h_tilde + lambda_eff * delta
        output = self.layer_norm(h_hat)
        output = F.gelu(output)
        
        return output


class QBNNAttentionLayered(nn.Module):
    """QBNN拡張Self-Attention（層状モード）"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, 
                 lambda_val: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.J_attn = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)
        self.lambda_attn = nn.Parameter(torch.tensor(float(lambda_val)))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 量子もつれ補正
        Q_norm = torch.tanh(Q)
        K_norm = torch.tanh(K)
        delta = torch.einsum('bhid,hde,bhje->bhij', Q_norm, self.J_attn, K_norm)
        attn_scores = attn_scores + self.lambda_attn * delta
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output


class QBNNTransformerBlockLayered(nn.Module):
    """QBNN-Transformer ブロック（層状モード）"""
    
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, 
                 dropout: float = 0.1, lambda_entangle: float = 0.5):
        super().__init__()
        
        self.attention = QBNNAttentionLayered(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            lambda_val=lambda_entangle
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            QBNNLayerLayered(embed_dim, hidden_dim, lambda_entangle),
            nn.Dropout(dropout),
            QBNNLayerLayered(hidden_dim, embed_dim, lambda_entangle),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attention(self.attn_norm(x), mask)
        x = x + self.dropout(attn_out)
        
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)
        
        return x


class NeuroQModelLayered(nn.Module):
    """
    NeuroQ Layered Mode: 層状QBNN-Transformer
    
    特徴:
    - QBNN-Attention: アテンションスコアへの量子補正
    - QBNN-FFN: FFN層でのもつれ補正
    - 学習可能な λ（もつれ強度）
    """
    
    def __init__(self, config: NeuroQConfig):
        super().__init__()
        self.config = config
        
        # 埋め込み層
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        # QBNN-Transformer ブロック
        self.transformer_blocks = nn.ModuleList([
            QBNNTransformerBlockLayered(
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                lambda_entangle=config.lambda_entangle
            ) for _ in range(config.num_layers)
        ])
        
        # 出力層
        self.norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # パラメータ初期化
        self.apply(self._init_weights)
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        
        token_embeds = self.token_embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        hidden_states = self.dropout(token_embeds + pos_embeds)
        
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=token_ids.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, mask)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def get_quantum_info(self) -> List[Dict]:
        """量子情報を取得"""
        info = []
        for i, block in enumerate(self.transformer_blocks):
            block_info = {
                'block': i,
                'attn_lambda': block.attention.lambda_attn.item(),
                'mode': 'layered',
            }
            info.append(block_info)
        return info


# ========================================================================
# Part 2: Brain Mode（脳型散在QBNN）
# ========================================================================

class BrainQuantumLayer(nn.Module):
    """
    脳型散在量子ビット層
    
    特徴:
    - ニューロンがバラバラに接続
    - スパースなグラフ構造
    - 時間ステップで信号伝播
    - 動的入出力対応
    """
    
    def __init__(self, num_neurons: int, input_dim: int, output_dim: int,
                 connection_density: float = 0.25, lambda_entangle: float = 0.35):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_entangle = nn.Parameter(torch.tensor(lambda_entangle))
        
        # 入力射影
        self.input_proj = nn.Linear(input_dim, num_neurons)
        
        # 各ニューロンのθパラメータ
        self.theta = nn.Parameter(torch.rand(num_neurons) * 1.0 + 0.25)
        
        # 接続マスク（スパース）
        mask = torch.rand(num_neurons, num_neurons) < connection_density
        mask.fill_diagonal_(False)  # 自己接続なし
        self.register_buffer('connection_mask', mask.float())
        
        # 重み行列
        self.weights = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.3)
        
        # もつれテンソル J
        self.J = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.1)
        
        # 出力射影
        self.output_proj = nn.Linear(num_neurons, output_dim)
    
    def get_r(self) -> torch.Tensor:
        return APQB.theta_to_r(self.theta)
    
    def get_T(self) -> torch.Tensor:
        return APQB.theta_to_T(self.theta)
    
    def forward(self, x: torch.Tensor, time_steps: int = 3) -> torch.Tensor:
        """
        前向き伝播
        
        Args:
            x: (batch, seq, input_dim) or (batch, input_dim)
            time_steps: 伝播ステップ数
        
        Returns:
            (batch, seq, output_dim) or (batch, output_dim)
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            batch, seq, _ = x.shape
            x = x.view(batch * seq, -1)
        else:
            batch = x.size(0)
            seq = None
        
        # 入力をニューロンに射影
        state = self.input_proj(x)  # (batch*seq, num_neurons)
        
        # 有効な重み（マスク適用）
        effective_weights = self.weights * self.connection_mask
        
        # 時間ステップで伝播
        for t in range(time_steps):
            # 通常の信号伝播
            signal = torch.matmul(state, effective_weights)
            
            # 量子もつれ補正
            s = torch.tanh(state)  # 正規化 [-1, 1]
            
            # もつれ計算（バッチ処理）
            J_masked = self.J * self.connection_mask
            delta = torch.einsum('bi,ij,bj->bj', s, J_masked, s)
            
            # 有効入力
            effective_input = signal + self.lambda_entangle * delta
            
            # 量子ゆらぎを追加
            T = self.get_T()
            noise = torch.randn_like(state) * T.unsqueeze(0) * 0.1
            
            state = torch.tanh(effective_input + noise)
        
        # 出力射影
        output = self.output_proj(state)
        
        # 元の形状に戻す
        if seq is not None:
            output = output.view(batch, seq, -1)
        
        return output
    
    def get_quantum_stats(self) -> Dict:
        """量子統計を取得"""
        with torch.no_grad():
            return {
                'theta_mean': self.theta.mean().item(),
                'r_mean': self.get_r().mean().item(),
                'T_mean': self.get_T().mean().item(),
                'lambda': self.lambda_entangle.item(),
                'connections': self.connection_mask.sum().item(),
            }


class BrainQuantumAttention(nn.Module):
    """脳型量子アテンション"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, 
                 num_neurons: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V 射影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 脳型量子層（アテンションスコア用）
        self.brain_layer = BrainQuantumLayer(
            num_neurons=num_neurons,
            input_dim=embed_dim,
            output_dim=embed_dim,
            connection_density=0.2,
            lambda_entangle=0.3
        )
        
        # 出力射影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 脳型量子処理でQ, K を変調
        Q = Q + 0.1 * self.brain_layer(Q, time_steps=2)
        K = K + 0.1 * self.brain_layer(K, time_steps=2)
        
        # マルチヘッド形式に変換
        Q = Q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # アテンションスコア
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Causalマスク
        causal_mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        
        return self.out_proj(context)


class BrainQuantumBlock(nn.Module):
    """脳型量子トランスフォーマーブロック"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4,
                 num_neurons: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.attention = BrainQuantumAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_neurons=num_neurons,
            dropout=dropout
        )
        
        self.ffn = BrainQuantumLayer(
            num_neurons=num_neurons * 2,
            input_dim=embed_dim,
            output_dim=embed_dim,
            connection_density=0.25,
            lambda_entangle=0.35
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        ffn_out = self.ffn(self.norm2(x), time_steps=2)
        x = x + self.dropout(ffn_out)
        
        return x


class NeuroQModelBrain(nn.Module):
    """
    NeuroQ Brain Mode: 脳型散在QBNN
    
    特徴:
    - 各ニューロンが独立した量子ビット（APQB）
    - ニューロン間の接続はグラフ構造（スパース）
    - 時間ステップで信号が伝播
    - 量子もつれが任意のニューロン間で発生
    """
    
    def __init__(self, config: NeuroQConfig):
        super().__init__()
        self.config = config
        
        # 埋め込み
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(config.max_seq_len, config.embed_dim) * 0.02)
        
        # 脳型量子ブロック
        self.blocks = nn.ModuleList([
            BrainQuantumBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                num_neurons=config.num_neurons,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        self.apply(self._init_weights)
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq = token_ids.shape
        
        tok_emb = self.token_embedding(token_ids)
        pos_emb = self.pos_embedding[:seq].unsqueeze(0)
        
        h = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            h = block(h)
        
        h = self.final_norm(h)
        logits = self.output_head(h)
        
        return logits
    
    def get_quantum_info(self) -> List[Dict]:
        """量子統計を取得"""
        info = []
        for i, block in enumerate(self.blocks):
            attn_stats = block.attention.brain_layer.get_quantum_stats()
            ffn_stats = block.ffn.get_quantum_stats()
            
            info.append({
                'block': i,
                'mode': 'brain',
                'attn_r': attn_stats['r_mean'],
                'attn_T': attn_stats['T_mean'],
                'attn_lambda': attn_stats['lambda'],
                'ffn_r': ffn_stats['r_mean'],
                'ffn_T': ffn_stats['T_mean'],
                'ffn_lambda': ffn_stats['lambda'],
                'connections': ffn_stats['connections'],
            })
        return info


# ========================================================================
# 統合 NeuroQ Model
# ========================================================================

class NeuroQModel(nn.Module):
    """
    NeuroQ: QBNN-LLM
    
    2つのモードをサポート:
    - 'brain': 脳型散在QBNN（ニューロンがバラバラに接続）
    - 'layered': 層状QBNN-Transformer（Attention + FFN）
    """
    
    def __init__(self, config: NeuroQConfig):
        super().__init__()
        self.config = config
        
        # モードに応じたモデルを作成
        if config.mode == 'brain':
            self.model = NeuroQModelBrain(config)
        else:  # 'layered'
            self.model = NeuroQModelLayered(config)
        
        self.num_params = self.model.num_params
    
    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(token_ids, mask)
    
    def get_quantum_info(self) -> List[Dict]:
        return self.model.get_quantum_info()
    
    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> 'NeuroQModel':
        """チェックポイントからモデルをロード"""
        checkpoint = torch.load(path, map_location=device)
        config = NeuroQConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.model.load_state_dict(checkpoint['model_state'])
        return model
    
    def save_checkpoint(self, path: str):
        """チェックポイントを保存"""
        torch.save({
            'config': self.config.to_dict(),
            'model_state': self.model.state_dict(),
        }, path)


# ========================================
# Tokenizer（共通）
# ========================================

class NeuroQTokenizer:
    """NeuroQ トークナイザー"""
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        self.actual_vocab_size = 4
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """語彙構築"""
        char_freq = Counter()
        for text in texts:
            char_freq.update(text)
        
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        sorted_chars = [char for char, freq in char_freq.most_common() if freq >= min_freq]
        sorted_chars = sorted_chars[:self.vocab_size - len(special_tokens)]
        
        all_tokens = special_tokens + sorted_chars
        self.char_to_idx = {c: i for i, c in enumerate(all_tokens)}
        self.idx_to_char = {i: c for i, c in enumerate(all_tokens)}
        
        self.actual_vocab_size = len(all_tokens)
        return self
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """エンコード"""
        tokens = []
        if add_special:
            tokens.append(self.bos_id)
        
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.unk_id))
        
        if add_special:
            tokens.append(self.eos_id)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """デコード"""
        chars = []
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        
        for t in token_ids:
            if skip_special and t in special_ids:
                continue
            char = self.idx_to_char.get(t, self.unk_token)
            if char not in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                chars.append(char)
        
        return ''.join(chars)
    
    def save(self, path: str):
        """保存"""
        data = {
            'char_to_idx': self.char_to_idx,
            'vocab_size': self.vocab_size,
            'actual_vocab_size': self.actual_vocab_size,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> 'NeuroQTokenizer':
        """読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(i): c for c, i in self.char_to_idx.items()}
        self.vocab_size = data['vocab_size']
        self.actual_vocab_size = data.get('actual_vocab_size', len(self.char_to_idx))
        return self


# ========================================
# NeuroQ Generator (推論用)
# ========================================

class NeuroQGenerator:
    """
    NeuroQ 推論用クラス
    
    RunPod Serverless から呼び出される
    Brain/Layered 両モード対応
    """
    
    def __init__(self, model: NeuroQModel, tokenizer: NeuroQTokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def load(cls, model_path: str, tokenizer_path: str, device: str = "cuda") -> 'NeuroQGenerator':
        """モデルとトークナイザーをロード"""
        model = NeuroQModel.load_from_checkpoint(model_path, device)
        tokenizer = NeuroQTokenizer().load(tokenizer_path)
        return cls(model, tokenizer, device)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> str:
        """
        テキスト生成
        
        Args:
            prompt: 入力プロンプト
            max_tokens: 最大生成トークン数
            temperature: サンプリング温度
            top_k: Top-K サンプリング
            top_p: Top-P (Nucleus) サンプリング
            repetition_penalty: 繰り返しペナルティ
        
        Returns:
            生成されたテキスト
        """
        tokens = self.tokenizer.encode(prompt, add_special=True)[:-1]
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        generated = tokens[0].tolist()
        max_seq_len = self.model.config.max_seq_len
        
        for _ in range(max_tokens):
            input_tokens = tokens[:, -max_seq_len:] if tokens.size(1) > max_seq_len else tokens
            
            logits = self.model(input_tokens)
            next_logits = logits[0, -1, :] / temperature
            
            # 繰り返しペナルティ
            recent_tokens = set(generated[-30:])
            for token_id in recent_tokens:
                next_logits[token_id] /= repetition_penalty
            
            # Top-K
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                indices_to_remove = next_logits < top_k_vals[-1]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-P
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == self.tokenizer.eos_id:
                break
            
            generated.append(next_token.item())
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        
        return self.tokenizer.decode(generated)
    
    def get_model_info(self) -> dict:
        """モデル情報を取得"""
        return {
            "mode": self.model.config.mode,
            "vocab_size": self.model.config.vocab_size,
            "embed_dim": self.model.config.embed_dim,
            "hidden_dim": self.model.config.hidden_dim,
            "num_heads": self.model.config.num_heads,
            "num_layers": self.model.config.num_layers,
            "num_neurons": self.model.config.num_neurons,
            "max_seq_len": self.model.config.max_seq_len,
            "num_params": self.model.num_params,
            "device": str(self.device),
        }


# ========================================
# モード別ファクトリ関数
# ========================================

def create_neuroq_brain(
    vocab_size: int = 8000,
    embed_dim: int = 128,
    num_neurons: int = 64,
    num_layers: int = 3,
    **kwargs
) -> NeuroQModel:
    """Brain モードのNeuroQを作成"""
    config = NeuroQConfig(
        mode='brain',
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_neurons=num_neurons,
        num_layers=num_layers,
        **kwargs
    )
    return NeuroQModel(config)


def create_neuroq_layered(
    vocab_size: int = 8000,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 3,
    **kwargs
) -> NeuroQModel:
    """Layered モードのNeuroQを作成"""
    config = NeuroQConfig(
        mode='layered',
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        **kwargs
    )
    return NeuroQModel(config)
