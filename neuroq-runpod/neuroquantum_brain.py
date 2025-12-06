#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗  ██████╗ ██╗   ██╗ █████╗      ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔═══██╗██║   ██║██╔══██╗     ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║   ██║██║   ██║███████║     ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║▄▄ ██║██║   ██║██╔══██║     ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║     ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝     ║
║                                                                               ║
║    ██████╗                                                                   ║
║   ██╔═══██╗                                                                  ║
║   ██║   ██║                                                                  ║
║   ██║   ██║                                                                  ║
║   ╚██████╔╝                                                                  ║
║    ╚═════╝                                                                   ║
║                                                                               ║
║   neuroQ Brain: 脳型散在QBNNによる生成AI                                      ║
║   独自の量子もつれニューラルネットワークによる生成AI                          ║
║                                                                               ║
║   参照元: qbnn_brain.py                                                       ║
║   - QBNNBrain: 純粋Python版の脳型散在ネットワーク                             ║
║   - QBNNBrainTorch: PyTorch版の脳型散在ネットワーク                          ║
║   - QuantumNeuron: 単一量子ビットニューロン                                   ║
║                                                                               ║
║   特徴:                                                                       ║
║   - 各ニューロンが独立した量子ビット（APQB）                                   ║
║   - ニューロン間の接続はグラフ構造（スパース）                                 ║
║   - 時間ステップで信号が伝播                                                   ║
║   - 量子もつれが任意のニューロン間で発生                                       ║
║   - 動的入出力（本物の脳のように入力/出力ニューロンが変化）                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# ========================================
# qbnn_brain.py からコアコンポーネントをインポート
# ========================================
try:
    from qbnn_brain import (
        QuantumNeuron,      # 単一量子ビットニューロン
        QBNNBrain,          # 純粋Python版脳型QBNN
        QBNNBrainTorch,     # PyTorch版脳型QBNN
    )
    QBNN_BRAIN_AVAILABLE = True
    print("✅ qbnn_brain.py からコアコンポーネントをインポートしました")
except ImportError:
    QBNN_BRAIN_AVAILABLE = False
    # オプションなので警告を表示しない（内蔵コンポーネントで動作します）


print("=" * 70)
print("🧠⚛️ ニューロQ Brain")
print("   脳型散在量子ビットネットワークによる生成AI")
print("   (qbnn_brain.py ベース)")
print("=" * 70)


# ========================================
# APQB（調整可能擬似量子ビット）- 共通ユーティリティ
# ========================================

class APQB:
    """
    APQB理論のコア
    
    qbnn_brain.py の QuantumNeuron と同じ理論に基づく:
    - θ: 内部角度パラメータ
    - r = cos(2θ): 相関係数
    - T = |sin(2θ)|: 温度（ゆらぎ）
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
        """量子測定（確率的に0 or 1）"""
        prob_1 = torch.sin(theta) ** 2
        return (torch.rand_like(prob_1) < prob_1).float()


# ========================================
# 脳型量子ニューロン層
# ========================================

class BrainQuantumLayer(nn.Module):
    """
    脳型散在量子ビット層
    
    qbnn_brain.py の QBNNBrainTorch を基盤として使用可能
    
    - ニューロンがバラバラに接続
    - スパースなグラフ構造
    - 時間ステップで信号伝播
    - 動的入出力対応（QBNN_BRAIN_AVAILABLE時）
    """
    
    def __init__(self, num_neurons: int, input_dim: int, output_dim: int,
                 connection_density: float = 0.25, lambda_entangle: float = 0.35,
                 use_qbnn_brain: bool = True):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_qbnn_brain = use_qbnn_brain and QBNN_BRAIN_AVAILABLE
        
        if self.use_qbnn_brain:
            # qbnn_brain.py の QBNNBrainTorch を使用
            self.qbnn_core = QBNNBrainTorch(
                num_neurons=num_neurons,
                max_input_size=input_dim,
                max_output_size=output_dim,
                connection_density=connection_density
            )
            self.qbnn_core.lambda_entangle = nn.Parameter(torch.tensor(lambda_entangle))
            
            # QBNNBrainTorchのパラメータを参照
            self.theta = self.qbnn_core.theta
            self.connection_mask = self.qbnn_core.connection_mask
            self.weights = self.qbnn_core.weights
            self.J = self.qbnn_core.J
            self.lambda_entangle = self.qbnn_core.lambda_entangle
        else:
            # フォールバック：内蔵実装
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
        if self.use_qbnn_brain:
            return self.qbnn_core.get_r()
        return APQB.theta_to_r(self.theta)
    
    def get_T(self) -> torch.Tensor:
        if self.use_qbnn_brain:
            return self.qbnn_core.get_T()
        return APQB.theta_to_T(self.theta)
    
    def forward(self, x: torch.Tensor, time_steps: int = 3, 
                dynamic_selection: bool = False) -> torch.Tensor:
        """
        前向き伝播
        
        qbnn_brain.py の QBNNBrainTorch を使用可能
        
        Args:
            x: (batch, seq, input_dim) or (batch, input_dim)
            time_steps: 伝播ステップ数
            dynamic_selection: 動的ニューロン選択を使用するか
        
        Returns:
            (batch, seq, output_dim) or (batch, output_dim)
        """
        # 入力形状を保持
        original_shape = x.shape
        if len(original_shape) == 3:
            batch, seq, _ = x.shape
            x = x.view(batch * seq, -1)
        else:
            batch = x.size(0)
            seq = None
        
        if self.use_qbnn_brain:
            # qbnn_brain.py の QBNNBrainTorch を使用
            output, in_neurons, out_neurons = self.qbnn_core(
                x, 
                input_size=self.input_dim,
                output_size=self.output_dim,
                time_steps=time_steps,
                dynamic_selection=dynamic_selection
            )
            
            # 元の形状に戻す
            if seq is not None:
                output = output.view(batch, seq, -1)
            
            return output
        else:
            # フォールバック：内蔵実装
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
            if self.use_qbnn_brain:
                # qbnn_brain.py の情報を取得
                info = self.qbnn_core.get_quantum_info()
                return {
                    'theta_mean': info['theta_mean'],
                    'r_mean': info['r_mean'],
                    'T_mean': info['T_mean'],
                    'lambda': info['lambda'],
                    'connections': info['connections'],
                    'sensitivity_mean': info.get('sensitivity_mean', 0),
                    'output_tendency_mean': info.get('output_tendency_mean', 0),
                    'source': 'qbnn_brain.py',
                }
            else:
                return {
                    'theta_mean': self.theta.mean().item(),
                    'r_mean': self.get_r().mean().item(),
                    'T_mean': self.get_T().mean().item(),
                    'lambda': self.lambda_entangle.item(),
                    'connections': self.connection_mask.sum().item(),
                    'source': 'builtin',
                }


# ========================================
# 脳型アテンション
# ========================================

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
        """
        GPT標準: Multi-Head Causal Self-Attention（脳型量子拡張版）
        
        Args:
            x: (batch, seq, embed_dim)
            mask: Optional attention mask (Noneの場合はCausal Maskを自動生成)
        
        Returns:
            (batch, seq, embed_dim)
        """
        batch, seq, _ = x.shape
        
        # Q, K, V 計算
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 脳型量子処理でQ, K を変調（量子拡張）
        Q = Q + 0.1 * self.brain_layer(Q, time_steps=2)
        K = K + 0.1 * self.brain_layer(K, time_steps=2)
        
        # マルチヘッド形式に変換
        Q = Q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        K = K.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # GPT標準: アテンションスコア計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, heads, seq, seq)
        
        # GPT標準: Causal Mask適用
        if mask is not None:
            # 提供されたマスクを使用
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            # Causal Mask自動生成（GPT標準）
            causal_mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # GPT標準: Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # GPT標準: アテンション適用
        context = torch.matmul(attn_weights, V)  # (batch, heads, seq, head_dim)
        
        # 元の形状に戻す
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        
        # GPT標準: 出力射影
        return self.out_proj(context)


# ========================================
# 脳型トランスフォーマーブロック
# ========================================

class BrainQuantumBlock(nn.Module):
    """
    GPTデコーダーブロック（脳型量子拡張版）
    
    GPT標準構造:
    1. Pre-norm LayerNorm
    2. Multi-Head Causal Self-Attention
    3. Residual Connection
    4. Pre-norm LayerNorm
    5. Feed-Forward Network (標準FFN + 脳型量子拡張)
    6. Residual Connection
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4,
                 num_neurons: int = 32, dropout: float = 0.1,
                 ffn_expansion: int = 4):
        super().__init__()
        
        # Pre-norm LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 脳型量子アテンション
        self.attention = BrainQuantumAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_neurons=num_neurons,
            dropout=dropout
        )
        
        # GPT標準FFN: Linear → GELU → Linear
        ffn_hidden = embed_dim * ffn_expansion
        self.ffn_standard = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 脳型量子拡張FFN（オプション）
        self.ffn_quantum = BrainQuantumLayer(
            num_neurons=num_neurons * 2,
            input_dim=embed_dim,
            output_dim=embed_dim,
            connection_density=0.25,
            lambda_entangle=0.35
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        GPTデコーダーフォワード
        
        Args:
            x: (batch, seq, embed_dim)
            mask: Optional attention mask
        
        Returns:
            (batch, seq, embed_dim)
        """
        # 1. Pre-norm + Multi-Head Causal Self-Attention + Residual
        residual = x
        x = self.norm1(x)
        attn_out = self.attention(x, mask)
        x = residual + self.dropout(attn_out)
        
        # 2. Pre-norm + Feed-Forward Network + Residual
        residual = x
        x = self.norm2(x)
        
        # 標準FFN + 脳型量子拡張（ブレンド）
        ffn_standard_out = self.ffn_standard(x)
        ffn_quantum_out = self.ffn_quantum(x, time_steps=2)
        
        # ブレンド比率: 標準FFN 70% + 量子拡張 30%
        ffn_out = 0.7 * ffn_standard_out + 0.3 * ffn_quantum_out
        
        x = residual + ffn_out
        
        return x


# ========================================
# ニューロQ Brain モデル
# ========================================

class NeuroQuantumBrain(nn.Module):
    """
    ニューロQ Brain - GPT型デコーダーのみのTransformer（脳型量子拡張版）
    
    処理フロー（図2-4に準拠）:
    1. 入力テキスト → トークン化 → トークンID
    2. トークンID → テキストエンベディング（Text Embedding + Position Embedding）
    3. テキストエンベディング → GPT型デコーダーのみのTransformer（N個のDecoder Blocks）
    4. Transformer出力 → 後処理ステップ（Final LayerNorm + Output Head）
    5. 後処理ステップ → 出力テキスト（ロジット）
    
    GPT標準構造:
    - Text Embedding + Position Embedding（テキストエンベディング）
    - Dropout
    - N個のGPT Decoder Blocks（Pre-norm + Attention + FFN）
    - Final LayerNorm
    - Output Head (Linear to vocab_size)
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 3,
                 num_neurons: int = 48, max_seq_len: int = 256,
                 dropout: float = 0.1, ffn_expansion: int = 4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # GPT標準: Text Embedding + Position Embedding
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)  # テキストエンベディング
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # GPT Decoder Blocks
        self.blocks = nn.ModuleList([
            BrainQuantumBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_neurons=num_neurons,
                dropout=dropout,
                ffn_expansion=ffn_expansion
            ) for _ in range(num_layers)
        ])
        
        # GPT標準: Final LayerNorm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # GPT標準: Output Head (weight tying可能だが、ここでは独立)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # GPT標準: Embedding Dropout
        self.dropout = nn.Dropout(dropout)
        
        # パラメータ初期化（GPT標準）
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """GPT標準の重み初期化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        GPT型デコーダーのみのTransformer フォワード（図2-4のフローに準拠）
        
        処理ステップ:
        1. トークンID → テキストエンベディング（Text Embedding + Position Embedding）
        2. テキストエンベディング → GPT型デコーダーのみのTransformer
        3. Transformer出力 → 後処理ステップ（Final LayerNorm + Output Head）
        4. 後処理ステップ → ロジット（出力テキスト生成用）
        
        Args:
            x: (batch, seq) トークンID（トークン化済みのテキスト）
            mask: Optional attention mask (Noneの場合はCausal Maskを自動生成)
        
        Returns:
            (batch, seq, vocab_size) ロジット（後処理ステップ後の出力）
        """
        batch, seq = x.shape
        
        # ステップ1: トークンID → テキストエンベディング
        # Text Embedding: トークンIDをベクトルに変換（テキストエンベディング）
        text_embeds = self.text_embedding(x)  # (batch, seq, embed_dim)
        # Position Embedding: 位置情報を追加
        positions = torch.arange(seq, device=x.device).unsqueeze(0).expand(batch, -1)
        pos_embeds = self.position_embedding(positions)  # (batch, seq, embed_dim)
        # 埋め込みの合成 + Dropout
        h = self.dropout(text_embeds + pos_embeds)
        
        # Causal Mask生成（maskがNoneの場合）
        if mask is None:
            mask = torch.tril(torch.ones(seq, seq, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        # ステップ2: テキストエンベディング → GPT型デコーダーのみのTransformer
        # N個のGPT Decoder Blocks（Pre-norm + Multi-Head Causal Self-Attention + FFN）
        for block in self.blocks:
            h = block(h, mask)
        
        # ステップ3: Transformer出力 → 後処理ステップ
        # Final LayerNorm
        h = self.final_norm(h)
        # Output Head: ベクトル → 語彙確率への変換
        logits = self.output_head(h)  # (batch, seq, vocab_size)
        
        # ステップ4: ロジット（出力テキスト生成用）
        return logits
    
    def get_quantum_report(self) -> str:
        """量子統計レポート"""
        report = "\n⚛️ 量子統計レポート\n" + "-" * 40 + "\n"
        
        for i, block in enumerate(self.blocks):
            # アテンションの統計
            attn_stats = block.attention.brain_layer.get_quantum_stats()
            # FFNの統計（新しい構造ではffn_quantumを使用）
            ffn_stats = block.ffn_quantum.get_quantum_stats()
            
            report += f"Block {i}:\n"
            report += f"  Attention: r={attn_stats['r_mean']:.3f}, T={attn_stats['T_mean']:.3f}, λ={attn_stats['lambda']:.3f}\n"
            report += f"  FFN:       r={ffn_stats['r_mean']:.3f}, T={ffn_stats['T_mean']:.3f}, λ={ffn_stats['lambda']:.3f}\n"
        
        return report
    
    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, max_length: int = 50,
                 temperature_min: float = 0.4, temperature_max: float = 0.9,
                 top_k: int = 40, top_p: float = 0.9,
                 repetition_penalty: float = 1.2) -> torch.Tensor:
        """
        テキスト生成（温度範囲制約版）
        
        Args:
            temperature_min: 最小温度（θ=π/4のとき、2θ=90°）
            temperature_max: 最大温度
            ※ 2θが45°〜135°の範囲で動くように制約
        """
        self.eval()
        
        tokens = start_tokens.clone()
        generated = []
        
        for step in range(max_length):
            # 入力準備
            x = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
            if x.size(1) > self.max_seq_len:
                x = x[:, -self.max_seq_len:]
            
            # 予測
            logits = self(x)
            
            # 量子状態から温度を動的に計算
            # 2θが45°〜135°の範囲（π/4〜3π/4）になるように制約
            if len(self.blocks) > 0:
                # ブロックのθから相関係数rを取得（新しい構造ではffn_quantumを使用）
                r_vals = []
                for block in self.blocks:
                    r_vals.append(block.ffn_quantum.get_r().mean().item())
                r_mean = np.mean(r_vals)
                
                # r ∈ [-1, 1] を温度範囲にマッピング
                # r = cos(2θ) なので、2θ ∈ [π/4, 3π/4] のとき r ∈ [-0.707, 0.707]
                # これを [temperature_min, temperature_max] にマッピング
                r_clamped = np.clip(r_mean, -0.707, 0.707)
                # 正規化: [-0.707, 0.707] → [0, 1]
                t_normalized = (r_clamped + 0.707) / 1.414
                # 温度にマッピング
                temperature = temperature_min + t_normalized * (temperature_max - temperature_min)
            else:
                temperature = (temperature_min + temperature_max) / 2
            
            next_logits = logits[0, -1] / temperature
            
            # 繰り返しペナルティ
            if len(generated) > 0:
                for prev_token in set(generated[-20:]):
                    next_logits[prev_token] /= repetition_penalty
            
            # 量子ゆらぎを追加（制約された範囲で）
            if len(self.blocks) > 0:
                T_mean = self.blocks[-1].ffn_quantum.get_T().mean()
                # T = |sin(2θ)|, 2θ ∈ [45°, 135°] なら T ∈ [0.707, 1.0]
                T_clamped = max(0.707, min(1.0, T_mean.item()))
                quantum_noise = torch.randn_like(next_logits) * T_clamped * 0.15
                next_logits = next_logits + quantum_noise
            
            # Top-K フィルタリング
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-P フィルタリング
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')
            
            # サンプリング
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # トークン連結
            if tokens.dim() == 0:
                tokens = next_token.view(1)
            else:
                tokens = torch.cat([tokens, next_token.view(-1)], dim=0)
            generated.append(next_token.item())
        
        return tokens


# ========================================
# トークナイザー（サブワード対応）
# ========================================

class BrainTokenizer:
    """
    サブワード対応トークナイザー（図2-4のトークン化ステップに準拠）
    
    処理フロー:
    1. 入力テキスト → トークン化（テキストを個々のトークンに分割）
    2. トークン → トークンID（各トークンを数値IDに変換）
    
    特徴:
    - 文字単位 + 頻出バイグラム/単語単位
    - 最大50,000トークン対応
    """
    
    def __init__(self, max_vocab: int = 50000, use_subword: bool = True):
        self.max_vocab = max_vocab
        self.use_subword = use_subword
        self.token2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx2token = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.vocab_size = 4
        self.subword_list = []
    
    def fit(self, texts: List[str]):
        """語彙を構築（文字 + サブワード）"""
        
        # 1. 文字カウント
        char_counts = Counter()
        for text in texts:
            char_counts.update(list(text))
        
        # 2. 単一文字を追加
        for char, _ in char_counts.most_common():
            if char not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[char] = idx
                self.idx2token[idx] = char
        
        # 3. サブワード（バイグラム、トライグラム、頻出単語）
        if self.use_subword:
            # バイグラム
            bigram_counts = Counter()
            for text in texts:
                chars = list(text)
                for i in range(len(chars) - 1):
                    bigram = chars[i] + chars[i + 1]
                    bigram_counts[bigram] += 1
            
            # トライグラム
            trigram_counts = Counter()
            for text in texts:
                chars = list(text)
                for i in range(len(chars) - 2):
                    trigram = chars[i] + chars[i + 1] + chars[i + 2]
                    trigram_counts[trigram] += 1
            
            # 頻出単語（日本語対応）
            word_counts = Counter()
            for text in texts:
                # 簡易的な単語分割（句読点、スペースで分割）
                words = re.split(r'[、。，．！？\s\n]+', text)
                for word in words:
                    if 2 <= len(word) <= 8:
                        word_counts[word] += 1
            
            # サブワードを追加（頻度順）
            available_slots = self.max_vocab - len(self.token2idx)
            
            # 各タイプから均等に追加
            subwords_to_add = []
            
            # 頻出単語（最大40%）
            for word, cnt in word_counts.most_common(available_slots * 2 // 5):
                if cnt >= 5 and word not in self.token2idx:
                    subwords_to_add.append((word, cnt * 3))  # 優先度高め
            
            # バイグラム（最大35%）
            for bigram, cnt in bigram_counts.most_common(available_slots * 7 // 20):
                if cnt >= 10 and bigram not in self.token2idx:
                    subwords_to_add.append((bigram, cnt * 2))
            
            # トライグラム（最大25%）
            for trigram, cnt in trigram_counts.most_common(available_slots // 4):
                if cnt >= 5 and trigram not in self.token2idx:
                    subwords_to_add.append((trigram, cnt))
            
            # 頻度でソートして追加
            subwords_to_add.sort(key=lambda x: -x[1])
            
            for token, _ in subwords_to_add:
                if len(self.token2idx) >= self.max_vocab:
                    break
                if token not in self.token2idx:
                    idx = len(self.token2idx)
                    self.token2idx[token] = idx
                    self.idx2token[idx] = token
                    self.subword_list.append(token)
        
        self.vocab_size = len(self.token2idx)
        print(f"   語彙サイズ: {self.vocab_size} (サブワード: {len(self.subword_list)})")
    
    def encode(self, text: str) -> List[int]:
        """
        エンコード（図2-4のトークン化ステップ）
        
        処理:
        1. 入力テキスト → トークン化（テキストを個々のトークンに分割）
        2. トークン → トークンID（各トークンを数値IDに変換）
        
        Args:
            text: 入力テキスト（例: "This is an example."）
        
        Returns:
            トークンIDのリスト（例: [40134, 2052, 133, 389, 12]）
        """
        tokens = []
        i = 0
        text_len = len(text)
        
        # トークン化: テキストを個々のトークンに分割
        while i < text_len:
            matched = False
            
            # 長いサブワードから優先的にマッチ
            for length in [8, 7, 6, 5, 4, 3, 2]:
                if i + length <= text_len:
                    substr = text[i:i+length]
                    if substr in self.token2idx:
                        # トークンID: 各トークンを数値IDに変換
                        tokens.append(self.token2idx[substr])
                        i += length
                        matched = True
                        break
            
            if not matched:
                # 単一文字
                char = text[i]
                tokens.append(self.token2idx.get(char, 1))  # UNK
                i += 1
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """トークンをテキストに変換"""
        result = []
        for t in tokens:
            if t not in [0, 1, 2, 3]:  # 特殊トークンスキップ
                token = self.idx2token.get(t, '')
                result.append(token)
        return ''.join(result)


# ========================================
# ニューロQ Brain AI
# ========================================

class NeuroQuantumBrainAI:
    """ニューロQ Brain 生成AI"""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 3, num_neurons: int = 75,
                 max_vocab: int = 50000):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.max_vocab = max_vocab
        
        self.tokenizer = BrainTokenizer(max_vocab)
        self.model = None
        
        # デバイス選択: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🍎 Apple Silicon GPU (MPS) を使用")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🎮 NVIDIA GPU (CUDA) を使用")
        else:
            self.device = torch.device("cpu")
            print("💻 CPU を使用")
    
    def train(self, texts: List[str], epochs: int = 20, batch_size: int = 16,
              lr: float = 0.001, seq_length: int = 64):
        """学習"""
        print("\n🎓 学習開始...")
        
        # トークナイザー構築
        self.tokenizer.fit(texts)
        
        # データ準備
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 2:
                all_tokens.extend(tokens)
        
        print(f"   総トークン数: {len(all_tokens)}")
        
        # モデル構築
        self.model = NeuroQuantumBrain(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_neurons=self.num_neurons,
            max_seq_len=256,
            dropout=0.1
        ).to(self.device)
        
        # オプティマイザ
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # データセット作成
        sequences = []
        for i in range(0, len(all_tokens) - seq_length - 1, seq_length // 2):
            x = all_tokens[i:i+seq_length]
            y = all_tokens[i+1:i+seq_length+1]
            if len(x) == seq_length:
                sequences.append((x, y))
        
        print(f"   シーケンス数: {len(sequences)}")
        
        # 学習ループ
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(sequences)
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                
                x = torch.tensor([s[0] for s in batch], dtype=torch.long).to(self.device)
                y = torch.tensor([s[1] for s in batch], dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits.view(-1, self.tokenizer.vocab_size), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / max(len(sequences) // batch_size, 1)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
        
        print("   学習完了！")
    
    def generate(self, prompt: str, max_length: int = 50,
                 temperature_min: float = 0.4, temperature_max: float = 0.9,
                 top_k: int = 40, top_p: float = 0.9) -> str:
        """
        テキスト生成（温度範囲制約版）
        
        Args:
            temperature_min: 最小温度（2θ=90°付近）
            temperature_max: 最大温度（2θが45°or135°付近）
            ※ 2θが45°〜135°の範囲で動くことで、適度な揺らぎを維持
        """
        if self.model is None:
            return "モデルが学習されていません"
        
        self.model.eval()
        
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) == 0:
            tokens = [2]  # <BOS>
        
        tokens = torch.tensor(tokens, dtype=torch.long).to(self.device)
        
        generated = self.model.generate(
            tokens, max_length=max_length,
            temperature_min=temperature_min, temperature_max=temperature_max,
            top_k=top_k, top_p=top_p
        )
        
        return self.tokenizer.decode(generated.cpu().tolist())
    
    def get_report(self) -> str:
        """モデルレポート"""
        if self.model is None:
            return "モデルなし"
        
        return self.model.get_quantum_report()


# ========================================
# 学習データ
# ========================================

def fetch_huggingface_data(max_samples: int = 5000) -> List[str]:
    """Hugging Faceからデータを取得"""
    texts = []
    
    try:
        from datasets import load_dataset
        print("   📡 Hugging Faceからデータ取得中...")
        
        # 日本語Wikipedia
        try:
            ds = load_dataset("range3/wiki40b-ja", split="train", streaming=True)
            count = 0
            for item in ds:
                if 'text' in item and len(item['text']) > 50:
                    texts.append(item['text'][:500])
                    count += 1
                    if count >= max_samples // 3:
                        break
        except:
            pass
        
        # 英語データ
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            for item in ds[:max_samples // 3]:
                if 'text' in item and len(item['text']) > 30:
                    texts.append(item['text'])
        except:
            pass
        
        # 日本語対話
        try:
            ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
            for item in ds[:max_samples // 3]:
                if 'output' in item:
                    texts.append(item['output'])
        except:
            pass
        
        print(f"   ✅ {len(texts):,} サンプル取得")
        
    except ImportError:
        print("   ⚠️ datasets未インストール、組み込みデータのみ使用")
    except Exception as e:
        print(f"   ⚠️ データ取得失敗: {e}")
    
    return texts


def get_training_data(use_huggingface: bool = False) -> List[str]:
    """学習データを取得（50,000トークン対応大規模版）"""
    
    base_texts = [
        # ===== 量子コンピューティング =====
        "量子コンピュータは、量子力学の原理を利用して情報を処理する革新的な計算機です。",
        "量子ビットは0と1の重ね合わせ状態を取ることができます。この性質により並列計算が可能になります。",
        "量子エンタングルメントは、複数の量子ビットが強く相関した状態です。量子通信や量子暗号に応用されています。",
        "量子コンピューティングは暗号解読や最適化問題で注目されています。将来的には創薬や材料開発にも貢献するでしょう。",
        "量子超越性とは、量子コンピュータが古典コンピュータよりも高速に計算できることを示す概念です。",
        "量子アルゴリズムには、ショアのアルゴリズムやグローバーのアルゴリズムなどがあります。",
        "量子誤り訂正は、量子計算の信頼性を高めるための重要な技術です。",
        "超伝導量子ビットやイオントラップ量子ビットなど、様々な実装方式が研究されています。",
        "量子アニーリングは、組み合わせ最適化問題を解くための量子計算手法です。",
        "量子テレポーテーションは、量子情報を離れた場所に転送する技術です。",
        "量子暗号通信は、盗聴が理論的に不可能な通信方式です。",
        "量子センサーは、極めて高精度な測定を可能にする技術です。",
        
        # ===== ニューラルネットワーク =====
        "ニューラルネットワークは、人間の脳の神経回路を模倣した計算システムです。入力層、隠れ層、出力層から構成されます。",
        "深層学習は、多層のニューラルネットワークを用いた機械学習手法です。画像認識や音声認識で大きな成功を収めています。",
        "注意機構は、入力の重要な部分に焦点を当てる技術です。機械翻訳の品質を大幅に向上させました。",
        "トランスフォーマーは、自己注意機構を用いた革新的なアーキテクチャです。GPTやBERTの基盤となっています。",
        "畳み込みニューラルネットワークは、画像認識に特化したアーキテクチャです。フィルタを用いて特徴を抽出します。",
        "再帰型ニューラルネットワークは、時系列データを処理するのに適しています。過去の情報を記憶して利用できます。",
        "長短期記憶ネットワークは、長期的な依存関係を学習できるRNNの一種です。",
        "バッチ正規化は、学習を安定させるための技術です。各層の入力を正規化します。",
        "ドロップアウトは、過学習を防ぐための正則化手法です。ランダムにニューロンを無効化します。",
        "活性化関数には、ReLU、シグモイド、tanhなどがあります。非線形性を導入する重要な役割を果たします。",
        "逆伝播法は、ニューラルネットワークの学習に使われる勾配計算アルゴリズムです。",
        "確率的勾配降下法は、ミニバッチを使って効率的に学習を行う最適化手法です。",
        "残差接続は、深いネットワークの学習を可能にする技術です。勾配消失問題を解決します。",
        "埋め込み層は、離散的なデータを連続的なベクトル空間に変換します。",
        
        # ===== 人工知能 =====
        "人工知能は、人間の知的活動をコンピュータで実現しようとする技術です。様々な分野で応用されています。",
        "機械学習は、データから学習して改善するシステムを実現します。教師あり学習、教師なし学習、強化学習などがあります。",
        "自然言語処理は、コンピュータが人間の言語を理解する技術です。翻訳、要約、質問応答などに応用されています。",
        "生成AIは、テキストや画像を生成できる人工知能です。創造的なタスクに革命をもたらしています。",
        "強化学習は、エージェントが環境との相互作用を通じて学習する手法です。ゲームやロボット制御に応用されています。",
        "転移学習は、ある課題で学習したモデルを別の課題に適用する技術です。少ないデータでも効果的に学習できます。",
        "教師あり学習は、正解ラベル付きのデータから学習する手法です。分類や回帰問題に使われます。",
        "教師なし学習は、ラベルなしのデータからパターンを発見する手法です。クラスタリングや次元削減に使われます。",
        "半教師あり学習は、少量のラベル付きデータと大量のラベルなしデータを組み合わせて学習します。",
        "メタ学習は、学習の仕方を学習する技術です。少ないデータで新しいタスクに素早く適応できます。",
        "説明可能なAIは、AIの判断理由を人間に説明できるようにする技術です。",
        "フェデレーテッドラーニングは、データを共有せずに分散学習を行う技術です。プライバシーを保護します。",
        "マルチモーダル学習は、テキスト、画像、音声など複数の形式のデータを統合的に処理する技術です。",
        "知識グラフは、エンティティ間の関係を表現するデータ構造です。",
        
        # ===== プログラミング =====
        "プログラミングは、コンピュータに命令を与えるための言語を使った活動です。創造的で論理的な思考が求められます。",
        "Pythonは、読みやすく書きやすいプログラミング言語です。機械学習やデータ分析で広く使われています。",
        "JavaScriptは、ウェブ開発で最も広く使われているプログラミング言語です。",
        "Rustは、安全性と性能を両立した新しいシステムプログラミング言語です。",
        "アルゴリズムは、問題を解決するための手順を定義したものです。効率的なアルゴリズムは計算時間を短縮します。",
        "データ構造は、データを効率的に格納・操作するための仕組みです。配列、リスト、木、グラフなどがあります。",
        "オブジェクト指向プログラミングは、データと処理をオブジェクトとしてまとめる設計手法です。",
        "関数型プログラミングは、関数を第一級オブジェクトとして扱うパラダイムです。副作用を避けることを重視します。",
        "ソフトウェア開発では、要件定義、設計、実装、テスト、保守というプロセスがあります。",
        "バージョン管理システムは、ソースコードの変更履歴を管理するツールです。Gitが広く使われています。",
        "デバッグは、プログラムのバグを見つけて修正する作業です。ログ出力やブレークポイントが役立ちます。",
        "テスト駆動開発は、テストを先に書いてから実装を行う開発手法です。",
        "アジャイル開発は、変化に柔軟に対応できる反復的な開発手法です。",
        "DevOpsは、開発と運用を統合した手法で、継続的なデリバリーを実現します。",
        
        # ===== 科学技術 =====
        "技術の進歩は私たちの生活を大きく変えています。スマートフォンやインターネットは日常に不可欠になりました。",
        "未来のコンピュータは量子原理を活用するでしょう。現在の限界を超えた計算能力が期待されています。",
        "人工知能は様々な分野で革新をもたらしています。医療、金融、製造業など幅広い応用があります。",
        "科学技術の発展は人類の可能性を広げます。宇宙探査や環境問題の解決にも貢献しています。",
        "インターネットは情報革命をもたらしました。世界中の人々がつながり、知識を共有できるようになりました。",
        "クラウドコンピューティングは、インターネット経由でコンピューティングリソースを提供する技術です。",
        "ビッグデータ分析は、大量のデータから価値ある洞察を抽出する技術です。",
        "IoTは、様々なモノがインターネットに接続される技術です。スマートホームや工場の自動化に応用されています。",
        "ブロックチェーンは、分散型台帳技術です。暗号通貨やスマートコントラクトの基盤となっています。",
        "5G通信は、高速・大容量・低遅延の通信を実現します。自動運転や遠隔医療を可能にします。",
        "エッジコンピューティングは、データ処理をネットワークの端で行う技術です。",
        "サイバーセキュリティは、デジタル資産を保護するための技術と実践です。",
        
        # ===== 対話・会話 =====
        "こんにちは、今日も良い天気ですね。お元気ですか？",
        "人工知能について教えてください。どのような仕組みで動いているのですか？",
        "量子コンピュータとは何ですか？普通のコンピュータとどう違うのですか？",
        "機械学習はどのように動作しますか？具体的な例を教えてください。",
        "未来の技術について考えてみましょう。どのような世界が待っているでしょうか。",
        "プログラミングは創造的な活動です。自分のアイデアを形にする喜びがあります。",
        "科学は私たちの世界を理解する手段です。好奇心から始まる探求の旅です。",
        "技術革新は社会を変革します。新しい可能性と課題の両方をもたらします。",
        "質問があればお気軽にどうぞ。できる限りお答えします。",
        "ありがとうございます。また何かあれば聞いてください。",
        "それは興味深い質問ですね。一緒に考えてみましょう。",
        "なるほど、そのような見方もあるのですね。",
        "素晴らしい観点です。もう少し詳しく教えていただけますか？",
        "その通りです。さらに詳しく説明しましょう。",
        "良い質問ですね。順番に説明していきます。",
        "お疲れ様です。今日も一日頑張りましょう。",
        "それは素晴らしいアイデアですね。実現に向けて頑張りましょう。",
        
        # ===== 数学・物理学 =====
        "数学は科学の言語です。自然法則を記述するために使われます。",
        "物理学は自然界の法則を研究する学問です。力学、電磁気学、量子力学などがあります。",
        "微分積分学は、変化と累積を扱う数学の分野です。物理学や工学で広く使われています。",
        "線形代数は、ベクトルや行列を扱う数学です。機械学習の基礎となっています。",
        "確率論は、不確実な現象を数学的に扱う分野です。統計学や機械学習と密接に関連しています。",
        "統計学は、データの収集・分析・解釈を行う学問です。科学的研究やビジネスで重要です。",
        "相対性理論は、時間と空間の概念を革新したアインシュタインの理論です。",
        "量子力学は、原子や分子のスケールで成り立つ物理法則です。",
        "熱力学は、エネルギーと熱の関係を研究する分野です。",
        "電磁気学は、電気と磁気の相互作用を研究する分野です。",
        "幾何学は、図形や空間の性質を研究する数学の分野です。",
        "数論は、整数の性質を研究する純粋数学の分野です。",
        
        # ===== 生物学・医学 =====
        "生物学は生命現象を研究する学問です。細胞、遺伝子、進化などを扱います。",
        "医学は病気の予防、診断、治療を研究する学問です。人々の健康を守ります。",
        "遺伝子工学は、DNAを操作して望ましい形質を得る技術です。",
        "免疫学は、体の防御システムを研究する分野です。ワクチン開発に貢献しています。",
        "神経科学は、脳と神経系を研究する分野です。意識や記憶の仕組みを解明しようとしています。",
        "バイオテクノロジーは、生物の機能を利用した技術です。医薬品や食品の生産に応用されています。",
        "ゲノム解析は、生物の全遺伝情報を解読する技術です。",
        "再生医療は、損傷した組織や臓器を再生する医療技術です。",
        
        # ===== 哲学・思考 =====
        "哲学は、存在、知識、倫理などの根本的な問いを探求する学問です。",
        "論理学は、正しい推論の規則を研究する学問です。",
        "倫理学は、善悪や正義について考える哲学の一分野です。",
        "認識論は、知識とは何か、どのように獲得されるかを研究します。",
        "批判的思考は、情報を客観的に分析し評価する能力です。",
        "創造的思考は、新しいアイデアや解決策を生み出す能力です。",
        "形而上学は、存在の本質について探求する哲学の分野です。",
        "美学は、美と芸術について研究する哲学の分野です。",
        
        # ===== 経済・ビジネス =====
        "経済学は、資源の配分と意思決定を研究する学問です。",
        "マーケティングは、顧客のニーズを満たす製品やサービスを提供するための活動です。",
        "ファイナンスは、お金の管理と投資に関する分野です。",
        "起業家精神は、新しいビジネスを創造し発展させる姿勢です。",
        "サプライチェーン管理は、製品の流れを効率的に管理する技術です。",
        "人事管理は、組織の人材を効果的に活用するための活動です。",
        
        # ===== 環境・エネルギー =====
        "再生可能エネルギーは、太陽光、風力、水力などの持続可能なエネルギー源です。",
        "気候変動は、人類が直面する最大の環境課題の一つです。",
        "持続可能な開発は、将来の世代のニーズを損なわない発展を目指します。",
        "電気自動車は、環境に優しい次世代の交通手段です。",
        "カーボンニュートラルは、二酸化炭素の排出と吸収を均衡させることです。",
        
        # ===== 宇宙・天文学 =====
        "天文学は、宇宙の構造と進化を研究する学問です。",
        "宇宙探査は、人類の知識の境界を広げる冒険です。",
        "ブラックホールは、光さえも逃げられない超重力天体です。",
        "銀河は、数十億から数千億の恒星が集まった巨大な天体系です。",
        "宇宙開発は、人類の未来を切り開く挑戦です。",
    ]
    
    # データ増幅（多様な組み合わせを生成）
    texts = base_texts * 50
    
    # 追加のバリエーションを生成
    prefixes = ["実際に、", "興味深いことに、", "重要なのは、", "特に、", "さらに、", "つまり、", "例えば、", "具体的には、", 
                "一般的に、", "結論として、", "要するに、", "言い換えると、", "なぜなら、", "したがって、", "もちろん、"]
    suffixes = ["これは革新的です。", "今後の発展が期待されます。", "多くの可能性があります。", "研究が進んでいます。",
                "注目されています。", "期待が高まっています。", "進化を続けています。", "重要な役割を果たしています。"]
    
    for text in base_texts[:80]:
        for prefix in prefixes:
            texts.append(prefix + text)
        for suffix in suffixes:
            texts.append(text + suffix)
    
    # Hugging Faceデータを追加（オプション）
    if use_huggingface:
        hf_texts = fetch_huggingface_data(max_samples=3000)
        texts.extend(hf_texts)
    
    return texts


# ========================================
# チャットモード
# ========================================

def chat_mode(ai: NeuroQuantumBrainAI):
    """対話モード"""
    print("\n" + "=" * 60)
    print("💬 ニューロQ Brain チャットモード")
    print("=" * 60)
    print("コマンド:")
    print("  /quit, /exit      - 終了")
    print("  /temp <min> <max> - 温度範囲 (例: /temp 0.3 0.8)")
    print("  /len <値>         - 生成長さ (10-100)")
    print("  /stats            - 量子統計")
    print("-" * 60)
    
    temp_min = 0.4
    temp_max = 0.9
    max_length = 40
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['/quit', '/exit', '/q']:
                print("\n👋 さようなら！")
                break
            
            if user_input.lower().startswith('/temp'):
                try:
                    parts = user_input.split()
                    if len(parts) >= 3:
                        temp_min = max(0.1, min(1.0, float(parts[1])))
                        temp_max = max(temp_min, min(1.5, float(parts[2])))
                    else:
                        val = float(parts[1])
                        temp_min = max(0.1, val - 0.2)
                        temp_max = min(1.5, val + 0.2)
                    print(f"   温度範囲を {temp_min:.2f}〜{temp_max:.2f} に設定")
                except:
                    print("   使用法: /temp <min> <max> または /temp <値>")
                continue
            
            if user_input.lower().startswith('/len'):
                try:
                    val = int(user_input.split()[1])
                    max_length = max(10, min(100, val))
                    print(f"   生成長さを {max_length} に設定")
                except:
                    print("   使用法: /len <10-100>")
                continue
            
            if user_input.lower() == '/stats':
                print(ai.get_report())
                continue
            
            # 生成（温度範囲制約）
            response = ai.generate(user_input, max_length=max_length, 
                                   temperature_min=temp_min, temperature_max=temp_max)
            print(f"\n🧠 ニューロQ: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 さようなら！")
            break
        except Exception as e:
            print(f"   エラー: {e}")


# ========================================
# メイン
# ========================================

def main(num_neurons: int = 100):
    """
    メイン関数
    
    Args:
        num_neurons: ニューロン数（デフォルト: 100）
    """
    print("\n🔧 ニューロQ Brain を構築中...")
    print(f"   ニューロン数: {num_neurons}")
    
    # 学習データ取得
    texts = get_training_data()
    print(f"\n📚 学習データ: {len(texts)} サンプル")
    
    # AI構築
    ai = NeuroQuantumBrainAI(
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        num_neurons=num_neurons,  # ニューロン数を指定
        max_vocab=2000
    )
    
    # 学習
    ai.train(texts, epochs=25, batch_size=16, lr=0.002, seq_length=48)
    
    # 量子統計
    print(ai.get_report())
    
    # テキスト生成テスト
    print("\n📝 テキスト生成テスト:")
    print("-" * 50)
    
    prompts = [
        "量子コンピュータは",
        "人工知能とは",
        "未来の技術",
        "こんにちは",
    ]
    
    for prompt in prompts:
        generated = ai.generate(prompt, max_length=40, temperature_min=0.4, temperature_max=0.9)
        print(f"   '{prompt}' → {generated}\n")
    
    # チャットモード
    print("\n" + "=" * 60)
    response = input("チャットモードを開始しますか？ (y/n): ").strip().lower()
    if response == 'y':
        chat_mode(ai)
    
    print("\n✅ ニューロQ Brain 完成！")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ニューロQ Brain - 脳型散在QBNNによる生成AI')
    parser.add_argument('--neurons', type=int, default=100, help='ニューロン数 (デフォルト: 100)')
    parser.add_argument('--chat', action='store_true', help='チャットモードで起動')
    args = parser.parse_args()
    
    main(num_neurons=args.neurons)

