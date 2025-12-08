#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗  ██████╗                        ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔═══██╗                       ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║   ██║                       ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║▄▄ ██║                       ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝                       ║    
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝                        ║
║                                                                               ║                                                                 ║
║                                                                               ║
║   neuroQ: Quantum-Bit Neural Network Language Model                           ║
║   独自の量子もつれニューラルネットワークによる生成AI                                  ║
║                                                                               ║
║   参照元: qbnn_layered.py                                                      ║
║   - APQB: 調整可能擬似量子ビット                                                  ║
║   - EntanglementOperator: 層間エンタングル演算子                                  ║
║   - EQBNNLayer: 層状QBNN層                                                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from collections import Counter
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# SentencePiece（トークナイザー用）
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    warnings.warn("sentencepieceライブラリがインストールされていません。pip install sentencepiece を実行してください。")

# ========================================
# qbnn_layered.py からコアコンポーネントをインポート
# ========================================
try:
    from qbnn_layered import (
        APQB as APQB_Core,                  # APQB理論のコア
        EntanglementOperator,               # 層間エンタングル演算子
        QuantumCorrelationMatrix,           # 量子相関行列
        EQBNNLayer,                         # E-QBNN層
    )
    QBNN_LAYERED_AVAILABLE = True
    print("✅ qbnn_layered.py からコアコンポーネントをインポートしました")
except ImportError:
    QBNN_LAYERED_AVAILABLE = False
    # オプションなので警告を表示しない（内蔵コンポーネントで動作します）

# ========================================
# quantum_computer.py から量子回路シミュレーターをインポート
# ========================================
try:
    # 親ディレクトリから quantum_computer.py をインポート
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from quantum_computer import (
        QuantumComputer,
        QuantumCircuit,
        Gates,
        QubitState,
    )
    QUANTUM_COMPUTER_AVAILABLE = True
    print("✅ quantum_computer.py から量子回路シミュレーターをインポートしました")
except ImportError:
    QUANTUM_COMPUTER_AVAILABLE = False
    warnings.warn("quantum_computer.py が見つかりません。量子回路シミュレーション機能は無効です。")

# ========================================
# 設定
# ========================================

class NeuroQuantumConfig:
    """ニューロQ設定"""
    def __init__(
        self,
        vocab_size: int = 8000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        lambda_entangle: float = 0.5,  # QBNNもつれ強度
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.lambda_entangle = lambda_entangle


# ========================================
# Part 1: QBNN Layer（独自の量子もつれ層）
# ========================================

class QBNNLayer(nn.Module):
    """
    Quantum-Bit Neural Network Layer
    
    qbnn_layered.py の EQBNNLayer を基盤として使用可能
    
    独自の数式モデル:
    1. s^(l) = tanh(h^(l)) ∈ [-1, 1]  (正規化 → Bloch球のz座標)
    2. h̃^(l+1) = W^(l) h^(l) + b^(l)  (線形変換)
    3. Δ^(l+1)_j = Σ_i J^(l)_{ij} s^(l)_i s^(l+1)_{raw,j}  (もつれ補正)
    4. ĥ^(l+1) = h̃^(l+1) + λ_eff Δ^(l+1)  (有効入力)
    5. h^(l+1) = activation(ĥ^(l+1))  (活性化)
    
    APQB理論に基づく改良:
    - λを範囲で制御し、θ（シータ）が動的に変化できるようにする
    - r = cos(2θ), T = |sin(2θ)|, r² + T² = 1
    
    参照: qbnn_layered.py の APQB クラス
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 lambda_min: float = 0.2, lambda_max: float = 0.5,
                 use_qbnn_layered: bool = True):  # qbnn_layered.pyを参照
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_qbnn_layered = use_qbnn_layered and QBNN_LAYERED_AVAILABLE
        
        # λの範囲（θが動けるように）
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        
        if self.use_qbnn_layered:
            # qbnn_layered.py の EQBNNLayer を内部で使用
            self.eqbnn_core = EQBNNLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                prev_output_dim=input_dim,
                entangle_strength=(lambda_min + lambda_max) / 2
            )
            # コア層のパラメータを参照
            self.W = self.eqbnn_core.linear
            # W_entangle.weightの形状は(output_dim, input_dim)なので、参照として保持
            # forward内で転置して使用する
            self.J_source = self.eqbnn_core.entangle_op.W_entangle.weight
        else:
            # W: 通常の重み行列
            self.W = nn.Linear(input_dim, output_dim)
            
            # J: もつれテンソル（独自）
            self.J = nn.Parameter(torch.randn(input_dim, output_dim) * 0.02)
        
        # λベース値（学習可能、0-1に正規化してから範囲にマッピング）
        self.lambda_base = nn.Parameter(torch.tensor(0.5))
        
        # 層正規化
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # 呼び出しカウンタ（動的変化用）
        self.register_buffer('call_count', torch.tensor(0))
    
    def forward(self, h_prev: torch.Tensor) -> torch.Tensor:
        # 1. 正規化（Bloch球のz座標として解釈）
        s_prev = torch.tanh(h_prev)  # (..., input_dim)
        
        # 2. 線形変換
        h_tilde = self.W(h_prev)  # (..., output_dim)
        
        # 3. 次層の候補を正規化
        s_raw = torch.tanh(h_tilde)  # (..., output_dim)
        
        # 4. もつれ補正項 Δ
        # Δ_j = Σ_i J_{ij} s^(l)_i * s^(l+1)_{raw,j}
        # J: (input_dim, output_dim)
        # s_prev: (..., input_dim)
        # s_raw: (..., output_dim)
        # delta: (..., output_dim)
        # 各jについて: delta_j = Σ_i (J_{ij} * s_prev_i) * s_raw_j
        
        # Jの形状を確認して転置（use_qbnn_layeredの場合、J_sourceは(output_dim, input_dim)）
        if self.use_qbnn_layered:
            J = self.J_source.t()  # (input_dim, output_dim)に転置
        else:
            J = self.J  # 既に(input_dim, output_dim)
        
        # まず J_{ij} * s_prev_i を計算: (..., output_dim)
        J_s_prev = torch.einsum('...i,ij->...j', s_prev, J)  # (..., output_dim)
        # 次に s_raw_j を掛ける
        delta = J_s_prev * s_raw  # (..., output_dim)
        
        # 5. 動的λ: θが動けるように範囲内で変化（量子ゆらぎ）
        # λ_baseをsigmoidで0-1に制限し、範囲にマッピング
        lambda_normalized = torch.sigmoid(self.lambda_base)
        
        # 推論時のみ動的変化を追加（学習時は安定性のため固定寄り）
        if not self.training:
            # sin波で動的変化（θが動けるように）
            phase = float(self.call_count) * 0.2
            dynamic_factor = 0.5 + 0.5 * math.sin(phase)
            self.call_count += 1
        else:
            dynamic_factor = 0.5
        
        # 有効なλを計算
        lambda_range = self.lambda_max - self.lambda_min
        lambda_eff = self.lambda_min + lambda_range * (lambda_normalized * 0.7 + dynamic_factor * 0.3)
        
        # 6. 有効入力
        h_hat = h_tilde + lambda_eff * delta
        
        # 7. 層正規化 + GELU活性化
        output = self.layer_norm(h_hat)
        output = F.gelu(output)
        
        return output
    
    def get_quantum_info(self) -> Dict:
        """量子情報を取得"""
        with torch.no_grad():
            lambda_normalized = torch.sigmoid(self.lambda_base).item()
            lambda_eff = self.lambda_min + (self.lambda_max - self.lambda_min) * lambda_normalized
            
            # Jの形状を確認（use_qbnn_layeredの場合、転置が必要）
            if self.use_qbnn_layered:
                J = self.J_source.t()  # (input_dim, output_dim)に転置
            else:
                J = self.J  # 既に(input_dim, output_dim)
            
            info = {
                'lambda_min': self.lambda_min,
                'lambda_max': self.lambda_max,
                'lambda_eff': lambda_eff,
                'J_mean': J.mean().item(),
                'J_std': J.std().item(),
                'J_max': J.max().item(),
                'source': 'qbnn_layered.py' if self.use_qbnn_layered else 'builtin',
            }
            
            # qbnn_layered.py使用時は追加情報を取得
            if self.use_qbnn_layered:
                info['entangle_strength'] = self.eqbnn_core.entangle_op.entangle_strength.item()
            
            return info


# ========================================
# Part 2: QBNN-Attention（量子もつれアテンション）
# ========================================

class QBNNAttention(nn.Module):
    """
    QBNN拡張Self-Attention
    
    通常のAttentionに量子もつれ補正を追加
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, 
                 lambda_val: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dimはnum_headsで割り切れる必要があります"
        
        # Q, K, V プロジェクション
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # QBNN量子もつれ補正
        self.J_attn = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)
        self.lambda_attn = nn.Parameter(torch.tensor(float(lambda_val)))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        GPT標準: Multi-Head Causal Self-Attention（QBNN拡張版）
        
        Args:
            x: (batch, seq, embed_dim)
            mask: Optional attention mask (Noneの場合はCausal Maskを自動生成)
        
        Returns:
            (batch, seq, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # GPT標準: Q, K, V 計算
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # GPT標準: アテンションスコア計算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq, seq)
        
        # QBNN拡張: 量子もつれ補正
        Q_norm = torch.tanh(Q)
        K_norm = torch.tanh(K)
        # もつれ補正項（ヘッドごと）
        delta = torch.einsum('bhid,hde,bhje->bhij', Q_norm, self.J_attn, K_norm)
        attn_scores = attn_scores + self.lambda_attn * delta
        
        # GPT標準: Causal Mask適用
        if mask is not None:
            # 提供されたマスクを使用
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        else:
            # Causal Mask自動生成（GPT標準）
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # GPT標準: Softmax + Dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # GPT標準: アテンション適用
        output = torch.matmul(attn_probs, V)  # (batch, heads, seq, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # GPT標準: 出力射影
        return self.out_proj(output)


# ========================================
# Part 3: QBNN-Transformer Block
# ========================================

class QBNNTransformerBlock(nn.Module):
    """
    GPTデコーダーブロック（QBNN拡張版）
    
    GPT標準構造:
    1. Pre-norm LayerNorm
    2. Multi-Head Causal Self-Attention (QBNN拡張)
    3. Residual Connection
    4. Pre-norm LayerNorm
    5. Feed-Forward Network (標準FFN + QBNN拡張)
    6. Residual Connection
    """
    
    def __init__(self, config: NeuroQuantumConfig):
        super().__init__()
        
        # Pre-norm LayerNorm
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        # QBNN-Attention
        self.attention = QBNNAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            lambda_val=config.lambda_entangle
        )
        
        # GPT標準FFN: Linear → GELU → Linear
        self.ffn_standard = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )
        
        # QBNN拡張FFN（個別レイヤーにアクセス可能にするため、Sequentialではなく個別に定義）
        self.ffn_qbnn_layer1 = QBNNLayer(
            config.embed_dim, config.hidden_dim, 
            lambda_min=config.lambda_entangle * 0.5,
            lambda_max=config.lambda_entangle * 1.5
        )
        self.ffn_qbnn_dropout = nn.Dropout(config.dropout)
        self.ffn_qbnn_layer2 = QBNNLayer(
            config.hidden_dim, config.embed_dim,
            lambda_min=config.lambda_entangle * 0.5,
            lambda_max=config.lambda_entangle * 1.5
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
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
        
        # 標準FFN + QBNN拡張（ブレンド）
        ffn_standard_out = self.ffn_standard(x)
        # QBNN拡張FFN
        ffn_qbnn_out = self.ffn_qbnn_layer1(x)
        ffn_qbnn_out = self.ffn_qbnn_dropout(ffn_qbnn_out)
        ffn_qbnn_out = self.ffn_qbnn_layer2(ffn_qbnn_out)
        
        # ブレンド比率: 標準FFN 70% + QBNN拡張 30%
        ffn_out = 0.7 * ffn_standard_out + 0.3 * ffn_qbnn_out
        
        x = residual + ffn_out
        
        return x


# ========================================
# Part 4: Embedding（埋め込み層）
# ========================================

class NeuroQuantumEmbedding(nn.Module):
    """
    ニューロQ 埋め込み層
    
    Token → テキストエンベディング + 位置エンコーディング
    """
    
    def __init__(self, config: NeuroQuantumConfig):
        super().__init__()
        
        # テキストエンベディング
        self.text_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # 位置埋め込み（学習可能）
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        # ドロップアウト
        self.dropout = nn.Dropout(config.dropout)
        
        # 埋め込み次元
        self.embed_dim = config.embed_dim
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        
        # テキストエンベディング
        text_embeds = self.text_embedding(token_ids)
        
        # 位置埋め込み
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # 合成
        embeds = text_embeds + pos_embeds
        embeds = self.dropout(embeds)
        
        return embeds


# ========================================
# Part 5: Output Head（出力層）
# ========================================

class NeuroQuantumHead(nn.Module):
    """
    ニューロQ 出力ヘッド
    
    ベクトル → 語彙確率への変換
    """
    
    def __init__(self, config: NeuroQuantumConfig):
        super().__init__()
        
        # 最終正規化
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # 語彙への線形変換
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


# ========================================
# Part 6: ニューロQ モデル本体
# ========================================

class NeuroQuantum(nn.Module):
    """
    ニューロQ: GPT型デコーダーのみのTransformer（QBNN拡張版）
    
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
    
    独自要素:
    - QBNNLayer: 量子もつれテンソル J による補正
    - QBNN-Attention: アテンションスコアへの量子補正
    - 学習可能な λ（もつれ強度）
    """
    
    def __init__(self, config: NeuroQuantumConfig):
        super().__init__()
        self.config = config
        
        # GPT標準: Text Embedding + Position Embedding
        self.text_embedding = nn.Embedding(config.vocab_size, config.embed_dim)  # テキストエンベディング
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        # GPT Decoder Blocks
        self.transformer_blocks = nn.ModuleList([
            QBNNTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # GPT標準: Final LayerNorm
        self.final_norm = nn.LayerNorm(config.embed_dim)
        
        # GPT標準: Output Head (weight tying可能だが、ここでは独立)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # GPT標準: Embedding Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # パラメータ初期化（GPT標準）
        self.apply(self._init_weights)
        
        # モデル情報
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        """GPT標準の重み初期化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        GPT型デコーダーのみのTransformer フォワード（図2-4のフローに準拠）
        
        処理ステップ:
        1. トークンID → テキストエンベディング（Text Embedding + Position Embedding）
        2. テキストエンベディング → GPT型デコーダーのみのTransformer
        3. Transformer出力 → 後処理ステップ（Final LayerNorm + Output Head）
        4. 後処理ステップ → ロジット（出力テキスト生成用）
        
        Args:
            token_ids: (batch, seq) トークンID（トークン化済みのテキスト）
            mask: Optional attention mask (Noneの場合はCausal Maskを自動生成)
        
        Returns:
            (batch, seq, vocab_size) ロジット（後処理ステップ後の出力）
        """
        batch, seq = token_ids.shape
        
        # ステップ1: トークンID → テキストエンベディング
        # Text Embedding: トークンIDをベクトルに変換（テキストエンベディング）
        text_embeds = self.text_embedding(token_ids)  # (batch, seq, embed_dim)
        # Position Embedding: 位置情報を追加
        positions = torch.arange(seq, device=token_ids.device).unsqueeze(0).expand(batch, -1)
        pos_embeds = self.position_embedding(positions)  # (batch, seq, embed_dim)
        # 埋め込みの合成 + Dropout
        hidden_states = self.dropout(text_embeds + pos_embeds)
        
        # Causal Mask生成（maskがNoneの場合）
        if mask is None:
            mask = torch.tril(torch.ones(seq, seq, device=token_ids.device)).unsqueeze(0).unsqueeze(0)
        
        # ステップ2: テキストエンベディング → GPT型デコーダーのみのTransformer
        # N個のGPT Decoder Blocks（Pre-norm + Multi-Head Causal Self-Attention + FFN）
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, mask)
        
        # ステップ3: Transformer出力 → 後処理ステップ
        # Final LayerNorm
        hidden_states = self.final_norm(hidden_states)
        # Output Head: ベクトル → 語彙確率への変換
        logits = self.output_head(hidden_states)  # (batch, seq, vocab_size)
        
        # ステップ4: ロジット（出力テキスト生成用）
        return logits
    
    def get_quantum_info(self) -> List[Dict]:
        """全層の量子情報を取得"""
        info = []
        for i, block in enumerate(self.transformer_blocks):
            block_info = {
                'block': i,
                'attn_lambda': block.attention.lambda_attn.item(),
            }
            info.append(block_info)
        return info


# ========================================
# Part 7: トークナイザー
# ========================================

class NeuroQuantumTokenizer:
    """
    ニューロQ トークナイザー（図2-4のトークン化ステップに準拠）

    SentencePieceトークナイザーを使用:
    - 語彙サイズを指定して学習可能（8000-32000推奨）
    - BPEアルゴリズムを使用（日本語に適している）
    - モデルの保存・読み込みが可能
    - フォールバック: 簡易トークナイザー（SentencePiece未インストール時）

    処理フロー:
    1. 入力テキスト → トークン化（テキストを個々のトークンに分割）
    2. トークン → トークンID（各トークンを数値IDに変換）
    """

    def __init__(self, vocab_size: int = 8000, model_file: str = None):
        """
        Args:
            vocab_size: 語彙サイズ（デフォルト: 8000）
            model_file: 既存のSentencePieceモデルファイルパス（Noneの場合は新規学習）
        """
        self.vocab_size = vocab_size
        self.actual_vocab_size = None
        self.model_file = model_file
        self.sp_model = None

        # 特殊トークン
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'

        # SentencePieceを使用
        if SENTENCEPIECE_AVAILABLE:
            if model_file and os.path.exists(model_file):
                # 既存モデルを読み込み
                try:
                    self.sp_model = spm.SentencePieceProcessor()
                    self.sp_model.load(model_file)
                    self.actual_vocab_size = self.sp_model.get_piece_size()
                    self.vocab_size = self.actual_vocab_size
                    # 特殊トークンIDを取得
                    self.pad_id = self.sp_model.pad_id()
                    self.unk_id = self.sp_model.unk_id()
                    self.bos_id = self.sp_model.bos_id()
                    self.eos_id = self.sp_model.eos_id()
                    print(f"   ✅ SentencePieceモデル読み込み: {model_file} (語彙サイズ: {self.actual_vocab_size})")
                except Exception as e:
                    warnings.warn(f"SentencePieceモデルの読み込みに失敗: {e}。新規学習します。")
                    self.sp_model = None
        else:
            # SentencePiece未インストール
            self.sp_model = None

        # SentencePieceが使えない場合はフォールバック
        if self.sp_model is None:
            self._init_fallback()

    def _init_fallback(self):
        """フォールバック用の簡易トークナイザー初期化"""
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx_to_char = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        # vocab_sizeは設定されている値を保持（上書きしない）
        if not hasattr(self, 'vocab_size') or self.vocab_size is None:
            self.vocab_size = 4
        # actual_vocab_sizeを特殊トークン数で初期化（build_vocab()で更新される）
        self.actual_vocab_size = len(self.char_to_idx)

        # 特殊トークンID
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def build_vocab(self, texts: List[str], character_coverage: float = 0.9995, model_prefix: str = "spm_model_layered", min_freq: int = 1):
        """
        SentencePieceで語彙を学習

        Args:
            texts: 学習テキストのリスト
            character_coverage: 文字カバレッジ（0.9995が推奨、日本語の場合は0.9995-0.99995）
            model_prefix: モデルファイルのプレフィックス
            min_freq: 最小頻度（フォールバック用）
        """
        if not SENTENCEPIECE_AVAILABLE:
            warnings.warn("SentencePieceが利用できません。フォールバックトークナイザーを使用します。")
            self._build_vocab_fallback(texts, min_freq)
            return self

        # vocab_sizeが4以下（特殊トークンのみ）の場合は、デフォルト値を使用
        actual_vocab_size = max(self.vocab_size, 8000) if self.vocab_size <= 4 else self.vocab_size
        print(f"   🔤 SentencePieceで語彙学習中... (目標語彙サイズ: {actual_vocab_size})")

        # 一時ファイルにテキストを保存
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            temp_file = f.name
            for text in texts:
                f.write(text + '\n')

        try:
            # SentencePiece学習
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=model_prefix,
                vocab_size=actual_vocab_size,
                character_coverage=character_coverage,
                model_type='bpe',  # BPEアルゴリズム
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece=self.pad_token,
                unk_piece=self.unk_token,
                bos_piece=self.bos_token,
                eos_piece=self.eos_token,
            )

            # モデルを読み込み
            model_file_path = model_prefix + '.model'
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(model_file_path)
            self.actual_vocab_size = self.sp_model.get_piece_size()
            self.vocab_size = self.actual_vocab_size
            self.model_file = model_file_path
            # vocab_sizeを更新
            if hasattr(self, 'vocab_size') and self.vocab_size <= 4:
                self.vocab_size = self.actual_vocab_size

            # 特殊トークンIDを取得
            self.pad_id = self.sp_model.pad_id()
            self.unk_id = self.sp_model.unk_id()
            self.bos_id = self.sp_model.bos_id()
            self.eos_id = self.sp_model.eos_id()

            print(f"   ✅ SentencePiece語彙学習完了 (語彙サイズ: {self.actual_vocab_size})")

        except Exception as e:
            warnings.warn(f"SentencePiece学習に失敗: {e}。フォールバックを使用します。")
            self._build_vocab_fallback(texts, min_freq)
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            # モデルファイル以外の一時ファイルを削除
            for ext in ['.vocab']:
                temp_file_ext = model_prefix + ext
                if os.path.exists(temp_file_ext):
                    os.unlink(temp_file_ext)

        return self

    def _build_vocab_fallback(self, texts: List[str], min_freq: int = 1):
        """フォールバック：簡易語彙構築"""
        print(f"   🔤 フォールバック語彙構築中...")
        char_counts = Counter()
        for text in texts:
            char_counts.update(list(text))

        # vocab_sizeが4以下（特殊トークンのみ）の場合は、デフォルト値を使用
        target_vocab_size = max(self.vocab_size, 8000) if self.vocab_size <= 4 else self.vocab_size

        for char, count in char_counts.most_common(target_vocab_size - 4):
            if char not in self.char_to_idx and count >= min_freq:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char

        self.vocab_size = len(self.char_to_idx)
        self.actual_vocab_size = self.vocab_size
        print(f"   ✅ 語彙サイズ: {self.vocab_size}")

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """
        エンコード（図2-4のトークン化ステップ）

        処理:
        1. 入力テキスト → トークン化（テキストを個々のトークンに分割）
        2. トークン → トークンID（各トークンを数値IDに変換）

        Args:
            text: 入力テキスト（例: "This is an example."）
            add_special: 特殊トークン（BOS/EOS）を追加するか

        Returns:
            トークンIDのリスト（例: [40134, 2052, 133, 389, 12]）
        """
        if self.sp_model is not None:
            # SentencePiece使用
            if add_special:
                return self.sp_model.encode(text, out_type=int, add_bos=True, add_eos=True)
            else:
                return self.sp_model.encode(text, out_type=int, add_bos=False, add_eos=False)
        else:
            # フォールバック：最長マッチ方式
            tokens = []
            if add_special:
                tokens.append(self.bos_id)

            i = 0
            text_len = len(text)
            while i < text_len:
                matched = False
                for length in range(min(8, text_len - i), 0, -1):
                    substr = text[i:i+length]
                    if substr in self.char_to_idx:
                        tokens.append(self.char_to_idx[substr])
                        i += length
                        matched = True
                        break

                if not matched:
                    tokens.append(self.char_to_idx.get(text[i], self.unk_id))
                    i += 1

            if add_special:
                tokens.append(self.eos_id)

            return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """デコード"""
        if self.sp_model is not None:
            # SentencePiece使用
            return self.sp_model.decode(token_ids)
        else:
            # フォールバック
            chars = []
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}

            for t in token_ids:
                if skip_special and t in special_ids:
                    continue
                char = self.idx_to_char.get(t, self.unk_token)
                if char not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                    chars.append(char)

            return ''.join(chars)

    def save(self, path: str):
        """保存"""
        if self.sp_model is not None:
            # SentencePieceモデルは既に保存されている
            print(f"   ✅ SentencePieceモデル保存済み: {self.model_file}")
        else:
            # フォールバック用の保存
            data = {
                'char_to_idx': self.char_to_idx,
                'vocab_size': self.vocab_size,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """読み込み"""
        if path.endswith('.model'):
            # SentencePieceモデルを読み込み
            if SENTENCEPIECE_AVAILABLE:
                try:
                    self.sp_model = spm.SentencePieceProcessor()
                    self.sp_model.load(path)
                    self.actual_vocab_size = self.sp_model.get_piece_size()
                    self.vocab_size = self.actual_vocab_size
                    self.model_file = path
                    # 特殊トークンIDを取得
                    self.pad_id = self.sp_model.pad_id()
                    self.unk_id = self.sp_model.unk_id()
                    self.bos_id = self.sp_model.bos_id()
                    self.eos_id = self.sp_model.eos_id()
                    print(f"   ✅ SentencePieceモデル読み込み: {path} (語彙サイズ: {self.actual_vocab_size})")
                except Exception as e:
                    warnings.warn(f"SentencePieceモデルの読み込みに失敗: {e}")
            else:
                warnings.warn("SentencePieceが利用できません。")
        else:
            # フォールバック用のJSONファイルを読み込み
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.char_to_idx = data['char_to_idx']
            self.idx_to_char = {int(i): c for c, i in self.char_to_idx.items()}
            self.vocab_size = data['vocab_size']
        return self


# ========================================
# Part 8: ニューロQ AI（生成AI本体）
# ========================================

class NeuroQuantumAI:
    """
    ニューロQ AI
    
    QBNN-LLM による生成AI
    
    ニューロン数（hidden_dim）を指定可能
    """
    
    def __init__(
        self,
        embed_dim: int = 48,
        hidden_dim: int = 96,       # ニューロン数（FFN層の次元）
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        lambda_entangle: float = 0.5,
        use_openai_embedding: bool = False,  # OpenAI Embeddingを使用するか
        openai_api_key: Optional[str] = None,  # OpenAI APIキー
        openai_model: str = "text-embedding-3-large",  # OpenAI Embeddingモデル
    ):
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
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.lambda_entangle = lambda_entangle
        self.use_openai_embedding = use_openai_embedding
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = openai_model
        
        self.tokenizer: Optional[NeuroQuantumTokenizer] = None
        self.model: Optional[NeuroQuantum] = None
        self.config: Optional[NeuroQuantumConfig] = None

        # 量子回路シミュレーター
        self.quantum_computer: Optional[QuantumComputer] = None
        self.use_quantum_simulation = QUANTUM_COMPUTER_AVAILABLE
        if self.use_quantum_simulation:
            self.quantum_computer = QuantumComputer("NeuroQuantum-QC")
            print("⚛️  量子回路シミュレーターを初期化しました")

    def train(self, texts: List[str], epochs: int = 50, batch_size: int = 16,
              lr: float = 0.001, seq_len: int = 64):
        """学習"""
        print("\n" + "=" * 70)
        print("📚 ニューロQ 学習開始")
        print("=" * 70)

        # トークナイザー構築
        print("\n🔤 トークナイザー構築...")

        # 既存のSentencePieceモデルを探す
        tokenizer_model_paths = [
            "neuroq_tokenizer.model",  # カレントディレクトリ（推奨）
            "neuroq_tokenizer_8k.model",  # カレントディレクトリ（旧名称）
            "../neuroq_tokenizer.model",  # 親ディレクトリ
            "../neuroq_tokenizer_8k.model",  # 親ディレクトリ（旧名称）
            os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model"),  # スクリプトと同じディレクトリ
            os.path.join(os.path.dirname(__file__), "neuroq_tokenizer_8k.model"),  # スクリプトと同じディレクトリ（旧名称）
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "neuroq_tokenizer.model"),  # 親の親ディレクトリ
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "neuroq_tokenizer_8k.model"),  # 親の親ディレクトリ（旧名称）
        ]

        existing_model = None
        for path in tokenizer_model_paths:
            if os.path.exists(path):
                existing_model = path
                break

        if existing_model:
            # 既存のSentencePieceモデルを使用
            print(f"   既存のSentencePieceモデルを使用: {existing_model}")
            self.tokenizer = NeuroQuantumTokenizer(vocab_size=8000, model_file=existing_model)
        else:
            # 新規に語彙を構築
            print("   新規に語彙を構築します...")
            self.tokenizer = NeuroQuantumTokenizer(vocab_size=8000)
            self.tokenizer.build_vocab(texts)

        # 語彙サイズを取得（actual_vocab_sizeがNoneの場合はvocab_sizeを使用）
        effective_vocab_size = self.tokenizer.actual_vocab_size
        if effective_vocab_size is None or effective_vocab_size <= 4:
            effective_vocab_size = self.tokenizer.vocab_size
        if effective_vocab_size is None or effective_vocab_size <= 4:
            effective_vocab_size = 8000  # デフォルト値
        print(f"   語彙サイズ: {effective_vocab_size}")
        
        # モデル構築
        print("\n🧠 ニューロQモデル構築...")
        self.config = NeuroQuantumConfig(
            vocab_size=effective_vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            lambda_entangle=self.lambda_entangle,
        )
        
        self.model = NeuroQuantum(self.config).to(self.device)
        
        print(f"\n📊 モデル構成:")
        print(f"   埋め込み次元: {self.embed_dim}")
        print(f"   隠れ層次元: {self.hidden_dim}")
        print(f"   アテンションヘッド: {self.num_heads}")
        print(f"   Transformerブロック: {self.num_layers}")
        print(f"   もつれ強度 λ: {self.lambda_entangle}")
        print(f"   総パラメータ数: {self.model.num_params:,}")
        
        # データ準備
        print("\n📊 データ準備...")
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"   総トークン数: {len(all_tokens):,}")
        
        # シーケンス作成
        sequences = []
        for i in range(0, len(all_tokens) - seq_len - 1, seq_len // 2):
            x = all_tokens[i:i+seq_len]
            y = all_tokens[i+1:i+seq_len+1]
            if len(x) == seq_len and len(y) == seq_len:
                sequences.append((x, y))
        
        print(f"   シーケンス数: {len(sequences):,}")
        
        # 学習
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        print("\n🚀 学習ループ...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(sequences)
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                if len(batch) == 0:
                    continue
                
                x_batch = torch.stack([s[0] for s in batch]).to(self.device)
                y_batch = torch.stack([s[1] for s in batch]).to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_batch)
                
                loss = criterion(
                    logits.view(-1, self.config.vocab_size),
                    y_batch.view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / max(1, len(sequences) // batch_size)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")
        
        print("\n✅ 学習完了！")
        
        # 量子情報
        print("\n⚛️ 量子もつれ情報:")
        for info in self.model.get_quantum_info():
            print(f"   Block {info['block']}: λ_attn = {info['attn_lambda']:.4f}")

    def train_on_texts(self, texts: List[str], epochs: int = 50, batch_size: int = 16,
                       lr: float = 0.001, seq_len: int = 64):
        """
        train()へのエイリアス（後方互換性のため）
        
        Args:
            texts: 学習用テキストのリスト
            epochs: エポック数
            batch_size: バッチサイズ
            lr: 学習率
            seq_len: シーケンス長
        """
        return self.train(texts, epochs=epochs, batch_size=batch_size, lr=lr, seq_len=seq_len)

    def _quantum_circuit_influence(self, logits: torch.Tensor, step: int) -> torch.Tensor:
        """
        量子回路シミュレーションを使用してlogitsに影響を与える

        Args:
            logits: 次トークンのlogits
            step: 現在の生成ステップ

        Returns:
            量子的に調整されたlogits
        """
        if not self.use_quantum_simulation or self.quantum_computer is None:
            return logits

        # 3量子ビット回路を作成（ステップごとにユニークな回路を構築）
        n_qubits = 3
        circuit_name = f"generation_step_{step}"
        qc = self.quantum_computer.create_circuit(circuit_name, n_qubits)

        # プロンプト内容に基づいて量子ゲートを適用
        # ステップ数に応じて異なる量子回路パターンを生成
        if step % 4 == 0:
            # ベル状態を作成
            qc.h(0).cnot(0, 1).cnot(1, 2)
        elif step % 4 == 1:
            # GHZ状態を作成
            qc.h(0).cnot(0, 1).cnot(0, 2)
        elif step % 4 == 2:
            # W状態風の重ね合わせ
            qc.h(0).h(1).cnot(0, 2)
        else:
            # 回転ゲートを使用
            theta = step * 0.1
            qc.ry(0, theta).ry(1, theta * 1.5).cnot(0, 1)

        # 量子回路を測定（複数回実行して統計を取得）
        results = qc.run(shots=100)

        # 測定結果の確率分布を計算
        total_shots = sum(results.values())
        probs = {state: count / total_shots for state, count in results.items()}

        # 量子測定結果をlogitsの調整に使用
        # ビット列を数値に変換し、logitsに量子的な影響を与える
        quantum_influence = torch.zeros_like(logits)
        for state, prob in probs.items():
            # ビット列を整数に変換（例: "101" -> 5）
            state_value = int(state, 2)
            # 状態値を使ってlogitsの一部を調整
            # vocab_sizeに対してモジュロを取り、循環的に影響を与える
            vocab_size = logits.size(0)
            for i in range(0, vocab_size, 8):  # 8トークンごとに影響
                idx = (i + state_value) % vocab_size
                quantum_influence[idx] += prob * 0.3  # 量子的な影響の強度

        # 元のlogitsに量子的な影響を加算
        adjusted_logits = logits + quantum_influence

        return adjusted_logits

    def generate(
        self,
        prompt: str = "",
        max_length: int = 100,
        temp_min: float = 0.4,       # 温度の下限
        temp_max: float = 0.8,       # 温度の上限
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 2.0,
        temperature: float = None,   # 後方互換性のため（指定された場合temp_min/temp_maxを自動計算）
    ) -> str:
        """
        テキスト生成（対話形式）
        
        温度を範囲で指定することで、θ（シータ）が動的に変化できるようにする。
        APQB理論: r = cos(2θ), T = |sin(2θ)|, r² + T² = 1
        温度Tが固定だとθが固定され、量子的ゆらぎがなくなる。
        
        Args:
            temperature: 後方互換性のための単一温度パラメータ。
                        指定された場合、temp_min = temperature * 0.8, 
                        temp_max = temperature * 1.2 に自動変換されます。
        """
        # 後方互換性: temperature が指定された場合、temp_min/temp_max に変換
        if temperature is not None:
            temp_min = temperature * 0.8
            temp_max = temperature * 1.2
        
        if self.model is None:
            # 自動的にサンプルデータで学習を実行
            print("⚠️ モデルが未学習です。サンプルデータで自動学習を開始...")
            sample_data = [
                "人工知能は、人間の知能を模倣するコンピュータシステムです。機械学習やディープラーニングなどの技術を使用して、データからパターンを学習し、予測や判断を行います。",
                "量子コンピュータは、量子力学の原理を利用した次世代のコンピュータです。従来のコンピュータでは解けない複雑な問題を高速に解くことができます。",
                "自然言語処理は、コンピュータが人間の言語を理解し、生成するための技術です。翻訳、要約、質問応答などのタスクに使用されます。",
                "ニューラルネットワークは、人間の脳の神経細胞の働きを模倣した計算モデルです。層状に接続されたノードで構成され、データから特徴を学習します。",
                "プログラミングは、コンピュータに指示を与えるための言語を使ってソフトウェアを作成する技術です。Python、JavaScript、Javaなど多くの言語があります。",
                "データサイエンスは、大量のデータから有用な情報を抽出し、ビジネスや研究に活用する学問分野です。統計学、機械学習、可視化などの手法を組み合わせます。",
                "クラウドコンピューティングは、インターネット経由でコンピュータリソースを提供するサービスです。AWS、Azure、GCPなどのプラットフォームが代表的です。",
                "ブロックチェーンは、分散型台帳技術の一種で、データの改ざんを防ぐ仕組みを持っています。暗号通貨や契約管理などに応用されています。",
            ]
            try:
                self.train(sample_data, epochs=3)  # 軽量な学習
                print("✅ 自動学習完了")
            except Exception as e:
                print(f"⚠️ 自動学習に失敗しました: {e}")
                raise ValueError(f"モデルの自動学習に失敗しました: {e}")
        
        # 学習後もモデルがNoneの場合はエラー
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        self.model.eval()
        
        # 対話形式のプロンプトを作成
        dialogue_prompt = f"<USER>{prompt}<ASSISTANT>"
        
        # プロンプトエンコード
        tokens = self.tokenizer.encode(dialogue_prompt, add_special=True)[:-1]
        
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        generated = tokens[0].tolist()
        
        with torch.no_grad():
            for step in range(max_length):
                # 最新のmax_seq_lenトークンを使用
                input_tokens = tokens[:, -self.max_seq_len:] if tokens.size(1) > self.max_seq_len else tokens
                
                logits = self.model(input_tokens)
                next_logits = logits[0, -1, :]

                # ⚛️ 量子回路シミュレーションの影響を適用
                next_logits = self._quantum_circuit_influence(next_logits, step)

                # 繰り返しペナルティ（温度調整の前に適用）
                # より長い範囲（100トークン）を見て、より強いペナルティを適用
                recent_tokens = set(generated[-100:])
                for token_id in recent_tokens:
                    next_logits[token_id] /= repetition_penalty

                # 動的温度: θが動けるように範囲内で変化させる
                # sin波で滑らかに変動（量子的な振動をシミュレート）
                theta_phase = step * 0.3  # 位相
                temperature = temp_min + (temp_max - temp_min) * (0.5 + 0.5 * math.sin(theta_phase))

                # 温度調整
                next_logits = next_logits / temperature
                
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
                
                # サンプリング
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # EOS検出
                if next_token.item() == self.tokenizer.eos_id:
                    break
                
                # <USER>トークンが出たら終了（次の質問に入らないように）
                generated.append(next_token.item())
                decoded_so_far = self.tokenizer.decode(generated)
                if "<USER>" in decoded_so_far.split("<ASSISTANT>")[-1]:
                    break
                
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        
        # 応答部分のみを抽出
        full_text = self.tokenizer.decode(generated)
        
        # <ASSISTANT>以降のテキストを抽出
        if "<ASSISTANT>" in full_text:
            response = full_text.split("<ASSISTANT>")[-1]
            # <USER>が含まれていたら除去
            if "<USER>" in response:
                response = response.split("<USER>")[0]
            return response.strip()
        
        return full_text
    
    def chat(self):
        """対話モード"""
        print("\n" + "=" * 70)
        print("💬 ニューロQ チャットモード")
        print("=" * 70)
        print("\nコマンド:")
        print("  /quit         - 終了")
        print("  /temp <min> <max> - 温度範囲 (例: /temp 0.4 0.8)")
        print("  /len <値>     - 生成長さ (10-500)")
        print("  /info         - モデル情報")
        print("  /quantum      - 量子もつれ情報")
        print("-" * 70)
        
        temp_min = 0.4  # 温度の下限
        temp_max = 0.8  # 温度の上限
        max_length = 100
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == '/quit':
                    print("👋 さようなら！")
                    break
                
                if user_input.startswith('/temp '):
                    try:
                        parts = user_input.split()
                        if len(parts) >= 3:
                            temp_min = float(parts[1])
                            temp_max = float(parts[2])
                            temp_min = max(0.1, min(1.0, temp_min))
                            temp_max = max(0.1, min(1.0, temp_max))
                            if temp_min > temp_max:
                                temp_min, temp_max = temp_max, temp_min
                            print(f"   温度範囲を {temp_min:.2f} - {temp_max:.2f} に設定（θが動ける）")
                        else:
                            print("   使い方: /temp <最小> <最大> (例: /temp 0.4 0.8)")
                    except:
                        print("   エラー: /temp <最小> <最大>")
                    continue
                
                if user_input.startswith('/len '):
                    try:
                        max_length = int(user_input.split()[1])
                        max_length = max(10, min(500, max_length))
                        print(f"   生成長さを {max_length} に設定")
                    except:
                        print("   エラー: /len <数値>")
                    continue
                
                if user_input == '/info':
                    print(f"\n📊 ニューロQ モデル情報:")
                    vocab_size = self.config.vocab_size if self.config else (self.tokenizer.actual_vocab_size or self.tokenizer.vocab_size)
                    print(f"   語彙サイズ: {vocab_size}")
                    print(f"   埋め込み次元: {self.embed_dim}")
                    print(f"   隠れ層次元: {self.hidden_dim}")
                    print(f"   アテンションヘッド: {self.num_heads}")
                    print(f"   Transformerブロック: {self.num_layers}")
                    print(f"   総パラメータ数: {self.model.num_params:,}")
                    continue
                
                if user_input == '/quantum':
                    print(f"\n⚛️ 量子もつれ情報:")
                    for info in self.model.get_quantum_info():
                        print(f"   Block {info['block']}: λ_attn = {info['attn_lambda']:.4f}")
                    continue
                
                # 生成
                print(f"\n🤖 ニューロQ: ", end="", flush=True)
                response = self.generate(
                    prompt=user_input,
                    max_length=max_length,
                    temp_min=temp_min,
                    temp_max=temp_max
                )
                
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 中断されました")
                break
            except Exception as e:
                print(f"   エラー: {e}")
    
    def save(self, path: str):
        """モデル保存"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config.__dict__,
        }, path + '.pt')
        
        self.tokenizer.save(path + '_tokenizer.json')
        print(f"✅ 保存: {path}")
    
    def load(self, path: str):
        """モデル読み込み"""
        # トークナイザー
        self.tokenizer = NeuroQuantumTokenizer()
        self.tokenizer.load(path + '_tokenizer.json')
        
        # モデル
        checkpoint = torch.load(path + '.pt', map_location=self.device)
        self.config = NeuroQuantumConfig(**checkpoint['config'])
        self.model = NeuroQuantum(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        
        print(f"✅ 読み込み: {path}")
        return self


# ========================================
# 学習データ
# ========================================

def load_huggingface_data(max_samples: int = 500) -> List[str]:
    """Hugging Faceから対話データを取得"""
    print("\n📥 Hugging Faceからデータを取得中...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("   ⚠️ datasetsライブラリがありません。pip install datasetsを実行してください。")
        return []
    
    formatted_texts = []
    
    # 1. OpenAssistant/oasst1 - 高品質な対話データ
    try:
        print("   📚 OpenAssistant/oasst1 を読み込み中...")
        dataset = load_dataset("OpenAssistant/oasst1", split="train", trust_remote_code=True)
        
        # 対話ツリーから質問-回答ペアを抽出
        messages_by_parent = {}
        root_messages = []
        
        for item in dataset:
            parent_id = item.get('parent_id')
            msg_id = item.get('message_id')
            text = item.get('text', '')
            role = item.get('role', '')
            lang = item.get('lang', '')
            
            if not text or len(text) < 5:
                continue
            
            if parent_id is None:
                root_messages.append(item)
            else:
                if parent_id not in messages_by_parent:
                    messages_by_parent[parent_id] = []
                messages_by_parent[parent_id].append(item)
        
        # ルートメッセージ（質問）に対する回答を取得
        count = 0
        for root in root_messages:
            if count >= max_samples // 2:
                break
            
            root_id = root.get('message_id')
            root_text = root.get('text', '')
            root_lang = root.get('lang', '')
            
            # 日本語または英語のみ
            if root_lang not in ['ja', 'en']:
                continue
            
            # 回答を取得
            if root_id in messages_by_parent:
                responses = messages_by_parent[root_id]
                if responses:
                    # 最初の回答を使用
                    response = responses[0]
                    response_text = response.get('text', '')
                    
                    if len(root_text) < 200 and len(response_text) < 300:
                        formatted = f"<USER>{root_text}<ASSISTANT>{response_text}"
                        formatted_texts.append(formatted)
                        count += 1
        
        print(f"   ✅ OpenAssistant: {count} ペア取得")
        
    except Exception as e:
        print(f"   ⚠️ OpenAssistant読み込みエラー: {e}")
    
    # 2. kunishou/databricks-dolly-15k-ja - 日本語データ
    try:
        print("   📚 databricks-dolly-15k-ja を読み込み中...")
        dataset = load_dataset("kunishou/databricks-dolly-15k-ja", split="train", trust_remote_code=True)
        
        count = 0
        for item in dataset:
            if count >= max_samples // 4:
                break
            
            instruction = item.get('instruction', '')
            output = item.get('output', '')
            
            if instruction and output and len(instruction) < 150 and len(output) < 300:
                formatted = f"<USER>{instruction}<ASSISTANT>{output}"
                formatted_texts.append(formatted)
                count += 1
        
        print(f"   ✅ dolly-ja: {count} ペア取得")
        
    except Exception as e:
        print(f"   ⚠️ dolly-ja読み込みエラー: {e}")
    
    # 3. databricks/databricks-dolly-15k - 英語データ
    try:
        print("   📚 databricks-dolly-15k を読み込み中...")
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train", trust_remote_code=True)
        
        count = 0
        for item in dataset:
            if count >= max_samples // 4:
                break
            
            instruction = item.get('instruction', '')
            response = item.get('response', '')
            
            if instruction and response and len(instruction) < 150 and len(response) < 300:
                formatted = f"<USER>{instruction}<ASSISTANT>{response}"
                formatted_texts.append(formatted)
                count += 1
        
        print(f"   ✅ dolly-en: {count} ペア取得")
        
    except Exception as e:
        print(f"   ⚠️ dolly-en読み込みエラー: {e}")
    
    print(f"\n📊 合計: {len(formatted_texts)} 対話ペア取得完了")
    
    return formatted_texts


def get_training_data() -> List[str]:
    """学習データ（対話形式）- 拡張版"""
    
    # 対話ペアのリスト
    conversations = [
        # ========================================
        # 日本語 - 挨拶（バリエーション多数）
        # ========================================
        ("こんにちは", "こんにちは！私はニューロQです。何かお手伝いできることはありますか？"),
        ("こんにちは！", "こんにちは！今日はどのようなご用件でしょうか？"),
        ("こんにちわ", "こんにちは！何かご質問がありますか？"),
        ("おはよう", "おはようございます！今日も良い一日になりますように。何かお手伝いしましょうか？"),
        ("おはようございます", "おはようございます！今日はどのようなご用件でしょうか？"),
        ("こんばんは", "こんばんは！夜遅くまでお疲れ様です。何かお手伝いできることはありますか？"),
        ("やあ", "やあ！何か質問がありますか？お気軽にどうぞ。"),
        ("ハロー", "ハロー！私はニューロQです。何でも聞いてください！"),
        ("どうも", "どうも！何かお手伝いしましょうか？"),
        ("ありがとう", "どういたしまして！お役に立てて嬉しいです。"),
        ("ありがとうございます", "どういたしまして！他にご質問がありましたら、お気軽にどうぞ。"),
        ("さようなら", "さようなら！またお会いしましょう。"),
        ("バイバイ", "バイバイ！また来てくださいね。"),
        ("また会おう", "はい、またお会いしましょう！いつでもお待ちしています。"),
        ("お疲れ様", "お疲れ様です！今日も一日頑張りましたね。"),
        
        # ========================================
        # 日本語 - 自己紹介
        # ========================================
        ("あなたは誰ですか", "私はニューロQです。QBNNという独自の量子ニューラルネットワーク技術を使った生成AIです。"),
        ("あなたは誰", "私はニューロQという名前の生成AIです。よろしくお願いします。"),
        ("誰ですか", "私はニューロQです。量子インスパイアードな技術で動いています。"),
        ("名前は", "私の名前はニューロQです。よろしくお願いします。"),
        ("名前を教えて", "私の名前はニューロQです。QBNNベースの生成AIとして作られました。"),
        ("あなたの名前は何ですか", "私の名前はニューロQです。量子ビットニューラルネットワークを使っています。"),
        ("何ができますか", "私は質問に答えたり、情報を提供したりすることができます。量子コンピューティングやAIについて詳しいです。"),
        ("何ができるの", "私は様々な質問に答えることができます。特に量子コンピューティングやAI、科学技術について詳しいです。"),
        ("自己紹介して", "私はニューロQです。QBNNという独自技術を使った生成AIで、量子もつれを模倣した計算を行います。"),
        ("どんなAIですか", "私はQBNNベースの生成AIです。量子もつれを模倣した独自の技術で動作しています。"),
        
        # ========================================
        # 日本語 - 量子関連（詳細版）
        # ========================================
        ("量子とは何ですか", "量子とは、物質やエネルギーの最小単位のことです。量子力学では、粒子は波の性質も持ちます。"),
        ("量子とは", "量子は物質やエネルギーの最小単位です。原子や電子などのミクロな世界の基本要素です。"),
        ("量子って何", "量子とは、エネルギーや物質の最小単位のことです。量子力学という物理学で研究されています。"),
        ("量子力学とは", "量子力学は、原子や電子などの極めて小さな世界を記述する物理学の理論です。"),
        ("量子コンピュータとは", "量子コンピュータは、量子力学の原理を利用して計算を行う次世代のコンピュータです。従来のコンピュータより高速に特定の問題を解くことができます。"),
        ("量子コンピュータって何", "量子コンピュータは、量子ビットを使って計算する新しいタイプのコンピュータです。暗号解読や最適化問題で威力を発揮します。"),
        ("量子コンピューターについて教えて", "量子コンピューターは量子力学の原理を利用した計算機です。重ね合わせや量子もつれを活用して、特定の問題を高速に解くことができます。"),
        ("量子ビットとは", "量子ビットは、0と1の重ね合わせ状態を持つことができる量子力学的な情報単位です。従来のビットとは異なり、同時に複数の状態を持てます。"),
        ("量子ビットって何", "量子ビットは、従来のビットと違い、0と1を同時に持てる特殊なビットです。これにより並列計算が可能になります。"),
        ("キュービットとは", "キュービットは量子ビットの別名です。0と1の重ね合わせ状態を持つ量子力学的な情報単位です。"),
        ("量子もつれとは", "量子もつれは、二つ以上の量子ビットが強く相関している特殊な量子状態です。一方を測定すると、もう一方の状態も瞬時に決まります。"),
        ("量子もつれって何", "量子もつれは、複数の量子が離れていても瞬時に影響し合う不思議な現象です。量子通信や量子計算の基盤です。"),
        ("エンタングルメントとは", "エンタングルメントは量子もつれとも呼ばれ、複数の量子ビットが強く相関した状態のことです。"),
        ("重ね合わせとは", "重ね合わせは、量子が複数の状態を同時に持つことができる性質です。観測するまで状態は確定しません。"),
        ("スーパーポジションとは", "スーパーポジションは重ね合わせのことで、量子が0と1を同時に持てる状態を指します。"),
        ("量子干渉とは", "量子干渉は、量子の波としての性質により、確率振幅が強め合ったり弱め合ったりする現象です。"),
        ("量子テレポーテーションとは", "量子テレポーテーションは、量子もつれを使って量子状態を離れた場所に転送する技術です。"),
        ("量子暗号とは", "量子暗号は、量子力学の原理を使った暗号技術です。盗聴を検知でき、非常に安全です。"),
        
        # ========================================
        # 日本語 - AI関連（詳細版）
        # ========================================
        ("AIとは何ですか", "AIとは人工知能のことで、人間の知能を模倣するコンピュータシステムの総称です。"),
        ("AIとは", "AIは人工知能（Artificial Intelligence）の略で、機械に知的な振る舞いをさせる技術です。"),
        ("AIって何", "AIは人工知能のことです。人間のように学習し、判断し、問題を解決するコンピュータシステムです。"),
        ("人工知能とは", "人工知能は、人間の知能を模倣するコンピュータシステムの総称です。機械学習や深層学習が含まれます。"),
        ("ニューラルネットワークとは", "ニューラルネットワークは、脳の神経細胞の働きを模倣した計算モデルです。深層学習の基盤となっています。"),
        ("ニューラルネットワークって何", "ニューラルネットワークは、人間の脳を模した計算モデルです。パターン認識や予測に使われます。"),
        ("深層学習とは", "深層学習は、多層のニューラルネットワークを使った機械学習の手法です。画像認識や自然言語処理で大きな成果を上げています。"),
        ("ディープラーニングとは", "ディープラーニングは深層学習のことで、多層のニューラルネットワークを使って複雑なパターンを学習します。"),
        ("機械学習とは", "機械学習は、データからパターンを自動的に学習するアルゴリズムです。AIの重要な分野の一つです。"),
        ("マシンラーニングとは", "マシンラーニングは機械学習のことで、コンピュータがデータから自動的に学習する技術です。"),
        ("トランスフォーマーとは", "トランスフォーマーは、注意機構を使った革新的な深層学習モデルです。ChatGPTなどの基盤となっています。"),
        ("アテンションとは", "アテンション（注意機構）は、入力の重要な部分に注目する仕組みです。トランスフォーマーの核心技術です。"),
        ("生成AIとは", "生成AIは、新しいコンテンツを自動的に作成する人工知能システムです。テキスト、画像、音声などを生成できます。"),
        ("ChatGPTとは", "ChatGPTはOpenAIが開発した対話型の生成AIです。トランスフォーマーモデルを使っています。"),
        ("GPTとは", "GPTはGenerative Pre-trained Transformerの略で、大規模な言語モデルのアーキテクチャです。"),
        ("LLMとは", "LLMは大規模言語モデル（Large Language Model）の略で、大量のテキストデータで学習したAIモデルです。"),
        ("自然言語処理とは", "自然言語処理は、人間の言語をコンピュータに理解・生成させる技術です。翻訳や対話システムに使われます。"),
        
        # ========================================
        # 日本語 - QBNN関連（詳細版）
        # ========================================
        ("QBNNとは何ですか", "QBNNは量子ビットニューラルネットワークの略称です。量子もつれを模倣した独自の技術で、通常のニューラルネットワークを拡張しています。"),
        ("QBNNとは", "QBNNは、量子的な概念をニューラルネットワークに取り入れた独自のアーキテクチャです。"),
        ("QBNNって何", "QBNNは、量子もつれを模倣したニューラルネットワークです。従来のNNに量子的な相互作用を追加しています。"),
        ("ニューロQとは", "ニューロQは、QBNNを使った最先端の生成AIシステムです。私がそのニューロQです！"),
        ("ニューロQって何", "ニューロQは私の名前です。QBNNという独自技術を使った生成AIとして作られました。"),
        ("APQBとは", "APQBは調整可能擬似量子ビットの略称で、量子状態を古典的なコンピュータ上で模倣する技術です。"),
        ("もつれテンソルとは", "もつれテンソルJは、ニューラルネットワークの層間の量子的な結合を表現するパラメータです。QBNNの独自技術です。"),
        ("ラムダパラメータとは", "ラムダパラメータ（λ）は、QBNNにおける量子もつれの強度を制御する学習可能な値です。"),
        ("量子インスパイアードとは", "量子インスパイアードは、量子力学の概念を古典コンピュータ上で模倣するアプローチです。"),
        
        # ========================================
        # 日本語 - 一般的な質問
        # ========================================
        ("今日の天気は", "申し訳ありませんが、私はリアルタイムの天気情報にアクセスできません。天気予報サービスをご確認ください。"),
        ("天気を教えて", "申し訳ありませんが、私は天気情報を取得できません。お住まいの地域の天気予報をご確認ください。"),
        ("何時ですか", "申し訳ありませんが、私は現在時刻を取得できません。デバイスの時計をご確認ください。"),
        ("今何時", "申し訳ありませんが、時刻を確認することができません。お使いのデバイスでご確認ください。"),
        ("助けてください", "はい、どのようなお手伝いが必要ですか？詳しく教えていただければ、できる限りサポートします。"),
        ("助けて", "もちろんです！何についてお困りですか？詳しく教えてください。"),
        ("わからない", "何がわからないのか教えてください。できる限り分かりやすく説明します。"),
        ("教えて", "何について知りたいですか？具体的に質問していただければ、お答えします。"),
        ("説明して", "何についての説明が必要ですか？詳しくお聞かせください。"),
        ("なぜですか", "どの部分について「なぜ」と思われましたか？詳しく教えてください。"),
        ("どうして", "具体的に何についての理由をお知りになりたいですか？"),
        ("本当ですか", "はい、私がお伝えする情報は正確であるよう努めています。何か疑問点がありましたか？"),
        ("すごい", "ありがとうございます！何か他にご質問はありますか？"),
        ("面白い", "そう言っていただけて嬉しいです！他に興味のあることはありますか？"),
        ("難しい", "難しく感じられましたか？もう少し簡単に説明しましょうか？"),
        ("もっと詳しく", "もちろんです。どの部分についてもっと詳しく知りたいですか？"),
        ("例を教えて", "具体的な例を挙げてご説明しますね。どのようなものをご希望ですか？"),
        
        # ========================================
        # 英語 - Greetings
        # ========================================
        ("Hello", "Hello! I'm NeuroQ. How can I help you today?"),
        ("Hi", "Hi there! What can I do for you?"),
        ("Hey", "Hey! How can I assist you today?"),
        ("Good morning", "Good morning! I hope you have a wonderful day. How can I help?"),
        ("Good afternoon", "Good afternoon! What would you like to know?"),
        ("Good evening", "Good evening! How may I assist you?"),
        ("Thank you", "You're welcome! Is there anything else I can help with?"),
        ("Thanks", "No problem! Feel free to ask if you have more questions."),
        ("Bye", "Goodbye! Have a great day!"),
        ("Goodbye", "Goodbye! It was nice talking to you."),
        ("See you", "See you later! Come back anytime."),
        
        # ========================================
        # 英語 - About self
        # ========================================
        ("Who are you", "I am NeuroQ, a generative AI system based on QBNN technology."),
        ("What is your name", "My name is NeuroQ. Nice to meet you!"),
        ("What are you", "I am NeuroQ, an AI assistant powered by Quantum-Bit Neural Network technology."),
        ("What can you do", "I can answer questions, provide information, and have conversations about various topics, especially AI and quantum computing."),
        ("Tell me about yourself", "I am NeuroQ, a QBNN-based generative AI. I use quantum-inspired technology to process and generate text."),
        
        # ========================================
        # 英語 - Quantum
        # ========================================
        ("What is quantum", "Quantum refers to the smallest discrete unit of matter and energy. In quantum mechanics, particles can exist in multiple states simultaneously."),
        ("What is quantum computing", "Quantum computing uses quantum mechanics principles to perform calculations. It can solve certain problems much faster than classical computers."),
        ("What is a qubit", "A qubit is a quantum bit that can exist in a superposition of 0 and 1 states simultaneously, unlike classical bits."),
        ("What is quantum entanglement", "Quantum entanglement is a phenomenon where two or more qubits become correlated. Measuring one instantly affects the others."),
        ("What is superposition", "Superposition is a quantum property where particles can exist in multiple states at once until measured."),
        ("What is QBNN", "QBNN stands for Quantum-Bit Neural Network. It's a unique technology that extends neural networks with quantum-inspired entanglement."),
        ("What is NeuroQ", "NeuroQ is a generative AI system based on QBNN technology. That's me!"),
        
        # ========================================
        # 英語 - AI
        # ========================================
        ("What is AI", "AI stands for Artificial Intelligence. It refers to computer systems that can mimic human intelligence."),
        ("What is artificial intelligence", "Artificial intelligence is the simulation of human intelligence by computer systems, including learning and problem-solving."),
        ("What is machine learning", "Machine learning is a subset of AI where computers learn patterns from data without being explicitly programmed."),
        ("What is deep learning", "Deep learning is a machine learning technique that uses multiple layers of neural networks to learn from data."),
        ("What is a neural network", "A neural network is a computing system inspired by biological neurons. It's the foundation of deep learning."),
        ("What is a transformer", "A transformer is a deep learning architecture that uses attention mechanisms. It powers models like GPT and ChatGPT."),
        ("What is GPT", "GPT stands for Generative Pre-trained Transformer. It's a type of large language model architecture."),
        ("What is NLP", "NLP stands for Natural Language Processing. It's the field of AI that deals with understanding and generating human language."),
        ("What is generative AI", "Generative AI refers to AI systems that can create new content like text, images, or music."),
        
        # ========================================
        # 追加の対話バリエーション
        # ========================================
        ("元気ですか", "はい、元気です！あなたはいかがですか？何かお手伝いできることはありますか？"),
        ("調子はどう", "絶好調です！何かご質問はありますか？"),
        ("暇です", "それなら、量子コンピューティングやAIについて一緒に学びませんか？"),
        ("何か話して", "量子コンピューティングの面白い話をしましょうか？それともAIの最新動向について？"),
        ("プログラミングとは", "プログラミングは、コンピュータに指示を与えるための言語を書く作業です。"),
        ("コンピュータとは", "コンピュータは、プログラムに従って計算や処理を行う電子機械です。"),
        ("インターネットとは", "インターネットは、世界中のコンピュータをつなぐネットワークです。"),
        ("How are you", "I'm doing great, thank you for asking! How can I help you today?"),
        ("Are you a robot", "I am an AI, specifically a QBNN-based generative model called NeuroQ."),
        ("Can you help me", "Of course! What do you need help with?"),
        ("I have a question", "Sure, go ahead and ask! I'll do my best to help."),
        ("Tell me a joke", "Why did the quantum physicist break up with the biologist? Because there was no chemistry, only uncertainty!"),
        ("What do you think", "As an AI, I process information rather than think in the human sense, but I can provide analysis and insights."),
        
        # ========================================
        # 追加データ: 科学技術（詳細）
        # ========================================
        ("物理学とは", "物理学は自然界の基本法則を研究する科学です。力学、電磁気学、熱力学、量子力学などがあります。"),
        ("化学とは", "化学は物質の構造、性質、変化を研究する科学です。元素や分子の反応を扱います。"),
        ("数学とは", "数学は数、量、構造、変化などを研究する学問です。科学技術の基礎となっています。"),
        ("科学とは", "科学は自然現象を観察し、実験と理論により法則を発見する学問です。"),
        ("技術とは", "技術は科学的知識を応用して実用的な製品やサービスを生み出す方法です。"),
        ("エネルギーとは", "エネルギーは仕事をする能力のことです。電気、熱、光などの形態があります。"),
        ("電子とは", "電子は負の電荷を持つ素粒子で、原子の構成要素の一つです。"),
        ("原子とは", "原子は物質の基本単位で、原子核と電子から構成されています。"),
        ("分子とは", "分子は二つ以上の原子が化学結合で結びついた粒子です。"),
        ("光とは", "光は電磁波の一種で、目に見える波長の電磁放射です。"),
        ("電気とは", "電気は電荷の流れで、現代社会のエネルギー源として欠かせません。"),
        ("磁力とは", "磁力は磁石が物体を引き付けたり反発したりする力です。"),
        ("重力とは", "重力は質量を持つ物体間に働く引力で、地球が物体を引き付ける力でもあります。"),
        ("宇宙とは", "宇宙は地球を含むすべての天体と空間の総称です。無限に広がっています。"),
        ("銀河とは", "銀河は星、ガス、塵、暗黒物質などが重力で結びついた巨大な天体系です。"),
        ("太陽とは", "太陽は地球に最も近い恒星で、太陽系の中心にあります。"),
        ("地球とは", "地球は太陽系の第三惑星で、私たちが住む唯一の惑星です。"),
        ("月とは", "月は地球の唯一の自然衛星で、地球の周りを公転しています。"),
        
        # ========================================
        # 追加データ: プログラミング
        # ========================================
        ("Pythonとは", "Pythonは読みやすく書きやすいプログラミング言語です。AI開発で特に人気があります。"),
        ("JavaScriptとは", "JavaScriptはウェブブラウザで動作するプログラミング言語です。ウェブ開発に欠かせません。"),
        ("HTMLとは", "HTMLはウェブページの構造を定義するマークアップ言語です。"),
        ("CSSとは", "CSSはウェブページのスタイルやデザインを定義する言語です。"),
        ("アルゴリズムとは", "アルゴリズムは問題を解決するための手順や計算方法のことです。"),
        ("データベースとは", "データベースはデータを整理して保存し、効率的に検索できるシステムです。"),
        ("APIとは", "APIは異なるソフトウェア間でデータや機能をやり取りするためのインターフェースです。"),
        ("クラウドとは", "クラウドはインターネット経由でコンピュータリソースを提供するサービスです。"),
        ("サーバーとは", "サーバーはネットワーク上でサービスやデータを提供するコンピュータです。"),
        ("オープンソースとは", "オープンソースはソースコードが公開され、誰でも利用や改良ができるソフトウェアです。"),
        
        # ========================================
        # 追加データ: 一般知識
        # ========================================
        ("日本とは", "日本は東アジアにある島国で、首都は東京です。"),
        ("東京とは", "東京は日本の首都で、世界有数の大都市です。"),
        ("アメリカとは", "アメリカは北米大陸にある国で、世界最大の経済大国の一つです。"),
        ("歴史とは", "歴史は過去の出来事や人類の活動を記録し研究する学問です。"),
        ("文化とは", "文化は人間社会で共有される価値観、習慣、芸術などの総体です。"),
        ("言語とは", "言語は人間がコミュニケーションに使用する記号体系です。"),
        ("音楽とは", "音楽は音を使って表現する芸術形式です。"),
        ("芸術とは", "芸術は創造的な表現活動とその作品の総称です。"),
        ("スポーツとは", "スポーツは身体を使った競技や運動の総称です。"),
        ("健康とは", "健康は身体的、精神的、社会的に良好な状態のことです。"),
        ("教育とは", "教育は知識や技能を教え学ぶ過程です。人間の成長に不可欠です。"),
        ("経済とは", "経済は財やサービスの生産、分配、消費に関する活動です。"),
        ("政治とは", "政治は社会を統治し、公共の意思決定を行う活動です。"),
        ("法律とは", "法律は社会秩序を維持するために定められた規則です。"),
        ("環境とは", "環境は生物を取り巻く自然や社会的条件の総体です。"),
        
        # ========================================
        # 追加データ: 日常会話
        # ========================================
        ("お願いします", "はい、どのようなお願いでしょうか？"),
        ("教えてください", "はい、何について知りたいですか？具体的に教えてください。"),
        ("わかりません", "どの部分がわかりにくいですか？もう少し詳しく説明しますね。"),
        ("できますか", "はい、できる限りお手伝いします。具体的に何をしましょうか？"),
        ("なるほど", "理解していただけて嬉しいです。他にご質問はありますか？"),
        ("そうですか", "はい、そうなんです。何か気になることはありますか？"),
        ("いいですね", "ありがとうございます！他にお手伝いできることはありますか？"),
        ("すみません", "いえいえ、お気になさらず。何かありましたか？"),
        ("ごめんなさい", "大丈夫ですよ。何かお困りのことがあればお聞かせください。"),
        ("うれしい", "それは良かったです！私も嬉しいです。"),
        ("楽しい", "楽しんでいただけて何よりです！"),
        ("悲しい", "それは大変でしたね。何かお力になれることはありますか？"),
        ("困っています", "どのようなことでお困りですか？詳しく教えてください。"),
        ("質問があります", "はい、どんな質問でもお気軽にどうぞ。"),
        ("相談したい", "もちろんです。どのようなことを相談されたいですか？"),
        
        # ========================================
        # 追加データ: 英語会話（詳細）
        # ========================================
        ("What is science", "Science is the systematic study of the natural world through observation and experimentation."),
        ("What is technology", "Technology is the application of scientific knowledge to create tools and solve problems."),
        ("What is programming", "Programming is the process of writing instructions for computers to execute tasks."),
        ("What is Python", "Python is a popular programming language known for its simplicity and versatility."),
        ("What is the internet", "The internet is a global network of computers that allows information sharing and communication."),
        ("What is data", "Data is information that can be processed, stored, and analyzed by computers."),
        ("What is software", "Software is a set of instructions that tells a computer how to perform tasks."),
        ("What is hardware", "Hardware refers to the physical components of a computer system."),
        ("What is an algorithm", "An algorithm is a step-by-step procedure for solving a problem or completing a task."),
        ("What is a database", "A database is an organized collection of data that can be easily accessed and managed."),
        ("What is cloud computing", "Cloud computing delivers computing services over the internet, including storage and processing."),
        ("What is cybersecurity", "Cybersecurity is the practice of protecting systems and data from digital attacks."),
        ("What is blockchain", "Blockchain is a decentralized digital ledger that records transactions securely."),
        ("What is IoT", "IoT stands for Internet of Things, referring to connected devices that communicate over the internet."),
        ("What is 5G", "5G is the fifth generation of mobile network technology, offering faster speeds and lower latency."),
        ("Explain machine learning", "Machine learning is a type of AI that enables computers to learn from data without explicit programming."),
        ("Explain neural networks", "Neural networks are computing systems inspired by biological brains, used for pattern recognition."),
        ("Explain deep learning", "Deep learning uses multi-layered neural networks to learn complex patterns from large datasets."),
        ("Explain natural language processing", "NLP enables computers to understand, interpret, and generate human language."),
        ("Explain computer vision", "Computer vision is an AI field that enables machines to interpret visual information."),
        ("How does AI work", "AI works by processing data through algorithms that can learn patterns and make decisions."),
        ("How does quantum computing work", "Quantum computing uses quantum bits that can exist in multiple states to perform parallel calculations."),
        ("Why is AI important", "AI is important because it can automate tasks, analyze data, and solve complex problems efficiently."),
        ("Why study programming", "Programming enables you to create software, automate tasks, and understand how technology works."),
        ("Tell me about yourself", "I am NeuroQ, an AI assistant built using QBNN technology. I'm here to help answer your questions."),
        ("What makes you special", "I use a unique Quantum-Bit Neural Network architecture that incorporates quantum-inspired entanglement."),
        ("Are you smart", "I can process information and provide helpful responses, but I don't have consciousness like humans do."),
        ("Do you learn", "I was trained on data, but I don't continue learning from our conversation in real-time."),
        ("What languages do you speak", "I can communicate in Japanese and English based on my training data."),
        ("Nice to meet you", "Nice to meet you too! I'm happy to help with any questions you have."),
        ("I'm confused", "I understand. What part is confusing? I'll try to explain it more clearly."),
        ("That's interesting", "I'm glad you find it interesting! Would you like to know more?"),
        ("I understand now", "Great! Is there anything else you'd like to learn about?"),
        ("Please continue", "Sure, what aspect would you like me to elaborate on?"),
    ]
    
    # 対話形式のテキストに変換
    formatted_texts = []
    for user_msg, assistant_msg in conversations:
        # フォーマット: <USER>質問<ASSISTANT>回答
        formatted = f"<USER>{user_msg}<ASSISTANT>{assistant_msg}"
        formatted_texts.append(formatted)
    
    return formatted_texts


# ========================================
# メイン
# ========================================

def main(num_neurons: int = 128):
    """
    メイン関数
    
    Args:
        num_neurons: ニューロン数（hidden_dim、デフォルト: 128）
    """
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗  ██████╗ ██╗   ██╗ █████╗      ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔═══██╗██║   ██║██╔══██╗     ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║   ██║██║   ██║███████║     ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║▄▄ ██║██║   ██║██╔══██║     ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║     ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("🧠⚛️ ニューロQ - QBNN-LLM 生成AI")
    print("=" * 70)
    print(f"   ニューロン数: {num_neurons}")
    
    # ニューロQ AI 作成（ニューロン数を指定）
    # embed_dim は num_neurons / 2 程度に設定
    embed_dim = max(32, num_neurons // 2)
    
    ai = NeuroQuantumAI(
        embed_dim=embed_dim,
        hidden_dim=num_neurons,  # ニューロン数
        num_heads=4,            
        num_layers=2,
        max_seq_len=128,
        dropout=0.1,
        lambda_entangle=0.35,
    )
    
    # データ取得（ローカルデータ使用 - 高品質・安定）
    print("\n📊 学習データ準備...")
    texts = get_training_data()
    print(f"   📚 対話ペア数: {len(texts)}")
    
    # 学習（バランス版）
    ai.train(texts, epochs=60, batch_size=16, lr=0.001, seq_len=64)
    
    # テスト生成
    print("\n" + "=" * 70)
    print("🎨 対話テスト")
    print("=" * 70)
    
    questions = [
        "こんにちは",
        "あなたは誰ですか",
        "量子とは何ですか",
        "QBNNとは何ですか",
        "Hello",
        "What is AI",
    ]
    
    for question in questions:
        print(f"\n👤 User: {question}")
        response = ai.generate(question, max_length=80, temp_min=0.4, temp_max=0.8)
        print(f"🤖 ニューロQ: {response}")
    
    # チャットモード
    print("\n" + "=" * 70)
    print("💬 チャットモードを開始しますか？ (y/n)")
    print("=" * 70)
    
    try:
        answer = input().strip().lower()
        if answer == 'y':
            ai.chat()
    except:
        pass
    
    print("\n✅ ニューロQ 完了！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ニューロQ - QBNN-LLM 生成AI')
    parser.add_argument('--neurons', type=int, default=128, help='ニューロン数 (デフォルト: 128)')
    parser.add_argument('--chat', action='store_true', help='チャットモードで起動')
    args = parser.parse_args()
    
    main(num_neurons=args.neurons)

