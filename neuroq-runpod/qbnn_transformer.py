#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗ ██████╗ ███╗   ██╗███╗   ██╗                                       ║
║  ██╔═══██╗██╔══██╗████╗  ██║████╗  ██║                                       ║
║  ██║   ██║██████╔╝██╔██╗ ██║██╔██╗ ██║                                       ║
║  ██║▄▄ ██║██╔══██╗██║╚██╗██║██║╚██╗██║                                       ║
║  ╚██████╔╝██████╔╝██║ ╚████║██║ ╚████║                                       ║
║   ╚══▀▀═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝                                       ║
║                                                                               ║
║  ████████╗██████╗  █████╗ ███╗   ██╗███████╗███████╗ ██████╗ ██████╗ ███╗   ███╗║
║  ╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔═══██╗██╔══██╗████╗ ████║║
║     ██║   ██████╔╝███████║██╔██╗ ██║███████╗█████╗  ██║   ██║██████╔╝██╔████╔██║║
║     ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║║
║     ██║   ██║  ██║██║  ██║██║ ╚████║███████║██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║║
║     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝║
║                                                                               ║
║   QBNN Transformer: 量子ビットニューラルネットワークによるTransformer          ║
║                                                                               ║
║   特徴:                                                                       ║
║   - 量子注意機構 (Quantum Attention) - APQBベースの自己注意                   ║
║   - 量子もつれマルチヘッド (Entangled Multi-Head) - 量子相関を活用            ║
║   - 量子位置エンコーディング (Quantum Positional Encoding)                    ║
║   - 幾何学的制約による正則化 (r² + T² = 1)                                    ║
║                                                                               ║
║   transformersライブラリ不要 - 純粋なPyTorch + QBNN実装                       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🧠⚛️ QBNN Transformer")
print("   量子ビットニューラルネットワークによるTransformer")
print("   (transformersライブラリ不要)")
print("=" * 70)


# ========================================================================
# 1. APQB (Adjustable Pseudo Quantum Bit) - 量子ビットのコア
# ========================================================================

class APQB:
    """
    調整可能擬似量子ビット (APQB)
    
    量子状態を古典コンピュータ上でシミュレート
    - θ: 内部角度パラメータ (0 ≤ θ ≤ π/2)
    - r = cos(2θ): 相関係数 (-1 ≤ r ≤ 1)
    - T = |sin(2θ)|: 温度/揺らぎ (0 ≤ T ≤ 1)
    - 制約: r² + T² = 1 (単位円上)
    """
    
    @staticmethod
    def theta_to_state(theta: torch.Tensor) -> torch.Tensor:
        """θ → 量子状態ベクトル [cos(θ), sin(θ)]"""
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    
    @staticmethod
    def theta_to_r(theta: torch.Tensor) -> torch.Tensor:
        """θ → 相関係数 r = cos(2θ)"""
        return torch.cos(2 * theta)
    
    @staticmethod
    def theta_to_T(theta: torch.Tensor) -> torch.Tensor:
        """θ → 温度 T = |sin(2θ)|"""
        return torch.abs(torch.sin(2 * theta))
    
    @staticmethod
    def theta_to_complex(theta: torch.Tensor) -> torch.Tensor:
        """θ → 複素数表現 z = e^{i2θ} = (cos(2θ), sin(2θ))"""
        return torch.stack([torch.cos(2 * theta), torch.sin(2 * theta)], dim=-1)
    
    @staticmethod
    def constraint_loss(theta: torch.Tensor) -> torch.Tensor:
        """幾何学的制約 r² + T² = 1 の損失"""
        r = APQB.theta_to_r(theta)
        T = APQB.theta_to_T(theta)
        return ((r**2 + T**2 - 1) ** 2).mean()
    
    @staticmethod
    def measure(theta: torch.Tensor) -> torch.Tensor:
        """量子測定（確率的に0 or 1を返す）"""
        prob_1 = torch.sin(theta) ** 2
        return (torch.rand_like(prob_1) < prob_1).float()


# ========================================================================
# 2. 量子回転ゲート (Quantum Rotation Gates)
# ========================================================================

class QuantumRotationGate(nn.Module):
    """
    量子回転ゲート
    
    Rx(θ), Ry(θ), Rz(θ) の実装
    位置エンコーディングや状態変換に使用
    """
    
    def __init__(self, dim: int, gate_type: str = 'Ry'):
        super().__init__()
        self.dim = dim
        self.gate_type = gate_type
        self.theta = nn.Parameter(torch.rand(dim) * np.pi / 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        回転ゲートを適用
        
        Args:
            x: 入力テンソル [..., dim]
        
        Returns:
            回転後のテンソル [..., dim]
        """
        if self.gate_type == 'Rx':
            # Rx(θ) = cos(θ/2)I - i*sin(θ/2)X
            cos_t = torch.cos(self.theta / 2)
            sin_t = torch.sin(self.theta / 2)
            return x * cos_t + x.roll(1, dims=-1) * sin_t
        
        elif self.gate_type == 'Ry':
            # Ry(θ) = cos(θ/2)I - i*sin(θ/2)Y  
            cos_t = torch.cos(self.theta / 2)
            sin_t = torch.sin(self.theta / 2)
            return x * cos_t + torch.flip(x, dims=[-1]) * sin_t
        
        elif self.gate_type == 'Rz':
            # Rz(θ) = e^{-iθ/2}|0⟩⟨0| + e^{iθ/2}|1⟩⟨1|
            phase = torch.exp(1j * self.theta / 2)
            return x * phase.real
        
        return x


# ========================================================================
# 3. 量子位置エンコーディング (Quantum Positional Encoding)
# ========================================================================

class QuantumPositionalEncoding(nn.Module):
    """
    量子位置エンコーディング
    
    従来のsin/cosエンコーディングに加え、
    量子回転ゲートによる位相情報を付加
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 従来の位置エンコーディング
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # 量子回転ゲートによる追加エンコーディング
        self.quantum_gate = QuantumRotationGate(d_model, 'Ry')
        
        # 量子位相パラメータ
        self.quantum_phase = nn.Parameter(torch.rand(max_len, d_model) * np.pi / 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        位置エンコーディングを適用
        
        Args:
            x: 入力 [batch, seq_len, d_model]
        
        Returns:
            位置エンコーディング付き [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # 従来の位置エンコーディング
        classical_pe = self.pe[:, :seq_len, :]
        
        # 量子位相エンコーディング
        quantum_pe = torch.cos(self.quantum_phase[:seq_len, :]) * 0.1
        
        # 結合
        x = x + classical_pe + quantum_pe.unsqueeze(0)
        
        # 量子回転ゲートを適用
        x = self.quantum_gate(x)
        
        return self.dropout(x)


# ========================================================================
# 4. 量子注意機構 (Quantum Attention)
# ========================================================================

class QuantumAttention(nn.Module):
    """
    量子注意機構 (Quantum Attention)
    
    APQBを使用した自己注意メカニズム
    - 量子状態ベースのQuery/Key/Value変換
    - 量子もつれによる相関計算
    - 確率的サンプリングオプション
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        
        # Q, K, V の線形変換
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 量子状態変換（θパラメータ生成）
        self.theta_proj = nn.Linear(d_model, d_model)
        
        # 出力投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 量子もつれ強度
        self.entangle_strength = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        use_quantum_sampling: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        量子注意の計算
        
        Args:
            x: 入力 [batch, seq_len, d_model]
            mask: アテンションマスク [batch, seq_len, seq_len]
            use_quantum_sampling: 量子サンプリングを使用するか
        
        Returns:
            output: 注意出力 [batch, seq_len, d_model]
            attn_weights: 注意重み [batch, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V を計算
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 量子状態θを計算
        theta_q = torch.sigmoid(self.theta_proj(Q)) * np.pi / 2
        theta_k = torch.sigmoid(self.theta_proj(K)) * np.pi / 2
        
        # 量子相関係数 r を計算
        r_q = APQB.theta_to_r(theta_q)  # [batch, seq_len, d_model]
        r_k = APQB.theta_to_r(theta_k)
        
        # 量子温度 T を計算（揺らぎ）
        T_q = APQB.theta_to_T(theta_q)
        T_k = APQB.theta_to_T(theta_k)
        
        # === 量子もつれアテンション ===
        
        # 1. 古典的なスケールドドットプロダクト
        classical_attn = torch.bmm(Q, K.transpose(-2, -1)) / self.scale
        
        # 2. 量子相関項（r_q と r_k の相互作用）
        quantum_corr = torch.bmm(r_q, r_k.transpose(-2, -1)) * self.entangle_strength
        
        # 3. 温度による揺らぎ
        if use_quantum_sampling:
            noise = torch.randn_like(classical_attn) * T_q.mean(dim=-1, keepdim=True)
            classical_attn = classical_attn + noise * 0.1
        
        # 4. 最終アテンションスコア
        attn_scores = classical_attn + quantum_corr
        
        # マスク適用
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Value との積
        output = torch.bmm(attn_weights, V)
        output = self.out_proj(output)
        
        return output, attn_weights
    
    def get_quantum_stats(self, x: torch.Tensor) -> dict:
        """量子統計情報を取得"""
        theta = torch.sigmoid(self.theta_proj(x)) * np.pi / 2
        r = APQB.theta_to_r(theta)
        T = APQB.theta_to_T(theta)
        
        return {
            'r_mean': r.mean().item(),
            'T_mean': T.mean().item(),
            'entangle_strength': self.entangle_strength.item(),
            'constraint': (r**2 + T**2).mean().item()
        }


# ========================================================================
# 5. 量子マルチヘッドアテンション (Quantum Multi-Head Attention)
# ========================================================================

class QuantumMultiHeadAttention(nn.Module):
    """
    量子マルチヘッドアテンション
    
    複数のQuantumAttentionヘッドを使用し、
    ヘッド間で量子もつれを共有
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V の線形変換
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 出力投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 量子状態θ投影（各ヘッド用）
        self.theta_proj = nn.Linear(self.d_k, self.d_k)
        
        # ヘッド間もつれパラメータ
        self.inter_head_entangle = nn.Parameter(torch.rand(num_heads, num_heads) * 0.1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # 各ヘッドのもつれ強度
        self.entangle_strengths = nn.Parameter(torch.ones(num_heads) * 0.5)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        use_quantum_sampling: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        マルチヘッド量子注意の計算
        
        Args:
            x: 入力 [batch, seq_len, d_model]
            mask: アテンションマスク
            use_quantum_sampling: 量子サンプリング使用
        
        Returns:
            output: 出力 [batch, seq_len, d_model]
            attn_weights: 注意重み [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V を計算してヘッドに分割
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # [batch, num_heads, seq_len, d_k]
        
        # 各ヘッドで量子注意を計算
        all_attn_scores = []
        all_quantum_corrs = []
        
        for h in range(self.num_heads):
            q_h = Q[:, h]  # [batch, seq_len, d_k]
            k_h = K[:, h]
            
            # 量子状態θを計算
            theta_q = torch.sigmoid(self.theta_proj(q_h)) * np.pi / 2
            theta_k = torch.sigmoid(self.theta_proj(k_h)) * np.pi / 2
            
            # 量子相関
            r_q = APQB.theta_to_r(theta_q)
            r_k = APQB.theta_to_r(theta_k)
            
            # 古典的アテンション
            classical_attn = torch.bmm(q_h, k_h.transpose(-2, -1)) / self.scale
            
            # 量子相関項
            quantum_corr = torch.bmm(r_q, r_k.transpose(-2, -1)) * self.entangle_strengths[h]
            
            all_attn_scores.append(classical_attn)
            all_quantum_corrs.append(quantum_corr)
        
        # スタックしてヘッド間もつれを適用
        attn_scores = torch.stack(all_attn_scores, dim=1)  # [batch, heads, seq, seq]
        quantum_corrs = torch.stack(all_quantum_corrs, dim=1)
        
        # ヘッド間もつれ（他のヘッドの量子相関を混合）
        inter_head_effect = torch.einsum('bhij,hg->bgij', quantum_corrs, self.inter_head_entangle)
        
        # 最終スコア
        final_scores = attn_scores + quantum_corrs + inter_head_effect * 0.1
        
        # 量子サンプリング
        if use_quantum_sampling:
            noise = torch.randn_like(final_scores) * 0.05
            final_scores = final_scores + noise
        
        # マスク適用
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq, seq]
            final_scores = final_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(final_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Valueとの積
        output = torch.matmul(attn_weights, V)  # [batch, heads, seq, d_k]
        
        # ヘッドを結合
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights


# ========================================================================
# 6. 量子フィードフォワードネットワーク (Quantum Feed-Forward)
# ========================================================================

class QuantumFeedForward(nn.Module):
    """
    量子フィードフォワードネットワーク
    
    APQBベースの非線形変換を使用
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 2層のFFN
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # 量子パラメータ
        self.theta_ff = nn.Parameter(torch.rand(d_ff) * np.pi / 4)
        
        self.dropout = nn.Dropout(dropout)
    
    def quantum_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        量子活性化関数
        
        GELU + 量子揺らぎの組み合わせ
        """
        # 基本のGELU
        gelu = F.gelu(x)
        
        # 量子温度による揺らぎ
        T = APQB.theta_to_T(self.theta_ff)
        quantum_noise = T * 0.1
        
        return gelu * (1 + quantum_noise.unsqueeze(0).unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        フィードフォワード計算
        
        Args:
            x: 入力 [batch, seq_len, d_model]
        
        Returns:
            出力 [batch, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.quantum_activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ========================================================================
# 7. QBNN Transformer エンコーダーレイヤー
# ========================================================================

class QBNNTransformerEncoderLayer(nn.Module):
    """
    QBNN Transformer エンコーダーレイヤー
    
    構成:
    1. 量子マルチヘッドアテンション
    2. 残差接続 + レイヤー正規化
    3. 量子フィードフォワード
    4. 残差接続 + レイヤー正規化
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = QuantumMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = QuantumFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        use_quantum_sampling: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        エンコーダーレイヤーの順伝播
        
        Args:
            x: 入力 [batch, seq_len, d_model]
            mask: アテンションマスク
            use_quantum_sampling: 量子サンプリング
        
        Returns:
            output: 出力 [batch, seq_len, d_model]
            attn_weights: 注意重み
        """
        # 量子自己注意
        attn_output, attn_weights = self.self_attn(x, mask, use_quantum_sampling)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 量子フィードフォワード
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


# ========================================================================
# 8. QBNN Transformer デコーダーレイヤー
# ========================================================================

class QBNNTransformerDecoderLayer(nn.Module):
    """
    QBNN Transformer デコーダーレイヤー
    
    構成:
    1. マスク付き量子自己注意
    2. エンコーダー-デコーダー注意（オプション）
    3. 量子フィードフォワード
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # マスク付き自己注意
        self.self_attn = QuantumMultiHeadAttention(d_model, num_heads, dropout)
        
        # エンコーダー-デコーダー注意
        self.cross_attn = QuantumMultiHeadAttention(d_model, num_heads, dropout)
        
        # フィードフォワード
        self.feed_forward = QuantumFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        use_quantum_sampling: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        デコーダーレイヤーの順伝播
        """
        # マスク付き自己注意
        attn_output, self_attn_weights = self.self_attn(x, self_attn_mask, use_quantum_sampling)
        x = self.norm1(x + self.dropout(attn_output))
        
        # エンコーダー-デコーダー注意
        cross_attn_weights = None
        if encoder_output is not None:
            cross_output, cross_attn_weights = self.cross_attn(
                x, cross_attn_mask, use_quantum_sampling
            )
            x = self.norm2(x + self.dropout(cross_output))
        
        # フィードフォワード
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights


# ========================================================================
# 9. QBNN Transformer (完全なモデル)
# ========================================================================

class QBNNTransformer(nn.Module):
    """
    QBNN Transformer
    
    量子ビットニューラルネットワークによる言語モデル
    transformersライブラリ不要
    
    特徴:
    - 量子注意機構
    - 量子もつれマルチヘッド
    - 量子位置エンコーディング
    - 幾何学的制約による正則化
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # 量子位置エンコーディング
        self.pos_encoding = QuantumPositionalEncoding(d_model, max_seq_len, dropout)
        
        # デコーダーレイヤー（言語モデル用）
        self.layers = nn.ModuleList([
            QBNNTransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 出力層
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # 重み共有（embedding と output）
        self.output_proj.weight = self.embedding.weight
        
        # 初期化
        self._init_weights()
        
        print(f"✅ QBNN Transformer 初期化完了")
        print(f"   vocab_size: {vocab_size}")
        print(f"   d_model: {d_model}")
        print(f"   num_heads: {num_heads}")
        print(f"   num_layers: {num_layers}")
        print(f"   パラメータ数: {self.count_parameters():,}")
    
    def _init_weights(self):
        """重みの初期化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def count_parameters(self) -> int:
        """パラメータ数をカウント"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """因果マスクを生成（未来のトークンを見えなくする）"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_quantum_sampling: bool = False
    ) -> torch.Tensor:
        """
        順伝播
        
        Args:
            input_ids: 入力トークンID [batch, seq_len]
            attention_mask: アテンションマスク [batch, seq_len]
            use_quantum_sampling: 量子サンプリングを使用
        
        Returns:
            logits: 出力ロジット [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 埋め込み + 位置エンコーディング
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 因果マスク（自己回帰用）
        causal_mask = self.generate_causal_mask(seq_len, device)
        
        # パディングマスクと組み合わせ
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = causal_mask.unsqueeze(0) + (1 - padding_mask) * float('-inf')
        else:
            combined_mask = causal_mask.unsqueeze(0)
        
        # Transformerレイヤーを通過
        for layer in self.layers:
            x, _, _ = layer(x, None, combined_mask, None, use_quantum_sampling)
        
        # 出力
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        use_quantum_sampling: bool = True,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        テキスト生成
        
        Args:
            input_ids: 開始トークン [batch, seq_len] or [seq_len]
            max_length: 最大生成長
            temperature: 温度（高いほどランダム）
            top_k: Top-K サンプリング
            top_p: Nucleus サンプリング
            repetition_penalty: 繰り返しペナルティ
            use_quantum_sampling: 量子サンプリング
            eos_token_id: 終了トークンID
        
        Returns:
            generated: 生成されたトークン [batch, new_seq_len]
        """
        self.eval()
        
        # 入力の形状を調整
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        device = input_ids.device
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        
        for _ in range(max_length):
            # 現在のシーケンスで予測
            logits = self(generated, use_quantum_sampling=use_quantum_sampling)
            
            # 最後のトークンの予測を取得
            next_token_logits = logits[:, -1, :] / temperature
            
            # 繰り返しペナルティ
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in generated[i].unique():
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Top-K フィルタリング
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-P (Nucleus) フィルタリング
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')
            
            # サンプリング
            probs = F.softmax(next_token_logits, dim=-1)
            
            if use_quantum_sampling:
                # 量子ノイズを追加
                quantum_noise = torch.randn_like(probs) * 0.01
                probs = probs + quantum_noise.abs()
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 生成結果に追加
            generated = torch.cat([generated, next_token], dim=1)
            
            # EOSチェック
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
    
    def get_quantum_stats(self) -> List[dict]:
        """各レイヤーの量子統計を取得"""
        stats = []
        for i, layer in enumerate(self.layers):
            layer_stats = {
                'layer': i,
                'entangle_strengths': layer.self_attn.entangle_strengths.tolist(),
                'inter_head_entangle_mean': layer.self_attn.inter_head_entangle.mean().item()
            }
            stats.append(layer_stats)
        return stats
    
    def get_constraint_loss(self) -> torch.Tensor:
        """全レイヤーの幾何学的制約損失"""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for layer in self.layers:
            ff = layer.feed_forward
            theta = ff.theta_ff
            total_loss += APQB.constraint_loss(theta)
        
        return total_loss / len(self.layers)


# ========================================================================
# 10. ユーティリティ関数
# ========================================================================

def create_qbnn_transformer(
    vocab_size: int,
    model_size: str = 'small'
) -> QBNNTransformer:
    """
    QBNN Transformer を作成
    
    Args:
        vocab_size: 語彙サイズ
        model_size: 'tiny', 'small', 'medium', 'large'
    
    Returns:
        QBNNTransformer モデル
    """
    configs = {
        'tiny': {'d_model': 128, 'num_heads': 4, 'num_layers': 2, 'd_ff': 512},
        'small': {'d_model': 256, 'num_heads': 8, 'num_layers': 4, 'd_ff': 1024},
        'medium': {'d_model': 512, 'num_heads': 8, 'num_layers': 6, 'd_ff': 2048},
        'large': {'d_model': 768, 'num_heads': 12, 'num_layers': 12, 'd_ff': 3072}
    }
    
    config = configs.get(model_size, configs['small'])
    
    return QBNNTransformer(
        vocab_size=vocab_size,
        **config
    )


# ========================================================================
# テスト
# ========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 QBNN Transformer テスト")
    print("=" * 70)
    
    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # モデル作成
    vocab_size = 8000
    model = create_qbnn_transformer(vocab_size, 'small').to(device)
    
    # テスト入力
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # 順伝播テスト
    print("\n📊 順伝播テスト...")
    logits = model(input_ids)
    print(f"   入力形状: {input_ids.shape}")
    print(f"   出力形状: {logits.shape}")
    
    # 生成テスト
    print("\n🎲 生成テスト...")
    start_tokens = torch.randint(0, vocab_size, (1, 5)).to(device)
    generated = model.generate(start_tokens, max_length=20, use_quantum_sampling=True)
    print(f"   開始トークン: {start_tokens.shape}")
    print(f"   生成結果: {generated.shape}")
    
    # 量子統計
    print("\n⚛️ 量子統計:")
    stats = model.get_quantum_stats()
    for s in stats[:2]:
        print(f"   Layer {s['layer']}: entangle_mean={s['inter_head_entangle_mean']:.4f}")
    
    # 制約損失
    constraint_loss = model.get_constraint_loss()
    print(f"\n📐 幾何学的制約損失: {constraint_loss.item():.6f}")
    
    print("\n✅ QBNN Transformer テスト完了!")
    print("=" * 70)
