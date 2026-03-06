#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗ ██████╗ ███╗   ██╗███╗   ██╗                                        ║
║  ██╔═══██╗██╔══██╗████╗  ██║████╗  ██║                                        ║
║  ██║   ██║██████╔╝██╔██╗ ██║██╔██╗ ██║                                        ║
║  ██║▄▄ ██║██╔══██╗██║╚██╗██║██║╚██╗██║                                        ║
║  ╚██████╔╝██████╔╝██║ ╚████║██║ ╚████║                                        ║
║   ╚══▀▀═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝                                        ║
║                                                                               ║
║  ███████╗██╗      █████╗ ███████╗██╗  ██╗                                     ║
║  ██╔════╝██║     ██╔══██╗██╔════╝██║  ██║                                     ║
║  █████╗  ██║     ███████║███████╗███████║                                     ║
║  ██╔══╝  ██║     ██╔══██║╚════██║██╔══██║                                     ║
║  ██║     ███████╗██║  ██║███████║██║  ██║                                     ║
║  ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝                                     ║
║                                                                               ║
║   QBNN Flash Transformer                                                      ║
║   量子ビットニューラルネットワーク + FlashAttention                              ║
║                                                                               ║
║   特徴:                                                                       ║
║   - FlashAttention: O(N)メモリ効率、高速アテンション                             ║
║   - QBNN: 量子もつれによる層間相関                                               ║
║   - APQB: 調整可能擬似量子ビット理論                                             ║
║   - 量子位相エンコーディング                                                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("⚛️⚡ QBNN Flash Transformer")
print("   量子ビットニューラルネットワーク + FlashAttention")
print("=" * 70)


# ========================================================================
# 設定
# ========================================================================

@dataclass
class QBNNFlashConfig:
    """QBNN Flash Transformer 設定"""
    vocab_size: int = 8000
    embed_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # QBNN固有パラメータ
    lambda_min: float = 0.2       # 量子もつれ強度の最小値
    lambda_max: float = 0.5       # 量子もつれ強度の最大値
    use_quantum_phase: bool = True  # 量子位相エンコーディング
    entangle_heads: bool = True     # ヘッド間エンタングルメント
    
    # FlashAttention設定
    use_flash_attention: bool = True
    attention_dropout: float = 0.0


# ========================================================================
# APQB（調整可能擬似量子ビット）
# ========================================================================

class APQB(nn.Module):
    """
    Adjustable Pseudo Quantum Bit
    
    量子ビットの数学的性質をニューラルネットワークで表現:
    - θ: 内部角度パラメータ
    - r = cos(2θ): 相関係数 ∈ [-1, 1]
    - T = |sin(2θ)|: 温度（量子ゆらぎ）∈ [0, 1]
    - 制約: r² + T² = 1
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 学習可能なθパラメータ
        self.theta = nn.Parameter(torch.rand(dim) * math.pi / 2)
    
    def get_r(self, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """相関係数 r = cos(2θ)"""
        t = theta if theta is not None else self.theta
        return torch.cos(2 * t)
    
    def get_T(self, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """温度 T = |sin(2θ)|"""
        t = theta if theta is not None else self.theta
        return torch.abs(torch.sin(2 * t))
    
    def get_quantum_state(self, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """量子状態ベクトル [cos(θ), sin(θ)]"""
        t = theta if theta is not None else self.theta
        return torch.stack([torch.cos(t), torch.sin(t)], dim=-1)
    
    def measure(self, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """量子測定（確率的）"""
        t = theta if theta is not None else self.theta
        prob_1 = torch.sin(t) ** 2
        return (torch.rand_like(prob_1) < prob_1).float()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力をAPQB空間に変換
        
        Returns:
            r: 相関係数
            T: 温度
        """
        # 入力をθに変換
        theta = torch.sigmoid(x) * math.pi / 2
        return self.get_r(theta), self.get_T(theta)


# ========================================================================
# 量子位相エンコーディング
# ========================================================================

class QuantumPhaseEncoding(nn.Module):
    """
    量子位相エンコーディング
    
    位置情報を量子位相として表現:
    - φ(pos) = 2π * pos / max_len
    - 回転ゲート R_z(φ) を模倣
    """
    
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 位相テーブル（事前計算）
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        # 量子位相: cos(φ) と sin(φ)
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.cos(position * div_term)  # 実部
        pe[:, 1::2] = torch.sin(position * div_term)  # 虚部
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # 学習可能な位相シフト
        self.phase_shift = nn.Parameter(torch.zeros(1, 1, dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            x + quantum_positional_encoding
        """
        seq_len = x.size(1)
        # 量子位相を加算
        phase = self.pe[:, :seq_len, :] + self.phase_shift
        return x + phase


# ========================================================================
# QBNN層間エンタングルメント
# ========================================================================

class QBNNEntanglement(nn.Module):
    """
    QBNN層間エンタングルメント
    
    隣接層間の量子もつれを表現:
    - e^(l) = f(q^(l), q^(l-1))
    - λ_eff ∈ [λ_min, λ_max] で動的に変化
    """
    
    def __init__(self, dim: int, lambda_min: float = 0.2, lambda_max: float = 0.5):
        super().__init__()
        self.dim = dim
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        
        # エンタングル重み
        self.W_entangle = nn.Linear(dim, dim, bias=False)
        
        # 位相パラメータ
        self.phase = nn.Parameter(torch.rand(dim) * math.pi / 2)
        
        # λベース（学習可能）
        self.lambda_base = nn.Parameter(torch.tensor(0.5))
        
        # 呼び出しカウンタ
        self.register_buffer('call_count', torch.tensor(0))
    
    def get_lambda_eff(self) -> float:
        """動的λを計算"""
        # λを[0,1]に正規化
        lambda_norm = torch.sigmoid(self.lambda_base)
        # [lambda_min, lambda_max]にマッピング
        lambda_eff = self.lambda_min + (self.lambda_max - self.lambda_min) * lambda_norm
        
        # 量子ゆらぎを追加
        if self.training:
            noise = torch.randn(1, device=lambda_norm.device) * 0.01
            lambda_eff = lambda_eff + noise
            lambda_eff = torch.clamp(lambda_eff, self.lambda_min, self.lambda_max)
        
        return lambda_eff
    
    def forward(self, h_current: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        層間エンタングル計算
        
        Args:
            h_current: 現在の層の状態 (batch, seq, dim)
            h_prev: 前の層の状態 (batch, seq, dim)
        
        Returns:
            エンタングルメント補正後の状態
        """
        self.call_count += 1
        
        # 1. 正規化（Bloch球座標）
        s_current = torch.tanh(h_current)
        s_prev = torch.tanh(h_prev)
        
        # 2. エンタングル相互作用
        entangle_signal = self.W_entangle(s_prev)
        
        # 3. 位相回転
        phase_factor = torch.cos(self.phase)
        
        # 4. 相関計算
        correlation = s_current * entangle_signal * phase_factor
        
        # 5. 動的λで重み付け
        lambda_eff = self.get_lambda_eff()
        delta = lambda_eff * correlation
        
        return h_current + delta


# ========================================================================
# QBNN Flash Attention
# ========================================================================

class QBNNFlashAttention(nn.Module):
    """
    QBNN Flash Attention
    
    FlashAttention + 量子もつれヘッド間相関:
    - FlashAttention: メモリ効率O(N)、高速計算
    - QBNN: ヘッド間の量子もつれ相関
    - 量子位相: アテンションスコアへの位相変調
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        lambda_min: float = 0.2,
        lambda_max: float = 0.5,
        entangle_heads: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.entangle_heads = entangle_heads
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 量子位相変調
        self.quantum_phase = nn.Parameter(torch.rand(num_heads) * math.pi)
        
        # ヘッド間エンタングルメント
        if entangle_heads and num_heads > 1:
            self.head_entangle = nn.Parameter(
                torch.randn(num_heads, num_heads) * 0.02
            )
            self.lambda_heads = nn.Parameter(torch.tensor(0.3))
        
        # APQB for attention modulation
        self.apqb = APQB(num_heads)
    
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, embed) -> (batch, heads, seq, head_dim)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def _apply_quantum_phase(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """アテンションスコアに量子位相変調を適用"""
        # attn_weights: (batch, heads, seq, seq)
        phase_modulation = torch.cos(self.quantum_phase).view(1, -1, 1, 1)
        return attn_weights * phase_modulation
    
    def _apply_head_entanglement(self, x: torch.Tensor) -> torch.Tensor:
        """ヘッド間エンタングルメントを適用"""
        if not self.entangle_heads or self.num_heads == 1:
            return x
        
        # x: (batch, heads, seq, head_dim)
        batch, heads, seq, head_dim = x.shape
        
        # ヘッド間相関
        # エンタングルメント行列を正規化
        entangle_matrix = torch.softmax(self.head_entangle, dim=-1)
        
        # ヘッド混合: 各ヘッドに他のヘッドの情報を注入
        x_flat = x.permute(0, 2, 3, 1)  # (batch, seq, head_dim, heads)
        x_entangled = torch.matmul(x_flat, entangle_matrix.t())  # (batch, seq, head_dim, heads)
        x_entangled = x_entangled.permute(0, 3, 1, 2)  # (batch, heads, seq, head_dim)
        
        # λで重み付け
        lambda_h = torch.sigmoid(self.lambda_heads)
        return x + lambda_h * (x_entangled - x)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with FlashAttention + QBNN
        
        Args:
            x: (batch, seq_len, embed_dim)
            attn_mask: optional attention mask
            is_causal: use causal masking
        
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V 投影
        q = self._reshape_for_attention(self.q_proj(x))
        k = self._reshape_for_attention(self.k_proj(x))
        v = self._reshape_for_attention(self.v_proj(x))
        
        # ヘッド間エンタングルメント（Q, K, Vに適用）
        q = self._apply_head_entanglement(q)
        k = self._apply_head_entanglement(k)
        
        # FlashAttention (PyTorch 2.0+)
        # scaled_dot_product_attention は自動的にFlashAttentionを使用
        try:
            # is_causal=True で因果マスキング
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        except Exception:
            # フォールバック: 手動アテンション計算
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # 量子位相変調
            attn_weights = self._apply_quantum_phase(attn_weights)
            
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, v)
        
        # (batch, heads, seq, head_dim) -> (batch, seq, embed)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)


# ========================================================================
# QBNN Feed Forward
# ========================================================================

class QBNNFeedForward(nn.Module):
    """
    QBNN Feed Forward Network
    
    量子もつれを組み込んだFFN:
    - SwiGLU活性化（LLaMA style）
    - QBNN補正項
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        lambda_min: float = 0.2,
        lambda_max: float = 0.5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # SwiGLU: gate * silu(x)
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # QBNN補正
        self.qbnn_entangle = QBNNEntanglement(hidden_dim, lambda_min, lambda_max)
        
        # 隠れ状態を保持
        self.register_buffer('prev_hidden', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU + QBNN
        """
        # SwiGLU: silu(w1(x)) * w3(x)
        hidden = F.silu(self.w1(x)) * self.w3(x)
        
        # QBNN補正（前の隠れ状態があれば）
        if self.prev_hidden is not None and self.prev_hidden.shape == hidden.shape:
            hidden = self.qbnn_entangle(hidden, self.prev_hidden)
        
        # 現在の隠れ状態を保存
        self.prev_hidden = hidden.detach()
        
        output = self.w2(hidden)
        return self.dropout(output)


# ========================================================================
# QBNN Transformer Block
# ========================================================================

class QBNNTransformerBlock(nn.Module):
    """
    QBNN Transformer Block
    
    構成:
    1. QBNN Flash Attention (Pre-Norm)
    2. QBNN Feed Forward (Pre-Norm)
    3. 層間エンタングルメント
    """
    
    def __init__(self, config: QBNNFlashConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Layer Normalization (Pre-Norm style)
        self.attn_norm = nn.LayerNorm(config.embed_dim)
        self.ffn_norm = nn.LayerNorm(config.embed_dim)
        
        # QBNN Flash Attention
        self.attention = QBNNFlashAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            lambda_min=config.lambda_min,
            lambda_max=config.lambda_max,
            entangle_heads=config.entangle_heads
        )
        
        # QBNN Feed Forward
        self.ffn = QBNNFeedForward(
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            lambda_min=config.lambda_min,
            lambda_max=config.lambda_max
        )
        
        # 層間エンタングルメント
        self.layer_entangle = QBNNEntanglement(
            config.embed_dim,
            config.lambda_min,
            config.lambda_max
        )
        
        # 前層の出力を保持
        self.register_buffer('prev_layer_output', None)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        prev_layer_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, embed_dim)
            attn_mask: attention mask
            is_causal: use causal masking
            prev_layer_output: 前層の出力（層間エンタングル用）
        
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        # 層間エンタングルメント
        if prev_layer_output is not None:
            x = self.layer_entangle(x, prev_layer_output)
        
        # Self-Attention (Pre-Norm + Residual)
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, attn_mask=attn_mask, is_causal=is_causal)
        x = residual + x
        
        # FFN (Pre-Norm + Residual)
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


# ========================================================================
# QBNN Flash Transformer (完全モデル)
# ========================================================================

class QBNNFlashTransformer(nn.Module):
    """
    QBNN Flash Transformer
    
    完全なトランスフォーマーモデル:
    - Token Embedding + Quantum Phase Encoding
    - N x QBNN Transformer Blocks
    - Output Head
    """
    
    def __init__(self, config: QBNNFlashConfig):
        super().__init__()
        self.config = config
        
        # Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Quantum Phase Position Encoding
        if config.use_quantum_phase:
            self.pos_encoding = QuantumPhaseEncoding(config.embed_dim, config.max_seq_len)
        else:
            self.pos_encoding = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        # Embedding Dropout
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            QBNNTransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Output
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Weight tying (embedding and output)
        self.output_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # 統計
        self._print_model_info()
    
    def _init_weights(self, module: nn.Module):
        """重み初期化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _print_model_info(self):
        """モデル情報を表示"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📊 QBNN Flash Transformer 構成:")
        print(f"   語彙サイズ: {self.config.vocab_size:,}")
        print(f"   埋め込み次元: {self.config.embed_dim}")
        print(f"   隠れ層次元: {self.config.hidden_dim}")
        print(f"   アテンションヘッド: {self.config.num_heads}")
        print(f"   トランスフォーマー層: {self.config.num_layers}")
        print(f"   量子もつれ強度: [{self.config.lambda_min}, {self.config.lambda_max}]")
        print(f"   総パラメータ数: {total_params:,}")
        print(f"   学習可能パラメータ: {trainable_params:,}")
        print()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attn_mask: optional attention mask
            is_causal: use causal masking (default: True for language modeling)
            return_hidden: return hidden states instead of logits
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Position encoding
        if self.config.use_quantum_phase:
            x = self.pos_encoding(x)
        else:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_encoding(positions)
        
        x = self.embed_dropout(x)
        
        # Transformer layers with inter-layer entanglement
        prev_output = None
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, is_causal=is_causal, prev_layer_output=prev_output)
            prev_output = x.detach()  # 勾配を切断してメモリ節約
        
        # Output
        x = self.final_norm(x)
        
        if return_hidden:
            return x
        
        logits = self.output_head(x)
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> torch.Tensor:
        """
        テキスト生成
        
        Args:
            input_ids: (batch, seq_len) 入力トークンID
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_k: Top-K サンプリング
            top_p: Top-P (Nucleus) サンプリング
            repetition_penalty: 繰り返しペナルティ
        
        Returns:
            generated_ids: (batch, seq_len + max_new_tokens)
        """
        self.eval()
        device = input_ids.device
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # 最大長を超えないように
            if generated.size(1) >= self.config.max_seq_len:
                # スライディングウィンドウ
                context = generated[:, -self.config.max_seq_len:]
            else:
                context = generated
            
            # Forward
            logits = self.forward(context, is_causal=True)
            next_logits = logits[:, -1, :] / temperature
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(generated.size(0)):
                    for token_id in set(generated[i].tolist()):
                        next_logits[i, token_id] /= repetition_penalty
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-P (Nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # EOS check (assuming token_id=1 is EOS)
            if (next_token == 1).all():
                break
        
        return generated


# ========================================================================
# ユーティリティ関数
# ========================================================================

def create_qbnn_flash_transformer(
    vocab_size: int = 8000,
    embed_dim: int = 256,
    hidden_dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    max_seq_len: int = 512,
    **kwargs
) -> QBNNFlashTransformer:
    """
    QBNN Flash Transformerを作成するヘルパー関数
    """
    config = QBNNFlashConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        **kwargs
    )
    return QBNNFlashTransformer(config)


def test_model():
    """テスト用関数"""
    print("\n🧪 QBNN Flash Transformer テスト\n")
    
    # 小さなモデルを作成
    config = QBNNFlashConfig(
        vocab_size=1000,
        embed_dim=64,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=128
    )
    
    model = QBNNFlashTransformer(config)
    
    # テスト入力
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    print("📊 Forward pass テスト...")
    logits = model(input_ids)
    print(f"   入力形状: {input_ids.shape}")
    print(f"   出力形状: {logits.shape}")
    print(f"   期待形状: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "出力形状が不正です"
    print("   ✅ Forward pass 成功!\n")
    
    # Generate test
    print("📊 Generate テスト...")
    generated = model.generate(input_ids[:1, :5], max_new_tokens=10)
    print(f"   入力長: 5")
    print(f"   生成後長: {generated.shape[1]}")
    print("   ✅ Generate 成功!\n")
    
    print("=" * 70)
    print("✅ すべてのテストに成功しました!")
    print("=" * 70)
    
    return model


# ========================================================================
# メイン
# ========================================================================

if __name__ == "__main__":
    test_model()
