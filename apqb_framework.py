#!/usr/bin/env python3
"""
APQBフレームワーク - ニューラルネットワークと量子多体系の統一理論

論文: ニューラルネットワークと量子多体系の統一理論としてのAPQBフレームワーク

このモジュールは以下を実装:
1. 単一APQBの数学的定義
2. N体系への一般化と多体相関
3. APQBニューラルネットワーク
4. 複素角度空間での学習
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import OrderedDict

# ========================================================================
# 1. 単一APQBの定義と基本関係式
# ========================================================================

class APQB:
    """
    調整可能擬似量子ビット (Adjustable Pseudo Quantum Bit)
    
    単一パラメータ θ で以下を統一的に記述:
    - 量子状態: |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
    - 相関係数: r = cos(2θ)
    - 温度（乱雑さ）: T = |sin(2θ)|
    
    制約条件: r² + T² = 1
    """
    
    def __init__(self, theta: float = np.pi/4):
        """
        Args:
            theta: 内部パラメータ角度 (0 ≤ θ ≤ π/2)
        """
        self.theta = np.clip(theta, 0, np.pi/2)
    
    @property
    def state(self) -> np.ndarray:
        """量子状態ベクトル |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩"""
        return np.array([np.cos(self.theta), np.sin(self.theta)])
    
    @property
    def r(self) -> float:
        """統計的相関係数: r = cos(2θ)"""
        return np.cos(2 * self.theta)
    
    @property
    def T(self) -> float:
        """温度（乱雑さ）: T = |sin(2θ)|"""
        return np.abs(np.sin(2 * self.theta))
    
    @property
    def z(self) -> complex:
        """複素表現: z = e^{i2θ}"""
        return np.exp(2j * self.theta)
    
    @property
    def p0(self) -> float:
        """状態|0⟩の観測確率: P(0) = cos²(θ)"""
        return np.cos(self.theta) ** 2
    
    @property
    def p1(self) -> float:
        """状態|1⟩の観測確率: P(1) = sin²(θ)"""
        return np.sin(self.theta) ** 2
    
    def measure(self) -> int:
        """量子測定: 確率的に0または1を返す"""
        return 1 if np.random.random() < self.p1 else 0
    
    def verify_constraint(self) -> float:
        """制約条件 r² + T² = 1 の検証"""
        return self.r**2 + self.T**2
    
    @classmethod
    def from_r(cls, r: float) -> 'APQB':
        """相関係数 r からAPQBを生成"""
        r = np.clip(r, -1, 1)
        theta = np.arccos(r) / 2
        return cls(theta)
    
    @classmethod
    def from_T(cls, T: float) -> 'APQB':
        """温度 T からAPQBを生成"""
        T = np.clip(T, 0, 1)
        theta = np.arcsin(T) / 2
        return cls(theta)
    
    def __repr__(self):
        return f"APQB(θ={self.theta:.4f}, r={self.r:.4f}, T={self.T:.4f})"


# ========================================================================
# 2. N体系への一般化と多体相関
# ========================================================================

class APQBMultiBody:
    """
    N量子ビットAPQB系と多体相関
    
    k体相関関数:
    - Q_k(θ) = cos(2kθ) (k偶数)
    - Q_k(θ) = sin(2kθ) (k奇数)
    
    一般化トレードオフ関係:
    C_n² + Σ_{k=2}^{n} s_k w_k Q_k(θ)² = W_0
    """
    
    def __init__(self, n: int, theta: float = np.pi/4):
        """
        Args:
            n: 量子ビット数
            theta: 内部パラメータ角度
        """
        self.n = n
        self.theta = theta
        self.apqbs = [APQB(theta) for _ in range(n)]
    
    def Q_k(self, k: int) -> float:
        """
        k体相関関数
        
        Q_k(θ) = cos(2kθ) if k is even
        Q_k(θ) = sin(2kθ) if k is odd
        """
        if k % 2 == 0:
            return np.cos(2 * k * self.theta)
        else:
            return np.sin(2 * k * self.theta)
    
    def get_all_correlations(self) -> dict:
        """全ての多体相関を取得"""
        correlations = {}
        for k in range(1, self.n + 1):
            correlations[f'Q_{k}'] = self.Q_k(k)
        return correlations
    
    def complex_polynomial_coefficients(self) -> np.ndarray:
        """
        複素多項式の係数を計算
        F(θ) = Σ A_k z^k where z = e^{i2θ}
        """
        z = np.exp(2j * self.theta)
        coefficients = np.zeros(self.n + 1, dtype=complex)
        for k in range(1, self.n + 1):
            # A_k は Q_k の複素表現から導出
            coefficients[k] = z ** k
        return coefficients
    
    def entanglement_measure(self) -> float:
        """
        エンタングルメント測度 C_n
        (簡略化モデル: 全体の量子相関の強さ)
        """
        # 2体以上の相関の二乗和から計算
        C_sq = 0
        for k in range(2, self.n + 1):
            C_sq += self.Q_k(k) ** 2 / k
        return np.sqrt(C_sq)
    
    def verify_tradeoff(self) -> dict:
        """
        一般化トレードオフ関係式の検証
        n=2: C² + T² = 1 (円)
        n=3: C² + αT² - βτ² = α (双曲面)
        """
        if self.n == 2:
            C = self.entanglement_measure()
            T = self.apqbs[0].T
            result = C**2 + T**2
            return {'type': 'circle', 'value': result, 'expected': 1.0}
        elif self.n == 3:
            C = self.entanglement_measure()
            T = self.Q_k(1)  # sin(2θ)
            tau = self.Q_k(3)  # sin(6θ)
            # 双曲面形式
            alpha, beta = 0.5, 0.3  # モデルパラメータ
            result = C**2 + alpha * T**2 - beta * tau**2
            return {'type': 'hyperboloid', 'value': result}
        else:
            return {'type': f'higher_dim_{self.n}', 'note': 'Complex structure'}


# ========================================================================
# 3. APQBニューラルネットワーク
# ========================================================================

class APQBLayer(nn.Module):
    """
    APQBベースのニューラルネットワーク層
    
    従来のNN: f(x) = Σ w_i x_i + Σ w_ij x_i x_j + ...
    APQB NN:  f(x) = Σ A_k z^k (複素角度空間)
    
    重み空間の次元: 2^n → O(n) (指数的削減)
    """
    
    def __init__(self, input_dim: int, output_dim: int, max_order: int = None):
        """
        Args:
            input_dim: 入力次元
            output_dim: 出力次元
            max_order: 多体相関の最大次数 (default: input_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_order = max_order or min(input_dim, 8)
        
        # 角度パラメータ θ (学習可能)
        # 各出力ニューロンに対して1つの θ
        self.theta = nn.Parameter(torch.rand(output_dim) * np.pi / 2)
        
        # 各次数の重み (複素数として扱うため実部と虚部)
        self.A_real = nn.Parameter(torch.randn(output_dim, self.max_order + 1) * 0.1)
        self.A_imag = nn.Parameter(torch.randn(output_dim, self.max_order + 1) * 0.1)
        
        # 入力埋め込み
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # バイアス
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        1. 入力 x を角度空間に写像
        2. 複素多項式 F(θ) = Σ A_k z^k を計算
        3. 実部を出力として返す
        """
        batch_size = x.shape[0]
        
        # 入力を投影
        x_proj = self.input_proj(x)  # (batch, output_dim)
        
        # θ を [0, π/2] に制約
        theta = torch.sigmoid(self.theta) * np.pi / 2
        
        # 複素数 z = e^{i2θ}
        z_real = torch.cos(2 * theta)
        z_imag = torch.sin(2 * theta)
        
        # 複素多項式を計算: F(θ) = Σ A_k z^k
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        z_k_real = torch.ones(self.output_dim, device=x.device)
        z_k_imag = torch.zeros(self.output_dim, device=x.device)
        
        for k in range(self.max_order + 1):
            # A_k (複素数)
            A_k_real = self.A_real[:, k]
            A_k_imag = self.A_imag[:, k]
            
            # A_k * z^k の実部
            term_real = A_k_real * z_k_real - A_k_imag * z_k_imag
            
            # 入力との相互作用
            output += x_proj * term_real.unsqueeze(0)
            
            # z^{k+1} = z^k * z
            new_real = z_k_real * z_real - z_k_imag * z_imag
            new_imag = z_k_real * z_imag + z_k_imag * z_real
            z_k_real = new_real
            z_k_imag = new_imag
        
        return output + self.bias
    
    def get_r_and_T(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """相関係数 r と温度 T を取得"""
        theta = torch.sigmoid(self.theta) * np.pi / 2
        r = torch.cos(2 * theta)
        T = torch.abs(torch.sin(2 * theta))
        return r, T


class APQBNeuralNetwork(nn.Module):
    """
    APQBニューラルネットワーク
    
    論文で提案された新しい計算モデル:
    - 重み空間: R^{2^n} → 複素角度空間
    - 自然な正則化: r² + T² = 1
    - 物理的解釈可能性
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'gelu', max_order: int = 4):
        """
        Args:
            input_dim: 入力次元
            hidden_dims: 隠れ層の次元リスト
            output_dim: 出力次元
            activation: 活性化関数 ('gelu', 'relu', 'tanh')
            max_order: 多体相関の最大次数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.max_order = max_order
        
        # 活性化関数
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU()
        }
        self.activation = activations.get(activation, nn.GELU())
        
        # ネットワーク層の構築
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append((f'apqb_{i}', APQBLayer(prev_dim, hidden_dim, max_order)))
            layers.append((f'act_{i}', self.activation))
            layers.append((f'norm_{i}', nn.LayerNorm(hidden_dim)))
            prev_dim = hidden_dim
        
        # 出力層
        layers.append(('output', APQBLayer(prev_dim, output_dim, max_order)))
        
        self.layers = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def get_quantum_statistics(self) -> dict:
        """量子統計情報を取得"""
        stats = {}
        for name, module in self.layers.named_modules():
            if isinstance(module, APQBLayer):
                r, T = module.get_r_and_T()
                stats[name] = {
                    'r_mean': r.mean().item(),
                    'r_std': r.std().item(),
                    'T_mean': T.mean().item(),
                    'T_std': T.std().item(),
                    'constraint': (r**2 + T**2).mean().item()  # Should be ~1
                }
        return stats
    
    def regularization_loss(self) -> torch.Tensor:
        """
        幾何学的制約に基づく正則化損失
        r² + T² = 1 からの逸脱をペナルティ
        """
        loss = 0.0
        for module in self.modules():
            if isinstance(module, APQBLayer):
                r, T = module.get_r_and_T()
                # 制約条件からの逸脱
                constraint_violation = (r**2 + T**2 - 1).abs().mean()
                loss += constraint_violation
        return loss


# ========================================================================
# 4. 複素角度空間での学習
# ========================================================================

class APQBOptimizer:
    """
    APQB用最適化器
    
    角度空間での勾配降下法:
    - θ の更新は周期境界条件を考慮
    - 自然な正則化の活用
    """
    
    def __init__(self, model: APQBNeuralNetwork, lr: float = 0.01,
                 reg_weight: float = 0.1):
        self.model = model
        self.lr = lr
        self.reg_weight = reg_weight
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    def step(self, loss: torch.Tensor) -> float:
        """最適化ステップ"""
        # 正則化損失を追加
        reg_loss = self.model.regularization_loss()
        total_loss = loss + self.reg_weight * reg_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return total_loss.item()


# ========================================================================
# 5. デモンストレーション
# ========================================================================

def demo_single_apqb():
    """単一APQBのデモ"""
    print("=" * 60)
    print("🔬 単一APQBのデモ")
    print("=" * 60)
    
    # 様々な θ でAPQBを生成
    print("\n📊 θ による状態変化:")
    print("-" * 50)
    print(f"{'θ':^10} {'r':^10} {'T':^10} {'r²+T²':^10}")
    print("-" * 50)
    
    for theta in np.linspace(0, np.pi/2, 9):
        apqb = APQB(theta)
        print(f"{theta:^10.4f} {apqb.r:^10.4f} {apqb.T:^10.4f} {apqb.verify_constraint():^10.4f}")
    
    print("\n🎲 量子測定シミュレーション (θ=π/4):")
    apqb = APQB(np.pi/4)
    measurements = [apqb.measure() for _ in range(1000)]
    print(f"   P(0) 理論値: {apqb.p0:.4f}, 実測値: {1 - np.mean(measurements):.4f}")
    print(f"   P(1) 理論値: {apqb.p1:.4f}, 実測値: {np.mean(measurements):.4f}")


def demo_multibody():
    """多体APQBのデモ"""
    print("\n" + "=" * 60)
    print("🔬 多体APQBのデモ")
    print("=" * 60)
    
    # N=2, 3, 4 の多体系
    for n in [2, 3, 4]:
        print(f"\n📊 N={n} 量子ビット系:")
        system = APQBMultiBody(n, theta=np.pi/6)
        
        print(f"   多体相関: {system.get_all_correlations()}")
        print(f"   エンタングルメント C_{n}: {system.entanglement_measure():.4f}")
        
        tradeoff = system.verify_tradeoff()
        print(f"   トレードオフ幾何: {tradeoff['type']}")


def demo_apqb_nn():
    """APQBニューラルネットワークのデモ"""
    print("\n" + "=" * 60)
    print("🧠 APQBニューラルネットワークのデモ")
    print("=" * 60)
    
    # モデル構築
    model = APQBNeuralNetwork(
        input_dim=10,
        hidden_dims=[32, 16],
        output_dim=2,
        activation='gelu',
        max_order=4
    )
    
    # パラメータ数の比較
    total_params = sum(p.numel() for p in model.parameters())
    traditional_params = 10 * 32 + 32 + 32 * 16 + 16 + 16 * 2 + 2  # 従来のNN
    
    print(f"\n📊 パラメータ比較:")
    print(f"   APQB NN パラメータ数: {total_params:,}")
    print(f"   従来の NN パラメータ数（同等構造）: {traditional_params:,}")
    
    # 順伝播テスト
    x = torch.randn(4, 10)
    y = model(x)
    print(f"\n🔄 順伝播:")
    print(f"   入力形状: {x.shape}")
    print(f"   出力形状: {y.shape}")
    
    # 量子統計
    stats = model.get_quantum_statistics()
    print(f"\n⚛️ 量子統計:")
    for layer_name, layer_stats in stats.items():
        print(f"   {layer_name}:")
        print(f"      r: {layer_stats['r_mean']:.4f} ± {layer_stats['r_std']:.4f}")
        print(f"      T: {layer_stats['T_mean']:.4f} ± {layer_stats['T_std']:.4f}")
        print(f"      制約 r²+T²: {layer_stats['constraint']:.4f}")


def demo_classification():
    """分類タスクのデモ"""
    print("\n" + "=" * 60)
    print("🎯 XOR問題の学習デモ")
    print("=" * 60)
    
    # XORデータ
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    # モデル
    model = APQBNeuralNetwork(
        input_dim=2,
        hidden_dims=[8, 4],
        output_dim=1,
        activation='tanh',
        max_order=3
    )
    
    optimizer = APQBOptimizer(model, lr=0.1, reg_weight=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    print("\n📚 学習中...")
    for epoch in range(200):
        output = model(X)
        loss = criterion(output, y)
        total_loss = optimizer.step(loss)
        
        if (epoch + 1) % 50 == 0:
            pred = (torch.sigmoid(output) > 0.5).float()
            acc = (pred == y).float().mean()
            print(f"   Epoch {epoch+1}: Loss={total_loss:.4f}, Acc={acc:.2%}")
    
    # 最終結果
    print("\n🎯 最終予測:")
    with torch.no_grad():
        pred = torch.sigmoid(model(X))
        for i in range(4):
            print(f"   入力 {X[i].tolist()} → 予測 {pred[i].item():.4f} (正解: {y[i].item()})")


def visualize_apqb_geometry():
    """APQBの幾何学を可視化"""
    print("\n" + "=" * 60)
    print("📈 APQBの幾何学的構造を可視化")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. r-T トレードオフ（円）
    theta_range = np.linspace(0, np.pi/2, 100)
    r_vals = np.cos(2 * theta_range)
    T_vals = np.abs(np.sin(2 * theta_range))
    
    axes[0].plot(r_vals, T_vals, 'b-', linewidth=2)
    axes[0].set_xlabel('r (相関係数)', fontsize=12)
    axes[0].set_ylabel('T (温度)', fontsize=12)
    axes[0].set_title('r² + T² = 1 (制約曲面)', fontsize=14)
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # 2. 多体相関 Q_k(θ)
    for k in range(1, 5):
        if k % 2 == 0:
            Q_k = np.cos(2 * k * theta_range)
        else:
            Q_k = np.sin(2 * k * theta_range)
        axes[1].plot(theta_range, Q_k, label=f'Q_{k}(θ)', linewidth=2)
    
    axes[1].set_xlabel('θ', fontsize=12)
    axes[1].set_ylabel('Q_k(θ)', fontsize=12)
    axes[1].set_title('多体相関関数', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 複素平面上の z = e^{i2θ}
    z = np.exp(2j * theta_range)
    axes[2].plot(z.real, z.imag, 'g-', linewidth=2)
    axes[2].scatter([1], [0], color='red', s=100, zorder=5, label='θ=0')
    axes[2].scatter([0], [1], color='blue', s=100, zorder=5, label='θ=π/4')
    axes[2].scatter([-1], [0], color='green', s=100, zorder=5, label='θ=π/2')
    
    axes[2].set_xlabel('Re(z)', fontsize=12)
    axes[2].set_ylabel('Im(z)', fontsize=12)
    axes[2].set_title('複素平面上の z = e^{i2θ}', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/Users/yuyahiguchi/Program/Qubit/apqb_geometry.png', dpi=150)
    print("✅ 可視化を apqb_geometry.png に保存しました")
    plt.close()


def main():
    """メイン関数"""
    print("=" * 60)
    print("🌟 APQBフレームワーク")
    print("   ニューラルネットワークと量子多体系の統一理論")
    print("=" * 60)
    
    demo_single_apqb()
    demo_multibody()
    demo_apqb_nn()
    demo_classification()
    visualize_apqb_geometry()
    
    print("\n" + "=" * 60)
    print("✅ 全デモ完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()

