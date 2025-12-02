#!/usr/bin/env python3
"""
Quantum Bit Neural Network (QBNN) シミュレーション

論文「ニューラルネットワークと量子多体系の統一理論としてのAPQBフレームワーク」に基づく

主要な特性:
1. 複素角度空間: z = e^{i2θ} でのパラメータ化
2. 幾何学的制約: r² + T² = 1 による自然な正則化
3. 多体相関: Q_k(θ) による高次特徴量
4. 構造的同型性: NN多項式 ≅ APQB複素多項式
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🧠⚛️ Quantum Bit Neural Network (QBNN) シミュレーション")
print("   論文: ニューラルネットワークと量子多体系の統一理論")
print("=" * 70)

# ========================================================================
# 1. QBNN コア - 論文の数学的定義
# ========================================================================

class QuantumBit:
    """
    量子ビット（論文のAPQB）
    
    単一パラメータ θ で以下を統一的に記述:
    - 量子状態: |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
    - 相関係数: r = cos(2θ)
    - 温度: T = |sin(2θ)|
    - 複素数: z = e^{i2θ}
    """
    
    def __init__(self, theta=np.pi/4):
        self.theta = np.clip(theta, 0, np.pi/2)
    
    @property
    def state(self):
        """量子状態 [cos(θ), sin(θ)]"""
        return np.array([np.cos(self.theta), np.sin(self.theta)])
    
    @property
    def r(self):
        """相関係数 r = cos(2θ)"""
        return np.cos(2 * self.theta)
    
    @property
    def T(self):
        """温度 T = |sin(2θ)|"""
        return np.abs(np.sin(2 * self.theta))
    
    @property
    def z(self):
        """複素数 z = e^{i2θ}"""
        return np.exp(2j * self.theta)
    
    @property
    def bloch_coords(self):
        """Bloch球上の座標 (x, y, z)"""
        return (
            np.sin(2 * self.theta),  # x
            0,                        # y
            np.cos(2 * self.theta)   # z = r
        )
    
    def Q_k(self, k):
        """k体相関"""
        if k % 2 == 0:
            return np.cos(2 * k * self.theta)
        else:
            return np.sin(2 * k * self.theta)
    
    def measure(self):
        """量子測定"""
        p1 = np.sin(self.theta) ** 2
        return 1 if np.random.random() < p1 else 0
    
    def constraint(self):
        """r² + T² = 1 の検証"""
        return self.r**2 + self.T**2


class QBNNNeuron:
    """
    QBNNニューロン
    
    論文の複素多項式: F(θ) = Σ A_k z^k
    """
    
    def __init__(self, input_dim, max_order=4):
        self.input_dim = input_dim
        self.max_order = max_order
        
        # 量子ビット（内部パラメータ θ）
        self.qbit = QuantumBit(np.random.rand() * np.pi / 2)
        
        # 複素係数 A_k = A_real + i * A_imag
        self.A_real = np.random.randn(max_order + 1) * 0.1
        self.A_imag = np.random.randn(max_order + 1) * 0.1
        
        # 入力重み
        self.weights = np.random.randn(input_dim) * 0.1
        self.bias = 0.0
    
    def forward(self, x):
        """
        順伝播: 複素多項式による変換
        
        F(θ) = Σ A_k z^k (実部)
        """
        # 入力の重み付き和
        pre_activation = np.dot(x, self.weights) + self.bias
        
        # θ を更新（入力に応じて）
        theta = (np.tanh(pre_activation) + 1) * np.pi / 4
        self.qbit.theta = theta
        
        # 複素多項式計算
        z = self.qbit.z
        result = 0.0
        z_k = 1.0
        
        for k in range(self.max_order + 1):
            A_k = self.A_real[k] + 1j * self.A_imag[k]
            result += (A_k * z_k).real
            z_k *= z
        
        return result
    
    def get_quantum_state(self):
        """量子状態を取得"""
        return {
            'theta': self.qbit.theta,
            'r': self.qbit.r,
            'T': self.qbit.T,
            'z': self.qbit.z,
            'constraint': self.qbit.constraint()
        }


class QBNNLayer:
    """QBNNの層"""
    
    def __init__(self, input_dim, output_dim, max_order=4):
        self.neurons = [QBNNNeuron(input_dim, max_order) for _ in range(output_dim)]
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x):
        return np.array([n.forward(x) for n in self.neurons])
    
    def get_layer_state(self):
        return [n.get_quantum_state() for n in self.neurons]


class QBNN:
    """
    Quantum Bit Neural Network
    
    論文の構造的同型性:
    NN多項式展開 ≅ APQB複素多項式 F(θ) = Σ A_k z^k
    """
    
    def __init__(self, architecture, max_order=4):
        """
        Args:
            architecture: [input_dim, hidden1, hidden2, ..., output_dim]
            max_order: 多体相関の最大次数
        """
        self.architecture = architecture
        self.max_order = max_order
        
        # 層の構築
        self.layers = []
        for i in range(len(architecture) - 1):
            layer = QBNNLayer(architecture[i], architecture[i+1], max_order)
            self.layers.append(layer)
    
    def forward(self, x):
        """順伝播"""
        for layer in self.layers:
            x = layer.forward(x)
            x = np.tanh(x)  # 活性化
        return x
    
    def get_network_state(self):
        """全ネットワークの量子状態"""
        return [layer.get_layer_state() for layer in self.layers]
    
    def total_constraint(self):
        """全ニューロンの制約条件"""
        constraints = []
        for layer in self.layers:
            for neuron in layer.neurons:
                constraints.append(neuron.qbit.constraint())
        return np.mean(constraints)


# ========================================================================
# 2. 可視化関数
# ========================================================================

def plot_quantum_neuron(neuron, ax, title="Quantum Neuron"):
    """単一量子ニューロンの可視化"""
    state = neuron.get_quantum_state()
    
    # Bloch球
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x, y, z, alpha=0.1, color='gray')
    
    # 状態ベクトル
    bx, by, bz = neuron.qbit.bloch_coords
    ax.quiver(0, 0, 0, bx, by, bz, color='red', arrow_length_ratio=0.1, linewidth=2)
    ax.scatter([bx], [by], [bz], color='red', s=100, zorder=5)
    
    # 軸
    ax.plot([-1, 1], [0, 0], [0, 0], 'k--', alpha=0.3)
    ax.plot([0, 0], [-1, 1], [0, 0], 'k--', alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1, 1], 'k--', alpha=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (r)')
    ax.set_title(f'{title}\nθ={state["theta"]:.2f}, r={state["r"]:.2f}, T={state["T"]:.2f}\nr²+T²={state["constraint"]:.4f}')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])


def plot_complex_polynomial(neuron, ax, title="Complex Polynomial"):
    """複素多項式の可視化"""
    # θ の範囲で F(θ) を計算
    thetas = np.linspace(0, np.pi/2, 100)
    F_values = []
    
    for theta in thetas:
        z = np.exp(2j * theta)
        result = 0.0
        z_k = 1.0
        for k in range(neuron.max_order + 1):
            A_k = neuron.A_real[k] + 1j * neuron.A_imag[k]
            result += A_k * z_k
            z_k *= z
        F_values.append(result)
    
    F_values = np.array(F_values)
    
    # 実部と虚部
    ax.plot(thetas, F_values.real, 'b-', label='Re(F)', linewidth=2)
    ax.plot(thetas, F_values.imag, 'r--', label='Im(F)', linewidth=2)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('θ')
    ax.set_ylabel('F(θ)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_multibody_correlations(neuron, ax, title="Multi-body Correlations"):
    """多体相関の可視化"""
    thetas = np.linspace(0, np.pi/2, 100)
    
    for k in range(1, neuron.max_order + 1):
        Q_k = []
        for theta in thetas:
            if k % 2 == 0:
                Q_k.append(np.cos(2 * k * theta))
            else:
                Q_k.append(np.sin(2 * k * theta))
        
        ax.plot(thetas, Q_k, label=f'Q_{k}(θ)', linewidth=2)
    
    # 現在の θ をマーク
    ax.axvline(neuron.qbit.theta, color='black', linestyle='--', alpha=0.5, label='Current θ')
    
    ax.set_xlabel('θ')
    ax.set_ylabel('Q_k(θ)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_constraint_surface(ax, title="Constraint: r² + T² = 1"):
    """幾何学的制約の可視化"""
    theta = np.linspace(0, np.pi/2, 100)
    r = np.cos(2 * theta)
    T = np.abs(np.sin(2 * theta))
    
    ax.plot(r, T, 'b-', linewidth=2)
    ax.fill_between(r, 0, T, alpha=0.2)
    
    # 単位円
    circle_theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), 'k--', alpha=0.3)
    
    ax.set_xlabel('r (Correlation)')
    ax.set_ylabel('T (Temperature/Entropy)')
    ax.set_title(title)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def visualize_qbnn(qbnn, save_path=None):
    """QBNNの完全な可視化"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. ネットワーク構造
    ax1 = fig.add_subplot(2, 3, 1)
    draw_network_structure(qbnn, ax1)
    
    # 2. 最初のニューロンのBloch球
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_quantum_neuron(qbnn.layers[0].neurons[0], ax2, "Layer 0, Neuron 0")
    
    # 3. 複素多項式
    ax3 = fig.add_subplot(2, 3, 3)
    plot_complex_polynomial(qbnn.layers[0].neurons[0], ax3, "Complex Polynomial F(θ)")
    
    # 4. 多体相関
    ax4 = fig.add_subplot(2, 3, 4)
    plot_multibody_correlations(qbnn.layers[0].neurons[0], ax4, "Multi-body Correlations Q_k(θ)")
    
    # 5. 幾何学的制約
    ax5 = fig.add_subplot(2, 3, 5)
    plot_constraint_surface(ax5, "Geometric Constraint")
    
    # ネットワーク上の全ニューロンをマーク
    for layer in qbnn.layers:
        for neuron in layer.neurons:
            r, T = neuron.qbit.r, neuron.qbit.T
            ax5.scatter([r], [T], s=50, zorder=5)
    
    # 6. 量子状態の統計
    ax6 = fig.add_subplot(2, 3, 6)
    plot_network_statistics(qbnn, ax6)
    
    plt.suptitle('Quantum Bit Neural Network (QBNN) Visualization\n'
                 f'Architecture: {qbnn.architecture}, Max Order: {qbnn.max_order}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   保存: {save_path}")
    
    plt.close()
    return fig


def draw_network_structure(qbnn, ax):
    """ネットワーク構造を描画"""
    arch = qbnn.architecture
    max_neurons = max(arch)
    layer_positions = np.linspace(0, 1, len(arch))
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    neuron_positions = []
    
    for layer_idx, num_neurons in enumerate(arch):
        x = layer_positions[layer_idx]
        y_positions = np.linspace(0.2, 0.8, num_neurons) if num_neurons > 1 else [0.5]
        
        layer_pos = []
        for y in y_positions:
            # ニューロンを描画
            if layer_idx < len(qbnn.layers):
                if layer_idx == 0:
                    neuron = qbnn.layers[0].neurons[0] if len(qbnn.layers[0].neurons) > 0 else None
                else:
                    neuron = qbnn.layers[layer_idx].neurons[0] if len(qbnn.layers) > layer_idx else None
                
                if neuron:
                    # 量子状態に応じた色
                    color = plt.cm.RdBu((neuron.qbit.r + 1) / 2)
                else:
                    color = 'lightblue'
            else:
                color = 'lightgreen'
            
            circle = Circle((x, y), 0.03, color=color, ec='black', linewidth=1.5)
            ax.add_patch(circle)
            layer_pos.append((x, y))
        
        neuron_positions.append(layer_pos)
    
    # 接続を描画
    for i in range(len(neuron_positions) - 1):
        for pos1 in neuron_positions[i]:
            for pos2 in neuron_positions[i + 1]:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'gray', alpha=0.3, linewidth=0.5)
    
    # レイヤーラベル
    for i, x in enumerate(layer_positions):
        if i == 0:
            label = f'Input\n({arch[i]})'
        elif i == len(arch) - 1:
            label = f'Output\n({arch[i]})'
        else:
            label = f'Hidden {i}\n({arch[i]})'
        ax.text(x, -0.05, label, ha='center', fontsize=9)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('QBNN Structure\n(Color = r value)')


def plot_network_statistics(qbnn, ax):
    """ネットワークの統計情報"""
    states = qbnn.get_network_state()
    
    all_r = []
    all_T = []
    all_constraints = []
    
    for layer_states in states:
        for state in layer_states:
            all_r.append(state['r'])
            all_T.append(state['T'])
            all_constraints.append(state['constraint'])
    
    # ヒストグラム
    ax.hist(all_r, bins=20, alpha=0.5, label=f'r (mean={np.mean(all_r):.3f})', color='blue')
    ax.hist(all_T, bins=20, alpha=0.5, label=f'T (mean={np.mean(all_T):.3f})', color='red')
    
    ax.axvline(np.mean(all_r), color='blue', linestyle='--')
    ax.axvline(np.mean(all_T), color='red', linestyle='--')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title(f'Network Statistics\nr²+T² = {np.mean(all_constraints):.4f}')
    ax.legend()


# ========================================================================
# 3. 学習シミュレーション
# ========================================================================

def simulate_xor_learning(qbnn, epochs=100):
    """XOR問題の学習シミュレーション"""
    
    # XORデータ
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    losses = []
    constraints = []
    
    print("\n📚 XOR学習シミュレーション")
    print("-" * 50)
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for x_i, y_i in zip(X, y):
            # 順伝播
            output = qbnn.forward(x_i)[0]
            pred = 1 / (1 + np.exp(-output))  # シグモイド
            
            # 損失
            loss = -(y_i * np.log(pred + 1e-8) + (1 - y_i) * np.log(1 - pred + 1e-8))
            epoch_loss += loss
            
            # 簡易的なパラメータ更新（確率的更新）
            for layer in qbnn.layers:
                for neuron in layer.neurons:
                    # θ を確率的に更新
                    grad = (pred - y_i) * 0.1
                    neuron.qbit.theta = np.clip(
                        neuron.qbit.theta - grad * np.random.randn() * 0.05,
                        0, np.pi/2
                    )
                    # 重みも更新（サイズを考慮）
                    weight_grad = grad * np.random.randn(len(neuron.weights)) * 0.01
                    neuron.weights -= weight_grad
                    neuron.A_real -= grad * np.random.randn(len(neuron.A_real)) * 0.01
        
        losses.append(epoch_loss / 4)
        constraints.append(qbnn.total_constraint())
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}: Loss={losses[-1]:.4f}, r²+T²={constraints[-1]:.4f}")
    
    return losses, constraints


def visualize_learning(losses, constraints, save_path=None):
    """学習過程の可視化"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 損失
    axes[0].plot(losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # 制約条件
    axes[1].plot(constraints, 'r-', linewidth=2)
    axes[1].axhline(1.0, color='green', linestyle='--', label='Target: 1.0')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('r² + T²')
    axes[1].set_title('Geometric Constraint')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('QBNN Learning Dynamics', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   保存: {save_path}")
    
    plt.close()
    return fig


# ========================================================================
# 4. 推論シミュレーション
# ========================================================================

def simulate_inference(qbnn, num_samples=100):
    """推論のシミュレーション（量子測定の統計）"""
    
    print("\n🔮 量子推論シミュレーション")
    print("-" * 50)
    
    # テスト入力
    test_inputs = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1])
    ]
    
    results = []
    
    for x in test_inputs:
        measurements = []
        
        for _ in range(num_samples):
            # 順伝播
            _ = qbnn.forward(x)
            
            # 各ニューロンの量子測定
            layer_measurements = []
            for layer in qbnn.layers:
                for neuron in layer.neurons:
                    m = neuron.qbit.measure()
                    layer_measurements.append(m)
            
            measurements.append(layer_measurements)
        
        measurements = np.array(measurements)
        mean_output = measurements.mean(axis=0)
        
        results.append({
            'input': x,
            'mean_measurements': mean_output,
            'p1': mean_output.mean()
        })
        
        print(f"   入力 {x} → P(1) = {mean_output.mean():.3f}")
    
    return results


# ========================================================================
# 5. メイン実行
# ========================================================================

def main():
    print("\n🏗️ QBNNを構築中...")
    
    # QBNNの作成
    architecture = [2, 4, 4, 1]  # 入力2, 隠れ4x2, 出力1
    qbnn = QBNN(architecture, max_order=4)
    
    print(f"   アーキテクチャ: {architecture}")
    print(f"   総ニューロン数: {sum(architecture[1:])}")
    print(f"   多体相関次数: {qbnn.max_order}")
    print(f"   初期制約 r²+T²: {qbnn.total_constraint():.4f}")
    
    # 1. ネットワーク可視化
    print("\n📊 1. ネットワーク可視化")
    print("-" * 50)
    visualize_qbnn(qbnn, '/Users/yuyahiguchi/Program/Qubit/qbnn_structure.png')
    
    # 2. 学習シミュレーション
    print("\n📊 2. 学習シミュレーション")
    print("-" * 50)
    losses, constraints = simulate_xor_learning(qbnn, epochs=100)
    visualize_learning(losses, constraints, '/Users/yuyahiguchi/Program/Qubit/qbnn_learning.png')
    
    # 3. 学習後の可視化
    print("\n📊 3. 学習後のネットワーク状態")
    print("-" * 50)
    visualize_qbnn(qbnn, '/Users/yuyahiguchi/Program/Qubit/qbnn_trained.png')
    
    # 4. 推論シミュレーション
    print("\n📊 4. 量子推論")
    print("-" * 50)
    results = simulate_inference(qbnn, num_samples=100)
    
    # 5. 詳細な量子状態
    print("\n📊 5. 各ニューロンの量子状態")
    print("-" * 50)
    for layer_idx, layer in enumerate(qbnn.layers):
        print(f"\n   Layer {layer_idx}:")
        for neuron_idx, neuron in enumerate(layer.neurons):
            state = neuron.get_quantum_state()
            print(f"      Neuron {neuron_idx}: θ={state['theta']:.3f}, "
                  f"r={state['r']:.3f}, T={state['T']:.3f}, "
                  f"r²+T²={state['constraint']:.4f}")
    
    # 6. 論文との対応まとめ
    print("\n" + "=" * 70)
    print("📚 論文との対応")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  Quantum Bit Neural Network (QBNN) - 論文の実装                │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. 各ニューロン = 量子ビット（APQB）                          │
    │     - 内部パラメータ θ                                         │
    │     - 量子状態 |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩                    │
    │                                                                 │
    │  2. 活性化 = 複素多項式                                        │
    │     - F(θ) = Σ A_k z^k where z = e^{i2θ}                       │
    │     - 論文: NN多項式 ≅ APQB複素多項式                          │
    │                                                                 │
    │  3. 正則化 = 幾何学的制約                                      │
    │     - r² + T² = 1                                              │
    │     - r = cos(2θ), T = |sin(2θ)|                               │
    │                                                                 │
    │  4. 高次特徴 = 多体相関                                        │
    │     - Q_k(θ) = cos(2kθ) or sin(2kθ)                            │
    │     - 論文: k体相互作用の表現                                  │
    │                                                                 │
    │  5. 出力 = 量子測定                                            │
    │     - 確率的な0/1出力                                          │
    │     - P(1) = sin²(θ)                                           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n✅ シミュレーション完了！")
    print("   生成ファイル:")
    print("   - qbnn_structure.png (初期構造)")
    print("   - qbnn_learning.png (学習曲線)")
    print("   - qbnn_trained.png (学習後構造)")


if __name__ == "__main__":
    main()

