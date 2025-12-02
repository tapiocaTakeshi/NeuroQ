"""
擬似量子ビットの視覚化

ブロッホ球と確率分布を表示
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pseudo_qubit import PseudoQubit


def plot_bloch_sphere(qubit: PseudoQubit, ax: plt.Axes = None) -> plt.Axes:
    """
    ブロッホ球上に量子状態をプロット
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # ブロッホ球を描画
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan')
    
    # 軸を描画
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', alpha=0.3)
    
    # 基底状態をラベル
    ax.text(0, 0, 1.3, '|0⟩', fontsize=12, ha='center')
    ax.text(0, 0, -1.3, '|1⟩', fontsize=12, ha='center')
    
    # 量子状態のベクトルを描画
    coords = qubit.to_bloch_coordinates()
    ax.quiver(0, 0, 0, coords[0], coords[1], coords[2], 
              color='red', arrow_length_ratio=0.1, linewidth=2)
    ax.scatter([coords[0]], [coords[1]], [coords[2]], 
               color='red', s=100, label=f'r = {qubit.correlation:.2f}')
    
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'ブロッホ球表現\n相関係数 r = {qubit.correlation:.4f}')
    ax.legend()
    
    return ax


def plot_probability_distribution(qubit: PseudoQubit, ax: plt.Axes = None) -> plt.Axes:
    """
    確率分布を棒グラフで表示
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    probs = qubit.probabilities
    states = ['|0⟩', '|1⟩']
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax.bar(states, probs, color=colors, edgecolor='black', linewidth=2)
    
    # 確率値を表示
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.4f}', ha='center', fontsize=12)
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('確率')
    ax.set_title(f'測定確率分布 (r = {qubit.correlation:.4f})')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='均等分布')
    ax.legend()
    
    return ax


def plot_correlation_mapping():
    """
    相関係数から量子状態へのマッピングを視覚化
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 相関係数の範囲
    r_values = np.linspace(-1, 1, 100)
    
    # θのマッピング
    theta_values = [np.pi * (1 - r) / 2 for r in r_values]
    axes[0, 0].plot(r_values, np.degrees(theta_values), 'b-', linewidth=2)
    axes[0, 0].set_xlabel('相関係数 r')
    axes[0, 0].set_ylabel('角度 θ (度)')
    axes[0, 0].set_title('相関係数 → 角度 マッピング')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=90, color='red', linestyle='--', alpha=0.5, label='θ = 90° (重ね合わせ)')
    axes[0, 0].legend()
    
    # 確率のマッピング
    p0_values = [np.cos(np.pi * (1 - r) / 4) ** 2 for r in r_values]
    p1_values = [np.sin(np.pi * (1 - r) / 4) ** 2 for r in r_values]
    axes[0, 1].plot(r_values, p0_values, 'b-', linewidth=2, label='P(|0⟩)')
    axes[0, 1].plot(r_values, p1_values, 'r-', linewidth=2, label='P(|1⟩)')
    axes[0, 1].set_xlabel('相関係数 r')
    axes[0, 1].set_ylabel('確率')
    axes[0, 1].set_title('相関係数 → 確率 マッピング')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # 特定の相関係数のブロッホ球
    ax_bloch = fig.add_subplot(2, 2, 3, projection='3d')
    
    # 複数の相関係数の状態を表示
    test_r = [1.0, 0.5, 0.0, -0.5, -1.0]
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    
    # ブロッホ球を描画
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax_bloch.plot_surface(x, y, z, alpha=0.05, color='cyan')
    
    for r, color in zip(test_r, colors):
        qubit = PseudoQubit(correlation=r)
        coords = qubit.to_bloch_coordinates()
        ax_bloch.scatter([coords[0]], [coords[1]], [coords[2]], 
                        color=color, s=100, label=f'r = {r}')
    
    ax_bloch.set_xlim([-1.2, 1.2])
    ax_bloch.set_ylim([-1.2, 1.2])
    ax_bloch.set_zlim([-1.2, 1.2])
    ax_bloch.set_title('異なる相関係数のブロッホ球表現')
    ax_bloch.legend(loc='upper left')
    
    # 測定シミュレーション結果
    ax_sim = axes[1, 1]
    n_measurements = 1000
    
    sim_r_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ratios_0 = []
    ratios_1 = []
    
    for r in sim_r_values:
        qubit = PseudoQubit(correlation=r)
        stats = qubit.measure_n(n_measurements)
        ratios_0.append(stats['ratio_0'])
        ratios_1.append(stats['ratio_1'])
    
    x_pos = np.arange(len(sim_r_values))
    width = 0.35
    
    bars1 = ax_sim.bar(x_pos - width/2, ratios_0, width, label='|0⟩', color='#2E86AB')
    bars2 = ax_sim.bar(x_pos + width/2, ratios_1, width, label='|1⟩', color='#A23B72')
    
    ax_sim.set_xlabel('相関係数 r')
    ax_sim.set_ylabel('測定比率')
    ax_sim.set_title(f'測定シミュレーション ({n_measurements}回)')
    ax_sim.set_xticks(x_pos)
    ax_sim.set_xticklabels([str(r) for r in sim_r_values])
    ax_sim.legend()
    ax_sim.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # マッピングの視覚化
    fig = plot_correlation_mapping()
    plt.savefig('qubit_mapping.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 個別の量子ビット表示
    for r in [0.8, 0.0, -0.6]:
        qubit = PseudoQubit(correlation=r)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'擬似量子ビット: r = {r}', fontsize=14)
        
        # ブロッホ球 (3D)
        ax1.remove()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        plot_bloch_sphere(qubit, ax1)
        
        # 確率分布
        plot_probability_distribution(qubit, ax2)
        
        plt.tight_layout()
        plt.savefig(f'qubit_r{r}.png', dpi=150, bbox_inches='tight')
        plt.show()

