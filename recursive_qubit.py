"""
再帰的擬似量子ビット シミュレーション

擬似量子ビットの相関係数を別の擬似量子ビットに依存させる
N回繰り返すとどうなるか？

Q_1 → 測定 → Q_2の相関係数 → 測定 → Q_3の相関係数 → ...
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
from pseudo_qubit import PseudoQubit
import time


# ============================================================
# 再帰的量子ビット
# ============================================================

class RecursiveQubit:
    """
    再帰的擬似量子ビット
    
    前の量子ビットの測定結果が次の量子ビットの相関係数を決定
    """
    
    def __init__(self, initial_correlation: float = 0.0,
                 dependency_function: Callable[[int, float], float] = None):
        """
        initial_correlation: 最初の量子ビットの相関係数
        dependency_function: 測定結果から次の相関係数を計算する関数
                            f(measurement, current_r) -> new_r
        """
        self.initial_correlation = initial_correlation
        
        # デフォルトの依存関数
        if dependency_function is None:
            # 測定結果が0なら正方向、1なら負方向にシフト
            self.dependency_function = lambda m, r: r + (0.2 if m == 0 else -0.2)
        else:
            self.dependency_function = dependency_function
        
        self.history: List[Dict] = []
    
    def run_chain(self, n_iterations: int) -> List[Dict]:
        """
        N回の連鎖を実行
        """
        self.history = []
        current_r = self.initial_correlation
        
        for i in range(n_iterations):
            # 現在の相関係数で量子ビットを生成
            qubit = PseudoQubit(correlation=current_r)
            
            # 測定
            measurement = qubit.measure()
            
            # 記録
            self.history.append({
                'iteration': i,
                'correlation': current_r,
                'prob_0': qubit.probabilities[0],
                'prob_1': qubit.probabilities[1],
                'measurement': measurement,
                'theta': qubit.theta,
            })
            
            # 次の相関係数を計算
            current_r = self.dependency_function(measurement, current_r)
            current_r = np.clip(current_r, -1, 1)  # 範囲制限
        
        return self.history
    
    def run_ensemble(self, n_iterations: int, n_runs: int = 100) -> np.ndarray:
        """
        複数回のアンサンブル実行
        """
        all_correlations = np.zeros((n_runs, n_iterations))
        all_measurements = np.zeros((n_runs, n_iterations))
        
        for run in range(n_runs):
            history = self.run_chain(n_iterations)
            all_correlations[run] = [h['correlation'] for h in history]
            all_measurements[run] = [h['measurement'] for h in history]
        
        return all_correlations, all_measurements


# ============================================================
# 様々な依存関数
# ============================================================

def linear_dependency(m: int, r: float, strength: float = 0.3) -> float:
    """線形依存：測定結果で相関係数をシフト"""
    return r + strength * (1 if m == 0 else -1)


def multiplicative_dependency(m: int, r: float, factor: float = 0.9) -> float:
    """乗法的依存：測定結果で相関係数をスケール"""
    sign = 1 if m == 0 else -1
    return r * factor + sign * 0.1


def feedback_dependency(m: int, r: float, alpha: float = 0.5) -> float:
    """フィードバック依存：測定結果と現在値の混合"""
    target = 1 if m == 0 else -1
    return alpha * r + (1 - alpha) * target


def chaotic_dependency(m: int, r: float) -> float:
    """カオス的依存：ロジスティック写像風"""
    # r を [0, 1] に変換
    x = (r + 1) / 2
    # ロジスティック写像
    mu = 3.9
    x_new = mu * x * (1 - x)
    # 測定結果で符号を決定
    sign = 1 if m == 0 else -1
    return sign * (2 * x_new - 1)


def quantum_inspired_dependency(m: int, r: float) -> float:
    """量子インスパイアード依存：角度を更新"""
    theta = np.arccos(r)
    delta_theta = np.pi / 8 * (1 if m == 0 else -1)
    new_theta = theta + delta_theta
    new_theta = np.clip(new_theta, 0, np.pi)
    return np.cos(new_theta)


def entanglement_dependency(m: int, r: float) -> float:
    """もつれ風依存：測定結果で反相関"""
    if m == 0:
        return max(-1, r - 0.5)  # |0⟩測定で相関減少
    else:
        return min(1, r + 0.5)   # |1⟩測定で相関増加


def oscillating_dependency(m: int, r: float, frequency: float = 0.3) -> float:
    """振動依存：周期的な変化"""
    phase = np.arccos(r) + frequency * np.pi
    sign = 1 if m == 0 else -1
    return sign * np.cos(phase)


# ============================================================
# 分析
# ============================================================

def analyze_convergence(correlations: np.ndarray) -> Dict:
    """収束性を分析"""
    n_runs, n_iterations = correlations.shape
    
    # 最終値の統計
    final_values = correlations[:, -1]
    
    # 収束判定（最後の10%で変動が小さいか）
    late_phase = correlations[:, -n_iterations//10:]
    is_converged = np.std(late_phase, axis=1) < 0.1
    
    # 周期性検出（自己相関）
    mean_correlation = np.mean(correlations, axis=0)
    autocorr = np.correlate(mean_correlation - np.mean(mean_correlation), 
                           mean_correlation - np.mean(mean_correlation), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= autocorr[0]
    
    return {
        'final_mean': np.mean(final_values),
        'final_std': np.std(final_values),
        'convergence_rate': np.mean(is_converged),
        'autocorrelation': autocorr[:min(50, len(autocorr))],
    }


def compute_entropy(measurements: np.ndarray) -> np.ndarray:
    """各ステップでの測定結果のエントロピーを計算"""
    n_runs, n_iterations = measurements.shape
    entropies = []
    
    for i in range(n_iterations):
        p0 = np.mean(measurements[:, i] == 0)
        p1 = 1 - p0
        if p0 > 0 and p1 > 0:
            entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
        else:
            entropy = 0
        entropies.append(entropy)
    
    return np.array(entropies)


# ============================================================
# 可視化
# ============================================================

def visualize_single_run(history: List[Dict], title: str = "再帰的量子ビット"):
    """単一実行の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0f')
    fig.suptitle(title, color='#00d4ff', fontsize=16)
    
    for ax in axes.flat:
        ax.set_facecolor('#0a0a0f')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    iterations = [h['iteration'] for h in history]
    correlations = [h['correlation'] for h in history]
    measurements = [h['measurement'] for h in history]
    prob_0 = [h['prob_0'] for h in history]
    
    # 1. 相関係数の推移
    ax1 = axes[0, 0]
    ax1.plot(iterations, correlations, color='#00d4ff', linewidth=2)
    ax1.axhline(0, color='white', linestyle='--', alpha=0.3)
    ax1.axhline(1, color='#10b981', linestyle='--', alpha=0.3)
    ax1.axhline(-1, color='#ec4899', linestyle='--', alpha=0.3)
    ax1.fill_between(iterations, correlations, 0, alpha=0.3, 
                     color=['#10b981' if c > 0 else '#ec4899' for c in correlations])
    ax1.set_xlabel('Iteration', color='white')
    ax1.set_ylabel('Correlation r', color='white')
    ax1.set_title('Correlation Evolution', color='white')
    ax1.set_ylim(-1.1, 1.1)
    
    # 2. 測定結果
    ax2 = axes[0, 1]
    colors = ['#00d4ff' if m == 0 else '#ec4899' for m in measurements]
    ax2.scatter(iterations, measurements, c=colors, s=30, alpha=0.7)
    ax2.set_xlabel('Iteration', color='white')
    ax2.set_ylabel('Measurement', color='white')
    ax2.set_title('Measurement Results', color='white')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['|0⟩', '|1⟩'])
    
    # 3. P(|0⟩) の推移
    ax3 = axes[1, 0]
    ax3.plot(iterations, prob_0, color='#8b5cf6', linewidth=2)
    ax3.axhline(0.5, color='white', linestyle='--', alpha=0.3)
    ax3.fill_between(iterations, prob_0, 0.5, alpha=0.3, color='#8b5cf6')
    ax3.set_xlabel('Iteration', color='white')
    ax3.set_ylabel('P(|0⟩)', color='white')
    ax3.set_title('Probability Evolution', color='white')
    ax3.set_ylim(0, 1)
    
    # 4. 相関係数のヒストグラム
    ax4 = axes[1, 1]
    ax4.hist(correlations, bins=30, color='#00d4ff', alpha=0.7, edgecolor='white')
    ax4.set_xlabel('Correlation', color='white')
    ax4.set_ylabel('Count', color='white')
    ax4.set_title('Correlation Distribution', color='white')
    
    plt.tight_layout()
    plt.savefig('recursive_qubit_single.png', facecolor='#0a0a0f', dpi=150)
    plt.show()


def visualize_ensemble(correlations: np.ndarray, measurements: np.ndarray, 
                       title: str = "Ensemble Analysis"):
    """アンサンブル実行の可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a0f')
    fig.suptitle(title, color='#00d4ff', fontsize=16)
    
    for ax in axes.flat:
        ax.set_facecolor('#0a0a0f')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    n_runs, n_iterations = correlations.shape
    iterations = np.arange(n_iterations)
    
    # 1. 全軌跡
    ax1 = axes[0, 0]
    for i in range(min(50, n_runs)):
        ax1.plot(iterations, correlations[i], alpha=0.2, linewidth=0.5, color='#00d4ff')
    ax1.plot(iterations, np.mean(correlations, axis=0), color='#ec4899', linewidth=2, label='Mean')
    ax1.axhline(0, color='white', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Iteration', color='white')
    ax1.set_ylabel('Correlation', color='white')
    ax1.set_title('All Trajectories', color='white')
    ax1.legend()
    
    # 2. 平均と標準偏差
    ax2 = axes[0, 1]
    mean = np.mean(correlations, axis=0)
    std = np.std(correlations, axis=0)
    ax2.plot(iterations, mean, color='#00d4ff', linewidth=2, label='Mean')
    ax2.fill_between(iterations, mean - std, mean + std, color='#00d4ff', alpha=0.3, label='±1σ')
    ax2.set_xlabel('Iteration', color='white')
    ax2.set_ylabel('Correlation', color='white')
    ax2.set_title('Mean ± Std', color='white')
    ax2.legend()
    
    # 3. エントロピー
    ax3 = axes[0, 2]
    entropy = compute_entropy(measurements)
    ax3.plot(iterations, entropy, color='#10b981', linewidth=2)
    ax3.axhline(1, color='white', linestyle='--', alpha=0.3, label='Max Entropy')
    ax3.set_xlabel('Iteration', color='white')
    ax3.set_ylabel('Entropy (bits)', color='white')
    ax3.set_title('Measurement Entropy', color='white')
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    
    # 4. 最終分布
    ax4 = axes[1, 0]
    ax4.hist(correlations[:, -1], bins=30, color='#8b5cf6', alpha=0.7, edgecolor='white')
    ax4.set_xlabel('Final Correlation', color='white')
    ax4.set_ylabel('Count', color='white')
    ax4.set_title('Final Distribution', color='white')
    
    # 5. 時間発展ヒートマップ
    ax5 = axes[1, 1]
    # 相関係数をビン分けしてヒートマップ
    bins = np.linspace(-1, 1, 41)
    heatmap = np.zeros((len(bins) - 1, n_iterations))
    for i in range(n_iterations):
        hist, _ = np.histogram(correlations[:, i], bins=bins)
        heatmap[:, i] = hist
    
    im = ax5.imshow(heatmap, aspect='auto', cmap='plasma', 
                    extent=[0, n_iterations, -1, 1], origin='lower')
    ax5.set_xlabel('Iteration', color='white')
    ax5.set_ylabel('Correlation', color='white')
    ax5.set_title('Distribution Evolution', color='white')
    plt.colorbar(im, ax=ax5, label='Count')
    
    # 6. 測定結果の偏り
    ax6 = axes[1, 2]
    p0_over_time = np.mean(measurements == 0, axis=0)
    ax6.plot(iterations, p0_over_time, color='#f59e0b', linewidth=2)
    ax6.axhline(0.5, color='white', linestyle='--', alpha=0.3)
    ax6.fill_between(iterations, p0_over_time, 0.5, alpha=0.3, color='#f59e0b')
    ax6.set_xlabel('Iteration', color='white')
    ax6.set_ylabel('P(|0⟩ measured)', color='white')
    ax6.set_title('Measurement Bias', color='white')
    ax6.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('recursive_qubit_ensemble.png', facecolor='#0a0a0f', dpi=150)
    plt.show()


def compare_dependencies(n_iterations: int = 100, n_runs: int = 500):
    """異なる依存関数を比較"""
    dependencies = {
        'Linear': lambda m, r: linear_dependency(m, r, 0.3),
        'Feedback': lambda m, r: feedback_dependency(m, r, 0.7),
        'Chaotic': chaotic_dependency,
        'Quantum-Inspired': quantum_inspired_dependency,
        'Entanglement': entanglement_dependency,
        'Oscillating': oscillating_dependency,
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a0f')
    fig.suptitle('Dependency Function Comparison (N={})'.format(n_iterations), 
                 color='#00d4ff', fontsize=16)
    
    for ax, (name, dep_func) in zip(axes.flat, dependencies.items()):
        ax.set_facecolor('#0a0a0f')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        
        # シミュレーション
        rq = RecursiveQubit(initial_correlation=0.0, dependency_function=dep_func)
        correlations, measurements = rq.run_ensemble(n_iterations, n_runs)
        
        # 可視化
        iterations = np.arange(n_iterations)
        mean = np.mean(correlations, axis=0)
        std = np.std(correlations, axis=0)
        
        for i in range(min(20, n_runs)):
            ax.plot(iterations, correlations[i], alpha=0.1, linewidth=0.5, color='#00d4ff')
        ax.plot(iterations, mean, color='#ec4899', linewidth=2)
        ax.fill_between(iterations, mean - std, mean + std, color='#ec4899', alpha=0.2)
        
        ax.axhline(0, color='white', linestyle='--', alpha=0.3)
        ax.set_title(name, color='white', fontsize=12)
        ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('recursive_qubit_comparison.png', facecolor='#0a0a0f', dpi=150)
    plt.show()


# ============================================================
# デモ
# ============================================================

def demo():
    """再帰的量子ビットのデモ"""
    print("=" * 70)
    print("  再帰的擬似量子ビット シミュレーション")
    print("  Recursive Pseudo-Qubit Simulation")
    print("=" * 70)
    
    print("\n" + "─" * 70)
    print("  概念: Q_n の測定結果 → Q_{n+1} の相関係数")
    print("─" * 70)
    print("""
  量子ビットの相関係数を前の量子ビットの測定結果に依存させる：
  
    Q₁(r₁) → 測定 m₁ → r₂ = f(m₁, r₁)
    Q₂(r₂) → 測定 m₂ → r₃ = f(m₂, r₂)
    ...
    Q_N(r_N) → 測定 m_N
  
  これにより量子ビットの連鎖が形成される
    """)
    
    # ============================================================
    # 1. 単一実行
    # ============================================================
    print("\n" + "─" * 70)
    print("  1. 単一実行 (N=100)")
    print("─" * 70)
    
    rq = RecursiveQubit(
        initial_correlation=0.0,
        dependency_function=lambda m, r: linear_dependency(m, r, 0.2)
    )
    
    history = rq.run_chain(100)
    
    print("\n  最初の10ステップ:")
    print("  " + "-" * 60)
    print(f"  {'Step':>5} | {'r':>8} | {'P(|0⟩)':>8} | {'測定':>6} | {'次のr':>8}")
    print("  " + "-" * 60)
    
    for h in history[:10]:
        next_r = history[h['iteration'] + 1]['correlation'] if h['iteration'] < len(history) - 1 else '-'
        measurement_str = '|0⟩' if h['measurement'] == 0 else '|1⟩'
        next_r_str = f"{next_r:.4f}" if isinstance(next_r, float) else next_r
        print(f"  {h['iteration']:>5} | {h['correlation']:>8.4f} | {h['prob_0']:>8.4f} | {measurement_str:>6} | {next_r_str:>8}")
    
    print("\n  [可視化中...]")
    visualize_single_run(history, "Linear Dependency (single run)")
    
    # ============================================================
    # 2. アンサンブル分析
    # ============================================================
    print("\n" + "─" * 70)
    print("  2. アンサンブル分析 (N=100, runs=500)")
    print("─" * 70)
    
    correlations, measurements = rq.run_ensemble(100, 500)
    
    analysis = analyze_convergence(correlations)
    
    print(f"\n  最終相関係数:")
    print(f"    平均: {analysis['final_mean']:.4f}")
    print(f"    標準偏差: {analysis['final_std']:.4f}")
    print(f"    収束率: {analysis['convergence_rate']:.1%}")
    
    print("\n  [アンサンブル可視化中...]")
    visualize_ensemble(correlations, measurements, "Linear Dependency Ensemble")
    
    # ============================================================
    # 3. 様々な依存関数の比較
    # ============================================================
    print("\n" + "─" * 70)
    print("  3. 依存関数の比較")
    print("─" * 70)
    
    dependencies = {
        'Linear': lambda m, r: linear_dependency(m, r, 0.3),
        'Feedback': lambda m, r: feedback_dependency(m, r, 0.7),
        'Chaotic': chaotic_dependency,
        'Quantum-Inspired': quantum_inspired_dependency,
        'Entanglement': entanglement_dependency,
        'Oscillating': oscillating_dependency,
    }
    
    print("\n  各依存関数の特性:")
    print("  " + "-" * 60)
    
    for name, dep_func in dependencies.items():
        rq_temp = RecursiveQubit(initial_correlation=0.0, dependency_function=dep_func)
        corrs, _ = rq_temp.run_ensemble(100, 200)
        analysis = analyze_convergence(corrs)
        
        print(f"\n  {name}:")
        print(f"    最終平均: {analysis['final_mean']:+.4f}")
        print(f"    最終標準偏差: {analysis['final_std']:.4f}")
        print(f"    収束率: {analysis['convergence_rate']:.1%}")
    
    print("\n  [比較可視化中...]")
    compare_dependencies(100, 200)
    
    # ============================================================
    # 4. 長期シミュレーション
    # ============================================================
    print("\n" + "─" * 70)
    print("  4. 長期シミュレーション (N=1000)")
    print("─" * 70)
    
    rq_long = RecursiveQubit(
        initial_correlation=0.5,
        dependency_function=lambda m, r: feedback_dependency(m, r, 0.8)
    )
    
    correlations_long, measurements_long = rq_long.run_ensemble(1000, 100)
    
    # 統計の時間発展
    print("\n  統計の時間発展:")
    print("  " + "-" * 50)
    checkpoints = [10, 50, 100, 500, 1000]
    
    for cp in checkpoints:
        if cp <= correlations_long.shape[1]:
            mean_cp = np.mean(correlations_long[:, cp-1])
            std_cp = np.std(correlations_long[:, cp-1])
            print(f"    N={cp:4d}: mean={mean_cp:+.4f}, std={std_cp:.4f}")
    
    print("\n  [長期シミュレーション可視化中...]")
    visualize_ensemble(correlations_long, measurements_long, 
                      "Long-term Simulation (N=1000, Feedback)")
    
    # ============================================================
    # 5. カオス的依存の詳細分析
    # ============================================================
    print("\n" + "─" * 70)
    print("  5. カオス的依存の分析")
    print("─" * 70)
    
    rq_chaos = RecursiveQubit(
        initial_correlation=0.1,
        dependency_function=chaotic_dependency
    )
    
    # 初期値感度テスト
    print("\n  初期値感度（バタフライ効果）:")
    
    initial_values = [0.100, 0.101, 0.102]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    
    colors = ['#00d4ff', '#ec4899', '#10b981']
    
    for iv, color in zip(initial_values, colors):
        rq_test = RecursiveQubit(
            initial_correlation=iv,
            dependency_function=chaotic_dependency
        )
        history = rq_test.run_chain(50)
        correlations = [h['correlation'] for h in history]
        ax.plot(correlations, color=color, linewidth=2, label=f'r₀={iv}')
    
    ax.set_xlabel('Iteration', color='white', fontsize=12)
    ax.set_ylabel('Correlation', color='white', fontsize=12)
    ax.set_title('Chaos: Initial Value Sensitivity', color='#00d4ff', fontsize=14)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('recursive_qubit_chaos.png', facecolor='#0a0a0f', dpi=150)
    plt.show()
    
    print("\n" + "=" * 70)
    print("  シミュレーション完了！")
    print("=" * 70)
    print("""
  発見された挙動:
  
  1. Linear: 測定結果に応じて相関係数が±1に振動
     → ランダムウォーク的な挙動
  
  2. Feedback: ターゲット値への収束傾向
     → 自己組織化的な挙動
  
  3. Chaotic: 初期値に敏感、予測不可能な軌跡
     → カオス的な挙動
  
  4. Quantum-Inspired: 角度ベースの滑らかな変化
     → 量子的な回転に類似
  
  5. Entanglement: 測定結果と逆方向に移動
     → 反相関的な挙動
  
  6. Oscillating: 周期的なパターン
     → 振動的な挙動
    """)


if __name__ == "__main__":
    demo()

