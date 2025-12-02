"""
擬似量子ビット (Pseudo-Qubit)

相関係数 r ∈ [-1, 1] を量子状態にマッピングする実装

マッピング原理:
- r = 1  → |0⟩ (純粋状態)
- r = -1 → |1⟩ (純粋状態)
- r = 0  → (|0⟩ + |1⟩)/√2 (完全な重ね合わせ)

量子状態: |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import random


@dataclass
class QuantumState:
    """量子状態を表すデータクラス"""
    alpha: complex  # |0⟩ の振幅
    beta: complex   # |1⟩ の振幅
    
    @property
    def prob_0(self) -> float:
        """|0⟩ を測定する確率"""
        return abs(self.alpha) ** 2
    
    @property
    def prob_1(self) -> float:
        """|1⟩ を測定する確率"""
        return abs(self.beta) ** 2
    
    def __str__(self) -> str:
        return f"|ψ⟩ = {self.alpha:.4f}|0⟩ + {self.beta:.4f}|1⟩"


class PseudoQubit:
    """
    相関係数を量子ビットにマッピングする擬似量子ビット
    
    数学的対応関係:
    - 相関係数 r ∈ [-1, 1] → 角度 θ ∈ [0, π]
    - θ = arccos(r) を使用して、r から角度を導出
    - 代替マッピング: θ = π * (1 - r) / 2
    """
    
    def __init__(self, correlation: float = 0.0):
        """
        Args:
            correlation: 相関係数 r ∈ [-1, 1]
        """
        self._validate_correlation(correlation)
        self._correlation = correlation
        self._theta = self._correlation_to_theta(correlation)
        self._state = self._compute_state()
    
    @staticmethod
    def _validate_correlation(r: float) -> None:
        """相関係数が有効範囲内か検証"""
        if not -1.0 <= r <= 1.0:
            raise ValueError(f"相関係数は [-1, 1] の範囲内である必要があります: {r}")
    
    @staticmethod
    def _correlation_to_theta(r: float) -> float:
        """
        相関係数を角度θに変換
        
        マッピング:
        - r = 1  → θ = 0   (|0⟩)
        - r = -1 → θ = π   (|1⟩)
        - r = 0  → θ = π/2 (重ね合わせ)
        """
        # 線形マッピング: θ = π * (1 - r) / 2
        return np.pi * (1 - r) / 2
    
    @staticmethod
    def _theta_to_correlation(theta: float) -> float:
        """角度θを相関係数に逆変換"""
        return 1 - 2 * theta / np.pi
    
    def _compute_state(self) -> QuantumState:
        """角度θから量子状態を計算"""
        alpha = complex(np.cos(self._theta / 2), 0)
        beta = complex(np.sin(self._theta / 2), 0)
        return QuantumState(alpha=alpha, beta=beta)
    
    @property
    def correlation(self) -> float:
        """相関係数を取得"""
        return self._correlation
    
    @correlation.setter
    def correlation(self, value: float) -> None:
        """相関係数を設定"""
        self._validate_correlation(value)
        self._correlation = value
        self._theta = self._correlation_to_theta(value)
        self._state = self._compute_state()
    
    @property
    def theta(self) -> float:
        """角度θを取得 (ラジアン)"""
        return self._theta
    
    @property
    def state(self) -> QuantumState:
        """量子状態を取得"""
        return self._state
    
    @property
    def probabilities(self) -> Tuple[float, float]:
        """(P(|0⟩), P(|1⟩)) を返す"""
        return (self._state.prob_0, self._state.prob_1)
    
    def measure(self) -> int:
        """
        量子ビットを測定（シミュレーション）
        
        Returns:
            0 または 1（確率的に決定）
        """
        return 0 if random.random() < self._state.prob_0 else 1
    
    def measure_n(self, n: int = 1000) -> dict:
        """
        n回測定を行い、統計を返す
        
        Args:
            n: 測定回数
            
        Returns:
            {'0': count_0, '1': count_1, 'ratio_0': ratio_0, 'ratio_1': ratio_1}
        """
        results = [self.measure() for _ in range(n)]
        count_0 = results.count(0)
        count_1 = results.count(1)
        
        return {
            '0': count_0,
            '1': count_1,
            'ratio_0': count_0 / n,
            'ratio_1': count_1 / n
        }
    
    def to_bloch_coordinates(self) -> Tuple[float, float, float]:
        """
        ブロッホ球上の座標を返す (x, y, z)
        
        この実装では位相 φ = 0 と仮定
        """
        x = np.sin(self._theta)
        y = 0.0
        z = np.cos(self._theta)
        return (x, y, z)
    
    def __str__(self) -> str:
        return (
            f"PseudoQubit(r={self._correlation:.4f})\n"
            f"  θ = {self._theta:.4f} rad ({np.degrees(self._theta):.2f}°)\n"
            f"  {self._state}\n"
            f"  P(|0⟩) = {self._state.prob_0:.4f}, P(|1⟩) = {self._state.prob_1:.4f}"
        )
    
    def __repr__(self) -> str:
        return f"PseudoQubit(correlation={self._correlation})"


# 便利な関数
def correlation_to_probability(r: float) -> Tuple[float, float]:
    """
    相関係数から確率への直接変換
    
    P(|0⟩) = cos²(θ/2) where θ = π(1-r)/2
    P(|1⟩) = sin²(θ/2)
    
    簡略化: P(|0⟩) = (1 + r) / 2, P(|1⟩) = (1 - r) / 2
    """
    theta = np.pi * (1 - r) / 2
    p0 = np.cos(theta / 2) ** 2
    p1 = np.sin(theta / 2) ** 2
    return (p0, p1)


def create_qubit_from_data(x: np.ndarray, y: np.ndarray) -> PseudoQubit:
    """
    二つのデータ配列から相関係数を計算し、擬似量子ビットを生成
    
    Args:
        x: データ配列1
        y: データ配列2
        
    Returns:
        PseudoQubit インスタンス
    """
    r = np.corrcoef(x, y)[0, 1]
    return PseudoQubit(correlation=r)


# デモンストレーション
if __name__ == "__main__":
    print("=" * 60)
    print("擬似量子ビット (Pseudo-Qubit) デモンストレーション")
    print("=" * 60)
    
    # 境界ケースのテスト
    test_correlations = [1.0, 0.5, 0.0, -0.5, -1.0]
    
    print("\n【相関係数から量子状態へのマッピング】\n")
    for r in test_correlations:
        qubit = PseudoQubit(correlation=r)
        print(qubit)
        print()
    
    # 測定のデモ
    print("=" * 60)
    print("【r = 0 の場合の測定シミュレーション (1000回)】")
    print("=" * 60)
    
    qubit = PseudoQubit(correlation=0.0)
    stats = qubit.measure_n(1000)
    print(f"理論値: P(|0⟩) = 0.5000, P(|1⟩) = 0.5000")
    print(f"実測値: P(|0⟩) = {stats['ratio_0']:.4f}, P(|1⟩) = {stats['ratio_1']:.4f}")
    print(f"カウント: |0⟩ = {stats['0']}, |1⟩ = {stats['1']}")
    
    # データからの量子ビット生成
    print("\n" + "=" * 60)
    print("【実データからの量子ビット生成】")
    print("=" * 60)
    
    # 強い正の相関を持つデータ
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 10, 12])
    
    qubit = create_qubit_from_data(x, y)
    print(f"\nデータの相関係数: r = {qubit.correlation:.4f}")
    print(qubit)
    
    # ブロッホ球座標
    print("\n【ブロッホ球座標】")
    coords = qubit.to_bloch_coordinates()
    print(f"(x, y, z) = ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})")

