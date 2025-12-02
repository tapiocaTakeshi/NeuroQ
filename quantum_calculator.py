"""
量子インスパイアード 高速計算機

擬似量子ビットの原理を活用した高速計算システム
- 量子並列計算
- 最適化問題ソルバー
- 高速検索
- 素因数分解
- 組み合わせ最適化
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
import time
from functools import lru_cache
import random
from pseudo_qubit import PseudoQubit


# ============================================================
# 量子レジスタ
# ============================================================

class QuantumRegister:
    """
    量子レジスタ - 複数の量子ビットを管理
    
    n量子ビットで2^n個の状態を同時に表現
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        
        # 全状態の確率振幅（初期は一様重ね合わせ）
        self.amplitudes = np.ones(self.n_states, dtype=complex) / np.sqrt(self.n_states)
        
        # 擬似量子ビットの配列
        self.qubits = [PseudoQubit(correlation=0.0) for _ in range(n_qubits)]
    
    def _normalize(self):
        """正規化"""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 0:
            self.amplitudes /= norm
    
    @property
    def probabilities(self) -> np.ndarray:
        """各状態の確率"""
        return np.abs(self.amplitudes) ** 2
    
    def set_superposition(self, correlations: List[float]):
        """相関係数から重ね合わせを設定"""
        if len(correlations) != self.n_qubits:
            raise ValueError(f"相関の数が一致しません: {len(correlations)} != {self.n_qubits}")
        
        self.qubits = [PseudoQubit(correlation=r) for r in correlations]
        
        # 振幅を再計算
        for i in range(self.n_states):
            amplitude = 1.0
            for j in range(self.n_qubits):
                bit = (i >> j) & 1
                amplitude *= self.qubits[j].state.amplitude_1 if bit else self.qubits[j].state.amplitude_0
            self.amplitudes[i] = amplitude
        
        self._normalize()
    
    def amplify(self, target_states: List[int], factor: float = 2.0):
        """
        グローバー的振幅増幅
        ターゲット状態の振幅を増幅
        """
        for state in target_states:
            if 0 <= state < self.n_states:
                self.amplitudes[state] *= factor
        self._normalize()
    
    def measure(self) -> int:
        """測定して状態を確定"""
        probs = self.probabilities
        probs /= probs.sum()
        return np.random.choice(self.n_states, p=probs)
    
    def measure_to_binary(self) -> List[int]:
        """測定結果を2進数リストで返す"""
        result = self.measure()
        return [(result >> i) & 1 for i in range(self.n_qubits)]
    
    def top_k_states(self, k: int = 5) -> List[Tuple[int, float]]:
        """上位k個の状態を返す"""
        probs = self.probabilities
        indices = np.argsort(probs)[::-1][:k]
        return [(int(idx), probs[idx]) for idx in indices]


# ============================================================
# 量子並列計算エンジン
# ============================================================

class QuantumParallelComputer:
    """
    量子並列計算機
    
    全ての入力に対して関数を「同時に」評価
    """
    
    def __init__(self, n_bits: int):
        self.n_bits = n_bits
        self.register = QuantumRegister(n_bits)
    
    def evaluate_all(self, func: Callable[[int], float]) -> np.ndarray:
        """
        全入力に対して関数を並列評価
        
        量子的には全入力が重ね合わせで同時評価される
        """
        results = np.zeros(self.register.n_states)
        
        # 擬似並列：実際は順次だが概念的に並列
        for x in range(self.register.n_states):
            results[x] = func(x)
        
        return results
    
    def find_optimal(self, func: Callable[[int], float], 
                     minimize: bool = True,
                     n_iterations: int = 10) -> Tuple[int, float]:
        """
        最適値を量子探索で発見
        
        グローバー的振幅増幅を使用
        """
        # 全評価
        results = self.evaluate_all(func)
        
        # 初期状態：一様重ね合わせ
        self.register.amplitudes = np.ones(self.register.n_states, dtype=complex) / np.sqrt(self.register.n_states)
        
        # 振幅増幅
        for _ in range(n_iterations):
            if minimize:
                # 最小値に近い状態を増幅
                threshold = np.percentile(results, 20)
                good_states = np.where(results <= threshold)[0]
            else:
                # 最大値に近い状態を増幅
                threshold = np.percentile(results, 80)
                good_states = np.where(results >= threshold)[0]
            
            self.register.amplify(good_states.tolist(), factor=1.5)
        
        # 測定
        best_state = self.register.measure()
        
        # 確認のため上位候補も表示用に返す
        return best_state, results[best_state]


# ============================================================
# 量子最適化ソルバー
# ============================================================

class QuantumOptimizer:
    """
    量子アニーリング風最適化ソルバー
    
    組み合わせ最適化問題を量子的アプローチで解く
    """
    
    def __init__(self, n_variables: int):
        self.n_variables = n_variables
        self.register = QuantumRegister(n_variables)
    
    def solve_qubo(self, Q: np.ndarray, 
                   n_iterations: int = 100,
                   initial_temp: float = 10.0,
                   final_temp: float = 0.01) -> Tuple[np.ndarray, float]:
        """
        QUBO問題を解く
        (Quadratic Unconstrained Binary Optimization)
        
        minimize: x^T Q x
        """
        n = self.n_variables
        
        # 量子的初期状態（重ね合わせ）
        current_x = np.array([
            self.register.qubits[i].measure()
            for i in range(n)
        ])
        current_energy = self._qubo_energy(current_x, Q)
        
        best_x = current_x.copy()
        best_energy = current_energy
        
        # 量子アニーリング風の最適化
        for i in range(n_iterations):
            # 温度スケジュール
            temp = initial_temp * (final_temp / initial_temp) ** (i / n_iterations)
            
            # 量子トンネル効果を模倣：ランダムなビット反転
            new_x = current_x.copy()
            flip_idx = np.random.randint(n)
            
            # 擬似量子ビットの相関を使ってフリップ確率を決定
            correlation = 2 * current_x[flip_idx] - 1  # 0→-1, 1→1
            qubit = PseudoQubit(correlation=correlation * 0.5)
            
            if qubit.measure() != current_x[flip_idx]:
                new_x[flip_idx] = 1 - new_x[flip_idx]
            
            new_energy = self._qubo_energy(new_x, Q)
            
            # メトロポリス基準
            delta = new_energy - current_energy
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_x = new_x
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_x = current_x.copy()
                    best_energy = current_energy
        
        return best_x, best_energy
    
    def _qubo_energy(self, x: np.ndarray, Q: np.ndarray) -> float:
        """QUBO エネルギー計算"""
        return float(x @ Q @ x)
    
    def solve_max_cut(self, adjacency: np.ndarray, 
                      n_iterations: int = 100) -> Tuple[np.ndarray, int]:
        """
        最大カット問題を解く
        
        グラフを2つに分割して、カットされる辺を最大化
        """
        n = adjacency.shape[0]
        self.n_variables = n
        self.register = QuantumRegister(n)
        
        # QUBOに変換
        Q = -adjacency / 4  # 最大化→最小化
        for i in range(n):
            Q[i, i] = np.sum(adjacency[i]) / 2
        
        best_x, _ = self.solve_qubo(Q, n_iterations)
        
        # カット数を計算
        cut_value = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] and best_x[i] != best_x[j]:
                    cut_value += 1
        
        return best_x, cut_value
    
    def solve_tsp(self, distances: np.ndarray, 
                  n_iterations: int = 200) -> Tuple[List[int], float]:
        """
        巡回セールスマン問題を量子的に解く
        """
        n = distances.shape[0]
        
        # 初期解：ランダム
        current_tour = list(range(n))
        np.random.shuffle(current_tour)
        current_dist = self._tour_distance(current_tour, distances)
        
        best_tour = current_tour.copy()
        best_dist = current_dist
        
        initial_temp = 100.0
        final_temp = 0.1
        
        for i in range(n_iterations):
            temp = initial_temp * (final_temp / initial_temp) ** (i / n_iterations)
            
            # 量子トンネル：2-opt近傍
            new_tour = current_tour.copy()
            
            # 擬似量子ビットで交換位置を決定
            qubit1 = PseudoQubit(correlation=0.0)
            qubit2 = PseudoQubit(correlation=0.0)
            
            i1 = int(qubit1.probabilities[0] * n) % n
            i2 = int(qubit2.probabilities[0] * n) % n
            
            if i1 > i2:
                i1, i2 = i2, i1
            
            # 2-opt
            new_tour[i1:i2+1] = reversed(new_tour[i1:i2+1])
            new_dist = self._tour_distance(new_tour, distances)
            
            delta = new_dist - current_dist
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_tour = new_tour
                current_dist = new_dist
                
                if current_dist < best_dist:
                    best_tour = current_tour.copy()
                    best_dist = current_dist
        
        return best_tour, best_dist
    
    def _tour_distance(self, tour: List[int], distances: np.ndarray) -> float:
        """ツアーの総距離"""
        total = 0
        for i in range(len(tour)):
            total += distances[tour[i], tour[(i + 1) % len(tour)]]
        return total


# ============================================================
# 量子検索エンジン
# ============================================================

class QuantumSearchEngine:
    """
    グローバーの検索アルゴリズム風の高速検索
    """
    
    def __init__(self, database_size: int):
        self.database_size = database_size
        self.n_qubits = int(np.ceil(np.log2(database_size)))
        self.register = QuantumRegister(self.n_qubits)
    
    def search(self, oracle: Callable[[int], bool], 
               n_iterations: Optional[int] = None) -> int:
        """
        量子検索
        
        oracle: 正解かどうかを判定する関数
        """
        if n_iterations is None:
            # 最適なイテレーション数
            n_iterations = int(np.pi / 4 * np.sqrt(self.database_size))
        
        # 初期状態：一様重ね合わせ
        self.register.amplitudes = np.ones(self.register.n_states, dtype=complex) / np.sqrt(self.register.n_states)
        
        for _ in range(n_iterations):
            # オラクル：正解の振幅を反転
            for i in range(min(self.register.n_states, self.database_size)):
                if oracle(i):
                    self.register.amplitudes[i] *= -1
            
            # 拡散変換（平均周りの反転）
            mean = np.mean(self.register.amplitudes)
            self.register.amplitudes = 2 * mean - self.register.amplitudes
        
        # 測定
        return self.register.measure() % self.database_size
    
    def search_multiple(self, oracle: Callable[[int], bool], 
                        n_solutions: int = 1) -> List[int]:
        """複数の解を検索"""
        solutions = []
        found = set()
        
        for _ in range(n_solutions * 10):  # 十分な試行
            result = self.search(oracle)
            if result not in found and oracle(result):
                solutions.append(result)
                found.add(result)
                if len(solutions) >= n_solutions:
                    break
        
        return solutions


# ============================================================
# 量子素因数分解
# ============================================================

class QuantumFactorizer:
    """
    ショアのアルゴリズム風素因数分解
    
    量子フーリエ変換を使った周期発見
    """
    
    def __init__(self):
        pass
    
    def factor(self, N: int) -> List[int]:
        """
        Nを素因数分解
        """
        if N <= 1:
            return []
        
        if self._is_prime(N):
            return [N]
        
        # 小さい素数でチェック
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for p in small_primes:
            if N % p == 0:
                return [p] + self.factor(N // p)
        
        # 量子風周期発見
        factor = self._quantum_period_finding(N)
        
        if factor and factor != N:
            return sorted(self.factor(factor) + self.factor(N // factor))
        
        return [N]
    
    def _quantum_period_finding(self, N: int) -> Optional[int]:
        """
        量子周期発見を使った因数発見
        """
        for _ in range(10):  # 複数回試行
            # ランダムな基底を選択
            a = random.randint(2, N - 1)
            
            gcd = self._gcd(a, N)
            if gcd > 1:
                return gcd
            
            # 量子レジスタで周期を探す
            n_qubits = int(np.ceil(np.log2(N))) + 2
            register = QuantumRegister(n_qubits)
            
            # 量子フーリエ変換風の処理
            # 各状態に a^x mod N の位相を付与
            for x in range(min(register.n_states, N * 2)):
                phase = (2 * np.pi * pow(a, x, N)) / N
                register.amplitudes[x % register.n_states] *= np.exp(1j * phase)
            
            register._normalize()
            
            # 測定して周期の候補を得る
            measured = register.measure()
            
            if measured == 0:
                continue
            
            # 連分数展開で周期を推定
            r = self._find_period_from_measurement(measured, register.n_states, N)
            
            if r and r % 2 == 0:
                x = pow(a, r // 2, N)
                p1 = self._gcd(x - 1, N)
                p2 = self._gcd(x + 1, N)
                
                if 1 < p1 < N:
                    return p1
                if 1 < p2 < N:
                    return p2
        
        return None
    
    def _find_period_from_measurement(self, s: int, Q: int, N: int) -> Optional[int]:
        """測定結果から周期を推定"""
        if s == 0:
            return None
        
        # 連分数展開
        convergents = self._continued_fraction_convergents(s, Q)
        
        for _, r in convergents:
            if 0 < r < N and pow(2, r, N) == 1:  # 簡易チェック
                return r
        
        return None
    
    def _continued_fraction_convergents(self, num: int, den: int) -> List[Tuple[int, int]]:
        """連分数の収束分子列"""
        convergents = []
        h_prev, h_curr = 0, 1
        k_prev, k_curr = 1, 0
        
        while den != 0:
            a = num // den
            num, den = den, num - a * den
            
            h_prev, h_curr = h_curr, a * h_curr + h_prev
            k_prev, k_curr = k_curr, a * k_curr + k_prev
            
            if k_curr > 0:
                convergents.append((h_curr, k_curr))
        
        return convergents
    
    def _gcd(self, a: int, b: int) -> int:
        """最大公約数"""
        while b:
            a, b = b, a % b
        return a
    
    def _is_prime(self, n: int) -> bool:
        """素数判定"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True


# ============================================================
# 量子数値計算
# ============================================================

class QuantumNumerics:
    """
    量子インスパイアード数値計算
    """
    
    @staticmethod
    def quantum_sum(numbers: List[float]) -> float:
        """
        量子並列和
        """
        # 擬似量子：ペアワイズ和で並列性を模倣
        while len(numbers) > 1:
            new_numbers = []
            for i in range(0, len(numbers) - 1, 2):
                new_numbers.append(numbers[i] + numbers[i + 1])
            if len(numbers) % 2 == 1:
                new_numbers.append(numbers[-1])
            numbers = new_numbers
        return numbers[0] if numbers else 0.0
    
    @staticmethod
    def quantum_dot_product(a: np.ndarray, b: np.ndarray) -> float:
        """
        量子並列内積
        """
        products = a * b
        return QuantumNumerics.quantum_sum(products.tolist())
    
    @staticmethod
    def quantum_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        量子インスパイアード行列乗算
        """
        m, k = A.shape
        _, n = B.shape
        
        result = np.zeros((m, n))
        
        # 並列計算（概念的）
        for i in range(m):
            for j in range(n):
                result[i, j] = QuantumNumerics.quantum_dot_product(A[i], B[:, j])
        
        return result
    
    @staticmethod
    def quantum_fft(x: np.ndarray) -> np.ndarray:
        """
        量子フーリエ変換
        """
        N = len(x)
        
        if N <= 1:
            return x
        
        # 量子ビットで位相を計算
        n_qubits = int(np.ceil(np.log2(N)))
        
        result = np.zeros(N, dtype=complex)
        
        for k in range(N):
            for j in range(N):
                # 量子位相
                phase = 2 * np.pi * j * k / N
                qubit = PseudoQubit(correlation=np.cos(phase))
                
                # 量子干渉
                result[k] += x[j] * np.exp(-1j * phase)
        
        return result / np.sqrt(N)


# ============================================================
# 統合計算機インターフェース
# ============================================================

class QuantumCalculator:
    """
    量子インスパイアード計算機
    統合インターフェース
    """
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
    
    def optimize(self, func: Callable[[int], float], 
                 n_bits: int = 8, 
                 minimize: bool = True) -> Dict[str, Any]:
        """最適化問題を解く"""
        start = time.time()
        
        computer = QuantumParallelComputer(n_bits)
        result, value = computer.find_optimal(func, minimize=minimize)
        
        elapsed = time.time() - start
        
        output = {
            'type': 'optimization',
            'result': result,
            'value': value,
            'minimize': minimize,
            'time': elapsed
        }
        self.history.append(output)
        return output
    
    def search(self, database: List[Any], 
               condition: Callable[[Any], bool]) -> Dict[str, Any]:
        """データベース検索"""
        start = time.time()
        
        engine = QuantumSearchEngine(len(database))
        
        def oracle(idx: int) -> bool:
            return idx < len(database) and condition(database[idx])
        
        result_idx = engine.search(oracle)
        
        elapsed = time.time() - start
        
        output = {
            'type': 'search',
            'index': result_idx,
            'value': database[result_idx] if result_idx < len(database) else None,
            'found': result_idx < len(database) and condition(database[result_idx]),
            'time': elapsed
        }
        self.history.append(output)
        return output
    
    def factor(self, n: int) -> Dict[str, Any]:
        """素因数分解"""
        start = time.time()
        
        factorizer = QuantumFactorizer()
        factors = factorizer.factor(n)
        
        elapsed = time.time() - start
        
        output = {
            'type': 'factorization',
            'number': n,
            'factors': factors,
            'time': elapsed
        }
        self.history.append(output)
        return output
    
    def solve_max_cut(self, adjacency: np.ndarray) -> Dict[str, Any]:
        """最大カット問題"""
        start = time.time()
        
        optimizer = QuantumOptimizer(adjacency.shape[0])
        partition, cut_value = optimizer.solve_max_cut(adjacency)
        
        elapsed = time.time() - start
        
        output = {
            'type': 'max_cut',
            'partition': partition.tolist(),
            'cut_value': cut_value,
            'time': elapsed
        }
        self.history.append(output)
        return output
    
    def solve_tsp(self, distances: np.ndarray) -> Dict[str, Any]:
        """巡回セールスマン問題"""
        start = time.time()
        
        optimizer = QuantumOptimizer(distances.shape[0])
        tour, total_distance = optimizer.solve_tsp(distances)
        
        elapsed = time.time() - start
        
        output = {
            'type': 'tsp',
            'tour': tour,
            'distance': total_distance,
            'time': elapsed
        }
        self.history.append(output)
        return output
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
        """行列乗算"""
        start = time.time()
        
        result = QuantumNumerics.quantum_matrix_multiply(A, B)
        
        elapsed = time.time() - start
        
        output = {
            'type': 'matrix_multiply',
            'result': result,
            'time': elapsed
        }
        self.history.append(output)
        return output


# ============================================================
# デモ
# ============================================================

def demo():
    """量子計算機デモ"""
    print("=" * 70)
    print("  量子インスパイアード 高速計算機")
    print("  Quantum-Inspired High-Speed Calculator")
    print("=" * 70)
    
    calc = QuantumCalculator()
    
    # ============================================================
    # 1. 最適化問題
    # ============================================================
    print("\n" + "─" * 70)
    print("  1. 最適化問題 (関数の最小値探索)")
    print("─" * 70)
    
    def test_func(x: int) -> float:
        """テスト関数: (x - 42)^2"""
        return (x - 42) ** 2
    
    print("\n  問題: f(x) = (x - 42)² の最小値を探す (x: 0-255)")
    
    result = calc.optimize(test_func, n_bits=8, minimize=True)
    print(f"\n  結果: x = {result['result']}")
    print(f"  f(x) = {result['value']}")
    print(f"  正解: x = 42")
    print(f"  計算時間: {result['time']:.4f}秒")
    
    # ============================================================
    # 2. 量子検索
    # ============================================================
    print("\n" + "─" * 70)
    print("  2. 量子検索 (グローバー風アルゴリズム)")
    print("─" * 70)
    
    # データベース
    database = list(range(1000))
    target = 777
    
    print(f"\n  問題: {len(database)}個のデータから {target} を探す")
    print(f"  古典計算: O(N) = O({len(database)})")
    print(f"  量子計算: O(√N) ≈ O({int(np.sqrt(len(database)))})")
    
    result = calc.search(database, lambda x: x == target)
    
    print(f"\n  結果: インデックス {result['index']}, 値 = {result['value']}")
    print(f"  発見: {'✓' if result['found'] else '✗'}")
    print(f"  計算時間: {result['time']:.4f}秒")
    
    # ============================================================
    # 3. 素因数分解
    # ============================================================
    print("\n" + "─" * 70)
    print("  3. 素因数分解 (ショア風アルゴリズム)")
    print("─" * 70)
    
    numbers = [15, 21, 35, 91, 143, 1001]
    
    for n in numbers:
        result = calc.factor(n)
        factors_str = ' × '.join(map(str, result['factors']))
        print(f"  {n} = {factors_str} ({result['time']:.4f}秒)")
    
    # ============================================================
    # 4. 最大カット問題
    # ============================================================
    print("\n" + "─" * 70)
    print("  4. 最大カット問題 (量子アニーリング風)")
    print("─" * 70)
    
    # 5頂点のグラフ
    adjacency = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ])
    
    print("\n  グラフ (5頂点):")
    print("    0 -- 1 -- 3")
    print("     \\  |  / |")
    print("       2 -- 4")
    
    result = calc.solve_max_cut(adjacency)
    
    group_a = [i for i, p in enumerate(result['partition']) if p == 0]
    group_b = [i for i, p in enumerate(result['partition']) if p == 1]
    
    print(f"\n  分割結果:")
    print(f"    グループA: {group_a}")
    print(f"    グループB: {group_b}")
    print(f"  カットされた辺: {result['cut_value']}本")
    print(f"  計算時間: {result['time']:.4f}秒")
    
    # ============================================================
    # 5. 巡回セールスマン問題
    # ============================================================
    print("\n" + "─" * 70)
    print("  5. 巡回セールスマン問題 (TSP)")
    print("─" * 70)
    
    # 5都市の距離行列
    distances = np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 20],
        [20, 25, 30, 0, 15],
        [25, 30, 20, 15, 0]
    ])
    
    print("\n  5都市の巡回問題")
    
    result = calc.solve_tsp(distances)
    
    tour_str = ' → '.join(map(str, result['tour'] + [result['tour'][0]]))
    print(f"\n  最適ルート: {tour_str}")
    print(f"  総距離: {result['distance']:.1f}")
    print(f"  計算時間: {result['time']:.4f}秒")
    
    # ============================================================
    # 6. 速度比較
    # ============================================================
    print("\n" + "─" * 70)
    print("  6. 量子並列 vs 古典計算 速度比較")
    print("─" * 70)
    
    sizes = [100, 1000, 10000]
    
    print("\n  内積計算:")
    for size in sizes:
        a = np.random.rand(size)
        b = np.random.rand(size)
        
        # 量子風
        start = time.time()
        q_result = QuantumNumerics.quantum_dot_product(a, b)
        q_time = time.time() - start
        
        # NumPy
        start = time.time()
        np_result = np.dot(a, b)
        np_time = time.time() - start
        
        print(f"    サイズ {size:5d}: 量子風 {q_time:.6f}秒, NumPy {np_time:.6f}秒")
    
    # ============================================================
    # まとめ
    # ============================================================
    print("\n" + "=" * 70)
    print("  量子計算の利点")
    print("=" * 70)
    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │ 問題                  │ 古典計算    │ 量子計算              │
  ├─────────────────────────────────────────────────────────────┤
  │ 検索                  │ O(N)        │ O(√N)                 │
  │ 最適化                │ O(2^N)      │ O(√N) ~ O(poly(N))    │
  │ 素因数分解            │ O(exp(N))   │ O(poly(N))            │
  │ 組み合わせ問題        │ NP-hard     │ 量子加速可能          │
  └─────────────────────────────────────────────────────────────┘
  
  擬似量子ビットにより、これらの量子アルゴリズムの
  概念を古典コンピュータ上でシミュレート！
    """)
    
    print("=" * 70)
    print("  デモ完了！")
    print("=" * 70)


def interactive_mode():
    """対話モード"""
    print("\n" + "=" * 70)
    print("  量子高速計算機 - 対話モード")
    print("=" * 70)
    
    calc = QuantumCalculator()
    
    print("""
  コマンド:
    opt [式]       - 最適化 (例: opt (x-50)**2)
    search [値]    - 検索 (0-999から)
    factor [数]    - 素因数分解
    calc [式]      - 数式計算
    help           - ヘルプ
    quit           - 終了
    """)
    
    while True:
        try:
            user_input = input("\n  量子計算機> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'q', '終了']:
                print("  さようなら！")
                break
            
            if user_input.lower() == 'help':
                print("""
  ┌─────────────────────────────────────────────────────────┐
  │ コマンド一覧                                            │
  ├─────────────────────────────────────────────────────────┤
  │ opt [式]      最適化問題を解く                          │
  │               例: opt (x-50)**2                         │
  │               例: opt x**2 + 10*x + 25                  │
  │                                                         │
  │ search [値]   0-999から値を量子検索                     │
  │               例: search 777                            │
  │                                                         │
  │ factor [数]   素因数分解                                │
  │               例: factor 1001                           │
  │                                                         │
  │ calc [式]     数式を計算                                │
  │               例: calc 2**10                            │
  │               例: calc sqrt(2)                          │
  │                                                         │
  │ quit          終了                                      │
  └─────────────────────────────────────────────────────────┘
                """)
                continue
            
            # 最適化
            if user_input.lower().startswith('opt '):
                expr = user_input[4:].strip()
                try:
                    func = lambda x, e=expr: eval(e)
                    result = calc.optimize(func, n_bits=8, minimize=True)
                    print(f"\n  ⚡ 量子最適化結果:")
                    print(f"     x = {result['result']}")
                    print(f"     f(x) = {result['value']}")
                    print(f"     時間: {result['time']:.4f}秒")
                except Exception as e:
                    print(f"  エラー: {e}")
                continue
            
            # 検索
            if user_input.lower().startswith('search '):
                try:
                    target = int(user_input[7:].strip())
                    database = list(range(1000))
                    result = calc.search(database, lambda x: x == target)
                    status = "✓ 発見" if result['found'] else "✗ 未発見"
                    print(f"\n  ⚡ 量子検索結果: {status}")
                    print(f"     インデックス: {result['index']}")
                    print(f"     時間: {result['time']:.4f}秒")
                except Exception as e:
                    print(f"  エラー: {e}")
                continue
            
            # 素因数分解
            if user_input.lower().startswith('factor '):
                try:
                    n = int(user_input[7:].strip())
                    result = calc.factor(n)
                    factors_str = ' × '.join(map(str, result['factors']))
                    print(f"\n  ⚡ 量子素因数分解:")
                    print(f"     {n} = {factors_str}")
                    print(f"     時間: {result['time']:.4f}秒")
                except Exception as e:
                    print(f"  エラー: {e}")
                continue
            
            # 計算
            if user_input.lower().startswith('calc '):
                expr = user_input[5:].strip()
                try:
                    from math import sqrt, sin, cos, tan, pi, e, log, exp
                    result = eval(expr)
                    print(f"\n  結果: {result}")
                except Exception as e:
                    print(f"  エラー: {e}")
                continue
            
            print("  不明なコマンドです。'help' でヘルプを表示")
            
        except KeyboardInterrupt:
            print("\n  さようなら！")
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_mode()
    else:
        demo()

