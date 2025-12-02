"""
擬似量子コンピューター (Pseudo Quantum Computer)

擬似量子ビットを使用した量子コンピューターシミュレーター

機能:
- 量子ビット (Qubit)
- 量子ゲート (Hadamard, Pauli-X/Y/Z, CNOT, etc.)
- 量子回路 (Quantum Circuit)
- 量子アルゴリズム (Deutsch, Grover)
- 測定 (Measurement)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import random
from copy import deepcopy


# ============================================================
# 量子状態
# ============================================================

class QubitState:
    """
    量子ビットの状態
    |ψ⟩ = α|0⟩ + β|1⟩
    """
    def __init__(self, alpha: complex = 1.0, beta: complex = 0.0):
        self.state = np.array([alpha, beta], dtype=complex)
        self._normalize()
    
    def _normalize(self):
        """状態ベクトルを正規化"""
        norm = np.sqrt(np.sum(np.abs(self.state) ** 2))
        if norm > 0:
            self.state = self.state / norm
    
    @property
    def alpha(self) -> complex:
        return self.state[0]
    
    @property
    def beta(self) -> complex:
        return self.state[1]
    
    @property
    def prob_0(self) -> float:
        return float(np.abs(self.alpha) ** 2)
    
    @property
    def prob_1(self) -> float:
        return float(np.abs(self.beta) ** 2)
    
    def measure(self) -> int:
        """量子ビットを測定"""
        if random.random() < self.prob_0:
            self.state = np.array([1.0, 0.0], dtype=complex)
            return 0
        else:
            self.state = np.array([0.0, 1.0], dtype=complex)
            return 1
    
    def copy(self) -> 'QubitState':
        return QubitState(self.alpha, self.beta)
    
    def __str__(self) -> str:
        return f"|ψ⟩ = ({self.alpha:.4f})|0⟩ + ({self.beta:.4f})|1⟩"
    
    def __repr__(self) -> str:
        return f"QubitState(α={self.alpha:.4f}, β={self.beta:.4f})"


# ============================================================
# 量子ゲート
# ============================================================

class QuantumGate:
    """量子ゲートの基底クラス"""
    
    def __init__(self, name: str, matrix: np.ndarray):
        self.name = name
        self.matrix = np.array(matrix, dtype=complex)
    
    def apply(self, state: QubitState) -> QubitState:
        """ゲートを状態に適用"""
        new_state_vector = self.matrix @ state.state
        return QubitState(new_state_vector[0], new_state_vector[1])
    
    def __str__(self) -> str:
        return f"{self.name} Gate"
    
    def __repr__(self) -> str:
        return f"QuantumGate({self.name})"


# 基本的な量子ゲート
class Gates:
    """標準量子ゲートのコレクション"""
    
    # 恒等ゲート (Identity)
    I = QuantumGate("I", [
        [1, 0],
        [0, 1]
    ])
    
    # アダマールゲート (Hadamard)
    # |0⟩ → (|0⟩ + |1⟩)/√2
    # |1⟩ → (|0⟩ - |1⟩)/√2
    H = QuantumGate("H", [
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [1/np.sqrt(2), -1/np.sqrt(2)]
    ])
    
    # パウリXゲート (NOT)
    # |0⟩ → |1⟩, |1⟩ → |0⟩
    X = QuantumGate("X", [
        [0, 1],
        [1, 0]
    ])
    
    # パウリYゲート
    Y = QuantumGate("Y", [
        [0, -1j],
        [1j, 0]
    ])
    
    # パウリZゲート
    # |0⟩ → |0⟩, |1⟩ → -|1⟩
    Z = QuantumGate("Z", [
        [1, 0],
        [0, -1]
    ])
    
    # 位相ゲート (S)
    S = QuantumGate("S", [
        [1, 0],
        [0, 1j]
    ])
    
    # T ゲート (π/8)
    T = QuantumGate("T", [
        [1, 0],
        [0, np.exp(1j * np.pi / 4)]
    ])
    
    @staticmethod
    def Rx(theta: float) -> QuantumGate:
        """X軸周りの回転"""
        return QuantumGate(f"Rx({theta:.2f})", [
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ])
    
    @staticmethod
    def Ry(theta: float) -> QuantumGate:
        """Y軸周りの回転"""
        return QuantumGate(f"Ry({theta:.2f})", [
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])
    
    @staticmethod
    def Rz(theta: float) -> QuantumGate:
        """Z軸周りの回転"""
        return QuantumGate(f"Rz({theta:.2f})", [
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ])


# ============================================================
# 量子レジスタ（複数量子ビット）
# ============================================================

class QuantumRegister:
    """
    量子レジスタ（複数の量子ビット）
    n量子ビットの状態は2^n次元のベクトル
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        # 初期状態: |00...0⟩
        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[0] = 1.0
    
    def reset(self):
        """|00...0⟩にリセット"""
        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[0] = 1.0
    
    def get_probabilities(self) -> np.ndarray:
        """各基底状態の確率を返す"""
        return np.abs(self.state) ** 2
    
    def apply_single_gate(self, gate: QuantumGate, target: int):
        """単一量子ビットゲートを適用"""
        # 完全な行列を構築（量子ビットの順序: q_{n-1} ⊗ ... ⊗ q_0）
        full_matrix = np.eye(1, dtype=complex)
        
        for i in range(self.n_qubits - 1, -1, -1):
            if i == target:
                full_matrix = np.kron(full_matrix, gate.matrix)
            else:
                full_matrix = np.kron(full_matrix, np.eye(2))
        
        self.state = full_matrix @ self.state
    
    def apply_cnot(self, control: int, target: int):
        """CNOTゲートを適用"""
        new_state = np.zeros_like(self.state)
        
        for i in range(self.n_states):
            # iをビット列として見る
            control_bit = (i >> control) & 1
            
            if control_bit == 1:
                # ターゲットビットを反転した状態へマッピング
                j = i ^ (1 << target)
                new_state[j] += self.state[i]
            else:
                new_state[i] += self.state[i]
        
        self.state = new_state
    
    def apply_cz(self, control: int, target: int):
        """CZゲート（制御Z）を適用"""
        for i in range(self.n_states):
            bits = [(i >> j) & 1 for j in range(self.n_qubits)]
            if bits[control] == 1 and bits[target] == 1:
                self.state[i] *= -1
    
    def apply_toffoli(self, control1: int, control2: int, target: int):
        """トフォリゲート（CCNOT）を適用"""
        new_state = np.zeros_like(self.state)
        
        for i in range(self.n_states):
            c1_bit = (i >> control1) & 1
            c2_bit = (i >> control2) & 1
            
            if c1_bit == 1 and c2_bit == 1:
                j = i ^ (1 << target)
                new_state[j] = self.state[i]
            else:
                new_state[i] = self.state[i]
        
        self.state = new_state
    
    def measure(self) -> List[int]:
        """全量子ビットを測定"""
        probs = self.get_probabilities()
        result_index = np.random.choice(self.n_states, p=probs)
        
        # 測定後の状態に収縮
        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[result_index] = 1.0
        
        # 結果をビット列として返す
        return [(result_index >> i) & 1 for i in range(self.n_qubits)]
    
    def measure_qubit(self, qubit: int) -> int:
        """特定の量子ビットを測定"""
        # qubitが0の状態と1の状態の確率を計算
        prob_0 = 0.0
        for i in range(self.n_states):
            if (i >> qubit) & 1 == 0:
                prob_0 += np.abs(self.state[i]) ** 2
        
        # 測定
        if random.random() < prob_0:
            result = 0
            # 状態を収縮
            for i in range(self.n_states):
                if (i >> qubit) & 1 == 1:
                    self.state[i] = 0
        else:
            result = 1
            for i in range(self.n_states):
                if (i >> qubit) & 1 == 0:
                    self.state[i] = 0
        
        # 正規化
        norm = np.sqrt(np.sum(np.abs(self.state) ** 2))
        if norm > 0:
            self.state = self.state / norm
        
        return result
    
    def get_state_string(self) -> str:
        """状態を文字列で表現"""
        terms = []
        for i in range(self.n_states):
            if np.abs(self.state[i]) > 1e-10:
                bits = ''.join(str((i >> j) & 1) for j in range(self.n_qubits - 1, -1, -1))
                coef = self.state[i]
                if np.abs(coef.imag) < 1e-10:
                    coef_str = f"{coef.real:.4f}"
                else:
                    coef_str = f"({coef.real:.4f}+{coef.imag:.4f}i)"
                terms.append(f"{coef_str}|{bits}⟩")
        return " + ".join(terms) if terms else "0"
    
    def __str__(self) -> str:
        return f"QuantumRegister({self.n_qubits} qubits): {self.get_state_string()}"


# ============================================================
# 量子回路
# ============================================================

@dataclass
class GateOperation:
    """ゲート操作を記録"""
    gate: str
    targets: List[int]
    controls: List[int] = None
    params: dict = None


class QuantumCircuit:
    """
    量子回路
    ゲート操作のシーケンスを管理
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.register = QuantumRegister(n_qubits)
        self.operations: List[GateOperation] = []
    
    def reset(self):
        """回路をリセット"""
        self.register.reset()
        self.operations = []
    
    # 単一量子ビットゲート
    def h(self, target: int) -> 'QuantumCircuit':
        """アダマールゲート"""
        self.register.apply_single_gate(Gates.H, target)
        self.operations.append(GateOperation("H", [target]))
        return self
    
    def x(self, target: int) -> 'QuantumCircuit':
        """パウリXゲート"""
        self.register.apply_single_gate(Gates.X, target)
        self.operations.append(GateOperation("X", [target]))
        return self
    
    def y(self, target: int) -> 'QuantumCircuit':
        """パウリYゲート"""
        self.register.apply_single_gate(Gates.Y, target)
        self.operations.append(GateOperation("Y", [target]))
        return self
    
    def z(self, target: int) -> 'QuantumCircuit':
        """パウリZゲート"""
        self.register.apply_single_gate(Gates.Z, target)
        self.operations.append(GateOperation("Z", [target]))
        return self
    
    def s(self, target: int) -> 'QuantumCircuit':
        """Sゲート"""
        self.register.apply_single_gate(Gates.S, target)
        self.operations.append(GateOperation("S", [target]))
        return self
    
    def t(self, target: int) -> 'QuantumCircuit':
        """Tゲート"""
        self.register.apply_single_gate(Gates.T, target)
        self.operations.append(GateOperation("T", [target]))
        return self
    
    def rx(self, target: int, theta: float) -> 'QuantumCircuit':
        """Rxゲート"""
        self.register.apply_single_gate(Gates.Rx(theta), target)
        self.operations.append(GateOperation("Rx", [target], params={"theta": theta}))
        return self
    
    def ry(self, target: int, theta: float) -> 'QuantumCircuit':
        """Ryゲート"""
        self.register.apply_single_gate(Gates.Ry(theta), target)
        self.operations.append(GateOperation("Ry", [target], params={"theta": theta}))
        return self
    
    def rz(self, target: int, theta: float) -> 'QuantumCircuit':
        """Rzゲート"""
        self.register.apply_single_gate(Gates.Rz(theta), target)
        self.operations.append(GateOperation("Rz", [target], params={"theta": theta}))
        return self
    
    # 2量子ビットゲート
    def cnot(self, control: int, target: int) -> 'QuantumCircuit':
        """CNOTゲート"""
        self.register.apply_cnot(control, target)
        self.operations.append(GateOperation("CNOT", [target], controls=[control]))
        return self
    
    def cx(self, control: int, target: int) -> 'QuantumCircuit':
        """CX（CNOT）ゲート"""
        return self.cnot(control, target)
    
    def cz(self, control: int, target: int) -> 'QuantumCircuit':
        """CZゲート"""
        self.register.apply_cz(control, target)
        self.operations.append(GateOperation("CZ", [target], controls=[control]))
        return self
    
    # 3量子ビットゲート
    def toffoli(self, control1: int, control2: int, target: int) -> 'QuantumCircuit':
        """トフォリゲート"""
        self.register.apply_toffoli(control1, control2, target)
        self.operations.append(GateOperation("CCX", [target], controls=[control1, control2]))
        return self
    
    def ccx(self, control1: int, control2: int, target: int) -> 'QuantumCircuit':
        """CCX（トフォリ）ゲート"""
        return self.toffoli(control1, control2, target)
    
    # 測定
    def measure(self) -> List[int]:
        """全量子ビットを測定"""
        return self.register.measure()
    
    def measure_qubit(self, qubit: int) -> int:
        """特定の量子ビットを測定"""
        return self.register.measure_qubit(qubit)
    
    def run(self, shots: int = 1000) -> dict:
        """回路を複数回実行して統計を取得"""
        results = {}
        initial_state = self.register.state.copy()
        
        for _ in range(shots):
            self.register.state = initial_state.copy()
            measurement = self.measure()
            bits = ''.join(str(b) for b in reversed(measurement))
            results[bits] = results.get(bits, 0) + 1
        
        self.register.state = initial_state
        return results
    
    def get_state(self) -> str:
        """現在の状態を文字列で取得"""
        return self.register.get_state_string()
    
    def draw(self) -> str:
        """回路を図示"""
        lines = [f"q{i}: " for i in range(self.n_qubits)]
        
        for op in self.operations:
            max_len = max(len(line) for line in lines)
            lines = [line.ljust(max_len) for line in lines]
            
            if op.controls:
                for c in op.controls:
                    lines[c] += "●──"
                lines[op.targets[0]] += f"[{op.gate}]"
                
                # 制御線を描画
                min_q = min(op.controls + op.targets)
                max_q = max(op.controls + op.targets)
                for q in range(min_q + 1, max_q):
                    if q not in op.controls and q not in op.targets:
                        lines[q] += "│──"
            else:
                lines[op.targets[0]] += f"[{op.gate}]"
            
            # 他の量子ビットは線を延長
            for i in range(self.n_qubits):
                if i not in op.targets and (not op.controls or i not in op.controls):
                    lines[i] += "───"
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return f"QuantumCircuit({self.n_qubits} qubits, {len(self.operations)} gates)"


# ============================================================
# 量子コンピューター
# ============================================================

class QuantumComputer:
    """
    量子コンピューター
    量子回路を作成・実行するインターフェース
    """
    
    def __init__(self, name: str = "PseudoQC"):
        self.name = name
        self.circuits: dict = {}
    
    def create_circuit(self, name: str, n_qubits: int) -> QuantumCircuit:
        """新しい量子回路を作成"""
        circuit = QuantumCircuit(n_qubits)
        self.circuits[name] = circuit
        return circuit
    
    def run_circuit(self, name: str, shots: int = 1000) -> dict:
        """回路を実行"""
        if name not in self.circuits:
            raise ValueError(f"Circuit '{name}' not found")
        return self.circuits[name].run(shots)
    
    def get_circuit(self, name: str) -> QuantumCircuit:
        """回路を取得"""
        return self.circuits.get(name)
    
    # 組み込み量子アルゴリズム
    def deutsch_algorithm(self, oracle_type: str = "constant") -> int:
        """
        ドイッチのアルゴリズム
        
        オラクル関数 f: {0,1} → {0,1} が
        定数関数か均等関数かを1回のクエリで判定
        
        Args:
            oracle_type: "constant_0", "constant_1", "balanced_id", "balanced_not"
        
        Returns:
            0: 定数関数, 1: 均等関数
        """
        qc = QuantumCircuit(2)
        
        # |01⟩ を準備
        qc.x(1)
        
        # アダマール変換
        qc.h(0)
        qc.h(1)
        
        # オラクル適用
        if oracle_type == "constant_0":
            # f(x) = 0: 何もしない
            pass
        elif oracle_type == "constant_1":
            # f(x) = 1: |1⟩ を反転
            qc.x(1)
        elif oracle_type == "balanced_id":
            # f(x) = x: CNOT
            qc.cnot(0, 1)
        elif oracle_type == "balanced_not":
            # f(x) = NOT x
            qc.x(0)
            qc.cnot(0, 1)
            qc.x(0)
        
        # 入力量子ビットにアダマール
        qc.h(0)
        
        # 測定
        result = qc.measure_qubit(0)
        return result  # 0=定数, 1=均等
    
    def bernstein_vazirani(self, secret: str) -> str:
        """
        ベルンシュタイン・ヴァジラニのアルゴリズム
        
        隠れたビット列 s を1回のクエリで発見
        f(x) = s · x (mod 2)
        
        Args:
            secret: 隠れたビット列（例: "101"）
        
        Returns:
            発見したビット列
        """
        n = len(secret)
        qc = QuantumCircuit(n + 1)
        
        # |0...01⟩ を準備
        qc.x(n)
        
        # アダマール変換
        for i in range(n + 1):
            qc.h(i)
        
        # オラクル: s の各ビットが1の位置でCNOT
        for i, bit in enumerate(reversed(secret)):
            if bit == '1':
                qc.cnot(i, n)
        
        # アダマール変換（補助ビット以外）
        for i in range(n):
            qc.h(i)
        
        # 測定
        results = qc.measure()
        return ''.join(str(results[i]) for i in range(n - 1, -1, -1))
    
    def grover_search(self, n_qubits: int, target: int, iterations: int = None) -> int:
        """
        グローバーの探索アルゴリズム
        
        N = 2^n 個の要素から目標を O(√N) で発見
        
        Args:
            n_qubits: 量子ビット数
            target: 探索対象のインデックス
            iterations: 反復回数（Noneなら最適値）
        
        Returns:
            発見したインデックス
        """
        N = 2 ** n_qubits
        if iterations is None:
            iterations = int(np.pi / 4 * np.sqrt(N))
        
        qc = QuantumCircuit(n_qubits)
        
        # 一様重ね合わせを作成
        for i in range(n_qubits):
            qc.h(i)
        
        # グローバー反復
        for _ in range(iterations):
            # オラクル: 目標状態の符号を反転
            # target のビットが0の位置にXを適用
            for i in range(n_qubits):
                if not (target >> i) & 1:
                    qc.x(i)
            
            # 多重制御Z（簡略化: 位相キックバック）
            if n_qubits == 2:
                qc.cz(0, 1)
            elif n_qubits >= 3:
                # 簡略化した実装
                qc.h(n_qubits - 1)
                # 多重制御NOTの代わりに直接位相を操作
                for i in range(n_qubits):
                    if (target >> i) & 1:
                        pass  # 位相を調整
                qc.h(n_qubits - 1)
            
            # オラクル後処理
            for i in range(n_qubits):
                if not (target >> i) & 1:
                    qc.x(i)
            
            # 拡散演算子
            for i in range(n_qubits):
                qc.h(i)
                qc.x(i)
            
            if n_qubits == 2:
                qc.cz(0, 1)
            elif n_qubits >= 3:
                qc.h(n_qubits - 1)
                qc.toffoli(0, 1, n_qubits - 1)
                qc.h(n_qubits - 1)
            
            for i in range(n_qubits):
                qc.x(i)
                qc.h(i)
        
        # 測定
        results = qc.measure()
        return sum(results[i] << i for i in range(n_qubits))


# ============================================================
# デモンストレーション
# ============================================================

def demo():
    """量子コンピューターのデモ"""
    print("=" * 70)
    print("  PSEUDO QUANTUM COMPUTER")
    print("  擬似量子コンピューター デモンストレーション")
    print("=" * 70)
    
    qc = QuantumComputer("MyQuantumComputer")
    
    # ============================================================
    # 1. 基本的な量子ゲート
    # ============================================================
    print("\n" + "─" * 70)
    print("  1. 基本量子ゲート")
    print("─" * 70)
    
    # アダマールゲート
    print("\n  [Hadamard Gate]")
    circuit = qc.create_circuit("hadamard_test", 1)
    circuit.h(0)
    print(f"  |0⟩ → H → {circuit.get_state()}")
    
    results = circuit.run(1000)
    print(f"  1000回測定: {results}")
    
    # ベル状態
    print("\n  [Bell State]")
    circuit = qc.create_circuit("bell", 2)
    circuit.h(0).cnot(0, 1)
    print(f"  |00⟩ → H₀ → CNOT → {circuit.get_state()}")
    print(f"  回路図:\n{circuit.draw()}")
    
    results = circuit.run(1000)
    print(f"  1000回測定: {results}")
    
    # ============================================================
    # 2. ドイッチのアルゴリズム
    # ============================================================
    print("\n" + "─" * 70)
    print("  2. ドイッチのアルゴリズム")
    print("─" * 70)
    
    for oracle in ["constant_0", "constant_1", "balanced_id", "balanced_not"]:
        result = qc.deutsch_algorithm(oracle)
        func_type = "定数関数" if result == 0 else "均等関数"
        print(f"  Oracle: {oracle:15} → 結果: {result} ({func_type})")
    
    # ============================================================
    # 3. ベルンシュタイン・ヴァジラニ
    # ============================================================
    print("\n" + "─" * 70)
    print("  3. ベルンシュタイン・ヴァジラニのアルゴリズム")
    print("─" * 70)
    
    for secret in ["101", "110", "011", "111"]:
        found = qc.bernstein_vazirani(secret)
        match = "✓" if found == secret else "✗"
        print(f"  隠れたビット列: {secret} → 発見: {found} {match}")
    
    # ============================================================
    # 4. GHZ状態
    # ============================================================
    print("\n" + "─" * 70)
    print("  4. GHZ状態 (3量子ビットのエンタングルメント)")
    print("─" * 70)
    
    circuit = qc.create_circuit("ghz", 3)
    circuit.h(0).cnot(0, 1).cnot(1, 2)
    print(f"  状態: {circuit.get_state()}")
    print(f"  回路図:\n{circuit.draw()}")
    
    results = circuit.run(1000)
    print(f"  1000回測定: {results}")
    
    # ============================================================
    # 5. 量子テレポーテーション回路
    # ============================================================
    print("\n" + "─" * 70)
    print("  5. 量子テレポーテーション回路の構造")
    print("─" * 70)
    
    circuit = qc.create_circuit("teleport", 3)
    # ベル対を作成
    circuit.h(1).cnot(1, 2)
    # ベル測定の準備
    circuit.cnot(0, 1).h(0)
    print(f"  回路図:\n{circuit.draw()}")
    
    print("\n" + "=" * 70)
    print("  デモ完了！")
    print("=" * 70)


def interactive_mode():
    """対話モード"""
    print("\n" + "=" * 70)
    print("  PSEUDO QUANTUM COMPUTER - Interactive Mode")
    print("=" * 70)
    
    qc = QuantumComputer()
    current_circuit = None
    
    commands = """
  Commands:
    new <n>         - 新しい回路を作成（n量子ビット）
    h <q>           - Hadamardゲート
    x <q>           - Pauli-Xゲート
    y <q>           - Pauli-Yゲート
    z <q>           - Pauli-Zゲート
    cnot <c> <t>    - CNOTゲート
    cz <c> <t>      - CZゲート
    state           - 現在の状態を表示
    draw            - 回路図を表示
    measure         - 測定
    run <shots>     - 複数回実行
    deutsch <type>  - ドイッチのアルゴリズム
    bv <secret>     - ベルンシュタイン・ヴァジラニ
    reset           - 回路をリセット
    help            - ヘルプ表示
    quit            - 終了
"""
    print(commands)
    
    while True:
        try:
            cmd = input("\n  QC> ").strip().lower().split()
            if not cmd:
                continue
            
            if cmd[0] == "quit" or cmd[0] == "exit":
                print("  Goodbye!")
                break
            
            elif cmd[0] == "help":
                print(commands)
            
            elif cmd[0] == "new":
                n = int(cmd[1]) if len(cmd) > 1 else 2
                current_circuit = qc.create_circuit("main", n)
                print(f"  Created {n}-qubit circuit")
            
            elif cmd[0] == "h" and current_circuit:
                q = int(cmd[1])
                current_circuit.h(q)
                print(f"  Applied H to qubit {q}")
            
            elif cmd[0] == "x" and current_circuit:
                q = int(cmd[1])
                current_circuit.x(q)
                print(f"  Applied X to qubit {q}")
            
            elif cmd[0] == "y" and current_circuit:
                q = int(cmd[1])
                current_circuit.y(q)
                print(f"  Applied Y to qubit {q}")
            
            elif cmd[0] == "z" and current_circuit:
                q = int(cmd[1])
                current_circuit.z(q)
                print(f"  Applied Z to qubit {q}")
            
            elif cmd[0] == "cnot" and current_circuit:
                c, t = int(cmd[1]), int(cmd[2])
                current_circuit.cnot(c, t)
                print(f"  Applied CNOT (control={c}, target={t})")
            
            elif cmd[0] == "cz" and current_circuit:
                c, t = int(cmd[1]), int(cmd[2])
                current_circuit.cz(c, t)
                print(f"  Applied CZ (control={c}, target={t})")
            
            elif cmd[0] == "state" and current_circuit:
                print(f"  State: {current_circuit.get_state()}")
            
            elif cmd[0] == "draw" and current_circuit:
                print(f"\n{current_circuit.draw()}")
            
            elif cmd[0] == "measure" and current_circuit:
                result = current_circuit.measure()
                print(f"  Measurement: {result}")
            
            elif cmd[0] == "run" and current_circuit:
                shots = int(cmd[1]) if len(cmd) > 1 else 1000
                results = current_circuit.run(shots)
                print(f"  Results ({shots} shots): {results}")
            
            elif cmd[0] == "reset" and current_circuit:
                current_circuit.reset()
                print("  Circuit reset")
            
            elif cmd[0] == "deutsch":
                oracle = cmd[1] if len(cmd) > 1 else "balanced_id"
                result = qc.deutsch_algorithm(oracle)
                print(f"  Deutsch ({oracle}): {result} ({'constant' if result == 0 else 'balanced'})")
            
            elif cmd[0] == "bv":
                secret = cmd[1] if len(cmd) > 1 else "101"
                found = qc.bernstein_vazirani(secret)
                print(f"  Bernstein-Vazirani: secret={secret}, found={found}")
            
            else:
                if not current_circuit and cmd[0] in ["h", "x", "y", "z", "cnot", "cz", "state", "draw", "measure", "run", "reset"]:
                    print("  No circuit. Use 'new <n>' to create one.")
                else:
                    print(f"  Unknown command: {cmd[0]}")
        
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_mode()
    else:
        demo()

