"""
量子化学シミュレーター

擬似量子ビットを使った分子・原子シミュレーション
- 原子軌道の可視化
- 分子結合シミュレーション
- 電子の確率分布
- 量子トンネル効果
- 化学反応シミュレーション
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass
from pseudo_qubit import PseudoQubit


# ============================================================
# 量子粒子
# ============================================================

@dataclass
class QuantumParticle:
    """量子粒子（電子など）"""
    x: float
    y: float
    z: float
    spin: int  # +1 or -1
    energy: float
    qubit: PseudoQubit
    
    def measure_position(self) -> Tuple[float, float, float]:
        """位置を測定（量子揺らぎあり）"""
        # 量子的な不確定性を追加
        uncertainty = 0.1 * (1 - abs(self.qubit.correlation))
        return (
            self.x + np.random.normal(0, uncertainty),
            self.y + np.random.normal(0, uncertainty),
            self.z + np.random.normal(0, uncertainty)
        )
    
    def measure_spin(self) -> int:
        """スピンを測定"""
        result = self.qubit.measure()
        return 1 if result == 0 else -1


# ============================================================
# 原子軌道
# ============================================================

class AtomicOrbital:
    """原子軌道のシミュレーション"""
    
    def __init__(self, n: int, l: int, m: int):
        """
        n: 主量子数 (1, 2, 3, ...)
        l: 方位量子数 (0=s, 1=p, 2=d, ...)
        m: 磁気量子数 (-l to +l)
        """
        self.n = n
        self.l = l
        self.m = m
        self.orbital_names = {
            (1, 0, 0): '1s',
            (2, 0, 0): '2s',
            (2, 1, -1): '2p_x',
            (2, 1, 0): '2p_z',
            (2, 1, 1): '2p_y',
            (3, 0, 0): '3s',
            (3, 1, -1): '3p_x',
            (3, 1, 0): '3p_z',
            (3, 1, 1): '3p_y',
            (3, 2, -2): '3d_xy',
            (3, 2, -1): '3d_xz',
            (3, 2, 0): '3d_z2',
            (3, 2, 1): '3d_yz',
            (3, 2, 2): '3d_x2-y2',
        }
    
    @property
    def name(self) -> str:
        return self.orbital_names.get((self.n, self.l, self.m), f'{self.n}{self.l}{self.m}')
    
    def wave_function(self, r: float, theta: float, phi: float) -> complex:
        """波動関数（簡略化版）"""
        # 動径部分
        a0 = 1.0  # ボーア半径（正規化）
        
        if self.n == 1 and self.l == 0:  # 1s
            R = 2 * np.exp(-r / a0)
        elif self.n == 2 and self.l == 0:  # 2s
            R = (1 / (2 * np.sqrt(2))) * (2 - r / a0) * np.exp(-r / (2 * a0))
        elif self.n == 2 and self.l == 1:  # 2p
            R = (1 / (2 * np.sqrt(6))) * (r / a0) * np.exp(-r / (2 * a0))
        elif self.n == 3 and self.l == 0:  # 3s
            R = (1 / (9 * np.sqrt(3))) * (6 - 6 * r / a0 + (r / a0) ** 2) * np.exp(-r / (3 * a0))
        else:
            R = np.exp(-r / (self.n * a0))
        
        # 角度部分（球面調和関数の簡略化）
        if self.l == 0:  # s軌道
            Y = 1 / np.sqrt(4 * np.pi)
        elif self.l == 1:  # p軌道
            if self.m == 0:
                Y = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
            elif self.m == 1:
                Y = -np.sqrt(3 / (8 * np.pi)) * np.sin(theta) * np.exp(1j * phi)
            else:
                Y = np.sqrt(3 / (8 * np.pi)) * np.sin(theta) * np.exp(-1j * phi)
        else:
            Y = 1.0
        
        return R * Y
    
    def probability_density(self, x: float, y: float, z: float) -> float:
        """確率密度"""
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-10
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        
        psi = self.wave_function(r, theta, phi)
        return np.abs(psi) ** 2
    
    def sample_position(self, n_samples: int = 1000) -> np.ndarray:
        """電子位置をサンプリング"""
        positions = []
        max_r = self.n * 5
        
        # メトロポリス-ヘイスティングス法
        x, y, z = 0.0, 0.0, 1.0
        
        for _ in range(n_samples * 10):
            # 提案分布
            x_new = x + np.random.normal(0, 0.5)
            y_new = y + np.random.normal(0, 0.5)
            z_new = z + np.random.normal(0, 0.5)
            
            # 受理確率
            p_old = self.probability_density(x, y, z)
            p_new = self.probability_density(x_new, y_new, z_new)
            
            if np.random.random() < min(1, p_new / (p_old + 1e-10)):
                x, y, z = x_new, y_new, z_new
            
            if len(positions) < n_samples:
                positions.append([x, y, z])
        
        return np.array(positions[:n_samples])


# ============================================================
# 分子
# ============================================================

class Molecule:
    """分子モデル"""
    
    def __init__(self, name: str):
        self.name = name
        self.atoms: List[Dict] = []
        self.bonds: List[Tuple[int, int, int]] = []  # (atom1, atom2, bond_order)
        self.electrons: List[QuantumParticle] = []
    
    def add_atom(self, symbol: str, x: float, y: float, z: float):
        """原子を追加"""
        atom_data = {
            'H': {'number': 1, 'mass': 1.008, 'color': 'white', 'radius': 0.3},
            'C': {'number': 6, 'mass': 12.011, 'color': 'gray', 'radius': 0.5},
            'N': {'number': 7, 'mass': 14.007, 'color': 'blue', 'radius': 0.5},
            'O': {'number': 8, 'mass': 15.999, 'color': 'red', 'radius': 0.5},
            'S': {'number': 16, 'mass': 32.065, 'color': 'yellow', 'radius': 0.6},
        }
        
        data = atom_data.get(symbol, {'number': 1, 'mass': 1, 'color': 'gray', 'radius': 0.4})
        self.atoms.append({
            'symbol': symbol,
            'x': x, 'y': y, 'z': z,
            **data
        })
    
    def add_bond(self, atom1: int, atom2: int, order: int = 1):
        """結合を追加"""
        self.bonds.append((atom1, atom2, order))
    
    def total_electrons(self) -> int:
        """総電子数"""
        return sum(a['number'] for a in self.atoms)
    
    def simulate_electrons(self):
        """電子をシミュレート"""
        self.electrons = []
        
        for atom in self.atoms:
            n_electrons = atom['number']
            for i in range(n_electrons):
                # 量子ビットで電子状態を表現
                correlation = np.random.uniform(-0.5, 0.5)
                qubit = PseudoQubit(correlation=correlation)
                
                # 原子周りにランダム配置
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                r = np.random.exponential(0.5)
                
                electron = QuantumParticle(
                    x=atom['x'] + r * np.sin(phi) * np.cos(theta),
                    y=atom['y'] + r * np.sin(phi) * np.sin(theta),
                    z=atom['z'] + r * np.cos(phi),
                    spin=1 if i % 2 == 0 else -1,
                    energy=-13.6 / (i + 1),  # 簡略化したエネルギー
                    qubit=qubit
                )
                self.electrons.append(electron)


# ============================================================
# 量子トンネル効果
# ============================================================

class QuantumTunneling:
    """量子トンネル効果シミュレーション"""
    
    def __init__(self, barrier_height: float = 1.0, barrier_width: float = 1.0):
        self.barrier_height = barrier_height
        self.barrier_width = barrier_width
    
    def transmission_probability(self, particle_energy: float, mass: float = 1.0) -> float:
        """透過確率を計算"""
        if particle_energy >= self.barrier_height:
            return 1.0
        
        # 量子トンネル確率（WKB近似）
        kappa = np.sqrt(2 * mass * (self.barrier_height - particle_energy))
        T = np.exp(-2 * kappa * self.barrier_width)
        
        return min(T, 1.0)
    
    def simulate_particle(self, energy: float, n_attempts: int = 100) -> Tuple[int, int]:
        """粒子のトンネリングをシミュレート"""
        T = self.transmission_probability(energy)
        
        # 擬似量子ビットで透過/反射を決定
        transmitted = 0
        reflected = 0
        
        for _ in range(n_attempts):
            # 相関係数をエネルギー比から設定
            correlation = 2 * T - 1  # T=0 -> r=-1, T=1 -> r=1
            qubit = PseudoQubit(correlation=correlation)
            
            if qubit.measure() == 0:  # |0⟩ = 透過
                transmitted += 1
            else:  # |1⟩ = 反射
                reflected += 1
        
        return transmitted, reflected


# ============================================================
# 化学反応シミュレーター
# ============================================================

class ChemicalReaction:
    """化学反応シミュレーター"""
    
    def __init__(self, name: str, activation_energy: float):
        self.name = name
        self.activation_energy = activation_energy
        self.temperature = 300  # K
        self.k_B = 8.617e-5  # eV/K (ボルツマン定数)
    
    def reaction_probability(self) -> float:
        """反応確率（アレニウス式）"""
        return np.exp(-self.activation_energy / (self.k_B * self.temperature))
    
    def simulate_reaction(self, n_molecules: int = 100) -> Tuple[int, int]:
        """反応をシミュレート"""
        prob = self.reaction_probability()
        correlation = 2 * prob - 1
        
        reacted = 0
        unreacted = 0
        
        for _ in range(n_molecules):
            qubit = PseudoQubit(correlation=correlation)
            if qubit.measure() == 0:
                reacted += 1
            else:
                unreacted += 1
        
        return reacted, unreacted


# ============================================================
# 可視化
# ============================================================

class QuantumChemistryVisualizer:
    """量子化学可視化"""
    
    @staticmethod
    def plot_orbital(orbital: AtomicOrbital, resolution: int = 50):
        """原子軌道を3Dプロット"""
        fig = plt.figure(figsize=(12, 5))
        fig.patch.set_facecolor('#0a0a0f')
        
        # 3D等値面
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_facecolor('#0a0a0f')
        
        # サンプリング
        positions = orbital.sample_position(2000)
        
        colors = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
        
        ax1.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=colors, cmap='plasma', s=2, alpha=0.5
        )
        
        ax1.set_xlabel('X', color='white')
        ax1.set_ylabel('Y', color='white')
        ax1.set_zlabel('Z', color='white')
        ax1.set_title(f'{orbital.name} 軌道 - 電子確率分布', color='#00d4ff', fontsize=14)
        ax1.tick_params(colors='white')
        
        # 2D断面
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor('#0a0a0f')
        
        x = np.linspace(-5, 5, resolution)
        y = np.linspace(-5, 5, resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = orbital.probability_density(X[i, j], Y[i, j], 0)
        
        im = ax2.contourf(X, Y, Z, levels=20, cmap='plasma')
        ax2.set_xlabel('X', color='white')
        ax2.set_ylabel('Y', color='white')
        ax2.set_title(f'{orbital.name} 軌道 - XY断面', color='#00d4ff', fontsize=14)
        ax2.tick_params(colors='white')
        ax2.set_aspect('equal')
        
        plt.colorbar(im, ax=ax2, label='|ψ|²')
        plt.tight_layout()
        plt.savefig(f'orbital_{orbital.name}.png', facecolor='#0a0a0f', dpi=150)
        plt.show()
    
    @staticmethod
    def plot_molecule(molecule: Molecule):
        """分子を3Dプロット"""
        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('#0a0a0f')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0a0a0f')
        
        # 原子をプロット
        for atom in molecule.atoms:
            color = atom['color']
            if color == 'white':
                color = '#e0e0e0'
            ax.scatter(
                atom['x'], atom['y'], atom['z'],
                c=color, s=atom['radius'] * 500,
                edgecolors='white', linewidths=1
            )
            ax.text(
                atom['x'], atom['y'], atom['z'] + 0.5,
                atom['symbol'], color='white', fontsize=12, ha='center'
            )
        
        # 結合をプロット
        for atom1_idx, atom2_idx, order in molecule.bonds:
            atom1 = molecule.atoms[atom1_idx]
            atom2 = molecule.atoms[atom2_idx]
            
            for i in range(order):
                offset = (i - (order - 1) / 2) * 0.1
                ax.plot(
                    [atom1['x'] + offset, atom2['x'] + offset],
                    [atom1['y'], atom2['y']],
                    [atom1['z'], atom2['z']],
                    color='#00d4ff', linewidth=2, alpha=0.8
                )
        
        # 電子雲をプロット
        if molecule.electrons:
            for electron in molecule.electrons:
                x, y, z = electron.measure_position()
                color = '#ec4899' if electron.spin == 1 else '#8b5cf6'
                ax.scatter(x, y, z, c=color, s=20, alpha=0.3)
        
        ax.set_xlabel('X (Å)', color='white')
        ax.set_ylabel('Y (Å)', color='white')
        ax.set_zlabel('Z (Å)', color='white')
        ax.set_title(f'{molecule.name}', color='#00d4ff', fontsize=16)
        ax.tick_params(colors='white')
        
        plt.tight_layout()
        plt.savefig(f'molecule_{molecule.name}.png', facecolor='#0a0a0f', dpi=150)
        plt.show()
    
    @staticmethod
    def plot_tunneling(tunneling: QuantumTunneling, energies: np.ndarray):
        """トンネル効果をプロット"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0a0a0f')
        
        for ax in axes:
            ax.set_facecolor('#0a0a0f')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
        
        # ポテンシャル障壁と透過確率
        ax1 = axes[0]
        
        x = np.linspace(-2, 4, 200)
        V = np.where((x > 0) & (x < tunneling.barrier_width), tunneling.barrier_height, 0)
        
        ax1.fill_between(x, V, alpha=0.3, color='#ec4899', label='ポテンシャル障壁')
        ax1.plot(x, V, color='#ec4899', linewidth=2)
        
        # 波動関数（概念的）
        for E in [0.3, 0.6, 0.9]:
            T = tunneling.transmission_probability(E)
            
            # 入射波
            x_left = x[x < 0]
            psi_left = np.sin(5 * x_left) * np.exp(0.5 * x_left)
            ax1.plot(x_left, psi_left * 0.3 + E, color='#00d4ff', alpha=0.7)
            
            # 透過波
            x_right = x[x > tunneling.barrier_width]
            psi_right = T * np.sin(5 * x_right) * np.exp(-0.1 * x_right)
            ax1.plot(x_right, psi_right * 0.3 + E, color='#10b981', alpha=0.7)
            
            ax1.axhline(E, color='white', linestyle='--', alpha=0.3)
            ax1.text(-1.8, E, f'E={E:.1f}', color='white', fontsize=10)
        
        ax1.set_xlabel('位置', color='white', fontsize=12)
        ax1.set_ylabel('エネルギー / 波動関数', color='white', fontsize=12)
        ax1.set_title('量子トンネル効果', color='#00d4ff', fontsize=14)
        ax1.legend(loc='upper right')
        
        # 透過確率
        ax2 = axes[1]
        
        probs = [tunneling.transmission_probability(E) for E in energies]
        ax2.plot(energies, probs, color='#00d4ff', linewidth=2, label='理論値')
        
        # シミュレーション結果
        sim_probs = []
        for E in energies[::5]:
            trans, refl = tunneling.simulate_particle(E, 100)
            sim_probs.append(trans / (trans + refl))
        
        ax2.scatter(energies[::5], sim_probs, color='#ec4899', s=50, label='量子シミュレーション')
        
        ax2.axvline(tunneling.barrier_height, color='#8b5cf6', linestyle='--', label='障壁高さ')
        ax2.set_xlabel('粒子エネルギー', color='white', fontsize=12)
        ax2.set_ylabel('透過確率', color='white', fontsize=12)
        ax2.set_title('透過確率 vs エネルギー', color='#00d4ff', fontsize=14)
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('tunneling.png', facecolor='#0a0a0f', dpi=150)
        plt.show()


# ============================================================
# 分子ライブラリ
# ============================================================

def create_water() -> Molecule:
    """水分子 H2O"""
    mol = Molecule("H₂O (水)")
    mol.add_atom('O', 0, 0, 0)
    mol.add_atom('H', 0.96, 0, 0)
    mol.add_atom('H', -0.24, 0.93, 0)
    mol.add_bond(0, 1, 1)
    mol.add_bond(0, 2, 1)
    mol.simulate_electrons()
    return mol

def create_methane() -> Molecule:
    """メタン CH4"""
    mol = Molecule("CH₄ (メタン)")
    mol.add_atom('C', 0, 0, 0)
    # 正四面体配置
    mol.add_atom('H', 1.09, 0, 0)
    mol.add_atom('H', -0.36, 1.03, 0)
    mol.add_atom('H', -0.36, -0.51, 0.89)
    mol.add_atom('H', -0.36, -0.51, -0.89)
    for i in range(1, 5):
        mol.add_bond(0, i, 1)
    mol.simulate_electrons()
    return mol

def create_co2() -> Molecule:
    """二酸化炭素 CO2"""
    mol = Molecule("CO₂ (二酸化炭素)")
    mol.add_atom('C', 0, 0, 0)
    mol.add_atom('O', -1.16, 0, 0)
    mol.add_atom('O', 1.16, 0, 0)
    mol.add_bond(0, 1, 2)  # 二重結合
    mol.add_bond(0, 2, 2)
    mol.simulate_electrons()
    return mol

def create_benzene() -> Molecule:
    """ベンゼン C6H6"""
    mol = Molecule("C₆H₆ (ベンゼン)")
    # 炭素環
    for i in range(6):
        angle = i * np.pi / 3
        mol.add_atom('C', 1.4 * np.cos(angle), 1.4 * np.sin(angle), 0)
    # 水素
    for i in range(6):
        angle = i * np.pi / 3
        mol.add_atom('H', 2.5 * np.cos(angle), 2.5 * np.sin(angle), 0)
    # C-C結合（交互に単結合と二重結合）
    for i in range(6):
        mol.add_bond(i, (i + 1) % 6, 1 if i % 2 == 0 else 2)
    # C-H結合
    for i in range(6):
        mol.add_bond(i, i + 6, 1)
    mol.simulate_electrons()
    return mol


# ============================================================
# デモ
# ============================================================

def demo():
    """量子化学デモ"""
    print("=" * 70)
    print("  量子化学シミュレーター")
    print("  Quantum Chemistry Simulator")
    print("=" * 70)
    
    # ============================================================
    # 1. 原子軌道
    # ============================================================
    print("\n" + "─" * 70)
    print("  1. 原子軌道の可視化")
    print("─" * 70)
    
    orbitals = [
        AtomicOrbital(1, 0, 0),  # 1s
        AtomicOrbital(2, 0, 0),  # 2s
        AtomicOrbital(2, 1, 0),  # 2p_z
    ]
    
    for orbital in orbitals:
        print(f"\n  {orbital.name} 軌道:")
        print(f"    主量子数 n = {orbital.n}")
        print(f"    方位量子数 l = {orbital.l}")
        print(f"    磁気量子数 m = {orbital.m}")
        
        # サンプル位置
        positions = orbital.sample_position(100)
        avg_r = np.mean(np.sqrt(np.sum(positions**2, axis=1)))
        print(f"    平均距離 = {avg_r:.2f} (ボーア半径)")
    
    print("\n  [軌道を可視化中...]")
    QuantumChemistryVisualizer.plot_orbital(orbitals[0])
    
    # ============================================================
    # 2. 分子モデル
    # ============================================================
    print("\n" + "─" * 70)
    print("  2. 分子モデルと電子雲")
    print("─" * 70)
    
    molecules = [
        create_water(),
        create_methane(),
        create_co2(),
    ]
    
    for mol in molecules:
        print(f"\n  {mol.name}:")
        print(f"    原子数: {len(mol.atoms)}")
        print(f"    結合数: {len(mol.bonds)}")
        print(f"    総電子数: {mol.total_electrons()}")
        
        # 電子スピンの統計
        up_spins = sum(1 for e in mol.electrons if e.spin == 1)
        down_spins = sum(1 for e in mol.electrons if e.spin == -1)
        print(f"    スピン↑: {up_spins}, スピン↓: {down_spins}")
    
    print("\n  [分子を可視化中...]")
    QuantumChemistryVisualizer.plot_molecule(molecules[0])
    
    # ============================================================
    # 3. 量子トンネル効果
    # ============================================================
    print("\n" + "─" * 70)
    print("  3. 量子トンネル効果")
    print("─" * 70)
    
    tunneling = QuantumTunneling(barrier_height=1.0, barrier_width=1.0)
    
    print(f"\n  障壁高さ: {tunneling.barrier_height} eV")
    print(f"  障壁幅: {tunneling.barrier_width} Å")
    
    energies_test = [0.3, 0.5, 0.7, 0.9, 1.1]
    
    print("\n  エネルギー  | 理論透過率 | シミュレーション")
    print("  " + "-" * 50)
    
    for E in energies_test:
        T_theory = tunneling.transmission_probability(E)
        trans, refl = tunneling.simulate_particle(E, 1000)
        T_sim = trans / (trans + refl)
        
        status = "古典的透過" if E >= tunneling.barrier_height else "量子トンネル"
        print(f"  {E:.1f} eV    |   {T_theory:.4f}   |   {T_sim:.4f}   ({status})")
    
    print("\n  [トンネル効果を可視化中...]")
    QuantumChemistryVisualizer.plot_tunneling(tunneling, np.linspace(0.1, 1.5, 50))
    
    # ============================================================
    # 4. 化学反応シミュレーション
    # ============================================================
    print("\n" + "─" * 70)
    print("  4. 化学反応シミュレーション")
    print("─" * 70)
    
    reactions = [
        ChemicalReaction("水素燃焼 (2H₂ + O₂ → 2H₂O)", 0.5),
        ChemicalReaction("メタン燃焼 (CH₄ + 2O₂ → CO₂ + 2H₂O)", 0.8),
        ChemicalReaction("酵素反応（低活性化エネルギー）", 0.1),
    ]
    
    for reaction in reactions:
        print(f"\n  {reaction.name}:")
        print(f"    活性化エネルギー: {reaction.activation_energy} eV")
        print(f"    温度: {reaction.temperature} K")
        
        prob = reaction.reaction_probability()
        print(f"    反応確率（理論）: {prob:.6f}")
        
        reacted, unreacted = reaction.simulate_reaction(1000)
        print(f"    シミュレーション: {reacted}/1000 分子が反応")
    
    # ============================================================
    # 5. 量子もつれ状態
    # ============================================================
    print("\n" + "─" * 70)
    print("  5. 電子のもつれ状態（ヘリウム原子）")
    print("─" * 70)
    
    print("\n  ヘリウム原子の2電子:")
    
    # 2つの電子のもつれ状態をシミュレート
    n_measurements = 100
    singlet_count = 0  # 反平行スピン
    triplet_count = 0  # 平行スピン
    
    for _ in range(n_measurements):
        # もつれた量子ビット（相関 = -1 で反相関）
        qubit1 = PseudoQubit(correlation=0.0)
        result1 = qubit1.measure()
        
        # もつれにより、2つ目は1つ目と反対になる確率が高い
        correlation2 = -0.9 if result1 == 0 else 0.9
        qubit2 = PseudoQubit(correlation=correlation2)
        result2 = qubit2.measure()
        
        spin1 = 1 if result1 == 0 else -1
        spin2 = 1 if result2 == 0 else -1
        
        if spin1 != spin2:
            singlet_count += 1
        else:
            triplet_count += 1
    
    print(f"    一重項状態（反平行スピン）: {singlet_count}%")
    print(f"    三重項状態（平行スピン）: {triplet_count}%")
    print(f"    → パウリの排他原理により、一重項状態が優勢")
    
    print("\n" + "=" * 70)
    print("  デモ完了！")
    print("=" * 70)


def interactive_mode():
    """対話モード"""
    print("\n" + "=" * 70)
    print("  量子化学シミュレーター - 対話モード")
    print("=" * 70)
    
    print("""
  コマンド:
    orbital [n] [l] [m]  - 原子軌道を表示 (例: orbital 2 1 0)
    molecule [名前]      - 分子を表示 (water, methane, co2, benzene)
    tunnel [E]           - トンネル効果シミュレーション
    reaction [Ea]        - 化学反応シミュレーション
    help                 - ヘルプ
    quit                 - 終了
    """)
    
    while True:
        try:
            user_input = input("\n  量子化学> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'q', '終了']:
                print("  さようなら！")
                break
            
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd == 'orbital':
                n = int(parts[1]) if len(parts) > 1 else 1
                l = int(parts[2]) if len(parts) > 2 else 0
                m = int(parts[3]) if len(parts) > 3 else 0
                
                orbital = AtomicOrbital(n, l, m)
                print(f"\n  {orbital.name} 軌道を可視化中...")
                QuantumChemistryVisualizer.plot_orbital(orbital)
            
            elif cmd == 'molecule':
                mol_name = parts[1].lower() if len(parts) > 1 else 'water'
                
                mol_map = {
                    'water': create_water,
                    'h2o': create_water,
                    'methane': create_methane,
                    'ch4': create_methane,
                    'co2': create_co2,
                    'benzene': create_benzene,
                    'c6h6': create_benzene,
                }
                
                if mol_name in mol_map:
                    mol = mol_map[mol_name]()
                    print(f"\n  {mol.name} を可視化中...")
                    QuantumChemistryVisualizer.plot_molecule(mol)
                else:
                    print(f"  利用可能: water, methane, co2, benzene")
            
            elif cmd == 'tunnel':
                E = float(parts[1]) if len(parts) > 1 else 0.5
                
                tunneling = QuantumTunneling(barrier_height=1.0, barrier_width=1.0)
                T = tunneling.transmission_probability(E)
                trans, refl = tunneling.simulate_particle(E, 1000)
                
                print(f"\n  エネルギー: {E} eV")
                print(f"  障壁高さ: 1.0 eV")
                print(f"  透過確率（理論）: {T:.4f}")
                print(f"  透過確率（シミュレーション）: {trans/(trans+refl):.4f}")
            
            elif cmd == 'reaction':
                Ea = float(parts[1]) if len(parts) > 1 else 0.5
                
                reaction = ChemicalReaction("カスタム反応", Ea)
                reacted, unreacted = reaction.simulate_reaction(1000)
                
                print(f"\n  活性化エネルギー: {Ea} eV")
                print(f"  反応確率: {reaction.reaction_probability():.6f}")
                print(f"  反応した分子: {reacted}/1000")
            
            elif cmd == 'help':
                print("""
  ┌─────────────────────────────────────────────────────────┐
  │ コマンド一覧                                            │
  ├─────────────────────────────────────────────────────────┤
  │ orbital [n] [l] [m]  原子軌道を可視化                   │
  │   例: orbital 1 0 0  → 1s軌道                          │
  │   例: orbital 2 1 0  → 2p_z軌道                        │
  │                                                         │
  │ molecule [名前]      分子を可視化                       │
  │   water, methane, co2, benzene                         │
  │                                                         │
  │ tunnel [エネルギー]  量子トンネル効果                   │
  │   例: tunnel 0.5                                       │
  │                                                         │
  │ reaction [活性化E]   化学反応シミュレーション            │
  │   例: reaction 0.3                                     │
  └─────────────────────────────────────────────────────────┘
                """)
            
            else:
                print("  不明なコマンド。'help' でヘルプを表示")
            
        except Exception as e:
            print(f"  エラー: {e}")
        except KeyboardInterrupt:
            print("\n  さようなら！")
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_mode()
    else:
        demo()

