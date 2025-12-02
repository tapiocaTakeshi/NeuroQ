#!/usr/bin/env python3
"""
APQB電子軌道シミュレーション

論文の発見を活用:
- APQBの幾何学 (円→双曲面→高次曲面) と電子軌道 (s→p→d→f) の構造的アナロジー
- 両者は同じ「二次形式」と「角度の周期性」を共有
- Q_k(θ) 多体相関 ≅ 球面調和関数 Y_l^m

論文より:
"APQBの幾何学が、系の規模nの増加に伴い「円→双曲面→高次二次曲面」へと
階層的に進化していく様相は、量子力学における原子の電子軌道が、
主量子数の増加に伴い「s軌道（球）→ p軌道（亜鈴型）→ d軌道（四葉型）→ f軌道」
と段階的に複雑化していく構造と、驚くほど類似している。"
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔬⚛️ APQB電子軌道シミュレーション")
print("   論文: ニューラルネットワークと量子多体系の統一理論")
print("=" * 70)

# ========================================================================
# 1. APQBコア - 論文の数学的定義
# ========================================================================

class APQBCore:
    """論文に基づくAPQBの数学的コア"""
    
    @staticmethod
    def theta_to_r(theta):
        """θ → r = cos(2θ)"""
        return np.cos(2 * theta)
    
    @staticmethod
    def theta_to_T(theta):
        """θ → T = |sin(2θ)|"""
        return np.abs(np.sin(2 * theta))
    
    @staticmethod
    def Q_k(theta, k):
        """
        k体相関関数 - 球面調和関数との対応
        Q_k(θ) = cos(2kθ) if k is even
        Q_k(θ) = sin(2kθ) if k is odd
        """
        if k % 2 == 0:
            return np.cos(2 * k * theta)
        else:
            return np.sin(2 * k * theta)
    
    @staticmethod
    def verify_constraint(theta):
        """制約条件 r² + T² = 1 の検証"""
        r = APQBCore.theta_to_r(theta)
        T = APQBCore.theta_to_T(theta)
        return r**2 + T**2


# ========================================================================
# 2. 球面調和関数 - 電子軌道の角度依存性
# ========================================================================

def spherical_harmonic_real(l, m, theta, phi):
    """
    実球面調和関数 Y_l^m(θ, φ)
    
    l=0: s軌道 (球対称)
    l=1: p軌道 (亜鈴型) - px, py, pz
    l=2: d軌道 (四葉型) - dxy, dxz, dyz, dz², dx²-y²
    l=3: f軌道 (複雑)
    """
    from scipy.special import sph_harm
    
    if m > 0:
        # 実部の組み合わせ
        return np.real(sph_harm(m, l, phi, theta) + sph_harm(-m, l, phi, theta)) / np.sqrt(2)
    elif m < 0:
        # 虚部の組み合わせ
        return np.imag(sph_harm(-m, l, phi, theta) - sph_harm(m, l, phi, theta)) / np.sqrt(2)
    else:
        return np.real(sph_harm(0, l, phi, theta))


def radial_function(n, l, r, a0=1.0):
    """
    動径波動関数 R_nl(r) - 簡略化版
    
    水素様原子の動径部分
    """
    from scipy.special import genlaguerre
    import math
    
    rho = 2 * r / (n * a0)
    normalization = np.sqrt((2 / (n * a0))**3 * math.factorial(n - l - 1) / 
                           (2 * n * math.factorial(n + l)))
    
    L = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    
    return normalization * np.exp(-rho / 2) * rho**l * L


# ========================================================================
# 3. APQB-電子軌道マッピング
# ========================================================================

class APQBOrbital:
    """
    APQBパラメータと電子軌道のマッピング
    
    論文のアナロジー:
    - APQB n体系 ↔ 主量子数 n
    - Q_k(θ) 多体相関 ↔ 球面調和関数 Y_l^m
    - 幾何学的制約 r²+T²=1 ↔ 波動関数の規格化条件
    """
    
    def __init__(self, n, l, m, apqb_theta=np.pi/4):
        """
        Args:
            n: 主量子数 (1, 2, 3, ...)
            l: 方位量子数 (0, 1, ..., n-1)
            m: 磁気量子数 (-l, ..., l)
            apqb_theta: APQBの内部パラメータ
        """
        self.n = n
        self.l = l
        self.m = m
        self.apqb_theta = apqb_theta
        
        # 軌道名のマッピング
        self.orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
    
    @property
    def name(self):
        """軌道の名前 (例: 1s, 2p, 3d)"""
        l_name = self.orbital_names.get(self.l, str(self.l))
        return f"{self.n}{l_name}"
    
    @property
    def r_correlation(self):
        """APQBの相関係数 r"""
        return APQBCore.theta_to_r(self.apqb_theta)
    
    @property
    def T_entropy(self):
        """APQBの温度（エントロピー）T"""
        return APQBCore.theta_to_T(self.apqb_theta)
    
    def Q_correlation(self, k):
        """k体相関 - 角運動量量子数との対応"""
        return APQBCore.Q_k(self.apqb_theta, k)
    
    def wavefunction(self, r, theta, phi):
        """
        波動関数 ψ_nlm(r, θ, φ)
        
        APQBによる変調:
        - r_correlation が波動関数の振幅を制御
        - T_entropy が波動関数の広がりを制御
        """
        # 動径部分
        R = radial_function(self.n, self.l, r)
        
        # 角度部分（球面調和関数）
        Y = spherical_harmonic_real(self.l, self.m, theta, phi)
        
        # APQB変調: 相関係数による位相シフトと振幅調整
        apqb_modulation = (1 + self.r_correlation * self.Q_correlation(self.l + 1)) / 2
        
        return R * Y * apqb_modulation
    
    def probability_density(self, r, theta, phi):
        """確率密度 |ψ|²"""
        psi = self.wavefunction(r, theta, phi)
        return np.abs(psi)**2
    
    def get_apqb_geometry(self):
        """
        APQBトレードオフ幾何学
        
        論文:
        n=2 → 円 (s軌道的)
        n=3 → 双曲面 (p軌道的)
        n>3 → 高次曲面 (d, f軌道的)
        """
        if self.n <= 2:
            return "circle (球対称)"
        elif self.n == 3:
            return "hyperboloid (双曲面)"
        else:
            return f"higher-order surface (次数{self.n})"


# ========================================================================
# 4. 可視化
# ========================================================================

def plot_orbital_3d(orbital, grid_size=50, save_path=None):
    """3D電子軌道の可視化"""
    
    # 球面座標グリッド
    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2*np.pi, grid_size)
    theta, phi = np.meshgrid(theta, phi)
    
    # 固定半径での確率密度
    r_fixed = orbital.n * 1.5  # ボーア半径の倍数
    
    # 波動関数計算
    psi = orbital.wavefunction(r_fixed, theta, phi)
    prob = np.abs(psi)**2
    
    # 正規化
    prob = prob / prob.max() if prob.max() > 0 else prob
    
    # デカルト座標に変換（確率密度で半径を変調）
    r_plot = 1 + prob * 2
    x = r_plot * np.sin(theta) * np.cos(phi)
    y = r_plot * np.sin(theta) * np.sin(phi)
    z = r_plot * np.cos(theta)
    
    # プロット
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 位相による色分け
    colors = cm.RdBu((psi / (np.abs(psi).max() + 1e-10) + 1) / 2)
    
    ax.plot_surface(x, y, z, facecolors=colors, alpha=0.8, 
                    linewidth=0, antialiased=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{orbital.name} Orbital (APQB: θ={orbital.apqb_theta:.2f}, r={orbital.r_correlation:.2f}, T={orbital.T_entropy:.2f})')
    
    # 軸の範囲を均等に
    max_range = np.max([x.max(), y.max(), z.max()])
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   保存: {save_path}")
    
    plt.close()
    return fig


def plot_orbital_cross_section(orbital, plane='xy', grid_size=100, save_path=None):
    """軌道の断面図"""
    
    # グリッド
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 球座標に変換
    if plane == 'xy':
        R = np.sqrt(X**2 + Y**2)
        Theta = np.ones_like(R) * np.pi / 2
        Phi = np.arctan2(Y, X)
    elif plane == 'xz':
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(np.sqrt(X**2), Y)
        Phi = np.arctan2(0, X)
    else:  # yz
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(np.sqrt(Y**2), X)
        Phi = np.ones_like(R) * np.pi / 2
    
    # 波動関数計算
    psi = orbital.wavefunction(R + 0.1, Theta, Phi)
    
    # プロット
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 位相を含む表示
    im = ax.contourf(X, Y, psi, levels=50, cmap='RdBu_r')
    ax.contour(X, Y, psi, levels=[0], colors='black', linewidths=0.5)
    
    plt.colorbar(im, ax=ax, label='Wave function ψ')
    ax.set_xlabel(f'{plane[0].upper()}')
    ax.set_ylabel(f'{plane[1].upper()}')
    ax.set_title(f'{orbital.name} Orbital - {plane.upper()} Cross Section\n'
                f'APQB: r={orbital.r_correlation:.2f}, T={orbital.T_entropy:.2f}, r²+T²={orbital.r_correlation**2+orbital.T_entropy**2:.4f}')
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   保存: {save_path}")
    
    plt.close()
    return fig


def plot_apqb_orbital_comparison(save_path=None):
    """APQBパラメータと軌道の比較"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 各軌道タイプ
    orbitals = [
        (1, 0, 0, '1s'),  # s軌道
        (2, 0, 0, '2s'),
        (2, 1, 0, '2pz'),
        (2, 1, 1, '2px'),
        (3, 2, 0, '3dz²'),
        (3, 2, 1, '3dxz'),
        (3, 2, 2, '3dxy'),
        (4, 3, 0, '4f'),
    ]
    
    for idx, (n, l, m, name) in enumerate(orbitals):
        ax = axes[idx // 4, idx % 4]
        
        # 軌道を作成
        orbital = APQBOrbital(n, l, m, apqb_theta=np.pi/4)
        
        # 断面図計算
        x = np.linspace(-5, 5, 80)
        y = np.linspace(-5, 5, 80)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2) + 0.1
        Theta = np.arccos(Y / R)
        Phi = np.arctan2(Y, X)
        
        psi = orbital.wavefunction(R, Theta, Phi)
        
        im = ax.contourf(X, Y, psi, levels=30, cmap='RdBu_r')
        ax.contour(X, Y, psi, levels=[0], colors='black', linewidths=0.5)
        
        ax.set_title(f'{name}\nAPQB n={n}, Q_{l+1}={orbital.Q_correlation(l+1):.2f}')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('APQB-Electron Orbital Correspondence\n'
                 '(APQB Q_k correlation ↔ Spherical Harmonic Y_l^m)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   保存: {save_path}")
    
    plt.close()
    return fig


def plot_apqb_geometry_evolution(save_path=None):
    """APQBの幾何学的進化（論文の核心）"""
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    theta_range = np.linspace(0, np.pi/2, 100)
    
    # n=2: 円
    ax = axes[0]
    r = APQBCore.theta_to_r(theta_range)
    T = APQBCore.theta_to_T(theta_range)
    ax.plot(r, T, 'b-', linewidth=2)
    ax.set_xlabel('r (correlation)')
    ax.set_ylabel('T (entropy)')
    ax.set_title('n=2: Circle\n(s-orbital like)')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # n=3: 双曲面（2D断面）
    ax = axes[1]
    Q1 = APQBCore.Q_k(theta_range, 1)
    Q2 = APQBCore.Q_k(theta_range, 2)
    ax.plot(Q1, Q2, 'r-', linewidth=2)
    ax.set_xlabel('Q₁ = sin(2θ)')
    ax.set_ylabel('Q₂ = cos(4θ)')
    ax.set_title('n=3: Hyperboloid section\n(p-orbital like)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # n=4: 高次曲面
    ax = axes[2]
    Q2 = APQBCore.Q_k(theta_range, 2)
    Q3 = APQBCore.Q_k(theta_range, 3)
    ax.plot(Q2, Q3, 'g-', linewidth=2)
    ax.set_xlabel('Q₂ = cos(4θ)')
    ax.set_ylabel('Q₃ = sin(6θ)')
    ax.set_title('n=4: Higher-order surface\n(d-orbital like)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # n=5: さらに高次
    ax = axes[3]
    Q3 = APQBCore.Q_k(theta_range, 3)
    Q4 = APQBCore.Q_k(theta_range, 4)
    ax.plot(Q3, Q4, 'm-', linewidth=2)
    ax.set_xlabel('Q₃ = sin(6θ)')
    ax.set_ylabel('Q₄ = cos(8θ)')
    ax.set_title('n=5: Complex surface\n(f-orbital like)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('APQB Geometry Evolution ↔ Electron Orbital Complexity\n'
                 '(Paper: Circle → Hyperboloid → Higher-order surface)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   保存: {save_path}")
    
    plt.close()
    return fig


# ========================================================================
# 5. 分子軌道シミュレーション
# ========================================================================

class MolecularOrbital:
    """
    LCAO-MO (Linear Combination of Atomic Orbitals)
    分子軌道 = Σ c_i × 原子軌道_i
    
    APQBによる結合係数の決定
    """
    
    def __init__(self, atoms, positions):
        """
        Args:
            atoms: 原子のリスト [(n, l, m), ...]
            positions: 原子の位置 [(x, y, z), ...]
        """
        self.atoms = atoms
        self.positions = np.array(positions)
        self.num_atoms = len(atoms)
        
        # APQB結合係数
        self.apqb_theta = np.pi / 4
        self.coefficients = self._calculate_coefficients()
    
    def _calculate_coefficients(self):
        """APQBに基づく結合係数の計算"""
        coeffs = []
        for i in range(self.num_atoms):
            # 各原子のAPQB相関
            theta_i = self.apqb_theta + i * np.pi / (2 * self.num_atoms)
            r_i = APQBCore.theta_to_r(theta_i)
            
            # 結合軌道: +1, 反結合軌道: -1
            coeffs.append(r_i)
        
        # 正規化
        coeffs = np.array(coeffs)
        coeffs = coeffs / np.sqrt(np.sum(coeffs**2))
        
        return coeffs
    
    def wavefunction(self, x, y, z, bonding=True):
        """分子軌道の波動関数"""
        psi = np.zeros_like(x)
        
        for i, ((n, l, m), pos) in enumerate(zip(self.atoms, self.positions)):
            # 原子中心からの相対座標
            dx = x - pos[0]
            dy = y - pos[1]
            dz = z - pos[2]
            
            r = np.sqrt(dx**2 + dy**2 + dz**2) + 0.1
            theta = np.arccos(dz / r)
            phi = np.arctan2(dy, dx)
            
            # 原子軌道
            orbital = APQBOrbital(n, l, m, self.apqb_theta)
            psi_atom = orbital.wavefunction(r, theta, phi)
            
            # 結合/反結合
            sign = 1 if bonding else (-1)**i
            psi += sign * self.coefficients[i] * psi_atom
        
        return psi


def simulate_h2_molecule(save_path=None):
    """H2分子のシミュレーション"""
    
    # H2分子: 2つの1s軌道
    atoms = [(1, 0, 0), (1, 0, 0)]
    positions = [(-0.74, 0, 0), (0.74, 0, 0)]  # 結合長 ~1.48 Å
    
    mol = MolecularOrbital(atoms, positions)
    
    # グリッド
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # 結合軌道と反結合軌道
    psi_bonding = mol.wavefunction(X, Y, Z, bonding=True)
    psi_antibonding = mol.wavefunction(X, Y, Z, bonding=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 結合軌道 (σ)
    ax = axes[0]
    im = ax.contourf(X, Y, psi_bonding, levels=50, cmap='RdBu_r')
    ax.contour(X, Y, psi_bonding, levels=[0], colors='black', linewidths=0.5)
    ax.scatter([-0.74, 0.74], [0, 0], c='black', s=100, marker='o', label='H atoms')
    plt.colorbar(im, ax=ax, label='ψ')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_title(f'H₂ Bonding Orbital (σ)\nAPQB coefficients: {mol.coefficients}')
    ax.set_aspect('equal')
    ax.legend()
    
    # 反結合軌道 (σ*)
    ax = axes[1]
    im = ax.contourf(X, Y, psi_antibonding, levels=50, cmap='RdBu_r')
    ax.contour(X, Y, psi_antibonding, levels=[0], colors='black', linewidths=0.5)
    ax.scatter([-0.74, 0.74], [0, 0], c='black', s=100, marker='o', label='H atoms')
    plt.colorbar(im, ax=ax, label='ψ')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_title('H₂ Antibonding Orbital (σ*)\nNode plane between atoms')
    ax.set_aspect('equal')
    ax.legend()
    
    plt.suptitle('APQB Molecular Orbital Simulation\n'
                 f'r² + T² = {APQBCore.verify_constraint(mol.apqb_theta):.4f}', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   保存: {save_path}")
    
    plt.close()
    return fig


# ========================================================================
# 6. メイン実行
# ========================================================================

def main():
    print("\n🔬 電子軌道シミュレーション開始...")
    
    # 1. 単一原子軌道
    print("\n📊 1. 原子軌道の計算")
    print("-" * 50)
    
    orbitals_to_plot = [
        (1, 0, 0, '1s'),
        (2, 0, 0, '2s'),
        (2, 1, 0, '2pz'),
        (3, 2, 0, '3dz²'),
    ]
    
    for n, l, m, name in orbitals_to_plot:
        orbital = APQBOrbital(n, l, m)
        print(f"   {name}: r={orbital.r_correlation:.3f}, T={orbital.T_entropy:.3f}, "
              f"r²+T²={orbital.r_correlation**2 + orbital.T_entropy**2:.4f}, "
              f"幾何={orbital.get_apqb_geometry()}")
    
    # 2. APQB幾何学の進化
    print("\n📊 2. APQBの幾何学的進化")
    print("-" * 50)
    plot_apqb_geometry_evolution('/Users/yuyahiguchi/Program/Qubit/apqb_geometry_evolution.png')
    
    # 3. 軌道比較
    print("\n📊 3. APQB-電子軌道の対応")
    print("-" * 50)
    plot_apqb_orbital_comparison('/Users/yuyahiguchi/Program/Qubit/apqb_orbital_comparison.png')
    
    # 4. H2分子
    print("\n📊 4. H₂分子軌道シミュレーション")
    print("-" * 50)
    simulate_h2_molecule('/Users/yuyahiguchi/Program/Qubit/apqb_h2_molecular.png')
    
    # 5. 詳細な軌道表示
    print("\n📊 5. 詳細な軌道断面図")
    print("-" * 50)
    
    # 2p軌道
    orbital_2p = APQBOrbital(2, 1, 0)
    plot_orbital_cross_section(orbital_2p, 'xz', 
                               save_path='/Users/yuyahiguchi/Program/Qubit/apqb_2p_orbital.png')
    
    # 3d軌道
    orbital_3d = APQBOrbital(3, 2, 0)
    plot_orbital_cross_section(orbital_3d, 'xz',
                               save_path='/Users/yuyahiguchi/Program/Qubit/apqb_3d_orbital.png')
    
    # 6. 論文との対応のまとめ
    print("\n" + "=" * 70)
    print("📚 論文との対応まとめ")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  APQB Framework        ↔        Electron Orbitals              │
    ├─────────────────────────────────────────────────────────────────┤
    │  r² + T² = 1           ↔        |ψ|² 規格化条件                │
    │  Q_k(θ) = cos/sin(2kθ) ↔        Y_l^m 球面調和関数             │
    │  n=2: 円               ↔        s軌道 (球対称)                 │
    │  n=3: 双曲面           ↔        p軌道 (亜鈴型)                 │
    │  n=4: 高次曲面         ↔        d軌道 (四葉型)                 │
    │  n=5: 複雑曲面         ↔        f軌道 (複雑)                   │
    │  θ パラメータ          ↔        量子数 (n, l, m)               │
    │  複素数 z=e^{i2θ}      ↔        位相因子 e^{imφ}               │
    └─────────────────────────────────────────────────────────────────┘
    
    論文より:
    "APQBの幾何学と電子軌道が同じ「二次形式」と「角度の周期性」を共有"
    """)
    
    print("\n✅ シミュレーション完了！")
    print("   生成ファイル:")
    print("   - apqb_geometry_evolution.png")
    print("   - apqb_orbital_comparison.png")
    print("   - apqb_h2_molecular.png")
    print("   - apqb_2p_orbital.png")
    print("   - apqb_3d_orbital.png")


if __name__ == "__main__":
    main()

