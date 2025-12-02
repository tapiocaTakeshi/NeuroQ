#!/usr/bin/env python3
"""
QBNN アーキテクチャ比較
======================
脳型散在構造 vs 層状構造 の回帰分析比較
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 脳型QBNNをインポート
from qbnn_brain import QBNNBrainTorch

# ========================================
# 層状QBNN（比較用に簡略化）
# ========================================

class APQB:
    """APQB理論のコア"""
    
    @staticmethod
    def theta_to_r(theta):
        return torch.cos(2 * theta)
    
    @staticmethod
    def theta_to_T(theta):
        return torch.abs(torch.sin(2 * theta))


class QBNNLayeredTorch(nn.Module):
    """層状QBNN（従来型）"""
    
    def __init__(self, input_size: int, hidden_dims: List[int], output_size: int,
                 entangle_strength: float = 0.5):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.entangle_strength = nn.Parameter(torch.tensor(entangle_strength))
        
        # 層を構築
        dims = [input_size] + hidden_dims + [output_size]
        
        self.layers = nn.ModuleList()
        self.theta_layers = nn.ModuleList()
        
        # 層間もつれテンソルを先に作成
        J_list = [nn.Parameter(torch.zeros(1))]  # 最初の層用のダミー
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            self.theta_layers.append(nn.Linear(dims[i+1], dims[i+1]))
            
            # 層間もつれテンソル
            if i > 0:
                J_list.append(nn.Parameter(torch.randn(dims[i], dims[i+1]) * 0.1))
        
        self.J_matrices = nn.ParameterList(J_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        q_prev = None
        
        for i, (layer, theta_layer) in enumerate(zip(self.layers, self.theta_layers)):
            # 線形変換
            h_linear = layer(h)
            
            # θ計算（量子状態）
            theta = torch.sigmoid(theta_layer(h_linear)) * np.pi / 2
            r = APQB.theta_to_r(theta)
            
            # エンタングルメント補正
            if q_prev is not None and i < len(self.J_matrices):
                J = self.J_matrices[i]
                if J.numel() > 1:
                    s_prev = torch.tanh(q_prev)
                    s_curr = torch.tanh(h_linear)
                    
                    # もつれ補正
                    entangle = torch.einsum('bi,ij,bj->bj', s_prev, J, s_curr)
                    h_linear = h_linear + self.entangle_strength * entangle
            
            # 活性化（最終層以外）
            if i < len(self.layers) - 1:
                h = torch.tanh(h_linear)
            else:
                h = h_linear
            
            q_prev = r
        
        return h
    
    def get_quantum_info(self) -> Dict:
        return {
            'entangle_strength': self.entangle_strength.item(),
            'num_layers': len(self.layers),
        }


# ========================================
# 比較テスト関数
# ========================================

def compare_linear_regression():
    """線形回帰比較"""
    print("\n" + "=" * 70)
    print("📈 線形回帰比較: y = 2x + 1")
    print("=" * 70)
    
    # データ
    np.random.seed(42)
    X_np = np.linspace(-1, 1, 50).reshape(-1, 1)
    y_np = 2 * X_np + 1 + np.random.randn(50, 1) * 0.1
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    results = {}
    
    # ===== 脳型QBNN =====
    print("\n🧠 脳型散在構造 (QBNN Brain)")
    print("-" * 40)
    
    model_brain = QBNNBrainTorch(
        num_neurons=30,
        input_size=1,
        output_size=1,
        connection_density=0.25
    )
    
    optimizer = torch.optim.Adam(model_brain.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    for epoch in range(150):
        optimizer.zero_grad()
        pred = model_brain(X, time_steps=2)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    brain_time = time.time() - start_time
    
    model_brain.eval()
    with torch.no_grad():
        pred = model_brain(X, time_steps=2)
        brain_mse = criterion(pred, y).item()
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        brain_r2 = 1 - (ss_res / ss_tot)
    
    results['brain'] = {'MSE': brain_mse, 'R²': brain_r2, 'Time': brain_time}
    print(f"   MSE: {brain_mse:.6f}")
    print(f"   R²: {brain_r2:.4f}")
    print(f"   学習時間: {brain_time:.2f}秒")
    
    # ===== 層状QBNN =====
    print("\n📊 層状構造 (QBNN Layered)")
    print("-" * 40)
    
    model_layered = QBNNLayeredTorch(
        input_size=1,
        hidden_dims=[16, 16],
        output_size=1,
        entangle_strength=0.35
    )
    
    optimizer = torch.optim.Adam(model_layered.parameters(), lr=0.02)
    
    start_time = time.time()
    for epoch in range(150):
        optimizer.zero_grad()
        pred = model_layered(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    layered_time = time.time() - start_time
    
    model_layered.eval()
    with torch.no_grad():
        pred = model_layered(X)
        layered_mse = criterion(pred, y).item()
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        layered_r2 = 1 - (ss_res / ss_tot)
    
    results['layered'] = {'MSE': layered_mse, 'R²': layered_r2, 'Time': layered_time}
    print(f"   MSE: {layered_mse:.6f}")
    print(f"   R²: {layered_r2:.4f}")
    print(f"   学習時間: {layered_time:.2f}秒")
    
    return results


def compare_nonlinear_regression():
    """非線形回帰比較"""
    print("\n" + "=" * 70)
    print("📈 非線形回帰比較: y = sin(πx)")
    print("=" * 70)
    
    # データ
    np.random.seed(42)
    X_np = np.linspace(-1, 1, 60).reshape(-1, 1)
    y_np = np.sin(np.pi * X_np) + np.random.randn(60, 1) * 0.05
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    results = {}
    criterion = nn.MSELoss()
    
    # ===== 脳型QBNN =====
    print("\n🧠 脳型散在構造 (QBNN Brain)")
    print("-" * 40)
    
    model_brain = QBNNBrainTorch(
        num_neurons=40,
        input_size=1,
        output_size=1,
        connection_density=0.25
    )
    
    optimizer = torch.optim.Adam(model_brain.parameters(), lr=0.015)
    
    start_time = time.time()
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model_brain(X, time_steps=3)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    brain_time = time.time() - start_time
    
    model_brain.eval()
    with torch.no_grad():
        pred = model_brain(X, time_steps=3)
        brain_mse = criterion(pred, y).item()
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        brain_r2 = 1 - (ss_res / ss_tot)
    
    results['brain'] = {'MSE': brain_mse, 'R²': brain_r2, 'Time': brain_time}
    print(f"   MSE: {brain_mse:.6f}")
    print(f"   R²: {brain_r2:.4f}")
    print(f"   学習時間: {brain_time:.2f}秒")
    
    # ===== 層状QBNN =====
    print("\n📊 層状構造 (QBNN Layered)")
    print("-" * 40)
    
    model_layered = QBNNLayeredTorch(
        input_size=1,
        hidden_dims=[32, 32],
        output_size=1,
        entangle_strength=0.35
    )
    
    optimizer = torch.optim.Adam(model_layered.parameters(), lr=0.015)
    
    start_time = time.time()
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model_layered(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    layered_time = time.time() - start_time
    
    model_layered.eval()
    with torch.no_grad():
        pred = model_layered(X)
        layered_mse = criterion(pred, y).item()
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        layered_r2 = 1 - (ss_res / ss_tot)
    
    results['layered'] = {'MSE': layered_mse, 'R²': layered_r2, 'Time': layered_time}
    print(f"   MSE: {layered_mse:.6f}")
    print(f"   R²: {layered_r2:.4f}")
    print(f"   学習時間: {layered_time:.2f}秒")
    
    return results


def compare_multivariate_regression():
    """多変量回帰比較"""
    print("\n" + "=" * 70)
    print("📈 多変量回帰比較: y = 0.5x₁ + 0.3x₂ - 0.2x₃")
    print("=" * 70)
    
    # データ
    np.random.seed(42)
    n_samples = 80
    X_np = np.random.randn(n_samples, 3)
    y_np = (0.5 * X_np[:, 0] + 0.3 * X_np[:, 1] - 0.2 * X_np[:, 2] 
            + np.random.randn(n_samples) * 0.1).reshape(-1, 1)
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    results = {}
    criterion = nn.MSELoss()
    
    # ===== 脳型QBNN =====
    print("\n🧠 脳型散在構造 (QBNN Brain)")
    print("-" * 40)
    
    model_brain = QBNNBrainTorch(
        num_neurons=35,
        input_size=3,
        output_size=1,
        connection_density=0.25
    )
    
    optimizer = torch.optim.Adam(model_brain.parameters(), lr=0.02)
    
    start_time = time.time()
    for epoch in range(150):
        optimizer.zero_grad()
        pred = model_brain(X, time_steps=3)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    brain_time = time.time() - start_time
    
    model_brain.eval()
    with torch.no_grad():
        pred = model_brain(X, time_steps=3)
        brain_mse = criterion(pred, y).item()
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        brain_r2 = 1 - (ss_res / ss_tot)
    
    results['brain'] = {'MSE': brain_mse, 'R²': brain_r2, 'Time': brain_time}
    print(f"   MSE: {brain_mse:.6f}")
    print(f"   R²: {brain_r2:.4f}")
    print(f"   学習時間: {brain_time:.2f}秒")
    
    # ===== 層状QBNN =====
    print("\n📊 層状構造 (QBNN Layered)")
    print("-" * 40)
    
    model_layered = QBNNLayeredTorch(
        input_size=3,
        hidden_dims=[24, 24],
        output_size=1,
        entangle_strength=0.35
    )
    
    optimizer = torch.optim.Adam(model_layered.parameters(), lr=0.02)
    
    start_time = time.time()
    for epoch in range(150):
        optimizer.zero_grad()
        pred = model_layered(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    layered_time = time.time() - start_time
    
    model_layered.eval()
    with torch.no_grad():
        pred = model_layered(X)
        layered_mse = criterion(pred, y).item()
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        layered_r2 = 1 - (ss_res / ss_tot)
    
    results['layered'] = {'MSE': layered_mse, 'R²': layered_r2, 'Time': layered_time}
    print(f"   MSE: {layered_mse:.6f}")
    print(f"   R²: {layered_r2:.4f}")
    print(f"   学習時間: {layered_time:.2f}秒")
    
    return results


def compare_polynomial_regression():
    """多項式回帰比較"""
    print("\n" + "=" * 70)
    print("📈 多項式回帰比較: y = x³ - x² + 0.5x")
    print("=" * 70)
    
    # データ
    np.random.seed(42)
    X_np = np.linspace(-1, 1, 50).reshape(-1, 1)
    y_np = X_np**3 - X_np**2 + 0.5*X_np + np.random.randn(50, 1) * 0.05
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    results = {}
    criterion = nn.MSELoss()
    
    # ===== 脳型QBNN =====
    print("\n🧠 脳型散在構造 (QBNN Brain)")
    print("-" * 40)
    
    model_brain = QBNNBrainTorch(
        num_neurons=45,
        input_size=1,
        output_size=1,
        connection_density=0.3
    )
    
    optimizer = torch.optim.Adam(model_brain.parameters(), lr=0.015)
    
    start_time = time.time()
    for epoch in range(250):
        optimizer.zero_grad()
        pred = model_brain(X, time_steps=4)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    brain_time = time.time() - start_time
    
    model_brain.eval()
    with torch.no_grad():
        pred = model_brain(X, time_steps=4)
        brain_mse = criterion(pred, y).item()
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        brain_r2 = 1 - (ss_res / ss_tot)
    
    results['brain'] = {'MSE': brain_mse, 'R²': brain_r2, 'Time': brain_time}
    print(f"   MSE: {brain_mse:.6f}")
    print(f"   R²: {brain_r2:.4f}")
    print(f"   学習時間: {brain_time:.2f}秒")
    
    # ===== 層状QBNN =====
    print("\n📊 層状構造 (QBNN Layered)")
    print("-" * 40)
    
    model_layered = QBNNLayeredTorch(
        input_size=1,
        hidden_dims=[32, 32, 16],
        output_size=1,
        entangle_strength=0.35
    )
    
    optimizer = torch.optim.Adam(model_layered.parameters(), lr=0.015)
    
    start_time = time.time()
    for epoch in range(250):
        optimizer.zero_grad()
        pred = model_layered(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    layered_time = time.time() - start_time
    
    model_layered.eval()
    with torch.no_grad():
        pred = model_layered(X)
        layered_mse = criterion(pred, y).item()
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        layered_r2 = 1 - (ss_res / ss_tot)
    
    results['layered'] = {'MSE': layered_mse, 'R²': layered_r2, 'Time': layered_time}
    print(f"   MSE: {layered_mse:.6f}")
    print(f"   R²: {layered_r2:.4f}")
    print(f"   学習時間: {layered_time:.2f}秒")
    
    return results


def compare_classification():
    """分類タスク比較（XOR）"""
    print("\n" + "=" * 70)
    print("🔢 分類比較: XOR問題")
    print("=" * 70)
    
    # データ
    X = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=torch.float32)
    
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    results = {}
    criterion = nn.MSELoss()
    
    # ===== 脳型QBNN =====
    print("\n🧠 脳型散在構造 (QBNN Brain)")
    print("-" * 40)
    
    model_brain = QBNNBrainTorch(
        num_neurons=25,
        input_size=2,
        output_size=1,
        connection_density=0.25
    )
    
    optimizer = torch.optim.Adam(model_brain.parameters(), lr=0.03)
    
    start_time = time.time()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model_brain(X, time_steps=3)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    brain_time = time.time() - start_time
    
    model_brain.eval()
    with torch.no_grad():
        pred = model_brain(X, time_steps=3)
        brain_correct = sum(1 for i in range(4) if (pred[i] > 0.5) == (y[i] > 0.5))
        brain_acc = brain_correct / 4 * 100
    
    results['brain'] = {'Accuracy': brain_acc, 'Time': brain_time}
    print(f"   正解率: {brain_acc:.0f}%")
    print(f"   学習時間: {brain_time:.2f}秒")
    
    # ===== 層状QBNN =====
    print("\n📊 層状構造 (QBNN Layered)")
    print("-" * 40)
    
    model_layered = QBNNLayeredTorch(
        input_size=2,
        hidden_dims=[16, 16],
        output_size=1,
        entangle_strength=0.35
    )
    
    optimizer = torch.optim.Adam(model_layered.parameters(), lr=0.03)
    
    start_time = time.time()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model_layered(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    layered_time = time.time() - start_time
    
    model_layered.eval()
    with torch.no_grad():
        pred = model_layered(X)
        layered_correct = sum(1 for i in range(4) if (pred[i] > 0.5) == (y[i] > 0.5))
        layered_acc = layered_correct / 4 * 100
    
    results['layered'] = {'Accuracy': layered_acc, 'Time': layered_time}
    print(f"   正解率: {layered_acc:.0f}%")
    print(f"   学習時間: {layered_time:.2f}秒")
    
    return results


def run_full_comparison():
    """全比較を実行"""
    print("=" * 70)
    print("🧠⚛️ QBNN アーキテクチャ比較")
    print("   脳型散在構造 vs 層状構造")
    print("=" * 70)
    
    all_results = {}
    
    # 1. 線形回帰
    all_results['linear'] = compare_linear_regression()
    
    # 2. 非線形回帰
    all_results['nonlinear'] = compare_nonlinear_regression()
    
    # 3. 多変量回帰
    all_results['multivariate'] = compare_multivariate_regression()
    
    # 4. 多項式回帰
    all_results['polynomial'] = compare_polynomial_regression()
    
    # 5. 分類
    all_results['classification'] = compare_classification()
    
    # ===== 総合サマリー =====
    print("\n" + "=" * 70)
    print("📊 総合比較サマリー")
    print("=" * 70)
    
    print("""
┌──────────────────────────────────────────────────────────────────────┐
│                    QBNN アーキテクチャ比較結果                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  タスク          │   脳型散在構造      │   層状構造         │ 勝者   │
│  ────────────────┼───────────────────┼──────────────────┼─────── │""")
    
    brain_wins = 0
    layered_wins = 0
    
    # 回帰タスク
    for task_name, display_name in [
        ('linear', '線形回帰'),
        ('nonlinear', '非線形回帰'),
        ('multivariate', '多変量回帰'),
        ('polynomial', '多項式回帰'),
    ]:
        if task_name in all_results:
            r = all_results[task_name]
            brain_r2 = r['brain']['R²']
            layered_r2 = r['layered']['R²']
            
            if brain_r2 > layered_r2:
                winner = "🧠"
                brain_wins += 1
            elif layered_r2 > brain_r2:
                winner = "📊"
                layered_wins += 1
            else:
                winner = "引分"
            
            print(f"│  {display_name:14} │ R²={brain_r2:.4f}        │ R²={layered_r2:.4f}       │  {winner}   │")
    
    # 分類タスク
    if 'classification' in all_results:
        r = all_results['classification']
        brain_acc = r['brain']['Accuracy']
        layered_acc = r['layered']['Accuracy']
        
        if brain_acc > layered_acc:
            winner = "🧠"
            brain_wins += 1
        elif layered_acc > brain_acc:
            winner = "📊"
            layered_wins += 1
        else:
            winner = "引分"
        
        print(f"│  {'XOR分類':14} │ Acc={brain_acc:.0f}%         │ Acc={layered_acc:.0f}%        │  {winner}   │")
    
    print("""│                                                                      │
├──────────────────────────────────────────────────────────────────────┤""")
    
    print(f"│  勝敗: 🧠 脳型={brain_wins}勝  📊 層状={layered_wins}勝                               │")
    
    # 平均比較
    brain_r2_avg = np.mean([all_results[t]['brain']['R²'] for t in ['linear', 'nonlinear', 'multivariate', 'polynomial']])
    layered_r2_avg = np.mean([all_results[t]['layered']['R²'] for t in ['linear', 'nonlinear', 'multivariate', 'polynomial']])
    
    brain_time_avg = np.mean([all_results[t]['brain']['Time'] for t in all_results])
    layered_time_avg = np.mean([all_results[t]['layered']['Time'] for t in all_results])
    
    print(f"│                                                                      │")
    print(f"│  平均R²: 🧠 {brain_r2_avg:.4f}  📊 {layered_r2_avg:.4f}                               │")
    print(f"│  平均時間: 🧠 {brain_time_avg:.2f}秒  📊 {layered_time_avg:.2f}秒                            │")
    
    print("""└──────────────────────────────────────────────────────────────────────┘
    """)
    
    # 特徴比較
    print("""
┌──────────────────────────────────────────────────────────────────────┐
│                    アーキテクチャ特徴比較                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  特徴              │   🧠 脳型散在構造    │   📊 層状構造          │
│  ──────────────────┼─────────────────────┼─────────────────────── │
│  構造              │   グラフ状・分散型   │   整列した層構造       │
│  接続              │   スパース・ランダム │   全結合・構造化       │
│  信号伝播          │   時間ステップ       │   順次伝播             │
│  量子もつれ        │   任意ニューロン間   │   隣接層間のみ         │
│  表現力            │   柔軟・適応的       │   階層的・構造化       │
│  計算コスト        │   やや高い           │   効率的               │
│  生物学的妥当性    │   高い（脳に近い）   │   低い（人工的）       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
    """)
    
    # 総評
    if brain_wins > layered_wins:
        overall = "🧠 脳型散在構造"
        comment = "より高い精度を達成"
    elif layered_wins > brain_wins:
        overall = "📊 層状構造"
        comment = "より高い精度を達成"
    else:
        overall = "引き分け"
        comment = "両者同等の性能"
    
    print(f"\n🏆 総合優勝: {overall}")
    print(f"   {comment}")
    
    return all_results


if __name__ == '__main__':
    run_full_comparison()

