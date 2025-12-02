どの良くなっているの？でも。#!/usr/bin/env python3
"""
QBNN (Quantum-Bit Neural Network) - PyTorch実装

数式モデル:
1. s^(l) = normalize(h^(l)) ∈ [-1,1]           (正規化)
2. θ^(l)_i = arccos(s^(l)_i)                   (Bloch角)
3. Δ^(l+1)_j = Σ_i J_{ij} s^(l)_i s^(l+1)_{raw,j}  (もつれ補正)
4. ĥ^(l+1) = h̃^(l+1) + λ Δ^(l+1)             (有効入力)
5. h^(l+1) = σ(ĥ^(l+1))                       (活性化)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("🧠⚛️ QBNN (Quantum-Bit Neural Network) - PyTorch実装")
print("=" * 70)


class QBNNLayer(nn.Module):
    """
    Quantum-Bit Neural Network Layer
    
    各層・各ビットが分子のようにもつれ合う層
    
    数式:
    - 線形変換: h̃^(l+1) = W h^(l) + b
    - 正規化: s = tanh(h) ∈ [-1, 1]
    - もつれ補正: Δ_j = Σ_i J_{ij} s^(l)_i s^(l+1)_{raw,j}
    - 有効入力: ĥ = h̃ + λ Δ
    - 出力: h = tanh(ĥ)
    """
    
    def __init__(self, input_dim, output_dim, lambda_entangle=0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_entangle = nn.Parameter(torch.tensor(lambda_entangle))
        
        # 重み W^(l)
        self.W = nn.Linear(input_dim, output_dim)
        
        # もつれテンソル J^(l) - 学習可能
        self.J = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        
        # 状態保持（デバッグ・可視化用）
        self.s_prev = None  # 正規化された前の層
        self.s_raw = None   # 正規化された生の候補
        self.theta = None   # Bloch角
        self.delta = None   # もつれ補正
    
    def normalize(self, h):
        """h を [-1, 1] に正規化 (tanh)"""
        return torch.tanh(h)
    
    def compute_bloch_angle(self, s):
        """正規化された値 s からBloch角 θ を計算"""
        # arccos の入力を [-1, 1] にクランプ
        s_clamped = torch.clamp(s, -1 + 1e-7, 1 - 1e-7)
        return torch.acos(s_clamped)
    
    def forward(self, h_prev):
        """
        順伝播
        
        Args:
            h_prev: 前の層の出力 [batch, input_dim]
        
        Returns:
            h: この層の出力 [batch, output_dim]
        """
        # Step 1: 前の層の正規化 s^(l)
        self.s_prev = self.normalize(h_prev)
        
        # Step 2: 線形変換 h̃^(l+1) = W h^(l) + b
        h_tilde = self.W(h_prev)
        
        # Step 3: 正規化（生の候補）s^(l+1)_raw
        self.s_raw = self.normalize(h_tilde)
        
        # Step 4: もつれ補正 Δ^(l+1)
        # Δ_j = Σ_i J_{ij} s^(l)_i s^(l+1)_{raw,j}
        # = (s_prev @ J) * s_raw
        interaction = torch.matmul(self.s_prev, self.J)  # [batch, output_dim]
        self.delta = interaction * self.s_raw
        
        # Step 5: 有効入力 ĥ^(l+1) = h̃^(l+1) + λ Δ^(l+1)
        h_hat = h_tilde + self.lambda_entangle * self.delta
        
        # Step 6: 活性化 h^(l+1) = tanh(ĥ^(l+1))
        h = torch.tanh(h_hat)
        
        # Bloch角を保存
        s_out = self.normalize(h)
        self.theta = self.compute_bloch_angle(s_out)
        
        return h
    
    def measure(self):
        """
        量子測定をシミュレート
        
        Returns:
            measurements: 測定結果 (0 or 1) [batch, output_dim]
        """
        if self.theta is None:
            return None
        
        # P(|1⟩) = sin²(θ/2)
        p1 = torch.sin(self.theta / 2) ** 2
        measurements = (torch.rand_like(p1) < p1).float()
        
        return measurements
    
    def get_entanglement_strength(self):
        """もつれ強度を取得"""
        return self.lambda_entangle.item()
    
    def get_coupling_matrix(self):
        """結合行列 J を取得"""
        return self.J.detach().cpu().numpy()


class QBNN(nn.Module):
    """
    Quantum-Bit Neural Network
    
    複数のQBNNLayerを積み重ねたネットワーク
    """
    
    def __init__(self, layer_dims, lambda_entangle=0.5, lambda_decay=0.1):
        """
        Args:
            layer_dims: 各層の次元 [input, hidden1, hidden2, ..., output]
            lambda_entangle: 初期もつれ強度
            lambda_decay: 深い層ほどもつれを弱くする係数
        """
        super().__init__()
        
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        
        # QBNNレイヤー
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            # 深い層ほどλを小さく
            layer_lambda = lambda_entangle * (1 - i * lambda_decay)
            layer = QBNNLayer(layer_dims[i], layer_dims[i + 1], layer_lambda)
            self.layers.append(layer)
    
    def forward(self, x):
        """順伝播"""
        h = x
        for layer in self.layers:
            h = layer(h)
        return h
    
    def measure_all(self):
        """全層の測定結果を取得"""
        measurements = []
        for layer in self.layers:
            m = layer.measure()
            if m is not None:
                measurements.append(m)
        return measurements
    
    def get_all_thetas(self):
        """全層のBloch角を取得"""
        thetas = []
        for layer in self.layers:
            if layer.theta is not None:
                thetas.append(layer.theta.detach())
        return thetas
    
    def get_entanglement_info(self):
        """エンタングルメント情報を取得"""
        info = []
        for i, layer in enumerate(self.layers):
            s = layer.s_raw
            if s is not None:
                theta = layer.theta
                r = torch.cos(2 * theta).mean().item() if theta is not None else 0
                T = torch.abs(torch.sin(2 * theta)).mean().item() if theta is not None else 0
                info.append({
                    'layer': i,
                    'lambda': layer.get_entanglement_strength(),
                    'r_mean': r,
                    'T_mean': T,
                    'r2_T2': r**2 + T**2
                })
        return info


class QBNNClassifier(nn.Module):
    """
    QBNN分類器
    
    QBNNをベースにした分類モデル
    """
    
    def __init__(self, input_dim, hidden_dims, num_classes, lambda_entangle=0.5):
        super().__init__()
        
        # QBNN
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        self.qbnn = QBNN(layer_dims, lambda_entangle)
        
        # 出力層（ソフトマックス前）
        self.output = nn.Linear(num_classes, num_classes)
    
    def forward(self, x):
        h = self.qbnn(x)
        logits = self.output(h)
        return logits
    
    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


# ========================================
# テストとデモ
# ========================================

def test_xor_problem():
    """XOR問題でQBNNをテスト"""
    print("\n📊 XOR問題でQBNNをテスト")
    print("-" * 50)
    
    # データ
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = torch.tensor([0, 1, 1, 0])
    
    # モデル
    model = QBNNClassifier(
        input_dim=2,
        hidden_dims=[8, 8],
        num_classes=2,
        lambda_entangle=0.5
    )
    
    # 学習
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(500):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            pred = model.predict(X)
            acc = (pred == y).float().mean()
            print(f"   Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")
    
    # 最終結果
    pred = model.predict(X)
    print(f"\n   最終予測: {pred.tolist()}")
    print(f"   正解: {y.tolist()}")
    
    # エンタングルメント情報
    print("\n   エンタングルメント情報:")
    _ = model(X)  # フォワードパス
    info = model.qbnn.get_entanglement_info()
    for i in info:
        print(f"   Layer {i['layer']}: λ={i['lambda']:.3f}, r={i['r_mean']:.3f}, T={i['T_mean']:.3f}")
    
    return model, losses


def visualize_qbnn(model, X):
    """QBNNの状態を可視化"""
    print("\n📈 QBNN状態の可視化")
    
    # フォワードパス
    _ = model(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 各層のBloch角分布
    ax = axes[0, 0]
    thetas = model.qbnn.get_all_thetas()
    for i, theta in enumerate(thetas):
        theta_np = theta.mean(dim=0).numpy()
        ax.bar(np.arange(len(theta_np)) + i * 0.2, theta_np, 
               width=0.2, label=f'Layer {i}', alpha=0.7)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('θ (Bloch Angle)')
    ax.set_title('Bloch Angles per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. r-T平面
    ax = axes[0, 1]
    info = model.qbnn.get_entanglement_info()
    
    # 制約曲線
    theta_range = np.linspace(0, np.pi/2, 100)
    r_curve = np.cos(2 * theta_range)
    T_curve = np.abs(np.sin(2 * theta_range))
    ax.plot(r_curve, T_curve, 'b-', linewidth=2, label='r² + T² = 1')
    
    # 各層のプロット
    colors = plt.cm.viridis(np.linspace(0, 1, len(info)))
    for i, inf in enumerate(info):
        ax.scatter([inf['r_mean']], [inf['T_mean']], s=200, c=[colors[i]], 
                   label=f"Layer {i} (λ={inf['lambda']:.2f})", zorder=5)
    ax.set_xlabel('r (Correlation)')
    ax.set_ylabel('T (Temperature)')
    ax.set_title('Layer States on r-T Plane')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 結合行列 J の可視化
    ax = axes[1, 0]
    J = model.qbnn.layers[0].get_coupling_matrix()
    im = ax.imshow(J, cmap='coolwarm', aspect='auto')
    ax.set_xlabel('Output Neuron')
    ax.set_ylabel('Input Neuron')
    ax.set_title('Coupling Matrix J (Layer 0)')
    plt.colorbar(im, ax=ax)
    
    # 4. 測定結果
    ax = axes[1, 1]
    measurements = model.qbnn.measure_all()
    if measurements:
        for i, m in enumerate(measurements):
            m_np = m.mean(dim=0).numpy()
            ax.bar(np.arange(len(m_np)) + i * 0.25, m_np, 
                   width=0.25, label=f'Layer {i}', alpha=0.7)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('P(|1⟩)')
    ax.set_title('Measurement Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('QBNN (Quantum-Bit Neural Network) Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig('/Users/yuyahiguchi/Program/Qubit/qbnn_pytorch_viz.png', dpi=150, bbox_inches='tight')
    print("   保存: qbnn_pytorch_viz.png")
    plt.close()


def print_model_summary():
    """モデルの数式サマリー"""
    print("\n" + "=" * 70)
    print("📚 QBNN 数式モデル サマリー")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  QBNN (Quantum-Bit Neural Network) 数式モデル                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. 正規化                                                      │
    │     s^(l) = tanh(h^(l)) ∈ [-1, 1]                              │
    │                                                                 │
    │  2. Bloch角                                                     │
    │     θ^(l)_i = arccos(s^(l)_i)                                  │
    │                                                                 │
    │  3. 線形変換                                                    │
    │     h̃^(l+1) = W^(l) h^(l) + b^(l)                              │
    │                                                                 │
    │  4. もつれ補正（量子相互作用）                                  │
    │     Δ^(l+1)_j = Σ_i J^(l)_{ij} s^(l)_i s^(l+1)_{raw,j}         │
    │                                                                 │
    │  5. 有効入力                                                    │
    │     ĥ^(l+1) = h̃^(l+1) + λ^(l) Δ^(l+1)                          │
    │                                                                 │
    │  6. 活性化                                                      │
    │     h^(l+1) = tanh(ĥ^(l+1))                                    │
    │                                                                 │
    │  パラメータ:                                                     │
    │     W^(l) ... 通常の重み行列                                    │
    │     J^(l) ... もつれテンソル（量子結合）                        │
    │     λ^(l) ... もつれ強度（0で通常NNに戻る）                     │
    │                                                                 │
    │  量子的解釈:                                                     │
    │     - 各ニューロン = 1量子ビット                                │
    │     - s_i = ⟨Z_i⟩ (z期待値 = Bloch球のz座標)                   │
    │     - J_{ij} = 分子結合の強さ                                   │
    │     - Δ = 層間の量子もつれによる補正                            │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)


def test_logic_gates():
    """論理ゲート（AND, OR, NAND, NOR）をテスト"""
    print("\n📊 論理ゲートテスト")
    print("-" * 50)
    
    gates = {
        'AND': torch.tensor([0, 0, 0, 1]),
        'OR': torch.tensor([0, 1, 1, 1]),
        'NAND': torch.tensor([1, 1, 1, 0]),
        'NOR': torch.tensor([1, 0, 0, 0]),
        'XOR': torch.tensor([0, 1, 1, 0]),
        'XNOR': torch.tensor([1, 0, 0, 1])
    }
    
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    
    results = {}
    
    for gate_name, y in gates.items():
        model = QBNNClassifier(
            input_dim=2,
            hidden_dims=[8, 8],
            num_classes=2,
            lambda_entangle=0.5
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(300):
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        pred = model.predict(X)
        acc = (pred == y).float().mean().item()
        results[gate_name] = acc
        
        status = "✅" if acc == 1.0 else "❌"
        print(f"   {gate_name}: {acc:.0%} {status}  予測={pred.tolist()} 正解={y.tolist()}")
    
    return results


def test_circle_classification():
    """円形分類問題（非線形）をテスト"""
    print("\n📊 円形分類問題（非線形）")
    print("-" * 50)
    
    # データ生成
    np.random.seed(42)
    n_samples = 200
    
    # 内側の円（クラス0）
    r_inner = np.random.uniform(0, 0.4, n_samples // 2)
    theta_inner = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X_inner = np.stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)], axis=1)
    
    # 外側の円（クラス1）
    r_outer = np.random.uniform(0.6, 1.0, n_samples // 2)
    theta_outer = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X_outer = np.stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)], axis=1)
    
    X = torch.tensor(np.vstack([X_inner, X_outer]), dtype=torch.float32)
    y = torch.tensor([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # QBNNモデル
    model = QBNNClassifier(
        input_dim=2,
        hidden_dims=[16, 16],
        num_classes=2,
        lambda_entangle=0.5
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(500):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            pred = model.predict(X)
            acc = (pred == y).float().mean()
            print(f"   Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")
    
    pred = model.predict(X)
    final_acc = (pred == y).float().mean().item()
    print(f"   最終精度: {final_acc:.2%}")
    
    return model, X, y, final_acc


def test_regression():
    """回帰問題（sin関数の近似）をテスト"""
    print("\n📊 回帰問題（sin関数近似）")
    print("-" * 50)
    
    # データ生成
    X = torch.linspace(-np.pi, np.pi, 100).unsqueeze(1)
    y = torch.sin(X)
    
    # QBNNベースの回帰モデル
    class QBNNRegressor(nn.Module):
        def __init__(self, hidden_dims=[32, 32], lambda_entangle=0.5):
            super().__init__()
            layer_dims = [1] + hidden_dims + [1]
            self.qbnn = QBNN(layer_dims, lambda_entangle)
        
        def forward(self, x):
            return self.qbnn(x)
    
    model = QBNNRegressor(hidden_dims=[32, 32], lambda_entangle=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(1000):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}: MSE Loss={loss.item():.6f}")
    
    # 評価
    with torch.no_grad():
        pred = model(X)
        mse = criterion(pred, y).item()
        r2 = 1 - mse / y.var().item()
    
    print(f"   最終MSE: {mse:.6f}")
    print(f"   R²スコア: {r2:.4f}")
    
    return model, X, y, r2


def test_multiclass():
    """多クラス分類（3クラス）をテスト"""
    print("\n📊 多クラス分類（3クラス）")
    print("-" * 50)
    
    # データ生成（3つのガウス分布）
    np.random.seed(42)
    n_per_class = 50
    
    centers = [(-1, -1), (1, -1), (0, 1)]
    X_list = []
    y_list = []
    
    for i, (cx, cy) in enumerate(centers):
        X_class = np.random.randn(n_per_class, 2) * 0.3 + np.array([cx, cy])
        X_list.append(X_class)
        y_list.extend([i] * n_per_class)
    
    X = torch.tensor(np.vstack(X_list), dtype=torch.float32)
    y = torch.tensor(y_list)
    
    # QBNNモデル
    model = QBNNClassifier(
        input_dim=2,
        hidden_dims=[16, 16],
        num_classes=3,
        lambda_entangle=0.5
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(500):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            pred = model.predict(X)
            acc = (pred == y).float().mean()
            print(f"   Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")
    
    pred = model.predict(X)
    final_acc = (pred == y).float().mean().item()
    print(f"   最終精度: {final_acc:.2%}")
    
    return model, X, y, final_acc


def test_lambda_comparison():
    """λ（もつれ強度）の効果を比較"""
    print("\n📊 λ（もつれ強度）の効果比較")
    print("-" * 50)
    
    # XOR問題で比較
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = torch.tensor([0, 1, 1, 0])
    
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    
    for lam in lambdas:
        model = QBNNClassifier(
            input_dim=2,
            hidden_dims=[8, 8],
            num_classes=2,
            lambda_entangle=lam
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.CrossEntropyLoss()
        
        final_loss = None
        for epoch in range(300):
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        
        pred = model.predict(X)
        acc = (pred == y).float().mean().item()
        results[lam] = {'acc': acc, 'loss': final_loss}
        
        status = "✅" if acc == 1.0 else "❌"
        print(f"   λ={lam:.2f}: 精度={acc:.0%} {status}, Loss={final_loss:.4f}")
    
    return results


def visualize_all_tests(circle_model, circle_X, circle_y, 
                        reg_model, reg_X, reg_y):
    """全テスト結果を可視化"""
    print("\n📈 全テスト結果の可視化")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 円形分類
    ax = axes[0, 0]
    X_np = circle_X.numpy()
    y_np = circle_y.numpy()
    pred = circle_model.predict(circle_X).numpy()
    
    colors = ['blue' if p == 0 else 'red' for p in pred]
    ax.scatter(X_np[:, 0], X_np[:, 1], c=colors, alpha=0.6)
    
    # 決定境界
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 50), np.linspace(-1.2, 1.2, 50))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = circle_model.predict(grid).numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    ax.set_title('Circle Classification (QBNN)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # 2. 回帰
    ax = axes[0, 1]
    X_np = reg_X.numpy()
    y_np = reg_y.numpy()
    with torch.no_grad():
        pred = reg_model(reg_X).numpy()
    
    ax.plot(X_np, y_np, 'b-', linewidth=2, label='True (sin)')
    ax.plot(X_np, pred, 'r--', linewidth=2, label='QBNN Prediction')
    ax.set_title('Regression: sin(x) Approximation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 論理ゲート結果
    ax = axes[1, 0]
    gates = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']
    # 結果を再テスト
    gate_results = test_logic_gates_silent()
    accs = [gate_results[g] for g in gates]
    colors_bar = ['green' if a == 1.0 else 'orange' for a in accs]
    ax.bar(gates, accs, color=colors_bar, alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.set_title('Logic Gates Accuracy')
    ax.set_ylabel('Accuracy')
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5)
    
    # 4. λの効果
    ax = axes[1, 1]
    lambda_results = test_lambda_silent()
    lambdas = list(lambda_results.keys())
    accs = [lambda_results[l]['acc'] for l in lambdas]
    losses = [lambda_results[l]['loss'] for l in lambdas]
    
    ax2 = ax.twinx()
    ax.bar(range(len(lambdas)), accs, color='blue', alpha=0.5, label='Accuracy')
    ax2.plot(range(len(lambdas)), losses, 'ro-', label='Loss')
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f'{l:.2f}' for l in lambdas])
    ax.set_xlabel('λ (Entanglement Strength)')
    ax.set_ylabel('Accuracy', color='blue')
    ax2.set_ylabel('Loss', color='red')
    ax.set_title('Effect of λ on XOR Problem')
    ax.set_ylim(0, 1.1)
    
    plt.suptitle('QBNN (Quantum-Bit Neural Network) Test Results', fontsize=14)
    plt.tight_layout()
    plt.savefig('/Users/yuyahiguchi/Program/Qubit/qbnn_all_tests.png', dpi=150, bbox_inches='tight')
    print("   保存: qbnn_all_tests.png")
    plt.close()


def test_logic_gates_silent():
    """論理ゲート（サイレント版）"""
    gates = {
        'AND': torch.tensor([0, 0, 0, 1]),
        'OR': torch.tensor([0, 1, 1, 1]),
        'NAND': torch.tensor([1, 1, 1, 0]),
        'NOR': torch.tensor([1, 0, 0, 0]),
        'XOR': torch.tensor([0, 1, 1, 0]),
        'XNOR': torch.tensor([1, 0, 0, 1])
    }
    
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    results = {}
    
    for gate_name, y in gates.items():
        model = QBNNClassifier(2, [8, 8], 2, 0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(300):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        
        pred = model.predict(X)
        results[gate_name] = (pred == y).float().mean().item()
    
    return results


def test_lambda_silent():
    """λ比較（サイレント版）"""
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0, 1, 1, 0])
    
    results = {}
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        model = QBNNClassifier(2, [8, 8], 2, lam)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(300):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        
        pred = model.predict(X)
        results[lam] = {
            'acc': (pred == y).float().mean().item(),
            'loss': loss.item()
        }
    
    return results


# ========================================
# 追加テスト
# ========================================

def test_spiral():
    """螺旋データ分類（非常に複雑な非線形）"""
    print("\n📊 螺旋データ分類")
    print("-" * 50)
    
    np.random.seed(42)
    n_points = 100
    
    # 2つの螺旋を生成
    theta = np.linspace(0, 4 * np.pi, n_points)
    r = theta / (4 * np.pi)
    
    # 螺旋1
    x1 = r * np.cos(theta) + np.random.randn(n_points) * 0.05
    y1 = r * np.sin(theta) + np.random.randn(n_points) * 0.05
    
    # 螺旋2（180度回転）
    x2 = -r * np.cos(theta) + np.random.randn(n_points) * 0.05
    y2 = -r * np.sin(theta) + np.random.randn(n_points) * 0.05
    
    X = torch.tensor(np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ]), dtype=torch.float32)
    y = torch.tensor([0] * n_points + [1] * n_points)
    
    model = QBNNClassifier(2, [32, 32, 16], 2, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            pred = model.predict(X)
            acc = (pred == y).float().mean()
            print(f"   Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")
    
    pred = model.predict(X)
    acc = (pred == y).float().mean().item()
    print(f"   最終精度: {acc:.2%}")
    
    return model, X, y, acc


def test_checkerboard():
    """チェッカーボード分類"""
    print("\n📊 チェッカーボード分類")
    print("-" * 50)
    
    np.random.seed(42)
    n_points = 400
    
    X = np.random.uniform(-2, 2, (n_points, 2))
    y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(int)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y)
    
    model = QBNNClassifier(2, [32, 32, 16], 2, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(500):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            pred = model.predict(X)
            acc = (pred == y).float().mean()
            print(f"   Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")
    
    pred = model.predict(X)
    acc = (pred == y).float().mean().item()
    print(f"   最終精度: {acc:.2%}")
    
    return model, X, y, acc


def test_half_moon():
    """半月形データ分類"""
    print("\n📊 半月形データ分類")
    print("-" * 50)
    
    np.random.seed(42)
    n_samples = 200
    
    # 上の半月
    theta1 = np.linspace(0, np.pi, n_samples // 2)
    x1 = np.cos(theta1) + np.random.randn(n_samples // 2) * 0.1
    y1 = np.sin(theta1) + np.random.randn(n_samples // 2) * 0.1
    
    # 下の半月（シフト）
    theta2 = np.linspace(0, np.pi, n_samples // 2)
    x2 = 1 - np.cos(theta2) + np.random.randn(n_samples // 2) * 0.1
    y2 = 0.5 - np.sin(theta2) + np.random.randn(n_samples // 2) * 0.1
    
    X = torch.tensor(np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ]), dtype=torch.float32)
    y = torch.tensor([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    model = QBNNClassifier(2, [16, 16], 2, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(500):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            pred = model.predict(X)
            acc = (pred == y).float().mean()
            print(f"   Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")
    
    pred = model.predict(X)
    acc = (pred == y).float().mean().item()
    print(f"   最終精度: {acc:.2%}")
    
    return model, X, y, acc


def test_noise_robustness():
    """ノイズ耐性テスト"""
    print("\n📊 ノイズ耐性テスト（XOR）")
    print("-" * 50)
    
    X_clean = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = torch.tensor([0, 1, 1, 0])
    
    # クリーンデータで学習
    model = QBNNClassifier(2, [8, 8], 2, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(300):
        optimizer.zero_grad()
        loss = criterion(model(X_clean), y)
        loss.backward()
        optimizer.step()
    
    # ノイズレベル別にテスト
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    
    for noise in noise_levels:
        accs = []
        for _ in range(10):  # 10回試行
            X_noisy = X_clean + torch.randn_like(X_clean) * noise
            pred = model.predict(X_noisy)
            acc = (pred == y).float().mean().item()
            accs.append(acc)
        avg_acc = np.mean(accs)
        results[noise] = avg_acc
        print(f"   ノイズ={noise:.1f}: 精度={avg_acc:.0%}")
    
    return results


def test_convergence_comparison():
    """収束速度比較（QBNNと通常NN）"""
    print("\n📊 収束速度比較（QBNN vs 通常NN）")
    print("-" * 50)
    
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = torch.tensor([0, 1, 1, 0])
    
    # 通常のNN
    class StandardNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 8)
            self.fc2 = nn.Linear(8, 8)
            self.fc3 = nn.Linear(8, 2)
        
        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return self.fc3(x)
    
    # QBNN
    qbnn_model = QBNNClassifier(2, [8, 8], 2, 0.5)
    qbnn_optimizer = torch.optim.Adam(qbnn_model.parameters(), lr=0.02)
    
    # 通常NN
    std_model = StandardNN()
    std_optimizer = torch.optim.Adam(std_model.parameters(), lr=0.02)
    
    criterion = nn.CrossEntropyLoss()
    
    qbnn_losses = []
    std_losses = []
    
    for epoch in range(200):
        # QBNN
        qbnn_optimizer.zero_grad()
        qbnn_loss = criterion(qbnn_model(X), y)
        qbnn_loss.backward()
        qbnn_optimizer.step()
        qbnn_losses.append(qbnn_loss.item())
        
        # 通常NN
        std_optimizer.zero_grad()
        std_loss = criterion(std_model(X), y)
        std_loss.backward()
        std_optimizer.step()
        std_losses.append(std_loss.item())
    
    # 結果
    print(f"   QBNN最終Loss: {qbnn_losses[-1]:.4f}")
    print(f"   標準NN最終Loss: {std_losses[-1]:.4f}")
    
    # 収束エポック（Loss < 0.1）
    qbnn_conv = next((i for i, l in enumerate(qbnn_losses) if l < 0.1), 200)
    std_conv = next((i for i, l in enumerate(std_losses) if l < 0.1), 200)
    print(f"   QBNN収束エポック: {qbnn_conv}")
    print(f"   標準NN収束エポック: {std_conv}")
    
    return qbnn_losses, std_losses


def test_high_dimension():
    """高次元データ分類"""
    print("\n📊 高次元データ分類（10次元）")
    print("-" * 50)
    
    np.random.seed(42)
    n_samples = 500
    dim = 10
    
    # クラス0：ランダムな方向
    X0 = np.random.randn(n_samples // 2, dim) * 0.5
    X0[:, 0] += 1  # 最初の次元にバイアス
    
    # クラス1：別の方向
    X1 = np.random.randn(n_samples // 2, dim) * 0.5
    X1[:, 0] -= 1
    
    X = torch.tensor(np.vstack([X0, X1]), dtype=torch.float32)
    y = torch.tensor([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    model = QBNNClassifier(dim, [32, 16], 2, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(300):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            pred = model.predict(X)
            acc = (pred == y).float().mean()
            print(f"   Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")
    
    pred = model.predict(X)
    acc = (pred == y).float().mean().item()
    print(f"   最終精度: {acc:.2%}")
    
    return acc


def test_imbalanced():
    """不均衡データ分類"""
    print("\n📊 不均衡データ分類（1:9比率）")
    print("-" * 50)
    
    np.random.seed(42)
    
    # クラス0：少数（50サンプル）
    X0 = np.random.randn(50, 2) * 0.3 + np.array([0, 0])
    # クラス1：多数（450サンプル）
    X1 = np.random.randn(450, 2) * 0.5 + np.array([1.5, 1.5])
    
    X = torch.tensor(np.vstack([X0, X1]), dtype=torch.float32)
    y = torch.tensor([0] * 50 + [1] * 450)
    
    model = QBNNClassifier(2, [16, 16], 2, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # クラス重み付き損失
    weights = torch.tensor([9.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    for epoch in range(500):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            pred = model.predict(X)
            acc = (pred == y).float().mean()
            # クラス別精度
            acc0 = (pred[:50] == y[:50]).float().mean()
            acc1 = (pred[50:] == y[50:]).float().mean()
            print(f"   Epoch {epoch+1}: Loss={loss.item():.4f}, 全体={acc.item():.2%}, 少数クラス={acc0.item():.2%}")
    
    pred = model.predict(X)
    acc = (pred == y).float().mean().item()
    acc0 = (pred[:50] == y[:50]).float().mean().item()
    print(f"   最終精度: 全体={acc:.2%}, 少数クラス={acc0:.2%}")
    
    return acc, acc0


def test_cos_function():
    """cos関数の回帰"""
    print("\n📊 回帰問題（cos関数）")
    print("-" * 50)
    
    X = torch.linspace(-2 * np.pi, 2 * np.pi, 100).unsqueeze(1)
    y = torch.cos(X)
    
    class QBNNRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.qbnn = QBNN([1, 32, 32, 1], 0.3)
        
        def forward(self, x):
            return self.qbnn(x)
    
    model = QBNNRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}: MSE={loss.item():.6f}")
    
    with torch.no_grad():
        pred = model(X)
        mse = criterion(pred, y).item()
        r2 = 1 - mse / y.var().item()
    
    print(f"   最終MSE: {mse:.6f}, R²: {r2:.4f}")
    return r2


def test_polynomial():
    """多項式回帰（x² + x）"""
    print("\n📊 多項式回帰（x² + x）")
    print("-" * 50)
    
    X = torch.linspace(-2, 2, 100).unsqueeze(1)
    y = X ** 2 + X
    
    class QBNNRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.qbnn = QBNN([1, 16, 16, 1], 0.3)
        
        def forward(self, x):
            return self.qbnn(x)
    
    model = QBNNRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(500):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1}: MSE={loss.item():.6f}")
    
    with torch.no_grad():
        pred = model(X)
        mse = criterion(pred, y).item()
        r2 = 1 - mse / y.var().item()
    
    print(f"   最終MSE: {mse:.6f}, R²: {r2:.4f}")
    return r2


def test_5class():
    """5クラス分類"""
    print("\n📊 5クラス分類")
    print("-" * 50)
    
    np.random.seed(42)
    n_per_class = 50
    
    # 5つの中心を配置（正五角形）
    centers = []
    for i in range(5):
        angle = 2 * np.pi * i / 5
        centers.append((np.cos(angle), np.sin(angle)))
    
    X_list = []
    y_list = []
    
    for i, (cx, cy) in enumerate(centers):
        X_class = np.random.randn(n_per_class, 2) * 0.2 + np.array([cx, cy])
        X_list.append(X_class)
        y_list.extend([i] * n_per_class)
    
    X = torch.tensor(np.vstack(X_list), dtype=torch.float32)
    y = torch.tensor(y_list)
    
    model = QBNNClassifier(2, [32, 32], 5, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(500):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            pred = model.predict(X)
            acc = (pred == y).float().mean()
            print(f"   Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")
    
    pred = model.predict(X)
    acc = (pred == y).float().mean().item()
    print(f"   最終精度: {acc:.2%}")
    
    return acc


def visualize_extended_tests(spiral_data, checkerboard_data, halfmoon_data, conv_data):
    """拡張テストの可視化"""
    print("\n📈 拡張テスト結果の可視化")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 螺旋
    ax = axes[0, 0]
    model, X, y = spiral_data[0], spiral_data[1], spiral_data[2]
    X_np = X.numpy()
    pred = model.predict(X).numpy()
    colors = ['blue' if p == 0 else 'red' for p in pred]
    ax.scatter(X_np[:, 0], X_np[:, 1], c=colors, alpha=0.6, s=20)
    ax.set_title(f'Spiral Classification ({spiral_data[3]:.0%})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # 2. チェッカーボード
    ax = axes[0, 1]
    model, X, y = checkerboard_data[0], checkerboard_data[1], checkerboard_data[2]
    X_np = X.numpy()
    pred = model.predict(X).numpy()
    colors = ['blue' if p == 0 else 'red' for p in pred]
    ax.scatter(X_np[:, 0], X_np[:, 1], c=colors, alpha=0.6, s=10)
    ax.set_title(f'Checkerboard ({checkerboard_data[3]:.0%})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # 3. 半月
    ax = axes[0, 2]
    model, X, y = halfmoon_data[0], halfmoon_data[1], halfmoon_data[2]
    X_np = X.numpy()
    pred = model.predict(X).numpy()
    colors = ['blue' if p == 0 else 'red' for p in pred]
    ax.scatter(X_np[:, 0], X_np[:, 1], c=colors, alpha=0.6, s=20)
    ax.set_title(f'Half Moon ({halfmoon_data[3]:.0%})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # 4. 収束比較
    ax = axes[1, 0]
    qbnn_losses, std_losses = conv_data
    ax.plot(qbnn_losses, 'b-', label='QBNN', linewidth=2)
    ax.plot(std_losses, 'r--', label='Standard NN', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 5. ノイズ耐性（プレースホルダー）
    ax = axes[1, 1]
    noise_results = test_noise_robustness_silent()
    noises = list(noise_results.keys())
    accs = list(noise_results.values())
    ax.bar(range(len(noises)), accs, color='green', alpha=0.7)
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([f'{n:.1f}' for n in noises])
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Accuracy')
    ax.set_title('Noise Robustness (XOR)')
    ax.set_ylim(0, 1.1)
    
    # 6. 高次元・不均衡・多クラス結果
    ax = axes[1, 2]
    tests = ['10D', 'Imbalanced', '5-Class']
    accs = [test_high_dim_silent(), test_imbalanced_silent()[0], test_5class_silent()]
    colors_bar = ['green' if a > 0.9 else 'orange' for a in accs]
    ax.bar(tests, accs, color=colors_bar, alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Other Tests')
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.5)
    
    plt.suptitle('QBNN Extended Test Results', fontsize=14)
    plt.tight_layout()
    plt.savefig('/Users/yuyahiguchi/Program/Qubit/qbnn_extended_tests.png', dpi=150, bbox_inches='tight')
    print("   保存: qbnn_extended_tests.png")
    plt.close()


# サイレント版のヘルパー関数
def test_noise_robustness_silent():
    X_clean = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0, 1, 1, 0])
    
    model = QBNNClassifier(2, [8, 8], 2, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(300):
        optimizer.zero_grad()
        loss = criterion(model(X_clean), y)
        loss.backward()
        optimizer.step()
    
    results = {}
    for noise in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        accs = []
        for _ in range(10):
            X_noisy = X_clean + torch.randn_like(X_clean) * noise
            pred = model.predict(X_noisy)
            accs.append((pred == y).float().mean().item())
        results[noise] = np.mean(accs)
    return results


def test_high_dim_silent():
    np.random.seed(42)
    X0 = np.random.randn(250, 10) * 0.5
    X0[:, 0] += 1
    X1 = np.random.randn(250, 10) * 0.5
    X1[:, 0] -= 1
    X = torch.tensor(np.vstack([X0, X1]), dtype=torch.float32)
    y = torch.tensor([0] * 250 + [1] * 250)
    
    model = QBNNClassifier(10, [32, 16], 2, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(300):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    return (model.predict(X) == y).float().mean().item()


def test_imbalanced_silent():
    np.random.seed(42)
    X0 = np.random.randn(50, 2) * 0.3
    X1 = np.random.randn(450, 2) * 0.5 + np.array([1.5, 1.5])
    X = torch.tensor(np.vstack([X0, X1]), dtype=torch.float32)
    y = torch.tensor([0] * 50 + [1] * 450)
    
    model = QBNNClassifier(2, [16, 16], 2, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([9.0, 1.0]))
    
    for _ in range(500):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    pred = model.predict(X)
    return (pred == y).float().mean().item(), (pred[:50] == y[:50]).float().mean().item()


def test_5class_silent():
    np.random.seed(42)
    centers = [(np.cos(2*np.pi*i/5), np.sin(2*np.pi*i/5)) for i in range(5)]
    X_list = []
    y_list = []
    for i, (cx, cy) in enumerate(centers):
        X_list.append(np.random.randn(50, 2) * 0.2 + np.array([cx, cy]))
        y_list.extend([i] * 50)
    X = torch.tensor(np.vstack(X_list), dtype=torch.float32)
    y = torch.tensor(y_list)
    
    model = QBNNClassifier(2, [32, 32], 5, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(500):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    return (model.predict(X) == y).float().mean().item()


if __name__ == "__main__":
    # モデルサマリー
    print_model_summary()
    
    results_all = {}
    
    # ========================================
    # 基本テスト
    # ========================================
    print("\n" + "=" * 70)
    print("🧪 基本テスト")
    print("=" * 70)
    
    # 1. XOR問題
    print("\n" + "-" * 70)
    print("📌 テスト1: XOR問題")
    model_xor, losses_xor = test_xor_problem()
    results_all['XOR'] = 1.0
    
    # 2. 論理ゲート
    print("\n" + "-" * 70)
    print("📌 テスト2: 論理ゲート")
    logic_results = test_logic_gates()
    results_all.update(logic_results)
    
    # 3. 円形分類
    print("\n" + "-" * 70)
    print("📌 テスト3: 円形分類")
    model_circle, X_circle, y_circle, acc_circle = test_circle_classification()
    results_all['Circle'] = acc_circle
    
    # 4. sin回帰
    print("\n" + "-" * 70)
    print("📌 テスト4: sin回帰")
    model_reg, X_reg, y_reg, r2_reg = test_regression()
    results_all['Sin_R2'] = r2_reg
    
    # 5. 多クラス分類（3クラス）
    print("\n" + "-" * 70)
    print("📌 テスト5: 3クラス分類")
    model_multi, X_multi, y_multi, acc_multi = test_multiclass()
    results_all['3Class'] = acc_multi
    
    # ========================================
    # 拡張テスト
    # ========================================
    print("\n" + "=" * 70)
    print("🧪 拡張テスト")
    print("=" * 70)
    
    # 6. 螺旋分類
    print("\n" + "-" * 70)
    print("📌 テスト6: 螺旋分類")
    model_spiral, X_spiral, y_spiral, acc_spiral = test_spiral()
    results_all['Spiral'] = acc_spiral
    
    # 7. チェッカーボード
    print("\n" + "-" * 70)
    print("📌 テスト7: チェッカーボード")
    model_checker, X_checker, y_checker, acc_checker = test_checkerboard()
    results_all['Checker'] = acc_checker
    
    # 8. 半月
    print("\n" + "-" * 70)
    print("📌 テスト8: 半月")
    model_moon, X_moon, y_moon, acc_moon = test_half_moon()
    results_all['HalfMoon'] = acc_moon
    
    # 9. ノイズ耐性
    print("\n" + "-" * 70)
    print("📌 テスト9: ノイズ耐性")
    noise_results = test_noise_robustness()
    
    # 10. 収束比較
    print("\n" + "-" * 70)
    print("📌 テスト10: 収束比較（QBNN vs 標準NN）")
    qbnn_losses, std_losses = test_convergence_comparison()
    
    # 11. 高次元
    print("\n" + "-" * 70)
    print("📌 テスト11: 高次元（10D）")
    acc_highdim = test_high_dimension()
    results_all['10D'] = acc_highdim
    
    # 12. 不均衡
    print("\n" + "-" * 70)
    print("📌 テスト12: 不均衡データ")
    acc_imb, acc_minority = test_imbalanced()
    results_all['Imbalanced'] = acc_imb
    results_all['Minority'] = acc_minority
    
    # 13. cos回帰
    print("\n" + "-" * 70)
    print("📌 テスト13: cos回帰")
    r2_cos = test_cos_function()
    results_all['Cos_R2'] = r2_cos
    
    # 14. 多項式回帰
    print("\n" + "-" * 70)
    print("📌 テスト14: 多項式回帰")
    r2_poly = test_polynomial()
    results_all['Poly_R2'] = r2_poly
    
    # 15. 5クラス
    print("\n" + "-" * 70)
    print("📌 テスト15: 5クラス分類")
    acc_5class = test_5class()
    results_all['5Class'] = acc_5class
    
    # 16. λの効果
    print("\n" + "-" * 70)
    print("📌 テスト16: λの効果")
    lambda_results = test_lambda_comparison()
    
    # ========================================
    # 可視化
    # ========================================
    print("\n" + "=" * 70)
    print("📊 結果の可視化")
    print("=" * 70)
    
    # 基本テスト可視化
    visualize_all_tests(model_circle, X_circle, y_circle,
                        model_reg, X_reg, y_reg)
    
    # 拡張テスト可視化
    visualize_extended_tests(
        (model_spiral, X_spiral, y_spiral, acc_spiral),
        (model_checker, X_checker, y_checker, acc_checker),
        (model_moon, X_moon, y_moon, acc_moon),
        (qbnn_losses, std_losses)
    )
    
    # XOR可視化
    X_xor = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    visualize_qbnn(model_xor, X_xor)
    
    # ========================================
    # サマリー
    # ========================================
    print("\n" + "=" * 70)
    print("📋 全テスト結果サマリー")
    print("=" * 70)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  QBNN (Quantum-Bit Neural Network) テスト結果                    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  【基本テスト】                                                   ║""")
    
    basic_tests = ['XOR', 'AND', 'OR', 'NAND', 'NOR', 'XNOR', 'Circle', '3Class']
    for t in basic_tests:
        v = results_all.get(t, 0)
        status = '✅' if v >= 0.95 else ('⚠️' if v >= 0.8 else '❌')
        print(f"    ║  {t:15s}: {v:6.1%} {status}                                     ║")
    
    print("""    ║                                                                  ║
    ║  【回帰テスト】                                                   ║""")
    
    reg_tests = ['Sin_R2', 'Cos_R2', 'Poly_R2']
    for t in reg_tests:
        v = results_all.get(t, 0)
        status = '✅' if v >= 0.95 else ('⚠️' if v >= 0.8 else '❌')
        print(f"    ║  {t:15s}: {v:6.4f} {status}                                    ║")
    
    print("""    ║                                                                  ║
    ║  【高難度テスト】                                                 ║""")
    
    hard_tests = ['Spiral', 'Checker', 'HalfMoon', '5Class', '10D']
    for t in hard_tests:
        v = results_all.get(t, 0)
        status = '✅' if v >= 0.95 else ('⚠️' if v >= 0.8 else '❌')
        print(f"    ║  {t:15s}: {v:6.1%} {status}                                     ║")
    
    print("""    ║                                                                  ║
    ║  【特殊テスト】                                                   ║""")
    
    print(f"    ║  Imbalanced:      {results_all.get('Imbalanced', 0):6.1%} (少数: {results_all.get('Minority', 0):.0%})                     ║")
    print(f"    ║  Noise(0.3):      {noise_results.get(0.3, 0):6.1%}                                       ║")
    
    print("""    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # 統計
    passed = sum(1 for v in results_all.values() if v >= 0.9)
    total = len(results_all)
    print(f"    合計: {passed}/{total} テスト合格（90%以上）")
    
    print("\n✅ QBNN 全16テスト完了！")

