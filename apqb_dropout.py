#!/usr/bin/env python3
"""
APQB Dropout - Adjustable Pseudo Quantum Bit ベースのドロップアウト層

従来のドロップアウト: 固定確率でランダムにニューロンを無効化
APQBドロップアウト: 量子確率分布に基づいてニューロンをドロップ

相関係数 r で確率分布を制御:
  r = 1  → ほぼドロップなし（すべて保持）
  r = -1 → ほぼ全ドロップ（すべて無効）
  r = 0  → 50%の確率でドロップ（通常のドロップアウト相当）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ========== APQB (Adjustable Pseudo Quantum Bit) ==========
class APQB:
    """調整可能な擬似量子ビット"""
    
    def __init__(self, r: float = 0.0):
        """
        Args:
            r: 相関係数 (-1 to 1)
               r = 1  → |0⟩ (確実に0 = ドロップ)
               r = -1 → |1⟩ (確実に1 = 保持)
               r = 0  → 等確率の重ね合わせ
        """
        self.r = np.clip(r, -1, 1)
    
    @property
    def theta(self) -> float:
        """相関係数から角度を計算"""
        return np.pi * (1 - self.r) / 2
    
    @property
    def p_keep(self) -> float:
        """保持確率 (|1⟩の確率)"""
        return np.sin(self.theta / 2) ** 2
    
    @property
    def p_drop(self) -> float:
        """ドロップ確率 (|0⟩の確率)"""
        return np.cos(self.theta / 2) ** 2
    
    def measure(self) -> int:
        """量子測定: 1=保持, 0=ドロップ"""
        return 1 if np.random.random() < self.p_keep else 0


# ========== APQBドロップアウト層 (PyTorch) ==========
class APQBDropout(nn.Module):
    """
    APQBベースのドロップアウト層
    
    従来のDropoutとの違い:
    - 相関係数 r でドロップ確率を制御
    - 量子的な確率分布を使用
    - 動的に r を変更可能
    """
    
    def __init__(self, r: float = 0.0, learnable: bool = False):
        """
        Args:
            r: 初期相関係数 (-1 to 1)
               r = 0 は従来の dropout(p=0.5) に相当
            learnable: Trueの場合、rを学習可能なパラメータにする
        """
        super().__init__()
        
        if learnable:
            # 学習可能なパラメータとして r を定義
            self._r = nn.Parameter(torch.tensor(float(r)))
        else:
            self.register_buffer('_r', torch.tensor(float(r)))
        
        self.learnable = learnable
    
    @property
    def r(self) -> float:
        """現在の相関係数"""
        return float(self._r.clamp(-1, 1))
    
    @r.setter
    def r(self, value: float):
        """相関係数を設定"""
        with torch.no_grad():
            self._r.fill_(np.clip(value, -1, 1))
    
    def get_keep_probability(self) -> float:
        """現在の保持確率を取得"""
        r = self.r
        theta = np.pi * (1 - r) / 2
        return np.sin(theta / 2) ** 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル
        
        Returns:
            APQBドロップアウトを適用したテンソル
        """
        if not self.training:
            return x
        
        # 相関係数から保持確率を計算
        r = self._r.clamp(-1, 1)
        theta = np.pi * (1 - r.item()) / 2
        p_keep = np.sin(theta / 2) ** 2
        
        # APQBベースのマスクを生成
        # 各要素について独立にAPQB測定を行う
        mask = self._generate_apqb_mask(x.shape, p_keep, x.device)
        
        # スケーリング（ドロップアウトと同様）
        if p_keep > 0:
            return x * mask / p_keep
        else:
            return x * 0
    
    def _generate_apqb_mask(self, shape, p_keep, device):
        """APQBベースのマスクを生成"""
        # 量子的なランダム性を模倣
        # 複数のAPQB測定を組み合わせてマスクを生成
        
        # 基本的な一様乱数
        rand = torch.rand(shape, device=device)
        
        # APQBの量子ゆらぎを追加
        # 複数の「量子ビット」からの干渉効果を模倣
        quantum_noise = torch.zeros(shape, device=device)
        for i in range(4):  # 4つのAPQBからの干渉
            phase = torch.rand(shape, device=device) * 2 * np.pi
            quantum_noise += torch.sin(phase) * 0.1
        
        # 干渉を加えた乱数でマスクを決定
        adjusted_rand = rand + quantum_noise
        mask = (adjusted_rand < p_keep).float()
        
        return mask
    
    def extra_repr(self) -> str:
        return f'r={self.r:.3f}, p_keep={self.get_keep_probability():.3f}, learnable={self.learnable}'


# ========== APQBを使ったニューラルネットワーク ==========
class APQBNeuralNet(nn.Module):
    """APQBドロップアウトを使用したニューラルネットワーク"""
    
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int,
                 dropout_r: float = 0.0, learnable_dropout: bool = False):
        """
        Args:
            input_size: 入力次元
            hidden_sizes: 隠れ層のサイズのリスト
            output_size: 出力次元
            dropout_r: APQBドロップアウトの相関係数
            learnable_dropout: ドロップアウト率を学習可能にするか
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # 隠れ層を構築
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(APQBDropout(r=dropout_r, learnable=learnable_dropout))
            prev_size = hidden_size
        
        # 出力層
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # 活性化関数
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = self.activation(x)
            x = dropout(x)
        
        x = self.output_layer(x)
        return x
    
    def set_dropout_r(self, r: float):
        """全APQBドロップアウト層の相関係数を設定"""
        for dropout in self.dropouts:
            dropout.r = r
    
    def get_dropout_stats(self) -> dict:
        """ドロップアウト層の統計を取得"""
        stats = []
        for i, dropout in enumerate(self.dropouts):
            stats.append({
                'layer': i,
                'r': dropout.r,
                'p_keep': dropout.get_keep_probability()
            })
        return stats


# ========== 従来のドロップアウトとの比較ネットワーク ==========
class StandardNeuralNet(nn.Module):
    """従来のドロップアウトを使用したニューラルネットワーク"""
    
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int,
                 dropout_p: float = 0.5):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(nn.Dropout(p=dropout_p))
            prev_size = hidden_size
        
        self.output_layer = nn.Linear(prev_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = self.activation(x)
            x = dropout(x)
        
        x = self.output_layer(x)
        return x


# ========== デモ: XOR問題での比較 ==========
def demo_xor():
    """XOR問題でAPQBドロップアウトと従来ドロップアウトを比較"""
    print("=" * 60)
    print("🧠⚛️ APQB Dropout デモ - XOR問題")
    print("=" * 60)
    
    # XORデータセット
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    # データを拡張（ノイズを追加）
    X_train = X.repeat(100, 1) + torch.randn(400, 2) * 0.1
    y_train = y.repeat(100, 1)
    
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # モデルを比較
    results = {}
    
    # 1. APQBドロップアウト (r=0, 50%ドロップ相当)
    print("\n📊 APQBドロップアウト (r=0.0)")
    apqb_model = APQBNeuralNet(2, [16, 16], 1, dropout_r=0.0)
    apqb_losses = train_model(apqb_model, dataloader, epochs=200)
    apqb_acc = evaluate_xor(apqb_model, X, y)
    results['APQB (r=0.0)'] = {'losses': apqb_losses, 'accuracy': apqb_acc}
    
    # 2. APQBドロップアウト (r=0.5, 低ドロップ率)
    print("\n📊 APQBドロップアウト (r=0.5)")
    apqb_model2 = APQBNeuralNet(2, [16, 16], 1, dropout_r=0.5)
    apqb_losses2 = train_model(apqb_model2, dataloader, epochs=200)
    apqb_acc2 = evaluate_xor(apqb_model2, X, y)
    results['APQB (r=0.5)'] = {'losses': apqb_losses2, 'accuracy': apqb_acc2}
    
    # 3. APQBドロップアウト (r=-0.5, 高ドロップ率)
    print("\n📊 APQBドロップアウト (r=-0.5)")
    apqb_model3 = APQBNeuralNet(2, [16, 16], 1, dropout_r=-0.5)
    apqb_losses3 = train_model(apqb_model3, dataloader, epochs=200)
    apqb_acc3 = evaluate_xor(apqb_model3, X, y)
    results['APQB (r=-0.5)'] = {'losses': apqb_losses3, 'accuracy': apqb_acc3}
    
    # 4. 従来のドロップアウト
    print("\n📊 従来のドロップアウト (p=0.5)")
    std_model = StandardNeuralNet(2, [16, 16], 1, dropout_p=0.5)
    std_losses = train_model(std_model, dataloader, epochs=200)
    std_acc = evaluate_xor(std_model, X, y)
    results['Standard (p=0.5)'] = {'losses': std_losses, 'accuracy': std_acc}
    
    # 5. ドロップアウトなし
    print("\n📊 ドロップアウトなし")
    no_drop_model = APQBNeuralNet(2, [16, 16], 1, dropout_r=1.0)  # r=1でほぼドロップなし
    no_drop_losses = train_model(no_drop_model, dataloader, epochs=200)
    no_drop_acc = evaluate_xor(no_drop_model, X, y)
    results['No Dropout'] = {'losses': no_drop_losses, 'accuracy': no_drop_acc}
    
    # 結果を表示
    print("\n" + "=" * 60)
    print("📈 結果比較")
    print("=" * 60)
    for name, data in results.items():
        print(f"{name:20s}: 精度 = {data['accuracy']*100:.1f}%, 最終損失 = {data['losses'][-1]:.4f}")
    
    # グラフを描画
    plot_comparison(results)
    
    return results


def train_model(model, dataloader, epochs=100, lr=0.01):
    """モデルを訓練"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return losses


def evaluate_xor(model, X, y):
    """XOR問題の精度を評価"""
    model.eval()
    with torch.no_grad():
        output = model(X)
        predictions = (torch.sigmoid(output) > 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    return accuracy


def plot_comparison(results):
    """結果を可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 損失曲線
    ax1 = axes[0]
    colors = ['#00d4ff', '#ff00ff', '#00ff88', '#ffaa00', '#ff6666']
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(data['losses'], label=name, color=colors[i % len(colors)], linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#1a1a2e')
    
    # 精度比較
    ax2 = axes[1]
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in names]
    bars = ax2.bar(range(len(names)), accuracies, color=colors[:len(names)])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Final Accuracy Comparison')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#1a1a2e')
    
    # 精度値をバーの上に表示
    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    fig.patch.set_facecolor('#0a0a1a')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    plt.tight_layout()
    plt.savefig('/Users/yuyahiguchi/Program/Qubit/apqb_dropout_comparison.png', 
                dpi=150, facecolor='#0a0a1a', edgecolor='none')
    print("\n📊 グラフを保存しました: apqb_dropout_comparison.png")
    plt.close()


# ========== APQBドロップアウトの確率分布を可視化 ==========
def visualize_apqb_dropout():
    """APQBドロップアウトの確率分布を可視化"""
    print("\n" + "=" * 60)
    print("📊 APQBドロップアウト確率分布")
    print("=" * 60)
    
    r_values = np.linspace(-1, 1, 100)
    p_keeps = []
    
    for r in r_values:
        apqb = APQB(r)
        p_keeps.append(apqb.p_keep)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(r_values, p_keeps, 'c-', linewidth=3, label='P(keep) = sin²(π(1-r)/4)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (標準ドロップアウト)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 特定のr値をマーク
    special_rs = [-1, -0.5, 0, 0.5, 1]
    for r in special_rs:
        apqb = APQB(r)
        ax.plot(r, apqb.p_keep, 'mo', markersize=10)
        ax.annotate(f'r={r}\nP={apqb.p_keep:.2f}', 
                   xy=(r, apqb.p_keep), xytext=(r+0.1, apqb.p_keep+0.1),
                   fontsize=9, color='white')
    
    ax.set_xlabel('相関係数 r', fontsize=12, color='white')
    ax.set_ylabel('保持確率 P(keep)', fontsize=12, color='white')
    ax.set_title('APQB Dropout: 相関係数と保持確率の関係', fontsize=14, color='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    fig.patch.set_facecolor('#0a0a1a')
    plt.tight_layout()
    plt.savefig('/Users/yuyahiguchi/Program/Qubit/apqb_dropout_probability.png',
                dpi=150, facecolor='#0a0a1a', edgecolor='none')
    print("📊 グラフを保存しました: apqb_dropout_probability.png")
    plt.close()
    
    # 確率表を表示
    print("\n相関係数 r と保持確率 P(keep) の対応:")
    print("-" * 40)
    for r in [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]:
        apqb = APQB(r)
        bar = '█' * int(apqb.p_keep * 20)
        print(f"r = {r:+.2f}: P(keep) = {apqb.p_keep:.3f} |{bar}")


# ========== メイン ==========
if __name__ == "__main__":
    import sys
    
    print("🧠⚛️ APQB Dropout - 量子確率的ドロップアウト層")
    print()
    
    # APQBドロップアウトの確率分布を可視化
    visualize_apqb_dropout()
    
    # XOR問題でのデモ
    demo_xor()
    
    print("\n✅ 完了！")

