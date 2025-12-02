#!/usr/bin/env python3
"""
QBNN Brain - 脳型散在量子ビットネットワーク
============================================
従来の層状構造ではなく、ニューロン（量子ビット）が
バラバラに散らばった脳のような構造

特徴:
- 各ニューロンが独立した量子ビット（APQB）
- ニューロン間の接続はグラフ構造（ランダム/学習可能）
- 層ではなく、時間ステップで信号が伝播
- 量子もつれが任意のニューロン間で発生
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt


# ========================================
# 単一量子ビットニューロン
# ========================================

class QuantumNeuron:
    """
    単一の量子ビットニューロン
    
    APQB理論に基づく:
    - θ: 内部角度パラメータ
    - r = cos(2θ): 相関係数
    - T = |sin(2θ)|: 温度（ゆらぎ）
    """
    
    def __init__(self, neuron_id: int):
        self.id = neuron_id
        
        # 量子状態パラメータ
        self.theta = random.uniform(0.3, 1.2)  # θ ∈ [0, π/2]
        
        # 活性化値（古典的な出力）
        self.activation = 0.0
        
        # 接続先ニューロンと重み {neuron_id: weight}
        self.connections: Dict[int, float] = {}
        
        # もつれテンソル（接続先との量子もつれ強度）
        self.entanglement: Dict[int, float] = {}
        
        # 入力バッファ
        self.input_buffer = 0.0
        
        # 発火履歴
        self.spike_history: List[float] = []
    
    @property
    def r(self) -> float:
        """相関係数"""
        return math.cos(2 * self.theta)
    
    @property
    def T(self) -> float:
        """温度（ゆらぎ）"""
        return abs(math.sin(2 * self.theta))
    
    @property
    def state_0_prob(self) -> float:
        """|0⟩状態の確率"""
        return math.cos(self.theta) ** 2
    
    @property
    def state_1_prob(self) -> float:
        """|1⟩状態の確率"""
        return math.sin(self.theta) ** 2
    
    def measure(self) -> int:
        """量子測定（0 or 1）"""
        return 1 if random.random() < self.state_1_prob else 0
    
    def receive_input(self, value: float, from_neuron: int):
        """他のニューロンから入力を受け取る"""
        weight = self.connections.get(from_neuron, 0.0)
        entangle = self.entanglement.get(from_neuron, 0.0)
        
        # 量子もつれ補正
        quantum_correction = entangle * self.r * value
        
        self.input_buffer += weight * value + quantum_correction
    
    def update(self, learning_rate: float = 0.01):
        """ニューロン状態を更新"""
        # 入力に基づいてθを更新
        # tanh活性化
        self.activation = math.tanh(self.input_buffer)
        
        # θの更新（量子状態の変化）
        delta_theta = learning_rate * self.input_buffer * (1 - self.activation ** 2)
        self.theta = max(0.1, min(1.47, self.theta + delta_theta))  # π/2 - ε
        
        # スパイク記録
        self.spike_history.append(self.activation)
        if len(self.spike_history) > 100:
            self.spike_history.pop(0)
        
        # バッファクリア
        self.input_buffer = 0.0
    
    def connect_to(self, target_id: int, weight: float = None, entangle: float = None):
        """他のニューロンに接続"""
        if weight is None:
            weight = random.gauss(0, 0.5)
        if entangle is None:
            entangle = random.uniform(0.1, 0.5)
        
        self.connections[target_id] = weight
        self.entanglement[target_id] = entangle


# ========================================
# 散在型QBNN（脳型ネットワーク）
# ========================================

class QBNNBrain:
    """
    脳型散在量子ビットネットワーク（動的入出力版）
    
    - ニューロンが層ではなくバラバラに存在
    - 接続はグラフ構造（スパース）
    - 入力/出力ニューロンは動的に変化（本物の脳のように）
    - 時間ステップで信号が伝播
    
    例：
    - 目からの入力 → ランダムなニューロン群が受信
    - 耳からの入力 → 別のランダムなニューロン群が受信
    - 出力も同様に、状況に応じて異なるニューロンから取得
    """
    
    def __init__(self, num_neurons: int = 100, 
                 connection_density: float = 0.15,
                 plasticity: float = 0.1):
        """
        Args:
            num_neurons: 総ニューロン数
            connection_density: 接続密度 (0-1)
            plasticity: 可塑性（接続の変化しやすさ）
        """
        self.num_neurons = num_neurons
        self.connection_density = connection_density
        self.plasticity = plasticity
        
        # ニューロン作成（すべて同等、入力にも出力にもなれる）
        self.neurons: Dict[int, QuantumNeuron] = {}
        for i in range(num_neurons):
            self.neurons[i] = QuantumNeuron(i)
        
        # 入力/出力ニューロンは固定しない（動的に選択）
        self.all_neuron_ids = list(range(num_neurons))
        
        # 各ニューロンの「感受性」（入力を受けやすさ）
        self.sensitivity = {i: random.uniform(0.3, 1.0) for i in range(num_neurons)}
        
        # 各ニューロンの「出力傾向」（出力に選ばれやすさ）
        self.output_tendency = {i: random.uniform(0.3, 1.0) for i in range(num_neurons)}
        
        # ランダム接続を生成
        self._create_connections(connection_density)
        
        # 学習可能パラメータ用
        self.global_lambda = 0.35  # もつれ強度
    
    def _create_connections(self, density: float):
        """ランダムなグラフ接続を生成（完全ランダム）"""
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and random.random() < density:
                    weight = random.gauss(0, 0.3)
                    entangle = random.uniform(0.1, 0.4)
                    self.neurons[i].connect_to(j, weight, entangle)
    
    def select_input_neurons(self, input_size: int, input_type: str = None) -> List[int]:
        """
        入力ニューロンを動的に選択（脳の感覚器官のように）
        
        Args:
            input_size: 必要な入力ニューロン数
            input_type: 入力タイプ（'visual', 'audio', 'touch'など）
        
        Returns:
            選択された入力ニューロンのID
        """
        # 感受性に基づいて確率的に選択
        weights = [self.sensitivity[i] for i in self.all_neuron_ids]
        total = sum(weights)
        probs = [w / total for w in weights]
        
        # input_typeに基づいてバイアスをかける（オプション）
        if input_type == 'visual':
            # 視覚：後半のニューロンにバイアス
            for i in range(len(probs)):
                if i > self.num_neurons * 0.6:
                    probs[i] *= 2.0
        elif input_type == 'audio':
            # 聴覚：中間のニューロンにバイアス
            for i in range(len(probs)):
                if self.num_neurons * 0.3 < i < self.num_neurons * 0.7:
                    probs[i] *= 2.0
        elif input_type == 'touch':
            # 触覚：前半のニューロンにバイアス
            for i in range(len(probs)):
                if i < self.num_neurons * 0.4:
                    probs[i] *= 2.0
        
        # 正規化
        total = sum(probs)
        probs = [p / total for p in probs]
        
        # 重複なしで選択
        selected = []
        available = list(self.all_neuron_ids)
        available_probs = list(probs)
        
        for _ in range(min(input_size, self.num_neurons)):
            if not available:
                break
            # 正規化
            total = sum(available_probs)
            if total == 0:
                break
            normalized = [p / total for p in available_probs]
            
            idx = random.choices(range(len(available)), weights=normalized, k=1)[0]
            selected.append(available[idx])
            available.pop(idx)
            available_probs.pop(idx)
        
        return selected
    
    def select_output_neurons(self, output_size: int, output_type: str = None) -> List[int]:
        """
        出力ニューロンを動的に選択（脳の運動野のように）
        
        Args:
            output_size: 必要な出力ニューロン数
            output_type: 出力タイプ（'motor', 'speech', 'emotion'など）
        """
        # 出力傾向に基づいて確率的に選択
        weights = [self.output_tendency[i] for i in self.all_neuron_ids]
        
        # 最も活性化しているニューロンにバイアス
        for i in self.all_neuron_ids:
            weights[i] *= (1.0 + abs(self.neurons[i].activation))
        
        total = sum(weights)
        probs = [w / total for w in weights]
        
        # output_typeに基づいてバイアス
        if output_type == 'motor':
            for i in range(len(probs)):
                if i < self.num_neurons * 0.3:
                    probs[i] *= 2.0
        elif output_type == 'speech':
            for i in range(len(probs)):
                if self.num_neurons * 0.4 < i < self.num_neurons * 0.6:
                    probs[i] *= 2.0
        
        # 正規化
        total = sum(probs)
        probs = [p / total for p in probs]
        
        # 選択
        selected = []
        available = list(self.all_neuron_ids)
        available_probs = list(probs)
        
        for _ in range(min(output_size, self.num_neurons)):
            if not available:
                break
            total = sum(available_probs)
            if total == 0:
                break
            normalized = [p / total for p in available_probs]
            
            idx = random.choices(range(len(available)), weights=normalized, k=1)[0]
            selected.append(available[idx])
            available.pop(idx)
            available_probs.pop(idx)
        
        return selected
    
    def forward(self, inputs: List[float], 
                input_neurons: List[int] = None,
                output_neurons: List[int] = None,
                output_size: int = 5,
                time_steps: int = 5,
                input_type: str = None,
                output_type: str = None) -> Tuple[List[float], List[int], List[int]]:
        """
        前向き伝播（動的入出力版）
        
        Args:
            inputs: 入力値のリスト
            input_neurons: 入力ニューロンのID（Noneなら自動選択）
            output_neurons: 出力ニューロンのID（Noneなら自動選択）
            output_size: 出力サイズ（output_neuronsがNoneの場合）
            time_steps: 信号伝播のステップ数
            input_type: 入力タイプ（選択にバイアス）
            output_type: 出力タイプ（選択にバイアス）
        
        Returns:
            (出力値, 使用した入力ニューロン, 使用した出力ニューロン)
        """
        # 入力ニューロンを動的に選択
        if input_neurons is None:
            input_neurons = self.select_input_neurons(len(inputs), input_type)
        
        # 入力ニューロンに値を設定
        for i, val in enumerate(inputs):
            if i < len(input_neurons):
                neuron_id = input_neurons[i]
                self.neurons[neuron_id].activation = val
                self.neurons[neuron_id].theta = 0.25 + 0.5 * (val + 1) / 2
                # 感受性を更新（使われたニューロンは感受性が上がる）
                self.sensitivity[neuron_id] = min(1.0, self.sensitivity[neuron_id] + self.plasticity * 0.1)
        
        # 時間ステップで信号伝播
        for t in range(time_steps):
            # 全ニューロンから信号を送信
            for neuron_id, neuron in self.neurons.items():
                for target_id, weight in neuron.connections.items():
                    if target_id in self.neurons:
                        # 量子測定に基づく確率的発火
                        if random.random() < 0.7 + 0.3 * abs(neuron.activation):
                            self.neurons[target_id].receive_input(
                                neuron.activation, neuron_id
                            )
            
            # 全ニューロンを更新
            for neuron in self.neurons.values():
                neuron.update(learning_rate=0.05)
            
            # 接続の可塑性（ヘブ則的な更新）
            if random.random() < self.plasticity:
                self._update_connections()
        
        # 出力ニューロンを動的に選択
        if output_neurons is None:
            output_neurons = self.select_output_neurons(output_size, output_type)
        
        # 出力傾向を更新（使われたニューロンは出力傾向が上がる）
        for neuron_id in output_neurons:
            self.output_tendency[neuron_id] = min(1.0, self.output_tendency[neuron_id] + self.plasticity * 0.1)
        
        # 出力を収集
        outputs = [self.neurons[i].activation for i in output_neurons]
        return outputs, input_neurons, output_neurons
    
    def _update_connections(self):
        """接続の可塑性更新（ヘブ則）"""
        for neuron_id, neuron in self.neurons.items():
            for target_id in list(neuron.connections.keys()):
                if target_id in self.neurons:
                    target = self.neurons[target_id]
                    
                    # ヘブ則：同時に発火するニューロン間の接続を強化
                    if abs(neuron.activation) > 0.5 and abs(target.activation) > 0.5:
                        # 同符号なら強化、異符号なら弱化
                        if neuron.activation * target.activation > 0:
                            neuron.connections[target_id] *= 1.01
                        else:
                            neuron.connections[target_id] *= 0.99
                    
                    # 接続の減衰
                    neuron.connections[target_id] *= 0.999
        
        # 新しい接続をランダムに追加（低確率）
        if random.random() < 0.01:
            i = random.randint(0, self.num_neurons - 1)
            j = random.randint(0, self.num_neurons - 1)
            if i != j and j not in self.neurons[i].connections:
                self.neurons[i].connect_to(j, random.gauss(0, 0.1), random.uniform(0.1, 0.3))
    
    def get_quantum_state(self) -> Dict:
        """全ニューロンの量子状態を取得"""
        states = {}
        for neuron_id, neuron in self.neurons.items():
            states[neuron_id] = {
                'theta': neuron.theta,
                'r': neuron.r,
                'T': neuron.T,
                'activation': neuron.activation,
                'p0': neuron.state_0_prob,
                'p1': neuron.state_1_prob,
                'sensitivity': self.sensitivity[neuron_id],
                'output_tendency': self.output_tendency[neuron_id],
            }
        return states
    
    def measure_all(self) -> List[int]:
        """全ニューロンを量子測定"""
        return [neuron.measure() for neuron in self.neurons.values()]
    
    def get_most_active_neurons(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """最も活性化しているニューロンを取得"""
        activations = [(i, abs(n.activation)) for i, n in self.neurons.items()]
        activations.sort(key=lambda x: -x[1])
        return activations[:top_k]
    
    def visualize(self, filename: str = None, 
                  last_input_neurons: List[int] = None,
                  last_output_neurons: List[int] = None):
        """ネットワーク構造の可視化（動的入出力対応）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. ニューロン配置と接続
        ax1 = axes[0]
        ax1.set_title('ネットワーク構造（動的入出力）')
        
        # ニューロン位置を円形に配置（脳のように）
        positions = {}
        for i in range(self.num_neurons):
            angle = 2 * np.pi * i / self.num_neurons
            radius = 0.8 + 0.2 * self.sensitivity[i]
            positions[i] = (np.cos(angle) * radius, np.sin(angle) * radius)
        
        # 接続を描画
        for neuron_id, neuron in self.neurons.items():
            x1, y1 = positions[neuron_id]
            for target_id, weight in neuron.connections.items():
                if target_id in positions:
                    x2, y2 = positions[target_id]
                    color = 'blue' if weight > 0 else 'red'
                    alpha = min(abs(weight), 1.0) * 0.2
                    ax1.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.3)
        
        # ニューロンを描画
        for i, (x, y) in positions.items():
            # 最後に使われた入力/出力ニューロンを色分け
            if last_input_neurons and i in last_input_neurons:
                color = 'green'
                size = 80
            elif last_output_neurons and i in last_output_neurons:
                color = 'red'
                size = 80
            else:
                # 活性化度に応じた色
                act = abs(self.neurons[i].activation)
                color = plt.cm.viridis(act)
                size = 30 + 50 * act
            ax1.scatter(x, y, c=[color], s=size, zorder=5, alpha=0.7)
        
        ax1.set_xlim(-1.3, 1.3)
        ax1.set_ylim(-1.3, 1.3)
        ax1.set_aspect('equal')
        
        # 2. 感受性と出力傾向
        ax2 = axes[1]
        ax2.set_title('感受性 vs 出力傾向')
        sens = [self.sensitivity[i] for i in range(self.num_neurons)]
        out_tend = [self.output_tendency[i] for i in range(self.num_neurons)]
        colors = [abs(self.neurons[i].activation) for i in range(self.num_neurons)]
        scatter = ax2.scatter(sens, out_tend, c=colors, cmap='plasma', alpha=0.6)
        ax2.set_xlabel('感受性（入力されやすさ）')
        ax2.set_ylabel('出力傾向（出力されやすさ）')
        plt.colorbar(scatter, ax=ax2, label='活性化度')
        
        # 3. r-T空間
        ax3 = axes[2]
        ax3.set_title('r-T空間（相関-温度）')
        rs = [n.r for n in self.neurons.values()]
        Ts = [n.T for n in self.neurons.values()]
        activations = [abs(n.activation) for n in self.neurons.values()]
        scatter = ax3.scatter(rs, Ts, c=activations, cmap='hot', alpha=0.6)
        ax3.set_xlabel('r (相関)')
        ax3.set_ylabel('T (温度)')
        plt.colorbar(scatter, ax=ax3, label='活性化度')
        
        # r²+T²=1 の円
        theta_circle = np.linspace(0, np.pi/2, 100)
        ax3.plot(np.cos(2*theta_circle), np.abs(np.sin(2*theta_circle)), 
                'k--', alpha=0.3, label='r²+T²=1')
        ax3.legend()
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📊 グラフを保存: {filename}")
        
        plt.close()
        return fig


# ========================================
# PyTorch版（学習可能、動的入出力対応）
# ========================================

class QBNNBrainTorch(nn.Module):
    """
    PyTorch版 脳型QBNN（学習可能、動的入出力）
    
    入力/出力ニューロンが固定ではなく、動的に選択される
    """
    
    def __init__(self, num_neurons: int = 50, 
                 max_input_size: int = 20, 
                 max_output_size: int = 10,
                 connection_density: float = 0.2):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.max_input_size = max_input_size
        self.max_output_size = max_output_size
        
        # 各ニューロンの量子状態パラメータ θ
        self.theta = nn.Parameter(torch.rand(num_neurons) * 1.0 + 0.25)
        
        # 各ニューロンの感受性（入力されやすさ）
        self.sensitivity = nn.Parameter(torch.rand(num_neurons) * 0.5 + 0.3)
        
        # 各ニューロンの出力傾向
        self.output_tendency = nn.Parameter(torch.rand(num_neurons) * 0.5 + 0.3)
        
        # 接続行列（スパース）
        mask = torch.rand(num_neurons, num_neurons) < connection_density
        mask.fill_diagonal_(False)  # 自己接続なし
        self.register_buffer('connection_mask', mask.float())
        
        # 重み行列
        self.weights = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.3)
        
        # もつれテンソル
        self.J = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.1)
        
        # もつれ強度
        self.lambda_entangle = nn.Parameter(torch.tensor(0.35))
        
        # 汎用入力射影（任意サイズの入力を受け取れる）
        self.input_proj = nn.Linear(max_input_size, num_neurons)
        self.output_proj = nn.Linear(num_neurons, max_output_size)
        
        # 最後に使用した入力/出力ニューロン
        self.last_input_neurons = None
        self.last_output_neurons = None
    
    def get_r(self) -> torch.Tensor:
        """相関係数 r = cos(2θ)"""
        return torch.cos(2 * self.theta)
    
    def get_T(self) -> torch.Tensor:
        """温度 T = |sin(2θ)|"""
        return torch.abs(torch.sin(2 * self.theta))
    
    def select_neurons(self, tendency: torch.Tensor, k: int, 
                       temperature: float = 1.0) -> torch.Tensor:
        """傾向に基づいてニューロンを確率的に選択"""
        # ソフトマックスで確率に変換
        probs = F.softmax(tendency / temperature, dim=-1)
        
        # Gumbel-Softmax的な選択（微分可能）
        noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
        scores = (torch.log(probs + 1e-8) + noise) / temperature
        
        # Top-K選択
        _, indices = torch.topk(scores, k)
        return indices
    
    def forward(self, x: torch.Tensor, 
                input_size: int = None,
                output_size: int = None,
                time_steps: int = 3,
                dynamic_selection: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向き伝播（動的入出力版）
        
        Args:
            x: (batch, input_dim) 入力
            input_size: 使用する入力ニューロン数
            output_size: 使用する出力ニューロン数
            time_steps: 伝播ステップ数
            dynamic_selection: 動的にニューロンを選択するか
        
        Returns:
            (output, input_neurons, output_neurons)
        """
        batch_size = x.size(0)
        actual_input_size = x.size(1)
        
        if input_size is None:
            input_size = min(actual_input_size, self.num_neurons // 2)
        if output_size is None:
            output_size = self.max_output_size
        
        # 入力をパディング
        if actual_input_size < self.max_input_size:
            padding = torch.zeros(batch_size, self.max_input_size - actual_input_size, device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x[:, :self.max_input_size]
        
        # 初期状態：入力をニューロンに射影
        state = self.input_proj(x_padded)  # (batch, num_neurons)
        
        # 動的に入力ニューロンを選択
        if dynamic_selection:
            input_neurons = self.select_neurons(self.sensitivity, input_size)
            # 選択されたニューロンに入力を集中
            input_mask = torch.zeros(self.num_neurons, device=x.device)
            input_mask[input_neurons] = 1.0
            state = state * input_mask
        else:
            input_neurons = torch.arange(input_size, device=x.device)
        
        self.last_input_neurons = input_neurons
        
        # 有効な重み（マスク適用）
        effective_weights = self.weights * self.connection_mask
        
        # 時間ステップで伝播
        for t in range(time_steps):
            # 通常の信号伝播
            signal = torch.matmul(state, effective_weights)  # (batch, num_neurons)
            
            # 量子もつれ補正（効率化版）
            s = torch.tanh(state)  # 正規化 [-1, 1]
            J_masked = self.J * self.connection_mask
            
            # バッチ処理
            entangle_correction = torch.einsum('bi,ij,bj->bj', s, J_masked, s)
            
            # 有効入力
            effective_input = signal + self.lambda_entangle * entangle_correction
            
            # 活性化（量子的ゆらぎを含む）
            T = self.get_T()  # (num_neurons,)
            noise = torch.randn_like(state) * T.unsqueeze(0) * 0.1
            state = torch.tanh(effective_input + noise)
        
        # 動的に出力ニューロンを選択
        if dynamic_selection:
            # 活性化度と出力傾向を組み合わせ
            combined_tendency = self.output_tendency + state.mean(0).abs()
            output_neurons = self.select_neurons(combined_tendency, output_size)
        else:
            output_neurons = torch.arange(self.num_neurons - output_size, self.num_neurons, device=x.device)
        
        self.last_output_neurons = output_neurons
        
        # 出力射影
        output = self.output_proj(state)
        output = output[:, :output_size]
        
        return output, input_neurons, output_neurons
    
    def get_quantum_info(self) -> Dict:
        """量子情報を取得"""
        with torch.no_grad():
            return {
                'theta_mean': self.theta.mean().item(),
                'theta_std': self.theta.std().item(),
                'r_mean': self.get_r().mean().item(),
                'T_mean': self.get_T().mean().item(),
                'lambda': self.lambda_entangle.item(),
                'connections': self.connection_mask.sum().item(),
                'sensitivity_mean': self.sensitivity.mean().item(),
                'output_tendency_mean': self.output_tendency.mean().item(),
            }


# ========================================
# デモ
# ========================================

def demo_brain_qbnn():
    """脳型QBNNのデモ（動的入出力版）"""
    print("=" * 60)
    print("🧠 QBNN Brain - 脳型散在量子ビットネットワーク")
    print("   入力/出力が動的に変化する本物の脳のようなモデル")
    print("=" * 60)
    
    # 純粋Python版
    print("\n📌 純粋Python版（50ニューロン、動的入出力）")
    print("-" * 40)
    
    brain = QBNNBrain(
        num_neurons=50,
        connection_density=0.15,
        plasticity=0.1
    )
    
    # 異なる入力タイプでテスト
    print("\n🔹 視覚入力テスト:")
    inputs = [0.5, -0.3, 0.8, -0.1, 0.6]
    outputs, in_neurons, out_neurons = brain.forward(
        inputs, 
        output_size=3,
        time_steps=5,
        input_type='visual',
        output_type='motor'
    )
    print(f"   入力: {inputs}")
    print(f"   使用した入力ニューロン: {in_neurons}")
    print(f"   使用した出力ニューロン: {out_neurons}")
    print(f"   出力: {[f'{o:.4f}' for o in outputs]}")
    
    print("\n🔹 聴覚入力テスト（同じ入力、異なるニューロン選択）:")
    outputs2, in_neurons2, out_neurons2 = brain.forward(
        inputs, 
        output_size=3,
        time_steps=5,
        input_type='audio',
        output_type='speech'
    )
    print(f"   使用した入力ニューロン: {in_neurons2}")
    print(f"   使用した出力ニューロン: {out_neurons2}")
    print(f"   出力: {[f'{o:.4f}' for o in outputs2]}")
    
    # 入力/出力ニューロンが変わることを確認
    print(f"\n⚛️ 動的選択の確認:")
    print(f"   入力ニューロンが同じ: {set(in_neurons) == set(in_neurons2)}")
    print(f"   出力ニューロンが同じ: {set(out_neurons) == set(out_neurons2)}")
    
    # 最も活性化しているニューロン
    print("\n🔥 最も活性化しているニューロン:")
    active = brain.get_most_active_neurons(top_k=5)
    for neuron_id, activation in active:
        print(f"   ニューロン{neuron_id}: 活性化={activation:.4f}")
    
    # 量子測定
    measurements = brain.measure_all()
    print(f"\n🔬 量子測定結果: {sum(measurements)}/{len(measurements)} が |1⟩")
    
    # 可視化
    brain.visualize(
        '/Users/yuyahiguchi/Program/Qubit/qbnn_brain_dynamic.png',
        last_input_neurons=in_neurons,
        last_output_neurons=out_neurons
    )
    
    # PyTorch版
    print("\n" + "=" * 60)
    print("📌 PyTorch版（学習可能、動的入出力）")
    print("-" * 40)
    
    model = QBNNBrainTorch(
        num_neurons=40,
        max_input_size=10,
        max_output_size=5,
        connection_density=0.2
    )
    
    # 推論テスト
    x = torch.tensor([[0.5, -0.3, 0.8, -0.1, 0.6]], dtype=torch.float32)
    output, in_idx, out_idx = model(x, input_size=5, output_size=3, time_steps=3)
    
    print(f"\n入力: {x[0].tolist()}")
    print(f"使用した入力ニューロン: {in_idx.tolist()}")
    print(f"使用した出力ニューロン: {out_idx.tolist()}")
    print(f"出力: {output[0].detach().tolist()[:3]}")
    
    # 同じ入力で再度実行（異なるニューロンが選ばれる）
    output2, in_idx2, out_idx2 = model(x, input_size=5, output_size=3, time_steps=3)
    print(f"\n🔄 再実行:")
    print(f"   使用した入力ニューロン: {in_idx2.tolist()}")
    print(f"   入力ニューロンが変化: {not torch.equal(in_idx, in_idx2)}")
    
    # 量子情報
    info = model.get_quantum_info()
    print(f"\n⚛️ 量子情報:")
    print(f"   θ平均: {info['theta_mean']:.4f}")
    print(f"   r平均: {info['r_mean']:.4f}")
    print(f"   T平均: {info['T_mean']:.4f}")
    print(f"   λ: {info['lambda']:.4f}")
    print(f"   感受性平均: {info['sensitivity_mean']:.4f}")
    print(f"   出力傾向平均: {info['output_tendency_mean']:.4f}")
    
    # 簡単な学習テスト
    print("\n📚 学習テスト（XOR問題）...")
    
    # XORデータ
    X = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ], dtype=torch.float32)
    
    y = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(100):
        optimizer.zero_grad()
        pred, _, _ = model(X, input_size=5, output_size=3, time_steps=3, dynamic_selection=False)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    # 最終結果
    print("\n🎯 学習後の予測:")
    model.eval()
    with torch.no_grad():
        pred, _, _ = model(X, input_size=5, output_size=3, time_steps=3, dynamic_selection=False)
        for i in range(4):
            input_str = f"({int(X[i,0])},{int(X[i,1])})"
            pred_val = pred[i, 0].item()
            expected = y[i, 0].item()
            status = "✅" if abs(pred_val - expected) < 0.3 else "❌"
            print(f"   {status} {input_str} → 予測: {pred_val:.3f}, 正解: {expected:.0f}")
    
    print("\n✅ デモ完了！")
    
    return brain, model


def compare_architectures():
    """層状 vs 散在型の比較"""
    print("=" * 60)
    print("📊 アーキテクチャ比較: 層状 vs 散在型")
    print("=" * 60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│              従来の層状ニューラルネットワーク                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   入力層        隠れ層1       隠れ層2       出力層            │
│                                                             │
│    ○──────────○──────────○──────────○                    │
│    ○──────────○──────────○──────────○                    │
│    ○──────────○──────────○──────────○                    │
│    ○──────────○──────────○                                │
│    ○──────────○──────────○                                │
│                                                             │
│   特徴: 整然とした層構造、情報は左→右に流れる               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              QBNN Brain（脳型散在ネットワーク）               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│           ○───────○                                        │
│          ╱│╲     ╱│╲                                       │
│    入力  ○─┼──○───┼──○  出力                               │
│          ╲│╱  │╲ ╱│  ╲                                     │
│           ○───┼─○───○                                      │
│            ╲  │╱    ╱                                       │
│             ○──○───○                                       │
│                                                             │
│   特徴: ニューロンがバラバラに散らばり、任意の接続が可能      │
│         量子もつれが任意のニューロン間で発生                  │
│         時間ステップで信号が伝播（非同期的）                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    """)
    
    print("\n⚛️ QBNN Brainの特徴:")
    print("   1. 各ニューロンが独立した量子ビット（APQB）")
    print("   2. θパラメータで状態を表現 (r=cos2θ, T=|sin2θ|)")
    print("   3. 接続はスパースなグラフ構造")
    print("   4. 量子もつれ（J テンソル）が任意のニューロン間で発生")
    print("   5. 時間ステップで信号が伝播")


if __name__ == '__main__':
    compare_architectures()
    brain, model = demo_brain_qbnn()

