#!/usr/bin/env python3
"""
QBNN パスファインダー
======================
QBNNを使った迷路の経路探索AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import time
from typing import List, Tuple, Optional
from collections import deque


# ========================================
# QBNN Layer
# ========================================

class QBNNLayer(nn.Module):
    """Quantum-Bit Neural Network Layer"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 lambda_min: float = 0.2, lambda_max: float = 0.5):
        super().__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        
        self.W = nn.Linear(input_dim, output_dim)
        self.J = nn.Parameter(torch.randn(input_dim, output_dim) * 0.02)
        self.lambda_base = nn.Parameter(torch.tensor(0.5))
        self.layer_norm = nn.LayerNorm(output_dim)
        self.call_count = 0
    
    def forward(self, h_prev: torch.Tensor) -> torch.Tensor:
        s_prev = torch.tanh(h_prev)
        h_tilde = self.W(h_prev)
        s_raw = torch.tanh(h_tilde)
        
        delta = torch.einsum('...i,ij,...j->...j', s_prev, self.J, s_raw)
        
        lambda_normalized = torch.sigmoid(self.lambda_base)
        if not self.training:
            phase = self.call_count * 0.2
            dynamic_factor = 0.5 + 0.5 * math.sin(phase)
            self.call_count += 1
        else:
            dynamic_factor = 0.5
        
        lambda_range = self.lambda_max - self.lambda_min
        lambda_eff = self.lambda_min + lambda_range * (lambda_normalized * 0.7 + dynamic_factor * 0.3)
        
        h_hat = h_tilde + lambda_eff * delta
        output = self.layer_norm(h_hat)
        return F.gelu(output)


# ========================================
# QBNN パスファインダー
# ========================================

class QBNNPathfinder(nn.Module):
    """QBNNベースの経路探索AI"""
    
    def __init__(self, grid_size: int = 10, hidden_dim: int = 128):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        
        # 入力: 現在位置(2) + ゴール位置(2) + 周囲8方向の壁情報(8) + 距離情報(1) = 13
        self.input_dim = 13
        # 出力: 4方向の移動確率 (上, 下, 左, 右)
        self.output_dim = 4
        
        # QBNNレイヤー
        self.qbnn1 = QBNNLayer(self.input_dim, hidden_dim)
        self.qbnn2 = QBNNLayer(hidden_dim, hidden_dim)
        self.qbnn3 = QBNNLayer(hidden_dim, hidden_dim // 2)
        
        # 出力層
        self.output = nn.Linear(hidden_dim // 2, self.output_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.qbnn1(state)
        x = self.qbnn2(x)
        x = self.qbnn3(x)
        return self.output(x)
    
    def get_action(self, state: torch.Tensor, valid_actions: List[int], 
                   temperature: float = 0.3) -> int:
        """次の行動を選択"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(state.unsqueeze(0))[0]
            
            # 無効な行動をマスク
            mask = torch.full((4,), float('-inf'))
            for a in valid_actions:
                mask[a] = 0
            logits = logits + mask
            
            # 温度付きソフトマックス
            probs = F.softmax(logits / temperature, dim=0)
            
            # 量子的サンプリング
            if random.random() < 0.1:  # 10%探索
                return random.choice(valid_actions)
            
            return torch.argmax(probs).item()


# ========================================
# 迷路クラス
# ========================================

class Maze:
    """迷路生成と管理"""
    
    # 方向: 0=上, 1=下, 2=左, 3=右
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    DIR_NAMES = ['↑', '↓', '←', '→']
    
    def __init__(self, size: int = 10):
        self.size = size
        self.grid = None
        self.start = None
        self.goal = None
        self.path = []
        self.visited = set()
    
    def generate(self, wall_density: float = 0.25):
        """迷路生成"""
        self.grid = [[0] * self.size for _ in range(self.size)]
        
        # ランダムに壁を配置
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < wall_density:
                    self.grid[i][j] = 1  # 壁
        
        # スタートとゴールを設定
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        
        # スタートとゴールは必ず空に
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0
        
        # 経路が存在することを確認
        if not self._has_path():
            self.generate(wall_density)  # 再生成
        
        self.path = []
        self.visited = set()
    
    def _has_path(self) -> bool:
        """BFSで経路存在確認"""
        queue = deque([self.start])
        visited = {self.start}
        
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal:
                return True
            
            for dr, dc in self.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.size and 0 <= nc < self.size and 
                    self.grid[nr][nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return False
    
    def get_shortest_path(self) -> List[Tuple[int, int]]:
        """BFSで最短経路を取得"""
        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == self.goal:
                return path
            
            for dr, dc in self.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.size and 0 <= nc < self.size and 
                    self.grid[nr][nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        
        return []
    
    def get_valid_actions(self, pos: Tuple[int, int]) -> List[int]:
        """有効な行動リスト"""
        r, c = pos
        valid = []
        for i, (dr, dc) in enumerate(self.DIRECTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr][nc] == 0:
                valid.append(i)
        return valid
    
    def get_state(self, pos: Tuple[int, int]) -> torch.Tensor:
        """現在状態をテンソルに変換"""
        r, c = pos
        gr, gc = self.goal
        
        # 位置情報（正規化）
        state = [
            r / self.size,
            c / self.size,
            gr / self.size,
            gc / self.size,
        ]
        
        # 周囲8方向の壁情報
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    state.append(float(self.grid[nr][nc]))
                else:
                    state.append(1.0)  # 範囲外は壁扱い
        
        # ゴールまでの距離（正規化）
        dist = abs(gr - r) + abs(gc - c)
        state.append(dist / (2 * self.size))
        
        return torch.tensor(state, dtype=torch.float32)
    
    def display(self, current_pos: Optional[Tuple[int, int]] = None, 
                path: Optional[List[Tuple[int, int]]] = None):
        """迷路表示"""
        path_set = set(path) if path else set()
        
        print("\n  " + " ".join([str(i % 10) for i in range(self.size)]))
        for i in range(self.size):
            row = f"{i % 10} "
            for j in range(self.size):
                if current_pos and (i, j) == current_pos:
                    row += "🔵"
                elif (i, j) == self.start:
                    row += "🟢"
                elif (i, j) == self.goal:
                    row += "🔴"
                elif (i, j) in path_set:
                    row += "🟡"
                elif self.grid[i][j] == 1:
                    row += "⬛"
                else:
                    row += "⬜"
            print(row)
        print()


# ========================================
# トレーナー
# ========================================

class PathfinderTrainer:
    """QBNNパスファインダーの訓練"""
    
    def __init__(self, pathfinder: QBNNPathfinder, grid_size: int = 10):
        self.pathfinder = pathfinder
        self.grid_size = grid_size
        self.optimizer = torch.optim.Adam(pathfinder.parameters(), lr=0.005)
    
    def generate_episode(self, maze: Maze) -> List[Tuple]:
        """1エピソードの経験を生成"""
        pos = maze.start
        trajectory = []
        visited = {pos}
        max_steps = maze.size * maze.size
        
        for _ in range(max_steps):
            state = maze.get_state(pos)
            valid_actions = maze.get_valid_actions(pos)
            
            if not valid_actions:
                break
            
            # ε-greedy
            if random.random() < 0.3:
                action = random.choice(valid_actions)
            else:
                action = self.pathfinder.get_action(state, valid_actions)
            
            # 移動
            dr, dc = Maze.DIRECTIONS[action]
            new_pos = (pos[0] + dr, pos[1] + dc)
            
            # 報酬計算
            if new_pos == maze.goal:
                reward = 10.0
            elif new_pos in visited:
                reward = -1.0  # 再訪問ペナルティ
            else:
                # ゴールに近づいたら報酬
                old_dist = abs(maze.goal[0] - pos[0]) + abs(maze.goal[1] - pos[1])
                new_dist = abs(maze.goal[0] - new_pos[0]) + abs(maze.goal[1] - new_pos[1])
                reward = (old_dist - new_dist) * 0.5
            
            trajectory.append((state, action, reward, valid_actions))
            
            visited.add(new_pos)
            pos = new_pos
            
            if pos == maze.goal:
                break
        
        return trajectory
    
    def train(self, epochs: int = 100, episodes_per_epoch: int = 50):
        """訓練"""
        print("\n🗺️ QBNN パスファインダー訓練")
        print("=" * 50)
        
        best_success_rate = 0
        
        for epoch in range(epochs):
            maze = Maze(self.grid_size)
            maze.generate(wall_density=0.2)
            
            all_trajectories = []
            successes = 0
            
            for _ in range(episodes_per_epoch):
                trajectory = self.generate_episode(maze)
                all_trajectories.append(trajectory)
                
                # 成功判定（最後にゴールに到達）
                if trajectory and trajectory[-1][2] == 10.0:
                    successes += 1
            
            # 学習
            total_loss = 0
            for trajectory in all_trajectories:
                if not trajectory:
                    continue
                
                # Return計算（割引累積報酬）
                returns = []
                G = 0
                gamma = 0.99
                for _, _, reward, _ in reversed(trajectory):
                    G = reward + gamma * G
                    returns.insert(0, G)
                
                returns = torch.tensor(returns, dtype=torch.float32)
                if len(returns) > 1:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
                # Policy Gradient
                for (state, action, _, valid_actions), ret in zip(trajectory, returns):
                    self.optimizer.zero_grad()
                    
                    logits = self.pathfinder(state.unsqueeze(0))[0]
                    
                    # マスク
                    mask = torch.full((4,), float('-inf'))
                    for a in valid_actions:
                        mask[a] = 0
                    logits = logits + mask
                    
                    log_probs = F.log_softmax(logits, dim=0)
                    loss = -log_probs[action] * ret
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
            
            success_rate = successes / episodes_per_epoch * 100
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg_loss = total_loss / max(sum(len(t) for t in all_trajectories), 1)
                print(f"   Epoch {epoch+1:3d}/{epochs}: Loss={avg_loss:.4f}, 成功率={success_rate:.1f}%")
        
        print(f"\n✅ 訓練完了！ 最高成功率: {best_success_rate:.1f}%")


# ========================================
# デモ
# ========================================

def demo_pathfinding():
    """パスファインディングのデモ"""
    print("=" * 60)
    print("🗺️ QBNN パスファインダー デモ")
    print("=" * 60)
    
    grid_size = 8
    
    # モデル作成
    pathfinder = QBNNPathfinder(grid_size=grid_size, hidden_dim=64)
    
    # 訓練
    trainer = PathfinderTrainer(pathfinder, grid_size=grid_size)
    trainer.train(epochs=100, episodes_per_epoch=30)
    
    # テスト
    print("\n" + "=" * 50)
    print("🧪 テスト実行")
    print("=" * 50)
    
    for test_num in range(3):
        print(f"\n--- テスト {test_num + 1} ---")
        
        maze = Maze(grid_size)
        maze.generate(wall_density=0.2)
        
        # 最短経路（BFS）
        shortest = maze.get_shortest_path()
        print(f"📏 最短経路長: {len(shortest) - 1} ステップ")
        
        # QBNN経路
        pos = maze.start
        qbnn_path = [pos]
        visited = {pos}
        max_steps = grid_size * grid_size
        
        start_time = time.time()
        
        for step in range(max_steps):
            if pos == maze.goal:
                break
            
            state = maze.get_state(pos)
            valid_actions = maze.get_valid_actions(pos)
            
            if not valid_actions:
                print("   ❌ 行き止まり！")
                break
            
            action = pathfinder.get_action(state, valid_actions, temperature=0.2)
            dr, dc = Maze.DIRECTIONS[action]
            pos = (pos[0] + dr, pos[1] + dc)
            qbnn_path.append(pos)
            
            if pos in visited:
                pass  # 再訪問
            visited.add(pos)
        
        elapsed = time.time() - start_time
        
        if pos == maze.goal:
            print(f"✅ QBNN経路長: {len(qbnn_path) - 1} ステップ")
            efficiency = (len(shortest) - 1) / (len(qbnn_path) - 1) * 100 if len(qbnn_path) > 1 else 0
            print(f"📊 効率: {efficiency:.1f}%")
        else:
            print(f"❌ ゴール未到達 ({len(qbnn_path) - 1} ステップで中断)")
        
        print(f"⏱️ 探索時間: {elapsed*1000:.2f}ms")
        
        # 迷路表示
        print("\n最短経路:")
        maze.display(path=shortest)
        
        print("QBNN経路:")
        maze.display(path=qbnn_path)


def interactive_demo():
    """インタラクティブデモ"""
    print("=" * 60)
    print("🗺️ QBNN パスファインダー - インタラクティブモード")
    print("=" * 60)
    
    grid_size = 8
    
    pathfinder = QBNNPathfinder(grid_size=grid_size, hidden_dim=64)
    trainer = PathfinderTrainer(pathfinder, grid_size=grid_size)
    trainer.train(epochs=80, episodes_per_epoch=40)
    
    while True:
        print("\n新しい迷路を生成します...")
        maze = Maze(grid_size)
        maze.generate(wall_density=0.25)
        
        print("\n🟢 = スタート, 🔴 = ゴール, ⬛ = 壁")
        maze.display()
        
        input("Enterでパスファインディング開始...")
        
        pos = maze.start
        path = [pos]
        step = 0
        
        while pos != maze.goal and step < grid_size * grid_size:
            state = maze.get_state(pos)
            valid_actions = maze.get_valid_actions(pos)
            
            if not valid_actions:
                print("行き止まり！")
                break
            
            action = pathfinder.get_action(state, valid_actions, temperature=0.2)
            dr, dc = Maze.DIRECTIONS[action]
            pos = (pos[0] + dr, pos[1] + dc)
            path.append(pos)
            step += 1
            
            print(f"\rステップ {step}: {Maze.DIR_NAMES[action]} → ({pos[0]},{pos[1]})", end="")
            time.sleep(0.1)
        
        print()
        
        if pos == maze.goal:
            print(f"\n🎉 ゴール到達！ {len(path)-1}ステップ")
        else:
            print(f"\n❌ 失敗")
        
        # 最短経路と比較
        shortest = maze.get_shortest_path()
        print(f"📏 最短: {len(shortest)-1}ステップ")
        
        maze.display(path=path)
        
        cont = input("\n続けますか? (y/n): ").strip().lower()
        if cont != 'y':
            break


if __name__ == '__main__':
    print("=" * 60)
    print("🗺️ QBNN パスファインダー")
    print("=" * 60)
    print("\n選択:")
    print("  1. デモ（自動テスト）")
    print("  2. インタラクティブ")
    
    choice = input("\n選択 (1/2): ").strip()
    
    if choice == '2':
        interactive_demo()
    else:
        demo_pathfinding()

