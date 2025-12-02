"""
量子迷路ソルバー (Quantum Maze Solver)

擬似量子ビットを使用して迷路をクリアするプログラム
AIは一切使用せず、量子の確率的性質のみで探索

アルゴリズム:
1. 量子ウォーク: 重ね合わせ状態で複数経路を同時探索
2. 測定による経路選択: 分岐点で量子ビットを測定
3. 干渉効果: 行き止まりの確率を減少、正解経路を増幅
"""

import numpy as np
import random
import time
import os
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass
from pseudo_qubit import PseudoQubit


@dataclass
class Position:
    """位置を表すクラス"""
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"({self.x}, {self.y})"


class QuantumWalker:
    """
    量子ウォーカー
    重ね合わせ状態で迷路を探索
    """
    def __init__(self, pos: Position, amplitude: complex = 1.0):
        self.pos = pos
        self.amplitude = amplitude  # 確率振幅
        self.path: List[Position] = [pos]
        self.dead_end = False
    
    @property
    def probability(self) -> float:
        """存在確率"""
        return abs(self.amplitude) ** 2
    
    def split(self, directions: int) -> List['QuantumWalker']:
        """分岐時に重ね合わせ状態に分裂"""
        new_amplitude = self.amplitude / np.sqrt(directions)
        return [
            QuantumWalker(self.pos, new_amplitude) 
            for _ in range(directions)
        ]


class Maze:
    """迷路クラス"""
    
    # 迷路の記号
    WALL = '█'
    PATH = ' '
    START = 'S'
    GOAL = 'G'
    WALKER = '◉'
    VISITED = '·'
    SOLUTION = '★'
    
    def __init__(self, width: int = 21, height: int = 21):
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.grid = [[self.WALL for _ in range(self.width)] for _ in range(self.height)]
        self.start = Position(1, 1)
        self.goal = Position(self.width - 2, self.height - 2)
        self._generate_maze()
    
    def _generate_maze(self):
        """深さ優先探索で迷路を生成"""
        # スタート位置を通路にする
        self.grid[self.start.y][self.start.x] = self.PATH
        
        stack = [(self.start.x, self.start.y)]
        
        while stack:
            x, y = stack[-1]
            
            # 移動可能な方向を探す（2マス先が壁かつ範囲内）
            directions = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.width - 1 and 0 < ny < self.height - 1:
                    if self.grid[ny][nx] == self.WALL:
                        directions.append((dx, dy))
            
            if directions:
                # ランダムに方向を選択
                dx, dy = random.choice(directions)
                # 壁を壊して通路を作る
                self.grid[y + dy // 2][x + dx // 2] = self.PATH
                self.grid[y + dy][x + dx] = self.PATH
                stack.append((x + dx, y + dy))
            else:
                stack.pop()
        
        # スタートとゴールを設定
        self.grid[self.start.y][self.start.x] = self.START
        self.grid[self.goal.y][self.goal.x] = self.GOAL
    
    def is_valid_move(self, pos: Position) -> bool:
        """移動可能かチェック"""
        if 0 <= pos.x < self.width and 0 <= pos.y < self.height:
            return self.grid[pos.y][pos.x] != self.WALL
        return False
    
    def get_neighbors(self, pos: Position) -> List[Position]:
        """隣接する移動可能なマスを取得"""
        neighbors = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            new_pos = Position(pos.x + dx, pos.y + dy)
            if self.is_valid_move(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def display(self, walkers: List[Position] = None, visited: Set[Position] = None, 
                solution: List[Position] = None):
        """迷路を表示"""
        display_grid = [row[:] for row in self.grid]
        
        # 訪問済みマスを表示
        if visited:
            for pos in visited:
                if display_grid[pos.y][pos.x] == self.PATH:
                    display_grid[pos.y][pos.x] = self.VISITED
        
        # 解答経路を表示
        if solution:
            for pos in solution:
                if display_grid[pos.y][pos.x] in [self.PATH, self.VISITED]:
                    display_grid[pos.y][pos.x] = self.SOLUTION
        
        # ウォーカーを表示
        if walkers:
            for pos in walkers:
                if display_grid[pos.y][pos.x] not in [self.START, self.GOAL]:
                    display_grid[pos.y][pos.x] = self.WALKER
        
        # 出力
        for row in display_grid:
            print(''.join(row))


class QuantumMazeSolver:
    """
    量子迷路ソルバー
    
    擬似量子ビットを使用した探索アルゴリズム:
    1. 各分岐点で量子ビットによる確率的方向選択
    2. 訪問済み経路の確率を減少（干渉効果）
    3. ゴールに近い方向の確率を増幅
    """
    
    def __init__(self, maze: Maze):
        self.maze = maze
        self.visited: Set[Position] = set()
        self.solution_path: List[Position] = []
        self.total_measurements = 0
        self.quantum_states: List[Tuple[Position, float]] = []  # (位置, 確率)
    
    def calculate_correlation(self, current: Position, candidates: List[Position]) -> List[float]:
        """
        各候補方向に対する相関係数を計算
        
        - ゴールに近づく方向: 正の相関 (r > 0)
        - ゴールから遠ざかる方向: 負の相関 (r < 0)
        - 訪問済み: 強い負の相関
        """
        correlations = []
        
        current_dist = abs(current.x - self.maze.goal.x) + abs(current.y - self.maze.goal.y)
        
        for pos in candidates:
            # ゴールまでの距離
            new_dist = abs(pos.x - self.maze.goal.x) + abs(pos.y - self.maze.goal.y)
            
            # 基本相関: ゴールに近づくほど正
            base_r = (current_dist - new_dist) / max(current_dist, 1)
            
            # 訪問済みペナルティ
            if pos in self.visited:
                base_r -= 0.5  # 訪問済みは確率を下げる
            
            # 範囲を [-1, 1] に制限
            r = max(-1.0, min(1.0, base_r))
            correlations.append(r)
        
        return correlations
    
    def quantum_choice(self, current: Position, candidates: List[Position]) -> Position:
        """
        量子ビットを使って次の移動先を選択
        """
        if len(candidates) == 1:
            return candidates[0]
        
        # 各候補の相関係数を計算
        correlations = self.calculate_correlation(current, candidates)
        
        # 量子ビットで確率を計算
        probabilities = []
        for r in correlations:
            qubit = PseudoQubit(correlation=r)
            # |0⟩ の確率を使用（正の相関 → 高確率）
            prob = qubit.probabilities[0]
            probabilities.append(prob)
        
        # 確率を正規化
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / len(candidates)] * len(candidates)
        
        # 量子測定（確率的選択）
        self.total_measurements += 1
        
        # 累積確率で選択
        rand = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand <= cumulative:
                return candidates[i]
        
        return candidates[-1]
    
    def solve(self, animate: bool = True, delay: float = 0.1) -> bool:
        """
        量子探索で迷路を解く
        
        Args:
            animate: アニメーション表示
            delay: アニメーション間隔
        
        Returns:
            解けたかどうか
        """
        current = self.maze.start
        path = [current]
        self.visited.add(current)
        
        max_steps = self.maze.width * self.maze.height * 2
        steps = 0
        
        while current != self.maze.goal and steps < max_steps:
            steps += 1
            
            # 隣接マスを取得
            neighbors = self.maze.get_neighbors(current)
            
            # 未訪問の隣接マスを優先
            unvisited = [n for n in neighbors if n not in self.visited]
            
            if unvisited:
                # 量子選択で次のマスを決定
                next_pos = self.quantum_choice(current, unvisited)
            elif neighbors:
                # 全て訪問済みの場合、バックトラック
                # 量子的な確率で戻る方向を選択
                next_pos = self.quantum_choice(current, neighbors)
            else:
                # 行き止まり
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                    continue
                else:
                    break
            
            current = next_pos
            self.visited.add(current)
            path.append(current)
            
            # アニメーション表示
            if animate:
                self.clear_screen()
                print(f"\n  QUANTUM MAZE SOLVER")
                print(f"  Steps: {steps}  |  Measurements: {self.total_measurements}")
                print(f"  Current: {current}  |  Goal: {self.maze.goal}")
                print()
                self.maze.display(walkers=[current], visited=self.visited)
                time.sleep(delay)
        
        # 結果
        if current == self.maze.goal:
            self.solution_path = path
            return True
        return False
    
    def quantum_parallel_solve(self, animate: bool = True, delay: float = 0.15) -> bool:
        """
        量子並列探索（重ね合わせ状態での探索をシミュレート）
        
        複数の「量子ウォーカー」が同時に探索
        """
        # 初期ウォーカー
        walkers = [QuantumWalker(self.maze.start, 1.0)]
        all_positions = {self.maze.start}
        self.visited = {self.maze.start}
        
        max_steps = self.maze.width * self.maze.height
        steps = 0
        found_path = None
        
        while steps < max_steps and not found_path:
            steps += 1
            new_walkers = []
            
            for walker in walkers:
                if walker.dead_end:
                    continue
                
                # 隣接マスを取得
                neighbors = self.maze.get_neighbors(walker.pos)
                unvisited = [n for n in neighbors if n not in self.visited]
                
                if not unvisited:
                    walker.dead_end = True
                    # 行き止まり: 確率振幅を減衰
                    walker.amplitude *= 0.5
                    continue
                
                # ゴールチェック
                for n in unvisited:
                    if n == self.maze.goal:
                        found_path = walker.path + [n]
                        break
                
                if found_path:
                    break
                
                # 量子分岐: 全ての方向に重ね合わせ
                if len(unvisited) > 1:
                    # 各方向への相関係数
                    correlations = self.calculate_correlation(walker.pos, unvisited)
                    
                    for i, next_pos in enumerate(unvisited):
                        # 量子ビットで振幅を調整
                        qubit = PseudoQubit(correlation=correlations[i])
                        amplitude_factor = np.sqrt(qubit.probabilities[0])
                        
                        new_walker = QuantumWalker(
                            next_pos, 
                            walker.amplitude * amplitude_factor / np.sqrt(len(unvisited))
                        )
                        new_walker.path = walker.path + [next_pos]
                        new_walkers.append(new_walker)
                        self.visited.add(next_pos)
                        all_positions.add(next_pos)
                else:
                    # 一方向のみ
                    next_pos = unvisited[0]
                    walker.pos = next_pos
                    walker.path.append(next_pos)
                    new_walkers.append(walker)
                    self.visited.add(next_pos)
                    all_positions.add(next_pos)
            
            # 確率が低いウォーカーを刈り込み
            walkers = [w for w in new_walkers if w.probability > 0.001]
            
            # 確率で正規化
            total_prob = sum(w.probability for w in walkers)
            if total_prob > 0:
                for w in walkers:
                    w.amplitude *= np.sqrt(1.0 / total_prob)
            
            # アニメーション
            if animate and walkers:
                self.clear_screen()
                print(f"\n  QUANTUM PARALLEL MAZE SOLVER")
                print(f"  Steps: {steps}  |  Active Walkers: {len(walkers)}")
                print(f"  Total Explored: {len(all_positions)}")
                print()
                
                # 確率が高い上位のウォーカーを表示
                top_walkers = sorted(walkers, key=lambda w: w.probability, reverse=True)[:5]
                walker_positions = [w.pos for w in walkers]
                
                self.maze.display(walkers=walker_positions, visited=self.visited)
                
                print(f"\n  Top Walkers (by probability):")
                for i, w in enumerate(top_walkers[:3]):
                    print(f"    {i+1}. {w.pos} - P = {w.probability:.4f}")
                
                time.sleep(delay)
        
        if found_path:
            self.solution_path = found_path
            return True
        
        return False
    
    @staticmethod
    def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')


def main():
    print("\n" + "=" * 60)
    print("  QUANTUM MAZE SOLVER")
    print("  Using Pseudo-Qubit for Probabilistic Pathfinding")
    print("=" * 60)
    
    # 迷路サイズの選択
    print("\n  Maze Size:")
    print("    1. Small  (11x11)")
    print("    2. Medium (21x21)")
    print("    3. Large  (31x31)")
    
    while True:
        try:
            choice = input("\n  Select size [1-3] (default=2): ").strip()
            if choice == "" or choice == "2":
                width, height = 21, 21
                break
            elif choice == "1":
                width, height = 11, 11
                break
            elif choice == "3":
                width, height = 31, 31
                break
        except:
            pass
    
    # アルゴリズムの選択
    print("\n  Algorithm:")
    print("    1. Quantum Walk (single walker)")
    print("    2. Quantum Parallel (superposition)")
    
    while True:
        try:
            algo = input("\n  Select algorithm [1-2] (default=1): ").strip()
            if algo == "" or algo == "1":
                parallel = False
                break
            elif algo == "2":
                parallel = True
                break
        except:
            pass
    
    # 迷路生成
    print("\n  Generating maze...")
    maze = Maze(width, height)
    
    print("\n  Initial Maze:")
    maze.display()
    
    input("\n  Press Enter to start solving...")
    
    # ソルバー実行
    solver = QuantumMazeSolver(maze)
    start_time = time.time()
    
    if parallel:
        success = solver.quantum_parallel_solve(animate=True, delay=0.1)
    else:
        success = solver.solve(animate=True, delay=0.05)
    
    elapsed = time.time() - start_time
    
    # 結果表示
    QuantumMazeSolver.clear_screen()
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    
    if success:
        print(f"\n  ✓ MAZE SOLVED!")
        print(f"  Path length: {len(solver.solution_path)}")
        print(f"  Total measurements: {solver.total_measurements}")
        print(f"  Explored cells: {len(solver.visited)}")
        print(f"  Time: {elapsed:.2f}s")
        
        print("\n  Solution:")
        maze.display(solution=solver.solution_path, visited=solver.visited)
    else:
        print(f"\n  ✗ Could not find solution")
        print(f"  Explored cells: {len(solver.visited)}")
        maze.display(visited=solver.visited)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

