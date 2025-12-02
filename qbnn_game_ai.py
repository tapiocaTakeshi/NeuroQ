#!/usr/bin/env python3
"""
QBNN ゲームAI - 三目並べ (Tic-Tac-Toe)
========================================
QBNNを使った敵CPU（AI）との対戦ゲーム
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import List, Tuple, Optional


# ========================================
# QBNN Layer（量子もつれ層）
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
        
        # もつれ補正
        delta = torch.einsum('...i,ij,...j->...j', s_prev, self.J, s_raw)
        
        # 動的λ
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
# QBNN ゲームAI
# ========================================

class QBNNGameAI(nn.Module):
    """三目並べ用のQBNN AI"""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # 入力: 9マス × 3状態(空/○/×) = 27次元
        # または 9マス (0=空, 1=自分, -1=相手) = 9次元
        self.input_dim = 9
        self.hidden_dim = hidden_dim
        self.output_dim = 9  # 各マスへの打つ確率
        
        # QBNNレイヤー
        self.qbnn1 = QBNNLayer(self.input_dim, hidden_dim)
        self.qbnn2 = QBNNLayer(hidden_dim, hidden_dim)
        self.qbnn3 = QBNNLayer(hidden_dim, self.output_dim)
        
        # 出力層
        self.output = nn.Linear(self.output_dim, self.output_dim)
    
    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        """
        board_state: (batch, 9) - 盤面状態
        return: (batch, 9) - 各マスへの評価値
        """
        x = self.qbnn1(board_state)
        x = self.qbnn2(x)
        x = self.qbnn3(x)
        return self.output(x)
    
    def get_move(self, board: List[int], temperature: float = 0.5) -> int:
        """
        盤面から次の手を選択
        board: [0]*9 (0=空, 1=AI, -1=プレイヤー)
        """
        self.eval()
        with torch.no_grad():
            state = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(state)[0]
            
            # 打てないマスをマスク
            mask = torch.tensor([float('-inf') if b != 0 else 0 for b in board])
            logits = logits + mask
            
            # 温度でソフトマックス
            probs = F.softmax(logits / temperature, dim=0)
            
            # サンプリング（量子的な不確定性を模倣）
            if random.random() < 0.3:  # 30%の確率で探索
                valid_moves = [i for i, b in enumerate(board) if b == 0]
                return random.choice(valid_moves)
            else:
                return torch.argmax(probs).item()


# ========================================
# 三目並べゲーム
# ========================================

class TicTacToe:
    """三目並べゲームロジック"""
    
    def __init__(self):
        self.board = [0] * 9  # 0=空, 1=○, -1=×
        self.current_player = 1  # 1=○(プレイヤー), -1=×(CPU)
        
    def reset(self):
        self.board = [0] * 9
        self.current_player = 1
        
    def display(self):
        """盤面表示"""
        symbols = {0: '·', 1: '○', -1: '×'}
        print("\n   1   2   3")
        for i in range(3):
            row = [symbols[self.board[i*3 + j]] for j in range(3)]
            print(f" {i+1} {row[0]} │ {row[1]} │ {row[2]}")
            if i < 2:
                print("   ──┼───┼──")
        print()
    
    def make_move(self, position: int, player: int) -> bool:
        """手を打つ (position: 0-8)"""
        if self.board[position] == 0:
            self.board[position] = player
            return True
        return False
    
    def check_winner(self) -> Optional[int]:
        """勝者をチェック (1=○勝利, -1=×勝利, 0=引き分け, None=続行)"""
        # 勝利ライン
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 横
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 縦
            [0, 4, 8], [2, 4, 6]              # 斜め
        ]
        
        for line in lines:
            s = sum(self.board[i] for i in line)
            if s == 3:
                return 1  # ○勝利
            if s == -3:
                return -1  # ×勝利
        
        # 引き分けチェック
        if 0 not in self.board:
            return 0
        
        return None  # 続行
    
    def get_valid_moves(self) -> List[int]:
        """打てるマスのリスト"""
        return [i for i, b in enumerate(self.board) if b == 0]


# ========================================
# AIトレーナー
# ========================================

class AITrainer:
    """自己対戦でAIを訓練"""
    
    def __init__(self, ai: QBNNGameAI):
        self.ai = ai
        self.optimizer = torch.optim.Adam(ai.parameters(), lr=0.01)
    
    def generate_training_data(self, num_games: int = 1000) -> List[Tuple]:
        """ランダムプレイでデータ生成"""
        data = []
        
        for _ in range(num_games):
            game = TicTacToe()
            history = []
            
            while True:
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break
                
                move = random.choice(valid_moves)
                history.append((game.board.copy(), move, game.current_player))
                game.make_move(move, game.current_player)
                
                winner = game.check_winner()
                if winner is not None:
                    # 結果に基づいて報酬
                    for board, m, player in history:
                        if winner == 0:
                            reward = 0.3  # 引き分け
                        elif winner == player:
                            reward = 1.0  # 勝利
                        else:
                            reward = -0.5  # 敗北
                        data.append((board, m, reward))
                    break
                
                game.current_player *= -1
        
        return data
    
    def train(self, epochs: int = 50, games_per_epoch: int = 200):
        """訓練"""
        print("\n🎮 QBNN ゲームAI 訓練開始")
        print("=" * 50)
        
        for epoch in range(epochs):
            # データ生成
            data = self.generate_training_data(games_per_epoch)
            
            # シャッフル
            random.shuffle(data)
            
            total_loss = 0
            batch_size = 32
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                
                boards = torch.tensor([d[0] for d in batch], dtype=torch.float32)
                moves = torch.tensor([d[1] for d in batch], dtype=torch.long)
                rewards = torch.tensor([d[2] for d in batch], dtype=torch.float32)
                
                self.optimizer.zero_grad()
                
                logits = self.ai(boards)
                log_probs = F.log_softmax(logits, dim=1)
                selected_log_probs = log_probs.gather(1, moves.unsqueeze(1)).squeeze()
                
                # Policy Gradient Loss
                loss = -(selected_log_probs * rewards).mean()
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / (len(data) // batch_size + 1)
                print(f"   Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")
        
        print("\n✅ 訓練完了！")


# ========================================
# ミニマックスAI（比較用）
# ========================================

class MinimaxAI:
    """完璧なAI（比較用）"""
    
    def get_move(self, board: List[int], player: int = -1) -> int:
        """ミニマックスで最適手を返す"""
        best_score = float('-inf')
        best_move = None
        
        for i in range(9):
            if board[i] == 0:
                board[i] = player
                score = self._minimax(board, 0, False, player)
                board[i] = 0
                
                if score > best_score:
                    best_score = score
                    best_move = i
        
        return best_move if best_move is not None else random.choice([i for i in range(9) if board[i] == 0])
    
    def _minimax(self, board: List[int], depth: int, is_maximizing: bool, player: int) -> int:
        winner = self._check_winner(board)
        if winner == player:
            return 10 - depth
        elif winner == -player:
            return depth - 10
        elif 0 not in board:
            return 0
        
        if is_maximizing:
            best = float('-inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = player
                    best = max(best, self._minimax(board, depth + 1, False, player))
                    board[i] = 0
            return best
        else:
            best = float('inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = -player
                    best = min(best, self._minimax(board, depth + 1, True, player))
                    board[i] = 0
            return best
    
    def _check_winner(self, board: List[int]) -> Optional[int]:
        lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for line in lines:
            s = sum(board[i] for i in line)
            if s == 3: return 1
            if s == -3: return -1
        return None


# ========================================
# ゲームプレイ
# ========================================

def play_game():
    """人間 vs QBNN AI"""
    print("=" * 60)
    print("🎮 三目並べ - QBNN AI対戦")
    print("=" * 60)
    print("\nQBNN（量子ビットニューラルネットワーク）AIと対戦します！")
    print("あなたは ○ 、CPUは × です。")
    print("マスは 行,列 (例: 1,1 = 左上) で指定します。")
    
    # AI作成と訓練
    ai = QBNNGameAI(hidden_dim=64)
    trainer = AITrainer(ai)
    trainer.train(epochs=30, games_per_epoch=300)
    
    # 統計
    stats = {'player': 0, 'cpu': 0, 'draw': 0}
    
    while True:
        game = TicTacToe()
        print("\n" + "=" * 40)
        print("🆕 新しいゲーム開始！")
        print("=" * 40)
        game.display()
        
        while True:
            if game.current_player == 1:
                # プレイヤーの手番
                print("👤 あなたの番です (○)")
                while True:
                    try:
                        inp = input("   マスを入力 (行,列 例: 2,2): ").strip()
                        if inp.lower() in ['q', 'quit', 'exit']:
                            print("\n📊 最終結果:")
                            print(f"   プレイヤー勝利: {stats['player']}")
                            print(f"   CPU勝利: {stats['cpu']}")
                            print(f"   引き分け: {stats['draw']}")
                            return
                        
                        row, col = map(int, inp.split(','))
                        pos = (row - 1) * 3 + (col - 1)
                        
                        if 0 <= pos < 9 and game.board[pos] == 0:
                            game.make_move(pos, 1)
                            break
                        else:
                            print("   ❌ そのマスには打てません！")
                    except:
                        print("   ❌ 入力形式: 行,列 (例: 1,2)")
            else:
                # CPUの手番
                print("🤖 QBNN CPU の番です (×)...")
                
                # AIから盤面を見た状態（AIは-1として打つ）
                ai_board = [-b for b in game.board]  # 視点変換
                pos = ai.get_move(ai_board, temperature=0.3)
                
                game.make_move(pos, -1)
                r, c = pos // 3 + 1, pos % 3 + 1
                print(f"   → CPU: ({r},{c})")
            
            game.display()
            
            # 勝敗チェック
            winner = game.check_winner()
            if winner is not None:
                if winner == 1:
                    print("🎉 あなたの勝ち！")
                    stats['player'] += 1
                elif winner == -1:
                    print("💻 CPUの勝ち！")
                    stats['cpu'] += 1
                else:
                    print("🤝 引き分け！")
                    stats['draw'] += 1
                
                print(f"\n📊 戦績: プレイヤー {stats['player']} - {stats['cpu']} CPU (引分: {stats['draw']})")
                break
            
            game.current_player *= -1
        
        # 続けるか
        cont = input("\n続けますか? (y/n): ").strip().lower()
        if cont != 'y':
            print("\n📊 最終結果:")
            print(f"   プレイヤー勝利: {stats['player']}")
            print(f"   CPU勝利: {stats['cpu']}")
            print(f"   引き分け: {stats['draw']}")
            break


def benchmark_ai():
    """AIの強さをベンチマーク"""
    print("=" * 60)
    print("📊 QBNN AI ベンチマーク")
    print("=" * 60)
    
    # QBNN AI
    qbnn_ai = QBNNGameAI(hidden_dim=64)
    trainer = AITrainer(qbnn_ai)
    trainer.train(epochs=50, games_per_epoch=500)
    
    # ランダムAI
    def random_ai(board):
        valid = [i for i, b in enumerate(board) if b == 0]
        return random.choice(valid) if valid else 0
    
    # ミニマックスAI
    minimax_ai = MinimaxAI()
    
    def play_match(ai1_func, ai2_func, num_games=100):
        """対戦"""
        wins = [0, 0, 0]  # ai1勝, ai2勝, 引分
        
        for _ in range(num_games):
            game = TicTacToe()
            current_ai = 0  # 0=ai1, 1=ai2
            
            while True:
                if current_ai == 0:
                    move = ai1_func(game.board.copy())
                else:
                    move = ai2_func(game.board.copy())
                
                player = 1 if current_ai == 0 else -1
                game.make_move(move, player)
                
                winner = game.check_winner()
                if winner is not None:
                    if winner == 1:
                        wins[0] += 1
                    elif winner == -1:
                        wins[1] += 1
                    else:
                        wins[2] += 1
                    break
                
                current_ai = 1 - current_ai
        
        return wins
    
    print("\n🎮 対戦結果:")
    
    # QBNN vs ランダム
    def qbnn_move(board):
        ai_board = [-b for b in board]
        return qbnn_ai.get_move(ai_board, temperature=0.2)
    
    results = play_match(qbnn_move, random_ai, 100)
    print(f"\n   QBNN vs ランダム: {results[0]}勝-{results[1]}敗-{results[2]}分")
    qbnn_vs_random = results[0] / (results[0] + results[1] + 0.001) * 100
    print(f"   → QBNN勝率: {qbnn_vs_random:.1f}%")
    
    # QBNN vs ミニマックス
    def minimax_move(board):
        return minimax_ai.get_move(board, player=-1)
    
    results = play_match(qbnn_move, minimax_move, 100)
    print(f"\n   QBNN vs ミニマックス(完璧AI): {results[0]}勝-{results[1]}敗-{results[2]}分")
    
    # ランダム vs ミニマックス（参考）
    results = play_match(random_ai, minimax_move, 100)
    print(f"\n   ランダム vs ミニマックス(参考): {results[0]}勝-{results[1]}敗-{results[2]}分")
    
    print("\n✅ ベンチマーク完了！")


if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("🎮 QBNN ゲームAI - 三目並べ")
    print("=" * 60)
    print("\n選択してください:")
    print("  1. 人間 vs CPU で対戦")
    print("  2. AIベンチマーク（強さ測定）")
    print("  3. 両方")
    
    choice = input("\n選択 (1/2/3): ").strip()
    
    if choice == '1':
        play_game()
    elif choice == '2':
        benchmark_ai()
    else:
        benchmark_ai()
        print("\n" + "=" * 60)
        play_game()

