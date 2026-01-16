#!/usr/bin/env python3
"""
QBNNのエポック数とテスト精度の関係をグラフ化するスクリプト

横軸：エポック数
縦軸：テスト精度（Accuracy / Loss）

各エポックで学習後、テストデータセットで評価を行い精度を測定

使い方:
    python qbnn_epoch_accuracy_plot.py
    python qbnn_epoch_accuracy_plot.py --max_epochs 100
    python qbnn_epoch_accuracy_plot.py --output qbnn_test_accuracy.png
"""

import os
import sys
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']

# 現在のディレクトリをパスに追加
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# neuroq-runpodディレクトリもパスに追加
runpod_dir = os.path.join(current_dir, 'neuroq-runpod')
if runpod_dir not in sys.path:
    sys.path.insert(0, runpod_dir)

print("=" * 70)
print("📊 QBNN エポック数 vs テスト精度 グラフ生成")
print("=" * 70)

# インポート
try:
    from neuroq_runpod.neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    print("✅ NeuroQuantum (neuroquantum_layered.py) をインポートしました")
except ImportError:
    try:
        sys.path.insert(0, os.path.join(current_dir, 'neuroq-runpod'))
        from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
        print("✅ NeuroQuantum (neuroquantum_layered.py) をインポートしました")
    except ImportError as e:
        print(f"❌ NeuroQuantumのインポートに失敗: {e}")
        sys.exit(1)

try:
    from neuroq_runpod.tiktoken_tokenizer import TikTokenTokenizer
except ImportError:
    try:
        from tiktoken_tokenizer import TikTokenTokenizer
    except ImportError as e:
        print(f"❌ TikTokenTokenizerのインポートに失敗: {e}")
        sys.exit(1)

print("✅ モジュールをインポートしました")

# モデル設定
MODEL_CONFIG = {
    'vocab_size': 200019,  # o200k_baseの語彙サイズ
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 3,
}

# 学習データ（英語対話）
TRAINING_DATA = [
    "<USER> Hello! <ASSISTANT> Hello! How can I help you today?",
    "<USER> What is AI? <ASSISTANT> AI, or Artificial Intelligence, is technology that enables machines to simulate human intelligence.",
    "<USER> Tell me about machine learning. <ASSISTANT> Machine learning is a subset of AI where computers learn from data without being explicitly programmed.",
    "<USER> What is deep learning? <ASSISTANT> Deep learning is a type of machine learning using neural networks with many layers.",
    "<USER> What is quantum computing? <ASSISTANT> Quantum computing uses quantum mechanics principles to process information using qubits.",
    "<USER> What are qubits? <ASSISTANT> Qubits are quantum bits that can exist in a superposition of both 0 and 1 states.",
    "<USER> What is Python? <ASSISTANT> Python is a popular programming language known for its simplicity and readability.",
    "<USER> How do I learn programming? <ASSISTANT> Start with a beginner-friendly language like Python. Practice regularly and work on projects.",
    "<USER> Thank you! <ASSISTANT> You're welcome! Feel free to ask if you have more questions.",
    "<USER> Can you help me? <ASSISTANT> Of course! I'd be happy to help. What do you need assistance with?",
    "<USER> What is neural network? <ASSISTANT> A neural network is a computing system inspired by biological neural networks in the brain.",
    "<USER> Explain backpropagation. <ASSISTANT> Backpropagation is an algorithm for training neural networks by computing gradients.",
    "<USER> What is TensorFlow? <ASSISTANT> TensorFlow is an open-source machine learning framework developed by Google.",
    "<USER> What is PyTorch? <ASSISTANT> PyTorch is a deep learning framework known for its flexibility and dynamic computation graphs.",
    "<USER> Define entropy. <ASSISTANT> Entropy is a measure of uncertainty or randomness in information theory.",
    "<USER> What is GPU? <ASSISTANT> GPU stands for Graphics Processing Unit, widely used for parallel computing tasks.",
    "<USER> Explain transformer architecture. <ASSISTANT> Transformer is a neural network architecture that uses self-attention mechanisms for sequence processing.",
    "<USER> What is NLP? <ASSISTANT> NLP stands for Natural Language Processing, a field of AI focused on understanding human language.",
]

# テストデータ（学習データとは異なる新しい対話）
TEST_DATA = [
    "<USER> Good morning! <ASSISTANT> Good morning! How are you doing today?",
    "<USER> What is computer vision? <ASSISTANT> Computer vision is an AI field that enables computers to interpret and understand visual information.",
    "<USER> How does attention work? <ASSISTANT> Attention mechanism allows models to focus on relevant parts of the input when making predictions.",
    "<USER> What is reinforcement learning? <ASSISTANT> Reinforcement learning is a type of machine learning where agents learn by interacting with an environment.",
    "<USER> Explain overfitting. <ASSISTANT> Overfitting occurs when a model learns training data too well, failing to generalize to new data.",
    "<USER> What is a CNN? <ASSISTANT> CNN stands for Convolutional Neural Network, primarily used for image recognition tasks.",
]


def evaluate_model(model, sequences, device, vocab_size):
    """
    モデルをテストデータで評価
    
    Returns:
        test_loss: テストデータでの平均損失
        accuracy: 次トークン予測の正解率
        perplexity: パープレキシティ
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    batch_count = 0
    
    with torch.no_grad():
        for seq in sequences:
            seq_tensor = torch.tensor([seq], device=device)
            
            input_ids = seq_tensor[:, :-1]
            target_ids = seq_tensor[:, 1:]
            
            logits = model(input_ids)
            
            # 損失計算
            loss = criterion(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1)
            )
            total_loss += loss.item()
            batch_count += 1
            
            # 正解率計算（次トークン予測）
            predictions = logits.argmax(dim=-1)
            correct = (predictions == target_ids).sum().item()
            total_correct += correct
            total_tokens += target_ids.numel()
    
    avg_loss = total_loss / max(batch_count, 1)
    accuracy = total_correct / max(total_tokens, 1) * 100  # パーセントに変換
    perplexity = np.exp(min(avg_loss, 20))  # オーバーフロー防止
    
    return avg_loss, accuracy, perplexity


def train_and_evaluate(max_epochs, batch_size=4, seq_length=64, lr=0.001):
    """
    モデルを学習し、各エポックでテストデータで評価
    
    Returns:
        epochs_list: エポック番号のリスト
        train_losses: 各エポックでの学習損失リスト
        test_losses: 各エポックでのテスト損失リスト
        test_accuracies: 各エポックでのテスト正解率リスト
        test_perplexities: 各エポックでのテストパープレキシティリスト
    """
    print("\n📥 トークナイザー初期化...")
    tokenizer = TikTokenTokenizer(encoding_name="o200k_base")
    
    # 学習データをトークン化
    print("🔄 学習データトークン化中...")
    train_tokens = []
    for text in TRAINING_DATA:
        tokens = tokenizer.encode(text, add_special=False)
        train_tokens.extend(tokens)
        train_tokens.append(tokenizer.eos_id)
    
    # テストデータをトークン化
    print("🔄 テストデータトークン化中...")
    test_tokens = []
    for text in TEST_DATA:
        tokens = tokenizer.encode(text, add_special=False)
        test_tokens.extend(tokens)
        test_tokens.append(tokenizer.eos_id)
    
    print(f"   学習トークン数: {len(train_tokens):,}")
    print(f"   テストトークン数: {len(test_tokens):,}")
    
    # シーケンスを作成
    train_sequences = []
    for i in range(0, len(train_tokens) - seq_length, seq_length // 2):
        train_sequences.append(train_tokens[i:i + seq_length])
    
    test_sequences = []
    for i in range(0, len(test_tokens) - seq_length, seq_length // 2):
        test_sequences.append(test_tokens[i:i + seq_length])
    
    print(f"   学習シーケンス数: {len(train_sequences):,}")
    print(f"   テストシーケンス数: {len(test_sequences):,}")
    
    # デバイス選択
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) を使用")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 NVIDIA GPU (CUDA) を使用")
    else:
        device = torch.device("cpu")
        print("💻 CPU を使用")
    
    # モデル構築
    print("\n🧠 モデル構築中...")
    config = NeuroQuantumConfig()
    config.vocab_size = MODEL_CONFIG['vocab_size']
    config.embed_dim = MODEL_CONFIG['embed_dim']
    config.num_heads = MODEL_CONFIG['num_heads']
    config.num_layers = MODEL_CONFIG['num_layers']
    config.max_seq_len = 256
    config.dropout = 0.1
    
    model = NeuroQuantum(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   総パラメータ数: {total_params:,}")
    
    # 学習設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 結果を記録
    epochs_list = []
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_perplexities = []
    
    print(f"\n🚀 学習開始 (最大 {max_epochs} エポック)...")
    print(f"   各エポック終了時にテストデータで評価します\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Test Loss':>10} | {'Test Acc%':>10} | {'Test PPL':>10}")
    print("-" * 60)
    
    for epoch in range(1, max_epochs + 1):
        # ===== 学習フェーズ =====
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        # シーケンスをシャッフル
        random.shuffle(train_sequences)
        
        for i in range(0, len(train_sequences) - batch_size, batch_size):
            batch = train_sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch, device=device)
            
            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]
            
            optimizer.zero_grad()
            logits = model(input_ids)
            
            loss = criterion(
                logits.reshape(-1, MODEL_CONFIG['vocab_size']),
                target_ids.reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = total_train_loss / max(batch_count, 1)
        
        # ===== テストフェーズ（毎エポック） =====
        test_loss, test_accuracy, test_perplexity = evaluate_model(
            model, test_sequences, device, MODEL_CONFIG['vocab_size']
        )
        
        # 記録
        epochs_list.append(epoch)
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_perplexities.append(test_perplexity)
        
        # 進捗表示
        print(f"{epoch:>6} | {avg_train_loss:>10.4f} | {test_loss:>10.4f} | {test_accuracy:>9.2f}% | {test_perplexity:>10.2f}")
    
    print("-" * 60)
    print(f"\n✅ 学習完了！")
    print(f"   最終テスト精度: {test_accuracies[-1]:.2f}%")
    print(f"   最終テストLoss: {test_losses[-1]:.4f}")
    print(f"   最終テストPerplexity: {test_perplexities[-1]:.2f}")
    
    return epochs_list, train_losses, test_losses, test_accuracies, test_perplexities


def plot_results(epochs, train_losses, test_losses, test_accuracies, test_perplexities, output_path="qbnn_test_accuracy.png"):
    """
    結果をグラフ化（3パネル構成）
    """
    print(f"\n📊 グラフ生成中...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. 損失グラフ（学習 vs テスト）
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=3, label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', linewidth=2, marker='s', markersize=3, label='Test Loss')
    ax1.set_xlabel('エポック数', fontsize=12)
    ax1.set_ylabel('損失 (Loss)', fontsize=12)
    ax1.set_title('QBNN: 学習Loss vs テストLoss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. テスト精度グラフ
    ax2 = axes[1]
    ax2.plot(epochs, test_accuracies, 'g-', linewidth=2, marker='^', markersize=4, label='Test Accuracy')
    ax2.set_xlabel('エポック数', fontsize=12)
    ax2.set_ylabel('テスト精度 (%)', fontsize=12)
    ax2.set_title('QBNN: エポック数 vs テスト精度', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, max(test_accuracies) * 1.1)  # 余白を持たせる
    
    # 3. パープレキシティグラフ
    ax3 = axes[2]
    ax3.plot(epochs, test_perplexities, 'm-', linewidth=2, marker='d', markersize=3, label='Test Perplexity')
    ax3.set_xlabel('エポック数', fontsize=12)
    ax3.set_ylabel('テストパープレキシティ', fontsize=12)
    ax3.set_title('QBNN: エポック数 vs テストPerplexity', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # ログスケールオプション
    if max(test_perplexities) > 1000:
        ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ グラフを保存しました: {output_path}")
    
    # 統計情報
    print(f"\n📈 統計情報:")
    print(f"   [学習Loss]")
    print(f"     - 初期: {train_losses[0]:.4f} → 最終: {train_losses[-1]:.4f} (改善率: {(1 - train_losses[-1]/train_losses[0]) * 100:.1f}%)")
    print(f"   [テストLoss]")
    print(f"     - 初期: {test_losses[0]:.4f} → 最終: {test_losses[-1]:.4f} (改善率: {(1 - test_losses[-1]/test_losses[0]) * 100:.1f}%)")
    print(f"   [テスト精度]")
    print(f"     - 初期: {test_accuracies[0]:.2f}% → 最終: {test_accuracies[-1]:.2f}%")
    print(f"   [テストPerplexity]")
    print(f"     - 初期: {test_perplexities[0]:.2f} → 最終: {test_perplexities[-1]:.2f}")
    
    # グラフを表示
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='QBNNのエポック数とテスト精度の関係をグラフ化'
    )
    parser.add_argument(
        '--max_epochs', '-e',
        type=int,
        default=50,
        help='最大エポック数 (デフォルト: 50)'
    )
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=4,
        help='バッチサイズ (デフォルト: 4)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='学習率 (デフォルト: 0.001)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='qbnn_test_accuracy.png',
        help='出力ファイル名 (デフォルト: qbnn_test_accuracy.png)'
    )
    
    args = parser.parse_args()
    
    # 学習と評価
    epochs, train_losses, test_losses, test_accuracies, test_perplexities = train_and_evaluate(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # グラフ生成
    plot_results(epochs, train_losses, test_losses, test_accuracies, test_perplexities, output_path=args.output)
    
    print("\n" + "=" * 70)
    print("✅ 完了！")
    print("=" * 70)


if __name__ == "__main__":
    main()
