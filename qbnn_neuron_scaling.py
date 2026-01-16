#!/usr/bin/env python3
"""
QBNN ニューロン数 vs 性能 比較ベンチマーク

ニューロン数（embed_dim）を変化させながらLLMベンチマークを実行し、
モデルサイズと性能の関係を可視化

使い方:
    python qbnn_neuron_scaling.py
    python qbnn_neuron_scaling.py --epochs 20
"""

import os
import sys
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# 日本語フォント設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']

# パス設定
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

runpod_dir = os.path.join(current_dir, 'neuroq-runpod')
if runpod_dir not in sys.path:
    sys.path.insert(0, runpod_dir)

print("=" * 70)
print("📊 QBNN ニューロン数スケーリング実験")
print("   - ニューロン数を増加させながら性能を測定")
print("=" * 70)

# インポート
try:
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    print("✅ NeuroQuantum をインポートしました")
except ImportError as e:
    print(f"❌ NeuroQuantumのインポートに失敗: {e}")
    sys.exit(1)

try:
    from tiktoken_tokenizer import TikTokenTokenizer
    print("✅ TikTokenTokenizer をインポートしました")
except ImportError as e:
    print(f"❌ TikTokenTokenizerのインポートに失敗: {e}")
    sys.exit(1)

# ===============================
# ニューロン数の設定（スケーリング実験）
# ===============================
NEURON_CONFIGS = [
    {'name': 'Tiny',   'embed_dim': 64,  'num_heads': 2, 'num_layers': 2},
    {'name': 'Small',  'embed_dim': 128, 'num_heads': 4, 'num_layers': 3},
    {'name': 'Medium', 'embed_dim': 256, 'num_heads': 8, 'num_layers': 4},
    {'name': 'Large',  'embed_dim': 512, 'num_heads': 8, 'num_layers': 6},
]

VOCAB_SIZE = 200019  # o200k_base

# ===============================
# 学習・テストデータ
# ===============================
TRAINING_DATA = [
    "<USER> Hello! <ASSISTANT> Hello! How can I help you today?",
    "<USER> What is AI? <ASSISTANT> AI stands for Artificial Intelligence.",
    "<USER> What is machine learning? <ASSISTANT> Machine learning is a type of AI that learns from data.",
    "<USER> What is Python? <ASSISTANT> Python is a programming language.",
    "<USER> What is the capital of France? <ASSISTANT> The capital of France is Paris.",
    "<USER> What is 2 + 2? <ASSISTANT> 2 + 2 equals 4.",
    "<USER> Who wrote Romeo and Juliet? <ASSISTANT> William Shakespeare wrote Romeo and Juliet.",
    "<USER> What is the largest planet? <ASSISTANT> Jupiter is the largest planet in our solar system.",
    "<USER> What color is the sky? <ASSISTANT> The sky is blue.",
    "<USER> How many days in a week? <ASSISTANT> There are 7 days in a week.",
    "<USER> What is H2O? <ASSISTANT> H2O is the chemical formula for water.",
    "<USER> Thank you! <ASSISTANT> You're welcome!",
    "<USER> Goodbye! <ASSISTANT> Goodbye! Have a great day!",
    "<USER> What is quantum computing? <ASSISTANT> Quantum computing uses quantum mechanics for computation.",
    "<USER> Explain neural networks. <ASSISTANT> Neural networks are computing systems inspired by the brain.",
]

# ベンチマーク用テスト
MULTIPLE_CHOICE_TEST = [
    {"question": "The capital of France is", "choices": ["Paris", "London", "Berlin", "Madrid"], "correct": 0},
    {"question": "Water is made of", "choices": ["CO2", "H2O", "O2", "N2"], "correct": 1},
    {"question": "The largest planet is", "choices": ["Mars", "Earth", "Jupiter", "Venus"], "correct": 2},
    {"question": "2 + 2 equals", "choices": ["3", "4", "5", "6"], "correct": 1},
    {"question": "Shakespeare wrote", "choices": ["Harry Potter", "Romeo and Juliet", "The Lord of the Rings", "Pride"], "correct": 1},
]

QA_TEST = [
    {"question": "What is the capital of Japan?", "keywords": ["tokyo", "capital"]},
    {"question": "What is 3 + 5?", "keywords": ["8", "eight"]},
    {"question": "What color is grass?", "keywords": ["green"]},
]


def generate_text(model, tokenizer, prompt, device, max_length=30, temperature=0.7):
    """テキスト生成"""
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special=False)
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            seq_tensor = torch.tensor([generated[-256:]], device=device)
            logits = model(seq_tensor)
            probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
            top_k = 40
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            top_probs = top_probs / top_probs.sum()
            idx = torch.multinomial(top_probs, num_samples=1)
            next_token = top_indices[idx].item()
            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)
    
    output = tokenizer.decode(generated, skip_special=True)
    if '<ASSISTANT>' in output:
        response = output.split('<ASSISTANT>')[-1].strip()
    else:
        response = output[len(prompt):].strip()
    if '<USER>' in response:
        response = response.split('<USER>')[0].strip()
    return response


def evaluate_qa(model, tokenizer, device):
    """QA評価"""
    correct = 0
    for test in QA_TEST:
        prompt = f"<USER> {test['question']} <ASSISTANT>"
        response = generate_text(model, tokenizer, prompt, device)
        if any(kw.lower() in response.lower() for kw in test['keywords']):
            correct += 1
    return (correct / len(QA_TEST)) * 100


def evaluate_multiple_choice(model, tokenizer, device):
    """選択式問題評価"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    correct = 0
    
    for test in MULTIPLE_CHOICE_TEST:
        question = test['question']
        choices = test['choices']
        correct_idx = test['correct']
        choice_scores = []
        
        for choice in choices:
            text = f"<USER> {question} <ASSISTANT> {choice}"
            tokens = tokenizer.encode(text, add_special=False)
            if len(tokens) < 2:
                choice_scores.append(float('inf'))
                continue
            
            seq_tensor = torch.tensor([tokens], device=device)
            input_ids = seq_tensor[:, :-1]
            target_ids = seq_tensor[:, 1:]
            
            with torch.no_grad():
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_ids.reshape(-1))
                avg_loss = loss.mean().item()
            choice_scores.append(avg_loss)
        
        if np.argmin(choice_scores) == correct_idx:
            correct += 1
    
    return (correct / len(MULTIPLE_CHOICE_TEST)) * 100


def train_and_benchmark_single(config, tokenizer, train_sequences, device, max_epochs, batch_size=4, lr=0.001):
    """単一のニューロン設定で学習とベンチマーク"""
    
    # モデル構築
    model_config = NeuroQuantumConfig()
    model_config.vocab_size = VOCAB_SIZE
    model_config.embed_dim = config['embed_dim']
    model_config.num_heads = config['num_heads']
    model_config.num_layers = config['num_layers']
    model_config.max_seq_len = 256
    model_config.dropout = 0.1
    
    model = NeuroQuantum(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n   📦 {config['name']}: embed_dim={config['embed_dim']}, heads={config['num_heads']}, layers={config['num_layers']}")
    print(f"      パラメータ数: {total_params:,}")
    
    # 学習設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 結果記録
    epoch_results = []
    
    for epoch in range(1, max_epochs + 1):
        # 学習
        model.train()
        total_loss = 0
        batch_count = 0
        random.shuffle(train_sequences)
        
        for i in range(0, len(train_sequences) - batch_size, batch_size):
            batch = train_sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch, device=device)
            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_ids.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / max(batch_count, 1)
        
        # ベンチマーク（5エポックごと + 最終エポック）
        if epoch % 5 == 0 or epoch == max_epochs:
            qa_acc = evaluate_qa(model, tokenizer, device)
            mc_acc = evaluate_multiple_choice(model, tokenizer, device)
            overall = (qa_acc + mc_acc) / 2
            
            epoch_results.append({
                'epoch': epoch,
                'loss': avg_loss,
                'qa_acc': qa_acc,
                'mc_acc': mc_acc,
                'overall': overall,
            })
            
            print(f"      Epoch {epoch:2d}: Loss={avg_loss:.4f}, QA={qa_acc:.0f}%, MC={mc_acc:.0f}%, Overall={overall:.1f}%")
    
    return {
        'name': config['name'],
        'embed_dim': config['embed_dim'],
        'params': total_params,
        'final_loss': epoch_results[-1]['loss'],
        'final_qa': epoch_results[-1]['qa_acc'],
        'final_mc': epoch_results[-1]['mc_acc'],
        'final_overall': epoch_results[-1]['overall'],
        'epoch_results': epoch_results,
    }


def run_scaling_experiment(max_epochs=20, batch_size=4, lr=0.001):
    """スケーリング実験を実行"""
    
    print("\n📥 トークナイザー初期化...")
    tokenizer = TikTokenTokenizer(encoding_name="o200k_base")
    
    # 学習データトークン化
    print("🔄 学習データトークン化中...")
    train_tokens = []
    for text in TRAINING_DATA:
        tokens = tokenizer.encode(text, add_special=False)
        train_tokens.extend(tokens)
        train_tokens.append(tokenizer.eos_id)
    
    seq_length = 64
    train_sequences = []
    for i in range(0, len(train_tokens) - seq_length, seq_length // 2):
        train_sequences.append(train_tokens[i:i + seq_length])
    
    print(f"   学習シーケンス数: {len(train_sequences)}")
    
    # デバイス
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) を使用")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 NVIDIA GPU (CUDA) を使用")
    else:
        device = torch.device("cpu")
        print("💻 CPU を使用")
    
    # 各ニューロン設定で実験
    all_results = []
    
    print(f"\n🚀 スケーリング実験開始（{len(NEURON_CONFIGS)}つの設定 x {max_epochs}エポック）")
    
    for config in NEURON_CONFIGS:
        result = train_and_benchmark_single(
            config, tokenizer, train_sequences, device, max_epochs, batch_size, lr
        )
        all_results.append(result)
    
    return all_results


def plot_scaling_results(results, output_path="qbnn_neuron_scaling.png"):
    """スケーリング結果をグラフ化"""
    
    print(f"\n📊 グラフ生成中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = [r['name'] for r in results]
    params = [r['params'] / 1e6 for r in results]  # 百万単位
    embed_dims = [r['embed_dim'] for r in results]
    final_qa = [r['final_qa'] for r in results]
    final_mc = [r['final_mc'] for r in results]
    final_overall = [r['final_overall'] for r in results]
    final_loss = [r['final_loss'] for r in results]
    
    # 1. パラメータ数 vs 総合スコア
    ax1 = axes[0, 0]
    bars = ax1.bar(names, final_overall, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
    ax1.set_ylabel('総合スコア (%)', fontsize=11)
    ax1.set_title('ニューロン数 vs 総合スコア', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    for bar, param in zip(bars, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{param:.1f}M', ha='center', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 各ベンチマーク結果
    ax2 = axes[0, 1]
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax2.bar(x - width/2, final_qa, width, label='QA精度', color='#27ae60')
    bars2 = ax2.bar(x + width/2, final_mc, width, label='選択式精度', color='#e67e22')
    ax2.set_ylabel('精度 (%)', fontsize=11)
    ax2.set_title('ベンチマーク別スコア', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. パラメータ数と性能の関係（散布図）
    ax3 = axes[1, 0]
    ax3.scatter(params, final_overall, s=200, c=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'], 
                edgecolors='black', linewidths=2)
    for i, name in enumerate(names):
        ax3.annotate(name, (params[i], final_overall[i]), 
                    textcoords="offset points", xytext=(10, 5), fontsize=10)
    ax3.set_xlabel('パラメータ数 (M)', fontsize=11)
    ax3.set_ylabel('総合スコア (%)', fontsize=11)
    ax3.set_title('スケーリング則：パラメータ数 vs 性能', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. エポックごとの学習曲線（各設定）
    ax4 = axes[1, 1]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    for i, result in enumerate(results):
        epochs = [r['epoch'] for r in result['epoch_results']]
        overalls = [r['overall'] for r in result['epoch_results']]
        ax4.plot(epochs, overalls, marker='o', linewidth=2, label=result['name'], color=colors[i])
    ax4.set_xlabel('エポック数', fontsize=11)
    ax4.set_ylabel('総合スコア (%)', fontsize=11)
    ax4.set_title('学習曲線（ニューロン数別）', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ グラフを保存しました: {output_path}")
    
    # 結果サマリー
    print(f"\n" + "=" * 60)
    print("📈 スケーリング実験結果サマリー")
    print("=" * 60)
    print(f"{'設定':<10} {'embed_dim':<10} {'パラメータ':<12} {'QA':<8} {'MC':<8} {'総合':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<10} {r['embed_dim']:<10} {r['params']/1e6:.2f}M{'':<6} {r['final_qa']:<7.1f}% {r['final_mc']:<7.1f}% {r['final_overall']:<7.1f}%")
    print("-" * 60)
    
    # 最良モデル
    best = max(results, key=lambda x: x['final_overall'])
    print(f"\n🏆 最高性能: {best['name']} (総合スコア: {best['final_overall']:.1f}%)")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='QBNN ニューロン数スケーリング実験')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='各設定のエポック数')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--output', '-o', type=str, default='qbnn_neuron_scaling.png', help='出力ファイル')
    
    args = parser.parse_args()
    
    # スケーリング実験
    results = run_scaling_experiment(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # グラフ生成
    plot_scaling_results(results, output_path=args.output)
    
    print("\n" + "=" * 70)
    print("✅ ニューロン数スケーリング実験完了！")
    print("=" * 70)


if __name__ == "__main__":
    main()
