#!/usr/bin/env python3
"""
NeuroQ Small/Large ベンチマーク比較

OpenAI text-embedding-3 の次元数に対応したモデルで
各エポックでLLMベンチマークを実行

使い方:
    python neuroq_benchmark_comparison.py
    python neuroq_benchmark_comparison.py --epochs 20
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
print("📊 NeuroQ Small/Large ベンチマーク比較")
print("   OpenAI text-embedding-3 次元数対応")
print("=" * 70)

# インポート
from neuroq_model_configs import NEUROQ_SMALL, NEUROQ_LARGE, create_model

try:
    from tiktoken_tokenizer import TikTokenTokenizer
    print("✅ TikTokenTokenizer をインポートしました")
except ImportError as e:
    print(f"❌ TikTokenTokenizerのインポートに失敗: {e}")
    sys.exit(1)

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

MULTIPLE_CHOICE_TEST = [
    {"question": "The capital of France is", "choices": ["Paris", "London", "Berlin", "Madrid"], "correct": 0},
    {"question": "Water is made of", "choices": ["CO2", "H2O", "O2", "N2"], "correct": 1},
    {"question": "The largest planet is", "choices": ["Mars", "Earth", "Jupiter", "Venus"], "correct": 2},
    {"question": "2 + 2 equals", "choices": ["3", "4", "5", "6"], "correct": 1},
    {"question": "Shakespeare wrote", "choices": ["Harry Potter", "Romeo and Juliet", "LOTR", "Pride"], "correct": 1},
]

QA_TEST = [
    {"question": "What is the capital of Japan?", "keywords": ["tokyo", "capital"]},
    {"question": "What is 3 + 5?", "keywords": ["8", "eight"]},
    {"question": "What color is grass?", "keywords": ["green"]},
]


def generate_text(model, tokenizer, prompt, device, vocab_size, max_length=30, temperature=0.7):
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


def evaluate_qa(model, tokenizer, device, vocab_size):
    """QA評価"""
    correct = 0
    for test in QA_TEST:
        prompt = f"<USER> {test['question']} <ASSISTANT>"
        response = generate_text(model, tokenizer, prompt, device, vocab_size)
        if any(kw.lower() in response.lower() for kw in test['keywords']):
            correct += 1
    return (correct / len(QA_TEST)) * 100


def evaluate_mc(model, tokenizer, device, vocab_size):
    """選択式問題評価"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    correct = 0
    
    for test in MULTIPLE_CHOICE_TEST:
        choice_scores = []
        for choice in test['choices']:
            text = f"<USER> {test['question']} <ASSISTANT> {choice}"
            tokens = tokenizer.encode(text, add_special=False)
            if len(tokens) < 2:
                choice_scores.append(float('inf'))
                continue
            
            seq_tensor = torch.tensor([tokens], device=device)
            input_ids = seq_tensor[:, :-1]
            target_ids = seq_tensor[:, 1:]
            
            with torch.no_grad():
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
                choice_scores.append(loss.mean().item())
        
        if np.argmin(choice_scores) == test['correct']:
            correct += 1
    
    return (correct / len(MULTIPLE_CHOICE_TEST)) * 100


def train_and_benchmark(config, tokenizer, train_sequences, device, max_epochs, batch_size=2, lr=0.0005):
    """単一モデルの学習とベンチマーク"""
    
    print(f"\n{'='*60}")
    print(f"🧠 {config['name']} (embed_dim={config['embed_dim']})")
    print(f"{'='*60}")
    
    # モデル作成
    model, device = create_model(config, device)
    vocab_size = config['vocab_size']
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   パラメータ数: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   デバイス: {device}")
    
    # 学習設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 結果記録
    results = {
        'epochs': [],
        'losses': [],
        'qa_scores': [],
        'mc_scores': [],
        'overall_scores': [],
    }
    
    print(f"\n{'Epoch':>5} | {'Loss':>8} | {'QA':>6} | {'MC':>6} | {'Overall':>8}")
    print("-" * 45)
    
    for epoch in range(1, max_epochs + 1):
        # 学習
        model.train()
        total_loss = 0
        batch_count = 0
        random.shuffle(train_sequences)
        
        for i in range(0, max(1, len(train_sequences) - batch_size), batch_size):
            batch = train_sequences[i:i + batch_size]
            if len(batch) == 0:
                continue
            batch_tensor = torch.tensor(batch, device=device)
            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / max(batch_count, 1)
        
        # ベンチマーク評価
        qa_acc = evaluate_qa(model, tokenizer, device, vocab_size)
        mc_acc = evaluate_mc(model, tokenizer, device, vocab_size)
        overall = (qa_acc + mc_acc) / 2
        
        # 記録
        results['epochs'].append(epoch)
        results['losses'].append(avg_loss)
        results['qa_scores'].append(qa_acc)
        results['mc_scores'].append(mc_acc)
        results['overall_scores'].append(overall)
        
        print(f"{epoch:>5} | {avg_loss:>8.4f} | {qa_acc:>5.0f}% | {mc_acc:>5.0f}% | {overall:>7.1f}%")
    
    print("-" * 45)
    print(f"✅ 完了: 最終Overall={results['overall_scores'][-1]:.1f}%")
    
    # メモリ解放
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def run_comparison(max_epochs=10, batch_size=2, lr=0.0005):
    """SmallとLargeの比較実験"""
    
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
    
    all_results = {}
    
    # NeuroQ-Small
    print("\n" + "=" * 70)
    print("📦 NeuroQ-Small (embed_dim=1536) をベンチマーク中...")
    print("=" * 70)
    all_results['small'] = train_and_benchmark(
        NEUROQ_SMALL, tokenizer, train_sequences, device, max_epochs, batch_size, lr
    )
    
    # NeuroQ-Large
    print("\n" + "=" * 70)
    print("📦 NeuroQ-Large (embed_dim=3072) をベンチマーク中...")
    print("=" * 70)
    all_results['large'] = train_and_benchmark(
        NEUROQ_LARGE, tokenizer, train_sequences, device, max_epochs, batch_size, lr
    )
    
    return all_results


def plot_comparison(results, output_path="neuroq_small_large_comparison.png"):
    """比較グラフを生成"""
    
    print(f"\n📊 グラフ生成中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    small = results['small']
    large = results['large']
    
    # 1. 学習Loss比較
    ax1 = axes[0, 0]
    ax1.plot(small['epochs'], small['losses'], 'b-', linewidth=2, marker='o', label='NeuroQ-Small (1536)')
    ax1.plot(large['epochs'], large['losses'], 'r-', linewidth=2, marker='s', label='NeuroQ-Large (3072)')
    ax1.set_xlabel('エポック数', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('学習損失の推移', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. QA精度比較
    ax2 = axes[0, 1]
    ax2.plot(small['epochs'], small['qa_scores'], 'b-', linewidth=2, marker='o', label='NeuroQ-Small')
    ax2.plot(large['epochs'], large['qa_scores'], 'r-', linewidth=2, marker='s', label='NeuroQ-Large')
    ax2.set_xlabel('エポック数', fontsize=11)
    ax2.set_ylabel('QA Accuracy (%)', fontsize=11)
    ax2.set_title('質問応答精度 (QA)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # 3. MC精度比較
    ax3 = axes[1, 0]
    ax3.plot(small['epochs'], small['mc_scores'], 'b-', linewidth=2, marker='o', label='NeuroQ-Small')
    ax3.plot(large['epochs'], large['mc_scores'], 'r-', linewidth=2, marker='s', label='NeuroQ-Large')
    ax3.set_xlabel('エポック数', fontsize=11)
    ax3.set_ylabel('MC Accuracy (%)', fontsize=11)
    ax3.set_title('選択式問題精度 (Multiple Choice)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)
    
    # 4. 総合スコア比較
    ax4 = axes[1, 1]
    ax4.plot(small['epochs'], small['overall_scores'], 'b-', linewidth=3, marker='o', label='NeuroQ-Small (1536)')
    ax4.plot(large['epochs'], large['overall_scores'], 'r-', linewidth=3, marker='s', label='NeuroQ-Large (3072)')
    ax4.fill_between(small['epochs'], small['overall_scores'], alpha=0.2, color='blue')
    ax4.fill_between(large['epochs'], large['overall_scores'], alpha=0.2, color='red')
    ax4.set_xlabel('エポック数', fontsize=11)
    ax4.set_ylabel('Overall Score (%)', fontsize=11)
    ax4.set_title('総合スコア比較', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ グラフを保存しました: {output_path}")
    
    # サマリー
    print(f"\n" + "=" * 60)
    print("📈 NeuroQ Small vs Large 比較サマリー")
    print("=" * 60)
    print(f"{'モデル':<20} {'最終Loss':<12} {'QA':<8} {'MC':<8} {'Overall':<10}")
    print("-" * 60)
    print(f"{'NeuroQ-Small (1536)':<20} {small['losses'][-1]:<12.4f} {small['qa_scores'][-1]:<7.0f}% {small['mc_scores'][-1]:<7.0f}% {small['overall_scores'][-1]:<9.1f}%")
    print(f"{'NeuroQ-Large (3072)':<20} {large['losses'][-1]:<12.4f} {large['qa_scores'][-1]:<7.0f}% {large['mc_scores'][-1]:<7.0f}% {large['overall_scores'][-1]:<9.1f}%")
    print("=" * 60)
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='NeuroQ Small/Large ベンチマーク比較')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='エポック数 (デフォルト: 10)')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.0005, help='学習率')
    parser.add_argument('--output', '-o', type=str, default='neuroq_small_large_comparison.png', help='出力ファイル')
    
    args = parser.parse_args()
    
    # 比較実験
    results = run_comparison(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # グラフ生成
    plot_comparison(results, output_path=args.output)
    
    print("\n" + "=" * 70)
    print("✅ NeuroQ Small/Large ベンチマーク比較完了！")
    print("=" * 70)


if __name__ == "__main__":
    main()
