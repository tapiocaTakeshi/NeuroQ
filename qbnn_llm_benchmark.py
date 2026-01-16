#!/usr/bin/env python3
"""
QBNN LLMベンチマーク評価スクリプト

Claude/ChatGPTのような大規模言語モデルが使用する評価手法を
小規模モデル向けに実装

評価項目:
1. 生成品質 (BLEU Score) - 生成テキストと正解の類似度
2. 質問応答 (QA Accuracy) - 質問に対する回答の正確性
3. 選択式問題 (Multiple Choice) - 常識推論テスト
4. 文の完成度 (Coherence) - 生成テキストの一貫性

使い方:
    python qbnn_llm_benchmark.py
    python qbnn_llm_benchmark.py --max_epochs 50
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
print("📊 QBNN LLMベンチマーク（Claude/ChatGPT方式の評価）")
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

# モデル設定
MODEL_CONFIG = {
    'vocab_size': 200019,
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 3,
}

# ===============================
# 学習データ
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
    "<USER> What is the speed of light? <ASSISTANT> The speed of light is approximately 300,000 km per second.",
    "<USER> Thank you! <ASSISTANT> You're welcome!",
    "<USER> Goodbye! <ASSISTANT> Goodbye! Have a great day!",
    "<USER> What is quantum computing? <ASSISTANT> Quantum computing uses quantum mechanics for computation.",
    "<USER> Explain neural networks. <ASSISTANT> Neural networks are computing systems inspired by the brain.",
]

# ===============================
# ベンチマーク用テストセット
# ===============================

# 1. 質問応答テスト (QA) - 質問と期待される回答キーワード
QA_TEST = [
    {"question": "What is the capital of Japan?", "keywords": ["tokyo", "capital"]},
    {"question": "What is 3 + 5?", "keywords": ["8", "eight"]},
    {"question": "What color is grass?", "keywords": ["green"]},
    {"question": "How many months in a year?", "keywords": ["12", "twelve"]},
    {"question": "What is the sun?", "keywords": ["star", "light", "hot"]},
]

# 2. 選択式問題 (Multiple Choice) - 正しい選択肢を選べるか
MULTIPLE_CHOICE_TEST = [
    {
        "question": "The capital of France is",
        "choices": ["Paris", "London", "Berlin", "Madrid"],
        "correct": 0  # Paris
    },
    {
        "question": "Water is made of",
        "choices": ["CO2", "H2O", "O2", "N2"],
        "correct": 1  # H2O
    },
    {
        "question": "The largest planet is",
        "choices": ["Mars", "Earth", "Jupiter", "Venus"],
        "correct": 2  # Jupiter
    },
    {
        "question": "2 + 2 equals",
        "choices": ["3", "4", "5", "6"],
        "correct": 1  # 4
    },
    {
        "question": "Shakespeare wrote",
        "choices": ["Harry Potter", "Romeo and Juliet", "The Lord of the Rings", "Pride and Prejudice"],
        "correct": 1  # Romeo and Juliet
    },
]

# 3. 文完成テスト - 参照回答とBLEUスコアで比較
COMPLETION_TEST = [
    {"prompt": "<USER> Hello! <ASSISTANT>", "reference": "Hello! How can I help you today?"},
    {"prompt": "<USER> Thank you! <ASSISTANT>", "reference": "You're welcome!"},
    {"prompt": "<USER> Goodbye! <ASSISTANT>", "reference": "Goodbye! Have a great day!"},
    {"prompt": "<USER> What is Python? <ASSISTANT>", "reference": "Python is a programming language."},
]


def compute_bleu(reference, hypothesis, n=4):
    """
    簡易BLEUスコア計算
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    # n-gramの一致率を計算
    scores = []
    for i in range(1, min(n + 1, len(hyp_tokens) + 1)):
        ref_ngrams = Counter([tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[j:j+i]) for j in range(len(hyp_tokens) - i + 1)])
        
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        
        if total > 0:
            scores.append(matches / total)
        else:
            scores.append(0.0)
    
    if not scores or all(s == 0 for s in scores):
        return 0.0
    
    # 幾何平均
    score = np.exp(np.mean([np.log(max(s, 1e-10)) for s in scores]))
    
    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    
    return bp * score * 100  # パーセントで返す


def generate_text(model, tokenizer, prompt, device, max_length=50, temperature=0.7):
    """
    テキスト生成
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special=False)
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            seq_tensor = torch.tensor([generated[-256:]], device=device)
            logits = model(seq_tensor)
            
            probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
            
            # Top-k sampling
            top_k = 40
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            top_probs = top_probs / top_probs.sum()
            idx = torch.multinomial(top_probs, num_samples=1)
            next_token = top_indices[idx].item()
            
            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)
    
    output = tokenizer.decode(generated, skip_special=True)
    
    # <ASSISTANT>以降を抽出
    if '<ASSISTANT>' in output:
        response = output.split('<ASSISTANT>')[-1].strip()
    else:
        response = output[len(prompt):].strip()
    
    # <USER>以降を削除
    if '<USER>' in response:
        response = response.split('<USER>')[0].strip()
    
    return response


def evaluate_qa(model, tokenizer, device):
    """
    質問応答の評価
    正解キーワードが含まれているかをチェック
    """
    correct = 0
    total = len(QA_TEST)
    
    for test in QA_TEST:
        prompt = f"<USER> {test['question']} <ASSISTANT>"
        response = generate_text(model, tokenizer, prompt, device, max_length=30)
        response_lower = response.lower()
        
        # いずれかのキーワードが含まれていれば正解
        if any(kw.lower() in response_lower for kw in test['keywords']):
            correct += 1
    
    return (correct / total) * 100


def evaluate_multiple_choice(model, tokenizer, device):
    """
    選択式問題の評価
    各選択肢のスコアを計算し、最高スコアの選択肢を選ぶ
    """
    correct = 0
    total = len(MULTIPLE_CHOICE_TEST)
    
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    for test in MULTIPLE_CHOICE_TEST:
        question = test['question']
        choices = test['choices']
        correct_idx = test['correct']
        
        choice_scores = []
        
        for choice in choices:
            # 質問+選択肢のシーケンスを作成
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
                loss = criterion(
                    logits.reshape(-1, MODEL_CONFIG['vocab_size']),
                    target_ids.reshape(-1)
                )
                avg_loss = loss.mean().item()
            
            choice_scores.append(avg_loss)
        
        # 最も低いロス（高い確率）の選択肢を選ぶ
        predicted = np.argmin(choice_scores)
        if predicted == correct_idx:
            correct += 1
    
    return (correct / total) * 100


def evaluate_bleu(model, tokenizer, device):
    """
    生成テキストのBLEUスコア評価
    """
    total_bleu = 0
    
    for test in COMPLETION_TEST:
        response = generate_text(model, tokenizer, test['prompt'], device, max_length=30)
        bleu = compute_bleu(test['reference'], response)
        total_bleu += bleu
    
    return total_bleu / len(COMPLETION_TEST)


def run_all_benchmarks(model, tokenizer, device):
    """
    全ベンチマークを実行
    """
    results = {
        'qa_accuracy': evaluate_qa(model, tokenizer, device),
        'mc_accuracy': evaluate_multiple_choice(model, tokenizer, device),
        'bleu_score': evaluate_bleu(model, tokenizer, device),
    }
    
    # 総合スコア（平均）
    results['overall'] = np.mean([results['qa_accuracy'], results['mc_accuracy'], results['bleu_score']])
    
    return results


def train_and_benchmark(max_epochs, batch_size=4, seq_length=64, lr=0.001):
    """
    学習しながら各エポックでベンチマーク評価
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
    
    print(f"   学習トークン数: {len(train_tokens):,}")
    
    # シーケンスを作成
    train_sequences = []
    for i in range(0, len(train_tokens) - seq_length, seq_length // 2):
        train_sequences.append(train_tokens[i:i + seq_length])
    
    print(f"   学習シーケンス数: {len(train_sequences):,}")
    
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
    qa_scores = []
    mc_scores = []
    bleu_scores = []
    overall_scores = []
    
    print(f"\n🚀 学習開始 (最大 {max_epochs} エポック)...")
    print(f"   各エポック終了時にLLMベンチマークを実行\n")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'QA Acc%':>8} | {'MC Acc%':>8} | {'BLEU':>8} | {'Overall':>8}")
    print("-" * 65)
    
    for epoch in range(1, max_epochs + 1):
        # ===== 学習フェーズ =====
        model.train()
        total_train_loss = 0
        batch_count = 0
        
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
        
        # ===== ベンチマーク評価 =====
        results = run_all_benchmarks(model, tokenizer, device)
        
        # 記録
        epochs_list.append(epoch)
        train_losses.append(avg_train_loss)
        qa_scores.append(results['qa_accuracy'])
        mc_scores.append(results['mc_accuracy'])
        bleu_scores.append(results['bleu_score'])
        overall_scores.append(results['overall'])
        
        # 進捗表示
        print(f"{epoch:>5} | {avg_train_loss:>8.4f} | {results['qa_accuracy']:>7.1f}% | {results['mc_accuracy']:>7.1f}% | {results['bleu_score']:>7.1f}% | {results['overall']:>7.1f}%")
    
    print("-" * 65)
    print(f"\n✅ 学習完了！")
    print(f"   最終QA精度: {qa_scores[-1]:.1f}%")
    print(f"   最終選択式精度: {mc_scores[-1]:.1f}%")
    print(f"   最終BLEUスコア: {bleu_scores[-1]:.1f}%")
    print(f"   最終総合スコア: {overall_scores[-1]:.1f}%")
    
    # サンプル生成を表示
    print(f"\n📝 サンプル生成:")
    test_prompts = [
        "<USER> Hello! <ASSISTANT>",
        "<USER> What is AI? <ASSISTANT>",
        "<USER> Thank you! <ASSISTANT>",
    ]
    for prompt in test_prompts:
        response = generate_text(model, tokenizer, prompt, device, max_length=30)
        q = prompt.replace("<USER> ", "").replace(" <ASSISTANT>", "")
        print(f"   Q: {q}")
        print(f"   A: {response[:80]}")
        print()
    
    return {
        'epochs': epochs_list,
        'train_losses': train_losses,
        'qa_scores': qa_scores,
        'mc_scores': mc_scores,
        'bleu_scores': bleu_scores,
        'overall_scores': overall_scores,
    }


def plot_benchmark_results(results, output_path="qbnn_llm_benchmark.png"):
    """
    ベンチマーク結果をグラフ化
    """
    print(f"\n📊 グラフ生成中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = results['epochs']
    
    # 1. 学習Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, results['train_losses'], 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('エポック数', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('学習損失の推移', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 質問応答精度
    ax2 = axes[0, 1]
    ax2.plot(epochs, results['qa_scores'], 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('エポック数', fontsize=11)
    ax2.set_ylabel('QA Accuracy (%)', fontsize=11)
    ax2.set_title('質問応答精度 (QA)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # 3. 選択式問題精度
    ax3 = axes[1, 0]
    ax3.plot(epochs, results['mc_scores'], 'r-', linewidth=2, marker='^', markersize=4)
    ax3.plot(epochs, results['bleu_scores'], 'm-', linewidth=2, marker='d', markersize=3, label='BLEU Score')
    ax3.set_xlabel('エポック数', fontsize=11)
    ax3.set_ylabel('Score (%)', fontsize=11)
    ax3.set_title('選択式問題精度 & BLEUスコア', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(['Multiple Choice', 'BLEU Score'])
    ax3.set_ylim(0, 105)
    
    # 4. 総合スコア
    ax4 = axes[1, 1]
    ax4.plot(epochs, results['overall_scores'], 'purple', linewidth=3, marker='*', markersize=6)
    ax4.fill_between(epochs, results['overall_scores'], alpha=0.3, color='purple')
    ax4.set_xlabel('エポック数', fontsize=11)
    ax4.set_ylabel('Overall Score (%)', fontsize=11)
    ax4.set_title('総合スコア (Claude/ChatGPT方式)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ グラフを保存しました: {output_path}")
    
    # 統計情報
    best_epoch = np.argmax(results['overall_scores']) + 1
    best_score = max(results['overall_scores'])
    
    print(f"\n📈 ベンチマーク統計:")
    print(f"   最高総合スコア: {best_score:.1f}% (エポック {best_epoch})")
    print(f"   最終QA精度: {results['qa_scores'][-1]:.1f}%")
    print(f"   最終MC精度: {results['mc_scores'][-1]:.1f}%")
    print(f"   最終BLEU: {results['bleu_scores'][-1]:.1f}%")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='QBNN LLMベンチマーク（Claude/ChatGPT方式の評価）'
    )
    parser.add_argument('--max_epochs', '-e', type=int, default=30, help='最大エポック数')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--output', '-o', type=str, default='qbnn_llm_benchmark.png', help='出力ファイル')
    
    args = parser.parse_args()
    
    # 学習とベンチマーク
    results = train_and_benchmark(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # グラフ生成
    plot_benchmark_results(results, output_path=args.output)
    
    print("\n" + "=" * 70)
    print("✅ LLMベンチマーク完了！")
    print("=" * 70)


if __name__ == "__main__":
    main()
