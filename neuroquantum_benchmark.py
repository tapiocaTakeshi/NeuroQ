#!/usr/bin/env python3
"""
NeuroQuantum ベンチマークテスト
================================
QBNN（Quantum-Bit Neural Network）の性能評価

テスト項目：
1. 学習速度
2. 推論速度
3. 生成品質
4. QBNNもつれ効果
5. メモリ効率
"""

import torch
import torch.nn as nn
import time
import math
import psutil
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

# NeuroQuantumからインポート
from neuroquantum import (
    NeuroQuantumAI, QBNNLayer, QBNNAttention, 
    QBNNTransformerBlock, NeuroQuantum, NeuroQuantumTokenizer
)

plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']


def get_memory_usage():
    """メモリ使用量を取得（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_qbnn_layer():
    """QBNNLayer単体のベンチマーク"""
    print("\n" + "=" * 60)
    print("🔬 テスト1: QBNNLayer ベンチマーク")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    # 異なるサイズでテスト
    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    
    for in_dim, out_dim in sizes:
        layer = QBNNLayer(in_dim, out_dim, lambda_min=0.2, lambda_max=0.5).to(device)
        x = torch.randn(32, 64, in_dim).to(device)  # (batch, seq, dim)
        
        # ウォームアップ
        for _ in range(10):
            _ = layer(x)
        
        # 計測
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = layer(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) / iterations * 1000  # ms
        
        # パラメータ数
        params = sum(p.numel() for p in layer.parameters())
        
        results.append({
            'size': f'{in_dim}x{out_dim}',
            'time_ms': elapsed,
            'params': params,
            'throughput': 32 * 64 / elapsed * 1000  # tokens/sec
        })
        
        print(f"   サイズ {in_dim}x{out_dim}: {elapsed:.3f}ms, パラメータ: {params:,}, スループット: {results[-1]['throughput']:.0f} tokens/s")
    
    return results


def benchmark_attention_comparison():
    """QBNN Attention vs 通常Attentionの比較"""
    print("\n" + "=" * 60)
    print("🔬 テスト2: QBNN Attention vs 標準 Attention")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_dim = 128
    num_heads = 4
    batch_size = 16
    seq_len = 64
    
    # QBNN Attention
    qbnn_attn = QBNNAttention(embed_dim, num_heads, lambda_val=0.35).to(device)
    
    # 標準 Attention
    std_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(device)
    
    x = torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    results = {}
    
    # QBNN Attention計測
    for _ in range(10):
        _ = qbnn_attn(x)
    
    start = time.perf_counter()
    for _ in range(50):
        _ = qbnn_attn(x)
    qbnn_time = (time.perf_counter() - start) / 50 * 1000
    
    # 標準Attention計測
    for _ in range(10):
        _, _ = std_attn(x, x, x)
    
    start = time.perf_counter()
    for _ in range(50):
        _, _ = std_attn(x, x, x)
    std_time = (time.perf_counter() - start) / 50 * 1000
    
    results['qbnn_attention'] = qbnn_time
    results['standard_attention'] = std_time
    results['overhead'] = (qbnn_time / std_time - 1) * 100
    
    print(f"   QBNN Attention:    {qbnn_time:.3f} ms")
    print(f"   標準 Attention:    {std_time:.3f} ms")
    print(f"   オーバーヘッド:    {results['overhead']:.1f}%")
    
    return results


def benchmark_training_speed():
    """学習速度のベンチマーク"""
    print("\n" + "=" * 60)
    print("🔬 テスト3: 学習速度ベンチマーク")
    print("=" * 60)
    
    # 小さなテストデータ
    test_data = [
        ("こんにちは", "こんにちは！お元気ですか？"),
        ("AIとは", "人工知能は機械学習を使った技術です。"),
        ("量子とは", "量子は物質の最小単位です。"),
        ("Hello", "Hello! How can I help you?"),
        ("What is AI", "AI is artificial intelligence."),
    ] * 10  # 50ペア
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 異なる設定でテスト
    configs = [
        {'embed_dim': 64, 'hidden_dim': 128, 'num_layers': 2, 'name': 'Small'},
        {'embed_dim': 128, 'hidden_dim': 256, 'num_layers': 4, 'name': 'Medium'},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n   設定: {config['name']} (embed={config['embed_dim']}, layers={config['num_layers']})")
        
        ai = NeuroQuantumAI(
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=4,
            num_layers=config['num_layers'],
            max_seq_len=64,
            dropout=0.1,
            lambda_entangle=0.35,
        )
        
        mem_before = get_memory_usage()
        
        start = time.perf_counter()
        ai.train(test_data, epochs=5, batch_size=8, lr=0.001, seq_len=32)
        train_time = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        
        # パラメータ数
        total_params = ai.model.num_params
        
        results.append({
            'name': config['name'],
            'train_time': train_time,
            'params': total_params,
            'memory_mb': mem_after - mem_before,
            'time_per_epoch': train_time / 5,
            'tokens_per_sec': len(test_data) * 32 * 5 / train_time
        })
        
        print(f"      学習時間: {train_time:.2f}秒 ({train_time/5:.3f}秒/エポック)")
        print(f"      パラメータ数: {total_params:,}")
        print(f"      メモリ増加: {mem_after - mem_before:.1f} MB")
        print(f"      スループット: {results[-1]['tokens_per_sec']:.0f} tokens/s")
    
    return results


def benchmark_generation_quality():
    """生成品質のベンチマーク"""
    print("\n" + "=" * 60)
    print("🔬 テスト4: 生成品質ベンチマーク")
    print("=" * 60)
    
    # テストデータで学習
    test_data = [
        ("こんにちは", "こんにちは！元気ですか？何かお手伝いできることはありますか？"),
        ("AIとは", "AIは人工知能の略で、機械学習などの技術を使って知的な処理を行うシステムです。"),
        ("量子コンピュータとは", "量子コンピュータは量子力学の原理を使った次世代のコンピュータです。"),
        ("プログラミングとは", "プログラミングはコンピュータに指示を与えるための言語を書く作業です。"),
        ("Hello", "Hello! How are you today? I'm here to help!"),
        ("What is machine learning", "Machine learning is a type of AI that enables computers to learn from data."),
        ("Explain quantum computing", "Quantum computing uses quantum bits to perform calculations in parallel."),
    ] * 20
    
    ai = NeuroQuantumAI(
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=128,
        dropout=0.1,
        lambda_entangle=0.35,
    )
    
    print("\n   学習中...")
    ai.train(test_data, epochs=30, batch_size=16, lr=0.001, seq_len=64)
    
    # テストプロンプト
    test_prompts = [
        "こんにちは",
        "AIとは",
        "Hello",
        "What is",
    ]
    
    print("\n   生成テスト:")
    results = []
    
    for prompt in test_prompts:
        # 異なる温度範囲でテスト
        temp_configs = [
            (0.3, 0.5, "低"),
            (0.4, 0.8, "中"),
            (0.6, 1.0, "高"),
        ]
        
        print(f"\n   プロンプト: '{prompt}'")
        
        for temp_min, temp_max, label in temp_configs:
            start = time.perf_counter()
            response = ai.generate(prompt, max_length=50, temp_min=temp_min, temp_max=temp_max)
            gen_time = time.perf_counter() - start
            
            # 品質指標
            unique_chars = len(set(response))
            repetition = 1 - (unique_chars / max(len(response), 1))
            
            results.append({
                'prompt': prompt,
                'temp_range': f'{temp_min}-{temp_max}',
                'response': response[:60],
                'gen_time': gen_time,
                'length': len(response),
                'unique_ratio': unique_chars / max(len(response), 1),
                'repetition': repetition
            })
            
            print(f"      温度{label}({temp_min}-{temp_max}): {response[:40]}...")
    
    return results


def benchmark_quantum_entanglement():
    """量子もつれ効果のベンチマーク"""
    print("\n" + "=" * 60)
    print("🔬 テスト5: 量子もつれ効果分析")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 異なるλ範囲でのQBNNレイヤー
    lambda_configs = [
        (0.0, 0.1, "微弱"),
        (0.2, 0.5, "中程度"),
        (0.5, 0.9, "強力"),
    ]
    
    results = []
    
    for lambda_min, lambda_max, label in lambda_configs:
        layer = QBNNLayer(128, 128, lambda_min=lambda_min, lambda_max=lambda_max).to(device)
        x = torch.randn(16, 32, 128).to(device)
        
        # Forward通過
        output = layer(x)
        
        # 量子情報取得
        q_info = layer.get_quantum_info()
        
        # 出力の統計
        output_mean = output.mean().item()
        output_std = output.std().item()
        
        results.append({
            'label': label,
            'lambda_range': f'{lambda_min}-{lambda_max}',
            'lambda_eff': q_info['lambda_eff'],
            'J_mean': q_info['J_mean'],
            'J_std': q_info['J_std'],
            'output_mean': output_mean,
            'output_std': output_std,
        })
        
        print(f"\n   {label}もつれ (λ={lambda_min}-{lambda_max}):")
        print(f"      有効λ: {q_info['lambda_eff']:.4f}")
        print(f"      J平均: {q_info['J_mean']:.6f}, J標準偏差: {q_info['J_std']:.4f}")
        print(f"      出力平均: {output_mean:.4f}, 出力標準偏差: {output_std:.4f}")
    
    return results


def benchmark_memory_efficiency():
    """メモリ効率のベンチマーク"""
    print("\n" + "=" * 60)
    print("🔬 テスト6: メモリ効率分析")
    print("=" * 60)
    
    results = []
    
    configs = [
        {'embed_dim': 64, 'layers': 2, 'name': 'Tiny'},
        {'embed_dim': 128, 'layers': 4, 'name': 'Small'},
        {'embed_dim': 256, 'layers': 6, 'name': 'Medium'},
    ]
    
    base_memory = get_memory_usage()
    
    for config in configs:
        # GC
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        mem_before = get_memory_usage()
        
        ai = NeuroQuantumAI(
            embed_dim=config['embed_dim'],
            hidden_dim=config['embed_dim'] * 2,
            num_heads=4,
            num_layers=config['layers'],
            max_seq_len=128,
            dropout=0.1,
            lambda_entangle=0.35,
        )
        
        # モデル初期化のためにダミー学習
        dummy_data = [("test", "test response")] * 5
        ai.train(dummy_data, epochs=1, batch_size=4, lr=0.001, seq_len=32)
        
        mem_after = get_memory_usage()
        
        params = ai.model.num_params
        memory_used = mem_after - mem_before
        
        results.append({
            'name': config['name'],
            'embed_dim': config['embed_dim'],
            'layers': config['layers'],
            'params': params,
            'memory_mb': memory_used,
            'bytes_per_param': memory_used * 1024 * 1024 / params if params > 0 else 0
        })
        
        print(f"   {config['name']}: {params:,} params, {memory_used:.1f} MB ({results[-1]['bytes_per_param']:.1f} bytes/param)")
        
        del ai
    
    return results


def create_benchmark_report(all_results: Dict):
    """ベンチマーク結果のビジュアルレポート作成"""
    print("\n" + "=" * 60)
    print("📊 ベンチマーク結果サマリー")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('NeuroQuantum (QBNN) ベンチマーク結果', fontsize=14, fontweight='bold')
    
    # 1. QBNNレイヤー速度
    ax1 = axes[0, 0]
    if 'qbnn_layer' in all_results:
        sizes = [r['size'] for r in all_results['qbnn_layer']]
        times = [r['time_ms'] for r in all_results['qbnn_layer']]
        ax1.bar(sizes, times, color='#4CAF50')
        ax1.set_xlabel('Layer Size')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('QBNNLayer 処理時間')
    
    # 2. Attention比較
    ax2 = axes[0, 1]
    if 'attention' in all_results:
        labels = ['QBNN\nAttention', 'Standard\nAttention']
        times = [all_results['attention']['qbnn_attention'], 
                 all_results['attention']['standard_attention']]
        colors = ['#2196F3', '#FF9800']
        ax2.bar(labels, times, color=colors)
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('Attention 比較')
        ax2.text(0.5, max(times) * 0.8, f"+{all_results['attention']['overhead']:.1f}%", 
                ha='center', fontsize=10, color='red')
    
    # 3. 学習速度
    ax3 = axes[0, 2]
    if 'training' in all_results:
        names = [r['name'] for r in all_results['training']]
        throughput = [r['tokens_per_sec'] for r in all_results['training']]
        ax3.bar(names, throughput, color='#9C27B0')
        ax3.set_ylabel('Tokens/sec')
        ax3.set_title('学習スループット')
    
    # 4. 量子もつれ効果
    ax4 = axes[1, 0]
    if 'entanglement' in all_results:
        labels = [r['label'] for r in all_results['entanglement']]
        lambda_eff = [r['lambda_eff'] for r in all_results['entanglement']]
        output_std = [r['output_std'] for r in all_results['entanglement']]
        
        x = np.arange(len(labels))
        width = 0.35
        ax4.bar(x - width/2, lambda_eff, width, label='λ_eff', color='#00BCD4')
        ax4.bar(x + width/2, output_std, width, label='出力σ', color='#E91E63')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_title('量子もつれ効果')
        ax4.legend()
    
    # 5. メモリ効率
    ax5 = axes[1, 1]
    if 'memory' in all_results:
        names = [r['name'] for r in all_results['memory']]
        memory = [r['memory_mb'] for r in all_results['memory']]
        ax5.bar(names, memory, color='#FF5722')
        ax5.set_ylabel('Memory (MB)')
        ax5.set_title('メモリ使用量')
    
    # 6. パラメータ数 vs 性能
    ax6 = axes[1, 2]
    if 'memory' in all_results and 'training' in all_results:
        for m, t in zip(all_results['memory'], all_results['training']):
            ax6.scatter(m['params'] / 1000, t['tokens_per_sec'], s=100, alpha=0.7)
            ax6.annotate(m['name'], (m['params'] / 1000, t['tokens_per_sec']),
                        xytext=(5, 5), textcoords='offset points')
        ax6.set_xlabel('Parameters (K)')
        ax6.set_ylabel('Throughput (tokens/s)')
        ax6.set_title('パラメータ数 vs 学習速度')
    
    plt.tight_layout()
    plt.savefig('/Users/yuyahiguchi/Program/Qubit/neuroquantum_benchmark.png', dpi=150, bbox_inches='tight')
    print("\n   📈 グラフを保存しました: neuroquantum_benchmark.png")
    
    return fig


def main():
    print("=" * 60)
    print("🚀 NeuroQuantum (QBNN) ベンチマークテスト")
    print("=" * 60)
    print(f"   デバイス: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   PyTorch: {torch.__version__}")
    
    all_results = {}
    
    # テスト実行
    all_results['qbnn_layer'] = benchmark_qbnn_layer()
    all_results['attention'] = benchmark_attention_comparison()
    all_results['training'] = benchmark_training_speed()
    all_results['entanglement'] = benchmark_quantum_entanglement()
    all_results['memory'] = benchmark_memory_efficiency()
    
    # 生成品質（時間がかかるので最後）
    all_results['generation'] = benchmark_generation_quality()
    
    # レポート作成
    create_benchmark_report(all_results)
    
    print("\n" + "=" * 60)
    print("✅ ベンチマーク完了！")
    print("=" * 60)
    
    return all_results


if __name__ == '__main__':
    main()

