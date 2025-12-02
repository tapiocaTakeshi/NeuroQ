#!/usr/bin/env python3
"""
QBNN Brain 動的入出力テスト
============================
数学問題・回帰分析・パスファインダーなど
動的入出力が有効に機能しているかを検証
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

from qbnn_brain import QBNNBrain, QBNNBrainTorch


# ========================================
# 1. 数学問題テスト
# ========================================

def test_math_problems():
    """数学問題テスト（動的入出力）"""
    print("=" * 60)
    print("📐 数学問題テスト（動的入出力 QBNN Brain）")
    print("=" * 60)
    
    results = {}
    
    # --- 1.1 XOR問題 ---
    print("\n🔹 1.1 XOR問題")
    print("-" * 40)
    
    model = QBNNBrainTorch(
        num_neurons=40,
        max_input_size=10,
        max_output_size=5,
        connection_density=0.25
    )
    
    # XORデータ
    X_xor = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    
    y_xor = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    
    print("   学習中...")
    for epoch in range(200):
        optimizer.zero_grad()
        # dynamic_selection=False で学習（安定性のため）
        pred, in_idx, out_idx = model(X_xor, input_size=2, output_size=1, 
                                       time_steps=3, dynamic_selection=False)
        loss = criterion(pred[:, :1], y_xor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    # テスト（動的入出力で）
    model.eval()
    correct = 0
    print("\n   動的入出力でテスト:")
    with torch.no_grad():
        for i in range(4):
            # 動的選択で推論
            pred, in_idx, out_idx = model(X_xor[i:i+1], input_size=2, output_size=1, 
                                          time_steps=3, dynamic_selection=True)
            pred_val = pred[0, 0].item()
            expected = y_xor[i, 0].item()
            correct += 1 if abs(pred_val - expected) < 0.5 else 0
            status = "✅" if abs(pred_val - expected) < 0.5 else "❌"
            print(f"   {status} ({int(X_xor[i,0])},{int(X_xor[i,1])}) → 予測:{pred_val:.3f}, 正解:{expected:.0f}")
            print(f"      入力ニューロン: {in_idx.tolist()}, 出力ニューロン: {out_idx.tolist()}")
    
    results['XOR'] = correct / 4.0
    print(f"\n   XOR正答率: {correct}/4 = {results['XOR']*100:.1f}%")
    
    # --- 1.2 加算問題 ---
    print("\n🔹 1.2 加算問題 (a + b)")
    print("-" * 40)
    
    model_add = QBNNBrainTorch(
        num_neurons=50,
        max_input_size=10,
        max_output_size=5,
        connection_density=0.2
    )
    
    # 加算データ（正規化）
    np.random.seed(42)
    X_add = []
    y_add = []
    for _ in range(100):
        a = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)
        X_add.append([a, b, 0, 0, 0, 0, 0, 0, 0, 0])
        y_add.append([(a + b) / 2])  # 0-1に正規化
    
    X_add = torch.tensor(X_add, dtype=torch.float32)
    y_add = torch.tensor(y_add, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model_add.parameters(), lr=0.01)
    
    print("   学習中...")
    for epoch in range(200):
        optimizer.zero_grad()
        pred, _, _ = model_add(X_add, input_size=2, output_size=1, 
                               time_steps=3, dynamic_selection=False)
        loss = criterion(pred[:, :1], y_add)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    # テスト
    model_add.eval()
    errors = []
    print("\n   動的入出力でテスト（5サンプル）:")
    with torch.no_grad():
        for i in range(5):
            pred, in_idx, out_idx = model_add(X_add[i:i+1], input_size=2, output_size=1,
                                               time_steps=3, dynamic_selection=True)
            pred_val = pred[0, 0].item() * 2  # 逆正規化
            expected = y_add[i, 0].item() * 2
            error = abs(pred_val - expected)
            errors.append(error)
            print(f"   {X_add[i,0]:.2f} + {X_add[i,1]:.2f} = 予測:{pred_val:.3f}, 正解:{expected:.3f}, 誤差:{error:.3f}")
            print(f"      入力ニューロン: {in_idx.tolist()[:3]}..., 出力ニューロン: {out_idx.tolist()}")
    
    results['加算_MAE'] = np.mean(errors)
    print(f"\n   加算MAE: {results['加算_MAE']:.4f}")
    
    # --- 1.3 乗算問題 ---
    print("\n🔹 1.3 乗算問題 (a × b)")
    print("-" * 40)
    
    model_mul = QBNNBrainTorch(
        num_neurons=60,
        max_input_size=10,
        max_output_size=5,
        connection_density=0.25
    )
    
    # 乗算データ
    X_mul = []
    y_mul = []
    for _ in range(100):
        a = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)
        X_mul.append([a, b, 0, 0, 0, 0, 0, 0, 0, 0])
        y_mul.append([a * b])
    
    X_mul = torch.tensor(X_mul, dtype=torch.float32)
    y_mul = torch.tensor(y_mul, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model_mul.parameters(), lr=0.01)
    
    print("   学習中...")
    for epoch in range(200):
        optimizer.zero_grad()
        pred, _, _ = model_mul(X_mul, input_size=2, output_size=1, 
                               time_steps=4, dynamic_selection=False)
        loss = criterion(pred[:, :1], y_mul)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    model_mul.eval()
    errors = []
    print("\n   動的入出力でテスト（5サンプル）:")
    with torch.no_grad():
        for i in range(5):
            pred, in_idx, out_idx = model_mul(X_mul[i:i+1], input_size=2, output_size=1,
                                               time_steps=4, dynamic_selection=True)
            pred_val = pred[0, 0].item()
            expected = y_mul[i, 0].item()
            error = abs(pred_val - expected)
            errors.append(error)
            print(f"   {X_mul[i,0]:.2f} × {X_mul[i,1]:.2f} = 予測:{pred_val:.3f}, 正解:{expected:.3f}")
    
    results['乗算_MAE'] = np.mean(errors)
    print(f"\n   乗算MAE: {results['乗算_MAE']:.4f}")
    
    return results


# ========================================
# 2. 回帰分析テスト
# ========================================

def test_regression():
    """回帰分析テスト（動的入出力）"""
    print("\n" + "=" * 60)
    print("📈 回帰分析テスト（動的入出力 QBNN Brain）")
    print("=" * 60)
    
    results = {}
    
    # --- 2.1 線形回帰 ---
    print("\n🔹 2.1 線形回帰 (y = 2x + 1)")
    print("-" * 40)
    
    model_lin = QBNNBrainTorch(
        num_neurons=30,
        max_input_size=10,
        max_output_size=5,
        connection_density=0.2
    )
    
    # データ生成
    np.random.seed(42)
    X_lin = np.random.uniform(-1, 1, (100, 1))
    y_lin = 2 * X_lin + 1 + np.random.normal(0, 0.1, (100, 1))
    
    # 正規化
    y_lin_norm = (y_lin - y_lin.min()) / (y_lin.max() - y_lin.min())
    
    X_lin_pad = np.pad(X_lin, ((0, 0), (0, 9)), 'constant')
    X_lin_t = torch.tensor(X_lin_pad, dtype=torch.float32)
    y_lin_t = torch.tensor(y_lin_norm, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model_lin.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("   学習中...")
    losses = []
    for epoch in range(300):
        optimizer.zero_grad()
        pred, _, _ = model_lin(X_lin_t, input_size=1, output_size=1, 
                               time_steps=3, dynamic_selection=False)
        loss = criterion(pred[:, :1], y_lin_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # テスト（動的入出力）
    model_lin.eval()
    with torch.no_grad():
        # 複数回テスト（動的選択の影響を見る）
        preds_all = []
        for trial in range(5):
            pred, in_idx, out_idx = model_lin(X_lin_t[:20], input_size=1, output_size=1,
                                              time_steps=3, dynamic_selection=True)
            preds_all.append(pred[:, 0].numpy())
        
        preds_all = np.array(preds_all)
        pred_mean = preds_all.mean(axis=0)
        pred_std = preds_all.std(axis=0)
    
    # R²スコア
    ss_res = np.sum((y_lin_t[:20, 0].numpy() - pred_mean) ** 2)
    ss_tot = np.sum((y_lin_t[:20, 0].numpy() - y_lin_t[:20, 0].numpy().mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    results['線形回帰_R2'] = r2
    results['線形回帰_予測std'] = pred_std.mean()
    print(f"\n   R²スコア: {r2:.4f}")
    print(f"   動的選択による予測の標準偏差: {pred_std.mean():.4f}")
    
    # --- 2.2 二次関数回帰 ---
    print("\n🔹 2.2 二次関数回帰 (y = x² + 0.5)")
    print("-" * 40)
    
    model_quad = QBNNBrainTorch(
        num_neurons=50,
        max_input_size=10,
        max_output_size=5,
        connection_density=0.25
    )
    
    X_quad = np.random.uniform(-1, 1, (100, 1))
    y_quad = X_quad ** 2 + 0.5 + np.random.normal(0, 0.05, (100, 1))
    
    # 正規化
    y_quad_norm = (y_quad - y_quad.min()) / (y_quad.max() - y_quad.min())
    
    X_quad_pad = np.pad(X_quad, ((0, 0), (0, 9)), 'constant')
    X_quad_t = torch.tensor(X_quad_pad, dtype=torch.float32)
    y_quad_t = torch.tensor(y_quad_norm, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model_quad.parameters(), lr=0.01)
    
    print("   学習中...")
    for epoch in range(300):
        optimizer.zero_grad()
        pred, _, _ = model_quad(X_quad_t, input_size=1, output_size=1, 
                                time_steps=4, dynamic_selection=False)
        loss = criterion(pred[:, :1], y_quad_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    model_quad.eval()
    with torch.no_grad():
        pred, _, _ = model_quad(X_quad_t[:20], input_size=1, output_size=1,
                                time_steps=4, dynamic_selection=True)
    
    ss_res = np.sum((y_quad_t[:20, 0].numpy() - pred[:, 0].numpy()) ** 2)
    ss_tot = np.sum((y_quad_t[:20, 0].numpy() - y_quad_t[:20, 0].numpy().mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    results['二次関数_R2'] = r2
    print(f"\n   R²スコア: {r2:.4f}")
    
    # --- 2.3 sin関数回帰 ---
    print("\n🔹 2.3 sin関数回帰 (y = sin(πx))")
    print("-" * 40)
    
    model_sin = QBNNBrainTorch(
        num_neurons=60,
        max_input_size=10,
        max_output_size=5,
        connection_density=0.3
    )
    
    X_sin = np.random.uniform(-1, 1, (100, 1))
    y_sin = np.sin(np.pi * X_sin) + np.random.normal(0, 0.05, (100, 1))
    
    # 正規化
    y_sin_norm = (y_sin - y_sin.min()) / (y_sin.max() - y_sin.min())
    
    X_sin_pad = np.pad(X_sin, ((0, 0), (0, 9)), 'constant')
    X_sin_t = torch.tensor(X_sin_pad, dtype=torch.float32)
    y_sin_t = torch.tensor(y_sin_norm, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model_sin.parameters(), lr=0.01)
    
    print("   学習中...")
    for epoch in range(400):
        optimizer.zero_grad()
        pred, _, _ = model_sin(X_sin_t, input_size=1, output_size=1, 
                               time_steps=5, dynamic_selection=False)
        loss = criterion(pred[:, :1], y_sin_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    model_sin.eval()
    with torch.no_grad():
        pred, _, _ = model_sin(X_sin_t[:20], input_size=1, output_size=1,
                               time_steps=5, dynamic_selection=True)
    
    ss_res = np.sum((y_sin_t[:20, 0].numpy() - pred[:, 0].numpy()) ** 2)
    ss_tot = np.sum((y_sin_t[:20, 0].numpy() - y_sin_t[:20, 0].numpy().mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    results['sin関数_R2'] = r2
    print(f"\n   R²スコア: {r2:.4f}")
    
    return results


# ========================================
# 3. 動的入出力の検証
# ========================================

def test_dynamic_io():
    """動的入出力の検証"""
    print("\n" + "=" * 60)
    print("🔄 動的入出力の検証")
    print("=" * 60)
    
    # 純粋Python版
    print("\n🔹 純粋Python版（QBNNBrain）")
    print("-" * 40)
    
    brain = QBNNBrain(num_neurons=50, connection_density=0.15, plasticity=0.1)
    
    inputs = [0.5, -0.3, 0.8, -0.1, 0.6]
    
    # 同じ入力で3回実行
    io_records = []
    for trial in range(3):
        outputs, in_neurons, out_neurons = brain.forward(
            inputs, 
            output_size=3,
            time_steps=3,
            input_type='visual',
            output_type='motor'
        )
        io_records.append({
            'input_neurons': set(in_neurons),
            'output_neurons': set(out_neurons),
            'outputs': outputs
        })
        print(f"\n   試行{trial+1}:")
        print(f"   入力ニューロン: {in_neurons}")
        print(f"   出力ニューロン: {out_neurons}")
        print(f"   出力: {[f'{o:.4f}' for o in outputs]}")
    
    # 入出力ニューロンの変化を確認
    in_changes = 0
    out_changes = 0
    for i in range(1, 3):
        if io_records[i]['input_neurons'] != io_records[i-1]['input_neurons']:
            in_changes += 1
        if io_records[i]['output_neurons'] != io_records[i-1]['output_neurons']:
            out_changes += 1
    
    print(f"\n   📊 動的選択の確認:")
    print(f"   入力ニューロンが変化した回数: {in_changes}/2")
    print(f"   出力ニューロンが変化した回数: {out_changes}/2")
    
    # PyTorch版
    print("\n🔹 PyTorch版（QBNNBrainTorch）")
    print("-" * 40)
    
    model = QBNNBrainTorch(
        num_neurons=40,
        max_input_size=10,
        max_output_size=5,
        connection_density=0.2
    )
    
    x = torch.tensor([[0.5, -0.3, 0.8, -0.1, 0.6]], dtype=torch.float32)
    
    # 同じ入力で3回実行
    io_records_torch = []
    for trial in range(3):
        with torch.no_grad():
            output, in_idx, out_idx = model(x, input_size=5, output_size=3, 
                                            time_steps=3, dynamic_selection=True)
        io_records_torch.append({
            'input_neurons': set(in_idx.tolist()),
            'output_neurons': set(out_idx.tolist()),
        })
        print(f"\n   試行{trial+1}:")
        print(f"   入力ニューロン: {in_idx.tolist()}")
        print(f"   出力ニューロン: {out_idx.tolist()}")
    
    in_changes = 0
    out_changes = 0
    for i in range(1, 3):
        if io_records_torch[i]['input_neurons'] != io_records_torch[i-1]['input_neurons']:
            in_changes += 1
        if io_records_torch[i]['output_neurons'] != io_records_torch[i-1]['output_neurons']:
            out_changes += 1
    
    print(f"\n   📊 動的選択の確認:")
    print(f"   入力ニューロンが変化した回数: {in_changes}/2")
    print(f"   出力ニューロンが変化した回数: {out_changes}/2")
    
    # 量子情報
    info = model.get_quantum_info()
    print(f"\n   ⚛️ 量子情報:")
    print(f"   θ平均: {info['theta_mean']:.4f}")
    print(f"   r平均（相関）: {info['r_mean']:.4f}")
    print(f"   T平均（温度）: {info['T_mean']:.4f}")
    print(f"   λ（もつれ強度）: {info['lambda']:.4f}")
    print(f"   感受性平均: {info['sensitivity_mean']:.4f}")
    print(f"   出力傾向平均: {info['output_tendency_mean']:.4f}")


# ========================================
# 4. 可視化
# ========================================

def visualize_results(math_results: dict, reg_results: dict):
    """結果の可視化"""
    print("\n" + "=" * 60)
    print("📊 結果サマリー")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 数学問題
    ax1 = axes[0]
    ax1.set_title('Math Problem Results', fontsize=12)
    
    labels = ['XOR\nAccuracy', 'Addition\nMAE', 'Multiplication\nMAE']
    values = [
        math_results.get('XOR', 0) * 100,
        (1 - min(math_results.get('加算_MAE', 1), 1)) * 100,
        (1 - min(math_results.get('乗算_MAE', 1), 1)) * 100
    ]
    colors = ['#2ecc71' if v > 50 else '#e74c3c' for v in values]
    
    bars = ax1.bar(labels, values, color=colors, edgecolor='black')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Score (%)')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', fontsize=10)
    
    # 回帰分析
    ax2 = axes[1]
    ax2.set_title('Regression R² Scores', fontsize=12)
    
    labels = ['Linear\ny=2x+1', 'Quadratic\ny=x²+0.5', 'Sine\ny=sin(πx)']
    values = [
        max(0, reg_results.get('線形回帰_R2', 0)) * 100,
        max(0, reg_results.get('二次関数_R2', 0)) * 100,
        max(0, reg_results.get('sin関数_R2', 0)) * 100
    ]
    colors = ['#3498db' if v > 50 else '#e74c3c' for v in values]
    
    bars = ax2.bar(labels, values, color=colors, edgecolor='black')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('R² Score (%)')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/Users/yuyahiguchi/Program/Qubit/qbnn_brain_dynamic_tests.png', 
                dpi=150, bbox_inches='tight')
    print("\n📊 グラフを保存: qbnn_brain_dynamic_tests.png")
    plt.close()
    
    # サマリー表示
    print("\n" + "-" * 40)
    print("📐 数学問題:")
    print(f"   XOR正答率: {math_results.get('XOR', 0)*100:.1f}%")
    print(f"   加算MAE: {math_results.get('加算_MAE', 0):.4f}")
    print(f"   乗算MAE: {math_results.get('乗算_MAE', 0):.4f}")
    
    print("\n📈 回帰分析:")
    print(f"   線形回帰 R²: {reg_results.get('線形回帰_R2', 0):.4f}")
    print(f"   二次関数 R²: {reg_results.get('二次関数_R2', 0):.4f}")
    print(f"   sin関数 R²: {reg_results.get('sin関数_R2', 0):.4f}")


# ========================================
# メイン
# ========================================

if __name__ == '__main__':
    print("=" * 60)
    print("🧠⚛️ QBNN Brain 動的入出力 総合テスト")
    print("=" * 60)
    
    start_time = time.time()
    
    # 動的入出力の検証
    test_dynamic_io()
    
    # 数学問題テスト
    math_results = test_math_problems()
    
    # 回帰分析テスト
    reg_results = test_regression()
    
    # 結果可視化
    visualize_results(math_results, reg_results)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ 総実行時間: {elapsed:.1f}秒")
    print("\n✅ 全テスト完了！")
