#!/usr/bin/env python3
"""
QBNN Brain 回帰分析テスト
========================
脳型散在QBNNによる各種回帰タスク
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import math

from qbnn_brain import QBNNBrain, QBNNBrainTorch


def test_linear_regression():
    """線形回帰テスト"""
    print("\n" + "=" * 60)
    print("📈 線形回帰テスト: y = 2x + 1")
    print("=" * 60)
    
    # データ生成
    np.random.seed(42)
    X_np = np.linspace(-1, 1, 50).reshape(-1, 1)
    y_np = 2 * X_np + 1 + np.random.randn(50, 1) * 0.1
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    # モデル
    model = QBNNBrainTorch(
        num_neurons=25,
        input_size=1,
        output_size=1,
        connection_density=0.25
    )
    
    # 学習
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    
    print("\n📚 学習中...")
    for epoch in range(150):
        optimizer.zero_grad()
        pred = model(X, time_steps=2)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 30 == 0:
            print(f"   Epoch {epoch+1}: MSE = {loss.item():.4f}")
    
    # 評価
    model.eval()
    with torch.no_grad():
        pred = model(X, time_steps=2)
        mse = criterion(pred, y).item()
        
        # R²スコア
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n🎯 結果:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²スコア: {r2:.4f}")
    
    return mse, r2


def test_quadratic_regression():
    """二次関数回帰テスト"""
    print("\n" + "=" * 60)
    print("📈 二次関数回帰テスト: y = x² - 0.5x + 0.2")
    print("=" * 60)
    
    # データ生成
    np.random.seed(42)
    X_np = np.linspace(-1, 1, 50).reshape(-1, 1)
    y_np = X_np**2 - 0.5*X_np + 0.2 + np.random.randn(50, 1) * 0.05
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    # モデル
    model = QBNNBrainTorch(
        num_neurons=35,
        input_size=1,
        output_size=1,
        connection_density=0.3
    )
    
    # 学習
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    
    print("\n📚 学習中...")
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X, time_steps=3)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 40 == 0:
            print(f"   Epoch {epoch+1}: MSE = {loss.item():.4f}")
    
    # 評価
    model.eval()
    with torch.no_grad():
        pred = model(X, time_steps=3)
        mse = criterion(pred, y).item()
        
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n🎯 結果:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²スコア: {r2:.4f}")
    
    return mse, r2


def test_sin_regression():
    """sin関数回帰テスト"""
    print("\n" + "=" * 60)
    print("📈 sin関数回帰テスト: y = sin(πx)")
    print("=" * 60)
    
    # データ生成
    np.random.seed(42)
    X_np = np.linspace(-1, 1, 60).reshape(-1, 1)
    y_np = np.sin(np.pi * X_np) + np.random.randn(60, 1) * 0.05
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    # モデル
    model = QBNNBrainTorch(
        num_neurons=40,
        input_size=1,
        output_size=1,
        connection_density=0.25
    )
    
    # 学習
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    criterion = nn.MSELoss()
    
    print("\n📚 学習中...")
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X, time_steps=3)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 40 == 0:
            print(f"   Epoch {epoch+1}: MSE = {loss.item():.4f}")
    
    # 評価
    model.eval()
    with torch.no_grad():
        pred = model(X, time_steps=3)
        mse = criterion(pred, y).item()
        
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n🎯 結果:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²スコア: {r2:.4f}")
    
    return mse, r2


def test_multivariate_regression():
    """多変量回帰テスト"""
    print("\n" + "=" * 60)
    print("📈 多変量回帰テスト: y = 0.5x₁ + 0.3x₂ - 0.2x₃ + 0.1")
    print("=" * 60)
    
    # データ生成
    np.random.seed(42)
    n_samples = 80
    X_np = np.random.randn(n_samples, 3)
    y_np = (0.5 * X_np[:, 0] + 0.3 * X_np[:, 1] - 0.2 * X_np[:, 2] + 0.1 
            + np.random.randn(n_samples) * 0.1).reshape(-1, 1)
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    # モデル
    model = QBNNBrainTorch(
        num_neurons=30,
        input_size=3,
        output_size=1,
        connection_density=0.25
    )
    
    # 学習
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    
    print("\n📚 学習中...")
    for epoch in range(150):
        optimizer.zero_grad()
        pred = model(X, time_steps=3)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 30 == 0:
            print(f"   Epoch {epoch+1}: MSE = {loss.item():.4f}")
    
    # 評価
    model.eval()
    with torch.no_grad():
        pred = model(X, time_steps=3)
        mse = criterion(pred, y).item()
        
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n🎯 結果:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²スコア: {r2:.4f}")
    
    return mse, r2


def test_polynomial_regression():
    """多項式回帰テスト"""
    print("\n" + "=" * 60)
    print("📈 多項式回帰テスト: y = x³ - x² + 0.5x - 0.3")
    print("=" * 60)
    
    # データ生成
    np.random.seed(42)
    X_np = np.linspace(-1, 1, 50).reshape(-1, 1)
    y_np = X_np**3 - X_np**2 + 0.5*X_np - 0.3 + np.random.randn(50, 1) * 0.05
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    # モデル（より複雑な関数に対応するため大きめ）
    model = QBNNBrainTorch(
        num_neurons=45,
        input_size=1,
        output_size=1,
        connection_density=0.3
    )
    
    # 学習
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    criterion = nn.MSELoss()
    
    print("\n📚 学習中...")
    for epoch in range(250):
        optimizer.zero_grad()
        pred = model(X, time_steps=4)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}: MSE = {loss.item():.4f}")
    
    # 評価
    model.eval()
    with torch.no_grad():
        pred = model(X, time_steps=4)
        mse = criterion(pred, y).item()
        
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n🎯 結果:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²スコア: {r2:.4f}")
    
    return mse, r2


def test_exponential_regression():
    """指数関数回帰テスト"""
    print("\n" + "=" * 60)
    print("📈 指数関数回帰テスト: y = exp(x) / 3")
    print("=" * 60)
    
    # データ生成
    np.random.seed(42)
    X_np = np.linspace(-1, 1, 50).reshape(-1, 1)
    y_np = np.exp(X_np) / 3 + np.random.randn(50, 1) * 0.05
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    # モデル
    model = QBNNBrainTorch(
        num_neurons=35,
        input_size=1,
        output_size=1,
        connection_density=0.25
    )
    
    # 学習
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    
    print("\n📚 学習中...")
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X, time_steps=3)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 40 == 0:
            print(f"   Epoch {epoch+1}: MSE = {loss.item():.4f}")
    
    # 評価
    model.eval()
    with torch.no_grad():
        pred = model(X, time_steps=3)
        mse = criterion(pred, y).item()
        
        ss_res = ((y - pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n🎯 結果:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²スコア: {r2:.4f}")
    
    return mse, r2


def test_multi_output_regression():
    """多出力回帰テスト"""
    print("\n" + "=" * 60)
    print("📈 多出力回帰テスト: y₁ = sin(x), y₂ = cos(x)")
    print("=" * 60)
    
    # データ生成
    np.random.seed(42)
    X_np = np.linspace(-np.pi, np.pi, 60).reshape(-1, 1)
    y_np = np.hstack([
        np.sin(X_np) + np.random.randn(60, 1) * 0.05,
        np.cos(X_np) + np.random.randn(60, 1) * 0.05
    ])
    
    X = torch.tensor(X_np / np.pi, dtype=torch.float32)  # 正規化
    y = torch.tensor(y_np, dtype=torch.float32)
    
    # モデル
    model = QBNNBrainTorch(
        num_neurons=45,
        input_size=1,
        output_size=2,
        connection_density=0.3
    )
    
    # 学習
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    criterion = nn.MSELoss()
    
    print("\n📚 学習中...")
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X, time_steps=3)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 40 == 0:
            print(f"   Epoch {epoch+1}: MSE = {loss.item():.4f}")
    
    # 評価
    model.eval()
    with torch.no_grad():
        pred = model(X, time_steps=3)
        mse = criterion(pred, y).item()
        
        # 各出力のR²
        r2_list = []
        for i in range(2):
            ss_res = ((y[:, i] - pred[:, i]) ** 2).sum().item()
            ss_tot = ((y[:, i] - y[:, i].mean()) ** 2).sum().item()
            r2_list.append(1 - (ss_res / ss_tot))
    
    print(f"\n🎯 結果:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²(sin): {r2_list[0]:.4f}")
    print(f"   R²(cos): {r2_list[1]:.4f}")
    print(f"   R²平均: {sum(r2_list)/2:.4f}")
    
    return mse, sum(r2_list)/2


def analyze_quantum_contribution():
    """量子もつれの寄与分析"""
    print("\n" + "=" * 60)
    print("⚛️ 量子もつれの寄与分析")
    print("=" * 60)
    
    # 同じデータで比較
    np.random.seed(42)
    X_np = np.linspace(-1, 1, 50).reshape(-1, 1)
    y_np = np.sin(np.pi * X_np) + np.random.randn(50, 1) * 0.05
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    results = []
    
    for lambda_val in [0.0, 0.1, 0.25, 0.5, 1.0]:
        print(f"\n📌 λ (もつれ強度) = {lambda_val}")
        
        # モデル
        model = QBNNBrainTorch(
            num_neurons=35,
            input_size=1,
            output_size=1,
            connection_density=0.25
        )
        
        # λを固定
        with torch.no_grad():
            model.lambda_entangle.fill_(lambda_val)
        
        # 学習
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.MSELoss()
        
        for epoch in range(150):
            optimizer.zero_grad()
            pred = model(X, time_steps=3)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            # λを固定に保つ
            with torch.no_grad():
                model.lambda_entangle.fill_(lambda_val)
        
        # 評価
        model.eval()
        with torch.no_grad():
            pred = model(X, time_steps=3)
            mse = criterion(pred, y).item()
            
            ss_res = ((y - pred) ** 2).sum().item()
            ss_tot = ((y - y.mean()) ** 2).sum().item()
            r2 = 1 - (ss_res / ss_tot)
        
        print(f"   MSE: {mse:.4f}, R²: {r2:.4f}")
        results.append((lambda_val, mse, r2))
    
    print("\n📊 量子もつれ効果まとめ:")
    print("   λ      | MSE     | R²")
    print("   -------|---------|-------")
    for lam, mse, r2 in results:
        print(f"   {lam:.2f}   | {mse:.4f}  | {r2:.4f}")
    
    return results


def run_all_regression_tests():
    """全回帰テストを実行"""
    print("=" * 60)
    print("🧠 QBNN Brain - 回帰分析テスト")
    print("=" * 60)
    
    results = {}
    
    # 各テスト実行
    mse, r2 = test_linear_regression()
    results['線形'] = {'MSE': mse, 'R²': r2}
    
    mse, r2 = test_quadratic_regression()
    results['二次関数'] = {'MSE': mse, 'R²': r2}
    
    mse, r2 = test_sin_regression()
    results['sin関数'] = {'MSE': mse, 'R²': r2}
    
    mse, r2 = test_multivariate_regression()
    results['多変量'] = {'MSE': mse, 'R²': r2}
    
    mse, r2 = test_polynomial_regression()
    results['多項式'] = {'MSE': mse, 'R²': r2}
    
    mse, r2 = test_exponential_regression()
    results['指数関数'] = {'MSE': mse, 'R²': r2}
    
    mse, r2 = test_multi_output_regression()
    results['多出力'] = {'MSE': mse, 'R²': r2}
    
    # 量子もつれ分析
    quantum_results = analyze_quantum_contribution()
    
    # サマリー
    print("\n" + "=" * 60)
    print("📊 回帰分析テスト結果サマリー")
    print("=" * 60)
    
    print("""
┌────────────────────────────────────────────────────┐
│           QBNN Brain 回帰分析結果                    │
├────────────────────────────────────────────────────┤""")
    
    for name, res in results.items():
        r2_stars = "⭐" * min(5, int(res['R²'] * 5 + 0.5))
        print(f"│ {name:10} | MSE: {res['MSE']:.4f} | R²: {res['R²']:.4f} {r2_stars:5}")
    
    print("""├────────────────────────────────────────────────────┤
│               量子もつれ効果分析                      │
├────────────────────────────────────────────────────┤""")
    
    best_lambda = max(quantum_results, key=lambda x: x[2])
    print(f"│ 最適λ: {best_lambda[0]:.2f} (R²={best_lambda[2]:.4f})              │")
    print("└────────────────────────────────────────────────────┘")
    
    # 平均R²
    avg_r2 = sum(r['R²'] for r in results.values()) / len(results)
    print(f"\n✅ 平均R²スコア: {avg_r2:.4f}")
    
    # 評価
    if avg_r2 >= 0.9:
        grade = "A (優秀)"
    elif avg_r2 >= 0.8:
        grade = "B (良好)"
    elif avg_r2 >= 0.7:
        grade = "C (普通)"
    else:
        grade = "D (要改善)"
    
    print(f"📝 総合評価: {grade}")
    
    return results


if __name__ == '__main__':
    run_all_regression_tests()

