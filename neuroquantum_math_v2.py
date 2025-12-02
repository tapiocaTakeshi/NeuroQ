#!/usr/bin/env python3
"""
NeuroQuantum 数学能力テスト v2
==============================
改善版: より明確なフォーマットで学習
"""

import torch
import random
import re
from typing import List, Tuple
from neuroquantum import NeuroQuantumAI


def generate_math_data() -> List[Tuple[str, str]]:
    """数学学習データ（改善フォーマット）"""
    data = []
    
    # 足し算（明確なフォーマット）
    for a in range(1, 15):
        for b in range(1, 15):
            data.append((f"計算:{a}たす{b}は", f"答え:{a+b}"))
            data.append((f"問題:{a}+{b}の答え", f"={a+b}"))
    
    # 引き算
    for a in range(5, 20):
        for b in range(1, min(a, 10)):
            data.append((f"計算:{a}ひく{b}は", f"答え:{a-b}"))
            data.append((f"問題:{a}-{b}の答え", f"={a-b}"))
    
    # 掛け算（九九）
    for a in range(1, 10):
        for b in range(1, 10):
            data.append((f"計算:{a}かける{b}は", f"答え:{a*b}"))
            data.append((f"問題:{a}×{b}の答え", f"={a*b}"))
    
    # 割り算
    for a in range(1, 10):
        for b in range(1, 10):
            result = a * b
            data.append((f"計算:{result}わる{a}は", f"答え:{b}"))
            data.append((f"問題:{result}÷{a}の答え", f"={b}"))
    
    # 直接的な答え形式
    for a in range(1, 12):
        for b in range(1, 12):
            data.append((f"{a}+{b}=?", f"{a+b}"))
            if a > b:
                data.append((f"{a}-{b}=?", f"{a-b}"))
            data.append((f"{a}*{b}=?", f"{a*b}"))
    
    return data


def extract_answer(text: str) -> str:
    """回答から数値を抽出"""
    # "答え:" や "=" の後の数字
    match = re.search(r'[答え:=]\s*(-?\d+)', text)
    if match:
        return match.group(1)
    # 最初の数字
    match = re.search(r'^(-?\d+)', text.strip())
    if match:
        return match.group(1)
    return text.strip()


def run_test():
    print("=" * 70)
    print("🧮 NeuroQuantum 数学テスト v2（改善版）")
    print("=" * 70)
    
    # データ生成
    print("\n📊 学習データ生成...")
    data = generate_math_data()
    print(f"   総データ数: {len(data)}")
    
    # シャッフル
    random.shuffle(data)
    training_data = data[:2000]  # 最大2000問
    
    # モデル作成
    print("\n🧠 モデル構築...")
    ai = NeuroQuantumAI(
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=32,
        dropout=0.1,
        lambda_entangle=0.35,
    )
    
    # 学習
    print("\n📚 学習中（100エポック）...")
    ai.train(training_data, epochs=100, batch_size=32, lr=0.002, seq_len=20)
    
    # テスト
    print("\n" + "=" * 70)
    print("📝 テスト")
    print("=" * 70)
    
    # テストケース（学習データと同じフォーマット）
    tests = {
        "足し算": [
            ("3+5=?", "8"),
            ("7+4=?", "11"),
            ("9+6=?", "15"),
            ("2+8=?", "10"),
            ("5+5=?", "10"),
            ("計算:6たす3は", "答え:9"),
            ("計算:8たす2は", "答え:10"),
            ("問題:4+7の答え", "=11"),
        ],
        "引き算": [
            ("10-3=?", "7"),
            ("15-7=?", "8"),
            ("12-5=?", "7"),
            ("計算:9ひく4は", "答え:5"),
            ("計算:11ひく6は", "答え:5"),
        ],
        "掛け算": [
            ("3*4=?", "12"),
            ("5*6=?", "30"),
            ("7*8=?", "56"),
            ("計算:4かける5は", "答え:20"),
            ("問題:6×7の答え", "=42"),
        ],
    }
    
    results = {}
    total_correct = 0
    total_questions = 0
    
    for category, test_cases in tests.items():
        print(f"\n📌 {category}")
        correct = 0
        
        for question, expected in test_cases:
            response = ai.generate(question, max_length=15, temp_min=0.1, temp_max=0.3)
            
            # 回答抽出
            predicted = extract_answer(response)
            expected_num = extract_answer(expected)
            
            is_correct = predicted == expected_num
            if is_correct:
                correct += 1
                total_correct += 1
            total_questions += 1
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} {question}")
            print(f"      予測: {response[:25]} → 抽出: {predicted}")
            print(f"      正解: {expected_num}")
        
        results[category] = {
            'correct': correct,
            'total': len(test_cases),
            'accuracy': correct / len(test_cases) * 100
        }
    
    # サマリー
    print("\n" + "=" * 70)
    print("📊 結果サマリー")
    print("=" * 70)
    
    print("\n   カテゴリ    | 正解率")
    print("   " + "-" * 35)
    for cat, r in results.items():
        bar = "█" * int(r['accuracy'] / 10) + "░" * (10 - int(r['accuracy'] / 10))
        print(f"   {cat:<8} | {r['correct']}/{r['total']} ({r['accuracy']:5.1f}%) {bar}")
    
    overall = total_correct / total_questions * 100
    print("   " + "-" * 35)
    print(f"   {'総合':<8} | {total_correct}/{total_questions} ({overall:5.1f}%)")
    
    # 量子もつれ情報
    print("\n⚛️ QBNNもつれ情報:")
    for info in ai.model.get_quantum_info():
        print(f"   Block {info['block']}: λ = {info['attn_lambda']:.4f}")
    
    return results


if __name__ == '__main__':
    run_test()

