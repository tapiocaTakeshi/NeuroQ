#!/usr/bin/env python3
"""
NeuroQuantum 数学能力テスト
============================
QBNNベースのLLMがどの程度数学問題を解けるかを検証
"""

import torch
import random
import re
from typing import List, Tuple, Dict
from neuroquantum import NeuroQuantumAI


def generate_arithmetic_data() -> List[Tuple[str, str]]:
    """四則演算の学習データを生成"""
    data = []
    
    # 足し算
    for a in range(1, 20):
        for b in range(1, 20):
            data.append((f"{a}+{b}=", f"{a+b}"))
            data.append((f"{a}と{b}の和は", f"{a+b}"))
    
    # 引き算
    for a in range(5, 30):
        for b in range(1, min(a, 15)):
            data.append((f"{a}-{b}=", f"{a-b}"))
            data.append((f"{a}から{b}を引くと", f"{a-b}"))
    
    # 掛け算
    for a in range(1, 13):
        for b in range(1, 13):
            data.append((f"{a}*{b}=", f"{a*b}"))
            data.append((f"{a}×{b}=", f"{a*b}"))
            data.append((f"{a}かける{b}は", f"{a*b}"))
    
    # 割り算（割り切れるもの）
    for a in range(1, 13):
        for b in range(1, 13):
            result = a * b
            data.append((f"{result}/{a}=", f"{b}"))
            data.append((f"{result}÷{a}=", f"{b}"))
    
    return data


def generate_sequence_data() -> List[Tuple[str, str]]:
    """数列の学習データ"""
    data = []
    
    # 等差数列
    for start in range(1, 10):
        for diff in range(1, 5):
            seq = [start + diff * i for i in range(5)]
            data.append((f"数列{seq[0]},{seq[1]},{seq[2]},{seq[3]}の次は", f"{seq[4]}"))
    
    # 等比数列（簡単なもの）
    for start in [1, 2]:
        for ratio in [2, 3]:
            seq = [start * (ratio ** i) for i in range(5)]
            if seq[4] <= 200:
                data.append((f"数列{seq[0]},{seq[1]},{seq[2]},{seq[3]}の次は", f"{seq[4]}"))
    
    # フィボナッチ風
    data.append(("数列1,1,2,3,5の次は", "8"))
    data.append(("数列1,1,2,3,5,8の次は", "13"))
    data.append(("数列2,2,4,6,10の次は", "16"))
    
    return data


def generate_algebra_data() -> List[Tuple[str, str]]:
    """簡単な代数の学習データ"""
    data = []
    
    # 方程式 x + a = b
    for a in range(1, 15):
        for x in range(1, 15):
            b = x + a
            data.append((f"x+{a}={b}のとき、x=", f"{x}"))
    
    # 方程式 x - a = b
    for a in range(1, 10):
        for x in range(a+1, 20):
            b = x - a
            data.append((f"x-{a}={b}のとき、x=", f"{x}"))
    
    # 方程式 a*x = b
    for a in range(2, 10):
        for x in range(1, 10):
            b = a * x
            data.append((f"{a}x={b}のとき、x=", f"{x}"))
    
    return data


def generate_comparison_data() -> List[Tuple[str, str]]:
    """比較の学習データ"""
    data = []
    
    for a in range(1, 30):
        for b in range(1, 30):
            if a > b:
                data.append((f"{a}と{b}、大きいのは", f"{a}"))
            elif a < b:
                data.append((f"{a}と{b}、大きいのは", f"{b}"))
    
    return data


def generate_word_problem_data() -> List[Tuple[str, str]]:
    """文章題の学習データ"""
    data = []
    
    # 足し算の文章題
    for a in range(1, 15):
        for b in range(1, 15):
            data.append((f"りんごが{a}個とみかんが{b}個あります。合計は", f"{a+b}個"))
            data.append((f"{a}人と{b}人で合わせて何人", f"{a+b}人"))
    
    # 引き算の文章題
    for a in range(10, 25):
        for b in range(1, min(a, 10)):
            data.append((f"{a}個のうち{b}個使うと残りは", f"{a-b}個"))
    
    # 掛け算の文章題
    for a in range(2, 10):
        for b in range(2, 10):
            data.append((f"1箱{a}個入りが{b}箱あると全部で", f"{a*b}個"))
    
    return data


def extract_number(text: str) -> str:
    """テキストから数値を抽出"""
    # 数字のみを抽出
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return numbers[0]
    return text.strip()


def run_math_test():
    """数学テストを実行"""
    print("=" * 70)
    print("🧮 NeuroQuantum 数学能力テスト")
    print("=" * 70)
    
    # 学習データ生成
    print("\n📊 学習データ生成中...")
    
    arithmetic = generate_arithmetic_data()
    sequences = generate_sequence_data()
    algebra = generate_algebra_data()
    comparison = generate_comparison_data()
    word_problems = generate_word_problem_data()
    
    # 各カテゴリからサンプリング
    all_data = []
    all_data.extend(random.sample(arithmetic, min(500, len(arithmetic))))
    all_data.extend(random.sample(sequences, min(100, len(sequences))))
    all_data.extend(random.sample(algebra, min(300, len(algebra))))
    all_data.extend(random.sample(comparison, min(200, len(comparison))))
    all_data.extend(random.sample(word_problems, min(300, len(word_problems))))
    
    random.shuffle(all_data)
    
    print(f"   四則演算: {len(arithmetic)} 問")
    print(f"   数列: {len(sequences)} 問")
    print(f"   代数: {len(algebra)} 問")
    print(f"   比較: {len(comparison)} 問")
    print(f"   文章題: {len(word_problems)} 問")
    print(f"   学習用: {len(all_data)} 問")
    
    # モデル作成と学習
    print("\n🧠 NeuroQuantum モデル構築...")
    ai = NeuroQuantumAI(
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=64,
        dropout=0.1,
        lambda_entangle=0.35,
    )
    
    print("\n📚 学習中...")
    ai.train(all_data, epochs=50, batch_size=32, lr=0.001, seq_len=32)
    
    # テスト
    print("\n" + "=" * 70)
    print("📝 テスト開始")
    print("=" * 70)
    
    results = {
        'arithmetic': {'correct': 0, 'total': 0, 'examples': []},
        'sequence': {'correct': 0, 'total': 0, 'examples': []},
        'algebra': {'correct': 0, 'total': 0, 'examples': []},
        'comparison': {'correct': 0, 'total': 0, 'examples': []},
        'word_problem': {'correct': 0, 'total': 0, 'examples': []},
    }
    
    # 四則演算テスト
    print("\n🔢 テスト1: 四則演算")
    test_arithmetic = [
        ("3+5=", "8"),
        ("12+7=", "19"),
        ("15-6=", "9"),
        ("20-8=", "12"),
        ("4*7=", "28"),
        ("9*6=", "54"),
        ("36/6=", "6"),
        ("48/8=", "6"),
        ("7+8=", "15"),
        ("11-3=", "8"),
        ("5*9=", "45"),
        ("72/9=", "8"),
        ("25+13=", "38"),
        ("30-17=", "13"),
        ("8*8=", "64"),
    ]
    
    for question, expected in test_arithmetic:
        response = ai.generate(question, max_length=10, temp_min=0.2, temp_max=0.4)
        predicted = extract_number(response)
        correct = predicted == expected
        results['arithmetic']['total'] += 1
        if correct:
            results['arithmetic']['correct'] += 1
        results['arithmetic']['examples'].append({
            'question': question,
            'expected': expected,
            'predicted': predicted,
            'response': response[:30],
            'correct': correct
        })
        status = "✅" if correct else "❌"
        print(f"   {status} {question} → 予測: {predicted}, 正解: {expected}")
    
    # 数列テスト
    print("\n📈 テスト2: 数列")
    test_sequences = [
        ("数列2,4,6,8の次は", "10"),
        ("数列1,3,5,7の次は", "9"),
        ("数列5,10,15,20の次は", "25"),
        ("数列1,2,4,8の次は", "16"),
        ("数列3,6,9,12の次は", "15"),
        ("数列1,1,2,3,5の次は", "8"),
        ("数列10,20,30,40の次は", "50"),
        ("数列2,5,8,11の次は", "14"),
    ]
    
    for question, expected in test_sequences:
        response = ai.generate(question, max_length=10, temp_min=0.2, temp_max=0.4)
        predicted = extract_number(response)
        correct = predicted == expected
        results['sequence']['total'] += 1
        if correct:
            results['sequence']['correct'] += 1
        results['sequence']['examples'].append({
            'question': question,
            'expected': expected,
            'predicted': predicted,
            'correct': correct
        })
        status = "✅" if correct else "❌"
        print(f"   {status} {question} → 予測: {predicted}, 正解: {expected}")
    
    # 代数テスト
    print("\n🔤 テスト3: 代数（方程式）")
    test_algebra = [
        ("x+5=12のとき、x=", "7"),
        ("x+3=10のとき、x=", "7"),
        ("x-4=6のとき、x=", "10"),
        ("2x=14のとき、x=", "7"),
        ("3x=15のとき、x=", "5"),
        ("x+8=15のとき、x=", "7"),
        ("5x=25のとき、x=", "5"),
        ("x-7=3のとき、x=", "10"),
    ]
    
    for question, expected in test_algebra:
        response = ai.generate(question, max_length=10, temp_min=0.2, temp_max=0.4)
        predicted = extract_number(response)
        correct = predicted == expected
        results['algebra']['total'] += 1
        if correct:
            results['algebra']['correct'] += 1
        results['algebra']['examples'].append({
            'question': question,
            'expected': expected,
            'predicted': predicted,
            'correct': correct
        })
        status = "✅" if correct else "❌"
        print(f"   {status} {question} → 予測: {predicted}, 正解: {expected}")
    
    # 比較テスト
    print("\n⚖️ テスト4: 比較")
    test_comparison = [
        ("15と8、大きいのは", "15"),
        ("7と12、大きいのは", "12"),
        ("25と19、大きいのは", "25"),
        ("3と9、大きいのは", "9"),
        ("100と50、大きいのは", "100"),
    ]
    
    for question, expected in test_comparison:
        response = ai.generate(question, max_length=10, temp_min=0.2, temp_max=0.4)
        predicted = extract_number(response)
        correct = predicted == expected
        results['comparison']['total'] += 1
        if correct:
            results['comparison']['correct'] += 1
        results['comparison']['examples'].append({
            'question': question,
            'expected': expected,
            'predicted': predicted,
            'correct': correct
        })
        status = "✅" if correct else "❌"
        print(f"   {status} {question} → 予測: {predicted}, 正解: {expected}")
    
    # 文章題テスト
    print("\n📖 テスト5: 文章題")
    test_word_problems = [
        ("りんごが5個とみかんが3個あります。合計は", "8個"),
        ("10個のうち4個使うと残りは", "6個"),
        ("1箱6個入りが3箱あると全部で", "18個"),
        ("7人と5人で合わせて何人", "12人"),
        ("15個のうち7個使うと残りは", "8個"),
        ("1箱4個入りが5箱あると全部で", "20個"),
    ]
    
    for question, expected in test_word_problems:
        response = ai.generate(question, max_length=15, temp_min=0.2, temp_max=0.4)
        predicted_num = extract_number(response)
        expected_num = extract_number(expected)
        correct = predicted_num == expected_num
        results['word_problem']['total'] += 1
        if correct:
            results['word_problem']['correct'] += 1
        results['word_problem']['examples'].append({
            'question': question,
            'expected': expected,
            'predicted': response[:20],
            'correct': correct
        })
        status = "✅" if correct else "❌"
        print(f"   {status} {question} → 予測: {response[:15]}, 正解: {expected}")
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("📊 テスト結果サマリー")
    print("=" * 70)
    
    total_correct = 0
    total_questions = 0
    
    categories = [
        ('四則演算', 'arithmetic'),
        ('数列', 'sequence'),
        ('代数', 'algebra'),
        ('比較', 'comparison'),
        ('文章題', 'word_problem'),
    ]
    
    print("\n   カテゴリ       | 正解 / 問題数 | 正答率")
    print("   " + "-" * 50)
    
    for name, key in categories:
        r = results[key]
        accuracy = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
        total_correct += r['correct']
        total_questions += r['total']
        bar = "█" * int(accuracy / 10) + "░" * (10 - int(accuracy / 10))
        print(f"   {name:<10} |   {r['correct']:2d} / {r['total']:2d}    | {accuracy:5.1f}% {bar}")
    
    overall = total_correct / total_questions * 100 if total_questions > 0 else 0
    print("   " + "-" * 50)
    print(f"   {'総合':<10} |   {total_correct:2d} / {total_questions:2d}    | {overall:5.1f}%")
    
    # 評価
    print("\n" + "=" * 70)
    print("🏆 評価")
    print("=" * 70)
    
    if overall >= 80:
        grade = "S - 優秀！"
    elif overall >= 60:
        grade = "A - 良好"
    elif overall >= 40:
        grade = "B - 普通"
    elif overall >= 20:
        grade = "C - 要改善"
    else:
        grade = "D - 学習が必要"
    
    print(f"\n   総合評価: {grade}")
    print(f"   正答率: {overall:.1f}%")
    
    # QBNN特性との関連
    print("\n⚛️ QBNNの量子もつれ情報:")
    for info in ai.model.get_quantum_info():
        print(f"   Block {info['block']}: λ_attn = {info['attn_lambda']:.4f}")
    
    return results


if __name__ == '__main__':
    results = run_math_test()

