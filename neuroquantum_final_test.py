#!/usr/bin/env python3
"""
NeuroQuantum 最終テスト
========================
明確なタスク形式でLLMの能力を評価
"""

import torch
import random
from neuroquantum import NeuroQuantumAI


def run_final_test():
    print("=" * 70)
    print("🧠 NeuroQuantum 最終能力テスト")
    print("=" * 70)
    
    # 明確なフォーマットの学習データ
    training_data = []
    
    # フォーマット: "タスク:入力" → "出力"
    
    # 感情分類（バランス良く）
    positive_words = ["嬉しい", "楽しい", "幸せ", "素晴らしい", "最高", "良い", "好き", "感動", "満足", "素敵"]
    negative_words = ["悲しい", "辛い", "嫌い", "最悪", "怒り", "不満", "残念", "退屈", "つまらない", "がっかり"]
    
    for word in positive_words:
        training_data.append((f"感情:{word}", "プラス"))
    for word in negative_words:
        training_data.append((f"感情:{word}", "マイナス"))
    
    # 果物/野菜
    fruits = ["りんご", "みかん", "バナナ", "いちご", "ぶどう", "メロン", "スイカ", "もも", "梨", "柿"]
    vegetables = ["にんじん", "キャベツ", "トマト", "きゅうり", "なす", "ピーマン", "大根", "白菜", "ほうれん草", "ブロッコリー"]
    
    for f in fruits:
        training_data.append((f"分類:{f}", "フルーツ"))
    for v in vegetables:
        training_data.append((f"分類:{v}", "ヤサイ"))
    
    # 言語判定
    english = ["Hello", "Thank you", "Good morning", "Goodbye", "Please", "Sorry", "Yes", "No", "OK", "Nice"]
    japanese = ["こんにちは", "ありがとう", "おはよう", "さようなら", "お願い", "ごめん", "はい", "いいえ", "大丈夫", "素敵"]
    
    for e in english:
        training_data.append((f"言語:{e}", "英語"))
    for j in japanese:
        training_data.append((f"言語:{j}", "日本語"))
    
    # 奇数/偶数
    for i in range(1, 31):
        result = "奇数" if i % 2 == 1 else "偶数"
        training_data.append((f"判定:{i}", result))
    
    # データを増やす
    training_data = training_data * 5
    random.shuffle(training_data)
    
    print(f"\n📊 学習データ: {len(training_data)} サンプル")
    
    # モデル作成
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
    print("\n📚 学習中...")
    ai.train(training_data, epochs=100, batch_size=32, lr=0.001, seq_len=16)
    
    # テスト
    print("\n" + "=" * 70)
    print("📝 テスト")
    print("=" * 70)
    
    results = {}
    
    # テスト1: 感情分類
    print("\n🎭 感情分類テスト")
    emotion_tests = [
        ("感情:嬉しい", "プラス"),
        ("感情:悲しい", "マイナス"),
        ("感情:楽しい", "プラス"),
        ("感情:辛い", "マイナス"),
        ("感情:幸せ", "プラス"),
        ("感情:怒り", "マイナス"),
        ("感情:好き", "プラス"),
        ("感情:嫌い", "マイナス"),
    ]
    
    correct = 0
    for q, expected in emotion_tests:
        response = ai.generate(q, max_length=8, temp_min=0.1, temp_max=0.3).strip()
        is_correct = expected in response
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"   {status} {q} → {response[:10]} (正解: {expected})")
    results['感情'] = (correct, len(emotion_tests))
    
    # テスト2: 果物/野菜
    print("\n🍎 果物/野菜分類テスト")
    food_tests = [
        ("分類:りんご", "フルーツ"),
        ("分類:にんじん", "ヤサイ"),
        ("分類:バナナ", "フルーツ"),
        ("分類:トマト", "ヤサイ"),
        ("分類:ぶどう", "フルーツ"),
        ("分類:大根", "ヤサイ"),
        ("分類:もも", "フルーツ"),
        ("分類:なす", "ヤサイ"),
    ]
    
    correct = 0
    for q, expected in food_tests:
        response = ai.generate(q, max_length=8, temp_min=0.1, temp_max=0.3).strip()
        is_correct = expected in response
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"   {status} {q} → {response[:10]} (正解: {expected})")
    results['果物野菜'] = (correct, len(food_tests))
    
    # テスト3: 言語判定
    print("\n🌐 言語判定テスト")
    lang_tests = [
        ("言語:Hello", "英語"),
        ("言語:こんにちは", "日本語"),
        ("言語:Thank you", "英語"),
        ("言語:ありがとう", "日本語"),
        ("言語:Please", "英語"),
        ("言語:お願い", "日本語"),
        ("言語:Goodbye", "英語"),
        ("言語:さようなら", "日本語"),
    ]
    
    correct = 0
    for q, expected in lang_tests:
        response = ai.generate(q, max_length=8, temp_min=0.1, temp_max=0.3).strip()
        is_correct = expected in response
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"   {status} {q} → {response[:10]} (正解: {expected})")
    results['言語'] = (correct, len(lang_tests))
    
    # テスト4: 奇数/偶数
    print("\n🔢 奇数/偶数テスト")
    num_tests = [
        ("判定:3", "奇数"),
        ("判定:4", "偶数"),
        ("判定:7", "奇数"),
        ("判定:10", "偶数"),
        ("判定:15", "奇数"),
        ("判定:20", "偶数"),
        ("判定:25", "奇数"),
        ("判定:30", "偶数"),
    ]
    
    correct = 0
    for q, expected in num_tests:
        response = ai.generate(q, max_length=8, temp_min=0.1, temp_max=0.3).strip()
        is_correct = expected in response
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"   {status} {q} → {response[:10]} (正解: {expected})")
    results['奇偶'] = (correct, len(num_tests))
    
    # サマリー
    print("\n" + "=" * 70)
    print("📊 結果サマリー")
    print("=" * 70)
    
    total_correct = 0
    total_questions = 0
    
    print("\n   カテゴリ    | 正解/問題数 | 正答率")
    print("   " + "-" * 45)
    
    for name, (c, t) in results.items():
        acc = c / t * 100
        bar = "█" * int(acc / 10) + "░" * (10 - int(acc / 10))
        print(f"   {name:<10} |   {c}/{t}    | {acc:5.1f}% {bar}")
        total_correct += c
        total_questions += t
    
    overall = total_correct / total_questions * 100
    print("   " + "-" * 45)
    print(f"   {'総合':<10} |   {total_correct}/{total_questions}    | {overall:5.1f}%")
    
    # 評価
    print("\n🏆 評価:")
    if overall >= 80:
        print("   S - 優秀！")
    elif overall >= 60:
        print("   A - 良好")
    elif overall >= 40:
        print("   B - 普通")
    elif overall >= 20:
        print("   C - 要改善")
    else:
        print("   D - 学習が必要")
    
    # QBNN情報
    print("\n⚛️ QBNNもつれ:")
    for info in ai.model.get_quantum_info():
        print(f"   Block {info['block']}: λ = {info['attn_lambda']:.4f}")
    
    return results


if __name__ == '__main__':
    run_final_test()

