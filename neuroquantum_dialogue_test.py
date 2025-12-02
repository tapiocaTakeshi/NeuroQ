#!/usr/bin/env python3
"""
NeuroQuantum 対話生成テスト
============================
LLMの本来の用途である「対話生成」能力を評価
"""

import torch
from neuroquantum import NeuroQuantumAI


def run_dialogue_test():
    print("=" * 70)
    print("💬 NeuroQuantum 対話生成テスト")
    print("=" * 70)
    print("\nLLMの本来の用途「対話生成」をテスト")
    
    # 対話学習データ
    training_data = [
        # 挨拶
        ("こんにちは", "こんにちは！何かお手伝いできますか？"),
        ("おはよう", "おはようございます！今日も良い一日を！"),
        ("こんばんは", "こんばんは！お疲れ様です。"),
        ("ありがとう", "どういたしまして！他に何かありますか？"),
        ("さようなら", "さようなら！またお会いしましょう！"),
        ("はじめまして", "はじめまして！NeuroQuantumです。よろしくお願いします！"),
        
        # 自己紹介
        ("あなたは誰", "私はNeuroQuantumです。QBNNベースのAIアシスタントです。"),
        ("名前は", "NeuroQuantumと申します。量子ニューラルネットワークで動いています。"),
        ("何ができる", "対話、質問への回答、情報提供などができます。"),
        
        # 質問応答
        ("今日の天気は", "申し訳ありませんが、天気情報は取得できません。"),
        ("何時ですか", "現在の時刻は取得できませんが、お手伝いします。"),
        ("お腹すいた", "何か食べるものを探しましょうか？"),
        ("眠い", "休憩をとることをおすすめします。"),
        ("暇だ", "何か面白いことを一緒に考えましょうか？"),
        
        # 技術的な質問
        ("AIとは", "AIは人工知能の略で、コンピュータが知的な処理を行う技術です。"),
        ("プログラミングとは", "プログラミングはコンピュータに指示を与えるコードを書くことです。"),
        ("量子コンピュータとは", "量子コンピュータは量子力学を利用した次世代の計算機です。"),
        ("機械学習とは", "機械学習はデータから学習して予測を行うAI技術です。"),
        ("ニューラルネットワークとは", "ニューラルネットワークは脳の神経回路を模した計算モデルです。"),
        
        # 英語
        ("Hello", "Hello! How can I help you today?"),
        ("Thank you", "You're welcome! Let me know if you need anything else."),
        ("Goodbye", "Goodbye! Have a great day!"),
        ("What is your name", "I'm NeuroQuantum, a QBNN-based AI assistant."),
        ("How are you", "I'm doing well, thank you for asking!"),
        
        # 雑談
        ("好きな食べ物は", "私はAIなので食べませんが、皆さんの好みを聞くのは楽しいです！"),
        ("趣味は", "対話をすることと学習することが私の趣味です！"),
        ("面白いこと教えて", "量子コンピュータは同時に0と1の状態を持てるんですよ！"),
        ("冗談言って", "なぜプログラマーは眼鏡をかけるのでしょう？C#が見えないからです！"),
    ]
    
    # データを増やす
    training_data = training_data * 10
    
    print(f"\n📊 学習データ: {len(training_data)} サンプル")
    
    # モデル作成
    ai = NeuroQuantumAI(
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=64,
        dropout=0.1,
        lambda_entangle=0.35,
    )
    
    # 学習
    print("\n📚 学習中...")
    ai.train(training_data, epochs=80, batch_size=16, lr=0.001, seq_len=48)
    
    # テスト
    print("\n" + "=" * 70)
    print("📝 対話テスト")
    print("=" * 70)
    
    test_prompts = [
        # 学習データにあるもの
        "こんにちは",
        "ありがとう",
        "Hello",
        "AIとは",
        
        # 類似だが違う
        "やあ",
        "サンキュー",
        "Hi",
        "人工知能って何",
        
        # 新しい質問
        "調子どう",
        "何してる",
        "教えて",
    ]
    
    print("\n🗣️ 対話生成結果:\n")
    
    results = []
    for prompt in test_prompts:
        print(f"👤 User: {prompt}")
        response = ai.generate(prompt, max_length=50, temp_min=0.3, temp_max=0.6)
        print(f"🤖 NeuroQuantum: {response}")
        print()
        
        # 応答の質を簡易評価
        quality = {
            'length': len(response),
            'has_content': len(response) > 3,
            'not_repetitive': len(set(response)) > len(response) * 0.3,
        }
        results.append({'prompt': prompt, 'response': response, 'quality': quality})
    
    # 評価
    print("=" * 70)
    print("📊 生成品質評価")
    print("=" * 70)
    
    valid_responses = sum(1 for r in results if r['quality']['has_content'])
    non_repetitive = sum(1 for r in results if r['quality']['not_repetitive'])
    avg_length = sum(r['quality']['length'] for r in results) / len(results)
    
    print(f"\n   有効な応答: {valid_responses}/{len(results)} ({valid_responses/len(results)*100:.1f}%)")
    print(f"   非反復的応答: {non_repetitive}/{len(results)} ({non_repetitive/len(results)*100:.1f}%)")
    print(f"   平均応答長: {avg_length:.1f} 文字")
    
    # 総合評価
    score = (valid_responses / len(results) * 50) + (non_repetitive / len(results) * 30) + min(avg_length / 50 * 20, 20)
    
    print(f"\n🏆 総合スコア: {score:.1f}/100")
    
    if score >= 80:
        print("   評価: S - 優秀！")
    elif score >= 60:
        print("   評価: A - 良好")
    elif score >= 40:
        print("   評価: B - 普通")
    elif score >= 20:
        print("   評価: C - 要改善")
    else:
        print("   評価: D - 学習が必要")
    
    # QBNN情報
    print("\n⚛️ QBNNもつれ状態:")
    for info in ai.model.get_quantum_info():
        print(f"   Block {info['block']}: λ = {info['attn_lambda']:.4f}")
    
    return results


if __name__ == '__main__':
    run_dialogue_test()

