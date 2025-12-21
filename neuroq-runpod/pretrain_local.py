#!/usr/bin/env python3
"""
NeuroQ ローカル事前学習スクリプト
==================================
大量データで学習し、学習済みモデルを保存します。
その後、RunPodにアップロードして使用します。

使い方:
    python pretrain_local.py

生成されるファイル:
    - neuroq_pretrained.pth (学習済みモデル)
"""

import torch
import os
import sys

# neuroquantum_layered.py をインポート
from neuroquantum_layered import NeuroQuantumAI

def generate_training_data():
    """学習用の大量テキストデータを生成"""
    
    # 基本的な知識ベース
    knowledge_base = [
        # 自己紹介
        "私はニューロQです。量子ビットニューラルネットワークを使った人工知能です。",
        "ニューロQは日本で開発された次世代のAIシステムです。",
        "私は量子コンピュータの原理を活用して動作しています。",
        "人工知能として、私はあなたの質問にお答えします。",
        "こんにちは。何かお手伝いできることはありますか？",
        
        # 量子コンピュータ
        "量子コンピュータは量子力学の原理を利用した計算機です。",
        "量子ビットは0と1の重ね合わせ状態を取ることができます。",
        "量子もつれは二つの粒子が相関を持つ現象です。",
        "量子コンピュータは暗号解読や最適化問題に優れています。",
        "量子ゲートは量子ビットを操作する基本単位です。",
        "量子アルゴリズムは従来のコンピュータより高速に問題を解けます。",
        "量子誤り訂正は量子計算の信頼性を高める技術です。",
        "超電導量子ビットは現在最も普及している量子ビットです。",
        "イオントラップ型量子コンピュータも研究が進んでいます。",
        "量子優位性は量子コンピュータが古典コンピュータを超える能力です。",
        
        # 人工知能
        "人工知能は人間の知能を模倣するシステムです。",
        "機械学習はデータからパターンを学習する技術です。",
        "ディープラーニングは多層ニューラルネットワークを使います。",
        "自然言語処理は言語を理解し生成する技術です。",
        "トランスフォーマーは現代のAIの基盤となるアーキテクチャです。",
        "アテンション機構は入力の重要な部分に注目します。",
        "GPTは大規模言語モデルの代表的な例です。",
        "BERTは双方向のトランスフォーマーモデルです。",
        "強化学習は試行錯誤から学ぶ機械学習の手法です。",
        "教師あり学習はラベル付きデータから学習します。",
        "教師なし学習はラベルなしデータからパターンを発見します。",
        
        # ニューラルネットワーク
        "ニューラルネットワークは脳の神経回路を模倣しています。",
        "ニューロンは入力を受け取り、活性化関数を通して出力します。",
        "重みとバイアスはニューラルネットワークのパラメータです。",
        "勾配降下法は損失関数を最小化するアルゴリズムです。",
        "バックプロパゲーションは勾配を計算する手法です。",
        "活性化関数は非線形性を導入します。",
        "ReLUは最も一般的な活性化関数です。",
        "ソフトマックス関数は確率分布を出力します。",
        "ドロップアウトは過学習を防ぐ正則化技術です。",
        "バッチ正規化は学習を安定させます。",
        
        # プログラミング
        "Pythonは人工知能開発で最も人気のある言語です。",
        "PyTorchは深層学習フレームワークの一つです。",
        "TensorFlowもGoogleが開発した深層学習フレームワークです。",
        "GPUは並列計算に優れており、深層学習に適しています。",
        "CUDAはNVIDIAのGPUプログラミングプラットフォームです。",
        
        # 科学技術
        "科学は観察と実験に基づく知識体系です。",
        "技術は科学を応用して問題を解決します。",
        "数学は論理と抽象的な構造を研究します。",
        "物理学は自然界の法則を研究します。",
        "化学は物質の性質と変化を研究します。",
        "生物学は生命の仕組みを研究します。",
        "コンピュータサイエンスは計算と情報を研究します。",
        
        # 会話パターン
        "質問があれば、お気軽にどうぞ。",
        "それは興味深い質問ですね。",
        "詳しく説明しましょう。",
        "理解できましたか？",
        "他に質問はありますか？",
        "お役に立てて嬉しいです。",
        "ありがとうございます。",
        "承知しました。",
        "なるほど、そういうことですね。",
        "もう少し詳しく教えてください。",
    ]
    
    # パターンを組み合わせて多様性を増やす
    extended_data = []
    
    # 基本データを追加
    extended_data.extend(knowledge_base)
    
    # 組み合わせパターン
    prefixes = [
        "つまり、",
        "例えば、",
        "具体的には、",
        "言い換えると、",
        "さらに、",
        "また、",
        "一方で、",
        "特に、",
        "実際に、",
        "結論として、",
    ]
    
    for prefix in prefixes:
        for sentence in knowledge_base[:30]:
            extended_data.append(f"{prefix}{sentence}")
    
    # Q&Aパターン
    qa_patterns = [
        ("量子コンピュータとは何ですか？", "量子コンピュータは量子力学の原理を利用した次世代の計算機です。"),
        ("人工知能とは何ですか？", "人工知能は人間の知能を模倣するコンピュータシステムです。"),
        ("ニューロQとは何ですか？", "ニューロQは量子ビットニューラルネットワークを使った人工知能です。"),
        ("機械学習とは何ですか？", "機械学習はデータからパターンを学習するAIの手法です。"),
        ("ディープラーニングとは？", "ディープラーニングは多層ニューラルネットワークを使った機械学習です。"),
        ("トランスフォーマーとは？", "トランスフォーマーは現代のAIの基盤となるアーキテクチャです。"),
        ("GPTとは何ですか？", "GPTは大規模言語モデルの代表的な例です。"),
        ("Pythonとは？", "Pythonは人工知能開発で最も人気のあるプログラミング言語です。"),
        ("こんにちは", "こんにちは！何かお手伝いできることはありますか？"),
        ("ありがとう", "どういたしまして。お役に立てて嬉しいです。"),
    ]
    
    for q, a in qa_patterns:
        extended_data.append(f"<USER>{q}<ASSISTANT>{a}")
        extended_data.append(f"質問: {q}\n回答: {a}")
    
    # データを複製して量を増やす
    final_data = extended_data * 20  # 20倍に増やす
    
    print(f"📊 学習データ準備完了:")
    print(f"   文数: {len(final_data)}")
    print(f"   総文字数: {sum(len(s) for s in final_data):,}")
    
    return final_data


def main():
    print("=" * 60)
    print("🧠 NeuroQ ローカル事前学習")
    print("=" * 60)
    
    # デバイス確認
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("💻 CPU")
    
    # 学習データ生成
    print("\n📚 学習データ準備中...")
    training_data = generate_training_data()
    
    # モデル作成
    print("\n🧠 モデル作成中...")
    model = NeuroQuantumAI(
        embed_dim=256,      # より大きな次元
        hidden_dim=512,     # より大きな隠れ層
        num_heads=8,        # より多いヘッド
        num_layers=6,       # より多い層
        max_seq_len=256,
        dropout=0.1,
        lambda_entangle=0.5
    )
    
    # 学習
    print("\n🚀 学習開始...")
    print("   （これには数分〜数十分かかる場合があります）")
    
    model.train(
        training_data,
        epochs=30,          # 多めのエポック
        seq_len=128,        # 長めのシーケンス
        batch_size=16,
        lr=0.001
    )
    
    # モデル保存
    save_path = "neuroq_pretrained.pth"
    print(f"\n💾 モデル保存中: {save_path}")
    
    # 保存データを準備
    save_data = {
        'model_state_dict': model.model.state_dict(),
        'config': {
            'vocab_size': model.config.vocab_size,
            'embed_dim': model.config.embed_dim,
            'hidden_dim': model.config.hidden_dim,
            'num_heads': model.config.num_heads,
            'num_layers': model.config.num_layers,
            'max_seq_len': model.config.max_seq_len,
            'dropout': model.config.dropout,
            'lambda_entangle': model.config.lambda_entangle,
        },
        'tokenizer_vocab_size': model.tokenizer.actual_vocab_size or model.tokenizer.vocab_size,
    }
    
    torch.save(save_data, save_path)
    
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ 保存完了: {save_path} ({file_size:.1f} MB)")
    
    # テスト生成
    print("\n🧪 生成テスト:")
    test_prompts = [
        "量子コンピュータ",
        "人工知能とは",
        "こんにちは",
    ]
    
    for prompt in test_prompts:
        result = model.generate(prompt, max_length=50, temperature=0.7)
        print(f"   入力: {prompt}")
        print(f"   出力: {result[:100]}...")
        print()
    
    print("=" * 60)
    print("✅ 事前学習完了！")
    print("=" * 60)
    print("\n次のステップ:")
    print("1. neuroq_pretrained.pth をRunPodにアップロード")
    print("2. handler.py を修正して学習済みモデルをロード")
    print("3. Rebuild & Deploy")


if __name__ == "__main__":
    main()
