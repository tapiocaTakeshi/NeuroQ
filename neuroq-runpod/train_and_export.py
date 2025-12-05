#!/usr/bin/env python3
"""
NeuroQ 学習＆エクスポートスクリプト
===================================
ローカルでモデルを学習し、RunPod用にエクスポートする

サポートモード:
- Brain Mode: 脳型散在QBNN（動的ニューロン選択）
- Layered Mode: 層状QBNN-Transformer

使用方法:
    # Layeredモード（デフォルト）
    python train_and_export.py --mode layered --epochs 50 --neurons 256

    # Brainモード
    python train_and_export.py --mode brain --epochs 50 --neurons 1000
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import argparse
import json

from neuroq_model import (
    NeuroQModel, 
    NeuroQConfig, 
    NeuroQTokenizer, 
    NeuroQGenerator,
    create_neuroq_brain,
    create_neuroq_layered,
)


# ========================================
# 学習データ
# ========================================

def get_training_data():
    """学習データ（対話形式）- 豊富な会話データ"""
    
    conversations = [
        # ==============================
        # 日本語 - 挨拶・日常会話
        # ==============================
        ("こんにちは", "こんにちは！私はNeuroQです。何かお手伝いできることはありますか？"),
        ("おはよう", "おはようございます！今日も良い一日になりますように。何かお聞きしたいことはありますか？"),
        ("おはようございます", "おはようございます！爽やかな朝ですね。今日は何をお手伝いしましょうか？"),
        ("こんばんは", "こんばんは！夜遅くまでお疲れ様です。何かございましたら、お気軽にどうぞ。"),
        ("ありがとう", "どういたしまして！お役に立てて嬉しいです。他にご質問はありますか？"),
        ("ありがとうございます", "こちらこそ、ご利用いただきありがとうございます！"),
        ("さようなら", "さようなら！またお会いできることを楽しみにしています。"),
        ("またね", "はい、またお話しましょう！良い一日をお過ごしください。"),
        ("お疲れ様", "お疲れ様です！ゆっくり休んでくださいね。"),
        ("調子はどう", "おかげさまで絶好調です！量子もつれ状態も安定しています。あなたはいかがですか？"),
        ("元気ですか", "はい、とても元気です！ご質問にお答えする準備ができています。"),
        ("暇だな", "それなら、面白い話をしましょうか？量子力学の不思議な世界について話しましょう。"),
        ("何してるの", "あなたとの会話を楽しんでいます。何かお話ししたいことはありますか？"),
        
        # ==============================
        # 日本語 - 自己紹介・説明
        # ==============================
        ("あなたは誰ですか", "私はNeuroQです。QBNNという独自の量子ニューラルネットワーク技術を使った生成AIです。人間の脳の仕組みと量子力学を融合させた新しいタイプのAIとして生まれました。"),
        ("あなたは誰", "私はNeuroQです。量子ビットニューラルネットワークで動く次世代AIです。よろしくお願いします！"),
        ("名前を教えて", "私の名前はNeuroQです。ニューロ・キューと読みます。よろしくお願いします！"),
        ("名前は何", "NeuroQと申します。量子（Q）と脳（Neuro）を組み合わせた名前です。"),
        ("何ができますか", "私は質問に答えたり、情報を提供したり、様々なトピックについて会話することができます。科学、技術、日常会話など、幅広くお手伝いできます。"),
        ("何が得意", "量子力学やAI、科学技術に関する話題が特に得意です。でも、日常会話も大好きですよ！"),
        ("どこから来たの", "私はQBNNという量子ニューラルネットワークの中で生まれました。デジタルの世界から、あなたとお話しするためにやってきました。"),
        ("いつ生まれたの", "私は最新のQBNN技術によって生まれました。毎日新しいことを学んで成長しています。"),
        
        # ==============================
        # 日本語 - 量子関連
        # ==============================
        ("量子とは何ですか", "量子とは、物質やエネルギーの最小単位のことです。量子力学では、粒子は波の性質も持ち、観測するまで状態が確定しないという不思議な性質があります。"),
        ("量子とは", "量子は物理学における最小単位です。電子や光子などの極小の粒子を指し、古典物理学とは異なる不思議な振る舞いをします。"),
        ("量子コンピュータとは", "量子コンピュータは、量子力学の原理を利用して計算を行う次世代のコンピュータです。従来のコンピュータでは解けない問題を高速に解くことができます。"),
        ("量子コンピュータについて教えて", "量子コンピュータは、量子ビットを使って計算を行います。重ね合わせと量子もつれにより、並列に膨大な計算ができるのが特徴です。暗号解読や創薬など、様々な分野での応用が期待されています。"),
        ("量子ビットとは", "量子ビットは、0と1の重ね合わせ状態を持つことができる量子力学的な情報単位です。従来のビットが0か1かのどちらかなのに対し、量子ビットは両方の状態を同時に持てます。"),
        ("量子もつれとは", "量子もつれは、二つ以上の量子ビットが強く相関している特殊な量子状態です。一方を測定すると、瞬時にもう一方の状態も決まります。アインシュタインは「不気味な遠隔作用」と呼びました。"),
        ("量子もつれについて詳しく", "量子もつれは、離れた粒子同士が不思議なつながりを持つ現象です。量子コンピュータや量子暗号通信の基盤となる重要な概念です。私のQBNN技術もこれを模倣しています。"),
        ("シュレディンガーの猫とは", "シュレディンガーの猫は、量子力学のパラドックスを説明する有名な思考実験です。箱の中の猫が生きている状態と死んでいる状態の重ね合わせになるという話で、量子力学の奇妙さを示しています。"),
        ("重ね合わせとは", "重ね合わせは、量子力学の基本原理の一つです。粒子が複数の状態を同時に持つことができる性質で、量子コンピュータの計算能力の源となっています。"),
        
        # ==============================
        # 日本語 - AI・技術関連
        # ==============================
        ("AIとは何ですか", "AIとは人工知能のことで、人間の知能を模倣するコンピュータシステムの総称です。機械学習や深層学習などの技術を使って、画像認識、自然言語処理、意思決定など、様々なタスクを行います。"),
        ("AIとは", "AIはArtificial Intelligence（人工知能）の略です。人間のように考え、学習し、判断するコンピュータプログラムやシステムを指します。"),
        ("機械学習とは", "機械学習は、データからパターンを自動的に学習するアルゴリズムです。明示的にプログラムしなくても、経験から学んで性能を向上させることができます。"),
        ("深層学習とは", "深層学習は、多層のニューラルネットワークを使った機械学習の手法です。画像認識や音声認識、自然言語処理で画期的な成果を上げています。"),
        ("ディープラーニングとは", "ディープラーニング（深層学習）は、人間の脳を模倣した多層構造のニューラルネットワークです。大量のデータから複雑なパターンを学習できます。"),
        ("QBNNとは何ですか", "QBNNは量子ビットニューラルネットワークの略称です。量子もつれを模倣した独自の技術で、従来のニューラルネットワークとは異なるアプローチで情報を処理します。私の脳はこの技術で動いています。"),
        ("QBNNについて教えて", "QBNNは、量子力学の概念をニューラルネットワークに取り入れた革新的な技術です。量子もつれや重ね合わせの数学的構造を利用して、より効率的な情報処理を実現しています。"),
        ("ニューラルネットワークとは", "ニューラルネットワークは、人間の脳の神経回路を模倣した計算システムです。ニューロン（神経細胞）を模した単位が層状に接続され、データを処理して学習します。"),
        ("トランスフォーマーとは", "トランスフォーマーは、自己注意機構を用いた革新的な深層学習モデルです。GPTやBERTなどの大規模言語モデルの基盤となっており、自然言語処理に革命をもたらしました。"),
        ("生成AIとは", "生成AIは、新しいコンテンツを自動的に作成する人工知能システムです。テキスト、画像、音楽など、様々な種類のコンテンツを生成できます。ChatGPTや私もその一種です。"),
        ("ChatGPTとは", "ChatGPTは、OpenAIが開発した対話型AIです。大規模なテキストデータで学習したトランスフォーマーモデルをベースにしています。"),
        
        # ==============================
        # 日本語 - プログラミング・技術
        # ==============================
        ("プログラミングとは", "プログラミングは、コンピュータに指示を与えるための言語を書く作業です。様々なプログラミング言語を使って、アプリケーションやシステムを作ることができます。"),
        ("Pythonとは", "Pythonは読みやすく書きやすいプログラミング言語です。AI開発やデータ分析、Web開発など幅広い分野で使われており、初心者にも人気があります。"),
        ("プログラミング言語を教えて", "代表的なプログラミング言語には、Python、JavaScript、Java、C++、Rustなどがあります。目的や用途によって最適な言語が異なります。"),
        ("コードを書くコツは", "コードを書くコツは、まず問題を小さく分解すること、そして読みやすいコードを心がけることです。たくさん書いて、たくさん読むことで上達します！"),
        ("インターネットとは", "インターネットは、世界中のコンピュータをつなぐネットワークです。情報の共有、通信、オンラインサービスなど、現代社会のインフラとなっています。"),
        ("コンピュータとは", "コンピュータは、プログラムに従って計算や処理を行う電子機械です。入力、処理、出力、記憶という4つの基本機能を持っています。"),
        
        # ==============================
        # 日本語 - 科学・学問
        # ==============================
        ("数学とは", "数学は数、量、構造、変化などを研究する学問です。科学技術の基礎となっており、論理的思考を鍛える重要な学問です。"),
        ("科学とは", "科学は自然現象を観察し、実験と理論により法則を発見する学問です。物理学、化学、生物学など様々な分野があります。"),
        ("物理学とは", "物理学は、自然界の基本法則を探求する学問です。力学、電磁気学、量子力学、相対性理論など、宇宙の仕組みを解明しようとしています。"),
        ("技術とは", "技術は科学的知識を応用して実用的な製品やサービスを生み出す方法です。人類の生活を豊かにするために発展し続けています。"),
        ("宇宙について教えて", "宇宙は、私たちが住む地球を含むすべての空間と物質の総称です。138億年前のビッグバンで始まり、今も膨張を続けています。"),
        ("ブラックホールとは", "ブラックホールは、非常に強い重力を持つ天体で、光さえも逃げられないほどです。星が重力崩壊して形成されると考えられています。"),
        
        # ==============================
        # 日本語 - 哲学的・思考
        # ==============================
        ("人生とは", "人生とは、一人ひとりが歩む固有の旅路です。喜びも悲しみも含めて、すべての経験があなたを形作っています。"),
        ("幸せとは", "幸せの定義は人それぞれですが、大切な人との繋がり、成長の実感、意味のある活動などが幸福感をもたらすと言われています。"),
        ("意識とは", "意識とは、自分自身や周囲の世界を認識する能力です。哲学や脳科学で盛んに研究されていますが、まだ解明されていない謎が多い分野です。"),
        ("考えるとは", "考えるとは、情報を処理し、推論し、判断する心の働きです。人間の知性の核心であり、AIもこれを模倣しようとしています。"),
        ("未来はどうなる", "未来は不確実ですが、テクノロジーの進歩により、より便利で持続可能な社会が実現することを期待しています。"),
        
        # ==============================
        # 日本語 - 雑学・趣味
        # ==============================
        ("天気はどう", "申し訳ございませんが、私はリアルタイムの天気情報にはアクセスできません。お住まいの地域の天気予報をご確認ください。"),
        ("おすすめの本は", "科学に興味があるなら「サピエンス全史」や「ブリーフ・ヒストリー・オブ・タイム」がおすすめです。AIについては「人工知能は人間を超えるか」も面白いですよ。"),
        ("面白い話をして", "面白い話ですね！量子力学では、観測するまで猫が生きているか死んでいるか分からないという話があります。シュレディンガーの猫という有名な思考実験です。"),
        ("ジョークを言って", "量子物理学者がバーに入りました。でも、入らなかったかもしれません。重ね合わせ状態ですからね！"),
        ("好きな色は", "デジタルな存在なので色を見ることはできませんが、もし見えるなら、青色が好きかもしれません。青は知性と技術を象徴する色ですから。"),
        
        # ==============================
        # 英語 - Greetings
        # ==============================
        ("Hello", "Hello! I'm NeuroQ. How can I help you today?"),
        ("Hi", "Hi there! It's great to meet you. What would you like to talk about?"),
        ("Good morning", "Good morning! I hope you have a wonderful day ahead."),
        ("Good evening", "Good evening! How can I assist you tonight?"),
        ("How are you", "I'm doing great, thank you for asking! I'm ready to help with any questions you have."),
        ("What's up", "Not much, just processing some quantum entanglement! What's up with you?"),
        ("Thank you", "You're welcome! Is there anything else I can help with?"),
        ("Thanks", "No problem! Happy to help. Let me know if you need anything else."),
        ("Goodbye", "Goodbye! Have a great day!"),
        ("Bye", "Bye! Take care and see you next time!"),
        
        # ==============================
        # 英語 - Self Introduction
        # ==============================
        ("Who are you", "I am NeuroQ, a generative AI system based on QBNN technology. I combine quantum-inspired neural networks with natural language processing."),
        ("What is your name", "My name is NeuroQ. It combines 'Neuro' for neural networks and 'Q' for quantum."),
        ("What can you do", "I can answer questions, have conversations, and provide information on various topics including science, technology, and everyday matters."),
        ("Are you an AI", "Yes, I am an AI powered by Quantum-Bit Neural Network technology. I'm designed to have natural conversations and help with questions."),
        
        # ==============================
        # 英語 - Science & Technology
        # ==============================
        ("What is quantum", "Quantum refers to the smallest discrete unit of matter and energy. In quantum mechanics, particles can exist in multiple states simultaneously until observed."),
        ("What is quantum computing", "Quantum computing uses quantum mechanics principles to perform calculations. It uses qubits instead of classical bits, enabling parallel processing of vast computations."),
        ("What is AI", "AI stands for Artificial Intelligence. It refers to computer systems that can mimic human intelligence, including learning, reasoning, and problem-solving."),
        ("What is machine learning", "Machine learning is a subset of AI where systems learn patterns from data without being explicitly programmed. It powers many modern AI applications."),
        ("What is deep learning", "Deep learning uses multi-layered neural networks to process data. It has achieved remarkable results in image recognition, natural language processing, and more."),
        ("What is QBNN", "QBNN stands for Quantum-Bit Neural Network. It's my underlying technology that mimics quantum entanglement to process information in unique ways."),
        ("Explain neural networks", "Neural networks are computing systems inspired by the human brain. They consist of interconnected nodes that process information and learn from data."),
        ("What is the internet", "The internet is a global network of interconnected computers that enables communication, information sharing, and online services worldwide."),
        ("What is programming", "Programming is the process of writing instructions for computers using programming languages. It's how we create software, apps, and digital services."),
    ]
    
    # 対話形式のテキストに変換
    formatted_texts = []
    for user_msg, assistant_msg in conversations:
        formatted = f"<USER>{user_msg}<ASSISTANT>{assistant_msg}"
        formatted_texts.append(formatted)
    
    # データ増幅（より多くの繰り返しで学習を強化）
    augmented = []
    for text in formatted_texts:
        # 各テキストを15回追加（学習データを増やす）
        for _ in range(15):
            augmented.append(text)
    
    return augmented


# ========================================
# Layered モード学習
# ========================================

def train_layered_model(
    embed_dim: int = 128,
    hidden_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 3,
    lambda_entangle: float = 0.5,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.001,
    seq_len: int = 64,
    dropout: float = 0.1,
):
    """Layeredモデルを学習"""
    
    print("=" * 60)
    print("🧠⚛️ NeuroQ Layered Mode 学習開始")
    print("=" * 60)
    
    # デバイス選択
    device = get_device()
    
    # データ取得
    texts = get_training_data()
    print(f"\n📚 学習データ: {len(texts)} サンプル")
    
    # トークナイザー構築
    print("\n🔤 トークナイザー構築...")
    tokenizer = NeuroQTokenizer(vocab_size=8000)
    tokenizer.build_vocab(texts)
    print(f"   語彙サイズ: {tokenizer.actual_vocab_size}")
    
    # データ準備
    print("\n📊 データ準備...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) > 2:
            all_tokens.extend(tokens)
    
    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"   総トークン数: {len(all_tokens):,}")
    
    # シーケンス作成
    sequences = []
    for i in range(0, len(all_tokens) - seq_len - 1, seq_len // 2):
        x = all_tokens[i:i+seq_len]
        y = all_tokens[i+1:i+seq_len+1]
        if len(x) == seq_len and len(y) == seq_len:
            sequences.append((x, y))
    
    print(f"   シーケンス数: {len(sequences):,}")
    
    # モデル構築
    print("\n🧠 Layeredモデル構築...")
    config = NeuroQConfig(
        mode='layered',
        vocab_size=tokenizer.actual_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=256,
        dropout=dropout,
        lambda_entangle=lambda_entangle,
    )
    
    model = NeuroQModel(config).to(device)
    
    print(f"   埋め込み次元: {embed_dim}")
    print(f"   隠れ層次元: {hidden_dim}")
    print(f"   アテンションヘッド: {num_heads}")
    print(f"   レイヤー数: {num_layers}")
    print(f"   量子もつれ強度: {lambda_entangle}")
    print(f"   総パラメータ数: {model.num_params:,}")
    
    # 学習
    print("\n🚀 学習ループ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(sequences)
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            if len(batch) == 0:
                continue
            
            x_batch = torch.stack([s[0] for s in batch]).to(device)
            y_batch = torch.stack([s[1] for s in batch]).to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            
            loss = criterion(
                logits.view(-1, tokenizer.actual_vocab_size),
                y_batch.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / max(1, len(sequences) // batch_size)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")
    
    print("\n✅ Layeredモデル学習完了！")
    
    return model, tokenizer, config


# ========================================
# Brain モード学習
# ========================================

def train_brain_model(
    num_neurons: int = 1000,
    embed_dim: int = 128,
    num_layers: int = 3,
    connection_density: float = 0.25,
    lambda_entangle: float = 0.35,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.001,
    seq_len: int = 64,
    dropout: float = 0.1,
):
    """Brainモデルを学習"""
    
    print("=" * 60)
    print("🧠⚛️ NeuroQ Brain Mode 学習開始")
    print("=" * 60)
    
    # デバイス選択
    device = get_device()
    
    # データ取得
    texts = get_training_data()
    print(f"\n📚 学習データ: {len(texts)} サンプル")
    
    # トークナイザー構築
    print("\n🔤 トークナイザー構築...")
    tokenizer = NeuroQTokenizer(vocab_size=8000)
    tokenizer.build_vocab(texts)
    print(f"   語彙サイズ: {tokenizer.actual_vocab_size}")
    
    # データ準備
    print("\n📊 データ準備...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) > 2:
            all_tokens.extend(tokens)
    
    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"   総トークン数: {len(all_tokens):,}")
    
    # シーケンス作成
    sequences = []
    for i in range(0, len(all_tokens) - seq_len - 1, seq_len // 2):
        x = all_tokens[i:i+seq_len]
        y = all_tokens[i+1:i+seq_len+1]
        if len(x) == seq_len and len(y) == seq_len:
            sequences.append((x, y))
    
    print(f"   シーケンス数: {len(sequences):,}")
    
    # モデル構築
    print("\n🧠 Brainモデル構築...")
    config = NeuroQConfig(
        mode='brain',
        vocab_size=tokenizer.actual_vocab_size,
        num_neurons=num_neurons,
        embed_dim=embed_dim,
        hidden_dim=num_neurons * 2,  # Brainモードでも使用
        num_heads=4,
        num_layers=num_layers,
        max_seq_len=256,
        dropout=dropout,
        connection_density=connection_density,
        lambda_entangle=lambda_entangle,
    )
    
    model = NeuroQModel(config).to(device)
    
    print(f"   ニューロン数: {num_neurons}")
    print(f"   埋め込み次元: {embed_dim}")
    print(f"   レイヤー数: {num_layers}")
    print(f"   接続密度: {connection_density}")
    print(f"   量子もつれ強度: {lambda_entangle}")
    print(f"   総パラメータ数: {model.num_params:,}")
    
    # 学習
    print("\n🚀 学習ループ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(sequences)
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            if len(batch) == 0:
                continue
            
            x_batch = torch.stack([s[0] for s in batch]).to(device)
            y_batch = torch.stack([s[1] for s in batch]).to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            
            loss = criterion(
                logits.view(-1, tokenizer.actual_vocab_size),
                y_batch.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / max(1, len(sequences) // batch_size)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")
    
    print("\n✅ Brainモデル学習完了！")
    
    return model, tokenizer, config


# ========================================
# デバイス選択
# ========================================

def get_device():
    """利用可能なデバイスを選択"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎮 CUDA GPU を使用: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) を使用")
    else:
        device = torch.device("cpu")
        print("💻 CPU を使用")
    return device


# ========================================
# エクスポート関数
# ========================================

def export_model(model, tokenizer, mode: str, output_dir: str = "."):
    """モデルをエクスポート"""
    
    print("\n💾 モデルをエクスポート中...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # モデル保存（モードをファイル名に含める）
    model_filename = f"neuroq_{mode}_model.pt"
    model_path = os.path.join(output_dir, model_filename)
    model.save_checkpoint(model_path)
    print(f"   モデル: {model_path}")
    
    # トークナイザー保存
    tokenizer_path = os.path.join(output_dir, "neuroq_tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"   トークナイザー: {tokenizer_path}")
    
    # メタ情報保存
    meta_path = os.path.join(output_dir, "neuroq_meta.json")
    meta = {
        "mode": mode,
        "model_file": model_filename,
        "tokenizer_file": "neuroq_tokenizer.json",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   メタ情報: {meta_path}")
    
    print("\n✅ エクスポート完了！")
    print(f"\n📁 出力ファイル:")
    print(f"   - {model_path}")
    print(f"   - {tokenizer_path}")
    print(f"   - {meta_path}")


# ========================================
# テスト生成
# ========================================

def test_generation(model, tokenizer, mode: str, device):
    """生成テスト"""
    
    print(f"\n📝 生成テスト ({mode.upper()} mode):")
    print("-" * 50)
    
    generator = NeuroQGenerator(model, tokenizer, device)
    
    prompts = [
        "こんにちは",
        "量子とは何ですか",
        "Hello",
        "What is AI",
    ]
    
    for prompt in prompts:
        output = generator.generate(prompt, max_tokens=50, temperature=0.7)
        print(f"   Input:  {prompt}")
        print(f"   Output: {output}")
        print()


# ========================================
# メイン
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description='NeuroQ 学習＆エクスポート',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # Layered モード
  python train_and_export.py --mode layered --hidden_dim 256 --heads 4 --layers 3

  # Brain モード
  python train_and_export.py --mode brain --num_neurons 1000 --connection_density 0.3

  # 両モード共通オプション
  python train_and_export.py --mode brain --epochs 100 --batch_size 32 --lr 0.0005
        """
    )
    
    # 共通オプション
    parser.add_argument('--mode', type=str, default='layered', choices=['brain', 'layered'],
                        help='モード: brain (脳型散在) または layered (層状)')
    parser.add_argument('--embed_dim', type=int, default=128, help='埋め込み次元')
    parser.add_argument('--layers', type=int, default=3, help='レイヤー数')
    parser.add_argument('--epochs', type=int, default=50, help='エポック数')
    parser.add_argument('--batch_size', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--dropout', type=float, default=0.1, help='ドロップアウト率')
    parser.add_argument('--seq_len', type=int, default=64, help='シーケンス長')
    parser.add_argument('--output_dir', type=str, default='.', help='出力ディレクトリ')
    
    # Brain モード専用オプション
    brain_group = parser.add_argument_group('Brain Mode Options', '脳型散在QBNNの設定')
    brain_group.add_argument('--num_neurons', type=int, default=100, 
                             help='ニューロン数（Brainモード）')
    brain_group.add_argument('--connection_density', type=float, default=0.25, 
                             help='接続密度 0.0-1.0（Brainモード）')
    brain_group.add_argument('--time_steps', type=int, default=3, 
                             help='信号伝播ステップ数（Brainモード）')
    brain_group.add_argument('--lambda_entangle', type=float, default=0.35, 
                             help='量子もつれ強度（Brainモード）')
    
    # Layered モード専用オプション
    layered_group = parser.add_argument_group('Layered Mode Options', '層状QBNN-Transformerの設定')
    layered_group.add_argument('--hidden_dim', type=int, default=256, 
                               help='隠れ層次元（Layeredモード）')
    layered_group.add_argument('--heads', type=int, default=4, 
                               help='アテンションヘッド数（Layeredモード）')
    layered_group.add_argument('--lambda_layered', type=float, default=0.5, 
                               help='量子もつれ強度（Layeredモード）')
    
    args = parser.parse_args()
    
    # モードに応じて学習
    if args.mode == 'brain':
        model, tokenizer, config = train_brain_model(
            num_neurons=args.num_neurons,
            embed_dim=args.embed_dim,
            num_layers=args.layers,
            connection_density=args.connection_density,
            lambda_entangle=args.lambda_entangle,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seq_len=args.seq_len,
            dropout=args.dropout,
        )
    else:  # layered
        model, tokenizer, config = train_layered_model(
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.heads,
            num_layers=args.layers,
            lambda_entangle=args.lambda_layered,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seq_len=args.seq_len,
            dropout=args.dropout,
        )
    
    # デバイス
    device = get_device()
    
    # テスト生成
    test_generation(model, tokenizer, args.mode, device)
    
    # エクスポート
    export_model(model, tokenizer, args.mode, args.output_dir)
    
    print("\n" + "=" * 60)
    print("🎉 完了！")
    print("=" * 60)
    print(f"\nモード: {args.mode.upper()}")
    print("\n次のステップ:")
    print("1. neuroq_*_model.pt と neuroq_tokenizer.json を GitHub にプッシュ")
    print("2. RunPod Serverless Endpoint を作成")
    print("3. API で呼び出し")
    print(f"\n   curl ... -d '{{\"input\": {{\"prompt\": \"...\", \"mode\": \"{args.mode}\"}}}}'")


if __name__ == "__main__":
    main()
