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
    """学習データ（対話形式）"""
    
    conversations = [
        # 日本語 - 挨拶
        ("こんにちは", "こんにちは！私はNeuroQです。何かお手伝いできることはありますか？"),
        ("おはよう", "おはようございます！今日も良い一日になりますように。"),
        ("こんばんは", "こんばんは！夜遅くまでお疲れ様です。"),
        ("ありがとう", "どういたしまして！お役に立てて嬉しいです。"),
        ("さようなら", "さようなら！またお会いしましょう。"),
        
        # 日本語 - 自己紹介
        ("あなたは誰ですか", "私はNeuroQです。QBNNという独自の量子ニューラルネットワーク技術を使った生成AIです。"),
        ("名前を教えて", "私の名前はNeuroQです。よろしくお願いします。"),
        ("何ができますか", "私は質問に答えたり、情報を提供したりすることができます。"),
        
        # 日本語 - 量子関連
        ("量子とは何ですか", "量子とは、物質やエネルギーの最小単位のことです。量子力学では、粒子は波の性質も持ちます。"),
        ("量子コンピュータとは", "量子コンピュータは、量子力学の原理を利用して計算を行う次世代のコンピュータです。"),
        ("量子ビットとは", "量子ビットは、0と1の重ね合わせ状態を持つことができる量子力学的な情報単位です。"),
        ("量子もつれとは", "量子もつれは、二つ以上の量子ビットが強く相関している特殊な量子状態です。"),
        
        # 日本語 - AI関連
        ("AIとは何ですか", "AIとは人工知能のことで、人間の知能を模倣するコンピュータシステムの総称です。"),
        ("機械学習とは", "機械学習は、データからパターンを自動的に学習するアルゴリズムです。"),
        ("深層学習とは", "深層学習は、多層のニューラルネットワークを使った機械学習の手法です。"),
        ("QBNNとは何ですか", "QBNNは量子ビットニューラルネットワークの略称です。量子もつれを模倣した独自の技術です。"),
        
        # 英語
        ("Hello", "Hello! I'm NeuroQ. How can I help you today?"),
        ("Who are you", "I am NeuroQ, a generative AI system based on QBNN technology."),
        ("What is quantum", "Quantum refers to the smallest discrete unit of matter and energy."),
        ("What is AI", "AI stands for Artificial Intelligence. It refers to computer systems that can mimic human intelligence."),
        ("Thank you", "You're welcome! Is there anything else I can help with?"),
        ("Goodbye", "Goodbye! Have a great day!"),
        
        # 追加データ
        ("プログラミングとは", "プログラミングは、コンピュータに指示を与えるための言語を書く作業です。"),
        ("Pythonとは", "Pythonは読みやすく書きやすいプログラミング言語です。AI開発で特に人気があります。"),
        ("数学とは", "数学は数、量、構造、変化などを研究する学問です。科学技術の基礎となっています。"),
        ("科学とは", "科学は自然現象を観察し、実験と理論により法則を発見する学問です。"),
        ("技術とは", "技術は科学的知識を応用して実用的な製品やサービスを生み出す方法です。"),
        ("インターネットとは", "インターネットは、世界中のコンピュータをつなぐネットワークです。"),
        ("コンピュータとは", "コンピュータは、プログラムに従って計算や処理を行う電子機械です。"),
        ("ニューラルネットワークとは", "ニューラルネットワークは、人間の脳の神経回路を模倣した計算システムです。"),
        ("トランスフォーマーとは", "トランスフォーマーは、自己注意機構を用いた革新的な深層学習モデルです。"),
        ("生成AIとは", "生成AIは、新しいコンテンツを自動的に作成する人工知能システムです。"),
    ]
    
    # 対話形式のテキストに変換
    formatted_texts = []
    for user_msg, assistant_msg in conversations:
        formatted = f"<USER>{user_msg}<ASSISTANT>{assistant_msg}"
        formatted_texts.append(formatted)
    
    # データ増幅
    augmented = []
    for text in formatted_texts:
        augmented.append(text)
        # 各テキストを複数回追加（学習データを増やす）
        for _ in range(10):
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
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.001,
    seq_len: int = 64,
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
        vocab_size=tokenizer.actual_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=256,
        dropout=0.1,
        lambda_entangle=0.5,
    )
    
    model = NeuroQModel(config).to(device)
    
    print(f"   埋め込み次元: {embed_dim}")
    print(f"   隠れ層次元: {hidden_dim}")
    print(f"   アテンションヘッド: {num_heads}")
    print(f"   レイヤー数: {num_layers}")
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
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.001,
    seq_len: int = 64,
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
        num_layers=3,
        max_seq_len=256,
        dropout=0.1,
        connection_density=0.25,
    )
    
    model = NeuroQModel(config).to(device)
    
    print(f"   ニューロン数: {num_neurons}")
    print(f"   埋め込み次元: {embed_dim}")
    print(f"   レイヤー数: {config.num_layers}")
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
    parser = argparse.ArgumentParser(description='NeuroQ 学習＆エクスポート')
    parser.add_argument('--mode', type=str, default='layered', choices=['brain', 'layered'],
                        help='モード: brain (脳型散在) または layered (層状)')
    parser.add_argument('--embed_dim', type=int, default=128, help='埋め込み次元')
    parser.add_argument('--neurons', type=int, default=256, 
                        help='ニューロン数 (brainモード) または隠れ層次元 (layeredモード)')
    parser.add_argument('--heads', type=int, default=4, help='アテンションヘッド数 (layeredモードのみ)')
    parser.add_argument('--layers', type=int, default=3, help='レイヤー数 (layeredモードのみ)')
    parser.add_argument('--epochs', type=int, default=50, help='エポック数')
    parser.add_argument('--batch_size', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--output_dir', type=str, default='.', help='出力ディレクトリ')
    args = parser.parse_args()
    
    # モードに応じて学習
    if args.mode == 'brain':
        model, tokenizer, config = train_brain_model(
            num_neurons=args.neurons,
            embed_dim=args.embed_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    else:  # layered
        model, tokenizer, config = train_layered_model(
            embed_dim=args.embed_dim,
            hidden_dim=args.neurons,
            num_heads=args.heads,
            num_layers=args.layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
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
