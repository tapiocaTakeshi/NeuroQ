#!/usr/bin/env python3
"""
NeuroQ 20エポック トレーニングスクリプト
=========================================
ローカルで20エポックのトレーニングを実行します。

使い方:
    python train_20epochs.py
"""

import torch
import random
import os
import sys

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neuroq-runpod'))

def main():
    print("=" * 70)
    print("   NeuroQ 20エポック トレーニング")
    print("=" * 70)

    # デバイス設定
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("CPU")

    # モジュールインポート
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    from tiktoken_tokenizer import TikTokenTokenizer
    from model_configs import get_model_config

    # 設定
    model_size = "micro"  # micro, small, large
    epochs = 20
    batch_size = 4
    lr = 0.0005
    seq_length = 64

    print(f"\n設定:")
    print(f"   model_size: {model_size}")
    print(f"   epochs: {epochs}")
    print(f"   batch_size: {batch_size}")
    print(f"   lr: {lr}")
    print(f"   device: {device}")

    # モデル設定を取得
    config = get_model_config(model_size)
    model_vocab_size = config['vocab_size']

    # トークナイザー
    print("\nトークナイザー初期化...")
    tokenizer = TikTokenTokenizer(encoding_name="o200k_base", max_vocab=model_vocab_size)

    # 学習データ
    print("\n学習データ準備...")
    texts = get_training_data()
    print(f"   サンプル数: {len(texts)}")

    # トークン化
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special=False)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    print(f"   総トークン数: {len(all_tokens):,}")

    # モデル構築
    print("\nモデル構築...")
    nq_config = NeuroQuantumConfig()
    nq_config.vocab_size = config['vocab_size']
    nq_config.embed_dim = config['embed_dim']
    nq_config.num_heads = config['num_heads']
    nq_config.num_layers = config['num_layers']
    nq_config.max_seq_len = config.get('max_seq_len', 512)
    nq_config.dropout = config.get('dropout', 0.1)

    model = NeuroQuantum(nq_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   {config['name']}: {total_params:,} パラメータ")

    # シーケンス作成
    sequences = []
    for i in range(0, len(all_tokens) - seq_length, seq_length // 2):
        sequences.append(all_tokens[i:i + seq_length])
    print(f"   シーケンス数: {len(sequences):,}")

    # 学習
    print("\n" + "=" * 70)
    print("   学習開始（20エポック）")
    print("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        random.shuffle(sequences)

        for i in range(0, len(sequences) - batch_size, batch_size):
            batch = sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch, device=device)

            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = criterion(
                logits.reshape(-1, config['vocab_size']),
                target_ids.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

        print(f"   Epoch {epoch+1:2d}/{epochs}: Loss = {avg_loss:.4f}")

    print("\n" + "=" * 70)
    print(f"   学習完了！ ベストLoss: {best_loss:.4f}")
    print("=" * 70)

    # チェックポイント保存
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/neuroq_{model_size}_20epochs.pt"

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': {
            'type': 'tiktoken',
            'encoding': 'o200k_base',
            'vocab_size': tokenizer.vocab_size,
        },
        'model_size': model_size,
        'epochs': epochs,
        'best_loss': best_loss,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"\nチェックポイント保存: {checkpoint_path}")

    # テスト生成
    print("\n" + "=" * 70)
    print("   テスト生成")
    print("=" * 70)

    model.eval()
    test_prompts = ["こんにちは", "量子コンピュータ", "人工知能"]

    for prompt in test_prompts:
        result = generate_text(model, tokenizer, prompt, device, max_length=30)
        print(f"   入力: {prompt}")
        print(f"   出力: {result}")
        print()


def get_training_data():
    """学習データを取得"""
    texts = [
        # 自己紹介
        "<USER>こんにちは<ASSISTANT>こんにちは！何かお手伝いできることはありますか？",
        "<USER>あなたは誰ですか？<ASSISTANT>私はNeuroQです。量子ビットニューラルネットワークを使った人工知能です。",
        "<USER>何ができますか？<ASSISTANT>質問への回答、文章の生成、会話のお手伝いなどができます。",

        # 量子コンピュータ
        "<USER>量子コンピュータとは何ですか？<ASSISTANT>量子コンピュータは量子力学の原理を利用した次世代の計算機です。量子ビットは0と1の重ね合わせ状態を取ることができます。",
        "<USER>量子ビットとは？<ASSISTANT>量子ビットは量子コンピュータの基本単位で、0と1の両方の状態を同時に持つことができます。",
        "<USER>量子もつれとは？<ASSISTANT>量子もつれは、二つの粒子が離れていても瞬時に相関を持つ量子力学的な現象です。",

        # 人工知能
        "<USER>人工知能とは何ですか？<ASSISTANT>人工知能は人間の知能を模倣するコンピュータシステムです。機械学習やディープラーニングが主な技術です。",
        "<USER>機械学習とは？<ASSISTANT>機械学習はデータからパターンを学習するAIの手法です。大量のデータから自動的に規則を発見します。",
        "<USER>ディープラーニングとは？<ASSISTANT>ディープラーニングは多層ニューラルネットワークを使った機械学習の手法です。",
        "<USER>トランスフォーマーとは？<ASSISTANT>トランスフォーマーはアテンション機構を使った現代のAIの基盤アーキテクチャです。",

        # プログラミング
        "<USER>Pythonとは？<ASSISTANT>Pythonは人工知能開発で最も人気のあるプログラミング言語です。シンプルで読みやすい構文が特徴です。",
        "<USER>PyTorchとは？<ASSISTANT>PyTorchはFacebookが開発した深層学習フレームワークです。動的計算グラフが特徴です。",

        # 一般会話
        "<USER>ありがとう<ASSISTANT>どういたしまして。お役に立てて嬉しいです。",
        "<USER>さようなら<ASSISTANT>さようなら。またお話しましょう！",
        "<USER>元気ですか？<ASSISTANT>はい、元気です！ご質問があれば何でもどうぞ。",

        # 科学
        "<USER>物理学とは？<ASSISTANT>物理学は自然界の法則を研究する学問です。力、エネルギー、時間、空間などを扱います。",
        "<USER>化学とは？<ASSISTANT>化学は物質の性質と変化を研究する学問です。原子や分子の構造と反応を扱います。",
        "<USER>生物学とは？<ASSISTANT>生物学は生命の仕組みを研究する学問です。細胞、遺伝子、進化などを扱います。",
        "<USER>数学とは？<ASSISTANT>数学は論理と抽象的な構造を研究する学問です。数、量、空間の関係を扱います。",

        # 技術
        "<USER>GPUとは？<ASSISTANT>GPUは並列計算に優れたプロセッサで、深層学習の計算を高速化します。",
        "<USER>ニューラルネットワークとは？<ASSISTANT>ニューラルネットワークは脳の神経回路を模倣した計算モデルです。",
    ]

    # データを増やす
    extended = []
    for text in texts:
        extended.append(text)
        # 少し変形したデータも追加
        if "<USER>" in text and "<ASSISTANT>" in text:
            parts = text.split("<ASSISTANT>")
            if len(parts) == 2:
                extended.append(parts[0] + "<ASSISTANT>はい。" + parts[1])

    # 20倍に増やす
    return extended * 20


def generate_text(model, tokenizer, prompt, device, max_length=50, temperature=0.7):
    """テキスト生成"""
    model.eval()

    # プロンプトをトークン化
    input_ids = tokenizer.encode(f"<USER>{prompt}<ASSISTANT>", add_special=False)
    generated = input_ids.copy()

    vocab_size = model.config.vocab_size

    with torch.no_grad():
        for _ in range(max_length):
            seq = generated[-512:]
            seq_tensor = torch.tensor([seq], device=device)

            logits = model(seq_tensor)
            next_token_logits = logits[0, -1, :]

            # 温度でスケーリング
            probs = torch.softmax(next_token_logits / temperature, dim=-1)

            # Top-k sampling
            top_k = 40
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            top_probs = top_probs / top_probs.sum()
            idx = torch.multinomial(top_probs, num_samples=1)
            next_token = top_indices[idx].item()

            # EOSなら終了
            if next_token == tokenizer.eos_id:
                break

            generated.append(next_token)

    # デコード
    result = tokenizer.decode(generated, skip_special=True)

    # <ASSISTANT>以降を抽出
    if '<ASSISTANT>' in result:
        result = result.split('<ASSISTANT>')[-1].strip()

    # <USER>以降を除去
    if '<USER>' in result:
        result = result.split('<USER>')[0].strip()

    return result


if __name__ == "__main__":
    main()
