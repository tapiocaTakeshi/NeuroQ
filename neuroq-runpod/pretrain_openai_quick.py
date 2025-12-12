#!/usr/bin/env python3
"""
NeuroQ OpenAIデータセット事前学習スクリプト（軽量版）
=======================================================
handler.py経由での実行に最適化
- エポック数削減: 5エポック
- データ量削減: 各データセットの一部のみ使用
- 高速実行: 10-20分で完了目標
"""

import torch
import os
import sys
from typing import List

print("=" * 60, flush=True)
print("🧠 NeuroQ OpenAIデータセット事前学習（軽量版）", flush=True)
print("=" * 60, flush=True)

# Hugging Face datasets をインポート
try:
    from datasets import load_dataset
    print("✅ datasets ライブラリをインポートしました", flush=True)
except ImportError:
    print("❌ datasets ライブラリがインストールされていません", flush=True)
    print("   実行: pip install datasets", flush=True)
    sys.exit(1)

# neuroquantum_layered.py をインポート
from neuroquantum_layered import NeuroQuantumAI


def load_humaneval_data_quick() -> List[str]:
    """HumanEvalデータセット（全データ）"""
    print("\n📥 openai/openai_humaneval データセットをロード中...", flush=True)
    texts = []

    try:
        dataset = load_dataset("openai/openai_humaneval", split="test")
        print(f"   ✅ {len(dataset)} サンプルをロードしました", flush=True)

        for item in dataset:
            if "prompt" in item and item["prompt"]:
                texts.append(f"プログラミング課題:\n{item['prompt']}")
            if "canonical_solution" in item and item["canonical_solution"]:
                texts.append(f"解答コード:\n{item['canonical_solution']}")

        print(f"   ✅ humaneval から {len(texts)} テキストを抽出", flush=True)
    except Exception as e:
        print(f"   ⚠️ humaneval ロードエラー: {e}", flush=True)

    return texts


def load_mmmlu_data_quick() -> List[str]:
    """MMMMLUデータセット（サンプル500件）"""
    print("\n📥 openai/MMMLU データセットをロード中（サンプル500件）...", flush=True)
    texts = []

    try:
        dataset = load_dataset("openai/MMMLU", "JA_JP", split="test")
        print(f"   ✅ MMMLU日本語: {len(dataset)} サンプルをロードしました", flush=True)

        # 最初の500件のみ使用
        for item in list(dataset)[:500]:
            if "Question" in item and item["Question"]:
                q = item["Question"]
                choices = []
                for key in ["A", "B", "C", "D"]:
                    if key in item and item[key]:
                        choices.append(f"{key}. {item[key]}")
                answer = item.get("Answer", "")

                if choices:
                    choices_text = "\n".join(choices)
                    texts.append(f"問題: {q}\n選択肢:\n{choices_text}\n正解: {answer}")

                if answer and answer in ["A", "B", "C", "D"]:
                    answer_text = item.get(answer, "")
                    if answer_text:
                        texts.append(f"<USER>{q}<ASSISTANT>{answer_text}")

        print(f"   ✅ MMMLU から {len(texts)} テキストを抽出", flush=True)
    except Exception as e:
        print(f"   ⚠️ MMMLU ロードエラー: {e}", flush=True)

    return texts


def generate_japanese_instruction_data() -> List[str]:
    """日本語指示データ"""
    instruction_data = [
        "<USER>ChatGPTについて教えて<ASSISTANT>ChatGPTはOpenAIが開発した大規模言語モデルです。",
        "<USER>量子コンピュータについて教えて<ASSISTANT>量子コンピュータは量子力学の原理を利用した次世代の計算機です。",
        "<USER>ニューロQとは何ですか？<ASSISTANT>ニューロQは量子ビットニューラルネットワークを使った次世代のAIです。",
        "<USER>こんにちは<ASSISTANT>こんにちは！私はニューロQです。何かお手伝いできることはありますか？",
        "<USER>Pythonについて教えて<ASSISTANT>PythonはAI開発で最も人気のあるプログラミング言語です。",
    ]
    return instruction_data * 20  # 100サンプル


def main():
    # デバイス確認
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = "cpu"
        print("💻 CPU", flush=True)

    # データセットをロード
    all_texts = []

    # 1. openai/openai_humaneval（全データ）
    humaneval_texts = load_humaneval_data_quick()
    all_texts.extend(humaneval_texts)

    # 2. openai/MMMLU（サンプル500件）
    mmmlu_texts = load_mmmlu_data_quick()
    all_texts.extend(mmmlu_texts)

    # 3. 日本語指示データ
    print("\n📚 日本語指示データを追加中...", flush=True)
    instruction_texts = generate_japanese_instruction_data()
    all_texts.extend(instruction_texts)

    print(f"\n📊 最終学習データ:", flush=True)
    print(f"   総テキスト数: {len(all_texts):,}", flush=True)
    print(f"   総文字数: {sum(len(s) for s in all_texts):,}", flush=True)

    # モデル作成
    print("\n🧠 モデル作成中...", flush=True)
    model = NeuroQuantumAI(
        embed_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=256,
        dropout=0.1,
        lambda_entangle=0.5
    )

    # 学習（軽量設定）
    print("\n🚀 学習開始（5エポック）...", flush=True)
    print("   （これには10分〜20分かかる場合があります）", flush=True)

    model.train(
        all_texts,
        epochs=5,  # 25→5に削減
        seq_len=128,
        batch_size=16,
        lr=0.0005
    )

    # モデル保存
    save_path = "neuroq_pretrained.pt"
    print(f"\n💾 モデル保存中: {save_path}", flush=True)

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
        'training_info': {
            'datasets': ['openai/openai_humaneval', 'openai/MMMLU (sample)', 'japanese_instructions'],
            'total_texts': len(all_texts),
            'epochs': 5,
        }
    }

    torch.save(save_data, save_path)

    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ 保存完了: {save_path} ({file_size:.1f} MB)", flush=True)

    # テスト生成
    print("\n🧪 生成テスト:", flush=True)
    test_prompts = [
        "ChatGPTについて教えて",
        "量子コンピュータとは",
    ]

    for prompt in test_prompts:
        result = model.generate(prompt, max_length=50, temperature=0.7)
        print(f"\n   入力: {prompt}", flush=True)
        print(f"   出力: {result[:100]}...", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("✅ OpenAIデータセット事前学習完了（軽量版）！", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
