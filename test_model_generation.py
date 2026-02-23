#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQ モデル生成テスト

学習済みモデルの生成品質をテストするスクリプト
文字化けを防ぐための推奨パラメータ設定付き
"""

import sys
import os
import torch
from pathlib import Path

# パスを追加
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import fugashi
except ImportError:
    print("fugashi がインストールされていません")
    sys.exit(1)

from train_oasst1_ja import SimpleTransformer, SimpleTokenizer


def load_model(checkpoint_path: str, tokenizer_path: str, device: torch.device):
    """モデルとトークナイザーを読み込む"""

    # トークナイザー
    tokenizer = SimpleTokenizer(tokenizer_path)

    # チェックポイント
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # モデル
    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        max_seq_len=160,
        dropout=0.0  # 推論時はドロップアウト無効
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.5,
    max_length: int = 80,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
):
    """
    応答を生成

    推奨パラメータ:
    - temperature: 0.3-0.7 (低いほど安定、高いほど多様)
    - max_length: 50-120 (長すぎると崩壊しやすい)
    - top_k: 30-50
    - top_p: 0.85-0.95
    - repetition_penalty: 1.1-1.2 (繰り返し防止)
    """
    device = next(model.parameters()).device

    # 会話形式でプロンプトを整形
    if not prompt.startswith("User:"):
        formatted_prompt = f"User: {prompt}\nAssistant:"
    else:
        formatted_prompt = prompt

    tokens = tokenizer.encode(formatted_prompt)

    model.eval()
    with torch.no_grad():
        result_tokens = list(tokens)

        for _ in range(max_length):
            if len(result_tokens) >= model.max_seq_len:
                break

            x = torch.tensor([result_tokens[-model.max_seq_len:]], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty > 1.0:
                for token_id in set(result_tokens[-50:]):  # 直近50トークンに対してペナルティ
                    next_logits[token_id] /= repetition_penalty

            # Top-k サンプリング
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Top-p サンプリング
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            result_tokens.append(next_token)

            # EOS または改行でUser:が来たら停止
            if next_token == tokenizer.eos_id:
                break

            # 生成されたテキストをチェック
            current_text = tokenizer.decode(result_tokens)
            if "\nUser:" in current_text[len(formatted_prompt):]:
                break

    output = tokenizer.decode(result_tokens)

    # Assistantの応答部分のみを抽出
    if "Assistant:" in output:
        response = output.split("Assistant:")[-1].strip()
        # 次のUser:があれば切る
        if "User:" in response:
            response = response.split("User:")[0].strip()
        return response

    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NeuroQ モデル生成テスト")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/oasst1_ja_model.pt")
    parser.add_argument("--tokenizer", type=str, default="neuroq_tokenizer_oasst1_ja.model")
    parser.add_argument("--temperature", type=float, default=0.5, help="生成温度 (推奨: 0.3-0.7)")
    parser.add_argument("--max-length", type=int, default=80, help="最大生成長")
    parser.add_argument("--interactive", action="store_true", help="対話モード")
    args = parser.parse_args()

    print("=" * 60)
    print("NeuroQ モデル生成テスト")
    print("=" * 60)

    # デバイス
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"デバイス: {device}")

    # モデル読み込み
    if not os.path.exists(args.checkpoint):
        print(f"チェックポイントが見つかりません: {args.checkpoint}")
        return

    if not os.path.exists(args.tokenizer):
        print(f"トークナイザーが見つかりません: {args.tokenizer}")
        return

    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    print("モデル読み込み完了")

    print(f"\n生成パラメータ:")
    print(f"  temperature: {args.temperature}")
    print(f"  max_length: {args.max_length}")
    print(f"  top_k: 40, top_p: 0.9, repetition_penalty: 1.1")

    if args.interactive:
        # 対話モード
        print("\n対話モード（終了: 'quit' または 'exit'）")
        print("-" * 60)

        while True:
            try:
                user_input = input("\nあなた: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if not user_input:
                    continue

                response = generate_response(
                    model, tokenizer, user_input,
                    temperature=args.temperature,
                    max_length=args.max_length
                )
                print(f"NeuroQ: {response}")

            except KeyboardInterrupt:
                break
    else:
        # テストプロンプト
        test_prompts = [
            "こんにちは",
            "今日の天気は？",
            "プログラミングを教えて",
            "量子コンピュータって何？",
            "あなたは誰ですか？",
        ]

        print("\nテスト生成:")
        print("-" * 60)

        for prompt in test_prompts:
            print(f"\n入力: {prompt}")
            response = generate_response(
                model, tokenizer, prompt,
                temperature=args.temperature,
                max_length=args.max_length
            )
            print(f"応答: {response}")

    print("\n" + "=" * 60)
    print("ヒント:")
    print("  - 文字化けが発生する場合: --temperature 0.3 を試す")
    print("  - より多様な応答が欲しい場合: --temperature 0.7 を試す")
    print("  - 対話モード: --interactive フラグを追加")
    print("=" * 60)


if __name__ == "__main__":
    main()
