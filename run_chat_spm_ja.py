#!/usr/bin/env python3
"""
NeuroQ QBNN - SentencePiece 日本語チャットデモ
================================================
先ほど学習した SentencePiece トークナイザー (`neuroq_tokenizer_oasst1_ja.model`)
を使用して、NeuroQuantum (QBNN) モデルとインタラクティブに対話するスクリプトです。

注意: モデル自体の重みはこの語彙サイズに対して学習されていない状態（ランダム初期化、
または簡易学習のみ）であるため、意味のある返答は生成されませんが、
チャットインターフェースと推論・生成のパイプラインとして動作します。
"""

import os
import sys
import torch
from pathlib import Path

# NeuroQモジュールへのパス追加
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
neuroq_dir = current_dir / 'neuroq-runpod'
if str(neuroq_dir) not in sys.path:
    sys.path.insert(0, str(neuroq_dir))

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

def main():
    print("=" * 70)
    print("🧠 NeuroQuantum ✖️ 日本語 SentencePiece トークナイザー チャット")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 1. トークナイザーの読み込み
    tokenizer_path = current_dir / "neuroq_tokenizer_oasst1_ja.model"
    if not tokenizer_path.exists():
        print(f"❌ トークナイザーが見つかりません: {tokenizer_path}")
        print("さきほど作成したトークナイザーモデルが必要です。")
        return

    print("\n[1] トークナイザーをロード中...")
    tokenizer = NeuroQuantumTokenizer(model_file=str(tokenizer_path))
    print(f"✅ ロード完了 (Vocab Size: {tokenizer.vocab_size})")

    # 2. モデルの初期化
    print("\n[2] NeuroQuantum (QBNN) モデルを初期化中...")
    config = NeuroQuantumConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        max_seq_len=256,
        lambda_entangle=0.5
    )
    model = NeuroQuantum(config).to(device)
    model.eval()
    print("✅ モデルの初期化完了 (学習前/ランダム重みです)")

    # (オプション) ここで簡単なダミー学習を数ステップ行うことで、
    # 完全にランダムな文字化けを防ぐこともできますが、今回は推論動作のみ確認します。

    print("\n" + "="*70)
    print("💬 チャットを開始します！ (終了するには 'quit' または 'exit' と入力)")
    print("="*70)

    while True:
        try:
            user_input = input("\n👤 あなた: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            # 入力エンコード
            input_ids = tokenizer.encode(user_input, add_special=True)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            print("🤖 NeuroQ: ", end="", flush=True)

            # 推論 (Generation loop)
            # max_new_tokens 回ループして1文字ずつ生成 (非常に単純な Greedy/Top-k デコード)
            current_ids = input_ids.copy()
            max_new_tokens = 50
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    seq_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
                    # seq_length が max_seq_len を超えないようにする
                    if seq_tensor.size(1) > config.max_seq_len:
                        seq_tensor = seq_tensor[:, -config.max_seq_len:]
                        
                    logits = model(seq_tensor)
                    next_token_logits = logits[0, -1, :]
                    
                    # Top-k sampling (簡易実装)
                    top_k = 40
                    temperature = 0.8
                    next_token_logits = next_token_logits / temperature
                    
                    values, indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(values, dim=-1)
                    next_token_idx = indices[torch.multinomial(probs, 1).item()].item()
                    
                    current_ids.append(next_token_idx)
                    
                    # 生成されたトークンを1文字ずつデコードして表示
                    # ※ SentencePieceのデコード仕様上、まとめてデコードした方が綺麗になりますが
                    # 簡易的に都度変換してみます
                    dec_str = tokenizer.decode([next_token_idx], skip_special=True)
                    print(dec_str, end="", flush=True)

                    # EOS トークンなどでストップさせるロジック (現状は特殊トークンマッピングがないため固定長まで回します)

            print() # 改行

        except KeyboardInterrupt:
            print("\nチャットを終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
