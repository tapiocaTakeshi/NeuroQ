#!/usr/bin/env python3
"""
OASST1 & OASST2 を使った NeuroQ モデルのローカル学習および Modal への自動アップロードスクリプト
"""

import os
import sys
import torch
import logging
import random
import argparse
import subprocess
from pathlib import Path
from datasets import load_dataset

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# パス設定
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    from tiktoken_tokenizer import TikTokenTokenizer
    from model_configs import AVAILABLE_MODELS, get_model_config
    logger.info("✅ コンポーネントのインポートに成功しました")
except ImportError as e:
    logger.error(f"❌ インポート失敗: {e}")
    sys.exit(1)

# 定数
ENCODING_NAME = "o200k_base"
MODAL_VOLUME_NAME = "neuroq-checkpoints"

def load_oasst_data(dataset_id="OpenAssistant/oasst1", max_samples=10000):
    """
    Open Assistant データセットから対話データを読み込み
    """
    logger.info(f"📥 {dataset_id} をロード中...")
    try:
        dataset = load_dataset(dataset_id, split="train")
    except Exception as e:
        logger.error(f"❌ {dataset_id} のロードに失敗: {e}")
        return []

    # 英語と日本語のメッセージを抽出
    messages = [msg for msg in dataset if msg.get('lang') in ['en', 'ja']]
    logger.info(f"   {dataset_id} (en/ja) メッセージ数: {len(messages):,}")

    # 簡易的な会話ペアの構築
    messages_by_id = {msg['message_id']: msg for msg in messages}
    conversations = []
    
    for msg in messages:
        if msg.get('role') == 'assistant' and msg.get('parent_id') in messages_by_id:
            parent = messages_by_id[msg['parent_id']]
            if parent.get('role') == 'prompter':
                user_text = parent.get('text', '').strip()
                assistant_text = msg.get('text', '').strip()
                if user_text and assistant_text:
                    # シンプルな形式に統合
                    conversations.append(f"{user_text}\n{assistant_text}")
    
    random.shuffle(conversations)
    return conversations[:max_samples]

def train_model(model_size, epochs, max_samples=10000, epoch_mode="early_stop"):
    """
    特定のサイズのモデルを学習
    """
    mode_label = "早期終了あり" if epoch_mode == "early_stop" else f"強制 {epochs} エポック"
    logger.info(f"\n🚀 {model_size.upper()} モデルの学習を開始します (Epochs: {epochs}, モード: {mode_label})")
    
    # 1. データ準備
    texts_1 = load_oasst_data("OpenAssistant/oasst1", max_samples=max_samples // 2)
    texts_2 = load_oasst_data("OpenAssistant/oasst2", max_samples=max_samples // 2)
    all_texts = texts_1 + texts_2
    logger.info(f"✅ 合計学習サンプル数: {len(all_texts):,}")

    if not all_texts:
        logger.error("❌ 学習データがありません")
        return None

    # 2. トークナイザーとデータ変換
    tokenizer = TikTokenTokenizer(encoding_name=ENCODING_NAME)
    all_tokens = []
    for text in all_texts:
        tokens = tokenizer.encode(text, add_special=False)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    
    logger.info(f"✅ 総トークン数: {len(all_tokens):,}")

    # 3. モデル構築
    config_dict = get_model_config(model_size)
    config = NeuroQuantumConfig()
    config.vocab_size = config_dict['vocab_size']
    config.embed_dim = config_dict['embed_dim']
    config.num_heads = config_dict['num_heads']
    config.num_layers = config_dict['num_layers']
    config.max_seq_len = config_dict.get('max_seq_len', 512)
    
    # デバイス選択 (MPS prioritize for Mac)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = NeuroQuantum(config).to(device)
    logger.info(f"✅ モデル構築完了 ({model_size}): {sum(p.numel() for p in model.parameters()):,} params")

    # 4. 学習設定
    seq_length = 128
    batch_size = 4 if model_size == 'micro' else 2
    lr = 0.0001
    
    # データをシーケンスに分割
    all_sequences = [all_tokens[i:i + seq_length] for i in range(0, len(all_tokens) - seq_length, seq_length)]
    random.shuffle(all_sequences)
    
    # 検証データの分割 (5%)
    val_size = max(1, int(len(all_sequences) * 0.05))
    train_sequences = all_sequences[:-val_size]
    val_sequences = all_sequences[-val_size:]
    
    logger.info(f"✅ 学習シーケンス数: {len(train_sequences):,}, 検証シーケンス数: {len(val_sequences):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 早期終了（Early Stopping）の設定
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    min_delta = 0.01
    
    # 5. 学習ループ
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        random.shuffle(train_sequences)
        
        # Training Phase
        for i in range(0, len(train_sequences) - batch_size, batch_size):
            batch = train_sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch, device=device)
            
            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, config.vocab_size), target_ids.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (i // batch_size) % 100 == 0:
                logger.info(f"   Epoch {epoch+1}/{epochs} | Batch {i//batch_size} | Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / (len(train_sequences) // batch_size)
        
        # Validation Phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_sequences) - batch_size, batch_size):
                batch = val_sequences[i:i + batch_size]
                batch_tensor = torch.tensor(batch, device=device)
                
                input_ids = batch_tensor[:, :-1]
                target_ids = batch_tensor[:, 1:]
                
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, config.vocab_size), target_ids.reshape(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / max(1, (len(val_sequences) // batch_size))
        logger.info(f"⭐ Epoch {epoch+1} 完了 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- 定点テスト生成 ---
        model.eval()
        test_prompts = ["こんにちは、元気ですか？", "What is AI?", "Pythonでfor文を書いて"]
        logger.info("📝 定点テスト生成:")
        for prompt in test_prompts:
            with torch.no_grad():
                input_ids = tokenizer.encode(prompt, add_special=False)
                generated = input_ids.copy()
                for _ in range(50):
                    seq = torch.tensor([generated[-128:]], device=device)
                    logits = model(seq)
                    next_token = torch.argmax(logits[0, -1, :]).item()
                    if next_token == tokenizer.eos_id: break
                    generated.append(next_token)
                response = tokenizer.decode(generated[len(input_ids):], skip_special=True)
                logger.info(f"   Q: {prompt} | A: {response.strip()[:100]}")

        # 早期終了の判定と保存
        is_best = False
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            is_best = True
            logger.info(f"   ✨ Best Loss 更新: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"   ⏳ 改善なし (Patience: {patience_counter}/{patience})")

        # 毎エポック保存（最新版）
        save_checkpoint(model, config_dict, model_size, f"neuroq_{model_size}_last.pt")
        # ベスト版を更新
        if is_best:
            save_checkpoint(model, config_dict, model_size, f"neuroq_{model_size}_best.pt")

        if epoch_mode == "early_stop" and patience_counter >= patience:
            logger.info("🛑 早期終了: ロスが安定したため学習を停止します。")
            break
        elif epoch_mode == "fixed" and patience_counter >= patience:
            logger.info(f"   ℹ️ 改善なしが {patience} 回続いていますが、強制モードのため学習を継続します。")

    # 学習完了後は best を正規のチェックポイント名としてアップロード対象にする
    best_path = Path("checkpoints") / f"neuroq_{model_size}_best.pt"
    # Modal用には標準名でもコピーしておく
    std_path = Path("checkpoints") / f"neuroq_{model_size}_checkpoint.pt"
    import shutil
    if best_path.exists():
        shutil.copy(best_path, std_path)
    
    return std_path

def save_checkpoint(model, config_dict, model_size, filename):
    path = Path("checkpoints") / filename
    path.parent.mkdir(exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config_dict,
        'model_size': model_size,
        'tokenizer': {'encoding': ENCODING_NAME}
    }
    torch.save(checkpoint, path)

def upload_to_modal(local_path):
    """
    ファイルを Modal ボリュームにアップロード
    """
    remote_name = local_path.name
    logger.info(f"📤 Modal へアップロード中: {remote_name}...")
    
    cmd = ["modal", "volume", "put", MODAL_VOLUME_NAME, str(local_path), f"/{remote_name}"]
    try:
        subprocess.run(cmd, check=True)
        # best と last もアップロードしたい場合はここに追加
        base_name = local_path.stem.replace("_checkpoint", "")
        for suffix in ["_best.pt", "_last.pt"]:
            extra_file = local_path.parent / f"{base_name}{suffix}"
            if extra_file.exists():
                subprocess.run(["modal", "volume", "put", MODAL_VOLUME_NAME, str(extra_file), f"/{extra_file.name}"], check=True)
        
        logger.info(f"✅ アップロード成功: {remote_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ アップロード失敗: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--sizes", nargs="+", default=["micro", "small", "large"])
    parser.add_argument("--epoch-mode", choices=["early_stop", "fixed"], default="early_stop",
                        help="early_stop: 改善なしで打ち切り（デフォルト） / fixed: 指定エポック数を強制的に回す")
    args = parser.parse_args()

    for size in args.sizes:
        path = train_model(size, args.epochs, args.samples, epoch_mode=args.epoch_mode)
        if path:
            upload_to_modal(path)
    
    logger.info("\n🎉 全ての工程が完了しました！")
