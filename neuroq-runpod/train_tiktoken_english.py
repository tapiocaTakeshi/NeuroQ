#!/usr/bin/env python3
"""
TikToken + OASST1英語データによるNeuroQモデル学習スクリプト

OASST1 (Open Assistant) データセットで対話モデルを学習します。
英語データでモデルを学習し、英語で思考するAIを作成します。
チャット時に翻訳(NLLB-200)を使って日本語と英語を変換します。

使い方:
    python train_tiktoken_english.py                    # Microモデル（デフォルト）
    python train_tiktoken_english.py --model-size small # Smallモデル
    python train_tiktoken_english.py -m large --epochs 20
"""

import os
import sys
import torch
import logging
import random
import argparse
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 現在のディレクトリをパスに追加
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=" * 70)
print("🚀 TikToken + OASST1英語データによる NeuroQ モデル学習")
print("=" * 70)

# インポート
try:
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    NEUROQUANTUM_AVAILABLE = True
    print("✅ NeuroQuantum (neuroquantum_layered.py) をインポートしました")
except ImportError as e:
    print(f"⚠️ neuroquantum_layered.py インポート失敗: {e}")
    NEUROQUANTUM_AVAILABLE = False

try:
    from neuroquantum_brain import NeuroQuantumBrain
    from tiktoken_tokenizer import TikTokenTokenizer
    print("✅ モジュールをインポートしました")
except ImportError as e:
    print(f"❌ インポートに失敗: {e}")
    sys.exit(1)

try:
    from datasets import load_dataset
    print("✅ Hugging Face datasets をインポートしました")
except ImportError:
    print("❌ datasets ライブラリが必要です: pip install datasets")
    sys.exit(1)

# モデル設定をインポート
try:
    from model_configs import AVAILABLE_MODELS, get_model_config, get_checkpoint_path
    MODEL_CONFIGS_AVAILABLE = True
    print("✅ model_configs.py をインポートしました")
except ImportError:
    MODEL_CONFIGS_AVAILABLE = False

# 設定
CHECKPOINT_PATH = "checkpoints/neuroq_tiktoken_english_checkpoint.pt"
ENCODING_NAME = "o200k_base"  # GPT-4oと同じエンコーディング

# モデル設定（tiktokenの語彙サイズに対応）
MODEL_CONFIG = {
    'vocab_size': 200019,  # o200k_baseの語彙サイズ
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 3,
    'num_neurons': 100,
}

# 学習設定
TRAIN_CONFIG = {
    'epochs': 30,         # QBNNの量子もつれパラメータ収束のため60エポック
    'batch_size': 8,      # メモリ節約
    'lr': 0.0005,         # 安定した学習率
    'seq_length': 128,    # より長いシーケンス
    'max_samples': 3000,  # 最大サンプル数
}


def load_oasst1_english_data(max_samples=5000):
    """
    OASST1データセットから英語の対話データを読み込み
    
    Returns:
        list: <USER>...<ASSISTANT>形式の対話リスト
    """
    print("\n📥 OASST1データセットをダウンロード中...")
    
    try:
        # OASST1データセットを読み込み
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        print(f"   総メッセージ数: {len(dataset):,}")
    except Exception as e:
        print(f"❌ データセットの読み込みに失敗: {e}")
        return []
    
    # メッセージをIDでインデックス化
    print("🔄 データ構造を解析中...")
    messages_by_id = {}
    for msg in dataset:
        messages_by_id[msg['message_id']] = msg
    
    # 英語のみフィルタ
    english_messages = [msg for msg in dataset if msg.get('lang') == 'en']
    print(f"   英語メッセージ数: {len(english_messages):,}")
    
    # 対話を構築（parent_idを使って会話ツリーを辿る）
    conversations = []
    
    # ルートメッセージ（prompter）を見つける
    root_messages = [msg for msg in english_messages 
                     if msg.get('parent_id') is None and msg.get('role') == 'prompter']
    
    print(f"   ルートメッセージ数: {len(root_messages):,}")
    
    for root in root_messages:
        if len(conversations) >= max_samples:
            break
            
        # 子メッセージを探す（assistant の返答）
        children = [msg for msg in english_messages 
                    if msg.get('parent_id') == root['message_id'] 
                    and msg.get('role') == 'assistant']
        
        if children:
            # 最初の返答を使用（もしくはランクが高いもの）
            children.sort(key=lambda x: x.get('rank', 999) or 999)
            response = children[0]
            
            user_text = root.get('text', '').strip()
            assistant_text = response.get('text', '').strip()
            
            if user_text and assistant_text and len(user_text) > 5 and len(assistant_text) > 10:
                # 長すぎるテキストはトリミング
                user_text = user_text[:500]
                assistant_text = assistant_text[:1000]
                
                conversation = f"<USER> {user_text} <ASSISTANT> {assistant_text}"
                conversations.append(conversation)
    
    print(f"✅ 対話数: {len(conversations):,}")
    
    # サンプルを表示
    if conversations:
        print("\n📋 サンプル対話:")
        for i, conv in enumerate(conversations[:2]):
            print(f"   {i+1}. {conv[:150]}...")
    
    return conversations


def get_fallback_training_data():
    """フォールバック用の英語学習データ"""
    
    conversations = [
        "<USER> Hello! <ASSISTANT> Hello! How can I help you today?",
        "<USER> What is AI? <ASSISTANT> AI, or Artificial Intelligence, is technology that enables machines to simulate human intelligence.",
        "<USER> Tell me about machine learning. <ASSISTANT> Machine learning is a subset of AI where computers learn from data without being explicitly programmed.",
        "<USER> What is deep learning? <ASSISTANT> Deep learning is a type of machine learning using neural networks with many layers.",
        "<USER> What is quantum computing? <ASSISTANT> Quantum computing uses quantum mechanics principles to process information using qubits.",
        "<USER> What are qubits? <ASSISTANT> Qubits are quantum bits that can exist in a superposition of both 0 and 1 states.",
        "<USER> What is Python? <ASSISTANT> Python is a popular programming language known for its simplicity and readability.",
        "<USER> How do I learn programming? <ASSISTANT> Start with a beginner-friendly language like Python. Practice regularly and work on projects.",
        "<USER> Thank you! <ASSISTANT> You're welcome! Feel free to ask if you have more questions.",
        "<USER> Can you help me? <ASSISTANT> Of course! I'd be happy to help. What do you need assistance with?",
    ]
    
    return conversations


def train_with_oasst1(epochs=None, model_size='micro'):
    # コマンドライン引数がない場合はデフォルト値を使用
    if epochs is not None:
        TRAIN_CONFIG['epochs'] = epochs
    
    # モデルサイズに応じて調整
    if model_size in ['small', 'large']:
        if TRAIN_CONFIG.get('batch_size', 8) > 4:
            TRAIN_CONFIG['batch_size'] = 4 if model_size == 'small' else 2
        TRAIN_CONFIG['lr'] = 0.0003  # 大きいモデルは学習率を下げる
    
    print(f"\n📦 モデルサイズ: {model_size.upper()}")
    
    print("\n" + "=" * 50)
    print("📚 Step 1: トークナイザー初期化")
    print("=" * 50)
    
    tokenizer = TikTokenTokenizer(encoding_name=ENCODING_NAME)
    
    print("\n" + "=" * 50)
    print("📊 Step 2: OASST1データセット準備")
    print("=" * 50)
    
    # OASST1データを読み込み
    texts = load_oasst1_english_data(max_samples=TRAIN_CONFIG['max_samples'])
    
    # データが少ない場合はフォールバック
    if len(texts) < 100:
        print("⚠️ データが少ないため、フォールバックデータを追加します")
        texts.extend(get_fallback_training_data())
    
    logger.info(f"学習データ: {len(texts)} サンプル")
    
    # データをトークン化
    print("\n🔄 データトークン化中...")
    all_tokens = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special=False)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)  # 各会話の終わりにEOS
        
        if i < 3:
            print(f"   Sample {i}: '{text[:80]}...' -> {len(tokens)} tokens")
        elif i == 3:
            print(f"   ... (残り {len(texts) - 3} サンプル)")
    
    logger.info(f"総トークン数: {len(all_tokens):,}")
    
    print("\n" + "=" * 50)
    print("🧠 Step 3: モデル構築")
    print("=" * 50)
    
    # デバイス選択
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) を使用")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 NVIDIA GPU (CUDA) を使用")
    else:
        device = torch.device("cpu")
        print("💻 CPU を使用")
    
    # モデル設定を取得
    if MODEL_CONFIGS_AVAILABLE and model_size in ['small', 'large'] and NEUROQUANTUM_AVAILABLE:
        model_config = get_model_config(model_size)
        checkpoint_path = get_checkpoint_path(model_size)
        
        print(f"\n📋 モデル設定 ({model_config['name']}):")
        print(f"   - vocab_size: {model_config['vocab_size']:,}")
        print(f"   - embed_dim: {model_config['embed_dim']:,}")
        print(f"   - num_heads: {model_config['num_heads']}")
        print(f"   - num_layers: {model_config['num_layers']}")
        
        config = NeuroQuantumConfig()
        config.vocab_size = model_config['vocab_size']
        config.embed_dim = model_config['embed_dim']
        config.num_heads = model_config['num_heads']
        config.num_layers = model_config['num_layers']
        config.max_seq_len = model_config.get('max_seq_len', 512)
        config.dropout = model_config.get('dropout', 0.1)
        
        model = NeuroQuantum(config).to(device)
        model_type = f"NeuroQuantum {model_config['name']}"
        used_model_config = model_config
        
    elif NEUROQUANTUM_AVAILABLE:
        # Microモデル（従来）
        print(f"\n📋 モデル設定:")
        for key, value in MODEL_CONFIG.items():
            print(f"   - {key}: {value:,}" if isinstance(value, int) else f"   - {key}: {value}")
        
        config = NeuroQuantumConfig()
        config.vocab_size = MODEL_CONFIG['vocab_size']
        config.embed_dim = MODEL_CONFIG['embed_dim']
        config.num_heads = MODEL_CONFIG['num_heads']
        config.num_layers = MODEL_CONFIG['num_layers']
        config.max_seq_len = 256
        config.dropout = 0.1
        
        model = NeuroQuantum(config).to(device)
        model_type = "NeuroQuantum (QBNN Transformer)"
        checkpoint_path = CHECKPOINT_PATH
        used_model_config = MODEL_CONFIG
        
    else:
        print(f"\n📋 モデル設定:")
        for key, value in MODEL_CONFIG.items():
            print(f"   - {key}: {value:,}" if isinstance(value, int) else f"   - {key}: {value}")
        
        model = NeuroQuantumBrain(
            vocab_size=MODEL_CONFIG['vocab_size'],
            embed_dim=MODEL_CONFIG['embed_dim'],
            num_heads=MODEL_CONFIG['num_heads'],
            num_layers=MODEL_CONFIG['num_layers'],
            num_neurons=MODEL_CONFIG['num_neurons'],
        ).to(device)
        model_type = "NeuroQuantumBrain"
        checkpoint_path = CHECKPOINT_PATH
        used_model_config = MODEL_CONFIG
    
    print(f"   ✅ {model_type} を使用")
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"総パラメータ数: {total_params:,} ({total_params/1e6:.1f}M)")
    
    print("\n" + "=" * 50)
    print("🎓 Step 4: 学習開始")
    print("=" * 50)
    
    print(f"\n📋 学習設定:")
    for key, value in TRAIN_CONFIG.items():
        print(f"   - {key}: {value}")
    
    # シーケンスを作成
    seq_length = TRAIN_CONFIG['seq_length']
    sequences = []
    for i in range(0, len(all_tokens) - seq_length, seq_length // 2):  # オーバーラップ
        sequences.append(all_tokens[i:i + seq_length])
    
    logger.info(f"シーケンス数: {len(sequences):,}")
    
    # 学習ループ
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    
    batch_size = TRAIN_CONFIG['batch_size']
    epochs = TRAIN_CONFIG['epochs']
    
    print("\n🚀 学習ループ開始...")
    
    # 総バッチ数を計算
    total_batches = (len(sequences) - batch_size) // batch_size
    print(f"   📊 エポックあたりのバッチ数: {total_batches:,}")
    print(f"   ⚠️  大きなモデルでは1バッチに数秒かかることがあります\n")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        # シーケンスをシャッフル
        random.shuffle(sequences)
        
        # 進捗表示間隔（10%ごと、最低10バッチごと）
        progress_interval = max(10, total_batches // 10)
        
        for i in range(0, len(sequences) - batch_size, batch_size):
            batch = sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch, device=device)
            
            # 入力と正解を作成
            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]
            
            # フォワードパス
            optimizer.zero_grad()
            logits = model(input_ids)
            
            # ロス計算
            loss = criterion(
                logits.reshape(-1, used_model_config['vocab_size']),
                target_ids.reshape(-1)
            )
            
            # バックワードパス
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # バッチ単位の進捗表示
            if batch_count % progress_interval == 0 or batch_count == 1:
                progress = batch_count / total_batches * 100
                current_avg_loss = total_loss / batch_count
                print(f"   Epoch {epoch + 1}/{epochs} - Batch {batch_count:,}/{total_batches:,} ({progress:.0f}%) - Loss: {current_avg_loss:.4f}")
        
        avg_loss = total_loss / max(batch_count, 1)
        
        # エポック完了時の表示
        print(f"   ✅ Epoch {epoch + 1}/{epochs} 完了: Loss={avg_loss:.4f}")
        logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")
        
        # ベストモデルを更新
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    print(f"\n   ✅ 学習完了！ベストロス: {best_loss:.4f}")
    
    print("\n" + "=" * 50)
    print("💾 Step 5: チェックポイント保存")
    print("=" * 50)
    
    # ディレクトリを作成
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # チェックポイントを保存
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': used_model_config,
        'tokenizer': {
            'type': 'tiktoken',
            'encoding': ENCODING_NAME,
            'vocab_size': tokenizer.vocab_size,
        },
        'language': 'english',
        'dataset': 'oasst1',
        'training_samples': len(texts),
        'model_size': model_size,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ チェックポイント保存完了: {checkpoint_path}")
    
    print("\n" + "=" * 50)
    print("🧪 Step 6: テスト生成 (英語)")
    print("=" * 50)
    
    model.eval()
    test_prompts = [
        "<USER> What is AI? <ASSISTANT>",
        "<USER> Hello, how are you? <ASSISTANT>",
        "<USER> Tell me about quantum computing. <ASSISTANT>",
        "<USER> Can you help me learn programming? <ASSISTANT>",
    ]
    
    for prompt in test_prompts:
        try:
            input_ids = tokenizer.encode(prompt)
            generated = input_ids.copy()
            
            with torch.no_grad():
                for _ in range(80):
                    seq_tensor = torch.tensor([generated[-256:]], device=device)
                    logits = model(seq_tensor)
                    
                    probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)
                    
                    # Top-k sampling
                    top_k = 50
                    top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                    top_probs = top_probs / top_probs.sum()
                    idx = torch.multinomial(top_probs, num_samples=1)
                    next_token = top_indices[idx].item()
                    
                    if next_token == tokenizer.eos_id:
                        break
                    generated.append(next_token)
            
            output = tokenizer.decode(generated, skip_special=True)
            # <ASSISTANT>以降を抽出
            if '<ASSISTANT>' in output:
                response = output.split('<ASSISTANT>')[-1].strip()
            else:
                response = output
            
            # <USER>以降を削除
            if '<USER>' in response:
                response = response.split('<USER>')[0].strip()
            
            print(f"\n   Q: '{prompt.replace('<USER> ', '').replace(' <ASSISTANT>', '')}'")
            print(f"   A: '{response[:200]}'")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 完了！")
    print("=" * 70)
    print(f"""
次のステップ:
1. チェックポイントを確認: {checkpoint_path}
2. チャットで使用: python chat.py --model-size {model_size} --translate
   （翻訳モードで日本語⇔英語変換）
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='TikToken + OASST1英語データによるNeuroQモデル学習'
    )
    parser.add_argument(
        '--model-size', '-m',
        choices=['micro', 'small', 'large'],
        default='micro',
        help='モデルサイズ: micro(128), small(1536), large(3072)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help=f'学習エポック数 (デフォルト: {TRAIN_CONFIG["epochs"]})'
    )
    args = parser.parse_args()
    
    train_with_oasst1(epochs=args.epochs, model_size=args.model_size)
