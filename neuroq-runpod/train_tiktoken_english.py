#!/usr/bin/env python3
"""
TikToken + 英語データによるNeuroQモデル学習スクリプト

英語データでモデルを学習し、英語で思考するAIを作成します。
チャット時に翻訳(NLLB-200)を使って日本語と英語を変換します。

使い方:
    python train_tiktoken_english.py
"""

import os
import sys
import torch
import logging
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
print("🚀 TikToken + 英語データによる NeuroQ モデル学習")
print("=" * 70)

# インポート
try:
    from neuroquantum_brain import NeuroQuantumBrain
    from tiktoken_tokenizer import TikTokenTokenizer
    print("✅ モジュールをインポートしました")
except ImportError as e:
    print(f"❌ インポートに失敗: {e}")
    sys.exit(1)

# 設定
CHECKPOINT_PATH = "checkpoints/neuroq_tiktoken_english_checkpoint.pt"
ENCODING_NAME = "cl100k_base"  # GPT-4と同じエンコーディング

# モデル設定（tiktokenの語彙サイズに対応）
MODEL_CONFIG = {
    'vocab_size': 100277,  # cl100k_baseの語彙サイズ
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 3,
    'num_neurons': 100,
}

# 学習設定
TRAIN_CONFIG = {
    'epochs': 50000,    # 50,000エポックで学習
    'batch_size': 16,
    'lr': 0.001,
    'seq_length': 64,
}


def get_english_training_data():
    """英語の学習データを取得"""
    
    # 対話形式の英語データ
    conversations = [
        # 挨拶
        "<USER> Hello! <ASSISTANT> Hello! How can I help you today?",
        "<USER> Hi there! <ASSISTANT> Hi! Nice to meet you. What would you like to know?",
        "<USER> Good morning! <ASSISTANT> Good morning! I hope you're having a great day.",
        "<USER> How are you? <ASSISTANT> I'm doing well, thank you for asking! How about you?",
        "<USER> Nice to meet you! <ASSISTANT> Nice to meet you too! How can I assist you?",
        
        # AI関連
        "<USER> What is AI? <ASSISTANT> AI, or Artificial Intelligence, is technology that enables machines to simulate human intelligence. It includes learning, reasoning, and self-correction capabilities.",
        "<USER> Tell me about machine learning. <ASSISTANT> Machine learning is a subset of AI where computers learn from data without being explicitly programmed. It uses algorithms to find patterns and make predictions.",
        "<USER> What is deep learning? <ASSISTANT> Deep learning is a type of machine learning using neural networks with many layers. It excels at processing images, speech, and natural language.",
        "<USER> How does neural network work? <ASSISTANT> Neural networks are inspired by the human brain. They consist of layers of interconnected nodes that process and transform data to make predictions.",
        "<USER> What is natural language processing? <ASSISTANT> Natural Language Processing, or NLP, enables computers to understand, interpret, and generate human language. It powers chatbots, translation, and text analysis.",
        
        # 量子コンピューティング
        "<USER> What is quantum computing? <ASSISTANT> Quantum computing uses quantum mechanics principles to process information. Unlike classical computers using bits, quantum computers use qubits that can be in multiple states simultaneously.",
        "<USER> What are qubits? <ASSISTANT> Qubits are quantum bits, the basic unit of quantum information. Unlike classical bits that are either 0 or 1, qubits can exist in a superposition of both states.",
        "<USER> Explain quantum entanglement. <ASSISTANT> Quantum entanglement is a phenomenon where particles become correlated so that the state of one instantly influences the other, regardless of distance.",
        "<USER> What is superposition? <ASSISTANT> Superposition allows quantum particles to exist in multiple states at once. This enables quantum computers to process many possibilities simultaneously.",
        
        # プログラミング
        "<USER> What is Python? <ASSISTANT> Python is a popular programming language known for its simplicity and readability. It's widely used for web development, data science, AI, and automation.",
        "<USER> How do I learn programming? <ASSISTANT> Start with a beginner-friendly language like Python. Practice regularly, work on projects, and use online resources like tutorials and coding exercises.",
        "<USER> What is an algorithm? <ASSISTANT> An algorithm is a step-by-step set of instructions to solve a problem or complete a task. It's the foundation of computer programming.",
        "<USER> Explain object-oriented programming. <ASSISTANT> Object-oriented programming organizes code into objects that contain data and methods. It promotes reusability, modularity, and easier maintenance.",
        
        # 科学
        "<USER> What is physics? <ASSISTANT> Physics is the natural science that studies matter, energy, and their interactions. It explores everything from subatomic particles to the entire universe.",
        "<USER> Explain the theory of relativity. <ASSISTANT> Einstein's theory of relativity describes how space and time are interconnected. Special relativity deals with constant velocities, while general relativity explains gravity as curved spacetime.",
        "<USER> What is mathematics? <ASSISTANT> Mathematics is the study of numbers, quantities, shapes, and patterns. It's fundamental to science, engineering, finance, and many other fields.",
        "<USER> What is chemistry? <ASSISTANT> Chemistry is the science of matter and its transformations. It studies atoms, molecules, and how substances interact and change.",
        
        # 一般的な質問
        "<USER> Thank you! <ASSISTANT> You're welcome! Feel free to ask if you have more questions.",
        "<USER> Can you help me? <ASSISTANT> Of course! I'd be happy to help. What do you need assistance with?",
        "<USER> I don't understand. <ASSISTANT> No problem! Let me explain it in a different way. What part is confusing?",
        "<USER> That's interesting! <ASSISTANT> I'm glad you find it interesting! Would you like to learn more about it?",
        "<USER> Tell me more. <ASSISTANT> Sure! Let me provide more details on this topic.",
        
        # 追加の会話
        "<USER> What is the internet? <ASSISTANT> The internet is a global network of interconnected computers that communicate using standardized protocols. It enables information sharing, communication, and online services.",
        "<USER> How does a computer work? <ASSISTANT> Computers process information using a central processing unit (CPU) that executes instructions, memory that stores data, and input/output devices for interaction.",
        "<USER> What is data science? <ASSISTANT> Data science combines statistics, programming, and domain knowledge to extract insights from data. It involves data collection, analysis, and visualization.",
        "<USER> Explain cloud computing. <ASSISTANT> Cloud computing delivers computing services over the internet, including storage, processing, and software. It offers flexibility, scalability, and cost efficiency.",
        "<USER> What is cybersecurity? <ASSISTANT> Cybersecurity protects computer systems, networks, and data from digital attacks. It involves encryption, firewalls, and secure practices to prevent unauthorized access.",
    ]
    
    # 知識ベースの文章
    knowledge = [
        "Artificial intelligence is transforming how we live and work. It enables machines to perform tasks that previously required human intelligence.",
        "Machine learning algorithms can identify patterns in data and make predictions. They improve their performance over time through experience.",
        "Deep learning uses neural networks with multiple layers to process complex data. It has revolutionized image recognition and natural language processing.",
        "Quantum computers leverage quantum mechanical phenomena like superposition and entanglement. They can solve certain problems exponentially faster than classical computers.",
        "Python is one of the most popular programming languages in the world. Its simple syntax makes it ideal for beginners and experts alike.",
        "The internet has connected billions of people worldwide. It enables instant communication, information sharing, and global collaboration.",
        "Data science involves extracting knowledge from large datasets. It combines statistics, programming, and domain expertise to solve complex problems.",
        "Neural networks are inspired by the biological neural networks in the human brain. They consist of interconnected nodes that process information.",
        "Cybersecurity is essential in our digital world. It protects sensitive information from unauthorized access and cyber attacks.",
        "Cloud computing has transformed how businesses operate. It provides scalable, on-demand access to computing resources.",
        "Natural language processing enables computers to understand human language. It powers virtual assistants, translation services, and text analysis tools.",
        "Algorithms are the building blocks of computer programs. They provide step-by-step instructions for solving problems.",
        "Software engineering is the systematic approach to designing, developing, and maintaining software. It ensures quality and reliability.",
        "The theory of relativity changed our understanding of space, time, and gravity. Einstein's revolutionary ideas continue to influence physics today.",
        "Quantum mechanics describes the behavior of matter at the atomic and subatomic level. It reveals a world that defies classical intuition.",
    ]
    
    # すべてのデータを結合
    all_data = conversations + knowledge
    
    return all_data


def train_with_english():
    print("\n" + "=" * 50)
    print("📚 Step 1: トークナイザー初期化")
    print("=" * 50)
    
    tokenizer = TikTokenTokenizer(encoding_name=ENCODING_NAME)
    
    # テスト
    print("\n🔤 トークン化テスト:")
    test_texts = ["Hello world", "What is AI?", "Quantum computing"]
    for text in test_texts:
        tokens = tokenizer.encode(text)
        print(f"   '{text}' -> {len(tokens)} tokens")
    
    print("\n" + "=" * 50)
    print("📊 Step 2: 英語学習データ準備")
    print("=" * 50)
    
    texts = get_english_training_data()
    logger.info(f"学習データ: {len(texts)} サンプル")
    
    # データをトークン化
    print("\n🔄 データトークン化中...")
    all_tokens = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special=False)
        all_tokens.extend(tokens)
        if i < 3:
            print(f"   Sample {i}: '{text[:50]}...' -> {len(tokens)} tokens")
    
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
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"総パラメータ数: {total_params:,}")
    
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
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        # シーケンスをシャッフル
        import random
        random.shuffle(sequences)
        
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
                logits.reshape(-1, MODEL_CONFIG['vocab_size']),
                target_ids.reshape(-1)
            )
            
            # バックワードパス
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / max(batch_count, 1)
        print(f"   Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}")
        logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")
    
    print("\n   学習完了！")
    
    print("\n" + "=" * 50)
    print("💾 Step 5: チェックポイント保存")
    print("=" * 50)
    
    # ディレクトリを作成
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    
    # チェックポイントを保存
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': MODEL_CONFIG,
        'tokenizer': {
            'type': 'tiktoken',
            'encoding': ENCODING_NAME,
            'vocab_size': tokenizer.vocab_size,
        },
        'language': 'english',
    }
    
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"✅ チェックポイント保存完了: {CHECKPOINT_PATH}")
    
    print("\n" + "=" * 50)
    print("🧪 Step 6: テスト生成 (英語)")
    print("=" * 50)
    
    model.eval()
    test_prompts = [
        "What is AI?",
        "Hello, how are you?",
        "Tell me about quantum computing.",
    ]
    
    for prompt in test_prompts:
        try:
            input_ids = tokenizer.encode(prompt)
            generated = input_ids.copy()
            
            with torch.no_grad():
                for _ in range(50):
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
            print(f"\n   Input: '{prompt}'")
            print(f"   Output: '{output}'")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 完了！")
    print("=" * 70)
    print(f"""
次のステップ:
1. チェックポイントを確認: {CHECKPOINT_PATH}
2. チャットで使用: python chat.py --tokenizer tiktoken --translate
   （翻訳モードで日本語⇔英語変換）
""")


if __name__ == "__main__":
    train_with_english()
