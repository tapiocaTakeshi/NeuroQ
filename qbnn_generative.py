#!/usr/bin/env python3
"""
QBNN Generative AI - 独自のQuantum-Bit Neural Networkを使った生成AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import Counter

print("=" * 70)
print("🧠⚛️ QBNN Generative AI - 独自の量子ビットNNで生成AI")
print("=" * 70)


# ========================================
# QBNNレイヤー（独自）
# ========================================

class QBNNLayer(nn.Module):
    """
    Quantum-Bit Neural Network Layer
    
    数式モデル:
    1. s^(l) = tanh(h^(l)) ∈ [-1, 1]  (正規化)
    2. θ^(l)_i = arccos(s^(l)_i)       (Bloch角)
    3. h̃^(l+1) = W^(l) h^(l) + b^(l)  (線形変換)
    4. Δ^(l+1)_j = Σ_i J^(l)_{ij} s^(l)_i s^(l+1)_{raw,j}  (もつれ補正)
    5. ĥ^(l+1) = h̃^(l+1) + λ^(l) Δ^(l+1)  (有効入力)
    6. h^(l+1) = tanh(ĥ^(l+1))  (活性化)
    """
    
    def __init__(self, input_dim, output_dim, lambda_val=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 通常の重みとバイアス
        self.W = nn.Linear(input_dim, output_dim)
        
        # もつれテンソル J（独自）
        self.J = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        
        # もつれ強度 λ（独自）
        self.lambda_val = nn.Parameter(torch.tensor(float(lambda_val)))
    
    def forward(self, h_prev):
        # 1. 正規化
        s_prev = torch.tanh(h_prev)
        
        # 2. 線形変換
        h_tilde = self.W(h_prev)
        
        # 3. 正規化（次層の候補）
        s_raw = torch.tanh(h_tilde)
        
        # 4. もつれ補正項 Δ
        # Δ_j = Σ_i J_{ij} s^(l)_i s^(l+1)_{raw,j}
        delta = torch.einsum('...i,ij,...j->...j', s_prev, self.J, s_raw)
        
        # 5. 有効入力
        h_hat = h_tilde + self.lambda_val * delta
        
        # 6. 活性化
        return torch.tanh(h_hat)
    
    def get_entanglement_info(self):
        """もつれ情報を取得"""
        with torch.no_grad():
            j_mean = self.J.mean().item()
            j_std = self.J.std().item()
            lam = self.lambda_val.item()
        return {'lambda': lam, 'J_mean': j_mean, 'J_std': j_std}


# ========================================
# QBNN言語モデル
# ========================================

class QBNNLanguageModel(nn.Module):
    """QBNNベースの言語モデル"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_layers=3, lambda_val=0.5, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        
        # QBNNレイヤー（独自アーキテクチャ）
        self.qbnn_layers = nn.ModuleList()
        
        # 入力層
        self.qbnn_layers.append(QBNNLayer(embed_dim, hidden_dim, lambda_val))
        
        # 中間層
        for _ in range(num_layers - 2):
            self.qbnn_layers.append(QBNNLayer(hidden_dim, hidden_dim, lambda_val))
        
        # 出力層
        self.qbnn_layers.append(QBNNLayer(hidden_dim, embed_dim, lambda_val))
        
        # 出力プロジェクション
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        print(f"\n📊 QBNNLanguageModel 構成:")
        print(f"   語彙サイズ: {vocab_size}")
        print(f"   埋め込み次元: {embed_dim}")
        print(f"   隠れ層次元: {hidden_dim}")
        print(f"   QBNNレイヤー数: {num_layers}")
        print(f"   もつれ強度 λ: {lambda_val}")
    
    def forward(self, x):
        # x: (batch, seq_len)
        
        # 埋め込み
        h = self.embedding(x)  # (batch, seq_len, embed_dim)
        h = self.pos_encoding(h)
        
        # QBNNレイヤーを通す
        for layer in self.qbnn_layers:
            h = self.dropout(layer(h))
        
        # 出力
        logits = self.output_proj(h)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def get_entanglement_info(self):
        """全層のもつれ情報"""
        info = []
        for i, layer in enumerate(self.qbnn_layers):
            layer_info = layer.get_entanglement_info()
            info.append(f"Layer {i}: λ={layer_info['lambda']:.3f}")
        return info


class PositionalEncoding(nn.Module):
    """位置エンコーディング"""
    
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ========================================
# トークナイザー
# ========================================

class CharTokenizer:
    """文字レベルトークナイザー"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        # 特殊トークン
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
    
    def build_vocab(self, texts):
        """語彙を構築"""
        # 特殊トークン
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # 全文字を収集
        chars = set()
        for text in texts:
            chars.update(text)
        
        # 辞書作成
        all_tokens = special_tokens + sorted(list(chars))
        self.char_to_idx = {c: i for i, c in enumerate(all_tokens)}
        self.idx_to_char = {i: c for i, c in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)
        
        print(f"   語彙サイズ: {self.vocab_size}")
        return self
    
    def encode(self, text):
        """テキストをトークン列に変換"""
        tokens = [self.char_to_idx.get(self.bos_token)]
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.char_to_idx[self.unk_token]))
        tokens.append(self.char_to_idx.get(self.eos_token))
        return tokens
    
    def decode(self, tokens):
        """トークン列をテキストに変換"""
        chars = []
        for t in tokens:
            char = self.idx_to_char.get(t, self.unk_token)
            if char not in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]:
                chars.append(char)
        return ''.join(chars)


# ========================================
# 生成AI
# ========================================

class QBNNGenerativeAI:
    """QBNN生成AI"""
    
    def __init__(self, embed_dim=128, hidden_dim=256, num_layers=3, 
                 lambda_val=0.5, dropout=0.1):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lambda_val = lambda_val
        self.dropout = dropout
        
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n🔧 デバイス: {self.device}")
    
    def train(self, texts, epochs=100, batch_size=32, lr=0.001, seq_len=64):
        """学習"""
        print("\n" + "=" * 50)
        print("📚 学習開始")
        print("=" * 50)
        
        # トークナイザー構築
        print("\n🔤 トークナイザー構築...")
        self.tokenizer = CharTokenizer()
        self.tokenizer.build_vocab(texts)
        
        # モデル構築
        print("\n🧠 QBNNモデル構築...")
        self.model = QBNNLanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            lambda_val=self.lambda_val,
            dropout=self.dropout
        ).to(self.device)
        
        # データ準備
        print("\n📊 データ準備...")
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"   総トークン数: {len(all_tokens)}")
        
        # データセット作成
        sequences = []
        for i in range(0, len(all_tokens) - seq_len - 1, seq_len // 2):
            x = all_tokens[i:i+seq_len]
            y = all_tokens[i+1:i+seq_len+1]
            if len(x) == seq_len and len(y) == seq_len:
                sequences.append((x, y))
        
        print(f"   シーケンス数: {len(sequences)}")
        
        # 学習
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        print("\n🚀 学習ループ...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(sequences)
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                if len(batch) == 0:
                    continue
                
                x_batch = torch.stack([s[0] for s in batch]).to(self.device)
                y_batch = torch.stack([s[1] for s in batch]).to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_batch)
                
                # (batch, seq, vocab) -> (batch * seq, vocab)
                loss = criterion(
                    logits.view(-1, self.tokenizer.vocab_size),
                    y_batch.view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / max(1, len(sequences) // batch_size)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")
        
        print("\n✅ 学習完了！")
        
        # もつれ情報
        print("\n⚛️ 量子もつれ情報:")
        for info in self.model.get_entanglement_info():
            print(f"   {info}")
    
    def generate(self, prompt="", max_length=100, temperature=1.0, 
                 top_k=50, top_p=0.9, repetition_penalty=1.2):
        """テキスト生成"""
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        self.model.eval()
        
        # プロンプトをエンコード
        if prompt:
            tokens = self.tokenizer.encode(prompt)[:-1]  # EOSを除く
        else:
            tokens = [self.tokenizer.char_to_idx[self.tokenizer.bos_token]]
        
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        generated = tokens[0].tolist()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 最後の64トークンを使用
                input_tokens = tokens[:, -64:] if tokens.size(1) > 64 else tokens
                
                logits = self.model(input_tokens)
                next_logits = logits[0, -1, :]  # 最後のトークンの予測
                
                # 温度調整
                next_logits = next_logits / temperature
                
                # 繰り返しペナルティ
                for token_id in set(generated[-20:]):
                    next_logits[token_id] /= repetition_penalty
                
                # Top-K
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Top-P (Nucleus)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = float('-inf')
                
                # サンプリング
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # EOS検出
                if next_token.item() == self.tokenizer.char_to_idx[self.tokenizer.eos_token]:
                    break
                
                generated.append(next_token.item())
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        
        return self.tokenizer.decode(generated)
    
    def chat(self):
        """対話モード"""
        print("\n" + "=" * 50)
        print("💬 QBNN チャットモード")
        print("=" * 50)
        print("コマンド:")
        print("  /quit  - 終了")
        print("  /temp <値> - 温度設定 (0.1-1.0)")
        print("  /len <値> - 生成長さ (10-500)")
        print("  /info - モデル情報")
        print("-" * 50)
        
        temperature = 1.0
        max_length = 100
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == '/quit':
                    print("👋 さようなら！")
                    break
                
                if user_input.startswith('/temp '):
                    try:
                        temperature = float(user_input.split()[1])
                        temperature = max(0.1, min(1.0, temperature))
                        print(f"   温度を {temperature} に設定")
                    except:
                        print("   エラー: /temp <数値> (0.1-1.0)")
                    continue
                
                if user_input.startswith('/len '):
                    try:
                        max_length = int(user_input.split()[1])
                        max_length = max(10, min(500, max_length))
                        print(f"   生成長さを {max_length} に設定")
                    except:
                        print("   エラー: /len <数値>")
                    continue
                
                if user_input == '/info':
                    print(f"\n📊 モデル情報:")
                    print(f"   語彙サイズ: {self.tokenizer.vocab_size}")
                    print(f"   埋め込み次元: {self.embed_dim}")
                    print(f"   隠れ層次元: {self.hidden_dim}")
                    print(f"   QBNNレイヤー数: {self.num_layers}")
                    print(f"   もつれ強度 λ: {self.lambda_val}")
                    print(f"\n⚛️ 量子もつれ情報:")
                    for info in self.model.get_entanglement_info():
                        print(f"   {info}")
                    continue
                
                # 生成
                print(f"\n🤖 QBNN: ", end="", flush=True)
                response = self.generate(
                    prompt=user_input,
                    max_length=max_length,
                    temperature=temperature
                )
                
                # プロンプト部分を除去
                if response.startswith(user_input):
                    response = response[len(user_input):]
                
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 中断されました")
                break
            except Exception as e:
                print(f"   エラー: {e}")


# ========================================
# デモ用学習データ
# ========================================

def get_demo_texts():
    """デモ用テキスト"""
    return [
        # 日本語
        "量子コンピュータは、量子力学の原理を利用して計算を行うコンピュータです。",
        "量子ビットは、0と1の重ね合わせ状態を持つことができます。",
        "量子もつれは、二つの量子ビットが強く相関している状態です。",
        "ニューラルネットワークは、脳の神経細胞を模倣した計算モデルです。",
        "深層学習は、多層のニューラルネットワークを使った機械学習です。",
        "人工知能は、人間の知能を模倣するコンピュータシステムです。",
        "機械学習は、データからパターンを学習するアルゴリズムです。",
        "自然言語処理は、人間の言語を理解し生成する技術です。",
        "トランスフォーマーは、注意機構を使った深層学習モデルです。",
        "生成AIは、新しいコンテンツを作成する人工知能です。",
        "QBNNは、量子ビットニューラルネットワークの略称です。",
        "APQBは、調整可能擬似量子ビットの略称です。",
        "相関係数は、二つの変数の関係の強さを表す指標です。",
        "ブロッホ球は、量子ビットの状態を視覚化する方法です。",
        "エンタングルメントは、量子もつれとも呼ばれます。",
        
        # 英語
        "Quantum computing uses quantum mechanics for computation.",
        "A qubit can exist in superposition of 0 and 1.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple layers of neural networks.",
        "Artificial intelligence aims to mimic human intelligence.",
        "Machine learning algorithms learn patterns from data.",
        "Natural language processing understands human language.",
        "Transformers use attention mechanisms for learning.",
        "Generative AI creates new content automatically.",
        "QBNN stands for Quantum-Bit Neural Network.",
    ]


# ========================================
# メイン
# ========================================

if __name__ == "__main__":
    # 生成AI作成
    ai = QBNNGenerativeAI(
        embed_dim=64,
        hidden_dim=128,
        num_layers=4,
        lambda_val=0.5,
        dropout=0.1
    )
    
    # デモテキストで学習
    texts = get_demo_texts()
    ai.train(texts, epochs=100, batch_size=8, lr=0.002, seq_len=32)
    
    # テスト生成
    print("\n" + "=" * 50)
    print("🎨 テキスト生成テスト")
    print("=" * 50)
    
    prompts = [
        "量子",
        "Neural",
        "深層",
        "AI",
        "QBNN"
    ]
    
    for prompt in prompts:
        print(f"\n📝 プロンプト: '{prompt}'")
        result = ai.generate(prompt, max_length=50, temperature=0.8)
        print(f"   生成: {result}")
    
    # チャットモード
    print("\n" + "=" * 50)
    print("チャットモードを開始しますか？ (y/n)")
    print("=" * 50)
    
    try:
        answer = input().strip().lower()
        if answer == 'y':
            ai.chat()
    except:
        pass
    
    print("\n✅ 完了！")

