#!/usr/bin/env python3
"""
Entangled Quantum Bit Neural Network (E-QBNN)
層間エンタングルメントを持つ量子ビットニューラルネットワーク

論文 + フィードバックに基づく実装:
1. 各層に量子状態 |ψ^(l)⟩ を持たせる
2. 層間のエンタングル演算 U を定義
3. もつれ項: e^(l) = f_entangle(q^(l), q^(l-1))
4. 次層入力: h^(l+1) = σ(W^(l) h^(l) + B^(l) + G(e^(l)))
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import requests
import gzip
import io
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🧠⚛️ Entangled Quantum Bit Neural Network (E-QBNN)")
print("   層間エンタングルメントを持つ量子生成AI")
print("=" * 70)

# ========================================================================
# 1. APQB コア（論文の数学的定義）
# ========================================================================

class APQB:
    """Adjustable Pseudo Quantum Bit - 論文の核心"""
    
    @staticmethod
    def theta_to_state(theta):
        """θ → 量子状態 [cos(θ), sin(θ)]"""
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    
    @staticmethod
    def theta_to_r(theta):
        """θ → 相関係数 r = cos(2θ)"""
        return torch.cos(2 * theta)
    
    @staticmethod
    def theta_to_T(theta):
        """θ → 温度 T = |sin(2θ)|"""
        return torch.abs(torch.sin(2 * theta))
    
    @staticmethod
    def theta_to_z(theta):
        """θ → 複素数 z = e^{i2θ} (実部, 虚部)"""
        return torch.stack([torch.cos(2 * theta), torch.sin(2 * theta)], dim=-1)
    
    @staticmethod
    def constraint(theta):
        """r² + T² = 1 の検証"""
        r = APQB.theta_to_r(theta)
        T = APQB.theta_to_T(theta)
        return r**2 + T**2
    
    @staticmethod
    def Q_k(theta, k):
        """k体相関 Q_k(θ)"""
        if k % 2 == 0:
            return torch.cos(2 * k * theta)
        else:
            return torch.sin(2 * k * theta)


# ========================================================================
# 2. 層間エンタングルメント
# ========================================================================

class EntanglementOperator(nn.Module):
    """
    層間エンタングル演算 U
    
    e^(l) = f_entangle(q^(l), q^(l-1))
    
    実装:
    - 相関行列ベースのもつれ
    - CNOTライクな相互作用
    - 位相キックバック
    """
    
    def __init__(self, current_dim, prev_dim=None, entangle_strength=0.5):
        super().__init__()
        self.current_dim = current_dim
        self.prev_dim = prev_dim if prev_dim else current_dim
        self.entangle_strength = nn.Parameter(torch.tensor(entangle_strength))
        
        # エンタングル重み（異なる次元間を接続）
        self.W_entangle = nn.Linear(self.prev_dim, current_dim)
        
        # 位相パラメータ
        self.phase = nn.Parameter(torch.rand(current_dim) * np.pi / 2)
    
    def forward(self, q_current, q_prev):
        """
        層間エンタングル計算
        
        Args:
            q_current: 現在の層の量子状態 [batch, current_dim]
            q_prev: 前の層の量子状態 [batch, prev_dim]
        
        Returns:
            エンタングルメント項 e [batch, current_dim]
        """
        # 1. 前の層の状態を現在の次元に変換
        q_prev_transformed = self.W_entangle(q_prev)
        
        # 2. 要素ごとの相関（CNOTライク）
        # 制御ビット（前の層）と現在の層の相互作用
        correlation = q_current * torch.tanh(q_prev_transformed)
        
        # 3. 位相キックバック
        phase_factor = torch.cos(self.phase.unsqueeze(0))
        
        # 4. 最終エンタングル項
        e = self.entangle_strength * correlation * phase_factor
        
        return e


class QuantumCorrelationMatrix(nn.Module):
    """
    量子相関行列の計算
    
    h^(l) から相関行列 R^(l) を作成し、APQB状態ベクトル q^(l) を生成
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.theta_proj = nn.Linear(dim, dim)
    
    def forward(self, h):
        """
        隠れ状態 → 量子状態ベクトル
        
        Args:
            h: 隠れ状態 [batch, dim]
        
        Returns:
            q: 量子状態ベクトル [batch, dim]
            theta: 内部パラメータ [batch, dim]
        """
        # 1. θ の計算（0〜π/2 に制限）
        theta = torch.sigmoid(self.theta_proj(h)) * np.pi / 2
        
        # 2. 量子状態（r 成分をベクトルとして使用）
        r = APQB.theta_to_r(theta)
        
        return r, theta


# ========================================================================
# 3. E-QBNN レイヤー
# ========================================================================

class EQBNNLayer(nn.Module):
    """
    Entangled QBNN Layer
    
    h^(l+1) = σ(W^(l) h^(l) + B^(l) + G(e^(l)))
    
    - 通常の線形変換
    - 量子もつれからの補正
    - 幾何学的制約の正則化
    """
    
    def __init__(self, input_dim, output_dim, prev_output_dim=None, entangle_strength=0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_output_dim = prev_output_dim if prev_output_dim else input_dim
        
        # 線形変換
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 量子相関
        self.quantum_corr = QuantumCorrelationMatrix(output_dim)
        
        # エンタングルメント演算子（異なる次元間）
        self.entangle_op = EntanglementOperator(output_dim, self.prev_output_dim, entangle_strength)
        
        # エンタングル補正の変換 G
        self.G = nn.Linear(output_dim, output_dim)
        
        # 量子状態保持
        self.q = None
        self.theta = None
    
    def forward(self, h, q_prev=None):
        """
        順伝播
        
        Args:
            h: 入力 [batch, input_dim]
            q_prev: 前の層の量子状態 [batch, output_dim] or None
        
        Returns:
            h_out: 出力 [batch, output_dim]
            q: この層の量子状態 [batch, output_dim]
        """
        # 1. 通常の線形変換
        h_linear = self.linear(h)
        
        # 2. 量子状態の計算
        self.q, self.theta = self.quantum_corr(h_linear)
        
        # 3. エンタングルメント補正
        if q_prev is not None:
            e = self.entangle_op(self.q, q_prev)
            entangle_correction = self.G(e)
        else:
            entangle_correction = 0
        
        # 4. 出力 = 線形変換 + エンタングル補正
        h_out = torch.tanh(h_linear + entangle_correction)
        
        return h_out, self.q
    
    def get_constraint_loss(self):
        """幾何学的制約 r² + T² = 1 の損失"""
        if self.theta is None:
            return 0
        constraint = APQB.constraint(self.theta)
        return ((constraint - 1) ** 2).mean()


# ========================================================================
# 4. E-QBNN 生成モデル
# ========================================================================

class EQBNNGenerativeModel(nn.Module):
    """
    Entangled QBNN 生成モデル
    
    特徴:
    - 層間エンタングルメント
    - 量子サンプリング
    - 幾何学的制約による正則化
    """
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dims=[256, 256, 256], 
                 entangle_strength=0.5, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.entangle_strength = entangle_strength
        
        # 埋め込み
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(512, embed_dim) * 0.02)
        
        # E-QBNNレイヤー
        self.layers = nn.ModuleList()
        dims = [embed_dim] + hidden_dims
        
        # 前の層のQ次元を追跡（Qは各層のoutput_dimと同じ）
        prev_q_dim = embed_dim  # 最初の層は入力次元
        
        for i in range(len(dims) - 1):
            current_output_dim = dims[i+1]
            # 2層目以降は前の層の出力次元をprev_q_dimとして使用
            layer = EQBNNLayer(dims[i], current_output_dim, prev_q_dim, entangle_strength)
            self.layers.append(layer)
            prev_q_dim = current_output_dim  # 次の層のために更新
        
        # 出力層
        self.output_proj = nn.Linear(hidden_dims[-1], vocab_size)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        順伝播
        
        Args:
            x: 入力トークン [batch, seq_len]
        
        Returns:
            logits: 出力ロジット [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # 埋め込み + 位置エンコーディング
        h = self.embedding(x) + self.pos_encoding[:seq_len].unsqueeze(0)
        h = self.dropout(h)
        
        # シーケンスを平均化（簡略化）
        h = h.mean(dim=1)  # [batch, embed_dim]
        
        # E-QBNNレイヤーを通過
        q_prev = None
        for layer in self.layers:
            h, q = layer(h, q_prev)
            q_prev = q
        
        # シーケンス次元を復元
        h = h.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 出力
        logits = self.output_proj(h)
        
        return logits
    
    def get_total_constraint_loss(self):
        """全層の幾何学的制約損失"""
        loss = 0
        for layer in self.layers:
            loss += layer.get_constraint_loss()
        return loss / len(self.layers)
    
    def get_entanglement_stats(self):
        """エンタングルメント統計"""
        stats = []
        for i, layer in enumerate(self.layers):
            if layer.q is not None:
                r_mean = APQB.theta_to_r(layer.theta).mean().item()
                T_mean = APQB.theta_to_T(layer.theta).mean().item()
                constraint = APQB.constraint(layer.theta).mean().item()
                stats.append({
                    'layer': i,
                    'r_mean': r_mean,
                    'T_mean': T_mean,
                    'constraint': constraint,
                    'entangle_strength': layer.entangle_op.entangle_strength.item()
                })
        return stats
    
    @torch.no_grad()
    def generate(self, start_tokens, max_length=50, temperature=1.0, 
                 use_quantum_sampling=True, top_k=40, top_p=0.9, 
                 repetition_penalty=1.2):
        """
        テキスト生成（改良版）
        
        Args:
            start_tokens: 開始トークン [seq_len]
            max_length: 最大生成長
            temperature: 温度パラメータ
            use_quantum_sampling: 量子サンプリングを使用するか
            top_k: トップKサンプリング
            top_p: ニュークレオサンプリング（top-p）
            repetition_penalty: 繰り返しペナルティ
        """
        self.eval()
        
        tokens = start_tokens.clone()
        generated_tokens = []
        
        for _ in range(max_length):
            # 入力準備
            x = tokens.unsqueeze(0)
            if x.size(1) > 512:
                x = x[:, -512:]
            
            # 順伝播
            logits = self(x)
            next_logits = logits[0, -1] / temperature
            
            # 繰り返しペナルティ
            if len(generated_tokens) > 0:
                for prev_token in set(generated_tokens[-20:]):  # 直近20トークン
                    next_logits[prev_token] /= repetition_penalty
            
            # 量子サンプリング
            if use_quantum_sampling and len(self.layers) > 0:
                last_layer = self.layers[-1]
                if last_layer.theta is not None:
                    T = APQB.theta_to_T(last_layer.theta).mean()
                    quantum_noise = torch.randn_like(next_logits) * T * 0.3
                    next_logits = next_logits + quantum_noise
            
            # Top-K フィルタリング
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-P (Nucleus) フィルタリング
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # top_p を超えるトークンを除外
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')
            
            # サンプリング
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            tokens = torch.cat([tokens, next_token], dim=0)
            generated_tokens.append(next_token.item())
        
        return tokens


# ========================================================================
# 5. データ取得（Common Crawl サンプル）
# ========================================================================

def fetch_common_crawl_sample(max_samples=1000, min_length=50, lang='en'):
    """
    学習データを取得
    
    Args:
        max_samples: 最大サンプル数
        min_length: 最小テキスト長
        lang: 'en' (英語) or 'ja' (日本語)
    """
    print(f"\n📥 データを取得中... (言語: {lang})")
    
    texts = []
    
    if lang == 'ja':
        # 日本語データ
        texts = fetch_japanese_data(max_samples, min_length)
    else:
        # 英語データ
        texts = fetch_english_data(max_samples, min_length)
    
    print(f"   取得テキスト数: {len(texts)}")
    
    return texts[:max_samples]


def fetch_japanese_data(max_samples=1000, min_length=30):
    """日本語データを取得"""
    texts = []
    
    # 1. Wikipedia日本語版からサンプル取得
    try:
        wiki_titles = [
            "量子コンピュータ", "ニューラルネットワーク", "機械学習",
            "人工知能", "深層学習", "自然言語処理",
            "コンピュータ", "物理学", "数学", "技術"
        ]
        
        for title in wiki_titles:
            url = f"https://ja.wikipedia.org/w/api.php?action=query&titles={title}&prop=extracts&explaintext&format=json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                for page in pages.values():
                    extract = page.get('extract', '')
                    if len(extract) > min_length:
                        paragraphs = extract.split('\n\n')
                        for p in paragraphs:
                            if len(p) > min_length:
                                texts.append(p.strip())
            
            if len(texts) >= max_samples:
                break
    except Exception as e:
        print(f"   Wikipedia日本語取得エラー: {e}")
    
    # 2. 追加の日本語サンプルテキスト
    japanese_texts = [
        "量子コンピュータは、量子力学の原理を利用して情報を処理する革新的な計算機です。従来のコンピュータが0と1のビットを使うのに対し、量子コンピュータは重ね合わせ状態を持つ量子ビットを使用します。",
        "ニューラルネットワークは、人間の脳の神経回路を模倣した計算システムです。入力層、隠れ層、出力層から構成され、学習によってパラメータを調整します。",
        "機械学習は、明示的にプログラムされることなく、データから学習して改善するシステムを実現する人工知能の一分野です。",
        "深層学習は、多層のニューラルネットワークを用いて、データから高度な特徴を自動的に学習する手法です。画像認識や音声認識で大きな成功を収めています。",
        "自然言語処理は、コンピュータが人間の言語を理解し、生成するための技術です。機械翻訳、感情分析、質問応答などに応用されています。",
        "人工知能は、人間の知的活動をコンピュータで実現しようとする技術の総称です。現在は機械学習が主流となっています。",
        "トランスフォーマーは、自己注意機構を用いた革新的なアーキテクチャです。GPTやBERTなど、多くの大規模言語モデルの基盤となっています。",
        "強化学習は、エージェントが環境との相互作用を通じて、報酬を最大化する行動を学習する手法です。ゲームやロボット制御に応用されています。",
        "転移学習は、ある課題で学習したモデルを、別の課題に適用する技術です。少ないデータでも効果的に学習できます。",
        "生成AIは、テキスト、画像、音声などを生成できる人工知能です。創造的なタスクに革命をもたらしています。",
        "量子ビットは、0と1の重ね合わせ状態を取ることができます。この性質により、量子コンピュータは並列計算が可能になります。",
        "エンタングルメントは、複数の量子ビットが強く相関した状態です。量子通信や量子暗号に応用されています。",
        "注意機構は、入力の重要な部分に焦点を当てる技術です。機械翻訳の品質を大幅に向上させました。",
        "畳み込みニューラルネットワークは、画像認識に特化したアーキテクチャです。フィルタを用いて特徴を抽出します。",
        "再帰型ニューラルネットワークは、時系列データを処理するのに適しています。過去の情報を記憶して利用できます。",
    ] * 70
    
    texts.extend(japanese_texts)
    
    return texts


def fetch_english_data(max_samples=1000, min_length=50):
    """英語データを取得"""
    texts = []
    
    # 1. Wikipediaからサンプル取得
    try:
        wiki_titles = [
            "Quantum_computing", "Neural_network", "Machine_learning",
            "Artificial_intelligence", "Deep_learning", "Natural_language_processing",
            "Computer_science", "Physics", "Mathematics", "Technology"
        ]
        
        for title in wiki_titles:
            url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=extracts&explaintext&format=json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                for page in pages.values():
                    extract = page.get('extract', '')
                    if len(extract) > min_length:
                        paragraphs = extract.split('\n\n')
                        for p in paragraphs:
                            if len(p) > min_length:
                                texts.append(p.strip())
            
            if len(texts) >= max_samples:
                break
    except Exception as e:
        print(f"   Wikipedia取得エラー: {e}")
    
    # 2. 追加の英語サンプルテキスト
    english_texts = [
        "Quantum computing harnesses the principles of quantum mechanics to process information in fundamentally new ways. Unlike classical computers that use bits representing 0 or 1, quantum computers use qubits that can exist in superposition of both states simultaneously.",
        "Neural networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes or neurons that process information using connectionist approaches to computation.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
        "Deep learning is a class of machine learning algorithms that uses multiple layers to progressively extract higher-level features from raw input. In image recognition, lower layers may identify edges, while higher layers may identify human-relevant concepts.",
        "Natural language processing combines computational linguistics with statistical, machine learning, and deep learning models. It enables computers to process and analyze large amounts of natural language data.",
        "The transformer architecture has revolutionized natural language processing since its introduction. It relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
        "Attention mechanisms allow neural networks to focus on specific parts of the input when producing output. This is particularly useful for tasks like machine translation where the relationship between input and output elements is not strictly sequential.",
        "Generative models learn the underlying distribution of the training data and can generate new samples from that distribution. Examples include variational autoencoders and generative adversarial networks.",
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. It differs from supervised learning in not requiring labeled input-output pairs.",
        "Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. This is popular in deep learning because it can train models with comparatively little data.",
    ] * 100
    
    texts.extend(english_texts)
    
    return texts


# ========================================================================
# 6. トークナイザー
# ========================================================================

class SimpleTokenizer:
    """シンプルなトークナイザー（日本語/英語対応）"""
    
    def __init__(self, max_vocab_size=5000, use_char=False):
        self.max_vocab_size = max_vocab_size
        self.use_char = use_char  # 文字単位トークナイズ
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.vocab_size = 4
        self.is_japanese = False
    
    def fit(self, texts):
        """語彙を構築"""
        word_counts = Counter()
        
        # 日本語検出
        sample_text = ' '.join(texts[:10])
        self.is_japanese = any('\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF' for c in sample_text)
        
        if self.is_japanese:
            print("   日本語モードを検出")
            self.use_char = True  # 日本語は文字単位
        
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        
        # 頻度順にソート
        most_common = word_counts.most_common(self.max_vocab_size - self.vocab_size)
        
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"   語彙サイズ: {self.vocab_size}")
    
    def _tokenize(self, text):
        """テキストをトークンに分割"""
        if self.use_char or self.is_japanese:
            # 文字単位（日本語向け）
            return list(text)
        else:
            # 単語単位（英語向け）
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.split()
            return words
    
    def encode(self, text, max_length=None):
        """テキスト → トークンID"""
        words = self._tokenize(text)
        tokens = [self.word2idx.get(w, 1) for w in words]  # 1 = <UNK>
        
        if max_length:
            tokens = tokens[:max_length]
        
        return tokens
    
    def decode(self, tokens):
        """トークンID → テキスト"""
        words = [self.idx2word.get(t, '') for t in tokens if t not in [0, 1, 2, 3]]
        if self.is_japanese or self.use_char:
            return ''.join(words)  # 日本語は結合
        return ' '.join(words)


# ========================================================================
# 7. データセット
# ========================================================================

class TextDataset(Dataset):
    """テキストデータセット"""
    
    def __init__(self, texts, tokenizer, seq_length=64):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # 全テキストをトークン化
        self.all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > 2:
                self.all_tokens.extend(tokens)
        
        print(f"   総トークン数: {len(self.all_tokens)}")
    
    def __len__(self):
        return max(0, len(self.all_tokens) - self.seq_length - 1)
    
    def __getitem__(self, idx):
        tokens = self.all_tokens[idx:idx + self.seq_length + 1]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        # パディング
        if len(x) < self.seq_length:
            pad_len = self.seq_length - len(x)
            x = F.pad(x, (0, pad_len), value=0)
            y = F.pad(y, (0, pad_len), value=0)
        
        return x, y


# ========================================================================
# 8. 生成AI クラス
# ========================================================================

class EQBNNGenerativeAI:
    """Entangled QBNN 生成AI"""
    
    def __init__(self, embed_dim=128, hidden_dims=[256, 256], 
                 entangle_strength=0.5, max_vocab_size=3000,
                 num_neurons: int = None):  # ニューロン数指定可能
        self.embed_dim = embed_dim
        # num_neuronsが指定されたらhidden_dimsを上書き
        if num_neurons is not None:
            self.hidden_dims = [num_neurons, num_neurons]
        else:
            self.hidden_dims = hidden_dims
        self.entangle_strength = entangle_strength
        self.max_vocab_size = max_vocab_size
        
        self.tokenizer = SimpleTokenizer(max_vocab_size)
        self.model = None
        
        # デバイス選択: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🍎 Apple Silicon GPU (MPS) を使用")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🎮 NVIDIA GPU (CUDA) を使用")
        else:
            self.device = torch.device("cpu")
            print("💻 CPU を使用")
    
    def train(self, texts, epochs=10, batch_size=32, lr=0.001, seq_length=64):
        """モデルを学習"""
        print("\n🎓 学習開始...")
        
        # トークナイザーを構築
        self.tokenizer.fit(texts)
        
        # データセット
        dataset = TextDataset(texts, self.tokenizer, seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # モデル構築
        self.model = EQBNNGenerativeModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            entangle_strength=self.entangle_strength
        ).to(self.device)
        
        # オプティマイザ
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 損失関数
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # 学習ループ
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_constraint = 0
            num_batches = 0
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # 順伝播
                logits = self.model(batch_x)
                
                # 損失計算
                ce_loss = criterion(logits.view(-1, self.tokenizer.vocab_size), batch_y.view(-1))
                constraint_loss = self.model.get_total_constraint_loss()
                
                # 総損失 = CE損失 + 幾何学的制約
                loss = ce_loss + 0.01 * constraint_loss
                
                # 逆伝播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += ce_loss.item()
                total_constraint += constraint_loss.item() if isinstance(constraint_loss, torch.Tensor) else constraint_loss
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            avg_constraint = total_constraint / max(num_batches, 1)
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, r²+T²={1+avg_constraint:.4f}")
        
        print("   学習完了！")
    
    def generate(self, prompt="The quantum", max_length=50, temperature=1.0, 
                 use_quantum=True, top_k=40, top_p=0.9, repetition_penalty=1.2):
        """テキスト生成"""
        if self.model is None:
            return "モデルが学習されていません"
        
        self.model.eval()
        
        # プロンプトをトークン化
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) == 0:
            tokens = [2]  # <BOS>
        
        tokens = torch.tensor(tokens, dtype=torch.long).to(self.device)
        
        # 生成
        generated = self.model.generate(
            tokens, 
            max_length=max_length, 
            temperature=temperature,
            use_quantum_sampling=use_quantum,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        return self.tokenizer.decode(generated.cpu().tolist())
    
    def get_entanglement_report(self):
        """エンタングルメントレポート"""
        if self.model is None:
            return "モデルなし"
        
        # ダミー入力で統計を取得
        dummy_input = torch.randint(0, self.tokenizer.vocab_size, (1, 10)).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        stats = self.model.get_entanglement_stats()
        
        report = "\n📊 エンタングルメントレポート\n" + "-" * 50 + "\n"
        for s in stats:
            report += f"   Layer {s['layer']}: r={s['r_mean']:.3f}, T={s['T_mean']:.3f}, "
            report += f"r²+T²={s['constraint']:.4f}, λ={s['entangle_strength']:.3f}\n"
        
        return report


# ========================================================================
# 9. 可視化
# ========================================================================

def visualize_entanglement(ai, save_path=None):
    """エンタングルメントの可視化"""
    import matplotlib.pyplot as plt
    
    if ai.model is None:
        print("モデルがありません")
        return
    
    # ダミー入力で統計を取得
    dummy_input = torch.randint(0, ai.tokenizer.vocab_size, (1, 10)).to(ai.device)
    with torch.no_grad():
        _ = ai.model(dummy_input)
    
    stats = ai.model.get_entanglement_stats()
    
    if not stats:
        print("統計がありません")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 各層の r と T
    ax = axes[0, 0]
    layers = [s['layer'] for s in stats]
    r_values = [s['r_mean'] for s in stats]
    T_values = [s['T_mean'] for s in stats]
    
    x = np.arange(len(layers))
    width = 0.35
    ax.bar(x - width/2, r_values, width, label='r (correlation)', color='blue', alpha=0.7)
    ax.bar(x + width/2, T_values, width, label='T (temperature)', color='red', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Value')
    ax.set_title('APQB Parameters per Layer')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 幾何学的制約
    ax = axes[0, 1]
    constraints = [s['constraint'] for s in stats]
    ax.bar(layers, constraints, color='green', alpha=0.7)
    ax.axhline(1.0, color='red', linestyle='--', label='Target: 1.0')
    ax.set_xlabel('Layer')
    ax.set_ylabel('r² + T²')
    ax.set_title('Geometric Constraint')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. エンタングルメント強度
    ax = axes[1, 0]
    entangle_strengths = [s['entangle_strength'] for s in stats]
    ax.plot(layers, entangle_strengths, 'o-', color='purple', linewidth=2, markersize=10)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Entanglement Strength (λ)')
    ax.set_title('Inter-layer Entanglement')
    ax.grid(True, alpha=0.3)
    
    # 4. r-T 平面上のプロット
    ax = axes[1, 1]
    
    # 制約曲線
    theta_range = np.linspace(0, np.pi/2, 100)
    r_curve = np.cos(2 * theta_range)
    T_curve = np.abs(np.sin(2 * theta_range))
    ax.plot(r_curve, T_curve, 'b-', linewidth=2, label='r² + T² = 1')
    
    # 各層のプロット
    colors = plt.cm.viridis(np.linspace(0, 1, len(stats)))
    for i, s in enumerate(stats):
        ax.scatter([s['r_mean']], [s['T_mean']], s=200, c=[colors[i]], 
                   label=f'Layer {s["layer"]}', zorder=5, edgecolors='black')
    
    ax.set_xlabel('r (Correlation)')
    ax.set_ylabel('T (Temperature)')
    ax.set_title('Layer States on r-T Plane')
    ax.legend()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Entangled Quantum Bit Neural Network (E-QBNN)\nInter-layer Entanglement Visualization', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   保存: {save_path}")
    
    plt.close()


# ========================================================================
# 10. メイン実行
# ========================================================================

def main(lang='en', num_neurons: int = 128):
    """
    メイン関数
    
    Args:
        lang: 言語 ('en' or 'ja')
        num_neurons: ニューロン数 (デフォルト: 128)
    """
    print("\n🔧 E-QBNN 生成AI を構築中...")
    print(f"   ニューロン数: {num_neurons}")
    
    # 1. データ取得
    texts = fetch_common_crawl_sample(max_samples=500, min_length=30, lang=lang)
    
    # 2. AI構築（ニューロン数を指定）
    ai = EQBNNGenerativeAI(
        embed_dim=64,
        num_neurons=num_neurons,  # ニューロン数を指定
        entangle_strength=0.5,
        max_vocab_size=2000
    )
    
    # 3. 学習
    ai.train(texts, epochs=10, batch_size=16, lr=0.002, seq_length=32)
    
    # 4. エンタングルメントレポート
    print(ai.get_entanglement_report())
    
    # 5. テキスト生成
    print("\n📝 テキスト生成:")
    print("-" * 50)
    
    if lang == 'ja':
        prompts = [
            "量子コンピュータは",
            "人工知能とは",
            "機械学習は",
            "未来の技術"
        ]
    else:
        prompts = [
            "Quantum computing",
            "Neural networks",
            "Machine learning",
            "The future of"
        ]
    
    for prompt in prompts:
        generated = ai.generate(
            prompt, 
            max_length=40, 
            temperature=1.0, 
            use_quantum=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.5
        )
        print(f"   Prompt: '{prompt}'")
        print(f"   → {generated}\n")
    
    # 6. 可視化
    print("\n📊 可視化を生成中...")
    visualize_entanglement(ai, '/Users/yuyahiguchi/Program/Qubit/eqbnn_entanglement.png')
    
    # 7. 論文との対応
    print("\n" + "=" * 70)
    print("📚 論文との対応（層間エンタングルメント版）")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  Entangled Quantum Bit Neural Network (E-QBNN)                  │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. 各層に量子状態を持たせる                                   │
    │     h^(l) → R^(l) → |ψ^(l)⟩ (APQB)                            │
    │                                                                 │
    │  2. 層間エンタングルメント                                     │
    │     e^(l) = f_entangle(q^(l), q^(l-1))                         │
    │     - 相関行列ベースのもつれ                                   │
    │     - CNOTライクな相互作用                                     │
    │     - 位相キックバック                                         │
    │                                                                 │
    │  3. 次層への伝播                                               │
    │     h^(l+1) = σ(W^(l) h^(l) + B^(l) + G(e^(l)))               │
    │     - 通常の線形変換 + エンタングル補正                        │
    │                                                                 │
    │  4. 幾何学的制約                                               │
    │     r² + T² = 1 を損失関数に追加                               │
    │                                                                 │
    │  5. 量子サンプリング                                           │
    │     生成時に T（温度）で量子ノイズを追加                       │
    │                                                                 │
    │  利点:                                                          │
    │     - 深い依存関係を表現                                       │
    │     - 自然な正則化（幾何学的制約）                             │
    │     - エンタングルメント強度 λ で制御可能                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n✅ E-QBNN 生成AI 完成！")
    print("   生成ファイル:")
    print("   - eqbnn_entanglement.png (エンタングルメント可視化)")


def chat_mode(lang='en'):
    """対話モード"""
    lang_name = "日本語" if lang == 'ja' else "English"
    print(f"\n🔧 E-QBNN チャットモードを起動中... ({lang_name})")
    
    # データ取得
    texts = fetch_common_crawl_sample(max_samples=500, min_length=30, lang=lang)
    
    # AI構築
    ai = EQBNNGenerativeAI(
        embed_dim=64,
        hidden_dims=[128, 128, 64],
        entangle_strength=0.5,
        max_vocab_size=2000
    )
    
    # 学習
    ai.train(texts, epochs=15, batch_size=16, lr=0.002, seq_length=32)
    
    print("\n" + "=" * 60)
    print("💬 E-QBNN チャットモード")
    print("=" * 60)
    print("コマンド:")
    print("  /quit, /exit  - 終了")
    print("  /temp <値>    - 温度設定 (0.1-2.0)")
    print("  /len <値>     - 生成長さ (10-100)")
    print("  /topk <値>    - Top-K (1-100)")
    print("  /topp <値>    - Top-P (0.1-1.0)")
    print("  /rep <値>     - 繰り返しペナルティ (1.0-2.0)")
    print("  /quantum      - 量子サンプリング ON/OFF")
    print("  /stats        - エンタングルメント統計")
    print("  /help         - ヘルプ表示")
    print("-" * 60)
    
    temperature = 1.0
    max_length = 30
    use_quantum = True
    top_k = 40
    top_p = 0.9
    repetition_penalty = 1.5
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # コマンド処理
            if user_input.lower() in ['/quit', '/exit', '/q']:
                print("\n👋 さようなら！")
                break
            
            elif user_input.lower() == '/help':
                print("コマンド一覧:")
                print("  /quit, /exit  - 終了")
                print("  /temp <値>    - 温度設定 (0.1-2.0)")
                print("  /len <値>     - 生成長さ (10-100)")
                print("  /quantum      - 量子サンプリング ON/OFF")
                print("  /stats        - エンタングルメント統計")
                continue
            
            elif user_input.lower().startswith('/temp'):
                try:
                    val = float(user_input.split()[1])
                    temperature = max(0.1, min(2.0, val))
                    print(f"   温度を {temperature} に設定しました")
                except:
                    print("   使用法: /temp <0.1-2.0>")
                continue
            
            elif user_input.lower().startswith('/len'):
                try:
                    val = int(user_input.split()[1])
                    max_length = max(10, min(100, val))
                    print(f"   生成長さを {max_length} に設定しました")
                except:
                    print("   使用法: /len <10-100>")
                continue
            
            elif user_input.lower() == '/quantum':
                use_quantum = not use_quantum
                status = "ON" if use_quantum else "OFF"
                print(f"   量子サンプリング: {status}")
                continue
            
            elif user_input.lower().startswith('/topk'):
                try:
                    val = int(user_input.split()[1])
                    top_k = max(1, min(100, val))
                    print(f"   Top-K を {top_k} に設定しました")
                except:
                    print("   使用法: /topk <1-100>")
                continue
            
            elif user_input.lower().startswith('/topp'):
                try:
                    val = float(user_input.split()[1])
                    top_p = max(0.1, min(1.0, val))
                    print(f"   Top-P を {top_p} に設定しました")
                except:
                    print("   使用法: /topp <0.1-1.0>")
                continue
            
            elif user_input.lower().startswith('/rep'):
                try:
                    val = float(user_input.split()[1])
                    repetition_penalty = max(1.0, min(2.0, val))
                    print(f"   繰り返しペナルティを {repetition_penalty} に設定しました")
                except:
                    print("   使用法: /rep <1.0-2.0>")
                continue
            
            elif user_input.lower() == '/stats':
                print(ai.get_entanglement_report())
                continue
            
            # テキスト生成
            response = ai.generate(
                user_input, 
                max_length=max_length, 
                temperature=temperature,
                use_quantum=use_quantum,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            print(f"\n🤖 E-QBNN: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 さようなら！")
            break
        except Exception as e:
            print(f"   エラー: {e}")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='E-QBNN 生成AI')
    parser.add_argument('--neurons', type=int, default=128, help='ニューロン数 (デフォルト: 128)')
    parser.add_argument('--ja', action='store_true', help='日本語モード')
    parser.add_argument('--chat', action='store_true', help='チャットモード')
    args = parser.parse_args()
    
    lang = 'ja' if args.ja else 'en'
    
    if args.chat:
        chat_mode(lang=lang)
    else:
        main(lang=lang, num_neurons=args.neurons)

