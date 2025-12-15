#!/usr/bin/env python3
"""
Hugging Face (GPT-2) + OpenAI Embedding + QBNN 統合パイプライン
================================================================

構成:
1. GPT-2 Tokenizer (Hugging Face): テキスト → トークンID
2. Hybrid Embedding: HF Embedding + OpenAI Embedding (optional)
3. GPT-2 Style Attention (Hugging Face): 因果的アテンション
4. QBNN Layers: 量子ビットニューラルネットワーク処理
5. Language Model Head: 次トークン予測

使用例:
    pipeline = HFOpenAIQBNNPipeline(
        model_name='gpt2',
        use_openai_embedding=True,
        openai_api_key='your-key'
    )
    pipeline.train(texts, epochs=20)
    output = pipeline.generate("量子コンピュータは")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Hugging Face Transformers
try:
    from transformers import GPT2Tokenizer, GPT2Config
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  transformers not installed. Install with: pip install transformers")

# OpenAI (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai not installed. Install with: pip install openai")


# ============================================================
# Configuration
# ============================================================

@dataclass
class HFQBNNConfig:
    """Hugging Face + QBNN 統合設定"""

    # Tokenizer設定
    tokenizer_name: str = 'gpt2'  # 'gpt2', 'gpt2-medium', 'gpt2-large'

    # Embedding設定
    embed_dim: int = 256
    use_openai_embedding: bool = False
    openai_model: str = 'text-embedding-3-small'
    openai_embed_dim: int = 1536

    # Model設定
    num_heads: int = 8
    num_layers: int = 6
    hidden_dim: int = 512
    max_seq_len: int = 512

    # Dropout & Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # QBNN設定
    lambda_entangle: float = 0.35
    use_qbnn_attention: bool = True

    # Training設定
    vocab_size: int = None  # Auto-detect from tokenizer


# ============================================================
# STAGE 1: Tokenizer (Hugging Face GPT-2)
# ============================================================

class HFTokenizerWrapper:
    """
    Hugging Face GPT-2 Tokenizer ラッパー

    機能:
    - GPT-2トークナイザーの使用
    - 特殊トークン管理
    - バッチ処理対応
    """

    def __init__(self, model_name: str = 'gpt2'):
        if not HF_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers")

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # GPT-2に特殊トークンを追加
        special_tokens = {
            'pad_token': '<PAD>',
            'bos_token': '<BOS>',
            'eos_token': '<EOS>',
        }
        self.tokenizer.add_special_tokens(special_tokens)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.vocab_size = len(self.tokenizer)

        print(f"✅ GPT-2 Tokenizer loaded: {model_name}")
        print(f"   Vocabulary size: {self.vocab_size}")
        print(f"   Special tokens: PAD={self.pad_token_id}, BOS={self.bos_token_id}, EOS={self.eos_token_id}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """テキスト → トークンID"""
        if add_special_tokens:
            return [self.bos_token_id] + self.tokenizer.encode(text) + [self.eos_token_id]
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """トークンID → テキスト"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """バッチエンコーディング"""
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length' if padding else False,
            truncation=True,
            return_tensors='pt'
        )
        return encoded


# ============================================================
# STAGE 2: Hybrid Embedding (HF + OpenAI)
# ============================================================

class HybridEmbedding(nn.Module):
    """
    ハイブリッド埋め込み層

    オプション1: Hugging Face Embeddingのみ
    オプション2: HF Embedding + OpenAI Embedding

    OpenAI Embeddingは、より豊かな意味表現を提供
    """

    def __init__(
        self,
        config: HFQBNNConfig,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None
    ):
        super().__init__()

        self.config = config
        self.use_openai = use_openai and OPENAI_AVAILABLE

        # Hugging Face Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Position Embedding
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # OpenAI Embedding統合用
        if self.use_openai:
            if openai_api_key:
                openai.api_key = openai_api_key

            # OpenAI embedding → HF embed_dim への射影
            self.openai_projection = nn.Linear(config.openai_embed_dim, config.embed_dim)

            # 統合用の重み（学習可能）
            self.alpha = nn.Parameter(torch.tensor(0.5))  # HF vs OpenAI のバランス

            print(f"✅ OpenAI Embedding enabled: {config.openai_model}")

        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def get_openai_embedding(self, text: str) -> torch.Tensor:
        """OpenAI APIでembeddingを取得"""
        try:
            response = openai.Embedding.create(
                model=self.config.openai_model,
                input=text
            )
            embedding = response['data'][0]['embedding']
            return torch.tensor(embedding, dtype=torch.float32)
        except Exception as e:
            print(f"⚠️  OpenAI API error: {e}")
            return None

    def forward(
        self,
        token_ids: torch.Tensor,
        texts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Embedding forward pass

        Args:
            token_ids: (batch, seq_len) トークンID
            texts: オプション - OpenAI embedding用の元テキスト

        Returns:
            (batch, seq_len, embed_dim) 埋め込みベクトル
        """
        batch_size, seq_len = token_ids.shape

        # 1. Hugging Face Token Embedding
        token_embeds = self.token_embedding(token_ids)

        # 2. Position Embedding
        positions = torch.arange(seq_len, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)

        # 3. 基本埋め込み
        embeddings = token_embeds + pos_embeds

        # 4. OpenAI Embedding統合（オプション）
        if self.use_openai and texts is not None:
            openai_embeds = []
            for text in texts:
                oe = self.get_openai_embedding(text)
                if oe is not None:
                    # (openai_embed_dim,) → (embed_dim,)
                    oe = self.openai_projection(oe.to(token_ids.device))
                    openai_embeds.append(oe)
                else:
                    # Fallback: zero embedding
                    openai_embeds.append(torch.zeros(self.config.embed_dim, device=token_ids.device))

            # (batch, embed_dim) → (batch, 1, embed_dim) → broadcast
            openai_embeds = torch.stack(openai_embeds).unsqueeze(1)

            # HF と OpenAI を重み付け平均
            alpha = torch.sigmoid(self.alpha)  # 0-1 に正規化
            embeddings = alpha * embeddings + (1 - alpha) * openai_embeds.expand_as(embeddings)

        # 5. 正規化とドロップアウト
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# ============================================================
# STAGE 3: GPT-2 Style Causal Attention (Hugging Face)
# ============================================================

class GPT2StyleAttention(nn.Module):
    """
    GPT-2スタイルの因果的アテンション

    特徴:
    - 因果的マスク（未来の情報を見ない）
    - マルチヘッドアテンション
    - Scaled Dot-Product Attention
    """

    def __init__(self, config: HFQBNNConfig):
        super().__init__()

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.scale = math.sqrt(self.head_dim)

        # Causal mask (下三角行列)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        GPT-2 Causal Attention

        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Q, K, V を計算
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: Q @ K^T / sqrt(d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Causal masking (未来を見ない)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Attention output
        output = torch.matmul(attn_probs, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        return output


# ============================================================
# STAGE 4: QBNN Layer (量子ビットニューラルネットワーク)
# ============================================================

class QBNNLayer(nn.Module):
    """
    QBNN層 - 量子もつれを模倣した処理

    数式:
    1. s = tanh(h) ∈ [-1, 1] (正規化)
    2. h̃ = W·h + b (線形変換)
    3. Δ = Σ J_ij · s_i · s_j (量子もつれ項)
    4. ĥ = h̃ + λ·Δ (もつれ補正)
    5. output = activation(ĥ)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lambda_entangle: float = 0.35
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 線形変換
        self.linear = nn.Linear(input_dim, output_dim)

        # 量子もつれテンソル J
        self.J = nn.Parameter(torch.randn(input_dim, output_dim) * 0.02)

        # もつれ強度 λ (学習可能)
        self.lambda_entangle = nn.Parameter(torch.tensor(lambda_entangle))

        # 正規化
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        QBNN層の処理

        Args:
            h: (batch, seq, input_dim) 入力

        Returns:
            (batch, seq, output_dim) 出力
        """
        # 1. 正規化 (疑似量子状態)
        s = torch.tanh(h)

        # 2. 線形変換
        h_tilde = self.linear(h)

        # 3. 量子もつれ項の計算
        s_out = torch.tanh(h_tilde)
        delta = torch.einsum('...i,ij,...j->...j', s, self.J, s_out)

        # 4. もつれ補正を加算
        h_hat = h_tilde + self.lambda_entangle * delta

        # 5. 正規化と活性化
        output = self.layer_norm(h_hat)
        output = F.gelu(output)

        return output


class QBNNAttention(nn.Module):
    """
    QBNN拡張アテンション

    GPT-2 Attention + 量子もつれ補正
    """

    def __init__(self, config: HFQBNNConfig):
        super().__init__()

        # Base GPT-2 attention
        self.gpt2_attention = GPT2StyleAttention(config)

        # 量子もつれテンソル (ヘッドごと)
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        self.J_attn = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.head_dim) * 0.02
        )
        self.lambda_attn = nn.Parameter(torch.tensor(config.lambda_entangle))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        QBNN Attention

        Args:
            x: (batch, seq, embed_dim)

        Returns:
            (batch, seq, embed_dim)
        """
        # Base attention
        attn_out = self.gpt2_attention(x)

        # 量子もつれ補正（簡易版）
        # ここでは出力に対して軽い補正を加える
        delta = torch.tanh(attn_out) * self.lambda_attn * 0.1

        return attn_out + delta


# ============================================================
# STAGE 5: Transformer Block
# ============================================================

class HFQBNNBlock(nn.Module):
    """
    Transformer Block = Attention + FFN

    オプション:
    - use_qbnn_attention: QBNN拡張アテンションを使用
    - それ以外: 標準GPT-2アテンション
    """

    def __init__(self, config: HFQBNNConfig):
        super().__init__()

        # Attention
        if config.use_qbnn_attention:
            self.attention = QBNNAttention(config)
        else:
            self.attention = GPT2StyleAttention(config)

        self.attn_norm = nn.LayerNorm(config.embed_dim)

        # Feed-Forward Network (QBNN拡張)
        self.ffn = nn.Sequential(
            QBNNLayer(config.embed_dim, config.hidden_dim, config.lambda_entangle),
            nn.Dropout(config.dropout),
            QBNNLayer(config.hidden_dim, config.embed_dim, config.lambda_entangle),
        )
        self.ffn_norm = nn.LayerNorm(config.embed_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer Block

        Pre-LN式: Norm → Attention → Residual
        """
        # Attention + 残差接続
        attn_out = self.attention(self.attn_norm(x))
        x = x + self.dropout(attn_out)

        # FFN + 残差接続
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)

        return x


# ============================================================
# STAGE 6: Language Model
# ============================================================

class HFOpenAIQBNNPipeline(nn.Module):
    """
    Hugging Face (GPT-2) + OpenAI Embedding + QBNN 統合パイプライン

    処理フロー:
    ┌─────────┐   ┌──────────┐   ┌────────┐   ┌──────┐   ┌────────┐
    │  テキスト │ → │ HF Token │ → │ Hybrid │ → │ QBNN │ → │  出力  │
    │         │   │   izer   │   │Embedding│   │Blocks│   │  Head  │
    └─────────┘   └──────────┘   └────────┘   └──────┘   └────────┘
    """

    def __init__(
        self,
        config: Optional[HFQBNNConfig] = None,
        model_name: str = 'gpt2',
        use_openai_embedding: bool = False,
        openai_api_key: Optional[str] = None
    ):
        super().__init__()

        if config is None:
            config = HFQBNNConfig(tokenizer_name=model_name)

        self.config = config

        # Tokenizer
        self.tokenizer = HFTokenizerWrapper(config.tokenizer_name)

        # Update vocab_size
        if config.vocab_size is None:
            config.vocab_size = self.tokenizer.vocab_size

        # Hybrid Embedding
        self.embedding = HybridEmbedding(
            config,
            use_openai=use_openai_embedding,
            openai_api_key=openai_api_key
        )

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            HFQBNNBlock(config)
            for _ in range(config.num_layers)
        ])

        # Final Layer Norm
        self.final_norm = nn.LayerNorm(config.embed_dim)

        # Language Model Head
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Device
        self.device = torch.device('cpu')

        # Tie weights (embedding と lm_head)
        self.lm_head.weight = self.embedding.token_embedding.weight

        # Initialize
        self._init_weights()

        # Parameter count
        self.num_params = sum(p.numel() for p in self.parameters())

        print(f"\n✅ HF-OpenAI-QBNN Pipeline initialized")
        print(f"   Model: {config.tokenizer_name}")
        print(f"   Vocab size: {config.vocab_size}")
        print(f"   Embed dim: {config.embed_dim}")
        print(f"   Num layers: {config.num_layers}")
        print(f"   Num heads: {config.num_heads}")
        print(f"   Total parameters: {self.num_params:,}")
        print(f"   QBNN attention: {config.use_qbnn_attention}")
        print(f"   OpenAI embedding: {use_openai_embedding}")

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def to(self, device):
        """Move to device"""
        super().to(device)
        self.device = device
        return self

    def forward(
        self,
        token_ids: torch.Tensor,
        texts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            token_ids: (batch, seq_len) トークンID
            texts: オプション - OpenAI embedding用

        Returns:
            (batch, seq_len, vocab_size) ロジット
        """
        # 1. Embedding
        x = self.embedding(token_ids, texts=texts)

        # 2. Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final Norm
        x = self.final_norm(x)

        # 4. Language Model Head
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        テキスト生成

        Args:
            prompt: 入力プロンプト
            max_length: 最大生成長
            temperature: サンプリング温度
            top_k: Top-Kサンプリング
            top_p: Top-Pサンプリング
            repetition_penalty: 繰り返しペナルティ

        Returns:
            生成されたテキスト
        """
        self.eval()

        # Encode prompt
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

        generated = tokens[0].tolist()

        for _ in range(max_length):
            # Forward pass
            input_tokens = tokens[:, -self.config.max_seq_len:]
            logits = self(input_tokens)

            # Next token logits
            next_logits = logits[0, -1, :] / temperature

            # Repetition penalty
            for token_id in set(generated[-20:]):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

            # Top-K filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_vals[-1]] = float('-inf')

            # Top-P filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')

            # Sampling
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # EOS check
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated.append(next_token.item())
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        return self.tokenizer.decode(generated)

    def train_model(
        self,
        texts: List[str],
        epochs: int = 20,
        batch_size: int = 8,
        lr: float = 5e-4,
        seq_length: int = 128
    ):
        """
        モデルを学習

        Args:
            texts: 学習テキストのリスト
            epochs: エポック数
            batch_size: バッチサイズ
            lr: 学習率
            seq_length: シーケンス長
        """
        print("\n" + "=" * 60)
        print("🎓 学習開始: HF + OpenAI + QBNN Pipeline")
        print("=" * 60)

        # データ準備
        print("\n📊 データ準備中...")
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(tokens)

        print(f"   総トークン数: {len(all_tokens):,}")

        # シーケンス作成
        sequences = []
        for i in range(0, len(all_tokens) - seq_length - 1, seq_length // 2):
            x = all_tokens[i:i + seq_length]
            y = all_tokens[i + 1:i + seq_length + 1]
            if len(x) == seq_length:
                sequences.append((x, y))

        print(f"   シーケンス数: {len(sequences):,}")
        print(f"   パラメータ数: {self.num_params:,}")

        # Optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # Training loop
        print("\n🔄 学習中...")
        self.train()

        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(sequences)

            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]

                x = torch.tensor([s[0] for s in batch], dtype=torch.long).to(self.device)
                y = torch.tensor([s[1] for s in batch], dtype=torch.long).to(self.device)

                optimizer.zero_grad()

                # Forward
                logits = self(x)

                # Loss
                loss = criterion(
                    logits.view(-1, self.config.vocab_size),
                    y.view(-1)
                )

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(len(sequences) // batch_size, 1)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch + 1:3d}/{epochs}: Loss = {avg_loss:.4f}")

        print("\n✅ 学習完了！")


# ============================================================
# Main Demo
# ============================================================

def get_training_data() -> List[str]:
    """サンプル学習データ"""
    return [
        # 量子コンピューティング
        "量子コンピュータは、量子力学の原理を利用して情報を処理する革新的な計算機です。",
        "量子ビットは0と1の重ね合わせ状態を取ることができます。",
        "量子エンタングルメントは、複数の量子ビットが強く相関した状態です。",

        # AI & ML
        "ニューラルネットワークは、人間の脳の神経回路を模倣した計算システムです。",
        "深層学習は、多層のニューラルネットワークを用いた機械学習手法です。",
        "Transformerは、自己注意機構を用いた革新的なアーキテクチャです。",

        # GPT-2
        "GPT-2 is a large-scale unsupervised language model.",
        "The model uses transformer architecture with causal attention.",
        "It can generate coherent and contextually relevant text.",

        # General
        "人工知能は急速に進化しています。",
        "機械学習によって様々な問題が解決されています。",
        "自然言語処理は重要な研究分野です。",
    ] * 100  # データを増幅


def main():
    """メイン関数"""
    print("=" * 70)
    print("🧠⚛️ Hugging Face (GPT-2) + OpenAI + QBNN Pipeline")
    print("=" * 70)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n🎮 CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("\n💻 CPU")

    # Config
    config = HFQBNNConfig(
        tokenizer_name='gpt2',
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        hidden_dim=512,
        max_seq_len=256,
        use_qbnn_attention=True,
        lambda_entangle=0.35
    )

    # Pipeline
    print("\n🔧 パイプライン構築中...")
    pipeline = HFOpenAIQBNNPipeline(
        config=config,
        use_openai_embedding=False,  # OpenAI APIキーがない場合はFalse
        openai_api_key=None
    ).to(device)

    # Training data
    texts = get_training_data()
    print(f"\n📚 学習データ: {len(texts)} サンプル")

    # Train
    pipeline.train_model(texts, epochs=20, batch_size=8, lr=5e-4)

    # Generation test
    print("\n" + "=" * 60)
    print("📝 テキスト生成テスト")
    print("=" * 60)

    prompts = [
        "量子コンピュータは",
        "The transformer model",
        "人工知能とは",
        "Deep learning is",
    ]

    for prompt in prompts:
        output = pipeline.generate(
            prompt,
            max_length=30,
            temperature=0.8
        )
        print(f"\n入力: '{prompt}'")
        print(f"出力: {output}")

    print("\n✅ 完了！")


if __name__ == '__main__':
    if not HF_AVAILABLE:
        print("\n❌ Hugging Face transformers が必要です")
        print("インストール: pip install transformers")
    else:
        main()
