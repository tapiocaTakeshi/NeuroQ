"""
量子インスパイアード LLM (PyTorch版)

擬似量子ビットの原理をPyTorchニューラルネットワークに統合
- 量子重ね合わせ的なアテンション
- 量子干渉による確率増幅
- 高速な並列処理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import math
from pseudo_qubit import PseudoQubit


# ============================================================
# 量子インスパイアード コンポーネント
# ============================================================

class QuantumSuperpositionLayer(nn.Module):
    """
    量子重ね合わせレイヤー
    
    全ての可能な状態を同時に表現
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 量子位相を学習
        self.phase = nn.Parameter(torch.randn(dim) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 量子的な位相変調
        phase_shift = torch.cos(self.phase) + 1j * torch.sin(self.phase)
        # 実数部のみ使用（擬似量子）
        amplitude = torch.cos(self.phase * x)
        return x * amplitude


class QuantumInterferenceLayer(nn.Module):
    """
    量子干渉レイヤー
    
    良い候補を増幅、悪い候補を減衰
    """
    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        self.interference_gate = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 干渉パターンを計算
        interference = torch.sigmoid(self.interference_gate(x))
        projected = self.projection(x)
        # 構造的干渉
        return projected * interference + x * (1 - interference)


class QuantumAttention(nn.Module):
    """
    量子インスパイアード アテンション
    
    全てのトークン関係を「重ね合わせ」で同時評価
    """
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 量子位相パラメータ
        self.quantum_phase = nn.Parameter(torch.randn(n_heads, 1, 1) * 0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Query, Key, Value
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 量子的アテンションスコア
        # cos(phase) による干渉効果
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        quantum_interference = torch.cos(self.quantum_phase * scores)
        scores = scores * (1 + 0.1 * quantum_interference)
        
        # マスク適用（因果的アテンション）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # ソフトマックスで確率化（量子測定に相当）
        attn_weights = F.softmax(scores, dim=-1)
        
        # 値の重み付け和
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out_proj(out)


class QuantumTransformerBlock(nn.Module):
    """
    量子インスパイアード Transformerブロック
    """
    def __init__(self, dim: int, n_heads: int = 4, ff_dim: int = None):
        super().__init__()
        ff_dim = ff_dim or dim * 4
        
        self.attention = QuantumAttention(dim, n_heads)
        self.superposition = QuantumSuperpositionLayer(dim)
        self.interference = QuantumInterferenceLayer(dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 量子アテンション + 残差接続
        x = x + self.attention(self.norm1(x), mask)
        
        # 量子重ね合わせ
        x = self.superposition(x)
        
        # フィードフォワード + 残差接続
        x = x + self.ff(self.norm2(x))
        
        # 量子干渉
        x = self.interference(x)
        
        return x


# ============================================================
# 量子LLMモデル
# ============================================================

class QuantumLLM(nn.Module):
    """
    量子インスパイアード 言語モデル
    
    PyTorch + 擬似量子ビット原理
    """
    def __init__(
        self, 
        vocab_size: int,
        dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 256
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        
        # 量子Transformerブロック
        self.blocks = nn.ModuleList([
            QuantumTransformerBlock(dim, n_heads)
            for _ in range(n_layers)
        ])
        
        # 出力層
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)
        
        # 因果マスク
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        
        # 埋め込み
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        # マスク
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # Transformerブロック
        for block in self.blocks:
            x = block(x, mask)
        
        # 出力
        x = self.norm(x)
        logits = self.output(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        テキスト生成（量子サンプリング）
        """
        self.eval()
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # 最大長を超えないように
            if generated.shape[1] >= self.max_seq_len:
                break
            
            # 予測
            logits = self(generated)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k フィルタリング
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) フィルタリング
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # 量子測定（確率的サンプリング）
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# ============================================================
# トークナイザー
# ============================================================

class SimpleTokenizer:
    """シンプルな文字レベルトークナイザー"""
    
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        
        # 特殊トークン
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
    def build_vocab(self, texts: List[str]):
        """語彙を構築"""
        # 特殊トークン
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            self.char_to_id[token] = len(self.char_to_id)
            self.id_to_char[self.char_to_id[token]] = token
        
        # 文字を収集
        chars = set()
        for text in texts:
            chars.update(text)
        
        for char in sorted(chars):
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)
                self.id_to_char[self.char_to_id[char]] = char
        
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text: str) -> List[int]:
        """テキストをID列に変換"""
        ids = [self.char_to_id.get(self.bos_token, 0)]
        for char in text:
            ids.append(self.char_to_id.get(char, self.char_to_id[self.unk_token]))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """ID列をテキストに変換"""
        chars = []
        for id in ids:
            char = self.id_to_char.get(id, '')
            if char not in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                chars.append(char)
        return ''.join(chars)


# ============================================================
# トレーナー
# ============================================================

class QuantumLLMTrainer:
    """量子LLMトレーナー"""
    
    def __init__(
        self,
        model: QuantumLLM,
        tokenizer: SimpleTokenizer,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def prepare_batch(self, texts: List[str], max_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """バッチデータを準備"""
        batch_ids = []
        
        for text in texts:
            ids = self.tokenizer.encode(text)
            # パディング/トランケート
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.tokenizer.char_to_id[self.tokenizer.pad_token]] * (max_len - len(ids))
            batch_ids.append(ids)
        
        batch = torch.tensor(batch_ids, device=self.device)
        # 入力: [:-1], ターゲット: [1:]
        return batch[:, :-1], batch[:, 1:]
    
    def train_step(self, texts: List[str]) -> float:
        """1ステップの学習"""
        self.model.train()
        
        inputs, targets = self.prepare_batch(texts)
        
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        
        # 損失計算
        loss = self.criterion(
            logits.reshape(-1, self.tokenizer.vocab_size),
            targets.reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, texts: List[str], epochs: int = 100, batch_size: int = 8):
        """学習ループ"""
        n_batches = (len(texts) + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            total_loss = 0
            
            # シャッフル
            indices = np.random.permutation(len(texts))
            
            for i in range(n_batches):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                batch_texts = [texts[j] for j in batch_indices]
                
                loss = self.train_step(batch_texts)
                total_loss += loss
            
            avg_loss = total_loss / n_batches
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8) -> str:
        """テキスト生成"""
        self.model.eval()
        
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        
        generated_ids = self.model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        return self.tokenizer.decode(generated_ids[0].tolist())


# ============================================================
# デモ
# ============================================================

def demo():
    """量子LLM デモ"""
    print("=" * 70)
    print("  QUANTUM-INSPIRED LLM (PyTorch)")
    print("  擬似量子ビット + ニューラルネットワーク")
    print("=" * 70)
    
    # 学習データ
    training_texts = [
        "こんにちは。今日はいい天気ですね。",
        "量子コンピュータは素晴らしい技術です。",
        "機械学習で未来を変えましょう。",
        "人工知能は日々進化しています。",
        "プログラミングは楽しいですね。",
        "新しい技術を学ぶことが大切です。",
        "量子ビットは0と1を同時に持てます。",
        "深層学習は画像認識に使われます。",
        "今日も一日頑張りましょう。",
        "ありがとうございます。",
        "質問があれば聞いてください。",
        "未来のコンピュータは量子技術を使います。",
        "私は量子AIです。何でも聞いてください。",
        "学習することは楽しいです。",
        "明日も良い日になりますように。",
    ]
    
    # トークナイザー構築
    print("\n" + "─" * 70)
    print("  1. トークナイザー構築")
    print("─" * 70)
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(training_texts)
    print(f"  語彙サイズ: {tokenizer.vocab_size}")
    
    # モデル構築
    print("\n" + "─" * 70)
    print("  2. 量子LLMモデル構築")
    print("─" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  デバイス: {device}")
    
    model = QuantumLLM(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=3,
        n_heads=4,
        max_seq_len=128
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数: {total_params:,}")
    
    # トレーナー
    trainer = QuantumLLMTrainer(model, tokenizer, learning_rate=1e-3, device=device)
    
    # 学習
    print("\n" + "─" * 70)
    print("  3. 学習開始")
    print("─" * 70)
    
    start_time = time.time()
    trainer.train(training_texts, epochs=100, batch_size=4)
    train_time = time.time() - start_time
    print(f"  学習時間: {train_time:.2f}秒")
    
    # 生成テスト
    print("\n" + "─" * 70)
    print("  4. テキスト生成テスト")
    print("─" * 70)
    
    prompts = ["こんにちは", "量子", "今日", "機械学習"]
    
    for prompt in prompts:
        print(f"\n  プロンプト: '{prompt}'")
        
        start = time.time()
        result = trainer.generate(prompt, max_new_tokens=40, temperature=0.8)
        gen_time = time.time() - start
        
        print(f"  生成結果: {result}")
        print(f"  生成時間: {gen_time:.3f}秒")
    
    # 温度比較
    print("\n" + "─" * 70)
    print("  5. 温度による生成の違い")
    print("─" * 70)
    
    prompt = "量子"
    for temp in [0.3, 0.8, 1.5]:
        result = trainer.generate(prompt, max_new_tokens=30, temperature=temp)
        print(f"  温度 {temp}: {result}")
    
    print("\n" + "=" * 70)
    print("  デモ完了！")
    print("=" * 70)
    
    return trainer


def interactive_mode():
    """対話モード"""
    print("\n" + "=" * 70)
    print("  量子LLM (PyTorch) - 対話モード")
    print("=" * 70)
    
    # 学習データ
    training_texts = [
        "こんにちは。今日はいい天気ですね。",
        "量子コンピュータは素晴らしい技術です。",
        "機械学習で未来を変えましょう。",
        "人工知能は日々進化しています。",
        "プログラミングは楽しいですね。",
        "新しい技術を学ぶことが大切です。",
        "量子ビットは0と1を同時に持てます。",
        "深層学習は画像認識に使われます。",
        "今日も一日頑張りましょう。",
        "ありがとうございます。",
        "質問があれば聞いてください。",
        "未来のコンピュータは量子技術を使います。",
        "私は量子AIです。何でも聞いてください。",
        "学習することは楽しいです。",
        "明日も良い日になりますように。",
    ]
    
    # 初期化
    print("\n  モデルを初期化中...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(training_texts)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = QuantumLLM(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=3,
        n_heads=4,
        max_seq_len=128
    )
    
    trainer = QuantumLLMTrainer(model, tokenizer, learning_rate=1e-3, device=device)
    
    # 学習
    print("  学習中...")
    trainer.train(training_texts, epochs=50, batch_size=4)
    
    # 設定
    temperature = 0.8
    max_tokens = 50
    
    print("\n  準備完了！")
    print("\n  コマンド:")
    print("    [テキスト] - 続きを生成")
    print("    add [テキスト] - 学習データを追加して再学習")
    print("    temp [値] - 温度設定 (0.1-2.0)")
    print("    len [値] - 最大生成長")
    print("    retrain - 再学習")
    print("    quit - 終了")
    
    while True:
        try:
            user_input = input(f"\n  [temp={temperature}] > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'q', '終了']:
                print("  さようなら！")
                break
            
            if user_input.lower().startswith('add '):
                new_text = user_input[4:].strip()
                if new_text:
                    training_texts.append(new_text)
                    # 語彙を再構築
                    tokenizer.build_vocab(training_texts)
                    # モデルを再構築
                    model = QuantumLLM(
                        vocab_size=tokenizer.vocab_size,
                        dim=64,
                        n_layers=3,
                        n_heads=4,
                        max_seq_len=128
                    )
                    trainer = QuantumLLMTrainer(model, tokenizer, learning_rate=1e-3, device=device)
                    print(f"  ✓ テキストを追加しました")
                    print("  再学習中...")
                    trainer.train(training_texts, epochs=50, batch_size=4)
                    print("  ✓ 再学習完了！")
                continue
            
            if user_input.lower().startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    temperature = max(0.1, min(2.0, temperature))
                    print(f"  ✓ 温度を {temperature} に設定")
                except:
                    print("  無効な値です")
                continue
            
            if user_input.lower().startswith('len '):
                try:
                    max_tokens = int(user_input.split()[1])
                    max_tokens = max(10, min(200, max_tokens))
                    print(f"  ✓ 最大長を {max_tokens} に設定")
                except:
                    print("  無効な値です")
                continue
            
            if user_input.lower() == 'retrain':
                print("  再学習中...")
                trainer.train(training_texts, epochs=50, batch_size=4)
                print("  ✓ 再学習完了！")
                continue
            
            # 生成
            print("  ⚡ 量子生成中...")
            start = time.time()
            result = trainer.generate(user_input, max_new_tokens=max_tokens, temperature=temperature)
            gen_time = time.time() - start
            
            print(f"\n  ┌{'─'*60}")
            print(f"  │ 🎯 結果: {result}")
            print(f"  └{'─'*60}")
            print(f"  ⏱ 生成時間: {gen_time:.3f}秒")
            
        except KeyboardInterrupt:
            print("\n  さようなら！")
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_mode()
    else:
        demo()

