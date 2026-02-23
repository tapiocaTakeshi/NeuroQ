#!/usr/bin/env python3
"""
APQB Generative AI - Adjustable Pseudo Quantum Bit ベースの生成AI

APQBニューラルネットワークを使用してテキストを生成する
量子確率的な活性化により、多様で創造的なテキスト生成が可能
"""

import numpy as np
import json
import re
from collections import defaultdict
import random

# ========== APQB (Adjustable Pseudo Quantum Bit) ==========
class APQB:
    """調整可能な擬似量子ビット"""
    
    def __init__(self, r: float = 0.0):
        """
        Args:
            r: 相関係数 (-1 to 1)
               r = 1  → |0⟩ (確実に0)
               r = -1 → |1⟩ (確実に1)
               r = 0  → 等確率の重ね合わせ
        """
        self.r = np.clip(r, -1, 1)
    
    @property
    def theta(self) -> float:
        """相関係数から角度を計算"""
        return np.pi * (1 - self.r) / 2
    
    @property
    def p0(self) -> float:
        """|0⟩の確率"""
        return np.cos(self.theta / 2) ** 2
    
    @property
    def p1(self) -> float:
        """|1⟩の確率"""
        return np.sin(self.theta / 2) ** 2
    
    def measure(self) -> int:
        """量子測定: 確率的に0または1を返す"""
        return 1 if np.random.random() < self.p1 else 0
    
    def __repr__(self):
        return f"APQB(r={self.r:.3f}, P(1)={self.p1*100:.1f}%)"


# ========== APQBニューロン ==========
class APQBNeuron:
    """APQBベースのニューロン"""
    
    def __init__(self, num_inputs: int, init_scale: float = 0.5):
        self.weights = np.random.randn(num_inputs) * init_scale
        self.bias = np.random.randn() * 0.2
        self.apqb = APQB(0)
        self.last_output = 0
        self.last_r = 0
    
    def forward(self, inputs: np.ndarray, base_r: float = 0.0) -> int:
        """順伝播: 入力から出力を計算"""
        # 重み付き和を計算
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        
        # tanh で相関係数 r に変換 (-1 to 1)
        r = np.tanh(weighted_sum) + base_r * 0.3
        r = np.clip(r, -1, 1)
        self.last_r = r
        
        # APQBを更新して測定
        self.apqb = APQB(r)
        self.last_output = self.apqb.measure()
        
        return self.last_output
    
    def get_probability(self, inputs: np.ndarray, base_r: float = 0.0) -> float:
        """確率を取得（測定なし）"""
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        r = np.tanh(weighted_sum) + base_r * 0.3
        r = np.clip(r, -1, 1)
        return APQB(r).p1


# ========== APQBニューラルネットワーク ==========
class APQBNeuralNetwork:
    """APQBベースのニューラルネットワーク"""
    
    def __init__(self, layer_sizes: list, init_scale: float = 0.5):
        """
        Args:
            layer_sizes: 各層のニューロン数 [入力, 隠れ1, ..., 出力]
        """
        self.layer_sizes = layer_sizes
        self.layers = []
        self.base_r = 0.0
        
        # 各層のニューロンを作成
        for i in range(1, len(layer_sizes)):
            layer = []
            for _ in range(layer_sizes[i]):
                layer.append(APQBNeuron(layer_sizes[i-1], init_scale))
            self.layers.append(layer)
    
    def forward(self, inputs: np.ndarray) -> tuple:
        """順伝播"""
        current = np.array(inputs, dtype=float)
        activations = [current.copy()]
        
        for layer in self.layers:
            outputs = np.array([neuron.forward(current, self.base_r) for neuron in layer])
            current = outputs
            activations.append(current.copy())
        
        return current, activations
    
    def get_output_probabilities(self, inputs: np.ndarray) -> np.ndarray:
        """出力層の確率を取得"""
        current = np.array(inputs, dtype=float)
        
        for layer in self.layers[:-1]:
            outputs = np.array([neuron.forward(current, self.base_r) for neuron in layer])
            current = outputs
        
        # 最終層は確率を取得
        probs = np.array([neuron.get_probability(current, self.base_r) for neuron in self.layers[-1]])
        return probs
    
    def randomize_weights(self, scale: float = 0.8):
        """重みをランダム化"""
        for layer in self.layers:
            for neuron in layer:
                neuron.weights = np.random.randn(len(neuron.weights)) * scale
                neuron.bias = np.random.randn() * 0.3


# ========== APQB生成AI ==========
class APQBGenerativeAI:
    """APQBニューラルネットワークベースの生成AI"""
    
    def __init__(self, vocab_size: int = 256, hidden_size: int = 128, 
                 embedding_dim: int = 32, context_size: int = 3):
        """
        Args:
            vocab_size: 語彙サイズ（文字/トークン数）
            hidden_size: 隠れ層のサイズ
            embedding_dim: 埋め込み次元
            context_size: コンテキストサイズ（何文字前まで見るか）
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        
        # 文字→インデックスのマッピング
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # N-gramモデル（APQBで拡張）
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.ngram_totals = defaultdict(int)
        
        # APQBネットワーク（埋め込み→隠れ層→出力）
        input_size = embedding_dim * context_size
        self.network = APQBNeuralNetwork([input_size, hidden_size, hidden_size // 2, vocab_size])
        
        # 簡易埋め込み行列
        self.embeddings = None
        
        # 学習データ
        self.trained = False
        
    def _build_vocab(self, text: str):
        """語彙を構築"""
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        
        # 埋め込み行列を初期化
        self.embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
        
        # ネットワークを再構築
        input_size = self.embedding_dim * self.context_size
        self.network = APQBNeuralNetwork([input_size, self.hidden_size, self.hidden_size // 2, self.vocab_size])
    
    def _get_embedding(self, char: str) -> np.ndarray:
        """文字の埋め込みベクトルを取得"""
        if char not in self.char_to_idx:
            return np.zeros(self.embedding_dim)
        idx = self.char_to_idx[char]
        return self.embeddings[idx]
    
    def _context_to_input(self, context: str) -> np.ndarray:
        """コンテキスト文字列を入力ベクトルに変換"""
        # コンテキストをパディング
        padded = context.rjust(self.context_size, ' ')[-self.context_size:]
        
        # 各文字の埋め込みを連結
        embeddings = [self._get_embedding(c) for c in padded]
        return np.concatenate(embeddings)
    
    def train(self, text: str, verbose: bool = True):
        """テキストで学習"""
        if verbose:
            print("🧠 APQB Generative AI 学習開始...")
        
        # 語彙を構築
        self._build_vocab(text)
        if verbose:
            print(f"   語彙サイズ: {self.vocab_size}文字")
        
        # N-gramカウントを構築
        for i in range(len(text) - self.context_size):
            context = text[i:i + self.context_size]
            next_char = text[i + self.context_size]
            self.ngram_counts[context][next_char] += 1
            self.ngram_totals[context] += 1
        
        if verbose:
            print(f"   N-gramパターン数: {len(self.ngram_counts)}")
        
        # APQBネットワークの重みを調整（N-gramに基づく）
        self._tune_network(text)
        
        self.trained = True
        if verbose:
            print("✅ 学習完了！")
    
    def _tune_network(self, text: str):
        """N-gramに基づいてネットワークを調整"""
        # 各コンテキストについて、最も頻出する次の文字に向けて重みを調整
        for context, next_chars in self.ngram_counts.items():
            if not next_chars:
                continue
            
            # 入力ベクトル
            input_vec = self._context_to_input(context)
            
            # 最頻出文字
            most_common = max(next_chars.items(), key=lambda x: x[1])[0]
            target_idx = self.char_to_idx.get(most_common, 0)
            
            # 出力層の対応するニューロンの重みを強化
            if target_idx < len(self.network.layers[-1]):
                neuron = self.network.layers[-1][target_idx]
                # 重みを入力方向に少し調整
                adjustment = input_vec * 0.01
                if len(adjustment) == len(neuron.weights):
                    neuron.weights += adjustment
    
    def generate(self, prompt: str = "", max_length: int = 100, 
                 temperature: float = 1.0, base_r: float = 0.0) -> str:
        """テキストを生成
        
        Args:
            prompt: 開始テキスト
            max_length: 生成する最大文字数
            temperature: 温度パラメータ（高いほどランダム）
            base_r: APQB基準相関係数（-1〜1）
        """
        if not self.trained:
            raise ValueError("先にtrain()で学習してください")
        
        self.network.base_r = base_r
        
        # 生成テキスト
        generated = prompt
        context = prompt[-self.context_size:] if len(prompt) >= self.context_size else prompt.rjust(self.context_size)
        
        for _ in range(max_length):
            next_char = self._generate_next_char(context, temperature)
            generated += next_char
            context = context[1:] + next_char
        
        return generated
    
    def _generate_next_char(self, context: str, temperature: float = 1.0) -> str:
        """次の文字を生成"""
        # N-gramの確率を取得
        ngram_probs = self._get_ngram_probs(context)
        
        # APQBネットワークの確率を取得
        input_vec = self._context_to_input(context)
        network_probs = self.network.get_output_probabilities(input_vec)
        
        # 確率を結合（N-gram優先、APQBで調整）
        combined_probs = np.zeros(self.vocab_size)
        
        for char, idx in self.char_to_idx.items():
            ngram_p = ngram_probs.get(char, 0.001)
            network_p = network_probs[idx] if idx < len(network_probs) else 0.5
            
            # APQBの確率でN-gramを調整
            combined_probs[idx] = ngram_p * (0.5 + network_p)
        
        # 温度でスケーリング
        if temperature != 1.0:
            combined_probs = np.power(combined_probs, 1.0 / temperature)
        
        # 正規化
        total = combined_probs.sum()
        if total > 0:
            combined_probs /= total
        else:
            combined_probs = np.ones(self.vocab_size) / self.vocab_size
        
        # APQBベースのサンプリング
        # 各候補についてAPQBを使って選択
        selected_idx = self._apqb_sample(combined_probs)
        
        return self.idx_to_char.get(selected_idx, ' ')
    
    def _get_ngram_probs(self, context: str) -> dict:
        """N-gramの確率を取得"""
        probs = {}
        total = self.ngram_totals.get(context, 0)
        
        if total > 0:
            for char, count in self.ngram_counts[context].items():
                probs[char] = count / total
        
        return probs
    
    def _apqb_sample(self, probs: np.ndarray) -> int:
        """APQBベースのサンプリング"""
        # 累積確率を計算
        cumsum = np.cumsum(probs)
        
        # APQBで乱数を生成（量子的なランダム性）
        # 複数のAPQBの測定結果を組み合わせて[0,1]の値を生成
        quantum_random = 0
        for i in range(8):  # 8ビット精度
            r = np.random.uniform(-1, 1)  # ランダムな相関係数
            apqb = APQB(r)
            bit = apqb.measure()
            quantum_random += bit * (2 ** -(i + 1))
        
        # 確率に基づいて選択
        idx = np.searchsorted(cumsum, quantum_random)
        return min(idx, len(probs) - 1)
    
    def generate_stream(self, prompt: str = "", max_length: int = 100,
                        temperature: float = 1.0, base_r: float = 0.0):
        """ストリーミング生成（1文字ずつyield）"""
        if not self.trained:
            raise ValueError("先にtrain()で学習してください")
        
        self.network.base_r = base_r
        context = prompt[-self.context_size:] if len(prompt) >= self.context_size else prompt.rjust(self.context_size)
        
        for char in prompt:
            yield char
        
        for _ in range(max_length):
            next_char = self._generate_next_char(context, temperature)
            yield next_char
            context = context[1:] + next_char


# ========== 対話型インターフェース ==========
def interactive_mode():
    """対話モード"""
    print("=" * 60)
    print("🧠⚛️ APQB Generative AI - 量子確率的テキスト生成")
    print("=" * 60)
    
    # サンプルテキストで学習
    sample_text = """
    人工知能は私たちの生活を大きく変えています。
    機械学習やディープラーニングの発展により、
    コンピュータは人間のように考え、学習することができるようになりました。
    量子コンピュータは、この革命をさらに加速させる可能性を秘めています。
    量子ビットは重ね合わせ状態を利用して、
    従来のコンピュータでは不可能だった計算を可能にします。
    APQBは、この量子の性質を擬似的に再現し、
    ニューラルネットワークに確率的な振る舞いをもたらします。
    未来のAIは、量子と古典の融合によって、
    より創造的で柔軟な思考ができるようになるでしょう。
    技術の進歩は止まることを知りません。
    私たちは常に新しい可能性を探求し続けています。
    """
    
    print("\n📚 サンプルテキストで学習中...")
    ai = APQBGenerativeAI(hidden_size=64, context_size=4)
    ai.train(sample_text)
    
    print("\n" + "=" * 60)
    print("コマンド:")
    print("  <テキスト> : プロンプトからテキストを生成")
    print("  /temp <値> : 温度を設定 (0.1-2.0)")
    print("  /r <値>    : 基準相関係数を設定 (-1.0-1.0)")
    print("  /train     : 追加テキストで学習")
    print("  /quit      : 終了")
    print("=" * 60)
    
    temperature = 1.0
    base_r = 0.0
    
    while True:
        try:
            user_input = input("\n🔮 > ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith("/quit"):
                print("👋 終了します")
                break
            
            elif user_input.startswith("/temp"):
                try:
                    temp = float(user_input.split()[1])
                    temperature = max(0.1, min(2.0, temp))
                    print(f"🌡️ 温度を {temperature} に設定しました")
                except (ValueError, IndexError):
                    print("❌ 使用法: /temp <0.1-2.0>")
            
            elif user_input.startswith("/r"):
                try:
                    r = float(user_input.split()[1])
                    base_r = max(-1.0, min(1.0, r))
                    print(f"⚛️ 基準相関係数を {base_r} に設定しました")
                except (ValueError, IndexError):
                    print("❌ 使用法: /r <-1.0-1.0>")
            
            elif user_input.startswith("/train"):
                print("📝 追加学習テキストを入力してください（空行で終了）:")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                if lines:
                    additional_text = "\n".join(lines)
                    ai.train(additional_text)
                    print("✅ 追加学習完了！")
            
            else:
                # テキスト生成
                print("\n📜 生成中... ", end="", flush=True)
                
                generated = ai.generate(
                    prompt=user_input,
                    max_length=150,
                    temperature=temperature,
                    base_r=base_r
                )
                
                print("\n" + "-" * 40)
                print(generated)
                print("-" * 40)
                print(f"[温度: {temperature}, 相関係数: {base_r}]")
        
        except KeyboardInterrupt:
            print("\n👋 終了します")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")


# ========== デモ ==========
def demo():
    """デモンストレーション"""
    print("=" * 60)
    print("🧠⚛️ APQB Generative AI デモ")
    print("=" * 60)
    
    # 日本語テキストで学習
    training_text = """
    桜の花が咲く春の季節、私たちは新しい始まりを迎えます。
    夏には太陽が輝き、海や山で楽しい時間を過ごします。
    秋になると紅葉が美しく色づき、穏やかな風が吹きます。
    冬は雪が降り、温かい家の中で家族と過ごす時間が増えます。
    日本の四季は本当に美しいものです。
    自然の移り変わりを感じながら、私たちは生きています。
    技術は進歩しても、自然の美しさは変わりません。
    量子コンピュータや人工知能が発展しても、
    人間の心の温かさは大切にしていきたいものです。
    未来への希望を持って、私たちは前に進みます。
    """
    
    print("\n📚 学習中...")
    ai = APQBGenerativeAI(hidden_size=64, context_size=3)
    ai.train(training_text)
    
    # 様々な設定でテキスト生成
    prompts = ["春の", "技術", "未来"]
    settings = [
        (0.5, 0.0, "低温度（保守的）"),
        (1.0, 0.0, "標準"),
        (1.5, 0.0, "高温度（創造的）"),
        (1.0, -0.5, "量子バイアス（1寄り）"),
        (1.0, 0.5, "量子バイアス（0寄り）"),
    ]
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"🔮 プロンプト: 「{prompt}」")
        print("=" * 60)
        
        for temp, base_r, desc in settings:
            generated = ai.generate(
                prompt=prompt,
                max_length=50,
                temperature=temp,
                base_r=base_r
            )
            print(f"\n[{desc}]")
            print(f"  {generated}")
    
    print("\n" + "=" * 60)
    print("✅ デモ完了！")


# ========== メイン ==========
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        interactive_mode()

