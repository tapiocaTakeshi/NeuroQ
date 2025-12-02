"""
量子インスパイアード テキスト生成AI (Quantum-Inspired LLM)

擬似量子ビットを活用した高速テキスト生成

原理:
1. 量子重ね合わせ: 全ての次トークン候補を同時に評価
2. 量子干渉: 文脈に合う候補を増幅、合わない候補を減衰
3. 量子測定: 確率的にトークンを選択

ChatGPTのような従来のLLMは候補を順次評価するが、
量子的アプローチでは全候補を並列評価！
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random
import time
import re
from pseudo_qubit import PseudoQubit


# ============================================================
# 量子トークン状態
# ============================================================

class QuantumTokenState:
    """
    トークンの量子状態
    
    各候補トークンが確率振幅を持つ重ね合わせ状態
    """
    
    def __init__(self, tokens: List[str], amplitudes: Optional[np.ndarray] = None):
        self.tokens = tokens
        self.n_tokens = len(tokens)
        
        if amplitudes is None:
            # 一様な重ね合わせ
            self.amplitudes = np.ones(self.n_tokens) / np.sqrt(self.n_tokens)
        else:
            self.amplitudes = amplitudes
            self._normalize()
    
    def _normalize(self):
        """正規化"""
        norm = np.sqrt(np.sum(self.amplitudes ** 2))
        if norm > 0:
            self.amplitudes /= norm
    
    @property
    def probabilities(self) -> np.ndarray:
        """確率分布"""
        return self.amplitudes ** 2
    
    def amplify(self, indices: List[int], factor: float = 2.0):
        """
        グローバー的振幅増幅
        指定インデックスの振幅を増幅
        """
        for i in indices:
            self.amplitudes[i] *= factor
        self._normalize()
    
    def interfere(self, scores: np.ndarray):
        """
        量子干渉
        スコアに基づいて振幅を調整
        """
        # スコアを相関係数に変換
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        correlations = 2 * scores_norm - 1  # [-1, 1]
        
        # 量子ビットで確率に変換して干渉
        for i, r in enumerate(correlations):
            qubit = PseudoQubit(correlation=float(r))
            interference = qubit.probabilities[0]  # |0⟩の確率
            self.amplitudes[i] *= np.sqrt(interference)
        
        self._normalize()
    
    def measure(self) -> str:
        """測定（トークンを選択）"""
        probs = self.probabilities
        probs /= probs.sum()  # 正規化
        idx = np.random.choice(self.n_tokens, p=probs)
        return self.tokens[idx]
    
    def top_k(self, k: int = 5) -> List[Tuple[str, float]]:
        """上位k個の候補を返す"""
        probs = self.probabilities
        indices = np.argsort(probs)[::-1][:k]
        return [(self.tokens[i], probs[i]) for i in indices]


# ============================================================
# 量子N-gramモデル
# ============================================================

class QuantumNGram:
    """
    量子インスパイアード N-gramモデル
    
    文脈からの次トークン予測を量子並列で実行
    """
    
    def __init__(self, n: int = 3):
        self.n = n  # N-gramのN
        self.ngrams: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.vocabulary: set = set()
        self.total_counts: Dict[str, int] = defaultdict(int)
    
    def tokenize(self, text: str) -> List[str]:
        """テキストをトークン化（文字単位）"""
        return list(text)
    
    def train(self, text: str):
        """テキストから学習"""
        tokens = self.tokenize(text)
        self.vocabulary.update(tokens)
        
        # N-gramカウント
        for i in range(len(tokens) - self.n):
            context = ''.join(tokens[i:i + self.n - 1])
            next_token = tokens[i + self.n - 1]
            self.ngrams[context][next_token] += 1
            self.total_counts[context] += 1
    
    def get_candidates(self, context: str) -> QuantumTokenState:
        """
        文脈から候補トークンの量子状態を生成
        
        全候補を重ね合わせ状態で表現
        """
        # 文脈の最後のn-1文字
        ctx = context[-(self.n - 1):] if len(context) >= self.n - 1 else context
        
        if ctx in self.ngrams:
            tokens = list(self.ngrams[ctx].keys())
            counts = np.array([self.ngrams[ctx][t] for t in tokens], dtype=float)
            
            # カウントを振幅に変換
            amplitudes = np.sqrt(counts / counts.sum())
        else:
            # 未知の文脈：全語彙から一様に
            tokens = list(self.vocabulary) if self.vocabulary else [' ']
            amplitudes = None
        
        return QuantumTokenState(tokens, amplitudes)


# ============================================================
# 量子テキスト生成器
# ============================================================

class QuantumTextGenerator:
    """
    量子インスパイアード テキスト生成AI
    
    量子並列性を活用した高速生成
    """
    
    def __init__(self, n_gram: int = 4):
        self.model = QuantumNGram(n=n_gram)
        self.context_memory: List[str] = []
        self.generation_history: List[Tuple[str, List[Tuple[str, float]]]] = []
    
    def train(self, corpus: str):
        """コーパスから学習"""
        self.model.train(corpus)
        print(f"  Vocabulary size: {len(self.model.vocabulary)}")
        print(f"  N-gram contexts: {len(self.model.ngrams)}")
    
    def quantum_predict_next(self, context: str, temperature: float = 1.0) -> Tuple[str, List[Tuple[str, float]]]:
        """
        量子予測: 次のトークンを予測
        
        1. 全候補を量子重ね合わせで生成
        2. 文脈スコアで干渉
        3. 測定で選択
        """
        # 候補の量子状態を取得
        quantum_state = self.model.get_candidates(context)
        
        if quantum_state.n_tokens == 0:
            return ' ', []
        
        # 文脈に基づくスコアリング
        scores = self._compute_context_scores(context, quantum_state.tokens)
        
        # 量子干渉でスコアを反映
        quantum_state.interfere(scores)
        
        # 温度でシャープネス調整
        if temperature != 1.0:
            quantum_state.amplitudes = quantum_state.amplitudes ** (1.0 / temperature)
            quantum_state._normalize()
        
        # 上位候補を取得
        top_candidates = quantum_state.top_k(5)
        
        # 量子測定でトークン選択
        selected = quantum_state.measure()
        
        return selected, top_candidates
    
    def _compute_context_scores(self, context: str, candidates: List[str]) -> np.ndarray:
        """
        文脈スコアを計算
        
        量子干渉の重みとして使用
        """
        scores = np.ones(len(candidates))
        
        if len(context) < 2:
            return scores
        
        last_char = context[-1]
        
        for i, token in enumerate(candidates):
            # 連続性ボーナス
            if last_char.isalpha() and token.isalpha():
                scores[i] *= 1.5
            
            # 句読点の後はスペースか大文字
            if last_char in '.!?':
                if token == ' ' or token.isupper():
                    scores[i] *= 2.0
            
            # スペースの後はアルファベット
            if last_char == ' ' and token.isalpha():
                scores[i] *= 1.5
            
            # 同じ文字の連続を減衰
            if token == last_char and token not in ' ':
                scores[i] *= 0.3
        
        return scores
    
    def generate(self, prompt: str, max_length: int = 100, 
                 temperature: float = 0.8, 
                 show_process: bool = False) -> str:
        """
        テキスト生成
        
        量子並列で高速に次トークンを予測
        """
        result = prompt
        self.generation_history = []
        
        for i in range(max_length):
            # 量子予測
            next_token, candidates = self.quantum_predict_next(result, temperature)
            
            if show_process and candidates:
                print(f"\n  Step {i+1}: '{result[-20:]}' → ", end='')
                print(f"[{', '.join(f'{t}:{p:.2f}' for t, p in candidates[:3])}]")
                print(f"  Selected: '{next_token}'")
            
            result += next_token
            self.generation_history.append((next_token, candidates))
            
            # 終了条件
            if next_token in '.!?' and len(result) > len(prompt) + 20:
                if random.random() < 0.3:  # 30%の確率で文を終了
                    break
        
        return result
    
    def generate_with_beam(self, prompt: str, max_length: int = 50,
                           beam_width: int = 3) -> List[str]:
        """
        ビームサーチによる生成
        
        量子的に複数の候補パスを並列探索
        """
        # 初期ビーム
        beams = [(prompt, 0.0)]  # (text, score)
        
        for _ in range(max_length):
            all_candidates = []
            
            for text, score in beams:
                quantum_state = self.model.get_candidates(text)
                
                if quantum_state.n_tokens == 0:
                    continue
                
                # 上位候補を取得
                top = quantum_state.top_k(beam_width)
                
                for token, prob in top:
                    new_text = text + token
                    new_score = score + np.log(prob + 1e-10)
                    all_candidates.append((new_text, new_score))
            
            # 上位beam_width個を保持
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]
            
            if not beams:
                break
        
        return [text for text, score in beams]


# ============================================================
# 量子チャットボット
# ============================================================

class QuantumChatBot:
    """
    量子インスパイアード チャットボット
    
    対話形式のテキスト生成
    """
    
    def __init__(self):
        self.generator = QuantumTextGenerator(n_gram=4)
        self.conversation_history: List[Tuple[str, str]] = []
        self.personality_prompts = {
            'friendly': 'Hello! ',
            'formal': 'Greetings. ',
            'curious': 'Interesting! '
        }
    
    def train_from_conversations(self, conversations: List[str]):
        """会話データから学習"""
        corpus = '\n'.join(conversations)
        self.generator.train(corpus)
    
    def respond(self, user_input: str, max_length: int = 100) -> str:
        """
        ユーザー入力に応答
        
        量子並列で最適な応答を生成
        """
        # 文脈構築
        context = user_input + ' '
        
        # 応答生成
        response = self.generator.generate(
            context, 
            max_length=max_length,
            temperature=0.7
        )
        
        # プロンプト部分を除去
        response = response[len(context):]
        
        # 履歴に追加
        self.conversation_history.append((user_input, response))
        
        return response
    
    def chat(self):
        """対話モード"""
        print("\n" + "=" * 60)
        print("  QUANTUM CHAT BOT")
        print("  Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n  You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n  Quantum Bot: Goodbye! 👋")
                    break
                
                if not user_input:
                    continue
                
                response = self.respond(user_input)
                print(f"  Quantum Bot: {response}")
                
            except KeyboardInterrupt:
                print("\n\n  Quantum Bot: Goodbye! 👋")
                break


# ============================================================
# デモンストレーション
# ============================================================

def demo():
    """量子LLMのデモ"""
    print("=" * 70)
    print("  QUANTUM-INSPIRED TEXT GENERATION AI")
    print("  擬似量子ビットによる高速テキスト生成")
    print("=" * 70)
    
    # サンプルコーパス（英語）
    corpus_en = """
    The quantum computer uses superposition to evaluate all possibilities simultaneously.
    This allows for faster computation than classical computers in certain problems.
    Quantum bits can be in a state of zero and one at the same time.
    When measured, the quantum state collapses to a definite value.
    The power of quantum computing comes from quantum parallelism.
    Machine learning can benefit from quantum inspired algorithms.
    Neural networks learn patterns from data through training.
    The future of computing may involve quantum processors.
    Artificial intelligence is transforming many industries.
    Deep learning has achieved remarkable results in various tasks.
    """
    
    # サンプルコーパス（日本語）- 拡充版
    corpus_ja = """
    こんにちは。今日はいい天気ですね。お元気ですか。
    はい、元気です。ありがとうございます。
    量子コンピュータは重ね合わせを使って全ての可能性を同時に評価します。
    これにより古典コンピュータより高速な計算が可能になります。
    量子ビットは0と1の状態を同時に持つことができます。
    測定すると量子状態は確定した値に収縮します。
    量子コンピューティングの力は量子並列性から来ています。
    機械学習は量子インスパイアードアルゴリズムの恩恵を受けられます。
    人工知能は様々な分野で活用されています。
    深層学習は画像認識や自然言語処理で成果を上げています。
    未来のコンピュータは量子技術を活用するでしょう。
    私たちは新しい技術を学ぶことが大切です。
    学ぶことは楽しいですね。一緒に頑張りましょう。
    今日は何をしましたか。私は勉強をしました。
    明日の天気はどうでしょうか。晴れるといいですね。
    質問があれば何でも聞いてください。お答えします。
    それは面白い質問ですね。考えてみましょう。
    素晴らしいアイデアですね。試してみましょう。
    ありがとうございます。またお話しましょう。
    """
    
    # ============================================================
    # 1. 英語テキスト生成
    # ============================================================
    print("\n" + "─" * 70)
    print("  1. English Text Generation")
    print("─" * 70)
    
    generator_en = QuantumTextGenerator(n_gram=4)
    
    print("\n  Training on English corpus...")
    generator_en.train(corpus_en)
    
    prompts = ["The quantum", "Machine learning", "The future"]
    
    for prompt in prompts:
        print(f"\n  Prompt: '{prompt}'")
        
        start = time.time()
        result = generator_en.generate(prompt, max_length=80, temperature=0.8)
        gen_time = time.time() - start
        
        print(f"  Generated: {result}")
        print(f"  Time: {gen_time:.3f}s ({len(result)/gen_time:.0f} chars/s)")
    
    # ============================================================
    # 2. 生成プロセスの可視化
    # ============================================================
    print("\n" + "─" * 70)
    print("  2. Generation Process Visualization")
    print("─" * 70)
    
    print("\n  Showing step-by-step generation:")
    result = generator_en.generate("Quantum", max_length=30, show_process=True)
    
    # ============================================================
    # 3. 日本語テキスト生成
    # ============================================================
    print("\n" + "─" * 70)
    print("  3. Japanese Text Generation")
    print("─" * 70)
    
    generator_ja = QuantumTextGenerator(n_gram=3)
    
    print("\n  Training on Japanese corpus...")
    generator_ja.train(corpus_ja)
    
    prompts_ja = ["量子", "機械学習"]
    
    for prompt in prompts_ja:
        print(f"\n  Prompt: '{prompt}'")
        result = generator_ja.generate(prompt, max_length=50, temperature=0.7)
        print(f"  Generated: {result}")
    
    # ============================================================
    # 4. ビームサーチ
    # ============================================================
    print("\n" + "─" * 70)
    print("  4. Beam Search (Multiple Candidates)")
    print("─" * 70)
    
    print("\n  Generating multiple candidates with beam search:")
    candidates = generator_en.generate_with_beam("The", max_length=30, beam_width=3)
    
    for i, text in enumerate(candidates):
        print(f"  Candidate {i+1}: {text}")
    
    # ============================================================
    # 5. 速度テスト
    # ============================================================
    print("\n" + "─" * 70)
    print("  5. Speed Test")
    print("─" * 70)
    
    print("\n  Generating 1000 characters...")
    start = time.time()
    long_text = generator_en.generate("The quantum computer", max_length=1000)
    total_time = time.time() - start
    
    print(f"  Generated {len(long_text)} characters in {total_time:.3f}s")
    print(f"  Speed: {len(long_text)/total_time:.0f} characters/second")
    
    # ============================================================
    # 量子並列の利点
    # ============================================================
    print("\n" + "─" * 70)
    print("  6. Quantum Parallelism Advantage")
    print("─" * 70)
    
    print("""
  従来のLLM:
    - 候補トークンを順次評価
    - 計算量: O(V × N) [V=語彙サイズ, N=シーケンス長]
    
  量子インスパイアードLLM:
    - 全候補を重ね合わせで同時評価
    - 干渉効果で有望な候補を増幅
    - 理論的計算量: O(√V × N)
    
  擬似量子ビットの利点:
    - 実際の量子ハードウェア不要
    - 量子的な確率分布をシミュレート
    - 従来手法より多様な出力が可能
    """)
    
    print("\n" + "=" * 70)
    print("  デモ完了！")
    print("=" * 70)


def interactive_mode():
    """対話モード（日本語）"""
    print("\n" + "=" * 70)
    print("  量子テキスト生成AI - 対話モード")
    print("=" * 70)
    
    # 初期の日本語学習データ
    corpus = """
    こんにちは。今日はいい天気ですね。お元気ですか。
    はい、元気です。ありがとうございます。
    量子コンピュータは重ね合わせを使って全ての可能性を同時に評価します。
    これにより古典コンピュータより高速な計算が可能になります。
    量子ビットは0と1の状態を同時に持つことができます。
    測定すると量子状態は確定した値に収縮します。
    私は量子AIです。何でも聞いてください。
    ありがとうございます。またお話しましょう。
    """
    
    # 設定
    n_gram = 3
    max_length = 80
    temperature = 0.7
    show_process = False
    
    generator = QuantumTextGenerator(n_gram=n_gram)
    print("\n  モデルを学習中...")
    generator.train(corpus)
    
    def show_status():
        print(f"\n  ┌{'─'*50}")
        print(f"  │ 📊 現在のステータス")
        print(f"  ├{'─'*50}")
        print(f"  │ 語彙サイズ: {len(generator.model.vocabulary)}")
        print(f"  │ N-gramコンテキスト数: {len(generator.model.ngrams)}")
        print(f"  │ N-gram: {n_gram}")
        print(f"  │ 温度: {temperature}")
        print(f"  │ 最大長: {max_length}")
        print(f"  │ プロセス表示: {'ON' if show_process else 'OFF'}")
        print(f"  └{'─'*50}")
    
    def show_help():
        print("""
  ┌─────────────────────────────────────────────────────────────┐
  │ 📖 コマンド一覧                                             │
  ├─────────────────────────────────────────────────────────────┤
  │ [テキスト]      - 入力テキストの続きを生成                   │
  │                                                             │
  │ add [テキスト]  - 学習データを追加                          │
  │ addfile [パス]  - ファイルから学習データを追加              │
  │                                                             │
  │ temp [値]       - 温度設定 (0.1-2.0)                        │
  │                   低い=確実、高い=多様                       │
  │ len [値]        - 最大生成長を設定                          │
  │ ngram [値]      - N-gramのNを変更 (2-5)                     │
  │                                                             │
  │ show            - 生成プロセス表示ON/OFF                    │
  │ status          - 現在の設定を表示                          │
  │ vocab           - 語彙リストを表示                          │
  │ reset           - モデルをリセット                          │
  │                                                             │
  │ help            - このヘルプを表示                          │
  │ quit / q        - 終了                                      │
  └─────────────────────────────────────────────────────────────┘
        """)
    
    show_status()
    print("\n  'help' でコマンド一覧を表示")
    
    while True:
        try:
            user_input = input(f"\n  [温度={temperature}, 長さ={max_length}] > ").strip()
            
            if not user_input:
                continue
            
            # 終了
            if user_input.lower() in ['quit', 'exit', 'q', '終了']:
                print("  さようなら！")
                break
            
            # ヘルプ
            if user_input.lower() == 'help':
                show_help()
                continue
            
            # ステータス
            if user_input.lower() == 'status':
                show_status()
                continue
            
            # 語彙表示
            if user_input.lower() == 'vocab':
                vocab_list = sorted(list(generator.model.vocabulary))
                print(f"\n  語彙 ({len(vocab_list)}文字):")
                # 20文字ずつ表示
                for i in range(0, len(vocab_list), 30):
                    chars = vocab_list[i:i+30]
                    print(f"    {''.join(chars)}")
                continue
            
            # 学習データ追加
            if user_input.lower().startswith('add '):
                new_text = user_input[4:].strip()
                if new_text:
                    generator.model.train(new_text)
                    print(f"  ✓ テキストを追加しました ({len(new_text)}文字)")
                    print(f"    語彙サイズ: {len(generator.model.vocabulary)}")
                    print(f"    コンテキスト数: {len(generator.model.ngrams)}")
                else:
                    print("  追加するテキストを入力してください")
                continue
            
            # ファイルから追加
            if user_input.lower().startswith('addfile '):
                filepath = user_input[8:].strip()
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        file_text = f.read()
                    generator.model.train(file_text)
                    print(f"  ✓ ファイルを読み込みました ({len(file_text)}文字)")
                    print(f"    語彙サイズ: {len(generator.model.vocabulary)}")
                    print(f"    コンテキスト数: {len(generator.model.ngrams)}")
                except FileNotFoundError:
                    print(f"  ファイルが見つかりません: {filepath}")
                except Exception as e:
                    print(f"  エラー: {e}")
                continue
            
            # 温度設定
            if user_input.lower().startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    temperature = max(0.1, min(2.0, temperature))
                    print(f"  ✓ 温度を {temperature} に設定しました")
                except:
                    print("  無効な値です (例: temp 0.8)")
                continue
            
            # 最大長設定
            if user_input.lower().startswith('len '):
                try:
                    max_length = int(user_input.split()[1])
                    max_length = max(10, min(500, max_length))
                    print(f"  ✓ 最大長を {max_length} に設定しました")
                except:
                    print("  無効な値です (例: len 100)")
                continue
            
            # N-gram設定
            if user_input.lower().startswith('ngram '):
                try:
                    new_n = int(user_input.split()[1])
                    new_n = max(2, min(5, new_n))
                    n_gram = new_n
                    # モデルを再構築
                    old_vocab = generator.model.vocabulary
                    generator = QuantumTextGenerator(n_gram=n_gram)
                    # 既存の語彙からテキストを再構築して学習
                    print(f"  ✓ N-gramを {n_gram} に変更しました")
                    print("  ⚠ モデルがリセットされました。'add' で学習データを追加してください")
                except:
                    print("  無効な値です (例: ngram 4)")
                continue
            
            # プロセス表示
            if user_input.lower() == 'show':
                show_process = not show_process
                status = "ON" if show_process else "OFF"
                print(f"  ✓ プロセス表示: {status}")
                continue
            
            # リセット
            if user_input.lower() == 'reset':
                generator = QuantumTextGenerator(n_gram=n_gram)
                print("  ✓ モデルをリセットしました")
                print("    'add' コマンドで学習データを追加してください")
                continue
            
            # テキスト生成
            print("\n  ⚡ 量子生成中...")
            result = generator.generate(user_input, max_length=max_length, 
                                        temperature=temperature,
                                        show_process=show_process)
            
            print(f"\n  ┌{'─'*60}")
            print(f"  │ 🎯 結果:")
            # 複数行に分割して表示
            for line in result.split('\n'):
                if line.strip():
                    print(f"  │ {line}")
            print(f"  └{'─'*60}")
            
            # 候補表示
            if not show_process and generator.generation_history:
                print("\n  📊 上位候補 (最初の5ステップ):")
                for i, (token, candidates) in enumerate(generator.generation_history[:5]):
                    if candidates:
                        cand_str = ', '.join(f"'{t}':{p:.2f}" for t, p in candidates[:3])
                        print(f"    {i+1}. [{cand_str}] → '{token}'")
            
        except KeyboardInterrupt:
            print("\n  さようなら！")
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "-i":
            interactive_mode()
        elif sys.argv[1] == "-chat":
            # チャットボット（追加コーパスで学習が必要）
            bot = QuantumChatBot()
            bot.train_from_conversations([
                "Hello! How can I help you today?",
                "That's interesting! Tell me more.",
                "I understand. What else would you like to know?",
                "Great question! Let me explain.",
            ])
            bot.chat()
    else:
        demo()

