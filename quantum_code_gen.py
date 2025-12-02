"""
量子インスパイアード コード生成AI

擬似量子ビットを活用した高速コード生成
- 量子重ね合わせで複数のコードパターンを同時評価
- 量子干渉で最適なコードを増幅
- 文脈理解による適切な補完
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import re
import time
from pseudo_qubit import PseudoQubit


# ============================================================
# 量子トークン状態
# ============================================================

class QuantumCodeState:
    """コードトークンの量子状態"""
    
    def __init__(self, tokens: List[str], scores: Optional[np.ndarray] = None):
        self.tokens = tokens
        self.n_tokens = len(tokens)
        
        if scores is None:
            self.amplitudes = np.ones(self.n_tokens) / np.sqrt(max(self.n_tokens, 1))
        else:
            self.amplitudes = np.sqrt(np.abs(scores) + 1e-10)
            self._normalize()
    
    def _normalize(self):
        norm = np.sqrt(np.sum(self.amplitudes ** 2))
        if norm > 0:
            self.amplitudes /= norm
    
    @property
    def probabilities(self) -> np.ndarray:
        return self.amplitudes ** 2
    
    def amplify(self, indices: List[int], factor: float = 2.0):
        """振幅増幅"""
        for i in indices:
            if 0 <= i < self.n_tokens:
                self.amplitudes[i] *= factor
        self._normalize()
    
    def interfere(self, correlations: np.ndarray):
        """量子干渉"""
        for i, r in enumerate(correlations):
            qubit = PseudoQubit(correlation=float(r))
            self.amplitudes[i] *= np.sqrt(qubit.probabilities[0] + 0.1)
        self._normalize()
    
    def measure(self) -> str:
        """測定"""
        if self.n_tokens == 0:
            return ""
        probs = self.probabilities
        probs /= probs.sum()
        idx = np.random.choice(self.n_tokens, p=probs)
        return self.tokens[idx]
    
    def top_k(self, k: int = 5) -> List[Tuple[str, float]]:
        """上位k個"""
        probs = self.probabilities
        indices = np.argsort(probs)[::-1][:k]
        return [(self.tokens[i], probs[i]) for i in indices if i < len(self.tokens)]


# ============================================================
# コードパターン学習
# ============================================================

class CodePatternLearner:
    """コードパターンを学習"""
    
    def __init__(self):
        # パターン: context -> next_tokens
        self.patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.keywords: Dict[str, Set[str]] = {
            'python': {'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif',
                      'for', 'while', 'try', 'except', 'with', 'as', 'yield', 'lambda',
                      'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'pass',
                      'break', 'continue', 'raise', 'finally', 'global', 'nonlocal',
                      'assert', 'async', 'await', 'print', 'len', 'range', 'list',
                      'dict', 'set', 'str', 'int', 'float', 'bool', 'self', '__init__'},
            'javascript': {'function', 'const', 'let', 'var', 'return', 'if', 'else',
                          'for', 'while', 'try', 'catch', 'throw', 'new', 'class',
                          'constructor', 'this', 'async', 'await', 'import', 'export',
                          'default', 'from', 'true', 'false', 'null', 'undefined',
                          'console', 'log', 'document', 'window', 'array', 'object'}
        }
        self.common_patterns = self._init_common_patterns()
    
    def _init_common_patterns(self) -> Dict[str, List[str]]:
        """よく使うコードパターン"""
        return {
            # Python patterns
            'def ': ['__init__(self', 'main()', 'get_', 'set_', 'is_', 'has_', 'calculate_'],
            'def __init__(self': ['):', ', name):', ', value):', ', *args, **kwargs):'],
            'class ': ['Solution:', 'Model:', 'Handler:', 'Manager:', 'Config:'],
            'import ': ['numpy as np', 'pandas as pd', 'torch', 'os', 'sys', 'json', 're'],
            'from ': ['typing import', 'collections import', 'dataclasses import'],
            'if ': ['__name__ == "__main__":', 'x is None:', 'len(', 'not ', 'x > ', 'x == '],
            'for ': ['i in range(', 'item in ', 'key, value in ', 'i, x in enumerate('],
            'return ': ['None', 'True', 'False', 'result', 'self.', '[]', '{}'],
            'print(': ['f"', '"', 'x)', 'result)'],
            'self.': ['name', 'value', 'data', 'config', '_', 'model'],
            
            # JavaScript patterns
            'function ': ['() {', 'main() {', 'handleClick() {', 'getData() {'],
            'const ': ['result = ', 'data = ', 'config = {', 'arr = ['],
            'let ': ['i = 0', 'result = ', 'temp = '],
            'console.': ['log(', 'error(', 'warn('],
            'document.': ['getElementById(', 'querySelector(', 'createElement('],
            'async ': ['function ', '() => {'],
            'await ': ['fetch(', 'response.json()', 'Promise.all('],
            
            # Common
            '(': [')', 'x)', 'self)', 'i)', 'data)', '"'],
            '[': [']', 'i]', '0]', '-1]', ':'],
            '{': ['}', '\n    ', '"key": '],
            '"': ['"', "'"],
            ':': ['\n    ', ' ', ''],
        }
    
    def tokenize(self, code: str) -> List[str]:
        """コードをトークン化"""
        # シンプルなトークン化
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[^\s]', code)
        return tokens
    
    def learn(self, code: str, context_size: int = 3):
        """コードからパターンを学習"""
        tokens = self.tokenize(code)
        
        for i in range(len(tokens) - context_size):
            context = ' '.join(tokens[i:i + context_size])
            next_token = tokens[i + context_size]
            self.patterns[context][next_token] += 1
    
    def get_suggestions(self, context: str, language: str = 'python') -> QuantumCodeState:
        """文脈から候補を量子状態で返す"""
        candidates = []
        scores = []
        
        # パターンマッチング
        for pattern, completions in self.common_patterns.items():
            if context.endswith(pattern):
                for completion in completions:
                    candidates.append(completion)
                    scores.append(2.0)
        
        # 学習したパターン
        tokens = self.tokenize(context)
        if len(tokens) >= 3:
            ctx = ' '.join(tokens[-3:])
            if ctx in self.patterns:
                for token, count in self.patterns[ctx].items():
                    if token not in candidates:
                        candidates.append(token)
                        scores.append(count)
        
        # キーワード補完
        last_word = tokens[-1] if tokens else ''
        keywords = self.keywords.get(language, set())
        for kw in keywords:
            if kw.startswith(last_word) and kw != last_word:
                if kw not in candidates:
                    candidates.append(kw)
                    scores.append(1.5)
        
        # デフォルト候補
        if not candidates:
            candidates = ['(', ')', ':', '\n', ' ', '=', ',', '.']
            scores = [1.0] * len(candidates)
        
        return QuantumCodeState(candidates, np.array(scores))


# ============================================================
# 量子コード生成器
# ============================================================

class QuantumCodeGenerator:
    """量子インスパイアード コード生成AI"""
    
    def __init__(self, language: str = 'python'):
        self.language = language
        self.learner = CodePatternLearner()
        self.templates = self._init_templates()
        self.indent_level = 0
    
    def _init_templates(self) -> Dict[str, str]:
        """コードテンプレート"""
        if self.language == 'python':
            return {
                'function': 'def {name}({params}):\n    {body}\n    return {return_value}',
                'class': 'class {name}:\n    def __init__(self{params}):\n        {init_body}',
                'if': 'if {condition}:\n    {body}',
                'for': 'for {var} in {iterable}:\n    {body}',
                'while': 'while {condition}:\n    {body}',
                'try': 'try:\n    {body}\nexcept {exception}:\n    {handler}',
                'import': 'import {module}',
                'from_import': 'from {module} import {items}',
                'list_comp': '[{expr} for {var} in {iterable}]',
                'dict_comp': '{{{key}: {value} for {var} in {iterable}}}',
                'lambda': 'lambda {params}: {expr}',
                'with': 'with {context} as {var}:\n    {body}',
                'main': 'if __name__ == "__main__":\n    main()',
            }
        else:  # javascript
            return {
                'function': 'function {name}({params}) {{\n    {body}\n    return {return_value};\n}}',
                'arrow': 'const {name} = ({params}) => {{\n    {body}\n}};',
                'class': 'class {name} {{\n    constructor({params}) {{\n        {init_body}\n    }}\n}}',
                'if': 'if ({condition}) {{\n    {body}\n}}',
                'for': 'for (let {var} = 0; {var} < {limit}; {var}++) {{\n    {body}\n}}',
                'foreach': '{array}.forEach(({item}) => {{\n    {body}\n}});',
                'try': 'try {{\n    {body}\n}} catch ({exception}) {{\n    {handler}\n}}',
                'import': 'import {items} from "{module}";',
                'export': 'export {type} {name};',
                'async': 'async function {name}({params}) {{\n    {body}\n}}',
                'fetch': 'const response = await fetch("{url}");\nconst data = await response.json();',
            }
    
    def train(self, code_samples: List[str]):
        """コードサンプルから学習"""
        for code in code_samples:
            self.learner.learn(code)
    
    def generate_from_template(self, template_name: str, **kwargs) -> str:
        """テンプレートからコード生成"""
        if template_name not in self.templates:
            return f"# Unknown template: {template_name}"
        
        template = self.templates[template_name]
        
        # デフォルト値を設定
        defaults = {
            'name': 'func',
            'params': '',
            'body': 'pass' if self.language == 'python' else '// TODO',
            'return_value': 'None' if self.language == 'python' else 'null',
            'condition': 'True' if self.language == 'python' else 'true',
            'var': 'i',
            'iterable': 'range(10)' if self.language == 'python' else '10',
            'exception': 'Exception' if self.language == 'python' else 'error',
            'handler': 'pass' if self.language == 'python' else 'console.error(error);',
            'module': 'os',
            'items': '*',
            'init_body': 'pass' if self.language == 'python' else 'this.value = value;',
        }
        defaults.update(kwargs)
        
        return template.format(**defaults)
    
    def complete(self, code: str, max_tokens: int = 20, 
                 temperature: float = 0.7) -> str:
        """コード補完"""
        result = code
        
        for _ in range(max_tokens):
            # 候補を取得
            state = self.learner.get_suggestions(result, self.language)
            
            if state.n_tokens == 0:
                break
            
            # 文脈スコアリング
            context_scores = self._compute_context_scores(result, state.tokens)
            state.interfere(context_scores)
            
            # 温度適用
            if temperature != 1.0:
                state.amplitudes = state.amplitudes ** (1.0 / temperature)
                state._normalize()
            
            # 量子測定
            next_token = state.measure()
            
            if not next_token:
                break
            
            result += next_token
            
            # 終了条件
            if next_token in ['\n\n', '```']:
                break
        
        return result
    
    def _compute_context_scores(self, context: str, candidates: List[str]) -> np.ndarray:
        """文脈スコア"""
        scores = np.zeros(len(candidates))
        
        for i, token in enumerate(candidates):
            score = 0.0
            
            # インデント整合性
            if context.endswith(':'):
                if token.startswith('\n') or token.startswith('    '):
                    score += 1.0
            
            # 括弧の整合性
            open_parens = context.count('(') - context.count(')')
            if open_parens > 0 and ')' in token:
                score += 0.8
            
            open_brackets = context.count('[') - context.count(']')
            if open_brackets > 0 and ']' in token:
                score += 0.8
            
            # コロン後のインデント
            lines = context.split('\n')
            if lines and lines[-1].strip().endswith(':'):
                if token == '    ' or token.startswith('    '):
                    score += 1.5
            
            # キーワード後の空白
            keywords = ['def', 'class', 'if', 'for', 'while', 'import', 'from', 'return']
            for kw in keywords:
                if context.rstrip().endswith(kw):
                    if token == ' ':
                        score += 2.0
            
            scores[i] = score
        
        # 正規化
        if np.max(scores) > 0:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
            scores = 2 * scores - 1  # [-1, 1]
        
        return scores
    
    def generate_function(self, description: str) -> str:
        """説明からコード生成"""
        # 説明を解析
        words = description.lower().split()
        
        # 関数名を推測
        name = 'process'
        for word in words:
            if word in ['calculate', 'compute', 'get', 'set', 'create', 'make', 
                       'find', 'search', 'sort', 'filter', 'check', 'validate']:
                name = word
                break
        
        # パラメータを推測
        params = []
        if 'list' in words or 'array' in words:
            params.append('data')
        if 'number' in words or 'value' in words:
            params.append('value')
        if 'string' in words or 'text' in words:
            params.append('text')
        if not params:
            params = ['x']
        
        # テンプレート生成
        code = self.generate_from_template('function', 
                                           name=name, 
                                           params=', '.join(params))
        
        # 説明をコメントとして追加
        if self.language == 'python':
            comment = f'"""{description}"""'
        else:
            comment = f'// {description}'
        
        lines = code.split('\n')
        lines.insert(1, f'    {comment}')
        
        return '\n'.join(lines)
    
    def suggest_completions(self, code: str, n: int = 5) -> List[Tuple[str, float]]:
        """補完候補を提案"""
        state = self.learner.get_suggestions(code, self.language)
        context_scores = self._compute_context_scores(code, state.tokens)
        state.interfere(context_scores)
        return state.top_k(n)


# ============================================================
# インタラクティブ コードエディタ
# ============================================================

class QuantumCodeEditor:
    """量子コードエディタ"""
    
    def __init__(self, language: str = 'python'):
        self.generator = QuantumCodeGenerator(language)
        self.code = ""
        self.history: List[str] = []
        
        # サンプルコードで学習
        self._train_with_samples()
    
    def _train_with_samples(self):
        """サンプルコードで学習"""
        python_samples = [
            '''def hello_world():
    print("Hello, World!")
    return None''',
            '''def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total''',
            '''class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
        return self.value''',
            '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)''',
            '''import numpy as np

def matrix_multiply(a, b):
    return np.dot(a, b)''',
            '''def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)''',
            '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',
        ]
        
        self.generator.train(python_samples)
    
    def complete_code(self, partial_code: str, max_tokens: int = 30) -> str:
        """コードを補完"""
        return self.generator.complete(partial_code, max_tokens)
    
    def generate_function(self, description: str) -> str:
        """説明から関数生成"""
        return self.generator.generate_function(description)
    
    def get_suggestions(self, code: str) -> List[Tuple[str, float]]:
        """補完候補を取得"""
        return self.generator.suggest_completions(code)


# ============================================================
# デモ
# ============================================================

def demo():
    """コード生成AIデモ"""
    print("=" * 70)
    print("  量子インスパイアード コード生成AI")
    print("  Quantum-Inspired Code Generation AI")
    print("=" * 70)
    
    editor = QuantumCodeEditor('python')
    
    # ============================================================
    # 1. コード補完
    # ============================================================
    print("\n" + "─" * 70)
    print("  1. コード補完 (Code Completion)")
    print("─" * 70)
    
    test_codes = [
        "def ",
        "def calculate_",
        "for i in ",
        "if __name__ == ",
        "import ",
        "class Solution",
        "return ",
    ]
    
    for code in test_codes:
        print(f"\n  入力: '{code}'")
        
        # 候補表示
        suggestions = editor.get_suggestions(code)
        print(f"  候補: {[f'{s[0]}({s[1]:.2f})' for s in suggestions[:5]]}")
        
        # 補完
        completed = editor.complete_code(code, max_tokens=15)
        print(f"  補完: {completed}")
    
    # ============================================================
    # 2. 関数生成
    # ============================================================
    print("\n" + "─" * 70)
    print("  2. 説明から関数生成 (Function Generation)")
    print("─" * 70)
    
    descriptions = [
        "calculate the sum of a list",
        "check if a number is prime",
        "sort an array in ascending order",
        "find the maximum value",
        "validate user input string",
    ]
    
    for desc in descriptions:
        print(f"\n  説明: '{desc}'")
        code = editor.generate_function(desc)
        print(f"  生成コード:")
        for line in code.split('\n'):
            print(f"    {line}")
    
    # ============================================================
    # 3. テンプレート生成
    # ============================================================
    print("\n" + "─" * 70)
    print("  3. テンプレートからコード生成")
    print("─" * 70)
    
    templates = [
        ('function', {'name': 'process_data', 'params': 'data, options'}),
        ('class', {'name': 'DataProcessor', 'params': ', config'}),
        ('for', {'var': 'item', 'iterable': 'items'}),
        ('try', {'body': 'result = risky_operation()', 'exception': 'ValueError as e'}),
        ('list_comp', {'expr': 'x * 2', 'var': 'x', 'iterable': 'numbers'}),
    ]
    
    for template_name, params in templates:
        print(f"\n  テンプレート: {template_name}")
        code = editor.generator.generate_from_template(template_name, **params)
        print(f"  生成:")
        for line in code.split('\n'):
            print(f"    {line}")
    
    # ============================================================
    # 4. 量子並列補完
    # ============================================================
    print("\n" + "─" * 70)
    print("  4. 量子並列補完プロセス")
    print("─" * 70)
    
    code = "def fibonacci("
    print(f"\n  入力: '{code}'")
    
    state = editor.generator.learner.get_suggestions(code, 'python')
    print(f"\n  量子状態 (重ね合わせ):")
    print(f"    候補数: {state.n_tokens}")
    print(f"    上位候補:")
    for token, prob in state.top_k(5):
        bar = '█' * int(prob * 30)
        print(f"      '{token}': {prob:.3f} {bar}")
    
    print(f"\n  量子測定 (5回):")
    for i in range(5):
        measured = state.measure()
        print(f"    測定{i+1}: '{measured}'")
    
    # ============================================================
    # 5. 速度テスト
    # ============================================================
    print("\n" + "─" * 70)
    print("  5. 生成速度テスト")
    print("─" * 70)
    
    test_prompts = ["def ", "class ", "for ", "import "]
    
    for prompt in test_prompts:
        start = time.time()
        for _ in range(100):
            editor.complete_code(prompt, max_tokens=20)
        elapsed = time.time() - start
        
        print(f"  '{prompt}' → {100/elapsed:.1f} 補完/秒")
    
    print("\n" + "=" * 70)
    print("  デモ完了！")
    print("=" * 70)


def interactive_mode():
    """対話モード"""
    print("\n" + "=" * 70)
    print("  量子コード生成AI - 対話モード")
    print("=" * 70)
    
    editor = QuantumCodeEditor('python')
    
    print("""
  コマンド:
    [コード]        - コードを補完
    gen [説明]      - 説明から関数生成
    temp [名前]     - テンプレートからコード生成
    suggest [コード] - 補完候補を表示
    lang [言語]     - 言語変更 (python/javascript)
    help            - ヘルプ
    quit            - 終了
    """)
    
    while True:
        try:
            user_input = input("\n  コード> ").rstrip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'q', '終了']:
                print("  さようなら！")
                break
            
            if user_input.lower() == 'help':
                print("""
  ┌─────────────────────────────────────────────────────────┐
  │ コマンド一覧                                            │
  ├─────────────────────────────────────────────────────────┤
  │ [コード]       コードの続きを量子生成                   │
  │                例: def calculate_                       │
  │                                                         │
  │ gen [説明]     説明から関数を生成                       │
  │                例: gen calculate the sum of a list      │
  │                                                         │
  │ temp [名前]    テンプレートからコード生成               │
  │                名前: function, class, for, while,       │
  │                      if, try, import, list_comp, lambda │
  │                                                         │
  │ suggest [コード] 補完候補を表示                         │
  │                                                         │
  │ lang [言語]    言語変更 (python / javascript)           │
  └─────────────────────────────────────────────────────────┘
                """)
                continue
            
            # 関数生成
            if user_input.lower().startswith('gen '):
                description = user_input[4:].strip()
                print("\n  ⚡ 量子生成中...")
                code = editor.generate_function(description)
                print("\n  ┌" + "─" * 50)
                print("  │ 生成コード:")
                for line in code.split('\n'):
                    print(f"  │ {line}")
                print("  └" + "─" * 50)
                continue
            
            # テンプレート
            if user_input.lower().startswith('temp '):
                template_name = user_input[5:].strip()
                code = editor.generator.generate_from_template(template_name)
                print("\n  ┌" + "─" * 50)
                print(f"  │ テンプレート: {template_name}")
                for line in code.split('\n'):
                    print(f"  │ {line}")
                print("  └" + "─" * 50)
                continue
            
            # 候補表示
            if user_input.lower().startswith('suggest '):
                code = user_input[8:]
                suggestions = editor.get_suggestions(code)
                print("\n  補完候補:")
                for i, (token, prob) in enumerate(suggestions):
                    bar = '█' * int(prob * 20)
                    print(f"    {i+1}. '{token}' ({prob:.3f}) {bar}")
                continue
            
            # 言語変更
            if user_input.lower().startswith('lang '):
                lang = user_input[5:].strip().lower()
                if lang in ['python', 'javascript', 'js']:
                    lang = 'python' if lang == 'python' else 'javascript'
                    editor = QuantumCodeEditor(lang)
                    print(f"  ✓ 言語を {lang} に変更しました")
                else:
                    print("  対応言語: python, javascript")
                continue
            
            # コード補完
            print("\n  ⚡ 量子補完中...")
            start = time.time()
            completed = editor.complete_code(user_input, max_tokens=40)
            elapsed = time.time() - start
            
            print("\n  ┌" + "─" * 50)
            print("  │ 補完結果:")
            for line in completed.split('\n'):
                print(f"  │ {line}")
            print("  └" + "─" * 50)
            print(f"  ⏱ 生成時間: {elapsed:.3f}秒")
            
            # 候補も表示
            suggestions = editor.get_suggestions(user_input)
            if suggestions:
                print("\n  📊 他の候補: ", end='')
                print(', '.join([f"'{s[0]}'" for s in suggestions[:3]]))
            
        except KeyboardInterrupt:
            print("\n  さようなら！")
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_mode()
    else:
        demo()

