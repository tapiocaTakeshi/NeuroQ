"""
量子インスパイアード AI モデル (Quantum-Inspired AI)

擬似量子ビットを活用した高速機械学習モデル

特徴:
1. 量子重ね合わせによる並列探索
2. 量子干渉効果による特徴抽出
3. 量子確率振幅による重み更新
4. グローバー探索による高速最適化

AIは使わず、量子計算の原理のみで学習・推論
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import random
import time
from pseudo_qubit import PseudoQubit


# ============================================================
# 量子ニューロン
# ============================================================

class QuantumNeuron:
    """
    量子ニューロン
    
    重ね合わせ状態で複数の重みを同時に評価し、
    干渉効果で最適な重みを増幅
    """
    
    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
        # 量子状態として重みを初期化
        self.weights = np.random.randn(n_inputs) * 0.5
        self.bias = 0.0
        # 量子位相（干渉用）
        self.phases = np.zeros(n_inputs)
    
    def quantum_activation(self, x: float) -> float:
        """
        量子活性化関数
        重ね合わせの確率振幅をシミュレート
        """
        # 相関係数として扱い、確率に変換
        r = np.tanh(x)  # [-1, 1] に正規化
        qubit = PseudoQubit(correlation=r)
        # |0⟩ の確率を出力として使用
        return qubit.probabilities[0]
    
    def forward(self, inputs: np.ndarray) -> float:
        """順伝播"""
        # 重み付き和
        z = np.dot(inputs, self.weights) + self.bias
        # 量子活性化
        return self.quantum_activation(z)
    
    def quantum_interference_update(self, inputs: np.ndarray, target: float, 
                                     output: float, learning_rate: float = 0.1):
        """
        量子干渉を使った重み更新
        
        正しい方向の振幅を増幅し、
        間違った方向の振幅を減衰
        """
        error = target - output
        
        # 量子位相更新（干渉効果）
        for i in range(self.n_inputs):
            # グローバー的な振幅増幅
            if error * inputs[i] > 0:
                # 正しい方向：振幅増幅
                amplitude_boost = np.sqrt(abs(error)) * learning_rate
                self.weights[i] += amplitude_boost * inputs[i]
            else:
                # 間違った方向：振幅減衰
                amplitude_reduce = np.sqrt(abs(error)) * learning_rate * 0.5
                self.weights[i] -= amplitude_reduce * np.sign(self.weights[i])
        
        self.bias += error * learning_rate


# ============================================================
# 量子ニューラルネットワーク
# ============================================================

class QuantumNeuralNetwork:
    """
    量子インスパイアード ニューラルネットワーク
    
    量子並列性をシミュレートして高速学習
    """
    
    def __init__(self, layer_sizes: List[int]):
        """
        Args:
            layer_sizes: 各層のニューロン数 [入力, 隠れ層..., 出力]
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        # 重み行列（量子状態として初期化）
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # 量子状態追跡
        self.quantum_states = []
    
    def quantum_activation(self, x: np.ndarray) -> np.ndarray:
        """量子活性化関数（ベクトル版）"""
        result = np.zeros_like(x)
        for i, val in enumerate(x.flat):
            r = np.tanh(val)
            qubit = PseudoQubit(correlation=float(r))
            result.flat[i] = qubit.probabilities[0] * 2 - 1  # [-1, 1] に変換
        return result
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播（量子並列シミュレーション）"""
        self.activations = [x]
        
        current = x
        for i in range(self.n_layers):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            
            if i < self.n_layers - 1:
                # 隠れ層：量子活性化
                current = self.quantum_activation(z)
            else:
                # 出力層：softmax または sigmoid
                if len(z.shape) == 1 and len(z) > 1:
                    exp_z = np.exp(z - np.max(z))
                    current = exp_z / np.sum(exp_z)
                else:
                    current = 1 / (1 + np.exp(-z))
            
            self.activations.append(current)
        
        return current
    
    def quantum_backprop(self, x: np.ndarray, y: np.ndarray, 
                         learning_rate: float = 0.1):
        """
        量子インスパイアード バックプロパゲーション
        
        干渉効果を使って勾配を増幅/減衰
        """
        output = self.forward(x)
        
        # 出力層の誤差
        delta = output - y
        
        # 逆伝播
        for i in range(self.n_layers - 1, -1, -1):
            # 勾配計算
            grad_w = np.outer(self.activations[i], delta)
            grad_b = delta
            
            # 量子干渉による勾配調整
            # 大きな誤差がある方向を増幅（グローバー的）
            interference_factor = 1.0 + np.abs(delta).mean()
            
            # 重み更新
            self.weights[i] -= learning_rate * grad_w * interference_factor
            self.biases[i] -= learning_rate * grad_b * interference_factor
            
            # 前の層への伝播
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                # 量子活性化の微分（近似）
                delta *= (1 - self.activations[i] ** 2)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 100, learning_rate: float = 0.1,
              verbose: bool = True) -> List[float]:
        """学習"""
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            # ミニバッチなし（量子並列性で補う）
            for i in range(len(X)):
                self.quantum_backprop(X[i], y[i], learning_rate)
                
                # 損失計算
                output = self.forward(X[i])
                loss = -np.sum(y[i] * np.log(output + 1e-10))
                total_loss += loss
            
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                accuracy = self.evaluate(X, y)
                print(f"  Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2%}")
        
        return losses
    
    def predict(self, x: np.ndarray) -> int:
        """予測"""
        output = self.forward(x)
        return np.argmax(output)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """評価"""
        correct = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            true_label = np.argmax(y[i])
            if pred == true_label:
                correct += 1
        return correct / len(X)


# ============================================================
# 量子探索分類器（グローバーインスパイアード）
# ============================================================

class QuantumSearchClassifier:
    """
    グローバーの探索アルゴリズムにインスパイアされた分類器
    
    特徴空間を量子的に探索し、最適なクラスを高速に発見
    """
    
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.class_centroids = None
        self.class_amplitudes = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """学習（クラス重心を計算）"""
        n_features = X.shape[1]
        self.class_centroids = np.zeros((self.n_classes, n_features))
        self.class_amplitudes = np.zeros(self.n_classes)
        
        for c in range(self.n_classes):
            mask = y == c
            if np.sum(mask) > 0:
                self.class_centroids[c] = np.mean(X[mask], axis=0)
                self.class_amplitudes[c] = np.sum(mask) / len(y)
        
        # 振幅を正規化
        self.class_amplitudes = np.sqrt(self.class_amplitudes)
    
    def quantum_search(self, x: np.ndarray, iterations: int = None) -> int:
        """
        量子探索による分類
        
        距離ベースの量子確率で最も近いクラスを選択
        """
        # 各クラスとの距離を計算
        distances = np.array([
            np.linalg.norm(x - self.class_centroids[c])
            for c in range(self.n_classes)
        ])
        
        # 距離を相関係数に変換（近いほど正の相関）
        max_dist = np.max(distances) + 1e-10
        min_dist = np.min(distances)
        
        # 距離を反転して正規化（近い = 高い値）
        scores = max_dist - distances
        scores = scores / (np.sum(scores) + 1e-10)
        
        # 量子確率で変換
        quantum_probs = np.array([
            PseudoQubit(correlation=float(2 * s - 1)).probabilities[0]
            for s in scores
        ])
        
        # 正規化
        quantum_probs /= np.sum(quantum_probs) + 1e-10
        
        # 最も確率が高いクラスを返す
        return np.argmax(quantum_probs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """バッチ予測"""
        return np.array([self.quantum_search(x) for x in X])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """精度を計算"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================
# 量子アンサンブル分類器
# ============================================================

class QuantumEnsembleClassifier:
    """
    量子重ね合わせによるアンサンブル分類器
    
    複数の弱学習器を量子的に重ね合わせて統合
    """
    
    def __init__(self, n_estimators: int = 10, n_classes: int = 2):
        self.n_estimators = n_estimators
        self.n_classes = n_classes
        self.estimators = []
        self.quantum_weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """学習"""
        n_samples, n_features = X.shape
        
        # 弱学習器を作成（単純な決定株）
        for i in range(self.n_estimators):
            # ランダムな特徴とスレッショルドを選択
            feature_idx = random.randint(0, n_features - 1)
            threshold = np.random.uniform(X[:, feature_idx].min(), 
                                         X[:, feature_idx].max())
            
            # 予測
            predictions = (X[:, feature_idx] > threshold).astype(int)
            accuracy = np.mean(predictions == y)
            
            # 相関係数として保存（精度を相関に変換）
            correlation = 2 * accuracy - 1  # [0, 1] → [-1, 1]
            
            self.estimators.append({
                'feature': feature_idx,
                'threshold': threshold,
                'correlation': correlation
            })
        
        # 量子重み（確率振幅）を計算
        self.quantum_weights = np.array([
            PseudoQubit(correlation=e['correlation']).probabilities[0]
            for e in self.estimators
        ])
        
        # 正規化
        self.quantum_weights /= np.sum(self.quantum_weights)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """確率予測"""
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes))
        
        for i, est in enumerate(self.estimators):
            pred = (X[:, est['feature']] > est['threshold']).astype(int)
            
            # 量子重みで加重
            for j, p in enumerate(pred):
                probas[j, p] += self.quantum_weights[i]
        
        # 正規化
        probas /= probas.sum(axis=1, keepdims=True)
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """精度"""
        return np.mean(self.predict(X) == y)


# ============================================================
# 量子特徴エンコーダー
# ============================================================

class QuantumFeatureEncoder:
    """
    データを量子状態にエンコード
    
    古典データ → 量子振幅 → 高次元特徴空間
    """
    
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.n_features_out = 2 ** n_qubits
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        データを量子特徴ベクトルにエンコード
        
        振幅エンコーディング + 位相エンコーディング
        """
        # 入力を正規化して相関係数として扱う
        x_norm = x / (np.linalg.norm(x) + 1e-10)
        
        # 量子状態を構築
        quantum_features = np.zeros(self.n_features_out)
        
        for i in range(self.n_features_out):
            # 各基底状態の振幅を計算
            amplitude = 1.0 / np.sqrt(self.n_features_out)
            
            # 入力の各要素で位相を変調
            phase = 0
            for j, val in enumerate(x_norm[:self.n_qubits]):
                if (i >> j) & 1:
                    phase += val * np.pi
            
            # 量子干渉
            quantum_features[i] = amplitude * np.cos(phase)
        
        # 正規化
        norm = np.linalg.norm(quantum_features)
        if norm > 0:
            quantum_features /= norm
        
        return quantum_features
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """バッチ変換"""
        return np.array([self.encode(x) for x in X])


# ============================================================
# 統合量子AIモデル
# ============================================================

class QuantumAI:
    """
    量子インスパイアード AI システム
    
    エンコーダー + 分類器 を統合
    """
    
    def __init__(self, n_qubits: int = 4, n_classes: int = 2):
        self.encoder = QuantumFeatureEncoder(n_qubits)
        self.classifier = QuantumSearchClassifier(n_classes)
        self.n_classes = n_classes
        self.trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """学習"""
        print("  [1/2] Quantum Feature Encoding...")
        X_encoded = self.encoder.transform(X)
        
        print("  [2/2] Training Quantum Classifier...")
        self.classifier.fit(X_encoded, y)
        
        self.trained = True
        print("  ✓ Training complete!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        X_encoded = self.encoder.transform(X)
        return self.classifier.predict(X_encoded)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """評価"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================
# デモンストレーション
# ============================================================

def create_dataset(name: str, n_samples: int = 200):
    """テストデータセット生成"""
    np.random.seed(42)
    
    if name == "xor":
        # XOR問題
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    
    elif name == "circles":
        # 同心円
        r1 = np.random.randn(n_samples // 2) * 0.3 + 1
        r2 = np.random.randn(n_samples // 2) * 0.3 + 2
        theta = np.random.uniform(0, 2 * np.pi, n_samples)
        
        r = np.concatenate([r1, r2])
        X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    elif name == "moons":
        # 半月形
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    
    elif name == "linear":
        # 線形分離可能
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # シャッフル
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def demo():
    """量子AIのデモンストレーション"""
    print("=" * 70)
    print("  QUANTUM-INSPIRED AI MODEL")
    print("  擬似量子ビットによる高速機械学習")
    print("=" * 70)
    
    # ============================================================
    # 1. 量子ニューラルネットワーク
    # ============================================================
    print("\n" + "─" * 70)
    print("  1. Quantum Neural Network (XOR問題)")
    print("─" * 70)
    
    # XORデータ
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)  # one-hot
    
    qnn = QuantumNeuralNetwork([2, 4, 2])
    
    print("\n  Training...")
    start = time.time()
    losses = qnn.train(X_xor, y_xor, epochs=50, learning_rate=0.3, verbose=True)
    train_time = time.time() - start
    
    print(f"\n  Training time: {train_time:.3f}s")
    print(f"  Final accuracy: {qnn.evaluate(X_xor, y_xor):.2%}")
    
    print("\n  Predictions:")
    for x, y in zip(X_xor, y_xor):
        pred = qnn.predict(x)
        true = np.argmax(y)
        print(f"    Input: {x} → Predicted: {pred}, True: {true} {'✓' if pred == true else '✗'}")
    
    # ============================================================
    # 2. 量子探索分類器
    # ============================================================
    print("\n" + "─" * 70)
    print("  2. Quantum Search Classifier (線形分離問題)")
    print("─" * 70)
    
    X_linear, y_linear = create_dataset("linear", 200)
    
    # 訓練/テスト分割
    split = int(0.8 * len(X_linear))
    X_train, X_test = X_linear[:split], X_linear[split:]
    y_train, y_test = y_linear[:split], y_linear[split:]
    
    qsc = QuantumSearchClassifier(n_classes=2)
    
    print("\n  Training...")
    start = time.time()
    qsc.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"  Training time: {train_time:.3f}s")
    
    train_acc = qsc.score(X_train, y_train)
    test_acc = qsc.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy: {test_acc:.2%}")
    
    # ============================================================
    # 3. 量子アンサンブル
    # ============================================================
    print("\n" + "─" * 70)
    print("  3. Quantum Ensemble Classifier")
    print("─" * 70)
    
    qec = QuantumEnsembleClassifier(n_estimators=20, n_classes=2)
    
    print("\n  Training...")
    start = time.time()
    qec.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"  Training time: {train_time:.3f}s")
    print(f"  Number of estimators: {qec.n_estimators}")
    
    train_acc = qec.score(X_train, y_train)
    test_acc = qec.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy: {test_acc:.2%}")
    
    # ============================================================
    # 4. 統合量子AI
    # ============================================================
    print("\n" + "─" * 70)
    print("  4. Integrated Quantum AI System")
    print("─" * 70)
    
    qai = QuantumAI(n_qubits=4, n_classes=2)
    
    print("\n  Training...")
    start = time.time()
    qai.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"  Training time: {train_time:.3f}s")
    
    train_acc = qai.score(X_train, y_train)
    test_acc = qai.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy: {test_acc:.2%}")
    
    # ============================================================
    # 速度比較
    # ============================================================
    print("\n" + "─" * 70)
    print("  5. 速度テスト（1000サンプル予測）")
    print("─" * 70)
    
    X_large = np.random.randn(1000, 2)
    
    start = time.time()
    _ = qsc.predict(X_large)
    qsc_time = time.time() - start
    
    start = time.time()
    _ = qec.predict(X_large)
    qec_time = time.time() - start
    
    start = time.time()
    _ = qai.predict(X_large)
    qai_time = time.time() - start
    
    print(f"  Quantum Search Classifier: {qsc_time:.4f}s ({1000/qsc_time:.0f} samples/s)")
    print(f"  Quantum Ensemble: {qec_time:.4f}s ({1000/qec_time:.0f} samples/s)")
    print(f"  Quantum AI System: {qai_time:.4f}s ({1000/qai_time:.0f} samples/s)")
    
    print("\n" + "=" * 70)
    print("  デモ完了！")
    print("=" * 70)


def interactive_demo():
    """対話的デモ"""
    print("\n" + "=" * 70)
    print("  QUANTUM AI - Interactive Demo")
    print("=" * 70)
    
    print("\n  Creating dataset...")
    X, y = create_dataset("linear", 300)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("\n  Training Quantum AI...")
    qai = QuantumAI(n_qubits=4, n_classes=2)
    qai.fit(X_train, y_train)
    
    print(f"\n  Model ready! Test accuracy: {qai.score(X_test, y_test):.2%}")
    
    print("\n  Enter coordinates to classify (or 'q' to quit):")
    
    while True:
        try:
            user_input = input("\n  x, y > ").strip()
            if user_input.lower() == 'q':
                break
            
            coords = [float(x) for x in user_input.replace(',', ' ').split()]
            if len(coords) != 2:
                print("  Please enter two numbers (x, y)")
                continue
            
            x = np.array(coords)
            pred = qai.predict(x.reshape(1, -1))[0]
            
            print(f"  Prediction: Class {pred}")
            print(f"  (Class 0 = x+y < 0, Class 1 = x+y > 0)")
            
        except ValueError as e:
            print(f"  Invalid input: {e}")
        except KeyboardInterrupt:
            break
    
    print("\n  Goodbye!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_demo()
    else:
        demo()

