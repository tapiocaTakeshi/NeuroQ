# 擬似量子ビット (Pseudo-Qubit)

相関係数 r ∈ [-1, 1] を量子ビットの状態にマッピングする数学的・プログラム的実装です。

## マッピング原理

| 相関係数 r | 角度 θ | 量子状態 | 説明 |
|-----------|--------|----------|------|
| r = 1 | θ = 0 | \|0⟩ | 完全な正の相関 → 純粋状態 |
| r = 0 | θ = π/2 | (|0⟩ + |1⟩)/√2 | 無相関 → 完全な重ね合わせ |
| r = -1 | θ = π | \|1⟩ | 完全な負の相関 → 純粋状態 |

## 数学的定式化

### 相関係数から角度への変換
```
θ = π × (1 - r) / 2
```

### 量子状態
```
|ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
```

### 測定確率
```
P(|0⟩) = cos²(θ/2)
P(|1⟩) = sin²(θ/2)
```

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使い方

```python
from pseudo_qubit import PseudoQubit

# 相関係数から擬似量子ビットを作成
qubit = PseudoQubit(correlation=0.5)

# 状態を確認
print(qubit)

# 確率を取得
p0, p1 = qubit.probabilities
print(f"P(|0⟩) = {p0}, P(|1⟩) = {p1}")

# 測定（シミュレーション）
result = qubit.measure()  # 0 または 1

# 統計測定
stats = qubit.measure_n(1000)
```

### データから量子ビットを生成

```python
import numpy as np
from pseudo_qubit import create_qubit_from_data

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

qubit = create_qubit_from_data(x, y)
```

## ファイル構成

- `pseudo_qubit.py` - 擬似量子ビットのコア実装
- `visualize_qubit.py` - ブロッホ球と確率分布の視覚化
- `requirements.txt` - 依存パッケージ

## 実行

```bash
# デモンストレーション
python pseudo_qubit.py

# 視覚化
python visualize_qubit.py
```

## 注意事項

この実装は **数学的・プログラム的な擬似量子ビット** であり、実際の量子コンピュータ上で動作する量子ビットではありません。真の量子ビットの実現には専門的な量子ハードウェアが必要です。

