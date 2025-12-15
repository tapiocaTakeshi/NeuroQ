# 調整可能擬似量子ビット (APQB: Adjustable Pseudo Quantum Bit)

**ニューラルネットワークと量子多体系の統一理論としてのAPQBフレームワーク**

統計、AI、量子論を統一的に記述する理論モデルです。

## 📚 理論の詳細

詳細な理論と論文は以下のNote記事をご覧ください：

- [調整可能擬似量子ビット（APQB）に基づく量子インスパイアニューラルネットワーク](https://note.com/bright_laelia447/n/n0751aaac9730)
- [APQB（Adjustable Pseudo Quantum Bit）理論](https://note.com/bright_laelia447/n/nc9e892c35fca)
- [私の理論と概念](https://note.com/bright_laelia447/n/n3b7e5124d73d)

## 🎯 基本概念

**調整可能擬似量子ビット（APQB）**は、単一パラメータ θ で以下を統一的に記述します：

- **量子状態**: |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
- **相関係数**: r = cos(2θ)
- **温度（乱雑さ）**: T = |sin(2θ)|
- **制約条件**: r² + T² = 1

## 📐 数学的定式化

### 基本関係式

| パラメータ | 角度 θ | 相関係数 r | 温度 T | 量子状態 |
|-----------|--------|-----------|--------|----------|
| θ = 0 | 0 | r = 1 | T = 0 | \|0⟩ (完全な正の相関) |
| θ = π/4 | π/4 | r = 0 | T = 1 | (\|0⟩ + \|1⟩)/√2 (無相関・最大ゆらぎ) |
| θ = π/2 | π/2 | r = -1 | T = 0 | \|1⟩ (完全な負の相関) |

### 相関係数から角度への変換

```
θ = arccos(r) / 2
```

または、相関係数 r ∈ [-1, 1] から直接：

```
θ = π × (1 - r) / 2
```

### 測定確率

```
P(|0⟩) = cos²(θ)
P(|1⟩) = sin²(θ)
```

## 🔬 理論的背景

APQBフレームワークは以下を実現します：

1. **単一APQBの数学的定義**
   - 量子状態、相関係数、温度を統一的に記述

2. **N体系への一般化と多体相関**
   - k体相関関数: Q_k(θ) = cos(2kθ) (k偶数) / sin(2kθ) (k奇数)
   - 一般化トレードオフ関係の確立

3. **APQBニューラルネットワーク**
   - 量子もつれを模倣した処理
   - 量子確率的な活性化による創造的な生成

4. **複素角度空間での学習**
   - 複素表現: z = e^{i2θ}
   - 最適化アルゴリズムの適用

## 💻 インストール

### 基本インストール

```bash
pip install -r requirements.txt
```

### Git LFSのセットアップ

このリポジトリは **Git LFS（Large File Storage）** を使用して、事前学習済みモデル（`neuroq_pretrained.pt`）を管理しています。

#### 1. Git LFSのインストール

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Windows
# https://git-lfs.github.com/ からダウンロード
```

#### 2. Git LFSの初期化とファイルの取得

```bash
# Git LFSを初期化
git lfs install

# LFSファイルをダウンロード
git lfs pull
```

**重要**: リポジトリをクローンした後、必ず `git lfs pull` を実行してください。これを実行しないと、モデルファイルが133バイトのポインタファイルのままで、実際のモデル（約58MB）がダウンロードされません。

#### 3. ファイルサイズの確認

```bash
# モデルファイルのサイズを確認（58MB程度あれば正常）
ls -lh neuroq_pretrained.pt
```

### Dockerでの実行

#### ビルド方法（推奨）

ヘルパースクリプトを使用すると、Git LFSファイルのチェックを自動的に行ってからビルドします：

```bash
cd neuroq-runpod
./build.sh
```

#### 手動ビルド方法

**方法1: ローカルのファイルを使用**（事前に `git lfs pull` が必要）

```bash
cd neuroq-runpod
docker build -t neuroq:latest .
```

**方法2: リポジトリURLを指定**（Git LFSファイルを自動取得）

```bash
cd neuroq-runpod
docker build \
  --build-arg GIT_REPO_URL=https://github.com/yourusername/NeuroQ.git \
  --build-arg GIT_BRANCH=main \
  -t neuroq:latest .
```

#### コンテナの実行

```bash
docker run --gpus all -p 8000:8000 neuroq:latest
```

## 🚀 使用方法

### Hugging Face (GPT-2) + OpenAI + QBNN パイプライン 【NEW!】

最新の統合パイプラインでは、Hugging FaceのGPT-2トークナイザー、OpenAI Embedding、QBNN処理を組み合わせることができます。

```python
from hf_openai_qbnn_pipeline import HFOpenAIQBNNPipeline, HFQBNNConfig

# Config設定
config = HFQBNNConfig(
    tokenizer_name='gpt2',      # GPT-2トークナイザー
    embed_dim=256,               # 埋め込み次元
    num_heads=8,                 # アテンションヘッド数
    num_layers=6,                # Transformerレイヤー数
    hidden_dim=512,              # 隠れ層次元
    use_qbnn_attention=True,     # QBNN拡張アテンション
    lambda_entangle=0.35         # 量子もつれ強度
)

# パイプライン作成
pipeline = HFOpenAIQBNNPipeline(
    config=config,
    use_openai_embedding=False,  # OpenAI APIを使う場合はTrue
    openai_api_key=None          # APIキーを設定
)

# 学習
texts = [
    "量子コンピュータは革新的な技術です。",
    "Transformerは強力なアーキテクチャです。",
    # ... more texts
]
pipeline.train_model(texts, epochs=20, batch_size=8)

# テキスト生成
output = pipeline.generate(
    "量子コンピュータは",
    max_length=50,
    temperature=0.8
)
print(output)
```

### 基本的な使い方（APQB Framework）

```python
from apqb_framework import APQB

# 角度からAPQBを作成
qubit = APQB(theta=np.pi/4)

# 相関係数と温度を取得
print(f"相関係数 r = {qubit.r:.4f}")
print(f"温度 T = {qubit.T:.4f}")
print(f"制約条件 r² + T² = {qubit.verify_constraint():.4f}")

# 確率を取得
p0, p1 = qubit.p0, qubit.p1
print(f"P(|0⟩) = {p0:.4f}, P(|1⟩) = {p1:.4f}")

# 測定（シミュレーション）
result = qubit.measure()  # 0 または 1
```

### 相関係数からAPQBを生成

```python
# 相関係数 r からAPQBを生成
qubit = APQB.from_r(r=0.5)
print(qubit)  # APQB(θ=..., r=0.5000, T=...)
```

### 温度からAPQBを生成

```python
# 温度 T からAPQBを生成
qubit = APQB.from_T(T=0.8)
print(qubit)
```

### N体系への拡張

```python
from apqb_framework import APQBMultiBody

# N量子ビットAPQB系
multi_body = APQBMultiBody(n=3, theta=np.pi/4)

# k体相関関数を取得
correlation_2 = multi_body.Q_k(k=2)  # 2体相関
correlation_3 = multi_body.Q_k(k=3)  # 3体相関

# 全ての多体相関を取得
all_correlations = multi_body.get_all_correlations()
```

## 📁 ファイル構成

### コア実装

- `apqb_framework.py` - APQBフレームワークのコア実装
  - 単一APQBの定義
  - N体系への一般化
  - APQBニューラルネットワーク

### 統合パイプライン 【NEW!】

- `hf_openai_qbnn_pipeline.py` - Hugging Face + OpenAI + QBNN 統合パイプライン
  - GPT-2トークナイザー (Hugging Face)
  - Hybrid Embedding (HF + OpenAI)
  - GPT-2スタイルの因果的アテンション
  - QBNN拡張レイヤー
  - テキスト生成機能
- `test_hf_qbnn_basic.py` - 統合パイプラインの基本テスト

### 応用例

- `apqb_generative_ai.py` - APQBベースの生成AI
- `apqb_dropout.py` - APQBベースのドロップアウト層
- `qbnn_layered.py` - レイヤード型量子ビットニューラルネットワーク
- `qbnn_brain.py` - 脳型散在量子ビットネットワーク
- `neuroquantum_layered.py` - NeuroQuantum Layered実装
- `neuroquantum_brain.py` - NeuroQuantum Brain実装

### RunPod Serverless

- `neuroq-runpod/` - RunPod Serverless用の実装
  - `handler.py` - RunPod Serverless Handler
  - `train_and_generate.py` - 学習→生成スクリプト
  - `train_request_examples_improved.json` - 改善されたリクエスト例

## 🎓 実行例

```bash
# APQBフレームワークのデモンストレーション
python apqb_framework.py

# 生成AIの実行
python apqb_generative_ai.py

# NeuroQuantum Layered
python neuroquantum_layered.py

# NeuroQuantum Brain
python neuroquantum_brain.py
```

## 📖 参考資料

- **理論論文**: [調整可能擬似量子ビット（APQB）に基づく量子インスパイアニューラルネットワーク](https://note.com/bright_laelia447/n/n0751aaac9730)
- **基本理論**: [APQB（Adjustable Pseudo Quantum Bit）理論](https://note.com/bright_laelia447/n/nc9e892c35fca)
- **作者のNote**: [https://note.com/bright_laelia447](https://note.com/bright_laelia447)
- **RSS Feed**: [https://note.com/bright_laelia447/rss](https://note.com/bright_laelia447/rss)

## ⚠️ 注意事項

この実装は **数学的・プログラム的な調整可能擬似量子ビット（APQB）** であり、実際の量子コンピュータ上で動作する量子ビットではありません。真の量子ビットの実現には専門的な量子ハードウェアが必要です。

APQBは、量子力学の概念を古典コンピュータ上で擬似的に実現し、ニューラルネットワークと量子多体系を統一的に記述するための理論フレームワークです。

## 📝 ライセンス

このプロジェクトは理論研究・教育目的で開発されています。

## 👤 作者

Yuya Higuchi (Riddle Official)

- Note: [https://note.com/bright_laelia447](https://note.com/bright_laelia447)
- RSS: [https://note.com/bright_laelia447/rss](https://note.com/bright_laelia447/rss)
