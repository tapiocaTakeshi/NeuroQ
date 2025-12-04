# NeuroQ - RunPod Serverless

QBNN（量子ビットニューラルネットワーク）ベースの生成AI

## 🧠 2つのモード

### Brain Mode（脳型散在QBNN）
- **特徴**: 人間の脳の神経回路を模倣した散在的なニューロン配置
- **動的選択**: 入力感度と出力傾向に基づいてニューロンを動的に選択
- **量子もつれ**: ニューロン間の相関を量子もつれとして表現
- **用途**: 創造的なタスク、柔軟な応答生成

### Layered Mode（層状QBNN-Transformer）
- **特徴**: Transformerアーキテクチャとの組み合わせ
- **自己注意**: 量子ビット重み付き注意機構
- **量子もつれ層**: レイヤー間の量子もつれ演算
- **用途**: 安定した応答生成、長文処理

---

## 🚀 API リファレンス

### 基本的な呼び出し

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "prompt": "こんにちは",
      "max_tokens": 64
    }
  }'
```

### Brain Mode でニューロン数を指定

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "prompt": "量子とは何ですか",
      "mode": "brain",
      "num_neurons": 500,
      "connection_density": 0.3,
      "lambda_entangle": 0.4,
      "max_tokens": 128,
      "temperature": 0.8
    }
  }'
```

### Layered Mode で隠れ層次元を指定

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "prompt": "Hello",
      "mode": "layered",
      "hidden_dim": 512,
      "num_heads": 8,
      "num_layers": 4,
      "max_tokens": 128,
      "temperature": 0.7
    }
  }'
```

---

## 📝 API パラメータ

### 生成パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `prompt` | string | (必須) | 入力プロンプト |
| `max_tokens` | int | 128 | 最大生成トークン数 |
| `temperature` | float | 0.7 | 温度（0.1-2.0） |
| `top_k` | int | 40 | Top-K サンプリング |
| `top_p` | float | 0.9 | Top-P (Nucleus) サンプリング |
| `repetition_penalty` | float | 1.2 | 繰り返しペナルティ |

### モデル設定パラメータ

#### 共通

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `mode` | string | "layered" | モード: "brain" または "layered" |
| `embed_dim` | int | 128 | 埋め込み次元 |
| `num_layers` | int | 3 | レイヤー数 |
| `dropout` | float | 0.1 | ドロップアウト率 |
| `max_seq_len` | int | 256 | 最大シーケンス長 |

#### Brain Mode 専用

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `num_neurons` | int | 100 | ニューロン数 |
| `connection_density` | float | 0.25 | 接続密度（0.0-1.0） |
| `lambda_entangle` | float | 0.35 | 量子もつれ強度 |

#### Layered Mode 専用

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `hidden_dim` | int | 256 | 隠れ層次元 |
| `num_heads` | int | 4 | アテンションヘッド数 |
| `lambda_entangle` | float | 0.5 | 量子もつれ強度 |

---

## 📁 ファイル構成

```
neuroq-runpod/
├── handler.py              # RunPod Serverless エントリポイント
├── neuroq_model.py         # モデル定義（Brain & Layered）
├── train_and_export.py     # 学習＆エクスポートスクリプト
├── requirements.txt        # 依存ライブラリ
├── Dockerfile              # Docker イメージ定義
├── neuroq_brain_model.pt   # Brain モデルチェックポイント
├── neuroq_layered_model.pt # Layered モデルチェックポイント
├── neuroq_tokenizer.json   # トークナイザー
└── neuroq_meta.json        # メタ情報
```

---

## 🛠️ 学習コマンド

### Brain Mode

```bash
# 基本
python train_and_export.py --mode brain --num_neurons 1000 --epochs 50

# フルオプション
python train_and_export.py \
  --mode brain \
  --num_neurons 2000 \
  --embed_dim 256 \
  --layers 4 \
  --connection_density 0.3 \
  --lambda_entangle 0.4 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.0005
```

### Layered Mode

```bash
# 基本
python train_and_export.py --mode layered --hidden_dim 512 --epochs 50

# フルオプション
python train_and_export.py \
  --mode layered \
  --hidden_dim 512 \
  --embed_dim 256 \
  --heads 8 \
  --layers 6 \
  --lambda_layered 0.5 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.0005
```

---

## 🐳 デプロイ手順

### 1. モデルを学習

```bash
source venv/bin/activate
python train_and_export.py --mode brain --num_neurons 500 --epochs 50
python train_and_export.py --mode layered --hidden_dim 256 --epochs 50
```

### 2. GitHub にプッシュ

```bash
git init
git add .
git commit -m "NeuroQ RunPod Serverless"
git remote add origin https://github.com/YOUR_USERNAME/neuroq-runpod.git
git push -u origin main
```

### 3. RunPod Serverless Endpoint を作成

1. [RunPod](https://runpod.io/) → **Serverless** → **New Endpoint**
2. **Docker Image** を選択
3. **GPU Type**: 24GB または 48GB
4. **環境変数** を設定（オプション）:
   - `NEUROQ_MODE`: デフォルトモード
   - `NEUROQ_NUM_NEURONS`: デフォルトニューロン数
   - `NEUROQ_HIDDEN_DIM`: デフォルト隠れ層次元

---

## 🔧 環境変数

Dockerfileまたは RunPod で設定可能:

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `NEUROQ_MODE` | layered | デフォルトモード |
| `NEUROQ_MODEL_PATH` | neuroq_model.pt | モデルファイルパス |
| `NEUROQ_TOKENIZER_PATH` | neuroq_tokenizer.json | トークナイザーパス |
| `NEUROQ_EMBED_DIM` | 128 | 埋め込み次元 |
| `NEUROQ_NUM_NEURONS` | 100 | ニューロン数（Brain） |
| `NEUROQ_HIDDEN_DIM` | 256 | 隠れ層次元（Layered） |
| `NEUROQ_NUM_HEADS` | 4 | アテンションヘッド数 |
| `NEUROQ_NUM_LAYERS` | 3 | レイヤー数 |
| `NEUROQ_CONNECTION_DENSITY` | 0.25 | 接続密度 |
| `NEUROQ_LAMBDA_BRAIN` | 0.35 | もつれ強度（Brain） |
| `NEUROQ_LAMBDA_LAYERED` | 0.5 | もつれ強度（Layered） |

---

## 📊 モード比較

| 項目 | Brain Mode | Layered Mode |
|-----|------------|--------------|
| アーキテクチャ | 脳型散在ネットワーク | Transformer + QBNN |
| ニューロン接続 | スパース（グラフ構造） | 層間接続 |
| 量子もつれ | 任意ニューロン間 | レイヤー間演算 |
| 時間ステップ | 信号伝播で複数ステップ | なし |
| 生成速度 | やや遅い | 高速 |
| 創造性 | 高い | 安定 |
| 推奨用途 | 創造的タスク | 一般的な対話 |

---

## 🔬 APQB理論

両モードともAPQB（調整可能擬似量子ビット）理論に基づいています：

```
θ (theta): 内部角度パラメータ
r = cos(2θ): 相関係数
T = |sin(2θ)|: 温度（ゆらぎ）

制約: r² + T² = 1
```

---

## 📜 ライセンス

MIT License
