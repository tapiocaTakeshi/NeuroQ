# NeuroQ - RunPod Serverless

QBNN（量子ビットニューラルネットワーク）ベースの生成AI

## 🧠 2つのモード

### Brain Mode（脳型散在QBNN）
- **特徴**: 人間の脳の神経回路を模倣した散在的なニューロン配置
- **動的選択**: 入力感度と出力傾向に基づいてニューロンを動的に選択
- **量子もつれ**: ニューロン間の相関を量子もつれとして表現
- **用途**: 創造的なタスク、柔軟な応答生成

```bash
# 学習
python train_and_export.py --mode brain --neurons 1000 --epochs 50
```

### Layered Mode（層状QBNN-Transformer）
- **特徴**: Transformerアーキテクチャとの組み合わせ
- **自己注意**: 量子ビット重み付き注意機構
- **量子もつれ層**: レイヤー間の量子もつれ演算
- **用途**: 安定した応答生成、長文処理

```bash
# 学習
python train_and_export.py --mode layered --neurons 256 --heads 4 --layers 3 --epochs 50
```

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

## 🚀 デプロイ手順

### 1. モデルを学習

```bash
# 仮想環境を有効化
source venv/bin/activate

# Layeredモード
python train_and_export.py --mode layered --epochs 50 --neurons 256

# Brainモード
python train_and_export.py --mode brain --epochs 50 --neurons 1000
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
2. **Docker Image** を選択:
   - リポジトリからビルドするか、DockerHubにプッシュしたイメージを指定
3. **GPU Type**: 24GB または 48GB
4. **環境変数** を設定:
   - `NEUROQ_MODE`: `layered` または `brain`
   - `NEUROQ_MODEL_PATH`: モデルファイルパス

### 4. API で呼び出し

```bash
# Layered モードで生成
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "prompt": "こんにちは",
      "mode": "layered",
      "max_tokens": 64,
      "temperature": 0.7
    }
  }'

# Brain モードで生成
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "prompt": "量子とは何ですか",
      "mode": "brain",
      "max_tokens": 64,
      "temperature": 0.8
    }
  }'
```

## 📝 API パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `prompt` | string | (必須) | 入力プロンプト |
| `mode` | string | "layered" | モード: "brain" または "layered" |
| `max_tokens` | int | 128 | 最大生成トークン数 |
| `temperature` | float | 0.7 | 温度（0.0-1.0） |
| `top_k` | int | 40 | Top-K サンプリング |
| `top_p` | float | 0.9 | Top-P (Nucleus) サンプリング |

## 🔧 Dockerfile

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV NEUROQ_MODE="layered"
ENV NEUROQ_MODEL_PATH="neuroq_layered_model.pt"
ENV NEUROQ_TOKENIZER_PATH="neuroq_tokenizer.json"

CMD ["python", "handler.py"]
```

## 🧪 ローカルテスト

```python
from neuroq_model import NeuroQGenerator, create_neuroq_layered, create_neuroq_brain

# Layered モード
model, tokenizer = create_neuroq_layered(
    model_path="neuroq_layered_model.pt",
    tokenizer_path="neuroq_tokenizer.json"
)
generator = NeuroQGenerator(model, tokenizer, "cuda")
print(generator.generate("こんにちは"))

# Brain モード
model, tokenizer = create_neuroq_brain(
    model_path="neuroq_brain_model.pt",
    tokenizer_path="neuroq_tokenizer.json"
)
generator = NeuroQGenerator(model, tokenizer, "cuda")
print(generator.generate("量子とは"))
```

## 📊 モード比較

| 項目 | Brain Mode | Layered Mode |
|-----|------------|--------------|
| アーキテクチャ | 脳型散在ネットワーク | Transformer + QBNN |
| ニューロン選択 | 動的（感度ベース） | 固定 |
| 量子もつれ | ニューロン間相関 | レイヤー間演算 |
| 生成速度 | やや遅い | 高速 |
| 創造性 | 高い | 安定 |
| 推奨用途 | 創造的タスク | 一般的な対話 |

## 🔬 APQB理論

両モードともAPQB（調整可能擬似量子ビット）理論に基づいています：

```
θ (theta): 内部角度パラメータ
r = cos(2θ): 相関係数
T = |sin(2θ)|: 温度（ゆらぎ）

制約: r² + T² = 1
```

## 📜 ライセンス

MIT License
