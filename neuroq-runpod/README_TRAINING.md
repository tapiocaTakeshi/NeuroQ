# NeuroQ OpenAssistant/oasst1 トレーニングガイド

## 概要

このガイドでは、OpenAssistant/oasst1データセットを使ってNeuroQモデルをトレーニングする方法を説明します。

## データセット構成

`pretrain_openai.py` は以下のデータセットを統合してトレーニングします：

1. **OpenAssistant/oasst1** (最優先 - 3倍に増やす)
   - 高品質な人間のフィードバックで評価された会話データセット
   - prompter（ユーザー）→ assistant のペアを抽出
   - 日本語、英語、その他の言語をサポート
   - 長すぎるメッセージはフィルタリング（user>500文字、assistant>1000文字）

2. **openai/mrcr** - Multi-turn Reasoning Chain Retrieval
   - 推論チェーンデータ

3. **openai/openai_humaneval** - HumanEval
   - コード生成評価データセット

4. **openai/MMMLU** - Multilingual MMLU
   - 多言語知識評価データセット（日本語優先）

5. **日本語指示データ** (500倍に増やす - 最重要)
   - ChatGPT、量子コンピュータ、AI、プログラミングに関する質問と回答
   - 挨拶パターン（多様性）
   - 一般的な質問

6. **一般知識データ** (10倍に増やす)
   - 科学、技術、AI、機械学習に関する知識

## トレーニング方法

### 方法1: ローカルで実行

```bash
cd neuroq-runpod
python3 pretrain_openai.py
```

**推奨環境:**
- GPU付きマシン（CUDA対応）
- または Apple Silicon Mac（MPS対応）
- 必要なパッケージ: `torch`, `datasets`, `sentencepiece`

**必要なパッケージをインストール:**
```bash
pip install -r requirements.txt
```

### 方法2: RunPod APIで実行

#### 1. トレーニング開始

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @request_train_oasst1.json
```

または、ファイルを使わずに直接実行：

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "action": "pretrain_openai"
    }
  }'
```

#### 2. トレーニングステータスを確認

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @request_train_status.json
```

または：

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "action": "pretrain_status"
    }
  }'
```

## リクエストJSONファイル

### request_train_oasst1.json
OpenAssistant/oasst1データセットを含む事前学習を開始します。

```json
{
  "input": {
    "action": "pretrain_openai"
  }
}
```

### request_train_status.json
トレーニングのステータスを確認します。

```json
{
  "input": {
    "action": "pretrain_status"
  }
}
```

## トレーニング後

トレーニングが完了すると、以下のファイルが生成されます：

- `neuroq_pretrained.pt` - 学習済みモデル
- `training_openai.log` - トレーニングログ

## モデルの使用

学習済みモデルは、RunPodハンドラーが自動的にロードして使用します。

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "action": "generate",
      "prompt": "OpenAssistant/oasst1データセットについて教えて",
      "max_length": 200,
      "temperature": 0.7
    }
  }'
```

## トレーニング設定

`pretrain_openai.py` のトレーニング設定：

```python
model = NeuroQuantumAI(
    embed_dim=256,
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
    max_seq_len=256,
    dropout=0.1,
    lambda_entangle=0.5
)

model.train(
    all_texts,
    epochs=25,
    seq_len=128,
    batch_size=16,
    lr=0.0005
)
```

## トラブルシューティング

### データセットのロードに失敗する

```bash
pip install --upgrade datasets
```

### GPU メモリ不足

`pretrain_openai.py` の設定を調整：
- `batch_size` を小さくする（16 → 8）
- `seq_len` を短くする（128 → 64）
- `embed_dim` を小さくする（256 → 128）

### トレーニングが遅い

- GPU環境で実行することを推奨
- CPUの場合は `epochs` を減らす（25 → 10）

## 参考情報

- OpenAssistant/oasst1: https://huggingface.co/datasets/OpenAssistant/oasst1
- NeuroQ リポジトリ: https://github.com/tapiocaTakeshi/NeuroQ
