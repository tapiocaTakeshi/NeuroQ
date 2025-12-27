# NeuroQ Handler - 使い方ガイド

## 重要：学習と推論の完全分離

この実装では、**学習（train）と推論（generate）を完全に分離**しています。

### アーキテクチャ

```
┌─────────────────┐
│  action: train  │ → 学習 → チェックポイント保存
└─────────────────┘

┌──────────────────┐
│ action: generate │ → チェックポイントロード → 推論（学習なし）
└──────────────────┘
```

---

## 1. 基本的な使い方

### Step 1: 学習（train）

**初回または更新時のみ実行**

```json
{
  "input": {
    "action": "train",
    "epochs": 25,
    "batch_size": 16,
    "lr": 0.002,
    "seq_length": 48
  }
}
```

**結果**：
- 学習データで25エポック学習
- `checkpoints/neuroq_checkpoint.pt` に保存

### Step 2: 推論（generate）

**学習後は何度でも実行可能**

```json
{
  "input": {
    "action": "generate",
    "prompt": "こんにちは",
    "max_length": 50,
    "temperature": 0.5
  }
}
```

**結果**：
- チェックポイントを自動ロード（初回のみ）
- 推論実行（学習は一切しない）
- 高速レスポンス

---

## 2. アクション一覧

### `action: "health"`
ヘルスチェック（即座に200を返す）

```json
{
  "input": {
    "action": "health"
  }
}
```

### `action: "status"`
ステータス確認

```json
{
  "input": {
    "action": "status"
  }
}
```

### `action: "train"`
学習実行 + チェックポイント保存

**パラメータ**：
- `texts` (optional): 学習データ（指定しない場合はデフォルト）
- `epochs` (default: 25): エポック数
- `batch_size` (default: 16): バッチサイズ
- `lr` (default: 0.002): 学習率
- `seq_length` (default: 48): シーケンス長
- `checkpoint_path` (optional): 保存先パス

### `action: "generate"`
推論実行（チェックポイント自動ロード）

**パラメータ**：
- `prompt` (required): 入力プロンプト
- `max_length` (default: 50): 最大生成長
- `temperature` (default: 0.5): 温度パラメータ
- `temp_min` / `temp_max` (optional): 温度範囲
- `session_id` (default: "default"): 会話セッションID

### `action: "clear_session"`
会話履歴をクリア

```json
{
  "input": {
    "action": "clear_session",
    "session_id": "default"
  }
}
```

---

## 3. RunPod serverless での運用パターン

### パターンA: イメージに同梱（推奨）

**1. ローカルで学習**
```bash
cd neuroq-runpod
python -c "import runpod; runpod.api_key='YOUR_KEY'; runpod.endpoint.Endpoint('ENDPOINT_ID').run({'action': 'train'})"
```

**2. チェックポイントをダウンロード**
```bash
# RunPodからダウンロード（またはローカル学習）
# checkpoints/neuroq_checkpoint.pt を取得
```

**3. Dockerイメージに同梱**
```dockerfile
# Dockerfile
COPY checkpoints/neuroq_checkpoint.pt /app/checkpoints/
COPY neuroq_tokenizer.model /app/
```

**4. デプロイ**
- 推論は `action: "generate"` のみ
- 学習不要で高速起動

### パターンB: RunPod Volume 使用

**1. 学習してVolumeに保存**
```json
{
  "input": {
    "action": "train",
    "checkpoint_path": "/workspace/checkpoints/neuroq_checkpoint.pt"
  }
}
```

**2. 推論時に自動ロード**
- handler.pyの `MODEL_CHECKPOINT_PATH` を `/workspace/checkpoints/neuroq_checkpoint.pt` に変更
- Volume mount で永続化

### パターンC: S3/GCS 使用

**1. 学習後にS3にアップロード**
```python
# 学習後
import boto3
s3 = boto3.client('s3')
s3.upload_file(
    'checkpoints/neuroq_checkpoint.pt',
    'my-bucket',
    'models/neuroq_checkpoint.pt'
)
```

**2. 起動時にダウンロード**
```python
# handler.py の起動時に追加
import boto3
s3 = boto3.client('s3')
s3.download_file(
    'my-bucket',
    'models/neuroq_checkpoint.pt',
    MODEL_CHECKPOINT_PATH
)
```

---

## 4. トラブルシューティング

### Q: 推論が遅い、または学習が走っている？

A: 以下を確認：
1. `action: "generate"` を使っているか？
2. チェックポイントが存在するか？
   - 存在しない場合は警告が出て未学習モデルが使われる
3. `initialize_model()` が呼ばれているか？
   - 初回のみ実行され、以降はキャッシュされる

### Q: チェックポイントが見つからない

A: 2つの方法：
1. **学習を実行**: `action: "train"` でチェックポイント作成
2. **既存をコピー**: ローカルで学習したものをコンテナに配置

### Q: 会話が続かない

A: `session_id` を指定：
```json
{
  "input": {
    "action": "generate",
    "prompt": "続きを教えて",
    "session_id": "user_123"
  }
}
```

---

## 5. ベストプラクティス

### 推論を高速化する

1. **事前学習したチェックポイントを使用**
   - Docker build時に同梱
   - 起動時のダウンロードをキャッシュ

2. **推論専用コンテナ**
   - `action: "generate"` のみサポート
   - `action: "train"` は無効化

3. **model.eval() + torch.no_grad()**
   - 実装済み（`generate_text()` 内）

### 学習を効率化する

1. **専用GPUで学習**
   - ローカルまたは専用インスタンス
   - serverlessではなくdedicated

2. **学習データの最適化**
   - `get_training_data()` をカスタマイズ
   - 高品質なデータセット使用

---

## 6. 設定ファイル

### handler.py の主要設定

```python
# チェックポイントパス
MODEL_CHECKPOINT_PATH = "checkpoints/neuroq_checkpoint.pt"

# トークナイザー
TOKENIZER_MODEL_PATH = "neuroq_tokenizer.model"

# モデル設定
VOCAB_SIZE = 8000
embed_dim = 128
num_heads = 4
num_layers = 3
num_neurons = 100
```

---

## まとめ

- **学習**: `action: "train"` → チェックポイント保存
- **推論**: `action: "generate"` → チェックポイントロード（学習なし）
- **運用**: 事前学習 → イメージ同梱 or Volume/S3 → 推論専用デプロイ

これで「API呼び出しのたびに学習」を完全に回避できます。
