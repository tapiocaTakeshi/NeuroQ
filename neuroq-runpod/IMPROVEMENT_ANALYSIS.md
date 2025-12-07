# 生成テキストの品質改善分析

## 現在の問題

生成されたテキストに以下の問題があります：

1. **特殊トークンが生成される**
   - `PARAGRAPH_ARTICLE`
   - `SECTION`
   - `START`
   - `NEWLINE`
   - など

2. **意味のない文字列**
   - 日本語の文法が崩れている
   - プロンプトとの関連性が低い

3. **語彙の問題**
   - 学習データに特殊トークンが含まれている
   - トークナイザーがこれらのトークンを学習してしまっている

## 原因分析

### 1. 学習データの質
- 学習データ（HuggingFaceのデータセット）に特殊トークンが含まれている
- データの前処理が不十分

### 2. 学習パラメータ
- エポック数が少ない（20エポック）
- データレコード数が少ない（100レコード）
- 学習率やバッチサイズの調整が必要

### 3. 生成パラメータ
- `repetition_penalty`は適切（2.5）
- `temperature`は適切（0.7）
- しかし、特殊トークンの生成を抑制する仕組みがない

## 改善策

### 1. 学習データの改善

#### データの前処理
```python
def preprocess_text(text):
    # 特殊トークンを除去
    text = text.replace("PARAGRAPH_ARTICLE", "")
    text = text.replace("SECTION", "")
    text = text.replace("START", "")
    text = text.replace("NEWLINE", "\n")
    # 連続する空白を除去
    text = " ".join(text.split())
    return text
```

#### より高品質なデータソース
- 日本語Wikipedia（前処理済み）
- 日本語コーパス
- 高品質な対話データ

### 2. 学習パラメータの調整

#### 推奨設定

**短時間学習（テスト用）**:
```json
{
  "epochs": 30,
  "batch_size": 16,
  "learning_rate": 0.0005,
  "max_records": 200,
  "seq_length": 64
}
```

**標準学習**:
```json
{
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "max_records": 500,
  "seq_length": 128
}
```

**長時間学習（高品質）**:
```json
{
  "epochs": 100,
  "batch_size": 64,
  "learning_rate": 0.00005,
  "max_records": 1000,
  "seq_length": 256
}
```

### 3. 生成パラメータの調整

#### 特殊トークンを抑制する設定
```json
{
  "max_length": 80,
  "temperature": 0.6,
  "top_k": 30,
  "top_p": 0.85,
  "repetition_penalty": 3.0
}
```

### 4. モデルパラメータの調整

#### より大きなモデル
```json
{
  "embed_dim": 256,
  "hidden_dim": 512,
  "num_heads": 8,
  "num_layers": 4,
  "max_vocab": 16000
}
```

## 推奨アクション

### 短期的な改善（すぐに実行可能）

1. **学習データを増やす**
   - `max_records`: 100 → 300-500

2. **エポック数を増やす**
   - `epochs`: 20 → 30-50

3. **生成パラメータを調整**
   - `repetition_penalty`: 2.5 → 3.0
   - `temperature`: 0.7 → 0.6

### 中期的な改善（コード変更が必要）

1. **データ前処理の追加**
   - 特殊トークンの除去
   - テキストの正規化

2. **トークナイザーの改善**
   - 特殊トークンのフィルタリング
   - より適切な語彙サイズ

3. **モデルアーキテクチャの調整**
   - より大きなモデル
   - より適切なハイパーパラメータ

### 長期的な改善（大幅な変更が必要）

1. **高品質なデータセットの構築**
   - 手動でクレンジングされたデータ
   - ドメイン固有のデータ

2. **ファインチューニング**
   - 事前学習済みモデルの使用
   - タスク固有のファインチューニング

## 改善されたJSONリクエスト例

詳細は `train_request_examples_improved.json` を参照してください。

