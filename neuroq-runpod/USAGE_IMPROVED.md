# 改善版学習→生成スクリプトの使用方法

## 問題点

現在の生成結果には以下の問題があります：

- 特殊トークン（`_NEWLINE`、`PARAGRAPH`、`SECTION`、`START`など）が生成される
- 意味のない文字列が生成される
- 同じテキストが繰り返される

## 改善版スクリプト

### `train_and_generate_improved.py`

改善されたパラメータを使用した学習→生成スクリプトです。

#### 使用方法

```bash
# 環境変数を設定
export RUNPOD_API_KEY="your_api_key"
export RUNPOD_ENDPOINT_ID="your_endpoint_id"

# 改善版スクリプトを実行
python neuroq-runpod/train_and_generate_improved.py
```

#### プリセット設定

スクリプト内で`preset`パラメータを変更できます：

1. **`"short"`** - 短時間改善版（約2-5分）
   - エポック数: 30
   - データレコード数: 300
   - 生成パラメータ: 特殊トークンを抑制

2. **`"standard"`** - 標準改善版（約5-10分）
   - エポック数: 50
   - データレコード数: 500
   - より大きなモデルサイズ

3. **`"high_quality"`** - 高品質版（約15-30分）
   - エポック数: 100
   - データレコード数: 1000
   - 最大モデルサイズ

#### カスタムパラメータ

スクリプト内の`main()`関数で以下を変更：

```python
preset = "short"  # "short", "standard", "high_quality"
prompt = "ChatGPTについて教えて"
```

または、`train_and_generate_improved()`関数を直接呼び出してカスタムパラメータを指定：

```python
result = train_and_generate_improved(
    prompt="あなたの質問",
    mode="layered",
    preset="short",
    epochs=40,  # プリセットを上書き
    max_records=400
)
```

## 主な改善点

### 学習パラメータ

| パラメータ | 改善前 | 改善後（短時間版） |
|----------|--------|------------------|
| データレコード数 | 100 | 300 |
| エポック数 | 20 | 30 |
| 学習率 | 0.001 | 0.0005 |

### 生成パラメータ

| パラメータ | 改善前 | 改善後 |
|----------|--------|--------|
| 温度 | 0.7 | 0.6（特殊トークンを抑制） |
| Top-K | 40 | 30（より保守的） |
| Top-P | 0.9 | 0.85（多様性を抑える） |
| 繰り返しペナルティ | 2.5 | 3.0（より強力） |

## JSONファイルから直接使用

改善版のJSONリクエスト例も利用できます：

```python
import json
import requests

# JSONファイルを読み込む
with open('neuroq-runpod/train_request_examples_improved.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 短時間改善版のリクエストを使用
payload = {
    "input": data["example_1_短時間改善版"]["input"]
}

# RunPodに送信
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    },
    json=payload
)
```

## 期待される改善

改善版パラメータを使用することで：

1. **特殊トークンの生成が減る**
   - `repetition_penalty: 3.0`で繰り返しを強く抑制
   - `temperature: 0.6`で一貫性を高める
   - `top_k: 30`で保守的な生成

2. **より意味のあるテキスト**
   - より多くのデータ（300レコード）で学習
   - より多くのエポック（30エポック）で学習

3. **プロンプトとの関連性向上**
   - より長いシーケンス長でコンテキストを保持

## トラブルシューティング

### まだ特殊トークンが生成される場合

1. **より多くのデータを使用**
   - `max_records: 300 → 500`
   - `preset: "short" → "standard"`

2. **より多くのエポックで学習**
   - `epochs: 30 → 50`
   - `preset: "short" → "standard"`

3. **より大きなモデルを使用**
   - `preset: "standard" → "high_quality"`

### 学習時間が長すぎる場合

1. **短時間版を使用**
   - `preset: "short"`

2. **データレコード数を減らす**
   - `max_records: 300 → 200`

3. **エポック数を減らす**
   - `epochs: 30 → 20`

## 詳細情報

- `IMPROVEMENT_ANALYSIS.md` - 詳細な分析と改善策
- `train_request_examples_improved.json` - 改善版JSONリクエスト例

