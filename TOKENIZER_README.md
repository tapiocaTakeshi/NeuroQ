# NeuroQuantum SentencePiece Tokenizer

## 概要

NeuroQuantum では、**SentencePiece** を使用した8,000語彙のトークナイザーを採用しています。これにより、以前の86語彙から大幅に改善され、より自然で正確なテキスト生成が可能になりました。

## 🔑 重要な改善点

### Before (vocab_size = 86)
- 文字レベルの分割により、単語の意味を正しく捉えられない
- 「ChatGPT」などの固有名詞が正しく表現できない
- 生成テキストが無限ループに陥りやすい
- 文字の切れ方が不自然（例: 「ニューラル」→「ニュータ」）

### After (vocab_size = 8,000)
- サブワード分割により、単語の意味を保持
- 固有名詞や専門用語を正しくトークナイズ
- より多様で自然なテキスト生成
- 日本語・英語の混在テキストにも対応

## 📦 トークナイザーファイル

学習済みのトークナイザーモデル：

- **neuroq_tokenizer.model** - SentencePieceモデル（メインファイル）
- **neuroq_tokenizer.vocab** - 語彙リスト（参考用）

## 🚀 使用方法

### 1. トークナイザーの準備

トークナイザーはすでに学習済みで、プロジェクトに含まれています。

```bash
ls neuroq_tokenizer.model  # モデルファイルを確認
```

### 2. 自動的に読み込まれる

`NeuroQuantumAI` クラスは、学習時に自動的にトークナイザーモデルを検索して読み込みます。

検索順序：
1. カレントディレクトリ: `neuroq_tokenizer.model`
2. 親ディレクトリ: `../neuroq_tokenizer.model`
3. スクリプトと同じディレクトリ
4. その他のパス

### 3. モデルの学習・生成

通常通り学習・生成を実行するだけで、自動的に8,000語彙のトークナイザーが使用されます：

```python
from neuroquantum_layered import NeuroQuantumAI

# モデルを初期化
model = NeuroQuantumAI(
    embed_dim=128,
    hidden_dim=256,
    num_heads=4,
    num_layers=3,
)

# 学習データ
texts = [
    "量子コンピュータについて教えて",
    "ChatGPTは大規模言語モデルです",
    # ... more texts ...
]

# 学習（トークナイザーが自動的に読み込まれる）
model.train(texts, epochs=20)

# テキスト生成
response = model.generate("量子コンピュータについて教えて", max_length=100)
print(response)
```

## 🔧 トークナイザーの再学習

新しいデータで トークナイザーを再学習したい場合：

```bash
python train_sentencepiece_tokenizer.py
```

これにより：
- Hugging Face から日本語Wikipedia、対話データを取得
- vocab_size=8,000 で SentencePiece モデルを学習
- `neuroq_tokenizer.model` と `neuroq_tokenizer.vocab` を生成

### カスタマイズ

`train_sentencepiece_tokenizer.py` の以下のパラメータを変更できます：

```python
train_sentencepiece_tokenizer(
    texts=texts,
    model_prefix="neuroq_tokenizer",
    vocab_size=8000,  # 語彙サイズ（8,000〜32,000推奨）
    character_coverage=0.9995,  # 文字カバレッジ（日本語: 0.9995〜0.99995）
)
```

## 📊 トークナイザーのテスト

トークナイザーの動作を確認：

```python
import sentencepiece as spm

# モデル読み込み
sp = spm.SentencePieceProcessor()
sp.load("neuroq_tokenizer.model")

# テキストをトークナイズ
text = "ChatGPTについて教えてください"
tokens = sp.encode(text, out_type=str)
print(f"トークン: {tokens}")
# 出力例: ['▁Ch', 'at', 'GP', 'T', 'について', '教', 'えて', 'く', 'だ', 'さ', 'い']

ids = sp.encode(text, out_type=int)
print(f"ID: {ids}")
# 出力例: [4234, 156, 3421, 89, 142, 567, 234, 98, 45, 67, 23]

# デコード
decoded = sp.decode(ids)
print(f"デコード: {decoded}")
# 出力: ChatGPTについて教えてください
```

## 🐳 RunPod / Docker デプロイ

RunPod や Docker 環境でデプロイする場合、トークナイザーモデルが自動的にコピーされます。

`Dockerfile` で以下が設定されています：

```dockerfile
# SentencePiece トークナイザーモデルをコピー
COPY neuroq_tokenizer.model /app/
COPY neuroq_tokenizer.vocab /app/

# 環境変数
ENV NEUROQ_TOKENIZER_PATH="neuroq_tokenizer.model"
```

## 📝 トークナイズの例

### 日本語テキスト

```
入力: 量子コンピュータについて教えて
トークン: ['▁', '量子', 'コンピュータ', 'について', '教', 'えて']
ID数: 6
```

### 英語テキスト

```
入力: ChatGPTとは何ですか？
トークン: ['▁Ch', 'at', 'GP', 'T', 'とは', '何', 'です', 'か', '?']
ID数: 9
```

### 混在テキスト

```
入力: ニューロQの特徴を説明してください
トークン: ['▁', 'ニュー', 'ロ', 'Q', 'の', '特徴', 'を', '説明', 'して', 'く', 'だ', 'さ', 'い']
ID数: 13
```

## 🛠️ 依存関係

必要なライブラリ（`requirements.txt` に含まれています）：

```
sentencepiece>=0.1.99
datasets>=2.0.0  # トークナイザー学習時のデータ取得用
```

インストール：

```bash
pip install sentencepiece datasets
```

## 📚 参考資料

- [SentencePiece 公式ドキュメント](https://github.com/google/sentencepiece)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

## ⚠️ トラブルシューティング

### トークナイザーが見つからない

**エラー**: `NeuroQuantum: 新規に語彙を構築します...`

**解決策**:
```bash
# トークナイザーが存在するか確認
ls neuroq_tokenizer.model

# 存在しない場合は学習
python train_sentencepiece_tokenizer.py
```

### vocab_size が 86 のまま

**原因**: 古いフォールバックトークナイザーが使用されている

**解決策**:
1. `sentencepiece` がインストールされているか確認
2. `neuroq_tokenizer.model` が正しい場所にあるか確認
3. モデルを再初期化

### 学習データが少ない

**エラー**: `Vocabulary size too high (8000). Please set it to a value <= XXXX.`

**解決策**:
```bash
# datasets ライブラリをインストール
pip install datasets

# 再度トークナイザーを学習
python train_sentencepiece_tokenizer.py
```

## 🎯 次のステップ

トークナイザーの準備ができたら：

1. **モデルの学習**: より多くの学習データで再学習
2. **生成パラメータの調整**: `repetition_penalty`, `top_k`, `top_p` を調整
3. **対話データの追加**: ChatGPT 関連の Q&A データを学習データに追加

これにより、さらに高品質なテキスト生成が可能になります！
