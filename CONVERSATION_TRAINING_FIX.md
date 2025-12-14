# NeuroQ 会話学習の改善

## 🔍 問題の診断

会話形式での生成がうまくいかない原因は、主に以下の3点でした：

### 1. 学習データが会話フォーマットになっていない
- 従来: `質問: ～\n回答: ～` という単純な形式
- 問題点: USER/ASSISTANT のターン構造が学習されていない

### 2. 役割トークン・EOS/stopが弱い
- stopトークンが事後チェックのみ
- `<USER>` が生成された後に止まる仕組み

### 3. 推論時プロンプトが会話指示になっていない
- プロンプトがそのまま渡される
- 会話履歴の管理がない
- システムプロンプトがない

---

## ✅ 実装した修正

### 1. 学習データの会話フォーマット強化 (`prepare_training_data.py`)

**修正内容:**
- USER/ASSISTANT 形式のターン構造を導入
- 単一ターン対話（基本パターン）: 4,000件
- マルチターン対話（2-4ターン）: 大量追加
- 指示追従パターン
- 不明確な質問への対応パターン
- 会話の終了パターン

**フォーマット例:**
```
<USER>こんにちは<ASSISTANT>こんにちは！何かお手伝いできることはありますか？
<USER>量子コンピュータとは何ですか<ASSISTANT>量子コンピュータは量子力学の原理を利用したコンピュータです。
```

### 2. 会話履歴管理機能の追加 (`handler.py`)

**追加機能:**
- **セッション管理**: `session_id` で会話を管理
- **履歴保存**: 最大10ターン（20メッセージ）を保持
- **プロンプト構築**: 履歴を含む会話形式のプロンプトを自動生成
- **システムプロンプト**: 会話指示を定義

**新しいAPI:**
```python
# 生成時にsession_idを指定
{
    "action": "generate",
    "prompt": "こんにちは",
    "session_id": "user_123",  # 会話セッションID
    "max_length": 100
}

# セッションクリア
{
    "action": "clear_session",
    "session_id": "user_123"
}
```

### 3. プロンプトテンプレートの改善

**会話プロンプトの構築:**
```python
# 履歴を含むプロンプト例
<USER>前回の質問<ASSISTANT>前回の回答<USER>今回の質問<ASSISTANT>
```

**システムプロンプト:**
```
あなたは親切で正確なアシスタントです。
1. ユーザーの質問に短く正確に答える
2. わからないことは質問する
3. 聞かれたことだけに答える
4. 前の文脈を踏まえて返答する
```

---

## 📊 期待される改善

### Before（修正前）
- 説明文は生成できるが、会話にならない
- 質問に対して延々と説明を続ける
- 追加質問への文脈理解がない
- ターン構造が学習されていない

### After（修正後）
- ✅ USER/ASSISTANT のターン構造を理解
- ✅ 質問に対して適切な長さで返答
- ✅ 追加質問に文脈を踏まえて返答
- ✅ 「聞かれたことだけ答える」ができる
- ✅ 不明な質問に対して質問返しができる

---

## 🚀 使い方

### 1. 学習データの準備（再生成推奨）

```bash
# 新しい会話フォーマットでデータを準備
python prepare_training_data.py --output data/training_data.txt --min-chars 1000000
```

### 2. モデルの再学習

会話フォーマットのデータで再学習することを推奨します：

```bash
# 事前学習を実行
python neuroq_pretrained.py
```

### 3. 推論時の使い方

**RunPod API:**
```python
import requests

# セッション付き会話
response = requests.post("YOUR_RUNPOD_ENDPOINT", json={
    "input": {
        "action": "generate",
        "prompt": "こんにちは",
        "session_id": "conversation_1",
        "max_length": 100
    }
})

# 追加質問（同じsession_id）
response = requests.post("YOUR_RUNPOD_ENDPOINT", json={
    "input": {
        "action": "generate",
        "prompt": "あなたは誰ですか",
        "session_id": "conversation_1",  # 同じセッション
        "max_length": 100
    }
})

# セッションクリア
response = requests.post("YOUR_RUNPOD_ENDPOINT", json={
    "input": {
        "action": "clear_session",
        "session_id": "conversation_1"
    }
})
```

---

## 📝 注意事項

### 学習データの比率

会話能力を向上させるためには、学習データの **60%以上を会話データ** にすることを推奨します。

現在の `prepare_training_data.py` での比率：
- 会話データ: 約65%（大幅増加）
- 説明文データ: 約35%

### 継続学習の推奨

既に説明文で事前学習済みのモデルがある場合は、会話データで **継続学習（fine-tuning）** することで、
説明能力を保ちながら会話能力を向上できます。

### RLHF/RLAIF（将来の改善）

さらに会話品質を向上させるには、強化学習（RLHF/RLAIF）の導入を検討してください。
簡易版として DPO（Direct Preference Optimization）が実装しやすいです。

---

## 🔧 今後の改善候補

1. **報酬モデルの追加**: DPO/PPOによる会話品質の強化
2. **システムプロンプトの学習**: より複雑な指示追従
3. **長期記憶**: セッションを永続化して長期対話をサポート
4. **マルチモーダル対応**: 画像・音声との統合

---

## 📚 参考資料

- [InstructGPT論文](https://arxiv.org/abs/2203.02155): RLHF の基礎
- [DPO論文](https://arxiv.org/abs/2305.18290): シンプルな強化学習手法
- [ChatGPT技術解説](https://openai.com/blog/chatgpt): 対話モデルの設計思想

---

**修正日**: 2025-12-14
**バージョン**: v1.0.0
