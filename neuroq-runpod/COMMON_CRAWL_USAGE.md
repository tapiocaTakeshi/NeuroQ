# Common Crawl データ取得機能

## 概要

NeuroQuantumにCommon Crawlからの大規模Web学習データ取得機能が実装されました。

Common Crawlは、Webクロールデータを公開しているオープンなプロジェクトで、ペタバイト規模のWebコンテンツにアクセスできます。

## インストール

必要なライブラリをインストール：

```bash
pip install -r neuroq-runpod/requirements.txt
```

または個別にインストール：

```bash
pip install warcio beautifulsoup4 requests
```

## 使用方法

### 基本的な使い方

RunPod APIを使用する場合：

```python
{
  "input": {
    "action": "generate",
    "prompt": "量子コンピュータについて",
    "train_before_generate": true,
    "data_sources": ["common_crawl"],
    "common_crawl_config": {
      "max_records": 100,
      "query": "*.jp"
    }
  }
}
```

### パラメータ

#### `data_sources`

データソースのリスト：
- `"common_crawl"`: Common Crawlから取得
- `"huggingface"`: Hugging Face Datasetsから取得

複数指定可能：`["common_crawl", "huggingface"]`

#### `common_crawl_config`

Common Crawl固有の設定：

- `max_records` (int): 取得する最大レコード数（デフォルト: 50）
- `query` (str): 検索クエリ（デフォルト: "*.jp"）
  - `"*.jp"`: 日本のドメイン
  - `"example.com/*"`: 特定ドメイン
  - `"*/blog/*"`: 特定のパス
- `index_name` (str, optional): 使用するCommon Crawl Index名（省略時は最新）
- `max_records_cc` (int): Common Crawl専用の最大レコード数

### クエリの例

#### 日本語サイトから取得
```python
"common_crawl_config": {
  "query": "*.jp",
  "max_records": 100
}
```

#### 特定ドメインから取得
```python
"common_crawl_config": {
  "query": "wikipedia.org/*",
  "max_records": 50
}
```

#### 特定のIndex（時期）を指定
```python
"common_crawl_config": {
  "query": "*.jp",
  "index_name": "CC-MAIN-2024-10",
  "max_records": 100
}
```

## スタンドアロンでの使用

`common_crawl_fetcher.py`を単独で使用することもできます：

```python
from common_crawl_fetcher import fetch_common_crawl_data

# データ取得
texts = fetch_common_crawl_data(
    max_records=100,
    query="*.jp",
    min_text_length=100
)

print(f"取得テキスト数: {len(texts)}")
for text in texts[:3]:
    print(text[:200])
```

### コマンドラインからのテスト

```bash
cd neuroq-runpod
python common_crawl_fetcher.py
```

## 技術詳細

### Common Crawl Index Server API

Common Crawlは、CDX (Capture inDeX) Server APIを提供しており、以下のように動作します：

1. **Index検索**: CDX Server APIで特定のURLパターンにマッチするレコードを検索
2. **WARCレコード取得**: S3からWARCファイルの該当部分をダウンロード
3. **テキスト抽出**: HTMLからテキストを抽出し、日本語コンテンツをフィルタリング

### データフロー

```
1. CDX Index Server
   ↓ (URLパターンで検索)
2. CDXレコード（WARC位置情報）
   ↓ (S3から取得)
3. WARCファイル
   ↓ (HTMLパース)
4. テキストデータ
   ↓ (日本語フィルタリング)
5. 学習データ
```

### 日本語フィルタリング

`is_japanese_text()` 関数により、以下の条件でフィルタリング：

- ひらがな、カタカナ、漢字の割合が10%以上
- テキスト長が最小100文字以上

## 注意事項

### パフォーマンス

- Common Crawlからのデータ取得は、ネットワーク経由でS3からWARCファイルを取得するため、時間がかかる場合があります
- 大量のレコードを取得する場合は、`max_records`を調整してください

### レート制限

- Common Crawl APIには明示的なレート制限はありませんが、大量のリクエストは控えめに行ってください
- エラーが発生した場合は、`max_records`を減らすか、時間をおいて再試行してください

### データ品質

- Webクロールデータには、様々な品質のコンテンツが含まれます
- 必要に応じて、追加のフィルタリングやクリーニングを実装してください

## トラブルシューティング

### `requests`が見つからない

```bash
pip install requests
```

### `warcio`が見つからない

```bash
pip install warcio
```

### `beautifulsoup4`が見つからない

```bash
pip install beautifulsoup4
```

### データが取得できない

1. インターネット接続を確認
2. クエリパターンを変更してみる（例: `"*.jp"` → `"*.com"`）
3. `max_records`を減らしてみる
4. 最新のIndexが使用されているか確認

### エラーログの確認

詳細なエラー情報は、コンソール出力に表示されます：

```
⚠️ Common Crawl取得失敗: [エラーメッセージ]
```

## 今後の拡張予定

- [ ] マルチスレッドでの並列取得
- [ ] キャッシング機能
- [ ] より高度なテキストフィルタリング
- [ ] 特定トピックへの絞り込み
- [ ] ローカルストレージへの保存

## 参考リンク

- [Common Crawl公式サイト](https://commoncrawl.org/)
- [Common Crawl Index Server](https://index.commoncrawl.org/)
- [CDX Server API ドキュメント](https://github.com/webrecorder/pywb/wiki/CDX-Server-API)
- [warcio ライブラリ](https://github.com/webrecorder/warcio)

## ライセンス

Common Crawlのデータは、クリエイティブ・コモンズ・ライセンスの下で提供されています。使用する際は、適切なライセンス表示を行ってください。
