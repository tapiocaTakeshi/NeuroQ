# 環境変数の設定方法

RunPod Serverless Handlerを使用するには、以下の環境変数を設定する必要があります。

## 必要な環境変数

1. **RUNPOD_API_KEY**: RunPod APIキー
2. **RUNPOD_ENDPOINT_ID**: RunPodエンドポイントID

## 設定方法

### 方法1: ターミナルで直接設定（一時的）

```bash
export RUNPOD_API_KEY='your_api_key_here'
export RUNPOD_ENDPOINT_ID='your_endpoint_id_here'
```

この方法は、現在のターミナルセッションでのみ有効です。

### 方法2: .envファイルを使用（推奨）

`neuroq-runpod`ディレクトリに`.env`ファイルを作成：

```bash
cd neuroq-runpod
cat > .env << EOF
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
EOF
```

`.env`ファイルを使用する場合は、スクリプトを修正して`python-dotenv`を使用する必要があります：

```python
from dotenv import load_dotenv
load_dotenv()
```

### 方法3: シェル設定ファイルに追加（永続的）

`~/.zshrc`（zshを使用している場合）または`~/.bashrc`（bashを使用している場合）に追加：

```bash
# ~/.zshrc または ~/.bashrc に追加
export RUNPOD_API_KEY='your_api_key_here'
export RUNPOD_ENDPOINT_ID='your_endpoint_id_here'
```

設定後、新しいターミナルを開くか、以下を実行：

```bash
source ~/.zshrc  # zshの場合
# または
source ~/.bashrc  # bashの場合
```

## RunPod APIキーとエンドポイントIDの取得方法

### APIキーの取得

1. [RunPod Dashboard](https://www.runpod.io/)にログイン
2. 右上のプロフィールアイコンをクリック
3. 「Settings」→「API Keys」に移動
4. 「Create API Key」をクリックして新しいキーを作成
5. キーをコピー（表示は一度だけ）

### エンドポイントIDの取得

1. RunPodダッシュボードにログイン
2. 「Serverless」→「Endpoints」に移動
3. エンドポイントを選択
4. エンドポイントIDはURLまたはエンドポイント詳細ページに表示されます
   - 例: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run`
   - `YOUR_ENDPOINT_ID`の部分がエンドポイントIDです

## 環境変数の確認

環境変数が正しく設定されているか確認：

```bash
echo $RUNPOD_API_KEY
echo $RUNPOD_ENDPOINT_ID
```

または、Pythonで確認：

```python
import os
print("API Key:", os.getenv("RUNPOD_API_KEY", "未設定"))
print("Endpoint ID:", os.getenv("RUNPOD_ENDPOINT_ID", "未設定"))
```

## セキュリティに関する注意

- APIキーは機密情報です。Gitにコミットしないでください
- `.env`ファイルは`.gitignore`に追加してください
- 他の人と共有しないでください

## トラブルシューティング

### 環境変数が読み込まれない

1. ターミナルを再起動する
2. `source ~/.zshrc`（または`source ~/.bashrc`）を実行
3. 環境変数が正しく設定されているか確認：`echo $RUNPOD_API_KEY`

### スクリプトが環境変数を認識しない

- 環境変数を設定したターミナルでスクリプトを実行しているか確認
- スクリプトの実行前に環境変数が設定されているか確認

