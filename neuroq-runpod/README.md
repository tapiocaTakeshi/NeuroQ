# NeuroQ RunPod Serverless

RunPod Serverless環境向けのNeuroQuantum AIサーバー実装です。

## 📋 前提条件

- Docker
- Git LFS
- NVIDIA GPU（ローカル実行時）

## 🚀 クイックスタート

### 1. Git LFSのセットアップ

```bash
# Git LFSをインストール（まだの場合）
sudo apt-get install git-lfs  # Ubuntu/Debian
# brew install git-lfs         # macOS

# Git LFSを初期化
git lfs install

# リポジトリのルートディレクトリでLFSファイルを取得
cd ..
git lfs pull
cd neuroq-runpod
```

### 2. Dockerイメージのビルド

**推奨方法**: ビルドヘルパースクリプトを使用

```bash
./build.sh
```

このスクリプトは以下を自動的に行います：
- Git LFSがインストールされているか確認
- モデルファイルのサイズをチェック
- 必要に応じてLFSファイルをプル
- Dockerイメージをビルド

**手動ビルド方法**:

方法1: ローカルのファイルを使用（`git lfs pull`実行済みの場合）
```bash
# リポジトリのルートディレクトリから実行
cd ..
docker build -f neuroq-runpod/Dockerfile -t neuroq:latest .
```

方法2: リポジトリURLを指定（Git LFSファイルを自動取得）**推奨**
```bash
# リポジトリのルートディレクトリから実行
cd ..
docker build \
  -f neuroq-runpod/Dockerfile \
  --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \
  --build-arg GIT_BRANCH=main \
  -t neuroq:latest .
```

### 3. コンテナの実行

```bash
docker run --gpus all -p 8000:8000 neuroq:latest
```

## 🔧 トラブルシューティング

### ❌ エラー: "File too small (133 bytes)" または "This is a Git LFS pointer file"

これは、Git LFSファイルが正しくダウンロードされていないことを示しています。

**推奨解決方法: リポジトリURLを使用したビルド（最も確実）**

ローカルでのLFSプルが失敗する場合、または環境にGit LFSがインストールされていない場合、Dockerビルド時にリポジトリから直接LFSファイルを取得できます：

```bash
# リポジトリのルートディレクトリから実行
cd ..
docker build \
  -f neuroq-runpod/Dockerfile \
  --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \
  --build-arg GIT_BRANCH=main \
  -t neuroq:latest .
```

この方法では：
- Dockerビルド中にリポジトリをクローン
- Docker内でGit LFSをインストール
- `git lfs pull`を自動実行
- 正しいモデルファイルを取得

**代替方法: ローカルでLFSファイルをプル**

1. Git LFSをインストールして初期化：
   ```bash
   git lfs install
   ```

2. LFSファイルをプル：
   ```bash
   cd ..  # リポジトリのルートへ
   git lfs pull
   ```

3. ファイルサイズを確認：
   ```bash
   ls -lh neuroq_pretrained.pt
   # 約58MBであれば正常
   ```

4. 再度ビルド：
   ```bash
   cd neuroq-runpod
   ./build.sh
   ```

## 📝 ファイル構成

```
neuroq-runpod/
├── Dockerfile              # Dockerイメージ定義
├── build.sh               # ビルドヘルパースクリプト
├── handler.py             # RunPod Serverless Handler
├── requirements.txt       # Python依存関係
├── neuroquantum_*.py      # NeuroQuantumコアファイル
├── qbnn_*.py             # QBNN実装
├── neuroq_pretrained.py  # 事前学習済みモデルモジュール
├── neuroq_pretrained.pt  # 事前学習済みモデル（Git LFS）
└── neuroq_tokenizer.*    # トークナイザーファイル
```

## 🎯 ビルド引数

| 引数 | デフォルト | 説明 |
|-----|----------|------|
| `GIT_REPO_URL` | （空） | Gitリポジトリurl（指定時はクローン＆LFS pull） |
| `GIT_BRANCH` | `main` | クローンするブランチ |

## 📊 環境変数

| 変数 | デフォルト | 説明 |
|-----|----------|------|
| `NEUROQ_MODE` | `layered` | 動作モード（`layered` or `brain`） |
| `NEUROQ_VOCAB_SIZE` | `8000` | 語彙サイズ |

## 🚢 RunPod Serverlessへのデプロイ

1. イメージをビルド：
   ```bash
   ./build.sh
   ```

2. イメージをDocker Hubにプッシュ：
   ```bash
   docker tag neuroq:latest yourusername/neuroq:latest
   docker push yourusername/neuroq:latest
   ```

3. RunPodコンソールで新しいテンプレートを作成し、イメージを指定

## 📖 参考資料

- [親ディレクトリのREADME](../README.md) - APQB理論の詳細
- [RunPod公式ドキュメント](https://docs.runpod.io/)
