# Docker起動時のモデル自動ダウンロード

## 概要

Docker起動時に事前学習済みモデル (`neuroq_pretrained.pt`) を自動的にダウンロードする機能を実装しました。

## 問題の背景

従来の問題：
- `neuroq_pretrained.pt` がGit LFSポインタファイル（133バイト）のままで、実際のモデル（58MB）がダウンロードされていない
- ビルド時にLFSファイルが取得できない場合、コンテナ起動に失敗する
- 簡易学習モードにフォールバックすると、生成テキストの品質が著しく低下

## 解決策

### 1. 起動時自動ダウンロード

`entrypoint.sh` スクリプトを追加し、Docker起動時に以下を実行：

1. モデルファイルの存在確認
2. ファイルサイズの検証（LFSポインタファイル検出）
3. 必要に応じてGit LFS経由でダウンロード
4. ダウンロード成功後、メインハンドラーを起動

### 2. Dockerfileの修正

- LFSファイルがビルド時に取得できなくてもエラーにしない
- `ENTRYPOINT` を使用して `entrypoint.sh` を実行
- 環境変数でダウンロード設定をカスタマイズ可能

## 使用方法

### 基本的な使用

```bash
# イメージをビルド
docker build -f neuroq-runpod/Dockerfile -t neuroq:latest .

# コンテナを起動（自動的にモデルをダウンロード）
docker run --gpus all -p 8000:8000 neuroq:latest
```

### 環境変数のカスタマイズ

```bash
docker run --gpus all -p 8000:8000 \
  -e GIT_REPO_URL="https://github.com/tapiocaTakeshi/NeuroQ.git" \
  -e GIT_BRANCH="main" \
  -e MODEL_FILE="neuroq_pretrained.pt" \
  neuroq:latest
```

### ビルド時にモデルを含める（高速起動）

起動時のダウンロード時間を短縮したい場合：

```bash
# ローカルでLFSファイルを取得
git lfs install
git lfs pull

# モデルファイルを含めてビルド
docker build -f neuroq-runpod/Dockerfile -t neuroq:latest .
```

または、ビルド引数でリポジトリURLを指定：

```bash
docker build -f neuroq-runpod/Dockerfile \
  --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \
  --build-arg GIT_BRANCH=main \
  -t neuroq:latest .
```

## 動作フロー

```
Docker起動
    ↓
entrypoint.sh 実行
    ↓
モデルファイル確認
    ↓
┌─────────────────────┐
│ファイルが存在する？│
└─────────────────────┘
    │              │
    YES            NO
    ↓              ↓
サイズ確認     ダウンロード
    ↓              ↓
┌──────────────┐   Git LFS pull
│10KB以上？    │       ↓
└──────────────┘   コピー
    │      │
    YES    NO (LFSポインタ)
    ↓      ↓
   OK   ダウンロード
    ↓      ↓
handler.py 起動
```

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|--------------|------|
| `MODEL_FILE` | `neuroq_pretrained.pt` | モデルファイル名 |
| `GIT_REPO_URL` | `https://github.com/tapiocaTakeshi/NeuroQ.git` | GitリポジトリURL |
| `GIT_BRANCH` | `main` | ブランチ名 |

## トラブルシューティング

### モデルのダウンロードに失敗する

```bash
# ログを確認
docker logs <container_id>

# Git LFSが正しくインストールされているか確認
docker exec <container_id> git lfs version
```

### ネットワークエラー

- プロキシ設定が必要な場合は、Dockerビルド時に指定：
  ```bash
  docker build --build-arg http_proxy=http://proxy:port \
               --build-arg https_proxy=http://proxy:port \
               -f neuroq-runpod/Dockerfile -t neuroq:latest .
  ```

### ダウンロードをスキップしたい

事前にモデルをマウント：

```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/neuroq_pretrained.pt:/app/neuroq_pretrained.pt:ro \
  neuroq:latest
```

## メリット

1. **ビルドの柔軟性**: LFSファイルがなくてもビルド成功
2. **イメージサイズの削減**: モデルファイル（58MB）をイメージに含めない選択が可能
3. **自動復旧**: LFSポインタファイルを検出して自動ダウンロード
4. **環境変数でカスタマイズ**: 異なるリポジトリやブランチを指定可能

## 関連ファイル

- `neuroq-runpod/entrypoint.sh` - 起動スクリプト
- `neuroq-runpod/Dockerfile` - Docker設定
- `neuroq-runpod/handler.py` - メインハンドラー
- `download_model.py` - モデルダウンロードユーティリティ
