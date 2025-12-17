#!/bin/bash
# NeuroQ Docker Entrypoint Script
# ================================
# Docker起動時に事前学習済みモデルを自動ダウンロード

set -e

echo "========================================"
echo "🚀 NeuroQ コンテナ起動中..."
echo "========================================"

# 環境変数（デフォルト値）
MODEL_FILE="${MODEL_FILE:-neuroq_pretrained.pt}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/tapiocaTakeshi/NeuroQ.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
MIN_MODEL_SIZE=10000  # 10KB以上が正常なモデルファイル

# モデルファイルの検証とダウンロード
echo ""
echo "📦 モデルファイルを確認中: $MODEL_FILE"
echo ""

# ファイルが存在するか確認
if [ ! -f "$MODEL_FILE" ]; then
    echo "⚠️  モデルファイルが存在しません"
    NEED_DOWNLOAD=true
else
    # ファイルサイズを確認
    FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || echo "0")
    echo "   ファイルサイズ: $FILE_SIZE bytes"

    if [ "$FILE_SIZE" -lt "$MIN_MODEL_SIZE" ]; then
        echo "⚠️  ファイルサイズが小さすぎます（Git LFSポインタの可能性）"

        # LFSポインタファイルか確認
        if head -n 1 "$MODEL_FILE" 2>/dev/null | grep -q "version https://git-lfs.github.com"; then
            echo "   → Git LFSポインタファイルを検出"
        fi

        NEED_DOWNLOAD=true
    else
        echo "✅ モデルファイルは正常です（$FILE_SIZE bytes）"
        NEED_DOWNLOAD=false
    fi
fi

# ダウンロードが必要な場合
if [ "$NEED_DOWNLOAD" = true ]; then
    echo ""
    echo "🔄 モデルファイルをダウンロード中..."
    echo "   リポジトリ: $GIT_REPO_URL"
    echo "   ブランチ: $GIT_BRANCH"
    echo ""

    # 一時ディレクトリを作成
    TMP_DIR=$(mktemp -d)
    trap "rm -rf $TMP_DIR" EXIT

    echo "📥 Git LFSでリポジトリをクローン中..."

    # Git LFSが利用可能か確認
    if ! command -v git-lfs &> /dev/null; then
        echo "❌ Git LFSがインストールされていません"
        echo "   apt-get install git-lfs を実行してください"
        exit 1
    fi

    # リポジトリをクローン（shallow clone + LFS）
    if git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO_URL" "$TMP_DIR/neuroq-repo" 2>&1; then
        cd "$TMP_DIR/neuroq-repo"

        echo "📦 Git LFS pullを実行中..."
        if git lfs pull 2>&1; then
            # モデルファイルを確認
            if [ -f "$MODEL_FILE" ]; then
                DOWNLOADED_SIZE=$(stat -c%s "$MODEL_FILE")

                if [ "$DOWNLOADED_SIZE" -ge "$MIN_MODEL_SIZE" ]; then
                    echo "✅ モデルファイルのダウンロード成功: $DOWNLOADED_SIZE bytes"

                    # 元のディレクトリにコピー
                    cp "$MODEL_FILE" "/app/$MODEL_FILE"

                    echo "✅ モデルファイルを /app/ にコピーしました"
                    ls -lh "/app/$MODEL_FILE"
                else
                    echo "❌ ダウンロードしたファイルが小さすぎます: $DOWNLOADED_SIZE bytes"
                    exit 1
                fi
            else
                echo "❌ モデルファイルが見つかりません: $MODEL_FILE"
                exit 1
            fi
        else
            echo "❌ Git LFS pullに失敗しました"
            exit 1
        fi
    else
        echo "❌ リポジトリのクローンに失敗しました"
        exit 1
    fi

    # 元のディレクトリに戻る
    cd /app
fi

echo ""
echo "========================================"
echo "✅ モデルファイルの準備完了"
echo "========================================"
echo ""

# 最終確認
if [ -f "/app/$MODEL_FILE" ]; then
    FINAL_SIZE=$(stat -c%s "/app/$MODEL_FILE")
    echo "📊 最終確認:"
    echo "   ファイル: /app/$MODEL_FILE"
    echo "   サイズ: $FINAL_SIZE bytes ($(echo "scale=2; $FINAL_SIZE / 1024 / 1024" | bc) MB)"
    echo ""
fi

echo "🚀 NeuroQ ハンドラーを起動中..."
echo ""

# メインのハンドラーを実行
exec python -u handler.py
