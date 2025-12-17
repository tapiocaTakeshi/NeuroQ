# NeuroQ RunPod Serverless - Optimized for Stability
# ===================================================
# 高速起動 & 安定動作のための最適化済みDockerfile

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 環境変数（起動高速化）
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Git LFSをインストール
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 依存関係をインストール（最小限）
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y transformers || true && \
    rm -rf /root/.cache/pip

# Git LFSファイルを取得するため、リポジトリをクローン
# （事前学習済みモデルなどのLFSファイル用）
RUN if [ -d "/tmp/neuroq-repo" ]; then rm -rf /tmp/neuroq-repo; fi

# ARGでリポジトリURLを指定可能にする（デフォルトはNeuroQリポジトリ）
ARG GIT_REPO_URL="https://github.com/tapiocaTakeshi/NeuroQ.git"
ARG GIT_BRANCH="main"

# リポジトリが指定されている場合はクローン＆LFS pull
RUN if [ -n "$GIT_REPO_URL" ]; then \
        echo "📦 Cloning repository with Git LFS..." && \
        git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO_URL" /tmp/neuroq-repo && \
        cd /tmp/neuroq-repo && \
        git lfs pull && \
        echo "✅ Git LFS files pulled successfully"; \
    else \
        echo "⚠️ No GIT_REPO_URL provided, skipping git clone"; \
    fi

# コアファイルのみコピー（軽量化）
# LFSファイルがクローンされている場合はそちらから、なければローカルから
COPY neuroquantum_layered.py /app/
COPY neuroquantum_brain.py /app/
COPY qbnn_layered.py /app/
COPY qbnn_brain.py /app/
COPY handler.py /app/
COPY pretrain_openai.py /app/

# 事前学習済みモデルモジュール（同じディレクトリに配置）
COPY neuroq_pretrained.py /app/

# トークナイザー
COPY neuroq_tokenizer.model /app/
COPY neuroq_tokenizer.vocab /app/

# Entrypointスクリプト（起動時にモデルをダウンロード）
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# 事前学習済みモデル（LFSファイル）
# NOTE: モデルファイルは起動時にentrypoint.shで自動ダウンロードされます
# ビルド時にLFSファイルが利用可能な場合はコピーしますが、必須ではありません
# オプション1: git cloneからコピー（GIT_REPO_URLが指定されている場合）
RUN if [ -f "/tmp/neuroq-repo/neuroq_pretrained.pt" ]; then \
        echo "📦 Using LFS-pulled pretrained model from git clone" && \
        MODEL_SIZE=$(stat -c%s /tmp/neuroq-repo/neuroq_pretrained.pt) && \
        if [ "$MODEL_SIZE" -ge 10000 ]; then \
            cp /tmp/neuroq-repo/neuroq_pretrained.pt /app/neuroq_pretrained.pt && \
            echo "✅ Model file from git clone: $MODEL_SIZE bytes" && \
            ls -lh /app/neuroq_pretrained.pt; \
        else \
            echo "⚠️ Model file from git clone is too small ($MODEL_SIZE bytes)"; \
            echo "   モデルファイルは起動時にダウンロードされます"; \
        fi; \
    else \
        echo "ℹ️ No model file from git clone"; \
        echo "   モデルファイルは起動時にダウンロードされます"; \
    fi

# オプション2: ローカルからコピー（利用可能な場合のみ）
# LFSポインタファイルの場合でもエラーにせず、起動時にダウンロードします
COPY --chown=root:root neuroq_pretrained.pt /tmp/neuroq_pretrained_local.pt || true

RUN if [ -f "/tmp/neuroq_pretrained_local.pt" ] && [ ! -f "/app/neuroq_pretrained.pt" ]; then \
        LOCAL_SIZE=$(stat -c%s /tmp/neuroq_pretrained_local.pt 2>/dev/null || echo "0") && \
        echo "📋 Checking neuroq_pretrained.pt from local build context..." && \
        echo "   Local file size: $LOCAL_SIZE bytes" && \
        if [ "$LOCAL_SIZE" -ge 10000 ]; then \
            cp /tmp/neuroq_pretrained_local.pt /app/neuroq_pretrained.pt && \
            echo "✅ Using local model file: $LOCAL_SIZE bytes"; \
        else \
            echo "⚠️ Local file is a Git LFS pointer ($LOCAL_SIZE bytes)"; \
            echo "   モデルファイルは起動時に自動ダウンロードされます"; \
        fi; \
    fi && \
    rm -f /tmp/neuroq_pretrained_local.pt

# モデルファイルの状態を確認（存在しない場合も継続）
RUN if [ -f "/app/neuroq_pretrained.pt" ]; then \
        FINAL_SIZE=$(stat -c%s /app/neuroq_pretrained.pt) && \
        echo "📊 Model file available in image: $FINAL_SIZE bytes" && \
        ls -lh /app/neuroq_pretrained.pt; \
    else \
        echo "ℹ️ Model file not included in image"; \
        echo "   ✅ 起動時にentrypoint.shが自動ダウンロードします"; \
    fi

# クローンしたリポジトリをクリーンアップ（不要なファイルを削除してイメージサイズ削減）
RUN if [ -d "/tmp/neuroq-repo" ]; then rm -rf /tmp/neuroq-repo; fi

# 環境変数（デフォルト設定）
ENV NEUROQ_MODE="layered"
ENV NEUROQ_VOCAB_SIZE="8000"
ENV MODEL_FILE="neuroq_pretrained.pt"
ENV GIT_REPO_URL="https://github.com/tapiocaTakeshi/NeuroQ.git"
ENV GIT_BRANCH="main"

# エントリポイント（起動時にモデルをダウンロード）
ENTRYPOINT ["/app/entrypoint.sh"]
