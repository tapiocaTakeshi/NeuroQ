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
COPY neuroq_tokenizer_8k.model /app/
COPY neuroq_tokenizer_8k.vocab /app/

# 事前学習済みモデルファイル (.pth)
COPY neuroq_pretrained.pth /app/

# Entrypointスクリプト（起動時にモデルをダウンロード）
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# クローンしたリポジトリをクリーンアップ（不要なファイルを削除してイメージサイズ削減）
RUN if [ -d "/tmp/neuroq-repo" ]; then rm -rf /tmp/neuroq-repo; fi

# 環境変数（デフォルト設定）
ENV NEUROQ_MODE="layered"
ENV NEUROQ_VOCAB_SIZE="8000"
ENV MODEL_FILE="neuroq_pretrained.pth"
ENV GIT_REPO_URL="https://github.com/tapiocaTakeshi/NeuroQ.git"
ENV GIT_BRANCH="main"

# エントリポイント（起動時にモデルをダウンロード）
ENTRYPOINT ["/app/entrypoint.sh"]
