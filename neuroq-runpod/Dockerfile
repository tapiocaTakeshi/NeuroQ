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

# ARGでリポジトリURLを指定可能にする（デフォルトは空）
ARG GIT_REPO_URL=""
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

# 事前学習済みモデル（LFSファイル）
# オプション1: git cloneからコピー（GIT_REPO_URLが指定されている場合）
# オプション2: ローカルからコピー（デフォルト）
RUN if [ -f "/tmp/neuroq-repo/neuroq_pretrained.pt" ]; then \
        echo "📦 Using LFS-pulled pretrained model from git clone" && \
        cp /tmp/neuroq-repo/neuroq_pretrained.pt /app/neuroq_pretrained.pt && \
        ls -lh /app/neuroq_pretrained.pt; \
    else \
        echo "⚠️ No git clone found, will use local COPY for neuroq_pretrained.pt"; \
    fi

# ローカルからコピー（git cloneしない場合のデフォルト）
COPY neuroq_pretrained.pt /app/

# ファイルサイズを確認（LFSポインタファイルでないことを確認）
RUN echo "📊 Checking neuroq_pretrained.pt file size..." && \
    ls -lh /app/neuroq_pretrained.pt && \
    FILE_SIZE=$(stat -c%s /app/neuroq_pretrained.pt) && \
    if [ "$FILE_SIZE" -lt 10000 ]; then \
        echo "❌ WARNING: neuroq_pretrained.pt is very small ($FILE_SIZE bytes)"; \
        echo "   This might be a Git LFS pointer file, not the actual model!"; \
        echo "   Run 'git lfs pull' locally before building, or use --build-arg GIT_REPO_URL=<url>"; \
    else \
        echo "✅ neuroq_pretrained.pt size OK: $FILE_SIZE bytes"; \
    fi

# クローンしたリポジトリをクリーンアップ（不要なファイルを削除してイメージサイズ削減）
RUN if [ -d "/tmp/neuroq-repo" ]; then rm -rf /tmp/neuroq-repo; fi

# 環境変数（デフォルト設定）
ENV NEUROQ_MODE="layered"
ENV NEUROQ_VOCAB_SIZE="8000"

# エントリポイント
CMD ["python", "-u", "handler.py"]
