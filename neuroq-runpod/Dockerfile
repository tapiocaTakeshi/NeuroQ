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

# 事前学習済みモデル（LFSファイル）
# オプション1: git cloneからコピー（GIT_REPO_URLが指定されている場合）
RUN if [ -f "/tmp/neuroq-repo/neuroq_pretrained.pt" ]; then \
        echo "📦 Using LFS-pulled pretrained model from git clone" && \
        MODEL_SIZE=$(stat -c%s /tmp/neuroq-repo/neuroq_pretrained.pt) && \
        if [ "$MODEL_SIZE" -ge 10000 ]; then \
            cp /tmp/neuroq-repo/neuroq_pretrained.pt /app/neuroq_pretrained.pt && \
            echo "✅ Model file from git clone: $MODEL_SIZE bytes" && \
            ls -lh /app/neuroq_pretrained.pt; \
        else \
            echo "⚠️ Model file from git clone is too small ($MODEL_SIZE bytes), trying local copy..."; \
        fi; \
    fi

# オプション2: ローカルからコピー（git cloneが成功しなかった場合）
# Copy model file from build context to temp location for validation
COPY --chown=root:root neuroq_pretrained.pt /tmp/neuroq_pretrained_local.pt

# Validate and copy from local build context if not already present
RUN if [ ! -f "/app/neuroq_pretrained.pt" ]; then \
        echo "📋 Validating neuroq_pretrained.pt from local build context..." && \
        LOCAL_SIZE=$(stat -c%s /tmp/neuroq_pretrained_local.pt) && \
        echo "   Local file size: $LOCAL_SIZE bytes" && \
        if [ "$LOCAL_SIZE" -lt 10000 ]; then \
            echo "" && \
            echo "❌ ERROR: neuroq_pretrained.pt is a Git LFS pointer file ($LOCAL_SIZE bytes)" && \
            echo "   The actual model file was not pulled from Git LFS." && \
            echo "" && \
            echo "   This happened because:" && \
            echo "   - No GIT_REPO_URL was provided to clone and pull LFS files" && \
            echo "   - The local file is an LFS pointer, not the actual model" && \
            echo "" && \
            echo "   To fix this, use ONE of the following methods:" && \
            echo "" && \
            echo "   1. Build with repository URL (RECOMMENDED):" && \
            echo "      docker build -f neuroq-runpod/Dockerfile \\" && \
            echo "        --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \\" && \
            echo "        --build-arg GIT_BRANCH=main \\" && \
            echo "        -t neuroq:latest ." && \
            echo "" && \
            echo "   2. Pull LFS files locally first:" && \
            echo "      git lfs install" && \
            echo "      git lfs pull" && \
            echo "      docker build -f neuroq-runpod/Dockerfile -t neuroq:latest ." && \
            echo "" && \
            echo "   3. Use the build helper script:" && \
            echo "      ./neuroq-runpod/build.sh" && \
            echo "" && \
            exit 1; \
        else \
            cp /tmp/neuroq_pretrained_local.pt /app/neuroq_pretrained.pt && \
            echo "✅ Using local model file: $LOCAL_SIZE bytes"; \
        fi; \
    fi && \
    rm -f /tmp/neuroq_pretrained_local.pt

# 最終確認
RUN echo "📊 Final validation of neuroq_pretrained.pt..." && \
    ls -lh /app/neuroq_pretrained.pt && \
    FINAL_SIZE=$(stat -c%s /app/neuroq_pretrained.pt) && \
    echo "✅ Model file ready: $FINAL_SIZE bytes"

# クローンしたリポジトリをクリーンアップ（不要なファイルを削除してイメージサイズ削減）
RUN if [ -d "/tmp/neuroq-repo" ]; then rm -rf /tmp/neuroq-repo; fi

# 環境変数（デフォルト設定）
ENV NEUROQ_MODE="layered"
ENV NEUROQ_VOCAB_SIZE="8000"

# エントリポイント
CMD ["python", "-u", "handler.py"]
