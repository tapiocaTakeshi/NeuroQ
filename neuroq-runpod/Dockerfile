# NeuroQ RunPod Serverless Worker
# ================================
# Brain & Layered モード対応
# Version: 2.0.1 (train_on_texts alias added)
#
# Build Context: neuroq-runpod ディレクトリ
# Usage: cd neuroq-runpod && docker build -t neuroq-runpod .
# または: docker build -t neuroq-runpod neuroq-runpod/

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# キャッシュバスト用（ビルドのたびに変更）
# Last update: 2025-12-08 07:20 JST - FORCE REBUILD
ARG CACHEBUST=3
RUN echo "Build timestamp: $(date) - CACHEBUST=3"

WORKDIR /app

# 依存関係を先にインストール（キャッシュ効率化）
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 参照元ファイルをコピー（train_on_textsメソッド含む）
COPY neuroquantum_layered.py /app/
RUN echo "neuroquantum_layered.py copied at $(date)"
COPY neuroquantum_brain.py /app/
RUN echo "neuroquantum_brain.py copied at $(date)"
COPY qbnn_layered.py /app/
COPY qbnn_brain.py /app/

# ハンドラーをコピー
COPY handler.py /app/
RUN echo "handler.py copied at $(date)"

# SentencePiece トークナイザーモデルをコピー
COPY neuroq_tokenizer.model /app/
COPY neuroq_tokenizer.vocab /app/

# ========================================
# 環境変数（デフォルト設定）
# ========================================

ENV NEUROQ_MODE="layered"
ENV NEUROQ_MODEL_PATH="neuroq_layered_model.pt"
ENV NEUROQ_TOKENIZER_PATH="neuroq_tokenizer.model"
ENV NEUROQ_EMBED_DIM="128"
ENV NEUROQ_NUM_LAYERS="3"
ENV NEUROQ_DROPOUT="0.1"
ENV NEUROQ_MAX_SEQ_LEN="256"
ENV NEUROQ_NUM_NEURONS="100"
ENV NEUROQ_CONNECTION_DENSITY="0.25"
ENV NEUROQ_LAMBDA_BRAIN="0.35"
ENV NEUROQ_HIDDEN_DIM="256"
ENV NEUROQ_NUM_HEADS="4"
ENV NEUROQ_LAMBDA_LAYERED="0.5"
# OPENAI_API_KEY（RunPodの環境変数で設定）
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY="${OPENAI_API_KEY}"

# エントリポイント
CMD ["python", "handler.py"]
