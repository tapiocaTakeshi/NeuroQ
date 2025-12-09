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
# Last update: 2025-12-08 07:15 JST
ARG CACHEBUST=2

WORKDIR /app

# 必要なファイルをコピー（カレントディレクトリから）
COPY handler.py /app/
COPY requirements.txt /app/

# 参照元ファイルをコピー
COPY neuroquantum_brain.py /app/
COPY neuroquantum_layered.py /app/
COPY qbnn_layered.py /app/
COPY qbnn_brain.py /app/

# SentencePiece トークナイザーモデルをコピー
COPY neuroq_tokenizer.model /app/
COPY neuroq_tokenizer.vocab /app/

# 依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

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
