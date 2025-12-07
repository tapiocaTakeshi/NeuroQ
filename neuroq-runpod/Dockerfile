# NeuroQ RunPod Serverless Worker
# ================================
# Brain & Layered モード対応
# ニューロン数・隠れ層次元など環境変数で設定可能
#
# RunPod Serverless では Build Context が "/" になるため、
# neuroq-runpod/ ディレクトリからのパスを指定する必要があります。

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# 必要なファイルをコピー（neuroq-runpodディレクトリから）
COPY neuroq-runpod/handler.py /app/
COPY neuroq-runpod/requirements.txt /app/

# 参照元ファイルをコピー（neuroq-runpodディレクトリ内に配置）
COPY neuroq-runpod/neuroquantum_brain.py /app/
COPY neuroq-runpod/neuroquantum_layered.py /app/
COPY neuroq-runpod/qbnn_layered.py /app/
COPY neuroq-runpod/qbnn_brain.py /app/

# # SentencePiece トークナイザーモデルをコピー
# COPY neuroq-runpod/neuroq_tokenizer.model /app/
# COPY neuroq-runpod/neuroq_tokenizer.vocab /app/

# 依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# ========================================
# 環境変数（デフォルト設定）
# ========================================

# モード設定
ENV NEUROQ_MODE="layered"
ENV NEUROQ_MODEL_PATH="neuroq_layered_model.pt"
ENV NEUROQ_TOKENIZER_PATH="neuroq_tokenizer.model"

# 共通設定
ENV NEUROQ_EMBED_DIM="128"
ENV NEUROQ_NUM_LAYERS="3"
ENV NEUROQ_DROPOUT="0.1"
ENV NEUROQ_MAX_SEQ_LEN="256"

# Brain Mode 設定
ENV NEUROQ_NUM_NEURONS="100"
ENV NEUROQ_CONNECTION_DENSITY="0.25"
ENV NEUROQ_LAMBDA_BRAIN="0.35"

# Layered Mode 設定
ENV NEUROQ_HIDDEN_DIM="256"
ENV NEUROQ_NUM_HEADS="4"
ENV NEUROQ_LAMBDA_LAYERED="0.5"

# RunPod API設定（実行時に環境変数として設定）
ARG RUNPOD_API_KEY=""
ARG RUNPOD_ENDPOINT_ID=""
ENV RUNPOD_API_KEY="${RUNPOD_API_KEY}"
ENV RUNPOD_ENDPOINT_ID="${RUNPOD_ENDPOINT_ID}"

# エントリポイント
CMD ["python", "handler.py"]
