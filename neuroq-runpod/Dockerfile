# NeuroQ RunPod Serverless Worker
# ================================
# Brain & Layered モード対応
# ニューロン数・隠れ層次元など環境変数で設定可能
#
# ビルド方法:
# プロジェクトルートから実行:
#   docker build -f neuroq-runpod/Dockerfile -t neuroq-runpod .
#
# または neuroq-runpod/ ディレクトリから実行:
#   docker build -t neuroq-runpod .

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# 必要なファイルをコピー
COPY handler.py /app/
COPY neuroq_model.py /app/
COPY requirements.txt /app/

# 参照元ファイルをコピー（neuroquantum_brain.py と neuroquantum_layered.py）
# ビルド前に neuroq-runpod/ ディレクトリにこれらのファイルをコピーしておく必要があります
# または、プロジェクトルートから docker build -f neuroq-runpod/Dockerfile . でビルドしてください
COPY neuroquantum_brain.py /app/
COPY neuroquantum_layered.py /app/

# モデルファイルとトークナイザーをコピー
# （ビルド前に学習済みモデルを配置してください）
COPY neuroq_*.pt /app/
COPY neuroq_*.json /app/

# 依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# ========================================
# 環境変数（デフォルト設定）
# ========================================

# モード設定
ENV NEUROQ_MODE="brain"
ENV NEUROQ_MODEL_PATH="neuroq_brain_model.pt"
ENV NEUROQ_TOKENIZER_PATH="neuroq_tokenizer.json"

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

# エントリポイント
CMD ["python", "handler.py"]
