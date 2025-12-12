# NeuroQ RunPod Serverless - Optimized for Stability
# ===================================================
# 高速起動 & 安定動作のための最適化済みDockerfile

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 環境変数（起動高速化）
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 依存関係をインストール（最小限）
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y transformers || true && \
    rm -rf /root/.cache/pip

# コアファイルのみコピー（軽量化）
COPY neuroquantum_layered.py /app/
COPY neuroquantum_brain.py /app/
COPY qbnn_layered.py /app/
COPY qbnn_brain.py /app/
COPY handler.py /app/
COPY pretrain_openai.py /app/

# トークナイザー
COPY neuroq_tokenizer.model /app/
COPY neuroq_tokenizer.vocab /app/

# 事前学習済みモデル
COPY neuroq_pretrained.pt /app/

# 環境変数（デフォルト設定）
ENV NEUROQ_MODE="layered"
ENV NEUROQ_VOCAB_SIZE="8000"

# エントリポイント
CMD ["python", "-u", "handler.py"]
