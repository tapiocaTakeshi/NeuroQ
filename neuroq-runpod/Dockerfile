# NeuroQ RunPod Serverless Worker
# ================================
# Brain & Layered モード対応

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# 必要なファイルをコピー
COPY handler.py /app/
COPY neuroq_model.py /app/
COPY requirements.txt /app/

# モデルファイルをコピー（存在する場合）
COPY neuroq_*.pt /app/
COPY neuroq_*.json /app/

# 依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# 環境変数
ENV NEUROQ_MODE="layered"
ENV NEUROQ_MODEL_PATH="neuroq_layered_model.pt"
ENV NEUROQ_TOKENIZER_PATH="neuroq_tokenizer.json"

# エントリポイント
CMD ["python", "handler.py"]
