import torch
from pathlib import Path
from neuroq_runpod.neuroquantum_brain import NeuroQuantumBrain
from neuroq_runpod.data import load_conversation_data  # 拡張

PROJECT_ROOT = Path("/Users/yuyahiguchi/Program/neuroQ")
DATA_DIR = PROJECT_ROOT / "data"

# 拡張データロード
train_data = []
train_data.extend(load_conversation_data(DATA_DIR / "oasst1_ja_conversations.txt"))
train_data.extend(load_conversation_data(DATA_DIR / "oasst1_en.jsonl"))  # 新規英語
train_data.extend(load_bookcorpus_sample(DATA_DIR / "bookcorpus"))  # 新規書籍

# LAIONマルチモーダル (画像テキスト埋め込み)
# multimodal_data = load_laion_pairs(DATA_DIR / "laion_sample")

model = NeuroQuantumBrain.from_pretrained("neuroq_pretrained")
model.train(
    train_data,
    epochs=10,
    batch_size=32,
    lr=1e-4,
    save_path=PROJECT_ROOT / "checkpoints" / "neuroq_public_trained.pt"
)
print("Training completed with public generative AI datasets.")
