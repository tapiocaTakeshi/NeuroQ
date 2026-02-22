import os
import requests
import json
from pathlib import Path
import subprocess

# neuroQプロジェクトルート
PROJECT_ROOT = Path("/Users/yuyahiguchi/Program/neuroQ")
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

def download_oasst1_en():
    """OASST1英語会話データ追加 (GPT系会話データ相当)"""
    url = "https://the-eye.eu/public/AI/pile_oasst1/oasst1_valid.jsonl.bz2"
    output = DATA_DIR / "oasst1_en.jsonl.bz2"
    subprocess.run(["curl", "-L", url, "-o", str(output)])
    print("Downloaded OASST1 English for conversation fine-tuning.")

def download_laion_aesthetics_subset():
    """Stable Diffusion/DALL-E系画像-テキストペア (サブセット)"""
    # LAION-Aesthetics V2 5%サンプル (数GB規模)
    url = "https://the-eye.eu/public/AI/LAION/laion-aesthetics-v2/laion-aesthetics_v2_5plusB/meta/laion-aesthetics_v2_5plusB_meta_00.tar"
    output = DATA_DIR / "laion_sample.tar"
    subprocess.run(["curl", "-L", url, "-o", str(output)])
    print("Downloaded LAION aesthetics sample for multimodal.")

def download_bookcorpus_sample():
    """GPT系書籍データサンプル"""
    # BookCorpus (HuggingFace)
    subprocess.run([
        "python", "-m", "datasets", "load_dataset", 
        "bookcorpus", "split=train", "--output-dir", str(DATA_DIR / "bookcorpus")
    ])
    print("Downloaded BookCorpus sample.")

def generate_synthetic_gpt_data():
    """GPT風合成テキスト生成 (neuroQモデル使用)"""
    from neuroq_runpod.neuroquantum_brain import NeuroQuantumBrain  # 仮定インポート
    model = NeuroQuantumBrain.load_checkpoint(str(CHECKPOINTS_DIR / "neuroq_checkpoint.pt"))
    synthetic_data = []
    prompts = ["Write a story about AI.", "Explain quantum computing."] * 1000
    for prompt in prompts:
        output = model.generate(prompt, max_length=512)
        synthetic_data.append({"prompt": prompt, "completion": output})
    with open(DATA_DIR / "synthetic_gpt_data.jsonl", "w") as f:
        for item in synthetic_data:
            f.write(json.dumps(item) + "\n")
    print("Generated 2000 synthetic GPT-like samples.")

if __name__ == "__main__":
    download_oasst1_en()
    download_laion_aesthetics_subset()
    download_bookcorpus_sample()
    generate_synthetic_gpt_data()
    print("All public datasets downloaded. Ready for training.")
