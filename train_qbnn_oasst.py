import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import langdetect
from langdetect.lang_detect_exception import LangDetectException

# Load HuggingFace Datasets
try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets library: pip install datasets")
    sys.exit(1)

# Import local QBNN components
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
runpod_dir = os.path.join(current_dir, 'neuroq-runpod')
if runpod_dir not in sys.path:
    sys.path.insert(0, runpod_dir)

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
from tiktoken_tokenizer import TikTokenTokenizer

# ==========================================
# Hyperparameters (Optimized for Stability)
# ==========================================
VOCAB_SIZE = 50257       # p50k_base
EMBED_DIM = 256          # d_model
HIDDEN_DIM = 512         # FFN internal dim
NUM_HEADS = 8            # Attention heads
NUM_LAYERS = 4           # Transformer blocks
MAX_SEQ_LEN = 256        # Sequence length
BATCH_SIZE = 8           # Batch size (per accumulation step)
ACCUMULATION_STEPS = 4   # Effective batch size = 32
EPOCHS = 3               # Training epochs
LEARNING_RATE = 2e-4     # Slightly lower LR for larger batch/dataset

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Dataset Preparation
# ==========================================
class OASSTDataset(Dataset):
    def __init__(self, tokenized_data, pad_id=0):
        self.data = tokenized_data
        self.pad_id = pad_id
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        pad_len = MAX_SEQ_LEN - 1 - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), self.pad_id, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
            
        return x, y

def is_en_ja(text):
    try:
        lang = langdetect.detect(text)
        return lang in ['en', 'ja']
    except LangDetectException:
        # Fallback if langdetect fails
        if any('\u3040' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF' for c in text):
            return True
        ascii_count = sum(1 for c in text if ord(c) < 128)
        return (ascii_count / max(1, len(text))) > 0.8

def prepare_data(tokenizer, max_samples=20000):
    print("Loading OASST datasets...")
    
    # Dataset 1: OASST1 (via guanaco variant which is already flattened to conversations)
    dataset1 = load_dataset("timdettmers/openassistant-guanaco", split="train")
    
    if max_samples:
        subset_size = min(max_samples // 2, len(dataset1))
        dataset1 = dataset1.select(range(subset_size))
        
    encoded_data = []
    print(f"Filtering and Tokenizing OASST1 ({len(dataset1)} samples)...")
    for item in tqdm(dataset1):
        text = item["text"]
        if not is_en_ja(text):
            continue
            
        tokens = tokenizer.encode(text)
        if len(tokens) >= 10:
            encoded_data.append(tokens[:MAX_SEQ_LEN])
            
    # Dataset 2: OASST2 (English Top1 variant flattened to conversations)
    try:
        dataset2 = load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="train")
        if max_samples:
             subset_size = min(max_samples // 2, len(dataset2))
             dataset2 = dataset2.select(range(subset_size))
             
        print(f"Filtering and Tokenizing OASST2 ({len(dataset2)} samples)...")
        for item in tqdm(dataset2):
            text = item["text"]
            if not is_en_ja(text):
                continue
                
            tokens = tokenizer.encode(text)
            if len(tokens) >= 10:
                encoded_data.append(tokens[:MAX_SEQ_LEN])
    except Exception as e:
        print(f"Warning: Could not load or format OASST2 correctly. Only using OASST1. Error: {e}")

    print(f"Total Prepared Sequences (OASST1 + OASST2, En/Ja Only): {len(encoded_data)}")
    return encoded_data

# ==========================================
# Training Loop
# ==========================================
def train():
    print("=" * 70)
    print("🚀 Starting QBNN Autoregressive Training on OASST1 & OASST2")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Gradient Accumulation Steps: {ACCUMULATION_STEPS} (Effective Batch: {BATCH_SIZE * ACCUMULATION_STEPS})")
    
    tokenizer = TikTokenTokenizer(encoding_name="p50k_base")
    
    # 1. Prepare Dataset (using 20,000 samples for substantial training)
    tokenized_texts = prepare_data(tokenizer, max_samples=20000)
    dataset = OASSTDataset(tokenized_texts, pad_id=tokenizer.pad_id)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    config = NeuroQuantumConfig(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        hidden_dim=HIDDEN_DIM, 
        num_heads=NUM_HEADS, 
        num_layers=NUM_LAYERS, 
        lambda_entangle=0.5,
        max_seq_len=MAX_SEQ_LEN
    )
    
    model = NeuroQuantum(config).to(DEVICE)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model architecture: d_model={EMBED_DIM}, layers={NUM_LAYERS}, vocab={VOCAB_SIZE}")
    print(f"Total trainable parameters: {trainable_params:,}")
    
    # 3. Training
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (x, y) in enumerate(progress):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            logits = model(x)
            
            # Calculate loss
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
            # Scale loss by accumulation steps
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            # Step optimizer every ACCUMULATION_STEPS
            if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(dataloader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUMULATION_STEPS # Original loss scaling for logging
            progress.set_postfix({'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed | Average Loss: {avg_loss:.4f}")
        
        # 4. Save checkpoint (Using dict format for Modal API compatibility directly)
        save_path = f"qbnn_oasst_full_epoch_{epoch+1}.pt"
        
        checkpoint_dict = {
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': VOCAB_SIZE, 'embed_dim': EMBED_DIM, 'hidden_dim': HIDDEN_DIM,
                'num_heads': NUM_HEADS, 'num_layers': NUM_LAYERS, 'lambda_entangle': 0.5,
                'max_seq_len': MAX_SEQ_LEN, 'dropout': 0.1
            },
            'tokenizer': {'type': 'tiktoken', 'encoding': 'p50k_base'}
        }
        
        torch.save(checkpoint_dict, save_path)
        print(f"💾 Checkpoint saved (Modal compatible format): {save_path}\n")
        
    print("✅ Training Complete!")

if __name__ == "__main__":
    train()
