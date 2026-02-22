import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

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
BATCH_SIZE = 8           # Batch size
EPOCHS = 3               # Training epochs
LEARNING_RATE = 3e-4     # Learning rate

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
        # Autoregressive shift: input is x[:-1], target is x[1:]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Pad to max length
        pad_len = MAX_SEQ_LEN - 1 - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), self.pad_id, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)]) # -100 ignores loss
            
        return x, y

def prepare_data(tokenizer, max_samples=2000):
    print("Loading OASST dataset (timdettmers/openassistant-guanaco)...")
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    
    # Take a subset for rapid experimentation
    subset = dataset.select(range(min(max_samples, len(dataset))))
    
    encoded_data = []
    print("Tokenizing data...")
    for item in tqdm(subset):
        text = item["text"]
        tokens = tokenizer.encode(text)
        
        # Filter too short texts
        if len(tokens) < 10:
            continue
            
        # Truncate to MAX_SEQ_LEN
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]
            
        encoded_data.append(tokens)
        
    print(f"Prepared {len(encoded_data)} sequences.")
    return encoded_data

# ==========================================
# Training Loop
# ==========================================
def train():
    print("=" * 70)
    print("🚀 Starting QBNN Autoregressive Training on OASST")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    tokenizer = TikTokenTokenizer(encoding_name="p50k_base")
    
    # 1. Prepare Dataset
    tokenized_texts = prepare_data(tokenizer, max_samples=3000)
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
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (x, y) in enumerate(progress):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass: (batch, seq, vocab_size)
            logits = model(x)
            
            # Calculate loss: Reshape to (batch * seq, vocab_size) and (batch * seq)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed | Average Loss: {avg_loss:.4f}")
        
        # 4. Save checkpoint
        save_path = f"qbnn_oasst_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint saved: {save_path}\n")
        
    print("✅ Training Complete!")

if __name__ == "__main__":
    train()
