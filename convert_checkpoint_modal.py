import torch
from pathlib import Path
import sys
import os

# Import NeuroQuantum components
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
runpod_dir = os.path.join(current_dir, 'neuroq-runpod')
if runpod_dir not in sys.path:
    sys.path.insert(0, runpod_dir)

from neuroquantum_layered import NeuroQuantumConfig, NeuroQuantum

def convert_checkpoint(input_path: str, output_path: str):
    print(f"Loading state dict from {input_path}...")
    state_dict = torch.load(input_path, map_location="cpu")
    
    print("Creating config and tokenizer info...")
    # These match the hyperparams used in train_qbnn_oasst.py
    config = {
        'vocab_size': 50257,
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 4,
        'lambda_entangle': 0.5,
        'max_seq_len': 256,
        'dropout': 0.1
    }
    
    tokenizer_info = {
        'type': 'tiktoken',
        'encoding': 'p50k_base'
    }
    
    checkpoint_dict = {
        'model_state_dict': state_dict,
        'config': config,
        'tokenizer': tokenizer_info
    }
    
    print(f"Saving formatted checkpoint to {output_path}...")
    torch.save(checkpoint_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    convert_checkpoint("qbnn_oasst_epoch_3.pt", "neuroq_oasst_modal_ready.pt")
