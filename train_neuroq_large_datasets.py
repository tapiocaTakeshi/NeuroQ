#!/usr/bin/env python3
"""
neuroQ + 大規模データセット プリトレーニング
LAION/C4/WebText/PileでQBnN学習
"""
import sys
sys.path.append('neuroq-runpod')
sys.path.append('.')

import torch
from torch.utils.data import DataLoader
import neuroquantum_brain  # 既存neuroQモデル
from neuroq_large_datasets_loader import LargeDatasetLoader
from neuroq_tokenizer.model import load_tokenizer  # 想定

def train_on_large_datasets():
    """大規模データセットでneuroQプリトレーニング"""
    
    # 1. トークナイザー初期化（既存neuroq_tokenizer使用）
    tokenizer = load_tokenizer('neuroq_tokenizer_8k.model')
    
    # 2. データローダー
    loader = LargeDatasetLoader()
    train_data = loader.prepare_neuroq_training_data(
        datasets=['common_crawl', 'webtext', 'the_pile', 'laion_5b'],
        output_file='data/large_datasets_for_neuroq.txt',
        max_tokens=5*10**9  # 50億トークン
    )
    
    # 3. neuroQモデル初期化（QBnN）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = neuroquantum_brain.NeuroQuantumBrain(
        vocab_size=tokenizer.vocab_size,
        n_qubits=512,  # 大規模対応
        layers=24,
        dim=2048
    ).to(device)
    
    # 4. トレーニング設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # 5. トレーニングループ（分散対応）
    model.train()
    for epoch in range(10):  # Pile基準10エポック
        data_loader = DataLoader(
            dataset=train_data,
            batch_size=8,  # A100 x8推奨
            shuffle=True,
            num_workers=4
        )
        
        for batch_idx, batch_texts in enumerate(data_loader):
            tokens = tokenizer(batch_texts, return_tensors='pt', padding=True).to(device)
            
            outputs = model(**tokens, labels=tokens['input_ids'])
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # チェックポイント保存
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, f'checkpoints/neuroq_large_datasets_epoch{epoch}.pt')
    
    print("Large dataset pretraining completed!")

if __name__ == "__main__":
    train_on_large_datasets()
