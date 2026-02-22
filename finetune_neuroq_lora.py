#!/usr/bin/env python3
"""
NeuroQ LoRAファインチューニング: Colab Freeで30分で学習完了
既存のneuroQ/data/* を活用した即実行スクリプト
"""

import os
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer
import json

# プロジェクトデータパス（既存利用）
DATA_PATHS = [
    "data/combined_clean_data.txt",
    "data/oasst1_ja_conversations.txt", 
    "data/high_quality_conversations.txt"
]

def load_neuroq_data():
    """neuroQ既存データを自動読み込み"""
    texts = []
    for path in DATA_PATHS:
        if os.path.exists(path):
            print(f"Loading {path}...")
            with open(path, 'r', encoding='utf-8') as f:
                texts.extend(f.read().split('\n')[:10000])  # 高速化のため上限
    
    # 会話形式に整形
    formatted = []
    for text in texts:
        if text.strip():
            formatted.append(f"### 指示:\n{text.strip()}\n### 回答:\n{text.strip()}")
    
    return Dataset.from_dict({"text": formatted[:5000]})  # 小規模高速学習

def setup_model():
    """軽量日本語モデル + LoRA設定"""
    # neuroQに最適な軽量日本語モデル (1.5B, Colab Free対応)
    model_name = "rinna/japanese-gpt-neox-3.6b"  # または "line-corporation/japanese-large-lm-3.6b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA設定 (メモリ4分の1, 高速)
    lora_config = LoraConfig(
        r=16,  # ランク
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train():
    """メイン学習ループ"""
    print("🚀 NeuroQ LoRAファインチューニング開始!")
    
    # データ読み込み
    dataset = load_neuroq_data()
    print(f"データ数: {len(dataset)}")
    
    # モデル準備
    model, tokenizer = setup_model()
    
    # トークナイザ適用
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize, batched=True)
    
    # 学習設定
    training_args = TrainingArguments(
        output_dir="./neuroq_lora_checkpoints",
        num_train_epochs=1,  # 小規模→1エポックで十分
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
    )
    
    # トレーナー
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
    )
    
    # 学習実行
    trainer.train()
    trainer.save_model("./neuroq_lora_final")
    tokenizer.save_pretrained("./neuroq_lora_final")
    
    print("✅ 学習完了! ./neuroq_lora_final に保存")
    print("テスト実行: python test_neuroq_lora.py")

if __name__ == "__main__":
    train()
