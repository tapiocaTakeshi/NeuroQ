#!/usr/bin/env python3
"""学習済みNeuroQ LoRAモデルのテスト"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def test_neuroq_lora():
    """ファインチューニング済みモデルの動作確認"""
    model_path = "./neuroq_lora_final"
    
    # ベースモデル + LoRAアダプタ読み込み
    base_model = AutoModelForCausalLM.from_pretrained(
        "rinna/japanese-gpt-neox-3.6b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # NeuroQらしい会話テスト
    prompts = [
        "### 指示:\n量子コンピュータとは何ですか？\n### 回答:\n",
        "### 指示:\nNeuroQの特徴を教えて\n### 回答:\n",
        "今日の気分は？"
    ]
    
    model.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n{prompt}")
        print(f"生成: {response[len(prompt):]}")
        print("-" * 50)

if __name__ == "__main__":
    test_neuroq_lora()
