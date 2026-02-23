#!/usr/bin/env python3
import os
import urllib.request
import json
import sentencepiece as spm
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def train_sentencepiece():
    vocab_size = 8000
    model_prefix = "neuroq_tokenizer_oasst1_ja"
    
    print("=" * 70)
    print("🔤 Training SentencePiece Tokenizer with OASST1 and OASST2")
    print("=" * 70)

    # 1. Download Datasets
    print("Loading OASST1 and OASST2 datasets...")
    datasets = []
    
    try:
        ds1 = load_dataset("kunishou/oasst1-89k-ja", split="train")
        datasets.append(ds1)
    except Exception as e:
        print(f"Failed to load oasst1-ja: {e}")
        
    try:
        ds2 = load_dataset("OpenAssistant/oasst2", split="train")
        datasets.append(ds2)
    except Exception as e:
        print(f"Failed to load OASST2: {e}")
        
    # 2. Extract Text
    print("Extracting strings...")
    corpus_path = "data/combined_oasst_ja_corpus.txt"
    os.makedirs("data", exist_ok=True)
    
    with open(corpus_path, "w", encoding="utf-8") as f:
        for ds in datasets:
            for item in tqdm(ds, desc="Processing text"):
                text = ""
                if "text" in item and item["text"]:
                    text = item["text"]
                elif "text_ja" in item and item["text_ja"]:
                    text = item["text_ja"]
                    
                if text:
                    # Simple heuristic: Keep only if it contains some Japanese characters
                    if any('\u3040' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF' for c in text):
                        f.write(text.replace("\n", " ") + "\n")
                        
    # 3. Train SentencePiece
    print(f"Training SentencePiece (Vocab Size: {vocab_size})...")
    
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="unigram",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=["<USER>", "<ASSISTANT>", "<SYSTEM>"],
        max_sentence_length=16384,
        num_threads=os.cpu_count()
    )
    
    print(f"✅ Tokenizer saved as {model_prefix}.model (Vocabulary: {model_prefix}.vocab)")
    
if __name__ == "__main__":
    train_sentencepiece()
