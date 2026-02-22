#!/usr/bin/env python3
"""
LLM学習データセット詳細調査結果をneuroQに統合
主要データセット：Common Crawl, C4, The Pile, RefinedWeb, SlimPajama
"""

import os
import json
import requests
from pathlib import Path

# neuroQプロジェクトルート
PROJECT_ROOT = Path("/Users/yuyahiguchi/Program/neuroQ")
DATA_DIR = PROJECT_ROOT / "data"
RESEARCH_FILE = DATA_DIR / "llm_datasets_research.json"

def save_llm_datasets_research():
    """LLMデータセット調査結果をJSON保存"""
    datasets = {
        "Common Crawl": {
            "source": "Web crawler monthly snapshots",
            "size": "Petabyte-scale (250TB+ compressed)",
            "type": "Raw HTML/text (multilingual)",
            "license": "CC0",
            "curation": "Minimal deduplication",
            "pii_protection": "None",
            "copyright_issues": "High (NY Times lawsuits)"
        },
        "C4": {
            "source": "Cleaned Common Crawl",
            "size": "750GB (trillions tokens)",
            "type": "Clean English text",
            "license": "ODCL",
            "curation": "Grammar, list pages, n-gram filtering",
            "pii_protection": "Basic PII masking",
            "copyright_issues": "Inherited from Common Crawl"
        },
        "The Pile": {
            "source": "22 diverse high-quality corpora",
            "size": "800GB (300B tokens)",
            "type": "Academic (22%), Books (17%), Code (14%)",
            "license": "Mixed (MIT, CC-BY)",
            "curation": "Source-specific filtering + dedup",
            "pii_protection": "Source-dependent",
            "copyright_issues": "BookCorpus licensing issues"
        },
        "RefinedWeb": {
            "source": "Common Crawl 2022 snapshot",
            "size": "5T tokens (600TB raw)",
            "type": "Multilingual web text",
            "license": "CC0 (research)",
            "curation": "Perplexity + language filtering (top 5%)",
            "pii_protection": "PII filtering",
            "copyright_issues": "Commercial web content"
        },
        "SlimPajama": {
            "source": "Common Crawl + C4 extension",
            "size": "1.3T tokens",
            "type": "Clean multilingual web",
            "license": "Apache 2.0",
            "curation": "10-stage filtering pipeline",
            "pii_protection": "Enhanced PII detection",
            "copyright_issues": "Common Crawl derived"
        }
    }
    
    RESEARCH_FILE.parent.mkdir(exist_ok=True)
    with open(RESEARCH_FILE, 'w', encoding='utf-8') as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)
    
    print(f"✅ LLMデータセット調査結果を保存: {RESEARCH_FILE}")
    return datasets

def update_training_pipeline():
    """既存トレーニングパイプラインにリサーチ結果を反映"""
    research = save_llm_datasets_research()
    
    # neuroq-runpod/training用のメタデータ更新
    metadata = {
        "llm_datasets_research": research,
        "recommended_augmentation": [
            "Common Crawl品質フィルタ追加",
            "C4式前処理パイプライン",
            "The Pile多様性指標",
            "PII自動検出強化"
        ],
        "neuroq_compatibility": "高品質日本語データセットと併用推奨"
    }
    
    meta_file = PROJECT_ROOT / "neuroq-runpod" / "llm_datasets_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ トレーニングメタデータ更新: {meta_file}")

if __name__ == "__main__":
    update_training_pipeline()
    print("\n🎯 neuroQにLLMデータセット知見を統合完了")
    print("📊 data/llm_datasets_research.json で詳細確認可能")
