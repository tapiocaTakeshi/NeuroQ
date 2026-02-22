"""
NeuroQ 学習データ法的コンプライアンスフィルタ
著作権・個人情報自動除去
"""
import re
from pathlib import Path
import json

class LegalDataFilter:
    def __init__(self):
        self.banned_patterns = {
            'copyright': [
                r'©\s*\d{4}.*?all rights? reserved',
                r'new york times',
                r'without permission',
                r'無断.*転載.*禁止'
            ],
            'pii': [  # GDPR対象
                r'\b\d{3}-\d{4}-\d{4}\b',
                r'[一-龯]{2,4}\s+[一-龯]{2,4}\s*@',
                r'住所[:：].*?[都道府県]'
            ]
        }
    
    def clean_dataset(self, input_file: str, output_file: str):
        """法的問題除去版データセット生成"""
        clean_lines = []
        total, removed = 0, 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                total += 1
                clean_line = line.strip()
                
                # フィルタ適用
                is_clean = True
                for category, patterns in self.banned_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, clean_line, re.IGNORECASE):
                            is_clean = False
                            break
                    if not is_clean:
                        break
                
                if is_clean and len(clean_line) > 10:
                    clean_lines.append(clean_line)
                else:
                    removed += 1
        
        # クリーン版保存
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(clean_lines))
        
        print(f"クリーン化完了: {total}→{len(clean_lines)}行 (除去率: {removed/total*100:.1f}%)")
        return len(clean_lines)

# neuroQデータセット即時クリーン化
filter = LegalDataFilter()
filter.clean_dataset("data/oasst1_ja_conversations.txt", "data/oasst1_ja_compliant.txt")
filter.clean_dataset("data/combined_clean_data.txt", "data/combined_compliant.txt")
