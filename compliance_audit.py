"""
NeuroQ 生成AI法的・倫理的コンプライアンス監査ツール
2026年2月14日作成 - すべての法的要件対応
"""
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

class NeuroQComplianceAuditor:
    def __init__(self, project_root: str = "/Users/yuyahiguchi/Program/neuroQ"):
        self.root = Path(project_root)
        self.risk_report = {}
        self.legal_frameworks = {
            "copyright": self._check_copyright,
            "gdpr": self._check_gdpr,
            "license": self._check_license,
            "bias": self._check_bias
        }
    
    def audit_all(self) -> Dict:
        """全ファイル法的・倫理的監査"""
        for file_path in self.root.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.txt', '.py', '.json', '.log']:
                self._audit_file(file_path)
        return self.risk_report
    
    def _audit_file(self, file_path: Path):
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        risks = {}
        
        # 著作権侵害チェック
        if self._check_copyright(content):
            risks["copyright"] = "高リスク: 保護コンテンツ検出"
        
        # GDPRパーソナルデータチェック
        if self._check_gdpr(content):
            risks["gdpr"] = "高リスク: 個人情報検出"
        
        # ライセンス違反チェック
        license_issue = self._check_license(file_path, content)
        if license_issue:
            risks["license"] = license_issue
        
        # バイアスチェック
        bias_score = self._check_bias(content)
        if bias_score > 0.7:
            risks["bias"] = f"高バイアススコア: {bias_score:.2f}"
        
        if risks:
            self.risk_report[str(file_path)] = risks
    
    def _check_copyright(self, content: str) -> bool:
        """著作権保護フレーズ検出"""
        copyright_patterns = [
            r'©\s*\d{4}',
            r'All rights reserved',
            r'著作権所有',
            r'無断複製禁止',
            r'published by.*press',
            r'new york times',
            r'code from github.*repository'
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in copyright_patterns)
    
    def _check_gdpr(self, content: str) -> bool:
        """GDPR対象個人情報検出"""
        gdpr_patterns = [
            r'\b\d{3}-\d{4}-\d{4}\b',  # 電話番号
            r'\b[0-9]{17}\b',         # マイナンバー
            r'[a-zA-Z一-龯]+\s+[一-龯]+[\s0-9@]+gmail\.com',  # 実名+メール
            r'住所[:：].{1,50}[都道府県]',  # 住所
            r'生年月日[:：].{1,20}'
        ]
        return any(re.search(pattern, content) for pattern in gdpr_patterns)
    
    def _check_license(self, file_path: Path, content: str) -> str:
        """ライセンスコンプライアンスチェック"""
        if 'oasst1' in str(file_path):
            if 'Apache 2.0' not in content and 'https://huggingface.co/datasets/OpenAssistant/oasst1' not in content:
                return "OASST1:帰属表記・派生作品条件違反"
        return ""
    
    def _check_bias(self, content: str) -> float:
        """バイアススコア計算"""
        bias_terms = {
            'male': 0.1, '男性': 0.1,
            'female': 0.1, '女性': 0.1,
            'engineer': 0.15, 'エンジニア': 0.15,
            'doctor': 0.12, '医師': 0.12
        }
        total = sum(content.count(term) * score for term, score in bias_terms.items())
        return min(total / max(len(content.split()), 1), 1.0)

# 実行
if __name__ == "__main__":
    auditor = NeuroQComplianceAuditor()
    report = auditor.audit_all()
    
    print("=== NeuroQ 法的・倫理的コンプライアンス報告 ===")
    high_risk_files = [f for f, risks in report.items() if any('高リスク' in r for r in risks.values())]
    
    print(f"高リスクファイル数: {len(high_risk_files)}")
    for file in high_risk_files[:10]:  # トップ10表示
        print(f"  ❌ {file}: {report[file]}")
    
    # 緊急修正提案
    with open("LEGAL_COMPLIANCE_FIX.md", "w", encoding="utf-8") as f:
        f.write("# NeuroQ緊急法的対応計画\n\n")
        f.write("## 即時対応（24時間以内）\n")
        f.write("- データセット完全クリーン化\n")
        f.write("- Apache 2.0ライセンスヘッダー追加\n")
        f.write("- GDPR準拠データフィルタ実装\n")
        f.write(f"## 影響ファイル: {len(high_risk_files)}件\n")
        for file in high_risk_files:
            f.write(f"- {file}\n")
