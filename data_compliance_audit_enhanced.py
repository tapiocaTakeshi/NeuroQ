"""
NeuroQ 生成AI学習データ 法的・倫理的コンプライアンス監査ツール (2026対応版)
"""
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Set
import spacy
from dataclasses import dataclass
import pandas as pd

@dataclass
class ComplianceIssue:
    issue_type: str  # 'copyright', 'pii', 'bias', 'poisoning'
    severity: str    # 'critical', 'high', 'medium', 'low'
    evidence: str
    line_numbers: List[int]
    recommendation: str

class NeuroQDataAuditor:
    def __init__(self):
        # 個人情報保護パターン（日本個人情報保護法・GDPR対応）
        self.pii_patterns = [
            r'\b[0-9]{3}-[0-9]{4}-[0-9]{4}\b',  # 電話番号
            r'\b\d{17}\b',                      # クレカ
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # メール
            r'^\d{4}[年\-]\d{1,2}[月\-]\d{1,2}[日]?\s*\d{1,2}:\d{2}',  # 日時+個人情報
        ]
        
        # 著作権疑似パターン（ニュース・書籍フレーズ）
        self.copyright_indicators = [
            r'本記事は.*?の著作物', r'©\s*\d{4}', r'All rights reserved',
            r'特集：', r'社説：', r'記者.*?報道'
        ]
        
        # バイアス検知キーワード（日本語）
        self.bias_keywords = {
            'gender': ['女々しい', '男らしい', 'お母さん', 'パパ'],
            'age': ['ジジイ', 'ババア', '若造'],
            'region': ['田舎者', '都会っ子']
        }
        
        try:
            self.nlp = spacy.load("ja_core_news_sm")
        except:
            self.nlp = None
            print("Warning: spaCy ja_core_news_sm not available")

    def audit_file(self, filepath: str) -> List[ComplianceIssue]:
        """単一ファイルの包括的監査"""
        issues = []
        content = Path(filepath).read_text(encoding='utf-8')
        lines = content.splitlines()
        
        # PIIスキャン
        for i, line in enumerate(lines):
            for pattern in self.pii_patterns:
                if re.search(pattern, line):
                    issues.append(ComplianceIssue(
                        'pii', 'critical', line[:100],
                        [i+1], '個人情報保護法違反：即時削除'
                    ))
        
        # 著作権スキャン
        for i, line in enumerate(lines):
            for indicator in self.copyright_indicators:
                if re.search(indicator, line):
                    issues.append(ComplianceIssue(
                        'copyright', 'high', line[:100],
                        [i+1], '著作権疑義：ライセンス確認要'
                    ))
        
        # バイアス分析
        bias_score = self._calculate_bias(content)
        if bias_score > 0.7:
            issues.append(ComplianceIssue(
                'bias', 'medium', f'Bias score: {bias_score:.2f}',
                list(range(1, len(lines)+1)), '多様性向上：データ拡張要'
            ))
        
        return issues

    def _calculate_bias(self, text: str) -> float:
        """簡易バイアススコア計算"""
        score = 0
        word_count = len(text.split())
        for category, keywords in self.bias_keywords.items():
            for kw in keywords:
                score += text.count(kw) * 10
        return min(score / max(word_count, 1), 1.0)

    def batch_audit(self, data_dir: str = 'data/') -> Dict:
        """全データディレクトリ監査"""
        results = {}
        data_path = Path(data_dir)
        for txt_file in data_path.glob('*.txt'):
            issues = self.audit_file(str(txt_file))
            results[str(txt_file)] = {
                'total_issues': len(issues),
                'critical': len([i for i in issues if i.severity == 'critical']),
                'issues': [i.__dict__ for i in issues]
            }
        return results

    def generate_compliance_report(self, audit_results: Dict) -> str:
        """HTMLコンプライアンスレポート生成"""
        html = """
        <!DOCTYPE html>
        <html><head><title>NeuroQ データコンプライアンス報告 (2026)</title>
        <style>table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}
        .critical{background:#ffebee}.high{background:#fff3e0}</style></head><body>
        """
        total_critical = sum(r['critical'] for r in audit_results.values())
        html += f"<h1>🚨 全{total_critical}件の重大問題検出</h1>"
        
        for filepath, result in audit_results.items():
            html += f"<h2>{filepath}</h2><table>"
            html += "<tr><th>タイプ</th><th>重大度</th><th>証拠</th><th>推奨</th></tr>"
            for issue in result['issues']:
                severity_class = issue['severity']
                html += f"<tr class='{severity_class}'><td>{issue['issue_type']}</td>"
                html += f"<td>{issue['severity']}</td><td>{issue['evidence']}</td>"
                html += f"<td>{issue['recommendation']}</td></tr>"
            html += "</table>"
        html += "</body></html>"
        return html

# 使用例と自動実行
if __name__ == "__main__":
    auditor = NeuroQDataAuditor()
    results = auditor.batch_audit('data/')
    
    # JSONレポート保存
    with open('compliance_audit_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # HTMLレポート生成
    report_html = auditor.generate_compliance_report(results)
    with open('compliance_report_2026.html', 'w', encoding='utf-8') as f:
        f.write(report_html)
    
    print("✅ コンプライアンス監査完了")
    print(f"   重大問題: {sum(r['critical'] for r in results.values())}件")
    print("   レポート: compliance_report_2026.html")
