"""
NeuroQ 法的コンプライアンス対応トレーニングラッパー
全トレーニングスクリプトを自動保護
"""
import os
import sys
from pathlib import Path

def legal_compliance_check():
    """トレーニング前の法的チェック"""
    from compliance_audit import NeuroQComplianceAuditor
    
    auditor = NeuroQComplianceAuditor()
    report = auditor.audit_all()
    
    high_risk = [f for f, risks in report.items() if any('高リスク' in r for r in risks.values())]
    
    if high_risk:
        print("⚠️  法的リスク検出 - クリーン化必須:")
        for file in high_risk[:5]:
            print(f"   {file}")
        print("トレーニング中止。data_compliance_filter.pyを実行してください。")
        sys.exit(1)
    print("✅ 法的コンプライアンス確認済み")

def main():
    legal_compliance_check()
    # 元のトレーニング実行
    os.system("python train_conversation.py")

if __name__ == "__main__":
    main()
