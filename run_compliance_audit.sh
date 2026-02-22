#!/bin/bash
# NeuroQ 全データコンプライアンス監査実行スクリプト
echo "🚀 NeuroQ 学習データ法的・倫理的監査開始 (2026基準)"

# 依存インストール
pip install spacy pandas
python -m spacy download ja_core_news_sm

# 監査実行
python data_compliance_audit_enhanced.py

# 結果表示
echo "📊 監査結果:"
cat compliance_audit_report.json | jq '.[] | select(.critical > 0) | .total_issues'

# クリーンアップ提案
echo "🧹 問題ファイル自動クリーンアップ:"
grep -l "critical" compliance_audit_report.json | xargs -I {} echo "rm -rf {} # 確認後実行"

echo "✅ 監査完了。compliance_report_2026.htmlを確認してください。"
