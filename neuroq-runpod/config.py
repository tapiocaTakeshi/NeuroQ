"""
NeuroQ RunPod 設定ファイル
========================
環境変数を .env ファイルから読み込みます
"""

from dotenv import load_dotenv
import os

# .envファイルを読み込み
load_dotenv()

# RunPod API設定
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

# RunPod URL (自動生成)
RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run" if RUNPOD_ENDPOINT_ID else None
RUNPOD_STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status" if RUNPOD_ENDPOINT_ID else None

# OpenAI API設定 (オプション)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# リクエストヘッダー
def get_headers():
    """認証ヘッダーを取得"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }

# 設定確認
def check_config():
    """設定が正しいか確認"""
    missing = []
    if not RUNPOD_API_KEY:
        missing.append("RUNPOD_API_KEY")
    if not RUNPOD_ENDPOINT_ID:
        missing.append("RUNPOD_ENDPOINT_ID")
    
    if missing:
        print(f"⚠️ 未設定の環境変数: {', '.join(missing)}")
        print("   .envファイルを確認してください")
        return False
    
    print("✅ 設定OK")
    print(f"   Endpoint: {RUNPOD_ENDPOINT_ID}")
    return True

if __name__ == "__main__":
    check_config()
