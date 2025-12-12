#!/usr/bin/env python3
"""
Handler.py のテストスクリプト
"""

import sys
import os

# handler.pyをインポート（runpod.serverless.startを実行させないため）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# runpod.serverless.startをモックして実行を防ぐ
import runpod
runpod.serverless.start = lambda x: None

from handler import handler

def test_pretrain_openai():
    """pretrain_openai アクションのテスト"""
    print("=" * 60)
    print("🧪 Testing pretrain_openai action")
    print("=" * 60)

    # pretrain_openai を実行
    job = {
        "input": {
            "action": "pretrain_openai"
        }
    }

    result = handler(job)
    print("\n📋 Result:")
    print(result)

    if result.get("status") == "success":
        print("\n✅ Pretraining started successfully!")
        print(f"   PID: {result.get('pid')}")
        print(f"   Log file: {result.get('log_file')}")

        # ステータス確認
        import time
        time.sleep(5)

        print("\n📊 Checking status after 5 seconds...")
        status_job = {
            "input": {
                "action": "pretrain_status"
            }
        }

        status_result = handler(status_job)
        print("\n📋 Status Result:")
        print(f"   Status: {status_result.get('pretrain_status')}")
        print(f"   Process running: {status_result.get('process_running')}")
        print(f"\n📝 Log tail:\n{status_result.get('log_tail')}")

    else:
        print(f"\n❌ Error: {result.get('error')}")

def test_available_actions():
    """利用可能なアクションのテスト"""
    print("\n" + "=" * 60)
    print("🧪 Testing available actions")
    print("=" * 60)

    job = {
        "input": {
            "action": "unknown_action"
        }
    }

    result = handler(job)
    print("\n📋 Available Actions:")
    for action in result.get("available_actions", []):
        print(f"   - {action}")

if __name__ == "__main__":
    # テスト実行
    test_available_actions()
    print("\n")
    test_pretrain_openai()
