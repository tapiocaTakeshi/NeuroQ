#!/usr/bin/env python3
"""
会話修正のテスト
==================
handler.py の修正をローカルでテスト
"""

import sys
from pathlib import Path

# パスを追加
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# handler から必要な関数をインポート
from handler import initialize_model, generate_text, conversation_sessions, build_conversation_prompt

def test_conversation():
    """会話機能のテスト"""
    print("=" * 60)
    print("🧪 会話修正テスト")
    print("=" * 60)

    # モデル初期化
    print("\n📦 モデル初期化中...")
    if not initialize_model():
        print("❌ モデル初期化失敗")
        return

    print("✅ モデル初期化完了\n")

    # テストケース
    test_cases = [
        "こんにちは",
        "人工知能とは？",
        "ありがとう",
        "量子コンピュータについて教えて",
        "わかりました"
    ]

    session_id = "test_session"

    print("\n" + "=" * 60)
    print("💬 会話テスト開始")
    print("=" * 60)

    for i, user_input in enumerate(test_cases, 1):
        print(f"\n--- ターン {i} ---")
        print(f"👤 USER: {user_input}")

        # プロンプト構築を確認
        prompt = build_conversation_prompt(session_id, user_input)
        print(f"\n📝 構築されたプロンプト:\n{prompt}")
        print(f"\n{'─' * 60}")

        # 生成
        response = generate_text(
            prompt=user_input,
            max_length=50,  # 短く制限
            temp_min=0.4,
            temp_max=0.7,
            session_id=session_id
        )

        print(f"🤖 ASSISTANT: {response}")
        print(f"\n📊 生成長さ: {len(response)} 文字")

        # 暴走チェック
        if len(response) > 200:
            print("⚠️  警告: 応答が長すぎます（暴走の可能性）")
        if "<USER>" in response:
            print("⚠️  警告: <USER> トークンが含まれています（stop失敗）")

        # 会話履歴確認
        if session_id in conversation_sessions:
            history_len = len(conversation_sessions[session_id])
            print(f"💾 会話履歴: {history_len // 2} ターン")

    print("\n" + "=" * 60)
    print("✅ テスト完了")
    print("=" * 60)

    # 最終的な会話履歴を表示
    if session_id in conversation_sessions:
        print("\n📚 最終的な会話履歴:")
        for msg in conversation_sessions[session_id]:
            role = "👤 USER" if msg["role"] == "user" else "🤖 ASSISTANT"
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            print(f"  {role}: {content}")


def test_single_response():
    """単発応答のテスト（暴走チェック）"""
    print("\n" + "=" * 60)
    print("🔬 単発応答テスト（暴走チェック）")
    print("=" * 60)

    if not initialize_model():
        print("❌ モデル初期化失敗")
        return

    test_prompts = [
        ("挨拶", "こんにちは"),
        ("質問", "人工知能とは？"),
        ("短い質問", "元気？"),
    ]

    for label, prompt in test_prompts:
        print(f"\n--- {label} ---")
        print(f"👤 入力: {prompt}")

        response = generate_text(
            prompt=prompt,
            max_length=50,
            temp_min=0.4,
            temp_max=0.7,
            session_id="single_test"
        )

        print(f"🤖 出力: {response}")
        print(f"📊 長さ: {len(response)} 文字")

        # 問題チェック
        issues = []
        if len(response) > 200:
            issues.append("長すぎる（暴走の可能性）")
        if "<USER>" in response or "<ASSISTANT>" in response:
            issues.append("トークンが残っている")
        if not response.strip():
            issues.append("空の応答")

        if issues:
            print(f"⚠️  問題: {', '.join(issues)}")
        else:
            print("✅ 正常")


if __name__ == "__main__":
    print("\n🚀 会話修正テスト開始\n")

    # 単発テスト
    test_single_response()

    # 会話テスト
    test_conversation()

    print("\n" + "=" * 60)
    print("🎉 すべてのテスト完了")
    print("=" * 60)
