#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kunishou/oasst1-89k-ja データセットを User/Assistant 形式に変換

出力形式:
User: こんにちは
Assistant: こんにちは！今日はどうしましたか？

User: 量子コンピュータって何？
Assistant: 量子力学を使って計算するコンピュータで…
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

try:
    from datasets import load_dataset
except ImportError:
    print("datasets がインストールされていません。pip install datasets を実行してください。")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def clean_text(text: str) -> str:
    """テキストクリーニング"""
    if not text or not isinstance(text, str):
        return ""

    # URL除去
    text = re.sub(r'https?://[^\s]+', '', text)

    # 制御文字除去
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # HTML タグ除去
    text = re.sub(r'<[^>]+>', '', text)

    # 余分な空白を整理
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def build_conversation_trees(dataset) -> Dict[str, List[Dict]]:
    """
    メッセージツリーを構築

    oasst1-89k-ja の構造:
    - message_id: メッセージのユニークID
    - parent_id: 親メッセージのID（ルートはnan）
    - role: 'prompter' または 'assistant'
    - text_ja: 日本語テキスト
    - message_tree_id: 会話ツリーのID
    - index: ツリー内のインデックス

    Returns:
        message_tree_id をキーとする、ソートされたメッセージリストの辞書
    """
    print("\n会話ツリーを構築中...")

    # メッセージをツリーIDごとにグループ化
    trees = defaultdict(list)

    for item in tqdm(dataset, desc="   メッセージ収集"):
        message_id = item.get('message_id', '')
        parent_id = item.get('parent_id', '')
        role = item.get('role', '')
        text_ja = item.get('text_ja', '')
        text = item.get('text', '')  # 英語テキスト（text_jaがない場合のフォールバック）
        message_tree_id = item.get('message_tree_id', '')
        index = item.get('index', 0)

        # text_ja を優先、なければ text を使用
        final_text = text_ja if text_ja else text

        if not final_text or not message_tree_id:
            continue

        trees[message_tree_id].append({
            'message_id': message_id,
            'parent_id': parent_id if parent_id and str(parent_id) != 'nan' else None,
            'role': role,
            'text': clean_text(final_text),
            'index': index
        })

    # 各ツリーをインデックスでソート
    for tree_id in trees:
        trees[tree_id].sort(key=lambda x: x.get('index', 0))

    print(f"   {len(trees)} 個の会話ツリーを構築")

    return dict(trees)


def extract_conversations_from_tree(messages: List[Dict]) -> List[str]:
    """
    会話ツリーから User/Assistant 形式の会話を抽出

    Args:
        messages: ソートされたメッセージのリスト

    Returns:
        User/Assistant 形式の会話リスト
    """
    if not messages:
        return []

    conversations = []

    # メッセージIDでインデックス化
    msg_by_id = {m['message_id']: m for m in messages}

    # ルートメッセージ（親がないメッセージ）を見つける
    roots = [m for m in messages if m['parent_id'] is None]

    # 各ルートからの会話パスを追跡
    for root in roots:
        # ルートがprompter（ユーザー）の場合のみ処理
        if root['role'] != 'prompter':
            continue

        # 子メッセージを見つけて会話を構築
        conversation_parts = []
        current = root
        visited = set()

        while current:
            if current['message_id'] in visited:
                break
            visited.add(current['message_id'])

            role = current['role']
            text = current['text']

            if text and len(text) > 5:
                if role == 'prompter':
                    conversation_parts.append(f"User: {text}")
                elif role == 'assistant':
                    conversation_parts.append(f"Assistant: {text}")

            # 子メッセージを探す
            children = [m for m in messages if m['parent_id'] == current['message_id']]

            if children:
                # 最初の子を選択（複数の分岐がある場合）
                current = children[0]
            else:
                current = None

        # User/Assistant ペアがある場合のみ追加
        if len(conversation_parts) >= 2:
            # ペアでグループ化
            pairs = []
            i = 0
            while i < len(conversation_parts):
                if conversation_parts[i].startswith("User:"):
                    user_msg = conversation_parts[i]
                    if i + 1 < len(conversation_parts) and conversation_parts[i + 1].startswith("Assistant:"):
                        assistant_msg = conversation_parts[i + 1]
                        pairs.append(f"{user_msg}\n{assistant_msg}")
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1

            if pairs:
                conversations.append("\n\n".join(pairs))

    return conversations


def convert_oasst1_to_user_assistant(max_samples: int = None) -> List[str]:
    """
    kunishou/oasst1-89k-ja を User/Assistant 形式に変換

    Args:
        max_samples: 最大会話数（Noneの場合は全件）

    Returns:
        変換された会話のリスト
    """
    print("=" * 70)
    print("kunishou/oasst1-89k-ja データセット変換")
    print("=" * 70)

    print("\nデータセットをダウンロード中...")

    try:
        dataset = load_dataset("kunishou/oasst1-89k-ja", split="train")
        print(f"データセット読み込み完了: {len(dataset)} 件のメッセージ")
    except Exception as e:
        print(f"データセット読み込みエラー: {e}")
        return []

    # データセットの構造を確認
    print("\nデータセットの構造:")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"  カラム: {list(sample.keys())}")

    # 会話ツリーを構築
    trees = build_conversation_trees(dataset)

    # 各ツリーから会話を抽出
    print("\n会話を抽出中...")
    all_conversations = []

    for tree_id, messages in tqdm(trees.items(), desc="   会話抽出"):
        convs = extract_conversations_from_tree(messages)
        all_conversations.extend(convs)

        if max_samples and len(all_conversations) >= max_samples:
            all_conversations = all_conversations[:max_samples]
            break

    print(f"\n変換完了: {len(all_conversations)} 件の会話")

    return all_conversations


def save_conversations(conversations: List[str], output_path: str):
    """
    会話をファイルに保存
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nファイルに保存中: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(conv + "\n\n")

    file_size = os.path.getsize(output_path)
    print(f"保存完了!")
    print(f"  ファイルサイズ: {file_size:,} バイト ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  会話数: {len(conversations):,}")

    # 統計情報を保存
    total_chars = sum(len(c) for c in conversations)
    stats = {
        'total_conversations': len(conversations),
        'total_characters': total_chars,
        'avg_conversation_length': total_chars / len(conversations) if conversations else 0,
        'file_size_bytes': file_size,
        'format': 'User/Assistant',
        'source': 'kunishou/oasst1-89k-ja'
    }

    stats_path = output_path.replace('.txt', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"  統計情報: {stats_path}")

    # サンプル表示
    print("\n" + "=" * 70)
    print("サンプル会話")
    print("=" * 70)
    for i, conv in enumerate(conversations[:3]):
        print(f"\n--- 例 {i+1} ---")
        # 長い場合は省略
        if len(conv) > 800:
            print(conv[:800] + "\n...")
        else:
            print(conv)
    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='kunishou/oasst1-89k-ja を User/Assistant 形式に変換'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/oasst1_ja_conversations.txt',
        help='出力ファイルパス'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大サンプル数（省略時は全件）'
    )

    args = parser.parse_args()

    conversations = convert_oasst1_to_user_assistant(args.max_samples)

    if conversations:
        save_conversations(conversations, args.output)
    else:
        print("変換された会話がありません。")


if __name__ == '__main__':
    main()
