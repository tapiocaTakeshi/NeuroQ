#!/usr/bin/env python3
"""
NeuroQ データセット設定ファイル（neuroq-runpod用）

dataset_id でトレーニングデータセットの設定を管理する。
model_configs.py と同様のパターンで、データセットごとの
デフォルトパラメータ、ファイルパス、トークナイザー設定を定義。

使用例:
    from dataset_configs import get_dataset_config, AVAILABLE_DATASETS, load_dataset_texts

    # 設定を取得
    config = get_dataset_config('oasst1_ja')

    # データセット一覧
    print(AVAILABLE_DATASETS.keys())

    # テキストデータを読み込み
    texts = load_dataset_texts('oasst1_ja')
"""

import os
from pathlib import Path

# ベースディレクトリ（neuroq-runpod/ の親ディレクトリ）
_BASE_DIR = str(Path(__file__).parent.parent)
_RUNPOD_DIR = str(Path(__file__).parent)


# ===============================
# データセット設定
# ===============================

DATASET_OASST1_JA = {
    'name': 'OASST1 Japanese',
    'description': 'kunishou/oasst1-89k-ja 日本語会話データセット（User/Assistant形式）',
    'source': 'kunishou/oasst1-89k-ja',
    'language': 'ja',
    'format': 'conversation',  # conversation | plain | mixed
    'data_files': [
        os.path.join(_BASE_DIR, 'data', 'oasst1_ja_conversations.txt'),
        os.path.join(_RUNPOD_DIR, 'data', 'oasst1_ja_conversations.txt'),
    ],
    'tokenizer_files': [
        os.path.join(_RUNPOD_DIR, 'neuroq_tokenizer_oasst1_ja.model'),
        os.path.join(_BASE_DIR, 'neuroq_tokenizer_oasst1_ja.model'),
    ],
    'requires_tokenizer': True,
    # デフォルトトレーニングパラメータ
    'default_params': {
        'epochs': 5,
        'batch_size': 8,
        'lr': 0.0003,
        'seq_length': 128,
    },
    # 会話パース設定
    'parse_config': {
        'block_separator': '\n\n',
        'required_markers': ['User:', 'Assistant:'],
    },
}

DATASET_OASST1_JA_CLEANED = {
    'name': 'OASST1 Japanese (Cleaned)',
    'description': 'クリーニング済み日本語会話データセット',
    'source': 'kunishou/oasst1-89k-ja (cleaned)',
    'language': 'ja',
    'format': 'plain',
    'data_files': [
        os.path.join(_BASE_DIR, 'data', 'oasst1_ja_cleaned.txt'),
        os.path.join(_RUNPOD_DIR, 'data', 'oasst1_ja_cleaned.txt'),
    ],
    'tokenizer_files': [
        os.path.join(_RUNPOD_DIR, 'neuroq_tokenizer_oasst1_ja.model'),
        os.path.join(_BASE_DIR, 'neuroq_tokenizer_oasst1_ja.model'),
    ],
    'requires_tokenizer': True,
    'default_params': {
        'epochs': 5,
        'batch_size': 8,
        'lr': 0.0003,
        'seq_length': 128,
    },
    'parse_config': {
        'block_separator': '\n\n',
        'required_markers': [],
    },
}

DATASET_TRAINING_DATA = {
    'name': 'General Training Data',
    'description': '汎用トレーニングデータ（3.6MB）',
    'source': 'local',
    'language': 'ja',
    'format': 'plain',
    'data_files': [
        os.path.join(_BASE_DIR, 'data', 'training_data.txt'),
        os.path.join(_RUNPOD_DIR, 'data', 'training_data.txt'),
    ],
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'default_params': {
        'epochs': 10,
        'batch_size': 8,
        'lr': 0.0005,
        'seq_length': 64,
    },
    'parse_config': {
        'block_separator': '\n\n',
        'required_markers': [],
    },
}

DATASET_COMBINED_CLEAN = {
    'name': 'Combined Clean Data',
    'description': '結合・クリーニング済みデータセット（16MB）',
    'source': 'local',
    'language': 'ja',
    'format': 'plain',
    'data_files': [
        os.path.join(_BASE_DIR, 'data', 'combined_clean_data.txt'),
        os.path.join(_RUNPOD_DIR, 'data', 'combined_clean_data.txt'),
    ],
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'default_params': {
        'epochs': 5,
        'batch_size': 8,
        'lr': 0.0003,
        'seq_length': 128,
    },
    'parse_config': {
        'block_separator': '\n\n',
        'required_markers': [],
    },
}

DATASET_HIGH_QUALITY = {
    'name': 'High Quality Conversations',
    'description': 'キュレーション済み高品質会話データ',
    'source': 'local',
    'language': 'ja',
    'format': 'conversation',
    'data_files': [
        os.path.join(_BASE_DIR, 'data', 'high_quality_conversations.txt'),
        os.path.join(_RUNPOD_DIR, 'data', 'high_quality_conversations.txt'),
    ],
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'default_params': {
        'epochs': 10,
        'batch_size': 4,
        'lr': 0.0003,
        'seq_length': 128,
    },
    'parse_config': {
        'block_separator': '\n\n',
        'required_markers': ['User:', 'Assistant:'],
    },
}

DATASET_JAPANESE_CORPUS = {
    'name': 'Japanese Training Corpus',
    'description': '日本語トレーニングコーパス',
    'source': 'local',
    'language': 'ja',
    'format': 'plain',
    'data_files': [
        os.path.join(_BASE_DIR, 'data', 'japanese_training_corpus.txt'),
        os.path.join(_RUNPOD_DIR, 'data', 'japanese_training_corpus.txt'),
    ],
    'tokenizer_files': [
        os.path.join(_RUNPOD_DIR, 'neuroq_tokenizer_oasst1_ja.model'),
        os.path.join(_BASE_DIR, 'neuroq_tokenizer_oasst1_ja.model'),
    ],
    'requires_tokenizer': False,
    'default_params': {
        'epochs': 10,
        'batch_size': 8,
        'lr': 0.0005,
        'seq_length': 64,
    },
    'parse_config': {
        'block_separator': '\n\n',
        'required_markers': [],
    },
}


# ===============================
# 利用可能なデータセット一覧
# ===============================
AVAILABLE_DATASETS = {
    'oasst1_ja': DATASET_OASST1_JA,
    'oasst1_ja_cleaned': DATASET_OASST1_JA_CLEANED,
    'training_data': DATASET_TRAINING_DATA,
    'combined_clean': DATASET_COMBINED_CLEAN,
    'high_quality': DATASET_HIGH_QUALITY,
    'japanese_corpus': DATASET_JAPANESE_CORPUS,
}


# ===============================
# ユーティリティ関数
# ===============================

def get_dataset_config(dataset_id: str) -> dict:
    """
    データセットIDに応じた設定を取得

    Args:
        dataset_id: データセットID

    Returns:
        データセット設定辞書

    Raises:
        ValueError: 不明なdataset_idが指定された場合
    """
    dataset_id = dataset_id.lower()
    if dataset_id not in AVAILABLE_DATASETS:
        available = ', '.join(AVAILABLE_DATASETS.keys())
        raise ValueError(
            f"不明なデータセットID: '{dataset_id}'. "
            f"利用可能なデータセット: {available}"
        )
    return AVAILABLE_DATASETS[dataset_id]


def find_data_file(dataset_config: dict) -> str:
    """
    データセット設定からデータファイルのパスを探す

    Args:
        dataset_config: データセット設定辞書

    Returns:
        見つかったデータファイルのパス

    Raises:
        FileNotFoundError: データファイルが見つからない場合
    """
    for path in dataset_config['data_files']:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"データファイルが見つかりません: {dataset_config['name']}. "
        f"検索パス: {dataset_config['data_files']}"
    )


def find_tokenizer_file(dataset_config: dict) -> str:
    """
    データセット設定からトークナイザーファイルのパスを探す

    Args:
        dataset_config: データセット設定辞書

    Returns:
        見つかったトークナイザーファイルのパス、または None

    Raises:
        FileNotFoundError: requires_tokenizer=True なのにファイルが見つからない場合
    """
    for path in dataset_config.get('tokenizer_files', []):
        if os.path.exists(path):
            return path

    if dataset_config.get('requires_tokenizer', False):
        raise FileNotFoundError(
            f"トークナイザーファイルが見つかりません: {dataset_config['name']}. "
            f"検索パス: {dataset_config.get('tokenizer_files', [])}"
        )

    return None


def load_dataset_texts(dataset_id: str) -> list:
    """
    データセットIDからテキストデータを読み込む

    Args:
        dataset_id: データセットID

    Returns:
        テキストのリスト
    """
    config = get_dataset_config(dataset_id)
    data_file = find_data_file(config)

    print(f"📖 データ読み込み: {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()

    parse_config = config.get('parse_config', {})
    separator = parse_config.get('block_separator', '\n\n')
    required_markers = parse_config.get('required_markers', [])

    # ブロック分割
    blocks = content.split(separator)

    texts = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 必須マーカーチェック
        if required_markers:
            if all(marker in block for marker in required_markers):
                texts.append(block)
        else:
            texts.append(block)

    print(f"   {len(texts)} 個のテキストブロックを読み込みました")
    return texts


def get_training_params(dataset_id: str, overrides: dict = None) -> dict:
    """
    データセットIDに応じたデフォルトトレーニングパラメータを取得
    overridesで上書き可能

    Args:
        dataset_id: データセットID
        overrides: 上書きパラメータ辞書（Noneの値はスキップ）

    Returns:
        トレーニングパラメータ辞書
    """
    config = get_dataset_config(dataset_id)
    params = config['default_params'].copy()

    if overrides:
        for key, value in overrides.items():
            if value is not None and key in params:
                params[key] = value

    return params


def print_available_datasets():
    """利用可能なデータセット一覧を表示"""
    print("=" * 60)
    print("📚 NeuroQ 利用可能なデータセット")
    print("=" * 60)

    for dataset_id, config in AVAILABLE_DATASETS.items():
        print(f"\n📦 {dataset_id}")
        print(f"   名前: {config['name']}")
        print(f"   説明: {config['description']}")
        print(f"   言語: {config['language']}")
        print(f"   形式: {config['format']}")

        # ファイル存在チェック
        try:
            data_file = find_data_file(config)
            file_size = os.path.getsize(data_file)
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f}MB"
            else:
                size_str = f"{file_size / 1024:.1f}KB"
            print(f"   データ: {size_str} ({data_file})")
        except FileNotFoundError:
            print(f"   データ: ❌ ファイルが見つかりません")

        # トークナイザーチェック
        try:
            tok_file = find_tokenizer_file(config)
            if tok_file:
                print(f"   トークナイザー: {tok_file}")
            else:
                print(f"   トークナイザー: デフォルト使用")
        except FileNotFoundError:
            print(f"   トークナイザー: ❌ 必須ファイルが見つかりません")

        # デフォルトパラメータ
        params = config['default_params']
        print(f"   デフォルト: epochs={params['epochs']}, "
              f"batch_size={params['batch_size']}, "
              f"lr={params['lr']}, "
              f"seq_length={params['seq_length']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_available_datasets()
