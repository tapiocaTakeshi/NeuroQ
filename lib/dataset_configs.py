#!/usr/bin/env python3
"""
NeuroQ データセット設定ファイル（neuroq-runpod用）

HuggingFace の load_dataset(id, config) パターンに対応。
各データセットは id（リポジトリパス）と config（サブセット名）で識別される。

使用例:
    from dataset_configs import get_dataset_config, AVAILABLE_DATASETS, load_dataset_texts

    # 設定を取得（内部キーで）
    config = get_dataset_config('oasst1_ja')

    # HuggingFace形式: load_dataset(id, config)
    # config['id']     -> 'kunishou/oasst1-89k-ja'
    # config['config'] -> None

    # データセット一覧（{id, config} のリスト）
    from dataset_configs import get_datasets_list
    datasets = get_datasets_list()
    # [{"id": "kunishou/oasst1-89k-ja", "config": null, ...}, ...]

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
    # HuggingFace load_dataset() の引数に対応
    'id': 'kunishou/oasst1-89k-ja',
    'config': None,
    # メタデータ
    'name': 'OASST1 Japanese',
    'description': 'kunishou/oasst1-89k-ja 日本語会話データセット（User/Assistant形式）',
    'language': 'ja',
    'format': 'conversation',  # conversation | plain | mixed
    # ローカルデータファイル（存在すればHFダウンロードより優先）
    'data_files': [
        os.path.join(_BASE_DIR, 'data', 'oasst1_ja_conversations.txt'),
        os.path.join(_RUNPOD_DIR, 'data', 'oasst1_ja_conversations.txt'),
    ],
    'tokenizer_files': [
        os.path.join(_RUNPOD_DIR, 'neuroq_tokenizer_oasst1_ja.model'),
        os.path.join(_BASE_DIR, 'neuroq_tokenizer_oasst1_ja.model'),
    ],
    'requires_tokenizer': True,
    # HuggingFaceからロードする際のテキストフィールド名
    'hf_text_field': 'text',
    'hf_max_samples': None,
    # デフォルトトレーニングパラメータ
    'default_params': {
        'epochs': 5,
        'batch_size': 8,
        'lr': 0.0003,
        'seq_length': 128,
    },
    # ローカルファイルのパース設定
    'parse_config': {
        'block_separator': '\n\n',
        'required_markers': ['User:', 'Assistant:'],
    },
}

DATASET_OASST1_JA_CLEANED = {
    'id': 'kunishou/oasst1-89k-ja',
    'config': 'cleaned',
    'name': 'OASST1 Japanese (Cleaned)',
    'description': 'クリーニング済み日本語会話データセット',
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
    'hf_text_field': 'text',
    'hf_max_samples': None,
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
    'id': None,  # ローカル専用データセット
    'config': None,
    'name': 'General Training Data',
    'description': '汎用トレーニングデータ（3.6MB）',
    'language': 'ja',
    'format': 'plain',
    'data_files': [
        os.path.join(_BASE_DIR, 'data', 'training_data.txt'),
        os.path.join(_RUNPOD_DIR, 'data', 'training_data.txt'),
    ],
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'hf_text_field': None,
    'hf_max_samples': None,
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
    'id': None,
    'config': None,
    'name': 'Combined Clean Data',
    'description': '結合・クリーニング済みデータセット（16MB）',
    'language': 'ja',
    'format': 'plain',
    'data_files': [
        os.path.join(_BASE_DIR, 'data', 'combined_clean_data.txt'),
        os.path.join(_RUNPOD_DIR, 'data', 'combined_clean_data.txt'),
    ],
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'hf_text_field': None,
    'hf_max_samples': None,
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
    'id': None,
    'config': None,
    'name': 'High Quality Conversations',
    'description': 'キュレーション済み高品質会話データ',
    'language': 'ja',
    'format': 'conversation',
    'data_files': [
        os.path.join(_BASE_DIR, 'data', 'high_quality_conversations.txt'),
        os.path.join(_RUNPOD_DIR, 'data', 'high_quality_conversations.txt'),
    ],
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'hf_text_field': None,
    'hf_max_samples': None,
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
    'id': None,
    'config': None,
    'name': 'Japanese Training Corpus',
    'description': '日本語トレーニングコーパス',
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
    'hf_text_field': None,
    'hf_max_samples': None,
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

DATASET_SWALLOW_NEMOTRON = {
    'id': 'tokyotech-llm/Swallow-Nemotron-Post-Training-Dataset-v1',
    'config': 'Nemotron-Post-Training-Dataset-v1',
    'name': 'Swallow Nemotron Post-Training',
    'description': 'Tokyo Tech Swallow Nemotron ポストトレーニングデータセット',
    'language': 'ja',
    'format': 'mixed',
    'data_files': [],  # HuggingFaceからダウンロード
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'hf_text_field': 'text',
    'hf_max_samples': 10000,
    'default_params': {
        'epochs': 3,
        'batch_size': 4,
        'lr': 0.0002,
        'seq_length': 256,
    },
    'parse_config': {
        'block_separator': '\n\n',
        'required_markers': [],
    },
}

DATASET_SWALLOW_CODE_V2 = {
    'id': 'tokyotech-llm/swallow-code-v2',
    'config': 'stage1-auto-format',
    'name': 'Swallow Code v2',
    'description': 'Tokyo Tech Swallow コードデータセット v2 (stage1-auto-format)',
    'language': 'multi',
    'format': 'plain',
    'data_files': [],
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'hf_text_field': 'text',
    'hf_max_samples': 10000,
    'default_params': {
        'epochs': 3,
        'batch_size': 4,
        'lr': 0.0002,
        'seq_length': 256,
    },
    'parse_config': {
        'block_separator': '\n\n',
        'required_markers': [],
    },
}

DATASET_SWALLOW_MATH_V2 = {
    'id': 'tokyotech-llm/swallow-math-v2',
    'config': 'swallow-math-v2-qa',
    'name': 'Swallow Math v2',
    'description': 'Tokyo Tech Swallow 数学データセット v2 (QA形式)',
    'language': 'ja',
    'format': 'mixed',
    'data_files': [],
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'hf_text_field': 'text',
    'hf_max_samples': 10000,
    'default_params': {
        'epochs': 3,
        'batch_size': 4,
        'lr': 0.0002,
        'seq_length': 256,
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
    'swallow_nemotron': DATASET_SWALLOW_NEMOTRON,
    'swallow_code_v2': DATASET_SWALLOW_CODE_V2,
    'swallow_math_v2': DATASET_SWALLOW_MATH_V2,
}


# ===============================
# ユーティリティ関数
# ===============================

def get_dataset_config(dataset_id: str) -> dict:
    """
    データセットIDに応じた設定を取得

    Args:
        dataset_id: データセットの内部キー（例: 'oasst1_ja', 'swallow_nemotron'）

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


def get_datasets_list() -> list:
    """
    利用可能なデータセットを {id, config, ...} のリスト形式で返す

    id と config は HuggingFace の load_dataset(id, config) に対応:
    - id: データセットリポジトリパス（例: 'kunishou/oasst1-89k-ja'）
    - config: サブセット/コンフィグ名（例: 'Nemotron-Post-Training-Dataset-v1'）

    Returns:
        [{"id": "kunishou/oasst1-89k-ja", "config": null, ...}, ...] 形式のリスト
    """
    result = []
    for key, ds in AVAILABLE_DATASETS.items():
        result.append({
            "id": ds["id"],
            "config": ds["config"],
            "key": key,
            "name": ds["name"],
            "description": ds["description"],
            "language": ds["language"],
            "format": ds["format"],
            "default_params": ds["default_params"],
        })
    return result


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


def _is_japanese_text(text: str, threshold: float = 0.3) -> bool:
    """
    テキストが日本語かどうかを判定する

    日本語文字（ひらがな、カタカナ、漢字）の割合が threshold 以上なら日本語と判定。
    小型モデルでは多言語混在を避けるため、日本語のみにフィルタリングすることが重要。

    Args:
        text: 判定対象のテキスト
        threshold: 日本語文字の最低割合（デフォルト: 0.3 = 30%）

    Returns:
        日本語テキストならTrue
    """
    if not text:
        return False

    japanese_count = 0
    total_chars = 0

    for char in text:
        if char.isspace():
            continue
        total_chars += 1
        # ひらがな: U+3040-U+309F
        # カタカナ: U+30A0-U+30FF
        # CJK統合漢字: U+4E00-U+9FFF
        # CJK統合漢字拡張A: U+3400-U+4DBF
        cp = ord(char)
        if (0x3040 <= cp <= 0x309F or  # ひらがな
            0x30A0 <= cp <= 0x30FF or  # カタカナ
            0x4E00 <= cp <= 0x9FFF or  # CJK統合漢字
            0x3400 <= cp <= 0x4DBF):   # CJK統合漢字拡張A
            japanese_count += 1

    if total_chars == 0:
        return False

    return (japanese_count / total_chars) >= threshold


def _load_from_huggingface(dataset_config: dict) -> list:
    """
    HuggingFace Hub からデータセットをダウンロードしてテキストリストを返す

    load_dataset(id, config) パターンで読み込む。
    言語設定が 'ja' の場合、日本語テキストのみをフィルタリングする。

    Args:
        dataset_config: データセット設定辞書

    Returns:
        テキストのリスト
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets ライブラリが必要です。"
            " pip install datasets でインストールしてください。"
        )

    hf_id = dataset_config['id']
    hf_config = dataset_config['config']
    text_field = dataset_config.get('hf_text_field', 'text')
    max_samples = dataset_config.get('hf_max_samples')
    target_language = dataset_config.get('language', 'ja')

    print(f"🌐 HuggingFace からデータセットをロード中...")
    print(f"   load_dataset('{hf_id}', '{hf_config}')" if hf_config
          else f"   load_dataset('{hf_id}')")

    if hf_config:
        ds = load_dataset(hf_id, hf_config)
    else:
        ds = load_dataset(hf_id)

    # split を自動選択（train > validation > test）
    if hasattr(ds, 'keys'):
        for split_name in ['train', 'validation', 'test']:
            if split_name in ds:
                ds = ds[split_name]
                print(f"   split: '{split_name}'")
                break

    # テキストフィールドを抽出
    texts = []
    filtered_count = 0
    for i, item in enumerate(ds):
        if max_samples and len(texts) >= max_samples:
            break

        # lang フィールドがある場合はそれでフィルタリング
        if 'lang' in item and target_language == 'ja':
            if item['lang'] != 'ja':
                filtered_count += 1
                continue

        if text_field and text_field in item:
            text = item[text_field]
        elif 'text' in item:
            text = item['text']
        elif 'content' in item:
            text = item['content']
        elif 'instruction' in item:
            # instruction + output 形式
            text = item['instruction']
            if 'output' in item and item['output']:
                text = f"User: {item['instruction']}\nAssistant: {item['output']}"
        else:
            # 最初の文字列フィールドを使用
            for v in item.values():
                if isinstance(v, str) and len(v) > 10:
                    text = v
                    break
            else:
                continue

        if isinstance(text, str) and text.strip():
            # 日本語データセットの場合、テキスト内容でも日本語判定を行う
            if target_language == 'ja' and 'lang' not in item:
                if not _is_japanese_text(text):
                    filtered_count += 1
                    continue
            texts.append(text.strip())

    print(f"   {len(texts)} 個のテキストを読み込みました")
    if filtered_count > 0:
        print(f"   ⚠️ {filtered_count} 個の非日本語テキストを除外しました")
    return texts


def load_dataset_texts(dataset_id: str) -> list:
    """
    データセットIDからテキストデータを読み込む

    ローカルファイルが存在すればそちらを優先。
    存在しなければ HuggingFace Hub からダウンロード。

    Args:
        dataset_id: データセットの内部キー

    Returns:
        テキストのリスト
    """
    config = get_dataset_config(dataset_id)

    # 1. ローカルファイルを探す
    try:
        data_file = find_data_file(config)
        print(f"📖 ローカルデータ読み込み: {data_file}")

        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()

        parse_config = config.get('parse_config', {})
        separator = parse_config.get('block_separator', '\n\n')
        required_markers = parse_config.get('required_markers', [])

        blocks = content.split(separator)

        texts = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if required_markers:
                if all(marker in block for marker in required_markers):
                    texts.append(block)
            else:
                texts.append(block)

        print(f"   {len(texts)} 個のテキストブロックを読み込みました")
        return texts

    except FileNotFoundError:
        pass

    # 2. HuggingFace Hub からダウンロード
    if config['id'] is not None:
        return _load_from_huggingface(config)

    # 3. どちらも利用不可
    raise FileNotFoundError(
        f"データセット '{dataset_id}' のデータが見つかりません。"
        f" ローカルファイルもHuggingFace IDも設定されていません。"
    )


def get_training_params(dataset_id: str, overrides: dict = None) -> dict:
    """
    データセットIDに応じたデフォルトトレーニングパラメータを取得
    overridesで上書き可能

    Args:
        dataset_id: データセットの内部キー
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

    for key, config in AVAILABLE_DATASETS.items():
        hf_id = config['id']
        hf_config = config['config']

        print(f"\n📦 {key}")
        if hf_id:
            id_str = f"load_dataset('{hf_id}', '{hf_config}')" if hf_config else f"load_dataset('{hf_id}')"
            print(f"   HuggingFace: {id_str}")
        else:
            print(f"   ソース: ローカルファイル")
        print(f"   名前: {config['name']}")
        print(f"   説明: {config['description']}")
        print(f"   言語: {config['language']}, 形式: {config['format']}")

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
            if hf_id:
                print(f"   データ: HuggingFace Hubからダウンロード")
            else:
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
