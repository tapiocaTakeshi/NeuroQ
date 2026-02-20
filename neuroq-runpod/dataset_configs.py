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
    'description': 'Tokyo Tech Swallow Nemotron ポストトレーニング会話データセット'
             '（code/math/stem、~5.14M rows）',
    'language': 'ja',
    'format': 'conversation',
    'data_files': [],  # HuggingFaceからダウンロード
    'tokenizer_files': [],
    'requires_tokenizer': False,
    # HuggingFace設定
    'hf_text_field': 'conversation',  # conversation形式（list of {role, content}）
    'hf_max_samples': 5000,
    'hf_split': None,  # None = 最初に見つかったsplitを使用（code/math/stem）
    'hf_requires_auth': False,
    'hf_streaming': True,  # 大規模データセットはストリーミングで読み込み
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
    'description': 'Tokyo Tech Swallow Pythonコードデータセット'
             '（auto-formatted、~54.8M rows）',
    'language': 'en',
    'format': 'code',
    'data_files': [],
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'hf_text_field': 'text',
    'hf_max_samples': 5000,
    'hf_split': 'train',
    'hf_requires_auth': True,
    'hf_streaming': True,
    'default_params': {
        'epochs': 3,
        'batch_size': 4,
        'lr': 0.0002,
        'seq_length': 512,
    },
    'parse_config': {
        'block_separator': '\n\n',
        'required_markers': [],
    },
}

DATASET_SWALLOW_MATH_V2 = {
    'id': 'tokyotech-llm/swallow-math-v2',
    'config': 'swallow-math-v2-qa',
    'name': 'Swallow Math v2 QA',
    'description': 'Tokyo Tech Swallow 数学QAデータセット'
             '（Question/Answer/Code形式、~8.47M rows）',
    'language': 'en',
    'format': 'qa',
    'data_files': [],
    'tokenizer_files': [],
    'requires_tokenizer': False,
    'hf_text_field': 'text',
    'hf_max_samples': 5000,
    'hf_split': 'train',
    'hf_requires_auth': True,
    'hf_streaming': True,
    'default_params': {
        'epochs': 3,
        'batch_size': 4,
        'lr': 0.0002,
        'seq_length': 512,
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
    'swallow_code': DATASET_SWALLOW_CODE_V2,
    'swallow_math': DATASET_SWALLOW_MATH_V2,
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


def _load_from_huggingface(dataset_config: dict) -> list:
    """
    HuggingFace Hub からデータセットをダウンロードしてテキストリストを返す

    load_dataset(id, config) パターンで読み込む。
    認証が必要なデータセットは HF_TOKEN 環境変数またはデータセット設定の
    hf_requires_auth フラグで対応する。

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
    requires_auth = dataset_config.get('hf_requires_auth', False)

    # HuggingFace認証トークン
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if requires_auth and not hf_token:
        raise EnvironmentError(
            f"データセット '{hf_id}' はログインが必要です。"
            " HF_TOKEN 環境変数を設定するか、"
            " huggingface-cli login を実行してください。"
        )

    print(f"🌐 HuggingFace からデータセットをロード中...")
    print(f"   load_dataset('{hf_id}', '{hf_config}')" if hf_config
          else f"   load_dataset('{hf_id}')")
    if requires_auth:
        print(f"   🔑 認証トークン使用")

    load_kwargs = {}
    if hf_token:
        load_kwargs['token'] = hf_token

    # ストリーミングモード（大規模データセット対応）
    use_streaming = dataset_config.get('hf_streaming', False)
    if use_streaming:
        load_kwargs['streaming'] = True
        print(f"   📡 ストリーミングモード")

    if hf_config:
        ds = load_dataset(hf_id, hf_config, **load_kwargs)
    else:
        ds = load_dataset(hf_id, **load_kwargs)

    # split を選択
    hf_split = dataset_config.get('hf_split')
    if hasattr(ds, 'keys'):
        if hf_split and hf_split in ds:
            ds = ds[hf_split]
            print(f"   split: '{hf_split}'")
        else:
            # 自動選択（train > 最初に見つかったsplit）
            for split_name in ['train', 'validation', 'test']:
                if split_name in ds:
                    ds = ds[split_name]
                    print(f"   split: '{split_name}'")
                    break
            else:
                # train/validation/test がない場合、最初のsplitを使用
                first_split = next(iter(ds.keys()))
                ds = ds[first_split]
                print(f"   split: '{first_split}'（自動選択）")

    # テキストフィールドを抽出
    texts = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        text = _extract_text_from_item(item, text_field)

        if text and text.strip():
            texts.append(text.strip())

    print(f"   {len(texts)} 個のテキストを読み込みました")
    return texts


def _extract_text_from_item(item: dict, text_field: str = 'text') -> str:
    """
    データセットのアイテムからテキストを抽出する

    conversation形式、text形式、instruction形式など
    様々なデータ形式に対応。

    Args:
        item: データセットの1レコード
        text_field: テキストフィールド名

    Returns:
        抽出されたテキスト文字列
    """
    # 1. conversation 形式（list of {role, content}）
    if text_field == 'conversation' and 'conversation' in item:
        return _format_conversation(item['conversation'])

    # 2. messages 形式（OpenAI互換）
    if text_field == 'messages' and 'messages' in item:
        return _format_conversation(item['messages'])

    # 3. 指定されたtext_fieldがある場合
    if text_field and text_field in item:
        value = item[text_field]
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return _format_conversation(value)

    # 4. フォールバック: よくあるフィールド名を試す
    for field_name in ['text', 'content', 'text_qwen3', 'text_gpt_oss']:
        if field_name in item and isinstance(item[field_name], str):
            return item[field_name]

    # 5. instruction + output 形式
    if 'instruction' in item:
        text = item['instruction']
        if 'output' in item and item['output']:
            return f"User: {item['instruction']}\nAssistant: {item['output']}"
        return text

    # 6. 最初の文字列フィールドを使用
    for v in item.values():
        if isinstance(v, str) and len(v) > 10:
            return v

    return ''


def _format_conversation(messages: list) -> str:
    """
    会話メッセージリストをテキスト形式に変換

    Args:
        messages: [{role: 'user', content: '...'}, ...] 形式のリスト

    Returns:
        'User: ...\nAssistant: ...' 形式のテキスト
    """
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role', '')
        content = msg.get('content', '')
        if not content:
            continue

        if role == 'user':
            parts.append(f"User: {content}")
        elif role == 'assistant':
            parts.append(f"Assistant: {content}")
        elif role == 'system':
            parts.append(f"System: {content}")
        else:
            parts.append(content)

    return '\n'.join(parts)


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
