#!/usr/bin/env python3
"""
NeuroQ RunPod Serverless Handler - Optimized Version
=====================================================
高速起動 & 安定動作のための最適化済みハンドラー

特徴:
- 起動時に重い処理をしない（高速起動）
- health checkは即座に200を返す
- モデルは初回リクエスト時にlazy load
- vocab_sizeの整合性を保証
"""

import runpod
import torch
import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from daily_time_limiter import DailyTimeLimiter

# 現在のディレクトリをパスに追加（neuroq_pretrained.pyをインポートするため）
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=" * 60)
print("⚛️ NeuroQ RunPod Serverless - Starting...")
print("=" * 60)

# NeuroQuantumBrainAI をインポート
try:
    from neuroquantum_brain import NeuroQuantumBrainAI, NeuroQuantumBrain, get_training_data
    from neuroquantum_layered import NeuroQuantumTokenizer, NeuroQuantum, NeuroQuantumConfig
    print("✅ neuroquantum_brain.py をインポートしました")
    print("✅ neuroquantum_layered.py をインポートしました")
except ImportError as e:
    print(f"❌ インポートに失敗: {e}")
    NeuroQuantumBrainAI = None
    NeuroQuantumBrain = None
    NeuroQuantumTokenizer = None
    NeuroQuantum = None
    NeuroQuantumConfig = None

# モデル設定をインポート
try:
    from model_configs import AVAILABLE_MODELS, get_model_config, get_checkpoint_path, create_model
    MODEL_CONFIGS_AVAILABLE = True
    print("✅ model_configs.py をインポートしました")
except ImportError:
    MODEL_CONFIGS_AVAILABLE = False

# データセット設定をインポート
try:
    from dataset_configs import (
        AVAILABLE_DATASETS, get_dataset_config, get_datasets_list,
        load_dataset_texts, find_data_file, find_tokenizer_file,
        get_training_params
    )
    DATASET_CONFIGS_AVAILABLE = True
    print("✅ dataset_configs.py をインポートしました")
except ImportError as e:
    DATASET_CONFIGS_AVAILABLE = False
    print(f"⚠️ dataset_configs.py のインポートに失敗: {e}")

# BPEトークナイザー学習モジュールをインポート
try:
    from bpe_tokenizer_trainer import BPETokenizerTrainer, TrainedBPETokenizer, train_bpe_tokenizer
    BPE_TRAINER_AVAILABLE = True
    print("✅ bpe_tokenizer_trainer.py をインポートしました")
except ImportError as e:
    BPE_TRAINER_AVAILABLE = False
    print(f"⚠️ bpe_tokenizer_trainer.py のインポートに失敗: {e}")

# 翻訳パイプラインをインポート
try:
    from translation_pipeline import TranslationPipeline
    TRANSLATION_AVAILABLE = True
    print("✅ translation_pipeline.py をインポートしました")
except ImportError as e:
    TRANSLATION_AVAILABLE = False
    TranslationPipeline = None
    print(f"⚠️ translation_pipeline.py のインポートに失敗: {e}")

# トークナイザーモデルのパス
TOKENIZER_MODEL_PATH = "neuroq_tokenizer.model"

# チェックポイントパス（モデルサイズ別）
# 全モデルを/model_checkpointsから参照
MODEL_CHECKPOINT_PATHS = {
    'micro': "/model_checkpoints/neuroq_micro_best.pt",
    'small': "/model_checkpoints/neuroq_small_best.pt",
    'large': "/model_checkpoints/neuroq_large_best.pt",
}
MODEL_CHECKPOINT_PATH = MODEL_CHECKPOINT_PATHS['micro']  # デフォルト

# 学習済みBPEトークナイザーの保存パス
TRAINED_TOKENIZER_PATHS = {
    'default': "/model_checkpoints/neuroq_bpe_tokenizer.json",
    'micro': "/model_checkpoints/neuroq_bpe_tokenizer_micro.json",
    'small': "/model_checkpoints/neuroq_bpe_tokenizer_small.json",
    'large': "/model_checkpoints/neuroq_bpe_tokenizer_large.json",
}

# ========================================
# グローバル変数（起動時は全てNone）
# ========================================
model = None  # NeuroQuantum または NeuroQuantumBrainAI インスタンス
model_config = None  # 現在のモデル設定
current_model_size = 'micro'  # 現在のモデルサイズ
is_initialized = False

# 日次時間制限（デフォルト: 1日 = 86400秒）
daily_limiter = DailyTimeLimiter(daily_limit_seconds=86400)

# 翻訳パイプライン（lazy loading）
translation_pipeline = None
translation_initialized = False

# 学習状態管理
pretrain_process = None
pretrain_status = "idle"  # idle, running, completed, error
pretrain_log_file = "training_openai.log"

# 会話履歴管理
conversation_sessions = {}  # session_id -> list of {role, content}

# 設定
VOCAB_SIZE = 8000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# デフォルトシステムプロンプト（会話指示 - 英語で生成→日本語に翻訳）
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant.
The user is a human.
Answer clearly and concisely.
Do not output random text.
Always reply in English.
You must respond in English only.

You are a helpful and accurate assistant.
User is the person using you, and Assistant is yourself.
Follow these rules:
1. Answer the user's questions briefly and accurately
2. Ask questions if you don't understand something
3. Only answer what is asked (don't add unnecessary information)
4. Respond based on the previous context"""

# セッションごとのシステムプロンプト
session_system_prompts = {}  # session_id -> system_prompt

print(f"📊 Device: {DEVICE}")
print(f"📊 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")


def remap_state_dict_keys(state_dict: dict, model_state_dict: dict) -> dict:
    """
    state_dictのキー名をモデルのキー名に変換する

    neuroq-runpod版: transformer_blocks, token_embedding, embedding_dropout
    メイン版: blocks, text_embedding, dropout

    Args:
        state_dict: チェックポイントのstate_dict
        model_state_dict: モデルのstate_dict（期待されるキー名の参照用）

    Returns:
        変換後のstate_dict
    """
    model_keys = set(model_state_dict.keys())
    checkpoint_keys = set(state_dict.keys())

    # キーが完全一致なら変換不要
    if model_keys == checkpoint_keys:
        return state_dict

    # モデルとチェックポイントのキープレフィックスを検出
    def has_prefix(keys, prefix):
        return any(k.startswith(prefix + '.') for k in keys)

    model_uses_transformer_blocks = has_prefix(model_keys, 'transformer_blocks')
    model_uses_blocks = has_prefix(model_keys, 'blocks')
    checkpoint_uses_transformer_blocks = has_prefix(checkpoint_keys, 'transformer_blocks')
    checkpoint_uses_blocks = has_prefix(checkpoint_keys, 'blocks')

    model_uses_token_embedding = has_prefix(model_keys, 'token_embedding')
    model_uses_text_embedding = has_prefix(model_keys, 'text_embedding')
    checkpoint_uses_token_embedding = has_prefix(checkpoint_keys, 'token_embedding')
    checkpoint_uses_text_embedding = has_prefix(checkpoint_keys, 'text_embedding')

    model_uses_embedding_dropout = has_prefix(model_keys, 'embedding_dropout')
    model_uses_dropout = has_prefix(model_keys, 'dropout')
    checkpoint_uses_embedding_dropout = has_prefix(checkpoint_keys, 'embedding_dropout')
    checkpoint_uses_dropout = has_prefix(checkpoint_keys, 'dropout')

    # 実際に必要な変換方向のみマッピングを構築
    key_mappings = {}

    # blocks <-> transformer_blocks
    if model_uses_transformer_blocks and checkpoint_uses_blocks and not checkpoint_uses_transformer_blocks:
        key_mappings['blocks'] = 'transformer_blocks'
    elif model_uses_blocks and checkpoint_uses_transformer_blocks and not checkpoint_uses_blocks:
        key_mappings['transformer_blocks'] = 'blocks'

    # text_embedding <-> token_embedding
    if model_uses_token_embedding and checkpoint_uses_text_embedding and not checkpoint_uses_token_embedding:
        key_mappings['text_embedding'] = 'token_embedding'
    elif model_uses_text_embedding and checkpoint_uses_token_embedding and not checkpoint_uses_text_embedding:
        key_mappings['token_embedding'] = 'text_embedding'

    # dropout <-> embedding_dropout
    if model_uses_embedding_dropout and checkpoint_uses_dropout and not checkpoint_uses_embedding_dropout:
        key_mappings['dropout'] = 'embedding_dropout'
    elif model_uses_dropout and checkpoint_uses_embedding_dropout and not checkpoint_uses_dropout:
        key_mappings['embedding_dropout'] = 'dropout'

    if not key_mappings:
        # マッピング不要（キーは異なるが変換対象外）
        print("⚠️ state_dictキー名が一致しませんが、変換マッピングが見つかりません")
        print(f"   モデルキー例: {list(model_keys)[:3]}")
        print(f"   チェックポイントキー例: {list(checkpoint_keys)[:3]}")
        return state_dict

    print("🔄 state_dictキー名を変換中...")
    print(f"   変換マッピング: {key_mappings}")

    new_state_dict = {}
    converted_count = 0
    for key, value in state_dict.items():
        new_key = key
        for old_name, new_name in key_mappings.items():
            if key.startswith(old_name + '.'):
                new_key = new_name + key[len(old_name):]
                converted_count += 1
                break
        new_state_dict[new_key] = value

    print(f"   ✅ {converted_count}個のキーを変換しました")
    return new_state_dict




# ========================================
# Checkpoint管理（保存・ロード）
# ========================================
def save_checkpoint(model_instance, checkpoint_path: str = MODEL_CHECKPOINT_PATH):
    """
    学習済みモデルをチェックポイントとして保存

    Args:
        model_instance: NeuroQuantumBrainAI インスタンス
        checkpoint_path: 保存先パス
    """
    try:
        # ディレクトリ作成
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        # 保存するデータ
        checkpoint = {
            'model_state_dict': model_instance.model.state_dict(),
            'config': {
                'embed_dim': model_instance.embed_dim,
                'num_heads': model_instance.num_heads,
                'num_layers': model_instance.num_layers,
                'num_neurons': model_instance.num_neurons,
                'max_vocab': model_instance.max_vocab,
            },
            'tokenizer_path': TOKENIZER_MODEL_PATH,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"✅ チェックポイント保存完了: {checkpoint_path}")
        return True

    except Exception as e:
        print(f"❌ チェックポイント保存エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_checkpoint(checkpoint_path: str = MODEL_CHECKPOINT_PATH):
    """
    チェックポイントから学習済みモデルをロード

    Args:
        checkpoint_path: チェックポイントのパス

    Returns:
        NeuroQuantumBrainAI インスタンス、またはNone
    """
    try:
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ チェックポイントが見つかりません: {checkpoint_path}")
            return None

        print(f"📦 チェックポイントをロード中: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        # 設定を復元
        config = checkpoint['config']

        # vocab_size を取得（'vocab_size' または 'max_vocab' キーをサポート）
        vocab_size = config.get('vocab_size') or config.get('max_vocab')
        if vocab_size is None:
            raise ValueError("チェックポイントに vocab_size または max_vocab が見つかりません")

        # モデルを構築
        model_instance = NeuroQuantumBrainAI(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_neurons=config['num_neurons'],
            max_vocab=vocab_size,
            use_sentencepiece=True
        )

        # トークナイザーをロード
        if os.path.exists(TOKENIZER_MODEL_PATH):
            print(f"✅ トークナイザーをロード: {TOKENIZER_MODEL_PATH}")
            model_instance.tokenizer = NeuroQuantumTokenizer(
                vocab_size=vocab_size,
                model_file=TOKENIZER_MODEL_PATH
            )

        # NeuroQuantumBrainモデルを作成
        print(f"📦 NeuroQuantumBrainモデルを構築中...")
        model_instance.model = NeuroQuantumBrain(
            vocab_size=vocab_size,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_neurons=config['num_neurons'],
            max_seq_len=256,
            dropout=0.1
        ).to(model_instance.device)

        # 重みをロード（キー名の互換性を保証）
        state_dict = checkpoint['model_state_dict']
        state_dict = remap_state_dict_keys(state_dict, model_instance.model.state_dict())
        model_instance.model.load_state_dict(state_dict)

        # 推論モードに設定
        model_instance.model.eval()

        print(f"✅ チェックポイントロード完了")
        return model_instance

    except Exception as e:
        print(f"❌ チェックポイントロードエラー: {e}")
        import traceback
        traceback.print_exc()
        return None


# ========================================
# Lazy Model Loading（初回リクエスト時のみ）
# ========================================
def initialize_model(model_size: str = 'micro'):
    """
    モデルを初期化（初回リクエスト時のみ呼ばれる）

    推論専用：チェックポイントをロードするだけで、学習は一切しない
    
    Args:
        model_size: 'micro', 'small', 'large'
    """
    global model, model_config, current_model_size, is_initialized

    # 既に初期化済みで同じサイズなら何もしない
    if is_initialized and current_model_size == model_size:
        return True
    
    # 異なるサイズが要求された場合は再初期化
    if is_initialized and current_model_size != model_size:
        print(f"🔄 モデルサイズ変更: {current_model_size} → {model_size}")
        is_initialized = False
        model = None

    print(f"🔄 モデル初期化開始（{model_size.upper()}）...")

    try:
        # チェックポイントパスを取得
        checkpoint_path = MODEL_CHECKPOINT_PATHS.get(model_size, MODEL_CHECKPOINT_PATHS['micro'])
        
        # MODEL_CONFIGS_AVAILABLEでSmall/Largeモデルを使用
        if MODEL_CONFIGS_AVAILABLE and model_size in ['small', 'large'] and NeuroQuantum is not None:
            print(f"📦 {model_size.upper()}モデルを構築中...")
            
            config = get_model_config(model_size)
            model_config = config
            
            # チェックポイントが存在するか確認
            if os.path.exists(checkpoint_path):
                print(f"💾 チェックポイントをロード: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                
                # モデルを構築
                nq_config = NeuroQuantumConfig()
                nq_config.vocab_size = config['vocab_size']
                nq_config.embed_dim = config['embed_dim']
                nq_config.num_heads = config['num_heads']
                nq_config.num_layers = config['num_layers']
                nq_config.max_seq_len = config.get('max_seq_len', 512)
                nq_config.dropout = config.get('dropout', 0.1)
                
                model = NeuroQuantum(nq_config).to(DEVICE)
                
                # 重みをロード（キー名の互換性を保証）
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = remap_state_dict_keys(checkpoint['model_state_dict'], model.state_dict())
                    model.load_state_dict(state_dict)
                else:
                    state_dict = remap_state_dict_keys(checkpoint, model.state_dict())
                    model.load_state_dict(state_dict, strict=False)
                
                model.eval()
                print(f"✅ {config['name']}をロードしました")
            else:
                print(f"⚠️ チェックポイントが見つかりません: {checkpoint_path}")
                print(f"⚠️ 未学習の{model_size.upper()}モデルを作成します")
                
                nq_config = NeuroQuantumConfig()
                nq_config.vocab_size = config['vocab_size']
                nq_config.embed_dim = config['embed_dim']
                nq_config.num_heads = config['num_heads']
                nq_config.num_layers = config['num_layers']
                nq_config.max_seq_len = config.get('max_seq_len', 512)
                nq_config.dropout = config.get('dropout', 0.1)
                
                model = NeuroQuantum(nq_config).to(DEVICE)
            
            current_model_size = model_size
            is_initialized = True
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ {config['name']}初期化完了 ({total_params:,}パラメータ)")
            return True
        
        else:
            # 従来のNeuroQuantumBrainAIを使用（micro）
            if NeuroQuantumBrainAI is None:
                raise ImportError("NeuroQuantumBrainAI がインポートできていません")

            # チェックポイントからロード
            model = load_checkpoint(checkpoint_path)

            if model is None:
                print("⚠️ チェックポイントが見つかりません。未学習モデルを作成します。")
                print("⚠️ 推論前に action='train' で学習を実行してください。")

                model = NeuroQuantumBrainAI(
                    embed_dim=128,
                    num_heads=4,
                    num_layers=3,
                    num_neurons=100,
                    max_vocab=8000,
                    use_sentencepiece=True
                )

                if os.path.exists(TOKENIZER_MODEL_PATH):
                    print(f"✅ トークナイザーをロード: {TOKENIZER_MODEL_PATH}")
                    model.tokenizer = NeuroQuantumTokenizer(
                        vocab_size=8000,
                        model_file=TOKENIZER_MODEL_PATH
                    )

            current_model_size = 'micro'
            is_initialized = True
            print("✅ モデル初期化完了（推論モード）!")
            return True

    except Exception as e:
        print(f"❌ モデル初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========================================
# 翻訳パイプライン初期化（Lazy Loading）
# ========================================
def initialize_translation_pipeline():
    """
    翻訳パイプラインを初期化（初回リクエスト時のみ呼ばれる）

    NLLB-200を使用した日本語↔英語の翻訳機能を提供
    """
    global translation_pipeline, translation_initialized

    if translation_initialized:
        return True

    if not TRANSLATION_AVAILABLE:
        print("⚠️ 翻訳パイプラインは利用できません（translation_pipeline.pyが見つかりません）")
        return False

    try:
        print("🔄 翻訳パイプライン初期化中...")
        translation_pipeline = TranslationPipeline(
            model_name="facebook/nllb-200-distilled-600M",
            device=DEVICE,
            use_tiktoken=True,
        )
        translation_initialized = True
        print("✅ 翻訳パイプライン初期化完了")
        return True
    except Exception as e:
        print(f"❌ 翻訳パイプライン初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========================================
# 会話履歴管理
# ========================================
def save_conversation_turn(session_id: str, user_message: str, assistant_response: str):
    """
    会話ターンを履歴に保存

    Args:
        session_id: セッションID
        user_message: ユーザーメッセージ
        assistant_response: アシスタントの応答
    """
    global conversation_sessions

    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []

    # ユーザーメッセージとアシスタントの応答を保存
    conversation_sessions[session_id].append({
        "role": "user",
        "content": user_message
    })
    conversation_sessions[session_id].append({
        "role": "assistant",
        "content": assistant_response
    })

    # 古い履歴を削除（最大10ターン = 20メッセージ）
    if len(conversation_sessions[session_id]) > 20:
        conversation_sessions[session_id] = conversation_sessions[session_id][-20:]


# ========================================
# テキスト生成
# ========================================
def generate_text(prompt: str, max_length: int = 50,
                  temp_min: float = None, temp_max: float = None,
                  temperature: float = None, session_id: str = "default",
                  system_prompt: str = None,
                  use_translation: bool = True) -> dict:
    """
    テキスト生成（会話対応版 - NeuroQuantumBrainAI使用）

    推論専用：model.eval() + torch.no_grad() で学習を一切しない

    Args:
        prompt: 入力プロンプト
        max_length: 最大生成長（デフォルト50 - 会話向けに短く制限）
        temp_min: 最低温度（指定された場合はtemp_min/temp_maxを使用）
        temp_max: 最高温度
        temperature: 互換性のための単一温度（指定された場合は自動的にtemp_min/temp_maxに変換）
        session_id: 会話セッションID（会話履歴管理用）
        system_prompt: カスタムシステムプロンプト（Noneの場合はセッションの既存設定またはデフォルトを使用）
        use_translation: 翻訳パイプラインを使用するか（日本語入力→英語生成→日本語出力）

    Returns:
        dict: {
            "generated": 生成されたテキスト,
            "translated_prompt": 翻訳された入力（翻訳使用時のみ）,
            "english_response": 英語の生成結果（翻訳使用時のみ）,
            "translation_used": 翻訳が使用されたか
        }
    """
    global model, session_system_prompts, translation_pipeline

    if model is None:
        return {"generated": "Error: Model not initialized", "translation_used": False}

    # モデルが未学習の場合（model.model が None または学習済みの NeuroQuantumBrain がない場合）
    if model.model is None:
        return {"generated": "Error: Model not trained. Please run action='train' or action='pretrain_openai' first to train the model before generating text.", "translation_used": False}

    # 翻訳パイプラインの初期化（必要な場合）
    translated_prompt = None
    english_response = None
    actual_prompt = prompt

    if use_translation:
        if not TRANSLATION_AVAILABLE:
            print("⚠️ 翻訳機能は利用できません")
            use_translation = False
        elif not translation_initialized:
            if not initialize_translation_pipeline():
                print("⚠️ 翻訳パイプラインの初期化に失敗しました")
                use_translation = False

    # 入力を日本語→英語に翻訳
    if use_translation and translation_pipeline is not None:
        try:
            translated_prompt = translation_pipeline.ja_to_en(prompt)
            actual_prompt = translated_prompt
            print(f"🌐 翻訳(JA→EN): {prompt[:30]}... → {translated_prompt[:30]}...")
        except Exception as e:
            print(f"⚠️ 入力翻訳エラー: {e}")
            use_translation = False

    try:
        # 推論モードに設定（重要：学習を防ぐ）
        model.model.eval()

        # システムプロンプトの処理
        # 1. リクエストで指定された場合はそれを使用し、セッションに保存
        # 2. 指定されていない場合はセッションの既存設定を使用
        # 3. セッションにも設定がない場合はデフォルトを使用
        if system_prompt is not None:
            session_system_prompts[session_id] = system_prompt
            active_system_prompt = system_prompt
        else:
            active_system_prompt = session_system_prompts.get(session_id, DEFAULT_SYSTEM_PROMPT)

        # temperatureが指定された場合、temp_min/temp_maxに変換
        if temperature is not None and temp_min is None:
            temp_min = temperature * 0.8
            temp_max = temperature * 1.2

        # デフォルト値（会話生成に最適化 - より保守的）
        if temp_min is None:
            temp_min = 0.4  # 会話向けにより保守的な温度
        if temp_max is None:
            temp_max = 0.7  # 0.8 → 0.7 に下げて暴走を防ぐ

        # 会話履歴を含むプロンプトを構築
        # NeuroQuantumBrainAI.generate() は内部で <USER>...<ASSISTANT> フォーマットを処理する
        # ただし、履歴を追加するためにここでプレフィックスを構築
        history = conversation_sessions.get(session_id, [])[-4:]  # 最新4ターン

        # システムプロンプトをコンテキストの先頭に追加
        context_parts = []
        if active_system_prompt:
            context_parts.append(f"<SYSTEM>{active_system_prompt}")

        # 会話履歴を追加
        for turn in history:
            if turn["role"] == "user":
                context_parts.append(f"<USER>{turn['content']}")
            elif turn["role"] == "assistant":
                context_parts.append(f"<ASSISTANT>{turn['content']}")

        # 履歴を含む完全なプロンプト（翻訳使用時は翻訳済みプロンプトを使用）
        history_context = "".join(context_parts)
        full_prompt = history_context + actual_prompt if history_context else actual_prompt

        # 推論実行（torch.no_grad()で勾配計算を無効化）
        with torch.no_grad():
            result = model.generate(
                prompt=full_prompt,
                max_length=max_length,
                temperature_min=temp_min,
                temperature_max=temp_max,
                top_k=40,
                top_p=0.9,
            )

        # 英語の生成結果を保存（翻訳使用時）
        if use_translation:
            english_response = result

        # 出力を英語→日本語に翻訳
        if use_translation and translation_pipeline is not None:
            try:
                result = translation_pipeline.en_to_ja(result)
                print(f"🌐 翻訳(EN→JA): {english_response[:30]}... → {result[:30]}...")
            except Exception as e:
                print(f"⚠️ 出力翻訳エラー: {e}")
                # エラー時は英語の結果をそのまま返す

        # 会話履歴に保存（元のプロンプトと最終結果）
        save_conversation_turn(session_id, prompt, result)

        return {
            "generated": result,
            "translated_prompt": translated_prompt,
            "english_response": english_response,
            "translation_used": use_translation and translation_pipeline is not None
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"generated": f"Error: {str(e)}", "translation_used": False}


# ========================================
# メインハンドラー（RunPod用）
# ========================================
def handler(job):
    """
    RunPod Serverless Handler

    重要: health checkは即座に返す！
    """
    global model, is_initialized, current_model_size, model_config
    global conversation_sessions, session_system_prompts
    global pretrain_process, pretrain_status
    global daily_limiter

    job_input = job.get("input", {})
    action = job_input.get("action", "generate")

    # ========================================
    # HEALTH CHECK（最優先・即座に返す）
    # ========================================
    if action == "health":
        return {
            "status": "healthy",
            "device": DEVICE,
            "cuda_available": torch.cuda.is_available(),
            "model_initialized": is_initialized
        }

    # ========================================
    # TIME_LIMIT_STATUS（日次時間制限ステータス確認）
    # ========================================
    if action == "time_limit_status":
        return {
            "status": "success",
            **daily_limiter.get_status()
        }

    # ========================================
    # SET_TIME_LIMIT（日次時間制限設定）
    # ========================================
    if action == "set_time_limit":
        new_limit = job_input.get("daily_limit_seconds")
        if new_limit is None:
            # 時間単位でも指定可能
            new_limit_hours = job_input.get("daily_limit_hours")
            if new_limit_hours is not None:
                new_limit = int(new_limit_hours * 3600)
        if new_limit is None:
            return {
                "status": "error",
                "error": "daily_limit_seconds or daily_limit_hours is required"
            }
        daily_limiter.set_limit(int(new_limit))
        return {
            "status": "success",
            "message": f"Daily time limit set to {new_limit} seconds ({new_limit/3600:.1f} hours)",
            **daily_limiter.get_status()
        }

    # ========================================
    # STATUS CHECK
    # ========================================
    if action == "status":
        time_status = daily_limiter.get_status()
        result = {
            "status": "ok",
            "initialized": is_initialized,
            "device": DEVICE,
            "vocab_size": VOCAB_SIZE,
            "current_model_size": current_model_size,
            "available_model_sizes": ["micro", "small", "large"],
            "model_configs_available": MODEL_CONFIGS_AVAILABLE,
            "translation_available": TRANSLATION_AVAILABLE,
            "translation_initialized": translation_initialized,
            "daily_time_limit": time_status
        }
        if DATASET_CONFIGS_AVAILABLE:
            result["datasets"] = get_datasets_list()
            result["dataset_configs_available"] = True
        else:
            result["dataset_configs_available"] = False
        return result
    
    # ========================================
    # GENERATE（モデルが必要な処理）
    # ========================================
    if action == "generate":
        # 日次時間制限チェック
        if not daily_limiter.start_request():
            limit_info = daily_limiter.get_status()
            return {
                "status": "error",
                "error": "Daily time limit exceeded",
                "message": f"本日の使用時間が制限（{limit_info['daily_limit_hours']}時間）に達しました。明日リセットされます。",
                "daily_time_limit": limit_info
            }

        request_start = time.time()

        # モデルサイズを取得
        model_size = job_input.get("model_size", "micro")

        # Lazy initialization（モデルサイズに応じて初期化）
        if not is_initialized or current_model_size != model_size:
            print(f"🔄 モデル初期化中 ({model_size.upper()})...")
            if not initialize_model(model_size):
                daily_limiter.end_request(time.time() - request_start)
                return {
                    "status": "error",
                    "error": f"Failed to initialize model ({model_size})"
                }

        prompt = job_input.get("prompt", "こんにちは")
        max_length = job_input.get("max_length", 50)
        session_id = job_input.get("session_id", "default")

        # 温度パラメータ
        temp_min = job_input.get("temp_min")
        temp_max = job_input.get("temp_max")
        temperature = job_input.get("temperature", 0.5)

        # システムプロンプト（カスタム設定可能）
        system_prompt = job_input.get("system_prompt")

        # 翻訳パラメータ（日本語入力→英語生成→日本語出力）
        # デフォルトでTrue: 日本語→英語→日本語の双方向翻訳を有効化
        use_translation = job_input.get("use_translation", True)

        print(f"📝 Generate: model={model_size}, session='{session_id}', prompt='{prompt[:30]}...', translation={use_translation}")
        if system_prompt:
            print(f"📝 System prompt: '{system_prompt[:50]}...'")

        result = generate_text(
            prompt=prompt,
            max_length=max_length,
            temp_min=temp_min,
            temp_max=temp_max,
            temperature=temperature,
            session_id=session_id,
            system_prompt=system_prompt,
            use_translation=use_translation
        )

        # 処理時間を記録
        elapsed = time.time() - request_start
        daily_limiter.end_request(elapsed)

        response = {
            "status": "success",
            "prompt": prompt,
            "generated": result.get("generated", ""),
            "session_id": session_id,
            "model_size": current_model_size,
            "system_prompt": session_system_prompts.get(session_id, DEFAULT_SYSTEM_PROMPT),
            "translation_used": result.get("translation_used", False),
            "processing_time_seconds": round(elapsed, 2)
        }

        # 翻訳が使用された場合、追加情報を含める
        if result.get("translation_used"):
            response["translated_prompt"] = result.get("translated_prompt")
            response["english_response"] = result.get("english_response")

        return response
    
    # ========================================
    # PRETRAIN_OPENAI（OpenAIデータセット事前学習）
    # ========================================
    if action == "pretrain_openai":
        # 既に実行中の場合
        if pretrain_status == "running":
            return {
                "status": "error",
                "error": "Pretraining is already running",
                "pretrain_status": pretrain_status
            }

        # ログファイルのパスを確認
        log_path = Path(pretrain_log_file)

        try:
            # バックグラウンドでpretrain_openai.pyを実行
            print("🚀 Starting OpenAI pretraining...")
            pretrain_status = "running"

            # python -u で unbuffered output
            cmd = [
                sys.executable, "-u",
                "pretrain_openai.py"
            ]

            # ログファイルを開いてsubprocessを起動
            # IMPORTANT: Don't use 'with' statement as the file needs to stay open
            # for the entire duration of the subprocess
            log_file = open(log_path, 'w', buffering=1)  # Line buffered
            pretrain_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            # 非同期でプロセスを監視
            def monitor_pretrain():
                global pretrain_status, pretrain_process
                pretrain_process.wait()

                # Close the log file after the process finishes
                try:
                    log_file.close()
                except:
                    pass

                if pretrain_process.returncode == 0:
                    pretrain_status = "completed"
                    print("✅ Pretraining completed successfully")
                else:
                    pretrain_status = "error"
                    print(f"❌ Pretraining failed with code {pretrain_process.returncode}")

            monitor_thread = threading.Thread(target=monitor_pretrain, daemon=True)
            monitor_thread.start()

            return {
                "status": "success",
                "message": "Pretraining started",
                "pretrain_status": pretrain_status,
                "log_file": str(log_path),
                "pid": pretrain_process.pid
            }

        except Exception as e:
            pretrain_status = "error"
            return {
                "status": "error",
                "error": str(e),
                "pretrain_status": pretrain_status
            }

    # ========================================
    # PRETRAIN_STATUS（事前学習ステータス確認）
    # ========================================
    if action == "pretrain_status":
        log_path = Path(pretrain_log_file)

        # ログファイルの最後の数行を読む
        log_tail = ""
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    log_tail = ''.join(lines[-20:])  # 最後の20行
            except Exception as e:
                log_tail = f"Error reading log: {e}"

        return {
            "status": "success",
            "pretrain_status": pretrain_status,
            "log_file": str(log_path),
            "log_exists": log_path.exists(),
            "log_tail": log_tail,
            "process_running": pretrain_process is not None and pretrain_process.poll() is None
        }

    # ========================================
    # TRAIN_TOKENIZER（BPEトークナイザー学習）
    # ========================================
    if action == "train_tokenizer":
        """
        BPEトークナイザー学習アクション

        tiktoken スタイルの BPE トークナイザーを学習データから学習します。

        パラメータ:
        - vocab_size: 語彙サイズ（デフォルト: 32000）
        - min_frequency: 最小出現頻度（デフォルト: 2）
        - texts: 学習テキスト（省略時はデフォルトデータ）
        - model_size: 保存先のモデルサイズ（'default', 'micro', 'small', 'large'）
        """

        if not BPE_TRAINER_AVAILABLE:
            return {
                "status": "error",
                "error": "BPE tokenizer trainer not available. Install 'tokenizers' library."
            }

        vocab_size = job_input.get("vocab_size", 32000)
        min_frequency = job_input.get("min_frequency", 2)
        texts = job_input.get("texts", None)
        model_size = job_input.get("model_size", "default")

        print(f"🔤 BPEトークナイザー学習開始")
        print(f"   語彙サイズ: {vocab_size:,}")
        print(f"   最小出現頻度: {min_frequency}")

        try:
            # 学習データ取得
            if texts is None:
                print("📚 デフォルト学習データを使用")
                texts = get_training_data()

            print(f"📊 学習データ: {len(texts)} サンプル")

            # トークナイザー学習
            trainer = BPETokenizerTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
            )
            trainer.train(texts)

            # 保存
            save_path = TRAINED_TOKENIZER_PATHS.get(model_size, TRAINED_TOKENIZER_PATHS['default'])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trainer.save(save_path)

            # テスト
            test_texts = [
                "こんにちは",
                "量子コンピュータ",
                "Hello World",
            ]

            test_results = []
            for text in test_texts:
                tokens = trainer.encode(text)
                test_results.append({
                    "text": text,
                    "token_count": len(tokens),
                    "tokens": trainer.get_tokens(text)[:10],  # 最初の10トークン
                })

            print(f"✅ トークナイザー学習完了！")
            print(f"💾 保存先: {save_path}")

            return {
                "status": "success",
                "message": "Tokenizer training completed",
                "vocab_size": trainer.get_vocab_size(),
                "requested_vocab_size": vocab_size,
                "num_samples": len(texts),
                "save_path": save_path,
                "test_results": test_results,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e)
            }

    # ========================================
    # TRAIN（学習 - チェックポイント保存付き）
    # ========================================
    if action == "train":
        """
        学習専用アクション

        パラメータ:
        - model_size: 'micro', 'small', 'large'（デフォルト: 'micro'）
        - dataset_id: データセットID（dataset_configs.pyで定義されたID）
        - epochs: エポック数
        - batch_size: バッチサイズ
        - lr: 学習率
        - seq_length: シーケンス長
        - texts: 学習テキスト（省略時はデフォルトデータ）
        - train_tokenizer: トークナイザーも学習するか（デフォルト: False）
        - tokenizer_vocab_size: トークナイザー語彙サイズ（デフォルト: 32000）
        - use_custom_tokenizer: カスタムトークナイザーを使用するか（デフォルト: False）

        dataset_id オプション（dataset_configs.pyで定義）:
        - 'oasst1_ja': kunishou/oasst1-89k-ja 日本語会話データセット
        - 'oasst1_ja_cleaned': クリーニング済み日本語会話データ
        - 'training_data': 汎用トレーニングデータ
        - 'combined_clean': 結合・クリーニング済みデータ
        - 'high_quality': キュレーション済み高品質会話データ
        - 'japanese_corpus': 日本語トレーニングコーパス
        """

        # 日次時間制限チェック
        if not daily_limiter.start_request():
            limit_info = daily_limiter.get_status()
            return {
                "status": "error",
                "error": "Daily time limit exceeded",
                "message": f"本日の使用時間が制限（{limit_info['daily_limit_hours']}時間）に達しました。明日リセットされます。",
                "daily_time_limit": limit_info
            }

        train_request_start = time.time()

        model_size = job_input.get("model_size", "micro")
        dataset_id = job_input.get("dataset_id", None)
        epochs = job_input.get("epochs", 30)
        batch_size = job_input.get("batch_size", 8)
        lr = job_input.get("lr", 0.0005)
        seq_length = job_input.get("seq_length", 64)
        texts = job_input.get("texts", None)
        train_tokenizer_flag = job_input.get("train_tokenizer", False)
        tokenizer_vocab_size = job_input.get("tokenizer_vocab_size", 32000)
        use_custom_tokenizer = job_input.get("use_custom_tokenizer", False)

        # ========================================
        # dataset_id による学習データ切り替え
        # ========================================
        dataset_tokenizer_path = None

        if dataset_id is not None and DATASET_CONFIGS_AVAILABLE:
            try:
                ds_config = get_dataset_config(dataset_id)
            except ValueError as e:
                daily_limiter.end_request(time.time() - train_request_start)
                datasets = get_datasets_list() if DATASET_CONFIGS_AVAILABLE else []
                return {
                    "status": "error",
                    "error": str(e),
                    "datasets": datasets
                }

            print("=" * 70)
            print(f"📚 dataset_id='{dataset_id}' - {ds_config['name']}で学習")
            print(f"   {ds_config['description']}")
            print("=" * 70)

            # データファイルを探す（HFデータセットの場合はローカルファイル不要）
            try:
                data_file = find_data_file(ds_config)
            except FileNotFoundError:
                if ds_config.get('id') is not None:
                    data_file = None  # HuggingFaceからダウンロード
                    print(f"   📡 ローカルファイルなし → HuggingFace Hubからロード")
                else:
                    daily_limiter.end_request(time.time() - train_request_start)
                    return {
                        "status": "error",
                        "error": f"データファイルが見つかりません: {ds_config['name']}",
                        "dataset_id": dataset_id
                    }

            # トークナイザーを探す
            try:
                dataset_tokenizer_path = find_tokenizer_file(ds_config)
            except FileNotFoundError as e:
                daily_limiter.end_request(time.time() - train_request_start)
                return {
                    "status": "error",
                    "error": str(e),
                    "dataset_id": dataset_id
                }

            # テキストデータを読み込み
            texts = load_dataset_texts(dataset_id)
            if dataset_tokenizer_path:
                print(f"🔤 トークナイザー: {dataset_tokenizer_path}")

            # データセットのデフォルトパラメータを適用（ユーザー指定がない場合のみ）
            ds_defaults = get_training_params(dataset_id, overrides={
                'epochs': job_input.get("epochs"),
                'batch_size': job_input.get("batch_size"),
                'lr': job_input.get("lr"),
                'seq_length': job_input.get("seq_length"),
            })
            epochs = ds_defaults['epochs']
            batch_size = ds_defaults['batch_size']
            lr = ds_defaults['lr']
            seq_length = ds_defaults['seq_length']

        elif dataset_id is not None and not DATASET_CONFIGS_AVAILABLE:
            # dataset_configs が利用不可でも oasst1_ja は後方互換のためサポート
            if dataset_id == "oasst1_ja":
                print("=" * 70)
                print("📚 dataset_id='oasst1_ja' - 日本語会話データセットで学習（レガシーモード）")
                print("=" * 70)

                data_file_candidates = [
                    "data/oasst1_ja_conversations.txt",
                    "../data/oasst1_ja_conversations.txt",
                    os.path.join(os.path.dirname(__file__), "data/oasst1_ja_conversations.txt"),
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/oasst1_ja_conversations.txt"),
                ]

                tokenizer_candidates = [
                    "neuroq_tokenizer_oasst1_ja.model",
                    "../neuroq_tokenizer_oasst1_ja.model",
                    os.path.join(os.path.dirname(__file__), "neuroq_tokenizer_oasst1_ja.model"),
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "neuroq_tokenizer_oasst1_ja.model"),
                ]

                data_file = None
                for path in data_file_candidates:
                    if os.path.exists(path):
                        data_file = path
                        break

                if data_file is None:
                    daily_limiter.end_request(time.time() - train_request_start)
                    return {
                        "status": "error",
                        "error": "oasst1_ja data file not found. Run convert_oasst1_ja.py first.",
                        "searched_paths": data_file_candidates
                    }

                for path in tokenizer_candidates:
                    if os.path.exists(path):
                        dataset_tokenizer_path = path
                        break

                if dataset_tokenizer_path is None:
                    daily_limiter.end_request(time.time() - train_request_start)
                    return {
                        "status": "error",
                        "error": "oasst1_ja tokenizer not found. Run train_japanese_tokenizer.py first.",
                        "searched_paths": tokenizer_candidates
                    }

                with open(data_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                texts = []
                blocks = content.split('\n\n')
                for block in blocks:
                    block = block.strip()
                    if block and 'User:' in block and 'Assistant:' in block:
                        texts.append(block)

                print(f"   {len(texts)} 個の会話を読み込みました")
                print(f"🔤 トークナイザー: {dataset_tokenizer_path}")

                if job_input.get("epochs") is None:
                    epochs = 5
                if job_input.get("seq_length") is None:
                    seq_length = 128
                if job_input.get("lr") is None:
                    lr = 0.0003
            else:
                daily_limiter.end_request(time.time() - train_request_start)
                return {
                    "status": "error",
                    "error": f"dataset_configs.py が利用できないため、dataset_id='{dataset_id}' を解決できません。"
                             f" dataset_configs.py をインストールしてください。",
                }

        # モデルサイズに応じて設定を調整
        if model_size == 'small':
            batch_size = job_input.get("batch_size", 4)
            lr = job_input.get("lr", 0.0003)
        elif model_size == 'large':
            batch_size = job_input.get("batch_size", 2)
            lr = job_input.get("lr", 0.0002)

        print(f"📦 学習モデルサイズ: {model_size.upper()}")
        
        # Small/LargeモデルはNeuroQuantumを使用
        if MODEL_CONFIGS_AVAILABLE and model_size in ['small', 'large'] and NeuroQuantum is not None:
            print(f"🔄 {model_size.upper()}モデルで学習開始...")

            try:
                # 学習データ取得（トークナイザー学習に必要）
                if texts is None:
                    texts = get_training_data()

                # モデル設定を取得（トークナイザー初期化前に必要）
                config = get_model_config(model_size)
                checkpoint_path = MODEL_CHECKPOINT_PATHS.get(model_size, MODEL_CHECKPOINT_PATHS['micro'])

                # トークナイザー初期化
                tokenizer = None
                tokenizer_type = 'sentencepiece'
                custom_tokenizer_path = None

                # データセット専用トークナイザー
                if dataset_tokenizer_path is not None:
                    print(f"🔤 データセット専用トークナイザーを使用: {dataset_tokenizer_path}")
                    tokenizer = NeuroQuantumTokenizer(vocab_size=8000, model_file=dataset_tokenizer_path)
                    tokenizer_type = f'sentencepiece_dataset'
                    custom_tokenizer_path = dataset_tokenizer_path

                # カスタムトークナイザー学習
                elif train_tokenizer_flag and BPE_TRAINER_AVAILABLE:
                    print(f"🔤 カスタムBPEトークナイザーを学習中...")
                    print(f"   語彙サイズ: {tokenizer_vocab_size:,}")

                    trainer = BPETokenizerTrainer(
                        vocab_size=tokenizer_vocab_size,
                        min_frequency=2,
                    )
                    trainer.train(texts)

                    # 保存
                    custom_tokenizer_path = TRAINED_TOKENIZER_PATHS.get(model_size, TRAINED_TOKENIZER_PATHS['default'])
                    os.makedirs(os.path.dirname(custom_tokenizer_path), exist_ok=True)
                    trainer.save(custom_tokenizer_path)

                    # カスタムトークナイザーを使用
                    tokenizer = TrainedBPETokenizer(custom_tokenizer_path)
                    tokenizer_type = 'custom_bpe'
                    print(f"✅ カスタムトークナイザー学習完了: {custom_tokenizer_path}")

                # 既存のカスタムトークナイザーを使用
                elif use_custom_tokenizer and BPE_TRAINER_AVAILABLE:
                    custom_tokenizer_path = TRAINED_TOKENIZER_PATHS.get(model_size, TRAINED_TOKENIZER_PATHS['default'])
                    if os.path.exists(custom_tokenizer_path):
                        tokenizer = TrainedBPETokenizer(custom_tokenizer_path)
                        tokenizer_type = 'custom_bpe'
                        print(f"✅ カスタムトークナイザーを読み込み: {custom_tokenizer_path}")
                    else:
                        print(f"⚠️ カスタムトークナイザーが見つかりません: {custom_tokenizer_path}")
                        print(f"   SentencePiece を使用します")

                # デフォルト: SentencePiece
                if tokenizer is None:
                    tokenizer = NeuroQuantumTokenizer(vocab_size=config.get('vocab_size', 8000))
                    tokenizer.build_vocab(texts, model_prefix="neuroq_tokenizer")
                    tokenizer_type = 'sentencepiece'

                # カスタムトークナイザー使用時は vocab_size を更新
                if tokenizer_type == 'custom_bpe':
                    config = config.copy()  # 元の設定を変更しない
                    config['vocab_size'] = tokenizer.vocab_size
                    print(f"📊 カスタムトークナイザー語彙サイズ: {tokenizer.vocab_size:,}")
                
                print(f"📋 モデル設定: {config['name']}")
                print(f"   embed_dim: {config['embed_dim']}")
                print(f"   num_heads: {config['num_heads']}")
                print(f"   num_layers: {config['num_layers']}")
                
                # モデル構築
                nq_config = NeuroQuantumConfig()
                nq_config.vocab_size = config['vocab_size']
                nq_config.embed_dim = config['embed_dim']
                nq_config.num_heads = config['num_heads']
                nq_config.num_layers = config['num_layers']
                nq_config.max_seq_len = config.get('max_seq_len', 512)
                nq_config.dropout = config.get('dropout', 0.1)
                
                train_model = NeuroQuantum(nq_config).to(DEVICE)
                
                total_params = sum(p.numel() for p in train_model.parameters())
                print(f"   総パラメータ数: {total_params:,} ({total_params/1e6:.1f}M)")

                print(f"🔄 データトークン化中... ({len(texts)}サンプル)")
                
                # トークン化
                all_tokens = []
                for text in texts:
                    tokens = tokenizer.encode(text, add_special=False)
                    all_tokens.extend(tokens)
                    all_tokens.append(tokenizer.eos_id)
                
                print(f"   総トークン数: {len(all_tokens):,}")
                
                # シーケンス作成
                sequences = []
                for i in range(0, len(all_tokens) - seq_length, seq_length // 2):
                    sequences.append(all_tokens[i:i + seq_length])
                
                print(f"   シーケンス数: {len(sequences):,}")
                
                if len(sequences) == 0:
                    daily_limiter.end_request(time.time() - train_request_start)
                    return {"status": "error", "error": "Not enough data for training"}
                
                # 学習ループ
                optimizer = torch.optim.AdamW(train_model.parameters(), lr=lr)
                criterion = torch.nn.CrossEntropyLoss()
                
                total_batches = (len(sequences) - batch_size) // batch_size
                print(f"\n🚀 学習開始: {epochs}エポック, バッチ数/エポック: {total_batches}")
                
                import random
                best_loss = float('inf')
                
                for epoch in range(epochs):
                    train_model.train()
                    total_loss = 0
                    batch_count = 0
                    
                    random.shuffle(sequences)
                    
                    for i in range(0, len(sequences) - batch_size, batch_size):
                        batch = sequences[i:i + batch_size]
                        batch_tensor = torch.tensor(batch, device=DEVICE)
                        
                        input_ids = batch_tensor[:, :-1]
                        target_ids = batch_tensor[:, 1:]
                        
                        optimizer.zero_grad()
                        logits = train_model(input_ids)
                        
                        loss = criterion(
                            logits.reshape(-1, config['vocab_size']),
                            target_ids.reshape(-1)
                        )
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                        optimizer.step()
                        
                        total_loss += loss.item()
                        batch_count += 1
                        
                        # 進捗表示（10%ごと）
                        if batch_count % max(1, total_batches // 10) == 0:
                            progress = batch_count / total_batches * 100
                            print(f"   Epoch {epoch+1}/{epochs} - {progress:.0f}% - Loss: {total_loss/batch_count:.4f}")
                    
                    avg_loss = total_loss / max(batch_count, 1)
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                    
                    print(f"   ✅ Epoch {epoch+1}/{epochs} 完了: Loss={avg_loss:.4f}")
                
                print(f"\n✅ 学習完了！ベストLoss: {best_loss:.4f}")

                # チェックポイント保存
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

                # トークナイザー情報を作成
                if tokenizer_type == 'custom_bpe':
                    tokenizer_info = {
                        'type': 'custom_bpe',
                        'path': custom_tokenizer_path,
                        'vocab_size': tokenizer.vocab_size,
                    }
                else:
                    tokenizer_info = {
                        'type': 'sentencepiece',
                        'vocab_size': tokenizer.vocab_size,
                    }

                checkpoint = {
                    'model_state_dict': train_model.state_dict(),
                    'config': config,
                    'tokenizer': tokenizer_info,
                    'model_size': model_size,
                }

                torch.save(checkpoint, checkpoint_path)
                print(f"💾 チェックポイント保存: {checkpoint_path}")

                result = {
                    "status": "success",
                    "message": f"Training completed ({epochs} epochs)",
                    "model_size": model_size,
                    "model_name": config['name'],
                    "parameters": total_params,
                    "best_loss": best_loss,
                    "checkpoint_path": checkpoint_path,
                    "num_samples": len(texts),
                    "tokenizer_type": tokenizer_type,
                }

                if dataset_id:
                    result["dataset_id"] = dataset_id
                if tokenizer_type == 'custom_bpe':
                    result["tokenizer_path"] = custom_tokenizer_path
                    result["tokenizer_vocab_size"] = tokenizer.vocab_size

                daily_limiter.end_request(time.time() - train_request_start)
                return result

            except Exception as e:
                import traceback
                traceback.print_exc()
                daily_limiter.end_request(time.time() - train_request_start)
                return {"status": "error", "error": str(e)}
        
        else:
            # 従来のNeuroQuantumBrainAIを使用（micro）
            if not is_initialized:
                print("🔄 学習用モデルを新規作成...")
                try:
                    model = NeuroQuantumBrainAI(
                        embed_dim=128,
                        num_heads=4,
                        num_layers=3,
                        num_neurons=100,
                        max_vocab=8000,
                        use_sentencepiece=True
                    )

                    # データセット専用トークナイザーを優先的に使用
                    if dataset_tokenizer_path is not None and os.path.exists(dataset_tokenizer_path):
                        print(f"✅ データセットトークナイザーをロード: {dataset_tokenizer_path}")
                        model.tokenizer = NeuroQuantumTokenizer(
                            vocab_size=8000,
                            model_file=dataset_tokenizer_path
                        )
                    elif os.path.exists(TOKENIZER_MODEL_PATH):
                        print(f"✅ トークナイザーをロード: {TOKENIZER_MODEL_PATH}")
                        model.tokenizer = NeuroQuantumTokenizer(
                            vocab_size=8000,
                            model_file=TOKENIZER_MODEL_PATH
                        )

                    is_initialized = True

                except Exception as e:
                    daily_limiter.end_request(time.time() - train_request_start)
                    return {
                        "status": "error",
                        "error": f"Failed to create model: {e}"
                    }

            if texts is None:
                print("📚 デフォルト学習データを使用")
                texts = get_training_data()

            if not texts:
                daily_limiter.end_request(time.time() - train_request_start)
                return {
                    "status": "error",
                    "error": "No training texts provided"
                }

            print(f"🔄 学習開始: {len(texts)}サンプル, {epochs}エポック")

            try:
                model.train(
                    texts,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    seq_length=seq_length
                )

                checkpoint_path = job_input.get("checkpoint_path", MODEL_CHECKPOINT_PATH)
                if save_checkpoint(model, checkpoint_path):
                    result = {
                        "status": "success",
                        "message": f"Training completed ({epochs} epochs)",
                        "model_size": "micro",
                        "checkpoint_path": checkpoint_path,
                        "num_samples": len(texts)
                    }
                    if dataset_id:
                        result["dataset_id"] = dataset_id
                    daily_limiter.end_request(time.time() - train_request_start)
                    return result
                else:
                    daily_limiter.end_request(time.time() - train_request_start)
                    return {
                        "status": "warning",
                        "message": "Training completed but checkpoint save failed",
                        "num_samples": len(texts)
                    }

            except Exception as e:
                import traceback
                traceback.print_exc()
                daily_limiter.end_request(time.time() - train_request_start)
                return {
                    "status": "error",
                    "error": str(e)
                }
    
    # ========================================
    # CLEAR_SESSION（会話履歴クリア）
    # ========================================
    if action == "clear_session":
        session_id = job_input.get("session_id", "default")
        clear_system_prompt = job_input.get("clear_system_prompt", True)

        cleared_items = []
        if session_id in conversation_sessions:
            del conversation_sessions[session_id]
            cleared_items.append("history")
        if clear_system_prompt and session_id in session_system_prompts:
            del session_system_prompts[session_id]
            cleared_items.append("system_prompt")

        if cleared_items:
            return {
                "status": "success",
                "message": f"Session '{session_id}' cleared ({', '.join(cleared_items)})"
            }
        else:
            return {
                "status": "success",
                "message": f"Session '{session_id}' not found (already empty)"
            }

    # ========================================
    # SET_SYSTEM_PROMPT（システムプロンプト設定）
    # ========================================
    if action == "set_system_prompt":
        session_id = job_input.get("session_id", "default")
        system_prompt = job_input.get("system_prompt")

        if system_prompt is None:
            return {
                "status": "error",
                "error": "system_prompt is required"
            }

        session_system_prompts[session_id] = system_prompt
        return {
            "status": "success",
            "message": f"System prompt set for session '{session_id}'",
            "session_id": session_id,
            "system_prompt": system_prompt
        }

    # ========================================
    # GET_SYSTEM_PROMPT（システムプロンプト取得）
    # ========================================
    if action == "get_system_prompt":
        session_id = job_input.get("session_id", "default")
        system_prompt = session_system_prompts.get(session_id, DEFAULT_SYSTEM_PROMPT)
        is_default = session_id not in session_system_prompts

        return {
            "status": "success",
            "session_id": session_id,
            "system_prompt": system_prompt,
            "is_default": is_default
        }

    # ========================================
    # UNKNOWN ACTION
    # ========================================
    return {
        "status": "error",
        "error": f"Unknown action: {action}",
        "available_actions": [
            "health",             # ヘルスチェック
            "status",             # ステータス確認
            "train",              # 学習（dataset_idでデータセット指定可能）
            "generate",           # 推論（use_translation=trueで翻訳パイプライン使用）
            "pretrain_openai",    # OpenAIデータセット事前学習
            "pretrain_status",    # 事前学習ステータス
            "clear_session",      # 会話履歴クリア
            "set_system_prompt",  # システムプロンプト設定
            "get_system_prompt",  # システムプロンプト取得
            "time_limit_status",  # 日次時間制限ステータス確認
            "set_time_limit"      # 日次時間制限設定
        ],
        "datasets": get_datasets_list() if DATASET_CONFIGS_AVAILABLE else [{"id": "kunishou/oasst1-89k-ja", "config": None, "key": "oasst1_ja", "name": "OASST1 Japanese", "description": "kunishou/oasst1-89k-ja 日本語会話データセット"}],
        "translation_note": "generate アクションで use_translation=true を指定すると、日本語入力→英語生成→日本語出力の翻訳パイプラインを使用できます"
    }


# ========================================
# 起動（何もしない = 高速起動）
# ========================================
print("=" * 60)
print("✅ NeuroQ Handler Ready")
print("   - Health check: instant response")
print("   - Model loading: lazy (on first request)")
print("=" * 60)

# RunPod起動
runpod.serverless.start({"handler": handler})
