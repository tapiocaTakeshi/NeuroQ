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

# BPEトークナイザー学習モジュールをインポート
try:
    from bpe_tokenizer_trainer import BPETokenizerTrainer, TrainedBPETokenizer, train_bpe_tokenizer
    BPE_TRAINER_AVAILABLE = True
    print("✅ bpe_tokenizer_trainer.py をインポートしました")
except ImportError as e:
    BPE_TRAINER_AVAILABLE = False
    print(f"⚠️ bpe_tokenizer_trainer.py のインポートに失敗: {e}")

# トークナイザーモデルのパス
TOKENIZER_MODEL_PATH = "neuroq_tokenizer.model"

# チェックポイントパス（モデルサイズ別）
# 全モデルを/model_checkpointsから参照
MODEL_CHECKPOINT_PATHS = {
    'micro': "/model_checkpoints/neuroq_micro_checkpoint.pt",
    'small': "/model_checkpoints/neuroq_small_checkpoint.pt",
    'large': "/model_checkpoints/neuroq_large_checkpoint.pt",
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

# 学習状態管理
pretrain_process = None
pretrain_status = "idle"  # idle, running, completed, error
pretrain_log_file = "training_openai.log"

# 会話履歴管理
conversation_sessions = {}  # session_id -> list of {role, content}

# 設定
VOCAB_SIZE = 8000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# デフォルトシステムプロンプト（会話指示）
DEFAULT_SYSTEM_PROMPT = """あなたは親切で正確なアシスタントです。
以下のルールに従ってください：
1. ユーザーの質問に短く正確に答える
2. わからないことは質問する
3. 聞かれたことだけに答える（余計な情報を追加しない）
4. 前の文脈を踏まえて返答する"""

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
                  system_prompt: str = None) -> str:
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

    Returns:
        生成されたテキスト
    """
    global model, session_system_prompts

    if model is None:
        return "Error: Model not initialized"

    # モデルが未学習の場合（model.model が None または学習済みの NeuroQuantumBrain がない場合）
    if model.model is None:
        return "Error: Model not trained. Please run action='train' or action='pretrain_openai' first to train the model before generating text."

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

        # 履歴を含む完全なプロンプト
        history_context = "".join(context_parts)
        full_prompt = history_context + prompt if history_context else prompt

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

        # 会話履歴に保存
        save_conversation_turn(session_id, prompt, result)

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


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
    # STATUS CHECK
    # ========================================
    if action == "status":
        return {
            "status": "ok",
            "initialized": is_initialized,
            "device": DEVICE,
            "vocab_size": VOCAB_SIZE,
            "current_model_size": current_model_size,
            "available_model_sizes": ["micro", "small", "large"],
            "model_configs_available": MODEL_CONFIGS_AVAILABLE
        }
    
    # ========================================
    # GENERATE（モデルが必要な処理）
    # ========================================
    if action == "generate":
        # モデルサイズを取得
        model_size = job_input.get("model_size", "micro")

        # Lazy initialization（モデルサイズに応じて初期化）
        if not is_initialized or current_model_size != model_size:
            print(f"🔄 モデル初期化中 ({model_size.upper()})...")
            if not initialize_model(model_size):
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

        print(f"📝 Generate: model={model_size}, session='{session_id}', prompt='{prompt[:30]}...'")
        if system_prompt:
            print(f"📝 System prompt: '{system_prompt[:50]}...'")

        result = generate_text(
            prompt=prompt,
            max_length=max_length,
            temp_min=temp_min,
            temp_max=temp_max,
            temperature=temperature,
            session_id=session_id,
            system_prompt=system_prompt
        )

        return {
            "status": "success",
            "prompt": prompt,
            "generated": result,
            "session_id": session_id,
            "model_size": current_model_size,
            "system_prompt": session_system_prompts.get(session_id, DEFAULT_SYSTEM_PROMPT)
        }
    
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
        - epochs: エポック数
        - batch_size: バッチサイズ
        - lr: 学習率
        - seq_length: シーケンス長
        - texts: 学習テキスト（省略時はデフォルトデータ）
        - train_tokenizer: トークナイザーも学習するか（デフォルト: False）
        - tokenizer_vocab_size: トークナイザー語彙サイズ（デフォルト: 32000）
        - use_custom_tokenizer: カスタムトークナイザーを使用するか（デフォルト: False）
        """

        model_size = job_input.get("model_size", "micro")
        epochs = job_input.get("epochs", 30)
        batch_size = job_input.get("batch_size", 8)
        lr = job_input.get("lr", 0.0005)
        seq_length = job_input.get("seq_length", 64)
        texts = job_input.get("texts", None)
        train_tokenizer_flag = job_input.get("train_tokenizer", False)
        tokenizer_vocab_size = job_input.get("tokenizer_vocab_size", 32000)
        use_custom_tokenizer = job_input.get("use_custom_tokenizer", False)

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

                # トークナイザー初期化
                tokenizer = None
                tokenizer_type = 'sentencepiece'
                custom_tokenizer_path = None

                # カスタムトークナイザー学習
                if train_tokenizer_flag and BPE_TRAINER_AVAILABLE:
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

                # モデル設定を取得
                config = get_model_config(model_size)
                checkpoint_path = MODEL_CHECKPOINT_PATHS.get(model_size, MODEL_CHECKPOINT_PATHS['micro'])

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

                if tokenizer_type == 'custom_bpe':
                    result["tokenizer_path"] = custom_tokenizer_path
                    result["tokenizer_vocab_size"] = tokenizer.vocab_size

                return result
                
            except Exception as e:
                import traceback
                traceback.print_exc()
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

                    if os.path.exists(TOKENIZER_MODEL_PATH):
                        print(f"✅ トークナイザーをロード: {TOKENIZER_MODEL_PATH}")
                        model.tokenizer = NeuroQuantumTokenizer(
                            vocab_size=8000,
                            model_file=TOKENIZER_MODEL_PATH
                        )

                    is_initialized = True

                except Exception as e:
                    return {
                        "status": "error",
                        "error": f"Failed to create model: {e}"
                    }

            if texts is None:
                print("📚 デフォルト学習データを使用")
                texts = get_training_data()

            if not texts:
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
                    return {
                        "status": "success",
                        "message": f"Training completed ({epochs} epochs)",
                        "model_size": "micro",
                        "checkpoint_path": checkpoint_path,
                        "num_samples": len(texts)
                    }
                else:
                    return {
                        "status": "warning",
                        "message": "Training completed but checkpoint save failed",
                        "num_samples": len(texts)
                    }

            except Exception as e:
                import traceback
                traceback.print_exc()
                return {
                    "status": "error",
                    "error": str(e)
                }
    
    # ========================================
    # TRAIN_OASST1_JA（日本語会話データで学習）
    # ========================================
    if action == "train_oasst1_ja":
        """
        kunishou/oasst1-89k-ja を変換した User/Assistant 形式のデータで学習

        パラメータ:
        - data_file: 会話データファイルパス（デフォルト: data/oasst1_ja_conversations.txt）
        - tokenizer_model: トークナイザーモデルパス（デフォルト: neuroq_tokenizer_oasst1_ja.model）
        - epochs: エポック数（デフォルト: 5）
        - batch_size: バッチサイズ（デフォルト: 8）
        - seq_length: シーケンス長（デフォルト: 128）
        - lr: 学習率（デフォルト: 0.0003）
        - embed_dim: 埋め込み次元（デフォルト: 256）
        - num_heads: アテンションヘッド数（デフォルト: 4）
        - num_layers: レイヤー数（デフォルト: 4）
        - checkpoint_path: チェックポイント保存先（デフォルト: /model_checkpoints/oasst1_ja_model.pt）
        """
        import sentencepiece as spm

        # パラメータ取得
        data_file = job_input.get("data_file", "../data/oasst1_ja_conversations.txt")
        tokenizer_model = job_input.get("tokenizer_model", "../neuroq_tokenizer_oasst1_ja.model")
        epochs = job_input.get("epochs", 5)
        batch_size = job_input.get("batch_size", 8)
        seq_length = job_input.get("seq_length", 128)
        lr = job_input.get("lr", 0.0003)
        embed_dim = job_input.get("embed_dim", 256)
        num_heads = job_input.get("num_heads", 4)
        num_layers = job_input.get("num_layers", 4)
        ff_dim = job_input.get("ff_dim", 512)
        checkpoint_path = job_input.get("checkpoint_path", "/model_checkpoints/oasst1_ja_model.pt")

        print("=" * 70)
        print("日本語会話データ (oasst1-89k-ja) で学習開始")
        print("=" * 70)

        try:
            # 会話データ読み込み
            print(f"\n会話データ読み込み: {data_file}")

            if not os.path.exists(data_file):
                return {
                    "status": "error",
                    "error": f"Data file not found: {data_file}"
                }

            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 空行で区切られた会話ブロックに分割
            conversations = []
            blocks = content.split('\n\n')
            for block in blocks:
                block = block.strip()
                if block and 'User:' in block and 'Assistant:' in block:
                    conversations.append(block)

            print(f"  {len(conversations)} 個の会話を読み込みました")

            if not conversations:
                return {
                    "status": "error",
                    "error": "No valid conversations found in data file"
                }

            # トークナイザー読み込み
            print(f"\nトークナイザー読み込み: {tokenizer_model}")

            if not os.path.exists(tokenizer_model):
                return {
                    "status": "error",
                    "error": f"Tokenizer model not found: {tokenizer_model}"
                }

            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_model)
            vocab_size = sp.get_piece_size()
            eos_id = sp.eos_id()

            print(f"  語彙サイズ: {vocab_size}")

            # データ準備
            print("\nデータ準備...")
            all_tokens = []
            for conv in conversations:
                tokens = sp.encode(conv, out_type=int)
                all_tokens.extend(tokens)
                all_tokens.append(eos_id)

            all_tokens = torch.tensor(all_tokens, dtype=torch.long)
            print(f"  総トークン数: {len(all_tokens):,}")

            # シーケンス作成
            sequences = []
            step = max(1, seq_length // 2)
            for i in range(0, len(all_tokens) - seq_length - 1, step):
                x = all_tokens[i:i + seq_length]
                y = all_tokens[i + 1:i + seq_length + 1]
                if len(x) == seq_length and len(y) == seq_length:
                    sequences.append((x, y))

            print(f"  シーケンス数: {len(sequences):,}")

            # モデル作成
            print("\nモデル構築...")
            import torch.nn as nn

            class SimpleTransformerForAPI(nn.Module):
                def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout=0.1):
                    super().__init__()
                    self.vocab_size = vocab_size
                    self.embed_dim = embed_dim
                    self.token_embedding = nn.Embedding(vocab_size, embed_dim)
                    self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                        dropout=dropout, activation='gelu', batch_first=True
                    )
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.layer_norm = nn.LayerNorm(embed_dim)
                    self.output_layer = nn.Linear(embed_dim, vocab_size)
                    self._init_weights()

                def _init_weights(self):
                    nn.init.normal_(self.token_embedding.weight, std=0.02)
                    nn.init.normal_(self.position_embedding.weight, std=0.02)
                    nn.init.normal_(self.output_layer.weight, std=0.02)
                    nn.init.zeros_(self.output_layer.bias)

                def forward(self, x):
                    batch_size, seq_len = x.shape
                    device = x.device
                    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                    x = self.token_embedding(x) + self.position_embedding(positions)
                    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
                    mask = mask.masked_fill(mask == 1, float('-inf'))
                    x = self.transformer(x, mask=mask)
                    x = self.layer_norm(x)
                    return self.output_layer(x)

            train_model = SimpleTransformerForAPI(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
                max_seq_len=seq_length + 32,
                dropout=0.1
            ).to(DEVICE)

            num_params = sum(p.numel() for p in train_model.parameters() if p.requires_grad)
            print(f"  総パラメータ数: {num_params:,}")

            # 学習設定
            optimizer = torch.optim.AdamW(train_model.parameters(), lr=lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.CrossEntropyLoss()

            # 学習ループ
            print(f"\n学習ループ開始...")
            print(f"  エポック数: {epochs}")
            print(f"  バッチサイズ: {batch_size}")
            print(f"  シーケンス長: {seq_length}")
            print(f"  学習率: {lr}")

            import numpy as np
            best_loss = float('inf')
            training_log = []

            for epoch in range(epochs):
                train_model.train()
                total_loss = 0
                num_batches = 0

                indices = np.random.permutation(len(sequences))

                for i in range(0, len(sequences), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    if len(batch_indices) == 0:
                        continue

                    x_batch = torch.stack([sequences[idx][0] for idx in batch_indices]).to(DEVICE)
                    y_batch = torch.stack([sequences[idx][1] for idx in batch_indices]).to(DEVICE)

                    optimizer.zero_grad()
                    logits = train_model(x_batch)

                    loss = criterion(
                        logits.view(-1, vocab_size),
                        y_batch.view(-1)
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                scheduler.step()
                avg_loss = total_loss / max(1, num_batches)
                training_log.append({"epoch": epoch + 1, "loss": avg_loss})

                print(f"  Epoch {epoch + 1:3d}/{epochs}: Loss = {avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss

            # チェックポイント保存
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            checkpoint = {
                'model_state_dict': train_model.state_dict(),
                'config': {
                    'vocab_size': vocab_size,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'num_layers': num_layers,
                    'ff_dim': ff_dim,
                    'max_seq_len': seq_length + 32,
                },
                'tokenizer': {
                    'type': 'sentencepiece',
                    'path': tokenizer_model,
                    'vocab_size': vocab_size,
                },
                'training_info': {
                    'epochs': epochs,
                    'best_loss': best_loss,
                    'num_conversations': len(conversations),
                    'num_sequences': len(sequences),
                },
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"\nチェックポイント保存: {checkpoint_path}")

            print("\n" + "=" * 70)
            print("学習完了!")
            print("=" * 70)

            return {
                "status": "success",
                "message": f"Training completed ({epochs} epochs)",
                "checkpoint_path": checkpoint_path,
                "best_loss": best_loss,
                "num_conversations": len(conversations),
                "num_sequences": len(sequences),
                "num_parameters": num_params,
                "vocab_size": vocab_size,
                "training_log": training_log[-5:],  # 最後の5エポック
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
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
            "train",              # 学習（チェックポイント保存）
            "train_oasst1_ja",    # 日本語会話データ学習
            "generate",           # 推論（チェックポイントロード）
            "pretrain_openai",    # OpenAIデータセット事前学習
            "pretrain_status",    # 事前学習ステータス
            "clear_session",      # 会話履歴クリア
            "set_system_prompt",  # システムプロンプト設定
            "get_system_prompt"   # システムプロンプト取得
        ]
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
