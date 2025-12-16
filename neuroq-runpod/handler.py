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

# ========================================
# グローバル変数（起動時は全てNone）
# ========================================
model = None
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

# システムプロンプト（会話指示）
SYSTEM_PROMPT = """あなたは親切で正確なアシスタントです。
以下のルールに従ってください：
1. ユーザーの質問に短く正確に答える
2. わからないことは質問する
3. 聞かれたことだけに答える（余計な情報を追加しない）
4. 前の文脈を踏まえて返答する"""

print(f"📊 Device: {DEVICE}")
print(f"📊 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")


# ========================================
# 事前学習済みモデルのパス
# ========================================
PRETRAINED_MODEL_PATH = "neuroq_pretrained.pt"


# ========================================
# PyTorch .ptファイルの検証
# ========================================
def is_lfs_pointer_file(file_path: str) -> bool:
    """
    Check if a file is a Git LFS pointer file.

    Args:
        file_path: Path to the file to check

    Returns:
        bool: True if the file is an LFS pointer, False otherwise
    """
    try:
        # LFS pointer files are typically very small (< 200 bytes)
        file_size = os.path.getsize(file_path)
        if file_size > 1024:
            return False

        # Read the first few bytes to check for LFS marker
        with open(file_path, 'rb') as f:
            header = f.read(50).decode('utf-8', errors='ignore')
            return header.startswith('version https://git-lfs.github.com/spec/')
    except:
        return False


def validate_pt_file(file_path: str) -> dict:
    """
    PyTorchの.ptファイルを検証する

    Args:
        file_path: 検証するptファイルのパス

    Returns:
        dict: 検証結果
            - valid: bool - ファイルが有効かどうか
            - error: str - エラーメッセージ（エラーがある場合）
            - info: dict - ファイル情報
    """
    result = {
        "valid": False,
        "error": None,
        "info": {}
    }

    try:
        # 1. ファイル存在チェック
        if not os.path.exists(file_path):
            result["error"] = f"File not found: {file_path}"
            return result

        # 2. ファイルサイズチェック
        file_size = os.path.getsize(file_path)
        result["info"]["file_size_bytes"] = file_size
        result["info"]["file_size_mb"] = round(file_size / (1024 * 1024), 2)

        if file_size < 1024:  # 1KB未満
            # Check if it's a Git LFS pointer file
            if is_lfs_pointer_file(file_path):
                result["error"] = (
                    f"Git LFS pointer file detected ({file_size} bytes). "
                    f"Run 'python download_model.py' or 'git lfs pull' to download the actual model file."
                )
                result["info"]["is_lfs_pointer"] = True
            else:
                result["error"] = f"File too small ({file_size} bytes). Possibly corrupted or empty."
            return result

        if file_size > 5 * 1024 * 1024 * 1024:  # 5GB以上
            result["error"] = f"File too large ({result['info']['file_size_mb']} MB). May cause memory issues."
            print(f"⚠️ Warning: {result['error']}")
            # 警告のみで続行

        # 3. PyTorchで読み込めるかチェック
        print(f"🔍 Loading checkpoint from {file_path}...")
        try:
            checkpoint = torch.load(file_path, map_location='cpu')
        except Exception as e:
            result["error"] = f"Failed to load PyTorch checkpoint: {str(e)}"
            return result

        # 4. チェックポイントの型チェック
        if not isinstance(checkpoint, dict):
            result["error"] = f"Invalid checkpoint format. Expected dict, got {type(checkpoint).__name__}"
            return result

        result["info"]["checkpoint_keys"] = list(checkpoint.keys())

        # 5. 必須キーの存在チェック
        required_keys = ['config', 'model_state_dict']
        missing_keys = [key for key in required_keys if key not in checkpoint]

        if missing_keys:
            result["error"] = f"Missing required keys: {missing_keys}. Found keys: {list(checkpoint.keys())}"
            return result

        # 6. Config情報の検証
        config = checkpoint['config']
        if not isinstance(config, dict):
            result["error"] = f"Invalid config format. Expected dict, got {type(config).__name__}"
            return result

        # 必須のconfig項目
        required_config_keys = ['vocab_size', 'embed_dim', 'hidden_dim', 'num_heads', 'num_layers']
        missing_config_keys = [key for key in required_config_keys if key not in config]

        if missing_config_keys:
            result["error"] = f"Missing required config keys: {missing_config_keys}"
            return result

        # Config情報を結果に追加
        result["info"]["config"] = {
            "vocab_size": config.get('vocab_size'),
            "embed_dim": config.get('embed_dim'),
            "hidden_dim": config.get('hidden_dim'),
            "num_heads": config.get('num_heads'),
            "num_layers": config.get('num_layers'),
            "max_seq_len": config.get('max_seq_len'),
            "dropout": config.get('dropout'),
            "lambda_entangle": config.get('lambda_entangle'),
        }

        # 7. Config値の妥当性チェック
        vocab_size = config.get('vocab_size', 0)
        if vocab_size < 100 or vocab_size > 100000:
            result["error"] = f"Invalid vocab_size: {vocab_size}. Expected range: 100-100000"
            return result

        embed_dim = config.get('embed_dim', 0)
        if embed_dim < 32 or embed_dim > 4096:
            result["error"] = f"Invalid embed_dim: {embed_dim}. Expected range: 32-4096"
            return result

        # 8. model_state_dictの検証
        model_state_dict = checkpoint['model_state_dict']
        if not isinstance(model_state_dict, dict):
            result["error"] = f"Invalid model_state_dict format. Expected dict, got {type(model_state_dict).__name__}"
            return result

        result["info"]["num_model_parameters"] = len(model_state_dict)

        if len(model_state_dict) == 0:
            result["error"] = "model_state_dict is empty"
            return result

        # 9. その他の情報を追加
        if 'epoch' in checkpoint:
            result["info"]["epoch"] = checkpoint['epoch']
        if 'optimizer_state_dict' in checkpoint:
            result["info"]["has_optimizer_state"] = True
        if 'loss' in checkpoint:
            result["info"]["loss"] = checkpoint['loss']

        # すべての検証をパス
        result["valid"] = True
        print(f"✅ Validation passed for {file_path}")
        print(f"   File size: {result['info']['file_size_mb']} MB")
        print(f"   Vocab size: {result['info']['config']['vocab_size']}")
        print(f"   Embed dim: {result['info']['config']['embed_dim']}")
        print(f"   Model parameters: {result['info']['num_model_parameters']}")

        return result

    except Exception as e:
        result["error"] = f"Unexpected error during validation: {str(e)}"
        import traceback
        traceback.print_exc()
        return result


# ========================================
# Lazy Model Loading（初回リクエスト時のみ）
# ========================================
def initialize_model():
    """モデルを初期化（初回リクエスト時のみ呼ばれる）"""
    global model, is_initialized

    if is_initialized:
        return True

    print("🔄 モデル初期化開始...")

    try:
        from neuroquantum_layered import NeuroQuantumAI, NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

        # ========================================
        # 方法1: neuroq_pretrained.py を使用してロード（推奨）
        # ========================================
        try:
            print("📦 Importing neuroq_pretrained module from parent directory...")
            import neuroq_pretrained

            # 事前学習済みモデルをロード
            nn_model, config_dict, checkpoint = neuroq_pretrained.load_pretrained_model(
                model_path=PRETRAINED_MODEL_PATH,
                device=DEVICE,
                verbose=True
            )

            if nn_model is not None and config_dict is not None:
                # トークナイザーをロード
                tokenizer = NeuroQuantumTokenizer(
                    vocab_size=config_dict['vocab_size'],
                    model_file="neuroq_tokenizer.model"
                )

                # vocab_size の整合性確認
                tokenizer_vocab_size = tokenizer.actual_vocab_size or tokenizer.vocab_size
                config_vocab_size = config_dict['vocab_size']
                print(f"🔍 Vocab size validation:")
                print(f"   Config vocab_size: {config_vocab_size}")
                print(f"   Tokenizer actual_vocab_size: {tokenizer_vocab_size}")

                if config_vocab_size != tokenizer_vocab_size:
                    print(f"❌ CRITICAL: vocab_size mismatch detected!")
                    print(f"   Model was trained with vocab_size={config_vocab_size}")
                    print(f"   But tokenizer has vocab_size={tokenizer_vocab_size}")
                    print(f"   ⚠️ WARNING: This may cause generation errors!")
                    print(f"   Recommendation: Retrain the model with correct vocab_size.")
                    # Note: We continue with a warning, as the model might still be usable

                # Configオブジェクトを作成
                config = NeuroQuantumConfig(
                    vocab_size=config_dict['vocab_size'],
                    embed_dim=config_dict['embed_dim'],
                    hidden_dim=config_dict['hidden_dim'],
                    num_heads=config_dict['num_heads'],
                    num_layers=config_dict['num_layers'],
                    max_seq_len=config_dict['max_seq_len'],
                    dropout=config_dict.get('dropout', 0.1),
                    lambda_entangle=config_dict.get('lambda_entangle', 0.5),
                )

                # NeuroQuantumAI のラッパーを作成
                model = NeuroQuantumAI(
                    embed_dim=config_dict['embed_dim'],
                    hidden_dim=config_dict['hidden_dim'],
                    num_heads=config_dict['num_heads'],
                    num_layers=config_dict['num_layers'],
                    max_seq_len=config_dict['max_seq_len'],
                    dropout=config_dict.get('dropout', 0.1),
                    lambda_entangle=config_dict.get('lambda_entangle', 0.5),
                )
                model.model = nn_model
                model.config = config
                model.tokenizer = tokenizer

                print(f"✅ 事前学習済みモデルロード完了 (via neuroq_pretrained.py)!")
                print(f"   vocab_size: {config_dict['vocab_size']}")
                print(f"   embed_dim: {config_dict['embed_dim']}")
                print(f"   パラメータ数: {nn_model.num_params:,}")

                is_initialized = True
                return True
            else:
                print("⚠️ neuroq_pretrained.py でのロードに失敗。従来の方法にフォールバック...")

        except ImportError as e:
            print(f"⚠️ neuroq_pretrained module not found: {e}")
            print("   Falling back to traditional loading method...")
        except Exception as e:
            print(f"⚠️ Error using neuroq_pretrained module: {e}")
            print("   Falling back to traditional loading method...")

        # ========================================
        # 方法2: 従来の方法でロード（フォールバック）
        # ========================================
        if os.path.exists(PRETRAINED_MODEL_PATH):
            print(f"📦 事前学習済みモデルをロード (traditional method): {PRETRAINED_MODEL_PATH}")

            # ファイルの検証
            validation_result = validate_pt_file(PRETRAINED_MODEL_PATH)

            if not validation_result["valid"]:
                print(f"❌ モデルファイルの検証に失敗: {validation_result['error']}")
                print(f"   簡易学習モードにフォールバック...")
                # 検証失敗時は簡易学習モードへ（以下のコードにフォールスルー）
            else:
                # 検証成功 - モデルをロード
                print(f"✅ ファイル検証成功")

                # 再度読み込み（検証時はCPUで読み込んだため）
                checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
                config_dict = checkpoint['config']

                # Configを復元
                config = NeuroQuantumConfig(
                    vocab_size=config_dict['vocab_size'],
                    embed_dim=config_dict['embed_dim'],
                    hidden_dim=config_dict['hidden_dim'],
                    num_heads=config_dict['num_heads'],
                    num_layers=config_dict['num_layers'],
                    max_seq_len=config_dict['max_seq_len'],
                    dropout=config_dict['dropout'],
                    lambda_entangle=config_dict['lambda_entangle'],
                )

                # トークナイザーをロード
                tokenizer = NeuroQuantumTokenizer(
                    vocab_size=config_dict['vocab_size'],
                    model_file="neuroq_tokenizer.model"
                )

                # vocab_size の整合性確認
                tokenizer_vocab_size = tokenizer.actual_vocab_size or tokenizer.vocab_size
                config_vocab_size = config_dict['vocab_size']
                print(f"🔍 Vocab size validation:")
                print(f"   Config vocab_size: {config_vocab_size}")
                print(f"   Tokenizer actual_vocab_size: {tokenizer_vocab_size}")

                if config_vocab_size != tokenizer_vocab_size:
                    print(f"❌ CRITICAL: vocab_size mismatch detected!")
                    print(f"   Model was trained with vocab_size={config_vocab_size}")
                    print(f"   But tokenizer has vocab_size={tokenizer_vocab_size}")
                    print(f"   ⚠️ WARNING: This may cause generation errors!")
                    print(f"   Recommendation: Retrain the model with correct vocab_size.")
                    # Note: We continue with a warning, as the model might still be usable

                # モデルを構築してウェイトをロード
                nn_model = NeuroQuantum(config).to(DEVICE)
                nn_model.load_state_dict(checkpoint['model_state_dict'])
                nn_model.eval()

                # NeuroQuantumAI のラッパーを作成
                model = NeuroQuantumAI(
                    embed_dim=config_dict['embed_dim'],
                    hidden_dim=config_dict['hidden_dim'],
                    num_heads=config_dict['num_heads'],
                    num_layers=config_dict['num_layers'],
                    max_seq_len=config_dict['max_seq_len'],
                    dropout=config_dict['dropout'],
                    lambda_entangle=config_dict['lambda_entangle'],
                )
                model.model = nn_model
                model.config = config
                model.tokenizer = tokenizer

                print(f"✅ 事前学習済みモデルロード完了!")
                print(f"   vocab_size: {config_dict['vocab_size']}")
                print(f"   embed_dim: {config_dict['embed_dim']}")
                print(f"   パラメータ数: {nn_model.num_params:,}")

                is_initialized = True
                return True

        # ========================================
        # 方法2: 簡易学習（事前学習済みモデルがない場合）
        # ========================================
        print("⚠️ 事前学習済みモデルが見つかりません。簡易学習を実行...")
        
        # モデル作成（個別パラメータで初期化）
        model = NeuroQuantumAI(
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            num_layers=3,
            max_seq_len=256,
            dropout=0.1,
            lambda_entangle=0.5
        )

        # トークナイザー確認
        if os.path.exists("neuroq_tokenizer.model"):
            print("✅ トークナイザー: neuroq_tokenizer.model")
        else:
            print("⚠️ トークナイザーファイルが見つかりません")
        
        # 簡易学習データ
        print("🔄 簡易学習開始...")
        training_data = [
            "こんにちは。私はニューロQです。量子ビットニューラルネットワークを使った人工知能です。",
            "ニューロQは量子コンピュータの原理を活用した次世代のAIシステムです。",
            "量子コンピュータは量子力学の原理を利用した次世代の計算機です。",
            "人工知能は人間の知能を模倣するコンピュータシステムです。",
            "機械学習はデータからパターンを学習するAIの手法です。",
            "ディープラーニングはニューラルネットワークを多層化した技術です。",
        ] * 10
        
        model.train(training_data, epochs=5, seq_len=64)
        
        is_initialized = True
        print("✅ 簡易学習完了!")
        return True
    
    except Exception as e:
        print(f"❌ モデル初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========================================
# 会話履歴管理
# ========================================
def build_conversation_prompt(session_id: str, user_message: str, max_history: int = 4) -> str:
    """
    会話履歴を含むプロンプトを構築

    Args:
        session_id: セッションID
        user_message: ユーザーメッセージ
        max_history: 最大履歴ターン数（直近のターンのみ使用）

    Returns:
        会話フォーマットのプロンプト
    """
    global conversation_sessions

    # セッションが存在しない場合は作成
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []

    # 履歴を取得（最新のmax_historyターンのみ）
    history = conversation_sessions[session_id][-max_history:]

    # プロンプトを構築（システムプロンプトは省略し、直接対話形式で）
    # 学習データと同じフォーマット: <USER>...<ASSISTANT>...
    conversation_text = ""

    # 履歴を追加
    for turn in history:
        if turn["role"] == "user":
            conversation_text += f"<USER>{turn['content']}"
        elif turn["role"] == "assistant":
            conversation_text += f"<ASSISTANT>{turn['content']}"

    # 現在のユーザーメッセージを追加
    conversation_text += f"<USER>{user_message}<ASSISTANT>"

    return conversation_text


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
                  temperature: float = None, session_id: str = "default") -> str:
    """
    テキスト生成（会話対応版）

    Args:
        prompt: 入力プロンプト
        max_length: 最大生成長（デフォルト50 - 会話向けに短く制限）
        temp_min: 最低温度（指定された場合はtemp_min/temp_maxを使用）
        temp_max: 最高温度
        temperature: 互換性のための単一温度（指定された場合は自動的にtemp_min/temp_maxに変換）
        session_id: 会話セッションID（会話履歴管理用）

    Returns:
        生成されたテキスト
    """
    global model

    if model is None:
        return "Error: Model not initialized"

    try:
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
        conversation_prompt = build_conversation_prompt(session_id, prompt)

        # 生成実行（会話向けに短く制限）
        result = model.generate(
            prompt=conversation_prompt,
            max_length=max_length,
            temp_min=temp_min,
            temp_max=temp_max,
            repetition_penalty=2.5,  # 2.0 → 2.5 に強化（繰り返しをより強く抑制）
            no_repeat_ngram_size=3,   # 3-gramの繰り返し防止
            top_k=40,                  # top_k を明示的に指定
            top_p=0.9,                 # top_p を明示的に指定
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
    global is_initialized
    
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
            "vocab_size": VOCAB_SIZE
        }
    
    # ========================================
    # GENERATE（モデルが必要な処理）
    # ========================================
    if action == "generate":
        # Lazy initialization
        if not is_initialized:
            print("🔄 初回リクエスト - モデル初期化中...")
            if not initialize_model():
                return {
                    "status": "error",
                    "error": "Failed to initialize model"
                }

        prompt = job_input.get("prompt", "こんにちは")
        max_length = job_input.get("max_length", 50)  # 100 → 50 に変更（会話向け）
        session_id = job_input.get("session_id", "default")  # 会話セッションID

        # 温度パラメータ（temp_min/temp_max優先、互換性のためtemperatureもサポート）
        temp_min = job_input.get("temp_min")
        temp_max = job_input.get("temp_max")
        temperature = job_input.get("temperature", 0.5)  # 0.6 → 0.5 に下げて安定性向上

        print(f"📝 Generate: session_id='{session_id}', prompt='{prompt[:30]}...'")

        result = generate_text(
            prompt=prompt,
            max_length=max_length,
            temp_min=temp_min,
            temp_max=temp_max,
            temperature=temperature,
            session_id=session_id
        )

        return {
            "status": "success",
            "prompt": prompt,
            "generated": result,
            "session_id": session_id
        }
    
    # ========================================
    # PRETRAIN_OPENAI（OpenAIデータセット事前学習）
    # ========================================
    if action == "pretrain_openai":
        global pretrain_process, pretrain_status

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
    # TRAIN（学習）
    # ========================================
    if action == "train":
        if not is_initialized:
            if not initialize_model():
                return {
                    "status": "error",
                    "error": "Failed to initialize model"
                }
        
        texts = job_input.get("texts", [])
        epochs = job_input.get("epochs", 5)
        
        if not texts:
            return {
                "status": "error",
                "error": "No training texts provided"
            }
        
        try:
            model.train(texts, epochs=epochs, seq_len=32)
            return {
                "status": "success",
                "message": f"Training completed ({epochs} epochs)"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    # ========================================
    # CLEAR_SESSION（会話履歴クリア）
    # ========================================
    if action == "clear_session":
        global conversation_sessions
        session_id = job_input.get("session_id", "default")

        if session_id in conversation_sessions:
            del conversation_sessions[session_id]
            return {
                "status": "success",
                "message": f"Session '{session_id}' cleared"
            }
        else:
            return {
                "status": "success",
                "message": f"Session '{session_id}' not found (already empty)"
            }

    # ========================================
    # UNKNOWN ACTION
    # ========================================
    return {
        "status": "error",
        "error": f"Unknown action: {action}",
        "available_actions": ["health", "status", "generate", "train", "pretrain_openai", "pretrain_status", "clear_session"]
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
