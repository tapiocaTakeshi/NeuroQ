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
    from neuroquantum_brain import NeuroQuantumBrainAI, get_training_data
    from neuroquantum_layered import NeuroQuantumTokenizer
    print("✅ neuroquantum_brain.py をインポートしました")
    print("✅ neuroquantum_layered.py をインポートしました")
except ImportError as e:
    print(f"❌ インポートに失敗: {e}")
    NeuroQuantumBrainAI = None
    NeuroQuantumTokenizer = None

# トークナイザーモデルのパス
TOKENIZER_MODEL_PATH = "neuroq_tokenizer.model"

# ========================================
# グローバル変数（起動時は全てNone）
# ========================================
model = None  # NeuroQuantumBrainAI インスタンス
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
# Lazy Model Loading（初回リクエスト時のみ）
# ========================================
def initialize_model():
    """モデルを初期化（初回リクエスト時のみ呼ばれる）"""
    global model, is_initialized

    if is_initialized:
        return True

    print("🔄 モデル初期化開始...")

    try:
        if NeuroQuantumBrainAI is None:
            raise ImportError("NeuroQuantumBrainAI がインポートできていません")

        # NeuroQuantumBrainAI を使用してモデルを構築
        print("📦 NeuroQuantumBrainAI を使用してモデルを構築...")
        
        model = NeuroQuantumBrainAI(
            embed_dim=128,
            num_heads=4,
            num_layers=3,
            num_neurons=100,
            max_vocab=8000,
            use_sentencepiece=True  # 学習済みSentencePieceモデルを使用
        )

        # トークナイザーを明示的にロード
        if os.path.exists(TOKENIZER_MODEL_PATH):
            print(f"✅ トークナイザーをロード: {TOKENIZER_MODEL_PATH}")
            model.tokenizer = NeuroQuantumTokenizer(
                vocab_size=8000,
                model_file=TOKENIZER_MODEL_PATH
            )
            print(f"   語彙サイズ: {model.tokenizer.actual_vocab_size or model.tokenizer.vocab_size}")
        else:
            print(f"⚠️ トークナイザーファイルが見つかりません: {TOKENIZER_MODEL_PATH}")
        
        # 学習データを取得して学習
        print("🔄 学習開始...")
        training_data = get_training_data()
        print(f"📚 学習データ: {len(training_data)} サンプル")
        
        model.train(training_data, epochs=25, batch_size=16, lr=0.002, seq_length=48)
        
        is_initialized = True
        print("✅ モデル初期化完了!")
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
                  temperature: float = None, session_id: str = "default") -> str:
    """
    テキスト生成（会話対応版 - NeuroQuantumBrainAI使用）

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
        # NeuroQuantumBrainAI.generate() は内部で <USER>...<ASSISTANT> フォーマットを処理する
        # ただし、履歴を追加するためにここでプレフィックスを構築
        history = conversation_sessions.get(session_id, [])[-4:]  # 最新4ターン
        
        history_context = ""
        for turn in history:
            if turn["role"] == "user":
                history_context += f"<USER>{turn['content']}"
            elif turn["role"] == "assistant":
                history_context += f"<ASSISTANT>{turn['content']}"
        
        # 履歴を含む完全なプロンプト
        full_prompt = history_context + prompt if history_context else prompt

        # 生成実行（NeuroQuantumBrainAI.generate()を使用）
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
            model.train(texts, epochs=epochs, seq_length=32)
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
