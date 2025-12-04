#!/usr/bin/env python3
"""
NeuroQ RunPod Serverless Handler
================================
RunPod Serverless Endpoint用のハンドラー

サポートモード:
- Brain Mode: 脳型散在QBNN
- Layered Mode: 層状QBNN-Transformer

使用方法:
1. このファイルと neuroq_model.py をRunPodにデプロイ
2. モデルファイル (neuroq_model.pt, neuroq_tokenizer.json) を配置
3. RunPod Endpoint から呼び出し

API パラメータ:
- 生成パラメータ: prompt, max_tokens, temperature, top_k, top_p
- モデル設定: mode, num_neurons, hidden_dim, connection_density, etc.
"""

import runpod
import torch
import os
import traceback
import json

from neuroq_model import (
    NeuroQGenerator, 
    NeuroQModel, 
    NeuroQTokenizer, 
    NeuroQConfig,
    create_neuroq_brain,
    create_neuroq_layered,
)

# ========================================
# グローバル設定
# ========================================

MODEL_PATH = os.environ.get("NEUROQ_MODEL_PATH", "neuroq_model.pt")
TOKENIZER_PATH = os.environ.get("NEUROQ_TOKENIZER_PATH", "neuroq_tokenizer.json")
DEFAULT_MODE = os.environ.get("NEUROQ_MODE", "layered")  # 'brain' or 'layered'

# デフォルトのモデル設定
DEFAULT_CONFIG = {
    # 共通
    "embed_dim": int(os.environ.get("NEUROQ_EMBED_DIM", "128")),
    "num_layers": int(os.environ.get("NEUROQ_NUM_LAYERS", "3")),
    "dropout": float(os.environ.get("NEUROQ_DROPOUT", "0.1")),
    "max_seq_len": int(os.environ.get("NEUROQ_MAX_SEQ_LEN", "256")),
    
    # Brain Mode
    "num_neurons": int(os.environ.get("NEUROQ_NUM_NEURONS", "100")),
    "connection_density": float(os.environ.get("NEUROQ_CONNECTION_DENSITY", "0.25")),
    "lambda_entangle_brain": float(os.environ.get("NEUROQ_LAMBDA_BRAIN", "0.35")),
    
    # Layered Mode
    "hidden_dim": int(os.environ.get("NEUROQ_HIDDEN_DIM", "256")),
    "num_heads": int(os.environ.get("NEUROQ_NUM_HEADS", "4")),
    "lambda_entangle_layered": float(os.environ.get("NEUROQ_LAMBDA_LAYERED", "0.5")),
}

# デフォルトの学習設定
DEFAULT_TRAINING_CONFIG = {
    # エポック・ステップ関連
    "epochs": int(os.environ.get("NEUROQ_EPOCHS", "10")),
    "batch_size": int(os.environ.get("NEUROQ_BATCH_SIZE", "32")),
    "gradient_accumulation_steps": int(os.environ.get("NEUROQ_GRAD_ACCUM_STEPS", "1")),
    
    # 学習率関連
    "learning_rate": float(os.environ.get("NEUROQ_LEARNING_RATE", "1e-4")),
    "min_learning_rate": float(os.environ.get("NEUROQ_MIN_LR", "1e-6")),
    "weight_decay": float(os.environ.get("NEUROQ_WEIGHT_DECAY", "0.01")),
    "warmup_steps": int(os.environ.get("NEUROQ_WARMUP_STEPS", "100")),
    "lr_scheduler": os.environ.get("NEUROQ_LR_SCHEDULER", "cosine"),  # cosine, linear, constant
    
    # Temperature関連（生成時のサンプリング温度スケジューリング）
    "max_temperature": float(os.environ.get("NEUROQ_MAX_TEMPERATURE", "1.5")),
    "min_temperature": float(os.environ.get("NEUROQ_MIN_TEMPERATURE", "0.1")),
    "temperature_decay": os.environ.get("NEUROQ_TEMP_DECAY", "linear"),  # linear, exponential, cosine
    
    # 正則化・最適化
    "max_grad_norm": float(os.environ.get("NEUROQ_MAX_GRAD_NORM", "1.0")),
    "label_smoothing": float(os.environ.get("NEUROQ_LABEL_SMOOTHING", "0.1")),
    
    # 評価・保存
    "eval_steps": int(os.environ.get("NEUROQ_EVAL_STEPS", "500")),
    "save_steps": int(os.environ.get("NEUROQ_SAVE_STEPS", "1000")),
    "logging_steps": int(os.environ.get("NEUROQ_LOGGING_STEPS", "100")),
    
    # 早期停止
    "early_stopping_patience": int(os.environ.get("NEUROQ_EARLY_STOPPING", "3")),
    "early_stopping_threshold": float(os.environ.get("NEUROQ_EARLY_STOPPING_THRESHOLD", "0.001")),
    
    # データ関連
    "train_split": float(os.environ.get("NEUROQ_TRAIN_SPLIT", "0.9")),
    "shuffle": True,
    "seed": int(os.environ.get("NEUROQ_SEED", "42")),
}

# デバイス選択
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🎮 CUDA GPU を使用: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🍎 Apple Silicon GPU (MPS) を使用")
else:
    DEVICE = "cpu"
    print("💻 CPU を使用")

# ========================================
# モデルキャッシュ
# ========================================

# モードごとにモデルをキャッシュ
model_cache = {}

def get_config_key(config_params: dict) -> str:
    """設定からキャッシュキーを生成"""
    return json.dumps(config_params, sort_keys=True)


def load_model(mode: str = None, config_params: dict = None):
    """
    モデルをロード（キャッシュあり）
    
    Args:
        mode: 'brain' or 'layered' (Noneの場合はDEFAULT_MODEを使用)
        config_params: カスタム設定（指定されていない場合はデフォルト使用）
    """
    global model_cache
    
    if mode is None:
        mode = DEFAULT_MODE
    
    # 設定をマージ
    params = DEFAULT_CONFIG.copy()
    if config_params:
        params.update(config_params)
    
    # キャッシュキーを生成
    cache_key = f"{mode}_{get_config_key(params)}"
    
    # キャッシュにあればそれを返す
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    print(f"📥 NeuroQ モデルをロード中...")
    print(f"   Mode: {mode}")
    print(f"   Config: {json.dumps(params, indent=2)}")
    print(f"   Device: {DEVICE}")
    
    # モード別のモデルファイルを探す
    mode_model_path = f"neuroq_{mode}_model.pt"
    actual_model_path = mode_model_path if os.path.exists(mode_model_path) else MODEL_PATH
    
    # ファイルが存在するか確認
    if not os.path.exists(actual_model_path):
        print(f"⚠️ モデルファイルが見つかりません: {actual_model_path}")
        print("   カスタム設定でモデルを作成します")
        
        # カスタム設定でモデルを作成
        if mode == 'brain':
            config = NeuroQConfig(
                mode='brain',
                vocab_size=2000,
                embed_dim=params['embed_dim'],
                num_neurons=params['num_neurons'],
                hidden_dim=params['num_neurons'] * 2,
                num_heads=params.get('num_heads', 4),
                num_layers=params['num_layers'],
                max_seq_len=params['max_seq_len'],
                dropout=params['dropout'],
                connection_density=params['connection_density'],
                lambda_entangle=params['lambda_entangle_brain'],
            )
        else:  # layered
            config = NeuroQConfig(
                mode='layered',
                vocab_size=2000,
                embed_dim=params['embed_dim'],
                hidden_dim=params['hidden_dim'],
                num_heads=params['num_heads'],
                num_layers=params['num_layers'],
                max_seq_len=params['max_seq_len'],
                dropout=params['dropout'],
                lambda_entangle=params['lambda_entangle_layered'],
            )
        
        model = NeuroQModel(config)
        tokenizer = NeuroQTokenizer(vocab_size=2000)
        
        # 基本的な語彙を構築
        basic_texts = [
            "こんにちは、私はNeuroQです。量子ビットニューラルネットワークベースの生成AIです。",
            "Hello, I am NeuroQ. A generative AI based on Quantum-Bit Neural Network.",
            "量子コンピュータは次世代の計算技術です。",
            "人工知能は私たちの生活を変革しています。",
            "QBNNは量子もつれを模倣したニューラルネットワークです。",
        ]
        tokenizer.build_vocab(basic_texts)
        
        generator = NeuroQGenerator(model, tokenizer, DEVICE)
        print(f"✅ カスタムモデル作成完了 (Mode: {mode})")
    else:
        # 学習済みモデルをロード
        generator = NeuroQGenerator.load(actual_model_path, TOKENIZER_PATH, DEVICE)
        print(f"✅ モデルロード完了: {actual_model_path}")
    
    # モデル情報を表示
    info = generator.get_model_info()
    print(f"   モード: {info.get('mode', 'unknown')}")
    print(f"   パラメータ数: {info['num_params']:,}")
    print(f"   埋め込み次元: {info['embed_dim']}")
    print(f"   隠れ層次元: {info['hidden_dim']}")
    print(f"   ニューロン数: {info.get('num_neurons', 'N/A')}")
    print(f"   レイヤー数: {info['num_layers']}")
    
    # キャッシュに保存
    model_cache[cache_key] = generator
    
    return generator


# ========================================
# RunPod Handler
# ========================================

def handler(job):
    """
    RunPod Serverless ハンドラー
    
    入力JSON形式:
    {
        "input": {
            // === 生成パラメータ ===
            "prompt": "生成したいテキストのプロンプト",  // 必須
            "max_tokens": 128,        // オプション（デフォルト: 128）
            "temperature": 0.7,       // オプション（デフォルト: 0.7）
            "top_k": 40,              // オプション（デフォルト: 40）
            "top_p": 0.9,             // オプション（デフォルト: 0.9）
            "repetition_penalty": 1.2 // オプション（デフォルト: 1.2）
            
            // === モデル設定 ===
            "mode": "brain",          // オプション: "brain" or "layered"
            
            // Brain Mode 専用
            "num_neurons": 100,       // ニューロン数
            "connection_density": 0.25, // 接続密度 (0.0-1.0)
            "lambda_entangle": 0.35,  // 量子もつれ強度
            
            // Layered Mode 専用
            "hidden_dim": 256,        // 隠れ層次元
            "num_heads": 4,           // アテンションヘッド数
            
            // 共通
            "embed_dim": 128,         // 埋め込み次元
            "num_layers": 3,          // レイヤー数
            
            // === 学習パラメータ ===
            "epochs": 10,             // エポック数（デフォルト: 10）
            "batch_size": 32,         // バッチサイズ（デフォルト: 32）
            "learning_rate": 1e-4,    // 学習率（デフォルト: 1e-4）
            "min_learning_rate": 1e-6, // 最小学習率（デフォルト: 1e-6）
            "weight_decay": 0.01,     // 重み減衰（デフォルト: 0.01）
            "warmup_steps": 100,      // ウォームアップステップ（デフォルト: 100）
            "lr_scheduler": "cosine", // LRスケジューラー: cosine, linear, constant
            "max_temperature": 1.5,   // 最大温度（デフォルト: 1.5）
            "min_temperature": 0.1,   // 最小温度（デフォルト: 0.1）
            "temperature_decay": "linear", // 温度減衰: linear, exponential, cosine
            "max_grad_norm": 1.0,     // 勾配クリッピング（デフォルト: 1.0）
            "label_smoothing": 0.1,   // ラベルスムージング（デフォルト: 0.1）
            "gradient_accumulation_steps": 1, // 勾配累積ステップ
            "eval_steps": 500,        // 評価間隔ステップ
            "save_steps": 1000,       // 保存間隔ステップ
            "logging_steps": 100,     // ログ間隔ステップ
            "early_stopping_patience": 3,    // 早期停止patience
            "early_stopping_threshold": 0.001, // 早期停止閾値
            "train_split": 0.9,       // 学習/検証分割比率
            "seed": 42,               // 乱数シード
        }
    }
    
    出力JSON形式:
    {
        "prompt": "入力プロンプト",
        "output": "生成されたテキスト",
        "model_info": {
            "mode": "brain" or "layered",
            "num_neurons": 100,
            "num_params": 123456,
            ...
        }
    }
    """
    try:
        # 入力を取得
        job_input = job.get("input", {})
        
        # モード設定
        mode = job_input.get("mode", DEFAULT_MODE)
        
        # モデル設定パラメータを抽出
        config_params = {}
        
        # 共通パラメータ
        if "embed_dim" in job_input:
            config_params["embed_dim"] = int(job_input["embed_dim"])
        if "num_layers" in job_input:
            config_params["num_layers"] = int(job_input["num_layers"])
        if "dropout" in job_input:
            config_params["dropout"] = float(job_input["dropout"])
        if "max_seq_len" in job_input:
            config_params["max_seq_len"] = int(job_input["max_seq_len"])
        
        # Brain Mode 専用
        if "num_neurons" in job_input:
            config_params["num_neurons"] = int(job_input["num_neurons"])
        if "connection_density" in job_input:
            config_params["connection_density"] = float(job_input["connection_density"])
        if "lambda_entangle" in job_input and mode == "brain":
            config_params["lambda_entangle_brain"] = float(job_input["lambda_entangle"])
        
        # Layered Mode 専用
        if "hidden_dim" in job_input:
            config_params["hidden_dim"] = int(job_input["hidden_dim"])
        if "num_heads" in job_input:
            config_params["num_heads"] = int(job_input["num_heads"])
        if "lambda_entangle" in job_input and mode == "layered":
            config_params["lambda_entangle_layered"] = float(job_input["lambda_entangle"])
        
        # 学習パラメータを抽出
        training_params = {}
        
        # エポック・バッチ関連
        if "epochs" in job_input:
            training_params["epochs"] = int(job_input["epochs"])
        if "batch_size" in job_input:
            training_params["batch_size"] = int(job_input["batch_size"])
        if "gradient_accumulation_steps" in job_input:
            training_params["gradient_accumulation_steps"] = int(job_input["gradient_accumulation_steps"])
        
        # 学習率関連
        if "learning_rate" in job_input:
            training_params["learning_rate"] = float(job_input["learning_rate"])
        if "min_learning_rate" in job_input:
            training_params["min_learning_rate"] = float(job_input["min_learning_rate"])
        if "weight_decay" in job_input:
            training_params["weight_decay"] = float(job_input["weight_decay"])
        if "warmup_steps" in job_input:
            training_params["warmup_steps"] = int(job_input["warmup_steps"])
        if "lr_scheduler" in job_input:
            training_params["lr_scheduler"] = str(job_input["lr_scheduler"])
        
        # Temperature関連
        if "max_temperature" in job_input:
            training_params["max_temperature"] = float(job_input["max_temperature"])
        if "min_temperature" in job_input:
            training_params["min_temperature"] = float(job_input["min_temperature"])
        if "temperature_decay" in job_input:
            training_params["temperature_decay"] = str(job_input["temperature_decay"])
        
        # 正則化・最適化
        if "max_grad_norm" in job_input:
            training_params["max_grad_norm"] = float(job_input["max_grad_norm"])
        if "label_smoothing" in job_input:
            training_params["label_smoothing"] = float(job_input["label_smoothing"])
        
        # 評価・保存
        if "eval_steps" in job_input:
            training_params["eval_steps"] = int(job_input["eval_steps"])
        if "save_steps" in job_input:
            training_params["save_steps"] = int(job_input["save_steps"])
        if "logging_steps" in job_input:
            training_params["logging_steps"] = int(job_input["logging_steps"])
        
        # 早期停止
        if "early_stopping_patience" in job_input:
            training_params["early_stopping_patience"] = int(job_input["early_stopping_patience"])
        if "early_stopping_threshold" in job_input:
            training_params["early_stopping_threshold"] = float(job_input["early_stopping_threshold"])
        
        # データ関連
        if "train_split" in job_input:
            training_params["train_split"] = float(job_input["train_split"])
        if "shuffle" in job_input:
            training_params["shuffle"] = bool(job_input["shuffle"])
        if "seed" in job_input:
            training_params["seed"] = int(job_input["seed"])
        
        # モデルをロード
        gen = load_model(mode, config_params if config_params else None)
        
        # 生成パラメータを取得
        prompt = job_input.get("prompt", "")
        max_tokens = int(job_input.get("max_tokens", 128))
        temperature = float(job_input.get("temperature", 0.7))
        top_k = int(job_input.get("top_k", 40))
        top_p = float(job_input.get("top_p", 0.9))
        repetition_penalty = float(job_input.get("repetition_penalty", 1.2))
        
        # バリデーション
        if not prompt:
            return {"error": "prompt is required"}
        
        if max_tokens < 1 or max_tokens > 1024:
            max_tokens = min(max(1, max_tokens), 1024)
        
        if temperature < 0.1 or temperature > 2.0:
            temperature = min(max(0.1, temperature), 2.0)
        
        print(f"📝 生成リクエスト:")
        print(f"   Mode: {mode}")
        print(f"   Prompt: {prompt[:50]}...")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Temperature: {temperature}")
        if config_params:
            print(f"   Custom config: {json.dumps(config_params)}")
        if training_params:
            print(f"   Training params: {json.dumps(training_params)}")
        
        # テキスト生成
        output_text = gen.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        print(f"✅ 生成完了: {len(output_text)} 文字")
        
        # 学習パラメータをデフォルト値とマージ
        merged_training_params = DEFAULT_TRAINING_CONFIG.copy()
        merged_training_params.update(training_params)
        
        # レスポンス
        return {
            "prompt": prompt,
            "output": output_text,
            "model_info": gen.get_model_info(),
            "config": {
                "mode": mode,
                **config_params
            },
            "training_config": merged_training_params
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}


# ========================================
# 学習エンドポイント
# ========================================

def train_handler(job):
    """
    モデル学習用ハンドラー
    
    入力JSON形式:
    {
        "input": {
            "training_data": ["テキスト1", "テキスト2", ...],  // 必須: 学習テキストのリスト
            "mode": "layered",        // オプション: "brain" or "layered"
            
            // 学習パラメータ
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "min_learning_rate": 1e-6,
            "warmup_steps": 100,
            "lr_scheduler": "cosine",
            "max_temperature": 1.5,
            "min_temperature": 0.1,
            "temperature_decay": "linear",
            "max_grad_norm": 1.0,
            "label_smoothing": 0.1,
            ...
        }
    }
    """
    try:
        job_input = job.get("input", {})
        
        # 学習データを取得
        training_data = job_input.get("training_data", [])
        if not training_data:
            return {"error": "training_data is required"}
        
        mode = job_input.get("mode", DEFAULT_MODE)
        
        # 学習パラメータを抽出してマージ
        training_config = DEFAULT_TRAINING_CONFIG.copy()
        
        training_param_keys = [
            "epochs", "batch_size", "gradient_accumulation_steps",
            "learning_rate", "min_learning_rate", "weight_decay", "warmup_steps", "lr_scheduler",
            "max_temperature", "min_temperature", "temperature_decay",
            "max_grad_norm", "label_smoothing",
            "eval_steps", "save_steps", "logging_steps",
            "early_stopping_patience", "early_stopping_threshold",
            "train_split", "shuffle", "seed"
        ]
        
        for key in training_param_keys:
            if key in job_input:
                value = job_input[key]
                # 型変換
                if key in ["epochs", "batch_size", "gradient_accumulation_steps", "warmup_steps", 
                           "eval_steps", "save_steps", "logging_steps", "early_stopping_patience", "seed"]:
                    training_config[key] = int(value)
                elif key in ["learning_rate", "min_learning_rate", "weight_decay", 
                             "max_temperature", "min_temperature", "max_grad_norm", 
                             "label_smoothing", "early_stopping_threshold", "train_split"]:
                    training_config[key] = float(value)
                elif key == "shuffle":
                    training_config[key] = bool(value)
                else:
                    training_config[key] = str(value)
        
        print(f"🎓 学習リクエスト:")
        print(f"   Mode: {mode}")
        print(f"   Training data size: {len(training_data)} samples")
        print(f"   Epochs: {training_config['epochs']}")
        print(f"   Batch size: {training_config['batch_size']}")
        print(f"   Learning rate: {training_config['learning_rate']}")
        print(f"   Max/Min temperature: {training_config['max_temperature']}/{training_config['min_temperature']}")
        print(f"   Temperature decay: {training_config['temperature_decay']}")
        print(f"   LR scheduler: {training_config['lr_scheduler']}")
        print(f"   Warmup steps: {training_config['warmup_steps']}")
        
        # モデル設定パラメータを抽出
        config_params = {}
        model_param_keys = [
            "embed_dim", "num_layers", "dropout", "max_seq_len",
            "num_neurons", "connection_density", "hidden_dim", "num_heads"
        ]
        for key in model_param_keys:
            if key in job_input:
                if key in ["dropout", "connection_density"]:
                    config_params[key] = float(job_input[key])
                else:
                    config_params[key] = int(job_input[key])
        
        if "lambda_entangle" in job_input:
            if mode == "brain":
                config_params["lambda_entangle_brain"] = float(job_input["lambda_entangle"])
            else:
                config_params["lambda_entangle_layered"] = float(job_input["lambda_entangle"])
        
        # モデルをロード
        gen = load_model(mode, config_params if config_params else None)
        
        # 注: 実際の学習処理はNeuroQGeneratorに実装が必要
        # ここでは学習パラメータの受け渡しと設定の確認のみ
        
        return {
            "status": "training_config_ready",
            "message": "Training parameters received and validated",
            "mode": mode,
            "training_data_count": len(training_data),
            "model_config": config_params if config_params else DEFAULT_CONFIG,
            "training_config": training_config,
            "model_info": gen.get_model_info(),
        }
        
    except Exception as e:
        error_msg = f"Training Error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}


# ========================================
# 量子情報エンドポイント
# ========================================

def quantum_info(job):
    """量子もつれ情報を取得"""
    try:
        job_input = job.get("input", {})
        mode = job_input.get("mode", DEFAULT_MODE)
        
        gen = load_model(mode)
        model_info = gen.get_model_info()
        quantum_info = gen.model.get_quantum_info()
        
        return {
            "status": "success",
            "mode": model_info.get('mode', 'unknown'),
            "model_info": model_info,
            "quantum_info": quantum_info,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ========================================
# モデル設定エンドポイント
# ========================================

def model_config(job):
    """
    現在のモデル設定と利用可能なオプションを取得
    """
    try:
        return {
            "status": "success",
            "default_mode": DEFAULT_MODE,
            "default_config": DEFAULT_CONFIG,
            "default_training_config": DEFAULT_TRAINING_CONFIG,
            "device": DEVICE,
            "cached_models": list(model_cache.keys()),
            "available_options": {
                "common": {
                    "embed_dim": "埋め込み次元（デフォルト: 128）",
                    "num_layers": "レイヤー数（デフォルト: 3）",
                    "dropout": "ドロップアウト率（デフォルト: 0.1）",
                    "max_seq_len": "最大シーケンス長（デフォルト: 256）",
                },
                "brain_mode": {
                    "num_neurons": "ニューロン数（デフォルト: 100）",
                    "connection_density": "接続密度 0.0-1.0（デフォルト: 0.25）",
                    "lambda_entangle": "量子もつれ強度（デフォルト: 0.35）",
                },
                "layered_mode": {
                    "hidden_dim": "隠れ層次元（デフォルト: 256）",
                    "num_heads": "アテンションヘッド数（デフォルト: 4）",
                    "lambda_entangle": "量子もつれ強度（デフォルト: 0.5）",
                },
                "training": {
                    "epochs": "エポック数（デフォルト: 10）",
                    "batch_size": "バッチサイズ（デフォルト: 32）",
                    "gradient_accumulation_steps": "勾配累積ステップ数（デフォルト: 1）",
                    "learning_rate": "学習率（デフォルト: 1e-4）",
                    "min_learning_rate": "最小学習率（デフォルト: 1e-6）",
                    "weight_decay": "重み減衰（デフォルト: 0.01）",
                    "warmup_steps": "ウォームアップステップ数（デフォルト: 100）",
                    "lr_scheduler": "学習率スケジューラー: cosine, linear, constant（デフォルト: cosine）",
                    "max_temperature": "最大サンプリング温度（デフォルト: 1.5）",
                    "min_temperature": "最小サンプリング温度（デフォルト: 0.1）",
                    "temperature_decay": "温度減衰方式: linear, exponential, cosine（デフォルト: linear）",
                    "max_grad_norm": "勾配クリッピング閾値（デフォルト: 1.0）",
                    "label_smoothing": "ラベルスムージング係数（デフォルト: 0.1）",
                    "eval_steps": "評価間隔ステップ数（デフォルト: 500）",
                    "save_steps": "モデル保存間隔ステップ数（デフォルト: 1000）",
                    "logging_steps": "ログ出力間隔ステップ数（デフォルト: 100）",
                    "early_stopping_patience": "早期停止patience（デフォルト: 3）",
                    "early_stopping_threshold": "早期停止閾値（デフォルト: 0.001）",
                    "train_split": "学習データ分割比率（デフォルト: 0.9）",
                    "shuffle": "データシャッフル有効化（デフォルト: true）",
                    "seed": "乱数シード（デフォルト: 42）",
                },
            },
            "example_request": {
                "input": {
                    "prompt": "こんにちは",
                    "mode": "brain",
                    "num_neurons": 200,
                    "connection_density": 0.3,
                    "max_tokens": 64,
                    "temperature": 0.7
                }
            },
            "example_training_request": {
                "input": {
                    "action": "train",
                    "training_data": ["テキスト1", "テキスト2", "..."],
                    "mode": "layered",
                    "epochs": 20,
                    "batch_size": 16,
                    "learning_rate": 5e-5,
                    "max_temperature": 1.2,
                    "min_temperature": 0.3,
                    "warmup_steps": 200,
                    "early_stopping_patience": 5
                }
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ========================================
# ヘルスチェック用エンドポイント
# ========================================

def health_check(job):
    """ヘルスチェック"""
    try:
        gen = load_model()
        return {
            "status": "healthy",
            "model_loaded": gen is not None,
            "device": DEVICE,
            "model_info": gen.get_model_info() if gen else None,
            "cached_models": len(model_cache),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


# ========================================
# メイン
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠⚛️ NeuroQ RunPod Serverless Worker")
    print("   Brain Mode: 脳型散在QBNN")
    print("   Layered Mode: 層状QBNN-Transformer")
    print("=" * 60)
    print("\n📋 デフォルトモデル設定:")
    print(f"   Mode: {DEFAULT_MODE}")
    for key, value in DEFAULT_CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n📚 デフォルト学習設定:")
    for key, value in DEFAULT_TRAINING_CONFIG.items():
        print(f"   {key}: {value}")
    print()
    
    # 起動時にデフォルトモデルをプリロード
    load_model()
    
    # RunPod Serverless を開始
    runpod.serverless.start({
        "handler": handler,
        "train": train_handler,
    })
