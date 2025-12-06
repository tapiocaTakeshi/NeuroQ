#!/usr/bin/env python3
"""
NeuroQ RunPod Serverless Handler
=================================
RunPod Serverless API用のハンドラーファイル

参照元:
- neuroquantum_layered.py: 層状QBNN-Transformer
- neuroquantum_brain.py: 脳型散在QBNN

エンドポイント:
- /generate: テキスト生成
- /health: ヘルスチェック
"""

import os
import sys
import json
import torch
from typing import Dict, Any, Optional

# 親ディレクトリをパスに追加（neuroquantum_*.py を参照するため）
# Dockerコンテナ内では同じディレクトリに配置されるので、親ディレクトリ参照は不要
# ただし、ローカル開発環境での互換性のため残す
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# まず現在のディレクトリを追加（Dockerコンテナ内ではこれで十分）
# 現在のディレクトリを最初に追加することで、現在のディレクトリのファイルが優先される
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
# 親ディレクトリも追加（ローカル開発用）
# ただし、現在のディレクトリの後に追加することで、現在のディレクトリが優先される
if PARENT_DIR not in sys.path:
    sys.path.insert(1, PARENT_DIR)  # インデックス1に挿入して現在のディレクトリを優先

# neuroquantum_layered.py からインポート
try:
    # 現在のディレクトリのファイルを明示的にインポート
    import importlib.util
    layered_path = os.path.join(CURRENT_DIR, "neuroquantum_layered.py")
    if os.path.exists(layered_path):
        spec = importlib.util.spec_from_file_location("neuroquantum_layered_local", layered_path)
        layered_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layered_module)
        NeuroQuantumAI = layered_module.NeuroQuantumAI
        NeuroQuantumTokenizer = layered_module.NeuroQuantumTokenizer
        NeuroQuantumConfig = layered_module.NeuroQuantumConfig
        NeuroQuantum = layered_module.NeuroQuantum
        NEUROQUANTUM_LAYERED_AVAILABLE = True
        print(f"✅ ローカルの neuroquantum_layered.py からコンポーネントをインポートしました ({layered_path})")
    else:
        # フォールバック: 通常のインポート
        from neuroquantum_layered import (
            NeuroQuantumAI,
            NeuroQuantumTokenizer,
            NeuroQuantumConfig,
            NeuroQuantum,
        )
        NEUROQUANTUM_LAYERED_AVAILABLE = True
        print("✅ neuroquantum_layered.py からコンポーネントをインポートしました")
except ImportError as e:
    NEUROQUANTUM_LAYERED_AVAILABLE = False
    print(f"⚠️ neuroquantum_layered.py が見つかりません: {e}")
except Exception as e:
    NEUROQUANTUM_LAYERED_AVAILABLE = False
    print(f"⚠️ neuroquantum_layered.py のインポートでエラーが発生しました: {e}")

# neuroquantum_brain.py からインポート
try:
    from neuroquantum_brain import (
        NeuroQuantumBrainAI,
        BrainTokenizer,
        NeuroQuantumBrain,
    )
    NEUROQUANTUM_BRAIN_AVAILABLE = True
    print("✅ neuroquantum_brain.py からコンポーネントをインポートしました")
except ImportError as e:
    NEUROQUANTUM_BRAIN_AVAILABLE = False
    print(f"⚠️ neuroquantum_brain.py が見つかりません: {e}")

# RunPod SDK
try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False
    print("⚠️ runpodライブラリがインストールされていません。pip install runpod を実行してください。")

# OpenAI API（ChatGPTエンベディング用）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAIライブラリがインストールされていません。pip install openai を実行してください。")

# ========================================
# グローバル変数
# ========================================

# モデルインスタンス（グローバルに保持）
model_layered: Optional[NeuroQuantumAI] = None
model_brain: Optional[NeuroQuantumBrainAI] = None

# デバイス選択
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🍎 Apple Silicon GPU (MPS) を使用")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🎮 NVIDIA GPU (CUDA) を使用")
else:
    DEVICE = torch.device("cpu")
    print("💻 CPU を使用")


# ========================================
# モデル初期化
# ========================================

def init_model(mode: str = "layered", **kwargs) -> Dict[str, Any]:
    """
    モデルを初期化
    
    Args:
        mode: 'layered' または 'brain'
        **kwargs: モデル設定パラメータ
    
    Returns:
        初期化結果
    """
    global model_layered, model_brain
    
    try:
        if mode == "layered":
            if not NEUROQUANTUM_LAYERED_AVAILABLE:
                return {"error": "neuroquantum_layered.py が利用できません"}
            
            # デフォルト設定
            embed_dim = kwargs.get("embed_dim", 64)
            # num_neuronsが指定されている場合はhidden_dimとして使用
            hidden_dim = kwargs.get("hidden_dim", kwargs.get("num_neurons", 128))
            num_heads = kwargs.get("num_heads", 4)
            num_layers = kwargs.get("num_layers", 2)
            max_seq_len = kwargs.get("max_seq_len", 128)
            dropout = kwargs.get("dropout", 0.1)
            lambda_entangle = kwargs.get("lambda_entangle", 0.35)
            use_openai_embedding = kwargs.get("use_openai_embedding", False)
            openai_api_key = kwargs.get("openai_api_key")
            openai_model = kwargs.get("openai_model", "text-embedding-3-large")
            
            model_layered = NeuroQuantumAI(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                dropout=dropout,
                lambda_entangle=lambda_entangle,
                use_openai_embedding=use_openai_embedding,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
            )
            model_layered.device = DEVICE
            
            return {
                "status": "success",
                "mode": "layered",
                "message": "Layered mode モデルを初期化しました"
            }
        
        elif mode == "brain":
            if not NEUROQUANTUM_BRAIN_AVAILABLE:
                return {"error": "neuroquantum_brain.py が利用できません"}
            
            # デフォルト設定
            embed_dim = kwargs.get("embed_dim", 128)
            num_heads = kwargs.get("num_heads", 4)
            num_layers = kwargs.get("num_layers", 3)
            num_neurons = kwargs.get("num_neurons", 75)
            max_vocab = kwargs.get("max_vocab", 50000)
            
            model_brain = NeuroQuantumBrainAI(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_neurons=num_neurons,
                max_vocab=max_vocab,
            )
            model_brain.device = DEVICE
            
            return {
                "status": "success",
                "mode": "brain",
                "message": "Brain mode モデルを初期化しました"
            }
        
        else:
            return {"error": f"不明なモード: {mode}"}
    
    except Exception as e:
        return {"error": f"モデル初期化エラー: {str(e)}"}


# ========================================
# データ取得
# ========================================

def fetch_training_data(
    data_sources: Optional[list] = None,
    common_crawl_config: Optional[Dict[str, Any]] = None,
    max_records: int = 100
) -> list:
    """
    学習データを取得
    
    Args:
        data_sources: データソースのリスト (例: ["common_crawl", "huggingface"])
        common_crawl_config: Common Crawl設定（将来の拡張用）
        max_records: 最大レコード数
    
    Returns:
        テキストのリスト
    """
    texts = []
    
    if data_sources is None:
        data_sources = ["huggingface"]  # デフォルト
    
    # Hugging Faceからデータ取得
    if "huggingface" in data_sources or "hugging_face" in data_sources:
        try:
            from datasets import load_dataset
            print("   📡 Hugging Faceからデータ取得中...")
            
            # 日本語Wikipedia
            try:
                ds = load_dataset("range3/wiki40b-ja", split="train", streaming=True)
                count = 0
                for item in ds:
                    if 'text' in item and len(item['text']) > 50:
                        texts.append(item['text'][:1000])  # 長さ制限
                        count += 1
                        if count >= max_records // 3:
                            break
            except Exception as e:
                print(f"   ⚠️ 日本語Wikipedia取得失敗: {e}")
            
            # 英語データ
            try:
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                for item in ds[:max_records // 3]:
                    if 'text' in item and len(item['text']) > 30:
                        texts.append(item['text'])
            except Exception as e:
                print(f"   ⚠️ WikiText取得失敗: {e}")
            
            # 日本語対話
            try:
                ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
                for item in ds[:max_records // 3]:
                    if 'output' in item:
                        texts.append(item['output'])
            except Exception as e:
                print(f"   ⚠️ 日本語対話データ取得失敗: {e}")
            
            print(f"   ✅ Hugging Face: {len(texts)} サンプル取得")
            
        except ImportError:
            print("   ⚠️ datasets未インストール")
        except Exception as e:
            print(f"   ⚠️ Hugging Face取得失敗: {e}")
    
    # Common Crawl（将来の拡張用）
    if "common_crawl" in data_sources:
        print("   ⚠️ Common Crawlは現在実装中です。Hugging Faceデータを使用します。")
        # TODO: Common Crawl API実装
    
    # 組み込みデータ（フォールバック）
    if len(texts) == 0:
        print("   📝 組み込みデータを使用...")
        texts = [
            "量子コンピュータは、量子力学の原理を利用して情報を処理する革新的な計算機です。",
            "人工知能は、人間の知的活動をコンピュータで実現しようとする技術です。",
            "機械学習は、データから学習して改善するシステムを実現します。",
            "深層学習は、多層のニューラルネットワークを用いた機械学習手法です。",
            "ニューラルネットワークは、人間の脳の神経回路を模倣した計算システムです。",
        ] * (max_records // 5)
    
    return texts[:max_records]


# ========================================
# 学習処理
# ========================================

def train_model(
    mode: str = "layered",
    texts: Optional[list] = None,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seq_length: int = 64,
    **kwargs
) -> Dict[str, Any]:
    """
    モデルを学習
    
    Args:
        mode: 'layered' または 'brain'
        texts: 学習データ（テキストのリスト）
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        seq_length: シーケンス長
        **kwargs: その他のパラメータ
    
    Returns:
        学習結果
    """
    global model_layered, model_brain
    
    try:
        if texts is None or len(texts) == 0:
            return {"error": "学習データが空です"}
        
        if mode == "layered":
            if model_layered is None:
                init_result = init_model(mode="layered", **kwargs)
                if "error" in init_result:
                    return init_result
            
            print(f"   🎓 Layered mode 学習開始: {len(texts)}サンプル, {epochs}エポック")
            model_layered.train(
                texts=texts,
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                seq_len=seq_length
            )
            
            return {
                "status": "success",
                "mode": "layered",
                "message": f"学習完了: {epochs}エポック"
            }
        
        elif mode == "brain":
            if model_brain is None:
                init_result = init_model(mode="brain", **kwargs)
                if "error" in init_result:
                    return init_result
            
            print(f"   🎓 Brain mode 学習開始: {len(texts)}サンプル, {epochs}エポック")
            model_brain.train(
                texts=texts,
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                seq_length=seq_length
            )
            
            return {
                "status": "success",
                "mode": "brain",
                "message": f"学習完了: {epochs}エポック"
            }
        
        else:
            return {"error": f"不明なモード: {mode}"}
    
    except Exception as e:
        return {"error": f"学習エラー: {str(e)}"}


# ========================================
# テキスト生成
# ========================================

def generate_text(
    prompt: str,
    mode: str = "layered",
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    **kwargs
) -> Dict[str, Any]:
    """
    テキスト生成
    
    Args:
        prompt: 入力プロンプト
        mode: 'layered' または 'brain'
        max_length: 最大生成長
        temperature: 温度パラメータ
        top_k: Top-K サンプリング
        top_p: Top-P サンプリング
        **kwargs: その他のパラメータ
    
    Returns:
        生成結果
    """
    global model_layered, model_brain
    
    try:
        if mode == "layered":
            if model_layered is None:
                # モデルが初期化されていない場合は初期化
                init_result = init_model(mode="layered", **kwargs)
                if "error" in init_result:
                    return init_result
            
            if model_layered.model is None:
                return {"error": "モデルが学習されていません"}
            
            # テキスト生成
            generated = model_layered.generate(
                prompt=prompt,
                max_length=max_length,
                temp_min=temperature * 0.8,
                temp_max=temperature * 1.2,
                top_k=top_k,
                top_p=top_p,
            )
            
            return {
                "status": "success",
                "mode": "layered",
                "prompt": prompt,
                "generated": generated,
            }
        
        elif mode == "brain":
            if model_brain is None:
                # モデルが初期化されていない場合は初期化
                init_result = init_model(mode="brain", **kwargs)
                if "error" in init_result:
                    return init_result
            
            if model_brain.model is None:
                return {"error": "モデルが学習されていません"}
            
            # テキスト生成
            generated = model_brain.generate(
                prompt=prompt,
                max_length=max_length,
                temperature_min=temperature * 0.8,
                temperature_max=temperature * 1.2,
                top_k=top_k,
                top_p=top_p,
            )
            
            return {
                "status": "success",
                "mode": "brain",
                "prompt": prompt,
                "generated": generated,
            }
        
        else:
            return {"error": f"不明なモード: {mode}"}
    
    except Exception as e:
        return {"error": f"生成エラー: {str(e)}"}


# ========================================
# RunPod Handler
# ========================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless Handler
    
    リクエスト形式:
    {
        "input": {
            "action": "generate" | "init" | "health",
            "mode": "layered" | "brain",
            "prompt": "テキストプロンプト",
            "max_length": 100,
            "temperature": 0.7,
            ...
        }
    }
    """
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "generate")
        
        if action == "health":
            return {
                "status": "healthy",
                "layered_available": NEUROQUANTUM_LAYERED_AVAILABLE,
                "brain_available": NEUROQUANTUM_BRAIN_AVAILABLE,
                "openai_available": OPENAI_AVAILABLE,
                "device": str(DEVICE),
            }
        
        elif action == "init":
            mode = input_data.get("mode", "layered")
            kwargs = {k: v for k, v in input_data.items() if k != "action" and k != "mode"}
            return init_model(mode=mode, **kwargs)
        
        elif action == "generate":
            prompt = input_data.get("prompt", "")
            if not prompt:
                return {"error": "promptが必要です"}
            
            mode = input_data.get("mode", "layered")
            max_length = input_data.get("max_length", input_data.get("max_tokens", 100))
            temperature = input_data.get("temperature", 0.7)
            top_k = input_data.get("top_k", 40)
            top_p = input_data.get("top_p", 0.9)
            
            # train_before_generate フラグの処理
            train_before_generate = input_data.get("train_before_generate", False)
            
            if train_before_generate:
                # データ取得
                data_sources = input_data.get("data_sources", ["huggingface"])
                common_crawl_config = input_data.get("common_crawl_config", {})
                max_records = common_crawl_config.get("max_records", 100)
                
                print(f"📥 学習データを取得中... (ソース: {data_sources}, 最大{max_records}レコード)")
                texts = fetch_training_data(
                    data_sources=data_sources,
                    common_crawl_config=common_crawl_config,
                    max_records=max_records
                )
                
                if len(texts) == 0:
                    return {"error": "学習データの取得に失敗しました"}
                
                # 学習パラメータ
                epochs = input_data.get("epochs", 20)
                batch_size = input_data.get("batch_size", 16)
                learning_rate = input_data.get("learning_rate", 0.001)
                seq_length = input_data.get("seq_length", 64)
                
                # モデル設定パラメータを抽出
                # num_neuronsはbrainモードではnum_neurons、layeredモードではhidden_dimとして使用
                model_kwargs = {
                    k: v for k, v in input_data.items()
                    if k in [
                        "embed_dim", "hidden_dim", "num_heads", "num_layers",
                        "num_neurons", "max_vocab", "max_seq_len", "dropout",
                        "lambda_entangle", "use_openai_embedding", "openai_api_key",
                        "openai_model"
                    ]
                }
                
                # num_neuronsが指定されている場合、モードに応じて変換
                if "num_neurons" in input_data and "num_neurons" not in model_kwargs:
                    num_neurons = input_data["num_neurons"]
                    if mode == "layered":
                        # layeredモードではhidden_dimとして使用
                        model_kwargs["hidden_dim"] = num_neurons
                    elif mode == "brain":
                        # brainモードではnum_neuronsとして使用
                        model_kwargs["num_neurons"] = num_neurons
                
                # 学習実行
                train_result = train_model(
                    mode=mode,
                    texts=texts,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    seq_length=seq_length,
                    **model_kwargs
                )
                
                if "error" in train_result:
                    return train_result
                
                print(f"✅ 学習完了: {train_result.get('message', '')}")
            
            # 生成パラメータ
            kwargs = {
                k: v for k, v in input_data.items()
                if k not in [
                    "action", "prompt", "mode", "max_length", "max_tokens",
                    "temperature", "top_k", "top_p", "train_before_generate",
                    "data_sources", "common_crawl_config", "epochs", "batch_size",
                    "learning_rate", "seq_length"
                ]
            }
            
            return generate_text(
                prompt=prompt,
                mode=mode,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs
            )
        
        else:
            return {"error": f"不明なアクション: {action}"}
    
    except Exception as e:
        return {"error": f"ハンドラーエラー: {str(e)}"}


# ========================================
# RunPod Serverless 起動
# ========================================

if __name__ == "__main__":
    if RUNPOD_AVAILABLE:
        print("🚀 RunPod Serverless Handler を起動します...")
        runpod.serverless.start({"handler": handler})
    else:
        print("⚠️ RunPod SDKが利用できません。ローカルテストモードで実行します。")
        print("\nテストリクエスト例:")
        print(json.dumps({
            "input": {
                "action": "health"
            }
        }, indent=2))

