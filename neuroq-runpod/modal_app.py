#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗  ██████╗                        ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔═══██╗                       ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║   ██║                       ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║▄▄ ██║                       ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝                       ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝                        ║
║                                                                               ║
║   NeuroQ Modal API - クラウドGPU推論サーバー                                    ║
║   Quantum-Bit Neural Network Language Model                                   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Modal.com を使用したNeuroQ APIサーバー

使用方法:
    開発モード（ホットリロード）:
        modal serve modal_app.py
    
    デプロイ:
        modal deploy modal_app.py
    
    ローカルテスト:
        modal run modal_app.py
"""

import modal
from pathlib import Path
from typing import Optional, Dict, Any, List

# ========================================
# パスの設定
# ========================================
# modal_app.py があるディレクトリを取得
LOCAL_DIR = Path(__file__).parent.resolve()

# ========================================
# Modal App & Image 設定
# ========================================

# Modalアプリケーション
app = modal.App("neuroq-api")

# コンテナイメージ: 必要なPythonパッケージをインストール
# ローカルファイルもイメージに含める
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "sentencepiece>=0.1.99",
        "tiktoken>=0.5.0",
        "fastapi[standard]",
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "datasets>=2.14.0",
        "duckduckgo-search>=6.0.0",  # Web RAG用
    )
    # ローカルのPythonファイルをコンテナにコピー
    .add_local_dir(
        str(LOCAL_DIR),
        remote_path="/root/neuroq",
    )
)

# ========================================
# NeuroQ 推論クラス（コンテナ内で実行）
# ========================================

# チェックポイント保存用ボリューム（推論時も使用）
checkpoints_volume = modal.Volume.from_name("neuroq-checkpoints", create_if_missing=True)

# デフォルトシステムプロンプト（英語で生成→日本語に翻訳）
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

@app.cls(
    image=image,
    gpu="T4",  # GPU選択: T4, A10G, A100, H100など
    timeout=604800,  # 1週間 (7日)
    scaledown_window=120,  # アイドル状態で2分後にコンテナを停止
    volumes={"/model_checkpoints": checkpoints_volume},
)
@modal.concurrent(max_inputs=10)  # 同時リクエスト数
class NeuroQInference:
    """
    NeuroQ推論サービス
    
    Modal.comのコンテナ内で実行されるクラス。
    @modal.enter デコレータでコンテナ起動時にモデルをロードし、
    各メソッドでAPIエンドポイントを提供。
    """
    
    @modal.enter()
    def initialize(self):
        """コンテナ起動時に呼ばれる初期化処理"""
        import sys
        import torch
        import os
        
        # パスを設定
        sys.path.insert(0, "/root/neuroq")
        os.chdir("/root/neuroq")
        
        print("=" * 60)
        print("⚛️ NeuroQ Modal API - Initializing...")
        print("=" * 60)
        
        # デバイス設定
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📊 Device: {self.device}")
        print(f"📊 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        
        # モデル設定
        self.model = None
        self.model_config = None
        self.current_model_size = None
        self.is_initialized = False
        self.conversation_sessions = {}
        self.session_system_prompts = {}  # セッションごとのシステムプロンプト
        self.vocab_size = 8000
        self.tokenizer = None  # Small/Large用

        # チェックポイントパス
        # 全てのモデルサイズでModalボリュームを参照（train_modelで保存された最新版を使用）
        self.checkpoint_paths = {
            'micro': "/model_checkpoints/neuroq_micro_best.pt",
            'small': "/model_checkpoints/neuroq_small_best.pt",
            'large': "/model_checkpoints/neuroq_large_best.pt",
        }
        
        # モジュールをインポート
        try:
            from neuroquantum_layered import NeuroQuantumTokenizer, NeuroQuantum, NeuroQuantumConfig
            self.NeuroQuantumTokenizer = NeuroQuantumTokenizer
            self.NeuroQuantum = NeuroQuantum
            self.NeuroQuantumConfig = NeuroQuantumConfig
            print("✅ neuroquantum_layered.py をインポートしました")
        except ImportError as e:
            print(f"❌ neuroquantum_layered.py インポートエラー: {e}")
            self.NeuroQuantum = None
        
        try:
            from neuroquantum_brain import NeuroQuantumBrainAI, NeuroQuantumBrain
            self.NeuroQuantumBrainAI = NeuroQuantumBrainAI
            self.NeuroQuantumBrain = NeuroQuantumBrain
            print("✅ neuroquantum_brain.py をインポートしました")
        except ImportError as e:
            print(f"❌ neuroquantum_brain.py インポートエラー: {e}")
            self.NeuroQuantumBrainAI = None
        
        try:
            from model_configs import AVAILABLE_MODELS, get_model_config, get_checkpoint_path
            self.get_model_config = get_model_config
            self.model_configs_available = True
            print("✅ model_configs.py をインポートしました")
        except ImportError:
            self.model_configs_available = False

        # Web RAGモジュールをインポート
        try:
            from web_rag import WebRAGProcessor, RAG_SYSTEM_PROMPT_JA, RAG_SYSTEM_PROMPT_EN
            self.WebRAGProcessor = WebRAGProcessor
            self.RAG_SYSTEM_PROMPT_JA = RAG_SYSTEM_PROMPT_JA
            self.RAG_SYSTEM_PROMPT_EN = RAG_SYSTEM_PROMPT_EN
            self.rag_processor = None  # Lazy initialization
            self.rag_available = True
            print("✅ web_rag.py をインポートしました")
        except ImportError as e:
            print(f"⚠️ web_rag.py のインポートに失敗: {e}")
            self.WebRAGProcessor = None
            self.rag_available = False

        # デフォルトでmicroモデルを事前ロード（高速化のため）
        self._load_model('micro')
        
        print("=" * 60)
        print("⚛️ NeuroQ Modal API - Ready!")
        print("=" * 60)

    def _remap_state_dict_keys(self, state_dict: dict, model_state_dict: dict) -> dict:
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

        # キー名のマッピングを定義（双方向）
        key_mappings = {
            'blocks': 'transformer_blocks',
            'transformer_blocks': 'blocks',
            'text_embedding': 'token_embedding',
            'token_embedding': 'text_embedding',
            'dropout': 'embedding_dropout',
            'embedding_dropout': 'dropout',
        }

        # 変換が必要かチェック
        needs_conversion = False
        for model_key in model_keys:
            if model_key not in checkpoint_keys:
                needs_conversion = True
                break

        if not needs_conversion:
            return state_dict

        print("🔄 state_dictキー名を変換中...")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for old_name, new_name in key_mappings.items():
                if key.startswith(old_name + '.'):
                    new_key = new_name + key[len(old_name):]
                    break
            new_state_dict[new_key] = value

        return new_state_dict

    def _load_model(self, model_size: str = 'micro') -> bool:
        """モデルをロード"""
        import torch
        import os
        import gc

        # 既にロード済みなら何もしない
        if self.is_initialized and self.current_model_size == model_size:
            return True

        print(f"🔄 モデルロード開始 ({model_size.upper()})...")

        # 既存のモデルがある場合はメモリを解放
        if self.model is not None:
            print("🗑️ 既存モデルのメモリを解放中...")
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_initialized = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("✅ メモリ解放完了")

        # GPUメモリ状態を表示
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"📊 GPU Memory: Total={total_mem:.2f}GB, Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

        try:
            checkpoint_path = self.checkpoint_paths.get(model_size, self.checkpoint_paths['micro'])
            
            # NeuroQuantumモデル（micro/small/large共通）
            # チェックポイントはNeuroQuantum (QBNNTransformerBlock)で学習されているため、
            # 全モデルサイズでNeuroQuantumを使用する
            if self.model_configs_available and self.NeuroQuantum is not None:
                config = self.get_model_config(model_size)
                
                if os.path.exists(checkpoint_path):
                    try:
                        print(f"💾 チェックポイントをロード: {checkpoint_path}")
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        
                        # チェックポイントから設定を取得（オーバーライド）
                        saved_config = checkpoint.get('config', {})
                        tokenizer_info = checkpoint.get('tokenizer', {})
                        
                        # 語彙サイズをチェックポイントから取得
                        # 注意: 古いチェックポイントは 'max_vocab' キー、新しいチェックポイントは 'vocab_size' キーを使用
                        vocab_size = saved_config.get('vocab_size') or saved_config.get('max_vocab') or config['vocab_size']
                        
                        nq_config = self.NeuroQuantumConfig()
                        nq_config.vocab_size = vocab_size
                        nq_config.embed_dim = saved_config.get('embed_dim', config['embed_dim'])
                        nq_config.num_heads = saved_config.get('num_heads', config['num_heads'])
                        nq_config.num_layers = saved_config.get('num_layers', config['num_layers'])
                        nq_config.max_seq_len = saved_config.get('max_seq_len', config.get('max_seq_len', 512))
                        nq_config.dropout = saved_config.get('dropout', config.get('dropout', 0.1))
                        
                        print(f"   vocab_size: {nq_config.vocab_size:,}")
                        print(f"   embed_dim: {nq_config.embed_dim}")
                        
                        self.model = self.NeuroQuantum(nq_config).to(self.device)

                        # 重みをロード（キー名の互換性を保証）
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            state_dict = self._remap_state_dict_keys(checkpoint['model_state_dict'], self.model.state_dict())
                            self.model.load_state_dict(state_dict)
                        else:
                            state_dict = self._remap_state_dict_keys(checkpoint, self.model.state_dict())
                            self.model.load_state_dict(state_dict, strict=False)

                        self.model.eval()
                        
                        # トークナイザーを初期化
                        if tokenizer_info.get('type') == 'tiktoken':
                            try:
                                from tiktoken_tokenizer import TikTokenTokenizer
                                encoding_name = tokenizer_info.get('encoding', 'o200k_base')
                                self.tokenizer = TikTokenTokenizer(encoding_name=encoding_name)
                                print(f"   ✅ TikTokenトークナイザー ({encoding_name}) をロード")
                            except ImportError as e:
                                print(f"   ⚠️ TikTokenTokenizerインポートエラー: {e}")
                        else:
                            # SentencePieceトークナイザー (デフォルト/フォールバック)
                            tokenizer_path = "/root/neuroq/neuroq_tokenizer.model"
                            if os.path.exists(tokenizer_path):
                                self.tokenizer = self.NeuroQuantumTokenizer(
                                    vocab_size=vocab_size,
                                    model_file=tokenizer_path
                                )
                                print(f"   ✅ SentencePieceトークナイザー (vocab_size={vocab_size}) をロード")
                        
                        total_params = sum(p.numel() for p in self.model.parameters())
                        print(f"✅ {config['name']} ロード完了 ({total_params:,}パラメータ)")
                    except Exception as e:
                        import traceback
                        print(f"❌ {model_size.upper()} モデルロード失敗: {e}")
                        traceback.print_exc()
                        return False
                else:
                    print(f"⚠️ チェックポイントが見つかりません: {checkpoint_path}")
                    return False
            
            # Legacy Microモデル (NeuroQuantumBrainAI)
            else:
                print("⚠️ フォールバックモード: NeuroQuantumが利用できないため、NeuroQuantumBrainを使用します")
                print("   ⚠️ 注意: チェックポイントとアーキテクチャが異なる可能性があります")

                if self.NeuroQuantumBrainAI is None:
                    print("❌ NeuroQuantumBrainAI がインポートできていません")
                    return False
                
                if os.path.exists(checkpoint_path):
                    print(f"💾 チェックポイントをロード: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    config = checkpoint.get('config', {})
                    tokenizer_info = checkpoint.get('tokenizer', {})

                    # チェックポイントの語彙サイズを使用
                    # 注意: 古いチェックポイントは 'max_vocab' キー、新しいチェックポイントは 'vocab_size' キーを使用
                    vocab_size = config.get('vocab_size') or config.get('max_vocab') or tokenizer_info.get('vocab_size', 200019)
                    print(f"   📊 config: {config}")
                    print(f"   📊 tokenizer_info: {tokenizer_info}")
                    print(f"   📊 vocab_size: {vocab_size}")
                    
                    self.model = self.NeuroQuantumBrainAI(
                        embed_dim=config.get('embed_dim', 128),
                        num_heads=config.get('num_heads', 4),
                        num_layers=config.get('num_layers', 3),
                        num_neurons=config.get('num_neurons', 100),
                        max_vocab=vocab_size,
                        use_sentencepiece=False  # TikTokenを使用
                    )
                    
                    # SentencePieceトークナイザーをロード
                    tokenizer_path = "/root/neuroq/neuroq_tokenizer.model"
                    if os.path.exists(tokenizer_path):
                        self.model.tokenizer = self.NeuroQuantumTokenizer(
                            vocab_size=vocab_size,
                            model_file=tokenizer_path
                        )
                        print(f"   ✅ SentencePieceトークナイザー (vocab_size={vocab_size}) をロード")
                    else:
                        print(f"   ⚠️ トークナイザーモデルが見つかりません: {tokenizer_path}")
                    
                    # モデル構築・重みロード
                    self.model.model = self.NeuroQuantumBrain(
                        vocab_size=vocab_size,
                        embed_dim=config.get('embed_dim', 128),
                        num_heads=config.get('num_heads', 4),
                        num_layers=config.get('num_layers', 3),
                        num_neurons=config.get('num_neurons', 100),
                        max_seq_len=256,
                        dropout=0.1
                    ).to(self.device)

                    # 重みをロード（strict=Falseでアーキテクチャ不一致を許容）
                    state_dict = self._remap_state_dict_keys(checkpoint['model_state_dict'], self.model.model.state_dict())
                    self.model.model.load_state_dict(state_dict, strict=False)
                    self.model.model.eval()
                    total_params = sum(p.numel() for p in self.model.model.parameters())
                    print(f"⚠️ NeuroQuantumBrainAI (フォールバック) ロード完了 ({total_params:,}パラメータ)")
                    print(f"   ⚠️ 一部の重みが初期化されている可能性があります")
                else:
                    print(f"⚠️ チェックポイントが見つかりません: {checkpoint_path}")
                    return False
            
            self.current_model_size = model_size
            self.is_initialized = True
            return True

        except Exception as e:
            import traceback
            print(f"❌ モデルロードエラー: {e}")
            traceback.print_exc()
            return False

    def _initialize_rag_processor(self) -> bool:
        """RAGプロセッサを初期化（Lazy Loading）"""
        if self.rag_processor is not None:
            return True

        if not self.rag_available or self.WebRAGProcessor is None:
            print("⚠️ RAGプロセッサは利用できません")
            return False

        try:
            print("🔄 RAGプロセッサ初期化中...")
            self.rag_processor = self.WebRAGProcessor(
                max_search_results=5,
                max_context_chars=1500,
                search_region="jp-jp"
            )
            print("✅ RAGプロセッサ初期化完了")
            return True
        except Exception as e:
            import traceback
            print(f"❌ RAGプロセッサ初期化エラー: {e}")
            traceback.print_exc()
            return False

    def _generate_text(self, prompt: str, max_length: int = 50,
                       temp_min: float = None, temp_max: float = None,
                       temperature: float = None, history: List[Dict] = None,
                       system_prompt: str = None, session_id: str = None,
                       use_rag: bool = False, force_search: bool = False) -> dict:
        """テキスト生成（ヒストリー方式 + Web RAG対応）

        Args:
            prompt: ユーザーの入力
            max_length: 最大生成トークン数
            temp_min: 最小温度（micro用）
            temp_max: 最大温度（micro用）
            temperature: 温度（small/large用）
            history: 会話履歴 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            system_prompt: カスタムシステムプロンプト（Noneの場合はセッションの既存設定またはデフォルトを使用）
            session_id: セッションID（システムプロンプトの保存用）
            use_rag: Web RAGを使用するか（Web検索結果をコンテキストに追加）
            force_search: 検索を強制するか（use_rag=Trueの場合のみ有効）

        Returns:
            dict: {
                "generated": 生成されたテキスト,
                "rag_used": RAGが使用されたか,
                "rag_sources": 検索ソースのURL（RAG使用時のみ）,
                "rag_query": 検索クエリ（RAG使用時のみ）
            }
        """
        import torch

        if self.model is None or not self.is_initialized:
            return {"generated": "Error: Model not initialized", "rag_used": False}

        # RAG処理
        rag_context = None
        rag_sources = []
        rag_query = None
        rag_used = False

        if use_rag:
            if not self.rag_available:
                print("⚠️ RAG機能は利用できません")
            elif not self._initialize_rag_processor():
                print("⚠️ RAGプロセッサの初期化に失敗しました")
            else:
                try:
                    # 元のプロンプトで検索（翻訳前の日本語クエリの方が適切な場合が多い）
                    rag_context = self.rag_processor.process(
                        prompt=prompt,
                        force_search=force_search
                    )
                    if rag_context:
                        rag_used = True
                        rag_sources = rag_context.sources
                        rag_query = rag_context.query
                        print(f"🔍 RAG: {len(rag_context.search_results)}件の検索結果を取得")
                except Exception as e:
                    print(f"⚠️ RAG処理エラー: {e}")

        try:
            # システムプロンプトの処理
            # 1. リクエストで指定された場合はそれを使用し、セッションに保存
            # 2. 指定されていない場合はセッションの既存設定を使用
            # 3. セッションにも設定がない場合はデフォルトを使用
            # 4. RAGが有効な場合はRAG用のシステムプロンプトを使用
            if rag_used and rag_context:
                # RAG使用時は専用のシステムプロンプト + 検索結果を使用
                active_system_prompt = self.rag_processor.augment_system_prompt(
                    self.RAG_SYSTEM_PROMPT_JA,
                    rag_context,
                    lang='ja'
                )
            elif system_prompt is not None and session_id:
                self.session_system_prompts[session_id] = system_prompt
                active_system_prompt = system_prompt
            elif session_id:
                active_system_prompt = self.session_system_prompts.get(session_id, DEFAULT_SYSTEM_PROMPT)
            elif system_prompt is not None:
                active_system_prompt = system_prompt
            else:
                active_system_prompt = DEFAULT_SYSTEM_PROMPT

            # 温度設定
            if temperature is not None and temp_min is None:
                temp_min = temperature * 0.8
                temp_max = temperature * 1.2

            if temp_min is None:
                temp_min = 0.4
            if temp_max is None:
                temp_max = 0.7

            # 推論モード
            if hasattr(self.model, 'model') and self.model.model is not None:
                self.model.model.eval()
            elif hasattr(self.model, 'eval'):
                self.model.eval()
            # 会話履歴をコンテキストに変換（最新4ターンまで）
            history_context = ""
            # システムプロンプトがある場合は最初に追加
            if active_system_prompt:
                history_context += f"### System: {active_system_prompt}\n"
            
            if history:
                for turn in history[-4:]:
                    role = turn.get("role")
                    content = turn.get("content", "")
                    if role == "user":
                        history_context += f"### Human: {content}\n"
                    elif role == "assistant":
                        history_context += f"### Assistant: {content}\n"

            full_prompt = history_context + f"### Human: {prompt}\n### Assistant:"
            
            # 生成
            with torch.no_grad():
                # 現時点のモデルは会話タグ（<USER>等）を学習していないため、シンプルに渡す
                formatted_prompt = full_prompt
                
                result = self._generate_neuroq(
                    prompt=formatted_prompt,
                    max_length=max_length,
                    temperature=temperature or 0.5
                )

            return {
                "generated": result,
                "rag_used": rag_used,
                "rag_sources": rag_sources if rag_used else None,
                "rag_query": rag_query if rag_used else None
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ 生成エラー詳細:\n{error_details}")
            return {
                "generated": f"Error: {str(e)}\nTraceback: {error_details}",
                "rag_used": False
            }
    
    def _generate_neuroq(self, prompt: str, max_length: int = 50, temperature: float = 0.5) -> str:
        """NeuroQuantum モデル用のテキスト生成（繰り返しペナルティ付き）"""
        import torch
        
        # トークナイザーを取得
        if self.tokenizer is None:
            return "Error: Tokenizer not initialized"
        
        # プロンプトをトークン化
        input_ids = self.tokenizer.encode(prompt)
        generated = input_ids.copy()
        
        # <USER>トークンのIDを取得（終了条件用）
        try:
            user_token_ids = self.tokenizer.encode("<USER>")
        except:
            user_token_ids = []
        
        # 繰り返しペナルティ設定
        repetition_penalty = 1.2
        no_repeat_ngram_size = 3
        
        # 生成ループ
        for step in range(max_length):
            # 最大シーケンス長に制限
            seq = generated[-512:]
            seq_tensor = torch.tensor([seq], device=self.device)
            
            # モデル推論
            logits = self.model(seq_tensor)
            next_token_logits = logits[0, -1, :].clone()
            
            # 繰り返しペナルティを適用
            vocab_size = next_token_logits.size(-1)
            for token_id in set(generated[-50:]):  # 最近50トークンに対してペナルティ
                if token_id < vocab_size:  # bounds check to prevent CUDA gather error
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty
            
            # n-gram繰り返し防止
            if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size:
                # 現在のn-1 gramを取得
                ngram_prefix = tuple(generated[-(no_repeat_ngram_size-1):])
                # 過去に同じn-gramがあった場合、次のトークンを禁止
                for i in range(len(generated) - no_repeat_ngram_size):
                    if tuple(generated[i:i+no_repeat_ngram_size-1]) == ngram_prefix:
                        banned_token = generated[i + no_repeat_ngram_size - 1]
                        if banned_token < vocab_size:  # bounds check to prevent CUDA gather error
                            next_token_logits[banned_token] = float('-inf')
            
            # 次のトークンを予測
            probs = torch.softmax(next_token_logits / temperature, dim=-1)

            # NaN/Inf対策
            if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() == 0:
                probs = torch.ones_like(probs) / probs.size(-1)

            # Top-k sampling
            top_k = 50
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            top_probs = top_probs / top_probs.sum()

            # top_probsのNaN対策
            if torch.isnan(top_probs).any() or top_probs.sum() == 0:
                top_probs = torch.ones_like(top_probs) / top_probs.size(-1)

            idx = torch.multinomial(top_probs, num_samples=1)
            next_token = top_indices[idx].item()
            
            # EOSトークンなら終了
            if next_token == self.tokenizer.eos_id:
                break
            
            generated.append(next_token)
            
            # 生成中のテキストに<USER>が含まれていたら終了（より確実な方法に変更）
            if step % 5 == 0:  # 5ステップごとにチェック
                # プロンプト以降の新規生成分のみを対象にチェック
                new_tokens = generated[len(input_ids):]
                if not new_tokens:
                    continue
                    
                partial_text = self.tokenizer.decode(new_tokens)
                
                # <USER> もしくは <ASSISTANT> が新しく生成されたら終了
                if '<USER>' in partial_text:
                    print(f"   🛑 Stop generation: <USER> tag detected in step {step}")
                    break
                
                if step > 10 and '<ASSISTANT>' in partial_text:
                    print(f"   🛑 Stop generation: Unexpected <ASSISTANT> tag detected in step {step}")
                    break
                
                # 同じフレーズの繰り返しを検出
                if len(partial_text) > 20:
                    half = len(partial_text) // 2
                    if partial_text[:half] == partial_text[half:half*2]:
                        break
        
        # 新規生成分のみを取得
        new_ids = generated[len(input_ids):]
        if not new_ids:
            return ""
            
        result = self.tokenizer.decode(new_ids)
        
        # クリーンアップ（念のため）
        for tag in ['<USER>', '<ASSISTANT>', '<|endoftext|>']:
            if tag in result:
                result = result.split(tag)[0]
        
        return result.strip()
    
    @modal.method()
    def health(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        import torch
        return {
            "status": "healthy",
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "model_initialized": self.is_initialized
        }
    
    @modal.method()
    def status(self) -> Dict[str, Any]:
        """ステータス確認"""
        return {
            "status": "ok",
            "initialized": self.is_initialized,
            "device": self.device,
            "vocab_size": self.vocab_size,
            "current_model_size": self.current_model_size or "none",
            "available_model_sizes": ["micro", "small", "large"],
            "rag_available": self.rag_available,
            "rag_initialized": self.rag_processor is not None
        }
    
    @modal.method()
    def get_embeddings(self, text: str, model_size: str = "micro") -> Dict[str, Any]:
        """テキストをトークン化し、各トークンの埋め込みベクトルを取得"""
        import torch
        
        # モデルサイズが変更された場合はリロード
        if self.current_model_size != model_size:
            if not self._load_model(model_size):
                return {
                    "status": "error",
                    "error": f"Failed to load model ({model_size})"
                }
        
        if self.model is None or not self.is_initialized:
            return {"status": "error", "error": "Model not initialized"}
        
        try:
            # モデルの vocab_size を取得
            if hasattr(self.model, 'model') and self.model.model is not None:
                model_vocab_size = self.model.model.token_embedding.weight.shape[0]
            elif hasattr(self.model, 'token_embedding'):
                model_vocab_size = self.model.token_embedding.weight.shape[0]
            else:
                model_vocab_size = self.vocab_size

            # トークナイザーを取得
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                tokenizer = self.model.tokenizer
            else:
                tokenizer_path = "/root/neuroq/neuroq_tokenizer.model"
                tokenizer = self.NeuroQuantumTokenizer(vocab_size=model_vocab_size, model_file=tokenizer_path)

            # テキストをトークン化
            token_ids = tokenizer.encode(text, add_special=False)
            
            # 各トークンをデコード（個別の単語/サブワード）
            tokens = []
            for tid in token_ids:
                try:
                    token_text = tokenizer.decode([tid], skip_special=True)
                except:
                    token_text = f"<{tid}>"
                tokens.append(token_text)
            
            # モデルから埋め込みを取得
            with torch.no_grad():
                token_tensor = torch.tensor([token_ids], device=self.device)
                
                # モデルの埋め込み層を取得
                if hasattr(self.model, 'model') and self.model.model is not None:
                    # NeuroQuantumBrainAI
                    embedding_layer = self.model.model.token_embedding
                elif hasattr(self.model, 'token_embedding'):
                    # NeuroQuantum
                    embedding_layer = self.model.token_embedding
                else:
                    return {"status": "error", "error": "Could not find embedding layer"}
                
                # 埋め込みベクトルを取得
                embeddings = embedding_layer(token_tensor)  # [1, seq_len, embed_dim]
                embeddings = embeddings.squeeze(0)  # [seq_len, embed_dim]
            
            # 結果を構築
            result = {
                "status": "success",
                "text": text,
                "num_tokens": len(token_ids),
                "embed_dim": embeddings.shape[1],
                "tokens": []
            }
            
            for i, (token_text, token_id) in enumerate(zip(tokens, token_ids)):
                embedding_vec = embeddings[i].cpu().tolist()
                # ベクトルを短縮表示（最初と最後の5要素）
                if len(embedding_vec) > 10:
                    vec_preview = embedding_vec[:5] + ["..."] + embedding_vec[-5:]
                else:
                    vec_preview = embedding_vec
                
                result["tokens"].append({
                    "index": i,
                    "token": token_text,
                    "token_id": token_id,
                    "embedding_preview": vec_preview,
                    "embedding_full": embedding_vec
                })
            
            return result
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    @modal.method()
    def decode_embeddings(self, embeddings: List[List[float]], model_size: str = "micro", top_k: int = 5) -> Dict[str, Any]:
        """埋め込みベクトルからテキストに戻す（最近傍トークンを検索）"""
        import torch
        import torch.nn.functional as F
        
        # モデルサイズが変更された場合はリロード
        if self.current_model_size != model_size:
            if not self._load_model(model_size):
                return {
                    "status": "error",
                    "error": f"Failed to load model ({model_size})"
                }
        
        if self.model is None or not self.is_initialized:
            return {"status": "error", "error": "Model not initialized"}
        
        try:
            # モデルの埋め込み層を取得
            if hasattr(self.model, 'model') and self.model.model is not None:
                embedding_layer = self.model.model.token_embedding
            elif hasattr(self.model, 'token_embedding'):
                embedding_layer = self.model.token_embedding
            else:
                return {"status": "error", "error": "Could not find embedding layer"}

            # 埋め込み行列全体を取得
            embedding_matrix = embedding_layer.weight  # [vocab_size, embed_dim]
            vocab_size = embedding_matrix.shape[0]

            # トークナイザーを取得（vocab_size に制限）
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                tokenizer = self.model.tokenizer
            else:
                tokenizer_path = "/root/neuroq/neuroq_tokenizer.model"
                tokenizer = self.NeuroQuantumTokenizer(vocab_size=vocab_size, model_file=tokenizer_path)
            
            # 入力ベクトルをテンソルに変換
            input_embeddings = torch.tensor(embeddings, device=self.device, dtype=torch.float32)  # [num_tokens, embed_dim]
            
            # 結果を構築
            result = {
                "status": "success",
                "num_input_vectors": len(embeddings),
                "embed_dim": input_embeddings.shape[1] if len(input_embeddings.shape) > 1 else len(embeddings[0]),
                "decoded_tokens": [],
                "reconstructed_text": ""
            }
            
            decoded_token_ids = []
            
            with torch.no_grad():
                # 正規化（コサイン類似度用）
                input_norm = F.normalize(input_embeddings, p=2, dim=1)
                embed_norm = F.normalize(embedding_matrix, p=2, dim=1)
                
                # コサイン類似度を計算
                similarities = torch.mm(input_norm, embed_norm.t())  # [num_tokens, vocab_size]
                
                for i, sim in enumerate(similarities):
                    # Top-k トークンを取得
                    top_values, top_indices = torch.topk(sim, min(top_k, vocab_size))
                    
                    # 最も類似度の高いトークン
                    best_token_id = top_indices[0].item()
                    best_similarity = top_values[0].item()
                    decoded_token_ids.append(best_token_id)
                    
                    # トークンをデコード
                    try:
                        best_token_text = tokenizer.decode([best_token_id], skip_special=True)
                    except:
                        best_token_text = f"<{best_token_id}>"
                    
                    # 候補リスト
                    candidates = []
                    for j, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist())):
                        try:
                            token_text = tokenizer.decode([idx], skip_special=True)
                        except:
                            token_text = f"<{idx}>"
                        candidates.append({
                            "rank": j + 1,
                            "token": token_text,
                            "token_id": idx,
                            "similarity": round(val, 4)
                        })
                    
                    result["decoded_tokens"].append({
                        "index": i,
                        "best_token": best_token_text,
                        "best_token_id": best_token_id,
                        "similarity": round(best_similarity, 4),
                        "candidates": candidates
                    })
            
            # 復元テキスト
            try:
                result["reconstructed_text"] = tokenizer.decode(decoded_token_ids, skip_special=True)
            except:
                result["reconstructed_text"] = "".join([t["best_token"] for t in result["decoded_tokens"]])
            
            return result
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    @modal.method()
    def generate(self, prompt: str, max_length: int = 50,
                 temperature: float = 0.5, temp_min: float = None,
                 temp_max: float = None, history: List[Dict] = None,
                 model_size: str = "micro", system_prompt: str = None,
                 session_id: str = None,
                 use_rag: bool = False, force_search: bool = False) -> Dict[str, Any]:
        """テキスト生成（ヒストリー方式 + Web RAG対応）

        Args:
            prompt: ユーザーの入力
            max_length: 最大生成トークン数
            temperature: 温度
            temp_min: 最小温度
            temp_max: 最大温度
            history: 会話履歴 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            model_size: モデルサイズ（micro, small, large）
            system_prompt: カスタムシステムプロンプト（Noneの場合はデフォルトを使用）
            session_id: セッションID（システムプロンプトの永続化用）
            use_rag: Web RAGを使用するか（Web検索結果をコンテキストに追加）
            force_search: 検索を強制するか（use_rag=Trueの場合のみ有効）
        """

        # モデルサイズが変更された場合はリロード
        if self.current_model_size != model_size:
            if not self._load_model(model_size):
                return {
                    "status": "error",
                    "error": f"Failed to load model ({model_size})"
                }

        try:
            result = self._generate_text(
                prompt=prompt,
                max_length=max_length,
                temp_min=temp_min,
                temp_max=temp_max,
                temperature=temperature,
                history=history,
                system_prompt=system_prompt,
                session_id=session_id,
                use_rag=use_rag,
                force_search=force_search
            )

            # エラーチェック
            generated_text = result.get("generated", "")
            if generated_text.startswith("Error:"):
                return {
                    "status": "error",
                    "error": generated_text,
                    "prompt": prompt,
                    "model_size": self.current_model_size
                }

            # 使用されたシステムプロンプトを取得
            if system_prompt is not None:
                used_system_prompt = system_prompt
            elif session_id:
                used_system_prompt = self.session_system_prompts.get(session_id, DEFAULT_SYSTEM_PROMPT)
            else:
                used_system_prompt = DEFAULT_SYSTEM_PROMPT

            response = {
                "status": "success",
                "prompt": prompt,
                "generated": generated_text,
                "model_size": self.current_model_size,
                "system_prompt": used_system_prompt,
                "session_id": session_id,
                "rag_used": result.get("rag_used", False)
            }

            # RAGが使用された場合、追加情報を含める
            if result.get("rag_used"):
                response["rag_sources"] = result.get("rag_sources")
                response["rag_query"] = result.get("rag_query")

            return response
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ Generate error: {e}")
            print(error_detail)
            return {
                "status": "error",
                "error": str(e),
                "error_detail": error_detail,
                "prompt": prompt,
                "model_size": self.current_model_size
            }
    
    @modal.method()
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """学習ジョブを開始（スタンドアロンのtrain_modelを使用）"""
        # A100 GPU上で非同期に学習を開始
        return train_model.spawn(*args, **kwargs)


# ========================================
# FastAPI Web エンドポイント
# ========================================

@app.function(
    image=image,
    gpu="T4",
    timeout=604800,  # 1週間 (7日)
    scaledown_window=120,
    volumes={"/model_checkpoints": checkpoints_volume},
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app(custom_domains=["api.neuroq.he-ro.jp"])
def fastapi_app():
    """
    FastAPI Webアプリケーション
    
    エンドポイント:
        GET  /           - ルート（API情報）
        GET  /health     - ヘルスチェック
        GET  /status     - ステータス確認
        POST /generate   - テキスト生成
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import sys
    import os
    
    # パス設定
    sys.path.insert(0, "/root/neuroq")
    os.chdir("/root/neuroq")
    
    # Pydanticモデル
    class HistoryItem(BaseModel):
        role: str  # "user" or "assistant"
        content: str
    
    class GenerateRequest(BaseModel):
        prompt: str
        max_length: int = 50
        temperature: float = 0.5
        temp_min: Optional[float] = None
        temp_max: Optional[float] = None
        history: Optional[List[HistoryItem]] = None  # 会話履歴
        model_size: str = "micro"
        system_prompt: Optional[str] = None  # カスタムシステムプロンプト
        session_id: Optional[str] = None  # セッションID（システムプロンプトの永続化用）
        use_rag: bool = False  # Web RAGを使用するか（Web検索結果をコンテキストに追加）
        force_search: bool = False  # 検索を強制するか（use_rag=Trueの場合のみ有効）
    
    class GenerateResponse(BaseModel):
        status: str
        prompt: str
        generated: str
        model_size: str
        system_prompt: Optional[str] = None
        session_id: Optional[str] = None
        rag_used: Optional[bool] = None
        rag_sources: Optional[List[str]] = None
        rag_query: Optional[str] = None
    
    class HealthResponse(BaseModel):
        status: str
        device: str
        cuda_available: bool
        model_initialized: bool
    
    class StatusResponse(BaseModel):
        status: str
        initialized: bool
        device: str
        vocab_size: int
        current_model_size: str
        available_model_sizes: List[str]
        rag_available: Optional[bool] = None
        rag_initialized: Optional[bool] = None
    
    web_app = FastAPI(
        title="NeuroQ API",
        description="Quantum-Bit Neural Network Language Model API",
        version="1.0.1"
    )
    
    # CORS設定（必要に応じて調整）
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 推論インスタンスを取得
    neuroq = NeuroQInference()
    
    # 埋め込みリクエスト用のモデル
    class EmbeddingsRequest(BaseModel):
        text: str
        model_size: str = "micro"
    
    @web_app.get("/")
    async def root():
        return {
            "name": "NeuroQ API",
            "version": "1.0.1",
            "description": "Quantum-Bit Neural Network Language Model",
            "endpoints": {
                "health": "GET /health",
                "status": "GET /status",
                "generate": "POST /generate (supports system_prompt and session_id)",
                "embeddings": "POST /embeddings",
                "decode_embeddings": "POST /decode_embeddings",
                "train": "POST /train",
                "train_status": "GET /train/status/{call_id}"
            },
            "generate_params": {
                "prompt": "ユーザー入力（必須）",
                "max_length": "最大生成トークン数（デフォルト: 50）",
                "temperature": "温度（デフォルト: 0.5）",
                "history": "会話履歴 [{role, content}, ...]",
                "model_size": "micro, small, large（デフォルト: micro）",
                "system_prompt": "カスタムシステムプロンプト（省略時はデフォルト）",
                "session_id": "セッションID（システムプロンプトの永続化用）",
                "use_rag": "Web RAGを使用（デフォルト: false、Web検索結果をコンテキストに追加）",
                "force_search": "検索を強制（デフォルト: false、use_rag=trueの場合のみ有効）"
            },
            "rag_note": "use_rag=true を指定すると、Web検索結果をコンテキストに追加して回答を生成します。force_search=true で検索を強制できます"
        }
    
    @web_app.get("/health", response_model=HealthResponse)
    async def health():
        result = neuroq.health.remote()
        return result
    
    @web_app.get("/status", response_model=StatusResponse)
    async def status():
        result = neuroq.status.remote()
        return result
    
    @web_app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """テキスト生成（ヒストリー方式）"""
        try:
            # historyをdict形式に変換
            history_list = None
            if request.history:
                history_list = [{"role": item.role, "content": item.content} for item in request.history]

            result = neuroq.generate.remote(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                temp_min=request.temp_min,
                temp_max=request.temp_max,
                history=history_list,
                model_size=request.model_size,
                system_prompt=request.system_prompt,
                session_id=request.session_id,
                use_rag=request.use_rag,
                force_search=request.force_search
            )

            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))

            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/embeddings")
    async def embeddings(request: EmbeddingsRequest):
        """テキストをトークン化し、各トークンの埋め込みベクトルを取得"""
        try:
            result = neuroq.get_embeddings.remote(
                text=request.text,
                model_size=request.model_size
            )
            
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # デコードリクエスト用のモデル
    class DecodeEmbeddingsRequest(BaseModel):
        embeddings: List[List[float]]
        model_size: str = "micro"
        top_k: int = 5
    
    @web_app.post("/decode_embeddings")
    async def decode_embeddings(request: DecodeEmbeddingsRequest):
        """埋め込みベクトルからテキストに復元（最近傍トークン検索）"""
        try:
            result = neuroq.decode_embeddings.remote(
                embeddings=request.embeddings,
                model_size=request.model_size,
                top_k=request.top_k
            )
            
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # 学習リクエスト用のモデル
    class TrainRequest(BaseModel):
        model_size: str = "micro"
        epochs: int = 10
        batch_size: Optional[int] = None
        learning_rate: Optional[float] = None
        dataset_ids: Optional[List[str]] = None  # Hugging Face dataset IDs リスト
        text_column: str = "text"
        split: str = "train"
        max_samples: Optional[int] = None
    
    @web_app.post("/train")
    async def train(request: TrainRequest):
        """
        モデルの学習を開始（バックグラウンドで実行）
        
        dataset_ids に Hugging Face データセットIDのリストを指定すると、それらのデータセットで学習します。
        例: ["wikitext", "openai/summarize_from_feedback", "databricks/dolly-15k"]
        """
        try:
            # train_model関数を呼び出し（A10G GPUで実行）
            call = train_model.spawn(
                model_size=request.model_size,
                epochs=request.epochs,
                batch_size=request.batch_size,
                lr=request.learning_rate,
                dataset_ids=request.dataset_ids,
                text_column=request.text_column,
                split=request.split,
                max_samples=request.max_samples
            )
            
            return {
                "status": "started",
                "message": f"Training {request.model_size.upper()} model started",
                "model_size": request.model_size,
                "epochs": request.epochs,
                "batch_size": request.batch_size or "auto",
                "learning_rate": request.learning_rate or "auto",
                "dataset_ids": request.dataset_ids or ["default"],
                "text_column": request.text_column,
                "split": request.split,
                "max_samples": request.max_samples or "all",
                "call_id": call.object_id
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # 学習ステータス確認用
    @web_app.get("/train/status/{call_id}")
    async def train_status(call_id: str):
        """学習ジョブのステータスを確認"""
        try:
            from modal import FunctionCall
            call = FunctionCall.from_id(call_id)
            
            try:
                # 完了している場合は結果を取得
                result = call.get(timeout=0)
                return {
                    "status": "completed",
                    "result": result
                }
            except TimeoutError:
                return {
                    "status": "running",
                    "message": "Training is still in progress"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    return web_app


# ========================================
# ローカルテスト用関数
# ========================================

@app.local_entrypoint()
def main():
    """ローカルエントリーポイント（テスト用）"""
    print("🚀 NeuroQ Modal API Test")
    print("=" * 40)
    
    neuroq = NeuroQInference()
    
    # ヘルスチェック
    print("\n📊 Health Check:")
    health = neuroq.health.remote()
    print(f"   Status: {health['status']}")
    print(f"   Device: {health['device']}")
    print(f"   CUDA: {health['cuda_available']}")
    
    # ステータス
    print("\n📊 Status:")
    status = neuroq.status.remote()
    print(f"   Initialized: {status['initialized']}")
    print(f"   Model Size: {status['current_model_size']}")
    
    # テキスト生成
    print("\n📝 Generate Test:")
    result = neuroq.generate.remote(
        prompt="Hello, how are you?",
        max_length=50,
        temperature=0.5
    )
    print(f"   Prompt: {result['prompt']}")
    print(f"   Generated: {result['generated']}")
    
    print("\n✅ Test completed!")


# テスト用関数（GPUなし）
@app.function(image=image)
def test_health():
    """ヘルスチェックテスト（GPUなし）"""
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


# ========================================
# 学習用関数（GPUで実行）
# ========================================

# checkpoints_volumeはファイル上部で既に定義済み

@app.function(
    image=image,
    gpu="A100",
    timeout=604800,  # 1週間 (7日)
    volumes={"/model_checkpoints": checkpoints_volume},
)
def train_model(
    model_size: str = "micro",
    epochs: int = 10,
    batch_size: int = None,
    lr: float = None,
    dataset_ids: list = None,
    text_column: str = "text",
    split: str = "train",
    max_samples: int = None,
    epoch_mode: str = "early_stop",
):
    import os
    import sys
    import torch
    import random
    from torch.cuda.amp import autocast, GradScaler
    
    # コンテナ内のパス解決
    if os.path.exists("/root/neuroq"):
        sys.path.insert(0, "/root/neuroq")
        os.chdir("/root/neuroq")
    elif os.path.exists("/root"):
        sys.path.insert(0, "/root")
        os.chdir("/root")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    mode_label = "早期終了あり" if epoch_mode == "early_stop" else f"強制 {epochs} エポック"
    print("\n" + "=" * 70)
    print(f"🚀 NeuroQ {model_size.upper()} クラウド学習 (モード: {mode_label})")
    print("=" * 70)

    # モジュール読込
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    from tiktoken_tokenizer import TikTokenTokenizer
    from model_configs import get_model_config
    
    config = get_model_config(model_size)
    # LR初期値をより安全に設定 (micro: 1e-4, small/large: 5e-5)
    lr = lr or (1e-4 if model_size == "micro" else 5e-5)
    batch_size = batch_size or (4 if model_size == "micro" else 2)
    seq_length = config.get('max_seq_len', 512)
    
    tokenizer = TikTokenTokenizer(encoding_name="o200k_base")
    
    # データ準備
    print("\n📥 データロード中...")
    from datasets import load_dataset, concatenate_datasets
    ds_list = []
    ids = dataset_ids or ["OpenAssistant/oasst1", "OpenAssistant/oasst2"]
    
    for ds_id in ids:
        try:
            print(f"   - {ds_id} をロード...")
            ds = load_dataset(ds_id, split=split)
            if max_samples:
                ds = ds.shuffle(seed=42).select(range(min(len(ds), max_samples)))
            ds_list.append(ds)
        except Exception as e:
            print(f"   ⚠️ {ds_id} 失敗: {e}")
            
    all_tokens = []
    if ds_list:
        combined = concatenate_datasets(ds_list)
        for i, ex in enumerate(combined):
            txt = ex.get(text_column, "")
            if txt:
                all_tokens.extend(tokenizer.encode(txt, add_special=False) + [tokenizer.eos_id])
            if i % 10000 == 0 and i > 0: print(f"      {i}件トークン化完了...")
    
    print(f"📊 総トークン数: {len(all_tokens):,}")
    
    # 3. シーケンス分割 & 検証用切り出し
    all_seqs = [all_tokens[i:i + seq_length] for i in range(0, len(all_tokens) - seq_length, seq_length // 2)]
    random.shuffle(all_seqs)
    val_sz = max(1, int(len(all_seqs) * 0.05))
    train_seqs = all_seqs[:-val_sz]
    val_seqs = all_seqs[-val_sz:]
    print(f"✅ 学習用: {len(train_seqs):,}, 検証用: {len(val_seqs):,}")

    # 4. モデル構築
    nq_config = NeuroQuantumConfig()
    for k, v in config.items():
        if hasattr(nq_config, k): setattr(nq_config, k, v)
    nq_config.max_seq_len = seq_length
    model = NeuroQuantum(nq_config).to(device)
    
    # 5. 学習ループ準備
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()
    accumulation_steps = 4
    
    best_val_loss = float('inf')
    patience, patience_counter = 3, 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss, batch_count = 0, 0
        random.shuffle(train_seqs)
        optimizer.zero_grad()
        
        for i in range(0, len(train_seqs) - batch_size, batch_size):
            batch = torch.tensor(train_seqs[i:i+batch_size], device=device)
            inp, tar = batch[:, :-1], batch[:, 1:]
            
            with autocast():
                lgt = model(inp)
                loss = criterion(lgt.reshape(-1, config['vocab_size']), tar.reshape(-1))
                loss = loss / accumulation_steps
            
            if torch.isnan(loss):
                continue
                
            scaler.scale(loss).backward()
            
            if (batch_count + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * accumulation_steps
            batch_count += 1
            if batch_count % 500 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | Batch {batch_count} | Loss: {loss.item()*accumulation_steps:.4f}")

        # Validation & Test Prompts
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for k in range(0, len(val_seqs) - batch_size, batch_size):
                v_batch = torch.tensor(val_seqs[k:k+batch_size], device=device)
                v_lgt = model(v_batch[:, :-1])
                v_loss += criterion(v_lgt.reshape(-1, config['vocab_size']), v_batch[:, 1:].reshape(-1)).item()
        
        avg_train = total_train_loss / max(1, batch_count)
        avg_val = v_loss / max(1, (len(val_seqs) // batch_size))
        print(f"\n⭐ Epoch {epoch+1} 終了 | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        # 保存用データの作成
        save_data = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'model_size': model_size,
            'tokenizer': {
                'type': 'tiktoken',
                'encoding': 'o200k_base'
            }
        }
        
        # 定点テスト
        print("📝 定点テスト生成 (Sampling Temp=0.7):")
        # OASST形式のプロンプトでモデルを誘導
        test_prompts = [
            "### Human: こんにちは、元気ですか？\n### Assistant:",
            "### Human: What is AI?\n### Assistant:",
            "### Human: Python code for a simple for loop:\n### Assistant:"
        ]
        for prompt in test_prompts:
            ids = tokenizer.encode(prompt, add_special=False)
            gen = ids.copy()
            for _ in range(60): # 生成トークン数を少し増やす
                with torch.no_grad():
                    lgt = model(torch.tensor([gen[-seq_length:]], device=device))
                    # argmaxではなくサンプリングを使用して表現力を高める
                    probs = torch.softmax(lgt[0, -1, :] / 0.7, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1).item()
                    
                if nxt == tokenizer.eos_id: break
                gen.append(nxt)
            
            resp_tokens = gen[len(ids):]
            resp_text = tokenizer.decode(resp_tokens, skip_special=True).strip()
            
            # 視認性向上のため、空文字や制御文字のみの場合は情報を出す
            display_prompt = prompt.replace("### Human: ", "").replace("\n### Assistant:", "")
            if not resp_text:
                if len(resp_tokens) > 0:
                    resp_text = f"(Non-printable or spaces. Tokens: {resp_tokens[:5]}...)"
                else:
                    resp_text = "(No tokens generated / Immediate EOS)"
                    
            print(f"   Q: {display_prompt} | A: {resp_text[:120]}")

        # Early Stopping / Fixed Mode 判定
        if avg_val < best_val_loss - 0.01:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # 毎エポックのBestを即座に保存
            torch.save(save_data, f"/model_checkpoints/neuroq_{model_size}_best.pt")
            print(f"   ✨ Best更新 & 保存完了")
        else:
            patience_counter += 1
            print(f"   ⏳ 改善なし ({patience_counter}/{patience})")
            if epoch_mode == "early_stop" and patience_counter >= patience:
                print("🛑 早期終了: ロスが安定したため学習を停止します。")
                break
            elif epoch_mode == "fixed" and patience_counter >= patience:
                print(f"   ℹ️ 改善なしが {patience} 回続いていますが、強制モードのため学習を継続します。")
    
    # 最終保存
    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_path = f"/model_checkpoints/neuroq_{model_size}_checkpoint.pt"
    
    # 最終保存用のデータ（best_stateを反映）
    final_save_data = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_size': model_size,
        'tokenizer': {
            'type': 'tiktoken',
            'encoding': 'o200k_base'
        }
    }
    torch.save(final_save_data, final_path)
    checkpoints_volume.commit()
    return {"status": "success", "best_val_loss": best_val_loss}


@app.function(
    image=image,
    volumes={"/model_checkpoints": checkpoints_volume},
)
def download_checkpoint(model_size: str = "small"):
    """
    学習済みチェックポイントをダウンロード

    使い方:
        modal run modal_app.py::download_checkpoint --model-size small
    """
    import os

    checkpoint_filename = f"neuroq_{model_size}_checkpoint.pt"
    checkpoint_path = f"/model_checkpoints/{checkpoint_filename}"
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            data = f.read()
        print(f"✅ チェックポイント読み込み: {checkpoint_path} ({len(data)/1e6:.1f} MB)")
        return data
    else:
        print(f"❌ チェックポイントが見つかりません: {checkpoint_path}")
        return None
