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
        "transformers>=4.30.0",
        "openai>=1.0.0",
        "datasets>=2.14.0",
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

# デフォルトシステムプロンプト
DEFAULT_SYSTEM_PROMPT = """あなたは親切で正確なアシスタントです。
以下のルールに従ってください：
1. ユーザーの質問に短く正確に答える
2. わからないことは質問する
3. 聞かれたことだけに答える（余計な情報を追加しない）
4. 前の文脈を踏まえて返答する"""

@app.cls(
    image=image,
    gpu="T4",  # GPU選択: T4, A10G, A100, H100など
    timeout=300,
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
        # 全モデルをModalボリューム（/model_checkpoints）から参照
        self.checkpoint_paths = {
            'micro': "/model_checkpoints/neuroq_micro_checkpoint.pt",
            'small': "/model_checkpoints/neuroq_small_checkpoint.pt",
            'large': "/model_checkpoints/neuroq_large_checkpoint.pt",
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
                    
                    # トークナイザーを初期化（SentencePiece優先）
                    tokenizer_path = "/root/neuroq/neuroq_tokenizer.model"
                    if os.path.exists(tokenizer_path):
                        self.tokenizer = self.NeuroQuantumTokenizer(
                            vocab_size=vocab_size,
                            model_file=tokenizer_path
                        )
                        print(f"   ✅ SentencePieceトークナイザー (vocab_size={vocab_size}) をロード")
                    else:
                        print(f"   ⚠️ トークナイザーモデルが見つかりません: {tokenizer_path}")
                    
                    total_params = sum(p.numel() for p in self.model.parameters())
                    print(f"✅ {config['name']} ロード完了 ({total_params:,}パラメータ)")
                else:
                    print(f"⚠️ チェックポイントが見つかりません: {checkpoint_path}")
                    return False
            
            # フォールバック: NeuroQuantumが利用できない場合（通常は発生しない）
            # 注意: このフォールバックはNeuroQuantumBrainを使用しますが、
            # 現在のチェックポイントはNeuroQuantum (QBNNTransformerBlock)で学習されているため、
            # アーキテクチャの不一致により正常に動作しない可能性があります。
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
    
    def _generate_text(self, prompt: str, max_length: int = 50,
                       temp_min: float = None, temp_max: float = None,
                       temperature: float = None, history: List[Dict] = None,
                       system_prompt: str = None, session_id: str = None) -> str:
        """テキスト生成（ヒストリー方式）

        Args:
            prompt: ユーザーの入力
            max_length: 最大生成トークン数
            temp_min: 最小温度（micro用）
            temp_max: 最大温度（micro用）
            temperature: 温度（small/large用）
            history: 会話履歴 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            system_prompt: カスタムシステムプロンプト（Noneの場合はセッションの既存設定またはデフォルトを使用）
            session_id: セッションID（システムプロンプトの保存用）
        """
        import torch

        if self.model is None or not self.is_initialized:
            return "Error: Model not initialized"

        try:
            # システムプロンプトの処理
            # 1. リクエストで指定された場合はそれを使用し、セッションに保存
            # 2. 指定されていない場合はセッションの既存設定を使用
            # 3. セッションにも設定がない場合はデフォルトを使用
            if system_prompt is not None and session_id:
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

            # コンテキストを構築
            context_parts = []

            # システムプロンプトをコンテキストの先頭に追加
            if active_system_prompt:
                context_parts.append(f"<SYSTEM>{active_system_prompt}")

            # 会話履歴をコンテキストに変換（最新4ターンまで）
            if history:
                for turn in history[-4:]:
                    if turn.get("role") == "user":
                        context_parts.append(f"<USER>{turn.get('content', '')}")
                    elif turn.get("role") == "assistant":
                        context_parts.append(f"<ASSISTANT>{turn.get('content', '')}")

            history_context = "".join(context_parts)
            full_prompt = history_context + prompt if history_context else prompt
            
            # 生成
            with torch.no_grad():
                # NeuroQuantumBrainAI (micro) の場合
                if hasattr(self.model, 'generate'):
                    result = self.model.generate(
                        prompt=full_prompt,
                        max_length=max_length,
                        temperature_min=temp_min,
                        temperature_max=temp_max,
                        top_k=40,
                        top_p=0.9,
                    )
                # NeuroQuantum (small/large) の場合
                else:
                    # 会話形式のプロンプトを構築
                    formatted_prompt = f"<USER>{full_prompt}<ASSISTANT>"
                    result = self._generate_neuroq(
                        prompt=formatted_prompt,
                        max_length=max_length,
                        temperature=temperature or 0.5,
                    )
            
            # 結果をクリーンアップ（<USER>タグ以降を除去）
            if '<USER>' in result:
                result = result.split('<USER>')[0].strip()
            
            return result
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
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
            
            # <USER>トークンの開始なら終了
            if user_token_ids and next_token == user_token_ids[0]:
                break
            
            generated.append(next_token)
            
            # 生成中のテキストに<USER>が含まれていたら終了
            if step % 5 == 0:  # 5ステップごとにチェック
                partial_text = self.tokenizer.decode(generated[-15:], skip_special=True)
                if '<USER>' in partial_text:
                    break
                
                # 同じフレーズの繰り返しを検出
                if len(partial_text) > 20:
                    half = len(partial_text) // 2
                    if partial_text[:half] == partial_text[half:half*2]:
                        break
        
        # デコード
        result = self.tokenizer.decode(generated, skip_special=True)
        
        # <ASSISTANT>以降を抽出
        if '<ASSISTANT>' in result:
            result = result.split('<ASSISTANT>')[-1].strip()
        
        # <USER>以降を除去
        if '<USER>' in result:
            result = result.split('<USER>')[0].strip()
        
        return result
    
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
            "available_model_sizes": ["micro", "small", "large"]
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
                 session_id: str = None) -> Dict[str, Any]:
        """テキスト生成（ヒストリー方式）

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
                session_id=session_id
            )

            # エラーチェック
            if result.startswith("Error:"):
                return {
                    "status": "error",
                    "error": result,
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

            return {
                "status": "success",
                "prompt": prompt,
                "generated": result,
                "model_size": self.current_model_size,
                "system_prompt": used_system_prompt,
                "session_id": session_id
            }
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
    def train(
        self,
        model_size: str = "micro",
        epochs: int = 10,
        batch_size: int = None,
        lr: float = None,
        dataset_ids: List[str] = None,
        text_column: str = "text",
        split: str = "train",
        max_samples: int = None
    ) -> Dict[str, Any]:
        """
        モデルを学習
        
        Args:
            model_size: 'micro', 'small', 'large'
            epochs: 学習エポック数
            batch_size: バッチサイズ
            lr: 学習率
            dataset_ids: Hugging FaceデータセットIDのリスト
            text_column: テキストカラム名
            split: データ分割
            max_samples: 各データセットの最大サンプル数
        """
        import torch
        import random
        import os
        
        print("=" * 70)
        print(f"🚀 NeuroQ {model_size.upper()} モデル学習")
        print("=" * 70)
        
        # 学習設定（T4 GPU 15GBメモリ対応）
        if batch_size is None:
            batch_size = 4 if model_size == 'micro' else (2 if model_size == 'small' else 1)
        if lr is None:
            lr = 0.0005 if model_size == 'micro' else 0.0003
        seq_length = 64
        
        print(f"\n📋 学習設定:")
        print(f"   model_size: {model_size}")
        print(f"   epochs: {epochs}")
        print(f"   batch_size: {batch_size}")
        print(f"   lr: {lr}")
        print(f"   device: {self.device}")
        if dataset_ids:
            print(f"   dataset_ids: {dataset_ids}")

        # モデル設定を先に取得（vocab_size を使うため）
        from model_configs import get_model_config
        config = get_model_config(model_size)
        model_vocab_size = config['vocab_size']

        # oasst1_ja がdataset_idsに含まれているかチェック
        use_oasst1_ja = dataset_ids and "oasst1_ja" in dataset_ids

        # SentencePieceトークナイザー
        from neuroquantum_layered import NeuroQuantumTokenizer
        if use_oasst1_ja:
            # oasst1_ja用のトークナイザーを使用
            tokenizer_candidates = [
                "/root/neuroq/neuroq_tokenizer_oasst1_ja.model",
                "neuroq_tokenizer_oasst1_ja.model",
                "../neuroq_tokenizer_oasst1_ja.model",
            ]
            tokenizer_path = None
            for path in tokenizer_candidates:
                if os.path.exists(path):
                    tokenizer_path = path
                    break
            if tokenizer_path is None:
                tokenizer_path = "/root/neuroq/neuroq_tokenizer.model"
                print(f"⚠️ oasst1_ja用トークナイザーが見つからないため、デフォルトを使用")
            else:
                print(f"🔤 oasst1_ja用トークナイザー: {tokenizer_path}")
            # oasst1_ja用のデフォルト設定
            seq_length = 128
            if lr is None or lr == 0.0005:
                lr = 0.0003
        else:
            tokenizer_path = "/root/neuroq/neuroq_tokenizer.model"
        tokenizer = NeuroQuantumTokenizer(vocab_size=model_vocab_size, model_file=tokenizer_path)
        print(f"\n📚 SentencePieceトークナイザー: 語彙サイズ {tokenizer.vocab_size:,}")
        
        # 学習データ準備
        print("\n📊 学習データ準備...")
        texts = []
        
        if dataset_ids and len(dataset_ids) > 0:
            from datasets import load_dataset

            for dataset_id in dataset_ids:
                print(f"   🤗 ロード中: {dataset_id}")
                try:
                    # oasst1_ja の場合はローカルファイルから読み込み（クリーニング済みデータ優先）
                    if dataset_id == "oasst1_ja":
                        data_file_candidates = [
                            "/root/neuroq/data/combined_clean_data.txt",
                            "/root/neuroq/data/oasst1_ja_cleaned.txt",
                            "/root/neuroq/data/oasst1_ja_conversations.txt",
                            "data/combined_clean_data.txt",
                            "data/oasst1_ja_cleaned.txt",
                            "data/oasst1_ja_conversations.txt",
                        ]
                        data_file = None
                        for path in data_file_candidates:
                            if os.path.exists(path):
                                data_file = path
                                break

                        if data_file is None:
                            print(f"   ❌ oasst1_ja: data file not found")
                            print(f"      検索パス: {data_file_candidates}")
                            continue

                        print(f"   📖 oasst1_ja会話データ読み込み: {data_file}")
                        with open(data_file, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 空行で区切られた会話ブロックに分割
                        blocks = content.split('\n\n')
                        oasst1_ja_count = 0
                        for block in blocks:
                            block = block.strip()
                            if block and 'User:' in block and 'Assistant:' in block:
                                texts.append(block)
                                oasst1_ja_count += 1

                        print(f"   ✅ oasst1_ja: {oasst1_ja_count} 会話サンプル")
                        continue

                    if "/" in dataset_id and dataset_id.count("/") >= 2:
                        parts = dataset_id.split("/")
                        ds_name = "/".join(parts[:-1])
                        ds_subset = parts[-1]
                        dataset = load_dataset(ds_name, ds_subset, split=split)
                    else:
                        try:
                            dataset = load_dataset(dataset_id, split=split)
                        except:
                            dataset = load_dataset(dataset_id, "default", split=split)
                    
                    if max_samples and len(dataset) > max_samples:
                        dataset = dataset.shuffle(seed=42).select(range(max_samples))
                    
                    # テキストカラムを探す
                    available_columns = dataset.column_names
                    if text_column in available_columns:
                        target_column = text_column
                    elif "text" in available_columns:
                        target_column = "text"
                    elif "content" in available_columns:
                        target_column = "content"
                    else:
                        target_column = available_columns[0]
                    
                    for item in dataset:
                        text = item.get(target_column, "")
                        if text and isinstance(text, str) and len(text) > 10:
                            texts.append(text)
                    
                    print(f"   ✅ {dataset_id}: {len(dataset)} サンプル")
                except Exception as e:
                    print(f"   ❌ {dataset_id}: {e}")
        
        if len(texts) == 0:
            from neuroquantum_brain import get_training_data
            texts = get_training_data()
            print(f"   📚 デフォルトデータ: {len(texts)} サンプル")
        
        # トークン化
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special=False)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_id)
        print(f"   総トークン数: {len(all_tokens):,}")
        
        # モデル構築
        print("\n🧠 モデル構築...")
        from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig

        nq_config = NeuroQuantumConfig()
        nq_config.vocab_size = config['vocab_size']
        nq_config.embed_dim = config['embed_dim']
        nq_config.num_heads = config['num_heads']
        nq_config.num_layers = config['num_layers']
        nq_config.max_seq_len = config.get('max_seq_len', 512)
        nq_config.dropout = config.get('dropout', 0.1)
        
        model = NeuroQuantum(nq_config).to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   {config['name']}: {total_params:,} パラメータ")
        
        # シーケンス作成
        sequences = []
        for i in range(0, len(all_tokens) - seq_length, seq_length // 2):
            sequences.append(all_tokens[i:i + seq_length])
        print(f"   シーケンス数: {len(sequences):,}")
        
        # 学習ループ
        print("\n🎓 学習開始...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        # 勾配累積設定（メモリ節約）
        accumulation_steps = 4
        effective_batch_size = batch_size * accumulation_steps
        print(f"   実効バッチサイズ: {effective_batch_size} (batch={batch_size} x accum={accumulation_steps})")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            random.shuffle(sequences)
            optimizer.zero_grad()
            
            for i in range(0, len(sequences) - batch_size, batch_size):
                batch = sequences[i:i + batch_size]
                batch_tensor = torch.tensor(batch, device=self.device)
                
                input_ids = batch_tensor[:, :-1]
                target_ids = batch_tensor[:, 1:]
                
                logits = model(input_ids)
                
                loss = criterion(
                    logits.reshape(-1, config['vocab_size']),
                    target_ids.reshape(-1)
                )
                
                # 勾配累積
                loss = loss / accumulation_steps
                loss.backward()
                
                total_loss += loss.item() * accumulation_steps
                batch_count += 1
                
                # 勾配累積ステップでオプティマイザ更新
                if batch_count % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 定期的にキャッシュクリア（メモリ節約）
                if batch_count % 50 == 0:
                    torch.cuda.empty_cache()
                
                # 進捗表示
                if batch_count % 100 == 0:
                    print(f"      Batch {batch_count}: Loss={loss.item() * accumulation_steps:.4f}")
            
            # 残りの勾配を適用
            if batch_count % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # キャッシュクリア
            torch.cuda.empty_cache()
            
            avg_loss = total_loss / max(batch_count, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
            print(f"   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
        
        # チェックポイント保存（Modalボリュームに保存）
        checkpoint_filename = f"neuroq_{model_size}_checkpoint.pt"
        checkpoint_path = f"/model_checkpoints/{checkpoint_filename}"
        os.makedirs("/model_checkpoints", exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'tokenizer': {
                'type': 'sentencepiece',
                'model_path': 'neuroq_tokenizer.model',
                'vocab_size': tokenizer.vocab_size,
            },
            'model_size': model_size,
        }
        
        torch.save(checkpoint, checkpoint_path)
        checkpoints_volume.commit()
        print(f"\n💾 チェックポイント保存: {checkpoint_path}")

        # モデルを更新
        self._load_model(model_size)
        
        print("\n✅ 学習完了！")
        
        return {
            "status": "success",
            "model_size": model_size,
            "parameters": total_params,
            "best_loss": best_loss,
            "epochs": epochs,
            "checkpoint_path": checkpoint_path
        }


# ========================================
# FastAPI Web エンドポイント
# ========================================

@app.function(
    image=image,
    gpu="T4",
    timeout=300,
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
    
    class GenerateResponse(BaseModel):
        status: str
        prompt: str
        generated: str
        model_size: str
        system_prompt: Optional[str] = None
        session_id: Optional[str] = None
    
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
    
    web_app = FastAPI(
        title="NeuroQ API",
        description="Quantum-Bit Neural Network Language Model API",
        version="1.0.0"
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
            "version": "1.0.0",
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
                "session_id": "セッションID（システムプロンプトの永続化用）"
            }
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
                session_id=request.session_id
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
    gpu="A100",  # 学習にはA100推奨（40GB/80GB VRAM）
    timeout=7200,  # 2時間
    volumes={"/model_checkpoints": checkpoints_volume},
)
def train_model(
    model_size: str = "small", 
    epochs: int = 10, 
    batch_size: int = None, 
    lr: float = None,
    dataset_ids: list = None,
    text_column: str = "text",
    split: str = "train",
    max_samples: int = None
):
    """
    Modal GPU上でNeuroQモデルを学習（A100 GPU + Mixed Precision）
    
    Args:
        model_size: 'micro', 'small', 'large'
        epochs: 学習エポック数
        batch_size: バッチサイズ（Noneならデフォルト）
        lr: 学習率（Noneならデフォルト）
        dataset_ids: Hugging FaceデータセットIDのリスト (例: ["wikitext", "openai/summarize_from_feedback"])
        text_column: テキストカラム名 (デフォルト: "text")
        split: データ分割 (デフォルト: "train")
        max_samples: 最大サンプル数/データセット (Noneなら全件)
    
    使い方:
        modal run modal_app.py::train_model --model-size small --epochs 10
        modal run modal_app.py::train_model --dataset-ids '["wikitext"]' --text-column text
    """
    import sys
    import os
    import torch
    import random
    from torch.cuda.amp import autocast, GradScaler
    
    sys.path.insert(0, "/root/neuroq")
    os.chdir("/root/neuroq")
    
    print("=" * 70)
    print(f"🚀 NeuroQ {model_size.upper()} モデル学習（A100 GPU + FP16）")
    print("=" * 70)
    
    # デバイス
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 Device: {device}")
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # メモリ断片化対策
        torch.cuda.empty_cache()
    
    # モジュールインポート
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer
    from neuroquantum_brain import get_training_data
    from model_configs import get_model_config, get_checkpoint_path, AVAILABLE_MODELS
    
    # 学習設定（A100用に調整）
    if batch_size is None:
        batch_size = 8 if model_size == 'micro' else (4 if model_size == 'small' else 2)
    if lr is None:
        lr = 0.0005 if model_size == 'micro' else 0.0003
    seq_length = 64
    
    print(f"\n📋 学習設定:")
    print(f"   model_size: {model_size}")
    print(f"   epochs: {epochs}")
    print(f"   batch_size: {batch_size}")
    print(f"   lr: {lr}")
    if dataset_ids:
        print(f"   dataset_ids: {dataset_ids}")
        print(f"   text_column: {text_column}")
        print(f"   split: {split}")
        if max_samples:
            print(f"   max_samples: {max_samples} (per dataset)")
    
    # モデル設定を先に取得（vocab_size を使うため）
    print("\n📚 SentencePieceトークナイザー初期化...")
    config = get_model_config(model_size)
    model_vocab_size = config['vocab_size']
    tokenizer_path = "neuroq_tokenizer.model"
    tokenizer = NeuroQuantumTokenizer(vocab_size=model_vocab_size, model_file=tokenizer_path)
    print(f"   語彙サイズ: {tokenizer.vocab_size:,}")
    
    # 学習データ
    print("\n📊 学習データ準備...")
    
    texts = []
    
    if dataset_ids and len(dataset_ids) > 0:
        # Hugging Faceデータセットからロード
        from datasets import load_dataset

        for dataset_id in dataset_ids:
            print(f"\n   🤗 ロード中: {dataset_id}")

            try:
                # oasst1_ja の場合はローカルファイルから読み込み（クリーニング済みデータ優先）
                if dataset_id == "oasst1_ja":
                    data_file_candidates = [
                        "/root/neuroq/data/combined_clean_data.txt",
                        "/root/neuroq/data/oasst1_ja_cleaned.txt",
                        "/root/neuroq/data/oasst1_ja_conversations.txt",
                        "data/combined_clean_data.txt",
                        "data/oasst1_ja_cleaned.txt",
                        "data/oasst1_ja_conversations.txt",
                    ]
                    data_file = None
                    for path in data_file_candidates:
                        if os.path.exists(path):
                            data_file = path
                            break

                    if data_file is None:
                        print(f"   ❌ oasst1_ja: data file not found")
                        print(f"      検索パス: {data_file_candidates}")
                        continue

                    print(f"   📖 oasst1_ja会話データ読み込み: {data_file}")
                    with open(data_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 空行で区切られた会話ブロックに分割
                    blocks = content.split('\n\n')
                    oasst1_ja_count = 0
                    for block in blocks:
                        block = block.strip()
                        if block and 'User:' in block and 'Assistant:' in block:
                            texts.append(block)
                            oasst1_ja_count += 1

                    print(f"   ✅ oasst1_ja: {oasst1_ja_count} 会話サンプル")
                    continue

                # データセット名にサブセットが含まれる場合（例: "wikitext/wikitext-2-raw-v1"）
                if "/" in dataset_id and dataset_id.count("/") >= 2:
                    parts = dataset_id.split("/")
                    ds_name = "/".join(parts[:-1])
                    ds_subset = parts[-1]
                    dataset = load_dataset(ds_name, ds_subset, split=split)
                else:
                    # 標準的なデータセット
                    try:
                        dataset = load_dataset(dataset_id, split=split)
                    except:
                        # サブセット指定が必要な場合
                        dataset = load_dataset(dataset_id, "default", split=split)
                
                print(f"   ✅ データセットロード完了: {len(dataset)} サンプル")
                
                # サンプル数制限
                if max_samples and len(dataset) > max_samples:
                    dataset = dataset.shuffle(seed=42).select(range(max_samples))
                    print(f"   📉 サンプル数を {max_samples} に制限")
                
                # テキストを抽出
                available_columns = dataset.column_names
                print(f"   利用可能なカラム: {available_columns}")
                
                # テキストカラムを探す
                if text_column in available_columns:
                    target_column = text_column
                elif "text" in available_columns:
                    target_column = "text"
                elif "content" in available_columns:
                    target_column = "content"
                elif "sentence" in available_columns:
                    target_column = "sentence"
                elif "prompt" in available_columns:
                    target_column = "prompt"
                elif "question" in available_columns:
                    target_column = "question"
                else:
                    # 最初の文字列カラムを使用
                    target_column = available_columns[0]
                    print(f"   ⚠️ テキストカラムが見つかりません。'{target_column}' を使用します")
                
                print(f"   📝 テキストカラム: {target_column}")
                
                dataset_texts = []
                for item in dataset:
                    text = item.get(target_column, "")
                    if text and isinstance(text, str) and len(text) > 10:
                        dataset_texts.append(text)
                
                print(f"   有効なテキスト: {len(dataset_texts)} 件")
                texts.extend(dataset_texts)
                
            except Exception as e:
                print(f"   ❌ データセットロードエラー ({dataset_id}): {e}")
        
        # データセットからテキストが取得できなかった場合はデフォルトにフォールバック
        if len(texts) == 0:
            print("\n   📚 デフォルトデータにフォールバック...")
            texts = get_training_data()
    else:
        # デフォルトの学習データを使用
        texts = get_training_data()
    
    print(f"   サンプル数: {len(texts)}")
    
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special=False)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    print(f"   総トークン数: {len(all_tokens):,}")
    
    # モデル構築
    print("\n🧠 モデル構築...")
    config = get_model_config(model_size)
    
    nq_config = NeuroQuantumConfig()
    nq_config.vocab_size = config['vocab_size']
    nq_config.embed_dim = config['embed_dim']
    nq_config.num_heads = config['num_heads']
    nq_config.num_layers = config['num_layers']
    nq_config.max_seq_len = config.get('max_seq_len', 512)
    nq_config.dropout = config.get('dropout', 0.1)
    
    model = NeuroQuantum(nq_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   {config['name']}: {total_params:,} パラメータ ({total_params/1e6:.1f}M)")
    print(f"   embed_dim: {config['embed_dim']}")
    print(f"   num_heads: {config['num_heads']}")
    print(f"   num_layers: {config['num_layers']}")
    
    # シーケンス作成
    sequences = []
    for i in range(0, len(all_tokens) - seq_length, seq_length // 2):
        sequences.append(all_tokens[i:i + seq_length])
    print(f"\n   シーケンス数: {len(sequences):,}")
    
    # 学習ループ (Mixed Precision FP16)
    print("\n🎓 学習開始（Mixed Precision FP16）...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()  # FP16用スケーラー
    
    total_batches = (len(sequences) - batch_size) // batch_size
    print(f"   バッチ数/エポック: {total_batches:,}")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        random.shuffle(sequences)
        
        for i in range(0, len(sequences) - batch_size, batch_size):
            batch = sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch, device=device)
            
            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward
            with autocast():
                logits = model(input_ids)
                loss = criterion(
                    logits.reshape(-1, config['vocab_size']),
                    target_ids.reshape(-1)
                )
            
            # Mixed Precision Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
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
    
    print(f"\n✅ 学習完了！ ベストLoss: {best_loss:.4f}")
    
    # チェックポイント保存
    checkpoint_filename = f"neuroq_{model_size}_checkpoint.pt"
    checkpoint_path = f"/model_checkpoints/{checkpoint_filename}"
    local_path = f"checkpoints/{checkpoint_filename}"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': {
            'type': 'sentencepiece',
            'model_path': 'neuroq_tokenizer.model',
            'vocab_size': tokenizer.vocab_size,
        },
        'model_size': model_size,
    }

    # ボリュームに保存
    torch.save(checkpoint, checkpoint_path)
    checkpoints_volume.commit()
    print(f"💾 チェックポイント保存: {checkpoint_path}")
    
    # ローカルにも保存（ダウンロード用）
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(checkpoint, local_path)
    print(f"💾 ローカルにもコピー: {local_path}")
    
    print("\n" + "=" * 70)
    print("✅ 学習完了！")
    print("=" * 70)
    
    return {
        "status": "success",
        "model_size": model_size,
        "parameters": total_params,
        "best_loss": best_loss,
        "checkpoint_path": checkpoint_path,
        "epochs": epochs,
    }


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
