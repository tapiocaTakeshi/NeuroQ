#!/usr/bin/env python3
"""
NeuroQ チャットインターフェース

学習済みモデルを使って対話的にテキスト生成を行います。
出力はマークダウン形式で表示されます。

トークナイザー:
  - tiktoken (デフォルト): OpenAI GPT-4/GPT-3.5と同じトークン化
  - sentencepiece: 従来のSentencePieceトークナイザー

モデル:
  - NeuroQuantum (デフォルト): neuroquantum_layered.pyのQBNN拡張Transformer
  - NeuroQuantumBrain: neuroquantum_brain.pyの脳型ネットワーク
"""

import sys
import os
import torch
import re
import argparse

# カレントディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# モデル設定をインポート
try:
    from model_configs import AVAILABLE_MODELS, get_model_config, get_checkpoint_path, create_model
    MODEL_CONFIGS_AVAILABLE = True
except ImportError:
    MODEL_CONFIGS_AVAILABLE = False

# モデルのインポート（neuroquantum_layered.pyを優先）
NEUROQUANTUM_LAYERED_AVAILABLE = False
NEUROQUANTUM_BRAIN_AVAILABLE = False

try:
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    NEUROQUANTUM_LAYERED_AVAILABLE = True
    print("✅ NeuroQuantum (neuroquantum_layered.py) をインポートしました")
except ImportError as e:
    print(f"⚠️ neuroquantum_layered.py インポート失敗: {e}")

try:
    from neuroquantum_brain import NeuroQuantumBrain
    NEUROQUANTUM_BRAIN_AVAILABLE = True
except ImportError:
    pass

# トークナイザーのインポート（両方サポート）
TIKTOKEN_AVAILABLE = False
SENTENCEPIECE_AVAILABLE = False
NLLB_AVAILABLE = False
DEEPL_AVAILABLE = False

try:
    from tiktoken_tokenizer import TikTokenTokenizer
    TIKTOKEN_AVAILABLE = True
except ImportError:
    pass

try:
    from neuroquantum_layered import NeuroQuantumTokenizer
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    pass

# 翻訳エンジン（NLLB-200を優先）
try:
    from nllb_translator import NLLB200Translator
    NLLB_AVAILABLE = True
except ImportError:
    pass

try:
    from deepl_translator import DeepLTranslator
    DEEPL_AVAILABLE = True
except ImportError:
    pass

# 設定
TOKENIZER_MODEL_PATH = "neuroq_tokenizer.model"
CHECKPOINT_PATH_SP = "checkpoints/neuroq_checkpoint.pt"  # SentencePiece用（日本語）
CHECKPOINT_PATH_TK = "checkpoints/neuroq_tiktoken_english_checkpoint.pt"  # TikToken用（英語・Micro）
CHECKPOINT_PATH_SMALL = "checkpoints/neuroq_small_checkpoint.pt"  # Small
CHECKPOINT_PATH_LARGE = "checkpoints/neuroq_large_checkpoint.pt"  # Large
DEFAULT_TOKENIZER = "tiktoken"  # デフォルトはtiktoken
DEFAULT_TRANSLATE = True  # 翻訳モード（tiktokenは英語なのでデフォルトオン）
DEFAULT_TRANSLATOR = "nllb"  # 翻訳エンジン: "nllb" or "deepl"


def clean_response(response: str, prompt: str) -> str:
    """
    応答をクリーンアップ
    - <ASSISTANT>タグ以降の実際の応答を抽出
    - 入力プロンプトの重複を削除
    - 特殊タグを削除
    - 余分な空白を整理
    - 文末で終わるように調整（十分な長さがある場合のみ）
    """
    original_response = response
    
    # <ASSISTANT>タグ以降の応答を抽出
    if '<ASSISTANT>' in response:
        # 最後の<ASSISTANT>タグ以降を取得
        parts = response.split('<ASSISTANT>')
        if len(parts) > 1:
            response = parts[-1].strip()
    
    # 入力プロンプトが先頭にある場合は削除
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    # 追加の<USER>タグ以降を削除（次の質問が入らないように）
    response = re.sub(r'<USER>.*', '', response, flags=re.DOTALL)
    response = re.sub(r'<ASSISTANT>.*', '', response, flags=re.DOTALL)
    response = re.sub(r'<[A-Z]+>.*', '', response, flags=re.DOTALL)
    
    # 余分な空白を整理
    response = re.sub(r'\s+', ' ', response).strip()
    
    # 文末記号で終わるように調整（最低20文字以上の場合のみ）
    if len(response) >= 20:
        sentence_endings = ['。', '！', '？', '!', '?', '.', '」', '）', ')']
        
        # 最後の文末記号を探す
        last_ending_pos = -1
        for ending in sentence_endings:
            pos = response.rfind(ending)
            if pos > last_ending_pos:
                last_ending_pos = pos
        
        # 文末記号が見つかった場合、そこで切る（ただし最低10文字は残す）
        if last_ending_pos > 10:
            response = response[:last_ending_pos + 1]
    
    return response


def format_markdown(prompt: str, response: str, turn_number: int) -> str:
    """マークダウン形式で出力を整形"""
    md = []
    md.append(f"### 💬 ターン {turn_number}")
    md.append("")
    md.append(f"**👤 あなた:**")
    md.append(f"> {prompt}")
    md.append("")
    md.append(f"**🧠 NeuroQ:**")
    md.append(f"> {response}")
    md.append("")
    md.append("---")
    return "\n".join(md)


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, device='cpu'):
    """
    テキスト生成
    
    Args:
        model: 学習済みモデル
        tokenizer: トークナイザー
        prompt: 入力プロンプト
        max_length: 最大生成トークン数（デフォルト100）
        temperature: 温度パラメータ（0.1-2.0推奨）
        device: デバイス
    """
    model.eval()
    
    # 温度の範囲をクリップ（極端な値を防ぐ）
    temperature = max(0.1, min(2.0, temperature))
    
    # プロンプトを学習形式にラップ（<USER>...<ASSISTANT>形式）
    formatted_prompt = f"<USER> {prompt} <ASSISTANT>"
    
    # プロンプトをエンコード
    input_ids = tokenizer.encode(formatted_prompt, add_special=True)
    
    # 生成
    generated = input_ids.copy()
    
    # 文末トークンのIDを取得（可能な場合）
    sentence_end_tokens = []
    for char in ['。', '！', '？', '!', '?', '.']:
        try:
            ids = tokenizer.encode(char, add_special=False)
            sentence_end_tokens.extend(ids)
        except:
            pass
    
    sentences_generated = 0
    min_sentences = 2  # 最低2文生成
    
    with torch.no_grad():
        for i in range(max_length):
            # 現在のシーケンスで予測
            seq_tensor = torch.tensor([generated[-256:]], device=device)  # 最大256トークン
            logits = model(seq_tensor)
            
            # 最後のトークンの予測
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-pサンプリング（より良い品質のため）
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Top-k filtering (k=50)
            top_k = 50
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            top_k_probs = top_k_probs / top_k_probs.sum()  # 再正規化
            
            # サンプリング
            idx = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_indices[idx].item()
            
            # EOSトークンで終了（ただし最低限の文は生成）
            if next_token == tokenizer.eos_id:
                if sentences_generated >= min_sentences:
                    break
                else:
                    continue  # EOSを無視して続ける
            
            generated.append(next_token)
            
            # 文末トークンをカウント
            if next_token in sentence_end_tokens:
                sentences_generated += 1
                # 十分な文を生成したら、次の文末で終了
                if sentences_generated >= min_sentences and i > max_length // 2:
                    break
    
    # デコード
    output = tokenizer.decode(generated, skip_special=True)
    return output


def print_header(translate_mode: bool = False, translator_engine: str = "nllb"):
    """ヘッダーをマークダウン形式で表示"""
    translate_status = "✅ ON" if translate_mode else "❌ OFF"
    translator_name = translator_engine.upper() if translate_mode else "-"
    print(f"""
# 🧠⚛️ NeuroQ チャット

**脳型量子ビットネットワークによる対話AI**

---

## 📋 使い方

| コマンド | 説明 |
|---------|------|
| テキスト入力 | AIが応答を生成 |
| `exit`, `quit`, `q` | チャット終了 |
| `set max_length=<数値>` | 最大生成長を変更 |
| `set temperature=<数値>` | 温度を変更 (0.1-2.0) |
| `markdown on/off` | マークダウン出力切替 |
| `translate on/off` | 翻訳モード切替 |
| `save` | 会話をファイルに保存 |

**翻訳モード**: {translate_status}
**翻訳エンジン**: {translator_name}
(日本語→英語→AI→英語→日本語)

---

## 💬 会話開始
""")


def main():
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='NeuroQ チャットインターフェース')
    parser.add_argument('--tokenizer', '-t', choices=['tiktoken', 'sentencepiece', 'sp'],
                        default=DEFAULT_TOKENIZER,
                        help='使用するトークナイザー (default: sentencepiece)')
    parser.add_argument('--encoding', '-e', default='o200k_base',
                        help='tiktokenのエンコーディング (default: o200k_base)')
    parser.add_argument('--translate', action='store_true',
                        default=DEFAULT_TRANSLATE,
                        help='翻訳モードを有効にする')
    parser.add_argument('--translator', choices=['nllb', 'deepl'],
                        default=DEFAULT_TRANSLATOR,
                        help='翻訳エンジン (default: nllb)')
    parser.add_argument('--model-size', '-m', choices=['micro', 'small', 'large'],
                        default='micro',
                        help='モデルサイズ: micro(128), small(1536), large(3072)')
    args = parser.parse_args()
    
    print()
    
    # デバイス選択
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_info = "🍎 Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_info = "🎮 NVIDIA GPU (CUDA)"
    else:
        device = torch.device("cpu")
        device_info = "💻 CPU"
    
    # トークナイザーを選択してロード
    tokenizer_type = args.tokenizer
    if tokenizer_type == 'sp':
        tokenizer_type = 'sentencepiece'
    
    print(f"📚 トークナイザーをロード中...")
    
    if tokenizer_type == 'tiktoken':
        if not TIKTOKEN_AVAILABLE:
            print("⚠️ tiktokenが利用できません。SentencePieceにフォールバックします。")
            tokenizer_type = 'sentencepiece'
        else:
            tokenizer = TikTokenTokenizer(encoding_name=args.encoding)
            # tiktokenの語彙サイズをモデルに合わせる
            vocab_size = 200019  # o200k_baseの語彙サイズ
            tokenizer_info = f"TikToken ({args.encoding})"
            # モデルサイズに応じたチェックポイント
            if args.model_size == 'small':
                checkpoint_path = CHECKPOINT_PATH_SMALL
            elif args.model_size == 'large':
                checkpoint_path = CHECKPOINT_PATH_LARGE
            else:
                checkpoint_path = CHECKPOINT_PATH_TK  # micro
    
    if tokenizer_type == 'sentencepiece':
        if not SENTENCEPIECE_AVAILABLE:
            print("❌ SentencePieceが利用できません。")
            sys.exit(1)
        if not os.path.exists(TOKENIZER_MODEL_PATH):
            print(f"❌ トークナイザーモデルが見つかりません: {TOKENIZER_MODEL_PATH}")
            sys.exit(1)
        tokenizer = NeuroQuantumTokenizer(vocab_size=8000, model_file=TOKENIZER_MODEL_PATH)
        vocab_size = tokenizer.vocab_size
        tokenizer_info = f"SentencePiece ({TOKENIZER_MODEL_PATH})"
        checkpoint_path = CHECKPOINT_PATH_SP
    
    # チェックポイントをロード
    print(f"💾 チェックポイントをロード: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ チェックポイントが見つかりません: {checkpoint_path}")
        sys.exit(1)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # チェックポイントの構造を確認
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            config = {}
        
        # モデル設定を取得（チェックポイントの設定を優先）
        model_vocab_size = config.get('vocab_size', 200019)  # o200k_baseのデフォルト
        embed_dim = config.get('embed_dim', 128)
        num_heads = config.get('num_heads', 4)
        num_layers = config.get('num_layers', 3)
        num_neurons = config.get('num_neurons', 100)
        
    except Exception as e:
        print(f"❌ チェックポイントのロードに失敗: {e}")
        sys.exit(1)
    
    # 翻訳エンジン初期化
    translator = None
    translate_mode = args.translate
    translator_engine = args.translator
    
    if translate_mode:
        # NLLB-200を優先
        if translator_engine == "nllb" and NLLB_AVAILABLE:
            try:
                print("\n🌐 NLLB-200 翻訳エンジンを初期化中...")
                translator = NLLB200Translator()
                translator_engine = "NLLB-200"
                print("✅ NLLB-200翻訳モードを有効化しました（オフライン・APIキー不要）")
            except Exception as e:
                print(f"⚠️ NLLB-200を初期化できません: {e}")
                print("   DeepLにフォールバックを試みます...")
                translator_engine = "deepl"
        
        # DeepLを試行（フォールバック）
        if translator is None and (translator_engine == "deepl" or translator_engine == "NLLB-200"):
            if DEEPL_AVAILABLE:
                try:
                    translator = DeepLTranslator()
                    translator_engine = "DeepL"
                    print("✅ DeepL翻訳モードを有効化しました")
                except Exception as e:
                    print(f"⚠️ DeepL翻訳を初期化できません: {e}")
                    translate_mode = False
            else:
                if not NLLB_AVAILABLE:
                    print("❌ 翻訳エンジンが利用できません。")
                    print("   NLLB-200: transformersをインストールしてください")
                    print("   DeepL: DEEPL_API_KEYを設定してください")
                    translate_mode = False
    
    # モデルを構築（NeuroQuantumを優先、フォールバックでNeuroQuantumBrain）
    print("\n🧠 モデルを構築中...")
    
    # モデルサイズを表示
    model_size = args.model_size
    print(f"   📦 モデルサイズ: {model_size.upper()}")
    
    if NEUROQUANTUM_LAYERED_AVAILABLE:
        # model_configsが利用可能な場合、設定を使用
        if MODEL_CONFIGS_AVAILABLE and model_size in ['small', 'large']:
            model_config = get_model_config(model_size)
            config_obj = NeuroQuantumConfig()
            config_obj.vocab_size = model_config['vocab_size']
            config_obj.embed_dim = model_config['embed_dim']
            config_obj.num_heads = model_config['num_heads']
            config_obj.num_layers = model_config['num_layers']
            config_obj.max_seq_len = model_config.get('max_seq_len', 512)
            config_obj.dropout = model_config.get('dropout', 0.1)
            
            # チェックポイントの設定を上書き
            embed_dim = model_config['embed_dim']
            num_heads = model_config['num_heads']
            num_layers = model_config['num_layers']
            model_vocab_size = model_config['vocab_size']
            
            model = NeuroQuantum(config_obj)
            model_type = f"NeuroQuantum {model_config['name']}"
        else:
            # 従来の方法（microまたはチェックポイントから読み込み）
            config_obj = NeuroQuantumConfig()
            config_obj.vocab_size = model_vocab_size
            config_obj.embed_dim = embed_dim
            config_obj.num_heads = num_heads
            config_obj.num_layers = num_layers
            config_obj.max_seq_len = 256
            config_obj.dropout = 0.1
            
            model = NeuroQuantum(config_obj)
            model_type = "NeuroQuantum (QBNN Transformer)"
        
        print(f"   ✅ {model_type} を使用")
    elif NEUROQUANTUM_BRAIN_AVAILABLE:
        # NeuroQuantumBrain (neuroquantum_brain.py) を使用
        model = NeuroQuantumBrain(
            vocab_size=model_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_neurons=num_neurons
        )
        model_type = "NeuroQuantumBrain (脳型ネットワーク)"
        print(f"   ✅ {model_type} を使用")
    else:
        print("❌ モデルが利用できません。")
        sys.exit(1)
    
    # 重みをロード
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # ヘッダー表示
    print_header(translate_mode, translator_engine)
    
    # 翻訳モード情報
    if translate_mode:
        translate_info = f"✅ {translator_engine} (日本語⇔英語)"
    else:
        translate_info = "❌ OFF"
    
    # モデル情報をマークダウンで表示
    print(f"""
### 📊 モデル情報

| 項目 | 値 |
|-----|-----|
| デバイス | {device_info} |
| モデル | {model_type} |
| トークナイザー | {tokenizer_info} |
| 翻訳モード | {translate_info} |
| トークナイザー語彙サイズ | {tokenizer.vocab_size:,} |
| モデル語彙サイズ | {model_vocab_size:,} |
| 埋め込み次元 | {embed_dim} |
| アテンションヘッド数 | {num_heads} |
| Transformerブロック数 | {num_layers} |
| 総パラメータ数 | {sum(p.numel() for p in model.parameters()):,} |

---
""")
    
    # デフォルト設定
    max_length = 100
    temperature = 0.8
    markdown_mode = True
    conversation_history = []
    turn_number = 0
    
    print("> 入力待ち... (`exit`で終了)")
    print()
    
    while True:
        try:
            user_input = input("**あなた:** ").strip()
            
            if not user_input:
                continue
            
            # 終了コマンド
            if user_input.lower() in ['exit', 'quit', 'q', '終了']:
                print()
                print("---")
                print()
                print("## 👋 セッション終了")
                print()
                print(f"**総ターン数:** {turn_number}")
                print()
                print("ありがとうございました！")
                break
            
            # 翻訳モード切替
            if user_input.lower() in ['translate on', 'nllb on', 'tr on']:
                if translator is None:
                    # NLLB-200を優先
                    if NLLB_AVAILABLE:
                        try:
                            print("> 🌐 NLLB-200を初期化中...")
                            translator = NLLB200Translator()
                            translator_engine = "NLLB-200"
                            translate_mode = True
                            print(f"> ✅ 翻訳モード: ON ({translator_engine})")
                        except Exception as e:
                            print(f"> ⚠️ NLLB-200初期化エラー: {e}")
                            # DeepLにフォールバック
                            if DEEPL_AVAILABLE:
                                try:
                                    translator = DeepLTranslator()
                                    translator_engine = "DeepL"
                                    translate_mode = True
                                    print(f"> ✅ 翻訳モード: ON ({translator_engine})")
                                except Exception as e2:
                                    print(f"> ❌ DeepL初期化エラー: {e2}")
                    elif DEEPL_AVAILABLE:
                        try:
                            translator = DeepLTranslator()
                            translator_engine = "DeepL"
                            translate_mode = True
                            print(f"> ✅ 翻訳モード: ON ({translator_engine})")
                        except Exception as e:
                            print(f"> ❌ DeepL初期化エラー: {e}")
                    else:
                        print("> ❌ 翻訳エンジンが利用できません。")
                else:
                    translate_mode = True
                    print(f"> ✅ 翻訳モード: ON ({translator_engine})")
                continue
            if user_input.lower() in ['translate off', 'nllb off', 'tr off']:
                translate_mode = False
                print("> ✅ 翻訳モード: OFF")
                continue
            
            # マークダウンモード切替
            if user_input.lower() in ['markdown on', 'md on']:
                markdown_mode = True
                print("> ✅ マークダウンモード: ON")
                continue
            if user_input.lower() in ['markdown off', 'md off']:
                markdown_mode = False
                print("> ✅ マークダウンモード: OFF")
                continue
            
            # 会話保存
            if user_input.lower() == 'save':
                filename = f"chat_log_{len(conversation_history)}.md"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("# NeuroQ チャットログ\n\n")
                    for item in conversation_history:
                        f.write(item + "\n\n")
                print(f"> ✅ 会話を保存しました: `{filename}`")
                continue
            
            # 設定コマンド
            if user_input.startswith('set '):
                try:
                    setting = user_input[4:].strip()
                    if '=' in setting:
                        key, value = setting.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'max_length':
                            max_length = int(value)
                            print(f"> ⚙️ `max_length` = **{max_length}**")
                        elif key == 'temperature':
                            temperature = float(value)
                            print(f"> ⚙️ `temperature` = **{temperature}**")
                        else:
                            print(f"> ❌ 不明な設定: `{key}`")
                except Exception as e:
                    print(f"> ❌ 設定エラー: {e}")
                continue
            
            # テキスト生成
            try:
                turn_number += 1
                
                # 翻訳モードの場合: 日本語→英語→AI→英語→日本語
                if translate_mode and translator:
                    # Step 1: 日本語を英語に翻訳
                    print("> 🔄 翻訳中 (日本語→英語)...")
                    english_prompt = translator.translate_to_english(user_input)
                    
                    # Step 2: 英語でAIに問い合わせ
                    print("> 🧠 AI処理中 (英語)...")
                    raw_response = generate_text(
                        model, 
                        tokenizer,
                        english_prompt, 
                        max_length=max_length,
                        temperature=temperature,
                        device=device
                    )
                    
                    # 応答をクリーンアップ
                    english_response = clean_response(raw_response, english_prompt)
                    
                    # Step 3: 英語を日本語に翻訳
                    print("> 🔄 翻訳中 (英語→日本語)...")
                    japanese_response = translator.translate_to_japanese(english_response)
                    
                    # マークダウン形式で翻訳過程も表示
                    if markdown_mode:
                        md_output = f"""### 💬 ターン {turn_number}

**👤 あなた (日本語):**
> {user_input}

**🔄 翻訳後 (英語):**
> {english_prompt}

**🧠 NeuroQ (英語):**
> {english_response}

**📝 最終応答 (日本語):**
> {japanese_response}

---"""
                        print()
                        print(md_output)
                        conversation_history.append(md_output)
                    else:
                        print(f"**NeuroQ:** {japanese_response}")
                        print()
                        conversation_history.append(f"あなた: {user_input}\nNeuroQ: {japanese_response}")
                
                else:
                    # 通常モード（翻訳なし）
                    raw_response = generate_text(
                        model, 
                        tokenizer,
                        user_input, 
                        max_length=max_length,
                        temperature=temperature,
                        device=device
                    )
                    
                    # 応答をクリーンアップ
                    clean_resp = clean_response(raw_response, user_input)
                    
                    if markdown_mode:
                        # マークダウン形式で出力
                        md_output = format_markdown(user_input, clean_resp, turn_number)
                        print()
                        print(md_output)
                        conversation_history.append(md_output)
                    else:
                        # プレーンテキストで出力
                        print(f"**NeuroQ:** {clean_resp}")
                        print()
                        conversation_history.append(f"あなた: {user_input}\nNeuroQ: {clean_resp}")
                
            except Exception as e:
                print(f"> ❌ 生成エラー: {e}")
                import traceback
                traceback.print_exc()
            
        except KeyboardInterrupt:
            print("\n\n## 👋 チャットを中断しました。")
            break
        except EOFError:
            print("\n\n## 👋 チャットを終了します。")
            break


if __name__ == "__main__":
    main()
