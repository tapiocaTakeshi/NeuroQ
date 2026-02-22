import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

# ==========================================
# 既存の独自モデル（NeuroQuantum）を読み込む準備
# ==========================================
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if os.path.join(current_dir, 'neuroq-runpod') not in sys.path:
    sys.path.insert(0, os.path.join(current_dir, 'neuroq-runpod'))

try:
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    from tiktoken_tokenizer import TikTokenTokenizer
    NEUROQ_AVAILABLE = True
except ImportError as e:
    NEUROQ_AVAILABLE = False
    print(f"Warning: neuroquantum_layered.py または tiktoken_tokenizer.py が見つかりません。{e}")

# ==========================================
# 1. Adapter (射影層) の定義
# ==========================================
class QBNNtoGemmaAdapter(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_features, out_features) // 2
            
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features, bias=False)
        )

    def forward(self, x):
        return self.proj(x)

# ==========================================
# 2. 特徴抽出関数
# ==========================================
def extract_neuroq_features(neuroq_model, token_ids):
    # NeuroQuantumから特徴量（hidden states）を抽出する
    x = neuroq_model.embedding(token_ids) if neuroq_model.embedding else neuroq_model.text_embedding(token_ids) + neuroq_model.position_embedding(torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0))
    x = neuroq_model.dropout(x)
    
    for block in neuroq_model.transformer_blocks:
        x = block(x)
        
    features = neuroq_model.final_norm(x)
    return features


# ==========================================
# 3. インタラクティブチャット
# ==========================================
def main():
    print("=" * 70)
    print("🧠 NeuroQ (QBNN) ✖️ Gemma 3 - インタラクティブチャット デモ")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ==================================
    # [設定] 使用するLLMモデル名
    # ==================================
    # 本来は "google/gemma-3-4b-it" などを指定します。
    # ここではテストコードがGPU上で即座に動くよう、極小サイズのモデル（SmolLMなど）やgpt2をプロキシとして使用します。
    # ユーザー環境にGemma3がダウンロードされている場合は、コメントを外して切り替えてください。
    
    llm_model_name = "gpt2" 
    # llm_model_name = "google/gemma-3-4b-it" # <- 実際にGemma 3を動かす場合はこちら

    print(f"\n[1] LLM ({llm_model_name}) の読み込み中...")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if hasattr(llm_tokenizer, "pad_token") and llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
            
        llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
        llm_model.eval()
        llm_hidden_size = llm_model.config.hidden_size 
        print(f"✅ LLM をロードしました。Hidden Size: {llm_hidden_size}")
    except Exception as e:
        print(f"❌ モデルのロードに失敗しました: {e}")
        return

    print("\n[2] NeuroQ の読み込み中...")
    if NEUROQ_AVAILABLE:
        neuroq_config = NeuroQuantumConfig(
            vocab_size=50257, embed_dim=256, hidden_dim=512, 
            num_heads=8, num_layers=4, lambda_entangle=0.8
        )
        neuroq_model = NeuroQuantum(neuroq_config).to(device)
        neuroq_model.eval()
        neuroq_tokenizer = TikTokenTokenizer(encoding_name="p50k_base")
        qbnn_output_dim = neuroq_config.embed_dim
        print("✅ NeuroQuantum をロードしました。")
    else:
        print("NeuroQが見つからないため、ダミーモデルを使用します。")
        neuroq_model = nn.Embedding(1000, 256).to(device)
        neuroq_model.eval()
        qbnn_output_dim = 256

    print("\n[3] Adapter の構築中...")
    adapter = QBNNtoGemmaAdapter(in_features=qbnn_output_dim, out_features=llm_hidden_size).to(device)
    adapter.eval()
    print("✅ Adapter をロードしました (今回は学習前(Random Weight)のAdapterを使用します)。")

    print("\n" + "="*70)
    print("💬 チャットを開始します！ (終了するには 'quit' または 'exit' と入力)")
    print("="*70)

    # チャット履歴（システムプロンプトの代わり）
    chat_history = ""

    while True:
        try:
            user_input = input("\n👤 ユーザー: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue
            
            # 1. 入力テキストの準備 (Gemma 3 Instructのプロンプト形式に合わせる場合もありますが、簡易的に繋げます)
            prompt = f"{chat_history}User: {user_input}\nAssistant:"
            
            # --- QBNN側の処理 ---
            if NEUROQ_AVAILABLE:
                # QBNNにはユーザー入力の特徴を計算させる
                neuroq_ids = torch.tensor([neuroq_tokenizer.encode(user_input)]).to(device)
                with torch.no_grad():
                    qbnn_features = extract_neuroq_features(neuroq_model, neuroq_ids)
            else:
                neuroq_ids = torch.tensor([[1, 2, 3]]).to(device) # ダミー
                with torch.no_grad():
                    qbnn_features = neuroq_model(neuroq_ids)
                    
            # --- Adapter層の適用 ---
            with torch.no_grad():
                adapted_embeds = adapter(qbnn_features)
            
            # --- LLM側の処理 ---
            llm_inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
            llm_ids = llm_inputs.input_ids
            
            with torch.no_grad():
                # Llama/Gemmaの場合、Word Embeddingの取得方法は異なります
                # 例: llm_model.model.embed_tokens(llm_ids)
                if hasattr(llm_model, "model") and hasattr(llm_model.model, "embed_tokens"):
                    llm_text_embeds = llm_model.model.embed_tokens(llm_ids)
                else:
                    # GPT-2等の場合は get_input_embeddings() 
                    llm_text_embeds = llm_model.get_input_embeddings()(llm_ids)
            
            # 特徴量結合 [QBNNの理解結果(Adapter通過後)] + [現在のプロンプト]
            # こうすることで、LLMは「QBNNの解析結果」を踏まえた上でテキストを生成し始めます
            combined_embeds = torch.cat([adapted_embeds, llm_text_embeds], dim=1)
            attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=device)
            
            print("🤖 Gemma3(NeuroQ連携) : ", end="", flush=True)
            
            # --- 推論 (Generation) ---
            with torch.no_grad():
                # inputs_embeds を使って generate する
                # ※ inputs_embeds を渡すと、生成途中に追加される新しいTokenは
                # モデル内部のEmbeddingを自動で使ってループしてくれます。
                generated_outputs = llm_model.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=llm_tokenizer.pad_token_id,
                    eos_token_id=llm_tokenizer.eos_token_id
                )
            
            # 出力のデコード
            # 生成されたトークンは元プロンプトを含まない(inputs_embedsで渡した分は出力されない仕様の場合もある)ため確認
            # output_ids には新しく生成されたトークンのみが含まれる機種と、そうでない機種があるため安全に処理
            response_text = llm_tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
            
            # 空の場合はエラー回避
            if not response_text.strip():
                response_text = "(モデルからのテキスト生成がありませんでした。無学習プロキシモデルのためです)"
                
            print(response_text)
            
            # 簡単な履歴の追加
            chat_history += f"User: {user_input}\nAssistant: {response_text}\n"

        except KeyboardInterrupt:
            print("\nチャットを終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
