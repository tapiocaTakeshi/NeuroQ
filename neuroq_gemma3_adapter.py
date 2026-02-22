import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# ==========================================
# 既存の独自モデル（NeuroQuantum）を読み込むための準備
# ==========================================
import sys
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
    from tiktoken_tokenizer import TikTokenTokenizer
    NEUROQ_AVAILABLE = True
except ImportError:
    NEUROQ_AVAILABLE = False
    print("Warning: neuroquantum_layered.py または tiktoken_tokenizer.py が見つかりません。")

# ==========================================
# 1. Adapter (射影層) の定義
# ==========================================
class QBNNtoGemmaAdapter(nn.Module):
    """
    QBNN（NeuroQuantum）の出力ベクトルを、
    Gemma 3が理解できる次元（hidden_size）に変換するアダプターネットワーク。
    ※このネットワークだけを学習させます。
    """
    def __init__(self, in_features, out_features, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_features, out_features) // 2
            
        # 2層のMLP（GELU活性化関数）で変換表現力を高める
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features, bias=False)
        )

    def forward(self, x):
        return self.proj(x)

# ==========================================
# 2. 統合モデル (NeuroQ + Adapter + Gemma3)
# ==========================================
class NeuroQ_Gemma_Pipeline(nn.Module):
    def __init__(self, neuroq_model, gemma_model, adapter):
        super().__init__()
        self.encoder = neuroq_model
        self.decoder = gemma_model
        self.adapter = adapter
        
        # Encoder(NeuroQ) と Decoder(Gemma) の重みを固定（Frozen）
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        for param in self.decoder.parameters():
            param.requires_grad = False
            
        # Adapterの重みだけを学習対象にする
        for param in self.adapter.parameters():
            param.requires_grad = True

    def extract_neuroq_features(self, token_ids):
        """NeuroQuantumから特徴量（hidden states）を抽出する"""
        x = self.encoder.embedding(token_ids) if self.encoder.embedding else self.encoder.text_embedding(token_ids) + self.encoder.position_embedding(torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0))
        x = self.encoder.dropout(x)
        
        for block in self.encoder.transformer_blocks:
            x = block(x)
            
        features = self.encoder.final_norm(x)
        return features

    def forward(self, neuroq_input_ids, gemma_input_ids, gemma_attention_mask=None, labels=None):
        """
        学習・推論のフォワードパス
        """
        # ① NeuroQuantum で特徴抽出（勾配計算を行わない）
        with torch.no_grad():
            qbnn_features = self.extract_neuroq_features(neuroq_input_ids)
            
        # ② Adapter で Gemma の次元に変換（★ここだけ学習される）
        adapted_embeds = self.adapter(qbnn_features)
        
        # ③ Gemma用の入力ベクトルを取得（勾配計算を行わない）
        with torch.no_grad():
            gemma_text_embeds = self.decoder.model.embed_tokens(gemma_input_ids)
            # Embedding層の出力をスケールする（Llama等と異なる場合がある）
            # gemma_text_embeds = gemma_text_embeds * (self.decoder.config.hidden_size ** 0.5) 
            
        # ④ QBNNの特徴ベクトル と Gemmaのテキストベクトル を結合
        # [batch, seq_qbnn, hidden] + [batch, seq_gemma, hidden] -> [batch, seq_combined, hidden]
        combined_embeds = torch.cat([adapted_embeds, gemma_text_embeds], dim=1)
        
        if labels is not None:
             # QBNNの出力部分（adapted_embeds）のラベルは-100（loss計算から除外）とし、Gemmaテキスト部分のLossのみ計算します
             qbnn_seq_len = adapted_embeds.size(1)
             ignore_labels = torch.full((labels.size(0), qbnn_seq_len), -100, dtype=torch.long, device=labels.device)
             combined_labels = torch.cat([ignore_labels, labels], dim=1)
             
             outputs = self.decoder(inputs_embeds=combined_embeds, labels=combined_labels)
             return outputs.loss
        else:
             outputs = self.decoder(inputs_embeds=combined_embeds)
             return outputs

# ==========================================
# 3. 実行・学習デモスクリプト
# ==========================================
def main():
    print("=" * 70)
    print("🧠 NeuroQ (QBNN) ✖️ Gemma 3 - アダプター学習デモ")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # -------------------------
    # 1. モデルの準備
    # -------------------------
    print("\n[1] Gemma 3 の読み込み中...")
    # 実際のGemma 3モデルに合わせて変更してください（例: google/gemma-3-4b）
    # デモ実行のため軽量な gpt2 を代用していますが、構造は同じです
    llm_model_name = "gpt2" 
    try:
        # LlamaやGemma向け
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
        llm_hidden_size = llm_model.config.hidden_size # Gemma 3のHidden Size
        
        # もしgpt2（デモ用）じゃなく本物のGemmaなら
        if "gemma" in llm_model_name.lower():
            print(f"✅ {llm_model_name} をロードしました。Hidden Size: {llm_hidden_size}")
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
        return

    print("\n[2] NeuroQ の読み込み中...")
    if NEUROQ_AVAILABLE:
        neuroq_config = NeuroQuantumConfig(
            vocab_size=200000, embed_dim=256, hidden_dim=512, 
            num_heads=8, num_layers=4, lambda_entangle=0.5
        )
        neuroq_model = NeuroQuantum(neuroq_config).to(device)
        neuroq_tokenizer = TikTokenTokenizer(encoding_name="o200k_base")
        qbnn_output_dim = neuroq_config.embed_dim
    else:
        print("NeuroQが見つからないため、ダミーモデルを使用します。")
        neuroq_model = nn.Embedding(1000, 256).to(device)
        qbnn_output_dim = 256

    print("\n[3] Adapter の構築...")
    adapter = QBNNtoGemmaAdapter(in_features=qbnn_output_dim, out_features=llm_hidden_size).to(device)
    
    pipeline = NeuroQ_Gemma_Pipeline(neuroq_model, llm_model, adapter)
    total_params = sum(p.numel() for p in pipeline.parameters())
    trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    
    print(f"総パラメータ数: {total_params:,}")
    print(f"学習対象パラメータ数 (Adapterのみ): {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # -------------------------
    # 2. 学習ループ（モック）
    # -------------------------
    print("\n[4] アダプターの学習ループ開始 (ダミーデータでのテスト)")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)

    dummy_qbnn_text = "<USER> QBNNの分析対象データ"
    dummy_llm_target = "このデータは正常なパターンを示しています。<|endoftext|>"

    for epoch in range(5):
        optimizer.zero_grad()
        
        if NEUROQ_AVAILABLE:
            neuroq_ids = torch.tensor([neuroq_tokenizer.encode(dummy_qbnn_text)]).to(device)
        else:
            neuroq_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)

        llm_inputs = llm_tokenizer(dummy_llm_target, return_tensors="pt").to(device)
        llm_ids = llm_inputs.input_ids
        labels = llm_ids.clone()
        
        loss = pipeline(
            neuroq_input_ids=neuroq_ids, 
            gemma_input_ids=llm_ids, 
            labels=labels
        )
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f}")

    print("\n✅ Gemma 3用アダプターの学習が完了しました！")
    print("本番環境では 'google/gemma-3-4b-it' などを指定して実行してください。")

if __name__ == "__main__":
    main()
