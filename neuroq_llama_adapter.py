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
class QBNNtoLlamaAdapter(nn.Module):
    """
    QBNN（NeuroQuantum）の出力ベクトルを、
    LLM（Llama等）が理解できる次元（hidden_size）に変換するアダプターネットワーク。
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
# 2. 統合モデル (NeuroQ + Adapter + LLM)
# ==========================================
class NeuroQ_LLM_Pipeline(nn.Module):
    def __init__(self, neuroq_model, llm_model, adapter):
        super().__init__()
        self.encoder = neuroq_model
        self.decoder = llm_model
        self.adapter = adapter
        
        # Encoder(NeuroQ) と Decoder(LLM) の重みを固定（Frozen）
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        for param in self.decoder.parameters():
            param.requires_grad = False
            
        # Adapterの重みだけを学習対象にする
        for param in self.adapter.parameters():
            param.requires_grad = True

    def extract_neuroq_features(self, token_ids):
        """NeuroQuantumから特徴量（hidden states）を抽出する"""
        # 注意: NeuroQuantumのforwardはデフォルトでロジットを返します。
        # 本来は最終層のLayerNorm後のベクトルが最適ですが、ここでは便宜上
        # embeddingからtransformer blocksを通した出力を取得するハックを行います。
        
        x = self.encoder.embedding(token_ids) if self.encoder.embedding else self.encoder.text_embedding(token_ids) + self.encoder.position_embedding(torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0))
        x = self.encoder.dropout(x)
        
        for block in self.encoder.transformer_blocks:
            x = block(x)
            
        features = self.encoder.final_norm(x)
        return features

    def forward(self, neuroq_input_ids, llm_input_ids, llm_attention_mask=None, labels=None):
        """
        学習・推論のフォワードパス
        
        Args:
            neuroq_input_ids: NeuroQに入力するデータ（例: QBNNに分析させたいデータ）
            llm_input_ids: LLMに入力するテキストデータ（例: プロンプト）
        """
        # ① NeuroQuantum で特徴抽出（勾配計算を行わない）
        with torch.no_grad():
            qbnn_features = self.extract_neuroq_features(neuroq_input_ids)
            
        # ② Adapter で LLM の次元に変換（★ここだけ学習される）
        adapted_embeds = self.adapter(qbnn_features)
        
        # ③ LLM用の入力ベクトルを取得（勾配計算を行わない）
        with torch.no_grad():
            llm_text_embeds = self.decoder.get_input_embeddings()(llm_input_ids)
            
        # ④ QBNNの特徴ベクトル と LLMのテキストベクトル を結合
        # [batch, seq_qbnn, hidden] + [batch, seq_llm, hidden] -> [batch, seq_combined, hidden]
        combined_embeds = torch.cat([adapted_embeds, llm_text_embeds], dim=1)
        
        # ラベルやアテンションマスクがある場合は結合幅に合わせる必要があります
        # ここではシンプルに inputs_embeds として流し込みます
        if labels is not None:
             # QBNNの出力部分（adapted_embeds）のラベルは-100（loss計算から除外）とし、LLMテキスト部分のLossのみ計算します
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
    print("🧠 NeuroQ (QBNN) ✖️ LLM (Llama/GPT) - アダプター学習デモ")
    print("=" * 70)

    # デバイス設定
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # -------------------------
    # 1. モデルの準備
    # -------------------------
    print("\n[1] LLMの読み込み中...")
    # デモ用として軽量な gpt2 を使用。Llama 3 8B を使う場合は "meta-llama/Meta-Llama-3-8B" に変更
    llm_model_name = "gpt2"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
    llm_hidden_size = llm_model.config.hidden_size # gpt2 = 768

    print("\n[2] NeuroQ の読み込み中...")
    if NEUROQ_AVAILABLE:
        neuroq_config = NeuroQuantumConfig(
            vocab_size=200000, embed_dim=256, hidden_dim=512, 
            num_heads=8, num_layers=4, lambda_entangle=0.5
        )
        neuroq_model = NeuroQuantum(neuroq_config).to(device)
        neuroq_tokenizer = TikTokenTokenizer(encoding_name="o200k_base")
        qbnn_output_dim = neuroq_config.embed_dim # 256
    else:
        print("NeuroQが見つからないため、ダミーモデルを使用します。")
        neuroq_model = nn.Embedding(1000, 256).to(device) # ダミー
        qbnn_output_dim = 256

    print("\n[3] Adapter の構築...")
    adapter = QBNNtoLlamaAdapter(in_features=qbnn_output_dim, out_features=llm_hidden_size).to(device)
    
    # パイプラインモデルの構築
    pipeline = NeuroQ_LLM_Pipeline(neuroq_model, llm_model, adapter)
    total_params = sum(p.numel() for p in pipeline.parameters())
    trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    
    print(f"総パラメータ数: {total_params:,}")
    print(f"学習対象パラメータ数 (Adapterのみ): {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # -------------------------
    # 2. 学習ループ（モック）
    # -------------------------
    print("\n[4] アダプターの学習ループ開始 (ダミーデータでのテスト)")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)

    # ダミーデータ: 
    # NeuroQ側には何かの観測データ（例: 量子状態や複雑なログ等）を入力し、
    # LLM側には「それを見た上でどう解釈するか」の模範解答を学習させるイメージ。
    dummy_qbnn_text = "<USER> QBNNの分析対象データ"
    dummy_llm_target = "このデータは正常なパターンを示しています。<|endoftext|>"

    for epoch in range(10):
        optimizer.zero_grad()
        
        # NeuroQ用入力
        if NEUROQ_AVAILABLE:
            neuroq_ids = torch.tensor([neuroq_tokenizer.encode(dummy_qbnn_text)]).to(device)
        else:
            neuroq_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)

        # LLM用テキスト入力・正解ラベル作成
        llm_inputs = llm_tokenizer(dummy_llm_target, return_tensors="pt").to(device)
        llm_ids = llm_inputs.input_ids
        # 次の単語を予測させるため、labelsをそのまま渡す
        labels = llm_ids.clone()
        
        # 推論・Loss計算
        loss = pipeline(
            neuroq_input_ids=neuroq_ids, 
            llm_input_ids=llm_ids, 
            labels=labels
        )
        
        # Adapterのみを更新
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f}")

    print("\n✅ アダプターの学習が完了しました！")
    print("ヒント: 実際のデータセットを用意し、Llama-3-8Bなどを読み込んで本番実行してください。")

if __name__ == "__main__":
    main()
