#!/usr/bin/env python3
"""
NeuroQ CLI - コマンドライン インターフェース
============================================
学習・データ取得・生成を統合したCLIツール

機能:
- Common Crawl からのデータ取得
- Wikipedia / ニュースサイトからのデータ取得
- Brain Mode / Layered Mode での学習
- テキスト生成

使用例:
python neuroq_cli.py --prompt "量子コンピュータについて" --mode brain --train --epochs 30
"""

import argparse
import json
import os
import sys
import random
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

# 親ディレクトリをパスに追加
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from neuroq_model import (
    NeuroQModel, 
    NeuroQConfig, 
    NeuroQTokenizer, 
    NeuroQGenerator,
    create_neuroq_brain,
    create_neuroq_layered,
)


# ========================================
# データ取得: Common Crawl
# ========================================

def fetch_common_crawl_data(
    domains: List[str] = None,
    max_records: int = 100,
    crawl_id: str = "CC-MAIN-2024-10",
    language: str = "ja",
) -> List[str]:
    """
    Common Crawl からデータを取得
    
    Args:
        domains: 取得するドメインのリスト（ワイルドカード対応）
        max_records: 最大レコード数
        crawl_id: クロールID
        language: 言語コード
    """
    print(f"\n📡 Common Crawl からデータ取得中...")
    print(f"   Crawl ID: {crawl_id}")
    print(f"   Domains: {domains}")
    print(f"   Max Records: {max_records}")
    
    texts = []
    
    try:
        # warcio と requests が必要
        import requests
        
        # Common Crawl Index API
        # 注: 実際の Common Crawl アクセスには追加の設定が必要
        # ここでは代替データを使用
        
        print("   ⚠️ Common Crawl への直接アクセスは設定が必要です")
        print("   📚 代替データソースを使用します")
        
        # 代替: Hugging Face からデータ取得
        texts = fetch_huggingface_data(max_samples=max_records, language=language)
        
    except ImportError:
        print("   ⚠️ requests/warcio がインストールされていません")
        print("   📚 組み込みデータを使用します")
        texts = get_builtin_training_data()
    except Exception as e:
        print(f"   ⚠️ Common Crawl 取得エラー: {e}")
        texts = get_builtin_training_data()
    
    print(f"   ✅ {len(texts)} サンプル取得完了")
    return texts


# ========================================
# データ取得: Hugging Face
# ========================================

def fetch_huggingface_data(
    max_samples: int = 500,
    language: str = "ja",
    datasets_list: List[str] = None,
) -> List[str]:
    """Hugging Face からデータを取得"""
    print(f"\n📡 Hugging Face からデータ取得中...")
    
    texts = []
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("   ⚠️ datasets ライブラリがありません")
        return get_builtin_training_data()
    
    # 日本語データセット
    if language == "ja":
        # Wikipedia 日本語
        try:
            print("   📚 Wikipedia 日本語...")
            ds = load_dataset("range3/wiki40b-ja", split="train", streaming=True)
            count = 0
            for item in ds:
                if 'text' in item and len(item['text']) > 50:
                    texts.append(item['text'][:500])
                    count += 1
                    if count >= max_samples // 3:
                        break
            print(f"      ✅ {count} サンプル")
        except Exception as e:
            print(f"      ⚠️ エラー: {e}")
        
        # 日本語対話データ
        try:
            print("   📚 databricks-dolly-15k-ja...")
            ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
            count = 0
            for item in ds:
                if count >= max_samples // 3:
                    break
                if 'output' in item and len(item['output']) > 20:
                    texts.append(item['output'])
                    count += 1
            print(f"      ✅ {count} サンプル")
        except Exception as e:
            print(f"      ⚠️ エラー: {e}")
    
    # 英語データセット
    else:
        try:
            print("   📚 wikitext-2...")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            count = 0
            for item in ds:
                if count >= max_samples:
                    break
                if 'text' in item and len(item['text']) > 30:
                    texts.append(item['text'])
                    count += 1
            print(f"      ✅ {count} サンプル")
        except Exception as e:
            print(f"      ⚠️ エラー: {e}")
    
    print(f"   ✅ 合計 {len(texts)} サンプル取得完了")
    return texts


# ========================================
# 組み込み学習データ
# ========================================

def get_builtin_training_data() -> List[str]:
    """組み込みの学習データを取得"""
    
    texts = [
        # 量子コンピューティング
        "量子コンピュータは、量子力学の原理を利用して情報を処理する革新的な計算機です。",
        "量子ビットは0と1の重ね合わせ状態を取ることができます。この性質により並列計算が可能になります。",
        "量子エンタングルメントは、複数の量子ビットが強く相関した状態です。量子通信や量子暗号に応用されています。",
        "量子コンピューティングは暗号解読や最適化問題で注目されています。",
        "量子超越性とは、量子コンピュータが古典コンピュータよりも高速に計算できることを示す概念です。",
        "量子アルゴリズムには、ショアのアルゴリズムやグローバーのアルゴリズムなどがあります。",
        "量子誤り訂正は、量子計算の信頼性を高めるための重要な技術です。",
        "超伝導量子ビットやイオントラップ量子ビットなど、様々な実装方式が研究されています。",
        "量子アニーリングは、組み合わせ最適化問題を解くための量子計算手法です。",
        "量子テレポーテーションは、量子情報を離れた場所に転送する技術です。",
        
        # AI・機械学習
        "人工知能は、人間の知的活動をコンピュータで実現しようとする技術です。",
        "機械学習は、データから学習して改善するシステムを実現します。",
        "深層学習は、多層のニューラルネットワークを用いた機械学習手法です。",
        "トランスフォーマーは、自己注意機構を用いた革新的なアーキテクチャです。",
        "自然言語処理は、コンピュータが人間の言語を理解する技術です。",
        "生成AIは、テキストや画像を生成できる人工知能です。",
        "ニューラルネットワークは、脳の神経回路を模倣した計算システムです。",
        "強化学習は、エージェントが環境との相互作用を通じて学習する手法です。",
        
        # QBNN関連
        "QBNNは量子ビットニューラルネットワークの略称です。",
        "APQBは調整可能擬似量子ビットで、量子状態を古典的に模倣します。",
        "ニューロQは、QBNNベースの生成AIシステムです。",
        "量子もつれを模倣したニューラルネットワークが研究されています。",
        
        # 対話
        "こんにちは、私はニューロQです。何かお手伝いできることはありますか？",
        "量子コンピュータについて知りたいですか？詳しく説明しますね。",
        "人工知能の仕組みについて質問がありますか？",
        "何でも聞いてください。できる限りお答えします。",
    ]
    
    # データ拡張
    expanded = texts * 20
    
    prefixes = ["実際に、", "興味深いことに、", "重要なのは、", "特に、", "さらに、"]
    for text in texts[:20]:
        for prefix in prefixes:
            expanded.append(prefix + text)
    
    return expanded


# ========================================
# 学習
# ========================================

def train_model(
    texts: List[str],
    mode: str = "brain",
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    num_neurons: int = 100,
    hidden_dim: int = 256,
    embed_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    seq_length: int = 64,
    connection_density: float = 0.25,
    lambda_entangle: float = 0.35,
    save_path: str = None,
) -> tuple:
    """
    モデルを学習
    
    Returns:
        (model, tokenizer, generator)
    """
    print("\n" + "=" * 60)
    print(f"🎓 NeuroQ 学習開始 (Mode: {mode})")
    print("=" * 60)
    
    # デバイス選択
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) を使用")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎮 CUDA GPU を使用: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("💻 CPU を使用")
    
    # トークナイザー構築
    print("\n🔤 トークナイザー構築...")
    tokenizer = NeuroQTokenizer(vocab_size=8000)
    tokenizer.build_vocab(texts)
    print(f"   語彙サイズ: {tokenizer.actual_vocab_size}")
    
    # モデル構築
    print(f"\n🧠 NeuroQ モデル構築 ({mode} mode)...")
    
    if mode == "brain":
        config = NeuroQConfig(
            mode='brain',
            vocab_size=tokenizer.actual_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=num_neurons * 2,
            num_neurons=num_neurons,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=256,
            dropout=0.1,
            connection_density=connection_density,
            lambda_entangle=lambda_entangle,
        )
    else:  # layered
        config = NeuroQConfig(
            mode='layered',
            vocab_size=tokenizer.actual_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=256,
            dropout=0.1,
            lambda_entangle=lambda_entangle,
        )
    
    model = NeuroQModel(config).to(device)
    
    print(f"\n📊 モデル構成:")
    print(f"   Mode: {mode}")
    print(f"   埋め込み次元: {embed_dim}")
    if mode == "brain":
        print(f"   ニューロン数: {num_neurons}")
        print(f"   接続密度: {connection_density}")
    else:
        print(f"   隠れ層次元: {hidden_dim}")
    print(f"   レイヤー数: {num_layers}")
    print(f"   総パラメータ数: {model.num_params:,}")
    
    # データ準備
    print("\n📊 データ準備...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    print(f"   総トークン数: {len(all_tokens):,}")
    
    # シーケンス作成
    sequences = []
    for i in range(0, len(all_tokens) - seq_length - 1, seq_length // 2):
        x = all_tokens[i:i+seq_length]
        y = all_tokens[i+1:i+seq_length+1]
        if len(x) == seq_length:
            sequences.append((x, y))
    
    print(f"   シーケンス数: {len(sequences):,}")
    
    # 学習
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🚀 学習開始 (Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate})")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(sequences)
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            if len(batch) == 0:
                continue
            
            x_batch = torch.tensor([s[0] for s in batch], dtype=torch.long).to(device)
            y_batch = torch.tensor([s[1] for s in batch], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            
            loss = criterion(
                logits.view(-1, tokenizer.actual_vocab_size),
                y_batch.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / max(1, len(sequences) // batch_size)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")
    
    print("\n✅ 学習完了！")
    
    # Generator 作成
    generator = NeuroQGenerator(model, tokenizer, str(device))
    
    # 保存
    if save_path:
        print(f"\n💾 モデル保存: {save_path}")
        model.save_checkpoint(save_path)
        tokenizer.save(save_path.replace('.pt', '_tokenizer.json'))
    
    return model, tokenizer, generator


# ========================================
# テキスト生成
# ========================================

def generate_text(
    generator: NeuroQGenerator,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
) -> str:
    """テキスト生成"""
    print(f"\n📝 テキスト生成:")
    print(f"   Prompt: {prompt}")
    print(f"   Max Tokens: {max_tokens}")
    print(f"   Temperature: {temperature}")
    
    output = generator.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    print(f"\n🤖 生成結果:")
    print(f"   {output}")
    
    return output


# ========================================
# メイン
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description='NeuroQ CLI - QBNN生成AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # Brain Mode で学習して生成
  python neuroq_cli.py --prompt "量子コンピュータについて" --mode brain --train --epochs 30
  
  # Common Crawl データで学習
  python neuroq_cli.py --prompt "AIとは" --train --data-source common_crawl
  
  # 既存モデルで生成のみ
  python neuroq_cli.py --prompt "こんにちは" --model-path neuroq_model.pt
        """
    )
    
    # 生成パラメータ
    parser.add_argument('--prompt', type=str, required=True, help='生成プロンプト')
    parser.add_argument('--max-tokens', type=int, default=100, help='最大生成トークン数')
    parser.add_argument('--temperature', type=float, default=0.7, help='サンプリング温度')
    parser.add_argument('--top-k', type=int, default=40, help='Top-K サンプリング')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-P サンプリング')
    
    # 学習フラグ
    parser.add_argument('--train', action='store_true', help='学習を実行')
    parser.add_argument('--train-before-generate', action='store_true', help='生成前に学習')
    
    # データソース
    parser.add_argument('--data-source', type=str, nargs='+', 
                        choices=['common_crawl', 'huggingface', 'builtin'],
                        default=['builtin'], help='データソース')
    parser.add_argument('--common-crawl-domains', type=str, nargs='+',
                        default=['*.wikipedia.org'], help='Common Crawl ドメイン')
    parser.add_argument('--common-crawl-max-records', type=int, default=100,
                        help='Common Crawl 最大レコード数')
    parser.add_argument('--common-crawl-id', type=str, default='CC-MAIN-2024-10',
                        help='Common Crawl ID')
    
    # モデル設定
    parser.add_argument('--mode', type=str, choices=['brain', 'layered'], 
                        default='brain', help='モデルモード')
    parser.add_argument('--num-neurons', type=int, default=100, help='ニューロン数 (Brain Mode)')
    parser.add_argument('--hidden-dim', type=int, default=256, help='隠れ層次元 (Layered Mode)')
    parser.add_argument('--embed-dim', type=int, default=128, help='埋め込み次元')
    parser.add_argument('--num-layers', type=int, default=3, help='レイヤー数')
    parser.add_argument('--num-heads', type=int, default=4, help='アテンションヘッド数')
    parser.add_argument('--connection-density', type=float, default=0.25, help='接続密度')
    parser.add_argument('--lambda-entangle', type=float, default=0.35, help='もつれ強度')
    
    # 学習設定
    parser.add_argument('--epochs', type=int, default=30, help='エポック数')
    parser.add_argument('--batch-size', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学習率')
    parser.add_argument('--seq-length', type=int, default=64, help='シーケンス長')
    
    # モデルパス
    parser.add_argument('--model-path', type=str, help='既存モデルパス')
    parser.add_argument('--save-path', type=str, help='モデル保存パス')
    
    # JSON入力
    parser.add_argument('--json-input', type=str, help='JSON形式の入力')
    
    args = parser.parse_args()
    
    # JSON入力がある場合は解析
    if args.json_input:
        try:
            json_data = json.loads(args.json_input)
            input_data = json_data.get('input', json_data)
            
            # JSON から引数を更新
            args.prompt = input_data.get('prompt', args.prompt)
            args.train = input_data.get('train_before_generate', args.train)
            args.mode = input_data.get('mode', args.mode)
            args.epochs = input_data.get('epochs', args.epochs)
            args.batch_size = input_data.get('batch_size', args.batch_size)
            args.learning_rate = input_data.get('learning_rate', args.learning_rate)
            args.num_neurons = input_data.get('num_neurons', args.num_neurons)
            args.max_tokens = input_data.get('max_tokens', args.max_tokens)
            args.temperature = input_data.get('temperature', args.temperature)
            
            if 'data_sources' in input_data:
                args.data_source = input_data['data_sources']
            
            if 'common_crawl_config' in input_data:
                cc_config = input_data['common_crawl_config']
                args.common_crawl_domains = cc_config.get('domains', args.common_crawl_domains)
                args.common_crawl_max_records = cc_config.get('max_records', args.common_crawl_max_records)
                args.common_crawl_id = cc_config.get('crawl_id', args.common_crawl_id)
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析エラー: {e}")
            sys.exit(1)
    
    print("=" * 60)
    print("🧠⚛️ NeuroQ CLI")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Prompt: {args.prompt}")
    print(f"Train: {args.train or args.train_before_generate}")
    
    generator = None
    
    # 学習が必要な場合
    if args.train or args.train_before_generate or not args.model_path:
        # データ取得
        texts = []
        
        for source in args.data_source:
            if source == 'common_crawl':
                cc_texts = fetch_common_crawl_data(
                    domains=args.common_crawl_domains,
                    max_records=args.common_crawl_max_records,
                    crawl_id=args.common_crawl_id,
                )
                texts.extend(cc_texts)
            elif source == 'huggingface':
                hf_texts = fetch_huggingface_data(max_samples=500)
                texts.extend(hf_texts)
            else:  # builtin
                texts.extend(get_builtin_training_data())
        
        print(f"\n📚 学習データ: {len(texts)} サンプル")
        
        # 学習
        model, tokenizer, generator = train_model(
            texts=texts,
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_neurons=args.num_neurons,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            seq_length=args.seq_length,
            connection_density=args.connection_density,
            lambda_entangle=args.lambda_entangle,
            save_path=args.save_path,
        )
    
    # 既存モデルをロード
    elif args.model_path:
        print(f"\n📥 モデルロード: {args.model_path}")
        tokenizer_path = args.model_path.replace('.pt', '_tokenizer.json')
        generator = NeuroQGenerator.load(args.model_path, tokenizer_path, "cpu")
    
    # テキスト生成
    if generator:
        output = generate_text(
            generator=generator,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        
        # 結果をJSON形式で出力
        result = {
            "prompt": args.prompt,
            "output": output,
            "model_info": generator.get_model_info(),
        }
        
        print("\n" + "=" * 60)
        print("📤 JSON出力:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n✅ 完了！")


if __name__ == "__main__":
    main()

