#!/usr/bin/env python3
"""
NeuroQ RunPod Serverless API Handler
=====================================
Common Crawlから事前学習するRunPod Serverless APIハンドラー
"""

import runpod
import torch
import requests
import re
from typing import Dict, Any, List
from io import BytesIO

# Common Crawl用ライブラリ
try:
    from warcio.archiveiterator import ArchiveIterator
    from bs4 import BeautifulSoup
    COMMON_CRAWL_AVAILABLE = True
except ImportError:
    COMMON_CRAWL_AVAILABLE = False
    print("⚠️ warcio/beautifulsoup4 が見つかりません")

# NeuroQuantumモデルをインポート
try:
    from neuroquantum_layered import NeuroQuantumAI, NeuroQuantumConfig
    LAYERED_AVAILABLE = True
except ImportError:
    LAYERED_AVAILABLE = False
    print("⚠️ neuroquantum_layered.py が見つかりません")

try:
    from neuroquantum_brain import NeuroQuantumBrainAI
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    print("⚠️ neuroquantum_brain.py が見つかりません")

# グローバルモデルインスタンス
layered_ai = None
brain_ai = None
is_pretrained = False


def fetch_common_crawl_data(max_records: int = 100, language: str = "ja") -> List[str]:
    """
    Common Crawlからテキストデータを取得
    
    Args:
        max_records: 取得する最大レコード数
        language: 言語フィルタ ("ja" for Japanese)
    
    Returns:
        テキストのリスト
    """
    if not COMMON_CRAWL_AVAILABLE:
        print("⚠️ Common Crawlライブラリが利用できません")
        return []
    
    texts = []
    
    # Common Crawl インデックスAPI
    # 日本語サイトを検索
    index_url = "https://index.commoncrawl.org/CC-MAIN-2024-10-index"
    
    try:
        # 日本語ドメインを検索
        search_url = f"{index_url}?url=*.jp/*&output=json&limit={max_records}"
        print(f"🔄 Common Crawlからデータを取得中... (最大{max_records}件)")
        
        response = requests.get(search_url, timeout=30)
        if response.status_code != 200:
            print(f"⚠️ Common Crawl API エラー: {response.status_code}")
            # フォールバック: サンプルデータを使用
            return get_sample_training_data()
        
        lines = response.text.strip().split('\n')
        
        for i, line in enumerate(lines[:max_records]):
            try:
                import json
                record = json.loads(line)
                
                # WARCファイルからコンテンツを取得
                warc_url = f"https://data.commoncrawl.org/{record['filename']}"
                offset = int(record['offset'])
                length = int(record['length'])
                
                headers = {'Range': f'bytes={offset}-{offset+length-1}'}
                warc_response = requests.get(warc_url, headers=headers, timeout=30)
                
                if warc_response.status_code in [200, 206]:
                    # WARCレコードをパース
                    stream = BytesIO(warc_response.content)
                    for warc_record in ArchiveIterator(stream):
                        if warc_record.rec_type == 'response':
                            content = warc_record.content_stream().read()
                            # HTMLからテキストを抽出
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # スクリプトとスタイルを削除
                            for script in soup(["script", "style"]):
                                script.decompose()
                            
                            text = soup.get_text(separator=' ', strip=True)
                            
                            # テキストをクリーンアップ
                            text = re.sub(r'\s+', ' ', text)
                            
                            # 日本語テキストのみフィルタ
                            if language == "ja" and re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text):
                                if len(text) > 100:  # 短すぎるテキストは除外
                                    texts.append(text[:2000])  # 最大2000文字
                                    print(f"  ✅ {i+1}/{max_records}: {len(text)}文字取得")
                            elif language != "ja" and len(text) > 100:
                                texts.append(text[:2000])
                                print(f"  ✅ {i+1}/{max_records}: {len(text)}文字取得")
                
            except Exception as e:
                print(f"  ⚠️ レコード {i+1} エラー: {e}")
                continue
        
        print(f"✅ Common Crawlから{len(texts)}件のテキストを取得")
        
    except Exception as e:
        print(f"⚠️ Common Crawl取得エラー: {e}")
        # フォールバック: サンプルデータを使用
        return get_sample_training_data()
    
    if not texts:
        return get_sample_training_data()
    
    return texts


def get_sample_training_data() -> List[str]:
    """サンプル学習データ（フォールバック用）"""
    return [
        "人工知能は、人間の知能を模倣するコンピュータシステムです。機械学習やディープラーニングなどの技術を使用して、データからパターンを学習し、予測や判断を行います。",
        "量子コンピュータは、量子力学の原理を利用した次世代のコンピュータです。従来のコンピュータでは解けない複雑な問題を高速に解くことができます。",
        "自然言語処理は、コンピュータが人間の言語を理解し、生成するための技術です。翻訳、要約、質問応答などのタスクに使用されます。",
        "ニューラルネットワークは、人間の脳の神経細胞の働きを模倣した計算モデルです。層状に接続されたノードで構成され、データから特徴を学習します。",
        "プログラミングは、コンピュータに指示を与えるための言語を使ってソフトウェアを作成する技術です。Python、JavaScript、Javaなど多くの言語があります。",
        "データサイエンスは、大量のデータから有用な情報を抽出し、ビジネスや研究に活用する学問分野です。統計学、機械学習、可視化などの手法を組み合わせます。",
        "クラウドコンピューティングは、インターネット経由でコンピュータリソースを提供するサービスです。AWS、Azure、GCPなどのプラットフォームが代表的です。",
        "ブロックチェーンは、分散型台帳技術の一種で、データの改ざんを防ぐ仕組みを持っています。暗号通貨や契約管理などに応用されています。",
    ]


def pretrain_model(model, max_records: int = 50, epochs: int = 5):
    """
    Common Crawlから事前学習を実行
    """
    global is_pretrained
    
    if is_pretrained:
        print("ℹ️ 既に事前学習済みです")
        return
    
    print("🔄 事前学習を開始...")
    
    # Common Crawlからデータ取得
    training_data = fetch_common_crawl_data(max_records=max_records)
    
    if training_data:
        print(f"📚 {len(training_data)}件のデータで学習開始 (エポック: {epochs})")
        try:
            model.train_on_texts(training_data, epochs=epochs)
            is_pretrained = True
            print("✅ 事前学習完了")
        except Exception as e:
            print(f"⚠️ 学習エラー: {e}")
    else:
        print("⚠️ 学習データが取得できませんでした")


def get_layered_model(pretrain: bool = True):
    """Layeredモデルを取得（事前学習付き）"""
    global layered_ai
    if layered_ai is None and LAYERED_AVAILABLE:
        print("🔄 Layeredモデルを初期化中...")
        layered_ai = NeuroQuantumAI()
        print("✅ Layeredモデル初期化完了")
        
        if pretrain:
            pretrain_model(layered_ai)
    
    return layered_ai


def get_brain_model(pretrain: bool = True):
    """Brainモデルを取得（事前学習付き）"""
    global brain_ai
    if brain_ai is None and BRAIN_AVAILABLE:
        print("🔄 Brainモデルを初期化中...")
        brain_ai = NeuroQuantumBrainAI()
        print("✅ Brainモデル初期化完了")
        
        if pretrain:
            pretrain_model(brain_ai)
    
    return brain_ai


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless ハンドラー
    
    リクエスト例:
    {
        "input": {
            "action": "generate",
            "prompt": "こんにちは",
            "mode": "layered",
            "max_length": 100,
            "temperature": 0.8,
            "pretrain": true
        }
    }
    """
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "generate")
        pretrain = input_data.get("pretrain", True)
        
        # ヘルスチェック
        if action == "health":
            return {
                "status": "healthy",
                "layered_available": LAYERED_AVAILABLE,
                "brain_available": BRAIN_AVAILABLE,
                "common_crawl_available": COMMON_CRAWL_AVAILABLE,
                "cuda_available": torch.cuda.is_available(),
                "is_pretrained": is_pretrained
            }
        
        # テキスト生成
        if action == "generate":
            mode = input_data.get("mode", "layered")
            prompt = input_data.get("prompt", "")
            max_length = input_data.get("max_length", 100)
            temperature = input_data.get("temperature", 0.8)
            top_k = input_data.get("top_k", 50)
            top_p = input_data.get("top_p", 0.9)
            
            if mode == "layered" and LAYERED_AVAILABLE:
                model = get_layered_model(pretrain=pretrain)
                result = model.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temp_min=temperature * 0.8,  # Convert temperature to temp_min/max range
                    temp_max=temperature * 1.2,
                    top_k=top_k,
                    top_p=top_p
                )
                return {
                    "status": "success",
                    "mode": "layered",
                    "prompt": prompt,
                    "generated_text": result,
                    "is_pretrained": is_pretrained
                }
            
            elif mode == "brain" and BRAIN_AVAILABLE:
                model = get_brain_model(pretrain=pretrain)
                result = model.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature_min=temperature * 0.8,  # Convert temperature to temperature_min/max range
                    temperature_max=temperature * 1.2
                )
                return {
                    "status": "success",
                    "mode": "brain",
                    "prompt": prompt,
                    "generated_text": result,
                    "is_pretrained": is_pretrained
                }
            
            else:
                return {
                    "status": "error",
                    "error": f"モード '{mode}' は利用できません"
                }
        
        # 追加学習
        if action == "train":
            mode = input_data.get("mode", "layered")
            training_data = input_data.get("training_data", [])
            epochs = input_data.get("epochs", 10)
            use_common_crawl = input_data.get("use_common_crawl", False)
            max_records = input_data.get("max_records", 50)
            
            # Common Crawlからデータを追加
            if use_common_crawl:
                cc_data = fetch_common_crawl_data(max_records=max_records)
                training_data.extend(cc_data)
            
            if not training_data:
                return {"status": "error", "error": "training_data が必要です"}
            
            if mode == "layered" and LAYERED_AVAILABLE:
                model = get_layered_model(pretrain=False)
                model.train_on_texts(training_data, epochs=epochs)
                return {
                    "status": "success",
                    "mode": "layered",
                    "message": f"{len(training_data)}件のデータで{epochs}エポック学習完了"
                }
            
            elif mode == "brain" and BRAIN_AVAILABLE:
                model = get_brain_model(pretrain=False)
                model.train_on_texts(training_data, epochs=epochs)
                return {
                    "status": "success",
                    "mode": "brain",
                    "message": f"{len(training_data)}件のデータで{epochs}エポック学習完了"
                }
            
            else:
                return {"status": "error", "error": f"モード '{mode}' は利用できません"}
        
        return {"status": "error", "error": f"不明なアクション: {action}"}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


# RunPod Serverless 起動
if __name__ == "__main__":
    print("🚀 NeuroQ RunPod Serverless Handler を起動します...")
    print(f"   Common Crawl: {'✅' if COMMON_CRAWL_AVAILABLE else '❌'}")
    print(f"   Layered: {'✅' if LAYERED_AVAILABLE else '❌'}")
    print(f"   Brain: {'✅' if BRAIN_AVAILABLE else '❌'}")
    runpod.serverless.start({"handler": handler})
