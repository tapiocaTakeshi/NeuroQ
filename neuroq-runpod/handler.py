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
import os
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


def get_extended_training_data() -> List[str]:
    """拡張サンプル学習データ（対話形式を含む）"""
    # 基本的なサンプルデータ
    base_data = get_sample_training_data()
    
    # 対話形式のデータを追加
    dialogues = [
        "<USER>こんにちは<ASSISTANT>こんにちは！私はニューロQです。何かお手伝いできることはありますか？",
        "<USER>あなたは誰ですか<ASSISTANT>私はニューロQという名前の生成AIです。よろしくお願いします。",
        "<USER>量子とは何ですか<ASSISTANT>量子とは、物質やエネルギーの最小単位のことです。量子力学では、粒子は波の性質も持ちます。",
        "<USER>AIとは何ですか<ASSISTANT>AIは人工知能（Artificial Intelligence）の略で、機械に知的な振る舞いをさせる技術です。",
        "<USER>Hello<ASSISTANT>Hello! I'm NeuroQ. How can I help you today?",
        "<USER>What is quantum computing<ASSISTANT>Quantum computing uses quantum mechanics principles to perform calculations much faster than classical computers.",
        "<USER>ありがとう<ASSISTANT>どういたしまして！お役に立てて嬉しいです。",
        "<USER>教えてください<ASSISTANT>はい、何について知りたいですか？具体的に教えてください。",
    ]
    
    # すべてのデータを結合
    all_data = base_data + dialogues
    
    return all_data


def pretrain_model(model, max_records: int = 50, epochs: int = 5):
    """
    Common Crawlから事前学習を実行
    
    Returns:
        bool: 学習が成功したかどうか
    """
    global is_pretrained
    
    print(f"🔍 pretrain_model開始: is_pretrained={is_pretrained}, model.model is None={model.model is None if model else 'N/A'}")
    
    # 既にモデルが学習済みかどうかを確認
    if is_pretrained and model.model is not None:
        print("ℹ️ 既に事前学習済みです")
        return True
    
    print("🔄 事前学習を開始...")
    print(f"   カレントディレクトリ: {os.getcwd()}")
    print(f"   トークナイザー存在: {os.path.exists('neuroq_tokenizer.model')}")
    
    # 拡張サンプルデータを取得（対話形式を含む）
    training_data = get_extended_training_data()
    
    # Common Crawlからデータ取得を試みる（追加データとして）
    try:
        cc_data = fetch_common_crawl_data(max_records=max_records)
        if cc_data:
            training_data.extend(cc_data)
            print(f"📡 Common Crawlから{len(cc_data)}件追加")
    except Exception as e:
        print(f"⚠️ Common Crawl取得スキップ: {e}")
    
    if training_data:
        # データを結合して長いテキストを作成（シーケンス作成のため）
        # 十分な長さを確保するために繰り返し回数を増やす
        long_text = " ".join(training_data) * 5
        combined_data = [long_text]
        print(f"📚 結合後テキスト長: {len(long_text)} で学習開始 (エポック: {epochs})")
        
        # 複数回の試行で学習を実行
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🔄 学習試行 {attempt + 1}/{max_retries}...")
                # train メソッドを使用（seq_lenを短く設定）
                model.train(combined_data, epochs=epochs, seq_len=16)
                
                # 学習後の確認
                if model.model is None:
                    print(f"⚠️ 学習試行 {attempt + 1} 後もmodel.modelがNoneです")
                    if attempt < max_retries - 1:
                        print("   再試行します...")
                        continue
                    else:
                        print("   すべての試行が失敗しました")
                        raise Exception("model.model is None after all training attempts")
                
                is_pretrained = True
                print(f"✅ 事前学習完了 (試行 {attempt + 1}, model.model is None: {model.model is None})")
                return True
            except Exception as e:
                print(f"⚠️ 学習試行 {attempt + 1} エラー: {e}")
                import traceback
                traceback.print_exc()
                
                if attempt < max_retries - 1:
                    print("   再試行します...")
                    continue
                else:
                    print("   すべての試行が失敗しました。最小サンプルデータで再試行します...")
            
            # 最小限のサンプルデータで再試行
            try:
                minimal_text = """
                人工知能は、人間の知能を模倣するコンピュータシステムです。
                量子コンピュータは、量子力学の原理を利用した次世代のコンピュータです。
                自然言語処理は、コンピュータが人間の言語を理解し生成するための技術です。
                こんにちは。私はニューロQです。何かお手伝いできることはありますか？
                ディープラーニングは、多層のニューラルネットワークを使用する機械学習の手法です。
                <USER>こんにちは<ASSISTANT>こんにちは！私はニューロQです。
                <USER>量子とは<ASSISTANT>量子は物質やエネルギーの最小単位です。
                <USER>Hello<ASSISTANT>Hello! I'm NeuroQ. How can I help you?
                """ * 20
                print("🔄 最小サンプルデータで再学習を試みます...")
                model.train([minimal_text], epochs=5, seq_len=16)
                
                # 学習後の確認
                if model.model is None:
                    print("⚠️ 最小データでの学習後もmodel.modelがNoneです")
                    return False
                
                is_pretrained = True
                print(f"✅ 最小サンプルデータでの学習完了 (model.model is None: {model.model is None})")
                return True
            except Exception as e2:
                print(f"⚠️ 最小サンプルデータでの学習も失敗: {e2}")
                import traceback
                traceback.print_exc()
                return False
    else:
        print("⚠️ 学習データが取得できませんでした")
        return False


def get_layered_model(pretrain: bool = True):
    """
    Layeredモデルを取得（事前学習付き）
    
    Returns:
        tuple: (model, is_trained) - モデルと学習済みかどうか
    """
    global layered_ai
    trained = False
    
    if layered_ai is None and LAYERED_AVAILABLE:
        print("🔄 Layeredモデルを初期化中...")
        layered_ai = NeuroQuantumAI()
        print("✅ Layeredモデル初期化完了")
        
        if pretrain:
            trained = pretrain_model(layered_ai)
    elif layered_ai is not None:
        # 既存のモデルがある場合、学習済みかどうかを確認
        trained = layered_ai.model is not None
        if not trained and pretrain:
            trained = pretrain_model(layered_ai)
    
    return layered_ai, trained


def get_brain_model(pretrain: bool = True):
    """
    Brainモデルを取得（事前学習付き）
    
    Returns:
        tuple: (model, is_trained) - モデルと学習済みかどうか
    """
    global brain_ai
    trained = False
    
    if brain_ai is None and BRAIN_AVAILABLE:
        print("🔄 Brainモデルを初期化中...")
        brain_ai = NeuroQuantumBrainAI()
        print("✅ Brainモデル初期化完了")
        
        if pretrain:
            trained = pretrain_model(brain_ai)
    elif brain_ai is not None:
        # 既存のモデルがある場合、学習済みかどうかを確認
        trained = brain_ai.model is not None
        if not trained and pretrain:
            trained = pretrain_model(brain_ai)
    
    return brain_ai, trained


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
            "temp_min": 0.4,
            "temp_max": 0.8,
            "pretrain": true
        }
    }
    """
    global is_pretrained, layered_ai, brain_ai
    
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "generate")
        pretrain = input_data.get("pretrain", True)
        
        # デバッグ情報を出力
        print(f"📥 リクエスト受信: action={action}, pretrain={pretrain}")
        print(f"   is_pretrained={is_pretrained}")
        print(f"   layered_ai is None={layered_ai is None}")
        print(f"   layered_ai.model is None={layered_ai.model is None if layered_ai else 'N/A'}")
        
        # ========================================
        # 毎回モデルの状態を確認し、必要に応じて初期化
        # ========================================
        if not is_pretrained or (layered_ai is not None and layered_ai.model is None) or (brain_ai is not None and brain_ai.model is None):
            print("⚠️ モデルが未初期化または未学習です。初期化を実行します...")
            print(f"   is_pretrained={is_pretrained}")
            print(f"   layered_ai is None={layered_ai is None}")
            print(f"   layered_ai.model is None={layered_ai.model is None if layered_ai else 'N/A'}")
            print(f"   brain_ai is None={brain_ai is None}")
            print(f"   brain_ai.model is None={brain_ai.model is None if brain_ai else 'N/A'}")
            initialize_models()
            
            # 初期化後の確認
            if layered_ai is not None and layered_ai.model is None:
                print("⚠️ 初期化後もlayered_ai.modelがNoneです。再度学習を試みます...")
                retrained = pretrain_model(layered_ai, max_records=30, epochs=5)
                if retrained and layered_ai.model is not None:
                    is_pretrained = True
                    print("✅ layered_aiの再学習が完了しました")
        
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
            
            print(f"🎯 生成開始: mode={mode}, prompt='{prompt[:50]}...'")
            
            # ========================================
            # 生成前の事前学習確認（強制）
            # ========================================
            print("🔍 生成前のモデル状態を確認中...")
            
            if mode == "layered":
                if layered_ai is None or layered_ai.model is None:
                    print("⚠️ Layeredモデルが未学習です。事前学習を実行します...")
                    if layered_ai is None:
                        layered_ai = NeuroQuantumAI()
                    success = pretrain_model(layered_ai, max_records=30, epochs=5)
                    if not success or layered_ai.model is None:
                        return {
                            "status": "error",
                            "error": "事前学習に失敗しました。モデルを初期化できませんでした。"
                        }
                    is_pretrained = True
                    print("✅ 生成前の事前学習が完了しました")
                else:
                    print("✅ Layeredモデルは既に学習済みです")
                    
            elif mode == "brain":
                if brain_ai is None or brain_ai.model is None:
                    print("⚠️ Brainモデルが未学習です。事前学習を実行します...")
                    if brain_ai is None:
                        brain_ai = NeuroQuantumBrainAI()
                    success = pretrain_model(brain_ai, max_records=30, epochs=5)
                    if not success or brain_ai.model is None:
                        return {
                            "status": "error",
                            "error": "事前学習に失敗しました。モデルを初期化できませんでした。"
                        }
                    is_pretrained = True
                    print("✅ 生成前の事前学習が完了しました")
                else:
                    print("✅ Brainモデルは既に学習済みです")
            max_length = input_data.get("max_length", 100)
            
            # 温度パラメータの処理（後方互換性対応）
            # temperature が指定された場合、temp_min/temp_max に変換
            temperature = input_data.get("temperature", None)
            if temperature is not None:
                # temperature を temp_min/temp_max の範囲に変換
                temp_min = temperature * 0.8
                temp_max = temperature * 1.2
            else:
                temp_min = input_data.get("temp_min", 0.4)
                temp_max = input_data.get("temp_max", 0.8)
            
            top_k = input_data.get("top_k", 50)
            top_p = input_data.get("top_p", 0.9)
            
            if mode == "layered" and LAYERED_AVAILABLE:
                model, trained = get_layered_model(pretrain=pretrain)
                
                # モデルが学習済みかどうかを確認
                if model is None:
                    return {
                        "status": "error",
                        "error": "モデルの初期化に失敗しました"
                    }
                
                # 事前学習が失敗した場合、ここで再度学習を試みる
                if model.model is None:
                    print("⚠️ モデルが未学習です。サンプルデータで学習を試みます...")
                    print(f"   カレントディレクトリ: {os.getcwd()}")
                    print(f"   トークナイザーファイル存在: {os.path.exists('neuroq_tokenizer.model')}")
                    print(f"   /app内のファイル: {os.listdir('/app') if os.path.exists('/app') else 'N/A'}")
                    
                    # 複数回の試行で学習を実行
                    max_retries = 3
                    training_success = False
                    
                    for attempt in range(max_retries):
                        try:
                            print(f"🔄 学習試行 {attempt + 1}/{max_retries}...")
                            # 拡張サンプルデータを使用（対話形式を含む）
                            sample_data = get_extended_training_data()
                            # 短いテキストを結合して長くする（十分な長さを確保）
                            long_text = " ".join(sample_data) * 10
                            combined_data = [long_text]
                            print(f"   結合後のテキスト長: {len(long_text)}")
                            # seq_lenを短く設定（16）、エポック数を増やす
                            model.train(combined_data, epochs=5, seq_len=16)
                            print("✅ サンプルデータでの学習完了")
                            print(f"   model.model is None: {model.model is None}")
                            
                            # 学習後の確認
                            if model.model is None:
                                if attempt < max_retries - 1:
                                    print(f"   学習試行 {attempt + 1} 後もmodel.modelがNoneです。再試行します...")
                                    continue
                                else:
                                    raise Exception("すべての学習試行後もmodel.modelがNoneです")
                            
                            training_success = True
                            global is_pretrained
                            is_pretrained = True
                            break
                            
                        except Exception as train_error:
                            print(f"⚠️ 学習試行 {attempt + 1} 失敗: {train_error}")
                            import traceback
                            traceback.print_exc()
                            
                            if attempt < max_retries - 1:
                                print("   再試行します...")
                                continue
                            else:
                                print("   すべての学習試行が失敗しました")
                    
                    if not training_success or model.model is None:
                        return {
                            "status": "error",
                            "error": "モデルの学習に失敗しました。すべての試行が失敗しました。"
                        }
                
                try:
                    # 新しいパラメータ形式 (temp_min/temp_max)
                    result = model.generate(
                        prompt=prompt,
                        max_length=max_length,
                        temp_min=temp_min,
                        temp_max=temp_max,
                        top_k=top_k,
                        top_p=top_p
                    )
                except TypeError as e:
                    # 後方互換性: 古いバージョンでは temperature を使用
                    if "temp_min" in str(e) or "temp_max" in str(e):
                        avg_temp = (temp_min + temp_max) / 2
                        result = model.generate(
                            prompt=prompt,
                            max_length=max_length,
                            temperature=avg_temp,
                            top_k=top_k,
                            top_p=top_p
                        )
                    else:
                        raise e
                except ValueError as e:
                    # 自動学習も失敗した場合、再度学習を試みる
                    error_msg = str(e)
                    print(f"⚠️ generate ValueError: {error_msg}")
                    
                    # モデルが未学習の場合、再度学習を試みる
                    if "モデルが学習されていません" in error_msg or "model.model is None" in error_msg or model.model is None:
                        print("🔄 モデルが未学習のため、再度学習を試みます...")
                        try:
                            # 拡張サンプルデータを使用（対話形式を含む）
                            sample_data = get_extended_training_data()
                            long_text = " ".join(sample_data) * 10
                            combined_data = [long_text]
                            model.train(combined_data, epochs=5, seq_len=16)
                            
                            if model.model is None:
                                return {
                                    "status": "error",
                                    "error": "モデルの学習に失敗しました。再学習後もmodel.modelがNoneです。"
                                }
                            
                            global is_pretrained
                            is_pretrained = True
                            
                            # 再学習後、再度生成を試みる
                            print("✅ 再学習完了。再度生成を試みます...")
                            result = model.generate(
                                prompt=prompt,
                                max_length=max_length,
                                temp_min=temp_min,
                                temp_max=temp_max,
                                top_k=top_k,
                                top_p=top_p
                            )
                        except Exception as retry_error:
                            print(f"⚠️ 再学習も失敗: {retry_error}")
                            import traceback
                            traceback.print_exc()
                            return {
                                "status": "error",
                                "error": f"モデルの再学習に失敗しました: {str(retry_error)}"
                            }
                    else:
                        return {
                            "status": "error",
                            "error": f"テキスト生成エラー: {error_msg}"
                        }
                except Exception as e:
                    # その他のエラー
                    error_msg = str(e)
                    print(f"⚠️ generate Exception: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    return {
                        "status": "error",
                        "error": f"生成時にエラーが発生しました: {error_msg}"
                    }
                
                # 生成結果がエラーメッセージかどうかを確認
                if result == "モデルが学習されていません" or not result:
                    return {
                        "status": "error",
                        "error": "モデルが学習されていません。学習を実行してください。"
                    }
                
                return {
                    "status": "success",
                    "mode": "layered",
                    "prompt": prompt,
                    "generated_text": result,
                    "is_pretrained": is_pretrained
                }
            
            elif mode == "brain" and BRAIN_AVAILABLE:
                model, trained = get_brain_model(pretrain=pretrain)
                
                # モデルが学習済みかどうかを確認
                if model is None:
                    return {
                        "status": "error",
                        "error": "モデルの初期化に失敗しました"
                    }
                
                # 事前学習が失敗した場合、ここで再度学習を試みる
                if model.model is None:
                    print("⚠️ Brainモデルが未学習です。サンプルデータで学習を試みます...")
                    try:
                        sample_data = get_sample_training_data()
                        model.train(sample_data, epochs=3)
                        print("✅ Brainモデルのサンプルデータでの学習完了")
                    except Exception as train_error:
                        print(f"⚠️ Brainモデルのサンプルデータでの学習失敗: {train_error}")
                        return {
                            "status": "error",
                            "error": f"Brainモデルの学習に失敗しました: {str(train_error)}"
                        }
                
                try:
                    result = model.generate(
                        prompt=prompt,
                        max_length=max_length,
                        temperature_min=temp_min,
                        temperature_max=temp_max
                    )
                except ValueError as e:
                    # 自動学習も失敗した場合
                    error_msg = str(e)
                    print(f"⚠️ Brain generate ValueError: {error_msg}")
                    return {
                        "status": "error",
                        "error": f"Brainモデルのテキスト生成エラー: {error_msg}"
                    }
                except Exception as e:
                    # その他のエラー
                    error_msg = str(e)
                    print(f"⚠️ Brain generate Exception: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    return {
                        "status": "error",
                        "error": f"Brain生成時にエラーが発生しました: {error_msg}"
                    }
                
                # 生成結果がエラーメッセージかどうかを確認
                if result == "モデルが学習されていません" or not result:
                    return {
                        "status": "error",
                        "error": "モデルが学習されていません。学習を実行してください。"
                    }
                
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
            
            # training_dataが空の場合はサンプルデータを使用
            if not training_data:
                print("⚠️ training_data が空です。サンプルデータを使用します。")
                training_data = get_sample_training_data()
            
            # データを結合して長いテキストを作成
            long_text = " ".join(training_data) * 3
            combined_data = [long_text]
            print(f"📚 結合後テキスト長: {len(long_text)}")
            
            if mode == "layered" and LAYERED_AVAILABLE:
                model, _ = get_layered_model(pretrain=False)
                if model is None:
                    return {"status": "error", "error": "モデルの初期化に失敗しました"}
                
                try:
                    model.train(combined_data, epochs=epochs, seq_len=16)
                    is_pretrained = True
                    return {
                        "status": "success",
                        "mode": "layered",
                        "message": f"{len(training_data)}件のデータで{epochs}エポック学習完了"
                    }
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return {"status": "error", "error": f"学習エラー: {str(e)}"}
            
            elif mode == "brain" and BRAIN_AVAILABLE:
                model, _ = get_brain_model(pretrain=False)
                if model is None:
                    return {"status": "error", "error": "モデルの初期化に失敗しました"}
                
                try:
                    model.train(combined_data, epochs=epochs, seq_len=16)
                    is_pretrained = True
                    return {
                        "status": "success",
                        "mode": "brain",
                        "message": f"{len(training_data)}件のデータで{epochs}エポック学習完了"
                    }
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return {"status": "error", "error": f"学習エラー: {str(e)}"}
            
            else:
                return {"status": "error", "error": f"モード '{mode}' は利用できません"}
        
        return {"status": "error", "error": f"不明なアクション: {action}"}
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"⚠️ Handler Exception: {error_msg}")
        traceback.print_exc()
        return {"status": "error", "error": f"ハンドラーエラー: {error_msg}"}


# 起動時に事前学習を実行
def initialize_models():
    """
    起動時にモデルを初期化し、事前学習を実行
    これにより、最初のリクエストでの遅延を回避
    """
    global is_pretrained
    
    print("🔄 モデルの初期化と事前学習を開始...")
    print(f"   カレントディレクトリ: {os.getcwd()}")
    print(f"   /app内のファイル: {os.listdir('/app') if os.path.exists('/app') else 'N/A'}")
    print(f"   トークナイザー存在: {os.path.exists('neuroq_tokenizer.model')}")
    print(f"   /app/トークナイザー存在: {os.path.exists('/app/neuroq_tokenizer.model')}")
    
    # Layeredモデルを初期化・学習
    if LAYERED_AVAILABLE:
        try:
            model, trained = get_layered_model(pretrain=True)
            if trained and model is not None and model.model is not None:
                print("✅ Layeredモデルの事前学習が完了しました")
                is_pretrained = True
            else:
                print(f"⚠️ Layeredモデル: trained={trained}, model is None={model is None}, model.model is None={model.model is None if model else 'N/A'}")
                # 再度学習を試みる
                if model is not None and model.model is None:
                    print("🔄 Layeredモデルの再学習を試みます...")
                    retrained = pretrain_model(model, max_records=30, epochs=5)
                    if retrained and model.model is not None:
                        print("✅ Layeredモデルの再学習が完了しました")
                        is_pretrained = True
        except Exception as e:
            print(f"⚠️ Layeredモデルの初期化エラー: {e}")
            import traceback
            traceback.print_exc()
    
    # Brainモデルを初期化・学習
    if BRAIN_AVAILABLE:
        try:
            model, trained = get_brain_model(pretrain=True)
            if trained and model is not None and model.model is not None:
                print("✅ Brainモデルの事前学習が完了しました")
            else:
                print(f"⚠️ Brainモデル: trained={trained}, model is None={model is None}, model.model is None={model.model is None if model else 'N/A'}")
                # 再度学習を試みる
                if model is not None and model.model is None:
                    print("🔄 Brainモデルの再学習を試みます...")
                    retrained = pretrain_model(model, max_records=30, epochs=5)
                    if retrained and model.model is not None:
                        print("✅ Brainモデルの再学習が完了しました")
        except Exception as e:
            print(f"⚠️ Brainモデルの初期化エラー: {e}")
    
    print(f"🏁 モデル初期化完了 (is_pretrained: {is_pretrained})")


# ========================================
# モジュール読み込み時に初期化を実行
# ========================================
# RunPod Serverlessでは__main__が実行されない場合があるため、
# モジュールレベルで初期化を行う

# カレントディレクトリを/appに変更（トークナイザーファイルが見つかるように）
if os.path.exists('/app'):
    os.chdir('/app')
    print(f"📁 カレントディレクトリを /app に変更しました")

print("🚀 NeuroQ RunPod Serverless Handler を起動します...")
print(f"   Common Crawl: {'✅' if COMMON_CRAWL_AVAILABLE else '❌'}")
print(f"   Layered: {'✅' if LAYERED_AVAILABLE else '❌'}")
print(f"   Brain: {'✅' if BRAIN_AVAILABLE else '❌'}")

# 起動時に事前学習を実行（重要！）
initialize_models()

# RunPod Serverless 起動
runpod.serverless.start({"handler": handler})
