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
    
    # まずサンプルデータで確実に学習（フォールバックとして）
    # これにより、Common Crawl取得に失敗しても動作を保証
    training_data = get_sample_training_data()
    
    # Common Crawlからデータ取得を試みる（追加データとして）
    try:
        cc_data = fetch_common_crawl_data(max_records=max_records)
        if cc_data:
            training_data.extend(cc_data)
            print(f"📡 Common Crawlから{len(cc_data)}件追加")
    except Exception as e:
        print(f"⚠️ Common Crawl取得スキップ: {e}")
    
    if training_data:
        print(f"📚 {len(training_data)}件のデータで学習開始 (エポック: {epochs})")
        try:
            # train メソッドを使用
            model.train(training_data, epochs=epochs)
            is_pretrained = True
            print("✅ 事前学習完了")
            return True
        except Exception as e:
            print(f"⚠️ 学習エラー: {e}")
            import traceback
            traceback.print_exc()
            
            # 最小限のサンプルデータで再試行
            print("🔄 最小サンプルデータで再学習を試みます...")
            try:
                minimal_data = [
                    "人工知能は、人間の知能を模倣するコンピュータシステムです。",
                    "量子コンピュータは、量子力学の原理を利用した次世代のコンピュータです。",
                    "自然言語処理は、コンピュータが人間の言語を理解し生成するための技術です。",
                    "こんにちは。私はニューロQです。何かお手伝いできることはありますか？",
                ]
                model.train(minimal_data, epochs=3)
                is_pretrained = True
                print("✅ 最小サンプルデータでの学習完了")
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
        
        # ========================================
        # 毎回モデルの状態を確認し、必要に応じて初期化
        # ========================================
        if not is_pretrained:
            print("⚠️ モデルが未初期化です。初期化を実行します...")
            initialize_models()
        
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
                    try:
                        sample_data = get_sample_training_data()
                        print(f"   サンプルデータ数: {len(sample_data)}")
                        model.train(sample_data, epochs=3)
                        print("✅ サンプルデータでの学習完了")
                        print(f"   model.model is None: {model.model is None}")
                    except Exception as train_error:
                        print(f"⚠️ サンプルデータでの学習失敗: {train_error}")
                        import traceback
                        traceback.print_exc()
                        return {
                            "status": "error",
                            "error": f"モデルの学習に失敗しました: {str(train_error)}"
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
                    # 自動学習も失敗した場合
                    error_msg = str(e)
                    print(f"⚠️ generate ValueError: {error_msg}")
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
            
            if mode == "layered" and LAYERED_AVAILABLE:
                model, _ = get_layered_model(pretrain=False)
                if model is None:
                    return {"status": "error", "error": "モデルの初期化に失敗しました"}
                
                # train メソッドを使用（train_on_texts は存在しない）
                try:
                    model.train(training_data, epochs=epochs)
                    is_pretrained = True
                    return {
                        "status": "success",
                        "mode": "layered",
                        "message": f"{len(training_data)}件のデータで{epochs}エポック学習完了"
                    }
                except Exception as e:
                    return {"status": "error", "error": f"学習エラー: {str(e)}"}
            
            elif mode == "brain" and BRAIN_AVAILABLE:
                model, _ = get_brain_model(pretrain=False)
                if model is None:
                    return {"status": "error", "error": "モデルの初期化に失敗しました"}
                
                # train メソッドを使用（train_on_texts は存在しない）
                try:
                    model.train(training_data, epochs=epochs)
                    is_pretrained = True
                    return {
                        "status": "success",
                        "mode": "brain",
                        "message": f"{len(training_data)}件のデータで{epochs}エポック学習完了"
                    }
                except Exception as e:
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
            if trained and model.model is not None:
                print("✅ Layeredモデルの事前学習が完了しました")
                is_pretrained = True
            else:
                print(f"⚠️ Layeredモデル: trained={trained}, model.model is None={model.model is None if model else 'N/A'}")
        except Exception as e:
            print(f"⚠️ Layeredモデルの初期化エラー: {e}")
            import traceback
            traceback.print_exc()
    
    # Brainモデルを初期化・学習
    if BRAIN_AVAILABLE:
        try:
            model, trained = get_brain_model(pretrain=True)
            if trained:
                print("✅ Brainモデルの事前学習が完了しました")
            else:
                print("⚠️ Brainモデルの事前学習が完了しましたが、確認してください")
        except Exception as e:
            print(f"⚠️ Brainモデルの初期化エラー: {e}")
    
    print(f"🏁 モデル初期化完了 (is_pretrained: {is_pretrained})")


# ========================================
# モジュール読み込み時に初期化を実行
# ========================================
# RunPod Serverlessでは__main__が実行されない場合があるため、
# モジュールレベルで初期化を行う
print("🚀 NeuroQ RunPod Serverless Handler を起動します...")
print(f"   Common Crawl: {'✅' if COMMON_CRAWL_AVAILABLE else '❌'}")
print(f"   Layered: {'✅' if LAYERED_AVAILABLE else '❌'}")
print(f"   Brain: {'✅' if BRAIN_AVAILABLE else '❌'}")

# 起動時に事前学習を実行（重要！）
initialize_models()

# RunPod Serverless 起動
runpod.serverless.start({"handler": handler})
