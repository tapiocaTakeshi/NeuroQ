#!/usr/bin/env python3
"""
NeuroQ RunPod Serverless Handler
================================
RunPod Serverless Endpoint用のハンドラー

サポートモード:
- Brain Mode: 脳型散在QBNN
- Layered Mode: 層状QBNN-Transformer

使用方法:
1. このファイルと neuroq_model.py をRunPodにデプロイ
2. モデルファイル (neuroq_model.pt, neuroq_tokenizer.json) を配置
3. RunPod Endpoint から呼び出し

API パラメータ:
- 生成パラメータ: prompt, max_tokens, temperature, top_k, top_p
- モデル設定: mode, num_neurons, hidden_dim, connection_density, etc.
"""

import runpod
import torch
import os
import traceback
import json
import requests
import gzip
import io
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse

from neuroq_model import (
    NeuroQGenerator, 
    NeuroQModel, 
    NeuroQTokenizer, 
    NeuroQConfig,
    create_neuroq_brain,
    create_neuroq_layered,
)

# ========================================
# グローバル設定
# ========================================

MODEL_PATH = os.environ.get("NEUROQ_MODEL_PATH", "neuroq_model.pt")
TOKENIZER_PATH = os.environ.get("NEUROQ_TOKENIZER_PATH", "neuroq_tokenizer.json")
DEFAULT_MODE = os.environ.get("NEUROQ_MODE", "layered")  # 'brain' or 'layered'

# デフォルトのモデル設定
DEFAULT_CONFIG = {
    # 共通
    "embed_dim": int(os.environ.get("NEUROQ_EMBED_DIM", "128")),
    "num_layers": int(os.environ.get("NEUROQ_NUM_LAYERS", "3")),
    "dropout": float(os.environ.get("NEUROQ_DROPOUT", "0.1")),
    "max_seq_len": int(os.environ.get("NEUROQ_MAX_SEQ_LEN", "256")),
    
    # Brain Mode
    "num_neurons": int(os.environ.get("NEUROQ_NUM_NEURONS", "100")),
    "connection_density": float(os.environ.get("NEUROQ_CONNECTION_DENSITY", "0.25")),
    "lambda_entangle_brain": float(os.environ.get("NEUROQ_LAMBDA_BRAIN", "0.35")),
    
    # Layered Mode
    "hidden_dim": int(os.environ.get("NEUROQ_HIDDEN_DIM", "256")),
    "num_heads": int(os.environ.get("NEUROQ_NUM_HEADS", "4")),
    "lambda_entangle_layered": float(os.environ.get("NEUROQ_LAMBDA_LAYERED", "0.5")),
}

# デフォルトの学習設定
DEFAULT_TRAINING_CONFIG = {
    # エポック・ステップ関連
    "epochs": int(os.environ.get("NEUROQ_EPOCHS", "10")),
    "batch_size": int(os.environ.get("NEUROQ_BATCH_SIZE", "32")),
    "gradient_accumulation_steps": int(os.environ.get("NEUROQ_GRAD_ACCUM_STEPS", "1")),
    
    # 学習率関連
    "learning_rate": float(os.environ.get("NEUROQ_LEARNING_RATE", "1e-4")),
    "min_learning_rate": float(os.environ.get("NEUROQ_MIN_LR", "1e-6")),
    "weight_decay": float(os.environ.get("NEUROQ_WEIGHT_DECAY", "0.01")),
    "warmup_steps": int(os.environ.get("NEUROQ_WARMUP_STEPS", "100")),
    "lr_scheduler": os.environ.get("NEUROQ_LR_SCHEDULER", "cosine"),  # cosine, linear, constant
    
    # Temperature関連（生成時のサンプリング温度スケジューリング）
    "max_temperature": float(os.environ.get("NEUROQ_MAX_TEMPERATURE", "1.5")),
    "min_temperature": float(os.environ.get("NEUROQ_MIN_TEMPERATURE", "0.1")),
    "temperature_decay": os.environ.get("NEUROQ_TEMP_DECAY", "linear"),  # linear, exponential, cosine
    
    # 正則化・最適化
    "max_grad_norm": float(os.environ.get("NEUROQ_MAX_GRAD_NORM", "1.0")),
    "label_smoothing": float(os.environ.get("NEUROQ_LABEL_SMOOTHING", "0.1")),
    
    # 評価・保存
    "eval_steps": int(os.environ.get("NEUROQ_EVAL_STEPS", "500")),
    "save_steps": int(os.environ.get("NEUROQ_SAVE_STEPS", "1000")),
    "logging_steps": int(os.environ.get("NEUROQ_LOGGING_STEPS", "100")),
    
    # 早期停止
    "early_stopping_patience": int(os.environ.get("NEUROQ_EARLY_STOPPING", "3")),
    "early_stopping_threshold": float(os.environ.get("NEUROQ_EARLY_STOPPING_THRESHOLD", "0.001")),
    
    # データ関連
    "train_split": float(os.environ.get("NEUROQ_TRAIN_SPLIT", "0.9")),
    "shuffle": True,
    "seed": int(os.environ.get("NEUROQ_SEED", "42")),
}

# デフォルトのデータソース設定
DEFAULT_DATA_SOURCE_CONFIG = {
    # Common Crawl設定
    "common_crawl": {
        "enabled": True,
        "index_url": "https://index.commoncrawl.org",
        "data_url": "https://data.commoncrawl.org",
        "crawl_id": os.environ.get("COMMONCRAWL_ID", "CC-MAIN-2024-10"),  # 最新のクロールID
        "max_records": int(os.environ.get("COMMONCRAWL_MAX_RECORDS", "1000")),
        "languages": ["ja", "en"],  # 日本語と英語
        "min_content_length": 100,
        "max_content_length": 10000,
    },
    # PubMed設定
    "pubmed": {
        "enabled": True,
        "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        "api_key": os.environ.get("PUBMED_API_KEY", ""),  # オプション: レート制限緩和用
        "max_records": int(os.environ.get("PUBMED_MAX_RECORDS", "1000")),
        "search_terms": ["quantum computing", "neural network", "machine learning", "artificial intelligence"],
        "retmax": 100,  # 1回のリクエストで取得する最大件数
        "include_abstract": True,
        "include_mesh_terms": True,
    },
}


# ========================================
# データローダークラス
# ========================================

class CommonCrawlLoader:
    """
    Common Crawlからテキストデータを取得するローダー
    https://commoncrawl.org/
    """
    
    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_DATA_SOURCE_CONFIG["common_crawl"]
        self.index_url = self.config["index_url"]
        self.data_url = self.config["data_url"]
        self.crawl_id = self.config["crawl_id"]
        
    def search_index(self, query: str, limit: int = 100) -> List[dict]:
        """
        Common Crawl Indexを検索
        """
        try:
            url = f"{self.index_url}/{self.crawl_id}-index"
            params = {
                "url": query,
                "output": "json",
                "limit": limit
            }
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                results = []
                for line in response.text.strip().split('\n'):
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                return results
            return []
        except Exception as e:
            print(f"⚠️ Common Crawl Index検索エラー: {e}")
            return []
    
    def fetch_warc_record(self, record: dict) -> Optional[str]:
        """
        WARCレコードからコンテンツを取得
        """
        try:
            filename = record.get("filename")
            offset = int(record.get("offset", 0))
            length = int(record.get("length", 0))
            
            if not filename or length == 0:
                return None
            
            url = f"{self.data_url}/{filename}"
            headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
            
            response = requests.get(url, headers=headers, timeout=60)
            
            if response.status_code in [200, 206]:
                # gzip解凍
                try:
                    decompressed = gzip.decompress(response.content)
                    content = decompressed.decode('utf-8', errors='ignore')
                    
                    # HTMLからテキストを抽出（簡易版）
                    text = self._extract_text_from_html(content)
                    return text
                except:
                    return None
            return None
        except Exception as e:
            print(f"⚠️ WARCレコード取得エラー: {e}")
            return None
    
    def _extract_text_from_html(self, html: str) -> str:
        """
        HTMLからテキストを抽出（簡易版）
        """
        # WARCヘッダーをスキップ
        if "\r\n\r\n" in html:
            parts = html.split("\r\n\r\n")
            if len(parts) > 2:
                html = "\r\n\r\n".join(parts[2:])
        
        # スクリプトとスタイルを除去
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # HTMLタグを除去
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # 特殊文字をデコード
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        
        # 余分な空白を整理
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_data(self, domains: List[str] = None, max_records: int = None) -> List[str]:
        """
        Common Crawlからデータをロード
        
        Args:
            domains: 検索するドメインのリスト（例: ["*.wikipedia.org", "*.news.yahoo.co.jp"]）
            max_records: 取得する最大レコード数
        """
        if max_records is None:
            max_records = self.config["max_records"]
        
        if domains is None:
            domains = ["*.wikipedia.org", "*.news.yahoo.co.jp", "*.nhk.or.jp"]
        
        print(f"📥 Common Crawlからデータをロード中...")
        print(f"   Crawl ID: {self.crawl_id}")
        print(f"   対象ドメイン: {domains}")
        print(f"   最大レコード数: {max_records}")
        
        texts = []
        records_per_domain = max_records // len(domains)
        
        for domain in domains:
            print(f"   🔍 検索中: {domain}")
            records = self.search_index(domain, limit=records_per_domain)
            
            for record in records[:records_per_domain]:
                text = self.fetch_warc_record(record)
                if text:
                    # 長さフィルター
                    if self.config["min_content_length"] <= len(text) <= self.config["max_content_length"]:
                        texts.append(text)
                
                if len(texts) >= max_records:
                    break
            
            if len(texts) >= max_records:
                break
            
            time.sleep(0.5)  # レート制限対策
        
        print(f"✅ Common Crawl: {len(texts)}件のテキストを取得")
        return texts


class PubMedLoader:
    """
    PubMedから論文データを取得するローダー
    https://pubmed.ncbi.nlm.nih.gov/
    """
    
    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_DATA_SOURCE_CONFIG["pubmed"]
        self.base_url = self.config["base_url"]
        self.api_key = self.config.get("api_key", "")
        
    def search(self, term: str, retmax: int = 100) -> List[str]:
        """
        PubMedを検索してPMIDのリストを取得
        """
        try:
            url = f"{self.base_url}/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": term,
                "retmax": retmax,
                "retmode": "json",
                "sort": "relevance"
            }
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("esearchresult", {}).get("idlist", [])
            return []
        except Exception as e:
            print(f"⚠️ PubMed検索エラー: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[dict]:
        """
        PMIDリストから論文情報を取得
        """
        if not pmids:
            return []
        
        try:
            url = f"{self.base_url}/efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "xml"
            }
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                return self._parse_pubmed_xml(response.text)
            return []
        except Exception as e:
            print(f"⚠️ PubMed論文取得エラー: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[dict]:
        """
        PubMed XMLをパース
        """
        articles = []
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    # タイトル
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    
                    # アブストラクト
                    abstract_parts = []
                    for abstract_text in article.findall(".//AbstractText"):
                        if abstract_text.text:
                            label = abstract_text.get("Label", "")
                            if label:
                                abstract_parts.append(f"{label}: {abstract_text.text}")
                            else:
                                abstract_parts.append(abstract_text.text)
                    abstract = " ".join(abstract_parts)
                    
                    # MeSH Terms
                    mesh_terms = []
                    if self.config.get("include_mesh_terms", True):
                        for mesh in article.findall(".//MeshHeading/DescriptorName"):
                            if mesh.text:
                                mesh_terms.append(mesh.text)
                    
                    # PMID
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    # 著者
                    authors = []
                    for author in article.findall(".//Author"):
                        lastname = author.find("LastName")
                        forename = author.find("ForeName")
                        if lastname is not None and lastname.text:
                            name = lastname.text
                            if forename is not None and forename.text:
                                name = f"{forename.text} {lastname.text}"
                            authors.append(name)
                    
                    # 出版年
                    year_elem = article.find(".//PubDate/Year")
                    year = year_elem.text if year_elem is not None else ""
                    
                    if title or abstract:
                        articles.append({
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract,
                            "authors": authors,
                            "year": year,
                            "mesh_terms": mesh_terms
                        })
                except Exception as e:
                    continue
                    
        except ET.ParseError as e:
            print(f"⚠️ XML解析エラー: {e}")
        
        return articles
    
    def load_data(self, search_terms: List[str] = None, max_records: int = None) -> List[str]:
        """
        PubMedからデータをロード
        
        Args:
            search_terms: 検索キーワードのリスト
            max_records: 取得する最大レコード数
        """
        if max_records is None:
            max_records = self.config["max_records"]
        
        if search_terms is None:
            search_terms = self.config["search_terms"]
        
        print(f"📥 PubMedからデータをロード中...")
        print(f"   検索キーワード: {search_terms}")
        print(f"   最大レコード数: {max_records}")
        
        texts = []
        records_per_term = max_records // len(search_terms)
        retmax = min(self.config["retmax"], records_per_term)
        
        for term in search_terms:
            print(f"   🔍 検索中: {term}")
            pmids = self.search(term, retmax=retmax)
            
            if pmids:
                articles = self.fetch_abstracts(pmids)
                
                for article in articles:
                    # タイトルとアブストラクトを結合してテキスト化
                    text_parts = []
                    
                    if article["title"]:
                        text_parts.append(f"Title: {article['title']}")
                    
                    if article["abstract"] and self.config.get("include_abstract", True):
                        text_parts.append(f"Abstract: {article['abstract']}")
                    
                    if article["mesh_terms"] and self.config.get("include_mesh_terms", True):
                        text_parts.append(f"Keywords: {', '.join(article['mesh_terms'][:10])}")
                    
                    if text_parts:
                        texts.append("\n".join(text_parts))
                
                if len(texts) >= max_records:
                    break
            
            time.sleep(0.34)  # NCBI API制限: 1秒あたり3リクエスト
        
        print(f"✅ PubMed: {len(texts)}件のテキストを取得")
        return texts[:max_records]


class DataSourceManager:
    """
    複数のデータソースを管理するマネージャー
    """
    
    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_DATA_SOURCE_CONFIG
        self.loaders = {}
        
        # ローダーを初期化
        if self.config.get("common_crawl", {}).get("enabled", True):
            self.loaders["common_crawl"] = CommonCrawlLoader(self.config.get("common_crawl"))
        
        if self.config.get("pubmed", {}).get("enabled", True):
            self.loaders["pubmed"] = PubMedLoader(self.config.get("pubmed"))
    
    def load_all(self, sources: List[str] = None, **kwargs) -> Dict[str, List[str]]:
        """
        指定されたすべてのデータソースからデータをロード
        
        Args:
            sources: ロードするソースのリスト（None=全て）
            **kwargs: 各ローダーへの追加パラメータ
        
        Returns:
            ソース名をキーとしたテキストリストの辞書
        """
        if sources is None:
            sources = list(self.loaders.keys())
        
        results = {}
        
        for source in sources:
            if source in self.loaders:
                try:
                    loader = self.loaders[source]
                    source_kwargs = kwargs.get(source, {})
                    texts = loader.load_data(**source_kwargs)
                    results[source] = texts
                except Exception as e:
                    print(f"⚠️ {source}のロードに失敗: {e}")
                    results[source] = []
        
        return results
    
    def load_combined(self, sources: List[str] = None, **kwargs) -> List[str]:
        """
        すべてのソースからデータをロードして結合
        """
        all_data = self.load_all(sources, **kwargs)
        combined = []
        for source, texts in all_data.items():
            combined.extend(texts)
        return combined
    
    def get_available_sources(self) -> List[str]:
        """利用可能なデータソースのリストを取得"""
        return list(self.loaders.keys())


# デバイス選択
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🎮 CUDA GPU を使用: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🍎 Apple Silicon GPU (MPS) を使用")
else:
    DEVICE = "cpu"
    print("💻 CPU を使用")

# ========================================
# モデルキャッシュ
# ========================================

# モードごとにモデルをキャッシュ
model_cache = {}

def get_config_key(config_params: dict) -> str:
    """設定からキャッシュキーを生成"""
    return json.dumps(config_params, sort_keys=True)


def load_model(mode: str = None, config_params: dict = None):
    """
    モデルをロード（キャッシュあり）
    
    Args:
        mode: 'brain' or 'layered' (Noneの場合はDEFAULT_MODEを使用)
        config_params: カスタム設定（指定されていない場合はデフォルト使用）
    """
    global model_cache
    
    if mode is None:
        mode = DEFAULT_MODE
    
    # 設定をマージ
    params = DEFAULT_CONFIG.copy()
    if config_params:
        params.update(config_params)
    
    # キャッシュキーを生成
    cache_key = f"{mode}_{get_config_key(params)}"
    
    # キャッシュにあればそれを返す
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    print(f"📥 NeuroQ モデルをロード中...")
    print(f"   Mode: {mode}")
    print(f"   Config: {json.dumps(params, indent=2)}")
    print(f"   Device: {DEVICE}")
    
    # モード別のモデルファイルを探す
    mode_model_path = f"neuroq_{mode}_model.pt"
    actual_model_path = mode_model_path if os.path.exists(mode_model_path) else MODEL_PATH
    
    # ファイルが存在するか確認
    if not os.path.exists(actual_model_path):
        print(f"⚠️ モデルファイルが見つかりません: {actual_model_path}")
        print("   カスタム設定でモデルを作成します")
        
        # カスタム設定でモデルを作成
        if mode == 'brain':
            config = NeuroQConfig(
                mode='brain',
                vocab_size=2000,
                embed_dim=params['embed_dim'],
                num_neurons=params['num_neurons'],
                hidden_dim=params['num_neurons'] * 2,
                num_heads=params.get('num_heads', 4),
                num_layers=params['num_layers'],
                max_seq_len=params['max_seq_len'],
                dropout=params['dropout'],
                connection_density=params['connection_density'],
                lambda_entangle=params['lambda_entangle_brain'],
            )
        else:  # layered
            config = NeuroQConfig(
                mode='layered',
                vocab_size=2000,
                embed_dim=params['embed_dim'],
                hidden_dim=params['hidden_dim'],
                num_heads=params['num_heads'],
                num_layers=params['num_layers'],
                max_seq_len=params['max_seq_len'],
                dropout=params['dropout'],
                lambda_entangle=params['lambda_entangle_layered'],
            )
        
        model = NeuroQModel(config)
        tokenizer = NeuroQTokenizer(vocab_size=2000)
        
        # 基本的な語彙を構築
        basic_texts = [
            "こんにちは、私はNeuroQです。量子ビットニューラルネットワークベースの生成AIです。",
            "Hello, I am NeuroQ. A generative AI based on Quantum-Bit Neural Network.",
            "量子コンピュータは次世代の計算技術です。",
            "人工知能は私たちの生活を変革しています。",
            "QBNNは量子もつれを模倣したニューラルネットワークです。",
        ]
        tokenizer.build_vocab(basic_texts)
        
        generator = NeuroQGenerator(model, tokenizer, DEVICE)
        print(f"✅ カスタムモデル作成完了 (Mode: {mode})")
    else:
        # 学習済みモデルをロード
        generator = NeuroQGenerator.load(actual_model_path, TOKENIZER_PATH, DEVICE)
        print(f"✅ モデルロード完了: {actual_model_path}")
    
    # モデル情報を表示
    info = generator.get_model_info()
    print(f"   モード: {info.get('mode', 'unknown')}")
    print(f"   パラメータ数: {info['num_params']:,}")
    print(f"   埋め込み次元: {info['embed_dim']}")
    print(f"   隠れ層次元: {info['hidden_dim']}")
    print(f"   ニューロン数: {info.get('num_neurons', 'N/A')}")
    print(f"   レイヤー数: {info['num_layers']}")
    
    # キャッシュに保存
    model_cache[cache_key] = generator
    
    return generator


# ========================================
# RunPod Handler
# ========================================

def handler(job):
    """
    RunPod Serverless ハンドラー
    
    入力JSON形式:
    {
        "input": {
            // === 生成パラメータ ===
            "prompt": "生成したいテキストのプロンプト",  // 必須
            "max_tokens": 128,        // オプション（デフォルト: 128）
            "temperature": 0.7,       // オプション（デフォルト: 0.7）
            "top_k": 40,              // オプション（デフォルト: 40）
            "top_p": 0.9,             // オプション（デフォルト: 0.9）
            "repetition_penalty": 1.2 // オプション（デフォルト: 1.2）
            
            // === モデル設定 ===
            "mode": "brain",          // オプション: "brain" or "layered"
            
            // Brain Mode 専用
            "num_neurons": 100,       // ニューロン数
            "connection_density": 0.25, // 接続密度 (0.0-1.0)
            "lambda_entangle": 0.35,  // 量子もつれ強度
            
            // Layered Mode 専用
            "hidden_dim": 256,        // 隠れ層次元
            "num_heads": 4,           // アテンションヘッド数
            
            // 共通
            "embed_dim": 128,         // 埋め込み次元
            "num_layers": 3,          // レイヤー数
            
            // === 学習パラメータ ===
            "epochs": 10,             // エポック数（デフォルト: 10）
            "batch_size": 32,         // バッチサイズ（デフォルト: 32）
            "learning_rate": 1e-4,    // 学習率（デフォルト: 1e-4）
            "min_learning_rate": 1e-6, // 最小学習率（デフォルト: 1e-6）
            "weight_decay": 0.01,     // 重み減衰（デフォルト: 0.01）
            "warmup_steps": 100,      // ウォームアップステップ（デフォルト: 100）
            "lr_scheduler": "cosine", // LRスケジューラー: cosine, linear, constant
            "max_temperature": 1.5,   // 最大温度（デフォルト: 1.5）
            "min_temperature": 0.1,   // 最小温度（デフォルト: 0.1）
            "temperature_decay": "linear", // 温度減衰: linear, exponential, cosine
            "max_grad_norm": 1.0,     // 勾配クリッピング（デフォルト: 1.0）
            "label_smoothing": 0.1,   // ラベルスムージング（デフォルト: 0.1）
            "gradient_accumulation_steps": 1, // 勾配累積ステップ
            "eval_steps": 500,        // 評価間隔ステップ
            "save_steps": 1000,       // 保存間隔ステップ
            "logging_steps": 100,     // ログ間隔ステップ
            "early_stopping_patience": 3,    // 早期停止patience
            "early_stopping_threshold": 0.001, // 早期停止閾値
            "train_split": 0.9,       // 学習/検証分割比率
            "seed": 42,               // 乱数シード
        }
    }
    
    出力JSON形式:
    {
        "prompt": "入力プロンプト",
        "output": "生成されたテキスト",
        "model_info": {
            "mode": "brain" or "layered",
            "num_neurons": 100,
            "num_params": 123456,
            ...
        }
    }
    """
    try:
        # 入力を取得
        job_input = job.get("input", {})
        
        # モード設定
        mode = job_input.get("mode", DEFAULT_MODE)
        
        # モデル設定パラメータを抽出
        config_params = {}
        
        # 共通パラメータ
        if "embed_dim" in job_input:
            config_params["embed_dim"] = int(job_input["embed_dim"])
        if "num_layers" in job_input:
            config_params["num_layers"] = int(job_input["num_layers"])
        if "dropout" in job_input:
            config_params["dropout"] = float(job_input["dropout"])
        if "max_seq_len" in job_input:
            config_params["max_seq_len"] = int(job_input["max_seq_len"])
        
        # Brain Mode 専用
        if "num_neurons" in job_input:
            config_params["num_neurons"] = int(job_input["num_neurons"])
        if "connection_density" in job_input:
            config_params["connection_density"] = float(job_input["connection_density"])
        if "lambda_entangle" in job_input and mode == "brain":
            config_params["lambda_entangle_brain"] = float(job_input["lambda_entangle"])
        
        # Layered Mode 専用
        if "hidden_dim" in job_input:
            config_params["hidden_dim"] = int(job_input["hidden_dim"])
        if "num_heads" in job_input:
            config_params["num_heads"] = int(job_input["num_heads"])
        if "lambda_entangle" in job_input and mode == "layered":
            config_params["lambda_entangle_layered"] = float(job_input["lambda_entangle"])
        
        # 学習パラメータを抽出
        training_params = {}
        
        # エポック・バッチ関連
        if "epochs" in job_input:
            training_params["epochs"] = int(job_input["epochs"])
        if "batch_size" in job_input:
            training_params["batch_size"] = int(job_input["batch_size"])
        if "gradient_accumulation_steps" in job_input:
            training_params["gradient_accumulation_steps"] = int(job_input["gradient_accumulation_steps"])
        
        # 学習率関連
        if "learning_rate" in job_input:
            training_params["learning_rate"] = float(job_input["learning_rate"])
        if "min_learning_rate" in job_input:
            training_params["min_learning_rate"] = float(job_input["min_learning_rate"])
        if "weight_decay" in job_input:
            training_params["weight_decay"] = float(job_input["weight_decay"])
        if "warmup_steps" in job_input:
            training_params["warmup_steps"] = int(job_input["warmup_steps"])
        if "lr_scheduler" in job_input:
            training_params["lr_scheduler"] = str(job_input["lr_scheduler"])
        
        # Temperature関連
        if "max_temperature" in job_input:
            training_params["max_temperature"] = float(job_input["max_temperature"])
        if "min_temperature" in job_input:
            training_params["min_temperature"] = float(job_input["min_temperature"])
        if "temperature_decay" in job_input:
            training_params["temperature_decay"] = str(job_input["temperature_decay"])
        
        # 正則化・最適化
        if "max_grad_norm" in job_input:
            training_params["max_grad_norm"] = float(job_input["max_grad_norm"])
        if "label_smoothing" in job_input:
            training_params["label_smoothing"] = float(job_input["label_smoothing"])
        
        # 評価・保存
        if "eval_steps" in job_input:
            training_params["eval_steps"] = int(job_input["eval_steps"])
        if "save_steps" in job_input:
            training_params["save_steps"] = int(job_input["save_steps"])
        if "logging_steps" in job_input:
            training_params["logging_steps"] = int(job_input["logging_steps"])
        
        # 早期停止
        if "early_stopping_patience" in job_input:
            training_params["early_stopping_patience"] = int(job_input["early_stopping_patience"])
        if "early_stopping_threshold" in job_input:
            training_params["early_stopping_threshold"] = float(job_input["early_stopping_threshold"])
        
        # データ関連
        if "train_split" in job_input:
            training_params["train_split"] = float(job_input["train_split"])
        if "shuffle" in job_input:
            training_params["shuffle"] = bool(job_input["shuffle"])
        if "seed" in job_input:
            training_params["seed"] = int(job_input["seed"])
        
        # モデルをロード
        gen = load_model(mode, config_params if config_params else None)
        
        # 生成パラメータを取得
        prompt = job_input.get("prompt", "")
        max_tokens = int(job_input.get("max_tokens", 128))
        temperature = float(job_input.get("temperature", 0.7))
        top_k = int(job_input.get("top_k", 40))
        top_p = float(job_input.get("top_p", 0.9))
        repetition_penalty = float(job_input.get("repetition_penalty", 1.2))
        
        # バリデーション
        if not prompt:
            return {"error": "prompt is required"}
        
        if max_tokens < 1 or max_tokens > 1024:
            max_tokens = min(max(1, max_tokens), 1024)
        
        if temperature < 0.1 or temperature > 2.0:
            temperature = min(max(0.1, temperature), 2.0)
        
        print(f"📝 生成リクエスト:")
        print(f"   Mode: {mode}")
        print(f"   Prompt: {prompt[:50]}...")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Temperature: {temperature}")
        if config_params:
            print(f"   Custom config: {json.dumps(config_params)}")
        if training_params:
            print(f"   Training params: {json.dumps(training_params)}")
        
        # テキスト生成
        output_text = gen.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        print(f"✅ 生成完了: {len(output_text)} 文字")
        
        # 学習パラメータをデフォルト値とマージ
        merged_training_params = DEFAULT_TRAINING_CONFIG.copy()
        merged_training_params.update(training_params)
        
        # レスポンス
        return {
            "prompt": prompt,
            "output": output_text,
            "model_info": gen.get_model_info(),
            "config": {
                "mode": mode,
                **config_params
            },
            "training_config": merged_training_params
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}


# ========================================
# 学習エンドポイント
# ========================================

def train_handler(job):
    """
    モデル学習用ハンドラー
    
    入力JSON形式:
    {
        "input": {
            // === データソース（いずれか必須） ===
            "training_data": ["テキスト1", "テキスト2", ...],  // 直接テキストを指定
            // または
            "data_sources": ["common_crawl", "pubmed"],  // データソースから取得
            
            // === データソース設定（オプション） ===
            "common_crawl_config": {
                "domains": ["*.wikipedia.org", "*.news.yahoo.co.jp"],
                "max_records": 500,
                "crawl_id": "CC-MAIN-2024-10"
            },
            "pubmed_config": {
                "search_terms": ["quantum computing", "neural network"],
                "max_records": 500
            },
            
            "mode": "layered",        // オプション: "brain" or "layered"
            
            // 学習パラメータ
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "min_learning_rate": 1e-6,
            "warmup_steps": 100,
            "lr_scheduler": "cosine",
            "max_temperature": 1.5,
            "min_temperature": 0.1,
            "temperature_decay": "linear",
            "max_grad_norm": 1.0,
            "label_smoothing": 0.1,
            ...
        }
    }
    """
    try:
        job_input = job.get("input", {})
        
        # 学習データを取得（直接指定またはデータソースから）
        training_data = job_input.get("training_data", [])
        data_sources = job_input.get("data_sources", [])
        
        # データソースからロード
        loaded_data_info = {}
        if data_sources:
            print(f"📊 データソースからデータをロード中...")
            print(f"   ソース: {data_sources}")
            
            # データソース設定を構築
            data_source_config = DEFAULT_DATA_SOURCE_CONFIG.copy()
            
            # Common Crawl設定をマージ
            if "common_crawl_config" in job_input:
                cc_config = job_input["common_crawl_config"]
                data_source_config["common_crawl"].update(cc_config)
            
            # PubMed設定をマージ
            if "pubmed_config" in job_input:
                pm_config = job_input["pubmed_config"]
                data_source_config["pubmed"].update(pm_config)
            
            # データローダーマネージャーを作成
            manager = DataSourceManager(data_source_config)
            
            # 各ソースからロード
            source_kwargs = {}
            
            if "common_crawl" in data_sources:
                cc_kwargs = {}
                if "common_crawl_config" in job_input:
                    if "domains" in job_input["common_crawl_config"]:
                        cc_kwargs["domains"] = job_input["common_crawl_config"]["domains"]
                    if "max_records" in job_input["common_crawl_config"]:
                        cc_kwargs["max_records"] = job_input["common_crawl_config"]["max_records"]
                source_kwargs["common_crawl"] = cc_kwargs
            
            if "pubmed" in data_sources:
                pm_kwargs = {}
                if "pubmed_config" in job_input:
                    if "search_terms" in job_input["pubmed_config"]:
                        pm_kwargs["search_terms"] = job_input["pubmed_config"]["search_terms"]
                    if "max_records" in job_input["pubmed_config"]:
                        pm_kwargs["max_records"] = job_input["pubmed_config"]["max_records"]
                source_kwargs["pubmed"] = pm_kwargs
            
            # データをロード
            loaded_data = manager.load_all(sources=data_sources, **source_kwargs)
            
            # ロード結果を記録
            for source, texts in loaded_data.items():
                loaded_data_info[source] = len(texts)
                training_data.extend(texts)
            
            print(f"✅ データソースからのロード完了:")
            for source, count in loaded_data_info.items():
                print(f"   {source}: {count}件")
        
        if not training_data:
            return {"error": "training_data or data_sources is required"}
        
        mode = job_input.get("mode", DEFAULT_MODE)
        
        # 学習パラメータを抽出してマージ
        training_config = DEFAULT_TRAINING_CONFIG.copy()
        
        training_param_keys = [
            "epochs", "batch_size", "gradient_accumulation_steps",
            "learning_rate", "min_learning_rate", "weight_decay", "warmup_steps", "lr_scheduler",
            "max_temperature", "min_temperature", "temperature_decay",
            "max_grad_norm", "label_smoothing",
            "eval_steps", "save_steps", "logging_steps",
            "early_stopping_patience", "early_stopping_threshold",
            "train_split", "shuffle", "seed"
        ]
        
        for key in training_param_keys:
            if key in job_input:
                value = job_input[key]
                # 型変換
                if key in ["epochs", "batch_size", "gradient_accumulation_steps", "warmup_steps", 
                           "eval_steps", "save_steps", "logging_steps", "early_stopping_patience", "seed"]:
                    training_config[key] = int(value)
                elif key in ["learning_rate", "min_learning_rate", "weight_decay", 
                             "max_temperature", "min_temperature", "max_grad_norm", 
                             "label_smoothing", "early_stopping_threshold", "train_split"]:
                    training_config[key] = float(value)
                elif key == "shuffle":
                    training_config[key] = bool(value)
                else:
                    training_config[key] = str(value)
        
        print(f"🎓 学習リクエスト:")
        print(f"   Mode: {mode}")
        print(f"   Training data size: {len(training_data)} samples")
        print(f"   Epochs: {training_config['epochs']}")
        print(f"   Batch size: {training_config['batch_size']}")
        print(f"   Learning rate: {training_config['learning_rate']}")
        print(f"   Max/Min temperature: {training_config['max_temperature']}/{training_config['min_temperature']}")
        print(f"   Temperature decay: {training_config['temperature_decay']}")
        print(f"   LR scheduler: {training_config['lr_scheduler']}")
        print(f"   Warmup steps: {training_config['warmup_steps']}")
        
        # モデル設定パラメータを抽出
        config_params = {}
        model_param_keys = [
            "embed_dim", "num_layers", "dropout", "max_seq_len",
            "num_neurons", "connection_density", "hidden_dim", "num_heads"
        ]
        for key in model_param_keys:
            if key in job_input:
                if key in ["dropout", "connection_density"]:
                    config_params[key] = float(job_input[key])
                else:
                    config_params[key] = int(job_input[key])
        
        if "lambda_entangle" in job_input:
            if mode == "brain":
                config_params["lambda_entangle_brain"] = float(job_input["lambda_entangle"])
            else:
                config_params["lambda_entangle_layered"] = float(job_input["lambda_entangle"])
        
        # モデルをロード
        gen = load_model(mode, config_params if config_params else None)
        
        # 注: 実際の学習処理はNeuroQGeneratorに実装が必要
        # ここでは学習パラメータの受け渡しと設定の確認のみ
        
        return {
            "status": "training_config_ready",
            "message": "Training parameters received and validated",
            "mode": mode,
            "training_data_count": len(training_data),
            "data_sources_used": loaded_data_info if loaded_data_info else None,
            "model_config": config_params if config_params else DEFAULT_CONFIG,
            "training_config": training_config,
            "model_info": gen.get_model_info(),
        }
        
    except Exception as e:
        error_msg = f"Training Error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}


# ========================================
# データソースエンドポイント
# ========================================

def fetch_data_handler(job):
    """
    データソースからデータを取得するハンドラー
    
    入力JSON形式:
    {
        "input": {
            "sources": ["common_crawl", "pubmed"],  // 取得するソース
            
            // Common Crawl設定
            "common_crawl_config": {
                "domains": ["*.wikipedia.org", "*.news.yahoo.co.jp"],
                "max_records": 100,
                "crawl_id": "CC-MAIN-2024-10"
            },
            
            // PubMed設定
            "pubmed_config": {
                "search_terms": ["quantum computing", "neural network"],
                "max_records": 100
            }
        }
    }
    """
    try:
        job_input = job.get("input", {})
        sources = job_input.get("sources", ["common_crawl", "pubmed"])
        
        print(f"📊 データソースからデータを取得中...")
        print(f"   ソース: {sources}")
        
        # データソース設定を構築
        data_source_config = DEFAULT_DATA_SOURCE_CONFIG.copy()
        
        if "common_crawl_config" in job_input:
            data_source_config["common_crawl"].update(job_input["common_crawl_config"])
        
        if "pubmed_config" in job_input:
            data_source_config["pubmed"].update(job_input["pubmed_config"])
        
        # データローダーマネージャーを作成
        manager = DataSourceManager(data_source_config)
        
        # 各ソースからロード
        source_kwargs = {}
        
        if "common_crawl" in sources:
            cc_kwargs = {}
            if "common_crawl_config" in job_input:
                if "domains" in job_input["common_crawl_config"]:
                    cc_kwargs["domains"] = job_input["common_crawl_config"]["domains"]
                if "max_records" in job_input["common_crawl_config"]:
                    cc_kwargs["max_records"] = job_input["common_crawl_config"]["max_records"]
            source_kwargs["common_crawl"] = cc_kwargs
        
        if "pubmed" in sources:
            pm_kwargs = {}
            if "pubmed_config" in job_input:
                if "search_terms" in job_input["pubmed_config"]:
                    pm_kwargs["search_terms"] = job_input["pubmed_config"]["search_terms"]
                if "max_records" in job_input["pubmed_config"]:
                    pm_kwargs["max_records"] = job_input["pubmed_config"]["max_records"]
            source_kwargs["pubmed"] = pm_kwargs
        
        # データをロード
        loaded_data = manager.load_all(sources=sources, **source_kwargs)
        
        # 結果を整理
        results = {}
        total_count = 0
        for source, texts in loaded_data.items():
            results[source] = {
                "count": len(texts),
                "sample": texts[:3] if texts else [],  # 最初の3件をサンプルとして
            }
            total_count += len(texts)
        
        print(f"✅ データ取得完了: 合計 {total_count}件")
        
        return {
            "status": "success",
            "total_count": total_count,
            "sources": results,
            "available_sources": manager.get_available_sources(),
        }
        
    except Exception as e:
        error_msg = f"Data Fetch Error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}


def data_source_config_handler(job):
    """
    データソース設定を取得するハンドラー
    """
    try:
        manager = DataSourceManager()
        
        return {
            "status": "success",
            "available_sources": manager.get_available_sources(),
            "default_config": DEFAULT_DATA_SOURCE_CONFIG,
            "usage_examples": {
                "fetch_from_common_crawl": {
                    "input": {
                        "sources": ["common_crawl"],
                        "common_crawl_config": {
                            "domains": ["*.wikipedia.org"],
                            "max_records": 100
                        }
                    }
                },
                "fetch_from_pubmed": {
                    "input": {
                        "sources": ["pubmed"],
                        "pubmed_config": {
                            "search_terms": ["quantum computing", "machine learning"],
                            "max_records": 100
                        }
                    }
                },
                "train_with_data_sources": {
                    "input": {
                        "data_sources": ["common_crawl", "pubmed"],
                        "common_crawl_config": {
                            "domains": ["*.wikipedia.org"],
                            "max_records": 500
                        },
                        "pubmed_config": {
                            "search_terms": ["neural network"],
                            "max_records": 500
                        },
                        "epochs": 10,
                        "batch_size": 32,
                        "learning_rate": 1e-4
                    }
                }
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ========================================
# 量子情報エンドポイント
# ========================================

def quantum_info(job):
    """量子もつれ情報を取得"""
    try:
        job_input = job.get("input", {})
        mode = job_input.get("mode", DEFAULT_MODE)
        
        gen = load_model(mode)
        model_info = gen.get_model_info()
        quantum_info = gen.model.get_quantum_info()
        
        return {
            "status": "success",
            "mode": model_info.get('mode', 'unknown'),
            "model_info": model_info,
            "quantum_info": quantum_info,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ========================================
# モデル設定エンドポイント
# ========================================

def model_config(job):
    """
    現在のモデル設定と利用可能なオプションを取得
    """
    try:
        manager = DataSourceManager()
        
        return {
            "status": "success",
            "default_mode": DEFAULT_MODE,
            "default_config": DEFAULT_CONFIG,
            "default_training_config": DEFAULT_TRAINING_CONFIG,
            "default_data_source_config": DEFAULT_DATA_SOURCE_CONFIG,
            "available_data_sources": manager.get_available_sources(),
            "device": DEVICE,
            "cached_models": list(model_cache.keys()),
            "available_options": {
                "common": {
                    "embed_dim": "埋め込み次元（デフォルト: 128）",
                    "num_layers": "レイヤー数（デフォルト: 3）",
                    "dropout": "ドロップアウト率（デフォルト: 0.1）",
                    "max_seq_len": "最大シーケンス長（デフォルト: 256）",
                },
                "brain_mode": {
                    "num_neurons": "ニューロン数（デフォルト: 100）",
                    "connection_density": "接続密度 0.0-1.0（デフォルト: 0.25）",
                    "lambda_entangle": "量子もつれ強度（デフォルト: 0.35）",
                },
                "layered_mode": {
                    "hidden_dim": "隠れ層次元（デフォルト: 256）",
                    "num_heads": "アテンションヘッド数（デフォルト: 4）",
                    "lambda_entangle": "量子もつれ強度（デフォルト: 0.5）",
                },
                "training": {
                    "epochs": "エポック数（デフォルト: 10）",
                    "batch_size": "バッチサイズ（デフォルト: 32）",
                    "gradient_accumulation_steps": "勾配累積ステップ数（デフォルト: 1）",
                    "learning_rate": "学習率（デフォルト: 1e-4）",
                    "min_learning_rate": "最小学習率（デフォルト: 1e-6）",
                    "weight_decay": "重み減衰（デフォルト: 0.01）",
                    "warmup_steps": "ウォームアップステップ数（デフォルト: 100）",
                    "lr_scheduler": "学習率スケジューラー: cosine, linear, constant（デフォルト: cosine）",
                    "max_temperature": "最大サンプリング温度（デフォルト: 1.5）",
                    "min_temperature": "最小サンプリング温度（デフォルト: 0.1）",
                    "temperature_decay": "温度減衰方式: linear, exponential, cosine（デフォルト: linear）",
                    "max_grad_norm": "勾配クリッピング閾値（デフォルト: 1.0）",
                    "label_smoothing": "ラベルスムージング係数（デフォルト: 0.1）",
                    "eval_steps": "評価間隔ステップ数（デフォルト: 500）",
                    "save_steps": "モデル保存間隔ステップ数（デフォルト: 1000）",
                    "logging_steps": "ログ出力間隔ステップ数（デフォルト: 100）",
                    "early_stopping_patience": "早期停止patience（デフォルト: 3）",
                    "early_stopping_threshold": "早期停止閾値（デフォルト: 0.001）",
                    "train_split": "学習データ分割比率（デフォルト: 0.9）",
                    "shuffle": "データシャッフル有効化（デフォルト: true）",
                    "seed": "乱数シード（デフォルト: 42）",
                },
                "data_sources": {
                    "common_crawl": {
                        "description": "Common Crawl - 大規模ウェブクロールデータセット (https://commoncrawl.org/)",
                        "domains": "検索するドメインのリスト（例: *.wikipedia.org）",
                        "max_records": "取得する最大レコード数",
                        "crawl_id": "クロールID（例: CC-MAIN-2024-10）",
                        "languages": "対象言語（デフォルト: ja, en）",
                    },
                    "pubmed": {
                        "description": "PubMed - 医学・生命科学論文データベース (https://pubmed.ncbi.nlm.nih.gov/)",
                        "search_terms": "検索キーワードのリスト",
                        "max_records": "取得する最大レコード数",
                        "api_key": "NCBI API Key（オプション、レート制限緩和用）",
                        "include_abstract": "アブストラクトを含める（デフォルト: true）",
                        "include_mesh_terms": "MeSHタームを含める（デフォルト: true）",
                    },
                },
            },
            "example_request": {
                "input": {
                    "prompt": "こんにちは",
                    "mode": "brain",
                    "num_neurons": 200,
                    "connection_density": 0.3,
                    "max_tokens": 64,
                    "temperature": 0.7
                }
            },
            "example_training_request": {
                "input": {
                    "action": "train",
                    "training_data": ["テキスト1", "テキスト2", "..."],
                    "mode": "layered",
                    "epochs": 20,
                    "batch_size": 16,
                    "learning_rate": 5e-5,
                    "max_temperature": 1.2,
                    "min_temperature": 0.3,
                    "warmup_steps": 200,
                    "early_stopping_patience": 5
                }
            },
            "example_training_with_data_sources": {
                "input": {
                    "data_sources": ["common_crawl", "pubmed"],
                    "common_crawl_config": {
                        "domains": ["*.wikipedia.org", "*.news.yahoo.co.jp"],
                        "max_records": 500
                    },
                    "pubmed_config": {
                        "search_terms": ["quantum computing", "neural network", "machine learning"],
                        "max_records": 500
                    },
                    "mode": "layered",
                    "epochs": 20,
                    "batch_size": 16,
                    "learning_rate": 5e-5,
                    "max_temperature": 1.2,
                    "min_temperature": 0.3
                }
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ========================================
# ヘルスチェック用エンドポイント
# ========================================

def health_check(job):
    """ヘルスチェック"""
    try:
        gen = load_model()
        return {
            "status": "healthy",
            "model_loaded": gen is not None,
            "device": DEVICE,
            "model_info": gen.get_model_info() if gen else None,
            "cached_models": len(model_cache),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


# ========================================
# コマンドライン引数
# ========================================

def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="NeuroQ RunPod Serverless Worker - QBNN-LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 生成パラメータ
    parser.add_argument(
        "--max-tokens", "-m",
        type=int,
        default=128,
        help="生成する最大トークン数"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="サンプリング温度 (0.1-2.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-K サンプリングのK値"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-P (Nucleus) サンプリングのP値"
    )
    
    # モデル設定
    parser.add_argument(
        "--mode",
        type=str,
        choices=["brain", "layered"],
        default=DEFAULT_MODE,
        help="モデルモード: brain (脳型散在QBNN) or layered (層状QBNN-Transformer)"
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=DEFAULT_CONFIG["embed_dim"],
        help="埋め込み次元"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_CONFIG["num_layers"],
        help="レイヤー数"
    )
    parser.add_argument(
        "--num-neurons",
        type=int,
        default=DEFAULT_CONFIG["num_neurons"],
        help="ニューロン数 (Brain Mode用)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_CONFIG["hidden_dim"],
        help="隠れ層次元 (Layered Mode用)"
    )
    
    # 実行モード
    parser.add_argument(
        "--test",
        action="store_true",
        help="テストモードで実行（RunPodサーバーを起動せず、テスト生成を行う）"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="こんにちは",
        help="テストモード時の入力プロンプト"
    )
    
    return parser.parse_args()


# ========================================
# メイン
# ========================================

if __name__ == "__main__":
    # コマンドライン引数をパース
    args = parse_args()
    
    print("=" * 60)
    print("🧠⚛️ NeuroQ RunPod Serverless Worker")
    print("   Brain Mode: 脳型散在QBNN")
    print("   Layered Mode: 層状QBNN-Transformer")
    print("=" * 60)
    
    print("\n📋 コマンドライン引数:")
    print(f"   --max-tokens: {args.max_tokens}")
    print(f"   --temperature: {args.temperature}")
    print(f"   --top-k: {args.top_k}")
    print(f"   --top-p: {args.top_p}")
    print(f"   --mode: {args.mode}")
    
    print("\n📋 デフォルトモデル設定:")
    print(f"   Mode: {DEFAULT_MODE}")
    for key, value in DEFAULT_CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n📚 デフォルト学習設定:")
    for key, value in DEFAULT_TRAINING_CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n📊 データソース設定:")
    print("   利用可能なソース:")
    print("   - common_crawl: Common Crawl (https://commoncrawl.org/)")
    print("   - pubmed: PubMed (https://pubmed.ncbi.nlm.nih.gov/)")
    for source, config in DEFAULT_DATA_SOURCE_CONFIG.items():
        print(f"   [{source}]")
        if source == "common_crawl":
            print(f"     crawl_id: {config.get('crawl_id')}")
            print(f"     max_records: {config.get('max_records')}")
        elif source == "pubmed":
            print(f"     search_terms: {config.get('search_terms')}")
            print(f"     max_records: {config.get('max_records')}")
    print()
    
    # モデル設定を引数から構築
    config_params = {
        "embed_dim": args.embed_dim,
        "num_layers": args.num_layers,
        "num_neurons": args.num_neurons,
        "hidden_dim": args.hidden_dim,
    }
    
    # 起動時にモデルをプリロード
    gen = load_model(args.mode, config_params)
    
    if args.test:
        # テストモード: 生成テストを実行
        print("\n🧪 テストモード: 生成テストを実行")
        print(f"   Prompt: {args.prompt}")
        print(f"   Max tokens: {args.max_tokens}")
        print(f"   Temperature: {args.temperature}")
        print()
        
        output = gen.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        
        print("📝 生成結果:")
        print("-" * 40)
        print(output)
        print("-" * 40)
    else:
        # RunPod Serverless を開始
        runpod.serverless.start({
            "handler": handler,
            "train": train_handler,
            "fetch_data": fetch_data_handler,
            "data_source_config": data_source_config_handler,
        })
