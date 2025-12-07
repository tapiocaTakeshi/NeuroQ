#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common Crawl データ取得モジュール

Common Crawl Index Server APIを使用して日本語コンテンツを取得します。

主な機能:
- Common Crawl Index APIクエリ
- WARCファイルからのテキスト抽出
- 日本語コンテンツのフィルタリング
"""

import re
import gzip
import json
from typing import List, Optional, Dict, Any
from io import BytesIO

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests が必要です: pip install requests")

try:
    from warcio.archiveiterator import ArchiveIterator
    WARCIO_AVAILABLE = True
except ImportError:
    WARCIO_AVAILABLE = False
    print("⚠️ warcio が必要です: pip install warcio")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("⚠️ beautifulsoup4 が必要です: pip install beautifulsoup4")


# Common Crawl Index Server
CC_INDEX_SERVER = "https://index.commoncrawl.org"


def get_latest_index() -> Optional[str]:
    """
    最新のCommon Crawl Index名を取得

    Returns:
        Index名 (例: "CC-MAIN-2024-10")、取得失敗時はNone
    """
    if not REQUESTS_AVAILABLE:
        return None

    try:
        response = requests.get(f"{CC_INDEX_SERVER}/collinfo.json", timeout=10)
        response.raise_for_status()

        collections = response.json()
        if collections and len(collections) > 0:
            # 最新のインデックスを取得
            latest = collections[0]['id']
            return latest

        return None

    except Exception as e:
        print(f"⚠️ Common Crawl Index取得エラー: {e}")
        return None


def search_common_crawl_index(
    query: str,
    index_name: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Common Crawl Index Serverでクエリを実行

    Args:
        query: 検索クエリ（ドメイン名など、例: "*.jp"）
        index_name: Index名（Noneの場合は最新を使用）
        limit: 取得する最大レコード数

    Returns:
        CDXレコードのリスト
    """
    if not REQUESTS_AVAILABLE:
        print("⚠️ requests がインストールされていません")
        return []

    # Index名の取得
    if index_name is None:
        index_name = get_latest_index()
        if index_name is None:
            print("⚠️ Common Crawl Index名の取得に失敗しました")
            return []

    print(f"   📡 Common Crawl Index: {index_name}")

    # CDX Server API
    cdx_api_url = f"{CC_INDEX_SERVER}/{index_name}-index"

    params = {
        'url': query,
        'output': 'json',
        'limit': limit
    }

    try:
        print(f"   🔍 検索クエリ: {query} (最大 {limit} レコード)")
        response = requests.get(cdx_api_url, params=params, timeout=30)
        response.raise_for_status()

        # レスポンスを行ごとに分割してJSONとしてパース
        lines = response.text.strip().split('\n')
        records = []

        for line in lines:
            if line.strip():
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue

        print(f"   ✅ {len(records)} レコード取得")
        return records

    except Exception as e:
        print(f"   ⚠️ Common Crawl検索エラー: {e}")
        return []


def fetch_warc_record(warc_path: str, offset: int, length: int) -> Optional[bytes]:
    """
    S3からWARCレコードを取得

    Args:
        warc_path: WARCファイルのパス
        offset: レコードのオフセット
        length: レコードの長さ

    Returns:
        レコードのバイトデータ、取得失敗時はNone
    """
    if not REQUESTS_AVAILABLE:
        return None

    # Common CrawlのS3バケット
    s3_base = "https://data.commoncrawl.org"
    warc_url = f"{s3_base}/{warc_path}"

    # Range リクエストでレコードの該当部分のみ取得
    headers = {
        'Range': f'bytes={offset}-{offset + length - 1}'
    }

    try:
        response = requests.get(warc_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content

    except Exception as e:
        print(f"   ⚠️ WARCレコード取得エラー: {e}")
        return None


def extract_text_from_html(html: str) -> str:
    """
    HTMLからテキストを抽出

    Args:
        html: HTMLコンテンツ

    Returns:
        抽出されたテキスト
    """
    if not BS4_AVAILABLE:
        # Beautiful Soupが使えない場合は簡易的にタグを除去
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    try:
        soup = BeautifulSoup(html, 'html.parser')

        # スクリプトとスタイルを除去
        for script in soup(['script', 'style', 'meta', 'link']):
            script.decompose()

        # テキストを取得
        text = soup.get_text(separator=' ', strip=True)

        # 余分な空白を削除
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    except Exception as e:
        print(f"   ⚠️ HTMLパースエラー: {e}")
        return ""


def is_japanese_text(text: str, min_ratio: float = 0.1) -> bool:
    """
    テキストが日本語を含むかチェック

    Args:
        text: チェックするテキスト
        min_ratio: 最小日本語文字比率

    Returns:
        日本語を含む場合はTrue
    """
    if not text:
        return False

    # 日本語文字（ひらがな、カタカナ、漢字）をカウント
    japanese_chars = len(re.findall(r'[ぁ-んァ-ヶー一-龯]', text))
    total_chars = len(text)

    if total_chars == 0:
        return False

    ratio = japanese_chars / total_chars
    return ratio >= min_ratio


def fetch_common_crawl_data(
    max_records: int = 100,
    query: str = "*.jp",
    index_name: Optional[str] = None,
    min_text_length: int = 100
) -> List[str]:
    """
    Common Crawlからデータを取得

    Args:
        max_records: 取得する最大レコード数
        query: 検索クエリ（デフォルト: "*.jp" で日本のドメイン）
        index_name: Index名（Noneの場合は最新を使用）
        min_text_length: 最小テキスト長

    Returns:
        抽出されたテキストのリスト
    """
    print(f"   🌐 Common Crawlからデータ取得中...")

    if not REQUESTS_AVAILABLE:
        print("   ⚠️ requests が必要です")
        return []

    if not WARCIO_AVAILABLE:
        print("   ⚠️ warcio が必要です")
        return []

    # CDX Index検索
    records = search_common_crawl_index(
        query=query,
        index_name=index_name,
        limit=max_records * 3  # フィルタリングを考慮して多めに取得
    )

    if not records:
        print("   ⚠️ レコードが見つかりませんでした")
        return []

    texts = []

    # 各レコードからテキストを抽出
    for i, record in enumerate(records):
        if len(texts) >= max_records:
            break

        try:
            # レコードの情報を取得
            filename = record.get('filename')
            offset = int(record.get('offset', 0))
            length = int(record.get('length', 0))

            if not filename or offset is None or length is None:
                continue

            # 進捗表示
            if (i + 1) % 10 == 0:
                print(f"   📥 処理中: {i + 1}/{len(records)} ({len(texts)} テキスト取得)")

            # WARCレコード取得
            warc_data = fetch_warc_record(filename, offset, length)
            if not warc_data:
                continue

            # WARCレコードをパース
            try:
                # gzip解凍が必要な場合
                if filename.endswith('.gz'):
                    warc_data = gzip.decompress(warc_data)

                # WARCファイルをパース
                stream = BytesIO(warc_data)
                for warc_record in ArchiveIterator(stream):
                    if warc_record.rec_type == 'response':
                        # HTTPレスポンスからコンテンツを取得
                        content = warc_record.content_stream().read()

                        try:
                            # HTMLをデコード
                            html = content.decode('utf-8', errors='ignore')

                            # テキストを抽出
                            text = extract_text_from_html(html)

                            # 日本語チェック
                            if is_japanese_text(text) and len(text) >= min_text_length:
                                texts.append(text[:5000])  # 長さ制限
                                break

                        except Exception as e:
                            continue

            except Exception as e:
                continue

        except Exception as e:
            continue

    print(f"   ✅ Common Crawl: {len(texts)} サンプル取得")
    return texts


# テスト用
if __name__ == "__main__":
    print("=" * 70)
    print("Common Crawl データ取得テスト")
    print("=" * 70)

    # 最新のインデックスを取得
    latest_index = get_latest_index()
    print(f"最新Index: {latest_index}")

    # データ取得テスト
    texts = fetch_common_crawl_data(max_records=5, query="*.jp")

    print(f"\n取得テキスト数: {len(texts)}")

    if texts:
        print("\nサンプル:")
        for i, text in enumerate(texts[:3], 1):
            print(f"\n--- サンプル {i} ---")
            print(text[:200] + "..." if len(text) > 200 else text)
