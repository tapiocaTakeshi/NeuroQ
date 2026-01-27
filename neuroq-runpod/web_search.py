#!/usr/bin/env python3
"""
Web検索モジュール

DuckDuckGo Search APIを使用してウェブ検索を実行し、
RAGコンテキスト用の情報を取得します。
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """検索結果を表すデータクラス"""
    title: str
    url: str
    snippet: str

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet
        }


class WebSearchClient:
    """
    Web検索クライアント

    DuckDuckGo Search APIを使用してウェブ検索を実行します。
    """

    def __init__(self, max_results: int = 5, region: str = "jp-jp"):
        """
        Args:
            max_results: 最大検索結果数
            region: 検索リージョン（jp-jp: 日本）
        """
        self.max_results = max_results
        self.region = region
        self._ddgs = None

    def _get_ddgs(self):
        """DuckDuckGo Search インスタンスをLazy初期化"""
        if self._ddgs is None:
            try:
                from duckduckgo_search import DDGS
                self._ddgs = DDGS()
            except ImportError:
                raise ImportError(
                    "duckduckgo-search パッケージが必要です。"
                    "pip install duckduckgo-search でインストールしてください。"
                )
        return self._ddgs

    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """
        ウェブ検索を実行

        Args:
            query: 検索クエリ
            max_results: 最大結果数（Noneの場合はデフォルト値を使用）

        Returns:
            SearchResult のリスト
        """
        if not query or not query.strip():
            return []

        max_results = max_results or self.max_results

        try:
            ddgs = self._get_ddgs()
            results = []

            # DuckDuckGo テキスト検索を実行
            search_results = ddgs.text(
                query,
                region=self.region,
                max_results=max_results
            )

            for item in search_results:
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", item.get("link", "")),
                    snippet=item.get("body", item.get("snippet", ""))
                )
                results.append(result)

            return results

        except Exception as e:
            print(f"⚠️ Web検索エラー: {e}")
            return []

    def search_news(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """
        ニュース検索を実行

        Args:
            query: 検索クエリ
            max_results: 最大結果数

        Returns:
            SearchResult のリスト
        """
        if not query or not query.strip():
            return []

        max_results = max_results or self.max_results

        try:
            ddgs = self._get_ddgs()
            results = []

            # DuckDuckGo ニュース検索を実行
            news_results = ddgs.news(
                query,
                region=self.region,
                max_results=max_results
            )

            for item in news_results:
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", item.get("link", "")),
                    snippet=item.get("body", item.get("excerpt", ""))
                )
                results.append(result)

            return results

        except Exception as e:
            print(f"⚠️ ニュース検索エラー: {e}")
            return []


def extract_search_query(prompt: str, is_japanese: bool = True) -> str:
    """
    ユーザープロンプトから検索クエリを抽出

    Args:
        prompt: ユーザーの入力テキスト
        is_japanese: 日本語かどうか

    Returns:
        検索用に最適化されたクエリ
    """
    # 質問形式を除去
    query = prompt.strip()

    # 日本語の場合
    if is_japanese:
        # 質問形式のパターンを除去
        patterns = [
            r"^(教えて|おしえて)[ください]*[。？]?",
            r"^(何|なに|なん)(ですか|でしょうか|について)[？。]?",
            r"(について教えて|について|とは)[ください]*[。？]?$",
            r"(ですか|でしょうか|かな)[？。]?$",
            r"^(どう|どんな|どの|いつ|どこ|誰|だれ)(が|は|を|に|で)?",
        ]
        for pattern in patterns:
            query = re.sub(pattern, "", query)
    else:
        # 英語の場合
        patterns = [
            r"^(what|who|where|when|why|how)\s+(is|are|was|were|do|does|did|can|could|would|should)\s+",
            r"^(tell me about|explain|describe)\s+",
            r"^(can you|could you|would you|please)\s+",
            r"\?$",
        ]
        for pattern in patterns:
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)

    return query.strip()


def format_search_results_for_context(
    results: List[SearchResult],
    max_chars: int = 2000
) -> str:
    """
    検索結果をRAGコンテキスト用にフォーマット

    Args:
        results: 検索結果のリスト
        max_chars: 最大文字数

    Returns:
        フォーマットされたコンテキスト文字列
    """
    if not results:
        return ""

    context_parts = []
    current_chars = 0

    for i, result in enumerate(results, 1):
        # 各結果をフォーマット
        entry = f"[{i}] {result.title}\n{result.snippet}\n(Source: {result.url})\n"

        if current_chars + len(entry) > max_chars:
            break

        context_parts.append(entry)
        current_chars += len(entry)

    if not context_parts:
        return ""

    return "=== Web Search Results ===\n" + "\n".join(context_parts)


# テスト用
if __name__ == "__main__":
    client = WebSearchClient(max_results=3)

    print("🔍 テスト検索: Python programming")
    results = client.search("Python programming")
    for r in results:
        print(f"  - {r.title}")
        print(f"    {r.snippet[:100]}...")
        print(f"    URL: {r.url}")
        print()

    print("\n🔍 テスト検索: 最新ニュース AI")
    news = client.search_news("AI 人工知能")
    for r in news:
        print(f"  - {r.title}")
        print(f"    {r.snippet[:100]}...")
        print()
