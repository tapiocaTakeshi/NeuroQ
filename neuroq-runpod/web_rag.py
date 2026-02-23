#!/usr/bin/env python3
"""
Web RAG (Retrieval-Augmented Generation) モジュール

ウェブ検索結果を使用してコンテキストを拡張し、
より正確で最新の情報に基づいた回答を生成します。
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class RAGContext:
    """RAGコンテキストを表すデータクラス"""
    query: str
    search_results: List[Dict]
    formatted_context: str
    sources: List[str]


class WebRAGProcessor:
    """
    Web RAG プロセッサ

    ウェブ検索とコンテキスト拡張を統合的に処理します。
    """

    def __init__(
        self,
        max_search_results: int = 5,
        max_context_chars: int = 1500,
        search_region: str = "jp-jp"
    ):
        """
        Args:
            max_search_results: 最大検索結果数
            max_context_chars: コンテキストの最大文字数
            search_region: 検索リージョン
        """
        self.max_search_results = max_search_results
        self.max_context_chars = max_context_chars
        self.search_region = search_region
        self._search_client = None

    def _get_search_client(self):
        """検索クライアントをLazy初期化"""
        if self._search_client is None:
            from web_search import WebSearchClient
            self._search_client = WebSearchClient(
                max_results=self.max_search_results,
                region=self.search_region
            )
        return self._search_client

    def detect_language(self, text: str) -> str:
        """
        テキストの言語を検出

        Args:
            text: 入力テキスト

        Returns:
            'ja' (日本語) or 'en' (英語)
        """
        # 日本語文字（ひらがな、カタカナ、漢字）が含まれているかチェック
        japanese_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]')
        if japanese_pattern.search(text):
            return 'ja'
        return 'en'

    def extract_search_query(self, prompt: str) -> str:
        """
        プロンプトから検索クエリを抽出

        Args:
            prompt: ユーザーの入力

        Returns:
            最適化された検索クエリ
        """
        query = prompt.strip()
        lang = self.detect_language(query)

        if lang == 'ja':
            # 日本語の質問形式を除去
            patterns = [
                r"(を?教えて|おしえて)(ください|くれ)?[。？]?$",
                r"(について|とは|って何|ってなに)[？。]?$",
                r"(ですか|でしょうか|かな|だろう)[？。]?$",
                r"^(今日|最近|最新)の",
            ]
            for pattern in patterns:
                query = re.sub(pattern, "", query)
        else:
            # 英語の質問形式を除去
            patterns = [
                r"^(what|who|where|when|why|how)\s+(is|are|was|were|do|does|did)\s+",
                r"^(tell me about|explain|describe|can you tell me)\s+",
                r"^(please|can you|could you)\s+",
                r"\?$",
            ]
            for pattern in patterns:
                query = re.sub(pattern, "", query, flags=re.IGNORECASE)

        return query.strip()

    def should_search(self, prompt: str) -> bool:
        """
        検索が必要かどうかを判定

        Args:
            prompt: ユーザーの入力

        Returns:
            検索が必要な場合True
        """
        prompt_lower = prompt.lower()
        lang = self.detect_language(prompt)

        # 検索が必要なキーワード（日本語）
        ja_search_triggers = [
            "最新", "今日", "現在", "ニュース", "速報",
            "天気", "株価", "為替", "調べて", "検索",
            "情報", "について", "とは", "って何", "詳しく",
            "どうなって", "何が", "どこで", "いつ"
        ]

        # 検索が必要なキーワード（英語）
        en_search_triggers = [
            "latest", "recent", "news", "current", "today",
            "weather", "price", "what is", "who is", "where is",
            "when", "how to", "search", "find", "look up",
            "information about", "tell me about"
        ]

        if lang == 'ja':
            return any(trigger in prompt for trigger in ja_search_triggers)
        else:
            return any(trigger in prompt_lower for trigger in en_search_triggers)

    def search_web(self, query: str, use_news: bool = False) -> List[Dict]:
        """
        ウェブ検索を実行

        Args:
            query: 検索クエリ
            use_news: ニュース検索を使用するか

        Returns:
            検索結果のリスト
        """
        try:
            client = self._get_search_client()

            if use_news:
                results = client.search_news(query, self.max_search_results)
            else:
                results = client.search(query, self.max_search_results)

            return [r.to_dict() for r in results]

        except Exception as e:
            print(f"⚠️ Web検索エラー: {e}")
            return []

    def format_context(
        self,
        search_results: List[Dict],
        lang: str = 'ja'
    ) -> Tuple[str, List[str]]:
        """
        検索結果をコンテキスト文字列にフォーマット

        Args:
            search_results: 検索結果のリスト
            lang: 言語コード

        Returns:
            (フォーマットされたコンテキスト, ソースURLリスト)
        """
        if not search_results:
            return "", []

        sources = []
        context_parts = []
        current_chars = 0

        # ヘッダー
        if lang == 'ja':
            header = "【参考情報（Web検索結果）】\n"
        else:
            header = "【Reference Information (Web Search Results)】\n"

        current_chars += len(header)

        for i, result in enumerate(search_results, 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = result.get("url", "")

            # スニペットを短縮
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."

            entry = f"[{i}] {title}\n{snippet}\n\n"

            if current_chars + len(entry) > self.max_context_chars:
                break

            context_parts.append(entry)
            current_chars += len(entry)
            if url:
                sources.append(url)

        if not context_parts:
            return "", []

        formatted = header + "".join(context_parts)
        return formatted, sources

    def process(
        self,
        prompt: str,
        force_search: bool = False,
        use_news: bool = False
    ) -> Optional[RAGContext]:
        """
        RAG処理を実行

        Args:
            prompt: ユーザーの入力
            force_search: 検索を強制するか
            use_news: ニュース検索を使用するか

        Returns:
            RAGContext または None（検索しなかった場合）
        """
        # 検索が必要か判定
        if not force_search and not self.should_search(prompt):
            return None

        # 検索クエリを抽出
        query = self.extract_search_query(prompt)
        if not query:
            query = prompt[:100]  # フォールバック

        # ニュースキーワードのチェック
        news_keywords = ["ニュース", "速報", "news", "latest", "最新", "今日"]
        if any(kw in prompt.lower() for kw in news_keywords):
            use_news = True

        # 検索実行
        print(f"🔍 Web検索中: {query[:50]}...")
        search_results = self.search_web(query, use_news=use_news)

        if not search_results:
            print("⚠️ 検索結果がありません")
            return None

        print(f"✅ {len(search_results)}件の検索結果を取得")

        # コンテキストをフォーマット
        lang = self.detect_language(prompt)
        formatted_context, sources = self.format_context(search_results, lang)

        return RAGContext(
            query=query,
            search_results=search_results,
            formatted_context=formatted_context,
            sources=sources
        )

    def augment_system_prompt(
        self,
        system_prompt: str,
        rag_context: RAGContext,
        lang: str = 'ja'
    ) -> str:
        """
        システムプロンプトにRAGコンテキストを追加

        Args:
            system_prompt: 元のシステムプロンプト
            rag_context: RAGコンテキスト
            lang: 言語コード

        Returns:
            拡張されたシステムプロンプト
        """
        if lang == 'ja':
            instruction = (
                "\n\n以下の参考情報を活用して回答してください。"
                "情報が古い可能性がある場合はその旨を伝えてください。\n"
            )
        else:
            instruction = (
                "\n\nUse the following reference information to answer. "
                "If the information might be outdated, please mention that.\n"
            )

        augmented = (
            system_prompt +
            instruction +
            rag_context.formatted_context
        )

        return augmented


# RAG用のシステムプロンプトテンプレート
RAG_SYSTEM_PROMPT_JA = """あなたはAIアシスタントです。
ユーザーの質問に対して、提供された参考情報を活用して回答してください。

回答のルール:
1. 参考情報に基づいて正確に回答する
2. 情報源を明示する（例: 「検索結果によると...」）
3. 情報が不確かな場合は、その旨を伝える
4. 簡潔で分かりやすい回答を心がける
5. 常に日本語で回答する"""

RAG_SYSTEM_PROMPT_EN = """あなたはAIアシスタントです。
提供された参考情報を活用して、ユーザーの質問に回答してください。

ルール:
1. 参考情報に基づいて正確に回答する
2. 情報源を明示する（例: 「検索結果によると...」）
3. 情報が不確かな場合は、その旨を伝える
4. 簡潔で分かりやすい回答を心がける
5. 常に日本語で回答する"""


# テスト用
if __name__ == "__main__":
    processor = WebRAGProcessor(max_search_results=3)

    # テストプロンプト
    test_prompts = [
        "今日の天気を教えて",
        "Pythonについて教えて",
        "最新のAIニュースは？",
        "What is machine learning?",
    ]

    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"📝 Prompt: {prompt}")
        print(f"   Should search: {processor.should_search(prompt)}")

        result = processor.process(prompt, force_search=True)
        if result:
            print(f"   Query: {result.query}")
            print(f"   Results: {len(result.search_results)}")
            print(f"   Sources: {result.sources}")
            print(f"   Context preview: {result.formatted_context[:200]}...")
