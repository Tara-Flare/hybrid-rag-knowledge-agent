from __future__ import annotations

from typing import Any

from rag.bm25_retriever import BM25Retriever
from rag.rrf_fusion import (
    RRFSearchResult,
    reciprocal_rank_fusion,
)

class HybridRetriever:
    """
    Dense 向量召回 + BM25 关键词召回 + RRF 排名融合。

    Dense 负责语义相关性；
    BM25 负责关键词和代码标识符匹配；
    RRF 负责融合两路排名。
    """

    def __init__(
        self,
        dense_retriever: Any | None = None,
        bm25_retriever: BM25Retriever | None = None,
        dense_top_k: int = 5,
        bm25_top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        if dense_top_k <= 0:
            raise ValueError(
                "dense_top_k 必须大于 0"
            )

        if bm25_top_k <= 0:
            raise ValueError(
                "bm25_top_k 必须大于 0"
            )

        if rrf_k < 0:
            raise ValueError(
                "rrf_k 不能小于 0"
            )

        # 没有从外部传入时，使用项目现有的
        # KnowledgeRetriever。
        if dense_retriever is None:
            from rag.knowledge_retriever import (
                KnowledgeRetriever,
            )

            self.dense_retriever = (
                KnowledgeRetriever()
            )
        else:
            self.dense_retriever = (
                dense_retriever
            )

        if bm25_retriever is None:
            self.bm25_retriever = (
                BM25Retriever(
                    chunks_path=(
                        "data/rag_chunks.json"
                    )
                )
            )
        else:
            self.bm25_retriever = (
                bm25_retriever
            )

        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k

    def _dense_search(
        self,
        query: str,
    ) -> list[Any]:
        """
        调用 Dense 检索器。

        同时兼容关键字参数和位置参数两种写法。
        """
        try:
            results = (
                self.dense_retriever.search(
                    query=query,
                    top_k=self.dense_top_k,
                )
            )
        except TypeError:
            results = (
                self.dense_retriever.search(
                    query,
                    self.dense_top_k,
                )
            )

        return list(results)

    def _bm25_search(
        self,
        query: str,
    ) -> list[Any]:
        """
        调用 BM25 检索器，并过滤零分结果。
        """
        results = (
            self.bm25_retriever.search(
                query=query,
                top_k=self.bm25_top_k,
            )
        )

        # BM25 分数为 0，通常意味着查询词
        # 与该 Chunk 没有有效匹配。
        positive_results = [
            result
            for result in results
            if result.score > 0
        ]

        return positive_results

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RRFSearchResult]:
        """
        分别执行 Dense 与 BM25 召回，
        再使用 RRF 融合排名。
        """
        if not isinstance(query, str):
            raise TypeError(
                "query 必须是字符串"
            )

        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        dense_results = self._dense_search(
            query=query
        )

        bm25_results = self._bm25_search(
            query=query
        )

        fused_results = (
            reciprocal_rank_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
                top_k=top_k,
                rrf_k=self.rrf_k,
            )
        )

        return fused_results