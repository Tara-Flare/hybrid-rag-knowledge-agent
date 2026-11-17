"""本地知识库检索服务。"""

from __future__ import annotations

from pathlib import Path

from rag.embedding_client import EmbeddingClient
from rag.vector_store import (
    InMemoryVectorStore,
    SearchResult,
)

class KnowledgeRetriever:
    """加载本地向量索引并执行语义检索。"""

    def __init__(
        self,
        index_path: str = "data/rag_index.npz",
        chunks_path: str = "data/rag_chunks.json",
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
    ) -> None:
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)

        self.embedding_client = EmbeddingClient(
            model=model,
            dimensions=dimensions,
            batch_size=10,
        )

        self.vector_store = InMemoryVectorStore()

        self._loaded = False
        self._load_error: str | None = None

        # 尝试加载索引。
        #
        # 即使索引不存在，也不让整个 Agent 初始化失败；
        # 真正调用检索工具时再返回明确错误。
        try:
            self.load_index()

        except Exception as exc:
            self._load_error = (
                f"{type(exc).__name__}: {exc}"
            )

    def load_index(self) -> None:
        """从磁盘加载知识库索引。"""

        self.vector_store.load(
            index_path=self.index_path,
            chunks_path=self.chunks_path,
        )

        self._loaded = True
        self._load_error = None

    def reload_index(self) -> None:
        """重新加载知识库索引。"""

        self.vector_store.clear()
        self.load_index()

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.25,
    ) -> list[SearchResult]:
        """
        搜索与查询最相关的知识库文本块。

        Args:
            query:
                用户查询。
            top_k:
                最多返回多少条结果。
            min_score:
                最低余弦相似度阈值。

        Returns:
            满足阈值的检索结果。
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "知识库查询不能为空。"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0。"
            )

        if top_k > 10:
            raise ValueError(
                "top_k 不能超过 10。"
            )

        if not -1.0 <= min_score <= 1.0:
            raise ValueError(
                "min_score 必须位于 -1 到 1 之间。"
            )

        if not self._loaded:
            detail = (
                self._load_error
                or "未知加载错误"
            )

            raise RuntimeError(
                "知识库索引尚未成功加载。"
                "请先运行 build_vector_index.py。"
                f"详细信息：{detail}"
            )

        query_embedding = (
            self.embedding_client.embed_query(
                query
            )
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # 过滤相似度过低的候选结果。
        return [
            result
            for result in results
            if result.score >= min_score
        ]