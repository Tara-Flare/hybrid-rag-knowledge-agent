"""千问文本向量模型客户端。"""

from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv
from openai import OpenAI

class EmbeddingClient:
    """负责将文本转换成 Embedding 向量。"""

    def __init__(
        self,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        batch_size: int = 10,
    ) -> None:
        """
        初始化向量模型客户端。

        Args:
            model:
                使用的向量模型名称。
            dimensions:
                输出向量维度。
            batch_size:
                每次 API 请求最多处理多少条文本。
        """

        load_dotenv()

        api_key = os.getenv(
            "DASHSCOPE_API_KEY"
        )

        base_url = os.getenv(
            "DASHSCOPE_BASE_URL"
        )

        if not api_key:
            raise RuntimeError(
                "未找到 DASHSCOPE_API_KEY，"
                "请检查项目根目录下的 .env 文件。"
            )

        if not base_url:
            raise RuntimeError(
                "未找到 DASHSCOPE_BASE_URL，"
                "请把 llm_client.py 中正在使用的 "
                "base_url 写入 .env。"
            )

        if dimensions <= 0:
            raise ValueError(
                "向量维度必须大于 0。"
            )

        if batch_size <= 0:
            raise ValueError(
                "批大小必须大于 0。"
            )

        # text-embedding-v4 的批量输入上限为 10 条。
        if batch_size > 10:
            raise ValueError(
                "text-embedding-v4 的 batch_size "
                "不能超过 10。"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

    @staticmethod
    def _validate_text(
        text: str,
    ) -> str:
        """检查并清理单条输入文本。"""

        if not isinstance(text, str):
            raise TypeError(
                "待向量化的内容必须是字符串。"
            )

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError(
                "待向量化的文本不能为空。"
            )

        return cleaned_text

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        批量生成文本向量。

        Args:
            texts:
                需要向量化的文本序列。

        Returns:
            与输入顺序一致的向量列表。
        """

        if not texts:
            return []

        cleaned_texts = [
            self._validate_text(text)
            for text in texts
        ]

        all_embeddings: list[list[float]] = []

        # 按 batch_size 分批调用接口。
        for start in range(
            0,
            len(cleaned_texts),
            self.batch_size,
        ):
            batch = cleaned_texts[
                start : start + self.batch_size
            ]

            try:
                response = (
                    self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        dimensions=self.dimensions,
                        encoding_format="float",
                    )
                )

            except Exception as exc:
                raise RuntimeError(
                    "调用千问向量模型失败："
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            # 根据 index 排序，保证返回顺序和输入一致。
            ordered_items = sorted(
                response.data,
                key=lambda item: item.index,
            )

            batch_embeddings = [
                item.embedding
                for item in ordered_items
            ]

            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    "向量模型返回数量与输入数量不一致。"
                )

            all_embeddings.extend(
                batch_embeddings
            )

        return all_embeddings

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """生成单个查询文本的向量。"""

        return self.embed_texts(
            [query]
        )[0]

    def embed_documents(
        self,
        documents: Sequence[str],
    ) -> list[list[float]]:
        """生成一组知识库文档向量。"""

        return self.embed_texts(
            documents
        )