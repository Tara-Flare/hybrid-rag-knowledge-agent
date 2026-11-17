"""简单的本地向量存储与检索模块。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rag.text_splitter import DocumentChunk

@dataclass(slots=True, frozen=True)
class SearchResult:
    """一条向量检索结果。"""

    chunk: DocumentChunk
    score: float
    rank: int

class InMemoryVectorStore:
    """
    基于 NumPy 的简单向量库。

    支持：
    1. 添加文本块和向量
    2. 余弦相似度检索
    3. 将向量和文本块保存到磁盘
    4. 从磁盘恢复索引
    """

    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []

        # 每一行对应一个 Chunk 的向量
        self.embedding_matrix: np.ndarray | None = None

    @staticmethod
    def _normalize_vectors(
        vectors: np.ndarray,
    ) -> np.ndarray:
        """把二维向量矩阵逐行归一化。"""

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        # 防止出现除以零
        norms = np.where(
            norms == 0,
            1.0,
            norms,
        )

        return vectors / norms

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """把文本块及其向量加入向量库。"""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "文本块数量与向量数量不一致。"
            )

        if not chunks:
            return

        new_matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if new_matrix.ndim != 2:
            raise ValueError(
                "embeddings 必须是二维向量列表。"
            )

        normalized_matrix = (
            self._normalize_vectors(
                new_matrix
            )
        )

        if self.embedding_matrix is None:
            self.embedding_matrix = normalized_matrix

        else:
            if (
                self.embedding_matrix.shape[1]
                != normalized_matrix.shape[1]
            ):
                raise ValueError(
                    "新旧向量维度不一致。"
                )

            self.embedding_matrix = np.vstack(
                [
                    self.embedding_matrix,
                    normalized_matrix,
                ]
            )

        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """根据查询向量检索相关文本块。"""

        if top_k <= 0:
            return []

        if (
            self.embedding_matrix is None
            or not self.chunks
        ):
            return []

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_vector.ndim != 1:
            raise ValueError(
                "查询向量必须是一维向量。"
            )

        if (
            query_vector.shape[0]
            != self.embedding_matrix.shape[1]
        ):
            raise ValueError(
                "查询向量与文档向量维度不一致。"
            )

        query_norm = np.linalg.norm(
            query_vector
        )

        if query_norm == 0:
            raise ValueError(
                "查询向量不能是零向量。"
            )

        normalized_query = (
            query_vector / query_norm
        )

        # 文档向量已归一化，
        # 因此点积就是余弦相似度。
        scores = (
            self.embedding_matrix
            @ normalized_query
        )

        result_count = min(
            top_k,
            len(self.chunks),
        )

        top_indices = np.argsort(
            scores
        )[::-1][:result_count]

        results: list[SearchResult] = []

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):
            index = int(index)

            results.append(
                SearchResult(
                    chunk=self.chunks[index],
                    score=float(scores[index]),
                    rank=rank,
                )
            )

        return results

    def save(
        self,
        index_path: str | Path,
        chunks_path: str | Path,
    ) -> None:
        """将向量矩阵和文本块保存到磁盘。"""

        if self.embedding_matrix is None:
            raise RuntimeError(
                "当前向量库为空，无法保存。"
            )

        if not self.chunks:
            raise RuntimeError(
                "当前没有文本块，无法保存。"
            )

        if len(self.chunks) != self.embedding_matrix.shape[0]:
            raise RuntimeError(
                "文本块数量与向量矩阵行数不一致。"
            )

        index_file = Path(index_path)
        chunks_file = Path(chunks_path)

        index_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        chunks_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 使用压缩的 NumPy 格式保存向量矩阵
        np.savez_compressed(
            index_file,
            embeddings=self.embedding_matrix,
        )

        chunk_records: list[dict[str, Any]] = []

        for chunk in self.chunks:
            chunk_records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "file_name": chunk.file_name,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                }
            )

        with chunks_file.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                chunk_records,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(
        self,
        index_path: str | Path,
        chunks_path: str | Path,
    ) -> None:
        """从磁盘加载向量矩阵和文本块。"""

        index_file = Path(index_path)
        chunks_file = Path(chunks_path)

        if not index_file.exists():
            raise FileNotFoundError(
                f"向量索引文件不存在：{index_file}"
            )

        if not chunks_file.exists():
            raise FileNotFoundError(
                f"文本块文件不存在：{chunks_file}"
            )

        try:
            index_data = np.load(
                index_file,
                allow_pickle=False,
            )

            embedding_matrix = index_data[
                "embeddings"
            ]

        except Exception as exc:
            raise RuntimeError(
                "读取向量索引失败："
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if embedding_matrix.ndim != 2:
            raise RuntimeError(
                "向量索引必须是二维矩阵。"
            )

        try:
            with chunks_file.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                chunk_records = json.load(file)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "文本块文件不是合法 JSON。"
            ) from exc

        if not isinstance(chunk_records, list):
            raise RuntimeError(
                "文本块文件的根数据必须是列表。"
            )

        loaded_chunks: list[DocumentChunk] = []

        for record in chunk_records:
            if not isinstance(record, dict):
                raise RuntimeError(
                    "文本块记录格式不正确。"
                )

            loaded_chunks.append(
                DocumentChunk(
                    chunk_id=str(
                        record["chunk_id"]
                    ),
                    content=str(
                        record["content"]
                    ),
                    source=str(
                        record["source"]
                    ),
                    file_name=str(
                        record["file_name"]
                    ),
                    chunk_index=int(
                        record["chunk_index"]
                    ),
                    metadata=dict(
                        record.get(
                            "metadata",
                            {},
                        )
                    ),
                )
            )

        if (
            len(loaded_chunks)
            != embedding_matrix.shape[0]
        ):
            raise RuntimeError(
                "加载的文本块数量与向量数量不一致。"
            )

        # 即使保存前已经归一化，
        # 加载时再次保证向量为单位向量。
        self.embedding_matrix = (
            self._normalize_vectors(
                np.asarray(
                    embedding_matrix,
                    dtype=np.float32,
                )
            )
        )

        self.chunks = loaded_chunks

    def clear(self) -> None:
        """清空当前内存中的向量索引。"""

        self.chunks.clear()
        self.embedding_matrix = None