from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

# 匹配三类内容：
# 1. 英文标识符：MemoryNotFoundError、max_steps、agent.py
# 2. 数字：1024、0.75
# 3. 连续中文
TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.-]*"
    r"|\d+(?:\.\d+)?"
    r"|[\u4e00-\u9fff]+"
)

CHINESE_PATTERN = re.compile(r"^[\u4e00-\u9fff]+$")

def tokenize(text: str) -> list[str]:
    """
    将文本切分成 BM25 使用的词列表。

    中文使用 jieba 分词；
    英文错误名、函数名、文件名等完整保留。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")

    tokens: list[str] = []

    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0).lower()

        if CHINESE_PATTERN.fullmatch(token):
            for word in jieba.lcut(token):
                word = word.strip()

                if word:
                    tokens.append(word)
        else:
            tokens.append(token)

    return tokens

@dataclass(slots=True)
class BM25SearchResult:
    """
    BM25 检索结果。
    """

    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any]

class BM25Retriever:
    """
    从 rag_chunks.json 加载 Chunk，
    并建立本地 BM25 关键词索引。
    """

    def __init__(
        self,
        chunks_path: str | Path = "data/rag_chunks.json",
    ) -> None:
        self.chunks_path = Path(chunks_path)

        self._chunks: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

        self._load_chunks()
        self._build_index()

    def _load_chunks(self) -> None:
        """
        从 JSON 文件加载知识库 Chunk。

        同时兼容 text、content、page_content 等常见字段名。
        """
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"找不到 Chunk 文件：{self.chunks_path}"
            )

        try:
            payload = json.loads(
                self.chunks_path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Chunk 文件不是有效 JSON：{self.chunks_path}"
            ) from exc

        if isinstance(payload, list):
            raw_chunks = payload

        elif isinstance(payload, dict):
            if isinstance(payload.get("chunks"), list):
                raw_chunks = payload["chunks"]
            else:
                raw_chunks = list(payload.values())

        else:
            raise ValueError(
                "rag_chunks.json 的顶层必须是列表或字典"
            )

        parsed_chunks: list[dict[str, Any]] = []

        for index, item in enumerate(raw_chunks):
            if not isinstance(item, dict):
                continue

            text = (
                item.get("text")
                or item.get("content")
                or item.get("page_content")
                or ""
            )

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

            if not text:
                continue

            raw_metadata = item.get("metadata", {})

            if isinstance(raw_metadata, dict):
                metadata = dict(raw_metadata)
            else:
                metadata = {}

            source = str(
                item.get("source")
                or metadata.get("source")
                or "unknown"
            )

            chunk_id = str(
                item.get("chunk_id")
                or metadata.get("chunk_id")
                or metadata.get("id")
                or f"{Path(source).name}::chunk-{index}"
            )

            parsed_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": source,
                    "metadata": metadata,
                }
            )

        if not parsed_chunks:
            raise ValueError(
                f"{self.chunks_path} 中没有可用的 Chunk"
            )

        self._chunks = parsed_chunks

    def _build_index(self) -> None:
        """
        对所有 Chunk 分词，并建立 BM25 索引。
        """
        self._tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self._chunks
        ]

        if not any(self._tokenized_corpus):
            raise ValueError(
                "所有 Chunk 分词结果都为空，无法建立 BM25 索引"
            )

        self._bm25 = BM25Okapi(
            self._tokenized_corpus
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[BM25SearchResult]:
        """
        使用 BM25 检索与查询关键词最相关的 Top-K Chunk。
        """
        if not isinstance(query, str):
            raise TypeError("query 必须是字符串")

        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if self._bm25 is None:
            raise RuntimeError("BM25 索引尚未建立")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = np.asarray(
            self._bm25.get_scores(query_tokens),
            dtype=np.float64,
        )

        result_count = min(
            top_k,
            len(self._chunks),
        )

        ranked_indices = np.argsort(scores)[::-1][
            :result_count
        ]

        results: list[BM25SearchResult] = []

        for index in ranked_indices:
            chunk = self._chunks[int(index)]

            results.append(
                BM25SearchResult(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    source=chunk["source"],
                    score=float(scores[index]),
                    metadata=chunk["metadata"],
                )
            )

        return results

    @property
    def chunk_count(self) -> int:
        """
        返回 BM25 索引中的 Chunk 数量。
        """
        return len(self._chunks)