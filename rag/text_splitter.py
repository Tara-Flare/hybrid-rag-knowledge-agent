"""RAG 文档切块模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.document_loader import SourceDocument

@dataclass(slots=True, frozen=True)
class DocumentChunk:
    """知识库中的一个文本块。"""

    chunk_id: str
    content: str
    source: str
    file_name: str
    chunk_index: int
    metadata: dict[str, Any]

class CharacterTextSplitter:
    """
    基于字符数量的滑动窗口切块器。

    这是第一版基线实现，后续会与标题切分、
    递归切分和代码结构切分进行对比。
    """

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size 必须大于 0。"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap 不能小于 0。"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap 必须小于 chunk_size。"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """统一换行符并去除首尾空白。"""

        return (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

    def split_document(
        self,
        document: SourceDocument,
    ) -> list[DocumentChunk]:
        """将一份文档切成多个文本块。"""

        text = self._normalize_text(
            document.content
        )

        if not text:
            return []

        # 顶层文档保持旧 ID 格式。
        # 嵌套文档包含相对目录，避免同名文件冲突。
        source_path = Path(document.source)

        chunk_key = (
            source_path.as_posix()
            if (
                not source_path.is_absolute()
                and len(source_path.parts) > 1
            )
            else document.file_name
        )

        chunks: list[DocumentChunk] = []

        step = (
            self.chunk_size
            - self.chunk_overlap
        )

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=(
                            f"{chunk_key}::chunk-{chunk_index}"
                        ),
                        content=chunk_text,
                        source=document.source,
                        file_name=document.file_name,
                        chunk_index=chunk_index,
                        metadata={
                            **document.metadata,
                            "start_char": start,
                            "end_char": end,
                        },
                    )
                )

                chunk_index += 1

            if end >= len(text):
                break

            start += step

        return chunks

    def split_documents(
        self,
        documents: list[SourceDocument],
    ) -> list[DocumentChunk]:
        """批量切分多份文档。"""

        all_chunks: list[DocumentChunk] = []

        for document in documents:
            chunks = self.split_document(
                document
            )

            all_chunks.extend(chunks)

        return all_chunks