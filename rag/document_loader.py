"""本地知识库文档加载模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(slots=True, frozen=True)
class SourceDocument:
    """一份从本地文件读取的原始文档。"""

    content: str
    source: str
    file_name: str
    file_type: str
    metadata: dict[str, Any]

class LocalDocumentLoader:
    """加载本地 TXT 和 Markdown 文档。"""

    SUPPORTED_SUFFIXES = {
        ".txt",
        ".md",
        ".markdown",
    }

    @staticmethod
    def _read_text_file(
        file_path: Path,
    ) -> str:
        """
        尝试使用常见编码读取文本文件。

        UTF-8 读取失败时，再尝试 GB18030。
        """

        encodings = [
            "utf-8",
            "utf-8-sig",
            "gb18030",
        ]

        last_error: UnicodeDecodeError | None = None

        for encoding in encodings:
            try:
                return file_path.read_text(
                    encoding=encoding
                )

            except UnicodeDecodeError as exc:
                last_error = exc

        raise RuntimeError(
            f"无法识别文件编码：{file_path}"
        ) from last_error

    def load_file(
        self,
        file_path: str | Path,
    ) -> SourceDocument:
        """加载一个文本文件。"""

        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"文件不存在：{path}"
            )

        if not path.is_file():
            raise ValueError(
                f"该路径不是文件：{path}"
            )

        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(
                f"暂不支持该文件类型：{suffix}"
            )

        content = self._read_text_file(path)
        content = content.strip()

        if not content:
            raise ValueError(
                f"文件内容为空：{path}"
            )

        return SourceDocument(
            content=content,
            source=str(path),
            file_name=path.name,
            file_type=suffix,
            metadata={
                "size_bytes": path.stat().st_size,
            },
        )

    def load_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
    ) -> list[SourceDocument]:
        """加载目录中的所有受支持文档。"""

        directory = Path(
            directory_path
        ).resolve()

        if not directory.exists():
            raise FileNotFoundError(
                f"目录不存在：{directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"该路径不是目录：{directory}"
            )

        pattern = "**/*" if recursive else "*"

        file_paths = sorted(
            path
            for path in directory.glob(pattern)
            if (
                path.is_file()
                and path.suffix.lower()
                in self.SUPPORTED_SUFFIXES
            )
        )

        documents: list[SourceDocument] = []

        for file_path in file_paths:
            try:
                document = self.load_file(
                    file_path
                )

            except (ValueError, RuntimeError) as exc:
                print(
                    f"[跳过文档] {file_path}：{exc}"
                )
                continue

            documents.append(document)

        return documents