"""安全的本地文件读取工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# 当前项目根目录。
# file_tools.py位于 tools/ 下，
# parents[0]是tools，parents[1]是agent_learning。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 单次最多列出的目录项数量。
_MAX_DIRECTORY_ITEMS = 100

# 单次最多读取的字符数，避免工具结果过长。
_DEFAULT_MAX_CHARS = 12_000
_MAX_ALLOWED_CHARS = 20_000

# 当前阶段允许读取的文本文件类型。
_ALLOWED_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".csv",
    ".log",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
}


def _is_inside_project(path: Path) -> bool:
    """
    判断规范化后的路径是否位于项目根目录内部。

    Args:
        path: 已经规范化的绝对路径。

    Returns:
        位于项目内部返回True，否则返回False。
    """
    try:
        path.relative_to(_PROJECT_ROOT)
    except ValueError:
        return False

    return True


def _resolve_safe_path(raw_path: str) -> Path:
    """
    将用户提供的路径转换成安全的绝对路径。

    相对路径会相对于项目根目录解析。
    绝对路径必须仍然位于项目根目录内部。

    Args:
        raw_path: 用户或模型提供的原始路径。

    Returns:
        校验通过后的绝对路径。

    Raises:
        ValueError: 路径为空或位于允许目录之外。
    """
    cleaned_path = raw_path.strip()

    if not cleaned_path:
        cleaned_path = "."

    candidate = Path(cleaned_path).expanduser()

    # 相对路径基于项目根目录进行拼接。
    if not candidate.is_absolute():
        candidate = _PROJECT_ROOT / candidate

    # strict=False表示路径暂时不存在时也可以完成规范化。
    resolved_path = candidate.resolve(strict=False)

    if not _is_inside_project(resolved_path):
        raise ValueError(
            "拒绝访问项目目录之外的路径。"
            f"当前允许访问的根目录是：{_PROJECT_ROOT}"
        )

    return resolved_path


def _format_size(size_bytes: int) -> str:
    """将字节数量转换为容易阅读的形式。"""
    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"

    return f"{size_bytes / (1024 * 1024):.1f} MB"


def list_directory(
    path: str = ".",
    max_items: int = 50,
) -> str:
    """
    列出项目目录内部指定文件夹的内容。

    Args:
        path: 相对于项目根目录的路径，默认是项目根目录。
        max_items: 最多返回多少个目录项。

    Returns:
        格式化后的目录内容。
    """
    safe_path = _resolve_safe_path(path)

    if not safe_path.exists():
        raise ValueError(f"目录不存在：{safe_path}")

    if not safe_path.is_dir():
        raise ValueError(f"给定路径不是目录：{safe_path}")

    if max_items <= 0:
        raise ValueError("max_items必须大于0。")

    max_items = min(max_items, _MAX_DIRECTORY_ITEMS)

    try:
        items = sorted(
            safe_path.iterdir(),
            key=lambda item: (
                not item.is_dir(),
                item.name.lower(),
            ),
        )
    except PermissionError as exc:
        raise ValueError(
            f"没有权限访问目录：{safe_path}"
        ) from exc

    relative_path = safe_path.relative_to(_PROJECT_ROOT)

    if str(relative_path) == ".":
        display_path = "项目根目录"
    else:
        display_path = str(relative_path)

    lines = [
        f"目录：{display_path}",
        f"绝对路径：{safe_path}",
        f"目录项总数：{len(items)}",
        "",
    ]

    displayed_items = items[:max_items]

    for index, item in enumerate(displayed_items, start=1):
        if item.is_dir():
            item_type = "[目录]"
            size_text = ""
        else:
            item_type = "[文件]"

            try:
                size_text = f"（{_format_size(item.stat().st_size)}）"
            except OSError:
                size_text = "（大小未知）"

        lines.append(
            f"{index}. {item_type} {item.name}{size_text}"
        )

    if len(items) > max_items:
        omitted_count = len(items) - max_items
        lines.append("")
        lines.append(
            f"其余{omitted_count}个目录项未显示。"
        )

    return "\n".join(lines)


def _read_text_with_fallback(file_path: Path) -> tuple[str, str]:
    """
    尝试使用常见编码读取文本文件。

    Returns:
        二元组：(文件内容, 实际使用的编码)。
    """
    encodings = (
        "utf-8",
        "utf-8-sig",
        "gb18030",
    )

    for encoding in encodings:
        try:
            content = file_path.read_text(
                encoding=encoding
            )
        except UnicodeDecodeError:
            continue
        except PermissionError as exc:
            raise ValueError(
                f"没有权限读取文件：{file_path}"
            ) from exc
        else:
            return content, encoding

    raise ValueError(
        "无法识别该文件的文本编码，"
        "目前尝试了UTF-8和GB18030。"
    )


def read_text_file(
    path: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """
    读取项目目录内部的文本文件。

    Args:
        path: 相对于项目根目录的文本文件路径。
        max_chars: 最多返回的字符数量。

    Returns:
        文件基本信息和文本内容。
    """
    safe_path = _resolve_safe_path(path)

    if not safe_path.exists():
        raise ValueError(f"文件不存在：{safe_path}")

    if not safe_path.is_file():
        raise ValueError(f"给定路径不是文件：{safe_path}")

    suffix = safe_path.suffix.lower()

    if suffix not in _ALLOWED_TEXT_SUFFIXES:
        allowed_types = "、".join(
            sorted(_ALLOWED_TEXT_SUFFIXES)
        )
        raise ValueError(
            f"暂不支持读取{suffix or '无扩展名'}文件。"
            f"当前支持：{allowed_types}"
        )

    if max_chars <= 0:
        raise ValueError("max_chars必须大于0。")

    max_chars = min(max_chars, _MAX_ALLOWED_CHARS)

    content, encoding = _read_text_with_fallback(
        safe_path
    )

    original_length = len(content)
    was_truncated = original_length > max_chars

    if was_truncated:
        returned_content = content[:max_chars]
    else:
        returned_content = content

    relative_path = safe_path.relative_to(
        _PROJECT_ROOT
    )

    result_lines = [
        f"文件：{relative_path}",
        f"编码：{encoding}",
        f"原始字符数：{original_length}",
        f"本次返回字符数：{len(returned_content)}",
        f"是否截断：{'是' if was_truncated else '否'}",
        "",
        "===== 文件内容开始 =====",
        returned_content,
        "===== 文件内容结束 =====",
    ]

    if was_truncated:
        result_lines.extend(
            [
                "",
                "提示：文件内容过长，"
                "本次只返回了前面的部分。",
            ]
        )

    return "\n".join(result_lines)


# 给大模型看的list_directory工具说明。
LIST_DIRECTORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": (
            "列出桌面智能助手项目目录内部指定文件夹的内容。"
            "可以查看文件和子目录名称。"
            "不能访问项目目录之外的位置。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "相对于项目根目录的目录路径。"
                        "项目根目录使用英文句点'.'，"
                        "例如：'.'、'tools'、'agent_core'。"
                    ),
                    "default": ".",
                },
                "max_items": {
                    "type": "integer",
                    "description": (
                        "最多返回的目录项数量，"
                        "建议不超过50。"
                    ),
                    "default": 50,
                },
            },
            "additionalProperties": False,
        },
    },
}


# 给大模型看的read_text_file工具说明。
READ_TEXT_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_text_file",
        "description": (
            "读取桌面智能助手项目目录内部的文本文件。"
            "适合读取TXT、Markdown、Python、JSON、"
            "CSV、YAML和日志文件。"
            "不能访问项目目录之外的位置，"
            "也不能读取PDF、Word、图片等二进制文件。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "相对于项目根目录的文件路径，"
                        "例如：'main.py'、"
                        "'agent_core/prompts.py'。"
                    ),
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "最多读取的字符数，"
                        "默认12000，最大20000。"
                    ),
                    "default": 12000,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}