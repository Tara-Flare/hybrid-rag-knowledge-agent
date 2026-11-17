"""Agent 核心层的结构化数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceInfo:
    """RAG 答案来源。"""

    file_name: str
    chunk_id: str
    score: float | None = None


@dataclass
class ToolCallInfo:
    """一次工具调用的摘要。"""

    tool_name: str
    success: bool
    error_type: str | None = None


@dataclass
class AgentResponse:
    """Agent 对外返回的统一结果。"""

    success: bool
    answer: str

    sources: list[SourceInfo] = field(
        default_factory=list
    )

    tool_calls: list[ToolCallInfo] = field(
        default_factory=list
    )

    error_type: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )
