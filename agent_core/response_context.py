"""Agent 单次请求的结构化结果收集上下文。"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from agent_core.models import SourceInfo, ToolCallInfo


@dataclass
class AgentRunCapture:
    """只保存当前请求产生的真实执行信息。"""

    tool_calls: list[ToolCallInfo] = field(
        default_factory=list
    )

    sources: list[SourceInfo] = field(
        default_factory=list
    )


ACTIVE_AGENT_RUN_CAPTURE: ContextVar[
    AgentRunCapture | None
] = ContextVar(
    "active_agent_run_capture",
    default=None,
)
