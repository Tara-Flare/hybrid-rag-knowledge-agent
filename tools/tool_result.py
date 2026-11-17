"""工具统一返回结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(slots=True)
class ToolResult:
    """
    工具统一返回结果。

    Attributes:
        success: 工具是否成功完成业务目标。
        message: 适合返回给大模型的文字结果。
        data: 可选的结构化结果。
        error_type: 失败时的错误类型。
    """

    success: bool
    message: str
    data: Any = None
    error_type: str | None = None

    @classmethod
    def ok(
        cls,
        message: str,
        data: Any = None,
    ) -> "ToolResult":
        """创建一个成功结果。"""

        return cls(
            success=True,
            message=message,
            data=data,
            error_type=None,
        )

    @classmethod
    def fail(
        cls,
        message: str,
        error_type: str,
        data: Any = None,
    ) -> "ToolResult":
        """创建一个失败结果。"""

        return cls(
            success=False,
            message=message,
            data=data,
            error_type=error_type,
        )

    def to_text(self) -> str:
        """转换成适合发给大模型的文本。"""

        return self.message