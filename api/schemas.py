"""FastAPI 请求和响应模型。"""

from typing import Literal

from pydantic import BaseModel, Field


KnowledgeBaseId = Literal["tech", "company"]


class ChatRequest(BaseModel):
    """聊天请求参数。"""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户发送给 Agent 的消息",
    )

    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="会话唯一标识",
    )

    knowledge_base_id: KnowledgeBaseId = Field(
        default="tech",
        description="选择的知识库，默认 tech",
    )


class ChatResponse(BaseModel):
    """聊天接口返回结果。"""

    success: bool
    answer: str
    duration_ms: float
