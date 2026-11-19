"""FastAPI 请求和响应模型。"""

from typing import Literal

from pydantic import BaseModel, Field


KnowledgeBaseId = Literal["tech", "company"]
ChatMode = Literal["general", "knowledge"]


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
    chat_mode: ChatMode = Field(
        default="general",
        description="general 为普通 Agent；knowledge 为严格知识库问答",
    )


class SourceResponse(BaseModel):
    """已验证的知识库引用，不包含服务器路径。"""

    file_name: str
    chunk_id: str


class ToolCallResponse(BaseModel):
    """安全的工具调用摘要，不包含调用参数。"""

    tool_name: str
    success: bool
    error_type: str | None = None


class ChatResponse(BaseModel):
    """聊天接口返回结果。"""

    success: bool
    answer: str
    duration_ms: float

    sources: list[SourceResponse] = Field(
        default_factory=list
    )

    tool_calls: list[ToolCallResponse] = Field(
        default_factory=list
    )

    error_type: str | None = None
