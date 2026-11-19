"""可配置知识库 Agent 的 FastAPI 服务入口。"""

import logging

from threading import Lock
from pathlib import PurePosixPath
from time import perf_counter

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Path,
    Query,
    Response,
)

from agent_core.agent import DesktopAgent
from api.schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeBaseId,
    SourceResponse,
    ToolCallResponse,
)
from api.security import require_api_key
from storage.sqlite_store import SQLiteStore
from storage.request_trace import RequestTrace, RequestTraceStore
from storage.trace_context import bind_request_trace
from api.session_manager import (
    SessionManager,
    SessionBusyError,
    SessionKnowledgeBaseConflictError,
)


app = FastAPI(
    title="Knowledge Base Agent API",
    description="多知识库 Function Calling Agent",
    version="1.1.0",
)


# 兼容不提供 session_id 的旧版接口。
# 该 Agent 固定使用 tech 知识库。
agent = DesktopAgent(
    knowledge_base_id="tech"
)


# 新版会话管理。
# 工厂同时支持旧版单参数调用和新版双参数调用。
session_manager = SessionManager(
    agent_factory=lambda session_id, knowledge_base_id="tech": (
        DesktopAgent(
            session_id=session_id,
            knowledge_base_id=knowledge_base_id,
        )
    ),
    ttl_seconds=1800,
    binding_store=SQLiteStore("data/agent.db"),
)


# 仅保护旧版无 session_id 的共享 Agent。
# 不同 Session 的请求不再使用这把全局锁。
agent_lock = Lock()

# 请求链路追踪存储
trace_store = RequestTraceStore("data/agent.db")


def finish_and_save_trace(
    trace: RequestTrace,
    error: Exception | None = None,
) -> None:
    """结束追踪并持久化。

    追踪存储失败时记录日志，不覆盖原始 API 结果。
    """
    trace.finish(error=error)

    try:
        trace_store.save(trace)
    except Exception:
        logging.exception(
            "RequestTrace 持久化失败：%s",
            trace.request_id,
        )



@app.get("/health")
def health() -> dict[str, str]:
    """服务健康检查。"""

    return {
        "status": "ok",
        "service": "desktop-agent",
    }


@app.get(
    "/sessions/{session_id}/history",
    dependencies=[Depends(require_api_key)],
)
def get_session_history(
    session_id: str = Path(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
    knowledge_base_id: KnowledgeBaseId = Query(
        default="tech",
    ),
) -> dict:
    """读取指定会话在指定知识库中的历史。"""

    try:
        with session_manager.use_agent(
            session_id,
            knowledge_base_id,
        ) as current_agent:

            history = current_agent.get_history(
                limit=100
            )

    except (
        SessionKnowledgeBaseConflictError,
        SessionBusyError,
    ) as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return {
        "session_id": session_id,
        "messages": history,
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
)
def chat(
    request: ChatRequest,
    response: Response,
) -> ChatResponse:
    """执行 Agent 对话并记录 RequestTrace。"""

    trace = RequestTrace(
        session_id=request.session_id,
        knowledge_base_id=request.knowledge_base_id,
    )

    try:
        # 第一阶段：输入校验
        with trace.phase("input_validation"):

            if not request.message.strip():
                raise HTTPException(
                    status_code=422,
                    detail="消息不能只包含空格",
                )

            if (
                request.session_id is None
                and request.knowledge_base_id != "tech"
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "使用非默认知识库时，"
                        "必须提供 session_id"
                    ),
                )

        # 第二阶段：Agent 执行
        with (
            bind_request_trace(trace),
            trace.phase("agent_execution") as execution_phase,
        ):

            if request.session_id is not None:

                with session_manager.use_agent(
                    request.session_id,
                    request.knowledge_base_id,
                ) as current_agent:

                    agent_response = current_agent.chat_detailed(
                        request.message,
                        knowledge_only=(
                            request.chat_mode == "knowledge"
                        ),
                        raise_errors=True,
                    )

            else:
                # 保留旧版无 Session 接口
                with agent_lock:
                    agent_response = agent.chat_detailed(
                        request.message,
                        knowledge_only=(
                            request.chat_mode == "knowledge"
                        ),
                        raise_errors=True,
                    )

            # 区分请求正常完成与 Agent 任务是否成功。
            execution_phase.outcome = (
                "agent_success"
                if agent_response.success
                else "agent_failure"
            )

    except (
        SessionKnowledgeBaseConflictError,
        SessionBusyError,
    ) as exc:

        finish_and_save_trace(trace, error=exc)

        raise HTTPException(
            status_code=409,
            detail=str(exc),
            headers={
                "X-Request-ID": trace.request_id,
            },
        ) from exc

    except HTTPException as exc:

        finish_and_save_trace(trace, error=exc)

        headers = dict(exc.headers or {})
        headers["X-Request-ID"] = trace.request_id
        exc.headers = headers

        raise

    except Exception as exc:

        finish_and_save_trace(trace, error=exc)

        logging.exception(
            "Agent 对话请求失败，request_id=%s",
            trace.request_id,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_type": type(exc).__name__,
                "message": "Agent 处理请求失败",
            },
            headers={
                "X-Request-ID": trace.request_id,
            },
        ) from exc

    # 成功请求：结束并保存追踪
    finish_and_save_trace(trace)

    response.headers["X-Request-ID"] = (
        trace.request_id
    )

    # Agent 内部仅在 Grounding 通过后收集 sources。
    # API 再进行成功状态与检索工具检查，避免失败响应泄露引用。
    search_succeeded = any(
        call.tool_name == "search_knowledge"
        and call.success
        for call in agent_response.tool_calls
    )

    grounded_sources = (
        agent_response.sources
        if agent_response.success and search_succeeded
        else []
    )

    return ChatResponse(
        success=agent_response.success,
        answer=agent_response.answer,
        duration_ms=trace.duration_ms,
        sources=[
            SourceResponse(
                file_name=PurePosixPath(
                    source.file_name.replace(
                        "\\", "/"
                    )
                ).name,
                chunk_id=source.chunk_id,
            )
            for source in grounded_sources
        ],
        tool_calls=[
            ToolCallResponse(
                tool_name=call.tool_name,
                success=call.success,
                error_type=call.error_type,
            )
            for call in agent_response.tool_calls
        ],
        error_type=agent_response.error_type,
    )
