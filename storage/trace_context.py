"""在请求执行上下文中安全传递 RequestTrace。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from storage.request_trace import RequestTrace, TracePhase


_active_trace: ContextVar[
    RequestTrace | None
] = ContextVar(
    "active_request_trace",
    default=None,
)


# 当前阶段以及它所属的请求。
# 两者同时保存，防止嵌套请求错误归属 Token。
_active_phase: ContextVar[
    tuple[RequestTrace, TracePhase] | None
] = ContextVar(
    "active_trace_phase",
    default=None,
)


@contextmanager
def bind_request_trace(
    trace: RequestTrace,
) -> Iterator[None]:
    """将 Trace 绑定到当前请求执行上下文。"""

    token = _active_trace.set(trace)

    try:
        yield

    finally:
        # 请求结束后恢复原来的上下文，
        # 防止请求之间的 Trace 污染。
        _active_trace.reset(token)


@contextmanager
def observed_phase(
    name: str,
) -> Iterator[TracePhase | None]:
    """记录阶段，允许填写业务结果。"""

    trace = _active_trace.get()

    if trace is None:
        yield None
        return

    with trace.phase(name) as phase:
        token = _active_phase.set((trace, phase))

        try:
            yield phase

        finally:
            _active_phase.reset(token)


def record_model_usage(usage: object | None) -> None:
    """将一次成功模型调用的 Token 用量记录到当前 Trace。

    无 Trace 时不做任何操作。
    缺失或无效的 Token 数据不参与用量求和。
    """

    trace = _active_trace.get()

    if trace is None:
        return

    if trace.token_usage is None:
        trace.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model_calls": 0,
            "usage_reported_calls": 0,
            "usage_missing_calls": 0,
        }

    stats = trace.token_usage

    # 同时统计请求总用量与当前执行阶段的用量。
    binding = _active_phase.get()

    targets = [stats]

    if binding is not None and binding[0] is trace:
        phase = binding[1]

        if phase.token_usage is None:
            phase.token_usage = {
                key: 0
                for key in stats
            }

        targets.append(phase.token_usage)

    for target in targets:
        target["model_calls"] += 1

    values = (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )

    # 不使用字符串、Mock、None 或负数冒充真实用量。
    valid = all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for value in values
    )

    if not valid:
        for target in targets:
            target["usage_missing_calls"] += 1
        return

    prompt, completion, total = values

    for target in targets:
        target["prompt_tokens"] += prompt
        target["completion_tokens"] += completion
        target["total_tokens"] += total
        target["usage_reported_calls"] += 1


def get_tool_log_context() -> dict[str, str | None]:
    """获取当前请求的工具日志关联字段。

    直接运行 Agent 时，没有 RequestTrace 则返回空字典，
    保持旧版 SQLiteStore.add_tool_log 调用兼容。
    """

    trace = _active_trace.get()

    if trace is None:
        return {}

    return {
        "request_id": trace.request_id,
        "session_id": trace.session_id,
        "knowledge_base_id": trace.knowledge_base_id,
    }
