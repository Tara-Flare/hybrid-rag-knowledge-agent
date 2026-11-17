"""工具执行阶段 RequestTrace 集成测试。"""

from agent_core.agent import DesktopAgent
from storage.request_trace import (
    RequestTrace,
    RequestTraceStore,
)
from storage.sqlite_store import SQLiteStore
from storage.trace_context import bind_request_trace
from tools.tool_result import ToolResult


def make_agent(store, functions):
    agent = DesktopAgent.__new__(DesktopAgent)
    agent.store = store
    agent.tool_functions = functions
    return agent


def get_tool_phase(trace):
    phases = [
        phase
        for phase in trace.phases
        if phase.name == "tool_execution"
    ]

    assert len(phases) == 1
    return phases[0]


def test_successful_tool_execution(tmp_path):
    """成功工具调用生成阶段追踪和关联日志。"""

    db_path = str(tmp_path / "agent.db")

    store = SQLiteStore(db_path)

    agent = make_agent(
        store,
        {"echo": lambda value: f"收到：{value}"},
    )

    trace = RequestTrace(
        session_id="tool_success",
        knowledge_base_id="tech",
    )

    with bind_request_trace(trace):
        result = agent._run_tool(
            "echo",
            {"value": "测试"},
        )

    phase = get_tool_phase(trace)

    assert "收到：测试" in result
    assert phase.success is True
    assert phase.outcome == "success"
    assert phase.error_type is None
    assert phase.duration_ms >= 0

    trace.finish()

    trace_store = RequestTraceStore(db_path)
    trace_store.save(trace)

    saved = trace_store.get(trace.request_id)

    assert saved is not None

    saved_phases = [
        p for p in saved["phases"]
        if p["name"] == "tool_execution"
    ]

    assert len(saved_phases) == 1
    assert saved_phases[0]["outcome"] == "success"

    logs = store.get_tool_logs_by_request(
        trace.request_id
    )

    assert len(logs) == 1
    assert logs[0]["success"] is True
    assert logs[0]["session_id"] == "tool_success"
    assert logs[0]["knowledge_base_id"] == "tech"


def test_business_failure_outcome(tmp_path):
    """ToolResult.fail 不应被误记为 Python 异常。"""

    store = SQLiteStore(
        str(tmp_path / "agent.db")
    )

    def denied():
        return ToolResult.fail(
            message="该操作未获授权",
            error_type="AccessDeniedError",
        )

    agent = make_agent(
        store,
        {"denied": denied},
    )

    trace = RequestTrace(
        session_id="tool_denied",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        agent._run_tool("denied", {})

    phase = get_tool_phase(trace)

    assert phase.success is True
    assert phase.outcome == "failed"
    assert phase.error_type == "AccessDeniedError"

    logs = store.get_tool_logs_by_request(
        trace.request_id
    )

    assert len(logs) == 1
    assert logs[0]["success"] is False
    assert logs[0]["error_type"] == "AccessDeniedError"


def test_technical_exception_trace(tmp_path):
    """工具执行异常应记录技术失败。"""

    store = SQLiteStore(
        str(tmp_path / "agent.db")
    )

    def broken_tool():
        raise RuntimeError("模拟工具异常")

    agent = make_agent(
        store,
        {"broken": broken_tool},
    )

    trace = RequestTrace(
        session_id="tool_error",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        agent._run_tool("broken", {})

    phase = get_tool_phase(trace)

    assert phase.success is False
    assert phase.outcome == "failed"
    assert phase.error_type == "RuntimeError"

    logs = store.get_tool_logs_by_request(
        trace.request_id
    )

    assert len(logs) == 1
    assert logs[0]["success"] is False
    assert logs[0]["error_type"] == "RuntimeError"


def test_standalone_tool_execution(tmp_path):
    """没有 RequestTrace 的旧版工具调用保持兼容。"""

    store = SQLiteStore(
        str(tmp_path / "agent.db")
    )

    agent = make_agent(
        store,
        {"echo": lambda value: value},
    )

    result = agent._run_tool(
        "echo",
        {"value": "standalone"},
    )

    assert "standalone" in result

    logs = store.get_recent_tool_logs()

    assert len(logs) == 1
    assert logs[0]["success"] is True
