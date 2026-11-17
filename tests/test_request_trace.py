"""RequestTrace 自动化测试。"""

import pytest

from storage.request_trace import (
    RequestTrace,
    RequestTraceStore,
)


def test_trace_success(tmp_path):
    """成功请求能够记录阶段并持久化。"""

    store = RequestTraceStore(
        str(tmp_path / "trace.db")
    )

    trace = RequestTrace(
        session_id="session_A",
        knowledge_base_id="tech",
    )

    with trace.phase("hybrid_retrieval"):
        result = [1, 2, 3]

    assert len(result) == 3

    trace.finish()
    store.save(trace)

    saved = store.get(trace.request_id)

    assert saved is not None
    assert saved["status"] == "success"
    assert saved["knowledge_base_id"] == "tech"
    assert saved["duration_ms"] >= 0

    assert len(saved["phases"]) == 1
    assert saved["phases"][0]["name"] == "hybrid_retrieval"
    assert saved["phases"][0]["success"] is True


def test_trace_failure(tmp_path):
    """执行阶段失败时记录错误类型。"""

    store = RequestTraceStore(
        str(tmp_path / "trace.db")
    )

    trace = RequestTrace(
        session_id="session_B",
        knowledge_base_id="company",
    )

    error = RuntimeError("模拟检索失败")

    with pytest.raises(RuntimeError):
        with trace.phase("retrieval"):
            raise error

    trace.finish(error=error)
    store.save(trace)

    saved = store.get(trace.request_id)

    assert saved["status"] == "error"
    assert saved["error_type"] == "RuntimeError"

    assert saved["phases"][0]["success"] is False
    assert saved["phases"][0]["error_type"] == "RuntimeError"


def test_running_trace_cannot_be_saved(tmp_path):
    """未结束的请求不能作为完整记录保存。"""

    store = RequestTraceStore(
        str(tmp_path / "trace.db")
    )

    trace = RequestTrace(
        session_id="session_C",
        knowledge_base_id="tech",
    )

    with pytest.raises(ValueError):
        store.save(trace)


def test_trace_store_does_not_remove_existing_tables(tmp_path):
    """创建追踪表不能破坏现有工具日志表。"""

    from storage.sqlite_store import SQLiteStore

    db_path = str(tmp_path / "agent.db")

    SQLiteStore(db_path)
    trace_store = RequestTraceStore(db_path)

    trace = RequestTrace(
        session_id="session_A",
        knowledge_base_id="tech",
    )

    trace.finish()
    trace_store.save(trace)

    store = SQLiteStore(db_path)

    assert store.get_recent_tool_logs() == []
    assert trace_store.get(trace.request_id) is not None
