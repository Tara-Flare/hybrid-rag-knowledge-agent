"""RequestTrace 上下文隔离测试。"""

import importlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from storage.request_trace import (
    RequestTrace,
    RequestTraceStore,
)
from storage.trace_context import (
    bind_request_trace,
    observed_phase,
)


def test_nested_trace_isolation():
    """不同 Trace 上下文不能混合记录。"""

    trace_a = RequestTrace(
        session_id="session_A",
        knowledge_base_id="tech",
    )

    trace_b = RequestTrace(
        session_id="session_B",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace_a):
        with observed_phase("phase_a"):
            pass

        with bind_request_trace(trace_b):
            with observed_phase("phase_b"):
                pass

        with observed_phase("phase_a_again"):
            pass

    assert [p.name for p in trace_a.phases] == [
        "phase_a",
        "phase_a_again",
    ]

    assert [p.name for p in trace_b.phases] == [
        "phase_b",
    ]


def test_observed_phase_without_trace():
    """没有活动 Trace 时也能正常执行。"""

    with observed_phase("standalone"):
        result = 1 + 1

    assert result == 2


def test_api_propagates_trace_to_agent(tmp_path, monkeypatch):
    """API 创建的 Trace 应能在 Agent 调用中使用。"""

    api_main = importlib.import_module("api.main")

    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "test-secret-123",
    )

    store = RequestTraceStore(
        str(tmp_path / "trace.db")
    )

    monkeypatch.setattr(
        api_main,
        "trace_store",
        store,
    )

    def fake_chat(message, *, knowledge_only=False):
        # 模拟 Agent 内部两个阶段
        with observed_phase("llm_call"):
            pass

        with observed_phase("hybrid_retrieval"):
            pass

        return "测试回答"

    client = TestClient(api_main.app)

    with patch(
        "api.main.agent.chat",
        side_effect=fake_chat,
    ):
        response = client.post(
            "/chat",
            json={"message": "你好"},
            headers={
                "X-API-Key": "test-secret-123",
            },
        )

    assert response.status_code == 200

    request_id = response.headers["X-Request-ID"]
    saved = store.get(request_id)

    assert saved is not None

    phase_names = [
        phase["name"]
        for phase in saved["phases"]
    ]

    assert "input_validation" in phase_names
    assert "agent_execution" in phase_names
    assert "llm_call" in phase_names
    assert "hybrid_retrieval" in phase_names

    # 调用结束后，上下文已被解除绑定。
    with observed_phase("after_request"):
        pass

    assert "after_request" not in [
        phase["name"]
        for phase in saved["phases"]
    ]
