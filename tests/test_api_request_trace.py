"""FastAPI RequestTrace 集成测试。"""

import importlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.session_manager import SessionBusyError
from storage.request_trace import RequestTraceStore
from agent_core.models import AgentResponse


api_main = importlib.import_module("api.main")
client = TestClient(api_main.app)

HEADERS = {
    "X-API-Key": "test-secret-123",
}


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    """每个测试使用独立的 SQLite 数据库。"""

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

    return store


def get_saved_trace(store, response):
    request_id = response.headers.get(
        "X-Request-ID"
    )

    assert request_id

    trace = store.get(request_id)

    assert trace is not None
    assert trace["request_id"] == request_id

    return trace


def test_chat_success_trace(trace_db):
    """成功聊天应保存完整 RequestTrace。"""

    with patch(
        "api.main.agent.chat_detailed",
        return_value=AgentResponse(
            success=True,
            answer="测试回答",
        ),
    ):
        response = client.post(
            "/chat",
            json={"message": "你好"},
            headers=HEADERS,
        )

    assert response.status_code == 200

    trace = get_saved_trace(
        trace_db, response
    )

    assert trace["status"] == "success"
    assert trace["knowledge_base_id"] == "tech"
    assert trace["session_id"] is None

    assert [p["name"] for p in trace["phases"]] == [
        "input_validation",
        "agent_execution",
    ]

    assert all(
        p["success"]
        for p in trace["phases"]
    )


def test_chat_failure_trace(trace_db):
    """Agent 异常时也必须保存追踪。"""

    with patch(
        "api.main.agent.chat_detailed",
        side_effect=RuntimeError("模拟失败"),
    ):
        response = client.post(
            "/chat",
            json={"message": "你好"},
            headers=HEADERS,
        )

    assert response.status_code == 500

    trace = get_saved_trace(
        trace_db, response
    )

    assert trace["status"] == "error"
    assert trace["error_type"] == "RuntimeError"

    assert trace["phases"][-1]["name"] == (
        "agent_execution"
    )

    assert trace["phases"][-1]["success"] is False


def test_session_busy_trace(trace_db):
    """Session 并发冲突返回 409，并保存失败追踪。"""

    with patch(
        "api.main.session_manager.use_agent",
        side_effect=SessionBusyError("模拟会话占用"),
    ):
        response = client.post(
            "/chat",
            json={
                "session_id": "trace_busy",
                "knowledge_base_id": "tech",
                "message": "你好",
            },
            headers=HEADERS,
        )

    assert response.status_code == 409

    trace = get_saved_trace(
        trace_db, response
    )

    assert trace["status"] == "error"
    assert trace["error_type"] == "SessionBusyError"
    assert trace["session_id"] == "trace_busy"


def test_whitespace_validation_trace(trace_db):
    """进入路由后的 422 校验错误也应被记录。"""

    response = client.post(
        "/chat",
        json={"message": "   "},
        headers=HEADERS,
    )

    assert response.status_code == 422

    trace = get_saved_trace(
        trace_db, response
    )

    assert trace["status"] == "error"

    assert trace["phases"][0]["name"] == (
        "input_validation"
    )

    assert trace["phases"][0]["success"] is False
