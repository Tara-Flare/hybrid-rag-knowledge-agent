"""FastAPI 结构化响应契约测试。"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from agent_core.models import (
    AgentResponse,
    SourceInfo,
    ToolCallInfo,
)

client = TestClient(app)
HEADERS = {"X-API-Key": "test-secret-123"}


@pytest.fixture(autouse=True)
def setup_key(monkeypatch):
    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "test-secret-123",
    )


def make_grounded_response():
    return AgentResponse(
        success=True,
        answer="根据制度，需提前申请。",
        sources=[
            SourceInfo(
                file_name="/private/server/leave_policy.md",
                chunk_id="leave_policy.md::chunk-0",
                score=0.99,
            )
        ],
        tool_calls=[
            ToolCallInfo(
                tool_name="search_knowledge",
                success=True,
            )
        ],
    )


def test_legacy_response_fields_preserved():
    """旧客户端仍可读取 success、answer、duration_ms。"""

    with patch(
        "api.main.agent.chat_detailed",
        return_value=AgentResponse(
            success=True,
            answer="你好",
        ),
    ):
        response = client.post(
            "/chat",
            json={"message": "你好"},
            headers=HEADERS,
        )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["answer"] == "你好"
    assert isinstance(data["duration_ms"], float)
    assert data["sources"] == []
    assert data["tool_calls"] == []
    assert data["error_type"] is None


def test_grounded_sources_returned_safely():
    """已验证引用正常返回，但不暴露绝对路径和分数。"""

    with patch(
        "api.main.agent.chat_detailed",
        return_value=make_grounded_response(),
    ):
        response = client.post(
            "/chat",
            json={
                "message": "年假如何申请？",
                "chat_mode": "knowledge",
            },
            headers=HEADERS,
        )

    assert response.status_code == 200
    data = response.json()

    assert data["sources"] == [
        {
            "file_name": "leave_policy.md",
            "chunk_id": "leave_policy.md::chunk-0",
        }
    ]

    assert data["tool_calls"] == [
        {
            "tool_name": "search_knowledge",
            "success": True,
            "error_type": None,
        }
    ]

    assert "/private/server" not in str(data)
    assert "score" not in str(data)


def test_failed_agent_hides_sources():
    """Agent 失败时，即使内部存在来源，也不向外暴露。"""

    result = make_grounded_response()
    result.success = False
    result.error_type = "GroundingNotVerifiedError"

    with patch(
        "api.main.agent.chat_detailed",
        return_value=result,
    ):
        response = client.post(
            "/chat",
            json={"message": "测试失败回答"},
            headers=HEADERS,
        )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is False
    assert data["sources"] == []
    assert data["error_type"] == "GroundingNotVerifiedError"


def test_no_search_call_hides_sources():
    """没有成功检索调用时不返回来源。"""

    result = make_grounded_response()
    result.tool_calls = []

    with patch(
        "api.main.agent.chat_detailed",
        return_value=result,
    ):
        response = client.post(
            "/chat",
            json={"message": "你好"},
            headers=HEADERS,
        )

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_session_knowledge_response():
    """Session + company 模式正确返回结构化引用。"""

    with patch(
        "api.main.session_manager.use_agent"
    ) as mock_manager:
        mock_agent = (
            mock_manager.return_value.__enter__.return_value
        )

        mock_agent.chat_detailed.return_value = (
            make_grounded_response()
        )

        response = client.post(
            "/chat",
            json={
                "message": "年假如何申请？",
                "session_id": "response_contract_test",
                "knowledge_base_id": "company",
                "chat_mode": "knowledge",
            },
            headers=HEADERS,
        )

    assert response.status_code == 200

    mock_manager.assert_called_once_with(
        "response_contract_test", "company"
    )

    assert (
        response.json()["sources"][0]["file_name"]
        == "leave_policy.md"
    )
