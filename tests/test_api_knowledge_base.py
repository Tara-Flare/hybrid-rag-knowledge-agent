"""多知识库 FastAPI 接口测试。"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app, session_manager
from agent_core.models import AgentResponse


client = TestClient(app)

HEADERS = {
    "X-API-Key": "test-secret-123",
}


@pytest.fixture(autouse=True)
def setup_api_key(monkeypatch):
    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "test-secret-123",
    )


def test_company_chat():
    """company 请求应创建 company Agent。"""

    with patch("api.main.DesktopAgent") as mock_class:

        mock_class.return_value.chat_detailed.return_value = AgentResponse(
            success=True,
            answer="年假需要提前2个工作日申请。",
        )

        response = client.post(
            "/chat",
            json={
                "session_id": "api_company_case",
                "knowledge_base_id": "company",
                "message": "年假如何申请？",
            },
            headers=HEADERS,
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

        mock_class.assert_called_once_with(
            session_id="api_company_case",
            knowledge_base_id="company",
        )


def test_session_kb_switch_returns_409():
    """同一 Session 不允许切换知识库。"""

    with patch("api.main.DesktopAgent") as mock_class:

        mock_class.return_value.chat_detailed.return_value = AgentResponse(
            success=True,
            answer="测试回答",
        )

        first = client.post(
            "/chat",
            json={
                "session_id": "api_switch_case",
                "knowledge_base_id": "tech",
                "message": "你好",
            },
            headers=HEADERS,
        )

        assert first.status_code == 200

        second = client.post(
            "/chat",
            json={
                "session_id": "api_switch_case",
                "knowledge_base_id": "company",
                "message": "你好",
            },
            headers=HEADERS,
        )

        assert second.status_code == 409

        # 第二次请求不应该执行模型
        assert mock_class.return_value.chat_detailed.call_count == 1


def test_same_session_busy_returns_409():
    """同一个 Session 已被占用时返回 409。"""

    with patch("api.main.DesktopAgent"):

        with session_manager.use_agent(
            "api_busy_case",
            "tech",
        ):

            response = client.post(
                "/chat",
                json={
                    "session_id": "api_busy_case",
                    "knowledge_base_id": "tech",
                    "message": "你好",
                },
                headers=HEADERS,
            )

            assert response.status_code == 409


def test_invalid_knowledge_base():
    """不合法的知识库 ID 返回 422。"""

    response = client.post(
        "/chat",
        json={
            "session_id": "api_invalid_case",
            "knowledge_base_id": "unknown",
            "message": "你好",
        },
        headers=HEADERS,
    )

    assert response.status_code == 422


def test_company_requires_session_id():
    """非默认知识库必须提供 Session ID。"""

    response = client.post(
        "/chat",
        json={
            "knowledge_base_id": "company",
            "message": "年假怎么申请？",
        },
        headers=HEADERS,
    )

    assert response.status_code == 422


def test_history_kb_conflict():
    """历史查询也不能绕过知识库绑定。"""

    with patch("api.main.DesktopAgent") as mock_class:

        mock_class.return_value.chat_detailed.return_value = AgentResponse(
            success=True,
            answer="测试回答",
        )

        first = client.post(
            "/chat",
            json={
                "session_id": "api_history_kb_case",
                "knowledge_base_id": "tech",
                "message": "你好",
            },
            headers=HEADERS,
        )

        assert first.status_code == 200

        response = client.get(
            "/sessions/api_history_kb_case/history",
            params={
                "knowledge_base_id": "company",
            },
            headers=HEADERS,
        )

        assert response.status_code == 409
