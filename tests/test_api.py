
"""FastAPI 接口自动化测试。"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
import pytest
from agent_core.models import AgentResponse

client = TestClient(app)


AUTH_HEADERS = {
    "X-API-Key": "test-secret-123"
}


@pytest.fixture(autouse=True)
def setup_api_key(monkeypatch):
    """为当前测试文件配置服务端 API Key。"""

    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "test-secret-123",
    )

def test_health():
    """测试健康检查接口。"""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "desktop-agent"


def test_chat_empty_message():
    """测试空字符串是否被拒绝。"""

    response = client.post(
        "/chat",
        json={"message": ""},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_chat_whitespace_message():
    """测试纯空格输入是否被拒绝。"""

    response = client.post(
        "/chat",
        json={"message": "   "},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422

    assert (
        response.json()["detail"]
        == "消息不能只包含空格"
    )


def test_chat_success():
    """模拟 Agent 回答，测试正常聊天接口。"""

    # Mock 替换真实模型调用，避免消耗 Token。
    with patch(
        "api.main.agent.chat_detailed",
        return_value=AgentResponse(
            success=True,
            answer="你好，我是测试助手。",
        ),
    ) as mock_chat:

        response = client.post(
            "/chat",
            json={"message": "你好"},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 200

        data = response.json()

        assert data["success"] is True
        assert data["answer"] == "你好，我是测试助手。"

        assert isinstance(
            data["duration_ms"],
            float,
        )

        mock_chat.assert_called_once_with(
            "你好",
            knowledge_only=False,
            raise_errors=True,
        )



def test_chat_agent_exception():
    """测试 Agent 发生异常时，API 是否返回 HTTP 500。"""

    # side_effect 用来模拟函数抛出异常。
    with patch(
        "api.main.agent.chat_detailed",
        side_effect=RuntimeError("模拟模型调用失败"),
    ) as mock_chat:

        response = client.post(
            "/chat",
            json={"message": "你好"},
            headers=AUTH_HEADERS,
        )

    # 服务器内部发生异常，应该返回 500。
    assert response.status_code == 500

    data = response.json()

    # 检查错误信息是否正确。
    assert data["detail"]["success"] is False
    assert data["detail"]["error_type"] == "RuntimeError"
    assert data["detail"]["message"] == "Agent 处理请求失败"

    # 确保 Agent 函数确实被调用过一次。
    mock_chat.assert_called_once_with(
            "你好",
            knowledge_only=False,
            raise_errors=True,
        )

def test_invalid_session_id():
    """不合法的会话 ID 应被 Pydantic 拒绝。"""

    response = client.post(
        "/chat",
        json={
            "session_id": "../../other",
            "message": "你好",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422



def test_chat_with_session_id():
    """验证 /chat 能根据 session_id 选择 Agent。"""

    with patch(
        "api.main.session_manager.use_agent"
    ) as mock_use_agent:

        # 模拟指定会话的 Agent 返回答案
        mock_agent = mock_use_agent.return_value.__enter__.return_value

        mock_agent.chat_detailed.return_value = AgentResponse(
            success=True,
            answer="这是 session_A 的回答",
        )

        # 发送包含 session_id 的 HTTP 请求
        response = client.post(
            "/chat",
            json={
                "session_id": "session_A",
                "message": "你好",
            },
            headers=AUTH_HEADERS,

        )

    # HTTP 请求应该成功
    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert data["answer"] == (
        "这是 session_A 的回答"
    )

    # 验证 FastAPI 使用了正确的会话 ID
    mock_use_agent.assert_called_once_with(
        "session_A", "tech"
    )

    # 验证对应 Agent 接收了正确的消息
    mock_agent.chat_detailed.assert_called_once_with(
        "你好",
        knowledge_only=False,
        raise_errors=True,
    )



def test_get_history_requires_api_key(monkeypatch):
    """没有 API Key 时，禁止读取历史。"""

    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "test-secret-123",
    )

    response = client.get(
        "/sessions/session_A/history"
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "API Key 无效"


def test_get_history_success(monkeypatch):
    """正确的 API Key 可以读取指定会话历史。"""

    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "test-secret-123",
    )

    fake_history = [
        {
            "role": "user",
            "content": "你好",
        },
        {
            "role": "assistant",
            "content": "你好，我是 Agent。",
        },
    ]

    with patch(
        "api.main.session_manager.use_agent"
    ) as mock_use_agent:

        mock_agent = mock_use_agent.return_value.__enter__.return_value
        mock_agent.get_history.return_value = fake_history

        response = client.get(
            "/sessions/session_A/history",
            headers={
                "X-API-Key": "test-secret-123"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["session_id"] == "session_A"
    assert data["messages"] == fake_history

    mock_use_agent.assert_called_once_with(
        "session_A", "tech"
    )

    mock_agent.get_history.assert_called_once_with(
        limit=100
    )


def test_get_history_wrong_api_key(monkeypatch):
    """错误的 API Key 应返回 HTTP 401。"""

    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "correct-secret",
    )

    with patch(
        "api.main.session_manager.use_agent"
    ) as mock_use_agent:

        response = client.get(
            "/sessions/session_A/history",
            headers={
                "X-API-Key": "wrong-secret"
            },
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "API Key 无效"

    # 鉴权失败时，不应该访问 Agent
    mock_use_agent.assert_not_called()


def test_get_history_api_key_not_configured(monkeypatch):
    """服务端没有配置 API Key 时应拒绝访问。"""

    monkeypatch.delenv(
        "AGENT_DEV_API_KEY",
        raising=False,
    )

    with patch(
        "api.main.session_manager.use_agent"
    ) as mock_use_agent:

        response = client.get(
            "/sessions/session_A/history",
            headers={
                "X-API-Key": "any-secret"
            },
        )

    assert response.status_code == 503

    assert response.json()["detail"] == (
        "API Key 未配置"
    )

    # 未配置服务端密钥时，不应该访问 Agent
    mock_use_agent.assert_not_called()

def test_chat_requires_api_key(monkeypatch):
    """没有 API Key 时，不允许调用 Agent。"""

    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "test-secret-123",
    )

    with patch("api.main.agent.chat_detailed") as mock_chat:
        response = client.post(
            "/chat",
            json={"message": "你好"},
        )

    assert response.status_code == 401
    mock_chat.assert_not_called()

def test_chat_mode_knowledge_without_session():
    """知识库模式应传递 knowledge_only=True。"""
    with patch(
        "api.main.agent.chat_detailed",
        return_value=AgentResponse(
            success=True,
            answer="模拟知识库回答",
        ),
    ) as mock_chat:
        response = client.post(
            "/chat",
            json={
                "message": "解释 RAG",
                "chat_mode": "knowledge",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    mock_chat.assert_called_once_with(
        "解释 RAG",
        knowledge_only=True,
        raise_errors=True,
    )


def test_chat_mode_knowledge_with_session():
    """company 知识库的严格模式应正确传递。"""
    with patch(
        "api.main.session_manager.use_agent"
    ) as mock_use_agent:
        mock_agent = (
            mock_use_agent.return_value.__enter__.return_value
        )
        mock_agent.chat_detailed.return_value = AgentResponse(
            success=True,
            answer="模拟年假制度回答",
        )

        response = client.post(
            "/chat",
            json={
                "message": "星桥科技员工年假怎么申请？",
                "session_id": "route_test_001",
                "knowledge_base_id": "company",
                "chat_mode": "knowledge",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    mock_use_agent.assert_called_once_with(
        "route_test_001", "company"
    )
    mock_agent.chat_detailed.assert_called_once_with(
        "星桥科技员工年假怎么申请？",
        knowledge_only=True,
        raise_errors=True,
    )


def test_invalid_chat_mode():
    """非法业务模式应返回 HTTP 422。"""
    response = client.post(
        "/chat",
        json={
            "message": "你好",
            "chat_mode": "invalid",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422
