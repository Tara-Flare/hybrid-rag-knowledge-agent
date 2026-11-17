"""FastAPI Session 持久化绑定集成测试。"""

import importlib
import sqlite3

from fastapi.testclient import TestClient

from api.session_manager import SessionManager
from agent_core.models import AgentResponse
from storage.sqlite_store import SQLiteStore


api_main = importlib.import_module("api.main")


class FakeAgent:
    """模拟 Agent，避免真实模型调用。"""

    def chat_detailed(
        self,
        message,
        *,
        raise_errors=False,
    ):
        return AgentResponse(
            success=True,
            answer=f"收到：{message}",
        )


def make_manager(db_path):
    return SessionManager(
        agent_factory=(
            lambda session_id, knowledge_base_id="tech":
            FakeAgent()
        ),
        binding_store=SQLiteStore(str(db_path)),
    )


def test_api_binding_survives_restart(
    tmp_path,
    monkeypatch,
):
    """API 重建 SessionManager 后仍拒绝跨库请求。"""

    # 确认正式 API 已配置 SQLite
    assert isinstance(
        api_main.session_manager.binding_store,
        SQLiteStore,
    )

    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "test-secret-123",
    )

    db_path = tmp_path / "api_sessions.db"

    monkeypatch.setattr(
        api_main,
        "session_manager",
        make_manager(db_path),
    )

    client = TestClient(api_main.app)

    headers = {
        "X-API-Key": "test-secret-123",
    }

    # 1. 首次绑定 company
    first = client.post(
        "/chat",
        json={
            "session_id": "restart_api_test",
            "knowledge_base_id": "company",
            "message": "年假如何申请？",
        },
        headers=headers,
    )

    assert first.status_code == 200

    # 2. 模拟服务重启：重新创建管理器
    monkeypatch.setattr(
        api_main,
        "session_manager",
        make_manager(db_path),
    )

    # 3. 尝试切换到 tech，必须返回 409
    conflict = client.post(
        "/chat",
        json={
            "session_id": "restart_api_test",
            "knowledge_base_id": "tech",
            "message": "什么是 RRF？",
        },
        headers=headers,
    )

    assert conflict.status_code == 409

    # 4. 继续访问原 company，应该成功
    allowed = client.post(
        "/chat",
        json={
            "session_id": "restart_api_test",
            "knowledge_base_id": "company",
            "message": "报销需要什么材料？",
        },
        headers=headers,
    )

    assert allowed.status_code == 200

    # 5. 检查 SQLite 中的实际绑定
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT knowledge_base_id
            FROM session_bindings
            WHERE session_id = ?
            """,
            ("restart_api_test",),
        ).fetchone()

    assert row is not None
    assert row[0] == "company"
