"""SessionManager 与 SQLite 持久化集成测试。"""

import pytest
from unittest.mock import Mock

from api.session_manager import (
    SessionManager,
    SessionKnowledgeBaseConflictError,
)
from storage.sqlite_store import SQLiteStore


def make_factory():
    return Mock(
        side_effect=lambda session_id, knowledge_base_id="tech": {
            "session_id": session_id,
            "knowledge_base_id": knowledge_base_id,
        }
    )


def test_binding_survives_manager_restart(tmp_path):
    """重新创建 SessionManager 后仍不能切换知识库。"""

    db_path = tmp_path / "sessions.db"

    first_manager = SessionManager(
        agent_factory=make_factory(),
        binding_store=SQLiteStore(str(db_path)),
    )

    first_agent = first_manager.get_agent(
        "restart_session",
        "company",
    )

    assert first_agent["knowledge_base_id"] == "company"

    # 模拟服务重启：重新创建管理器和数据库连接
    second_factory = make_factory()

    second_manager = SessionManager(
        agent_factory=second_factory,
        binding_store=SQLiteStore(str(db_path)),
    )

    with pytest.raises(
        SessionKnowledgeBaseConflictError
    ):
        second_manager.get_agent(
            "restart_session",
            "tech",
        )

    # 冲突请求不能创建 Agent
    second_factory.assert_not_called()

    # 使用原知识库仍然允许重新创建 Agent
    agent = second_manager.get_agent(
        "restart_session",
        "company",
    )

    assert agent["knowledge_base_id"] == "company"

    second_factory.assert_called_once_with(
        "restart_session",
        "company",
    )


def test_multiple_managers_share_binding(tmp_path):
    """两个管理器使用同一 SQLite 时必须遵守绑定。"""

    db_path = tmp_path / "shared.db"

    manager_a = SessionManager(
        agent_factory=make_factory(),
        binding_store=SQLiteStore(str(db_path)),
    )

    manager_b = SessionManager(
        agent_factory=make_factory(),
        binding_store=SQLiteStore(str(db_path)),
    )

    manager_a.get_agent(
        "shared_session",
        "tech",
    )

    with pytest.raises(
        SessionKnowledgeBaseConflictError
    ):
        manager_b.get_agent(
            "shared_session",
            "company",
        )

    # 不同 Session 仍然可以绑定不同知识库
    company_agent = manager_b.get_agent(
        "another_session",
        "company",
    )

    assert company_agent["knowledge_base_id"] == "company"

    # 数据库中原有绑定不应改变
    store = SQLiteStore(str(db_path))

    assert store.get_or_bind_session(
        "shared_session",
        "company",
    ) == "tech"
