"""Session 多知识库、TTL 和并发测试。"""

import pytest

from api.session_manager import (
    SessionManager,
    SessionBusyError,
    SessionKnowledgeBaseConflictError,
)


def make_factory():
    """创建不依赖真实模型的测试工厂。"""

    def factory(
        session_id,
        knowledge_base_id="tech",
    ):
        return {
            "session_id": session_id,
            "knowledge_base_id": knowledge_base_id,
        }

    return factory


def test_session_kb_binding():
    """同一 Session 不允许切换知识库。"""

    manager = SessionManager(make_factory())

    agent = manager.get_agent(
        "session_A",
        "tech",
    )

    assert agent["knowledge_base_id"] == "tech"

    with pytest.raises(
        SessionKnowledgeBaseConflictError
    ):
        manager.get_agent(
            "session_A",
            "company",
        )


def test_different_kb_sessions():
    """不同 Session 可使用不同知识库。"""

    manager = SessionManager(make_factory())

    tech = manager.get_agent(
        "session_A",
        "tech",
    )

    company = manager.get_agent(
        "session_B",
        "company",
    )

    assert tech is not company
    assert tech["knowledge_base_id"] == "tech"
    assert company["knowledge_base_id"] == "company"


def test_session_ttl():
    """TTL 后应重新创建 Agent，但保留知识库绑定。"""

    now = [100.0]

    manager = SessionManager(
        agent_factory=make_factory(),
        ttl_seconds=10,
        clock=lambda: now[0],
    )

    old_agent = manager.get_agent(
        "session_A",
        "tech",
    )

    now[0] = 111.0

    new_agent = manager.get_agent(
        "session_A",
        "tech",
    )

    assert old_agent is not new_agent

    with pytest.raises(
        SessionKnowledgeBaseConflictError
    ):
        manager.get_agent(
            "session_A",
            "company",
        )


def test_same_session_concurrent_rejected():
    """同 Session 的第二个请求应立即失败。"""

    manager = SessionManager(make_factory())

    with manager.use_agent("session_A", "tech"):

        with pytest.raises(SessionBusyError):

            with manager.use_agent(
                "session_A",
                "tech",
            ):
                pass


def test_different_sessions_can_run_together():
    """不同 Session 不应该共用请求锁。"""

    manager = SessionManager(make_factory())

    with manager.use_agent(
        "session_A",
        "tech",
    ) as agent_a:

        with manager.use_agent(
            "session_B",
            "company",
        ) as agent_b:

            assert agent_a is not agent_b
            assert agent_a["knowledge_base_id"] == "tech"
            assert agent_b["knowledge_base_id"] == "company"
