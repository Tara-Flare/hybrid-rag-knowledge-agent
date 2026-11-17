
"""SessionManager 单元测试。"""

from unittest.mock import Mock

from api.session_manager import SessionManager


def test_same_session_returns_same_agent():
    """同一个 session_id 应复用同一个对象。"""

    factory = Mock(
        side_effect=lambda session_id: object()
    )

    manager = SessionManager(factory)

    agent1 = manager.get_agent("session_A")
    agent2 = manager.get_agent("session_A")

    assert agent1 is agent2

    # 创建函数应该只调用一次
    factory.assert_called_once_with("session_A")


def test_different_sessions_return_different_agents():
    """不同 session_id 应获得不同对象。"""

    factory = Mock(
        side_effect=lambda session_id: object()
    )

    manager = SessionManager(factory)

    agent_a = manager.get_agent("session_A")
    agent_b = manager.get_agent("session_B")

    assert agent_a is not agent_b

    assert factory.call_count == 2
