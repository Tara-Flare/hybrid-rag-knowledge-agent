"""AgentResponse 结构化失败场景测试。"""

from agent_core.agent import DesktopAgent
from agent_core.models import ToolCallInfo
from agent_core.response_context import (
    ACTIVE_AGENT_RUN_CAPTURE,
)


def make_agent(fake_chat):
    agent = DesktopAgent.__new__(DesktopAgent)
    agent.chat = fake_chat
    return agent


def test_tool_failure_is_recorded():
    """工具失败必须反映在结构化响应中。"""

    def fake_chat(message, *, knowledge_only=False):
        capture = ACTIVE_AGENT_RUN_CAPTURE.get()

        capture.tool_calls.append(
            ToolCallInfo(
                tool_name="calculator",
                success=False,
                error_type="ValueError",
            )
        )

        return "工具参数错误"

    agent = make_agent(fake_chat)

    result = agent.chat_detailed("测试工具失败")

    assert result.success is False
    assert result.error_type == "ValueError"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "calculator"
    assert ACTIVE_AGENT_RUN_CAPTURE.get() is None


def test_knowledge_without_grounding_sources_fails():
    """知识库回答缺少核验通过的来源时不能标记成功。"""

    def fake_chat(message, *, knowledge_only=False):
        capture = ACTIVE_AGENT_RUN_CAPTURE.get()

        capture.tool_calls.append(
            ToolCallInfo(
                tool_name="search_knowledge",
                success=True,
            )
        )

        return "当前知识库证据不足"

    agent = make_agent(fake_chat)

    result = agent.chat_detailed(
        "根据知识库回答问题",
        knowledge_only=True,
    )

    assert result.success is False
    assert result.error_type == "GroundingNotVerifiedError"
    assert result.sources == []
    assert len(result.tool_calls) == 1
    assert ACTIVE_AGENT_RUN_CAPTURE.get() is None


def test_agent_exception_fails_closed():
    """Agent 异常时安全返回，且请求上下文必须恢复。"""

    def fake_chat(message, *, knowledge_only=False):
        raise RuntimeError("内部模拟异常")

    agent = make_agent(fake_chat)

    result = agent.chat_detailed("测试异常")

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert result.answer == "Agent 处理请求失败。"
    assert "内部模拟异常" not in result.answer
    assert result.sources == []
    assert ACTIVE_AGENT_RUN_CAPTURE.get() is None
