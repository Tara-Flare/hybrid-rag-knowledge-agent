"""DesktopAgent 在线 Grounding 集成测试。"""

from types import SimpleNamespace
from unittest.mock import Mock

from agent_core.agent import DesktopAgent
from rag.grounding_checker import GroundingCheckError
from storage.request_trace import RequestTrace
from storage.trace_context import bind_request_trace
from tools.tool_result import ToolResult


RECORDS = [
    {
        "chunk_id": "leave_policy.md::chunk-0",
        "content": "员工年假需提前2个工作日申请。",
    }
]


def make_agent(passed=True, error=None):
    """创建无需真实模型的测试 Agent。"""

    agent = DesktopAgent.__new__(DesktopAgent)

    if error is None:
        checker = Mock()
        checker.check.return_value = SimpleNamespace(
            passed=passed
        )
    else:
        checker = Mock()
        checker.check.side_effect = error

    agent.grounding_checker = checker
    return agent


def test_grounded_answer_allowed():
    agent = make_agent(passed=True)

    answer = "员工年假需提前2个工作日申请。"

    result = agent._verify_grounded_answer(
        answer,
        RECORDS,
    )

    assert result == answer
    agent.grounding_checker.check.assert_called_once()


def test_unsupported_answer_blocked():
    agent = make_agent(passed=False)

    answer = "员工每年最多只能申请2次年假。"

    trace = RequestTrace(
        session_id="grounding_test",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        result = agent._verify_grounded_answer(
            answer,
            RECORDS,
        )

    assert result != answer
    assert "不返回原答案" in result

    phases = [
        p for p in trace.phases
        if p.name == "grounding_check"
    ]

    assert len(phases) == 1
    assert phases[0].success is True
    assert phases[0].outcome == "rejected"


def test_grounding_error_fails_closed():
    agent = make_agent(
        error=GroundingCheckError("模拟核验超时")
    )

    answer = "这是一条尚未验证的回答。"

    trace = RequestTrace(
        session_id="grounding_error",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        result = agent._verify_grounded_answer(
            answer,
            RECORDS,
        )

    assert result != answer
    assert "未经验证" in result

    phases = [
        p for p in trace.phases
        if p.name == "grounding_check"
    ]

    assert len(phases) == 1
    assert phases[0].success is False
    assert phases[0].error_type == "GroundingCheckError"


def test_chat_invokes_grounding():
    """模拟完整 Function Calling Loop，验证最终答案必经核验。"""

    agent = make_agent(passed=False)

    agent.max_steps = 2
    agent.messages = []
    agent.memory = Mock()
    agent.tools = []
    agent._last_tool_result = None

    agent._extract_memory_delete_key = lambda text: None
    agent._is_explicit_knowledge_request = lambda text: True

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="search_knowledge",
            arguments='{"query": "年假如何申请"}',
        ),
    )

    def make_message(content, calls):
        return SimpleNamespace(
            content=content,
            tool_calls=calls,
            model_dump=lambda **kwargs: {
                "role": "assistant",
                "content": content,
            },
        )

    first_message = make_message(
        "",
        [tool_call],
    )

    final_answer = "员工每年最多只能申请2次年假。"

    final_message = make_message(
        final_answer,
        None,
    )

    agent.client = Mock()

    agent.client.chat.side_effect = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=first_message)
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=final_message)
            ]
        ),
    ]

    def fake_run_tool(name, arguments):
        assert name == "search_knowledge"

        agent._last_tool_result = ToolResult.ok(
            message="检索成功",
            data={"results": RECORDS},
        )

        return agent._last_tool_result.to_text()

    agent._run_tool = fake_run_tool

    result = agent.chat(
        "根据知识库，年假如何申请？"
    )

    # 证明 chat() 确实执行了 Grounding
    agent.grounding_checker.check.assert_called_once()

    # 不支持的原始答案不能返回
    assert result != final_answer
    assert "不返回原答案" in result

    # 不能把未经核验的原答案存入聊天历史
    assert agent.messages[-1]["content"] == result

    agent.memory.save.assert_called_once()
