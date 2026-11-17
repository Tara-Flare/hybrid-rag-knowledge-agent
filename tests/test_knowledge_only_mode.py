"""知识库受限执行模式测试，不调用真实模型。"""

from types import SimpleNamespace

import pytest

from agent_core.agent import DesktopAgent
from tools.tool_result import ToolResult


class FakeMemory:
    def get_recent_messages(self, messages):
        return messages[-10:]

    def save(self, messages):
        pass


def response(content=None, tool_name=None):
    if tool_name is None:
        calls = None
    else:
        calls = [
            SimpleNamespace(
                id="call_001",
                function=SimpleNamespace(
                    name=tool_name,
                    arguments="{}",
                ),
            )
        ]

    message = SimpleNamespace(
        content=content,
        tool_calls=calls,
        model_dump=lambda exclude_none=True: {
            "role": "assistant",
            "content": content,
        },
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message)
        ]
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.requests.append({
            "tools": tools,
            "tool_choice": tool_choice,
        })

        if not self.responses:
            raise AssertionError(
                "模型调用次数超过预期"
            )

        return self.responses.pop(0)


def make_agent(responses, explicit=False):
    agent = DesktopAgent.__new__(DesktopAgent)

    agent.max_steps = 3
    agent.messages = []
    agent.memory = FakeMemory()
    agent._last_tool_result = None

    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
            },
        },
    ]

    agent.client = FakeClient(responses)

    agent._extract_memory_delete_key = (
        lambda message: None
    )
    agent._is_explicit_knowledge_request = (
        lambda message: explicit
    )
    agent._build_system_message = (
        lambda: {
            "role": "system",
            "content": "通用 Agent",
        }
    )

    executed_tools = []

    def run_tool(name, arguments):
        executed_tools.append(name)

        agent._last_tool_result = ToolResult.ok(
            message="模拟检索成功",
            data={
                "results": [
                    {
                        "chunk_id": "policy::chunk-0",
                        "content": "模拟制度事实",
                    }
                ]
            },
        )

        return "模拟知识库证据"

    agent._run_tool = run_tool

    agent._verify_grounded_answer = (
        lambda answer, evidence, *, user_question="": answer
    )

    return agent, executed_tools


def test_knowledge_only_forces_search():
    """普通表述的知识库问题也必须先检索。"""

    agent, executed = make_agent([
        response(tool_name="search_knowledge"),
        response(
            content=(
                "根据制度，答案是模拟事实。\n"
                "来源文件：policy.md；"
                "Chunk ID：policy::chunk-0"
            )
        ),
    ])

    answer = agent.chat(
        "普通员工应该如何申请？",
        knowledge_only=True,
    )

    assert "模拟事实" in answer
    assert executed == ["search_knowledge"]

    requests = agent.client.requests

    assert len(requests) == 2

    first = requests[0]

    assert first["tool_choice"] == {
        "type": "function",
        "function": {
            "name": "search_knowledge",
        },
    }

    assert len(first["tools"]) == 1
    assert (
        first["tools"][0]["function"]["name"]
        == "search_knowledge"
    )

    # 第二轮生成最终答案，不能再使用工具。
    assert requests[1]["tools"] is None
    assert requests[1]["tool_choice"] == "none"


def test_forbidden_file_tool_not_executed():
    """模型即使违规返回文件工具，也不得实际执行。"""

    agent, executed = make_agent([
        response(tool_name="list_directory"),
    ])

    with pytest.raises(RuntimeError):
        agent.chat(
            "请解释公司制度",
            knowledge_only=True,
        )

    assert executed == []


def test_first_step_cannot_skip_retrieval():
    """严格模式下直接回答必须失败，不接受无依据输出。"""

    agent, executed = make_agent([
        response(
            content="根据我的常识，这应该是答案。"
        ),
    ])

    with pytest.raises(RuntimeError):
        agent.chat(
            "员工应当通知谁？",
            knowledge_only=True,
        )

    assert executed == []


def test_general_agent_keeps_original_behavior():
    """通用 Agent 不强制搜索知识库。"""

    agent, executed = make_agent([
        response(
            content="这是普通 Agent 的直接回答。"
        ),
    ])

    answer = agent.chat("你好")

    assert answer == "这是普通 Agent 的直接回答。"
    assert executed == []

    first = agent.client.requests[0]

    assert first["tool_choice"] == "auto"
    assert len(first["tools"]) == 2


def test_explicit_knowledge_request_still_works():
    """原有显式知识库模式继续兼容。"""

    agent, executed = make_agent(
        [
            response(tool_name="search_knowledge"),
            response(content="模拟最终答案"),
        ],
        explicit=True,
    )

    answer = agent.chat("根据我的知识库回答问题")

    assert answer == "模拟最终答案"
    assert executed == ["search_knowledge"]
    assert len(agent.client.requests[0]["tools"]) == 1


def test_insufficient_evidence_returns_refusal():
    """Answerability 证据不足应直接拒答，不触发 Grounding。"""

    from eval.score_answers import classify_answer

    agent, executed = make_agent([
        response(tool_name="search_knowledge"),
    ])

    refusal = (
        "知识库中没有足以回答此问题的依据。"
    )

    def failed_search(name, arguments):
        executed.append(name)

        agent._last_tool_result = ToolResult.fail(
            message=refusal,
            error_type="KnowledgeNotFoundError",
        )

        return agent._last_tool_result.to_text()

    def forbidden_grounding(*args, **kwargs):
        raise AssertionError(
            "证据不足时不允许继续执行 Grounding"
        )

    agent._run_tool = failed_search
    agent._verify_grounded_answer = forbidden_grounding

    answer = agent.chat(
        "公司每次报销的最高金额是多少？",
        knowledge_only=True,
    )

    assert answer == refusal
    assert executed == ["search_knowledge"]

    # 证据不足时直接返回，不需要第二次 LLM 调用
    assert len(agent.client.requests) == 1

    assert classify_answer(
        answer,
        tool_error_type=(
            agent._last_tool_result.error_type
        ),
    ) == "refuse"


def test_retrieval_technical_failure_is_not_refusal():
    """检索服务故障不能被算成正确拒答。"""

    from eval.score_answers import classify_answer

    agent, executed = make_agent([
        response(tool_name="search_knowledge"),
    ])

    error_message = (
        "知识库检索服务暂时不可用，请稍后重试。"
    )

    def unavailable_search(name, arguments):
        executed.append(name)

        agent._last_tool_result = ToolResult.fail(
            message=error_message,
            error_type="RetrievalUnavailableError",
        )

        return agent._last_tool_result.to_text()

    def forbidden_grounding(*args, **kwargs):
        raise AssertionError(
            "检索失败时不允许执行 Grounding"
        )

    agent._run_tool = unavailable_search
    agent._verify_grounded_answer = forbidden_grounding

    answer = agent.chat(
        "员工可以报销哪些费用？",
        knowledge_only=True,
    )

    assert answer == error_message
    assert executed == ["search_knowledge"]
    assert len(agent.client.requests) == 1

    assert classify_answer(
        answer,
        tool_error_type=(
            agent._last_tool_result.error_type
        ),
    ) == "error"
