"""Grounding 自动重写与二次核验测试。"""

from unittest.mock import Mock

from agent_core.agent import DesktopAgent
from rag.grounding_checker import (
    ClaimVerdict,
    GroundingCheckResult,
)


RECORDS = [
    {
        "chunk_id": "leave_policy.md::chunk-0",
        "content": "员工年假需提前2个工作日申请。",
    }
]

QUESTION = "员工如何申请年假？"

ORIGINAL = "员工每年最多只能申请2次年假。"

REWRITTEN = "员工年假需提前2个工作日申请。"


def make_result(passed: bool) -> GroundingCheckResult:
    """构造真实结构的 Grounding 核验结果。"""

    claim = REWRITTEN if passed else ORIGINAL

    return GroundingCheckResult(
        passed=passed,
        faithfulness=1.0 if passed else 0.0,
        claims=[claim],
        verdicts=[
            ClaimVerdict(
                claim=claim,
                label=(
                    "supported"
                    if passed
                    else "unsupported"
                ),
                supported_chunk_ids=(
                    ["leave_policy.md::chunk-0"]
                    if passed
                    else []
                ),
                reason=(
                    "原始证据支持"
                    if passed
                    else "原始证据未说明申请次数"
                ),
            )
        ],
    )


def make_agent(*results) -> DesktopAgent:
    """构造不调用真实模型的 Agent。"""

    agent = DesktopAgent.__new__(DesktopAgent)

    agent.grounding_checker = Mock()
    agent.grounding_checker.check.side_effect = list(results)

    agent._rewrite_grounded_answer = Mock(
        return_value=REWRITTEN
    )

    return agent


def test_rewrite_success():
    """首次不通过，重写后通过。"""

    agent = make_agent(
        make_result(False),
        make_result(True),
    )

    answer = agent._verify_grounded_answer(
        ORIGINAL,
        RECORDS,
        user_question=QUESTION,
    )

    assert answer == REWRITTEN

    agent._rewrite_grounded_answer.assert_called_once()

    kwargs = (
        agent._rewrite_grounded_answer.call_args.kwargs
    )

    assert kwargs["user_question"] == QUESTION
    assert kwargs["original_answer"] == ORIGINAL
    assert kwargs["unsupported_claims"] == [ORIGINAL]

    assert agent.grounding_checker.check.call_count == 2

    second_call = (
        agent.grounding_checker.check.call_args_list[1]
    )

    assert second_call.kwargs["answer"] == REWRITTEN


def test_second_grounding_failure():
    """重写后仍未通过，必须拒答。"""

    agent = make_agent(
        make_result(False),
        make_result(False),
    )

    answer = agent._verify_grounded_answer(
        ORIGINAL,
        RECORDS,
        user_question=QUESTION,
    )

    assert answer != ORIGINAL
    assert answer != REWRITTEN

    assert "不返回原答案" in answer

    agent._rewrite_grounded_answer.assert_called_once()

    assert agent.grounding_checker.check.call_count == 2


def test_rewrite_exception_fails_closed():
    """重写异常时，不得返回未经验证的答案。"""

    agent = make_agent(make_result(False))

    agent._rewrite_grounded_answer.side_effect = (
        RuntimeError("模拟重写服务失败")
    )

    answer = agent._verify_grounded_answer(
        ORIGINAL,
        RECORDS,
        user_question=QUESTION,
    )

    assert answer != ORIGINAL
    assert "未经验证" in answer

    agent._rewrite_grounded_answer.assert_called_once()

    # 重写失败后不应执行第二次 Grounding。
    agent.grounding_checker.check.assert_called_once()
