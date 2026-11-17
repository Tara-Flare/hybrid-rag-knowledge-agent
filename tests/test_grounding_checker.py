"""GroundingChecker 自动化测试。"""

import json
from types import SimpleNamespace

import pytest

from rag.grounding_checker import (
    GroundingChecker,
    GroundingCheckError,
)


def make_evidence():
    return [
        SimpleNamespace(
            chunk_id="leave_policy.md::chunk-0",
            text=(
                "员工年假需要提前2个工作日申请。"
                "申请经直属主管批准后生效。"
            ),
        )
    ]


def make_generator(claims, verdicts):
    """模拟两次 LLM 调用：拆分事实、判断事实。"""

    responses = iter([
        json.dumps(
            {"claims": claims},
            ensure_ascii=False,
        ),
        json.dumps(
            {"verdicts": verdicts},
            ensure_ascii=False,
        ),
    ])

    return lambda prompt: next(responses)


def test_grounding_all_supported():
    """全部事实有依据时允许通过。"""

    claim = "员工年假需要提前2个工作日申请"

    generator = make_generator(
        claims=[claim],
        verdicts=[
            {
                "claim": claim,
                "label": "supported",
                "supported_chunk_ids": [
                    "leave_policy.md::chunk-0"
                ],
                "reason": "原始证据明确说明申请时间",
            },
        ],
    )

    checker = GroundingChecker(
        llm_generate=generator
    )

    result = checker.check(
        answer=claim + "。",
        evidence=make_evidence(),
    )

    assert result.passed is True
    assert result.faithfulness == 1.0
    assert len(result.verdicts) == 1


def test_grounding_rejects_unsupported():
    """包含编造事实时不得通过。"""

    claims = [
        "员工年假需要提前2个工作日申请",
        "员工每年最多只能申请2次年假",
    ]

    generator = make_generator(
        claims=claims,
        verdicts=[
            {
                "claim": claims[0],
                "label": "supported",
                "supported_chunk_ids": [
                    "leave_policy.md::chunk-0"
                ],
                "reason": "原文明确支持",
            },
            {
                "claim": claims[1],
                "label": "unsupported",
                "supported_chunk_ids": [],
                "reason": "原文没有限制申请次数",
            },
        ],
    )

    checker = GroundingChecker(
        llm_generate=generator
    )

    result = checker.check(
        answer="。".join(claims) + "。",
        evidence=make_evidence(),
    )

    assert result.passed is False
    assert result.faithfulness == 0.5


def test_grounding_rejects_missing_claim():
    """LLM 漏掉某条事实时必须失败。"""

    claims = [
        "员工需要提前申请",
        "每年最多申请2次",
    ]

    generator = make_generator(
        claims=claims,
        verdicts=[
            {
                "claim": claims[0],
                "label": "supported",
                "supported_chunk_ids": [
                    "leave_policy.md::chunk-0"
                ],
                "reason": "有证据",
            },
        ],
    )

    checker = GroundingChecker(
        llm_generate=generator
    )

    with pytest.raises(GroundingCheckError):
        checker.check(
            answer="。".join(claims) + "。",
            evidence=make_evidence(),
        )


def test_grounding_rejects_fake_chunk_id():
    """模型不能引用不存在的 Chunk。"""

    claim = "员工需要提前申请"

    generator = make_generator(
        claims=[claim],
        verdicts=[
            {
                "claim": claim,
                "label": "supported",
                "supported_chunk_ids": [
                    "fake_policy.md::chunk-99"
                ],
                "reason": "模拟伪造引用",
            },
        ],
    )

    checker = GroundingChecker(
        llm_generate=generator
    )

    with pytest.raises(GroundingCheckError):
        checker.check(
            answer=claim + "。",
            evidence=make_evidence(),
        )


def test_grounding_model_error():
    """模型异常时必须失败，不能默认通过。"""

    def broken_model(prompt):
        raise TimeoutError("模拟模型超时")

    checker = GroundingChecker(
        llm_generate=broken_model
    )

    with pytest.raises(GroundingCheckError):
        checker.check(
            answer="员工需要提前申请。",
            evidence=make_evidence(),
        )
