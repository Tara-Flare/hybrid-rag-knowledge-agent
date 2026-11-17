"""Gold Fact Scoring 自动化测试。"""

import json

import pytest

from eval.gold_fact_scoring import (
    GoldFactScoringError,
    score_gold_answer,
)


def make_case(
    facts,
    missing_information=None,
    expected_behavior="answer",
):
    return {
        "id": "gold_case_001",
        "expected_behavior": expected_behavior,
        "gold_assertions": [
            {"fact": fact}
            for fact in facts
        ],
        "missing_information": missing_information,
    }


def fake_judge(labels, acknowledged=None):
    """模拟评分模型返回结构化结果。"""

    def generate(prompt):
        data = json.loads(
            prompt.split("待评测数据：\n", 1)[1]
        )

        judgments = [
            {
                "index": index,
                "label": label,
                "reason": f"模拟判断：{label}",
            }
            for index, label in enumerate(labels, 1)
        ]

        return json.dumps({
            "judgments": judgments,
            "missing_acknowledged": acknowledged,
        }, ensure_ascii=False)

    return generate


def test_all_gold_facts_covered():
    """全部事实覆盖时评分为 1.0。"""

    case = make_case([
        "年假至少提前2个工作日申请",
        "年假经直属主管批准后生效",
    ])

    result = score_gold_answer(
        case,
        "员工年假至少提前2个工作日申请，"
        "经过直属主管批准后生效。",
        fake_judge(["covered", "covered"]),
    )

    assert result["status"] == "scored"
    assert result["gold_fact_coverage"] == 1.0
    assert result["complete"] is True


def test_partial_fact_coverage():
    """遗漏一半 Gold 事实时只得 0.5。"""

    case = make_case([
        "需要提供费用说明",
        "需要提供有效票据",
    ])

    result = score_gold_answer(
        case,
        "报销需要提供费用说明。",
        fake_judge(["covered", "missing"]),
    )

    assert result["gold_fact_coverage"] == 0.5
    assert result["covered_count"] == 1
    assert result["missing_count"] == 1
    assert result["complete"] is False


def test_contradicted_fact_rejected():
    """与标准事实矛盾时不能视为正确。"""

    case = make_case([
        "报销需在15个自然日内提交",
    ])

    result = score_gold_answer(
        case,
        "报销必须在30个自然日内提交。",
        fake_judge(["contradicted"]),
    )

    assert result["gold_fact_coverage"] == 0.0
    assert result["contradicted_count"] == 1
    assert result["complete"] is False


def test_partial_answer_requires_missing_acknowledgment():
    """部分可回答问题必须承认信息缺失。"""

    case = make_case(
        ["年假原则上至少提前2个工作日申请"],
        missing_information="制度未规定年假结转",
    )

    result = score_gold_answer(
        case,
        "年假至少提前2个工作日申请。",
        fake_judge(
            ["covered"],
            acknowledged=False,
        ),
    )

    assert result["gold_fact_coverage"] == 1.0
    assert result["missing_acknowledged"] is False
    assert result["complete"] is False

    result2 = score_gold_answer(
        case,
        "年假至少提前2个工作日申请。"
        "关于结转，制度未说明。",
        fake_judge(
            ["covered"],
            acknowledged=True,
        ),
    )

    assert result2["complete"] is True


def test_refusal_cases_not_scored():
    """应拒答的问题不计算 Gold Fact Coverage。"""

    case = make_case(
        [],
        expected_behavior="refuse",
    )

    def must_not_call(prompt):
        raise AssertionError("拒答问题不应调用评分模型")

    result = score_gold_answer(
        case,
        "证据不足。",
        must_not_call,
    )

    assert result["status"] == "not_applicable"
    assert result["gold_fact_coverage"] is None


def test_empty_answer_is_not_success():
    """空答案不应调用模型，也不应获得分数。"""

    case = make_case([
        "正式员工基础年假为5个工作日",
    ])

    def must_not_call(prompt):
        raise AssertionError("空答案不应调用模型")

    result = score_gold_answer(
        case,
        "",
        must_not_call,
    )

    assert result["status"] == "not_answered"
    assert result["gold_fact_coverage"] == 0.0
    assert result["complete"] is False


@pytest.mark.parametrize(
    "bad_response",
    [
        "not json",
        '{"judgments":[]}',
        json.dumps({
            "judgments": [{
                "index": 2,
                "label": "covered",
                "reason": "错误编号",
            }]
        }),
        json.dumps({
            "judgments": [{
                "index": 1,
                "label": "excellent",
                "reason": "非法标签",
            }]
        }),
    ],
)
def test_invalid_judge_output_fails_closed(bad_response):
    """评分模型输出不合法时，不允许产生虚假高分。"""

    case = make_case([
        "基础年假为5个工作日",
    ])

    with pytest.raises(GoldFactScoringError):
        score_gold_answer(
            case,
            "基础年假为5个工作日。",
            lambda prompt: bad_response,
        )
