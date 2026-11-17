"""离线 Gold Fact Benchmark 集成测试。"""

import json

import pytest

from eval.score_benchmark import (
    grade_records,
    summarize_gold,
    validate_and_join,
)


def make_case(case_id, behavior, facts=None):
    return {
        "id": case_id,
        "question": f"问题-{case_id}",
        "knowledge_base_id": "company",
        "expected_behavior": behavior,
        "gold_assertions": [
            {"fact": fact}
            for fact in (facts or [])
        ],
        "missing_information": None,
    }


def make_result(case, answer, predicted, status="success"):
    return {
        "id": case["id"],
        "question": case["question"],
        "knowledge_base_id": "company",
        "expected_behavior": case["expected_behavior"],
        "status": status,
        "predicted_behavior": predicted,
        "answer": answer,
        "request_id": f"request_{case['id']}",
    }


def covered_judge(prompt):
    data = json.loads(
        prompt.split("待评测数据：\n", 1)[1]
    )

    return json.dumps({
        "judgments": [
            {
                "index": item["index"],
                "label": "covered",
                "reason": "模拟语义覆盖",
            }
            for item in data["gold_facts"]
        ],
        "missing_acknowledged": None,
    }, ensure_ascii=False)


def test_gold_scoring_full_pipeline():
    """完整连接结果文件与 Gold 事实评分。"""

    cases = [
        make_case("a", "answer", ["年假为5个工作日"]),
        make_case("b", "answer", ["报销限期15个自然日"]),
        make_case("c", "refuse"),
    ]

    results = [
        make_result(cases[0], "年假5个工作日", "answer"),
        make_result(cases[1], "证据不足", "refuse"),
        make_result(cases[2], "证据不足", "refuse"),
    ]

    joined, missing = validate_and_join(cases, results)

    calls = []

    def judge(prompt):
        calls.append(prompt)
        return covered_judge(prompt)

    graded = grade_records(joined, judge)
    summary = summarize_gold(cases, graded, missing)

    # 仅第一题需要模型评分
    assert len(calls) == 1

    assert graded[0]["gold_score"]["complete"] is True
    assert graded[1]["gold_score"]["status"] == "not_answered"
    assert graded[2]["gold_score"]["status"] == "not_applicable"

    assert summary["evaluation_complete"] is True
    assert summary["gold_fact_micro_coverage"] == 0.5
    assert summary["complete_answer_rate"] == 0.5


def test_grading_error_does_not_fake_score():
    """评分模型故障不能产生虚假总体指标。"""

    case = make_case("a", "answer", ["年假5天"])
    record = make_result(case, "年假5天", "answer")

    joined, missing = validate_and_join([case], [record])

    graded = grade_records(
        joined,
        lambda prompt: "not json",
    )

    summary = summarize_gold([case], graded, missing)

    assert graded[0]["gold_score"]["status"] == "judge_error"
    assert summary["evaluation_complete"] is False
    assert summary["gold_fact_micro_coverage"] is None
    assert summary["complete_answer_rate"] is None


def test_duplicate_or_wrong_knowledge_base_rejected():
    """不允许重复结果或跨知识库错配。"""

    case = make_case("a", "answer", ["年假5天"])
    record = make_result(case, "年假5天", "answer")

    with pytest.raises(ValueError):
        validate_and_join(
            [case],
            [record, record],
        )

    wrong_kb = dict(record)
    wrong_kb["knowledge_base_id"] = "tech"

    with pytest.raises(ValueError):
        validate_and_join(
            [case],
            [wrong_kb],
        )


def test_missing_cases_do_not_produce_final_metrics():
    """缺失测试题目时不输出虚假的完整评测结果。"""

    cases = [
        make_case("a", "answer", ["事实A"]),
        make_case("b", "answer", ["事实B"]),
    ]

    record = make_result(
        cases[0],
        "事实A",
        "answer",
    )

    joined, missing = validate_and_join(
        cases, [record]
    )

    assert missing == ["b"]

    graded = grade_records(
        joined,
        covered_judge,
    )

    summary = summarize_gold(
        cases, graded, missing
    )

    assert summary["evaluation_complete"] is False
    assert summary["missing_case_count"] == 1
    assert summary["gold_fact_micro_coverage"] is None
