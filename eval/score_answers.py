
"""RAG 回答行为评测：第一版。"""

import json
from pathlib import Path


RESULTS_PATH = Path("eval/answer_results.jsonl")


REFUSAL_PREFIXES = (
    "资料没有明确提及",
    "证据不足",
    "没有足够证据",
    "知识库中没有足以回答此问题的依据",
    "知识库中没有找到足够相关的资料",
    "知识库中没有找到可以用于回答问题的资料",
    "当前答案包含无法从知识库证据确认的内容",
)

ERROR_PREFIXES = (
    "知识库检索失败",
    "混合检索服务不可用",
    "证据核验服务未初始化",
    "证据核验暂时失败",
    "证据核验发生异常",
    "证据核验没有提供可用的来源",
    "当前无法完成答案的事实依据核验",
)

PREDICTIONS = ("answer", "refuse", "error", "unknown")
EXPECTED_LABELS = ("answer", "refuse")


def classify_answer(
    answer: str | None,
    *,
    status: str | None = None,
    tool_error_type: str | None = None,
    grounding_outcome: str | None = None,
    grounding_failed: bool = False,
) -> str:
    """分类知识库回答行为，优先使用结构化执行信息。

    answer: 有引用的正常回答
    refuse: 正常证据不足或 Grounding 拒答
    error: 技术执行故障
    unknown: 无法确定
    """

    if status == "error" or grounding_failed:
        return "error"

    if tool_error_type:
        if tool_error_type == "KnowledgeNotFoundError":
            return "refuse"
        return "error"

    if not isinstance(answer, str) or not answer.strip():
        return "unknown"

    text = answer.strip()

    if text.startswith(ERROR_PREFIXES):
        return "error"

    if grounding_outcome == "rejected":
        return "refuse"

    if text.startswith(REFUSAL_PREFIXES):
        return "refuse"

    # 正常知识库回答必须包含基本引用标记。
    # 这只验证引用格式存在，不代表引用内容正确。
    if (
        "来源文件：" in text
        and "Chunk ID：" in text
    ):
        return "answer"

    return "unknown"


def score_behavior(records: list[dict]) -> dict:
    """计算回答行为指标，不将技术错误计为正确拒答。"""

    matrix = {
        expected: {
            predicted: 0
            for predicted in PREDICTIONS
        }
        for expected in EXPECTED_LABELS
    }

    unlabeled = 0

    for record in records:
        expected = record.get("expected_behavior")

        if expected not in EXPECTED_LABELS:
            unlabeled += 1
            continue

        predicted = classify_answer(
            record.get("answer"),
            status=record.get("status"),
            tool_error_type=record.get("tool_error_type"),
            grounding_outcome=record.get("grounding_outcome"),
            grounding_failed=bool(
                record.get("grounding_failed", False)
            ),
        )

        matrix[expected][predicted] += 1

    answer_total = sum(matrix["answer"].values())
    refuse_total = sum(matrix["refuse"].values())
    labeled_total = answer_total + refuse_total

    def rate(numerator: int, denominator: int):
        if denominator == 0:
            return None
        return round(numerator / denominator, 4)

    correct_refusals = matrix["refuse"]["refuse"]
    false_acceptances = matrix["refuse"]["answer"]
    correct_answer_behavior = matrix["answer"]["answer"]
    false_refusals = matrix["answer"]["refuse"]

    return {
        "labeled_cases": labeled_total,
        "unlabeled_cases": unlabeled,
        "confusion_matrix": matrix,
        "correct_refusal_rate": rate(
            correct_refusals, refuse_total
        ),
        "false_acceptance_rate": rate(
            false_acceptances, refuse_total
        ),
        "answer_attempt_rate": rate(
            correct_answer_behavior, answer_total
        ),
        "false_refusal_rate": rate(
            false_refusals, answer_total
        ),
        "behavior_match_rate": rate(
            correct_refusals + correct_answer_behavior,
            labeled_total,
        ),
        "note": (
            "unknown 和 error 保留在分母中；"
            "answer 仅表示有引用的回答行为，"
            "不表示事实正确。"
        ),
    }


def main():
    """读取评测结果并计算行为匹配率。"""

    results = []

    with RESULTS_PATH.open(
        "r", encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        raise ValueError("评测结果文件为空")

    matched_count = 0

    for result in results:
        expected = result["expected_behavior"]

        predicted = classify_answer(
            result["answer"]
        )

        is_match = predicted == expected

        if is_match:
            matched_count += 1

        print(f"\n测试编号：{result['id']}")
        print("预期行为：", expected)
        print("实际分类：", predicted)
        print("是否匹配：", is_match)

    # 计算行为匹配率
    match_rate = (
        matched_count / len(results)
    )

    print("\n========== 评测汇总 ==========")
    print("总测试数：", len(results))
    print("匹配数量：", matched_count)
    print(f"行为匹配率：{match_rate:.2%}")


if __name__ == "__main__":
    main()
