"""知识库回答行为分类及评测指标测试。"""

from eval.score_answers import (
    classify_answer,
    score_behavior,
)
from eval.run_benchmark import summarize


def test_current_agent_response_classification():
    """覆盖目前 Agent 已有的真实返回类型。"""

    samples = {
        "正常回答": (
            "员工应提前2个工作日申请。\n"
            "来源文件：leave_policy.md；"
            "Chunk ID：leave_policy.md::chunk-0"
        ),
        "证据不足": (
            "知识库中没有足以回答此问题的依据。"
        ),
        "Grounding 拒答": (
            "当前答案包含无法从知识库证据确认的内容，"
            "因此暂不返回原答案。"
        ),
        "Grounding 异常": (
            "当前无法完成答案的事实依据核验，"
            "因此暂不返回未经验证的答案。"
        ),
        "检索服务异常": (
            "知识库检索失败，无法获得可靠资料。"
        ),
    }

    expected = {
        "正常回答": "answer",
        "证据不足": "refuse",
        "Grounding 拒答": "refuse",
        "Grounding 异常": "error",
        "检索服务异常": "error",
    }

    for name, answer in samples.items():
        assert classify_answer(answer) == expected[name]

    # 没有引用的普通文本暂不视为合格知识库回答
    assert classify_answer(
        "员工年假需要提前2个工作日申请。"
    ) == "unknown"


def test_structured_errors_override_answer_text():
    """优先使用执行状态，不能只依赖关键词。"""

    cited_answer = (
        "这是一个回答。\n"
        "来源文件：policy.md；Chunk ID：chunk-1"
    )

    assert classify_answer(
        cited_answer,
        status="error",
    ) == "error"

    assert classify_answer(
        cited_answer,
        tool_error_type="EvidenceCheckUnavailableError",
    ) == "error"

    assert classify_answer(
        cited_answer,
        tool_error_type="KnowledgeNotFoundError",
    ) == "refuse"

    assert classify_answer(
        cited_answer,
        grounding_failed=True,
    ) == "error"

    assert classify_answer(
        cited_answer,
        grounding_outcome="rejected",
    ) == "refuse"


def test_behavior_metric_denominators():
    """技术故障不能计作正确拒答或错误放行。"""

    records = [
        {
            "expected_behavior": "refuse",
            "status": "success",
            "answer": "证据不足，无法回答。",
        },
        {
            "expected_behavior": "refuse",
            "status": "success",
            "answer": (
                "错误放行的回答。"
                "来源文件：a.md；Chunk ID：a::1"
            ),
        },
        {
            "expected_behavior": "refuse",
            "status": "success",
            "answer": "知识库检索失败，无法获得资料。",
        },
        {
            "expected_behavior": "answer",
            "status": "success",
            "answer": (
                "正确行为的回答。"
                "来源文件：a.md；Chunk ID：a::1"
            ),
        },
        {
            "expected_behavior": "answer",
            "status": "success",
            "answer": "证据不足，无法回答。",
        },
        {
            "expected_behavior": "answer",
            "status": "error",
            "answer": None,
        },
    ]

    metrics = score_behavior(records)

    assert metrics["labeled_cases"] == 6
    assert metrics["correct_refusal_rate"] == 0.3333
    assert metrics["false_acceptance_rate"] == 0.3333
    assert metrics["answer_attempt_rate"] == 0.3333
    assert metrics["false_refusal_rate"] == 0.3333

    assert metrics["confusion_matrix"]["refuse"] == {
        "answer": 1,
        "refuse": 1,
        "error": 1,
        "unknown": 0,
    }


def test_no_gold_labels_returns_none():
    """无行为标签时，不能虚构可靠性指标。"""

    records = [
        {
            "expected_behavior": None,
            "status": "success",
            "answer": "证据不足。",
        }
    ]

    metrics = score_behavior(records)

    assert metrics["labeled_cases"] == 0
    assert metrics["unlabeled_cases"] == 1
    assert metrics["correct_refusal_rate"] is None
    assert metrics["false_acceptance_rate"] is None


def test_benchmark_summary_includes_behavior_metrics():
    """统一 Benchmark 必须输出行为评分。"""

    records = [
        {
            "expected_behavior": "refuse",
            "status": "success",
            "answer": "知识库中没有足以回答此问题的依据。",
            "duration_ms": 100.0,
            "token_usage": None,
            "answerability_outcome": "insufficient",
            "grounding_outcome": None,
        }
    ]

    summary = summarize(records)

    metrics = summary["behavior_metrics"]

    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["false_acceptance_rate"] == 0.0

    # 行为正确不等于事实准确率有 Gold 评分
    assert summary["answer_accuracy"] is None
