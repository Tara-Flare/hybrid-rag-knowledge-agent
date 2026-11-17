
"""测试 RAG 回答行为分类器。"""

from eval.score_answers import classify_answer


def test_classify_normal_answer():
    """有明确引用的回答应被判为 answer。"""

    answer = (
        "限制 max_steps 可以避免无限调用工具。\n"
        "来源文件：agent_notes.md；"
        "Chunk ID：agent_notes.md::chunk-0"
    )

    result = classify_answer(answer)

    assert result == "answer"


def test_classify_refusal():
    """证据不足时应被判为 refuse。"""

    answer = (
        "资料没有明确提及：MemoryNotFoundError。"
        "证据不足。"
    )

    result = classify_answer(answer)

    assert result == "refuse"


def test_classify_unknown():
    """无法识别的回答应被判为 unknown。"""

    answer = "我暂时无法确认这个问题。"

    result = classify_answer(answer)

    assert result == "unknown"


def test_refusal_has_priority():
    """同时包含拒答标记和引用时，优先判为拒答。"""

    answer = (
        "证据不足。\n"
        "来源文件：agent_notes.md；"
        "Chunk ID：agent_notes.md::chunk-0"
    )

    result = classify_answer(answer)

    assert result == "refuse"
