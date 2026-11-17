
"""候选事实句提取测试。"""

from eval.claim_extraction import (
    extract_candidate_claims,
)


def test_multiple_sentences():
    """可以把多个句子分开。"""

    answer = (
        "Agent 可以调用工具。"
        "max_steps 可以避免无限循环。"
    )

    claims = extract_candidate_claims(answer)

    assert len(claims) == 2
    assert claims[0] == "Agent 可以调用工具"
    assert claims[1] == "max_steps 可以避免无限循环"


def test_remove_citation_metadata():
    """来源文件和 Chunk ID 不应成为事实句。"""

    answer = (
        "Agent 可以调用工具。\n"
        "来源文件：agent_notes.md；"
        "Chunk ID：agent_notes.md::chunk-0"
    )

    claims = extract_candidate_claims(answer)

    assert claims == ["Agent 可以调用工具"]


def test_empty_answer():
    """空回答没有候选事实句。"""

    assert extract_candidate_claims("") == []
    assert extract_candidate_claims("   ") == []
