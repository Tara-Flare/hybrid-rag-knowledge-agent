"""知识库无答案与证据充分性测试。"""

from unittest.mock import Mock

from agent_core.agent import DesktopAgent

from rag.evidence_checker import (
    EvidenceCheckError,
    EvidenceCheckResult,
)


def make_candidate(
    text: str,
    chunk_id: str = "chunk-001",
) -> Mock:
    """构造模拟检索结果。"""

    candidate = Mock()

    candidate.chunk_id = chunk_id
    candidate.source = "knowledge/demo.md"
    candidate.text = text

    candidate.rrf_score = 0.02
    candidate.dense_rank = 1
    candidate.bm25_rank = 1
    candidate.dense_score = 0.75
    candidate.bm25_score = 3.0

    return candidate


def make_agent(
    candidates: list[Mock],
) -> DesktopAgent:
    """创建不调用真实模型的 Agent。"""

    agent = DesktopAgent.__new__(
        DesktopAgent
    )

    agent.hybrid_retriever = Mock()

    agent.hybrid_retriever.search.return_value = (
        candidates
    )

    agent.knowledge_reranker = None

    agent._reranker_init_error = (
        "测试中不启用 Rerank"
    )

    agent.evidence_checker = Mock()

    return agent


def test_no_candidates_returns_not_found():
    """没有候选资料时应该拒答。"""

    agent = make_agent([])

    result = agent._search_knowledge(
        "不存在的术语"
    )

    assert result.success is False

    assert (
        result.error_type
        == "KnowledgeNotFoundError"
    )

    assert result.data["results"] == []

    agent.evidence_checker.check.assert_not_called()


def test_irrelevant_candidates_should_not_be_accepted():
    """缺少精确错误名称时应该拒答。"""

    agent = make_agent(
        [
            make_candidate(
                "SQLite 可以存储工具日志。"
            )
        ]
    )

    result = agent._search_knowledge(
        "MemoryNotFoundError 是什么意思？"
    )

    assert result.success is False

    assert (
        result.error_type
        == "KnowledgeNotFoundError"
    )

    agent.evidence_checker.check.assert_not_called()


def test_relevant_candidates_should_be_accepted():
    """包含充分证据时应该接受资料。"""

    text = (
        "MemoryNotFoundError 表示"
        "被访问的长期记忆不存在。"
    )

    agent = make_agent(
        [
            make_candidate(text)
        ]
    )

    agent.evidence_checker.check.return_value = (
        EvidenceCheckResult(
            status="sufficient",
            reason="有明确的定义",
            supported_chunk_ids=["chunk-001"],
        )
    )

    result = agent._search_knowledge(
        "MemoryNotFoundError 是什么意思？"
    )

    assert result.success is True

    assert (
        result.data["evidence_status"]
        == "sufficient"
    )

    assert "MemoryNotFoundError" in result.message
    assert "chunk-001" in result.message


def test_identifier_without_explanation_should_be_rejected():
    """仅提到错误名称，没有解释时应该拒答。"""

    text = (
        "今天修复了 MemoryNotFoundError，"
        "并完成 SQLite 日志测试。"
    )

    agent = make_agent(
        [
            make_candidate(text)
        ]
    )

    agent.evidence_checker.check.return_value = (
        EvidenceCheckResult(
            status="insufficient",
            reason="只提及错误名称，没有解释",
            supported_chunk_ids=[],
        )
    )

    result = agent._search_knowledge(
        "MemoryNotFoundError 是什么意思？"
    )

    assert result.success is False

    assert (
        result.error_type
        == "KnowledgeNotFoundError"
    )


def test_partial_evidence_returns_only_supported_chunks():
    """部分充分时只能使用有支持依据的 Chunk。"""

    agent = make_agent(
        [
            make_candidate(
                "问题 A 的答案是 X",
                "chunk-a",
            ),
            make_candidate(
                "无关的说明",
                "chunk-b",
            ),
        ]
    )

    agent.evidence_checker.check.return_value = (
        EvidenceCheckResult(
            status="partial",
            reason="只能回答 A",
            supported_chunk_ids=["chunk-a"],
        )
    )

    result = agent._search_knowledge(
        "解释 A 和 B"
    )

    assert result.success is True

    assert (
        result.data["evidence_status"]
        == "partial"
    )

    assert [
        item["chunk_id"]
        for item in result.data["results"]
    ] == ["chunk-a"]


def test_checker_failure_is_not_knowledge_not_found():
    """证据核验故障不能被当成知识库无答案。"""

    agent = make_agent(
        [
            make_candidate(
                "MemoryNotFoundError 已修复。"
            )
        ]
    )

    agent.evidence_checker.check.side_effect = (
        EvidenceCheckError("API 超时")
    )

    result = agent._search_knowledge(
        "MemoryNotFoundError 是什么意思？"
    )

    assert result.success is False

    assert (
        result.error_type
        == "EvidenceCheckUnavailableError"
    )