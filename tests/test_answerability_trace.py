"""Answerability 阶段追踪测试。"""

from types import SimpleNamespace
from unittest.mock import Mock

from agent_core.agent import DesktopAgent

from rag.evidence_checker import (
    EvidenceCheckError,
    EvidenceCheckResult,
)

from storage.request_trace import RequestTrace
from storage.trace_context import bind_request_trace


def make_agent(checker):
    """构造不依赖真实模型的最小 Agent。"""

    agent = DesktopAgent.__new__(DesktopAgent)

    candidate = SimpleNamespace(
        chunk_id="policy.md::chunk-0",
        text="员工请假需要提前申请。",
        source="policy.md",
        rrf_score=0.03,
        dense_rank=1,
        bm25_rank=1,
        dense_score=0.8,
        bm25_score=1.0,
    )

    agent.hybrid_retriever = SimpleNamespace(
        search=Mock(return_value=[candidate])
    )

    # 不调用真实 Rerank
    agent.knowledge_reranker = None
    agent._reranker_init_error = "测试中禁用 Rerank"

    agent.evidence_checker = checker
    agent._checker_init_error = None

    return agent


def test_insufficient_evidence_trace():
    """证据不足属于正常业务判断，不是技术异常。"""

    checker = SimpleNamespace(
        check=Mock(
            return_value=EvidenceCheckResult(
                status="insufficient",
                reason="资料没有说明具体天数",
                supported_chunk_ids=[],
            )
        )
    )

    agent = make_agent(checker)

    trace = RequestTrace(
        session_id="trace_test_A",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        result = agent._search_knowledge(
            query="年假有多少天？",
            top_k=1,
        )

    assert result.success is False

    phases = [
        p for p in trace.phases
        if p.name == "answerability_check"
    ]

    assert len(phases) == 1

    phase = phases[0]

    assert phase.success is True
    assert phase.error_type is None
    assert phase.outcome == "insufficient"


def test_evidence_exception_trace():
    """证据核验异常应记录为技术失败。"""

    checker = SimpleNamespace(
        check=Mock(
            side_effect=EvidenceCheckError(
                "模拟核验模型异常"
            )
        )
    )

    agent = make_agent(checker)

    trace = RequestTrace(
        session_id="trace_test_B",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        result = agent._search_knowledge(
            query="年假有多少天？",
            top_k=1,
        )

    assert result.success is False
    assert result.error_type == (
        "EvidenceCheckUnavailableError"
    )

    phases = [
        p for p in trace.phases
        if p.name == "answerability_check"
    ]

    assert len(phases) == 1

    phase = phases[0]

    assert phase.success is False
    assert phase.error_type == "EvidenceCheckError"
    assert phase.outcome is None
