"""LLM Reranker 超时配置、Token 追踪与降级测试。"""

import json
from types import SimpleNamespace

from rag import llm_reranker as reranker_module
from rag.llm_reranker import LLMReranker
from agent_core.agent import DesktopAgent

from storage.request_trace import RequestTrace
from storage.trace_context import (
    bind_request_trace,
    observed_phase,
)


def test_reranker_timeout_and_retries(monkeypatch):
    """Reranker 应限制等待时间并关闭自动重试。"""

    monkeypatch.setenv(
        "DASHSCOPE_API_KEY",
        "unit-test-placeholder",
    )
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://example.invalid/v1",
    )

    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        reranker_module,
        "OpenAI",
        fake_openai,
    )

    LLMReranker()

    assert captured["timeout"] == 20.0
    assert captured["max_retries"] == 0


def test_reranker_token_usage_is_traced():
    """Rerank 模型 Token 应进入全局和阶段统计。"""

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "ranking": [
                            "doc::chunk-2",
                            "doc::chunk-1",
                        ],
                        "reason": "第二个片段更相关",
                    }, ensure_ascii=False)
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
        ),
    )

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return response

    reranker = LLMReranker.__new__(
        LLMReranker
    )

    reranker.model = "mock-qwen"
    reranker.max_chars_per_chunk = 800
    reranker.last_reason = ""

    reranker.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=fake_completion
            )
        )
    )

    candidates = [
        {
            "chunk_id": "doc::chunk-1",
            "source": "policy.md",
            "text": "无关的制度片段",
        },
        {
            "chunk_id": "doc::chunk-2",
            "source": "policy.md",
            "text": "相关的制度片段",
        },
    ]

    trace = RequestTrace(
        session_id="reranker_test",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        with observed_phase("llm_rerank") as phase:
            results = reranker.rerank(
                "制度是什么？",
                candidates,
                top_k=2,
            )

    assert len(calls) == 1

    assert results[0]["chunk_id"] == (
        "doc::chunk-2"
    )

    assert trace.token_usage["model_calls"] == 1
    assert trace.token_usage["prompt_tokens"] == 120
    assert trace.token_usage["completion_tokens"] == 30
    assert trace.token_usage["total_tokens"] == 150

    assert phase is not None
    assert phase.success is True
    assert phase.token_usage["total_tokens"] == 150


def test_reranker_timeout_falls_back_to_rrf():
    """Rerank 超时时，Agent 应继续使用 RRF 候选。"""

    candidate = SimpleNamespace(
        chunk_id="policy::chunk-0",
        text="正式员工每年有5个工作日基础年假。",
        source="leave_policy.md",
        rrf_score=0.03,
        dense_rank=1,
        bm25_rank=1,
        dense_score=0.9,
        bm25_score=0.8,
    )

    class FakeHybridRetriever:
        def search(self, query, top_k):
            return [candidate]

    class TimeoutReranker:
        last_reason = ""

        def rerank(self, query, candidates, top_k):
            raise TimeoutError("模拟 Rerank 超时")

    class FakeEvidenceChecker:
        def check(self, query, candidates):
            return SimpleNamespace(
                status="sufficient",
                reason="证据足够",
                supported_chunk_ids=[
                    "policy::chunk-0"
                ],
            )

    agent = DesktopAgent.__new__(DesktopAgent)

    agent.hybrid_retriever = FakeHybridRetriever()
    agent.knowledge_reranker = TimeoutReranker()
    agent.evidence_checker = FakeEvidenceChecker()

    agent._reranker_init_error = None
    agent._eval_evidence = []

    result = agent._search_knowledge(
        "正式员工有多少天年假？",
        top_k=1,
    )

    assert result.success is True
    assert result.data["rerank_used"] is False

    assert result.data["retrieval_mode"] == (
        "dense_bm25_rrf_fallback"
    )

    assert "TimeoutError" in (
        result.data["rerank_error"]
    )

    assert (
        result.data["results"][0]["chunk_id"]
        == "policy::chunk-0"
    )
