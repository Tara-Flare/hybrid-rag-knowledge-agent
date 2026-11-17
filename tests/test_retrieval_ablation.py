"""检索消融实验指标测试。"""

import pytest

from eval.retrieval_ablation import (
    extract_ids,
    score_rankings,
)


def test_extract_ids_deduplicates():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]

    assert extract_ids(results, top_k=3) == [
        "a", "b", "c"
    ]


def test_fact_coverage_and_mrr():
    rows = [
        {
            "ranked_chunk_ids": ["a", "c", "x"],
            "latency_ms": 100,
            "fact_labels": [
                {
                    "acceptable_chunk_ids": [
                        "a", "a_alt"
                    ],
                },
                {
                    "acceptable_chunk_ids": ["c"],
                },
            ],
        },
        {
            "ranked_chunk_ids": ["x", "d", "y"],
            "latency_ms": 200,
            "fact_labels": [
                {
                    "acceptable_chunk_ids": ["d"],
                },
                {
                    "acceptable_chunk_ids": ["z"],
                },
            ],
        },
    ]

    metrics = score_rankings(rows)

    assert metrics["gold_facts_total"] == 4
    assert metrics["gold_facts_hit"] == 3

    assert metrics["gold_fact_recall_at_3"] == (
        0.75
    )

    assert metrics[
        "complete_evidence_rate_at_3"
    ] == 0.5

    # 第一题首次命中第1名，第二题第2名
    assert metrics["mrr_at_3"] == pytest.approx(
        0.75
    )

    assert metrics["latency_p50_ms"] == 100
    assert metrics["latency_p95_ms"] == 200


def test_get_chunk_id_from_metadata():
    """兼容 Dense SearchResult 的 metadata 格式。"""

    from types import SimpleNamespace
    from eval.retrieval_ablation import get_chunk_id

    result = SimpleNamespace(
        text="员工原则上提前2个工作日申请年假",
        score=0.91,
        metadata={
            "chunk_id": "leave_policy.md::chunk-0",
            "source": "leave_policy.md",
        },
    )

    assert get_chunk_id(result) == (
        "leave_policy.md::chunk-0"
    )


def test_get_chunk_id_from_top_level():
    """BM25 和 RRF 的顶层 ID 格式继续兼容。"""

    from types import SimpleNamespace
    from eval.retrieval_ablation import get_chunk_id

    assert get_chunk_id({
        "chunk_id": "policy::chunk-0"
    }) == "policy::chunk-0"

    result = SimpleNamespace(
        chunk_id="policy::chunk-1"
    )

    assert get_chunk_id(result) == (
        "policy::chunk-1"
    )


def test_get_chunk_id_rejects_missing_id():
    """缺失真实 ID 时必须明确报错，不能伪造。"""

    from types import SimpleNamespace
    from eval.retrieval_ablation import get_chunk_id

    result = SimpleNamespace(
        text="测试内容",
        metadata={"source": "policy.md"},
    )

    with pytest.raises(
        ValueError,
        match="缺少有效 chunk_id",
    ):
        get_chunk_id(result)


def test_real_search_result_nested_chunk():
    """兼容项目真实 SearchResult 的嵌套结构。"""

    from types import SimpleNamespace

    from rag.vector_store import SearchResult
    from eval.retrieval_ablation import get_chunk_id

    chunk = SimpleNamespace(
        chunk_id="leave_policy.md::chunk-0",
        content="正式员工享有5个工作日基础年假",
        source="leave_policy.md",
    )

    result = SearchResult(
        chunk=chunk,
        score=0.95,
        rank=1,
    )

    assert get_chunk_id(result) == (
        "leave_policy.md::chunk-0"
    )


def test_filter_positive_bm25():
    """零分或负分候选不得参与 BM25 排名。"""

    from types import SimpleNamespace

    from eval.retrieval_ablation import (
        filter_positive_bm25,
    )

    results = [
        SimpleNamespace(chunk_id="a", score=2.5),
        SimpleNamespace(chunk_id="b", score=0.0),
        SimpleNamespace(chunk_id="c", score=-0.5),
        SimpleNamespace(chunk_id="d", score=0.8),
    ]

    filtered = filter_positive_bm25(results)

    assert [
        item.chunk_id for item in filtered
    ] == ["a", "d"]


def test_cached_rrf_matches_original_hybrid():
    """共享候选 RRF 与原 Hybrid 排名保持一致。"""

    from rag.bm25_retriever import BM25SearchResult
    from rag.hybrid_retriever import HybridRetriever
    from eval.retrieval_ablation import (
        fuse_cached_results,
        get_chunk_id,
    )

    def item(chunk_id, score):
        return BM25SearchResult(
            chunk_id=chunk_id,
            text=f"{chunk_id} 的内容",
            source="policy.md",
            score=score,
            metadata={},
        )

    dense_items = [
        item("a", 0.9),
        item("b", 0.8),
        item("c", 0.7),
    ]

    bm25_items = [
        item("b", 2.0),
        item("a", 1.0),
        item("d", 0.0),
    ]

    class FakeRetriever:
        def __init__(self, results):
            self.results = results
            self.calls = 0

        def search(self, query, top_k):
            self.calls += 1
            return self.results[:top_k]

    dense = FakeRetriever(dense_items)
    bm25 = FakeRetriever(bm25_items)

    original = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_top_k=5,
        bm25_top_k=5,
        rrf_k=60,
    )

    expected = original.search(
        query="测试制度",
        top_k=3,
    )

    actual = fuse_cached_results(
        dense_results=dense_items,
        bm25_results=bm25_items,
        top_k=3,
        rrf_k=60,
    )

    assert [
        get_chunk_id(x) for x in actual
    ] == [
        get_chunk_id(x) for x in expected
    ]

    assert dense.calls == 1
    assert bm25.calls == 1
