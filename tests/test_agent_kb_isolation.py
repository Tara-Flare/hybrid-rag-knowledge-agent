"""DesktopAgent 多知识库隔离测试。"""

from pathlib import Path

import pytest

from agent_core.agent import DesktopAgent
from rag.knowledge_base_config import (
    get_knowledge_base_config,
)


@pytest.mark.parametrize(
    "kb_id, query",
    [
        ("tech", "RRF 排名融合"),
        ("company", "星桥科技 年假申请"),
    ],
)
def test_agent_knowledge_base_isolation(kb_id, query):
    """验证 Agent 的 Dense 和 BM25 使用指定知识库。"""

    config = get_knowledge_base_config(kb_id)

    agent = DesktopAgent(
        session_id=f"pytest_isolation_{kb_id}",
        knowledge_base_id=kb_id,
    )

    # 1. Agent 绑定正确的知识库
    assert agent.knowledge_base_id == kb_id

    # 2. Dense 加载正确的索引
    dense = agent.knowledge_retriever

    assert dense._loaded, dense._load_error
    assert dense.index_path.resolve() == config.index_path.resolve()
    assert dense.chunks_path.resolve() == config.chunks_path.resolve()
    assert len(dense.vector_store.chunks) > 0

    # 3. HybridRetriever 复用正确的 Dense
    hybrid = agent.hybrid_retriever

    assert hybrid is not None, agent._hybrid_init_error
    assert hybrid.dense_retriever is dense

    # 4. 验证 BM25 返回的文档来源
    results = hybrid.bm25_retriever.search(
        query=query,
        top_k=3,
    )

    assert results

    for result in results:
        source = Path(result.source).resolve()

        assert source.is_relative_to(
            config.knowledge_dir.resolve()
        ), f"发现跨知识库文档：{source}"


def test_unknown_knowledge_base_rejected():
    """未知知识库必须明确拒绝，不允许回退到默认库。"""

    with pytest.raises(ValueError):
        DesktopAgent(
            knowledge_base_id="unknown",
        )
