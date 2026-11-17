from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class KnowledgeBaseConfig:
    """单个知识库的配置。"""

    knowledge_base_id: str
    knowledge_dir: Path
    index_path: Path
    chunks_path: Path

    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024

    dense_top_k: int = 5
    bm25_top_k: int = 5
    rrf_k: int = 60
    top_k: int = 3

    rerank_enabled: bool = True


KNOWLEDGE_BASES = {
    "tech": KnowledgeBaseConfig(
        knowledge_base_id="tech",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        index_path=PROJECT_ROOT / "data/rag_index.npz",
        chunks_path=PROJECT_ROOT / "data/rag_chunks.json",
    ),

    "company": KnowledgeBaseConfig(
        knowledge_base_id="company",
        knowledge_dir=PROJECT_ROOT / "knowledge_bases/company",
        index_path=PROJECT_ROOT / "data/knowledge_bases/company/rag_index.npz",
        chunks_path=PROJECT_ROOT / "data/knowledge_bases/company/rag_chunks.json",
    ),
}


def get_knowledge_base_config(
    knowledge_base_id: str,
) -> KnowledgeBaseConfig:
    """根据知识库 ID 获取配置。"""

    if knowledge_base_id not in KNOWLEDGE_BASES:
        raise ValueError(
            f"未知知识库：{knowledge_base_id!r}。"
            f"可选知识库：{list(KNOWLEDGE_BASES)}"
        )

    return KNOWLEDGE_BASES[knowledge_base_id]


def create_hybrid_retriever(knowledge_base_id: str):
    """为指定知识库创建独立的混合检索器。"""

    config = get_knowledge_base_config(knowledge_base_id)

    # 检查索引是否存在，避免静默切换到默认知识库。
    for path in [config.index_path, config.chunks_path]:
        if not path.is_file():
            raise FileNotFoundError(
                f"知识库 {knowledge_base_id!r} 缺少文件：{path}"
            )

    from rag.knowledge_retriever import KnowledgeRetriever
    from rag.bm25_retriever import BM25Retriever
    from rag.hybrid_retriever import HybridRetriever

    dense = KnowledgeRetriever(
        index_path=str(config.index_path),
        chunks_path=str(config.chunks_path),
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )

    if not dense._loaded:
        raise RuntimeError(
            f"知识库 {knowledge_base_id!r} 向量索引加载失败："
            f"{dense._load_error}"
        )

    bm25 = BM25Retriever(
        chunks_path=str(config.chunks_path),
    )

    return HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_top_k=config.dense_top_k,
        bm25_top_k=config.bm25_top_k,
        rrf_k=config.rrf_k,
    )
