import pytest

from rag.knowledge_base_config import (
    get_knowledge_base_config,
    create_hybrid_retriever,
)


def test_tech_config():
    config = get_knowledge_base_config("tech")

    assert config.knowledge_base_id == "tech"
    assert config.index_path.name == "rag_index.npz"
    assert config.chunks_path.name == "rag_chunks.json"


def test_kb_paths_are_isolated():
    tech = get_knowledge_base_config("tech")
    company = get_knowledge_base_config("company")

    assert tech.index_path != company.index_path
    assert tech.chunks_path != company.chunks_path
    assert tech.knowledge_dir != company.knowledge_dir


def test_unknown_kb_rejected():
    with pytest.raises(ValueError):
        get_knowledge_base_config("unknown")


def test_missing_company_index_rejected():
    company = get_knowledge_base_config("company")

    if company.index_path.exists() and company.chunks_path.exists():
        pytest.skip("company 索引已经构建")

    with pytest.raises(FileNotFoundError):
        create_hybrid_retriever("company")
