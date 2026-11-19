"""知识库来源路径与旧索引兼容性测试。"""

from pathlib import Path

from rag.document_loader import LocalDocumentLoader
from rag.text_splitter import (
    CharacterTextSplitter,
    DocumentChunk,
)
from rag.vector_store import InMemoryVectorStore


def make_documents(root):
    (root / "team").mkdir(parents=True)

    (root / "guide.md").write_text(
        "通用使用指南",
        encoding="utf-8",
    )

    (root / "team" / "guide.md").write_text(
        "团队使用指南",
        encoding="utf-8",
    )


def test_directory_sources_are_relative(tmp_path):
    kb = tmp_path / "knowledge"
    make_documents(kb)

    documents = LocalDocumentLoader().load_directory(kb)

    sources = {doc.source for doc in documents}

    assert sources == {
        "guide.md",
        "team/guide.md",
    }

    assert all(
        not Path(doc.source).is_absolute()
        for doc in documents
    )


def test_same_filename_has_unique_chunk_ids(tmp_path):
    kb = tmp_path / "knowledge"
    make_documents(kb)

    documents = LocalDocumentLoader().load_directory(kb)

    chunks = CharacterTextSplitter().split_documents(
        documents
    )

    ids = {chunk.chunk_id for chunk in chunks}

    assert ids == {
        "guide.md::chunk-0",
        "team/guide.md::chunk-0",
    }


def test_sources_stable_across_directories(tmp_path):
    first = tmp_path / "server_a" / "knowledge"
    second = tmp_path / "server_b" / "knowledge"

    make_documents(first)
    make_documents(second)

    def identity(root):
        docs = LocalDocumentLoader().load_directory(root)
        chunks = CharacterTextSplitter().split_documents(docs)
        return sorted(
            (chunk.source, chunk.chunk_id, chunk.content)
            for chunk in chunks
        )

    assert identity(first) == identity(second)


def test_legacy_absolute_source_index_still_loads(tmp_path):
    legacy_source = str(
        (tmp_path / "old_server" / "guide.md").resolve()
    )

    chunk = DocumentChunk(
        chunk_id="guide.md::chunk-0",
        content="原有文档内容",
        source=legacy_source,
        file_name="guide.md",
        chunk_index=0,
        metadata={},
    )

    index_path = tmp_path / "index.npz"
    chunks_path = tmp_path / "chunks.json"

    store = InMemoryVectorStore()
    store.add([chunk], [[1.0, 0.0]])
    store.save(index_path, chunks_path)

    restored = InMemoryVectorStore()
    restored.load(index_path, chunks_path)

    assert len(restored.chunks) == 1
    assert (
        restored.chunks[0].chunk_id
        == "guide.md::chunk-0"
    )
    assert restored.chunks[0].source == legacy_source
