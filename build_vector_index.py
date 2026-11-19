"""根据知识库配置构建独立向量索引。"""

from __future__ import annotations

import argparse

from rag.document_loader import LocalDocumentLoader
from rag.embedding_client import EmbeddingClient
from rag.text_splitter import CharacterTextSplitter
from rag.vector_store import InMemoryVectorStore

from rag.knowledge_base_config import (
    KNOWLEDGE_BASES,
    get_knowledge_base_config,
)


def build_index(
    knowledge_base_id: str,
    overwrite: bool = False,
) -> None:
    """为指定知识库构建独立索引。"""

    config = get_knowledge_base_config(
        knowledge_base_id
    )

    print(f"\n========== 构建知识库：{knowledge_base_id} ==========")
    print(f"文档目录：{config.knowledge_dir}")
    print(f"向量索引：{config.index_path}")
    print(f"Chunk文件：{config.chunks_path}")

    # 1. 检查文档目录
    if not config.knowledge_dir.is_dir():
        raise FileNotFoundError(
            f"知识库目录不存在：{config.knowledge_dir}"
        )

    # 2. 防止覆盖已有索引
    existing_files = [
        path
        for path in [
            config.index_path,
            config.chunks_path,
        ]
        if path.exists()
    ]

    if existing_files and not overwrite:
        raise FileExistsError(
            "目标知识库已有索引文件，"
            "为保护现有数据，已取消构建。"
            "如确实需要重新构建，请显式添加 --overwrite。"
        )

    # 3. 加载文档
    print("\n[1/4] 加载文档……")

    loader = LocalDocumentLoader()

    documents = loader.load_directory(
        directory_path=str(config.knowledge_dir),
        recursive=True,
    )

    if not documents:
        raise ValueError(
            f"知识库 {knowledge_base_id} 没有可加载的文档"
        )

    print(f"文档数量：{len(documents)}")

    # 4. 文本切块
    print("\n[2/4] 文本切块……")

    splitter = CharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
    )

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:
        raise ValueError("没有生成有效 Chunk")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "生成了重复 Chunk ID，"
            "请检查知识库中是否存在冲突的文档路径"
        )

    print(f"Chunk数量：{len(chunks)}")

    # 5. 生成 Embedding
    print("\n[3/4] 生成 Embedding……")

    embedding_client = EmbeddingClient(
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
        batch_size=10,
    )

    chunk_texts = [
        chunk.content
        for chunk in chunks
    ]

    embeddings = embedding_client.embed_documents(
        chunk_texts
    )

    # 6. 建立向量索引
    print("\n[4/4] 保存索引……")

    vector_store = InMemoryVectorStore()

    vector_store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    # 确保输出目录存在
    config.index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.chunks_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save(
        index_path=str(config.index_path),
        chunks_path=str(config.chunks_path),
    )

    print("\n========== 构建完成 ==========")
    print(f"知识库ID：{knowledge_base_id}")
    print(f"文档数量：{len(documents)}")
    print(f"Chunk数量：{len(chunks)}")
    print(
        f"向量维度："
        f"{vector_store.embedding_matrix.shape[1]}"
    )
    print(f"索引文件：{config.index_path}")
    print(f"Chunk文件：{config.chunks_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建指定知识库的向量索引"
    )

    parser.add_argument(
        "--knowledge-base",
        choices=sorted(KNOWLEDGE_BASES),
        default="tech",
        help="知识库ID，默认tech",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有索引，使用前请确认",
    )

    args = parser.parse_args()

    build_index(
        knowledge_base_id=args.knowledge_base,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
