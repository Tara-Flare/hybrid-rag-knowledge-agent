"""构建独立的 Company Challenge 向量索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from rag.document_loader import LocalDocumentLoader
from rag.embedding_client import EmbeddingClient
from rag.text_splitter import CharacterTextSplitter
from rag.vector_store import InMemoryVectorStore
from rag.knowledge_base_config import (
    get_knowledge_base_config,
)


ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT
    / "eval"
    / "challenge_corpus"
    / "company_v1_draft"
)

DOCS_DIR = BASE / "docs"
MANIFEST_PATH = BASE / "manifest.json"
INDEX_DIR = BASE / "index"

ORIGINAL_DIR = ROOT / "knowledge_bases" / "company"

ORIGINAL_FILES = (
    "leave_policy.md",
    "reimbursement_policy.md",
    "account_policy.md",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def validate_documents() -> int:
    """检查文档清单、SHA256 和原始制度一致性。"""

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    expected = {
        item["name"]: item["sha256"]
        for item in manifest["files"]
    }

    actual = {
        path.name: file_sha256(path)
        for path in DOCS_DIR.glob("*.md")
    }

    if expected != actual:
        raise RuntimeError(
            "挑战文档与 Manifest 不一致，禁止构建"
        )

    for name in ORIGINAL_FILES:
        original = ORIGINAL_DIR / name
        copied = DOCS_DIR / name

        if original.read_bytes() != copied.read_bytes():
            raise RuntimeError(
                f"原始制度内容不一致：{name}"
            )

    return len(actual)


def build(run: bool = False) -> None:
    """校验、切块，并按需生成独立索引。"""

    print("========== Challenge 索引构建 ==========")

    if INDEX_DIR.exists() or INDEX_DIR.is_symlink():
        raise FileExistsError(
            f"索引目录已存在，拒绝覆盖：{INDEX_DIR}"
        )

    document_count = validate_documents()

    loader = LocalDocumentLoader()

    documents = loader.load_directory(
        directory_path=str(DOCS_DIR),
        recursive=True,
    )

    if len(documents) != document_count:
        raise RuntimeError(
            "实际加载文档数与 Manifest 不一致"
        )

    splitter = CharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
    )

    chunks = splitter.split_documents(documents)

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    if not chunks:
        raise RuntimeError("没有生成 Chunk")

    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("存在重复 Chunk ID")

    if any(
        not chunk.content.strip()
        for chunk in chunks
    ):
        raise RuntimeError("存在空 Chunk")

    config = get_knowledge_base_config("company")

    print("文档数量：", document_count)
    print("Chunk 数量：", len(chunks))
    print("Embedding 模型：", config.embedding_model)
    print("Embedding 维度：", config.embedding_dimensions)
    print("目标目录：", INDEX_DIR)

    if not run:
        print("\nDRY RUN PASS")
        print("未调用 Embedding API")
        print("未写入向量索引")
        print("确认后可使用 --run 正式构建")
        return

    print("\n========== 生成 Embedding ==========")

    embedding_client = EmbeddingClient(
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
        batch_size=10,
    )

    embeddings = embedding_client.embed_documents(
        [chunk.content for chunk in chunks]
    )

    store = InMemoryVectorStore()

    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    # 先写临时目录，完成后再发布正式索引。
    staging = Path(
        tempfile.mkdtemp(
            prefix=".index_staging_",
            dir=BASE,
        )
    )

    try:
        store.save(
            index_path=str(staging / "rag_index.npz"),
            chunks_path=str(staging / "rag_chunks.json"),
        )

        if (
            not (staging / "rag_index.npz").is_file()
            or not (staging / "rag_chunks.json").is_file()
        ):
            raise RuntimeError("索引文件生成不完整")

        if INDEX_DIR.exists() or INDEX_DIR.is_symlink():
            raise FileExistsError(
                "目标索引目录已存在，拒绝覆盖"
            )

        staging.rename(INDEX_DIR)

    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print("\n========== 构建成功 ==========")
    print("Chunk 数量：", len(chunks))
    print("索引文件：", INDEX_DIR / "rag_index.npz")
    print("Chunk 文件：", INDEX_DIR / "rag_chunks.json")
    print("原始 Company 索引未修改")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        action="store_true",
        help="实际调用 Embedding API 构建索引",
    )

    args = parser.parse_args()

    build(run=args.run)


if __name__ == "__main__":
    main()
