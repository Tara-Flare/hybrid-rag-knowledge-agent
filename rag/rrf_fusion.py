from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

@dataclass(slots=True)
class RRFSearchResult:
    """
    Dense 和 BM25 经过 RRF 融合后的统一结果。
    """

    chunk_id: str
    text: str
    source: str
    metadata: dict[str, Any]

    rrf_score: float

    dense_rank: int | None
    bm25_rank: int | None

    dense_score: float | None
    bm25_score: float | None

def _get_value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    同时兼容：

    1. 普通 Python 对象：obj.chunk_id
    2. 字典：obj["chunk_id"]
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)

def _get_nested_content(result: Any) -> Any:
    """
    Dense 检索结果通常会把原始 Chunk 放在：

    result.document
    result.chunk
    result.item

    BM25 的结果通常直接把字段放在最外层。
    """
    possible_keys = [
        "document",
        "chunk",
        "item",
        "payload",
    ]

    for key in possible_keys:
        value = _get_value(
            result,
            key,
            None,
        )

        if value is not None:
            return value

    return None

def _extract_metadata(
    result: Any,
    nested: Any,
) -> dict[str, Any]:
    """
    合并最外层和嵌套对象的 metadata。
    """
    metadata: dict[str, Any] = {}

    nested_metadata = _get_value(
        nested,
        "metadata",
        {},
    )

    if isinstance(nested_metadata, dict):
        metadata.update(nested_metadata)

    result_metadata = _get_value(
        result,
        "metadata",
        {},
    )

    if isinstance(result_metadata, dict):
        metadata.update(result_metadata)

    return metadata

def _extract_source(
    result: Any,
    nested: Any,
    metadata: dict[str, Any],
) -> str:
    """
    提取来源文件路径。
    """
    possible_values = [
        _get_value(result, "source"),
        _get_value(nested, "source"),
        metadata.get("source"),
        metadata.get("file_path"),
        metadata.get("path"),
    ]

    for value in possible_values:
        if value is not None:
            source = str(value).strip()

            if source:
                return source

    return "unknown"

def _extract_chunk_id(
    result: Any,
    nested: Any,
    metadata: dict[str, Any],
    source: str,
) -> str:
    """
    从不同结构中提取 Chunk ID。
    """
    possible_values = [
        _get_value(result, "chunk_id"),
        _get_value(result, "id"),
        _get_value(nested, "chunk_id"),
        _get_value(nested, "id"),
        metadata.get("chunk_id"),
        metadata.get("id"),
    ]

    for value in possible_values:
        if value is not None:
            chunk_id = str(value).strip()

            if chunk_id:
                return chunk_id

    # 某些 Chunk 只保存 chunk_index，
    # 根据 source 和 chunk_index 还原统一 ID。
    possible_index_values = [
        _get_value(result, "chunk_index"),
        _get_value(nested, "chunk_index"),
        metadata.get("chunk_index"),
        metadata.get("index"),
    ]

    for value in possible_index_values:
        if value is None:
            continue

        try:
            chunk_index = int(value)
        except (TypeError, ValueError):
            continue

        source_name = Path(source).name

        return (
            f"{source_name}::chunk-{chunk_index}"
        )

    result_type = type(result).__name__
    nested_type = (
        type(nested).__name__
        if nested is not None
        else "None"
    )

    raise ValueError(
        "检索结果缺少有效的 chunk_id。"
        f" result 类型={result_type}，"
        f"嵌套对象类型={nested_type}，"
        f"metadata={metadata}"
    )

def _extract_text(
    result: Any,
    nested: Any,
) -> str:
    """
    提取 Chunk 原始文本。
    """
    possible_keys = [
        "text",
        "content",
        "page_content",
    ]

    for obj in [result, nested]:
        for key in possible_keys:
            value = _get_value(
                obj,
                key,
                None,
            )

            if value is not None:
                text = str(value).strip()

                if text:
                    return text

    return ""

def _extract_score(
    result: Any,
) -> float:
    """
    提取原始检索分数。
    """
    possible_keys = [
        "score",
        "similarity",
        "similarity_score",
    ]

    for key in possible_keys:
        value = _get_value(
            result,
            key,
            None,
        )

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return 0.0

def _normalize_result(
    result: Any,
) -> dict[str, Any]:
    """
    将 Dense 或 BM25 的不同结果结构，
    转换为统一字典。
    """
    nested = _get_nested_content(result)

    metadata = _extract_metadata(
        result=result,
        nested=nested,
    )

    source = _extract_source(
        result=result,
        nested=nested,
        metadata=metadata,
    )

    chunk_id = _extract_chunk_id(
        result=result,
        nested=nested,
        metadata=metadata,
        source=source,
    )

    text = _extract_text(
        result=result,
        nested=nested,
    )

    score = _extract_score(result)

    return {
        "chunk_id": chunk_id,
        "text": text,
        "source": source,
        "metadata": metadata,
        "score": score,
    }

def reciprocal_rank_fusion(
    dense_results: Sequence[Any],
    bm25_results: Sequence[Any],
    top_k: int = 3,
    rrf_k: int = 60,
) -> list[RRFSearchResult]:
    """
    使用 RRF 融合 Dense 和 BM25 的排名结果。
    """
    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0"
        )

    if rrf_k < 0:
        raise ValueError(
            "rrf_k 不能小于 0"
        )

    fused: dict[str, dict[str, Any]] = {}

    def ensure_chunk(
        result: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized = _normalize_result(result)

        chunk_id = normalized["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "text": normalized["text"],
                "source": normalized["source"],
                "metadata": normalized["metadata"],
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
            }

        # 若先加入的结果没有文本，
        # 使用另一路召回的文本进行补充。
        if (
            not fused[chunk_id]["text"]
            and normalized["text"]
        ):
            fused[chunk_id]["text"] = (
                normalized["text"]
            )

        if (
            fused[chunk_id]["source"]
            == "unknown"
            and normalized["source"]
            != "unknown"
        ):
            fused[chunk_id]["source"] = (
                normalized["source"]
            )

        return fused[chunk_id], normalized

    # Dense 排名
    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        chunk, normalized = ensure_chunk(result)

        chunk["dense_rank"] = rank
        chunk["dense_score"] = (
            normalized["score"]
        )

        chunk["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

    # BM25 排名
    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        chunk, normalized = ensure_chunk(result)

        chunk["bm25_rank"] = rank
        chunk["bm25_score"] = (
            normalized["score"]
        )

        chunk["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

    def sort_key(
        item: dict[str, Any],
    ) -> tuple[float, int, str]:
        ranks = [
            rank
            for rank in [
                item["dense_rank"],
                item["bm25_rank"],
            ]
            if rank is not None
        ]

        best_rank = (
            min(ranks)
            if ranks
            else 10**9
        )

        return (
            -item["rrf_score"],
            best_rank,
            item["chunk_id"],
        )

    ranked_items = sorted(
        fused.values(),
        key=sort_key,
    )

    final_results: list[RRFSearchResult] = []

    for item in ranked_items[:top_k]:
        final_results.append(
            RRFSearchResult(
                chunk_id=item["chunk_id"],
                text=item["text"],
                source=item["source"],
                metadata=item["metadata"],
                rrf_score=float(
                    item["rrf_score"]
                ),
                dense_rank=item["dense_rank"],
                bm25_rank=item["bm25_rank"],
                dense_score=item["dense_score"],
                bm25_score=item["bm25_score"],
            )
        )

    return final_results