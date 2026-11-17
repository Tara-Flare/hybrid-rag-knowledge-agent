"""Company RAG 检索层消融实验。

比较：
A Dense
B BM25
C Hybrid RRF
D Hybrid RRF + LLM Rerank

默认只检查输入，不调用模型。
添加 --run 才真正运行实验。
"""

import argparse
import hashlib
import json
import math
import time

from datetime import datetime, timezone
from pathlib import Path


MODES = (
    "dense",
    "bm25",
    "hybrid_rrf",
    "hybrid_rerank",
)


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def get_chunk_id(item):
    """从不同检索结果结构中提取真实 Chunk ID。

    支持：
    - 顶层 chunk_id
    - 顶层 metadata.chunk_id
    - SearchResult.chunk.chunk_id
    - SearchResult.chunk.metadata.chunk_id
    """
    from collections.abc import Mapping

    def read_field(obj, name):
        if isinstance(obj, Mapping):
            return obj.get(name)
        return getattr(obj, name, None)

    def valid_id(value):
        return (
            isinstance(value, str)
            and bool(value.strip())
        )

    # Dense SearchResult 使用嵌套的 DocumentChunk。
    holders = [
        item,
        read_field(item, "chunk"),
    ]

    for holder in holders:
        if holder is None:
            continue

        value = read_field(holder, "chunk_id")

        if valid_id(value):
            return value.strip()

        metadata = read_field(holder, "metadata")

        if isinstance(metadata, Mapping):
            value = metadata.get("chunk_id")

            if valid_id(value):
                return value.strip()

    raise ValueError(
        "检索结果缺少有效 chunk_id；"
        f"结果类型={type(item).__name__}"
    )


def extract_ids(results, top_k=3):
    """提取去重后的前 K 个 Chunk ID。"""

    ids = []
    seen = set()

    for item in results:
        chunk_id = get_chunk_id(item)

        if chunk_id in seen:
            continue

        ids.append(chunk_id)
        seen.add(chunk_id)

        if len(ids) >= top_k:
            break

    return ids


def validate_labels(
    dataset_path,
    chunks_path,
    labels_path,
):
    """验证标签与冻结数据版本一致。"""

    dataset = load_jsonl(dataset_path)

    report = json.loads(
        labels_path.read_text(
            encoding="utf-8"
        )
    )

    if report["dataset_sha256"] != sha256(
        dataset_path
    ):
        raise ValueError(
            "Gold 数据集 SHA256 不匹配"
        )

    if report["chunks_sha256"] != sha256(
        chunks_path
    ):
        raise ValueError(
            "Chunk 数据 SHA256 不匹配"
        )

    if report["unmatched_gold_facts"] != 0:
        raise ValueError(
            "存在尚未映射的 Gold Facts"
        )

    cases = {
        item["id"]: item
        for item in dataset
    }

    labels = report["cases"]

    if len(cases) != len(dataset):
        raise ValueError(
            "Gold 数据集中存在重复 ID"
        )

    label_ids = [
        item["id"]
        for item in labels
    ]

    if len(label_ids) != len(set(label_ids)):
        raise ValueError(
            "相关性标签存在重复 ID"
        )

    if set(label_ids) != set(cases):
        raise ValueError(
            "相关性标签与 Gold 题目不一致"
        )

    for label in labels:
        case = cases[label["id"]]

        if not label["all_facts_mapped"]:
            raise ValueError(
                f"题目 {label['id']} 的证据未完整映射"
            )

        if (
            case["expected_behavior"]
            != label["expected_behavior"]
        ):
            raise ValueError(
                f"题目 {label['id']} 的行为标签不一致"
            )

        if case["expected_behavior"] == "answer":
            if not label["fact_labels"]:
                raise ValueError(
                    f"题目 {label['id']} 没有 Gold Facts"
                )

            if any(
                not fact["acceptable_chunk_ids"]
                for fact in label["fact_labels"]
            ):
                raise ValueError(
                    f"题目 {label['id']} 缺少有效 Chunk 标签"
                )

    return cases, labels, report


def filter_positive_bm25(results):
    """与 HybridRetriever 保持一致：过滤零分候选。"""
    return [
        item
        for item in results
        if item.score > 0
    ]


def fuse_cached_results(
    dense_results,
    bm25_results,
    top_k=5,
    rrf_k=60,
):
    """使用已经召回的候选执行原项目 RRF。

    不再调用 Embedding 或 BM25 检索器。
    """
    from rag.hybrid_retriever import (
        reciprocal_rank_fusion,
    )

    return reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=filter_positive_bm25(
            bm25_results
        ),
        top_k=top_k,
        rrf_k=rrf_k,
    )


def score_rankings(rows):
    """计算 Chunk-level Gold Fact Recall 和 MRR。

    rows 中每项包含：
    fact_labels、ranked_chunk_ids、latency_ms。

    一个 Gold Fact 可对应多个等价 Chunk。
    """

    if not rows:
        return {
            "evaluated_queries": 0,
            "gold_facts_total": 0,
            "gold_facts_hit": 0,
            "gold_fact_recall_at_3": None,
            "complete_evidence_rate_at_3": None,
            "mrr_at_3": None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
        }

    total_facts = 0
    hit_facts = 0
    complete_queries = 0

    reciprocal_ranks = []
    latencies = []

    for row in rows:
        ranked = row["ranked_chunk_ids"][:3]
        facts = row["fact_labels"]

        query_hits = 0
        first_hit_rank = None

        for fact in facts:
            acceptable = set(
                fact["acceptable_chunk_ids"]
            )

            rank = next(
                (
                    i
                    for i, chunk_id
                    in enumerate(ranked, start=1)
                    if chunk_id in acceptable
                ),
                None,
            )

            total_facts += 1

            if rank is not None:
                hit_facts += 1
                query_hits += 1

                if (
                    first_hit_rank is None
                    or rank < first_hit_rank
                ):
                    first_hit_rank = rank

        if facts and query_hits == len(facts):
            complete_queries += 1

        reciprocal_ranks.append(
            1 / first_hit_rank
            if first_hit_rank is not None
            else 0.0
        )

        latencies.append(
            float(row["latency_ms"])
        )

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    # 与现有 Benchmark 保持一致：nearest-rank
    p50 = sorted_latencies[
        math.ceil(0.50 * n) - 1
    ]

    p95 = sorted_latencies[
        math.ceil(0.95 * n) - 1
    ]

    return {
        "evaluated_queries": len(rows),
        "gold_facts_total": total_facts,
        "gold_facts_hit": hit_facts,
        "gold_fact_recall_at_3": (
            hit_facts / total_facts
            if total_facts else None
        ),
        "complete_evidence_rate_at_3": (
            complete_queries / len(rows)
        ),
        "mrr_at_3": (
            sum(reciprocal_ranks) / len(rows)
        ),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
    }


def run_experiment(args, cases, labels, report):
    """执行实际检索，仅在 --run 时调用。"""

    from rag.knowledge_retriever import (
        KnowledgeRetriever,
    )
    from rag.bm25_retriever import BM25Retriever
    from rag.llm_reranker import LLMReranker

    from storage.request_trace import RequestTrace
    from storage.trace_context import (
        bind_request_trace,
        observed_phase,
    )

    dense = KnowledgeRetriever(
        index_path=str(args.index),
        chunks_path=str(args.chunks),
    )

    bm25 = BM25Retriever(
        chunks_path=args.chunks,
    )

    reranker = LLMReranker(
        model=args.rerank_model,
        max_chars_per_chunk=800,
    )

    evaluated_labels = [
        label
        for label in labels
        if label["expected_behavior"] == "answer"
    ]

    if args.limit is not None:
        evaluated_labels = evaluated_labels[
            :args.limit
        ]

    results = []
    fallback_count = 0

    total_rerank_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model_calls": 0,
        "usage_missing_calls": 0,
    }

    for index, label in enumerate(
        evaluated_labels, start=1
    ):
        case_id = label["id"]
        query = cases[case_id]["question"]

        print(
            f"\n[{index}/{len(evaluated_labels)}] "
            f"{case_id}"
        )

        # A. Dense Only
        start = time.perf_counter()

        dense_results = dense.search(
            query=query,
            top_k=args.dense_top_k,
        )

        dense_ms = (
            time.perf_counter() - start
        ) * 1000

        # B. BM25 Only
        start = time.perf_counter()

        bm25_results = filter_positive_bm25(
            bm25.search(
                query=query,
                top_k=args.bm25_top_k,
            )
        )

        bm25_ms = (
            time.perf_counter() - start
        ) * 1000

        # C. Hybrid RRF
        # 复用 A、B 已经召回的结果，不重复请求。
        start = time.perf_counter()

        hybrid_results = fuse_cached_results(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=args.candidate_top_k,
            rrf_k=args.rrf_k,
        )

        fusion_ms = (
            time.perf_counter() - start
        ) * 1000

        # 混合检索耗时包括两路召回和融合。
        hybrid_ms = (
            dense_ms + bm25_ms + fusion_ms
        )

        # D. 基于同一批 Hybrid 候选执行 LLM Rerank
        trace = RequestTrace(
            session_id=f"ablation_{case_id}",
            knowledge_base_id="company",
        )

        start = time.perf_counter()
        fallback = False
        rerank_error = None

        try:
            with bind_request_trace(trace):
                with observed_phase("llm_rerank"):
                    reranked_results = reranker.rerank(
                        query=query,
                        candidates=hybrid_results,
                        top_k=3,
                    )

        except Exception as exc:
            fallback = True
            fallback_count += 1

            rerank_error = (
                f"{type(exc).__name__}: {exc}"
            )

            reranked_results = hybrid_results[:3]

        rerank_ms = (
            time.perf_counter() - start
        ) * 1000

        usage = dict(trace.token_usage)

        for key in total_rerank_usage:
            total_rerank_usage[key] += int(
                usage.get(key, 0) or 0
            )

        outputs = {
            "dense": (
                dense_results,
                dense_ms,
            ),
            "bm25": (
                bm25_results,
                bm25_ms,
            ),
            "hybrid_rrf": (
                hybrid_results,
                hybrid_ms,
            ),
            "hybrid_rerank": (
                reranked_results,
                hybrid_ms + rerank_ms,
            ),
        }

        for mode, (items, latency_ms) in outputs.items():
            results.append({
                "id": case_id,
                "mode": mode,
                "question": query,
                "fact_labels": label["fact_labels"],
                "ranked_chunk_ids": extract_ids(
                    items, top_k=3
                ),
                "latency_ms": round(
                    latency_ms, 2
                ),
                "rerank_fallback": (
                    fallback
                    if mode == "hybrid_rerank"
                    else False
                ),
                "rerank_error": (
                    rerank_error
                    if mode == "hybrid_rerank"
                    else None
                ),
                "rerank_token_usage": (
                    usage
                    if mode == "hybrid_rerank"
                    else None
                ),
            })

        print(
            "Dense:",
            extract_ids(dense_results),
        )
        print(
            "BM25:",
            extract_ids(bm25_results),
        )
        print(
            "RRF:",
            extract_ids(hybrid_results),
        )
        print(
            "Rerank:",
            extract_ids(reranked_results),
            "fallback=", fallback,
        )

    metrics = {}

    for mode in MODES:
        rows = [
            row
            for row in results
            if row["mode"] == mode
        ]

        metrics[mode] = score_rankings(rows)

    summary = {
        "dataset": str(args.dataset),
        "dataset_sha256": report["dataset_sha256"],
        "chunks": str(args.chunks),
        "chunks_sha256": report["chunks_sha256"],
        "index": str(args.index),
        "index_sha256": sha256(args.index),
        "labels": str(args.labels),
        "label_status": report["label_status"],
        "evaluation_scope": (
            "answerable_dev_cases_only"
        ),
        "recall_denominator": (
            "gold_facts_not_queries"
        ),
        "mrr_definition": (
            "mean_reciprocal_rank_of_first_relevant_chunk"
        ),
        "latency_note": (
            "Hybrid RRF 延迟为共享候选的 Dense "
            "+ BM25 + RRF 融合耗时之和；"
            "Hybrid+Rerank 额外加入重排耗时；"
            "不包含 Agent 生成与 Grounding"
        ),
        "config": {
            "top_k": 3,
            "candidate_top_k": args.candidate_top_k,
            "dense_top_k": args.dense_top_k,
            "bm25_top_k": args.bm25_top_k,
            "rrf_k": args.rrf_k,
            "rerank_model": args.rerank_model,
        },
        "metrics": metrics,
        "rerank_fallback_count": fallback_count,
        "rerank_token_usage": total_rerank_usage,
        "token_note": (
            "仅统计 Rerank Chat Completions Token；"
            "Dense Embedding Token 未纳入"
        ),
    }

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S_%fZ")

    details_path = output_dir / (
        f"retrieval_ablation_{stamp}.jsonl"
    )

    summary_path = output_dir / (
        f"retrieval_ablation_{stamp}.summary.json"
    )

    with details_path.open(
        "x", encoding="utf-8"
    ) as f:
        for row in results:
            f.write(
                json.dumps(
                    row, ensure_ascii=False
                ) + "\n"
            )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== 消融实验汇总 ==========")

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n明细：", details_path)
    print("汇总：", summary_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path(
            "data/knowledge_bases/company/rag_chunks.json"
        ),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            "data/knowledge_bases/company/rag_index.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results/ablation"),
    )

    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--rerank-model",
        default="qwen-plus",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="实际执行检索和 Rerank",
    )

    args = parser.parse_args()

    for path in (
        args.dataset,
        args.labels,
        args.chunks,
        args.index,
    ):
        if not path.is_file():
            parser.error(
                f"文件不存在：{path}"
            )

    if min(
        args.dense_top_k,
        args.bm25_top_k,
        args.candidate_top_k,
    ) < 3:
        parser.error(
            "各候选数量必须至少为 3"
        )

    if args.rrf_k < 0:
        parser.error("rrf-k 不能为负数")

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        parser.error("limit 必须大于 0")

    cases, labels, report = validate_labels(
        args.dataset,
        args.chunks,
        args.labels,
    )

    answer_labels = [
        label
        for label in labels
        if label["expected_behavior"] == "answer"
    ]

    print("========== 数据检查 ==========")
    print("Gold 题数：", len(cases))
    print("有效检索题数：", len(answer_labels))
    print("Gold Facts：", report["total_gold_facts"])
    print("匹配 Facts：", report["matched_gold_facts"])
    print("标签状态：", report["label_status"])
    print("实验组：", list(MODES))
    print("Top K：3")

    if not args.run:
        print(
            "\n仅完成数据校验；"
            "没有调用检索 API 或 LLM。"
        )
        return

    run_experiment(
        args, cases, labels, report
    )


if __name__ == "__main__":
    main()
