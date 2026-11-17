"""统一 Agent Benchmark：运行、追踪和基础性能统计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agent_core.agent import DesktopAgent
from eval.score_answers import classify_answer, score_behavior
from storage.request_trace import RequestTrace, RequestTraceStore
from storage.trace_context import (
    bind_request_trace,
    observed_phase,
)


def load_cases(path: Path) -> list[dict]:
    """加载并验证 JSONL 或 JSON 评测数据集。"""

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix == ".jsonl":
        cases = [
            json.loads(line)
            for line in path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

    elif path.suffix == ".json":
        cases = json.loads(
            path.read_text(encoding="utf-8-sig")
        )

        if isinstance(cases, dict):
            cases = cases.get("cases")

    else:
        raise ValueError("数据集只支持 JSON 或 JSONL")

    if not isinstance(cases, list) or not cases:
        raise ValueError("评测数据集必须是非空列表")

    seen = set()

    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise ValueError(f"第 {index} 条不是对象")

        case_id = case.get("id")
        question = case.get("question")

        if (
            not isinstance(case_id, str)
            or not case_id.strip()
        ):
            raise ValueError(
                f"第 {index} 条缺少有效 id"
            )

        if case_id in seen:
            raise ValueError(
                f"重复的评测 ID：{case_id}"
            )

        seen.add(case_id)

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(
                f"{case_id} 缺少有效 question"
            )

    return cases


def percentile(values: list[float], p: float):
    """Nearest-rank 分位数，小样本也可计算。"""

    if not values:
        return None

    if not 0 < p <= 100:
        raise ValueError("分位数必须在 (0, 100] 内")

    ordered = sorted(values)
    index = max(
        0,
        math.ceil(len(ordered) * p / 100) - 1,
    )
    return round(ordered[index], 2)


def run_case(
    case: dict,
    knowledge_base_id: str,
    trace_store: RequestTraceStore,
) -> dict:
    """执行一次真实问答，保留成功或失败的追踪信息。"""

    session_id = "bench_" + uuid4().hex

    trace = RequestTrace(
        session_id=session_id,
        knowledge_base_id=knowledge_base_id,
    )

    agent = None
    answer = None

    try:
        with bind_request_trace(trace):
            with observed_phase("agent_initialization"):
                agent = DesktopAgent(
                    session_id=session_id,
                    knowledge_base_id=knowledge_base_id,
                )

                agent._eval_evidence = []

            with observed_phase("agent_execution"):
                answer = agent.chat(
                    case["question"],
                    knowledge_only=True,
                )

        trace.finish()

    except Exception as exc:
        trace.finish(error=exc)

    # 所有结果都写入 RequestTrace SQLite，
    # 使 tool_logs.request_id 能关联到真实请求。
    trace_store.save(trace)

    evidence_ids = []

    if agent is not None:
        for item in getattr(agent, "_eval_evidence", []):
            if isinstance(item, dict):
                chunk_id = item.get("chunk_id")

                if isinstance(chunk_id, str):
                    evidence_ids.append(chunk_id)

    phases = [
        asdict(phase)
        for phase in trace.phases
    ]

    def last_outcome(name: str):
        outcomes = [
            phase["outcome"]
            for phase in phases
            if phase["name"] == name
        ]
        return outcomes[-1] if outcomes else None

    last_tool = (
        getattr(agent, "_last_tool_result", None)
        if agent is not None
        else None
    )

    tool_error_type = getattr(
        last_tool, "error_type", None
    )

    grounding_outcome = last_outcome("grounding_check")

    grounding_failed = any(
        phase["name"] == "grounding_check"
        and phase["success"] is False
        for phase in phases
    )

    predicted_behavior = classify_answer(
        answer,
        status=trace.status,
        tool_error_type=tool_error_type,
        grounding_outcome=grounding_outcome,
        grounding_failed=grounding_failed,
    )

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_behavior": case.get(
            "expected_behavior"
        ),
        "knowledge_base_id": knowledge_base_id,
        "session_id": session_id,
        "request_id": trace.request_id,
        "status": trace.status,
        "error_type": trace.error_type,
        "tool_error_type": tool_error_type,
        "predicted_behavior": predicted_behavior,
        "grounding_failed": grounding_failed,
        "answer": answer,
        "duration_ms": trace.duration_ms,
        "token_usage": trace.token_usage,
        "answerability_outcome": last_outcome(
            "answerability_check"
        ),
        "grounding_outcome": grounding_outcome,
        "evidence_chunk_ids": evidence_ids,
        "phases": phases,
    }


def summarize(records: list[dict]) -> dict:
    """汇总执行性能，不把执行成功率误称为回答正确率。"""

    statuses = Counter(
        item["status"]
        for item in records
    )

    successful = [
        item
        for item in records
        if item["status"] == "success"
    ]

    latencies = [
        float(item["duration_ms"])
        for item in successful
    ]

    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model_calls": 0,
        "usage_reported_calls": 0,
        "usage_missing_calls": 0,
    }

    for item in records:
        usage = item.get("token_usage") or {}

        for key in total_usage:
            total_usage[key] += int(
                usage.get(key, 0)
            )

    grounding = Counter(
        item.get("grounding_outcome") or "not_executed"
        for item in records
    )

    answerability = Counter(
        item.get("answerability_outcome")
        or "not_executed"
        for item in records
    )

    return {
        "total_cases": len(records),
        "execution_status_counts": dict(statuses),
        "successful_execution_count": len(successful),
        "latency_ms": {
            "sample_count": len(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "method": "nearest_rank",
        },
        "token_usage_reported": total_usage,
        "grounding_outcomes": dict(grounding),
        "answerability_outcomes": dict(answerability),
        "behavior_metrics": score_behavior(records),
        "answer_accuracy": None,
        "answer_accuracy_note": (
            "尚未根据人工标注 Gold 进行回答正确性评分"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/answer_cases.jsonl"),
    )

    parser.add_argument(
        "--knowledge-base",
        choices=["tech", "company"],
        default="tech",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 表示全部问题",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="真正调用 Agent 和模型；不指定则只检查数据集",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit 不能为负数")

    cases = load_cases(args.dataset)

    if args.limit:
        cases = cases[:args.limit]

    labels = Counter(
        str(case.get("expected_behavior", "unlabeled"))
        for case in cases
    )

    print("========== Benchmark 数据集 ==========")
    print("数据文件：", args.dataset)
    print("知识库：", args.knowledge_base)
    print("本次问题数：", len(cases))
    print("已有行为标签分布：", dict(labels))

    if not args.run:
        print("\n当前为数据检查模式，不调用模型。")
        print("指定 --run 才会执行真实评测。")
        return

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S_%fZ")

    output = args.output or Path(
        f"eval/results/benchmark_"
        f"{args.knowledge_base}_{stamp}.jsonl"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 不覆盖任何既有结果。
    trace_store = RequestTraceStore("data/agent.db")

    records = []

    with output.open("x", encoding="utf-8") as f:
        for index, case in enumerate(cases, 1):
            print(
                f"\n[{index}/{len(cases)}] "
                f"{case['id']}"
            )

            record = run_case(
                case,
                args.knowledge_base,
                trace_store,
            )

            records.append(record)

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )
            f.flush()

            print("状态：", record["status"])
            print(
                "耗时(ms)：",
                record["duration_ms"],
            )
            print(
                "Grounding：",
                record["grounding_outcome"],
            )

    summary = summarize(records)
    summary["knowledge_base_id"] = args.knowledge_base
    summary["dataset"] = str(args.dataset)
    summary["dataset_sha256"] = hashlib.sha256(
        args.dataset.read_bytes()
    ).hexdigest()

    summary_path = output.with_suffix(
        ".summary.json"
    )

    with summary_path.open("x", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n========== Benchmark 完成 ==========")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("明细：", output)
    print("汇总：", summary_path)


if __name__ == "__main__":
    main()
