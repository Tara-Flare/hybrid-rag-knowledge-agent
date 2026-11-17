"""对已保存的 Benchmark 结果进行 Gold Fact 离线评分。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from eval.gold_fact_scoring import (
    GoldFactScoringError,
    score_gold_answer,
)
from eval.qwen_adapter import qwen_generate
from eval.run_benchmark import load_cases


def load_results(path: Path) -> list[dict]:
    """加载已执行的 Benchmark JSONL 结果。"""

    if not path.is_file():
        raise FileNotFoundError(path)

    records = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        record = json.loads(line)

        if not isinstance(record, dict):
            raise ValueError(
                f"第 {line_number} 条结果不是对象"
            )

        records.append(record)

    if not records:
        raise ValueError("Benchmark 结果文件为空")

    return records


def validate_and_join(
    cases: list[dict],
    results: list[dict],
) -> tuple[list[tuple[dict, dict]], list[str]]:
    """检查结果与 Gold 数据集是否严格对应。"""

    cases_by_id = {case["id"]: case for case in cases}
    seen = set()
    joined = []

    for case in cases:
        expected = case.get("expected_behavior")

        if expected not in {"answer", "refuse"}:
            raise ValueError(
                f"无效行为标签：{case['id']}"
            )

        if expected == "answer":
            facts = case.get("gold_assertions")
            if not isinstance(facts, list) or not facts:
                raise ValueError(
                    f"缺少 Gold Facts：{case['id']}"
                )

    for record in results:
        case_id = record.get("id")

        if case_id not in cases_by_id:
            raise ValueError(
                f"结果包含未知题目：{case_id}"
            )

        if case_id in seen:
            raise ValueError(
                f"结果中存在重复题目：{case_id}"
            )

        seen.add(case_id)
        case = cases_by_id[case_id]

        if record.get("question") != case["question"]:
            raise ValueError(
                f"题目内容不一致：{case_id}"
            )

        if (
            record.get("expected_behavior")
            != case["expected_behavior"]
        ):
            raise ValueError(
                f"预期行为不一致：{case_id}"
            )

        expected_kb = case.get("knowledge_base_id")

        if (
            expected_kb is not None
            and record.get("knowledge_base_id") != expected_kb
        ):
            raise ValueError(
                f"知识库不一致：{case_id}"
            )

        joined.append((case, record))

    missing_ids = [
        case["id"]
        for case in cases
        if case["id"] not in seen
    ]

    return joined, missing_ids


def grade_records(
    joined: list[tuple[dict, dict]],
    llm_generate: Callable[[str], str],
) -> list[dict[str, Any]]:
    """评分；拒答和执行错误不调用评分模型。"""

    output = []

    for case, record in joined:
        expected = case["expected_behavior"]
        predicted = record.get("predicted_behavior")
        status = record.get("status")
        answer = record.get("answer")

        if expected == "refuse":
            # 正确拒答率由原 Benchmark 行为评分统计。
            # 拒答题没有需要覆盖的 Gold Facts。
            score = score_gold_answer(
                case,
                answer,
                llm_generate,
            )

        elif (
            status != "success"
            or predicted in {"refuse", "error"}
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            # 应回答却未能正常回答，事实覆盖计为 0。
            # 不花费额外 Token 让模型评分拒答文本。
            score = score_gold_answer(
                case,
                None,
                llm_generate,
            )

        else:
            try:
                score = score_gold_answer(
                    case,
                    answer,
                    llm_generate,
                )
            except GoldFactScoringError as exc:
                # 评分服务故障不能伪装成事实得分 0。
                score = {
                    "id": case["id"],
                    "status": "judge_error",
                    "error_type": type(exc).__name__,
                    "gold_fact_coverage": None,
                    "complete": None,
                    "judgments": [],
                }

        output.append({
            "id": case["id"],
            "request_id": record.get("request_id"),
            "expected_behavior": expected,
            "predicted_behavior": predicted,
            "execution_status": status,
            "scenario": case.get("scenario"),
            "gold_score": score,
        })

    return output


def summarize_gold(
    cases: list[dict],
    graded: list[dict],
    missing_ids: list[str],
) -> dict:
    """仅在评分完整时计算整体 Gold 指标。"""

    counts = Counter(
        item["gold_score"]["status"]
        for item in graded
    )

    answer_cases = [
        case for case in cases
        if case["expected_behavior"] == "answer"
    ]

    answer_results = [
        item for item in graded
        if item["expected_behavior"] == "answer"
    ]

    judge_errors = counts["judge_error"]

    # 数据集不完整或评分模型发生错误时，
    # 不输出貌似完整的总体指标。
    complete_evaluation = (
        not missing_ids
        and judge_errors == 0
        and len(graded) == len(cases)
    )

    micro_coverage = None
    complete_answer_rate = None

    if complete_evaluation and answer_cases:
        total_facts = sum(
            len(case["gold_assertions"])
            for case in answer_cases
        )

        covered = sum(
            item["gold_score"].get("covered_count", 0)
            for item in answer_results
        )

        fully_complete = sum(
            item["gold_score"].get("complete") is True
            for item in answer_results
        )

        micro_coverage = round(
            covered / total_facts,
            4,
        ) if total_facts else None

        complete_answer_rate = round(
            fully_complete / len(answer_cases),
            4,
        )

    return {
        "total_dataset_cases": len(cases),
        "processed_cases": len(graded),
        "missing_case_count": len(missing_ids),
        "expected_answer_cases": len(answer_cases),
        "expected_refuse_cases": (
            len(cases) - len(answer_cases)
        ),
        "gold_score_status_counts": dict(counts),
        "evaluation_complete": complete_evaluation,
        "gold_fact_micro_coverage": micro_coverage,
        "complete_answer_rate": complete_answer_rate,
        "grading_method": "llm_judge",
        "note": (
            "事实覆盖率按 Gold 事实数量加权；"
            "应回答却拒答的题目计为零覆盖；"
            "评分故障或题目缺失时，不输出总体指标。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--results",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="调用评分模型；默认只检查数据",
    )

    args = parser.parse_args()

    cases = load_cases(args.dataset)
    results = load_results(args.results)

    joined, missing_ids = validate_and_join(
        cases,
        results,
    )

    grade_candidates = sum(
        case["expected_behavior"] == "answer"
        and record.get("status") == "success"
        and record.get("predicted_behavior")
        not in {"refuse", "error"}
        and isinstance(record.get("answer"), str)
        and bool(record["answer"].strip())
        for case, record in joined
    )

    print("========== Gold Fact 评分检查 ==========")
    print("Gold 数据集：", args.dataset)
    print("Benchmark 结果：", args.results)
    print("数据集题数：", len(cases))
    print("已有结果：", len(results))
    print("缺失题数：", len(missing_ids))
    print("预计需要模型评分的回答：", grade_candidates)

    if not args.run:
        print("\n检查完成，未调用评分模型。")
        print("添加 --run 才开始真正评分。")
        return

    output = (
        args.output
        if args.output is not None
        else args.results.with_suffix(".gold.jsonl")
    )

    summary_path = output.with_suffix(
        ".summary.json"
    )

    if output.exists() or summary_path.exists():
        raise FileExistsError(
            "评分结果已存在，拒绝覆盖"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graded = grade_records(
        joined,
        qwen_generate,
    )

    summary = summarize_gold(
        cases,
        graded,
        missing_ids,
    )

    summary["dataset_sha256"] = hashlib.sha256(
        args.dataset.read_bytes()
    ).hexdigest()

    summary["results_sha256"] = hashlib.sha256(
        args.results.read_bytes()
    ).hexdigest()

    # 保留原始 Benchmark 文件，单独保存评分输出。
    with output.open("x", encoding="utf-8") as f:
        for item in graded:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                ) + "\n"
            )

    with summary_path.open("x", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n========== Gold Fact 评分结果 ==========")
    print(json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ))
    print("评分明细：", output)
    print("评分汇总：", summary_path)


if __name__ == "__main__":
    main()
