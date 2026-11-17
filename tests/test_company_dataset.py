"""Company 评测集一致性和数据完整性测试。"""

import hashlib
import json

from eval.build_company_dataset import (
    OUTPUT_DIR,
    SOURCE_DIR,
    DOCUMENTS,
    validate_cases,
)
from eval.run_benchmark import load_cases


def get_cases():
    dev = load_cases(
        OUTPUT_DIR / "company_dev_v1.jsonl"
    )
    holdout = load_cases(
        OUTPUT_DIR / "company_holdout_v1.jsonl"
    )
    return dev, holdout


def test_dataset_counts_and_unique_questions():
    dev, holdout = get_cases()

    assert len(dev) == 16
    assert len(holdout) == 10

    # 包括跨数据集的重复检查
    validate_cases(dev + holdout)

    assert all(c["split"] == "dev" for c in dev)
    assert all(c["split"] == "holdout" for c in holdout)


def test_all_gold_evidence_exists_in_source():
    dev, holdout = get_cases()

    texts = {
        filename: (SOURCE_DIR / filename).read_text(
            encoding="utf-8"
        )
        for filename in DOCUMENTS.values()
    }

    for case in dev + holdout:
        for fact in case["gold_assertions"]:
            assert (
                fact["evidence_quote"]
                in texts[fact["source_file"]]
            )


def test_behavior_and_partial_labels():
    dev, holdout = get_cases()
    all_cases = dev + holdout

    answer_cases = [
        c for c in all_cases
        if c["expected_behavior"] == "answer"
    ]
    refuse_cases = [
        c for c in all_cases
        if c["expected_behavior"] == "refuse"
    ]
    partial_cases = [
        c for c in all_cases
        if c["scenario"] == "partial"
    ]

    assert len(answer_cases) == 21
    assert len(refuse_cases) == 5
    assert len(partial_cases) == 2

    for c in refuse_cases:
        assert c["gold_assertions"] == []
        assert c["missing_information"]

    for c in partial_cases:
        assert c["gold_assertions"]
        assert c["missing_information"]


def test_frozen_dataset_manifest():
    manifest_path = (
        OUTPUT_DIR / "company_v1_manifest.json"
    )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["counts"] == {
        "dev": 16,
        "holdout": 10,
    }

    for filename, expected_hash in (
        manifest["source_sha256"].items()
    ):
        actual = hashlib.sha256(
            (SOURCE_DIR / filename).read_bytes()
        ).hexdigest()
        assert actual == expected_hash

    for filename, expected_hash in (
        manifest["dataset_sha256"].items()
    ):
        actual = hashlib.sha256(
            (OUTPUT_DIR / filename).read_bytes()
        ).hexdigest()
        assert actual == expected_hash
