"""统一 Benchmark Runner 自动化测试。"""

import json

import pytest

from eval import run_benchmark as benchmark
from storage.request_trace import RequestTraceStore
from storage.trace_context import observed_phase


def test_load_jsonl_cases(tmp_path):
    path = tmp_path / "cases.jsonl"

    cases = [
        {
            "id": "case_1",
            "question": "问题一",
            "expected_behavior": "answer",
        },
        {
            "id": "case_2",
            "question": "问题二",
            "expected_behavior": "refuse",
        },
    ]

    path.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in cases
        ),
        encoding="utf-8",
    )

    loaded = benchmark.load_cases(path)

    assert len(loaded) == 2
    assert loaded[0]["id"] == "case_1"


def test_duplicate_case_rejected(tmp_path):
    path = tmp_path / "duplicate.jsonl"

    path.write_text(
        '{"id":"same","question":"A"}\n'
        '{"id":"same","question":"B"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        benchmark.load_cases(path)


def test_summary_does_not_fake_accuracy():
    """执行成功不等于答案正确。"""

    records = [
        {
            "status": "success",
            "duration_ms": 100.0,
            "token_usage": {
                "total_tokens": 120,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "model_calls": 1,
                "usage_reported_calls": 1,
                "usage_missing_calls": 0,
            },
            "grounding_outcome": "passed",
            "answerability_outcome": "sufficient",
        },
        {
            "status": "success",
            "duration_ms": 300.0,
            "token_usage": None,
            "grounding_outcome": None,
            "answerability_outcome": "insufficient",
        },
    ]

    summary = benchmark.summarize(records)

    assert summary["total_cases"] == 2
    assert summary["latency_ms"]["p50"] == 100
    assert summary["latency_ms"]["p95"] == 300
    assert (
        summary["token_usage_reported"]["total_tokens"]
        == 120
    )

    # 没有人工 Gold 时，不能凭 HTTP/Agent 成功判正确。
    assert summary["answer_accuracy"] is None


def test_run_case_trace_integration(
    tmp_path,
    monkeypatch,
):
    """使用假 Agent 测试真实 Trace 写入和阶段统计。"""

    class FakeAgent:
        def __init__(self, session_id, knowledge_base_id):
            self._eval_evidence = []

        def chat(self, question, *, knowledge_only=False):
            assert knowledge_only is True
            # 模拟真实 Agent 在本轮检索后记录证据
            self._eval_evidence.append(
                {"chunk_id": "mock::chunk-1"}
            )

            with observed_phase("grounding_check") as phase:
                if phase is not None:
                    phase.outcome = "passed"

            return "经过核验的测试回答"

    monkeypatch.setattr(
        benchmark,
        "DesktopAgent",
        FakeAgent,
    )

    store = RequestTraceStore(
        str(tmp_path / "trace.db")
    )

    record = benchmark.run_case(
        {
            "id": "case_test",
            "question": "模拟问题",
            "expected_behavior": "answer",
        },
        "company",
        store,
    )

    assert record["status"] == "success"
    assert record["knowledge_base_id"] == "company"
    assert record["grounding_outcome"] == "passed"
    assert record["evidence_chunk_ids"] == [
        "mock::chunk-1"
    ]

    saved = store.get(record["request_id"])

    assert saved is not None
    assert saved["status"] == "success"
    assert saved["knowledge_base_id"] == "company"
