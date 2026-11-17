"""分阶段 Token 用量统计测试。"""

from types import SimpleNamespace
from unittest.mock import Mock

from agent_core.llm_client import QwenClient

from storage.request_trace import (
    RequestTrace,
    RequestTraceStore,
)

from storage.trace_context import (
    bind_request_trace,
    observed_phase,
)


MESSAGES = [
    {"role": "user", "content": "测试"}
]


def make_response(prompt, completion):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
    )


def make_client(responses):
    """使用 Mock 替代真正的 Qwen API。"""

    client = QwenClient.__new__(QwenClient)
    client.model = "mock-model"

    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=Mock(side_effect=responses),
            )
        )
    )

    return client


def test_phase_tokens_persisted(tmp_path):
    """每个模型阶段记录各自的真实 Token。"""

    client = make_client([
        make_response(100, 20),
        make_response(200, 30),
        make_response(300, 40),
    ])

    trace = RequestTrace(
        session_id="phase_A",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        with observed_phase("llm_call"):
            client.chat(MESSAGES)

        with observed_phase("answerability_check"):
            client.chat(MESSAGES)

        with observed_phase("grounding_check"):
            client.chat(MESSAGES)

    trace.finish()

    store = RequestTraceStore(
        str(tmp_path / "trace.db")
    )

    store.save(trace)

    saved = store.get(trace.request_id)

    assert saved is not None

    total = saved["token_usage"]

    assert total["prompt_tokens"] == 600
    assert total["completion_tokens"] == 90
    assert total["total_tokens"] == 690
    assert total["model_calls"] == 3

    phases = {
        p["name"]: p
        for p in saved["phases"]
    }

    assert (
        phases["llm_call"]["token_usage"]["total_tokens"]
        == 120
    )

    assert (
        phases["answerability_check"]
        ["token_usage"]["total_tokens"]
        == 230
    )

    assert (
        phases["grounding_check"]
        ["token_usage"]["total_tokens"]
        == 340
    )


def test_nested_phase_no_double_count():
    """嵌套阶段只归属最内层调用。"""

    client = make_client([
        make_response(10, 2),
        make_response(20, 3),
    ])

    trace = RequestTrace(
        session_id="phase_nested",
        knowledge_base_id="tech",
    )

    with bind_request_trace(trace):
        with observed_phase("outer"):
            client.chat(MESSAGES)

            with observed_phase("inner"):
                client.chat(MESSAGES)

    phases = {
        p.name: p
        for p in trace.phases
    }

    assert trace.token_usage["total_tokens"] == 35

    assert (
        phases["outer"].token_usage["total_tokens"]
        == 12
    )

    assert (
        phases["inner"].token_usage["total_tokens"]
        == 23
    )


def test_nested_request_not_misattributed():
    """另一个请求不能继承前一个请求的阶段用量。"""

    client = make_client([
        make_response(50, 5),
    ])

    trace_a = RequestTrace(
        session_id="session_A",
        knowledge_base_id="tech",
    )

    trace_b = RequestTrace(
        session_id="session_B",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace_a):
        with observed_phase("phase_a"):
            with bind_request_trace(trace_b):
                client.chat(MESSAGES)

    assert trace_a.token_usage is None

    assert trace_a.phases[0].token_usage is None

    assert trace_b.token_usage["total_tokens"] == 55


def test_missing_usage_reported_per_phase():
    """缺失 Token 用量时不能伪造实际消耗。"""

    client = make_client([
        SimpleNamespace(usage=None),
    ])

    trace = RequestTrace(
        session_id="phase_missing",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        with observed_phase("llm_rerank"):
            client.chat(MESSAGES)

    stats = trace.phases[0].token_usage

    assert stats["model_calls"] == 1
    assert stats["usage_reported_calls"] == 0
    assert stats["usage_missing_calls"] == 1
    assert stats["total_tokens"] == 0
