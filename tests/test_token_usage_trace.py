"""QwenClient Token 用量追踪测试。"""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from agent_core.llm_client import QwenClient
from storage.request_trace import (
    RequestTrace,
    RequestTraceStore,
)
from storage.trace_context import bind_request_trace


def make_response(prompt, completion, total):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )
    )


def make_client(responses):
    """绕过真实 SDK 初始化，使用模拟模型响应。"""

    client = QwenClient.__new__(QwenClient)
    client.model = "mock-model"

    create = Mock(side_effect=responses)

    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=create,
            )
        )
    )

    return client, create


def test_token_aggregation_and_persistence(tmp_path):
    """chat 与 create_completion 不应重复统计。"""

    client, create = make_client([
        make_response(100, 20, 120),
        make_response(200, 30, 230),
    ])

    trace = RequestTrace(
        session_id="token_A",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        client.create_completion(
            messages=[{"role": "user", "content": "A"}]
        )

        client.chat(
            messages=[{"role": "user", "content": "B"}]
        )

    assert create.call_count == 2

    assert trace.token_usage == {
        "prompt_tokens": 300,
        "completion_tokens": 50,
        "total_tokens": 350,
        "model_calls": 2,
        "usage_reported_calls": 2,
        "usage_missing_calls": 0,
    }

    trace.finish()

    store = RequestTraceStore(
        str(tmp_path / "trace.db")
    )

    store.save(trace)

    saved = store.get(trace.request_id)

    assert saved["token_usage"] == trace.token_usage


def test_token_context_isolation():
    """不同请求的 Token 不能混用。"""

    client, _ = make_client([
        make_response(10, 2, 12),
        make_response(30, 5, 35),
        make_response(99, 1, 100),
    ])

    trace_a = RequestTrace(
        session_id="token_A",
        knowledge_base_id="tech",
    )

    trace_b = RequestTrace(
        session_id="token_B",
        knowledge_base_id="company",
    )

    messages = [{"role": "user", "content": "测试"}]

    with bind_request_trace(trace_a):
        client.chat(messages)

        with bind_request_trace(trace_b):
            client.chat(messages)

    # 没有绑定 Trace 的调用不应污染已有记录。
    client.chat(messages)

    assert trace_a.token_usage["total_tokens"] == 12
    assert trace_b.token_usage["total_tokens"] == 35

    assert trace_a.token_usage["model_calls"] == 1
    assert trace_b.token_usage["model_calls"] == 1


def test_missing_usage_not_fabricated():
    """缺失或非法的模型用量不能被当成真实 Token。"""

    invalid_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens="unknown",
            completion_tokens=5,
            total_tokens=10,
        )
    )

    client, _ = make_client([
        SimpleNamespace(usage=None),
        invalid_response,
    ])

    trace = RequestTrace(
        session_id="token_missing",
        knowledge_base_id="tech",
    )

    with bind_request_trace(trace):
        client.chat([{"role": "user", "content": "A"}])
        client.chat([{"role": "user", "content": "B"}])

    stats = trace.token_usage

    assert stats["model_calls"] == 2
    assert stats["usage_reported_calls"] == 0
    assert stats["usage_missing_calls"] == 2
    assert stats["total_tokens"] == 0


def test_api_token_usage_saved(tmp_path, monkeypatch):
    """FastAPI 请求应自动将模型 Token 写入 SQLite。"""

    api_main = import_module("api.main")

    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "test-secret-123",
    )

    store = RequestTraceStore(
        str(tmp_path / "trace.db")
    )

    monkeypatch.setattr(
        api_main,
        "trace_store",
        store,
    )

    qwen, _ = make_client([
        make_response(100, 10, 110),
        make_response(200, 20, 220),
    ])

    def fake_chat(message, *, knowledge_only=False):
        messages = [
            {"role": "user", "content": message}
        ]

        qwen.chat(messages)
        qwen.chat(messages)

        return "测试回答"

    with patch(
        "api.main.agent.chat",
        side_effect=fake_chat,
    ):
        with TestClient(api_main.app) as client:
            response = client.post(
                "/chat",
                json={"message": "你好"},
                headers={
                    "X-API-Key": "test-secret-123",
                },
            )

    assert response.status_code == 200

    request_id = response.headers["X-Request-ID"]

    saved = store.get(request_id)

    assert saved is not None
    assert saved["status"] == "success"

    stats = saved["token_usage"]

    assert stats["prompt_tokens"] == 300
    assert stats["completion_tokens"] == 30
    assert stats["total_tokens"] == 330
    assert stats["model_calls"] == 2
    assert stats["usage_reported_calls"] == 2
    assert stats["usage_missing_calls"] == 0
