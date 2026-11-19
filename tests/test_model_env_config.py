
"""模型配置加载测试：不调用真实 API。"""

from agent_core import llm_client
from rag import embedding_client
from rag import llm_reranker


def test_qwen_uses_configured_model_and_endpoint(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.setenv("QWEN_MODEL", "mock-model")
    monkeypatch.setenv(
        "QWEN_BASE_URL", "https://chat.example/v1"
    )
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL", "https://embed.example/v1"
    )
    monkeypatch.setattr(llm_client, "OpenAI", fake_openai)

    client = llm_client.QwenClient()

    assert client.model == "mock-model"
    assert captured["base_url"] == "https://chat.example/v1"


def test_qwen_falls_back_to_dashscope_endpoint(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL", "https://shared.example/v1"
    )
    monkeypatch.setattr(llm_client, "OpenAI", fake_openai)

    client = llm_client.QwenClient(model="explicit-model")

    assert client.model == "explicit-model"
    assert captured["base_url"] == "https://shared.example/v1"


def test_embedding_keeps_original_endpoint(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL", "https://embed.example/v1"
    )
    monkeypatch.setattr(
        embedding_client, "OpenAI", fake_openai
    )

    client = embedding_client.EmbeddingClient()

    assert client.model == "text-embedding-v4"
    assert client.dimensions == 1024
    assert captured["base_url"] == "https://embed.example/v1"


def test_reranker_keeps_original_endpoint(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL", "https://embed.example/v1"
    )
    monkeypatch.setenv(
        "QWEN_BASE_URL", "https://chat.example/v1"
    )
    monkeypatch.setattr(llm_reranker, "OpenAI", fake_openai)

    client = llm_reranker.LLMReranker(
        model="mock-model"
    )

    assert client.model == "mock-model"
    assert captured["base_url"] == "https://embed.example/v1"
